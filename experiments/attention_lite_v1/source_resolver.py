#!/usr/bin/env python3
"""Resolve the exact validated Attention-Lite all-in-one source on Kaggle.

The paper runner intentionally does not reconstruct the successful model from the
modular repository.  This resolver makes that dependency explicit and friendly:

* accepts an explicit .py or .ipynb path;
* searches Kaggle inputs/working and the current repository;
* can extract a true all-in-one code cell from a Kaggle notebook input;
* validates protocol-specific golden markers before returning a source path.

It never converts a partial notebook or an unrelated NestSAR script into a paper
source.  Failure is explicit and includes the exact filenames that must be attached.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

CANONICAL_FILENAMES = {
    "xsub": "NestSAR_HOPE_Attention_Lite_TPUv5e8_XSUB_E40_ALL_IN_ONE.py",
    "xset": "NestSAR_HOPE_Attention_Lite_TPUv5e8_XSET_E40_ALL_IN_ONE.py",
}

NOTEBOOK_HINTS = {
    "xsub": (
        "NestSAR_HOPE_Attention_Lite_TPUv5e8_XSUB_E40_ONE_CELL.ipynb",
        "nestsar_attention_lite_E40.ipynb",
    ),
    "xset": (
        "NestSAR_HOPE_Attention_Lite_TPUv5e8_XSET_E40_ONE_CELL.ipynb",
    ),
}

COMMON_MARKERS = (
    "NESTSAR-HOPE-ATTENTION-LITE",
    "BUNDLE_B64",
    "EXPECTED_PARAMS = 2_381_028",
    "EXPECTED_LEAVES = 705",
    "ATTENTION_DIM = 64",
    "ATTENTION_HEADS = 4",
)


def _protocol(value: str) -> str:
    p = str(value).strip().lower()
    if p not in CANONICAL_FILENAMES:
        raise ValueError("protocol must be 'xsub' or 'xset'")
    return p


def _source_markers(protocol: str) -> tuple[str, ...]:
    p = _protocol(protocol)
    return COMMON_MARKERS + (f"REAL NTU120 {p.upper()}",)


def validate_canonical_source_text(text: str, protocol: str, *, origin: str = "source") -> None:
    missing = [marker for marker in _source_markers(protocol) if marker not in text]
    if missing:
        raise RuntimeError(
            f"{origin} is not the validated Attention-Lite {protocol.upper()} all-in-one source. "
            f"Missing markers: {', '.join(missing)}"
        )


def _read_python(path: Path, protocol: str) -> Path:
    text = path.read_text(encoding="utf-8", errors="strict")
    validate_canonical_source_text(text, protocol, origin=str(path))
    return path.resolve()


def _cell_text(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(str(x) for x in src)
    return str(src)


def _extract_notebook(path: Path, protocol: str, cache_dir: Path) -> Optional[Path]:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    candidates: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text = _cell_text(cell)
        if all(marker in text for marker in _source_markers(protocol)):
            candidates.append(text)

    if not candidates:
        return None

    # A true one-cell source is normally much larger than helper/audit cells.  If a
    # notebook contains more than one matching cell, keep the most complete candidate.
    text = max(candidates, key=len)
    validate_canonical_source_text(text, protocol, origin=str(path))

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / CANONICAL_FILENAMES[_protocol(protocol)]
    target.write_text(text, encoding="utf-8")
    return target.resolve()


def _walk_exact(roots: Iterable[Path], filename: str) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        direct = root / filename
        if direct.is_file():
            key = str(direct.resolve())
            if key not in seen:
                seen.add(key)
                found.append(direct)
        try:
            matches = root.rglob(filename)
            for path in matches:
                if not path.is_file():
                    continue
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(path)
        except (OSError, PermissionError):
            pass
    return found


def resolve_canonical_source(
    protocol: str,
    explicit: str | Path | None = None,
    *,
    cache_dir: str | Path = "/kaggle/working/NestSAR_attention_sources",
    verbose: bool = True,
) -> Path:
    """Return a validated protocol-specific all-in-one Python source.

    ``explicit`` may point to the exact .py file or to a one-cell .ipynb that embeds
    the exact source.  With ``explicit=None`` the resolver searches common Kaggle
    locations automatically.
    """
    p = _protocol(protocol)
    cache = Path(cache_dir).expanduser()

    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit canonical source does not exist: {path}")
        if path.suffix.lower() == ".py":
            result = _read_python(path, p)
        elif path.suffix.lower() == ".ipynb":
            result = _extract_notebook(path, p, cache)
            if result is None:
                raise RuntimeError(
                    f"Notebook does not contain a complete Attention-Lite {p.upper()} all-in-one cell: {path}"
                )
        else:
            raise ValueError("canonical source must be a .py or .ipynb file")
        if verbose:
            print(f"ATTENTION-LITE SOURCE {p.upper()}: {result}", flush=True)
        return result

    env_name = f"NESTSAR_{p.upper()}_CANONICAL_SOURCE"
    env_value = os.environ.get(env_name, "").strip() or os.environ.get("NESTSAR_CANONICAL_SOURCE", "").strip()
    if env_value:
        return resolve_canonical_source(p, env_value, cache_dir=cache, verbose=verbose)

    repo_dir = Path(__file__).resolve().parent
    roots = (
        repo_dir / "canonical",
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path.cwd(),
    )

    exact = CANONICAL_FILENAMES[p]
    rejected: list[str] = []
    for candidate in _walk_exact(roots, exact):
        try:
            result = _read_python(candidate, p)
        except Exception as exc:
            rejected.append(f"{candidate}: {exc}")
            continue
        if verbose:
            print(f"ATTENTION-LITE SOURCE {p.upper()}: {result}", flush=True)
        return result

    # If the exact .py was not attached, try an exact/known notebook input and extract
    # the self-contained code cell into /kaggle/working.
    notebook_names = NOTEBOOK_HINTS[p]
    for notebook_name in notebook_names:
        for candidate in _walk_exact(roots, notebook_name):
            result = _extract_notebook(candidate, p, cache)
            if result is not None:
                if verbose:
                    print(f"ATTENTION-LITE SOURCE {p.upper()} extracted from: {candidate}", flush=True)
                    print(f"ATTENTION-LITE SOURCE {p.upper()}: {result}", flush=True)
                return result

    lines = [
        f"Validated Attention-Lite {p.upper()} source was not found.",
        "",
        "Attach ONE of these to the Kaggle notebook as an Input, or pass its path explicitly:",
        f"  - {exact}",
    ]
    for name in notebook_names:
        lines.append(f"  - {name}")
    lines += [
        "",
        "Searched under /kaggle/input, /kaggle/working, the current directory, and the repo canonical folder.",
        "The runner will not silently substitute another architecture for a paper run.",
    ]
    if rejected:
        lines += ["", "Rejected same-name files:"] + [f"  - {x}" for x in rejected[:10]]
    raise FileNotFoundError("\n".join(lines))


def resolve_both_sources(
    *,
    xsub: str | Path | None = None,
    xset: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    return {
        "xsub": resolve_canonical_source("xsub", xsub, verbose=verbose),
        "xset": resolve_canonical_source("xset", xset, verbose=verbose),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resolve validated Attention-Lite Kaggle sources")
    parser.add_argument("--protocol", choices=("xsub", "xset", "both"), default="both")
    parser.add_argument("--source", default=None)
    args = parser.parse_args()

    if args.protocol == "both":
        resolved = resolve_both_sources()
        for key, value in resolved.items():
            print(f"{key.upper()}: {value}")
    else:
        print(resolve_canonical_source(args.protocol, args.source))

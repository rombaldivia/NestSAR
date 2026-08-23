#!/usr/bin/env python3
"""Resolve the validated Attention-Lite XSUB/XSET all-in-one trainers.

The repository already contains a SHA-verified canonical payload builder in
``canonical_payload/build.py``.  That builder is now the ONLY repository-native
source of truth used here:

* reconstruct exact validated XSUB from 8 committed LZMA+base64 chunks;
* verify XSUB source SHA256;
* derive XSET using the exact protocol-only substitutions recovered from the
  validated notebook;
* verify XSET source SHA256;
* syntax-compile both generated trainers before exposing them.

This replaces the older v2 gzip payload path that caused Kaggle to fail before
training.  External sources remain supported only as an explicit/compatibility
fallback; helper files such as this resolver can never be mistaken for a
canonical trainer.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, Optional

from .canonical_payload.build import (
    XSET_SHA256,
    XSUB_SHA256,
    ensure_canonical_sources,
)

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

PROTOCOL_GUARDS = {
    "xsub": (
        "REAL NTU120 XSUB",
        "TRAIN_EXPECTED = 63_026",
        "VAL_EXPECTED = 50_919",
        "assert MICROSTEPS_PER_EPOCH == 1970",
        "assert TOTAL_MICROSTEPS == 78_800",
        "assert TOTAL_STEPS == 19_700",
        'protocol="xsub"',
    ),
    "xset": (
        "REAL NTU120 XSET",
        "TRAIN_EXPECTED = 54_468",
        "VAL_EXPECTED = 59_477",
        "assert MICROSTEPS_PER_EPOCH == 1703",
        "assert TOTAL_MICROSTEPS == 68_120",
        "assert TOTAL_STEPS == 17_030",
        'protocol="xset"',
    ),
}

EXPECTED_SOURCE_SHA256 = {
    "xsub": XSUB_SHA256,
    "xset": XSET_SHA256,
}

EXPECTED_EMBEDDED_BUNDLE_SHA256 = (
    "c720c2afd9c32648ece4ac4b23e916f325039ba41684ebe4127ae285c6e216dd"
)

TEXT_SUFFIXES = {".py", ".txt"}
NOTEBOOK_SUFFIXES = {".ipynb"}
MAX_GENERIC_FILE_BYTES = 12 * 1024 * 1024


def _protocol(value: str) -> str:
    p = str(value).strip().lower()
    if p not in CANONICAL_FILENAMES:
        raise ValueError("protocol must be 'xsub' or 'xset'")
    return p


def _source_markers(protocol: str) -> tuple[str, ...]:
    return COMMON_MARKERS + PROTOCOL_GUARDS[_protocol(protocol)]


def _extract_embedded_bundle(text: str) -> str:
    match = re.search(r'BUNDLE_B64\s*=\s*r?"""', text)
    if match is None:
        raise RuntimeError("BUNDLE_B64 opening delimiter was not found")
    end = text.find('"""', match.end())
    if end < 0:
        raise RuntimeError("BUNDLE_B64 closing delimiter was not found")
    return text[match.end():end]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_embedded_bundle(text: str, *, origin: str) -> None:
    bundle_b64 = "".join(_extract_embedded_bundle(text).split())
    try:
        raw = base64.b64decode(bundle_b64, validate=True)
    except Exception as exc:
        raise RuntimeError(f"{origin}: embedded BUNDLE_B64 is not valid base64") from exc

    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_EMBEDDED_BUNDLE_SHA256:
        raise RuntimeError(
            f"{origin}: embedded v4.1 bundle SHA256 mismatch: "
            f"{digest} != {EXPECTED_EMBEDDED_BUNDLE_SHA256}"
        )


def validate_canonical_source_text(
    text: str,
    protocol: str,
    *,
    origin: str = "source",
    require_exact_source_sha: bool = False,
) -> None:
    """Validate architecture/protocol guards and embedded model bundle.

    ``require_exact_source_sha`` is used for repository-built canonical files.
    It is intentionally optional for explicit external sources because a
    byte-identical trainer may have harmless newline/container differences.
    """
    p = _protocol(protocol)
    missing = [marker for marker in _source_markers(p) if marker not in text]
    if missing:
        raise RuntimeError(
            f"{origin} is not the validated Attention-Lite {p.upper()} all-in-one source. "
            f"Missing markers: {', '.join(missing)}"
        )

    _validate_embedded_bundle(text, origin=origin)
    compile(text, origin, "exec")

    if require_exact_source_sha:
        digest = _sha256_text(text)
        expected = EXPECTED_SOURCE_SHA256[p]
        if digest != expected:
            raise RuntimeError(
                f"{origin}: canonical source SHA256 mismatch: {digest} != {expected}"
            )


def materialize_repo_canonical_sources(
    cache_dir: str | Path = "/kaggle/working/NestSAR_attention_sources",
    *,
    verbose: bool = True,
) -> dict[str, Path]:
    """Build exact repository canonical trainers and optionally mirror to cache.

    ``ensure_canonical_sources`` performs full XSUB/XSET source SHA verification
    itself.  We then perform an independent architecture/bundle validation here.
    """
    built = ensure_canonical_sources(verbose=verbose)

    verified: dict[str, Path] = {}
    for protocol in ("xsub", "xset"):
        p = Path(built[protocol]).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Canonical payload builder did not create {protocol}: {p}")

        text = p.read_text(encoding="utf-8")
        validate_canonical_source_text(
            text,
            protocol,
            origin=str(p),
            require_exact_source_sha=True,
        )
        verified[protocol] = p

    # The model bundle must be byte-for-byte identical across protocols.
    xsub_text = verified["xsub"].read_text(encoding="utf-8")
    xset_text = verified["xset"].read_text(encoding="utf-8")
    if _extract_embedded_bundle(xsub_text) != _extract_embedded_bundle(xset_text):
        raise RuntimeError("Repository XSUB/XSET embedded v4.1 bundles differ")

    # Optional stable cache used by the rest of the Kaggle orchestration.
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for protocol, source in verified.items():
        target = cache / CANONICAL_FILENAMES[protocol]
        data = source.read_bytes()
        if not target.is_file() or target.read_bytes() != data:
            target.write_bytes(data)
        result[protocol] = target.resolve()
        if verbose:
            print(
                f"ATTENTION-LITE REPO SOURCE {protocol.upper()}: {result[protocol]} "
                f"| sha256={hashlib.sha256(data).hexdigest()}",
                flush=True,
            )

    return result


def _read_text_source(path: Path, protocol: str, cache_dir: Path) -> Path:
    text = path.read_text(encoding="utf-8", errors="strict")
    validate_canonical_source_text(text, protocol, origin=str(path))
    if path.suffix.lower() == ".py":
        return path.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / CANONICAL_FILENAMES[_protocol(protocol)]
    target.write_text(text, encoding="utf-8")
    return target.resolve()


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
            for path in root.rglob(filename):
                if not path.is_file():
                    continue
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(path)
        except (OSError, PermissionError):
            pass
    return found


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _generic_candidates(roots: Iterable[Path], repo_root: Path) -> list[Path]:
    """Return external candidates while excluding repo helper/implementation code."""
    found: list[Path] = []
    seen: set[str] = set()
    canonical_dir = Path(__file__).resolve().parent / "canonical"

    for root in roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in (TEXT_SUFFIXES | NOTEBOOK_SUFFIXES):
                    continue
                if _is_within(path, repo_root) and not _is_within(path, canonical_dir):
                    continue
                try:
                    if path.stat().st_size > MAX_GENERIC_FILE_BYTES:
                        continue
                except OSError:
                    continue
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                found.append(path)
        except (OSError, PermissionError):
            pass
    return found


def _try_generic_candidate(path: Path, protocol: str, cache: Path) -> Optional[Path]:
    suffix = path.suffix.lower()
    try:
        if suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not all(marker in text for marker in _source_markers(protocol)):
                return None
            validate_canonical_source_text(text, protocol, origin=str(path))
            if suffix == ".py":
                return path.resolve()
            cache.mkdir(parents=True, exist_ok=True)
            target = cache / CANONICAL_FILENAMES[_protocol(protocol)]
            target.write_text(text, encoding="utf-8")
            return target.resolve()
        if suffix in NOTEBOOK_SUFFIXES:
            return _extract_notebook(path, protocol, cache)
    except Exception:
        return None
    return None


def _resolve_external_source(protocol: str, cache: Path, *, verbose: bool) -> Path:
    p = _protocol(protocol)
    repo_dir = Path(__file__).resolve().parent
    repo_root = repo_dir.parents[1]
    canonical_dir = repo_dir / "canonical"
    roots = (
        canonical_dir,
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path.cwd(),
    )

    exact = CANONICAL_FILENAMES[p]
    rejected: list[str] = []

    for candidate in _walk_exact(roots, exact):
        if _is_within(candidate, repo_root) and not _is_within(candidate, canonical_dir):
            continue
        try:
            result = _read_text_source(candidate, p, cache)
        except Exception as exc:
            rejected.append(f"{candidate}: {exc}")
            continue
        if verbose:
            print(f"ATTENTION-LITE EXTERNAL SOURCE {p.upper()}: {result}", flush=True)
        return result

    for notebook_name in NOTEBOOK_HINTS[p]:
        for candidate in _walk_exact(roots, notebook_name):
            if _is_within(candidate, repo_root) and not _is_within(candidate, canonical_dir):
                continue
            result = _extract_notebook(candidate, p, cache)
            if result is not None:
                if verbose:
                    print(f"ATTENTION-LITE SOURCE {p.upper()} extracted from: {candidate}", flush=True)
                return result

    scanned = 0
    for candidate in _generic_candidates(roots, repo_root):
        scanned += 1
        result = _try_generic_candidate(candidate, p, cache)
        if result is not None:
            if verbose:
                print(f"ATTENTION-LITE SOURCE {p.upper()} discovered in: {candidate}", flush=True)
            return result

    lines = [
        f"Validated Attention-Lite {p.upper()} source was not found.",
        f"External discovery scanned {scanned} candidate containers.",
    ]
    if rejected:
        lines += ["Rejected same-name files:"] + [f"  - {x}" for x in rejected[:10]]
    raise FileNotFoundError("\n".join(lines))


def resolve_canonical_source(
    protocol: str,
    explicit: str | Path | None = None,
    *,
    cache_dir: str | Path = "/kaggle/working/NestSAR_attention_sources",
    verbose: bool = True,
) -> Path:
    p = _protocol(protocol)
    cache = Path(cache_dir).expanduser()

    # Explicit override remains available for audits/debugging.
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit canonical source does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            result = _read_text_source(path, p, cache)
        elif suffix in NOTEBOOK_SUFFIXES:
            result = _extract_notebook(path, p, cache)
            if result is None:
                raise RuntimeError(f"Notebook lacks complete Attention-Lite {p.upper()} source: {path}")
        else:
            raise ValueError("canonical source must be a .py, .txt, or .ipynb file")
        if verbose:
            print(f"ATTENTION-LITE EXPLICIT SOURCE {p.upper()}: {result}", flush=True)
        return result

    env_name = f"NESTSAR_{p.upper()}_CANONICAL_SOURCE"
    env_value = (
        os.environ.get(env_name, "").strip()
        or os.environ.get("NESTSAR_CANONICAL_SOURCE", "").strip()
    )
    if env_value:
        return resolve_canonical_source(p, env_value, cache_dir=cache, verbose=verbose)

    # Normal zero-input path: the SHA-verified repository payload.
    try:
        return materialize_repo_canonical_sources(cache, verbose=verbose)[p]
    except Exception as repo_exc:
        if verbose:
            print(f"ATTENTION-LITE repository canonical build failed: {repo_exc}", flush=True)
            print("Trying external compatibility source...", flush=True)
        try:
            return _resolve_external_source(p, cache, verbose=verbose)
        except Exception as external_exc:
            raise RuntimeError(
                "Attention-Lite source integration failed.\n"
                f"Repository canonical builder error: {repo_exc}\n"
                f"External fallback error: {external_exc}"
            ) from repo_exc


def resolve_both_sources(
    *,
    xsub: str | Path | None = None,
    xset: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    # With no explicit overrides, build exactly once and return both verified files.
    if xsub is None and xset is None:
        return materialize_repo_canonical_sources(verbose=verbose)

    return {
        "xsub": resolve_canonical_source("xsub", xsub, verbose=verbose),
        "xset": resolve_canonical_source("xset", xset, verbose=verbose),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resolve validated Attention-Lite sources")
    parser.add_argument("--protocol", choices=("xsub", "xset", "both"), default="both")
    parser.add_argument("--source", default=None)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()

    if args.materialize or args.protocol == "both":
        resolved = resolve_both_sources()
        for key, value in resolved.items():
            print(f"{key.upper()}: {value}")
    else:
        print(resolve_canonical_source(args.protocol, args.source))

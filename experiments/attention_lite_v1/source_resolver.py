#!/usr/bin/env python3
"""Resolve the validated Attention-Lite all-in-one source.

Resolution order:
1. explicit path / environment override;
2. repository-bundled canonical payload (zero extra Kaggle input required);
3. external Kaggle source discovery as a compatibility fallback.

The repository payload stores the validated XSET all-in-one source as a split
base64-encoded gzip artifact. XSUB is materialized deterministically from that
source by changing only protocol-specific text/constants *outside* the embedded
BUNDLE_B64 v4.1 model bundle. The embedded bundle is kept byte-for-byte
identical.

Important safety properties:
- the embedded v4.1 bundle must match its known SHA-256;
- the generated source must contain the locked architecture/schedule guards;
- the generated source must compile as Python;
- fallback discovery is forbidden from selecting this resolver/package itself;
- if the repository gzip wrapper has a damaged CRC/footer, a raw-DEFLATE
  recovery path is allowed only when the recovered Python source and embedded
  model bundle pass all validation guards.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import re
import zlib
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

TEXT_SUFFIXES = {".py", ".txt"}
NOTEBOOK_SUFFIXES = {".ipynb"}
MAX_GENERIC_FILE_BYTES = 12 * 1024 * 1024

EXPECTED_EMBEDDED_BUNDLE_SHA256 = (
    "c720c2afd9c32648ece4ac4b23e916f325039ba41684ebe4127ae285c6e216dd"
)

# Repository-bundled canonical artifact. These four chunks together contain
# base64(gzip(validated XSET all-in-one source)).
BUNDLED_PAYLOAD_DIR = Path(__file__).resolve().parent / "canonical_payloads" / "v2"
BUNDLED_XSET_PARTS = tuple(f"xset_{i:02d}.b64" for i in range(1, 5))
BUNDLED_XSET_B64_LENGTH = 50_020


def _protocol(value: str) -> str:
    p = str(value).strip().lower()
    if p not in CANONICAL_FILENAMES:
        raise ValueError("protocol must be 'xsub' or 'xset'")
    return p


def _source_markers(protocol: str) -> tuple[str, ...]:
    p = _protocol(protocol)
    return COMMON_MARKERS + PROTOCOL_GUARDS[p]


def _extract_embedded_bundle(text: str) -> str:
    match = re.search(r'BUNDLE_B64\s*=\s*r?"""', text)
    if match is None:
        raise RuntimeError("BUNDLE_B64 opening delimiter was not found")
    end = text.find('"""', match.end())
    if end < 0:
        raise RuntimeError("BUNDLE_B64 closing delimiter was not found")
    return text[match.end():end]


def _validate_embedded_bundle(text: str, *, origin: str) -> None:
    bundle_b64 = _extract_embedded_bundle(text)
    compact = "".join(bundle_b64.split())
    try:
        raw = base64.b64decode(compact, validate=True)
    except Exception as exc:
        raise RuntimeError(f"{origin}: embedded BUNDLE_B64 is not valid base64") from exc

    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_EMBEDDED_BUNDLE_SHA256:
        raise RuntimeError(
            f"{origin}: embedded v4.1 bundle SHA-256 mismatch: "
            f"{digest} != {EXPECTED_EMBEDDED_BUNDLE_SHA256}"
        )


def validate_canonical_source_text(text: str, protocol: str, *, origin: str = "source") -> None:
    p = _protocol(protocol)
    missing = [marker for marker in _source_markers(p) if marker not in text]
    if missing:
        raise RuntimeError(
            f"{origin} is not the validated Attention-Lite {p.upper()} all-in-one source. "
            f"Missing markers: {', '.join(missing)}"
        )
    _validate_embedded_bundle(text, origin=origin)
    compile(text, origin, "exec")


def _raw_deflate_from_gzip(blob: bytes) -> bytes:
    """Return the raw DEFLATE payload from a gzip member without checking CRC.

    This is used only as a recovery path if Python's strict gzip decoder rejects
    the wrapper/footer. The recovered source still has to pass source compilation,
    protocol guards and the exact embedded-bundle SHA-256 check.
    """
    if len(blob) < 18 or blob[:2] != b"\x1f\x8b" or blob[2] != 8:
        raise RuntimeError("repository payload is not a supported gzip member")

    flags = blob[3]
    pos = 10

    if flags & 0x04:  # FEXTRA
        if pos + 2 > len(blob) - 8:
            raise RuntimeError("truncated gzip FEXTRA length")
        xlen = int.from_bytes(blob[pos:pos + 2], "little")
        pos += 2 + xlen

    if flags & 0x08:  # FNAME
        end = blob.find(b"\x00", pos, len(blob) - 8)
        if end < 0:
            raise RuntimeError("truncated gzip FNAME")
        pos = end + 1

    if flags & 0x10:  # FCOMMENT
        end = blob.find(b"\x00", pos, len(blob) - 8)
        if end < 0:
            raise RuntimeError("truncated gzip FCOMMENT")
        pos = end + 1

    if flags & 0x02:  # FHCRC
        pos += 2

    if pos >= len(blob) - 8:
        raise RuntimeError("gzip member has no DEFLATE body")

    return blob[pos:-8]


def _decode_repository_payload(encoded: str) -> str:
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise RuntimeError(f"repository XSET payload base64 decode failed: {exc}") from exc

    if compressed[:2] != b"\x1f\x8b":
        raise RuntimeError("repository XSET payload decoded, but gzip magic is missing")

    strict_error: Exception | None = None
    try:
        return gzip.decompress(compressed).decode("utf-8")
    except Exception as exc:
        strict_error = exc

    # Recovery for a damaged gzip wrapper/footer. We never trust recovery by
    # itself: validate_canonical_source_text() performs the strong source and
    # embedded-bundle checks immediately after this function returns.
    try:
        body = _raw_deflate_from_gzip(compressed)
        recovered = zlib.decompress(body, -zlib.MAX_WBITS).decode("utf-8")
    except Exception as recovery_exc:
        raise RuntimeError(
            "repository XSET payload could not be decoded. "
            f"strict gzip error={strict_error!r}; raw-deflate recovery error={recovery_exc!r}"
        ) from recovery_exc

    print(
        "ATTENTION-LITE PAYLOAD: strict gzip wrapper check failed, but raw-DEFLATE "
        "recovery succeeded; validating exact embedded bundle and source guards...",
        flush=True,
    )
    return recovered


def _read_bundled_xset_text() -> str:
    missing = [name for name in BUNDLED_XSET_PARTS if not (BUNDLED_PAYLOAD_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Repository canonical XSET payload is incomplete. Missing: " + ", ".join(missing)
        )

    parts = [
        (BUNDLED_PAYLOAD_DIR / name).read_text(encoding="ascii").strip()
        for name in BUNDLED_XSET_PARTS
    ]
    encoded = "".join(parts)

    if len(encoded) != BUNDLED_XSET_B64_LENGTH:
        raise RuntimeError(
            f"Repository XSET payload length mismatch: {len(encoded)} != {BUNDLED_XSET_B64_LENGTH}; "
            f"part lengths={[len(x) for x in parts]}"
        )
    if not encoded.startswith("H4sI"):
        raise RuntimeError("Repository XSET payload does not look like gzip+base64")

    text = _decode_repository_payload(encoded)
    validate_canonical_source_text(text, "xset", origin="repo-bundled XSET")
    return text


def _replace_exact_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"XSUB materialization expected exactly one {label} marker, found {count}: {old!r}"
        )
    return text.replace(old, new, 1)


def _xset_to_xsub_outside_bundle(xset_text: str) -> str:
    """Materialize XSUB without touching the embedded v4.1 source bundle."""
    match = re.search(r'BUNDLE_B64\s*=\s*r?"""', xset_text)
    if match is None:
        raise RuntimeError("Cannot isolate BUNDLE_B64 while materializing XSUB")
    bundle_end = xset_text.find('"""', match.end())
    if bundle_end < 0:
        raise RuntimeError("Cannot find BUNDLE_B64 end while materializing XSUB")
    bundle_end += 3

    prefix = xset_text[:match.start()]
    bundle_section = xset_text[match.start():bundle_end]
    suffix = xset_text[bundle_end:]

    # Protocol words/names are safe to change only outside the opaque bundle.
    prefix = prefix.replace("XSET", "XSUB").replace("xset", "xsub")
    suffix = suffix.replace("XSET", "XSUB").replace("xset", "xsub")

    outside = prefix + "\0BUNDLE\0" + suffix

    # Exact official NTU120 split sizes and the resulting E40 schedule.
    exact_changes = (
        ("TRAIN_EXPECTED = 54_468", "TRAIN_EXPECTED = 63_026", "train count"),
        ("VAL_EXPECTED = 59_477", "VAL_EXPECTED = 50_919", "validation count"),
        ("assert MICROSTEPS_PER_EPOCH == 1703", "assert MICROSTEPS_PER_EPOCH == 1970", "microsteps/epoch"),
        ("assert TOTAL_MICROSTEPS == 68_120", "assert TOTAL_MICROSTEPS == 78_800", "total microsteps"),
        ("assert TOTAL_STEPS == 17_030", "assert TOTAL_STEPS == 19_700", "optimizer steps"),
    )
    for old, new, label in exact_changes:
        outside = _replace_exact_once(outside, old, new, label)

    # Human-readable comments/prints only.
    outside = outside.replace("54,468", "63,026")
    outside = outside.replace("59,477", "50,919")
    outside = outside.replace("68,120", "78,800")
    outside = outside.replace("17,030", "19,700")

    prefix, suffix = outside.split("\0BUNDLE\0", 1)
    xsub_text = prefix + bundle_section + suffix

    if _extract_embedded_bundle(xsub_text) != _extract_embedded_bundle(xset_text):
        raise RuntimeError("XSUB materialization changed BUNDLE_B64; refusing to continue")

    validate_canonical_source_text(xsub_text, "xsub", origin="repo-materialized XSUB")
    return xsub_text


def materialize_repo_canonical_sources(
    cache_dir: str | Path = "/kaggle/working/NestSAR_attention_sources",
    *,
    verbose: bool = True,
) -> dict[str, Path]:
    """Decode repository artifact and emit validated executable XSUB/XSET files."""
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    xset_text = _read_bundled_xset_text()
    xsub_text = _xset_to_xsub_outside_bundle(xset_text)

    texts = {"xsub": xsub_text, "xset": xset_text}
    result: dict[str, Path] = {}
    for p, text in texts.items():
        target = cache / CANONICAL_FILENAMES[p]
        target.write_text(text, encoding="utf-8")
        written = target.read_text(encoding="utf-8")
        validate_canonical_source_text(written, p, origin=str(target))
        result[p] = target.resolve()
        if verbose:
            print(f"ATTENTION-LITE REPO SOURCE {p.upper()}: {result[p]}", flush=True)

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


def _generic_candidates(roots: Iterable[Path], *, excluded_repo_root: Path | None = None) -> list[Path]:
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

                # Never let a marker-rich implementation/helper file masquerade as
                # the canonical trainer. Repo-local generic discovery is allowed only
                # in the dedicated canonical/ directory.
                if excluded_repo_root is not None and _is_within(path, excluded_repo_root):
                    if not _is_within(path, canonical_dir):
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
    roots = (
        repo_dir / "canonical",
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path.cwd(),
    )
    exact = CANONICAL_FILENAMES[p]
    rejected: list[str] = []

    for candidate in _walk_exact(roots, exact):
        # Same exclusion for exact-name files: a repo implementation file may not
        # substitute for the canonical source unless it is under canonical/.
        if _is_within(candidate, repo_root) and not _is_within(candidate, repo_dir / "canonical"):
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
            if _is_within(candidate, repo_root) and not _is_within(candidate, repo_dir / "canonical"):
                continue
            result = _extract_notebook(candidate, p, cache)
            if result is not None:
                if verbose:
                    print(f"ATTENTION-LITE SOURCE {p.upper()} extracted from: {candidate}", flush=True)
                return result

    scanned = 0
    for candidate in _generic_candidates(roots, excluded_repo_root=repo_root):
        scanned += 1
        result = _try_generic_candidate(candidate, p, cache)
        if result is not None:
            if verbose:
                print(f"ATTENTION-LITE SOURCE {p.upper()} discovered by markers in: {candidate}", flush=True)
            return result

    lines = [
        f"Validated Attention-Lite {p.upper()} source was not found.",
        f"Repository materialization failed and external discovery scanned {scanned} containers.",
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
    env_value = os.environ.get(env_name, "").strip() or os.environ.get("NESTSAR_CANONICAL_SOURCE", "").strip()
    if env_value:
        return resolve_canonical_source(p, env_value, cache_dir=cache, verbose=verbose)

    try:
        bundled = materialize_repo_canonical_sources(cache, verbose=verbose)
        return bundled[p]
    except Exception as repo_exc:
        if verbose:
            print(f"ATTENTION-LITE repository source materialization failed: {repo_exc}", flush=True)
            print("Falling back to external source discovery...", flush=True)
        try:
            return _resolve_external_source(p, cache, verbose=verbose)
        except Exception as external_exc:
            raise RuntimeError(
                "Attention-Lite source integration failed.\n"
                f"Repository artifact error: {repo_exc}\n"
                f"External fallback error: {external_exc}"
            ) from repo_exc


def resolve_both_sources(
    *,
    xsub: str | Path | None = None,
    xset: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    if xsub is None and xset is None:
        try:
            return materialize_repo_canonical_sources(verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"Repository both-source materialization failed: {exc}", flush=True)
                print("Trying compatibility source resolution per protocol...", flush=True)
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

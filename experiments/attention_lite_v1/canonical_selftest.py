#!/usr/bin/env python3
"""Static self-test for the exact GitHub-integrated Attention-Lite sources."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .canonical_integrated import (
    XSET_SHA256,
    XSUB_SHA256,
    ensure_canonical_sources,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_embedded_bundle(text: str) -> str:
    match = re.search(r'BUNDLE_B64\s*=\s*r?"""', text)
    if match is None:
        raise RuntimeError("BUNDLE_B64 opening delimiter was not found")
    end = text.find('"""', match.end())
    if end < 0:
        raise RuntimeError("BUNDLE_B64 closing delimiter was not found")
    return text[match.end():end]


def _check_source(path: Path, protocol: str, expected_sha: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)

    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"{protocol.upper()} SHA256 mismatch in selftest:\n"
            f"expected: {expected_sha}\n"
            f"actual:   {actual_sha}"
        )

    text = path.read_text(encoding="utf-8")
    required = (
        "NESTSAR-HOPE-ATTENTION-LITE",
        "BUNDLE_B64",
        "EXPECTED_PARAMS = 2_381_028",
        "EXPECTED_LEAVES = 705",
        "ATTENTION_DIM = 64",
        "ATTENTION_HEADS = 4",
        f"REAL NTU120 {protocol.upper()}",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(
            f"{protocol.upper()} source missing marker(s): " + ", ".join(missing)
        )

    compile(text, str(path), "exec")
    return text


def main() -> int:
    # No external fallback is used here. A PASS proves the repository integration.
    sources = ensure_canonical_sources(verbose=True)

    xsub_path = Path(sources["xsub"])
    xset_path = Path(sources["xset"])

    xsub_text = _check_source(xsub_path, "xsub", XSUB_SHA256)
    xset_text = _check_source(xset_path, "xset", XSET_SHA256)

    xsub_bundle = _extract_embedded_bundle(xsub_text)
    xset_bundle = _extract_embedded_bundle(xset_text)
    if xsub_bundle != xset_bundle:
        raise RuntimeError("XSUB/XSET embedded v4.1 bundles differ")

    print("=" * 108)
    print("NESTSAR ATTENTION-LITE — EXACT GITHUB CANONICAL SELFTEST: PASS")
    print("=" * 108)
    print(f"XSUB | {xsub_path}")
    print(f"       sha256={XSUB_SHA256} | bytes={xsub_path.stat().st_size:,}")
    print(f"XSET | {xset_path}")
    print(f"       sha256={XSET_SHA256} | bytes={xset_path.stat().st_size:,}")
    print("Embedded BUNDLE_B64: IDENTICAL across protocols")
    print("Python compilation: PASS")
    print("Architecture guards: 2,381,028 params / 705 leaves / T16 / D64-H4-Dh16")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

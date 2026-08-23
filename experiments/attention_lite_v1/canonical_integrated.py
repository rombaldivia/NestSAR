#!/usr/bin/env python3
"""Single integration point for exact validated Attention-Lite sources.

The canonical payload builder reconstructs the exact validated XSUB trainer from
committed LZMA+base64 chunks, derives XSET using the exact protocol-only edits, and
SHA256-verifies both files.  This wrapper adds independent read-back/architecture
checks and is the only canonical entry point used by the Kaggle both-protocol runner.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .canonical_payload.build import (
    XSET_SHA256,
    XSUB_SHA256,
    ensure_canonical_sources as _build_canonical_sources,
)

COMMON_MARKERS = (
    "NESTSAR-HOPE-ATTENTION-LITE",
    "BUNDLE_B64",
    "EXPECTED_PARAMS = 2_381_028",
    "EXPECTED_LEAVES = 705",
    "ATTENTION_DIM = 64",
    "ATTENTION_HEADS = 4",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(path: Path, protocol: str, expected_sha: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    actual = _sha256(path)
    if actual != expected_sha:
        raise RuntimeError(
            f"Attention-Lite {protocol.upper()} integration SHA256 mismatch.\n"
            f"expected: {expected_sha}\n"
            f"actual:   {actual}"
        )

    text = path.read_text(encoding="utf-8")
    required = COMMON_MARKERS + (f"REAL NTU120 {protocol.upper()}",)
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(
            f"Attention-Lite {protocol.upper()} source missing marker(s): "
            + ", ".join(missing)
        )

    compile(text, str(path), "exec")


def ensure_canonical_sources(*, verbose: bool = False) -> dict[str, Path]:
    sources = _build_canonical_sources(verbose=verbose)

    xsub = Path(sources["xsub"]).resolve()
    xset = Path(sources["xset"]).resolve()
    _verify(xsub, "xsub", XSUB_SHA256)
    _verify(xset, "xset", XSET_SHA256)

    if verbose:
        print("=" * 108, flush=True)
        print("ATTENTION-LITE GITHUB CANONICAL INTEGRATION: PASS", flush=True)
        print(f"XSUB SHA256: {XSUB_SHA256}", flush=True)
        print(f"XSET SHA256: {XSET_SHA256}", flush=True)
        print("Architecture: T16 / D128 / Attention D64-H4-Dh16", flush=True)
        print("Guards: 2,381,028 params / 705 leaves", flush=True)
        print("=" * 108, flush=True)

    return {"xsub": xsub, "xset": xset}


if __name__ == "__main__":
    ensure_canonical_sources(verbose=True)

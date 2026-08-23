#!/usr/bin/env python3
"""Materialize exact validated Attention-Lite XSUB/XSET sources from GitHub.

This is the canonical integration path used by Kaggle.  It intentionally avoids the
older experimental gzip payload directories.  The committed LZMA/base64 XSUB payload
under ``canonical_payload/xsub`` is reconstructed and SHA256-verified by the original
payload builder.  XSET is then produced using the exact protocol-only replacements
between the two validated all-in-one trainers, with an independent SHA256 guard.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .canonical_payload import build as payload_build

XSUB_SHA256 = "e1080c4e02af96cf9dd0562415e73374d9d582ffa5e74c389ca3e47a05549aa6"
XSET_SHA256 = "8b6bffc91840055c84b0415bfd2c28cefc815d4a087c712c9e2d369c89541c07"

CANONICAL_DIR = Path(__file__).resolve().parent / "canonical"
XSUB_NAME = "NestSAR_HOPE_Attention_Lite_TPUv5e8_XSUB_E40_ALL_IN_ONE.py"
XSET_NAME = "NestSAR_HOPE_Attention_Lite_TPUv5e8_XSET_E40_ALL_IN_ONE.py"

COMMON_MARKERS = (
    "NESTSAR-HOPE-ATTENTION-LITE",
    "BUNDLE_B64",
    "EXPECTED_PARAMS = 2_381_028",
    "EXPECTED_LEAVES = 705",
    "ATTENTION_DIM = 64",
    "ATTENTION_HEADS = 4",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate(text: str, protocol: str, expected_sha: str) -> bytes:
    data = text.encode("utf-8")
    actual = _sha256_bytes(data)
    if actual != expected_sha:
        raise RuntimeError(
            f"Attention-Lite {protocol.upper()} SHA256 mismatch.\n"
            f"expected: {expected_sha}\n"
            f"actual:   {actual}"
        )

    required = COMMON_MARKERS + (f"REAL NTU120 {protocol.upper()}",)
    missing = [m for m in required if m not in text]
    if missing:
        raise RuntimeError(
            f"Attention-Lite {protocol.upper()} source missing markers: "
            + ", ".join(missing)
        )

    compile(text, f"<Attention-Lite-{protocol.upper()}>", "exec")
    return data


def _derive_exact_xset(xsub_text: str) -> str:
    """Apply only the exact XSUB->XSET textual differences.

    The legacy payload builder additionally removed the final separator print.  The
    validated XSET all-in-one source keeps that separator, so this integration does
    not perform that truncation.
    """
    text = xsub_text
    for old, new in payload_build.XSET_REPLACEMENTS:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(
                "Attention-Lite XSET derivation guard failed.\n"
                f"Expected exactly one occurrence of: {old!r}\n"
                f"Found: {count}"
            )
        text = text.replace(old, new, 1)
    return text


def ensure_canonical_sources(*, verbose: bool = False) -> dict[str, Path]:
    # This function already verifies the exact validated XSUB SHA256 before returning.
    xsub_bytes = payload_build._read_xsub_payload()
    if _sha256_bytes(xsub_bytes) != XSUB_SHA256:
        raise RuntimeError("Committed XSUB payload is not the validated source")

    try:
        xsub_text = xsub_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("Committed XSUB payload is not UTF-8") from exc

    _validate(xsub_text, "xsub", XSUB_SHA256)

    xset_text = _derive_exact_xset(xsub_text)
    xset_bytes = _validate(xset_text, "xset", XSET_SHA256)

    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    xsub_path = CANONICAL_DIR / XSUB_NAME
    xset_path = CANONICAL_DIR / XSET_NAME
    xsub_path.write_bytes(xsub_bytes)
    xset_path.write_bytes(xset_bytes)

    # Read-back validation catches partial/corrupt writes before TPU time is spent.
    _validate(xsub_path.read_text(encoding="utf-8"), "xsub", XSUB_SHA256)
    _validate(xset_path.read_text(encoding="utf-8"), "xset", XSET_SHA256)

    if verbose:
        print("=" * 108, flush=True)
        print("ATTENTION-LITE GITHUB CANONICAL INTEGRATION: PASS", flush=True)
        print(f"XSUB SHA256: {XSUB_SHA256}", flush=True)
        print(f"XSET SHA256: {XSET_SHA256}", flush=True)
        print(f"XSUB source: {xsub_path}", flush=True)
        print(f"XSET source: {xset_path}", flush=True)
        print("Architecture: T16 / D128 / Attention D64-H4-Dh16", flush=True)
        print("Guards: 2,381,028 params / 705 leaves", flush=True)
        print("=" * 108, flush=True)

    return {"xsub": xsub_path.resolve(), "xset": xset_path.resolve()}


if __name__ == "__main__":
    ensure_canonical_sources(verbose=True)

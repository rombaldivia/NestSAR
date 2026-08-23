#!/usr/bin/env python3
"""Materialize the exact validated Attention-Lite XSUB/XSET paper sources.

The repository stores one compressed canonical XSUB source in small base64 chunks.
XSUB is reconstructed byte-for-byte and SHA256-verified. XSET is then derived from
that exact source using the protocol-only replacements recovered from the validated
notebook, and is independently SHA256-verified. Both files are syntax-compiled before
being exposed to the trainer.
"""
from __future__ import annotations

import base64
import hashlib
import lzma
from pathlib import Path

XSUB_SHA256 = "e1080c4e02af96cf9dd0562415e73374d9d582ffa5e74c389ca3e47a05549aa6"
XSET_SHA256 = "8a446753a85bb8edba9c4c033cb49e7a9ebbbb317832c533d0f514b90720af0b"
NUM_XSUB_PARTS = 8

HERE = Path(__file__).resolve().parent
XSUB_PAYLOAD_DIR = HERE / "xsub"
CANONICAL_DIR = HERE.parent / "canonical"

XSUB_NAME = "NestSAR_HOPE_Attention_Lite_TPUv5e8_XSUB_E40_ALL_IN_ONE.py"
XSET_NAME = "NestSAR_HOPE_Attention_Lite_TPUv5e8_XSET_E40_ALL_IN_ONE.py"

# Exact protocol-only edits between the validated XSUB and XSET sources.
# Every replacement is guarded by an expected occurrence count of exactly one.
XSET_REPLACEMENTS = (
    (
        "# NESTSAR-HOPE-ATTENTION-LITE D128 — TPU v5e-8 — XSUB — E40\n",
        "# NESTSAR-HOPE-ATTENTION-LITE D128 — TPU v5e-8 — XSET — E40\n",
    ),
    (
        "# with TOTAL_STEPS=19,700. Define placeholders only so the namespace is complete.\n",
        "# with the protocol-specific E40 TOTAL_STEPS. Define placeholders only so the namespace is complete.\n",
    ),
    (
        "# REAL NTU120 XSUB — 40 EPOCH FULL RUN\n",
        "# REAL NTU120 XSET — 40 EPOCH FULL RUN\n",
    ),
    (
        "#   official XSUB      = 63,026 train / 50,919 val\n",
        "#   official XSET      = 54,468 train / 59,477 val\n",
    ),
    (
        "TRAIN_EXPECTED = 63_026\nVAL_EXPECTED = 50_919\n",
        "TRAIN_EXPECTED = 54_468\nVAL_EXPECTED = 59_477\n",
    ),
    (
        "assert MICROSTEPS_PER_EPOCH == 1970\nassert TOTAL_MICROSTEPS == 78_800\nassert TOTAL_STEPS == 19_700\n",
        "assert MICROSTEPS_PER_EPOCH == 1703\nassert TOTAL_MICROSTEPS == 68_120\nassert TOTAL_STEPS == 17_030\n",
    ),
    (
        "# the real E1->E40 run uses the correct 19,700-step schedule.\n",
        "# the real XSET E1->E40 run uses the correct 17,030-step schedule.\n",
    ),
    (
        '    f"40-epoch optimizer schedules rebuilt: TOTAL_STEPS={TOTAL_STEPS:,}"\n',
        '    f"XSET 40-epoch optimizer schedules rebuilt: TOTAL_STEPS={TOTAL_STEPS:,}"\n',
    ),
    (
        '    "NestSAR_HOPE_ATTENTION_LITE_D128_XSUB_E40"\n',
        '    "NestSAR_HOPE_ATTENTION_LITE_D128_XSET_E40"\n',
    ),
    (
        '    "REAL NTU120 XSUB 40-EPOCH FULL RUN"\n',
        '    "REAL NTU120 XSET 40-EPOCH FULL RUN"\n',
    ),
    (
        '    protocol="xsub",\n',
        '    protocol="xset",\n',
    ),
    (
        '        f"XSUB train mismatch: "\n',
        '        f"XSET train mismatch: "\n',
    ),
    (
        '        f"XSUB val mismatch: "\n',
        '        f"XSET val mismatch: "\n',
    ),
    (
        'print("OFFICIAL NTU120 XSUB")\n',
        'print("OFFICIAL NTU120 XSET")\n',
    ),
    (
        '        "xsub",\n',
        '        "xset",\n',
    ),
    (
        '    "XSUB 40-EPOCH FULL RUN COMPLETE"\n',
        '    "XSET 40-EPOCH FULL RUN COMPLETE"\n',
    ),
)

XSUB_TRAILING_ONLY = '\nprint("=" * 132)'


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_xsub_payload() -> bytes:
    parts = [XSUB_PAYLOAD_DIR / f"part_{i:02d}.b64" for i in range(NUM_XSUB_PARTS)]
    missing = [str(p) for p in parts if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing committed Attention-Lite XSUB payload part(s):\n  - "
            + "\n  - ".join(missing)
        )

    encoded = "".join(p.read_text(encoding="ascii").strip() for p in parts)
    try:
        compressed = base64.b64decode(encoded, validate=True)
        source_bytes = lzma.decompress(compressed)
    except Exception as exc:
        raise RuntimeError("Could not decode/decompress committed Attention-Lite XSUB payload") from exc

    digest = _sha256(source_bytes)
    if digest != XSUB_SHA256:
        raise RuntimeError(
            "Attention-Lite XSUB canonical SHA256 mismatch.\n"
            f"expected: {XSUB_SHA256}\n"
            f"actual:   {digest}"
        )
    return source_bytes


def _derive_xset(xsub_text: str) -> str:
    text = xsub_text
    for old, new in XSET_REPLACEMENTS:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(
                "Attention-Lite XSET derivation guard failed.\n"
                f"Expected exactly one occurrence of: {old!r}\n"
                f"Found: {count}"
            )
        text = text.replace(old, new, 1)

    if not text.endswith(XSUB_TRAILING_ONLY):
        raise RuntimeError(
            "Attention-Lite XSET trailing-line derivation guard failed: "
            "validated XSUB source does not end with the expected separator print"
        )
    text = text[:-len(XSUB_TRAILING_ONLY)]
    return text


def _validate_source(text: str, *, protocol: str, expected_sha256: str) -> bytes:
    data = text.encode("utf-8")
    digest = _sha256(data)
    if digest != expected_sha256:
        raise RuntimeError(
            f"Attention-Lite {protocol.upper()} canonical SHA256 mismatch.\n"
            f"expected: {expected_sha256}\n"
            f"actual:   {digest}"
        )

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
            f"Attention-Lite {protocol.upper()} source missing marker(s): "
            + ", ".join(missing)
        )

    try:
        compile(text, f"<Attention-Lite-{protocol.upper()}>", "exec")
    except SyntaxError as exc:
        raise RuntimeError(
            f"Attention-Lite {protocol.upper()} canonical source does not compile: {exc}"
        ) from exc
    return data


def ensure_canonical_sources(*, verbose: bool = False) -> dict[str, Path]:
    """Build and verify exact XSUB/XSET sources from the committed payload."""
    xsub_bytes = _read_xsub_payload()
    try:
        xsub_text = xsub_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("Committed Attention-Lite XSUB payload is not UTF-8") from exc

    _validate_source(
        xsub_text,
        protocol="xsub",
        expected_sha256=XSUB_SHA256,
    )

    xset_text = _derive_xset(xsub_text)
    _validate_source(
        xset_text,
        protocol="xset",
        expected_sha256=XSET_SHA256,
    )

    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    xsub_path = CANONICAL_DIR / XSUB_NAME
    xset_path = CANONICAL_DIR / XSET_NAME

    if not xsub_path.is_file() or xsub_path.read_bytes() != xsub_bytes:
        xsub_path.write_bytes(xsub_bytes)

    xset_bytes = xset_text.encode("utf-8")
    if not xset_path.is_file() or xset_path.read_bytes() != xset_bytes:
        xset_path.write_bytes(xset_bytes)

    if verbose:
        print("=" * 108, flush=True)
        print("ATTENTION-LITE CANONICAL PAYLOAD: PASS", flush=True)
        print(f"XSUB SHA256: {XSUB_SHA256}", flush=True)
        print(f"XSET SHA256: {XSET_SHA256}", flush=True)
        print(f"XSUB source: {xsub_path}", flush=True)
        print(f"XSET source: {xset_path}", flush=True)
        print("=" * 108, flush=True)

    return {
        "xsub": xsub_path.resolve(),
        "xset": xset_path.resolve(),
    }


if __name__ == "__main__":
    ensure_canonical_sources(verbose=True)

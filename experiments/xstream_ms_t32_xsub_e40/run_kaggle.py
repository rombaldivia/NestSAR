#!/usr/bin/env python3
from pathlib import Path
import hashlib
import os
import sys

# Hash of the normalized source payload currently committed on this branch.
EXPECTED_SHA256 = "9185ad324e490de1f12f49feea5eb94d6a514fee05a06d7ab724bd32cb040f72"
NUM_PARTS = 11

# The original chunk upload introduced one extra trailing character in three
# chunks. These are the intended UTF-8 byte lengths after boundary repair.
EXPECTED_PART_BYTES = [
    18010,
    18000,
    18000,
    18000,
    18024,
    18002,
    18002,
    18006,
    18004,
    18000,
    11158,
]

HERE = Path(__file__).resolve().parent
CHUNK_DIR = HERE / "chunks"
PARTS = [CHUNK_DIR / f"part_{i:02d}.txt" for i in range(NUM_PARTS)]

missing = [p.name for p in PARTS if not p.is_file()]
if missing:
    raise FileNotFoundError(
        "XStream source chunks are missing: " + ", ".join(missing)
    )

normalized_parts = []
for i, path in enumerate(PARTS):
    text = path.read_text(encoding="utf-8")
    expected_bytes = EXPECTED_PART_BYTES[i]
    actual_bytes = len(text.encode("utf-8"))

    if actual_bytes == expected_bytes:
        normalized = text
    elif actual_bytes == expected_bytes + 1:
        candidate = text[:-1]
        if len(candidate.encode("utf-8")) != expected_bytes:
            raise RuntimeError(
                f"Chunk {i:02d} has an unexpected +1 byte corruption: "
                f"{actual_bytes} bytes"
            )
        normalized = candidate
        print(
            f"Normalized chunk part_{i:02d}.txt: "
            f"{actual_bytes} -> {expected_bytes} bytes"
        )
    else:
        raise RuntimeError(
            f"Chunk part_{i:02d}.txt size mismatch: "
            f"expected {expected_bytes}, got {actual_bytes}"
        )

    normalized_parts.append(normalized)

source = "".join(normalized_parts)
source_bytes = source.encode("utf-8")
digest = hashlib.sha256(source_bytes).hexdigest()

print("=" * 100)
print("NESTSAR XSTREAM-MS T32 XSUB E40 — KAGGLE LAUNCHER")
print("=" * 100)
print("Chunks:       ", NUM_PARTS)
print("Source bytes: ", len(source_bytes))
print("SHA256:       ", digest)

if digest != EXPECTED_SHA256:
    raise RuntimeError(
        "Assembled XStream source checksum mismatch.\n"
        f"expected: {EXPECTED_SHA256}\n"
        f"actual:   {digest}"
    )

# Static syntax check before starting JAX/TPU initialization.
try:
    compile(source, "<NestSAR-XStream-MS-T32>", "exec")
except SyntaxError as exc:
    raise RuntimeError(f"Assembled XStream source does not compile: {exc}") from exc

# Architecture/training markers that must be present in this exact experiment.
REQUIRED_MARKERS = [
    "class CrossStreamMultiScaleHint",
    "class NestSARHOPEXStreamMST32",
    "CROSS_DIM = 64",
    "CROSS_HEADS = 4",
    "EXPECTED_PARAMS = 2_428_764",
    "EXPECTED_LEAVES = 724",
    "TRAIN_EXPECTED = 63_026",
    "VAL_EXPECTED = 50_919",
    'protocol="xsub"',
    "frames=32",
    "cross_stream_mixer",
]

missing_markers = [marker for marker in REQUIRED_MARKERS if marker not in source]
if missing_markers:
    raise RuntimeError(
        "Assembled source is missing required XStream experiment markers:\n  - "
        + "\n  - ".join(missing_markers)
    )

print("Syntax audit:  PASS")
print("Marker audit:  PASS")

# This production launcher is intentionally Kaggle-only.
if Path("/content").exists() and not Path("/kaggle").exists():
    raise RuntimeError(
        "This launcher is locked to Kaggle TPU v5e-8, but this runtime is Google Colab "
        "(/content exists and /kaggle does not). Open a Kaggle notebook, enable TPU v5e-8, "
        "attach ntu120_3danno.pkl, and run the same two launcher cells there."
    )

if not Path("/kaggle/working").exists():
    raise RuntimeError("/kaggle/working was not found; this does not look like a Kaggle runtime.")

out_root = Path("/kaggle/working")
target = out_root / "NestSAR_HOPE_XStream_MS_T32_D128_XSUB_E40_ALL_IN_ONE.py"
target.write_text(source, encoding="utf-8")

print("Source audit:  PASS")
print("Assembled:    ", target)
print("Launching in a fresh Python process...")
print("=" * 100, flush=True)

# Do not import JAX here. The assembled experiment configures XLA before importing JAX,
# then performs parameter, optimizer-tier, finite, TPU-count and gradient-sharding guards.
os.execv(sys.executable, [sys.executable, "-u", str(target)])

#!/usr/bin/env python3
from pathlib import Path
import hashlib
import os
import sys

EXPECTED_SHA256 = "f1b3e79e7499adcc09bec2d6d87ed3f68b48d2c9c18813767b053c3179f2dbb1"
NUM_PARTS = 11

# Exact UTF-8 byte lengths of the source chunks produced from the validated
# all-in-one experiment file. Three GitHub chunk writes picked up one extra
# boundary character; normalize only that one-byte boundary difference, then
# require the original full-source SHA256 before execution.
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
        # The only accepted repair is removing exactly one trailing character
        # whose removal restores the validated byte length.
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
digest = hashlib.sha256(source.encode("utf-8")).hexdigest()

print("=" * 100)
print("NESTSAR XSTREAM-MS T32 XSUB E40 — KAGGLE LAUNCHER")
print("=" * 100)
print("Chunks:       ", NUM_PARTS)
print("Source bytes: ", len(source.encode("utf-8")))
print("SHA256:       ", digest)

if digest != EXPECTED_SHA256:
    raise RuntimeError(
        "Assembled XStream source checksum mismatch after boundary normalization.\n"
        f"expected: {EXPECTED_SHA256}\n"
        f"actual:   {digest}"
    )

if Path("/content").exists() and not Path("/kaggle").exists():
    raise RuntimeError(
        "This launcher is locked to Kaggle TPU v5e-8, but this runtime looks like Google Colab "
        "(/content exists and /kaggle does not). Open a Kaggle notebook, enable TPU v5e-8, "
        "attach ntu120_3danno.pkl, then run this launcher there."
    )

out_root = Path("/kaggle/working") if Path("/kaggle/working").exists() else HERE
target = out_root / "NestSAR_HOPE_XStream_MS_T32_D128_XSUB_E40_ALL_IN_ONE.py"
target.write_text(source, encoding="utf-8")

print("Source audit:  PASS")
print("Assembled:    ", target)
print("Launching in a fresh Python process...")
print("=" * 100, flush=True)

# Important: do not import JAX in this launcher. The assembled experiment sets its XLA
# environment variables before importing JAX, then performs all parameter/TPU/gradient guards.
os.execv(sys.executable, [sys.executable, "-u", str(target)])

#!/usr/bin/env python3
from pathlib import Path
import hashlib
import os
import sys

EXPECTED_SHA256 = "f1b3e79e7499adcc09bec2d6d87ed3f68b48d2c9c18813767b053c3179f2dbb1"
NUM_PARTS = 11

HERE = Path(__file__).resolve().parent
CHUNK_DIR = HERE / "chunks"
PARTS = [CHUNK_DIR / f"part_{i:02d}.txt" for i in range(NUM_PARTS)]

missing = [p.name for p in PARTS if not p.is_file()]
if missing:
    raise FileNotFoundError(
        "XStream source chunks are missing: " + ", ".join(missing)
    )

source = "".join(p.read_text(encoding="utf-8") for p in PARTS)
digest = hashlib.sha256(source.encode("utf-8")).hexdigest()

print("=" * 100)
print("NESTSAR XSTREAM-MS T32 XSUB E40 — KAGGLE LAUNCHER")
print("=" * 100)
print("Chunks:       ", NUM_PARTS)
print("Source bytes: ", len(source.encode("utf-8")))
print("SHA256:       ", digest)

if digest != EXPECTED_SHA256:
    raise RuntimeError(
        "Assembled XStream source checksum mismatch.\n"
        f"expected: {EXPECTED_SHA256}\n"
        f"actual:   {digest}"
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

#!/usr/bin/env python3
from pathlib import Path
import hashlib
import os
import re
import shutil
import sys
import urllib.request

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

# This production launcher is intentionally Kaggle-only.
if Path("/content").exists() and not Path("/kaggle").exists():
    raise RuntimeError(
        "This launcher is locked to Kaggle TPU, but this runtime is Google Colab "
        "(/content exists and /kaggle does not). Open a Kaggle notebook and enable TPU."
    )

if not Path("/kaggle/working").exists():
    raise RuntimeError("/kaggle/working was not found; this does not look like a Kaggle runtime.")

# ------------------------------------------------------------------------------------------
# Resolve NTU120 robustly across Kaggle accounts.
# Priority:
#   1) NESTSAR_DATASET environment override
#   2) any attached Kaggle input containing ntu120_3danno.pkl
#   3) /kaggle/working cache
#   4) OpenMMLab download when Kaggle internet is enabled
# ------------------------------------------------------------------------------------------
DATASET_NAME = "ntu120_3danno.pkl"
DATASET_URL = "https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl"

candidates = []
manual = os.environ.get("NESTSAR_DATASET", "").strip()
if manual:
    candidates.append(Path(manual))

input_root = Path("/kaggle/input")
if input_root.exists():
    candidates.extend(input_root.rglob(DATASET_NAME))

working_dataset = Path("/kaggle/working") / DATASET_NAME
candidates.append(working_dataset)

dataset = next((p for p in candidates if p.is_file()), None)

if dataset is None:
    print("NTU120 dataset was not attached; attempting OpenMMLab download...")
    tmp = working_dataset.with_suffix(".pkl.part")
    try:
        req = urllib.request.Request(
            DATASET_URL,
            headers={"User-Agent": "NestSAR-Kaggle/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as response, tmp.open("wb") as f:
            shutil.copyfileobj(response, f, length=16 * 1024 * 1024)
        tmp.replace(working_dataset)
        dataset = working_dataset
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        found_pkls = []
        if input_root.exists():
            found_pkls = [str(p) for p in input_root.rglob("*.pkl")][:20]
        raise FileNotFoundError(
            "ntu120_3danno.pkl was not found and automatic OpenMMLab download failed.\n"
            "Attach the NTU120 dataset to this Kaggle notebook, or set NESTSAR_DATASET to its path.\n"
            f"Download error: {type(exc).__name__}: {exc}\n"
            "Visible .pkl files:\n  - " + ("\n  - ".join(found_pkls) if found_pkls else "NONE")
        ) from exc

print("Dataset:      ", dataset)
print("Dataset bytes:", f"{dataset.stat().st_size:,}")
os.environ["NESTSAR_DATASET"] = str(dataset)

# ------------------------------------------------------------------------------------------
# Deterministic runtime portability patches.
# The committed source payload is verified ABOVE before any patch is applied.
# ------------------------------------------------------------------------------------------
# 1) Make the resolved dataset an explicit first candidate without changing model/training code.
dataset_anchor = "DATASET_CANDIDATES = ["
if source.count(dataset_anchor) != 1:
    raise RuntimeError(
        f"Expected exactly one DATASET_CANDIDATES block, found {source.count(dataset_anchor)}"
    )
source = source.replace(
    dataset_anchor,
    'DATASET_CANDIDATES = [\n\n    Path(os.environ["NESTSAR_DATASET"]),',
    1,
)

# 2) Newer Kaggle TPU runtimes may expose the v5e slice as fewer logical JAX devices.
# Keep TPU-only execution, but require only that the global batch shards evenly.
device_guard = re.compile(
    r'if\s*\(\s*len\(\s*DEVICES\s*\)\s*!=\s*8\s*\)\s*:\s*'
    r'raise RuntimeError\(\s*f"Expected 8 TPU devices; found \{len\(DEVICES\)\}"\s*\)',
    re.MULTILINE,
)
source, device_patch_count = device_guard.subn(
    'if len(DEVICES) < 1 or (GLOBAL_BATCH % len(DEVICES)) != 0:\n\n'
    '    raise RuntimeError(\n'
    '        f"Global batch {GLOBAL_BATCH} must be divisible by visible TPU device count {len(DEVICES)}"\n'
    '    )',
    source,
    count=1,
)
if device_patch_count != 1:
    raise RuntimeError(
        f"TPU device guard portability patch failed; matches={device_patch_count}"
    )

# Keep result metadata accurate when the runtime exposes a non-8 logical topology.
local_batch_pattern = re.compile(
    r'("local_batch"\s*:\s*GLOBAL_BATCH\s*//\s*)8(\s*,)',
    re.MULTILINE,
)
source, local_batch_patch_count = local_batch_pattern.subn(
    r'\1len(DEVICES)\2',
    source,
    count=1,
)
if local_batch_patch_count not in (0, 1):
    raise RuntimeError("Unexpected local_batch metadata patch count")

runtime_bytes = source.encode("utf-8")
runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()

# Static syntax check before starting JAX/TPU initialization.
try:
    compile(source, "<NestSAR-XStream-MS-T32>", "exec")
except SyntaxError as exc:
    raise RuntimeError(f"Runtime-patched XStream source does not compile: {exc}") from exc

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
        "Runtime source is missing required XStream experiment markers:\n  - "
        + "\n  - ".join(missing_markers)
    )

print("Committed source audit: PASS")
print("Runtime syntax audit:   PASS")
print("Marker audit:           PASS")
print("Runtime SHA256:         ", runtime_sha)

out_root = Path("/kaggle/working")
target = out_root / "NestSAR_HOPE_XStream_MS_T32_D128_XSUB_E40_ALL_IN_ONE.py"
target.write_text(source, encoding="utf-8")

print("Assembled:             ", target)
print("Launching in a fresh Python process...")
print("=" * 100, flush=True)

# Do not import JAX here. The assembled experiment configures XLA before importing JAX,
# then performs parameter, optimizer-tier, finite and gradient-sharding guards.
os.execv(sys.executable, [sys.executable, "-u", str(target)])

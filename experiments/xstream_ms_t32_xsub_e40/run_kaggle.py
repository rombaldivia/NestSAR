#!/usr/bin/env python3
from pathlib import Path
import hashlib
import os
import shutil
import sys
import urllib.request

# Hash of the normalized source payload currently committed on this branch.
EXPECTED_SHA256 = "9185ad324e490de1f12f49feea5eb94d6a514fee05a06d7ab724bd32cb040f72"
NUM_PARTS = 11

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
                f"Chunk {i:02d} has an unexpected +1 byte corruption: {actual_bytes} bytes"
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
print("NESTSAR XSTREAM-MS T32 XSUB E40 — COLAB/KAGGLE LAUNCHER")
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

# ------------------------------------------------------------------------------------------
# Runtime platform
# IMPORTANT: detect Colab first. A Colab session can contain a stale /kaggle directory
# created by an earlier launcher attempt; /content is the authoritative Colab marker.
# ------------------------------------------------------------------------------------------
if Path("/content").exists():
    PLATFORM = "colab"
    RUNTIME_ROOT = Path("/content/nestsar_xstream_runtime")
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    SEARCH_ROOTS = [
        Path("/content"),
        Path("/content/drive/MyDrive"),
        RUNTIME_ROOT,
    ]
elif Path("/kaggle/working").exists():
    PLATFORM = "kaggle"
    RUNTIME_ROOT = Path("/kaggle/working")
    SEARCH_ROOTS = [Path("/kaggle/input"), RUNTIME_ROOT]
else:
    raise RuntimeError(
        "Unsupported runtime: expected Google Colab (/content) or Kaggle (/kaggle/working)."
    )

print("Platform:     ", PLATFORM)
print("Runtime root: ", RUNTIME_ROOT)

# ------------------------------------------------------------------------------------------
# Resolve NTU120 robustly.
# Priority:
#   1) NESTSAR_DATASET environment override
#   2) common local/attached paths
#   3) recursive search for the exact filename
#   4) OpenMMLab download
# ------------------------------------------------------------------------------------------
DATASET_NAME = "ntu120_3danno.pkl"
DATASET_URL = "https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl"

candidates = []
manual = os.environ.get("NESTSAR_DATASET", "").strip()
if manual:
    candidates.append(Path(manual))

if PLATFORM == "colab":
    candidates += [
        Path("/content") / DATASET_NAME,
        Path("/content/drive/MyDrive") / DATASET_NAME,
        RUNTIME_ROOT / DATASET_NAME,
        # Reuse a previously downloaded copy from a stale Kaggle-style directory if present.
        Path("/kaggle/working") / DATASET_NAME,
    ]
else:
    candidates.append(RUNTIME_ROOT / DATASET_NAME)

for root in SEARCH_ROOTS:
    if root.exists():
        try:
            candidates.extend(root.rglob(DATASET_NAME))
        except Exception:
            pass

dataset = next((p for p in candidates if p.is_file()), None)
working_dataset = RUNTIME_ROOT / DATASET_NAME

if dataset is None:
    print("NTU120 dataset was not found locally; attempting OpenMMLab download...")
    tmp = working_dataset.with_suffix(".pkl.part")
    try:
        req = urllib.request.Request(
            DATASET_URL,
            headers={"User-Agent": "NestSAR-Colab-Kaggle/1.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as f:
            shutil.copyfileobj(response, f, length=16 * 1024 * 1024)
        tmp.replace(working_dataset)
        dataset = working_dataset
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise FileNotFoundError(
            "ntu120_3danno.pkl was not found and automatic OpenMMLab download failed.\n"
            "Upload/mount the dataset, or set NESTSAR_DATASET to its full path.\n"
            f"Download error: {type(exc).__name__}: {exc}"
        ) from exc

print("Dataset:      ", dataset)
print("Dataset bytes:", f"{dataset.stat().st_size:,}")
os.environ["NESTSAR_DATASET"] = str(dataset)

# ------------------------------------------------------------------------------------------
# Deterministic runtime portability patches.
# The committed source payload is verified ABOVE before any patch is applied.
# ------------------------------------------------------------------------------------------
# 1) Make the resolved dataset an explicit first candidate.
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

# 2) Redirect every hard-coded Kaggle working path to the active runtime root.
source = source.replace("/kaggle/working", str(RUNTIME_ROOT))

# 3) TPU runtimes may expose different logical device counts. The committed all-in-one
# source formats the old 8-device guard across several lines, so patch that exact block
# rather than relying on a fragile regex.
old_device_guard = '''if len(
    DEVICES
) != 8:

    raise RuntimeError(
        f"Expected 8 TPU devices; found {len(DEVICES)}"
    )'''
new_device_guard = '''if len(DEVICES) < 1 or (GLOBAL_BATCH % len(DEVICES)) != 0:

    raise RuntimeError(
        f"Global batch {GLOBAL_BATCH} must be divisible by visible TPU device count {len(DEVICES)}"
    )'''

device_guard_count = source.count(old_device_guard)
if device_guard_count != 1:
    raise RuntimeError(
        f"TPU device guard portability patch failed; expected exactly 1 block, found {device_guard_count}"
    )
source = source.replace(old_device_guard, new_device_guard, 1)

# Keep result metadata accurate for non-8 logical topologies. The source also formats
# this value over several lines, so patch the exact block.
old_local_batch = '''"local_batch":
        GLOBAL_BATCH
        //
        8,'''
new_local_batch = '''"local_batch":
        GLOBAL_BATCH
        //
        len(DEVICES),'''

local_batch_count = source.count(old_local_batch)
if local_batch_count == 1:
    source = source.replace(old_local_batch, new_local_batch, 1)
elif local_batch_count != 0:
    raise RuntimeError(
        f"Unexpected local_batch metadata block count: {local_batch_count}"
    )

runtime_bytes = source.encode("utf-8")
runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()

try:
    compile(source, "<NestSAR-XStream-MS-T32>", "exec")
except SyntaxError as exc:
    raise RuntimeError(f"Runtime-patched XStream source does not compile: {exc}") from exc

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
print("Device guard patch:     PASS")
print("Runtime syntax audit:   PASS")
print("Marker audit:           PASS")
print("Runtime SHA256:         ", runtime_sha)

target = RUNTIME_ROOT / "NestSAR_HOPE_XStream_MS_T32_D128_XSUB_E40_ALL_IN_ONE.py"
target.write_text(source, encoding="utf-8")

print("Assembled:             ", target)
print("Launching in a fresh Python process...")
print("=" * 100, flush=True)

# Do not import JAX here. The assembled experiment configures XLA before importing JAX,
# then performs parameter, optimizer-tier, finite and gradient-sharding guards.
os.execv(sys.executable, [sys.executable, "-u", str(target)])

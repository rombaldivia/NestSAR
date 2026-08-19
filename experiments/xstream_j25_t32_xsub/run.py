#!/usr/bin/env python3
from __future__ import annotations

import base64, gzip, hashlib, math, os, py_compile, shutil, subprocess, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAYLOAD = HERE / "payload"
EXPECTED_SHA = "140ac4b532d81f91fa43d34f2fd06edf68cfbbbe85842c2554fce713bb3cb22a"
EXPECTED_BYTES = 197552
TRAIN_EXPECTED = 63_026
VAL_EXPECTED = 50_919
EFFECTIVE_BATCH = 128
EPOCHS = int(os.environ.get("NESTSAR_EPOCHS", "3"))
if EPOCHS < 1:
    raise ValueError("NESTSAR_EPOCHS must be >= 1")

# Platform selection: /content wins if both paths happen to exist.
if Path("/content").exists():
    PLATFORM = "colab"
    RUNTIME_ROOT = Path("/content/nestsar_j25_runtime")
elif Path("/kaggle/working").exists():
    PLATFORM = "kaggle"
    RUNTIME_ROOT = Path("/kaggle/working")
else:
    raise RuntimeError("Neither Colab (/content) nor Kaggle (/kaggle/working) was detected.")
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

# Reconstruct exact committed source from gzip+base64 ASCII chunks.
parts = sorted(PAYLOAD.glob("part_*.b64"))
if not parts:
    raise FileNotFoundError(f"No payload chunks found in {PAYLOAD}")
b64 = "".join(p.read_text(encoding="ascii").strip() for p in parts)
source_bytes = gzip.decompress(base64.b64decode(b64.encode("ascii")))
sha = hashlib.sha256(source_bytes).hexdigest()
if len(source_bytes) != EXPECTED_BYTES or sha != EXPECTED_SHA:
    raise RuntimeError(
        "Committed J25 source checksum mismatch.\n"
        f"expected bytes={EXPECTED_BYTES}, sha={EXPECTED_SHA}\n"
        f"actual   bytes={len(source_bytes)}, sha={sha}"
    )
source = source_bytes.decode("utf-8")

# Find/reuse NTU120.
def locate_dataset():
    explicit = os.environ.get("NESTSAR_DATASET")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    candidates = [
        RUNTIME_ROOT / "ntu120_3danno.pkl",
        Path("/content/nestsar_xstream_runtime/ntu120_3danno.pkl"),
        Path("/kaggle/working/ntu120_3danno.pkl"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    roots = [Path("/content/drive"), Path("/content"), Path("/kaggle/input")]
    for root in roots:
        if root.exists():
            try:
                hits = list(root.rglob("ntu120_3danno.pkl"))
            except Exception:
                hits = []
            if hits:
                return hits[0]
    return None

dataset = locate_dataset()
if dataset is None:
    dataset = RUNTIME_ROOT / "ntu120_3danno.pkl"
    url = "https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl"
    print("NTU120 not found locally; downloading official OpenMMLab file...")
    tmp = dataset.with_suffix(".download")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dataset)

# Detect TPU topology in a disposable process so the training process starts clean.
probe = subprocess.run(
    [sys.executable, "-c", "import jax; print(jax.default_backend()); print(len(jax.devices()))"],
    text=True, capture_output=True, check=True,
)
probe_lines = [x.strip() for x in probe.stdout.splitlines() if x.strip()]
if len(probe_lines) < 2:
    raise RuntimeError(f"Could not detect JAX TPU topology:\n{probe.stdout}\n{probe.stderr}")
backend = probe_lines[-2]
num_devices = int(probe_lines[-1])
if backend != "tpu":
    raise RuntimeError(f"This experiment requires TPU; detected backend={backend!r}")
if num_devices < 1 or EFFECTIVE_BATCH % num_devices:
    raise RuntimeError(f"Unsupported TPU topology: devices={num_devices}")

# One physical sample per visible TPU device. J25 internally expands each sample to 25 joint tracks.
global_batch = num_devices
grad_accum = EFFECTIVE_BATCH // global_batch
eval_batch = global_batch
microsteps = math.ceil(TRAIN_EXPECTED / global_batch)
total_microsteps = microsteps * EPOCHS
total_steps = math.ceil(total_microsteps / grad_accum)

# Runtime root + exact dataset.
source = source.replace('/kaggle/working', str(RUNTIME_ROOT))
source = source.replace(
    'DATASET_CANDIDATES = [',
    f'DATASET_CANDIDATES = [\n    Path(r"{dataset}"),',
    1,
)

# Patch the physical batch only; effective batch stays exactly 128.
config_old = """    batch_size=32,\n    grad_accum_steps=4,\n    eval_batch_size=32,"""
config_new = f"""    batch_size={global_batch},\n    grad_accum_steps={grad_accum},\n    eval_batch_size={eval_batch},"""
if source.count(config_old) != 1:
    raise RuntimeError(f"Config batch patch guard failed; matches={source.count(config_old)}")
source = source.replace(config_old, config_new, 1)

# Two GRAD_ACCUM constants: optimizer bootstrap + trainer.
if source.count("GRAD_ACCUM = 4") != 2:
    raise RuntimeError(f"GRAD_ACCUM patch guard failed; matches={source.count('GRAD_ACCUM = 4')}")
source = source.replace("GRAD_ACCUM = 4", f"GRAD_ACCUM = {grad_accum}")

for old, new in [
    ("EPOCHS = 40", f"EPOCHS = {EPOCHS}"),
    ("GLOBAL_BATCH = 32", f"GLOBAL_BATCH = {global_batch}"),
    ("EFFECTIVE_BATCH = 128", f"EFFECTIVE_BATCH = {EFFECTIVE_BATCH}"),
    ("assert ns.CFG.batch_size == 32", f"assert ns.CFG.batch_size == {global_batch}"),
    ("assert ns.CFG.grad_accum_steps == 4", f"assert ns.CFG.grad_accum_steps == {grad_accum}"),
    ("assert MICROSTEPS_PER_EPOCH == 1970", f"assert MICROSTEPS_PER_EPOCH == {microsteps}"),
    ("assert TOTAL_MICROSTEPS == 78_800", f"assert TOTAL_MICROSTEPS == {total_microsteps}"),
    ("assert TOTAL_STEPS == 19_700", f"assert TOTAL_STEPS == {total_steps}"),
]:
    if source.count(old) != 1:
        raise RuntimeError(f"Runtime patch guard failed for {old!r}; matches={source.count(old)}")
    source = source.replace(old, new, 1)

# Distinguish output folders for probe/full runs.
source = source.replace(
    'NestSAR_HOPE_XSTREAM_J25_T32_D128_XSUB_E40',
    f'NestSAR_HOPE_XSTREAM_J25_T32_D128_XSUB_E{EPOCHS:02d}',
)

# J25 backward fix: non-joint hint cotangents have shape
# [3, B, T, J, D].  The original XStream code used the old 4-D
# transpose for [3, B, T, D].  Preserve the joint axis and move the
# non-joint stream axis beside it: [B, T, 3, J, D].
transpose_old = "jnp.transpose(cn, (1, 2, 0, 3))"
transpose_new = "jnp.transpose(cn, (1, 2, 0, 3, 4))"
if source.count(transpose_old) != 1:
    raise RuntimeError(
        "J25 cotangent transpose patch guard failed; "
        f"matches={source.count(transpose_old)}"
    )
source = source.replace(transpose_old, transpose_new, 1)

# Marker audit: verify the science-defining invariants before execution.
markers = [
    "NUM_JOINT_TOKENS = 25",
    "class JointTokenFrontEnd",
    "class JointSpatialMixer",
    "class CrossStreamMultiScaleHint",
    "FIRST spatial collapse: only now, after L1/L2/L3/L4.",
    '"spatial_collapse_before_l4":',
    "False,",
    transpose_new,
]
missing = [m for m in markers if m not in source]
if missing:
    raise RuntimeError("J25 marker audit failed: " + repr(missing))

assembled = RUNTIME_ROOT / f"NestSAR_HOPE_XStream_J25_T32_D128_XSUB_E{EPOCHS:02d}_ALL_IN_ONE.py"
assembled.write_text(source, encoding="utf-8")
py_compile.compile(str(assembled), doraise=True)

print("=" * 108)
print("NESTSAR XSTREAM-J25 T32 — NO EARLY SPATIAL COLLAPSE")
print("=" * 108)
print("Platform:             ", PLATFORM)
print("Backend/devices:      ", backend, num_devices)
print("Dataset:              ", dataset)
print("Dataset bytes:        ", f"{dataset.stat().st_size:,}")
print("Source audit:          PASS", EXPECTED_SHA)
print("Runtime syntax audit:  PASS")
print("J25 backward transpose:PASS  [3,B,T,J,D] -> [B,T,3,J,D]")
print("Spatial tokens:        25 joints preserved through L1/L2/L3/L4")
print("Cross-stream:          joint-aligned S=4 attention at L1/L2/L3 hints")
print("Physical global batch:", global_batch)
print("Grad accumulation:     ", grad_accum)
print("Effective batch:       ", EFFECTIVE_BATCH)
print("Microsteps/epoch:      ", f"{microsteps:,}")
print("Epochs:                ", EPOCHS)
print("Optimizer steps:       ", f"{total_steps:,}")
print("Assembled:             ", assembled)
print("=" * 108)
print("Launching in a fresh Python process...")
sys.stdout.flush()

os.execv(sys.executable, [sys.executable, "-u", str(assembled)])
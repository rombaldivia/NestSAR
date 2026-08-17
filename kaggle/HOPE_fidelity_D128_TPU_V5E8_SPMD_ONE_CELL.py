# ======================================================================================
# NestSAR-HOPE-Fidelity D128 v1 — KAGGLE TPU v5e-8 — TRUE 8-CHIP SPMD
# ======================================================================================
#
# Uses all 8 TPU chips for normal train/eval batches:
#   parameters/optimizer/EMA : replicated across 8 chips
#   batch axis               : sharded across 8 chips
#   JAX execution            : NamedSharding + jax.jit SPMD
#
# TPU-efficient default:
#   global physical batch = 128
#   local batch per chip  = 16
#   grad accumulation     = 1
#   effective batch       = 128
#
# This keeps the nominal outer CMS sample windows at:
#   f1/fast        128
#   f2/medium      256
#   f4/slow        512
#   f8/consolidate 1024
#
# Required Kaggle inputs:
#   1) ntu120_3danno.pkl
#   2) NestSAR_HOPE_v4_1_SHORTL3FIX_Kaggle_16F_ONE_CELL.py
#
# Kaggle setting:
#   Accelerator -> TPU v5e-8
# ======================================================================================

from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

# --------------------------------------------------------------------------------------
# USER SWITCHES
# --------------------------------------------------------------------------------------
PROBE_ONLY = True          # True: 3 epochs | False: 40 epochs
FORCE_FRESH = True
SEED = 128

# TPU-optimized while preserving effective batch 128.
BATCH = 128
ACCUM = 1
EVAL_BATCH = 256

BRANCH = "hope-fidelity-d128-v1"
REPO_URL = "https://github.com/rombaldivia/NestSAR.git"
REPO = Path("/kaggle/working/NestSAR_GitHub")
ROOT = Path("/kaggle/working/NestSAR_HOPE_FIDELITY_TPU8")
OUT = ROOT / (
    "runs_hope_fidelity_d128_v1_tpu8_probe_xsub"
    if PROBE_ONLY
    else "runs_hope_fidelity_d128_v1_tpu8_e40_xsub"
)
LOG = ROOT / (
    "hope_fidelity_d128_v1_tpu8_probe.log"
    if PROBE_ONLY
    else "hope_fidelity_d128_v1_tpu8_e40.log"
)
EXPECTED_PARAMS = 2_083_236

print("=" * 120)
print("NESTSAR-HOPE-FIDELITY D128 v1 — TPU v5e-8 TRUE 8-CHIP SPMD")
print("=" * 120)

# --------------------------------------------------------------------------------------
# 0) VERIFY TPU IN A CHILD PROCESS
# --------------------------------------------------------------------------------------
check_env = os.environ.copy()
check_env["JAX_PLATFORMS"] = "tpu"
check_env["PYTHONUNBUFFERED"] = "1"
check_env["JAX_THREEFRY_PARTITIONABLE"] = "true"

check_code = r'''
import jax
print("JAX:", jax.__version__)
print("Backend:", jax.default_backend())
print("Device count:", jax.device_count())
print("Local device count:", jax.local_device_count())
for i, d in enumerate(jax.devices()):
    print(f"  TPU[{i}] = {d}")
assert jax.default_backend() == "tpu", "Select Kaggle Accelerator -> TPU v5e-8"
assert jax.device_count() >= 8, f"Expected 8 TPU chips, got {jax.device_count()}"
'''
check = subprocess.run(
    [sys.executable, "-u", "-c", check_code],
    env=check_env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(check.stdout)
if check.returncode != 0:
    raise RuntimeError("TPU v5e-8 detection failed")

subprocess.run(["bash", "-lc", "free -h || true"], check=False)
subprocess.run(["bash", "-lc", "df -h /kaggle/working || true"], check=False)

# --------------------------------------------------------------------------------------
# 1) FIND DATASET + EXACT v4.1 SELF-CONTAINED SOURCE
# --------------------------------------------------------------------------------------
datasets = sorted(Path("/kaggle/input").rglob("ntu120_3danno.pkl"))
if not datasets:
    datasets = sorted(Path("/kaggle/input").rglob("ntu120_3danno_clean.pkl"))
if not datasets:
    raise FileNotFoundError("Attach ntu120_3danno.pkl to the Kaggle notebook")
DATASET = datasets[0].resolve()

baseline_cells = sorted(
    Path("/kaggle/input").rglob(
        "NestSAR_HOPE_v4_1_SHORTL3FIX_Kaggle_16F_ONE_CELL.py"
    )
)
if not baseline_cells:
    raise FileNotFoundError(
        "Attach NestSAR_HOPE_v4_1_SHORTL3FIX_Kaggle_16F_ONE_CELL.py"
    )
BASELINE_CELL = baseline_cells[0].resolve()

print("Dataset:    ", DATASET)
print("v4.1 bundle:", BASELINE_CELL)

# --------------------------------------------------------------------------------------
# 2) EXTRACT EXACT AUDITED v4.1 SOURCE FILES ONLY
# --------------------------------------------------------------------------------------
print("\n[1/7] Extracting exact v4.1 source bundle...")
source = BASELINE_CELL.read_text(encoding="utf-8")
anchor = "audit_code = r'''"
if source.count(anchor) != 1:
    raise RuntimeError(
        f"Unexpected v4.1 one-cell: audit anchor count={source.count(anchor)}"
    )
source = source.replace(
    anchor,
    "raise SystemExit('__NESTSAR_EXTRACT_ONLY__')\n\n" + anchor,
    1,
)
try:
    exec(compile(source, str(BASELINE_CELL), "exec"), {"__name__": "__main__"})
except SystemExit as exc:
    if str(exc) != "__NESTSAR_EXTRACT_ONLY__":
        raise

candidates = sorted(
    Path("/kaggle/working").rglob("nestsar_hope_fullselfref_v3_3_shortl3fix.py")
)
if not candidates:
    raise RuntimeError("Could not find extracted exact v4.1 source tree")
V41_EXTRACTED = max(candidates, key=lambda p: p.stat().st_mtime).parent

if ROOT.exists() and FORCE_FRESH:
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True, exist_ok=True)

required = [
    "nestsar.py",
    "nestsar_fcjm_b2.py",
    "nestsar_selfweight_clean.py",
    "nestsar_sms_s1c_v2.py",
    "nestsar_m4_geom_h4.py",
    "nestsar_m4_regmask_ema_v3_safe.py",
    "nestsar_m4_geom_h4_sasm_l3statefix_v1.py",
    "nestsar_hope_fullselfref_v3_3_shortl3fix.py",
    "nestsar_hope_v4_1_cms_dmgd_l2_shortl3fix.py",
]
for name in required:
    src = V41_EXTRACTED / name
    if not src.is_file():
        raise FileNotFoundError(f"Exact v4.1 extraction missing: {src}")
    shutil.copy2(src, ROOT / name)
print("Exact v4.1 extraction: PASS")

# --------------------------------------------------------------------------------------
# 3) CLONE OUR GUARDED FIDELITY + TPU8 BRANCH
# --------------------------------------------------------------------------------------
print("\n[2/7] Pulling HOPE-fidelity + TPU8 SPMD code from GitHub...")
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO)],
    check=True,
)
BUILDER = REPO / "experiments/hope_fidelity_d128_v1/build_from_v41.py"
TPU8 = REPO / "experiments/hope_fidelity_d128_v1/nestsar_tpu_v5e8_spmd.py"
if not BUILDER.is_file():
    raise FileNotFoundError(BUILDER)
if not TPU8.is_file():
    raise FileNotFoundError(TPU8)

# --------------------------------------------------------------------------------------
# 4) GENERATE THE NEW 2,083,236-PARAM FIDELITY MODEL/TRAINER
# --------------------------------------------------------------------------------------
print("\n[3/7] Generating HOPE-fidelity D128 core/trainer...")
subprocess.run(
    [sys.executable, "-u", str(BUILDER), "--root", str(ROOT)],
    check=True,
)
CORE = ROOT / "nestsar_hope_fidelity_d128_v1_core.py"
TRAINER = ROOT / "nestsar_hope_fidelity_d128_v1_train.py"
subprocess.run(
    [sys.executable, "-m", "py_compile", str(CORE), str(TRAINER), str(TPU8)],
    check=True,
)
print("Generated sources + TPU8 wrapper compile: PASS")

# --------------------------------------------------------------------------------------
# 5) TPU CHILD ENVIRONMENT
# --------------------------------------------------------------------------------------
env = os.environ.copy()
for key in (
    "CUDA_VISIBLE_DEVICES",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
):
    env.pop(key, None)

env.update({
    "JAX_PLATFORMS": "tpu",
    "JAX_THREEFRY_PARTITIONABLE": "true",
    "PYTHONUNBUFFERED": "1",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "PYTHONPATH": (
        str(ROOT)
        + os.pathsep
        + str(TPU8.parent)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    ),
    "JAX_COMPILATION_CACHE_DIR": str(ROOT / ".jax_cache_tpu8"),
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "1",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",

    "NESTSAR_SELFREF_DGD_SCALE": "1.0",
    "NESTSAR_SELFREF_ETA_MAX": "0.05",
    "NESTSAR_SELFREF_RESIDUAL_BETA": "0.10",
    "NESTSAR_SELFREF_MATRIX_NORM_CAP": "4.0",
    "NESTSAR_SELFREF_VECTOR_NORM_CAP": "1.0",
    "NESTSAR_SELFREF_ALPHA_MIN": "0.90",
    "NESTSAR_SELFREF_ALPHA_MAX": "0.999",
    "NESTSAR_SHORT_L3_POSTWRITE_BLEND": "1.0",

    "NESTSAR_CMS_PERIOD_L1": "1",
    "NESTSAR_CMS_PERIOD_L2": "2",
    "NESTSAR_CMS_PERIOD_L3": "4",
    "NESTSAR_CMS_PERIOD_L4": "8",

    "NESTSAR_DMGD_MOMENTUM": "0.90",
    "NESTSAR_DMGD_MEMORY_LR": "0.01",
    "NESTSAR_DMGD_MIX": "0.10",
    "NESTSAR_DMGD_PROJECTION_CAP": "2.0",
})
(ROOT / ".jax_cache_tpu8").mkdir(exist_ok=True)

# --------------------------------------------------------------------------------------
# 6) SHARDING AUDIT: PROVE B128 -> 8 x B16
# --------------------------------------------------------------------------------------
print("\n[4/7] Auditing true 8-chip batch sharding...")
sharding_code = rf'''
import dataclasses
import numpy as np
import jax
import nestsar as ns
ns.CFG = dataclasses.replace(
    ns.CFG,
    frames=16, persons=2, joints=25, coords=3, num_classes=120,
    model_dim=128, memory_dim=64, controller_rank=32,
    frame_blocks=2, chunk_blocks=2, clip_blocks=2, controller_blocks=2,
    chunk_size=4, clip_size=8, dropout=0.22,
    batch_size={BATCH}, grad_accum_steps={ACCUM}, eval_batch_size={EVAL_BATCH},
    learning_rate=1e-3, weight_decay=0.05, warmup_fraction=0.10,
    label_smoothing=0.05, grad_clip=1.0, predictive_loss_weight=0.10,
    memory_residual_scale=0.25, initial_eta=0.02, initial_alpha=0.95,
    seed=128,
)
import nestsar_tpu_v5e8_spmd as tpu8
x = tpu8._place_batch(np.zeros(({BATCH},16,150), np.float32))
y = tpu8._place_batch(np.zeros(({BATCH},), np.int32))
print("x global shape:", x.shape)
print("x sharding:", x.sharding)
print("x addressable shards:", len(x.addressable_shards))
print("x local shard shapes:", [s.data.shape for s in x.addressable_shards])
print("y local shard shapes:", [s.data.shape for s in y.addressable_shards])
assert len(x.addressable_shards) == 8
assert all(tuple(s.data.shape) == ({BATCH // 8},16,150) for s in x.addressable_shards)
assert all(tuple(s.data.shape) == ({BATCH // 8},) for s in y.addressable_shards)
print("8-CHIP BATCH SHARDING AUDIT: PASS")
'''
shard_audit = subprocess.run(
    [sys.executable, "-u", "-c", sharding_code],
    cwd=ROOT,
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(shard_audit.stdout)
if shard_audit.returncode != 0:
    raise RuntimeError("8-chip sharding audit failed")

# --------------------------------------------------------------------------------------
# COMMON TRAIN ARGUMENTS
# --------------------------------------------------------------------------------------
def make_cmd(output_dir: Path, epochs: int, patience: int, max_train=0, max_val=0):
    cmd = [
        sys.executable, "-u", str(TPU8),
        "--model", "nestsar_hope_fidelity_d128_v1",
        "--protocol", "xsub",
        "--dataset", str(DATASET),
        "--output-dir", str(output_dir),
        "--seed", str(SEED),
        "--frames", "16",
        "--num-classes", "120",
        "--model-dim", "128",
        "--memory-dim", "64",
        "--frame-blocks", "2",
        "--chunk-blocks", "2",
        "--clip-blocks", "2",
        "--controller-blocks", "2",
        "--chunk-size", "4",
        "--clip-size", "8",
        "--controller-rank", "32",
        "--dropout", "0.22",
        "--batch-size", str(BATCH),
        "--grad-accum-steps", str(ACCUM),
        "--eval-batch-size", str(EVAL_BATCH),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--learning-rate", "1e-3",
        "--weight-decay", "0.05",
        "--warmup-fraction", "0.10",
        "--label-smoothing", "0.05",
        "--grad-clip", "1.0",
        "--memory-residual-scale", "0.25",
        "--predictive-loss-weight", "0.10",
        "--initial-eta", "0.02",
        "--initial-alpha", "0.95",
        "--log-every-batches", "50",
        "--resume", "none",
    ]
    if max_train:
        cmd += ["--max-train-samples", str(max_train)]
    if max_val:
        cmd += ["--max-val-samples", str(max_val)]
    return cmd

# --------------------------------------------------------------------------------------
# 7) REAL DISTRIBUTED BACKWARD/OPTIMIZER PREFLIGHT
# --------------------------------------------------------------------------------------
print("\n[5/7] Compiling/running real 8-chip backward + optimizer preflight...")
PREFLIGHT_OUT = ROOT / "runs_tpu8_preflight"
if PREFLIGHT_OUT.exists():
    shutil.rmtree(PREFLIGHT_OUT)
preflight_cmd = make_cmd(
    PREFLIGHT_OUT,
    epochs=1,
    patience=1,
    max_train=BATCH,
    max_val=EVAL_BATCH,
)
pre = subprocess.run(
    preflight_cmd,
    cwd=ROOT,
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(pre.stdout)
if pre.returncode != 0:
    raise RuntimeError(
        f"8-chip TPU backward/integration preflight failed, rc={pre.returncode}"
    )
print("8-CHIP BACKWARD/OPTIMIZER PREFLIGHT: PASS")

# --------------------------------------------------------------------------------------
# 8) SCRATCH 3-EPOCH PROBE OR 40-EPOCH RUN
# --------------------------------------------------------------------------------------
print("\n[6/7] Preparing scratch XSUB distributed run...")
if FORCE_FRESH and OUT.exists():
    shutil.rmtree(OUT)
if FORCE_FRESH and LOG.exists():
    LOG.unlink()

EPOCHS = 3 if PROBE_ONLY else 40
PATIENCE = 3 if PROBE_ONLY else 12
cmd = make_cmd(OUT, EPOCHS, PATIENCE)

print("=" * 120)
print("TPU v5e-8 RUN CONFIG")
print("=" * 120)
print("Execution:            TRUE 8-chip SPMD data parallel")
print("Batch sharding:       leading axis over all 8 TPU chips")
print("Parameter/state:      replicated over all 8 TPU chips")
print(f"Global train batch:   {BATCH}")
print(f"Local batch/chip:     {BATCH // 8}")
print(f"Grad accumulation:    {ACCUM}")
print(f"Effective batch:      {BATCH * ACCUM}")
print(f"Global eval batch:    {EVAL_BATCH}")
print(f"Local eval/chip:      {EVAL_BATCH // 8}")
print("Outer CMS windows:    128 / 256 / 512 / 1024 samples")
print("Protocol:             NTU120 XSUB")
print("Frames / D / M / R:   16 / 128 / 64 / 32")
print("Parameters:           2,083,236")
print("Reference v4.1:       73.24771% | 2,033,988 | 0.067242094 GFLOPs")
print("Initialization:       SCRATCH")
print(f"Epochs:               {EPOCHS}")
print("New-model GFLOPs:     PROFILE EXACTLY BEFORE PAPER CLAIM")
print("=" * 120)

print("\n[7/7] TRAINING")
start = time.time()
with LOG.open("a", encoding="utf-8", buffering=1) as log:
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
    rc = proc.wait()

print("RETURN CODE:", rc)
print("Elapsed hours:", (time.time() - start) / 3600.0)
print("Output:", OUT)
print("Log:", LOG)
if rc != 0:
    raise RuntimeError(f"TPU8 training failed with return code {rc}")

# ======================================================================================
# NestSAR-HOPE-Fidelity D128 v1 — KAGGLE TPU v5e-8 ONE-CELL
# ======================================================================================
# This is the STABILITY-FIRST TPU launcher.
#
# IMPORTANT:
# - It requires Kaggle Accelerator = TPU v5e-8.
# - It verifies that JAX sees the TPU backend and 8 TPU devices.
# - The existing NestSAR trainer is single-device jax.jit code, therefore this
#   first stable port intentionally executes the training graph on one TPU chip.
#   The other seven chips remain unused until we replace the trainer with an
#   explicit shard_map/pmap data-parallel loop.
# - That is intentional: first prove the new 2,083,236-param architecture trains
#   without the local 16-GB host-RAM OOM; then distribute it.
# - Scratch training only: no 73.24771% checkpoint is loaded.
# - Softmax attention: NONE.
#
# Required Kaggle inputs:
#   1) ntu120_3danno.pkl
#   2) NestSAR_HOPE_v4_1_SHORTL3FIX_Kaggle_16F_ONE_CELL.py
#
# GitHub branch:
#   rombaldivia/NestSAR : hope-fidelity-d128-v1
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
PROBE_ONLY = True          # True = 3 epochs; False = 40 epochs
FORCE_FRESH = True
BATCH = 32
ACCUM = 4                 # effective batch = 128, same as audited v4.1
EVAL_BATCH = 64
SEED = 128

BRANCH = "hope-fidelity-d128-v1"
REPO_URL = "https://github.com/rombaldivia/NestSAR.git"
REPO = Path("/kaggle/working/NestSAR_GitHub")
ROOT = Path("/kaggle/working/NestSAR_HOPE_FIDELITY_TPU")
OUT = ROOT / (
    "runs_hope_fidelity_d128_v1_tpu_probe_xsub"
    if PROBE_ONLY
    else "runs_hope_fidelity_d128_v1_tpu_e40_xsub"
)
LOG = ROOT / (
    "hope_fidelity_d128_v1_tpu_probe.log"
    if PROBE_ONLY
    else "hope_fidelity_d128_v1_tpu_e40.log"
)
EXPECTED_PARAMS = 2_083_236

# --------------------------------------------------------------------------------------
# 0) VERIFY KAGGLE TPU BEFORE DOING ANY WORK
# --------------------------------------------------------------------------------------
print("=" * 120)
print("NESTSAR-HOPE-FIDELITY D128 v1 — KAGGLE TPU v5e-8")
print("=" * 120)

# Do TPU detection in a CHILD process so this notebook parent does not lock the
# backend before we build the exact source bundle.
check_env = os.environ.copy()
check_env["JAX_PLATFORMS"] = "tpu"
check_env["PYTHONUNBUFFERED"] = "1"

check_code = r'''
import os
import jax
print("JAX:", jax.__version__)
print("Backend:", jax.default_backend())
print("Device count:", jax.device_count())
print("Local device count:", jax.local_device_count())
for i, d in enumerate(jax.devices()):
    print(f"  {i}: {d}")
if jax.default_backend() != "tpu":
    raise SystemExit("ERROR: select Accelerator -> TPU v5e-8 in Kaggle.")
if jax.device_count() < 8:
    raise SystemExit(f"ERROR: expected >=8 TPU devices, got {jax.device_count()}.")
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
    raise RuntimeError("TPU v5e-8 preflight failed.")

print("HOST MEMORY")
subprocess.run(["bash", "-lc", "free -h || true"], check=False)
print("DISK")
subprocess.run(["bash", "-lc", "df -h /kaggle/working || true"], check=False)

# --------------------------------------------------------------------------------------
# 1) FIND DATASET AND EXACT AUDITED v4.1 SOURCE BUNDLE
# --------------------------------------------------------------------------------------
datasets = sorted(Path("/kaggle/input").rglob("ntu120_3danno.pkl"))
if not datasets:
    datasets = sorted(Path("/kaggle/input").rglob("ntu120_3danno_clean.pkl"))
if not datasets:
    raise FileNotFoundError("Attach ntu120_3danno.pkl to the Kaggle notebook.")
DATASET = datasets[0].resolve()

baseline_cells = sorted(
    Path("/kaggle/input").rglob(
        "NestSAR_HOPE_v4_1_SHORTL3FIX_Kaggle_16F_ONE_CELL.py"
    )
)
if not baseline_cells:
    raise FileNotFoundError(
        "Attach NestSAR_HOPE_v4_1_SHORTL3FIX_Kaggle_16F_ONE_CELL.py "
        "as a Kaggle input."
    )
BASELINE_CELL = baseline_cells[0].resolve()

print("Dataset:       ", DATASET)
print("v4.1 source:   ", BASELINE_CELL)

# --------------------------------------------------------------------------------------
# 2) EXTRACT EXACT v4.1 SOURCE FILES, BUT DO NOT START THE OLD GPU TRAINING
# --------------------------------------------------------------------------------------
print("\n[1/6] Extracting exact audited v4.1 source bundle...")
source = BASELINE_CELL.read_text(encoding="utf-8")

# The historical one-cell defines its full source bundle and patches before the
# audit/training block. Stop at that exact boundary.
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

# The exact one-cell writes its bundle to its own hard-coded working directory.
# Execute it, then discover that directory by locating the extracted Short-L3 file.
try:
    exec(compile(source, str(BASELINE_CELL), "exec"), {"__name__": "__main__"})
except SystemExit as exc:
    if str(exc) != "__NESTSAR_EXTRACT_ONLY__":
        raise

candidates = sorted(
    Path("/kaggle/working").rglob("nestsar_hope_fullselfref_v3_3_shortl3fix.py")
)
if not candidates:
    raise RuntimeError("Could not find extracted exact v4.1 source tree.")

# Prefer the newest extracted tree and copy it into a clean TPU working root.
base_core = max(candidates, key=lambda p: p.stat().st_mtime)
V41_EXTRACTED = base_core.parent

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

print("Exact v4.1 source extraction: PASS")
print("Source root:", V41_EXTRACTED)
print("TPU root:   ", ROOT)

# --------------------------------------------------------------------------------------
# 3) PULL THE GUARDED HOPE-FIDELITY BUILDER
# --------------------------------------------------------------------------------------
print("\n[2/6] Pulling HOPE-fidelity builder from GitHub...")
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(
    [
        "git", "clone", "--depth", "1", "--branch", BRANCH,
        REPO_URL, str(REPO),
    ],
    check=True,
)
BUILDER = REPO / "experiments/hope_fidelity_d128_v1/build_from_v41.py"
if not BUILDER.is_file():
    raise FileNotFoundError(BUILDER)

# --------------------------------------------------------------------------------------
# 4) GENERATE THE NEW 2,083,236-PARAM MODEL
# --------------------------------------------------------------------------------------
print("\n[3/6] Generating HOPE-fidelity D128 model...")
subprocess.run(
    [sys.executable, "-u", str(BUILDER), "--root", str(ROOT)],
    check=True,
)
CORE = ROOT / "nestsar_hope_fidelity_d128_v1_core.py"
TRAINER = ROOT / "nestsar_hope_fidelity_d128_v1_train.py"
subprocess.run(
    [sys.executable, "-m", "py_compile", str(CORE), str(TRAINER)],
    check=True,
)
print("Generated sources compile: PASS")

# --------------------------------------------------------------------------------------
# 5) TPU CHILD ENVIRONMENT
# --------------------------------------------------------------------------------------
env = os.environ.copy()

# Remove GPU-only controls inherited from old notebooks.
for key in (
    "CUDA_VISIBLE_DEVICES",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
):
    env.pop(key, None)

env.update({
    "JAX_PLATFORMS": "tpu",
    "PYTHONUNBUFFERED": "1",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "PYTHONPATH": str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
    "JAX_COMPILATION_CACHE_DIR": str(ROOT / ".jax_cache_tpu"),
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "1",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",

    # Self-referential memory.
    "NESTSAR_SELFREF_DGD_SCALE": "1.0",
    "NESTSAR_SELFREF_ETA_MAX": "0.05",
    "NESTSAR_SELFREF_RESIDUAL_BETA": "0.10",
    "NESTSAR_SELFREF_MATRIX_NORM_CAP": "4.0",
    "NESTSAR_SELFREF_VECTOR_NORM_CAP": "1.0",
    "NESTSAR_SELFREF_ALPHA_MIN": "0.90",
    "NESTSAR_SELFREF_ALPHA_MAX": "0.999",
    "NESTSAR_SHORT_L3_POSTWRITE_BLEND": "1.0",

    # Outer continuum.
    "NESTSAR_CMS_PERIOD_L1": "1",
    "NESTSAR_CMS_PERIOD_L2": "2",
    "NESTSAR_CMS_PERIOD_L3": "4",
    "NESTSAR_CMS_PERIOD_L4": "8",

    # DMGD-L2.
    "NESTSAR_DMGD_MOMENTUM": "0.90",
    "NESTSAR_DMGD_MEMORY_LR": "0.01",
    "NESTSAR_DMGD_MIX": "0.10",
    "NESTSAR_DMGD_PROJECTION_CAP": "2.0",
})
(ROOT / ".jax_cache_tpu").mkdir(exist_ok=True)

# --------------------------------------------------------------------------------------
# 6) PARAMETER + TPU FORWARD PREFLIGHT
# --------------------------------------------------------------------------------------
print("\n[4/6] TPU parameter/forward preflight...")
audit_code = r'''
import dataclasses
import jax
import jax.numpy as jnp
import nestsar as ns

print("JAX:", jax.__version__)
print("Backend:", jax.default_backend())
print("TPU devices:", len(jax.devices()))
assert jax.default_backend() == "tpu"
assert len(jax.devices()) >= 8

ns.CFG = dataclasses.replace(
    ns.CFG,
    frames=16,
    persons=2,
    joints=25,
    coords=3,
    num_classes=120,
    model_dim=128,
    memory_dim=64,
    controller_rank=32,
    frame_blocks=2,
    chunk_blocks=2,
    clip_blocks=2,
    controller_blocks=2,
    chunk_size=4,
    clip_size=8,
    dropout=0.22,
    batch_size=32,
    grad_accum_steps=4,
    eval_batch_size=64,
    learning_rate=1e-3,
    weight_decay=0.05,
    warmup_fraction=0.10,
    label_smoothing=0.05,
    grad_clip=1.0,
    predictive_loss_weight=0.10,
    memory_residual_scale=0.25,
    initial_eta=0.02,
    initial_alpha=0.95,
    seed=128,
)

import nestsar_hope_fidelity_d128_v1_core as core
model = core.build_model(core.MODEL_ID)
x = jnp.zeros((1, 16, 150), jnp.float32)
rng = jax.random.PRNGKey(128)
variables = model.init({"params": rng, "dropout": rng}, x, training=False)
n = sum(int(v.size) for v in jax.tree_util.tree_leaves(variables["params"]))
print("Parameters:", f"{n:,}")
assert n == 2_083_236, n
out = jax.jit(lambda vv, xx: model.apply(vv, xx, training=False))(variables, x)
jax.block_until_ready(out["logits"])
print("Logits:", out["logits"].shape)
print("Device:", out["logits"].devices())
assert out["logits"].shape == (1,120)
assert bool(jnp.all(jnp.isfinite(out["logits"])))
print("TPU PARAMETER/FORWARD PREFLIGHT: PASS")
'''

audit = subprocess.run(
    [sys.executable, "-u", "-c", audit_code],
    cwd=ROOT,
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(audit.stdout)
if audit.returncode != 0:
    raise RuntimeError("TPU parameter/forward preflight failed.")

# --------------------------------------------------------------------------------------
# 7) FIRST REAL BACKWARD PREFLIGHT — ONLY 128 TRAIN / 64 VAL SAMPLES
# --------------------------------------------------------------------------------------
print("\n[5/6] TPU backward/integration preflight...")
PREFLIGHT_OUT = ROOT / "runs_tpu_preflight"
if PREFLIGHT_OUT.exists():
    shutil.rmtree(PREFLIGHT_OUT)

preflight_cmd = [
    sys.executable, "-u", str(TRAINER),
    "--model", "nestsar_hope_fidelity_d128_v1",
    "--protocol", "xsub",
    "--dataset", str(DATASET),
    "--output-dir", str(PREFLIGHT_OUT),
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
    "--epochs", "1",
    "--patience", "1",
    "--learning-rate", "1e-3",
    "--weight-decay", "0.05",
    "--warmup-fraction", "0.10",
    "--label-smoothing", "0.05",
    "--grad-clip", "1.0",
    "--memory-residual-scale", "0.25",
    "--predictive-loss-weight", "0.10",
    "--initial-eta", "0.02",
    "--initial-alpha", "0.95",
    "--max-train-samples", "128",
    "--max-val-samples", "64",
    "--log-every-batches", "1",
    "--resume", "none",
]

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
        f"TPU backward/integration preflight failed, return code={pre.returncode}"
    )
print("TPU BACKWARD PREFLIGHT: PASS")

# --------------------------------------------------------------------------------------
# 8) SCRATCH XSUB PROBE / FULL TRAIN
# --------------------------------------------------------------------------------------
print("\n[6/6] Launching scratch XSUB run on TPU...")
if FORCE_FRESH and OUT.exists():
    shutil.rmtree(OUT)
if FORCE_FRESH and LOG.exists():
    LOG.unlink()

EPOCHS = 3 if PROBE_ONLY else 40
PATIENCE = 3 if PROBE_ONLY else 12

cmd = [
    sys.executable, "-u", str(TRAINER),
    "--model", "nestsar_hope_fidelity_d128_v1",
    "--protocol", "xsub",
    "--dataset", str(DATASET),
    "--output-dir", str(OUT),
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
    "--epochs", str(EPOCHS),
    "--patience", str(PATIENCE),
    "--learning-rate", "1e-3",
    "--weight-decay", "0.05",
    "--warmup-fraction", "0.10",
    "--label-smoothing", "0.05",
    "--grad-clip", "1.0",
    "--memory-residual-scale", "0.25",
    "--predictive-loss-weight", "0.10",
    "--initial-eta", "0.02",
    "--initial-alpha", "0.95",
    "--log-every-batches", "100",
    "--resume", "none" if FORCE_FRESH else "auto",
]

print("=" * 120)
print("RUN CONFIG")
print("Backend target:       TPU v5e-8")
print("Visible TPU chips:    8 expected")
print("Execution mode:       stability-first single-chip JIT")
print("Protocol:             NTU120 XSUB")
print("Frames / D / M / R:   16 / 128 / 64 / 32")
print("Parameters:           2,083,236")
print("Reference v4.1:       73.24771% | 2,033,988 params | 0.067242094 GFLOPs")
print("Local temporal conv:  causal depthwise k=4")
print("Self-ref memory:      K/V/Q/eta/alpha/main-memory")
print("Sequential CMS:       f1 -> f2 -> f4 -> f8")
print("Softmax attention:    NONE")
print(f"Batch:                {BATCH} x accum {ACCUM} = effective {BATCH*ACCUM}")
print(f"Epochs:               {EPOCHS}")
print("Initialization:       SCRATCH")
print("New-model GFLOPs:     PROFILE EXACTLY BEFORE PAPER CLAIM")
print("=" * 120)

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
    raise RuntimeError(f"TPU training failed with return code {rc}")

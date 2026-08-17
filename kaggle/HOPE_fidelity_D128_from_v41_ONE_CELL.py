# ======================================================================================
# NestSAR-HOPE-Fidelity D128 v1 — KAGGLE GPU ONE-CELL BOOTSTRAP
# ======================================================================================
# PURPOSE
#   1) Reuse the exact self-contained v4.1 Short-L3 one-cell as the trusted source bundle.
#   2) Extract the exact v4.1 source without starting the old training run.
#   3) Pull the guarded HOPE-fidelity builder from GitHub.
#   4) Generate the 2,083,236-param D128/T16 fidelity model.
#   5) Launch a fresh 3-epoch XSUB probe.
#
# KAGGLE SETUP
#   - Accelerator: GPU (P100/T4/L4/etc.)
#   - Internet: ON (for git clone)
#   - Attach ntu120_3danno.pkl
#   - Attach the exact file:
#       NestSAR_HOPE_v4_1_SHORTL3FIX_Kaggle_16F_ONE_CELL.py
#
# IMPORTANT
#   - This is SCRATCH training. No 73.24771% checkpoint is loaded.
#   - Softmax attention: NONE.
# ======================================================================================

from pathlib import Path
import os, shutil, subprocess, sys, textwrap, time

# ------------------------- USER SWITCHES -------------------------
PROBE_ONLY = True
FORCE_FRESH = True
BATCH = 32
ACCUM = 4
EVAL_BATCH = 64

BRANCH = "hope-fidelity-d128-v1"
REPO_URL = "https://github.com/rombaldivia/NestSAR.git"
REPO = Path("/kaggle/working/NestSAR_GitHub")
V41_ROOT = Path("/kaggle/working/NestSAR_HOPE_v4_1_SHORTL3FIX_OFFLINE")
OUT = V41_ROOT / (
    "runs_hope_fidelity_d128_v1_probe_xsub"
    if PROBE_ONLY
    else "runs_hope_fidelity_d128_v1_e40_xsub"
)
LOG = V41_ROOT / (
    "hope_fidelity_d128_v1_probe.log"
    if PROBE_ONLY
    else "hope_fidelity_d128_v1_e40.log"
)

EXPECTED_PARAMS = 2_083_236

print("=" * 120)
print("NESTSAR-HOPE-FIDELITY D128 v1 — KAGGLE GPU")
print("=" * 120)
subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
    check=False,
)

# ------------------------- FIND DATASET -------------------------
datasets = sorted(Path("/kaggle/input").rglob("ntu120_3danno.pkl"))
if not datasets:
    datasets = sorted(Path("/kaggle/input").rglob("ntu120_3danno_clean.pkl"))
if not datasets:
    raise FileNotFoundError("Attach ntu120_3danno.pkl to this Kaggle notebook.")
DATASET = datasets[0].resolve()
print("Dataset:", DATASET)

# ------------------------- FIND EXACT v4.1 ONE-CELL -------------------------
one_cells = sorted(
    Path("/kaggle/input").rglob("NestSAR_HOPE_v4_1_SHORTL3FIX_Kaggle_16F_ONE_CELL.py")
)
if not one_cells:
    raise FileNotFoundError(
        "Attach the exact baseline source file "
        "NestSAR_HOPE_v4_1_SHORTL3FIX_Kaggle_16F_ONE_CELL.py"
    )
BASELINE_CELL = one_cells[0].resolve()
print("Exact v4.1 bundle:", BASELINE_CELL)

# ------------------------- EXTRACT EXACT v4.1 SOURCES ONLY -------------------------
print("\n[1/5] Extracting exact audited v4.1 source bundle...")
source = BASELINE_CELL.read_text(encoding="utf-8")

# Stop the historical one-cell immediately before its GPU/autodiff audit/training section.
anchor = "audit_code = r'''"
if source.count(anchor) != 1:
    raise RuntimeError(
        f"Unexpected baseline one-cell: extract anchor count={source.count(anchor)}"
    )
source = source.replace(
    anchor,
    "raise SystemExit('__NESTSAR_EXTRACT_ONLY__')\n\n" + anchor,
    1,
)

namespace = {"__name__": "__main__"}
try:
    exec(compile(source, str(BASELINE_CELL), "exec"), namespace, namespace)
except SystemExit as exc:
    if str(exc) != "__NESTSAR_EXTRACT_ONLY__":
        raise

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
missing = [name for name in required if not (V41_ROOT / name).is_file()]
if missing:
    raise RuntimeError(f"Exact v4.1 extraction incomplete: {missing}")
print("Exact v4.1 source extraction: PASS")

# ------------------------- CLONE BUILDER BRANCH -------------------------
print("\n[2/5] Pulling guarded fidelity builder from GitHub...")
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO)],
    check=True,
)
BUILDER = REPO / "experiments/hope_fidelity_d128_v1/build_from_v41.py"
if not BUILDER.is_file():
    raise FileNotFoundError(BUILDER)

# ------------------------- GENERATE NEW MODEL/TRAINER -------------------------
print("\n[3/5] Generating HOPE-fidelity core and trainer...")
subprocess.run(
    [sys.executable, "-u", str(BUILDER), "--root", str(V41_ROOT)],
    check=True,
)

NEW_CORE = V41_ROOT / "nestsar_hope_fidelity_d128_v1_core.py"
NEW_TRAIN = V41_ROOT / "nestsar_hope_fidelity_d128_v1_train.py"
subprocess.run(
    [sys.executable, "-m", "py_compile", str(NEW_CORE), str(NEW_TRAIN)],
    check=True,
)
print("Generated sources compile: PASS")

# ------------------------- ENVIRONMENT -------------------------
env = os.environ.copy()
env.update({
    "CUDA_VISIBLE_DEVICES": "0",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
    "MALLOC_ARENA_MAX": "2",
    "PYTHONUNBUFFERED": "1",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "PYTHONPATH": str(V41_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
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

# ------------------------- PARAM/FORWARD PREFLIGHT -------------------------
print("\n[4/5] Parameter/forward preflight...")
audit_code = r'''
import dataclasses
import jax
import jax.numpy as jnp
import nestsar as ns

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
x = jnp.zeros((1,16,150), jnp.float32)
variables = model.init(
    {"params":jax.random.PRNGKey(128), "dropout":jax.random.PRNGKey(129)},
    x,
    training=False,
)
n = sum(int(v.size) for v in jax.tree_util.tree_leaves(variables["params"]))
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
print("Parameters:", f"{n:,}")
assert n == 2_083_236, n
out = model.apply(variables, x, training=False)
assert out["logits"].shape == (1,120)
assert bool(jnp.all(jnp.isfinite(out["logits"])))
print("PARAMETER/FORWARD PREFLIGHT: PASS")
'''
audit = subprocess.run(
    [sys.executable, "-u", "-c", audit_code],
    cwd=V41_ROOT,
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(audit.stdout)
if audit.returncode != 0:
    raise RuntimeError("HOPE-fidelity parameter/forward preflight failed")

# ------------------------- SCRATCH TRAIN -------------------------
print("\n[5/5] Launching scratch XSUB training...")
if FORCE_FRESH and OUT.exists():
    shutil.rmtree(OUT)
if FORCE_FRESH and LOG.exists():
    LOG.unlink()

EPOCHS = 3 if PROBE_ONLY else 40
PATIENCE = 3 if PROBE_ONLY else 12
resume = "none" if FORCE_FRESH else "auto"

cmd = [
    sys.executable, "-u", str(NEW_TRAIN),
    "--model", "nestsar_hope_fidelity_d128_v1",
    "--protocol", "xsub",
    "--dataset", str(DATASET),
    "--output-dir", str(OUT),
    "--seed", "128",
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
    "--log-every-batches", "200",
    "--resume", resume,
]

print("=" * 120)
print("RUN CONFIG — HOPE-FIDELITY D128 v1")
print("Protocol:           NTU120 XSUB")
print("Frames / D / M:     16 / 128 / 64")
print("Expected params:    2,083,236")
print("Baseline reference: 73.24771% | 2,033,988 params | 0.067242094 GFLOPs")
print("New local conv:     causal depthwise k=4")
print("Self-ref memory:    K/V/Q/eta/alpha/main-memory")
print("Forward CMS:        f1 -> f2 -> f4 -> f8")
print("Softmax attention:  NONE")
print(f"Batch:              {BATCH} x {ACCUM} => effective {BATCH*ACCUM}")
print(f"Epochs:             {EPOCHS}")
print("Initialization:     SCRATCH")
print("GFLOPs:             PROFILE EXACTLY BEFORE PAPER CLAIM")
print("=" * 120)

start = time.time()
with LOG.open("a", encoding="utf-8", buffering=1) as log:
    proc = subprocess.Popen(
        cmd,
        cwd=V41_ROOT,
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
    raise RuntimeError(f"Training failed with return code {rc}")

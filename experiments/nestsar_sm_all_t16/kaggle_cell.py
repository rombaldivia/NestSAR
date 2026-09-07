"""Run inside the notebook kernel using runpy.run_path with NESTSAR_SETTINGS."""
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys

settings = dict(globals().get("NESTSAR_SETTINGS", {}))
allowed = {"dataset", "outdir", "cache_dir", "config", "raw_layout", "audit_first", "smoke_test", "runtime_mode"}
if set(settings) - allowed:
    raise ValueError(f"Unknown NESTSAR_SETTINGS keys: {sorted(set(settings) - allowed)}")

display_packages = []
if importlib.util.find_spec("tqdm") is None:
    display_packages.append("tqdm==4.67.1")
if importlib.util.find_spec("ipywidgets") is None:
    display_packages.append("ipywidgets>=8,<9")
if display_packages:
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *display_packages],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Display dependency install failed: " + result.stderr[-2500:])

# Evict old 'experiments' imports from earlier notebook checkouts. The parent
# does not import JAX; each child selects its GPU before importing the model.
root = Path(__file__).resolve().parents[2]
for name in list(sys.modules):
    if name == "experiments" or name.startswith("experiments."):
        del sys.modules[name]
sys.path.insert(0, str(root))
importlib.invalidate_caches()
from experiments.nestsar_sm_all_t16.streaming.launch import run

config = dict(settings.get("config", {}))
outdir = settings.get("outdir", "/kaggle/working/NestSAR_SM_ALL_T16_SharedCache_v2")
if settings.get("smoke_test", False):
    config.update(epochs=2, micro_batch=32, accumulation_steps=2,
                  eval_batch=64, max_train_samples=256, max_val_samples=256)
    outdir += "_smoke"

nestsar_results = run(
    dataset=settings.get("dataset"), outdir=outdir,
    cache_dir=settings.get("cache_dir", "/kaggle/working/NestSAR_SM_ALL_SharedCache_v2"),
    config=config, raw_layout=settings.get("raw_layout", "MTVC"),
    audit_first=settings.get("audit_first", True),
    runtime_mode=settings.get("runtime_mode", "host"),
)

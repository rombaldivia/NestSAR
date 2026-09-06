"""Execute in the Kaggle notebook kernel via runpy.run_path, not a shell pipe.

Optional init_globals: {"NESTSAR_SETTINGS": {"dataset": None, "config": {...}}}.
The parent owns XSUB/GPU0 and XSET/GPU1 bars with BEST score and epoch.
"""
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys

settings = dict(globals().get("NESTSAR_SETTINGS", {}))
allowed = {"dataset", "outdir", "cache_dir", "config", "raw_layout", "audit_first", "smoke_test"}
if set(settings) - allowed:
    raise ValueError(f"Unknown NESTSAR_SETTINGS keys: {sorted(set(settings) - allowed)}")

# Display packages belong to the notebook kernel; the numerical runtime is
# installed separately by the launcher using pip --python, without ensurepip.
display_requirements = []
if importlib.util.find_spec("tqdm") is None:
    display_requirements.append("tqdm==4.67.1")
if importlib.util.find_spec("ipywidgets") is None:
    display_requirements.append("ipywidgets>=8,<9")
if display_requirements:
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *display_requirements],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Display dependency install failed: " + result.stderr[-2500:])

source = Path(__file__).resolve().parent
for name in list(sys.modules):
    if name == "nestsar_fixed" or name.startswith("nestsar_fixed."):
        del sys.modules[name]
sys.path.insert(0, str(source))
importlib.invalidate_caches()
from nestsar_fixed.launch import run

config = dict(settings.get("config", {}))
outdir = settings.get("outdir", "/kaggle/working/NestSAR_Fixed_T16_2xT4")
if settings.get("smoke_test", False):
    config.update(epochs=2, micro_batch=32, accumulation_steps=2,
                  eval_batch=64, max_train_samples=256, max_val_samples=256)
    outdir += "_smoke"

nestsar_results = run(
    dataset=settings.get("dataset"),
    outdir=outdir,
    cache_dir=settings.get("cache_dir", "/kaggle/working/NestSAR_RawCache_v1"),
    config=config,
    raw_layout=settings.get("raw_layout", "MTVC"),
    audit_first=settings.get("audit_first", True),
)

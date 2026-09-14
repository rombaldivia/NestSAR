"""Call with runpy.run_path(..., init_globals={'NESTSAR_B_SETTINGS': {...}})."""
import importlib
from pathlib import Path
import sys

settings = dict(globals().get("NESTSAR_B_SETTINGS", {}))
allowed = {"dataset", "outdir", "cache_dir", "config", "audit_first"}
if set(settings) - allowed:
    raise ValueError(f"Unknown sampler B settings: {sorted(set(settings) - allowed)}")

root = Path(__file__).resolve().parents[2]
# Avoid mixed imports from an earlier Kaggle checkout; never import JAX here.
for name in list(sys.modules):
    if name == "experiments" or name.startswith("experiments."):
        del sys.modules[name]
sys.path.insert(0, str(root))
importlib.invalidate_caches()

from experiments.nestsar_sm_all_t16.run_sampler_b_dual_t4 import run_sampler_b

settings.setdefault("outdir", "/kaggle/working/NestSAR_SamplerB_T16_Seed128_v1")
settings.setdefault("audit_first", not (Path(settings["outdir"]) / "compute_audit.json").is_file())
nestsar_results = run_sampler_b(**settings)

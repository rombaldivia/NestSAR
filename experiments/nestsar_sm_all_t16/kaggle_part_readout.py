"""Run through runpy with NESTSAR_PART_SETTINGS; no JAX import in the notebook."""
import importlib
from pathlib import Path
import sys

settings = dict(globals().get("NESTSAR_PART_SETTINGS", {}))
allowed = {"dataset", "outdir", "cache_dir", "sampler_cache_dir", "config", "audit_first"}
if set(settings) - allowed:
    raise ValueError(f"Unknown part readout settings: {sorted(set(settings) - allowed)}")
root = Path(__file__).resolve().parents[2]
for name in list(sys.modules):
    if name == "experiments" or name.startswith("experiments."):
        del sys.modules[name]
sys.path.insert(0, str(root))
importlib.invalidate_caches()

from experiments.nestsar_sm_all_t16.run_part_readout_dual_t4 import run_part_readout

settings.setdefault("outdir", "/kaggle/working/NestSAR_PartReadout_T16_Seed128_v1")
settings.setdefault("audit_first", not (Path(settings["outdir"]) / "compute_audit.json").is_file())
nestsar_results = run_part_readout(**settings)

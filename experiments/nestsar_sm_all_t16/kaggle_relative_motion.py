"""Execute in the notebook kernel, which owns the same two persistent bars."""
from experiments.nestsar_sm_all_t16.relative_motion.launch import run

settings = dict(globals().get("NESTSAR_RELATIVE_SETTINGS", {}))
unknown = set(settings) - {"cache_dir", "auxiliary_dir", "outdir", "config", "dataset"}
if unknown:
    raise ValueError(f"Unknown NESTSAR_RELATIVE_SETTINGS: {sorted(unknown)}")
nestsar_relative_results = run(**settings)

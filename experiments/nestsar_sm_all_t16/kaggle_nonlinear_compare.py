"""Run inside the notebook kernel so it owns both persistent progress bars."""
from experiments.nestsar_sm_all_t16.nonlinear_compare.launch import run

settings = dict(globals().get("NESTSAR_COMPARE_SETTINGS", {}))
unknown = set(settings)-{"cache_dir", "outdir", "config", "smoke_test", "dataset"}
if unknown:
    raise ValueError(f"Unknown NESTSAR_COMPARE_SETTINGS: {sorted(unknown)}")
config = dict(settings.get("config", {}))
outdir = settings.get("outdir", "/kaggle/working/NestSAR_P2_Nonlinear_Grouped_v1")
if settings.get("smoke_test", False):
    config.update(pairs=[[71, 72]], seeds=[128], epochs=2, warmup_epochs=1,
                  trials=[dict(learning_rate=1e-3, weight_decay=1e-4, dropout=.1)])
    outdir += "_smoke"

nestsar_comparison = run(cache_dir=settings.get("cache_dir"), outdir=outdir,
                         config=config, dataset=settings.get("dataset"))

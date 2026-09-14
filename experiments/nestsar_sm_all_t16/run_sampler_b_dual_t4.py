"""Sampler B only, complete official NTU120 splits, one protocol per GPU."""
from pathlib import Path

from .sampler_b import VERSION
from .streaming import launch
from .streaming import notebook_progress


def run_sampler_b(dataset=None,
                  outdir="/kaggle/working/NestSAR_SamplerB_T16_Seed128_v1",
                  cache_dir="/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3",
                  config=None, audit_first=True):
    overrides = dict(config or {})
    if overrides.get("pose_sampler", VERSION) != VERSION:
        raise ValueError("This entry point runs sampler B only")
    overrides["pose_sampler"] = VERSION
    c = launch.validate_config(overrides)
    # Latest full-official run's complete settings are inherited from DEFAULTS:
    # seed128, 60 epochs, patience5 after warmup, batch64x4, EMA.995, fresh views.
    print(f"Sampler B | XSUB GPU0: 63,026 train / 50,919 val | XSET GPU1: 54,468 train / 59,477 val")
    print(f"T16 / M4-G4 | seed {c['seed']} | epochs {c['epochs']} | patience {c['patience']} "
          f"after warm-up | batch {c['micro_batch']} x {c['accumulation_steps']}")
    old_make, old_update = launch.make_bars, launch.update_bar
    launch.make_bars, launch.update_bar = notebook_progress.make_bars, notebook_progress.update_bar
    try:
        results = launch.run(dataset=dataset, outdir=outdir, cache_dir=cache_dir,
                             config=overrides, raw_layout="MTVC", audit_first=audit_first)
    finally:
        launch.make_bars, launch.update_bar = old_make, old_update
    for protocol, result in results.items():
        print(f"{protocol.upper()} BEST: {100 * result['best_val_accuracy']:.4f}% "
              f"@ E{result['best_epoch']:02d}")
    print(f"Checkpoints, best_scores.json and sampler diagnostics: {Path(outdir)}")
    return results

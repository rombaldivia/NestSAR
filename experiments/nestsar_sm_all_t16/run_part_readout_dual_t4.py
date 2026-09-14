"""One architectural ablation on the existing sampler B training pipeline."""
from pathlib import Path

from .part_readout_config import VERSION, extra_parameters
from .run_sampler_b_dual_t4 import run_sampler_b


def choose_sampler_cache(explicit=None):
    if explicit is not None:
        return str(Path(explicit))
    # Reuse the last run's completed overlays instead of making another 2 GiB copy.
    previous = Path("/kaggle/working/NestSAR_SamplerB_T16_Seed128_v1/sampler_b")
    if (previous / "manifest.json").is_file():
        return str(previous)
    return "/kaggle/working/NestSAR_SamplerB_SharedPoseCache_v1"


def run_part_readout(dataset=None,
                     outdir="/kaggle/working/NestSAR_PartReadout_T16_Seed128_v1",
                     cache_dir="/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3",
                     sampler_cache_dir=None, config=None, audit_first=True):
    overrides = dict(config or {})
    if overrides.get("part_readout", VERSION) != VERSION:
        raise ValueError("This entry point runs the local-subspace-v1 part readout")
    if "sampler_cache_dir" in overrides:
        raise ValueError("Pass sampler_cache_dir as a launcher setting, not inside config")
    overrides.update(part_readout=VERSION, sampler_cache_dir=choose_sampler_cache(sampler_cache_dir))
    print(f"Nonlinear part readout | +{extra_parameters():,} parameters | T16 and M4/G4 retained")
    print(f"Shared B poses: {overrides['sampler_cache_dir']} | FLOPs measured before training")
    return run_sampler_b(dataset=dataset, outdir=outdir, cache_dir=cache_dir,
                         config=overrides, audit_first=audit_first)

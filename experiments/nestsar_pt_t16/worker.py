"""NestSAR-PT inside the proven R4 streaming pipeline.

Same cache, preprocessing, augmentation, losses (main + stream aux +
consistency), optimizer, EMA, checkpointing and resume as R4. Only the model
factory, its identity and the expected parameter count change, and only inside
this worker process (the R4 worker module is not modified on disk).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from experiments.nestsar_sm_all_t16.streaming import worker as base
from experiments.nestsar_sm_all_t16.streaming import launch as r4_launch
from experiments.nestsar_pt_t16 import MODEL_NAME, VERSION
from experiments.nestsar_pt_t16.model import NestSARPTT16

# Measured with the exact model (jax.make_jaxpr + param count), batch 1, T=16.
EXPECTED_PARAMS = {32: 1_173_436, 40: 1_276_828, 48: 1_394_556}
STRICT_MFLOPS = {32: 75.40, 40: 97.89, 48: 124.96}   # 1 MAC = 2 FLOPs
DEFAULT_PART_DIM = 32


def validate_config(config):
    config = dict(config or {})
    part_dim = int(config.pop("part_dim", DEFAULT_PART_DIM))
    if part_dim not in EXPECTED_PARAMS:
        raise ValueError(f"part_dim must be one of {sorted(EXPECTED_PARAMS)}")
    c = r4_launch.validate_config(config)      # every R4 safety rule still applies
    c["part_dim"] = part_dim
    return c


def make_model(config):
    return NestSARPTT16(
        spatial_dim=config["spatial_dim"], part_dim=config["part_dim"],
        model_dim=config["model_dim"], dropout=config["dropout"],
        controller_dim=config["controller_dim"], fast_rank=config["fast_rank"],
        head_rank=config["head_rank"], sm_residual_scale=config["sm_residual_scale"],
        head_residual_scale=config["head_residual_scale"],
    )


def implementation_identity():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("model.py", "worker.py", "__init__.py"):
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    return {"version": VERSION, "source_sha256": digest.hexdigest()}


def run(config, protocol, cache, outdir, allow_cpu=False):
    c = validate_config(config)
    base.EXPECTED_PARAMS = EXPECTED_PARAMS[c["part_dim"]]   # this process only
    return base.run(c, protocol, cache, outdir, allow_cpu,
                    model_factory=make_model, model_name=MODEL_NAME,
                    model_identity=implementation_identity(),
                    config_validator=validate_config)


if __name__ == "__main__":
    print(MODEL_NAME, flush=True)
    base.main(run)

"""Parallel model with explicit injection into the shared training pipeline.

Importing this module never changes the R4 worker or its model factory.
"""
from experiments.nestsar_sm_all_t16.model_parallel import NestSARParallelT16
from experiments.nestsar_sm_all_t16.parallel_config import (
    MODEL_NAME, EXPECTED_PARAMS, implementation_identity,
)
from experiments.nestsar_sm_all_t16.streaming import worker as base


def make_model(config):
    return NestSARParallelT16(**{k: config[k] for k in (
        "spatial_dim", "model_dim", "dropout", "controller_dim", "fast_rank",
        "head_rank", "sm_residual_scale", "head_residual_scale")})


def run(config, protocol, cache, outdir, allow_cpu=False):
    return base.run(config, protocol, cache, outdir, allow_cpu,
                    model_factory=make_model, model_name=MODEL_NAME,
                    model_identity=implementation_identity())


if __name__ == "__main__":
    print(MODEL_NAME, flush=True)
    print("Memory: associative prefix scans; streams: lifted vmap", flush=True)
    base.main(run)

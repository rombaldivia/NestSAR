"""D128 multi-timescale parallel model in the proven streaming pipeline."""
from experiments.nestsar_sm_all_t16.model_parallel import NestSARParallelT16
from experiments.nestsar_sm_all_t16.parallel_d128_config import (
    MODEL_NAME,
    EXPECTED_PARAMS,
    MODEL_DIM,
    M4_HALF_LIVES,
    G4_HALF_LIVES,
    implementation_identity,
    validate_d128_config,
)
from experiments.nestsar_sm_all_t16.streaming import worker as base


def make_model(config):
    if int(config["model_dim"]) != MODEL_DIM:
        raise ValueError(
            f"This experiment requires model_dim={MODEL_DIM}, got {config['model_dim']}"
        )
    return NestSARParallelT16(
        spatial_dim=config["spatial_dim"],
        model_dim=MODEL_DIM,
        dropout=config["dropout"],
        controller_dim=config["controller_dim"],
        fast_rank=config["fast_rank"],
        head_rank=config["head_rank"],
        sm_residual_scale=config["sm_residual_scale"],
        head_residual_scale=config["head_residual_scale"],
        m4_half_lives=M4_HALF_LIVES,
        g4_half_lives=G4_HALF_LIVES,
    )


def run(config, protocol, cache, outdir, allow_cpu=False):
    # base.run/create_state intentionally keep the historical constant for R4.
    # Override it only inside this dedicated worker process.
    base.EXPECTED_PARAMS = EXPECTED_PARAMS
    return base.run(
        config,
        protocol,
        cache,
        outdir,
        allow_cpu,
        model_factory=make_model,
        model_name=MODEL_NAME,
        model_identity=implementation_identity(),
        config_validator=validate_d128_config,
    )


if __name__ == "__main__":
    print(MODEL_NAME, flush=True)
    print(f"model_dim={MODEL_DIM} params={EXPECTED_PARAMS:,}", flush=True)
    print(f"M4 half-lives={M4_HALF_LIVES}", flush=True)
    print(f"G4 half-lives={G4_HALF_LIVES}", flush=True)
    base.main(run)

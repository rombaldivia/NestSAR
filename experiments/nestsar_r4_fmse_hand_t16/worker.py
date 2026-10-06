from __future__ import annotations

"""Exact R4/FMSE training pipeline with the hand-relation model factory."""

from experiments.nestsar_sm_all_t16.streaming import worker as base_worker

from . import MODEL_IDENTITY, MODEL_NAME
from .model import NestSARR4FMSEHandT16


EXPECTED_PARAMS = 1_832_732

# The historical worker parameter-count/checkpoint/result machinery reads this
# module-level constant.  Each protocol runs in its own process, so this change
# is isolated to the hand-relation experiment worker.
base_worker.EXPECTED_PARAMS = EXPECTED_PARAMS


def make_model(config):
    return NestSARR4FMSEHandT16(
        **{
            k: config[k]
            for k in (
                "spatial_dim",
                "model_dim",
                "dropout",
                "controller_dim",
                "fast_rank",
                "head_rank",
                "sm_residual_scale",
                "head_residual_scale",
            )
        }
    )


def run(
    config,
    protocol,
    cache,
    outdir,
    allow_cpu=False,
):
    base_worker.EXPECTED_PARAMS = EXPECTED_PARAMS

    return base_worker.run(
        config,
        protocol,
        cache,
        outdir,
        allow_cpu,
        model_factory=make_model,
        model_name=MODEL_NAME,
        model_identity=MODEL_IDENTITY,
    )


def main():
    base_worker.EXPECTED_PARAMS = EXPECTED_PARAMS
    base_worker.main(
        run_fn=run
    )


if __name__ == "__main__":
    main()

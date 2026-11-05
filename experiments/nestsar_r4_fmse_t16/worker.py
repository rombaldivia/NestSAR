from __future__ import annotations

"""R4 FMSE worker.

Reuses the exact historical R4 training/evaluation/checkpoint pipeline.
Only model_factory/model identity change.
"""

from experiments.nestsar_sm_all_t16.streaming import worker as base_worker

from . import MODEL_IDENTITY, MODEL_NAME
from .model import NestSARR4FMSET16


EXPECTED_PARAMS = base_worker.EXPECTED_PARAMS


def make_model(config):
    return NestSARR4FMSET16(
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
    base_worker.main(
        run_fn=run
    )


if __name__ == "__main__":
    main()

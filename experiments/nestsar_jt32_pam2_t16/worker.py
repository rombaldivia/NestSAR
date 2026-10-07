from __future__ import annotations

"""Historical R4 training harness adapted to NestSAR-JT32-PAM2-T16."""

import jax
import jax.numpy as jnp

from experiments.nestsar_sm_all_t16.preprocessing_corrected import (
    FRAMES,
    FEATURES,
)
from experiments.nestsar_sm_all_t16.streaming import worker as base_worker

from . import MODEL_IDENTITY, MODEL_NAME
from .model import NestSARJT32PAM2T16


def make_model(config):
    return NestSARJT32PAM2T16(
        dim=32,
        heads=4,
        blocks=2,
        memory_rank=4,
        dropout=config["dropout"],
    )


def parameter_count(config):
    model = make_model(config)
    key = jax.random.PRNGKey(config["seed"])
    params = model.init(
        {
            "params": key,
            "dropout": key,
        },
        jnp.zeros(
            (1, FRAMES, FEATURES),
            dtype=jnp.float32,
        ),
        training=False,
    )["params"]

    return int(
        sum(
            x.size
            for x in jax.tree.leaves(params)
        )
    )


def run(
    config,
    protocol,
    cache,
    outdir,
    allow_cpu=False,
):
    # The historical worker uses a module-global expected count for its
    # reproducibility signature, checkpoint metadata and initialization check.
    # Set it to the exact count of this architecture before entering the run.
    count = parameter_count(config)
    base_worker.EXPECTED_PARAMS = count

    return base_worker.run(
        config,
        protocol,
        cache,
        outdir,
        allow_cpu,
        model_factory=make_model,
        model_name=MODEL_NAME,
        model_identity={
            **MODEL_IDENTITY,
            "parameters": count,
        },
    )


def main():
    base_worker.main(
        run_fn=run
    )


if __name__ == "__main__":
    main()

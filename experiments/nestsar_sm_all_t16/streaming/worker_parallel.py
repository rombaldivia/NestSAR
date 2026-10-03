#!/usr/bin/env python3
from __future__ import annotations

"""Streaming worker for the fully-parallel NestSAR T16 experiment.

Reuses the proven R4 data pipeline, optimizer, augmentation, checkpoint/resume,
EMA, validation, and early stopping.  Only the model implementation is swapped.
"""

import json
from pathlib import Path

import jax

from experiments.nestsar_sm_all_t16.model_parallel import NestSARParallelT16
from experiments.nestsar_sm_all_t16.streaming import worker as base


MODEL_NAME = "NestSAR-FULL-PARALLEL-T16-v1"
EXPECTED_PARAMS = 1_831_932


def make_model(config):
    return NestSARParallelT16(
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


def publish_best(out, metadata):
    name = metadata["best_checkpoint"]
    if Path(name).name != name or not name.startswith("best_epoch_"):
        raise ValueError("Invalid best-checkpoint filename")
    payload = (out / name).read_bytes()
    base.atomic_bytes(out / "best.msgpack", payload)
    base.atomic_json(
        out / "best.json",
        dict(
            model=MODEL_NAME,
            epoch=metadata["best_epoch"],
            val_accuracy=metadata["best"],
            params=EXPECTED_PARAMS,
            preprocessing_version=base.PREPROCESSING_VERSION,
            pipeline_version=base.VERSION,
            config_hash=metadata["config_hash"],
        ),
    )


def write_result(out, protocol, metadata, digest, resumed=False):
    result = {
        "model": MODEL_NAME,
        "protocol": protocol,
        "best_val_accuracy": metadata["best"],
        "best_accuracy": metadata["best"],
        "best_epoch": metadata["best_epoch"],
        "last_epoch": metadata["epoch"],
        "epochs_run": metadata["epoch"],
        "params": EXPECTED_PARAMS,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.local_devices()],
        "config_hash": digest,
        "checkpoint": str(out / "best.msgpack"),
        "preprocessing_version": base.PREPROCESSING_VERSION,
        "pipeline_version": base.VERSION,
        "stopped_early": metadata["stopped_early"],
        "resumed_completed": resumed,
        "parallel_memory": {
            "base_sweep": "associative_affine_prefix",
            "fast_weight": "exact_associative_affine_matrix_prefix",
            "stream_execution": "lifted_vmap",
        },
    }
    base.atomic_json(out / "result.json", result)
    base.atomic_json(out.parent / f"result_{protocol}.json", result)
    return result


# Monkeypatch the proven R4 streaming worker.
base.make_model = make_model
base.EXPECTED_PARAMS = EXPECTED_PARAMS
base.publish_best = publish_best
base.write_result = write_result


if __name__ == "__main__":
    print("=" * 100)
    print(MODEL_NAME)
    print("=" * 100)
    print("Base memory : associative affine prefix scan")
    print("Fast weight : exact associative prefix scan")
    print("Streams     : vectorized with lifted vmap")
    print(f"Expected parameters: {EXPECTED_PARAMS:,}")
    base.main()

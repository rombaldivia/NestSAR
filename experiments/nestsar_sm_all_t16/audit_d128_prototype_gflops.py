#!/usr/bin/env python3
from __future__ import annotations

"""Inference GFLOP audit for D128-MTS + prototype-margin training.

The prototype bank/loss exists only in the training worker.  This audit proves
that the deployed inference graph is exactly the original D128-MTS graph and
measures its XLA forward FLOPs at batch 1 and batch 256.

No dataset or checkpoint is required.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.nestsar_sm_all_t16.model_parallel import NestSARParallelT16
from experiments.nestsar_sm_all_t16.parallel_d128_config import (
    EXPECTED_PARAMS,
    MODEL_DIM,
    M4_HALF_LIVES,
    G4_HALF_LIVES,
)
from experiments.nestsar_sm_all_t16.streaming import launch as launch_base
from experiments.nestsar_sm_all_t16.streaming import worker_parallel_d128_prototype as proto
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json


FRAMES = 16
FEATURES = 750
NUM_CLASSES = 120


def count_params(tree):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def tree_signature(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return str(treedef), [(tuple(np.asarray(x).shape), str(np.asarray(x).dtype)) for x in leaves]


def primitive_counts(obj, counts=None):
    counts = {} if counts is None else counts
    if hasattr(obj, "jaxpr"):
        primitive_counts(obj.jaxpr, counts)
    elif hasattr(obj, "eqns"):
        for eq in obj.eqns:
            name = eq.primitive.name
            counts[name] = counts.get(name, 0) + 1
            primitive_counts(eq.params, counts)
    elif isinstance(obj, dict):
        for value in obj.values():
            primitive_counts(value, counts)
    elif isinstance(obj, (tuple, list)):
        for value in obj:
            primitive_counts(value, counts)
    return counts


def make_d112():
    return NestSARParallelT16(
        spatial_dim=24,
        model_dim=112,
        dropout=0.10,
        controller_dim=16,
        fast_rank=4,
        head_rank=2,
        sm_residual_scale=0.08,
        head_residual_scale=0.15,
        m4_half_lives=None,
        g4_half_lives=None,
    )


def make_d128():
    return NestSARParallelT16(
        spatial_dim=24,
        model_dim=MODEL_DIM,
        dropout=0.10,
        controller_dim=16,
        fast_rank=4,
        head_rank=2,
        sm_residual_scale=0.08,
        head_residual_scale=0.15,
        m4_half_lives=M4_HALF_LIVES,
        g4_half_lives=G4_HALF_LIVES,
    )


def prototype_config():
    c = dict(launch_base.DEFAULTS)
    c["model_dim"] = MODEL_DIM
    c.update(proto.PROTOTYPE_DEFAULTS)
    return c


def init(model, seed=128):
    key = jax.random.PRNGKey(seed)
    x = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    return model.init(
        {"params": key, "dropout": key},
        x,
        training=False,
    )["params"]


def xla_cost(model, params, batch):
    x = jnp.zeros((batch, FRAMES, FEATURES), jnp.float32)
    fn = jax.jit(
        lambda p, xx: model.apply(
            {"params": p},
            xx,
            training=False,
        )["logits"]
    )
    compiled = fn.lower(params, x).compile()
    ca = compiled.cost_analysis()
    if isinstance(ca, list):
        ca = ca[0] if ca else {}
    flops = float(ca.get("flops", float("nan")))
    if not np.isfinite(flops) or flops <= 0:
        raise RuntimeError(f"Invalid XLA cost analysis: {ca}")
    return flops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise RuntimeError(
            f"Expected exactly one isolated GPU; backend={jax.default_backend()} "
            f"devices={jax.local_devices()}"
        )

    d112 = make_d112()
    d128 = make_d128()
    d128_proto = proto.make_model(prototype_config())

    p112 = init(d112)
    p128 = init(d128)
    pproto = init(d128_proto)

    n112 = count_params(p112)
    n128 = count_params(p128)
    nproto = count_params(pproto)

    if n112 != 1_831_932:
        raise AssertionError(f"D112 parameter drift: {n112:,}")
    if n128 != EXPECTED_PARAMS:
        raise AssertionError(f"D128 parameter drift: {n128:,}")
    if nproto != EXPECTED_PARAMS:
        raise AssertionError(f"Prototype inference parameter drift: {nproto:,}")

    sig128 = tree_signature(p128)
    sigproto = tree_signature(pproto)
    if sig128 != sigproto:
        raise AssertionError(
            "Prototype worker changed the inference parameter tree."
        )

    # Same seed + exact same inference module/config should initialize identically.
    leaves_a = jax.tree_util.tree_leaves(p128)
    leaves_b = jax.tree_util.tree_leaves(pproto)
    max_init_diff = max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(leaves_a, leaves_b)
    )
    if max_init_diff != 0.0:
        raise AssertionError(
            f"Prototype worker inference initialization differs: {max_init_diff}"
        )

    x_graph = jnp.zeros((2, FRAMES, FEATURES), jnp.float32)
    fn_graph = lambda p, xx: d128_proto.apply(
        {"params": p},
        xx,
        training=False,
    )["logits"]
    counts = primitive_counts(
        jax.make_jaxpr(fn_graph)(pproto, x_graph)
    )

    if counts.get("scan", 0) or counts.get("while", 0):
        raise AssertionError(
            f"Serial recurrent primitive remains in inference graph: {counts}"
        )

    f112_b1 = xla_cost(d112, p112, 1)
    f128_b1 = xla_cost(d128, p128, 1)
    fproto_b1 = xla_cost(d128_proto, pproto, 1)

    f128_b256 = xla_cost(d128, p128, 256)
    fproto_b256 = xla_cost(d128_proto, pproto, 256)

    # Cost analysis can change fusion by batch size, so report both raw total
    # and per-clip values instead of assuming perfect linear scaling.
    proto_bank_values = NUM_CLASSES * MODEL_DIM
    proto_bank_bytes_fp32 = proto_bank_values * 4

    if fproto_b1 != f128_b1:
        raise AssertionError(
            "Prototype training changed batch-1 inference FLOPs: "
            f"{fproto_b1} != {f128_b1}"
        )

    result = {
        "model": proto.MODEL_NAME,
        "backend": jax.default_backend(),
        "device": str(jax.local_devices()[0]),
        "input": [1, FRAMES, FEATURES],
        "parameters": {
            "d112_parallel": n112,
            "d128_mts": n128,
            "d128_prototype_inference": nproto,
            "prototype_extra_inference_params": nproto - n128,
        },
        "graph": {
            "scan_count": int(counts.get("scan", 0)),
            "while_count": int(counts.get("while", 0)),
        },
        "compute": {
            "d112_batch1_gflops": f112_b1 / 1e9,
            "d128_batch1_gflops": f128_b1 / 1e9,
            "prototype_batch1_gflops": fproto_b1 / 1e9,
            "prototype_extra_inference_gflops": (fproto_b1 - f128_b1) / 1e9,
            "d128_vs_d112_delta_gflops": (f128_b1 - f112_b1) / 1e9,
            "d128_vs_d112_ratio": f128_b1 / f112_b1,
            "d128_batch256_total_gflops": f128_b256 / 1e9,
            "d128_batch256_per_clip_gflops": (f128_b256 / 256.0) / 1e9,
            "prototype_batch256_total_gflops": fproto_b256 / 1e9,
            "prototype_batch256_per_clip_gflops": (fproto_b256 / 256.0) / 1e9,
            "convention": "JAX/XLA compiled forward cost_analysis, logits only",
        },
        "training_only_prototype_state": {
            "bank_shape": [NUM_CLASSES, MODEL_DIM],
            "fp32_values": proto_bank_values,
            "fp32_bytes": proto_bank_bytes_fp32,
            "kib": proto_bank_bytes_fp32 / 1024.0,
            "used_at_inference": False,
        },
        "proof": {
            "same_parameter_tree_as_plain_d128": True,
            "same_seed_initialization_max_abs_diff": max_init_diff,
            "same_batch1_forward_flops_as_plain_d128": True,
            "prototype_loss_used_at_inference": False,
        },
    }

    atomic_json(Path(args.output), result)

    print("=" * 118)
    print("NESTSAR D128 PROTOTYPE — INFERENCE GFLOP AUDIT")
    print("=" * 118)
    print("Device:", jax.local_devices()[0])
    print()
    print("PARAMETERS")
    print(f"D112 Parallel-v2       : {n112:,}")
    print(f"D128 MTS               : {n128:,}")
    print(f"D128 + prototype train : {nproto:,} inference params")
    print(f"Extra inference params : {nproto - n128:+,}")
    print()
    print("GRAPH")
    print(f"scan  : {counts.get('scan', 0)}")
    print(f"while : {counts.get('while', 0)}")
    print()
    print("BATCH-1 FORWARD COMPUTE")
    print(f"D112 Parallel-v2       : {f112_b1/1e9:.9f} GFLOPs/clip")
    print(f"D128 MTS               : {f128_b1/1e9:.9f} GFLOPs/clip")
    print(f"D128 prototype         : {fproto_b1/1e9:.9f} GFLOPs/clip")
    print(f"Prototype inference Δ  : {(fproto_b1-f128_b1)/1e9:+.9f} GFLOPs")
    print(
        f"D128 vs D112           : {(f128_b1-f112_b1)/1e9:+.9f} GFLOPs "
        f"({f128_b1/f112_b1:.4f}x)"
    )
    print()
    print("BATCH-256 FORWARD COMPUTE")
    print(
        f"D128 total             : {f128_b256/1e9:.6f} GFLOPs/batch | "
        f"{(f128_b256/256)/1e9:.9f} GFLOPs/clip"
    )
    print(
        f"Prototype total        : {fproto_b256/1e9:.6f} GFLOPs/batch | "
        f"{(fproto_b256/256)/1e9:.9f} GFLOPs/clip"
    )
    print()
    print("TRAINING-ONLY PROTOTYPE STATE")
    print(
        f"120 x 128 FP32 = {proto_bank_values:,} values = "
        f"{proto_bank_bytes_fp32/1024:.2f} KiB"
    )
    print("Inference use           : NO")
    print()
    print("CONCLUSION")
    print("Prototype training changes inference params : NO")
    print("Prototype training changes inference GFLOPs : NO")
    print("Prototype bank enters deployed graph         : NO")
    print()
    print("Saved:", args.output)
    print("=" * 118)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

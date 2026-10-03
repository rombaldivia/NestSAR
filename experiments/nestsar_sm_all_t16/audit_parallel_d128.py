#!/usr/bin/env python3
from __future__ import annotations

"""Hard audit for NestSAR D128 + per-stream multi-timescale memory.

No training data required. Verifies:
  * exact parameter count;
  * zero recurrent scan/while in the forward graph;
  * corrected per-stream channel-wise half-life layout;
  * GPU XLA forward FLOPs versus Parallel-v2 D112;
  * synchronized batch-1 / batch-256 inference latency.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

from experiments.nestsar_sm_all_t16.model_parallel import NestSARParallelT16
from experiments.nestsar_sm_all_t16.parallel_d128_config import (
    MODEL_NAME,
    EXPECTED_PARAMS,
    MODEL_DIM,
    M4_HALF_LIVES,
    G4_HALF_LIVES,
    implementation_identity,
)
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json


FRAMES = 16
FEATURES = 750


def count_params(params):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)))


def make_model(dim, *, mts):
    return NestSARParallelT16(
        spatial_dim=24,
        model_dim=dim,
        dropout=0.10,
        controller_dim=16,
        fast_rank=4,
        head_rank=2,
        sm_residual_scale=0.08,
        head_residual_scale=0.15,
        m4_half_lives=M4_HALF_LIVES if mts else None,
        g4_half_lives=G4_HALF_LIVES if mts else None,
    )


def init(model, seed=128):
    key = jax.random.PRNGKey(seed)
    x = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    params = model.init({"params": key, "dropout": key}, x, training=False)["params"]
    return params


def primitive_counts(obj, counts=None):
    counts = {} if counts is None else counts
    if hasattr(obj, "jaxpr"):
        primitive_counts(obj.jaxpr, counts)
    elif hasattr(obj, "eqns"):
        for eq in obj.eqns:
            counts[eq.primitive.name] = counts.get(eq.primitive.name, 0) + 1
            primitive_counts(eq.params, counts)
    elif isinstance(obj, dict):
        for value in obj.values():
            primitive_counts(value, counts)
    elif isinstance(obj, (tuple, list)):
        for value in obj:
            primitive_counts(value, counts)
    return counts


def graph_counts(model, params):
    x = jax.random.normal(jax.random.PRNGKey(4), (2, FRAMES, FEATURES)) * 0.1
    fn = lambda p, xx: model.apply({"params": p}, xx, training=False)["logits"]
    counts = primitive_counts(jax.make_jaxpr(fn)(params, x))
    return counts


def cost(model, params, batch=1):
    x = jnp.zeros((batch, FRAMES, FEATURES), jnp.float32)
    fn = jax.jit(lambda p, xx: model.apply({"params": p}, xx, training=False)["logits"])
    compiled = fn.lower(params, x).compile()
    ca = compiled.cost_analysis()
    if isinstance(ca, list):
        ca = ca[0] if ca else {}
    flops = float(ca.get("flops", float("nan")))
    if not np.isfinite(flops) or flops <= 0:
        raise RuntimeError(f"Invalid cost analysis: {ca}")
    return flops


def benchmark(model, params, batch, warmup, repeats):
    x = jax.random.normal(jax.random.PRNGKey(19 + batch), (batch, FRAMES, FEATURES)) * 0.1
    fn = jax.jit(lambda p, xx: model.apply({"params": p}, xx, training=False)["logits"])
    compiled = fn.lower(params, x).compile()
    for _ in range(warmup):
        jax.block_until_ready(compiled(params, x))
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        y = compiled(params, x)
        jax.block_until_ready(y)
        samples.append(time.perf_counter() - t0)
    a = np.asarray(samples) * 1000.0
    return {
        "median_ms": float(np.median(a)),
        "mean_ms": float(np.mean(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "clips_per_s": float(batch / (np.median(a) / 1000.0)),
    }


def expected_biases(dim, half_lives):
    if dim % len(half_lives):
        raise ValueError("Dimension must divide evenly across half-lives")
    per = dim // len(half_lives)
    values = []
    for h in half_lives:
        a = 0.5 ** (1.0 / float(h))
        bias = np.log(a / (1.0 - a))
        values.extend([bias] * per)
    return np.asarray(values, np.float32)


def check_timescales(params):
    flat = flatten_dict(params)
    m4_expected = expected_biases(MODEL_DIM, M4_HALF_LIVES)
    g4_expected = expected_biases(MODEL_DIM, G4_HALF_LIVES)
    found = {"m4": [], "g4": []}

    for path, value in flat.items():
        text = "/".join(map(str, path))
        if "forget" not in path or str(path[-1]) != "bias":
            continue
        arr = np.asarray(value)
        if "frame_memory_group" in path:
            if arr.shape != (4, MODEL_DIM):
                raise AssertionError(f"Unexpected M4 forget shape {arr.shape}: {text}")
            for s in range(4):
                np.testing.assert_allclose(arr[s], m4_expected, rtol=0, atol=1e-6)
            found["m4"].append(text)
        elif "descriptor_group" in path:
            if arr.shape != (4, MODEL_DIM):
                raise AssertionError(f"Unexpected G4 forget shape {arr.shape}: {text}")
            for s in range(4):
                np.testing.assert_allclose(arr[s], g4_expected, rtol=0, atol=1e-6)
            found["g4"].append(text)

    if len(found["m4"]) != 2 or len(found["g4"]) != 2:
        raise AssertionError(f"Missing expected M4/G4 forget biases: {found}")

    return {
        "m4_paths": found["m4"],
        "g4_paths": found["g4"],
        "channels_per_timescale": MODEL_DIM // 4,
        "m4_half_lives": list(M4_HALF_LIVES),
        "g4_half_lives": list(G4_HALF_LIVES),
        "same_schedule_in_all_four_streams": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--repeats", type=int, default=60)
    args = p.parse_args()

    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise RuntimeError(
            f"Expected exactly one isolated GPU; backend={jax.default_backend()} "
            f"devices={jax.local_devices()}"
        )

    d112 = make_model(112, mts=False)
    d128 = make_model(MODEL_DIM, mts=True)
    p112 = init(d112, 128)
    p128 = init(d128, 128)

    n112 = count_params(p112)
    n128 = count_params(p128)
    if n112 != 1_831_932:
        raise AssertionError(f"D112 drift: {n112}")
    if n128 != EXPECTED_PARAMS:
        raise AssertionError(f"D128 params: {n128} != {EXPECTED_PARAMS}")

    counts = graph_counts(d128, p128)
    if counts.get("scan", 0) or counts.get("while", 0):
        raise AssertionError(f"Recurrent primitive remains: {counts}")

    timescales = check_timescales(p128)

    print("=" * 112)
    print(MODEL_NAME)
    print("=" * 112)
    print("Device:", jax.local_devices()[0])
    print(f"D112 params: {n112:,}")
    print(f"D128 params: {n128:,}")
    print(f"M4 half-lives: {M4_HALF_LIVES}")
    print(f"G4 half-lives: {G4_HALF_LIVES}")
    print(f"Channels/scale/stream: {MODEL_DIM//4}")
    print("Forward recurrent scan:", counts.get("scan", 0))
    print("Forward while:", counts.get("while", 0))

    f112 = cost(d112, p112)
    f128 = cost(d128, p128)
    print("\nCOMPUTE")
    print(f"D112 Parallel-v2 : {f112/1e9:.9f} GFLOPs")
    print(f"D128 D128-MTS    : {f128/1e9:.9f} GFLOPs")
    print(f"Delta            : {(f128-f112)/1e9:+.9f} GFLOPs")
    print(f"Ratio            : {f128/f112:.4f}x")

    inference = {}
    for batch in (1, 256):
        print(f"\nTiming B={batch}...")
        a = benchmark(d112, p112, batch, args.warmup, args.repeats)
        b = benchmark(d128, p128, batch, args.warmup, args.repeats)
        inference[str(batch)] = {
            "d112": a,
            "d128_mts": b,
            "d128_over_d112_latency": b["median_ms"] / a["median_ms"],
        }
        print(
            f"D112={a['median_ms']:.4f} ms | "
            f"D128={b['median_ms']:.4f} ms | "
            f"ratio={b['median_ms']/a['median_ms']:.3f}x"
        )

    result = {
        "model": MODEL_NAME,
        "implementation": implementation_identity(),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.local_devices()],
        "params": {"d112": n112, "d128_mts": n128},
        "graph": {
            "scan_count": int(counts.get("scan", 0)),
            "while_count": int(counts.get("while", 0)),
        },
        "timescales": timescales,
        "compute": {
            "d112_gflops": f112 / 1e9,
            "d128_mts_gflops": f128 / 1e9,
            "delta_gflops": (f128 - f112) / 1e9,
            "ratio": f128 / f112,
            "convention": "same-backend XLA forward cost analysis",
        },
        "inference": inference,
    }
    atomic_json(args.output, result)
    print("\nPASS — D128-MTS hard audit complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""Hard audit for NestSAR D192 + per-stream multi-timescale memory.

No training data required. Verifies:
  * exact parameter count;
  * zero recurrent scan/while in the D192 forward graph;
  * corrected per-stream channel-wise half-life layout;
  * same-backend GPU XLA FLOPs for D112, D128, D192;
  * synchronized batch-1 / batch-256 inference latency.
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

from experiments.nestsar_sm_all_t16.model_parallel import NestSARParallelT16
from experiments.nestsar_sm_all_t16.parallel_d192_config import (
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
D112_PARAMS = 1_831_932
D128_PARAMS = 2_338_668


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
    params = model.init(
        {"params": key, "dropout": key},
        x,
        training=False,
    )["params"]
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
    return primitive_counts(jax.make_jaxpr(fn)(params, x))


def cost(model, params, batch=1):
    x = jnp.zeros((batch, FRAMES, FEATURES), jnp.float32)
    fn = jax.jit(
        lambda p, xx: model.apply({"params": p}, xx, training=False)["logits"]
    )
    compiled = fn.lower(params, x).compile()
    ca = compiled.cost_analysis()
    if isinstance(ca, list):
        ca = ca[0] if ca else {}
    flops = float(ca.get("flops", float("nan")))
    if not np.isfinite(flops) or flops <= 0:
        raise RuntimeError(f"Invalid XLA FLOP estimate: {ca}")
    return flops


def benchmark(model, params, batch, warmup, repeats):
    x = jax.random.normal(
        jax.random.PRNGKey(19 + batch),
        (batch, FRAMES, FEATURES),
    ) * 0.1

    fn = jax.jit(
        lambda p, xx: model.apply({"params": p}, xx, training=False)["logits"]
    )
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
    median = float(np.median(a))
    return {
        "median_ms": median,
        "mean_ms": float(np.mean(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "per_clip_median_ms": median / batch,
        "clips_per_s": float(batch / (median / 1000.0)),
    }


def expected_biases(dim, half_lives):
    if dim % len(half_lives):
        raise ValueError("Dimension must divide evenly across half-lives")
    per = dim // len(half_lives)
    values = []
    for h in half_lives:
        retention = 0.5 ** (1.0 / float(h))
        bias = np.log(retention / (1.0 - retention))
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
            for stream in range(4):
                np.testing.assert_allclose(
                    arr[stream], m4_expected, rtol=0, atol=1e-6
                )
            found["m4"].append(text)

        elif "descriptor_group" in path:
            if arr.shape != (4, MODEL_DIM):
                raise AssertionError(f"Unexpected G4 forget shape {arr.shape}: {text}")
            for stream in range(4):
                np.testing.assert_allclose(
                    arr[stream], g4_expected, rtol=0, atol=1e-6
                )
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

    models = {
        "d112": make_model(112, mts=False),
        "d128_mts": make_model(128, mts=True),
        "d192_mts": make_model(MODEL_DIM, mts=True),
    }
    params = {name: init(model, 128) for name, model in models.items()}
    counts = {name: count_params(params[name]) for name in models}

    expected = {
        "d112": D112_PARAMS,
        "d128_mts": D128_PARAMS,
        "d192_mts": EXPECTED_PARAMS,
    }
    if counts != expected:
        raise AssertionError(f"Parameter count mismatch: {counts} != {expected}")

    graph = graph_counts(models["d192_mts"], params["d192_mts"])
    if graph.get("scan", 0) or graph.get("while", 0):
        raise AssertionError(f"Recurrent primitive remains in D192 graph: {graph}")

    timescales = check_timescales(params["d192_mts"])

    print("=" * 118)
    print(MODEL_NAME)
    print("=" * 118)
    print("Device:", jax.local_devices()[0])
    print(f"D112 params: {counts['d112']:,}")
    print(f"D128 params: {counts['d128_mts']:,}")
    print(f"D192 params: {counts['d192_mts']:,}")
    print(f"M4 half-lives: {M4_HALF_LIVES}")
    print(f"G4 half-lives: {G4_HALF_LIVES}")
    print(f"Channels/scale/stream: {MODEL_DIM // 4}")
    print("Forward recurrent scan:", graph.get("scan", 0))
    print("Forward while:", graph.get("while", 0))

    flops = {name: cost(models[name], params[name]) for name in models}

    print("\nCOMPUTE")
    print(f"D112 Parallel-v2 : {flops['d112']/1e9:.9f} GFLOPs")
    print(f"D128 D128-MTS    : {flops['d128_mts']/1e9:.9f} GFLOPs")
    print(f"D192 D192-MTS    : {flops['d192_mts']/1e9:.9f} GFLOPs")
    print(
        f"D192 vs D128     : "
        f"{flops['d192_mts']/flops['d128_mts']:.4f}x "
        f"({100*(flops['d192_mts']/flops['d128_mts']-1):+.2f}%)"
    )
    print(
        f"D192 vs D112     : "
        f"{flops['d192_mts']/flops['d112']:.4f}x "
        f"({100*(flops['d192_mts']/flops['d112']-1):+.2f}%)"
    )

    inference = {}
    for batch in (1, 256):
        print(f"\nTiming B={batch}...")
        inference[str(batch)] = {}
        for name in ("d112", "d128_mts", "d192_mts"):
            row = benchmark(
                models[name], params[name], batch, args.warmup, args.repeats
            )
            inference[str(batch)][name] = row
            print(
                f"{name:9s} | median={row['median_ms']:.4f} ms | "
                f"p95={row['p95_ms']:.4f} ms | "
                f"{row['clips_per_s']:.1f} clips/s"
            )

        inference[str(batch)]["d192_over_d128_latency"] = (
            inference[str(batch)]["d192_mts"]["median_ms"]
            / inference[str(batch)]["d128_mts"]["median_ms"]
        )
        inference[str(batch)]["d192_over_d112_latency"] = (
            inference[str(batch)]["d192_mts"]["median_ms"]
            / inference[str(batch)]["d112"]["median_ms"]
        )

    result = {
        "model": MODEL_NAME,
        "implementation": implementation_identity(),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.local_devices()],
        "params": counts,
        "graph": {
            "scan_count": int(graph.get("scan", 0)),
            "while_count": int(graph.get("while", 0)),
        },
        "timescales": timescales,
        "compute": {
            "d112_gflops": flops["d112"] / 1e9,
            "d128_mts_gflops": flops["d128_mts"] / 1e9,
            "d192_mts_gflops": flops["d192_mts"] / 1e9,
            "d192_over_d128_ratio": flops["d192_mts"] / flops["d128_mts"],
            "d192_over_d112_ratio": flops["d192_mts"] / flops["d112"],
            "convention": "same-backend XLA forward cost analysis",
        },
        "inference": inference,
    }

    atomic_json(args.output, result)
    print("\nPASS — D192-MTS hard audit complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

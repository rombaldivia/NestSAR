#!/usr/bin/env python3
from __future__ import annotations

"""Audit the fully-parallel NestSAR T16 prototype.

Checks:
  1. Exact parameter count versus R4.
  2. Exact associative equivalence of the rank-4 fast-weight recurrence.
  3. Scan-corrected R4 GFLOPs versus parallel-model GFLOPs.
  4. End-to-end inference latency for batch 1 and batch 256.

Run on one isolated GPU.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from experiments.nestsar_sm_all_t16 import model as r4
from experiments.nestsar_sm_all_t16 import model_parallel as par
from experiments.nestsar_sm_all_t16.compute_unrolled_audit import (
    UnrolledFastWeightDeltaResidual,
    UnrolledGatedSweep,
)


EXPECTED_PARAMS = 1_831_932


def count_params(params):
    return int(
        sum(
            np.prod(x.shape)
            for x in jax.tree_util.tree_leaves(params)
        )
    )


def cost_flops(model, params, x):
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
    return float(ca.get("flops", float("nan")))


def serial_fast_reference(k, q, v, eta, alpha, memory0):
    b, _, rank = k.shape
    dim = v.shape[-1]

    mem = jnp.broadcast_to(
        memory0[None, :, :],
        (b, rank, dim),
    )

    def step(mem, inputs):
        key_t, query_t, value_t, eta_t, alpha_t = inputs
        pred_t = jnp.einsum(
            "br,brd->bd",
            key_t,
            mem,
        )
        err_t = value_t - pred_t
        delta_t = jnp.einsum(
            "br,bd->brd",
            key_t,
            err_t,
        )
        mem = (
            alpha_t[..., None] * mem
            + eta_t[..., None] * delta_t
        )
        read_t = jnp.einsum(
            "br,brd->bd",
            query_t,
            mem,
        )
        return mem, read_t

    _, reads = jax.lax.scan(
        step,
        mem,
        (
            jnp.swapaxes(k, 0, 1),
            jnp.swapaxes(q, 0, 1),
            jnp.swapaxes(v, 0, 1),
            jnp.swapaxes(eta, 0, 1),
            jnp.swapaxes(alpha, 0, 1),
        ),
    )

    return jnp.swapaxes(
        reads,
        0,
        1,
    )


def check_fast_weight_equivalence():
    key = jax.random.PRNGKey(7)
    keys = jax.random.split(key, 6)

    b, t, rank, dim = 3, 16, 4, 112

    k = jax.random.normal(keys[0], (b, t, rank))
    q = jax.random.normal(keys[1], (b, t, rank))
    v = jax.random.normal(keys[2], (b, t, dim))

    k = k / jnp.maximum(
        jnp.linalg.norm(k, axis=-1, keepdims=True),
        1e-6,
    )
    q = q / jnp.maximum(
        jnp.linalg.norm(q, axis=-1, keepdims=True),
        1e-6,
    )

    eta = 0.20 * jax.nn.sigmoid(
        jax.random.normal(keys[3], (b, t, 1))
    )
    alpha = 0.90 + 0.099 * jax.nn.sigmoid(
        jax.random.normal(keys[4], (b, t, 1))
    )
    memory0 = 0.01 * jax.random.normal(
        keys[5],
        (rank, dim),
    )

    serial = serial_fast_reference(
        k,
        q,
        v,
        eta,
        alpha,
        memory0,
    )
    parallel = par.associative_fast_weight_reads(
        k,
        q,
        v,
        eta,
        alpha,
        memory0,
    )

    diff = float(
        jnp.max(
            jnp.abs(
                serial - parallel
            )
        )
    )

    print(
        f"Fast-weight serial/parallel max |diff| = {diff:.9e}"
    )

    if diff > 2e-5:
        raise RuntimeError(
            "Associative fast-weight implementation is not numerically equivalent."
        )

    return diff


def model_from(cls):
    return cls(
        spatial_dim=24,
        model_dim=112,
        dropout=0.10,
        controller_dim=16,
        fast_rank=4,
        head_rank=2,
        sm_residual_scale=0.08,
        head_residual_scale=0.15,
    )


def init_model(model, key, batch=1):
    x = jnp.zeros(
        (
            batch,
            par.FRAMES,
            par.FEATURES,
        ),
        jnp.float32,
    )
    params = model.init(
        {
            "params": key,
            "dropout": key,
        },
        x,
        training=False,
    )["params"]
    return params, x


def scan_corrected_r4_flops(key):
    original_gated = r4.base.GatedSweep
    original_fast = r4.FastWeightDeltaResidual

    r4.base.GatedSweep = UnrolledGatedSweep
    r4.FastWeightDeltaResidual = UnrolledFastWeightDeltaResidual

    try:
        model = model_from(r4.NestSARSMAllT16)
        params, x = init_model(model, key)
        count = count_params(params)
        flops = cost_flops(
            model,
            params,
            x,
        )
        return count, flops

    finally:
        r4.base.GatedSweep = original_gated
        r4.FastWeightDeltaResidual = original_fast


def parallel_flops(key):
    model = model_from(par.NestSARParallelT16)
    params, x = init_model(model, key)
    return (
        model,
        params,
        count_params(params),
        cost_flops(model, params, x),
    )


def benchmark(model, params, batch, repeats):
    x = jnp.zeros(
        (
            batch,
            par.FRAMES,
            par.FEATURES,
        ),
        jnp.float32,
    )

    fn = jax.jit(
        lambda p, xx: model.apply(
            {"params": p},
            xx,
            training=False,
        )["logits"]
    )

    compiled = fn.lower(
        params,
        x,
    ).compile()

    for _ in range(15):
        jax.block_until_ready(
            compiled(
                params,
                x,
            )
        )

    samples = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        y = compiled(
            params,
            x,
        )
        jax.block_until_ready(y)
        samples.append(
            time.perf_counter() - t0
        )

    a = np.asarray(samples) * 1000.0

    return {
        "median_ms": float(np.median(a)),
        "mean_ms": float(np.mean(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "clips_per_s": float(
            batch / (np.median(a) / 1000.0)
        ),
    }


def main():
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"Expected GPU, got {jax.default_backend()}"
        )

    print("=" * 112)
    print("NESTSAR FULL-PARALLEL T16 AUDIT")
    print("=" * 112)
    print("JAX   :", jax.__version__)
    print("Device:", jax.local_devices()[0])

    diff = check_fast_weight_equivalence()

    key = jax.random.PRNGKey(128)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    r4_count, r4_flops = scan_corrected_r4_flops(k1)

    parallel_model, parallel_params, parallel_count, parallel_flops_value = (
        parallel_flops(k2)
    )

    print("\nCOMPUTE")
    print("-" * 112)
    print(
        f"{'R4 serial':20s} "
        f"params={r4_count:10,d} "
        f"GFLOPs={r4_flops/1e9:.9f}"
    )
    print(
        f"{'Full parallel':20s} "
        f"params={parallel_count:10,d} "
        f"GFLOPs={parallel_flops_value/1e9:.9f}"
    )
    print(
        f"Delta GFLOPs: "
        f"{(parallel_flops_value-r4_flops)/1e9:+.9f} "
        f"({100*(parallel_flops_value/r4_flops-1):+.3f}%)"
    )

    if r4_count != EXPECTED_PARAMS:
        raise RuntimeError(
            f"R4 parameter count drift: {r4_count} != {EXPECTED_PARAMS}"
        )

    if parallel_count != EXPECTED_PARAMS:
        raise RuntimeError(
            f"Parallel parameter count mismatch: "
            f"{parallel_count} != {EXPECTED_PARAMS}"
        )

    # Actual serial model for latency.
    serial_model = model_from(r4.NestSARSMAllT16)
    serial_params, _ = init_model(
        serial_model,
        k3,
    )

    print("\nLATENCY")
    print("-" * 112)

    for batch, repeats in (
        (1, 200),
        (256, 50),
    ):
        s = benchmark(
            serial_model,
            serial_params,
            batch,
            repeats,
        )
        p = benchmark(
            parallel_model,
            parallel_params,
            batch,
            repeats,
        )

        speedup = (
            s["median_ms"]
            / p["median_ms"]
        )

        print(
            f"B={batch:<3d} | "
            f"serial={s['median_ms']:.4f} ms "
            f"parallel={p['median_ms']:.4f} ms "
            f"speedup={speedup:.3f}x | "
            f"parallel={p['clips_per_s']:.1f} clips/s"
        )

    print("\nVERDICT")
    print("-" * 112)
    print(f"fast_weight_equivalence_max_abs={diff:.9e}")
    print(f"params_parallel={parallel_count:,}")
    print(f"gflops_parallel={parallel_flops_value/1e9:.9f}")
    print("=" * 112)


if __name__ == "__main__":
    main()

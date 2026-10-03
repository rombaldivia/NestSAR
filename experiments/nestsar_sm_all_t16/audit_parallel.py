#!/usr/bin/env python3
from __future__ import annotations

"""Audit the fully-parallel NestSAR T16 prototype.

Checks:
  1. Exact parameter count versus R4.
  2. Exact associative equivalence of the rank-4 fast-weight recurrence.
  3. Scan-corrected R4 GFLOPs versus parallel-model GFLOPs.
  4. Masked training, finite gradients, and recurrence-free forward graph.
  5. Device inference latency, including an identical-equation serial control.

Run on one isolated GPU.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.nestsar_sm_all_t16 import model as r4
from experiments.nestsar_sm_all_t16 import model_parallel as par
from experiments.nestsar_sm_all_t16.parallel_config import implementation_identity
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json


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
    flops = float(ca.get("flops", float("nan")))
    if not np.isfinite(flops) or flops <= 0:
        raise RuntimeError(f"Invalid XLA FLOP estimate: {ca}")
    return flops


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

    if not np.isfinite(diff) or diff > 2e-5:
        raise RuntimeError(
            "Associative fast-weight implementation is not numerically equivalent."
        )

    return diff


def model_from(cls, **kwargs):
    return cls(
        spatial_dim=24,
        model_dim=112,
        dropout=0.10,
        controller_dim=16,
        fast_rank=4,
        head_rank=2,
        sm_residual_scale=0.08,
        head_residual_scale=0.15,
        **kwargs,
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
    from experiments.nestsar_sm_all_t16.streaming.audit import audit_model
    model = model_from(r4.NestSARSMAllT16)
    params, _ = init_model(model, key)
    audit = audit_model(model, params)
    return audit['params'], audit['flops']


def parallel_flops(key):
    model = model_from(par.NestSARParallelT16)
    params, x = init_model(model, key)
    return (
        model,
        params,
        count_params(params),
        cost_flops(model, params, x),
    )


def benchmark(model, params, batch, repeats, warmup=15):
    x = jax.random.normal(jax.random.PRNGKey(23), (batch, par.FRAMES, par.FEATURES)) * .1
    jax.block_until_ready(x)

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

    for _ in range(warmup):
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

    if not np.isfinite(np.asarray(y)).all():
        raise FloatingPointError("Nonfinite benchmark predictions")
    a = np.asarray(samples) * 1000.0

    return {
        "median_ms": float(np.median(a)),
        "mean_ms": float(np.mean(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "clips_per_s": float(
            batch / (np.median(a) / 1000.0)
        ),
    }


def primitive_counts(obj, counts=None):
    counts = {} if counts is None else counts
    if hasattr(obj, 'jaxpr'):
        primitive_counts(obj.jaxpr, counts)
    elif hasattr(obj, 'eqns'):
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


def check_parallel_graph_and_outputs():
    model = model_from(par.NestSARParallelT16)
    serial = model_from(par.NestSARParallelT16, parallel=False)
    x = jax.random.normal(jax.random.PRNGKey(81), (2, 16, 750)) * .1
    x = x.reshape(2, 16, 2, 25, 15).at[0, :, 1].set(0).at[1].set(0).reshape(2, 16, 750)
    params = model.init(jax.random.PRNGKey(128), x)['params']
    fn = lambda m, p, x: m.apply({'params': p}, x, training=False)['logits']
    a, b = fn(model, params, x), fn(serial, params, x)
    if not np.isfinite(np.asarray(a)).all():
        raise FloatingPointError("Nonfinite masked/empty-clip predictions")
    np.testing.assert_allclose(a, b, rtol=3e-4, atol=3e-5, equal_nan=False)
    counts = primitive_counts(jax.make_jaxpr(lambda p, x: fn(model, p, x))(params, x))
    serial_counts = primitive_counts(jax.make_jaxpr(lambda p, x: fn(serial, p, x))(params, x))
    if counts.get('scan', 0) or counts.get('while', 0):
        raise AssertionError(f"Token recurrence remains in parallel graph: {counts}")
    if not serial_counts.get('scan', 0):
        raise AssertionError("Serial timing control does not contain recurrent scans")
    if count_params(params) != EXPECTED_PARAMS:
        raise AssertionError("Parallel parameter count changed")
    return {'params': count_params(params), 'scan_count': 0, 'while_count': 0,
            'serial_control_scan_count': serial_counts['scan'],
            'same_equation_forward_max_abs': float(jnp.max(jnp.abs(a-b)))}


def check_padded_training():
    """Regression for NaNs from a masked zero sample at fresh initialization."""
    from experiments.nestsar_sm_all_t16.streaming import worker, worker_parallel
    from experiments.nestsar_sm_all_t16.streaming.launch import validate_config
    c = validate_config(dict(epochs=3, micro_batch=2, accumulation_steps=1, eval_batch=2))
    model, state, key, _, _ = worker.create_state(c, 1, worker_parallel.make_model)
    original = state.params
    train, evaluate = worker.build_steps(model, c)
    x = jax.random.normal(jax.random.PRNGKey(82), (2, 16, 750)) * .1
    x = x.at[1].set(0)
    batch = dict(x=x, xa=x*.97, y=jnp.array([3, 0]), mask=jnp.array([1., 0.]))
    for _ in range(2):
        state, key, metrics = jax.block_until_ready(train(state, key, batch))
        for value in [metrics, *jax.tree.leaves(state)]:
            if not np.isfinite(np.asarray(value)).all():
                raise FloatingPointError("Masked padding produced nonfinite training state")
    if not any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(original), jax.tree.leaves(state.params))):
        raise AssertionError("Training did not update the parameters")
    if not np.isfinite(np.asarray(evaluate(state.ema_params, batch))).all():
        raise FloatingPointError("Nonfinite EMA evaluation")
    return {'padded_training_steps': int(state.step), 'finite_optimizer_and_ema': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true", help="Synthetic validation; no GPU speed claim")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 256])
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--output", help="Save machine-readable audit results")
    args = parser.parse_args()
    if min([args.repeats, args.warmup, *args.batches]) < 1:
        raise ValueError("Batch sizes, warmup, and repeats must be positive")
    if jax.default_backend() != "gpu" and not args.allow_cpu:
        raise RuntimeError(f"Expected GPU, got {jax.default_backend()}; --allow-cpu is for synthetic checks")
    if len(jax.local_devices()) != 1:
        raise RuntimeError("Run this audit on exactly one isolated device")
    source = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]),
                             "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]),
                            "status", "--porcelain"], capture_output=True, text=True)
    report = dict(backend=jax.default_backend(), devices=[str(d) for d in jax.local_devices()],
                  jax=jax.__version__, source_commit=source.stdout.strip(),
                  source_tree_dirty=bool(dirty.stdout.strip()),
                  implementation=implementation_identity(),
                  real_ntu_accuracy_measured=False, gpu_speed_measured=jax.default_backend()=="gpu",
                  timings_exclude_compile_preprocessing_and_transfer=True)
    print("Checking fast-memory equivalence, graph, and masked training...", flush=True)
    report['fast_weight_max_abs'] = check_fast_weight_equivalence()
    report['graph'] = check_parallel_graph_and_outputs()
    report['training'] = check_padded_training()
    jax.clear_caches()
    print("Auditing matched-backend forward compute...", flush=True)
    key = jax.random.PRNGKey(128)
    r4_count, r4_flops = scan_corrected_r4_flops(key)
    parallel_model, parallel_params, parallel_count, parallel_flops_value = parallel_flops(key)
    if r4_count != EXPECTED_PARAMS or parallel_count != EXPECTED_PARAMS:
        raise AssertionError("Parameter counts differ from R4")
    report['compute'] = dict(r4_gflops=r4_flops/1e9, parallel_gflops=parallel_flops_value/1e9,
                            delta_percent=100*(parallel_flops_value/r4_flops-1),
                            convention="XLA forward estimate; R4 scans unrolled and verified; 1 MAC=2 FLOPs")
    original = model_from(r4.NestSARSMAllT16)
    original_params, _ = init_model(original, key)
    serial_control = model_from(par.NestSARParallelT16, parallel=False)
    report['inference'] = {}
    for batch in args.batches:
        print(f"Timing batch {batch}...", flush=True)
        old = benchmark(original, original_params, batch, args.repeats, args.warmup)
        serial = benchmark(serial_control, parallel_params, batch, args.repeats, args.warmup)
        parallel = benchmark(parallel_model, parallel_params, batch, args.repeats, args.warmup)
        report['inference'][str(batch)] = dict(
            r4=old, same_equation_serial=serial, parallel=parallel,
            same_equation_speedup=serial['median_ms']/parallel['median_ms'],
            vs_r4_speedup=old['median_ms']/parallel['median_ms'])
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

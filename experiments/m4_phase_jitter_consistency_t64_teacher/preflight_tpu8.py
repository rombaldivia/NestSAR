#!/usr/bin/env python3
from __future__ import annotations

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from experiments.m4_phase_jitter_consistency_t64_teacher import train_tpu as tr
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_t16 import train_tpu as cons


def main() -> None:
    tr.install_overrides()
    print("=" * 120)
    print("M4-PHASE-JITTER-CONSISTENCY-T64-TEACHER | STRICT TPU8 PREFLIGHT")
    print("=" * 120)
    print(f"JAX_VERSION={jax.__version__}")
    print(f"BACKEND={jax.default_backend()}")
    print(f"LOCAL_DEVICE_COUNT={jax.local_device_count()}")
    print(f"DEVICES={jax.local_devices()}")
    if jax.default_backend() != "tpu" or jax.local_device_count() != 8:
        raise RuntimeError("Expected TPU v5e-8")

    model = tr.M4PhaseUniformT64Teacher(24, 112, 0.10)
    key = jax.random.PRNGKey(128)
    dummy = jnp.zeros((1, tr.FRAMES, tr.FEATURES), jnp.float32)
    params = model.init({"params": key, "dropout": key}, dummy, training=False)["params"]
    nparams = ju.count_params(params)
    print(f"MODEL_INIT=PASS | PARAMS={nparams:,} | INPUT=(B,{tr.FRAMES},{tr.FEATURES})")
    if nparams != tr.EXPECTED_PARAMS:
        raise RuntimeError(f"Parameter mismatch: {nparams:,} != {tr.EXPECTED_PARAMS:,}")
    print(f"PARAMETER_AUDIT=PASS | EXPECTED={tr.EXPECTED_PARAMS:,}")

    out = model.apply({"params": params}, dummy, training=False)
    fw = np.asarray(out["fusion_weights"])
    chunks = np.asarray(out["chunk_states"])
    if not np.allclose(fw, 0.25):
        raise RuntimeError(f"Uniform fusion failed: {fw}")
    if chunks.shape[2] != tr.CHUNKS:
        raise RuntimeError(f"Chunk hierarchy failed: shape={chunks.shape}")
    print(f"UNIFORM_FUSION=PASS | weights={fw[0].tolist()}")
    print(f"T64_HIERARCHY=PASS | frame_tokens={tr.FRAMES} | chunks={tr.CHUNKS}x{tr.CHUNK_SIZE}")

    devices = list(jax.local_devices())
    reps = jax.device_put_replicated(params, devices)
    rngs = jax.random.split(jax.random.PRNGKey(7), len(devices))
    xcan = jnp.zeros((len(devices), 1, tr.FRAMES, tr.FEATURES), jnp.float32)
    xjit = xcan.at[:, :, 1, 0].set(0.01)
    y = jnp.zeros((len(devices), 1), jnp.int32)

    @partial(jax.pmap, axis_name="d", devices=devices)
    def probe(p, rng, a, b, yb):
        rng, r1, r2 = jax.random.split(rng, 3)
        def loss_fn(pp):
            oa = model.apply({"params": pp}, a, training=True, rngs={"dropout": r1})
            ob = model.apply({"params": pp}, b, training=True, rngs={"dropout": r2})
            ce = jnp.mean(ju.smooth_ce(oa["logits"], yb, 0.05))
            aux = jnp.mean(ju.smooth_ce(
                oa["stream_logits"].reshape(-1, tr.NUM_CLASSES),
                jnp.repeat(yb, tr.NUM_STREAMS),
                0.05,
            ))
            skl = cons.symmetric_kl(oa["logits"], ob["logits"], 1.0)
            return ce + 0.15 * aux + 0.08 * skl
        loss, grads = jax.value_and_grad(loss_fn)(p)
        grads = jax.lax.pmean(grads, "d")
        gn = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)))
        return jax.lax.pmean(loss, "d"), gn

    loss, gn = probe(reps, rngs, xcan, xjit, y)
    print(
        "PMAP_DUAL_VIEW_FORWARD_BACKWARD=PASS | 8/8 cores | "
        f"loss={float(np.asarray(loss[0])):.6f} | grad_norm={float(np.asarray(gn[0])):.6f}"
    )

    dataset = ju.base.find_dataset(None)
    anns, split = ju.base.load_ntu(dataset)
    print(f"DATASET_FOUND={dataset}")
    for protocol in ("xsub", "xset"):
        _, train_ids, val_ids = ju.resolve_protocol_ids(anns, split, protocol)
        print(f"{protocol.upper()}_SPLIT=PASS | train={len(train_ids):,} | val={len(val_ids):,}")

    sample = next(a for a in anns if isinstance(a, dict))
    kp = ju.base.annotation_keypoints(sample)
    can = tr.segment_phase_tokens64(kp)
    rng = np.random.default_rng(128)
    jit = tr.jitter_phase_tokens64(kp, 1, rng)
    if can.shape != (tr.FRAMES, tr.FEATURES):
        raise RuntimeError(f"Bad canonical representation: {can.shape}")
    if not np.all(np.isfinite(can)) or not np.all(np.isfinite(jit)):
        raise RuntimeError("Non-finite T64 representation")
    print(
        "REAL_T64_PHASE_VIEW=PASS | "
        f"shape={can.shape} | mean_abs_jitter_delta={np.mean(np.abs(can-jit)):.8f}"
    )

    flops = ju.audit_flops(model, params)
    if flops is not None:
        print(f"XLA_GFLOPS_EXACT={flops / 1e9:.9f}")

    xsub_cache = (63026 * 2 + 50919) * tr.FRAMES * tr.FEATURES * np.dtype(tr.CACHE_DTYPE).itemsize
    xset_cache = (54468 * 2 + 59477) * tr.FRAMES * tr.FEATURES * np.dtype(tr.CACHE_DTYPE).itemsize
    print(f"HOST_CACHE_ESTIMATE_XSUB={xsub_cache / 1024**3:.2f} GiB")
    print(f"HOST_CACHE_ESTIMATE_XSET={xset_cache / 1024**3:.2f} GiB")
    print("STRICT_PREFLIGHT=PASS")


if __name__ == "__main__":
    main()

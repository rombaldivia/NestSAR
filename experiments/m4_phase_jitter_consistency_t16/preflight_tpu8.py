#!/usr/bin/env python3
from __future__ import annotations

from functools import partial
import numpy as np

from experiments.m4_phase_jitter_consistency_t16 import train_tpu as tr

jax = tr.jax
jnp = tr.jnp

EXPECTED_DEVICES = 8
SPATIAL_DIM = 24
MODEL_DIM = 112
DROPOUT = 0.10
SEED = 128


def main() -> int:
    print("=" * 120, flush=True)
    print("M4-PHASE-JITTER-CONSISTENCY-T16 | TPU8 PREFLIGHT", flush=True)
    print("=" * 120, flush=True)

    backend = jax.default_backend()
    devices = list(jax.local_devices())
    print(f"JAX_VERSION={jax.__version__}", flush=True)
    print(f"BACKEND={backend}", flush=True)
    print(f"LOCAL_DEVICE_COUNT={len(devices)}", flush=True)
    if backend != "tpu" or len(devices) != EXPECTED_DEVICES:
        raise RuntimeError(f"Expected TPU8, got backend={backend} devices={len(devices)}")

    model = tr.ju.M4PhaseUniformT16(SPATIAL_DIM, MODEL_DIM, DROPOUT)
    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)
    dummy = jnp.zeros((1, tr.FRAMES, tr.FEATURES), jnp.float32)
    params = model.init({"params": init_key, "dropout": init_key}, dummy, training=False)["params"]
    nparams = tr.ju.count_params(params)
    print(f"MODEL_INIT=PASS | PARAMS={nparams:,} | INPUT=(B,{tr.FRAMES},{tr.FEATURES})", flush=True)
    if nparams != tr.EXPECTED_PARAMS:
        raise RuntimeError(f"Expected {tr.EXPECTED_PARAMS:,} params, got {nparams:,}")
    print(f"PARAMETER_AUDIT=PASS | EXPECTED={tr.EXPECTED_PARAMS:,}", flush=True)

    # Pure consistency math check.
    a = jnp.asarray([[2.0, 0.0, -1.0]], jnp.float32)
    b = jnp.asarray([[1.5, 0.4, -0.8]], jnp.float32)
    c_same = float(tr.symmetric_kl(a, a))
    c_diff = float(tr.symmetric_kl(a, b))
    if not np.isfinite(c_same) or not np.isfinite(c_diff):
        raise RuntimeError("Consistency loss is non-finite")
    if abs(c_same) > 1e-6 or c_diff <= 0.0:
        raise RuntimeError(f"Bad symmetric KL: same={c_same} diff={c_diff}")
    print(f"SYMMETRIC_KL=PASS | identical={c_same:.8f} | different={c_diff:.8f}", flush=True)

    per_device_batch = 2
    shape = (EXPECTED_DEVICES, per_device_batch, tr.FRAMES, tr.FEATURES)
    xcan = np.zeros(shape, np.float32)
    xjit = np.zeros(shape, np.float32)
    # Make the second view genuinely different without affecting shapes.
    xjit[..., 6:12] = 1e-3
    yb = np.zeros((EXPECTED_DEVICES, per_device_batch), np.int32)
    rngs = jax.random.split(key, EXPECTED_DEVICES)
    params_repl = jax.device_put_replicated(params, devices)

    @partial(jax.pmap, axis_name="d", devices=devices)
    def smoke_step(p, rng, xc, xj, y):
        rng, dc, dj = jax.random.split(rng, 3)

        def loss_fn(pp):
            oc = model.apply({"params": pp}, xc, training=True, rngs={"dropout": dc})
            oj = model.apply({"params": pp}, xj, training=True, rngs={"dropout": dj})
            ce = 0.5 * (
                jnp.mean(tr.ju.smooth_ce(oc["logits"], y, 0.05))
                + jnp.mean(tr.ju.smooth_ce(oj["logits"], y, 0.05))
            )
            cons = tr.symmetric_kl(oc["logits"], oj["logits"], 1.0)
            return ce + 0.08 * cons, cons

        (loss, cons), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        grads = jax.lax.pmean(grads, "d")
        loss = jax.lax.pmean(loss, "d")
        cons = jax.lax.pmean(cons, "d")
        grad_sq = sum(jnp.sum(jnp.square(g)).astype(jnp.float32) for g in jax.tree_util.tree_leaves(grads))
        return loss, cons, jnp.sqrt(grad_sq)

    loss, cons, grad = smoke_step(params_repl, rngs, xcan, xjit, yb)
    jax.block_until_ready(loss)
    loss_np = np.asarray(jax.device_get(loss))
    cons_np = np.asarray(jax.device_get(cons))
    grad_np = np.asarray(jax.device_get(grad))
    if not (np.all(np.isfinite(loss_np)) and np.all(np.isfinite(cons_np)) and np.all(np.isfinite(grad_np))):
        raise RuntimeError("Non-finite PMAP consistency smoke test")
    print(
        f"PMAP_CANONICAL_JITTER_BACKWARD=PASS | 8/8 cores | "
        f"loss={float(loss_np[0]):.6f} | cons={float(cons_np[0]):.6f} | grad_norm={float(grad_np[0]):.6f}",
        flush=True,
    )

    dataset = tr.ju.base.find_dataset(None)
    print(f"DATASET_FOUND={dataset}", flush=True)
    anns, split = tr.ju.base.load_ntu(dataset)
    for protocol in ("xsub", "xset"):
        tk, vk = tr.ju.base.resolve_split(split, protocol)
        print(f"{protocol.upper()}_SPLIT=PASS | train={len(split[tk]):,} | val={len(split[vk]):,}", flush=True)

    sample = next((a for a in anns if isinstance(a, tr.ju.Mapping)), None)
    if sample is None:
        raise RuntimeError("No mapping-style NTU sample found")
    kp = tr.ju.base.annotation_keypoints(sample)
    can = tr.ju.phase.segment_phase_tokens(kp)
    rng = np.random.default_rng(9173)
    jit = tr.ju.jitter_phase_tokens(kp, 1, rng)
    if can.shape != (tr.FRAMES, tr.FEATURES) or jit.shape != can.shape:
        raise RuntimeError(f"Unexpected view shapes: canonical={can.shape}, jitter={jit.shape}")
    delta = float(np.mean(np.abs(can - jit)))
    print(f"REAL_CANONICAL_JITTER_PAIR=PASS | mean_abs_delta={delta:.8f}", flush=True)

    print("STRICT_PREFLIGHT=PASS", flush=True)
    print("=" * 120, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

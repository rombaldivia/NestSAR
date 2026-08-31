#!/usr/bin/env python3
from __future__ import annotations

from functools import partial
import numpy as np

from experiments.m4_phasepath_jitter_uniform_t16 import train_tpu as tr

jax = tr.jax
jnp = tr.jnp
EXPECTED_DEVICES = 8


def main() -> int:
    print("=" * 120, flush=True)
    print("M4-PHASEPATH-JITTER-UNIFORM-T16 | STRICT TPU8 PREFLIGHT", flush=True)
    print("=" * 120, flush=True)

    backend = jax.default_backend()
    devices = list(jax.local_devices())
    print(f"JAX_VERSION={jax.__version__}", flush=True)
    print(f"BACKEND={backend}", flush=True)
    print(f"LOCAL_DEVICE_COUNT={len(devices)}", flush=True)
    print(f"DEVICES={devices}", flush=True)
    if backend != "tpu" or len(devices) != EXPECTED_DEVICES:
        raise RuntimeError(
            f"Expected TPU with 8 local devices; got backend={backend}, devices={len(devices)}"
        )

    model = tr.M4PhasePathUniformT16(24, 112, 0.10)
    key = jax.random.PRNGKey(128)
    key, init_key = jax.random.split(key)
    dummy = jnp.zeros((1, tr.FRAMES, tr.FEATURES), jnp.float32)
    params = model.init(
        {"params": init_key, "dropout": init_key},
        dummy,
        training=False,
    )["params"]
    nparams = tr.ju.count_params(params)
    print(
        f"MODEL_INIT=PASS | PARAMS={nparams:,} | "
        f"INPUT=(B,{tr.FRAMES},{tr.FEATURES})",
        flush=True,
    )
    if nparams != tr.EXPECTED_PARAMS:
        raise RuntimeError(
            f"Expected {tr.EXPECTED_PARAMS:,} params, got {nparams:,}"
        )
    print(
        f"PARAMETER_AUDIT=PASS | EXPECTED={tr.EXPECTED_PARAMS:,}",
        flush=True,
    )

    out0 = model.apply({"params": params}, dummy, training=False)
    fw = np.asarray(jax.device_get(out0["fusion_weights"]))
    if not np.allclose(fw, 0.25, atol=1e-7):
        raise RuntimeError(f"Uniform fusion failed: {fw}")
    print(
        f"UNIFORM_FUSION=PASS | weights={fw[0].tolist()}",
        flush=True,
    )

    ndev = len(devices)
    per_device_batch = 2
    xb = np.zeros(
        (ndev, per_device_batch, tr.FRAMES, tr.FEATURES), np.float32
    )
    yb = np.zeros((ndev, per_device_batch), np.int32)
    rngs = jax.random.split(key, ndev)
    repl = jax.device_put_replicated(params, devices)

    @partial(jax.pmap, axis_name="d", devices=devices)
    def smoke(p, rng, x, y):
        rng, drop = jax.random.split(rng)

        def loss_fn(pp):
            out = model.apply(
                {"params": pp},
                x,
                training=True,
                rngs={"dropout": drop},
            )
            main = jnp.mean(tr.ju.smooth_ce(out["logits"], y, 0.05))
            sl = out["stream_logits"]
            aux = jnp.mean(
                tr.ju.smooth_ce(
                    sl.reshape(-1, tr.NUM_CLASSES),
                    jnp.repeat(y, tr.NUM_STREAMS),
                    0.05,
                )
            )
            return main + 0.15 * aux, out["logits"]

        (loss, logits), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(p)
        grads = jax.lax.pmean(grads, "d")
        loss = jax.lax.pmean(loss, "d")
        grad_sq = sum(
            jnp.sum(jnp.square(g)).astype(jnp.float32)
            for g in jax.tree_util.tree_leaves(grads)
        )
        return rng, loss, jnp.sqrt(grad_sq), logits

    rngs, loss, grad_norm, logits = smoke(repl, rngs, xb, yb)
    jax.block_until_ready(logits)
    loss_np = np.asarray(jax.device_get(loss))
    grad_np = np.asarray(jax.device_get(grad_norm))
    logits_np = np.asarray(jax.device_get(logits))

    if logits_np.shape != (ndev, per_device_batch, tr.NUM_CLASSES):
        raise RuntimeError(f"Bad logits shape {logits_np.shape}")
    if (
        not np.all(np.isfinite(loss_np))
        or not np.all(np.isfinite(grad_np))
        or not np.all(np.isfinite(logits_np))
    ):
        raise RuntimeError("Non-finite smoke-test values")
    print(
        "PMAP_FORWARD_BACKWARD=PASS | 8/8 cores | "
        f"loss={float(loss_np[0]):.6f} | grad_norm={float(grad_np[0]):.6f}",
        flush=True,
    )

    dataset = tr.base.find_dataset(None)
    print(f"DATASET_FOUND={dataset}", flush=True)
    annotations, split = tr.base.load_ntu(dataset)
    for protocol in ("xsub", "xset"):
        tk, vk = tr.base.resolve_split(split, protocol)
        print(
            f"{protocol.upper()}_SPLIT=PASS | "
            f"train={len(split[tk]):,} | val={len(split[vk]):,}",
            flush=True,
        )

    sample = next(a for a in annotations if isinstance(a, tr.Mapping))
    kp = tr.base.annotation_keypoints(sample)
    canonical = tr.segment_phasepath_tokens(kp)
    rng = np.random.default_rng(20260830)
    jittered = tr.jitter_phasepath_tokens(kp, 1, rng)

    if canonical.shape != (tr.FRAMES, tr.FEATURES):
        raise RuntimeError(f"Unexpected canonical shape {canonical.shape}")
    if jittered.shape != canonical.shape:
        raise RuntimeError(f"Unexpected jittered shape {jittered.shape}")
    if not np.all(np.isfinite(canonical)) or not np.all(np.isfinite(jittered)):
        raise RuntimeError("Non-finite PhasePath tokens")

    tok = canonical.reshape(
        tr.FRAMES, tr.PERSONS, tr.JOINTS, tr.TOKEN_CHANNELS
    )
    full_disp = tok[..., 3:6]
    phase_a = tok[..., 6:9]
    phase_b = tok[..., 9:12]
    path_a = tok[..., 12:15]
    path_b = tok[..., 15:18]

    phase_identity_error = float(
        np.max(np.abs(full_disp - (phase_a + phase_b)))
    )
    if phase_identity_error > 1e-5:
        raise RuntimeError(
            f"Phase identity failed: {phase_identity_error:.3e}"
        )
    if np.min(path_a) < -1e-7 or np.min(path_b) < -1e-7:
        raise RuntimeError("Path channels must be non-negative")

    delta = float(np.mean(np.abs(canonical - jittered)))
    path_a_energy = float(np.mean(np.abs(path_a)))
    path_b_energy = float(np.mean(np.abs(path_b)))

    print(
        "REAL_PHASEPATH_JITTER_VIEW=PASS | "
        f"shape={canonical.shape} | mean_abs_delta={delta:.8f} | "
        f"phase_identity_error={phase_identity_error:.3e} | "
        f"pathA_energy={path_a_energy:.6f} | pathB_energy={path_b_energy:.6f}",
        flush=True,
    )

    print("STRICT_PREFLIGHT=PASS", flush=True)
    print("=" * 120, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

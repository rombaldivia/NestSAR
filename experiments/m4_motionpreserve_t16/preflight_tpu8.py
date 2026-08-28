#!/usr/bin/env python3
from __future__ import annotations

from functools import partial
import numpy as np

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as tr

jax = tr.jax
jnp = tr.jnp

EXPECTED_DEVICES = 8
SPATIAL_DIM = 24
MODEL_DIM = 112
DROPOUT = 0.10
SEED = 128


def main() -> int:
    print("=" * 116, flush=True)
    print("M4-MOTIONPRESERVE-T16 | ISOLATED TPU8 PREFLIGHT", flush=True)
    print("=" * 116, flush=True)

    backend = jax.default_backend()
    devices = list(jax.local_devices())
    ndev = len(devices)
    print(f"JAX_VERSION={jax.__version__}", flush=True)
    print(f"BACKEND={backend}", flush=True)
    print(f"LOCAL_DEVICE_COUNT={ndev}", flush=True)
    print(f"DEVICES={devices}", flush=True)

    if backend != "tpu":
        raise RuntimeError(f"Expected TPU backend, got {backend!r}")
    if ndev != EXPECTED_DEVICES:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_DEVICES} local TPU devices, found {ndev}"
        )

    model = tr.M4MotionPreserveT16(
        spatial_dim=SPATIAL_DIM,
        model_dim=MODEL_DIM,
        dropout=DROPOUT,
    )
    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)
    dummy = jnp.zeros((1, tr.FRAMES, tr.FEATURES), jnp.float32)
    params = model.init(
        {"params": init_key, "dropout": init_key},
        dummy,
        training=False,
    )["params"]
    nparams = tr.count_params(params)
    print(
        f"MODEL_INIT=PASS | PARAMS={nparams:,} | "
        f"INPUT=(B,{tr.FRAMES},{tr.FEATURES})",
        flush=True,
    )

    per_device_batch = 2
    xb = np.zeros(
        (ndev, per_device_batch, tr.FRAMES, tr.FEATURES),
        dtype=np.float32,
    )
    yb = np.zeros((ndev, per_device_batch), dtype=np.int32)
    rngs = jax.random.split(key, ndev)
    params_repl = jax.device_put_replicated(params, devices)

    @partial(jax.pmap, axis_name="d", devices=devices)
    def smoke_step(p, rng, x, y):
        rng, drop = jax.random.split(rng)

        def loss_fn(pp):
            out = model.apply(
                {"params": pp},
                x,
                training=True,
                rngs={"dropout": drop},
            )
            main = jnp.mean(tr.smooth_ce(out["logits"], y, 0.05))
            sl = out["stream_logits"]
            aux = jnp.mean(
                tr.smooth_ce(
                    sl.reshape(-1, tr.NUM_CLASSES),
                    jnp.repeat(y, tr.NUM_STREAMS),
                    0.05,
                )
            )
            return main + 0.15 * aux, (
                out["logits"],
                out["router_weights"],
            )

        (loss, (logits, router)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(p)
        grads = jax.lax.pmean(grads, "d")
        loss = jax.lax.pmean(loss, "d")
        grad_sq = sum(
            jnp.sum(jnp.square(g)).astype(jnp.float32)
            for g in jax.tree_util.tree_leaves(grads)
        )
        return rng, loss, jnp.sqrt(grad_sq), logits, router

    rngs, loss, grad_norm, logits, router = smoke_step(
        params_repl, rngs, xb, yb
    )
    jax.block_until_ready(logits)
    loss_np = np.asarray(jax.device_get(loss))
    grad_np = np.asarray(jax.device_get(grad_norm))
    logits_np = np.asarray(jax.device_get(logits))
    router_np = np.asarray(jax.device_get(router))

    expected_logits = (ndev, per_device_batch, tr.NUM_CLASSES)
    expected_router = (
        ndev,
        per_device_batch,
        tr.FRAMES,
        tr.NUM_STREAMS,
    )
    if logits_np.shape != expected_logits:
        raise RuntimeError(
            f"Unexpected logits shape {logits_np.shape}; expected {expected_logits}"
        )
    if router_np.shape != expected_router:
        raise RuntimeError(
            f"Unexpected router shape {router_np.shape}; expected {expected_router}"
        )
    if not np.all(np.isfinite(loss_np)):
        raise RuntimeError(f"Non-finite loss: {loss_np}")
    if not np.all(np.isfinite(grad_np)):
        raise RuntimeError(f"Non-finite grad norm: {grad_np}")
    if not np.all(np.isfinite(logits_np)):
        raise RuntimeError("Non-finite logits")

    print(
        "PMAP_FORWARD_BACKWARD=PASS | "
        f"{ndev}/{EXPECTED_DEVICES} cores | "
        f"loss={float(loss_np[0]):.6f} | "
        f"grad_norm={float(grad_np[0]):.6f}",
        flush=True,
    )

    dataset = tr.find_dataset(None)
    print(f"DATASET_FOUND={dataset}", flush=True)
    annotations, split = tr.load_ntu(dataset)
    if not annotations:
        raise RuntimeError("NTU120 annotations are empty")

    for protocol in ("xsub", "xset"):
        tk, vk = tr.resolve_split(split, protocol)
        train_count = len(split[tk])
        val_count = len(split[vk])
        if train_count <= 0 or val_count <= 0:
            raise RuntimeError(
                f"Empty {protocol} split: train={train_count} val={val_count}"
            )
        print(
            f"{protocol.upper()}_SPLIT=PASS | "
            f"train={train_count:,} | val={val_count:,}",
            flush=True,
        )

    sample = next(
        (a for a in annotations if isinstance(a, tr.Mapping)),
        None,
    )
    if sample is None:
        raise RuntimeError("Could not find a mapping-style NTU annotation")

    sample_x = tr.preprocess_keypoints(
        tr.annotation_keypoints(sample), "segment"
    )
    sample_y = int(tr.annotation_label(sample))
    if sample_x.shape != (tr.FRAMES, tr.FEATURES):
        raise RuntimeError(f"Unexpected sample shape: {sample_x.shape}")
    if not np.all(np.isfinite(sample_x)):
        raise RuntimeError("Preprocessed sample contains NaN/Inf")
    if not 0 <= sample_y < tr.NUM_CLASSES:
        raise RuntimeError(f"Sample label out of range: {sample_y}")

    token = sample_x.reshape(
        tr.FRAMES, tr.PERSONS, tr.JOINTS, tr.TOKEN_CHANNELS
    )
    motion_energy = float(np.mean(np.abs(token[..., 3:9])))
    print(
        f"REAL_SEGMENT_TOKEN=PASS | shape={sample_x.shape} | "
        f"label={sample_y} | motion_energy={motion_energy:.6f}",
        flush=True,
    )
    print("STRICT_PREFLIGHT=PASS", flush=True)
    print("=" * 116, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

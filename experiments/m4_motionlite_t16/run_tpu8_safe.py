#!/usr/bin/env python3
from __future__ import annotations

"""Safe first-run launcher for one Kaggle TPU v5e-8.

One Kaggle TPU accelerator should expose 8 local JAX TPU devices. This launcher
uses exactly one Python process and all 8 local devices with pmap. Before any
long preprocessing/training it performs:
  1) backend/device-count validation,
  2) model init on T16,
  3) real pmap forward + backward + cross-device gradient reduction,
  4) finite-value/shape validation,
  5) NTU120 dataset discovery, split validation, and one-sample preprocessing.

Only after every preflight passes does it enter the normal XSUB -> XSET trainer.
"""

import importlib.util
import sys
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRAINER_PATH = HERE / "train_m4_motionlite_t16_tpu.py"

spec = importlib.util.spec_from_file_location("m4_motionlite_t16_trainer", TRAINER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import trainer: {TRAINER_PATH}")
tr = importlib.util.module_from_spec(spec)
# Python 3.12 + Flax dataclass processing requires dynamically imported modules
# to be registered before exec_module(). Without this, Flax's @nn.Module
# dataclass transform sees sys.modules[cls.__module__] == None.
sys.modules[spec.name] = tr
spec.loader.exec_module(tr)

jax = tr.jax
jnp = tr.jnp

EXPECTED_LOCAL_DEVICES = 8
SPATIAL_DIM = 24
MODEL_DIM = 112
DROPOUT = 0.10
SEED = 128


def banner(title: str) -> None:
    print("=" * 112, flush=True)
    print(title, flush=True)
    print("=" * 112, flush=True)


def strict_tpu_preflight() -> None:
    banner("NESTSAR M4-MOTIONLITE-T16 — SINGLE TPU / 8-CORE STRICT PREFLIGHT")

    backend = jax.default_backend()
    devices = jax.local_devices()
    ndev = len(devices)

    print(f"JAX_VERSION={jax.__version__}", flush=True)
    print(f"BACKEND={backend}", flush=True)
    print(f"LOCAL_DEVICE_COUNT={ndev}", flush=True)
    print(f"DEVICES={devices}", flush=True)

    if backend != "tpu":
        raise RuntimeError(
            "TPU backend is not active. In Kaggle select Accelerator -> TPU VM / TPU v5e-8 "
            "and restart the notebook session before running this cell."
        )
    if ndev != EXPECTED_LOCAL_DEVICES:
        raise RuntimeError(
            f"This safe launcher expects one Kaggle v5e-8 exposing exactly "
            f"{EXPECTED_LOCAL_DEVICES} local TPU devices; JAX reports {ndev}."
        )
    if any(getattr(d, "platform", "") != "tpu" for d in devices):
        raise RuntimeError(f"Non-TPU device found in local device list: {devices}")

    model = tr.M4MotionLiteT16(
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
    print(f"MODEL_INIT=PASS | PARAMS={nparams:,} | INPUT=(B,{tr.FRAMES},{tr.FEATURES})", flush=True)

    per_device_batch = 2
    xb = np.zeros(
        (ndev, per_device_batch, tr.FRAMES, tr.FEATURES),
        dtype=np.float32,
    )
    yb = np.zeros((ndev, per_device_batch), dtype=np.int32)
    rngs = jax.random.split(key, ndev)
    params_repl = jax.device_put_replicated(params, devices)

    @partial(jax.pmap, axis_name="d")
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
            stream_logits = out["stream_logits"]
            aux = jnp.mean(
                tr.smooth_ce(
                    stream_logits.reshape(-1, tr.NUM_CLASSES),
                    jnp.repeat(y, tr.NUM_STREAMS),
                    0.05,
                )
            )
            loss = main + 0.15 * aux
            return loss, out["logits"]

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        grads = jax.lax.pmean(grads, "d")
        loss = jax.lax.pmean(loss, "d")
        grad_sq = sum(
            jnp.sum(jnp.square(g)).astype(jnp.float32)
            for g in jax.tree_util.tree_leaves(grads)
        )
        grad_norm = jnp.sqrt(grad_sq)
        return rng, loss, grad_norm, logits

    rngs, loss, grad_norm, logits = smoke_step(params_repl, rngs, xb, yb)
    loss_np = np.asarray(jax.device_get(loss))
    grad_np = np.asarray(jax.device_get(grad_norm))
    logits_np = np.asarray(jax.device_get(logits))

    if logits_np.shape != (ndev, per_device_batch, tr.NUM_CLASSES):
        raise RuntimeError(f"Unexpected pmap logits shape: {logits_np.shape}")
    if not np.all(np.isfinite(loss_np)):
        raise RuntimeError(f"Non-finite synthetic loss: {loss_np}")
    if not np.all(np.isfinite(grad_np)):
        raise RuntimeError(f"Non-finite synthetic gradient norm: {grad_np}")
    if not np.all(np.isfinite(logits_np)):
        raise RuntimeError("Non-finite synthetic logits")

    print(
        f"PMAP_FORWARD_BACKWARD=PASS | {ndev}/{ndev} cores | "
        f"loss={float(loss_np[0]):.6f} | grad_norm={float(grad_np[0]):.6f}",
        flush=True,
    )

    dataset = tr.find_dataset(None)
    print(f"DATASET_FOUND={dataset}", flush=True)
    annotations, split = tr.load_ntu(dataset)
    if not annotations:
        raise RuntimeError("NTU120 annotation list is empty")

    for protocol in ("xsub", "xset"):
        tk, vk = tr.resolve_split(split, protocol)
        if len(split[tk]) == 0 or len(split[vk]) == 0:
            raise RuntimeError(f"Empty {protocol} split: train={tk}, val={vk}")
        print(
            f"{protocol.upper()}_SPLIT=PASS | train={len(split[tk]):,} | val={len(split[vk]):,}",
            flush=True,
        )

    sample = next((a for a in annotations if isinstance(a, tr.Mapping)), None)
    if sample is None:
        raise RuntimeError("Could not find a mapping-style NTU annotation")
    sample_x = tr.preprocess_keypoints(tr.annotation_keypoints(sample), "motion")
    sample_y = tr.annotation_label(sample)
    if sample_x.shape != (tr.FRAMES, tr.FEATURES):
        raise RuntimeError(f"Unexpected preprocessed sample shape: {sample_x.shape}")
    if not np.all(np.isfinite(sample_x)):
        raise RuntimeError("Preprocessed sample contains NaN/Inf")
    if not 0 <= int(sample_y) < tr.NUM_CLASSES:
        raise RuntimeError(f"Sample label out of range: {sample_y}")

    print(
        f"REAL_SAMPLE_PREPROCESS=PASS | shape={sample_x.shape} | label={int(sample_y)}",
        flush=True,
    )
    print("STRICT_PREFLIGHT=PASS — entering full training", flush=True)
    print("=" * 112, flush=True)


def run_full_training() -> None:
    sys.argv = [
        str(TRAINER_PATH),
        "--protocol", "both",
        "--selector", "motion",
        "--epochs", "60",
        "--patience", "12",
        "--batch-size", "256",
        "--eval-batch-size", "512",
        "--learning-rate", "6e-4",
        "--min-learning-rate", "2e-5",
        "--warmup-fraction", "0.08",
        "--weight-decay", "0.03",
        "--label-smoothing", "0.05",
        "--grad-clip", "1.0",
        "--ema-decay", "0.995",
        "--stream-aux-weight", "0.15",
        "--spatial-dim", str(SPATIAL_DIM),
        "--model-dim", str(MODEL_DIM),
        "--dropout", str(DROPOUT),
        "--seed", str(SEED),
        "--progress-every", "5",
        "--audit-first",
        "--outdir", "/kaggle/working/NestSAR_M4_MotionLite_T16_TPU",
    ]
    tr.main()


if __name__ == "__main__":
    strict_tpu_preflight()
    run_full_training()

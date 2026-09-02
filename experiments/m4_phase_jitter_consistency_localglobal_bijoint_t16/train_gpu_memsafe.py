#!/usr/bin/env python3
from __future__ import annotations

"""Memory-safe single-T4 trainer for LocalGlobal V2 + BiJoint M4/G4.

Why this exists
---------------
The original clean BiJoint worker kept the champion physical batch of 256 and
ran canonical+jitter forwards together.  On a 15-GB T4 the backward pass tried
to allocate an additional ~4.7 GB and failed.  This worker keeps the SAME
*effective* batch of 256, optimizer update count, LR schedule, loss, EMA, and
architecture, but evaluates the effective batch as four 64-sample
microbatches, accumulates their gradients, averages them, and performs one
optimizer update.  Thus the optimization batch remains 256 while peak GPU
activation memory is much lower.

Host RAM is handled by the companion launcher, which runs XSUB and XSET
sequentially so only one protocol's canonical+jitter/validation arrays exist at
once.
"""

import gc
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_t16 import train_tpu as cons
from experiments.m4_phase_jitter_consistency_localglobal_t16 import train_tpu as lg_train
from experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16.model import (
    EXPECTED_PARAMS,
    M4PhaseUniformBiJointT16,
)

NUM_CLASSES = ju.NUM_CLASSES
NUM_STREAMS = ju.NUM_STREAMS
FRAMES = ju.FRAMES
FEATURES = ju.FEATURES

# Keep the champion effective batch 256 while using much smaller GPU microbatches.
DEFAULT_MICROBATCH = 64
DEFAULT_EVAL_MICROBATCH = 256

BASELINE_ACCURACY = {
    "xsub": 0.7531176967340285,
    "xset": 0.7592682885821410,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_scale(a, scale: float):
    return jax.tree_util.tree_map(lambda x: x * scale, a)


def train_protocol_memsafe(args, annotations, split, protocol: str):
    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU; backend={jax.default_backend()} "
            f"count={jax.local_device_count()}"
        )

    effective_batch = int(args.batch_size)
    microbatch = int(os.environ.get("NESTSAR_MICROBATCH", DEFAULT_MICROBATCH))
    eval_microbatch = int(
        os.environ.get("NESTSAR_EVAL_MICROBATCH", DEFAULT_EVAL_MICROBATCH)
    )

    if effective_batch <= 0 or microbatch <= 0:
        raise ValueError("Batch sizes must be positive")
    if effective_batch % microbatch != 0:
        raise ValueError(
            f"Effective batch {effective_batch} must be divisible by microbatch {microbatch}"
        )

    accum_steps = effective_batch // microbatch
    protocol_seed = args.seed + (0 if protocol == "xsub" else 100000)

    log(
        f"{protocol.upper()} preprocessing START | "
        "LocalGlobal canonical+jitter train + canonical val"
    )
    tprep = time.time()
    Xcan, Xjit, ytr, Xva, yva = ju.build_protocol_views(
        annotations,
        split,
        protocol,
        args.jitter_max_shift,
        protocol_seed,
        args.max_train_samples,
        args.max_val_samples,
    )
    log(
        f"{protocol.upper()} preprocessing READY | time={time.time()-tprep:.1f}s | "
        f"Xcan={Xcan.nbytes/2**30:.2f}GiB Xjit={Xjit.nbytes/2**30:.2f}GiB "
        f"Xval={Xva.nbytes/2**30:.2f}GiB"
    )

    steps_per_epoch = len(ytr) // effective_batch
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_fraction))

    lr = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=warmup,
        decay_steps=total_steps,
        end_value=args.min_learning_rate,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(lr, weight_decay=args.weight_decay),
    )

    model = M4PhaseUniformBiJointT16(
        args.spatial_dim,
        args.model_dim,
        args.dropout,
    )

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    dummy = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    params = model.init(
        {"params": init_key, "dropout": init_key},
        dummy,
        training=False,
    )["params"]

    nparams = ju.count_params(params)
    log(f"{protocol.upper()} params={nparams:,}")
    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(
            f"Parameter mismatch: got {nparams:,}, expected {EXPECTED_PARAMS:,}"
        )

    state = ju.State.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        ema_params=params,
    )

    @jax.jit
    def grad_microstep(params, rng, xcan, xjit, yb):
        rng, drop_can, drop_jit = jax.random.split(rng, 3)

        def loss_fn(p):
            out_can = model.apply(
                {"params": p},
                xcan,
                training=True,
                rngs={"dropout": drop_can},
            )
            out_jit = model.apply(
                {"params": p},
                xjit,
                training=True,
                rngs={"dropout": drop_jit},
            )

            ce_can = jnp.mean(
                ju.smooth_ce(out_can["logits"], yb, args.label_smoothing)
            )
            ce_jit = jnp.mean(
                ju.smooth_ce(out_jit["logits"], yb, args.label_smoothing)
            )
            main = 0.5 * (ce_can + ce_jit)

            sl_can = out_can["stream_logits"]
            sl_jit = out_jit["stream_logits"]
            labels4 = jnp.repeat(yb, NUM_STREAMS)

            aux_can = jnp.mean(
                ju.smooth_ce(
                    sl_can.reshape(-1, NUM_CLASSES),
                    labels4,
                    args.label_smoothing,
                )
            )
            aux_jit = jnp.mean(
                ju.smooth_ce(
                    sl_jit.reshape(-1, NUM_CLASSES),
                    labels4,
                    args.label_smoothing,
                )
            )
            aux = 0.5 * (aux_can + aux_jit)

            consistency = cons.symmetric_kl(
                out_can["logits"],
                out_jit["logits"],
                args.consistency_temperature,
            )

            loss = (
                main
                + args.stream_aux_weight * aux
                + args.consistency_weight * consistency
            )

            acc_can = jnp.mean(jnp.argmax(out_can["logits"], axis=-1) == yb)
            acc_jit = jnp.mean(jnp.argmax(out_jit["logits"], axis=-1) == yb)
            agreement = jnp.mean(
                jnp.argmax(out_can["logits"], axis=-1)
                == jnp.argmax(out_jit["logits"], axis=-1)
            )

            metrics = jnp.asarray(
                [loss, main, aux, consistency, acc_can, acc_jit, agreement],
                dtype=jnp.float32,
            )
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(params)
        return grads, metrics, rng

    @jax.jit
    def apply_accumulated(state, grads):
        state = state.apply_gradients(grads=grads)
        ema = jax.tree_util.tree_map(
            lambda e, p: args.ema_decay * e + (1.0 - args.ema_decay) * p,
            state.ema_params,
            state.params,
        )
        return state.replace(ema_params=ema)

    @jax.jit
    def eval_step(ema_params, xb, yb, mask):
        out = model.apply({"params": ema_params}, xb, training=False)
        ce = ju.smooth_ce(out["logits"], yb, 0.0)
        pred = jnp.argmax(out["logits"], axis=-1)
        loss_sum = jnp.sum(ce * mask)
        correct = jnp.sum((pred == yb).astype(jnp.float32) * mask)
        count = jnp.sum(mask)
        return jnp.asarray([loss_sum, correct, count], jnp.float32)

    outdir = Path(args.outdir) / protocol
    outdir.mkdir(parents=True, exist_ok=True)

    best = -1.0
    best_epoch = 0
    stale = 0

    log(
        f"{protocol.upper()} MEMORY-SAFE TRAINING | effective_batch={effective_batch} "
        f"microbatch={microbatch} accum_steps={accum_steps} "
        f"optimizer_updates/epoch={steps_per_epoch}"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        sums = np.zeros(7, np.float64)
        n_updates = 0

        for xcan_eff, xjit_eff, y_eff in cons.iter_train_pairs(
            Xcan,
            Xjit,
            ytr,
            effective_batch,
            args.seed + epoch,
        ):
            accum_grads = None
            metric_sum = np.zeros(7, np.float64)
            seen = 0

            for s in range(0, effective_batch, microbatch):
                e = s + microbatch
                xcan_mb = jnp.asarray(xcan_eff[s:e])
                xjit_mb = jnp.asarray(xjit_eff[s:e])
                y_mb = jnp.asarray(y_eff[s:e])

                grads, metrics, key = grad_microstep(
                    state.params,
                    key,
                    xcan_mb,
                    xjit_mb,
                    y_mb,
                )

                accum_grads = grads if accum_grads is None else tree_add(accum_grads, grads)
                metric_sum += np.asarray(jax.device_get(metrics), np.float64) * microbatch
                seen += microbatch

            accum_grads = tree_scale(accum_grads, 1.0 / accum_steps)
            state = apply_accumulated(state, accum_grads)

            sums += metric_sum / max(seen, 1)
            n_updates += 1

            if n_updates % max(1, args.progress_every * 4) == 0:
                mean = sums / n_updates
                log(
                    f"{protocol.upper()} E{epoch:03d} U{n_updates:04d}/{steps_per_epoch} "
                    f"loss={mean[0]:.3f} cons={mean[3]:.4f} "
                    f"can={100*mean[4]:.2f}% jit={100*mean[5]:.2f}% "
                    f"agr={100*mean[6]:.1f}% best={100*best:.2f}%"
                )

        # Evaluation is also microbatched for predictable T4 memory use.
        eval_loss = 0.0
        eval_correct = 0.0
        eval_count = 0.0

        for xb, yb, mask in ju.iter_eval(Xva, yva, eval_microbatch):
            vals = np.asarray(
                jax.device_get(
                    eval_step(
                        state.ema_params,
                        jnp.asarray(xb),
                        jnp.asarray(yb),
                        jnp.asarray(mask),
                    )
                )
            )
            eval_loss += float(vals[0])
            eval_correct += float(vals[1])
            eval_count += float(vals[2])

        val_acc = eval_correct / max(eval_count, 1.0)
        val_loss = eval_loss / max(eval_count, 1.0)
        mean = sums / max(n_updates, 1)

        log(
            f"{protocol.upper()} E{epoch:03d} "
            f"train_can={100*mean[4]:.3f}% train_jit={100*mean[5]:.3f}% "
            f"agree={100*mean[6]:.2f}% cons={mean[3]:.5f} "
            f"val={100*val_acc:.3f}% loss={val_loss:.4f} "
            f"time={time.time()-t0:.1f}s"
        )

        improved = val_acc > best + 1e-6
        if improved:
            best = val_acc
            best_epoch = epoch
            stale = 0
            single = jax.device_get(state)

            payload = {
                "model": "M4LocalGlobalBiJointT16MemorySafe",
                "protocol": protocol,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "params": single.params,
                "ema_params": single.ema_params,
                "opt_state": single.opt_state,
                "step": single.step,
                "config": vars(args),
                "representation": {
                    "frames": FRAMES,
                    "preprocessing": "local_pose_global_motion_v2",
                    "spatial_joint_memory": "bidirectional_bimemory",
                    "attention": False,
                    "qkv": False,
                    "training_from_scratch": True,
                    "effective_batch": effective_batch,
                    "microbatch": microbatch,
                    "gradient_accumulation_steps": accum_steps,
                    "memory_safe": True,
                },
            }
            (outdir / "best.msgpack").write_bytes(serialization.to_bytes(payload))
            (outdir / "best.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "val_accuracy": val_acc,
                        "params": nparams,
                        "effective_batch": effective_batch,
                        "microbatch": microbatch,
                        "gradient_accumulation_steps": accum_steps,
                        "spatial_joint_memory": "bidirectional_bimemory",
                        "attention": False,
                    },
                    indent=2,
                )
            )
        else:
            stale += 1

        if stale >= args.patience:
            log(
                f"{protocol.upper()} early stop: best={100*best:.3f}% @ E{best_epoch}"
            )
            break

    del Xcan, Xjit, ytr, Xva, yva
    gc.collect()

    return best, best_epoch, microbatch, accum_steps


def main() -> None:
    # Exact LocalGlobal V2 preprocessing; only the spatial joint memory changes.
    lg_train.install_preprocessing_override()

    args = cons.parse_args()
    if args.protocol not in ("xsub", "xset"):
        raise ValueError("Memory-safe worker runs exactly one protocol")

    print("=" * 120, flush=True)
    print(
        f"NESTSAR LOCALGLOBAL V2 + BI-JOINT M4/G4 — MEMORY SAFE | "
        f"{args.protocol.upper()} | SINGLE T4",
        flush=True,
    )
    print("=" * 120, flush=True)
    print("JAX:", jax.__version__, flush=True)
    print("BACKEND:", jax.default_backend(), flush=True)
    print("DEVICES:", jax.local_devices(), flush=True)
    print("EXPECTED PARAMS:", f"{EXPECTED_PARAMS:,}", flush=True)
    print("EFFECTIVE BATCH:", args.batch_size, flush=True)
    print("MICROBATCH:", os.environ.get("NESTSAR_MICROBATCH", DEFAULT_MICROBATCH), flush=True)
    print("ATTENTION: NONE", flush=True)
    print("TRAINING: FROM SCRATCH", flush=True)

    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU; backend={jax.default_backend()} "
            f"count={jax.local_device_count()}"
        )

    dataset = ju.base.find_dataset(args.dataset)
    print("DATASET:", dataset, flush=True)
    annotations, split = ju.base.load_ntu(dataset)

    best, epoch, microbatch, accum_steps = train_protocol_memsafe(
        args,
        annotations,
        split,
        args.protocol,
    )

    baseline = BASELINE_ACCURACY[args.protocol]
    result = {
        "protocol": args.protocol,
        "best_val_accuracy": best,
        "best_epoch": epoch,
        "baseline_localglobal_v2": baseline,
        "delta_vs_baseline_pp": 100.0 * (best - baseline),
        "expected_params": EXPECTED_PARAMS,
        "backend": "gpu",
        "effective_batch": args.batch_size,
        "microbatch": microbatch,
        "gradient_accumulation_steps": accum_steps,
        "seed": args.seed,
        "preprocessing": "local_pose_global_motion_v2",
        "spatial_joint_memory": "bidirectional_bimemory",
        "attention": False,
        "training_from_scratch": True,
        "memory_safe": True,
    }

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"result_{args.protocol}.json").write_text(compact_json(result))

    print("=" * 120, flush=True)
    print("MEMORY-SAFE GPU WORKER DONE", compact_json(result), flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()

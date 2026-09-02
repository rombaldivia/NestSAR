#!/usr/bin/env python3
from __future__ import annotations

"""LocalGlobal V2 champion architecture + hard-negative ranking loss.

This is a from-scratch training experiment. It keeps the exact LocalGlobal V2
representation and the Phase+Jitter+Consistency champion architecture/training
schedule, and adds only a training-time hard-negative term to the final logits.
Inference architecture, parameter count, and inference FLOPs are unchanged.
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_t16 import train_tpu as cons
from experiments.m4_phase_jitter_consistency_localglobal_t16 import train_tpu as lg_train
from experiments.m4_phase_jitter_consistency_localglobal_t16 import train_gpu as lg_gpu

EXPECTED_PARAMS = 1_816_130
NUM_CLASSES = ju.NUM_CLASSES
NUM_STREAMS = ju.NUM_STREAMS
FRAMES = ju.FRAMES
FEATURES = ju.FEATURES

BASELINE_ACCURACY = {
    "xsub": 0.7531176967340285,
    "xset": 0.7592682885821410,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hard_negative_loss(logits: jnp.ndarray, labels: jnp.ndarray, margin: float) -> jnp.ndarray:
    """Smooth hardest-negative margin loss on 120-way logits."""
    labels = labels.astype(jnp.int32)
    true_logits = jnp.take_along_axis(logits, labels[:, None], axis=-1)[:, 0]
    onehot = jax.nn.one_hot(labels, NUM_CLASSES, dtype=jnp.bool_)
    wrong_logits = jnp.where(onehot, jnp.asarray(-1e9, logits.dtype), logits)
    hard_wrong = jnp.max(wrong_logits, axis=-1)
    return jnp.mean(jax.nn.softplus(hard_wrong - true_logits + margin))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument("--protocol", choices=["xsub", "xset"], required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=6e-4)
    p.add_argument("--min-learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.08)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--stream-aux-weight", type=float, default=0.15)
    p.add_argument("--spatial-dim", type=int, default=24)
    p.add_argument("--model-dim", type=int, default=112)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--jitter-max-shift", type=int, default=1)
    p.add_argument("--consistency-weight", type=float, default=0.08)
    p.add_argument("--consistency-temperature", type=float, default=1.0)
    p.add_argument("--hardneg-weight", type=float, default=0.04)
    p.add_argument("--hardneg-margin", type=float, default=0.20)
    p.add_argument("--outdir", required=True)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--audit-first", action="store_true")
    return p.parse_args()


def train_protocol(args: argparse.Namespace, annotations, split, protocol: str):
    devices = list(jax.local_devices())
    ndev = len(devices)
    if ndev != 1:
        raise RuntimeError(f"Expected one isolated GPU per worker, got {ndev}: {devices}")
    if args.batch_size % ndev or args.eval_batch_size % ndev:
        raise ValueError("Train/eval batch must be divisible by visible device count")

    protocol_seed = args.seed + (0 if protocol == "xsub" else 100000)
    Xcan, Xjit, ytr, Xva, yva = ju.build_protocol_views(
        annotations,
        split,
        protocol,
        args.jitter_max_shift,
        protocol_seed,
        args.max_train_samples,
        args.max_val_samples,
    )

    steps_per_epoch = len(ytr) // args.batch_size
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

    # FROM SCRATCH: exact champion initialization path, no checkpoint loading.
    model = ju.M4PhaseUniformT16(args.spatial_dim, args.model_dim, args.dropout)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    dummy = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    params = model.init(
        {"params": init_key, "dropout": init_key}, dummy, training=False
    )["params"]
    nparams = ju.count_params(params)
    log(f"{protocol.upper()} params={nparams:,}")
    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(f"Parameter mismatch: got {nparams:,}, expected {EXPECTED_PARAMS:,}")
    if args.audit_first:
        ju.audit_flops(model, params)

    state = ju.State.create(apply_fn=model.apply, params=params, tx=tx, ema_params=params)
    state = jax.device_put_replicated(state, devices)
    rngs = jax.random.split(key, ndev)

    @partial(jax.pmap, axis_name="d", devices=devices)
    def p_train_step(state, rng, xcan, xjit, yb):
        rng, drop_can, drop_jit = jax.random.split(rng, 3)

        def loss_fn(p):
            out_can = model.apply(
                {"params": p}, xcan, training=True, rngs={"dropout": drop_can}
            )
            out_jit = model.apply(
                {"params": p}, xjit, training=True, rngs={"dropout": drop_jit}
            )

            ce_can = jnp.mean(ju.smooth_ce(out_can["logits"], yb, args.label_smoothing))
            ce_jit = jnp.mean(ju.smooth_ce(out_jit["logits"], yb, args.label_smoothing))
            main = 0.5 * (ce_can + ce_jit)

            sl_can = out_can["stream_logits"]
            sl_jit = out_jit["stream_logits"]
            labels4 = jnp.repeat(yb, NUM_STREAMS)
            aux_can = jnp.mean(
                ju.smooth_ce(sl_can.reshape(-1, NUM_CLASSES), labels4, args.label_smoothing)
            )
            aux_jit = jnp.mean(
                ju.smooth_ce(sl_jit.reshape(-1, NUM_CLASSES), labels4, args.label_smoothing)
            )
            aux = 0.5 * (aux_can + aux_jit)

            consistency = cons.symmetric_kl(
                out_can["logits"], out_jit["logits"], args.consistency_temperature
            )

            hn_can = hard_negative_loss(out_can["logits"], yb, args.hardneg_margin)
            hn_jit = hard_negative_loss(out_jit["logits"], yb, args.hardneg_margin)
            hardneg = 0.5 * (hn_can + hn_jit)

            loss = (
                main
                + args.stream_aux_weight * aux
                + args.consistency_weight * consistency
                + args.hardneg_weight * hardneg
            )

            acc_can = jnp.mean(jnp.argmax(out_can["logits"], axis=-1) == yb)
            acc_jit = jnp.mean(jnp.argmax(out_jit["logits"], axis=-1) == yb)
            agreement = jnp.mean(
                jnp.argmax(out_can["logits"], axis=-1)
                == jnp.argmax(out_jit["logits"], axis=-1)
            )
            return loss, (main, aux, consistency, hardneg, acc_can, acc_jit, agreement)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, "d")
        loss = jax.lax.pmean(loss, "d")
        metrics = jax.lax.pmean(metrics, "d")

        state = state.apply_gradients(grads=grads)
        ema = jax.tree_util.tree_map(
            lambda e, p: args.ema_decay * e + (1.0 - args.ema_decay) * p,
            state.ema_params,
            state.params,
        )
        state = state.replace(ema_params=ema)
        return state, rng, (loss, *metrics)

    @partial(jax.pmap, axis_name="d", devices=devices)
    def p_eval_step(ema_params, xb, yb, mask):
        out = model.apply({"params": ema_params}, xb, training=False)
        ce = ju.smooth_ce(out["logits"], yb, 0.0)
        pred = jnp.argmax(out["logits"], axis=-1)
        correct = jnp.sum((pred == yb).astype(jnp.float32) * mask)
        loss_sum = jnp.sum(ce * mask)
        count = jnp.sum(mask)
        return jax.lax.psum(jnp.asarray([loss_sum, correct, count]), "d")

    outdir = Path(args.outdir) / protocol
    outdir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    best_epoch = 0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # loss, main, aux, consistency, hardneg, acc_can, acc_jit, agreement
        sums = np.zeros(8, np.float64)
        nstep = 0

        for xcan, xjit, yb in cons.iter_train_pairs(
            Xcan, Xjit, ytr, args.batch_size, args.seed + epoch
        ):
            xcan = ju.shard(xcan, ndev)
            xjit = ju.shard(xjit, ndev)
            yb = ju.shard(yb, ndev)
            state, rngs, metrics = p_train_step(state, rngs, xcan, xjit, yb)
            vals = [float(np.asarray(v[0])) for v in jax.device_get(metrics)]
            sums += np.asarray(vals, np.float64)
            nstep += 1

        eval_loss = eval_correct = eval_count = 0.0
        for xb, yb, mask in ju.iter_eval(Xva, yva, args.eval_batch_size):
            xb = ju.shard(xb, ndev)
            yb = ju.shard(yb, ndev)
            mask = ju.shard(mask, ndev)
            vals = np.asarray(jax.device_get(p_eval_step(state.ema_params, xb, yb, mask)[0]))
            eval_loss += float(vals[0])
            eval_correct += float(vals[1])
            eval_count += float(vals[2])

        val_acc = eval_correct / max(eval_count, 1.0)
        val_loss = eval_loss / max(eval_count, 1.0)
        mean = sums / max(nstep, 1)
        log(
            f"{protocol.upper()} E{epoch:03d} "
            f"train_can={100*mean[5]:.3f}% train_jit={100*mean[6]:.3f}% "
            f"agree={100*mean[7]:.2f}% cons={mean[3]:.5f} "
            f"hardneg={mean[4]:.5f} val={100*val_acc:.3f}% "
            f"loss={val_loss:.4f} time={time.time()-t0:.1f}s"
        )

        improved = val_acc > best + 1e-6
        if improved:
            best = val_acc
            best_epoch = epoch
            stale = 0
            single = jax.tree_util.tree_map(lambda z: jax.device_get(z[0]), state)
            payload = {
                "model": "M4PhaseJitterConsistencyLocalGlobalHardNegT16",
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
                    "features_per_token": FEATURES,
                    "preprocessing": "local_pose_global_motion_v2",
                    "pose_coordinate_frame": "exact_base_canonicalize_raw_framewise_root_centered",
                    "motion_coordinate_frame": "constant_first_valid_person0_root",
                    "final_fusion": "uniform_mean",
                    "consistency": "symmetric_kl",
                    "consistency_weight": args.consistency_weight,
                    "hard_negative": "softplus_hardest_wrong_minus_true_plus_margin",
                    "hardneg_weight": args.hardneg_weight,
                    "hardneg_margin": args.hardneg_margin,
                    "architecture_change": False,
                    "inference_change": False,
                    "training_from_scratch": True,
                },
            }
            (outdir / "best.msgpack").write_bytes(serialization.to_bytes(payload))
            (outdir / "best.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "val_accuracy": val_acc,
                        "baseline_localglobal_v2": BASELINE_ACCURACY[protocol],
                        "delta_vs_baseline_pp": 100.0 * (val_acc - BASELINE_ACCURACY[protocol]),
                        "params": EXPECTED_PARAMS,
                        "hardneg_weight": args.hardneg_weight,
                        "hardneg_margin": args.hardneg_margin,
                        "training_from_scratch": True,
                    },
                    indent=2,
                )
            )
        else:
            stale += 1

        if stale >= args.patience:
            log(f"{protocol.upper()} early stop: best={100*best:.3f}% @ E{best_epoch}")
            break

    del Xcan, Xjit, ytr, Xva, yva
    return best, best_epoch


def main() -> None:
    args = parse_args()

    # Exact LocalGlobal V2 representation used by the current champion.
    lg_train.install_preprocessing_override()
    # Keep Dual-T4 logs clean while preserving the same data-building path.
    lg_gpu.install_clean_progress(args.protocol)

    ju.EXPECTED_PARAMS = EXPECTED_PARAMS
    cons.EXPECTED_PARAMS = EXPECTED_PARAMS

    print("=" * 120, flush=True)
    print(f"NESTSAR LOCALGLOBAL V2 + HARDNEG | {args.protocol.upper()} | FROM SCRATCH", flush=True)
    print("=" * 120, flush=True)
    print("JAX:", jax.__version__, flush=True)
    print("BACKEND:", jax.default_backend(), flush=True)
    print("LOCAL DEVICES:", jax.local_device_count(), flush=True)
    print("DEVICES:", jax.local_devices(), flush=True)
    print("HARDNEG WEIGHT:", args.hardneg_weight, flush=True)
    print("HARDNEG MARGIN:", args.hardneg_margin, flush=True)

    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, got backend={jax.default_backend()} "
            f"count={jax.local_device_count()}"
        )

    dataset = ju.base.find_dataset(args.dataset)
    print("DATASET:", dataset, flush=True)
    annotations, split = ju.base.load_ntu(dataset)
    best, epoch = train_protocol(args, annotations, split, args.protocol)

    result = {
        "protocol": args.protocol,
        "best_val_accuracy": best,
        "best_epoch": epoch,
        "baseline_localglobal_v2": BASELINE_ACCURACY[args.protocol],
        "delta_vs_baseline_pp": 100.0 * (best - BASELINE_ACCURACY[args.protocol]),
        "expected_params": EXPECTED_PARAMS,
        "backend": "gpu",
        "visible_devices": [str(d) for d in jax.local_devices()],
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "seed": args.seed,
        "preprocessing": "local_pose_global_motion_v2",
        "hardneg_weight": args.hardneg_weight,
        "hardneg_margin": args.hardneg_margin,
        "training_from_scratch": True,
    }
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"result_{args.protocol}.json").write_text(json.dumps(result, separators=(",", ":")))

    print("=" * 120, flush=True)
    print("GPU WORKER DONE", json.dumps(result, separators=(",", ":")), flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()

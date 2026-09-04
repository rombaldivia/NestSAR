#!/usr/bin/env python3
from __future__ import annotations

"""Single-visible-GPU worker for NestSAR-SM-ALL-T16.

This worker intentionally renders NO tqdm bars. It writes machine-readable
STATUS lines to stdout; the dual-T4 parent owns the only notebook progress bars.
That prevents child-process carriage returns from creating broken Kaggle lines.
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

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_t16 import train_tpu as cons
from experiments.m4_phase_jitter_consistency_localglobal_t16 import preprocessing as lg
from experiments.nestsar_sm_all_t16.model import (
    FEATURES,
    FRAMES,
    NUM_CLASSES,
    NUM_STREAMS,
    NestSARSMAllT16,
)


def compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def emit_status(
    protocol: str,
    epoch: int,
    epochs: int,
    phase: str,
    done: int = 0,
    total: int = 0,
    **metrics,
) -> None:
    fields = [
        "STATUS",
        f"protocol={protocol}",
        f"epoch={epoch}",
        f"epochs={epochs}",
        f"phase={phase}",
        f"done={done}",
        f"total={total}",
    ]
    for key, value in metrics.items():
        fields.append(f"{key}={value}")
    print("|".join(fields), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", choices=["xsub", "xset"], required=True)
    p.add_argument("--outdir", required=True)

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
    p.add_argument("--consistency-weight", type=float, default=0.08)
    p.add_argument("--consistency-temperature", type=float, default=1.0)

    p.add_argument("--spatial-dim", type=int, default=24)
    p.add_argument("--model-dim", type=int, default=112)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--controller-dim", type=int, default=16)
    p.add_argument("--fast-rank", type=int, default=2)
    p.add_argument("--head-rank", type=int, default=2)
    p.add_argument("--sm-residual-scale", type=float, default=0.08)
    p.add_argument("--head-residual-scale", type=float, default=0.15)

    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--jitter-max-shift", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--eval-progress-every", type=int, default=10)
    p.add_argument("--preprocess-progress-every", type=int, default=2500)

    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--audit-first", action="store_true")
    p.add_argument("--max-gflops", type=float, default=0.025)
    return p.parse_args()


def make_model(args) -> NestSARSMAllT16:
    return NestSARSMAllT16(
        spatial_dim=args.spatial_dim,
        model_dim=args.model_dim,
        dropout=args.dropout,
        controller_dim=args.controller_dim,
        fast_rank=args.fast_rank,
        head_rank=args.head_rank,
        sm_residual_scale=args.sm_residual_scale,
        head_residual_scale=args.head_residual_scale,
    )


def init_and_audit(args, protocol: str):
    model = make_model(args)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    dummy = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    params = model.init(
        {"params": init_key, "dropout": init_key},
        dummy,
        training=False,
    )["params"]
    nparams = ju.count_params(params)

    flops = None
    if args.audit_first:
        fn = jax.jit(
            lambda p, xx: model.apply(
                {"params": p}, xx, training=False
            )["logits"]
        )
        compiled = fn.lower(params, dummy).compile()
        ca = compiled.cost_analysis()
        if isinstance(ca, list):
            ca = ca[0] if ca else {}
        flops = float(ca.get("flops", float("nan")))
        gflops = flops / 1e9
        print(
            f"AUDIT|protocol={protocol}|params={nparams}|flops={flops:.0f}|gflops={gflops:.9f}",
            flush=True,
        )
        if np.isfinite(gflops) and gflops > args.max_gflops:
            raise RuntimeError(
                f"GFLOPs hard limit exceeded: {gflops:.9f} > {args.max_gflops:.9f}"
            )

    return model, params, key, nparams, flops


def build_protocol_arrays(
    args,
    annotations,
    split,
    protocol: str,
):
    """Variable raw length -> fixed 16 LocalGlobal temporal tokens.

    Every raw clip is segmented into exactly 16 temporal regions. Each token
    summarizes its region (pose + full/phase/path motion), so the network input
    shape and inference compute do not depend on the original raw frame count.
    """
    by_id, train_ids, val_ids = ju.resolve_protocol_ids(
        annotations,
        split,
        protocol,
    )
    if args.max_train_samples:
        train_ids = train_ids[: args.max_train_samples]
    if args.max_val_samples:
        val_ids = val_ids[: args.max_val_samples]

    Xcan = np.empty((len(train_ids), FRAMES, FEATURES), np.float32)
    Xjit = np.empty_like(Xcan)
    ytr = np.empty((len(train_ids),), np.int32)

    protocol_seed = args.seed + (0 if protocol == "xsub" else 100000)
    changed = 0

    emit_status(protocol, 0, args.epochs, "PREP_TRAIN", 0, len(train_ids))
    for i, sid in enumerate(train_ids):
        a = by_id[sid]
        kp = base.annotation_keypoints(a)

        Xcan[i] = lg.segment_phase_tokens_localglobal(kp)

        total_raw = int(base.canonicalize_raw(kp).shape[0])
        rng = np.random.default_rng(
            np.random.SeedSequence([protocol_seed, i, 9173])
        )
        bounds = ju.jittered_segment_bounds(
            total_raw,
            FRAMES,
            args.jitter_max_shift,
            rng,
        )
        Xjit[i] = lg.phase_tokens_from_bounds_localglobal(kp, bounds)
        ytr[i] = base.annotation_label(a)

        changed += int(np.any(np.abs(Xcan[i] - Xjit[i]) > 1e-7))
        done = i + 1
        if (
            done % args.preprocess_progress_every == 0
            or done == len(train_ids)
        ):
            emit_status(
                protocol,
                0,
                args.epochs,
                "PREP_TRAIN",
                done,
                len(train_ids),
            )

    Xva = np.empty((len(val_ids), FRAMES, FEATURES), np.float32)
    yva = np.empty((len(val_ids),), np.int32)

    emit_status(protocol, 0, args.epochs, "PREP_VAL", 0, len(val_ids))
    for i, sid in enumerate(val_ids):
        a = by_id[sid]
        kp = base.annotation_keypoints(a)
        Xva[i] = lg.segment_phase_tokens_localglobal(kp)
        yva[i] = base.annotation_label(a)

        done = i + 1
        if (
            done % args.preprocess_progress_every == 0
            or done == len(val_ids)
        ):
            emit_status(
                protocol,
                0,
                args.epochs,
                "PREP_VAL",
                done,
                len(val_ids),
            )

    changed_fraction = changed / max(len(train_ids), 1)
    print(
        f"PREPROCESS_DONE|protocol={protocol}|train={len(ytr)}|val={len(yva)}|"
        f"processing_frames={FRAMES}|jitter_changed={changed_fraction:.6f}",
        flush=True,
    )
    return Xcan, Xjit, ytr, Xva, yva


def iter_train_pairs(Xcan, Xjit, y, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    usable = (len(idx) // batch_size) * batch_size
    idx = idx[:usable]
    for start in range(0, usable, batch_size):
        ii = idx[start : start + batch_size]
        yield Xcan[ii], Xjit[ii], y[ii]


def iter_eval(X, y, batch_size: int):
    for start in range(0, len(y), batch_size):
        n = min(batch_size, len(y) - start)
        xb = np.zeros((batch_size, *X.shape[1:]), dtype=X.dtype)
        yb = np.zeros((batch_size,), dtype=np.int32)
        mask = np.zeros((batch_size,), dtype=np.float32)
        xb[:n] = X[start : start + n]
        yb[:n] = y[start : start + n]
        mask[:n] = 1.0
        yield xb, yb, mask


def train_protocol(args, annotations, split, protocol: str):
    devices = list(jax.local_devices())
    if jax.default_backend() != "gpu" or len(devices) != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, got backend={jax.default_backend()} devices={devices}"
        )
    if args.batch_size % len(devices):
        raise ValueError("batch-size must be divisible by local device count")
    if args.eval_batch_size % len(devices):
        raise ValueError("eval-batch-size must be divisible by local device count")
    if args.fast_rank < 1 or args.head_rank < 1:
        raise ValueError("fast-rank and head-rank must be >= 1")

    # Compile/audit BEFORE expensive dataset materialization.
    model, params, key, nparams, flops = init_and_audit(args, protocol)

    Xcan, Xjit, ytr, Xva, yva = build_protocol_arrays(
        args,
        annotations,
        split,
        protocol,
    )

    steps_per_epoch = len(ytr) // args.batch_size
    if steps_per_epoch < 1:
        raise RuntimeError(
            f"Not enough train samples ({len(ytr)}) for batch {args.batch_size}"
        )

    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_fraction))
    lr = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=args.min_learning_rate,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(lr, weight_decay=args.weight_decay),
    )

    state = ju.State.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        ema_params=params,
    )
    state = jax.device_put_replicated(state, devices)
    rngs = jax.random.split(key, len(devices))

    @partial(jax.pmap, axis_name="d", devices=devices)
    def p_train_step(state, rng, xcan, xjit, yb):
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

            labels4 = jnp.repeat(yb, NUM_STREAMS)
            aux_can = jnp.mean(
                ju.smooth_ce(
                    out_can["stream_logits"].reshape(-1, NUM_CLASSES),
                    labels4,
                    args.label_smoothing,
                )
            )
            aux_jit = jnp.mean(
                ju.smooth_ce(
                    out_jit["stream_logits"].reshape(-1, NUM_CLASSES),
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

            acc_can = jnp.mean(
                jnp.argmax(out_can["logits"], axis=-1) == yb
            )
            acc_jit = jnp.mean(
                jnp.argmax(out_jit["logits"], axis=-1) == yb
            )
            agreement = jnp.mean(
                jnp.argmax(out_can["logits"], axis=-1)
                == jnp.argmax(out_jit["logits"], axis=-1)
            )
            eta = jnp.mean(out_can["sm_eta_mean"])
            alpha = jnp.mean(out_can["sm_alpha_mean"])
            return loss, (
                main,
                aux,
                consistency,
                acc_can,
                acc_jit,
                agreement,
                eta,
                alpha,
            )

        (loss, metrics), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(state.params)
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
        out = model.apply(
            {"params": ema_params},
            xb,
            training=False,
        )
        ce = ju.smooth_ce(out["logits"], yb, 0.0)
        pred = jnp.argmax(out["logits"], axis=-1)
        correct = jnp.sum((pred == yb).astype(jnp.float32) * mask)
        loss_sum = jnp.sum(ce * mask)
        count = jnp.sum(mask)
        eta_sum = jnp.sum(out["sm_eta_mean"] * mask)
        alpha_sum = jnp.sum(out["sm_alpha_mean"] * mask)
        vals = jnp.asarray(
            [loss_sum, correct, count, eta_sum, alpha_sum],
            dtype=jnp.float32,
        )
        return jax.lax.psum(vals, "d")

    outdir = Path(args.outdir) / protocol
    outdir.mkdir(parents=True, exist_ok=True)

    best = -1.0
    best_epoch = 0
    stale = 0
    last_epoch = 0

    ndev = len(devices)
    eval_total = (len(yva) + args.eval_batch_size - 1) // args.eval_batch_size

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        t0 = time.time()
        sums = np.zeros(9, np.float64)
        nstep = 0

        for xcan, xjit, yb in iter_train_pairs(
            Xcan,
            Xjit,
            ytr,
            args.batch_size,
            args.seed + epoch,
        ):
            xcan = ju.shard(xcan, ndev)
            xjit = ju.shard(xjit, ndev)
            yb_shard = ju.shard(yb, ndev)

            state, rngs, metrics = p_train_step(
                state,
                rngs,
                xcan,
                xjit,
                yb_shard,
            )
            vals = [float(np.asarray(v[0])) for v in jax.device_get(metrics)]
            sums += np.asarray(vals, np.float64)
            nstep += 1

            if (
                nstep % args.progress_every == 0
                or nstep == steps_per_epoch
            ):
                mean = sums / max(nstep, 1)
                emit_status(
                    protocol,
                    epoch,
                    args.epochs,
                    "TRAIN",
                    nstep,
                    steps_per_epoch,
                    loss=f"{mean[0]:.4f}",
                    acc=f"{100*mean[4]:.2f}",
                    jit=f"{100*mean[5]:.2f}",
                    agr=f"{100*mean[6]:.1f}",
                    eta=f"{mean[7]:.4f}",
                    alpha=f"{mean[8]:.4f}",
                    best=f"{100*best:.3f}" if best >= 0 else "NA",
                )

        eval_loss = 0.0
        eval_correct = 0.0
        eval_count = 0.0
        eval_eta = 0.0
        eval_alpha = 0.0
        eval_step = 0

        for xb, yb, mask in iter_eval(
            Xva,
            yva,
            args.eval_batch_size,
        ):
            xb = ju.shard(xb, ndev)
            yb = ju.shard(yb, ndev)
            mask = ju.shard(mask, ndev)
            vals = np.asarray(
                jax.device_get(
                    p_eval_step(state.ema_params, xb, yb, mask)[0]
                )
            )
            eval_loss += float(vals[0])
            eval_correct += float(vals[1])
            eval_count += float(vals[2])
            eval_eta += float(vals[3])
            eval_alpha += float(vals[4])
            eval_step += 1

            if (
                eval_step % args.eval_progress_every == 0
                or eval_step == eval_total
            ):
                partial_acc = eval_correct / max(eval_count, 1.0)
                emit_status(
                    protocol,
                    epoch,
                    args.epochs,
                    "VAL",
                    eval_step,
                    eval_total,
                    acc=f"{100*partial_acc:.3f}",
                    best=f"{100*best:.3f}" if best >= 0 else "NA",
                )

        val_acc = eval_correct / max(eval_count, 1.0)
        val_loss = eval_loss / max(eval_count, 1.0)
        val_eta = eval_eta / max(eval_count, 1.0)
        val_alpha = eval_alpha / max(eval_count, 1.0)
        mean = sums / max(nstep, 1)

        improved = val_acc > best + 1e-6
        if improved:
            best = val_acc
            best_epoch = epoch
            stale = 0

            single = jax.tree_util.tree_map(
                lambda z: jax.device_get(z[0]),
                state,
            )
            payload = {
                "model": "NestSAR-SM-ALL-T16-v1",
                "protocol": protocol,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "params": single.params,
                "ema_params": single.ema_params,
                "opt_state": single.opt_state,
                "step": single.step,
                "config": vars(args),
                "representation": {
                    "raw_input_frames": "variable",
                    "processing_frames": FRAMES,
                    "raw_to_processing": "16 whole-clip LocalGlobal motion-preserving segments",
                    "token_features": FEATURES,
                    "attention": False,
                    "gcn": False,
                    "tcn": False,
                    "transformer": False,
                    "m4": "base BiMemory + low-rank self-modifying delta residual",
                    "g4": "base chunk BiMemory + slower low-rank self-modifying delta residual",
                    "fast_rank": args.fast_rank,
                    "head_rank": args.head_rank,
                    "shared_eta_alpha_controller": True,
                    "fast_weight_reset": "reset to meta-learned S0 for every clip",
                },
                "audit": {
                    "params": nparams,
                    "flops": flops,
                    "gflops": None if flops is None else flops / 1e9,
                },
            }
            (outdir / "best.msgpack").write_bytes(
                serialization.to_bytes(payload)
            )
            (outdir / "best.json").write_text(
                json.dumps(
                    {
                        "model": "NestSAR-SM-ALL-T16-v1",
                        "epoch": epoch,
                        "val_accuracy": val_acc,
                        "params": nparams,
                        "flops": flops,
                        "gflops": None if flops is None else flops / 1e9,
                        "processing_frames": FRAMES,
                        "raw_input_frames": "variable",
                        "fast_rank": args.fast_rank,
                        "head_rank": args.head_rank,
                    },
                    indent=2,
                )
            )
        else:
            stale += 1

        emit_status(
            protocol,
            epoch,
            args.epochs,
            "EPOCH",
            1,
            1,
            val=f"{100*val_acc:.3f}",
            valloss=f"{val_loss:.4f}",
            train=f"{100*mean[4]:.2f}",
            eta=f"{val_eta:.4f}",
            alpha=f"{val_alpha:.4f}",
            best=f"{100*best:.3f}",
            best_e=best_epoch,
            stale=stale,
            sec=f"{time.time()-t0:.1f}",
        )

        if stale >= args.patience:
            print(
                f"EARLY_STOP|protocol={protocol}|last_epoch={epoch}|best_epoch={best_epoch}|best={best:.9f}",
                flush=True,
            )
            break

    result = {
        "model": "NestSAR-SM-ALL-T16-v1",
        "protocol": protocol,
        "best_val_accuracy": best,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "params": nparams,
        "flops": flops,
        "gflops": None if flops is None else flops / 1e9,
        "processing_frames": FRAMES,
        "raw_input_frames": "variable",
        "fast_rank": args.fast_rank,
        "head_rank": args.head_rank,
        "controller_dim": args.controller_dim,
        "sm_residual_scale": args.sm_residual_scale,
        "head_residual_scale": args.head_residual_scale,
        "seed": args.seed,
    }
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"result_{protocol}.json").write_text(compact_json(result))

    del Xcan, Xjit, ytr, Xva, yva
    print("RESULT|" + compact_json(result), flush=True)
    return result


def main() -> None:
    args = parse_args()

    print("=" * 118, flush=True)
    print(
        f"NESTSAR-SM-ALL-T16 v1 | {args.protocol.upper()} | FROM SCRATCH | SINGLE T4",
        flush=True,
    )
    print(
        "RAW CLIP LENGTH VARIABLE -> FIXED 16 PROCESSING TOKENS | NO ATTENTION/GCN/TCN/TRANSFORMER",
        flush=True,
    )
    print(
        f"FAST_RANK={args.fast_rank} HEAD_RANK={args.head_rank} CONTROLLER={args.controller_dim} "
        f"GFLOP_LIMIT={args.max_gflops:.6f}",
        flush=True,
    )
    print("JAX", jax.__version__, "BACKEND", jax.default_backend(), flush=True)
    print("DEVICES", jax.local_devices(), flush=True)
    print("=" * 118, flush=True)

    dataset = base.find_dataset(args.dataset)
    annotations, split = base.load_ntu(dataset)
    train_protocol(args, annotations, split, args.protocol)


if __name__ == "__main__":
    main()

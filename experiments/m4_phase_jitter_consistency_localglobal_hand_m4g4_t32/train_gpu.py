#!/usr/bin/env python3
from __future__ import annotations

"""From-scratch Dual-T4 worker for LocalGlobal V2 + Hand-M4/G4-Lite T32.

Main path:
  exact LocalGlobal V2 T16, 4 streams, fixed uniform fusion.

New path:
  T32 hand-only [local xyz + global velocity] -> dim32 projection
  -> BiMemory -> 8x4 chunks -> BiMemory -> hand classifier.

No attention.  The same GatedSweep/BiMemory primitive used by the NestSAR core
is reused by the hand specialist.

Training retains the champion recipe:
  canonical + +/-1 main-view boundary jitter
  symmetric-KL consistency
  stream auxiliary CE
  EMA
and adds only a small hand-head auxiliary CE so the new specialist learns a
meaningful discriminative representation.
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

from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.model import (
    EXPECTED_PARAMS_D32,
    M4LocalGlobalHandM4G4T32,
)
from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.preprocessing import (
    HAND_FEATURES,
    HAND_FRAMES,
    HAND_JOINT_IDS,
    hand_tokens_t32,
)

FRAMES = ju.FRAMES
FEATURES = ju.FEATURES
NUM_CLASSES = ju.NUM_CLASSES
NUM_STREAMS = ju.NUM_STREAMS
EXPECTED_PARAMS = EXPECTED_PARAMS_D32

BASELINE_ACCURACY = {
    "xsub": 0.7531176967340285,
    "xset": 0.7592682885821410,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
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
    p.add_argument("--hand-aux-weight", type=float, default=0.05)
    p.add_argument("--consistency-weight", type=float, default=0.08)
    p.add_argument("--consistency-temperature", type=float, default=1.0)

    p.add_argument("--spatial-dim", type=int, default=24)
    p.add_argument("--model-dim", type=int, default=112)
    p.add_argument("--hand-dim", type=int, default=32)
    p.add_argument("--hand-residual-scale", type=float, default=0.10)
    p.add_argument("--dropout", type=float, default=0.10)

    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--jitter-max-shift", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=5)

    p.add_argument("--outdir", required=True)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--audit-first", action="store_true")
    return p.parse_args()


def build_protocol_arrays(
    annotations,
    split,
    protocol: str,
    max_shift: int,
    seed: int,
    max_train: int = 0,
    max_val: int = 0,
):
    """Build exact LocalGlobal main views plus deterministic T32 hand view."""
    by_id, train_ids, val_ids = ju.resolve_protocol_ids(
        annotations,
        split,
        protocol,
    )

    if max_train:
        train_ids = train_ids[:max_train]
    if max_val:
        val_ids = val_ids[:max_val]

    Xcan = np.empty((len(train_ids), FRAMES, FEATURES), np.float32)
    Xjit = np.empty_like(Xcan)
    Htr = np.empty((len(train_ids), HAND_FRAMES, HAND_FEATURES), np.float32)
    ytr = np.empty((len(train_ids),), np.int32)

    print(
        f"{protocol.upper()} PREPROCESSING START | "
        f"LocalGlobal canonical+jitter T16 + Hand-M4/G4 T32",
        flush=True,
    )
    t0 = time.time()
    changed_count = 0
    hand_nonzero_count = 0

    for i, sid in enumerate(train_ids):
        a = by_id[sid]
        kp = base.annotation_keypoints(a)

        Xcan[i] = lg.segment_phase_tokens_localglobal(kp)

        total = int(base.canonicalize_raw(kp).shape[0])
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, i, 9173])
        )
        bounds = ju.jittered_segment_bounds(
            total,
            FRAMES,
            max_shift,
            rng,
        )
        Xjit[i] = lg.phase_tokens_from_bounds_localglobal(kp, bounds)

        Htr[i] = hand_tokens_t32(kp)
        ytr[i] = base.annotation_label(a)

        changed_count += int(
            np.any(np.abs(Xcan[i] - Xjit[i]) > 1e-7)
        )
        hand_nonzero_count += int(
            np.any(np.abs(Htr[i]) > 1e-7)
        )

        if (i + 1) % 10_000 == 0 or i + 1 == len(train_ids):
            print(
                f"{protocol.upper()} TRAIN PREPROCESS "
                f"{i+1:,}/{len(train_ids):,}",
                flush=True,
            )

    Xva = np.empty((len(val_ids), FRAMES, FEATURES), np.float32)
    Hva = np.empty((len(val_ids), HAND_FRAMES, HAND_FEATURES), np.float32)
    yva = np.empty((len(val_ids),), np.int32)

    for i, sid in enumerate(val_ids):
        a = by_id[sid]
        kp = base.annotation_keypoints(a)

        Xva[i] = lg.segment_phase_tokens_localglobal(kp)
        Hva[i] = hand_tokens_t32(kp)
        yva[i] = base.annotation_label(a)

        if (i + 1) % 10_000 == 0 or i + 1 == len(val_ids):
            print(
                f"{protocol.upper()} VAL PREPROCESS "
                f"{i+1:,}/{len(val_ids):,}",
                flush=True,
            )

    changed = changed_count / max(len(train_ids), 1)
    hand_nonzero = hand_nonzero_count / max(len(train_ids), 1)

    print(
        f"{protocol.upper()} PREPROCESSING READY | "
        f"time={time.time()-t0:.1f}s | "
        f"main jitter changed={100*changed:.2f}% | "
        f"hand nonzero={100*hand_nonzero:.2f}%",
        flush=True,
    )

    return Xcan, Xjit, Htr, ytr, Xva, Hva, yva


def iter_train_pairs(
    Xcan,
    Xjit,
    Htr,
    y,
    global_batch: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    usable = (len(idx) // global_batch) * global_batch
    idx = idx[:usable]

    for start in range(0, usable, global_batch):
        ii = idx[start:start + global_batch]
        yield Xcan[ii], Xjit[ii], Htr[ii], y[ii]


def iter_eval(X, H, y, global_batch: int):
    for start in range(0, len(y), global_batch):
        n = min(global_batch, len(y) - start)

        xb = np.zeros(
            (global_batch, *X.shape[1:]),
            dtype=X.dtype,
        )
        hb = np.zeros(
            (global_batch, *H.shape[1:]),
            dtype=H.dtype,
        )
        yb = np.zeros((global_batch,), dtype=np.int32)
        mask = np.zeros((global_batch,), dtype=np.float32)

        xb[:n] = X[start:start+n]
        hb[:n] = H[start:start+n]
        yb[:n] = y[start:start+n]
        mask[:n] = 1.0

        yield xb, hb, yb, mask


def audit_flops(model, params):
    dummy_main = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    dummy_hand = jnp.zeros((1, HAND_FRAMES, HAND_FEATURES), jnp.float32)

    try:
        fn = jax.jit(
            lambda p, x, h: model.apply(
                {"params": p},
                x,
                h,
                training=False,
            )["logits"]
        )
        compiled = fn.lower(
            params,
            dummy_main,
            dummy_hand,
        ).compile()
        ca = compiled.cost_analysis()
        if isinstance(ca, list):
            ca = ca[0] if ca else {}

        flops = float(ca.get("flops", float("nan")))
        log(
            f"XLA inference audit: {flops:,.0f} FLOPs/clip "
            f"= {flops/1e9:.9f} GFLOPs"
        )
        return flops

    except Exception as exc:
        log(f"GFLOPs audit unavailable: {exc}")
        return None


def train_protocol(
    args,
    annotations,
    split,
    protocol: str,
):
    devices = list(jax.local_devices())
    ndev = len(devices)

    if args.batch_size % ndev:
        raise ValueError(
            f"Batch {args.batch_size} not divisible by {ndev}"
        )
    if args.eval_batch_size % ndev:
        raise ValueError(
            f"Eval batch {args.eval_batch_size} not divisible by {ndev}"
        )
    if args.hand_dim != 32:
        raise ValueError(
            "This audited branch fixes hand_dim=32 so the exact parameter "
            "preflight remains deterministic."
        )

    protocol_seed = (
        args.seed
        + (0 if protocol == "xsub" else 100000)
    )

    (
        Xcan,
        Xjit,
        Htr,
        ytr,
        Xva,
        Hva,
        yva,
    ) = build_protocol_arrays(
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
    warmup = max(
        1,
        int(total_steps * args.warmup_fraction),
    )

    lr = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=warmup,
        decay_steps=total_steps,
        end_value=args.min_learning_rate,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(
            lr,
            weight_decay=args.weight_decay,
        ),
    )

    model = M4LocalGlobalHandM4G4T32(
        spatial_dim=args.spatial_dim,
        model_dim=args.model_dim,
        dropout=args.dropout,
        hand_dim=args.hand_dim,
        hand_residual_scale=args.hand_residual_scale,
    )

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    dummy_main = jnp.zeros(
        (1, FRAMES, FEATURES),
        jnp.float32,
    )
    dummy_hand = jnp.zeros(
        (1, HAND_FRAMES, HAND_FEATURES),
        jnp.float32,
    )

    params = model.init(
        {
            "params": init_key,
            "dropout": init_key,
        },
        dummy_main,
        dummy_hand,
        training=False,
    )["params"]

    nparams = ju.count_params(params)
    log(f"{protocol.upper()} params={nparams:,}")

    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(
            f"Parameter mismatch: got {nparams:,}, "
            f"expected {EXPECTED_PARAMS:,}"
        )

    if args.audit_first:
        audit_flops(model, params)

    state = ju.State.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        ema_params=params,
    )
    state = jax.device_put_replicated(
        state,
        devices,
    )

    rngs = jax.random.split(
        key,
        ndev,
    )

    @partial(
        jax.pmap,
        axis_name="d",
        devices=devices,
    )
    def p_train_step(
        state,
        rng,
        xcan,
        xjit,
        hand,
        yb,
    ):
        rng, drop_can, drop_jit = jax.random.split(
            rng,
            3,
        )

        def loss_fn(p):
            out_can = model.apply(
                {"params": p},
                xcan,
                hand,
                training=True,
                rngs={"dropout": drop_can},
            )
            out_jit = model.apply(
                {"params": p},
                xjit,
                hand,
                training=True,
                rngs={"dropout": drop_jit},
            )

            ce_can = jnp.mean(
                ju.smooth_ce(
                    out_can["logits"],
                    yb,
                    args.label_smoothing,
                )
            )
            ce_jit = jnp.mean(
                ju.smooth_ce(
                    out_jit["logits"],
                    yb,
                    args.label_smoothing,
                )
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

            hand_aux_can = jnp.mean(
                ju.smooth_ce(
                    out_can["hand_logits"],
                    yb,
                    args.label_smoothing,
                )
            )
            hand_aux_jit = jnp.mean(
                ju.smooth_ce(
                    out_jit["hand_logits"],
                    yb,
                    args.label_smoothing,
                )
            )
            hand_aux = 0.5 * (
                hand_aux_can + hand_aux_jit
            )

            consistency = cons.symmetric_kl(
                out_can["logits"],
                out_jit["logits"],
                args.consistency_temperature,
            )

            loss = (
                main
                + args.stream_aux_weight * aux
                + args.hand_aux_weight * hand_aux
                + args.consistency_weight * consistency
            )

            acc_can = jnp.mean(
                jnp.argmax(
                    out_can["logits"],
                    axis=-1,
                ) == yb
            )
            acc_jit = jnp.mean(
                jnp.argmax(
                    out_jit["logits"],
                    axis=-1,
                ) == yb
            )
            hand_acc_can = jnp.mean(
                jnp.argmax(
                    out_can["hand_logits"],
                    axis=-1,
                ) == yb
            )
            hand_acc_jit = jnp.mean(
                jnp.argmax(
                    out_jit["hand_logits"],
                    axis=-1,
                ) == yb
            )
            agreement = jnp.mean(
                jnp.argmax(
                    out_can["logits"],
                    axis=-1,
                )
                == jnp.argmax(
                    out_jit["logits"],
                    axis=-1,
                )
            )

            return loss, (
                main,
                aux,
                hand_aux,
                consistency,
                acc_can,
                acc_jit,
                hand_acc_can,
                hand_acc_jit,
                agreement,
            )

        (loss, metrics), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(state.params)

        grads = jax.lax.pmean(
            grads,
            "d",
        )
        loss = jax.lax.pmean(
            loss,
            "d",
        )
        metrics = jax.lax.pmean(
            metrics,
            "d",
        )

        state = state.apply_gradients(
            grads=grads
        )

        ema = jax.tree_util.tree_map(
            lambda e, p: (
                args.ema_decay * e
                + (1.0 - args.ema_decay) * p
            ),
            state.ema_params,
            state.params,
        )
        state = state.replace(
            ema_params=ema
        )

        return state, rng, (loss, *metrics)

    @partial(
        jax.pmap,
        axis_name="d",
        devices=devices,
    )
    def p_eval_step(
        ema_params,
        xb,
        hb,
        yb,
        mask,
    ):
        out = model.apply(
            {"params": ema_params},
            xb,
            hb,
            training=False,
        )

        pred = jnp.argmax(
            out["logits"],
            axis=-1,
        )
        main_pred = jnp.argmax(
            out["main_logits"],
            axis=-1,
        )
        hand_pred = jnp.argmax(
            out["hand_logits"],
            axis=-1,
        )

        ce = ju.smooth_ce(
            out["logits"],
            yb,
            0.0,
        )

        correct = jnp.sum(
            (pred == yb).astype(jnp.float32)
            * mask
        )
        main_correct = jnp.sum(
            (main_pred == yb).astype(jnp.float32)
            * mask
        )
        hand_correct = jnp.sum(
            (hand_pred == yb).astype(jnp.float32)
            * mask
        )
        loss_sum = jnp.sum(
            ce * mask
        )
        count = jnp.sum(mask)

        return jax.lax.psum(
            jnp.asarray(
                [
                    loss_sum,
                    correct,
                    main_correct,
                    hand_correct,
                    count,
                ]
            ),
            "d",
        )

    outdir = Path(args.outdir) / protocol
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best = -1.0
    best_epoch = 0
    stale = 0

    baseline = BASELINE_ACCURACY[protocol]

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        sums = np.zeros(10, dtype=np.float64)
        nstep = 0

        for (
            xcan,
            xjit,
            hand,
            yb,
        ) in iter_train_pairs(
            Xcan,
            Xjit,
            Htr,
            ytr,
            args.batch_size,
            args.seed + epoch,
        ):
            xcan = ju.shard(xcan, ndev)
            xjit = ju.shard(xjit, ndev)
            hand = ju.shard(hand, ndev)
            yb = ju.shard(yb, ndev)

            (
                state,
                rngs,
                metrics,
            ) = p_train_step(
                state,
                rngs,
                xcan,
                xjit,
                hand,
                yb,
            )

            vals = [
                float(np.asarray(v[0]))
                for v in jax.device_get(metrics)
            ]
            sums += np.asarray(
                vals,
                np.float64,
            )
            nstep += 1

        eval_loss = 0.0
        eval_correct = 0.0
        eval_main_correct = 0.0
        eval_hand_correct = 0.0
        eval_count = 0.0

        for xb, hb, yb, mask in iter_eval(
            Xva,
            Hva,
            yva,
            args.eval_batch_size,
        ):
            xb = ju.shard(xb, ndev)
            hb = ju.shard(hb, ndev)
            yb = ju.shard(yb, ndev)
            mask = ju.shard(mask, ndev)

            vals = np.asarray(
                jax.device_get(
                    p_eval_step(
                        state.ema_params,
                        xb,
                        hb,
                        yb,
                        mask,
                    )[0]
                )
            )

            eval_loss += float(vals[0])
            eval_correct += float(vals[1])
            eval_main_correct += float(vals[2])
            eval_hand_correct += float(vals[3])
            eval_count += float(vals[4])

        val_acc = (
            eval_correct
            / max(eval_count, 1.0)
        )
        val_main_acc = (
            eval_main_correct
            / max(eval_count, 1.0)
        )
        val_hand_acc = (
            eval_hand_correct
            / max(eval_count, 1.0)
        )
        val_loss = (
            eval_loss
            / max(eval_count, 1.0)
        )

        mean = sums / max(nstep, 1)

        log(
            f"{protocol.upper()} E{epoch:03d} "
            f"train_can={100*mean[5]:.3f}% "
            f"train_jit={100*mean[6]:.3f}% "
            f"hand_can={100*mean[7]:.3f}% "
            f"hand_jit={100*mean[8]:.3f}% "
            f"agree={100*mean[9]:.2f}% "
            f"cons={mean[4]:.5f} "
            f"hand_aux={mean[3]:.4f} "
            f"val={100*val_acc:.3f}% "
            f"main_only={100*val_main_acc:.3f}% "
            f"hand_only={100*val_hand_acc:.3f}% "
            f"baseline={100*baseline:.3f}% "
            f"best={100*best:.3f}% "
            f"loss={val_loss:.4f} "
            f"time={time.time()-t0:.1f}s"
        )

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
                "model": "M4LocalGlobalHandM4G4LiteT32",
                "protocol": protocol,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "val_main_only_accuracy": val_main_acc,
                "val_hand_only_accuracy": val_hand_acc,
                "baseline_localglobal_v2": baseline,
                "delta_vs_baseline_pp": 100.0 * (val_acc - baseline),

                "params": single.params,
                "ema_params": single.ema_params,
                "opt_state": single.opt_state,
                "step": single.step,

                "config": vars(args),

                "representation": {
                    "main_frames": FRAMES,
                    "main_features_per_token": FEATURES,
                    "main_preprocessing": "local_pose_global_motion_v2",
                    "main_jitter": "same +/-1 raw-frame boundary jitter as champion",

                    "hand_frames": HAND_FRAMES,
                    "hand_features_per_token": HAND_FEATURES,
                    "hand_joint_ids_zero_based": HAND_JOINT_IDS.tolist(),
                    "hand_features": "local_xyz_plus_global_velocity_xyz",
                    "hand_sampling": "uniform_t32_over_raw_sequence",
                    "hand_view_jitter": False,

                    "hand_memory_core": "same base.BiMemory/GatedSweep as NestSAR main core",
                    "hand_memory_hierarchy": "T32 frame BiMemory -> 8x4 chunk BiMemory",
                    "attention": False,
                    "hand_dim": args.hand_dim,
                    "hand_residual_scale": args.hand_residual_scale,
                    "hand_aux_weight": args.hand_aux_weight,

                    "main_final_fusion": "uniform_mean_4_streams",
                    "final_combination": "main_logits + fixed_scale * hand_logits",
                    "consistency": "symmetric_kl",
                    "consistency_weight": args.consistency_weight,
                    "training_from_scratch": True,
                },
            }

            (
                outdir / "best.msgpack"
            ).write_bytes(
                serialization.to_bytes(payload)
            )

            (
                outdir / "best.json"
            ).write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "val_accuracy": val_acc,
                        "val_main_only_accuracy": val_main_acc,
                        "val_hand_only_accuracy": val_hand_acc,
                        "baseline_localglobal_v2": baseline,
                        "delta_vs_baseline_pp": 100.0 * (val_acc - baseline),
                        "params": EXPECTED_PARAMS,
                        "main_frames": FRAMES,
                        "hand_frames": HAND_FRAMES,
                        "hand_features": HAND_FEATURES,
                        "hand_dim": args.hand_dim,
                        "hand_residual_scale": args.hand_residual_scale,
                        "hand_aux_weight": args.hand_aux_weight,
                        "attention": False,
                        "training_from_scratch": True,
                    },
                    indent=2,
                )
            )

        else:
            stale += 1

        if stale >= args.patience:
            log(
                f"{protocol.upper()} early stop: "
                f"best={100*best:.3f}% @ E{best_epoch}"
            )
            break

    result = {
        "protocol": protocol,
        "best_val_accuracy": best,
        "best_epoch": best_epoch,
        "baseline_localglobal_v2": baseline,
        "delta_vs_baseline_pp": 100.0 * (best - baseline),
        "expected_params": EXPECTED_PARAMS,
        "backend": jax.default_backend(),
        "visible_devices": [str(d) for d in jax.local_devices()],
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "seed": args.seed,
        "main_preprocessing": "local_pose_global_motion_v2",
        "hand_branch": "m4g4_lite_t32_dim32_no_attention",
        "hand_residual_scale": args.hand_residual_scale,
        "hand_aux_weight": args.hand_aux_weight,
        "training_from_scratch": True,
    }

    root = Path(args.outdir)
    root.mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        root / f"result_{protocol}.json"
    ).write_text(
        compact_json(result)
    )

    del (
        Xcan,
        Xjit,
        Htr,
        ytr,
        Xva,
        Hva,
        yva,
    )

    return best, best_epoch


def main() -> None:
    args = parse_args()

    print("=" * 120, flush=True)
    print(
        f"NESTSAR LOCALGLOBAL V2 + HAND-M4/G4-LITE T32 | "
        f"{args.protocol.upper()} | FROM SCRATCH",
        flush=True,
    )
    print("=" * 120, flush=True)

    print("JAX:", jax.__version__, flush=True)
    print("BACKEND:", jax.default_backend(), flush=True)
    print("LOCAL DEVICES:", jax.local_device_count(), flush=True)
    print("DEVICES:", jax.local_devices(), flush=True)
    print("EXPECTED PARAMS:", f"{EXPECTED_PARAMS:,}", flush=True)
    print("HAND FRAMES:", HAND_FRAMES, flush=True)
    print("HAND FEATURES:", HAND_FEATURES, flush=True)
    print("HAND DIM:", args.hand_dim, flush=True)
    print("HAND RESIDUAL SCALE:", args.hand_residual_scale, flush=True)
    print("HAND AUX WEIGHT:", args.hand_aux_weight, flush=True)
    print("ATTENTION: NONE", flush=True)

    if (
        jax.default_backend() != "gpu"
        or jax.local_device_count() != 1
    ):
        raise RuntimeError(
            f"Expected exactly one visible GPU, "
            f"got backend={jax.default_backend()} "
            f"count={jax.local_device_count()}"
        )

    dataset = base.find_dataset(
        args.dataset
    )
    print("DATASET:", dataset, flush=True)

    annotations, split = base.load_ntu(
        dataset
    )

    best, epoch = train_protocol(
        args,
        annotations,
        split,
        args.protocol,
    )

    result_path = (
        Path(args.outdir)
        / f"result_{args.protocol}.json"
    )
    result = json.loads(
        result_path.read_text()
    )

    print("=" * 120, flush=True)
    print(
        "GPU WORKER DONE",
        compact_json(result),
        flush=True,
    )
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()

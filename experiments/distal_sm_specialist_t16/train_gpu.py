#!/usr/bin/env python3
from __future__ import annotations

"""Single-visible-T4 worker for the standalone distal specialist.

No base/champion checkpoint is loaded. The network is initialized randomly and
trained on all 120 NTU classes using only distal joints. Child workers emit plain
STATUS lines; the parent launcher owns all notebook tqdm rendering.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_t16 import train_tpu as cons
from experiments.distal_sm_specialist_t16.model import (
    DistalSMSpecialistT16,
    NUM_CLASSES,
)
from experiments.distal_sm_specialist_t16.preprocessing import (
    DISTAL_JOINT_IDS,
    FEATURES,
    FRAMES,
    distal_tokens_t16,
    jittered_distal_tokens_t16,
)


def compact_json(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def emit_status(protocol, epoch, epochs, phase, done=0, total=0, **metrics):
    fields = [
        "STATUS",
        f"protocol={protocol}",
        f"epoch={epoch}",
        f"epochs={epochs}",
        f"phase={phase}",
        f"done={done}",
        f"total={total}",
    ]
    fields += [f"{k}={v}" for k, v in metrics.items()]
    print("|".join(fields), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", choices=["xsub", "xset"], required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)

    p.add_argument("--learning-rate", type=float, default=8e-4)
    p.add_argument("--min-learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.08)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--consistency-weight", type=float, default=0.05)
    p.add_argument("--consistency-temperature", type=float, default=1.0)

    p.add_argument("--spatial-dim", type=int, default=16)
    p.add_argument("--model-dim", type=int, default=64)
    p.add_argument("--controller-dim", type=int, default=16)
    p.add_argument("--fast-rank", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--sm-residual-scale", type=float, default=0.08)

    p.add_argument("--jitter-max-shift", type=int, default=1)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--eval-progress-every", type=int, default=10)
    p.add_argument("--preprocess-progress-every", type=int, default=2500)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--audit-first", action="store_true")
    return p.parse_args()


def make_model(args):
    return DistalSMSpecialistT16(
        spatial_dim=args.spatial_dim,
        model_dim=args.model_dim,
        controller_dim=args.controller_dim,
        fast_rank=args.fast_rank,
        dropout=args.dropout,
        sm_residual_scale=args.sm_residual_scale,
    )


def build_arrays(args, annotations, split, protocol):
    by_id, train_ids, val_ids = ju.resolve_protocol_ids(annotations, split, protocol)
    if args.max_train_samples:
        train_ids = train_ids[:args.max_train_samples]
    if args.max_val_samples:
        val_ids = val_ids[:args.max_val_samples]

    Xcan = np.empty((len(train_ids), FRAMES, FEATURES), np.float32)
    Xjit = np.empty_like(Xcan)
    ytr = np.empty((len(train_ids),), np.int32)

    protocol_seed = args.seed + (0 if protocol == "xsub" else 100000)
    emit_status(protocol, 0, args.epochs, "PREP_TRAIN", 0, len(train_ids))
    changed = 0
    for i, sid in enumerate(train_ids):
        a = by_id[sid]
        kp = base.annotation_keypoints(a)
        Xcan[i] = distal_tokens_t16(kp)
        rng = np.random.default_rng(np.random.SeedSequence([protocol_seed, i, 7717]))
        Xjit[i] = jittered_distal_tokens_t16(kp, args.jitter_max_shift, rng)
        ytr[i] = base.annotation_label(a)
        changed += int(np.any(np.abs(Xcan[i] - Xjit[i]) > 1e-7))
        done = i + 1
        if done % args.preprocess_progress_every == 0 or done == len(train_ids):
            emit_status(protocol, 0, args.epochs, "PREP_TRAIN", done, len(train_ids))

    Xva = np.empty((len(val_ids), FRAMES, FEATURES), np.float32)
    yva = np.empty((len(val_ids),), np.int32)
    emit_status(protocol, 0, args.epochs, "PREP_VAL", 0, len(val_ids))
    for i, sid in enumerate(val_ids):
        a = by_id[sid]
        Xva[i] = distal_tokens_t16(base.annotation_keypoints(a))
        yva[i] = base.annotation_label(a)
        done = i + 1
        if done % args.preprocess_progress_every == 0 or done == len(val_ids):
            emit_status(protocol, 0, args.epochs, "PREP_VAL", done, len(val_ids))

    print(
        f"PREPROCESS_DONE|protocol={protocol}|train={len(ytr)}|val={len(yva)}|"
        f"raw_frames=variable|processing_frames={FRAMES}|features={FEATURES}|"
        f"jitter_changed={changed/max(len(ytr),1):.6f}",
        flush=True,
    )
    return Xcan, Xjit, ytr, Xva, yva


def iter_train(Xcan, Xjit, y, batch_size, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    usable = (len(idx) // batch_size) * batch_size
    idx = idx[:usable]
    for s in range(0, usable, batch_size):
        ii = idx[s:s+batch_size]
        yield Xcan[ii], Xjit[ii], y[ii]


def iter_eval(X, y, batch_size):
    for s in range(0, len(y), batch_size):
        n = min(batch_size, len(y)-s)
        xb = np.zeros((batch_size, *X.shape[1:]), X.dtype)
        yb = np.zeros((batch_size,), np.int32)
        mask = np.zeros((batch_size,), np.float32)
        xb[:n] = X[s:s+n]
        yb[:n] = y[s:s+n]
        mask[:n] = 1.0
        yield xb, yb, mask


def train_protocol(args, annotations, split, protocol):
    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, backend={jax.default_backend()} "
            f"devices={jax.local_devices()}"
        )

    model = make_model(args)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    dummy = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    params = model.init({"params": init_key, "dropout": init_key}, dummy, training=False)["params"]
    nparams = ju.count_params(params)

    raw_flops = None
    if args.audit_first:
        fn = jax.jit(lambda p, xx: model.apply({"params": p}, xx, training=False)["logits"])
        compiled = fn.lower(params, dummy).compile()
        ca = compiled.cost_analysis()
        if isinstance(ca, list):
            ca = ca[0] if ca else {}
        raw_flops = float(ca.get("flops", float("nan")))
        print(
            f"AUDIT|protocol={protocol}|params={nparams}|raw_xla_flops={raw_flops:.0f}|"
            f"raw_xla_gflops={raw_flops/1e9:.9f}|note=scan_counter_not_paper_number",
            flush=True,
        )

    Xcan, Xjit, ytr, Xva, yva = build_arrays(args, annotations, split, protocol)
    steps_per_epoch = len(ytr) // args.batch_size
    if steps_per_epoch < 1:
        raise RuntimeError("Not enough training samples for configured batch size")

    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_fraction))
    lr = optax.warmup_cosine_decay_schedule(
        0.0, args.learning_rate, warmup, total_steps,
        end_value=args.min_learning_rate,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(lr, weight_decay=args.weight_decay),
    )
    state = ju.State.create(apply_fn=model.apply, params=params, tx=tx, ema_params=params)

    @jax.jit
    def train_step(state, rng, xcan, xjit, yb):
        rng, d1, d2 = jax.random.split(rng, 3)
        def loss_fn(p):
            a = model.apply({"params": p}, xcan, training=True, rngs={"dropout": d1})
            b = model.apply({"params": p}, xjit, training=True, rngs={"dropout": d2})
            cea = jnp.mean(ju.smooth_ce(a["logits"], yb, args.label_smoothing))
            ceb = jnp.mean(ju.smooth_ce(b["logits"], yb, args.label_smoothing))
            ce = 0.5 * (cea + ceb)
            skl = cons.symmetric_kl(a["logits"], b["logits"], args.consistency_temperature)
            loss = ce + args.consistency_weight * skl
            acc = jnp.mean(jnp.argmax(a["logits"], -1) == yb)
            accj = jnp.mean(jnp.argmax(b["logits"], -1) == yb)
            agr = jnp.mean(jnp.argmax(a["logits"], -1) == jnp.argmax(b["logits"], -1))
            eta = jnp.mean(a["eta_mean"])
            alpha = jnp.mean(a["alpha_mean"])
            return loss, (ce, skl, acc, accj, agr, eta, alpha)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        ema = jax.tree_util.tree_map(
            lambda e, p: args.ema_decay * e + (1.0-args.ema_decay) * p,
            state.ema_params, state.params,
        )
        state = state.replace(ema_params=ema)
        return state, rng, (loss, *metrics)

    @jax.jit
    def eval_step(ema_params, xb, yb, mask):
        out = model.apply({"params": ema_params}, xb, training=False)
        ce = ju.smooth_ce(out["logits"], yb, 0.0)
        pred = jnp.argmax(out["logits"], -1)
        return jnp.asarray([
            jnp.sum(ce * mask),
            jnp.sum((pred == yb).astype(jnp.float32) * mask),
            jnp.sum(mask),
            jnp.sum(out["eta_mean"] * mask),
            jnp.sum(out["alpha_mean"] * mask),
        ])

    outdir = Path(args.outdir) / protocol
    outdir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    best_epoch = 0
    stale = 0
    last_epoch = 0
    eval_total = (len(yva) + args.eval_batch_size - 1) // args.eval_batch_size

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        t0 = time.time()
        sums = np.zeros(8, np.float64)
        nstep = 0

        for xcan, xjit, yb in iter_train(
            Xcan, Xjit, ytr, args.batch_size, args.seed + epoch
        ):
            state, key, metrics = train_step(
                state, key,
                jnp.asarray(xcan), jnp.asarray(xjit), jnp.asarray(yb),
            )
            vals = np.asarray(jax.device_get(metrics), np.float64)
            sums += vals
            nstep += 1
            if nstep % args.progress_every == 0 or nstep == steps_per_epoch:
                m = sums / nstep
                emit_status(
                    protocol, epoch, args.epochs, "TRAIN", nstep, steps_per_epoch,
                    loss=f"{m[0]:.4f}", acc=f"{100*m[3]:.2f}",
                    jit=f"{100*m[4]:.2f}", agr=f"{100*m[5]:.1f}",
                    eta=f"{m[6]:.4f}", alpha=f"{m[7]:.4f}",
                    best=f"{100*best:.3f}" if best >= 0 else "NA",
                )

        ev = np.zeros(5, np.float64)
        estep = 0
        for xb, yb, mask in iter_eval(Xva, yva, args.eval_batch_size):
            vals = np.asarray(jax.device_get(eval_step(
                state.ema_params, jnp.asarray(xb), jnp.asarray(yb), jnp.asarray(mask)
            )), np.float64)
            ev += vals
            estep += 1
            if estep % args.eval_progress_every == 0 or estep == eval_total:
                emit_status(
                    protocol, epoch, args.epochs, "VAL", estep, eval_total,
                    acc=f"{100*ev[1]/max(ev[2],1):.3f}",
                    best=f"{100*best:.3f}" if best >= 0 else "NA",
                )

        val_acc = ev[1] / max(ev[2], 1.0)
        val_loss = ev[0] / max(ev[2], 1.0)
        eta = ev[3] / max(ev[2], 1.0)
        alpha = ev[4] / max(ev[2], 1.0)

        if val_acc > best + 1e-6:
            best = val_acc
            best_epoch = epoch
            stale = 0
            payload = {
                "model": "DistalSMSpecialistT16-v1",
                "protocol": protocol,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "params": state.params,
                "ema_params": state.ema_params,
                "config": vars(args),
                "representation": {
                    "raw_frames": "variable",
                    "processing_frames": FRAMES,
                    "features_per_token": FEATURES,
                    "distal_joint_ids_zero_based": DISTAL_JOINT_IDS.tolist(),
                    "distal_focus": "wrists,hands,hand-tips,thumbs,ankles,feet",
                    "attention": False,
                    "training_from_scratch": True,
                    "classes": NUM_CLASSES,
                },
                "raw_xla_flops": raw_flops,
            }
            (outdir / "best.msgpack").write_bytes(serialization.to_bytes(payload))
            (outdir / "best.json").write_text(json.dumps({
                "model": "DistalSMSpecialistT16-v1",
                "epoch": epoch,
                "val_accuracy": val_acc,
                "params": nparams,
                "raw_xla_flops": raw_flops,
                "processing_frames": FRAMES,
                "features_per_token": FEATURES,
                "distal_joint_ids_zero_based": DISTAL_JOINT_IDS.tolist(),
            }, indent=2))
        else:
            stale += 1

        emit_status(
            protocol, epoch, args.epochs, "EPOCH", 1, 1,
            val=f"{100*val_acc:.3f}", valloss=f"{val_loss:.4f}",
            eta=f"{eta:.4f}", alpha=f"{alpha:.4f}",
            best=f"{100*best:.3f}", best_e=best_epoch,
            stale=stale, sec=f"{time.time()-t0:.1f}",
        )

        if stale >= args.patience:
            print(
                f"EARLY_STOP|protocol={protocol}|last_epoch={epoch}|"
                f"best_epoch={best_epoch}|best={best:.9f}", flush=True,
            )
            break

    result = {
        "model": "DistalSMSpecialistT16-v1",
        "protocol": protocol,
        "best_val_accuracy": best,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "params": nparams,
        "raw_xla_flops": raw_flops,
        "processing_frames": FRAMES,
        "features_per_token": FEATURES,
        "distal_joint_ids_zero_based": DISTAL_JOINT_IDS.tolist(),
        "seed": args.seed,
    }
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"result_{protocol}.json").write_text(compact_json(result))
    print("RESULT|" + compact_json(result), flush=True)
    return result


def main():
    args = parse_args()
    print("="*118, flush=True)
    print(f"DISTAL-SM SPECIALIST T16 | {args.protocol.upper()} | FROM SCRATCH | SINGLE T4", flush=True)
    print("JOINTS: WRISTS/HANDS/HAND-TIPS/THUMBS + ANKLES/FEET", flush=True)
    print("RAW VARIABLE LENGTH -> FIXED T16 | 120-WAY SPECIALIST | NO ATTENTION", flush=True)
    print("JAX", jax.__version__, "BACKEND", jax.default_backend(), "DEVICES", jax.local_devices(), flush=True)
    print("="*118, flush=True)
    dataset = base.find_dataset(args.dataset)
    annotations, split = base.load_ntu(dataset)
    train_protocol(args, annotations, split, args.protocol)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, traverse_util

from experiments.parttrace_v3_attention_lite.audit_model import (
    FRAMES, NUM_CLASSES, EXPECTED_BASE_PARAMS, EXPECTED_BASE_LEAVES,
    load_canonical_prefix, tree_numel, tree_leaves,
)
from experiments.parttrace_v3_attention_lite.model_v32 import make_wrapper_v32

EXPECTED_COUNTS = {"xsub": (63_026, 50_919), "xset": (54_468, 59_477)}


def parse_args():
    p = argparse.ArgumentParser(description="Train Attention-Lite + PartTrace v3.2 on one T4")
    p.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    p.add_argument("--dataset", default="auto")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)

    # Residual branch widths. Canonical Attention-Lite stays fixed at D128.
    p.add_argument("--part-dim", type=int, default=64)
    p.add_argument("--part-heads", type=int, default=4)
    p.add_argument("--global-dim", type=int, default=128)
    p.add_argument("--dense-dim", type=int, default=192,
                   help="Final PartTrace fusion Dense width (D_dense)")
    p.add_argument("--branch-dropout", type=float, default=0.12)

    # Differential learning rates.
    p.add_argument("--base-lr", type=float, default=4e-4)
    p.add_argument("--branch-lr", type=float, default=6e-4)
    p.add_argument("--controller-lr", type=float, default=1.5e-4)
    p.add_argument("--base-min-lr", type=float, default=1e-5)
    p.add_argument("--branch-min-lr", type=float, default=2e-5)
    p.add_argument("--controller-min-lr", type=float, default=5e-6)
    p.add_argument("--warmup-fraction", type=float, default=0.08)

    p.add_argument("--base-weight-decay", type=float, default=0.03)
    p.add_argument("--branch-weight-decay", type=float, default=0.04)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)

    p.add_argument("--predictive-loss-weight", type=float, default=0.10)
    p.add_argument("--parttrace-aux-weight", type=float, default=0.10)
    p.add_argument("--controller-kl-weight", type=float, default=0.02)
    p.add_argument("--freeze-branch-epochs", type=int, default=2)
    p.add_argument("--branch-ramp-epochs", type=int, default=4)

    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--outdir", default="/kaggle/working/NestSAR_PartTrace_v32")
    p.add_argument("--save-online", action="store_true")
    p.add_argument("--audit-first", action="store_true")

    # Notebook parent reads this file and renders tqdm in-place.
    p.add_argument("--progress-json", default=None)
    p.add_argument("--progress-every", type=int, default=5)
    return p.parse_args()


def find_dataset(explicit: str) -> Path:
    if explicit != "auto":
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    preferred = Path("/kaggle/input/models/paolamaydana/ntudanno/other/default/1/ntu120_3danno.pkl")
    if preferred.is_file():
        return preferred
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            hits = list(root.rglob("ntu120_3danno.pkl"))
            if hits:
                return hits[0]
    raise FileNotFoundError("ntu120_3danno.pkl not found")


def smoothed_ce(logits, y, smoothing: float):
    targets = jax.nn.one_hot(y, NUM_CLASSES)
    targets = targets * (1.0 - smoothing) + smoothing / NUM_CLASSES
    return -jnp.mean(jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1))


def branch_scale_for_epoch(epoch: int, freeze_epochs: int, ramp_epochs: int) -> float:
    if epoch <= freeze_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (epoch - freeze_epochs) / ramp_epochs)))


def make_param_labels(params):
    flat = traverse_util.flatten_dict(params)
    labels = {}
    counts = {"base": 0, "branch": 0, "controller": 0}
    for path, value in flat.items():
        name = "/".join(path)
        if "stream_controller" in name or "parttrace_residual_gate_logit" in name:
            label = "controller"
        elif "parttrace_branch" in name:
            label = "branch"
        else:
            label = "base"
        labels[path] = label
        counts[label] += int(np.asarray(value).size)
    return traverse_util.unflatten_dict(labels), counts


def warm_cosine(peak, end, warmup_steps, total_steps):
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=end,
    )


def _write_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _compiled_flops(compiled) -> float:
    c = compiled.cost_analysis()
    if isinstance(c, list) and c:
        c = c[0]
    return float(c.get("flops", 0.0)) if isinstance(c, dict) else 0.0


def main() -> int:
    a = parse_args()
    if a.part_dim <= 0 or a.global_dim <= 0 or a.dense_dim <= 0:
        raise ValueError("part/global/dense dimensions must be positive")
    if a.part_heads <= 0 or a.part_dim % a.part_heads:
        raise ValueError("--part-dim must be divisible by --part-heads")
    if not 0.0 <= a.branch_dropout < 1.0:
        raise ValueError("--branch-dropout must be in [0,1)")

    progress_path = Path(a.progress_json) if a.progress_json else None
    _write_progress(progress_path, {
        "protocol": a.protocol, "phase": "initializing", "epoch": 0,
        "n": 0, "total": 1, "message": "loading canonical Attention-Lite",
    })

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required; got {jax.default_backend()}")
    devices = list(jax.local_devices())
    if len(devices) != 1:
        raise RuntimeError(f"Expected exactly one process-visible GPU; got {devices}")

    mod, source = load_canonical_prefix(a.protocol)
    ns = mod.ns
    base = mod.build_model()
    model = make_wrapper_v32(
        base,
        part_dim=a.part_dim,
        part_heads=a.part_heads,
        global_dim=a.global_dim,
        dense_dim=a.dense_dim,
        branch_dropout=a.branch_dropout,
    )

    rng = jax.random.PRNGKey(a.seed)
    rng, brng, bdrop = jax.random.split(rng, 3)
    dummy = jnp.zeros((1, FRAMES, 150), jnp.float32)
    base_vars = base.init({"params": brng, "dropout": bdrop}, dummy, training=True)
    bp, bl = tree_numel(base_vars["params"]), tree_leaves(base_vars["params"])
    if (bp, bl) != (EXPECTED_BASE_PARAMS, EXPECTED_BASE_LEAVES):
        raise RuntimeError(
            f"CANONICAL BASE GUARD FAILED params={bp}/{EXPECTED_BASE_PARAMS} leaves={bl}/{EXPECTED_BASE_LEAVES}"
        )

    rng, irng, drng = jax.random.split(rng, 3)
    params = model.init(
        {"params": irng, "dropout": drng},
        dummy,
        training=True,
        branch_scale=0.0,
    )["params"]
    tp, tl = tree_numel(params), tree_leaves(params)
    added = tp - bp
    # Wide enough for intentional D sweeps, while still catching accidental backbone replacement.
    if not (50_000 <= added <= 1_500_000):
        raise RuntimeError(f"V3.2 SIZE GUARD FAILED: added params={added:,}")

    labels, group_counts = make_param_labels(params)
    if min(group_counts.values()) <= 0:
        raise RuntimeError(f"Optimizer group guard failed: {group_counts}")

    dataset = find_dataset(a.dataset)
    raw = ns.load_pickle(dataset)
    train_samples, val_samples = ns.build_samples(
        raw,
        protocol=a.protocol,
        max_train=a.max_train_samples,
        max_val=a.max_val_samples,
        seed=a.seed,
    )
    if a.max_train_samples == 0 and a.max_val_samples == 0:
        expected = EXPECTED_COUNTS[a.protocol]
        if (len(train_samples), len(val_samples)) != expected:
            raise RuntimeError(
                f"Split mismatch got={(len(train_samples),len(val_samples))}, expected={expected}"
            )

    train_ds = ns.SkeletonDataset(train_samples)
    val_ds = ns.SkeletonDataset(val_samples)
    train_steps = math.ceil(len(train_ds) / a.batch_size)
    val_steps = math.ceil(len(val_ds) / a.eval_batch_size)
    total_steps = max(1, train_steps * a.epochs)
    warmup_steps = max(1, int(total_steps * a.warmup_fraction))

    base_sched = warm_cosine(a.base_lr, a.base_min_lr, warmup_steps, total_steps)
    branch_sched = warm_cosine(a.branch_lr, a.branch_min_lr, warmup_steps, total_steps)
    controller_sched = warm_cosine(a.controller_lr, a.controller_min_lr, warmup_steps, total_steps)
    multi = optax.multi_transform(
        {
            "base": optax.adamw(base_sched, b1=0.9, b2=0.999, eps=1e-8, weight_decay=a.base_weight_decay),
            "branch": optax.adamw(branch_sched, b1=0.9, b2=0.999, eps=1e-8, weight_decay=a.branch_weight_decay),
            "controller": optax.adamw(controller_sched, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0),
        },
        labels,
    )
    tx = optax.chain(optax.clip_by_global_norm(a.grad_clip), multi)
    opt_state = tx.init(params)
    ema_params = params

    print("=" * 118, flush=True)
    print("NESTSAR ATTENTION-LITE + PARTTRACE V3.2 — CONFIGURABLE DENSE TRAINER", flush=True)
    print("=" * 118, flush=True)
    print(f"Protocol: {a.protocol.upper()} | GPU: {devices[0]}", flush=True)
    print(f"Canonical source: {source}", flush=True)
    print(f"Base: {bp:,} params / {bl} leaves | Added: {added:,} | Total: {tp:,} / {tl} leaves", flush=True)
    print(
        f"Widths: part_dim={a.part_dim} heads={a.part_heads} "
        f"global_dim={a.global_dim} dense_dim={a.dense_dim}", flush=True
    )
    print(f"Optimizer groups: {group_counts}", flush=True)
    print(f"LR base={a.base_lr:g} branch={a.branch_lr:g} controller={a.controller_lr:g}", flush=True)
    print(f"WD base={a.base_weight_decay:g} branch={a.branch_weight_decay:g} | EMA={a.ema_decay:g}", flush=True)
    print("=" * 118, flush=True)

    audit = {}
    if a.audit_first:
        base_comp = jax.jit(
            lambda p, xx: base.apply({"params": p}, xx, training=False)["logits"]
        ).lower(base_vars["params"], dummy).compile()
        full_comp = jax.jit(
            lambda p, xx: model.apply({"params": p}, xx, training=False, branch_scale=1.0)["logits"]
        ).lower(params, dummy).compile()
        bf, ff = _compiled_flops(base_comp), _compiled_flops(full_comp)
        audit = {
            "base_gflops": bf / 1e9,
            "v32_gflops": ff / 1e9,
            "added_gflops": (ff - bf) / 1e9,
        }
        print(f"XLA base GFLOPs:     {audit['base_gflops']:.9f}", flush=True)
        print(f"XLA v3.2 GFLOPs:     {audit['v32_gflops']:.9f}", flush=True)
        print(f"XLA added GFLOPs:    {audit['added_gflops']:.9f}", flush=True)

    @jax.jit
    def train_step(params, ema_params, opt_state, x, y, step_rng, branch_scale):
        def loss_fn(p):
            out = model.apply(
                {"params": p}, x,
                training=True,
                branch_scale=branch_scale,
                rngs={"dropout": step_rng},
            )
            main_ce = smoothed_ce(out["logits"], y, a.label_smoothing)
            branch_ce = smoothed_ce(out["parttrace_logits"], y, a.label_smoothing)
            predictive = jnp.array(0.0, jnp.float32)
            if "prediction" in out and "motion_target" in out:
                predictive = jnp.mean(
                    jnp.square(out["prediction"] - jax.lax.stop_gradient(out["motion_target"]))
                )
            dyn = jnp.clip(out["dynamic_fusion_weights"], 1e-8, 1.0)
            base_w = jnp.asarray(out["fusion_weights"])
            if base_w.ndim == 1:
                base_w = jnp.broadcast_to(base_w[None, :], dyn.shape)
            base_w = jnp.clip(base_w, 1e-8, 1.0)
            controller_kl = jnp.mean(jnp.sum(dyn * (jnp.log(dyn) - jnp.log(base_w)), axis=-1))
            loss = (
                main_ce
                + a.parttrace_aux_weight * branch_ce
                + a.predictive_loss_weight * predictive
                + a.controller_kl_weight * controller_kl
            )
            acc = jnp.mean(jnp.argmax(out["logits"], axis=-1) == y)
            return loss, (
                main_ce, branch_ce, predictive, controller_kl, acc,
                out["parttrace_gate"], out["effective_parttrace_gate"],
                jnp.mean(out["dynamic_fusion_weights"], axis=0),
                out["joint_to_part_entropy"],
                out["temporal_entropy_global"],
                out["temporal_entropy_left"],
                out["temporal_entropy_right"],
            )

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = jax.tree_util.tree_map(
            lambda e, p: a.ema_decay * e + (1.0 - a.ema_decay) * p,
            ema_params, params,
        )
        return params, ema_params, opt_state, loss, aux, grad_norm

    @jax.jit
    def eval_step(params, x, y, branch_scale):
        out = model.apply({"params": params}, x, training=False, branch_scale=branch_scale)
        pred = jnp.argmax(out["logits"], axis=-1)
        branch_pred = jnp.argmax(out["parttrace_logits"], axis=-1)
        return (
            jnp.sum(pred == y),
            jnp.sum(branch_pred == y),
            out["parttrace_gate"],
            out["effective_parttrace_gate"],
            jnp.mean(out["dynamic_fusion_weights"], axis=0),
        )

    outdir = Path(a.outdir) / a.protocol
    outdir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    start = time.time()
    every = max(1, a.progress_every)

    for epoch in range(1, a.epochs + 1):
        scale = branch_scale_for_epoch(epoch, a.freeze_branch_epochs, a.branch_ramp_epochs)
        scale_j = jnp.asarray(scale, jnp.float32)
        tr_correct = tr_count = 0
        sums = np.zeros(5, np.float64)
        last_gate = last_eff = last_gn = 0.0
        last_weights = np.zeros(4, np.float32)
        last_ent = np.zeros(4, np.float64)
        epoch_rng = jax.random.fold_in(rng, epoch)

        _write_progress(progress_path, {
            "protocol": a.protocol, "phase": "train", "epoch": epoch,
            "epochs": a.epochs, "n": 0, "total": train_steps,
            "accuracy": 0.0, "best": best, "best_epoch": best_epoch,
        })

        for bi, (x_np, y_np) in enumerate(ns.batch_iterator(
            train_ds, batch_size=a.batch_size, shuffle=True, seed=a.seed + epoch, drop_last=False
        )):
            x = jnp.asarray(x_np, jnp.float32)
            y = jnp.asarray(y_np, jnp.int32)
            step_rng = jax.random.fold_in(epoch_rng, bi)
            params, ema_params, opt_state, loss, aux, gn = train_step(
                params, ema_params, opt_state, x, y, step_rng, scale_j
            )
            (
                main_ce, branch_ce, predictive, controller_kl, acc,
                gate, eff_gate, weights, ent_part, ent_g, ent_l, ent_r,
            ) = aux
            n = len(y_np)
            tr_count += n
            tr_correct += int(round(float(acc) * n))
            vals = [loss, main_ce, branch_ce, predictive, controller_kl]
            sums += np.asarray([float(v) for v in vals]) * n
            last_gate, last_eff, last_gn = float(gate), float(eff_gate), float(gn)
            last_weights = np.asarray(weights)
            last_ent = np.asarray([ent_part, ent_g, ent_l, ent_r], dtype=np.float64)

            if (bi + 1) % every == 0 or (bi + 1) == train_steps:
                _write_progress(progress_path, {
                    "protocol": a.protocol, "phase": "train", "epoch": epoch,
                    "epochs": a.epochs, "n": bi + 1, "total": train_steps,
                    "accuracy": tr_correct / max(1, tr_count),
                    "loss": sums[0] / max(1, tr_count),
                    "best": best, "best_epoch": best_epoch,
                })

        val_correct = val_branch_correct = val_count = 0
        val_gate = val_eff = 0.0
        val_weights_sum = np.zeros(4, np.float64)
        val_batches = 0
        _write_progress(progress_path, {
            "protocol": a.protocol, "phase": "val", "epoch": epoch,
            "epochs": a.epochs, "n": 0, "total": val_steps,
            "accuracy": 0.0, "best": best, "best_epoch": best_epoch,
        })

        for vi, (x_np, y_np) in enumerate(ns.batch_iterator(
            val_ds, batch_size=a.eval_batch_size, shuffle=False, seed=0, drop_last=False
        )):
            x = jnp.asarray(x_np, jnp.float32)
            y = jnp.asarray(y_np, jnp.int32)
            correct, bcorrect, gate, eff_gate, weights = eval_step(ema_params, x, y, scale_j)
            val_correct += int(correct)
            val_branch_correct += int(bcorrect)
            val_count += len(y_np)
            val_gate, val_eff = float(gate), float(eff_gate)
            val_weights_sum += np.asarray(weights)
            val_batches += 1

            if (vi + 1) % every == 0 or (vi + 1) == val_steps:
                _write_progress(progress_path, {
                    "protocol": a.protocol, "phase": "val", "epoch": epoch,
                    "epochs": a.epochs, "n": vi + 1, "total": val_steps,
                    "accuracy": val_correct / max(1, val_count),
                    "best": best, "best_epoch": best_epoch,
                })

        train_acc = tr_correct / max(1, tr_count)
        val_acc = val_correct / max(1, val_count)
        branch_val_acc = val_branch_correct / max(1, val_count)
        val_weights = val_weights_sum / max(1, val_batches)

        improved = val_acc > best
        if improved:
            best, best_epoch, stale = val_acc, epoch, 0
            payload = {
                "model": "AttentionLitePartTraceV32",
                "protocol": a.protocol,
                "epoch": epoch,
                "seed": a.seed,
                "val_accuracy": val_acc,
                "ema_params": ema_params,
                "args": vars(a),
            }
            if a.save_online:
                payload["params"] = params
            (outdir / "best_ema.msgpack").write_bytes(serialization.to_bytes(payload))
        else:
            stale += 1

        rec = {
            "epoch": epoch,
            "branch_scale": scale,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "parttrace_val_accuracy": branch_val_acc,
            "best_val_accuracy": best,
            "best_epoch": best_epoch,
            "loss": sums[0] / max(1, tr_count),
            "main_ce": sums[1] / max(1, tr_count),
            "branch_ce": sums[2] / max(1, tr_count),
            "predictive_loss": sums[3] / max(1, tr_count),
            "controller_kl": sums[4] / max(1, tr_count),
            "parttrace_gate": val_gate,
            "effective_parttrace_gate": val_eff,
            "dynamic_fusion_weights": val_weights.tolist(),
            "joint_to_part_entropy": float(last_ent[0]),
            "temporal_entropy_global": float(last_ent[1]),
            "temporal_entropy_left": float(last_ent[2]),
            "temporal_entropy_right": float(last_ent[3]),
            "grad_norm_last": last_gn,
            "stale_epochs": stale,
        }
        history.append(rec)
        (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        _write_progress(progress_path, {
            "protocol": a.protocol, "phase": "epoch_done", "epoch": epoch,
            "epochs": a.epochs, "n": 1, "total": 1,
            "accuracy": val_acc, "train_accuracy": train_acc,
            "parttrace_accuracy": branch_val_acc,
            "best": best, "best_epoch": best_epoch,
            "stale": stale, "patience": a.patience,
        })

        wtxt = "/".join(f"{w:.2f}" for w in val_weights)
        print(
            f"{a.protocol.upper()} E{epoch:03}/{a.epochs} | "
            f"train={100*train_acc:.2f}% | val={100*val_acc:.2f}% | "
            f"PT={100*branch_val_acc:.2f}% | BEST={100*best:.2f}%@E{best_epoch:03} | "
            f"scale={scale:.2f} gate={val_eff:.4f} | W={wtxt} | stale={stale}/{a.patience}",
            flush=True,
        )
        if a.patience > 0 and stale >= a.patience:
            break

    result = {
        "model": "AttentionLitePartTraceV32",
        "protocol": a.protocol,
        "seed": a.seed,
        "base_params": bp,
        "added_params": added,
        "params": tp,
        "leaves": tl,
        "widths": {
            "part_dim": a.part_dim,
            "part_heads": a.part_heads,
            "global_dim": a.global_dim,
            "dense_dim": a.dense_dim,
        },
        "audit": audit,
        "optimizer_group_params": group_counts,
        "best_val_accuracy": best,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "wall_hours": (time.time() - start) / 3600.0,
        "args": vars(a),
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_progress(progress_path, {
        "protocol": a.protocol, "phase": "done", "epoch": len(history),
        "epochs": a.epochs, "n": 1, "total": 1,
        "accuracy": best, "best": best, "best_epoch": best_epoch,
    })
    print(f"{a.protocol.upper()} COMPLETE | best={100*best:.5f}% @ epoch {best_epoch}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

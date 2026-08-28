#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-T4 worker for NestSAR v3.5 Cross-Stream Multi-Resolution Memory.

This is the GPU counterpart of train_v35_tpu.py and follows the validated
v3.4 dual-T4 method:
- one process sees exactly one physical T4;
- XSUB/XSET are launched concurrently by the notebook parent;
- exact canonical Attention-Lite guards;
- protocol-safe pretrained EMA transplant + epoch-0 baseline validation;
- branch warm-up, protected base unfreeze, differential LR, EMA;
- progress JSON for persistent notebook tqdm rows;
- best confusion matrix and per-class accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from experiments.parttrace_v3_attention_lite import train_v35_tpu as core
from experiments.parttrace_v3_attention_lite.audit_model import (
    FRAMES, NUM_CLASSES, EXPECTED_BASE_PARAMS, EXPECTED_BASE_LEAVES,
    load_canonical_prefix, tree_numel, tree_leaves,
)
from experiments.parttrace_v3_attention_lite.model_v35_crossstream_memory import (
    make_wrapper_v35,
)

EXPECTED_COUNTS = core.EXPECTED_COUNTS
MODEL_NAME = core.MODEL_NAME


def parse_args():
    p = argparse.ArgumentParser(description="NestSAR v3.5 single-T4 worker")
    p.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    p.add_argument("--dataset", default="auto")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--base-checkpoint", default="auto")
    p.add_argument("--allow-scratch", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)

    p.add_argument("--bridge-dim", type=int, default=32)
    p.add_argument("--local-stream-dim", type=int, default=16)
    p.add_argument("--memory-dim", type=int, default=32)
    p.add_argument("--fine-dim", type=int, default=24)
    p.add_argument("--readout-tokens", type=int, default=8)
    p.add_argument("--readout-heads", type=int, default=4)
    p.add_argument("--dense-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.08)
    p.add_argument("--stream-reweight-strength", type=float, default=0.08)

    p.add_argument("--base-lr", type=float, default=7.5e-5)
    p.add_argument("--new-lr", type=float, default=5e-4)
    p.add_argument("--base-min-lr", type=float, default=3e-6)
    p.add_argument("--new-min-lr", type=float, default=2e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.06)
    p.add_argument("--base-weight-decay", type=float, default=0.015)
    p.add_argument("--new-weight-decay", type=float, default=0.03)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997)

    p.add_argument("--predictive-loss-weight", type=float, default=0.03)
    p.add_argument("--memory-aux-warmup-weight", type=float, default=0.45)
    p.add_argument("--memory-aux-final-weight", type=float, default=0.15)
    p.add_argument("--diversity-loss-weight", type=float, default=0.03)
    p.add_argument("--stream-kl-weight", type=float, default=0.01)
    p.add_argument("--freeze-base-epochs", type=int, default=3)
    p.add_argument("--base-unfreeze-ramp-epochs", type=int, default=3)
    p.add_argument("--freeze-branch-epochs", type=int, default=2)
    p.add_argument("--branch-ramp-epochs", type=int, default=4)

    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--outdir", default="/kaggle/working/NestSAR_CrossStreamMemory_v35_DualT4")
    p.add_argument("--save-online", action="store_true")
    p.add_argument("--audit-first", action="store_true")
    p.add_argument("--progress-json", default=None)
    p.add_argument("--progress-every", type=int, default=5)
    return p.parse_args()


def _as_text(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def load_safe_base(spec: str, protocol: str, template):
    """Protocol-safe, lineage-safe selection of the strongest canonical EMA.

    In ``auto`` mode, all compatible canonical Attention-Lite checkpoints are
    inspected and the one with the highest stored validation accuracy is chosen.
    This avoids silently taking the first seed/path returned by rglob.
    """
    errors = []
    auto = spec.lower() == "auto"
    excluded = ("tokenpreserve", "parttrace", "crossstream", "cross_stream", "v35")
    matches = []
    for path in core._checkpoint_candidates(spec, protocol):
        try:
            payload = serialization.msgpack_restore(path.read_bytes())
            declared = None
            model_name = ""
            stored_val = -1.0
            if isinstance(payload, Mapping):
                if payload.get("protocol") is not None:
                    declared = _as_text(payload.get("protocol")).lower().strip()
                    if declared and declared != protocol:
                        errors.append(
                            f"{path}: declared protocol={declared!r}, requested={protocol!r}"
                        )
                        continue
                if payload.get("model") is not None:
                    model_name = _as_text(payload.get("model")).lower()
                for key in ("val_accuracy", "best_val_accuracy", "best_accuracy"):
                    if payload.get(key) is not None:
                        try:
                            stored_val = float(payload.get(key))
                            break
                        except (TypeError, ValueError):
                            pass

            path_text = str(path).lower()
            if auto:
                if any(tag in path_text or tag in model_name for tag in excluded):
                    errors.append(f"{path}: excluded non-canonical experiment lineage")
                    continue
                if declared is None and protocol not in path_text:
                    errors.append(f"{path}: ambiguous protocol in auto mode")
                    continue

            roots = []
            if isinstance(payload, Mapping):
                for key in ("ema_params", "params"):
                    if key in payload:
                        roots.append(payload[key])
            roots.append(payload)
            found = False
            for root in roots:
                match_path, subtree = core._find_matching_subtree(root, template)
                if subtree is not None:
                    loaded = jax.tree_util.tree_map(lambda z: jnp.asarray(z), subtree)
                    matches.append((
                        stored_val,
                        core._rank_checkpoint(path, protocol),
                        str(path),
                        loaded,
                        path,
                        match_path,
                    ))
                    found = True
                    break
            if not found:
                errors.append(f"{path}: no canonical-shaped Attention-Lite subtree")
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    if matches:
        matches.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best = matches[0]
        meta = {
            "subtree": best[5],
            "stored_val_accuracy": best[0],
            "compatible_candidates": len(matches),
        }
        return best[3], best[4], meta
    return None, None, errors


def _checkpoint_payload(a, epoch, val_acc, baseline_acc, base_checkpoint,
                        ema_params, branch_scale, base_grad_scale, params=None):
    payload = {
        "model": MODEL_NAME,
        "protocol": a.protocol,
        "epoch": int(epoch),
        "seed": int(a.seed),
        "val_accuracy": float(val_acc),
        "baseline_accuracy": float(baseline_acc),
        "base_checkpoint": str(base_checkpoint),
        "branch_scale": float(branch_scale),
        "base_grad_scale": float(base_grad_scale),
        "checkpoint_semantics":
            "Evaluate this EMA with the stored branch_scale; epoch-0 is canonical base only.",
        "ema_params": ema_params,
        "args": vars(a),
    }
    if params is not None:
        payload["params"] = params
    return payload


def main() -> int:
    a = parse_args()
    if a.frames != 16:
        raise ValueError("v3.5 is intentionally T16")
    if a.memory_dim % a.readout_heads:
        raise ValueError("--memory-dim must be divisible by --readout-heads")
    if a.bridge_dim % 4 or a.local_stream_dim % 4:
        raise ValueError("--bridge-dim and --local-stream-dim must be divisible by 4")
    if not 1 <= a.readout_tokens <= 32:
        raise ValueError("--readout-tokens must be in [1,32]")
    if not 0.0 <= a.dropout < 1.0:
        raise ValueError("--dropout must be in [0,1)")

    progress_path = Path(a.progress_json) if a.progress_json else None
    core._write_progress(progress_path, {
        "protocol": a.protocol, "phase": "initializing", "epoch": 0,
        "epochs": a.epochs, "n": 0, "total": 1,
        "message": "building exact Attention-Lite + v3.5 on one T4",
    })

    backend = jax.default_backend()
    devices = list(jax.local_devices())
    if backend != "gpu":
        raise RuntimeError(f"Kaggle GPU required; backend={backend}")
    if len(devices) != 1:
        raise RuntimeError(
            f"Expected exactly one process-visible GPU; found {len(devices)}: {devices}"
        )

    mod, source = load_canonical_prefix(a.protocol)
    ns = mod.ns
    base = mod.build_model()
    model = make_wrapper_v35(
        base,
        bridge_dim=a.bridge_dim,
        local_stream_dim=a.local_stream_dim,
        memory_dim=a.memory_dim,
        fine_dim=a.fine_dim,
        readout_tokens=a.readout_tokens,
        readout_heads=a.readout_heads,
        dense_dim=a.dense_dim,
        dropout=a.dropout,
        stream_reweight_strength=a.stream_reweight_strength,
    )

    rng = jax.random.PRNGKey(a.seed)
    rng, brng, bdrop = jax.random.split(rng, 3)
    dummy = jnp.zeros((1, FRAMES, 150), jnp.float32)
    base_vars = base.init({"params": brng, "dropout": bdrop}, dummy, training=True)
    base_template = base_vars["params"]
    bp, bl = tree_numel(base_template), tree_leaves(base_template)
    if (bp, bl) != (EXPECTED_BASE_PARAMS, EXPECTED_BASE_LEAVES):
        raise RuntimeError(
            f"CANONICAL BASE GUARD FAILED params={bp}/{EXPECTED_BASE_PARAMS} "
            f"leaves={bl}/{EXPECTED_BASE_LEAVES}"
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
    if not (30_000 <= added <= 500_000):
        raise RuntimeError(f"V3.5 SIZE GUARD FAILED: added params={added:,}")

    loaded_base, loaded_path, load_meta = load_safe_base(
        a.base_checkpoint, a.protocol, base_template
    )
    if loaded_base is None:
        if not a.allow_scratch:
            preview = "\n".join(str(x) for x in load_meta[:16]) if load_meta else \
                "no compatible best_ema.msgpack found"
            raise RuntimeError(
                "No compatible pretrained canonical Attention-Lite checkpoint found. "
                "Pass --base-checkpoint PATH or attach a checkpoint dataset.\n" + preview
            )
        loaded_path_text = "SCRATCH"
        print("WARNING: using scratch canonical base", flush=True)
    else:
        full_base_path, _ = core._find_matching_subtree(params, base_template)
        if full_base_path is None:
            raise RuntimeError("Could not locate canonical base subtree inside v3.5 params")
        params = core._replace_path(params, full_base_path, loaded_base)
        loaded_path_text = str(loaded_path)
        print(f"PRETRAINED BASE LOADED: {loaded_path}", flush=True)
        print(f"Checkpoint subtree: {load_meta}", flush=True)

    labels, group_counts, base_mask = core.make_param_labels(params)
    if min(group_counts.values()) <= 0:
        raise RuntimeError(f"Optimizer group guard failed: {group_counts}")

    dataset = core.find_dataset(a.dataset)
    raw = ns.load_pickle(dataset)
    train_samples, val_samples = ns.build_samples(
        raw,
        protocol=a.protocol,
        max_train=a.max_train_samples,
        max_val=a.max_val_samples,
        seed=a.seed,
    )
    if a.max_train_samples == 0 and a.max_val_samples == 0:
        got = (len(train_samples), len(val_samples))
        if got != EXPECTED_COUNTS[a.protocol]:
            raise RuntimeError(
                f"Split mismatch got={got}, expected={EXPECTED_COUNTS[a.protocol]}"
            )
    train_ds = ns.SkeletonDataset(train_samples)
    val_ds = ns.SkeletonDataset(val_samples)
    train_steps = len(train_ds) // a.batch_size
    val_steps = math.ceil(len(val_ds) / a.eval_batch_size)
    total_steps = max(1, train_steps * a.epochs)
    warmup_steps = max(1, int(total_steps * a.warmup_fraction))

    base_sched = core.warm_cosine(
        a.base_lr, a.base_min_lr, warmup_steps, total_steps
    )
    new_sched = core.warm_cosine(
        a.new_lr, a.new_min_lr, warmup_steps, total_steps
    )
    multi = optax.multi_transform({
        "base": optax.adamw(
            base_sched, b1=0.9, b2=0.999, eps=1e-8,
            weight_decay=a.base_weight_decay,
        ),
        "new": optax.adamw(
            new_sched, b1=0.9, b2=0.999, eps=1e-8,
            weight_decay=a.new_weight_decay,
        ),
    }, labels)
    tx = optax.chain(optax.clip_by_global_norm(a.grad_clip), multi)
    opt_state = tx.init(params)
    ema_params = params

    print("=" * 120, flush=True)
    print("NESTSAR v3.5 — CROSS-STREAM MULTI-RESOLUTION MEMORY — SINGLE T4 WORKER",
          flush=True)
    print("=" * 120, flush=True)
    print(f"Protocol: {a.protocol.upper()} | backend={backend} | GPU={devices[0]}",
          flush=True)
    print(f"Canonical source: {source}", flush=True)
    print(f"Pretrained base: {loaded_path_text}", flush=True)
    print(f"Base={bp:,}/{bl} | Added={added:,} | Total={tp:,}/{tl}", flush=True)
    print(
        f"Memory: coarse=320xD{a.memory_dim} + fine=320xD{a.fine_dim} | "
        f"K={a.readout_tokens} H={a.readout_heads}",
        flush=True,
    )
    print(
        f"Bridge D{a.bridge_dim} | local 4-stream D{a.local_stream_dim} | "
        f"dense D{a.dense_dim}",
        flush=True,
    )
    print(f"Batch={a.batch_size} | eval={a.eval_batch_size} | optimizer={group_counts}",
          flush=True)
    print(
        f"LR base={a.base_lr:g}->{a.base_min_lr:g} | "
        f"new={a.new_lr:g}->{a.new_min_lr:g} | EMA={a.ema_decay}",
        flush=True,
    )
    print("=" * 120, flush=True)

    audit = {}
    if a.audit_first:
        _, base_for_audit = core._find_matching_subtree(params, base_template)
        base_comp = jax.jit(
            lambda p, xx: base.apply({"params": p}, xx, training=False)["logits"]
        ).lower(base_for_audit, dummy).compile()
        full_comp = jax.jit(
            lambda p, xx: model.apply(
                {"params": p}, xx, training=False, branch_scale=1.0
            )["logits"]
        ).lower(params, dummy).compile()
        bf, ff = core._compiled_flops(base_comp), core._compiled_flops(full_comp)
        audit = {
            "base_gflops": bf / 1e9,
            "v35_gflops": ff / 1e9,
            "added_gflops": (ff - bf) / 1e9 if bf and ff else 0.0,
        }
        print(f"XLA base GFLOPs:     {audit['base_gflops']:.9f}", flush=True)
        print(f"XLA v3.5 GFLOPs:     {audit['v35_gflops']:.9f}", flush=True)
        print(f"XLA added GFLOPs:    {audit['added_gflops']:.9f}", flush=True)

    outdir = Path(a.outdir) / a.protocol
    outdir.mkdir(parents=True, exist_ok=True)

    _, current_base = core._find_matching_subtree(params, base_template)

    @jax.jit
    def base_eval_step(bp_, x, y):
        out = base.apply({"params": bp_}, x, training=False)
        pred = jnp.argmax(out["logits"], axis=-1)
        return jnp.sum(pred == y), pred

    baseline_correct = 0
    baseline_count = 0
    baseline_labels = []
    baseline_preds = []
    core._write_progress(progress_path, {
        "protocol": a.protocol, "phase": "baseline", "epoch": 0,
        "epochs": a.epochs, "n": 0, "total": val_steps,
        "accuracy": 0.0, "best": -1.0, "best_epoch": 0,
    })
    every = max(1, a.progress_every)
    for vi, (x_np, y_np) in enumerate(ns.batch_iterator(
        val_ds,
        batch_size=a.eval_batch_size,
        shuffle=False,
        seed=0,
        drop_last=False,
    )):
        x = jnp.asarray(x_np, jnp.float32)
        y = jnp.asarray(y_np, jnp.int32)
        c, pred = base_eval_step(current_base, x, y)
        baseline_correct += int(c)
        baseline_count += len(y_np)
        baseline_labels.append(np.asarray(y_np, np.int32))
        baseline_preds.append(np.asarray(pred, np.int32))
        if (vi + 1) % every == 0 or vi + 1 == val_steps:
            core._write_progress(progress_path, {
                "protocol": a.protocol, "phase": "baseline", "epoch": 0,
                "epochs": a.epochs, "n": vi + 1, "total": val_steps,
                "accuracy": baseline_correct / max(1, baseline_count),
                "best": -1.0, "best_epoch": 0,
            })

    baseline_acc = baseline_correct / max(1, baseline_count)
    baseline_labels = np.concatenate(baseline_labels)
    baseline_preds = np.concatenate(baseline_preds)
    baseline_macro = core._save_class_metrics(
        outdir, baseline_labels, baseline_preds, "baseline"
    )
    print(
        f"PRETRAINED BASELINE VAL: {100*baseline_acc:.5f}% | "
        f"macro={100*baseline_macro:.3f}%",
        flush=True,
    )

    @jax.jit
    def train_step(params_, ema_, opt_, x, y, step_rng,
                   branch_scale, base_grad_scale, aux_weight):
        def loss_fn(p):
            out = model.apply(
                {"params": p},
                x,
                training=True,
                branch_scale=branch_scale,
                rngs={"dropout": step_rng},
            )
            main_ce = core.smoothed_ce(
                out["logits"], y, a.label_smoothing
            )
            memory_ce = core.smoothed_ce(
                out["memory_logits"], y, a.label_smoothing
            )
            predictive = jnp.array(0.0, jnp.float32)
            if "prediction" in out and "motion_target" in out:
                predictive = jnp.mean(jnp.square(
                    out["prediction"] -
                    jax.lax.stop_gradient(out["motion_target"])
                ))
            dyn = jnp.clip(out["dynamic_fusion_weights"], 1e-8, 1.0)
            bw = jnp.asarray(out["fusion_weights"])
            if bw.ndim == 1:
                bw = jnp.broadcast_to(bw[None], dyn.shape)
            bw = jnp.clip(bw, 1e-8, 1.0)
            stream_kl = jnp.mean(jnp.sum(
                dyn * (jnp.log(dyn) - jnp.log(bw)), axis=-1
            ))
            div = out["query_diversity_loss"]
            loss = (
                main_ce
                + aux_weight * memory_ce
                + a.predictive_loss_weight * predictive
                + a.diversity_loss_weight * div
                + a.stream_kl_weight * stream_kl
            )
            acc = jnp.mean(jnp.argmax(out["logits"], axis=-1) == y)
            mem_acc = jnp.mean(
                jnp.argmax(out["memory_logits"], axis=-1) == y
            )
            eff_gate_mean = jnp.mean(out["effective_class_gate"])
            return loss, jnp.asarray([
                main_ce, memory_ce, predictive, div, stream_kl,
                acc, mem_acc,
                out["class_gate_mean"], eff_gate_mean,
                out["query_overlap_mean"], out["query_overlap_max"],
            ], jnp.float32)

        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params_)
        grads = jax.tree_util.tree_map(
            lambda g, m: g * (1.0 - m + m * base_grad_scale),
            grads, base_mask,
        )
        grad_norm = optax.global_norm(grads)
        updates, opt_ = tx.update(grads, opt_, params_)
        params_ = optax.apply_updates(params_, updates)
        ema_ = jax.tree_util.tree_map(
            lambda e, p: a.ema_decay * e + (1.0 - a.ema_decay) * p,
            ema_, params_,
        )
        return params_, ema_, opt_, loss, metrics, grad_norm

    @jax.jit
    def eval_step(params_, x, y, branch_scale):
        out = model.apply(
            {"params": params_}, x,
            training=False, branch_scale=branch_scale,
        )
        pred = jnp.argmax(out["logits"], axis=-1)
        mem_pred = jnp.argmax(out["memory_logits"], axis=-1)
        dyn_sum = jnp.sum(out["dynamic_fusion_weights"], axis=0)
        diagnostics = jnp.asarray([
            out["class_gate_mean"],
            jnp.mean(out["effective_class_gate"]),
            out["class_gate_max"],
            out["class_gate_min"],
            out["query_overlap_mean"],
            out["query_overlap_max"],
        ], jnp.float32)
        return (
            jnp.sum(pred == y),
            jnp.sum(mem_pred == y),
            dyn_sum,
            diagnostics,
            pred,
        )

    initial_payload = _checkpoint_payload(
        a, 0, baseline_acc, baseline_acc, loaded_path_text,
        params, 0.0, 0.0,
        params=params if a.save_online else None,
    )
    (outdir / "best_ema.msgpack").write_bytes(
        serialization.to_bytes(initial_payload)
    )

    best = baseline_acc
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    start = time.time()
    global_step = 0

    for epoch in range(1, a.epochs + 1):
        epoch_start = time.time()
        branch_scale = core.branch_scale_for_epoch(
            epoch, a.freeze_branch_epochs, a.branch_ramp_epochs
        )
        base_scale = core.base_grad_scale_for_epoch(
            epoch, a.freeze_base_epochs, a.base_unfreeze_ramp_epochs
        )
        aux_weight = (
            a.memory_aux_warmup_weight * (1.0 - branch_scale)
            + a.memory_aux_final_weight * branch_scale
        )
        bs_j = jnp.asarray(branch_scale, jnp.float32)
        bg_j = jnp.asarray(base_scale, jnp.float32)
        aux_j = jnp.asarray(aux_weight, jnp.float32)

        tr_count = 0
        tr_correct = 0.0
        tr_mem_correct = 0.0
        metric_sum = np.zeros(12, np.float64)
        last_grad = 0.0
        core._write_progress(progress_path, {
            "protocol": a.protocol, "phase": "train", "epoch": epoch,
            "epochs": a.epochs, "n": 0, "total": train_steps,
            "accuracy": 0.0, "memory_accuracy": 0.0,
            "best": best, "best_epoch": best_epoch,
            "branch_scale": branch_scale,
            "base_grad_scale": base_scale,
        })

        for bi, (x_np, y_np) in enumerate(ns.batch_iterator(
            train_ds,
            batch_size=a.batch_size,
            shuffle=True,
            seed=a.seed + epoch,
            drop_last=True,
        )):
            x = jnp.asarray(x_np, jnp.float32)
            y = jnp.asarray(y_np, jnp.int32)
            step_rng = jax.random.fold_in(rng, global_step)
            params, ema_params, opt_state, loss, metrics, gn = train_step(
                params, ema_params, opt_state,
                x, y, step_rng, bs_j, bg_j, aux_j,
            )
            global_step += 1
            met = np.asarray(metrics)
            n = len(y_np)
            tr_count += n
            tr_correct += float(met[5]) * n
            tr_mem_correct += float(met[6]) * n
            metric_sum += np.concatenate([[float(loss)], met]) * n
            last_grad = float(gn)

            if (bi + 1) % every == 0 or bi + 1 == train_steps:
                core._write_progress(progress_path, {
                    "protocol": a.protocol, "phase": "train",
                    "epoch": epoch, "epochs": a.epochs,
                    "n": bi + 1, "total": train_steps,
                    "accuracy": tr_correct / max(1, tr_count),
                    "memory_accuracy": tr_mem_correct / max(1, tr_count),
                    "loss": metric_sum[0] / max(1, tr_count),
                    "best": best, "best_epoch": best_epoch,
                    "gate": metric_sum[9] / max(1, tr_count),
                    "qoverlap": metric_sum[10] / max(1, tr_count),
                    "branch_scale": branch_scale,
                    "base_grad_scale": base_scale,
                })

        val_correct = 0
        val_mem_correct = 0
        val_count = 0
        val_weights = np.zeros(4, np.float64)
        diag_sum = np.zeros(6, np.float64)
        val_labels = []
        val_preds = []

        core._write_progress(progress_path, {
            "protocol": a.protocol, "phase": "val", "epoch": epoch,
            "epochs": a.epochs, "n": 0, "total": val_steps,
            "accuracy": 0.0, "memory_accuracy": 0.0,
            "best": best, "best_epoch": best_epoch,
            "branch_scale": branch_scale,
            "base_grad_scale": base_scale,
        })

        for vi, (x_np, y_np) in enumerate(ns.batch_iterator(
            val_ds,
            batch_size=a.eval_batch_size,
            shuffle=False,
            seed=0,
            drop_last=False,
        )):
            x = jnp.asarray(x_np, jnp.float32)
            y = jnp.asarray(y_np, jnp.int32)
            c, mc, ws, dg, pred = eval_step(
                ema_params, x, y, bs_j
            )
            n = len(y_np)
            val_correct += int(c)
            val_mem_correct += int(mc)
            val_count += n
            val_weights += np.asarray(ws)
            diag_sum += np.asarray(dg) * n
            val_labels.append(np.asarray(y_np, np.int32))
            val_preds.append(np.asarray(pred, np.int32))

            if (vi + 1) % every == 0 or vi + 1 == val_steps:
                core._write_progress(progress_path, {
                    "protocol": a.protocol, "phase": "val",
                    "epoch": epoch, "epochs": a.epochs,
                    "n": vi + 1, "total": val_steps,
                    "accuracy": val_correct / max(1, val_count),
                    "memory_accuracy":
                        val_mem_correct / max(1, val_count),
                    "best": best, "best_epoch": best_epoch,
                    "branch_scale": branch_scale,
                    "base_grad_scale": base_scale,
                })

        train_acc = tr_correct / max(1, tr_count)
        train_mem_acc = tr_mem_correct / max(1, tr_count)
        val_acc = val_correct / max(1, val_count)
        mem_val_acc = val_mem_correct / max(1, val_count)
        weights_mean = val_weights / max(1, val_count)
        diag_mean = diag_sum / max(1, val_count)
        val_labels = np.concatenate(val_labels)
        val_preds = np.concatenate(val_preds)
        _, per_class = core._confusion_and_per_class(
            val_labels, val_preds
        )
        macro = float(per_class.mean())

        improved = val_acc > best
        if improved:
            best, best_epoch, stale = val_acc, epoch, 0
            payload = _checkpoint_payload(
                a, epoch, val_acc, baseline_acc, loaded_path_text,
                ema_params, branch_scale, base_scale,
                params=params if a.save_online else None,
            )
            (outdir / "best_ema.msgpack").write_bytes(
                serialization.to_bytes(payload)
            )
            core._save_class_metrics(
                outdir, val_labels, val_preds, "best"
            )
        else:
            stale += 1

        epoch_seconds = time.time() - epoch_start
        rec = {
            "epoch": epoch,
            "branch_scale": branch_scale,
            "base_grad_scale": base_scale,
            "memory_aux_weight": aux_weight,
            "train_accuracy": train_acc,
            "memory_train_accuracy": train_mem_acc,
            "val_accuracy": val_acc,
            "memory_val_accuracy": mem_val_acc,
            "macro_val_accuracy": macro,
            "best_val_accuracy": best,
            "best_epoch": best_epoch,
            "loss": metric_sum[0] / max(1, tr_count),
            "main_ce": metric_sum[1] / max(1, tr_count),
            "memory_ce": metric_sum[2] / max(1, tr_count),
            "predictive_loss": metric_sum[3] / max(1, tr_count),
            "diversity_loss": metric_sum[4] / max(1, tr_count),
            "stream_kl": metric_sum[5] / max(1, tr_count),
            "class_gate_mean": float(diag_mean[0]),
            "effective_class_gate_mean": float(diag_mean[1]),
            "class_gate_max": float(diag_mean[2]),
            "class_gate_min": float(diag_mean[3]),
            "query_overlap_mean": float(diag_mean[4]),
            "query_overlap_max": float(diag_mean[5]),
            "dynamic_fusion_weights": weights_mean.tolist(),
            "grad_norm_last": last_grad,
            "epoch_seconds": epoch_seconds,
            "train_samples_per_sec":
                tr_count / max(epoch_seconds, 1e-6),
            "stale_epochs": stale,
        }
        history.append(rec)
        (outdir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        core._write_progress(progress_path, {
            "protocol": a.protocol, "phase": "epoch_done",
            "epoch": epoch, "epochs": a.epochs,
            "n": 1, "total": 1,
            "accuracy": val_acc,
            "memory_accuracy": mem_val_acc,
            "macro_accuracy": macro,
            "best": best, "best_epoch": best_epoch,
            "stale": stale, "patience": a.patience,
            "loss": rec["loss"],
            "gate": rec["effective_class_gate_mean"],
            "qoverlap": rec["query_overlap_mean"],
            "branch_scale": branch_scale,
            "base_grad_scale": base_scale,
        })

        wtxt = "/".join(f"{w:.2f}" for w in weights_mean)
        print(
            f"{a.protocol.upper()} E{epoch:03}/{a.epochs} | "
            f"train={100*train_acc:.2f}% MEMtr={100*train_mem_acc:.2f}% | "
            f"val={100*val_acc:.2f}% MEM={100*mem_val_acc:.2f}% "
            f"macro={100*macro:.2f}% | "
            f"BEST={100*best:.2f}%@E{best_epoch:03} | "
            f"bs={branch_scale:.2f} bg={base_scale:.2f} "
            f"gate={rec['effective_class_gate_mean']:.3f} "
            f"Qov={rec['query_overlap_mean']:.3f} | "
            f"W={wtxt} | stale={stale}/{a.patience} | "
            f"{epoch_seconds:.1f}s",
            flush=True,
        )

        if a.patience > 0 and stale >= a.patience:
            break

    result = {
        "model": MODEL_NAME,
        "protocol": a.protocol,
        "seed": a.seed,
        "backend": backend,
        "gpu": str(devices[0]),
        "frames": 16,
        "coarse_tokens": 320,
        "fine_tokens": 320,
        "memory_tokens": 640,
        "readout_tokens": a.readout_tokens,
        "base_checkpoint": loaded_path_text,
        "baseline_accuracy": baseline_acc,
        "baseline_macro_accuracy": baseline_macro,
        "base_params": bp,
        "added_params": added,
        "params": tp,
        "leaves": tl,
        "audit": audit,
        "optimizer_group_params": group_counts,
        "best_val_accuracy": best,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "wall_hours": (time.time() - start) / 3600.0,
        "args": vars(a),
    }
    (outdir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    core._write_progress(progress_path, {
        "protocol": a.protocol, "phase": "done",
        "epoch": len(history), "epochs": a.epochs,
        "n": 1, "total": 1,
        "accuracy": best, "best": best, "best_epoch": best_epoch,
    })
    print(
        f"{a.protocol.upper()} COMPLETE | "
        f"best={100*best:.5f}% @ epoch {best_epoch}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

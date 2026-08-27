#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Attention-Lite + TokenPreserve v3.4 on one GPU.

v3.4 fixes the main v3.3 experiment mismatch:
1) initialize the canonical Attention-Lite base from a compatible trained EMA;
2) warm up the new branch while the base is frozen, then gently unfreeze it;
3) use stronger branch supervision and bounded residual fusion;
4) penalize readout-query collapse;
5) reduce masking on the token-preservation path;
6) report an epoch-0 pretrained-base validation before any optimization.
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

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, traverse_util

from experiments.parttrace_v3_attention_lite.audit_model import (
    FRAMES, NUM_CLASSES, EXPECTED_BASE_PARAMS, EXPECTED_BASE_LEAVES,
    load_canonical_prefix, tree_numel, tree_leaves,
)
from experiments.parttrace_v3_attention_lite.model_v34_tokenpreserve import make_wrapper_v34

EXPECTED_COUNTS = {"xsub": (63_026, 50_919), "xset": (54_468, 59_477)}
MODEL_NAME = "AttentionLiteTokenPreserveV34"


def parse_args():
    p = argparse.ArgumentParser(description="Train pretrained Attention-Lite + TokenPreserve v3.4")
    p.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    p.add_argument("--dataset", default="auto")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--base-checkpoint", default=os.environ.get("NESTSAR_BASE_CHECKPOINT", "auto"))
    p.add_argument("--allow-scratch", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)

    p.add_argument("--part-dim", type=int, default=64)
    p.add_argument("--part-heads", type=int, default=4)
    p.add_argument("--global-dim", type=int, default=128,
                   help="Tiny readout-mixer FFN width")
    p.add_argument("--dense-dim", type=int, default=192)
    p.add_argument("--readout-tokens", type=int,
                   default=int(os.environ.get("NESTSAR_READOUT_TOKENS", "8")))
    p.add_argument("--branch-dropout", type=float, default=0.10)
    p.add_argument("--frame-mask-rate", type=float,
                   default=float(os.environ.get("NESTSAR_FRAME_MASK", "0.03")))
    p.add_argument("--joint-mask-rate", type=float,
                   default=float(os.environ.get("NESTSAR_JOINT_MASK", "0.04")))
    p.add_argument("--part-mask-rate", type=float,
                   default=float(os.environ.get("NESTSAR_PART_MASK", "0.01")))

    # Pretrained-base fine-tuning is intentionally gentler than v3.3 scratch training.
    p.add_argument("--base-lr", type=float, default=1e-4)
    p.add_argument("--branch-lr", type=float, default=5e-4)
    p.add_argument("--gate-lr", type=float, default=1e-4)
    p.add_argument("--base-min-lr", type=float, default=5e-6)
    p.add_argument("--branch-min-lr", type=float, default=2e-5)
    p.add_argument("--gate-min-lr", type=float, default=5e-6)
    p.add_argument("--warmup-fraction", type=float, default=0.08)
    p.add_argument("--base-weight-decay", type=float, default=0.02)
    p.add_argument("--branch-weight-decay", type=float, default=0.03)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)

    p.add_argument("--predictive-loss-weight", type=float, default=0.05)
    p.add_argument("--branch-aux-warmup-weight", type=float,
                   default=float(os.environ.get("NESTSAR_AUX_WARM", "0.50")))
    p.add_argument("--branch-aux-final-weight", type=float,
                   default=float(os.environ.get("NESTSAR_AUX_FINAL", "0.20")))
    p.add_argument("--diversity-loss-weight", type=float,
                   default=float(os.environ.get("NESTSAR_DIVERSITY_WEIGHT", "0.05")))
    p.add_argument("--freeze-base-epochs", type=int,
                   default=int(os.environ.get("NESTSAR_FREEZE_BASE", "3")))
    p.add_argument("--base-unfreeze-ramp-epochs", type=int,
                   default=int(os.environ.get("NESTSAR_BASE_RAMP", "3")))
    p.add_argument("--freeze-branch-epochs", type=int, default=2)
    p.add_argument("--branch-ramp-epochs", type=int, default=4)

    # Accepted for CLI compatibility with older launch cells; unused in v3.4.
    p.add_argument("--controller-lr", type=float, default=1e-4)
    p.add_argument("--controller-min-lr", type=float, default=5e-6)
    p.add_argument("--controller-kl-weight", type=float, default=0.0)

    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--outdir", default="/kaggle/working/NestSAR_TokenPreserve_v34_T16")
    p.add_argument("--save-online", action="store_true")
    p.add_argument("--audit-first", action="store_true")
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


def branch_scale_for_epoch(epoch, freeze_epochs, ramp_epochs):
    if epoch <= freeze_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (epoch - freeze_epochs) / ramp_epochs)))


def base_grad_scale_for_epoch(epoch, freeze_epochs, ramp_epochs):
    if epoch <= freeze_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (epoch - freeze_epochs) / ramp_epochs)))


def _write_progress(path: Path | None, payload: dict[str, Any]):
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


def warm_cosine(peak, end, warmup_steps, total_steps):
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak, warmup_steps=warmup_steps,
        decay_steps=total_steps, end_value=end,
    )


def make_param_labels(params):
    flat = traverse_util.flatten_dict(params)
    labels, counts, base_mask = {}, {"base": 0, "branch": 0, "gate": 0}, {}
    for path, value in flat.items():
        name = "/".join(path)
        if "parttrace_residual_gate_logit" in name:
            label = "gate"
        elif "parttrace_branch" in name:
            label = "branch"
        else:
            label = "base"
        labels[path] = label
        base_mask[path] = np.float32(1.0 if label == "base" else 0.0)
        counts[label] += int(np.asarray(value).size)
    return (
        traverse_util.unflatten_dict(labels),
        counts,
        traverse_util.unflatten_dict(base_mask),
    )


def _flat_shape_signature(tree):
    if not isinstance(tree, Mapping):
        return None
    try:
        flat = traverse_util.flatten_dict(dict(tree))
        return {k: tuple(np.asarray(v).shape) for k, v in flat.items()}
    except Exception:
        return None


def _same_param_tree(candidate, template) -> bool:
    a, b = _flat_shape_signature(candidate), _flat_shape_signature(template)
    return a is not None and b is not None and a == b


def _iter_mapping_subtrees(tree, path=()):
    if not isinstance(tree, Mapping):
        return
    yield path, tree
    for k, v in tree.items():
        if isinstance(v, Mapping):
            yield from _iter_mapping_subtrees(v, path + (k,))


def _find_matching_subtree(tree, template):
    for path, subtree in _iter_mapping_subtrees(tree):
        if _same_param_tree(subtree, template):
            return path, subtree
    return None, None


def _replace_path(tree, path, replacement):
    if not path:
        return replacement
    out = dict(tree)
    out[path[0]] = _replace_path(out[path[0]], path[1:], replacement)
    return out


def _rank_checkpoint(path: Path, protocol: str):
    s = str(path).lower()
    score = 0
    if "attention_lite" in s or "attention-lite" in s:
        score += 10
    if protocol in s:
        score += 6
    if "best_ema" in s:
        score += 2
    if "tokenpreserve" in s or "parttrace" in s:
        score -= 8
    if "t64" in s:
        score -= 8
    return score


def _checkpoint_candidates(spec: str, protocol: str):
    if spec.lower() not in ("auto", "none", "scratch"):
        p = Path(spec.format(protocol=protocol)).expanduser()
        if not p.is_file():
            raise FileNotFoundError(p)
        return [p]
    if spec.lower() in ("none", "scratch"):
        return []
    hits = []
    for root in (Path("/kaggle/working"), Path("/kaggle/input")):
        if root.exists():
            try:
                hits.extend(root.rglob("best_ema.msgpack"))
            except OSError:
                pass
    return sorted(set(hits), key=lambda p: (_rank_checkpoint(p, protocol), str(p)), reverse=True)


def load_compatible_base(spec: str, protocol: str, template):
    errors = []
    for path in _checkpoint_candidates(spec, protocol):
        try:
            payload = serialization.msgpack_restore(path.read_bytes())
            roots = []
            if isinstance(payload, Mapping):
                for key in ("ema_params", "params"):
                    if key in payload:
                        roots.append(payload[key])
            roots.append(payload)
            for root in roots:
                match_path, subtree = _find_matching_subtree(root, template)
                if subtree is not None:
                    loaded = jax.tree_util.tree_map(lambda x: jnp.asarray(x), subtree)
                    return loaded, path, match_path
            errors.append(f"{path}: no canonical-shaped subtree")
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return None, None, errors


def main() -> int:
    a = parse_args()
    if a.frames != 16:
        raise ValueError(f"v3.4 is T16 only; got --frames={a.frames}")
    if a.part_dim <= 0 or a.part_heads <= 0 or a.part_dim % a.part_heads:
        raise ValueError("part_dim must be positive and divisible by part_heads")
    if not 1 <= a.readout_tokens <= 32:
        raise ValueError("readout_tokens must be in [1,32]")
    for name in ("branch_dropout", "frame_mask_rate", "joint_mask_rate", "part_mask_rate"):
        v = float(getattr(a, name))
        if not 0.0 <= v < 1.0:
            raise ValueError(f"--{name.replace('_','-')} must be in [0,1)")

    progress_path = Path(a.progress_json) if a.progress_json else None
    _write_progress(progress_path, {
        "protocol": a.protocol, "phase": "initializing", "epoch": 0,
        "n": 0, "total": 1, "message": "loading pretrained canonical Attention-Lite",
    })

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required; got {jax.default_backend()}")
    devices = list(jax.local_devices())
    if len(devices) != 1:
        raise RuntimeError(f"Expected exactly one process-visible GPU; got {devices}")

    mod, source = load_canonical_prefix(a.protocol)
    ns = mod.ns
    base = mod.build_model()
    model = make_wrapper_v34(
        base,
        part_dim=a.part_dim,
        part_heads=a.part_heads,
        global_dim=a.global_dim,
        dense_dim=a.dense_dim,
        branch_dropout=a.branch_dropout,
        readout_tokens=a.readout_tokens,
        frame_mask_rate=a.frame_mask_rate,
        joint_mask_rate=a.joint_mask_rate,
        part_mask_rate=a.part_mask_rate,
    )

    rng = jax.random.PRNGKey(a.seed)
    rng, brng, bdrop = jax.random.split(rng, 3)
    dummy = jnp.zeros((1, FRAMES, 150), jnp.float32)
    base_vars = base.init({"params": brng, "dropout": bdrop}, dummy, training=True)
    base_template = base_vars["params"]
    bp, bl = tree_numel(base_template), tree_leaves(base_template)
    if (bp, bl) != (EXPECTED_BASE_PARAMS, EXPECTED_BASE_LEAVES):
        raise RuntimeError(
            f"CANONICAL BASE GUARD FAILED params={bp}/{EXPECTED_BASE_PARAMS} leaves={bl}/{EXPECTED_BASE_LEAVES}"
        )

    rng, irng, drng = jax.random.split(rng, 3)
    params = model.init(
        {"params": irng, "dropout": drng}, dummy, training=True, branch_scale=0.0
    )["params"]
    tp, tl = tree_numel(params), tree_leaves(params)
    added = tp - bp
    if not (50_000 <= added <= 500_000):
        raise RuntimeError(f"V3.4 SIZE GUARD FAILED: added params={added:,}")

    loaded_base, loaded_path, load_meta = load_compatible_base(a.base_checkpoint, a.protocol, base_template)
    if loaded_base is None:
        if not a.allow_scratch:
            preview = "\n".join(str(x) for x in load_meta[:12]) if load_meta else "no best_ema.msgpack candidates found"
            raise RuntimeError(
                "No compatible pretrained Attention-Lite checkpoint found. "
                "Pass --base-checkpoint /path/to/best_ema.msgpack or --allow-scratch.\n" + preview
            )
        print("WARNING: no compatible pretrained base found; using scratch initialization", flush=True)
        loaded_path_text = "SCRATCH"
    else:
        full_base_path, _ = _find_matching_subtree(params, base_template)
        if full_base_path is None:
            raise RuntimeError("Could not locate canonical base subtree inside v3.4 params")
        params = _replace_path(params, full_base_path, loaded_base)
        if not _same_param_tree(_find_matching_subtree(params, base_template)[1], base_template):
            raise RuntimeError("Pretrained-base transplant verification failed")
        loaded_path_text = str(loaded_path)
        print(f"PRETRAINED BASE LOADED: {loaded_path}", flush=True)
        print(f"Checkpoint subtree: {load_meta if isinstance(load_meta, tuple) else load_meta}", flush=True)

    labels, group_counts, base_mask = make_param_labels(params)
    if min(group_counts.values()) <= 0:
        raise RuntimeError(f"Optimizer group guard failed: {group_counts}")

    dataset = find_dataset(a.dataset)
    raw = ns.load_pickle(dataset)
    train_samples, val_samples = ns.build_samples(
        raw, protocol=a.protocol, max_train=a.max_train_samples,
        max_val=a.max_val_samples, seed=a.seed,
    )
    if a.max_train_samples == 0 and a.max_val_samples == 0:
        if (len(train_samples), len(val_samples)) != EXPECTED_COUNTS[a.protocol]:
            raise RuntimeError(
                f"Split mismatch got={(len(train_samples),len(val_samples))}, expected={EXPECTED_COUNTS[a.protocol]}"
            )
    train_ds, val_ds = ns.SkeletonDataset(train_samples), ns.SkeletonDataset(val_samples)
    train_steps = math.ceil(len(train_ds) / a.batch_size)
    val_steps = math.ceil(len(val_ds) / a.eval_batch_size)
    total_steps = max(1, train_steps * a.epochs)
    warmup_steps = max(1, int(total_steps * a.warmup_fraction))

    base_sched = warm_cosine(a.base_lr, a.base_min_lr, warmup_steps, total_steps)
    branch_sched = warm_cosine(a.branch_lr, a.branch_min_lr, warmup_steps, total_steps)
    gate_sched = warm_cosine(a.gate_lr, a.gate_min_lr, warmup_steps, total_steps)
    multi = optax.multi_transform({
        "base": optax.adamw(base_sched, weight_decay=a.base_weight_decay),
        "branch": optax.adamw(branch_sched, weight_decay=a.branch_weight_decay),
        "gate": optax.adamw(gate_sched, weight_decay=0.0),
    }, labels)
    tx = optax.chain(optax.clip_by_global_norm(a.grad_clip), multi)
    opt_state = tx.init(params)
    ema_params = params

    print("=" * 118, flush=True)
    print("NESTSAR ATTENTION-LITE + TOKENPRESERVE V3.4 — PRETRAINED DIVERSITY TRAINER", flush=True)
    print("=" * 118, flush=True)
    print(f"Protocol: {a.protocol.upper()} | GPU: {devices[0]} | T=16", flush=True)
    print(f"Canonical source: {source}", flush=True)
    print(f"Pretrained base: {loaded_path_text}", flush=True)
    print(f"Base: {bp:,}/{bl} | Added: {added:,} | Total: {tp:,}/{tl}", flush=True)
    print(
        f"K={a.readout_tokens} Dtoken={a.part_dim} H={a.part_heads} Dmixer={a.global_dim} Ddense={a.dense_dim}",
        flush=True,
    )
    print(
        f"Mask frame/joint/part={a.frame_mask_rate:.3f}/{a.joint_mask_rate:.3f}/{a.part_mask_rate:.3f}",
        flush=True,
    )
    print(f"Optimizer groups: {group_counts}", flush=True)
    print(
        f"LR base={a.base_lr:g} branch={a.branch_lr:g} gate={a.gate_lr:g} | "
        f"freeze base={a.freeze_base_epochs} ramp={a.base_unfreeze_ramp_epochs}",
        flush=True,
    )
    print(
        f"Aux warm/final={a.branch_aux_warmup_weight:g}/{a.branch_aux_final_weight:g} | "
        f"diversity={a.diversity_loss_weight:g}", flush=True,
    )
    print("=" * 118, flush=True)

    audit = {}
    if a.audit_first:
        base_path, base_for_audit = _find_matching_subtree(params, base_template)
        base_comp = jax.jit(
            lambda p, xx: base.apply({"params": p}, xx, training=False)["logits"]
        ).lower(base_for_audit, dummy).compile()
        full_comp = jax.jit(
            lambda p, xx: model.apply({"params": p}, xx, training=False, branch_scale=1.0)["logits"]
        ).lower(params, dummy).compile()
        bf, ff = _compiled_flops(base_comp), _compiled_flops(full_comp)
        audit = {"base_gflops": bf / 1e9, "v34_gflops": ff / 1e9,
                 "added_gflops": (ff - bf) / 1e9}
        print(f"XLA base GFLOPs:     {audit['base_gflops']:.9f}", flush=True)
        print(f"XLA v3.4 GFLOPs:     {audit['v34_gflops']:.9f}", flush=True)
        print(f"XLA added GFLOPs:    {audit['added_gflops']:.9f}", flush=True)

    # Validate loaded base before training. This is the checkpoint sanity check.
    base_path, current_base_params = _find_matching_subtree(params, base_template)
    @jax.jit
    def base_eval_step(base_params, x, y):
        out = base.apply({"params": base_params}, x, training=False)
        return jnp.sum(jnp.argmax(out["logits"], axis=-1) == y)

    baseline_correct = baseline_count = 0
    _write_progress(progress_path, {
        "protocol": a.protocol, "phase": "baseline", "epoch": 0, "epochs": a.epochs,
        "n": 0, "total": val_steps, "accuracy": 0.0, "best": -1.0, "best_epoch": 0,
    })
    for vi, (x_np, y_np) in enumerate(ns.batch_iterator(
        val_ds, batch_size=a.eval_batch_size, shuffle=False, seed=0, drop_last=False
    )):
        c = base_eval_step(current_base_params, jnp.asarray(x_np, jnp.float32), jnp.asarray(y_np, jnp.int32))
        baseline_correct += int(c)
        baseline_count += len(y_np)
        if (vi + 1) % max(1, a.progress_every) == 0 or vi + 1 == val_steps:
            _write_progress(progress_path, {
                "protocol": a.protocol, "phase": "baseline", "epoch": 0, "epochs": a.epochs,
                "n": vi + 1, "total": val_steps,
                "accuracy": baseline_correct / max(1, baseline_count), "best": -1.0, "best_epoch": 0,
            })
    baseline_acc = baseline_correct / max(1, baseline_count)
    print(f"PRETRAINED BASELINE VAL: {100*baseline_acc:.5f}%", flush=True)

    @jax.jit
    def train_step(params, ema_params, opt_state, x, y, step_rng,
                   branch_scale, base_grad_scale, aux_weight):
        def loss_fn(p):
            out = model.apply(
                {"params": p}, x, training=True, branch_scale=branch_scale,
                rngs={"dropout": step_rng},
            )
            main_ce = smoothed_ce(out["logits"], y, a.label_smoothing)
            branch_ce = smoothed_ce(out["parttrace_logits"], y, a.label_smoothing)
            predictive = jnp.array(0.0, jnp.float32)
            if "prediction" in out and "motion_target" in out:
                predictive = jnp.mean(jnp.square(
                    out["prediction"] - jax.lax.stop_gradient(out["motion_target"])))
            div = out["query_diversity_loss"]
            loss = (main_ce + aux_weight * branch_ce
                    + a.predictive_loss_weight * predictive
                    + a.diversity_loss_weight * div)
            acc = jnp.mean(jnp.argmax(out["logits"], axis=-1) == y)
            return loss, (
                main_ce, branch_ce, predictive, div, acc,
                out["parttrace_gate"], out["effective_parttrace_gate"],
                out["joint_to_part_entropy"], out["temporal_entropy_global"],
                out["temporal_entropy_left"], out["temporal_entropy_right"],
                out["query_overlap_mean"], out["query_overlap_max"],
            )

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        # Freeze/ramp only canonical-base gradients. Branch and gate still learn.
        grads = jax.tree_util.tree_map(
            lambda g, m: g * (1.0 - m + m * base_grad_scale), grads, base_mask
        )
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
        return (
            jnp.sum(jnp.argmax(out["logits"], axis=-1) == y),
            jnp.sum(jnp.argmax(out["parttrace_logits"], axis=-1) == y),
            out["parttrace_gate"], out["effective_parttrace_gate"],
            out["query_diversity_loss"], out["query_overlap_mean"], out["query_overlap_max"],
        )

    outdir = Path(a.outdir) / a.protocol
    outdir.mkdir(parents=True, exist_ok=True)
    best, best_epoch, stale = baseline_acc, 0, 0
    history: list[dict[str, Any]] = []
    start = time.time()
    every = max(1, a.progress_every)

    for epoch in range(1, a.epochs + 1):
        branch_scale = branch_scale_for_epoch(epoch, a.freeze_branch_epochs, a.branch_ramp_epochs)
        base_scale = base_grad_scale_for_epoch(epoch, a.freeze_base_epochs, a.base_unfreeze_ramp_epochs)
        aux_weight = (
            a.branch_aux_warmup_weight * (1.0 - branch_scale)
            + a.branch_aux_final_weight * branch_scale
        )
        bs_j = jnp.asarray(branch_scale, jnp.float32)
        bg_j = jnp.asarray(base_scale, jnp.float32)
        aux_j = jnp.asarray(aux_weight, jnp.float32)
        tr_correct = tr_count = 0
        sums = np.zeros(5, np.float64)
        last_diag = np.zeros(9, np.float64)
        epoch_rng = jax.random.fold_in(rng, epoch)

        _write_progress(progress_path, {
            "protocol": a.protocol, "phase": "train", "epoch": epoch, "epochs": a.epochs,
            "n": 0, "total": train_steps, "accuracy": 0.0,
            "best": best, "best_epoch": best_epoch,
        })
        for bi, (x_np, y_np) in enumerate(ns.batch_iterator(
            train_ds, batch_size=a.batch_size, shuffle=True,
            seed=a.seed + epoch, drop_last=False
        )):
            x, y = jnp.asarray(x_np, jnp.float32), jnp.asarray(y_np, jnp.int32)
            step_rng = jax.random.fold_in(epoch_rng, bi)
            params, ema_params, opt_state, loss, aux, gn = train_step(
                params, ema_params, opt_state, x, y, step_rng, bs_j, bg_j, aux_j
            )
            (main_ce, branch_ce, predictive, div, acc, gate, eff,
             ent_part, ent_g, ent_l, ent_r, overlap_mean, overlap_max) = aux
            n = len(y_np)
            tr_count += n
            tr_correct += int(round(float(acc) * n))
            sums += np.asarray([float(loss), float(main_ce), float(branch_ce),
                                float(predictive), float(div)]) * n
            last_diag = np.asarray([
                float(gate), float(eff), float(gn), float(ent_part), float(ent_g),
                float(ent_l), float(ent_r), float(overlap_mean), float(overlap_max)
            ])
            if (bi + 1) % every == 0 or bi + 1 == train_steps:
                _write_progress(progress_path, {
                    "protocol": a.protocol, "phase": "train", "epoch": epoch, "epochs": a.epochs,
                    "n": bi + 1, "total": train_steps,
                    "accuracy": tr_correct / max(1, tr_count),
                    "loss": sums[0] / max(1, tr_count),
                    "best": best, "best_epoch": best_epoch,
                })

        val_correct = val_branch_correct = val_count = 0
        val_div = val_overlap = val_overlap_max = 0.0
        val_gate = val_eff = 0.0
        val_batches = 0
        _write_progress(progress_path, {
            "protocol": a.protocol, "phase": "val", "epoch": epoch, "epochs": a.epochs,
            "n": 0, "total": val_steps, "accuracy": 0.0,
            "best": best, "best_epoch": best_epoch,
        })
        for vi, (x_np, y_np) in enumerate(ns.batch_iterator(
            val_ds, batch_size=a.eval_batch_size, shuffle=False, seed=0, drop_last=False
        )):
            x, y = jnp.asarray(x_np, jnp.float32), jnp.asarray(y_np, jnp.int32)
            c, bc, gate, eff, div, overlap, overlap_max = eval_step(ema_params, x, y, bs_j)
            val_correct += int(c)
            val_branch_correct += int(bc)
            val_count += len(y_np)
            val_gate, val_eff = float(gate), float(eff)
            val_div += float(div); val_overlap += float(overlap); val_overlap_max += float(overlap_max)
            val_batches += 1
            if (vi + 1) % every == 0 or vi + 1 == val_steps:
                _write_progress(progress_path, {
                    "protocol": a.protocol, "phase": "val", "epoch": epoch, "epochs": a.epochs,
                    "n": vi + 1, "total": val_steps,
                    "accuracy": val_correct / max(1, val_count),
                    "best": best, "best_epoch": best_epoch,
                })

        train_acc = tr_correct / max(1, tr_count)
        val_acc = val_correct / max(1, val_count)
        branch_val_acc = val_branch_correct / max(1, val_count)
        improved = val_acc > best
        if improved:
            best, best_epoch, stale = val_acc, epoch, 0
            payload = {
                "model": MODEL_NAME, "protocol": a.protocol, "epoch": epoch,
                "seed": a.seed, "val_accuracy": val_acc,
                "baseline_accuracy": baseline_acc,
                "base_checkpoint": loaded_path_text,
                "ema_params": ema_params, "args": vars(a),
            }
            if a.save_online:
                payload["params"] = params
            (outdir / "best_ema.msgpack").write_bytes(serialization.to_bytes(payload))
        else:
            stale += 1

        rec = {
            "epoch": epoch,
            "branch_scale": branch_scale,
            "base_grad_scale": base_scale,
            "branch_aux_weight": aux_weight,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "tokenpreserve_val_accuracy": branch_val_acc,
            "best_val_accuracy": best,
            "best_epoch": best_epoch,
            "loss": sums[0] / max(1, tr_count),
            "main_ce": sums[1] / max(1, tr_count),
            "branch_ce": sums[2] / max(1, tr_count),
            "predictive_loss": sums[3] / max(1, tr_count),
            "diversity_loss": sums[4] / max(1, tr_count),
            "parttrace_gate": val_gate,
            "effective_parttrace_gate": val_eff,
            "joint_to_part_entropy": float(last_diag[3]),
            "readout_entropy": float(last_diag[4]),
            "left_entropy": float(last_diag[5]),
            "right_entropy": float(last_diag[6]),
            "query_overlap_mean": val_overlap / max(1, val_batches),
            "query_overlap_max": val_overlap_max / max(1, val_batches),
            "grad_norm_last": float(last_diag[2]),
            "stale_epochs": stale,
        }
        history.append(rec)
        (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        _write_progress(progress_path, {
            "protocol": a.protocol, "phase": "epoch_done", "epoch": epoch, "epochs": a.epochs,
            "n": 1, "total": 1, "accuracy": val_acc,
            "train_accuracy": train_acc, "parttrace_accuracy": branch_val_acc,
            "best": best, "best_epoch": best_epoch,
            "stale": stale, "patience": a.patience,
        })
        print(
            f"{a.protocol.upper()} E{epoch:03}/{a.epochs} | train={100*train_acc:.2f}% | "
            f"val={100*val_acc:.2f}% | TP={100*branch_val_acc:.2f}% | "
            f"BEST={100*best:.2f}%@E{best_epoch:03} | bscale={branch_scale:.2f} "
            f"basegrad={base_scale:.2f} aux={aux_weight:.2f} gate={val_eff:.4f} | "
            f"Qoverlap={rec['query_overlap_mean']:.3f} | stale={stale}/{a.patience}",
            flush=True,
        )
        if a.patience > 0 and stale >= a.patience:
            break

    result = {
        "model": MODEL_NAME,
        "protocol": a.protocol,
        "seed": a.seed,
        "frames": 16,
        "fine_tokens": 320,
        "readout_tokens": a.readout_tokens,
        "base_checkpoint": loaded_path_text,
        "baseline_accuracy": baseline_acc,
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
    (outdir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_progress(progress_path, {
        "protocol": a.protocol, "phase": "done", "epoch": len(history), "epochs": a.epochs,
        "n": 1, "total": 1, "accuracy": best, "best": best, "best_epoch": best_epoch,
    })
    print(f"{a.protocol.upper()} COMPLETE | best={100*best:.5f}% @ epoch {best_epoch}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

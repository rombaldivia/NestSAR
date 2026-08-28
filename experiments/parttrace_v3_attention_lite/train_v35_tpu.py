#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TPU trainer for NestSAR v3.5 Cross-Stream Multi-Resolution Memory.

Designed for Kaggle TPU v5e-8:
- one protocol at a time, all process-visible TPU cores via jax.pmap;
- exact canonical Attention-Lite guards (2,381,028 params / 705 leaves);
- pretrained Attention-Lite EMA transplant with epoch-0 validation;
- branch warm-up, protected base unfreeze, differential LR, EMA;
- persistent progress JSON for the notebook tqdm launcher;
- per-class accuracy and confusion matrix whenever validation improves.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, traverse_util

from experiments.parttrace_v3_attention_lite.audit_model import (
    FRAMES, NUM_CLASSES, EXPECTED_BASE_PARAMS, EXPECTED_BASE_LEAVES,
    load_canonical_prefix, tree_numel, tree_leaves,
)
from experiments.parttrace_v3_attention_lite.model_v35_crossstream_memory import make_wrapper_v35

EXPECTED_COUNTS = {"xsub": (63_026, 50_919), "xset": (54_468, 59_477)}
MODEL_NAME = "AttentionLiteCrossStreamMultiResolutionMemoryV35"


def parse_args():
    p = argparse.ArgumentParser(description="NestSAR v3.5 TPU trainer")
    p.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    p.add_argument("--dataset", default="auto")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--base-checkpoint", default="auto")
    p.add_argument("--allow-scratch", action="store_true")
    p.add_argument("--expected-devices", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Global batch across all TPU cores")
    p.add_argument("--eval-batch-size", type=int, default=512,
                   help="Global validation batch across all TPU cores")

    p.add_argument("--bridge-dim", type=int, default=32)
    p.add_argument("--local-stream-dim", type=int, default=16)
    p.add_argument("--memory-dim", type=int, default=32)
    p.add_argument("--fine-dim", type=int, default=24)
    p.add_argument("--readout-tokens", type=int, default=8)
    p.add_argument("--readout-heads", type=int, default=4)
    p.add_argument("--dense-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.08)
    p.add_argument("--stream-reweight-strength", type=float, default=0.08)

    # Pretrained-base schedule: new modules learn quickly, canonical base moves gently.
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
    p.add_argument("--outdir", default="/kaggle/working/NestSAR_CrossStreamMemory_v35_TPU")
    p.add_argument("--save-online", action="store_true")
    p.add_argument("--audit-first", action="store_true")
    p.add_argument("--progress-json", default=None)
    p.add_argument("--progress-every", type=int, default=2)
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
    try:
        c = compiled.cost_analysis()
        if isinstance(c, list) and c:
            c = c[0]
        return float(c.get("flops", 0.0)) if isinstance(c, dict) else 0.0
    except Exception:
        return 0.0


def warm_cosine(peak, end, warmup_steps, total_steps):
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak, warmup_steps=warmup_steps,
        decay_steps=total_steps, end_value=end,
    )


def make_param_labels(params):
    flat = traverse_util.flatten_dict(params)
    labels, counts, base_mask = {}, {"base": 0, "new": 0}, {}
    for path, value in flat.items():
        name = "/".join(path)
        is_new = ("cross_stream_bridge" in name or "cross_stream_memory" in name)
        label = "new" if is_new else "base"
        labels[path] = label
        base_mask[path] = np.float32(0.0 if is_new else 1.0)
        counts[label] += int(np.asarray(value).size)
    return traverse_util.unflatten_dict(labels), counts, traverse_util.unflatten_dict(base_mask)


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
        score += 15
    if protocol in s:
        score += 8
    if "best_ema" in s:
        score += 3
    if any(x in s for x in ("tokenpreserve", "parttrace", "crossstream", "v35")):
        score -= 12
    if "t64" in s:
        score -= 12
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
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
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
                    loaded = jax.tree_util.tree_map(lambda z: jnp.asarray(z), subtree)
                    return loaded, path, match_path
            errors.append(f"{path}: no canonical-shaped subtree")
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return None, None, errors


def _shard_train(x_np, y_np, ndev):
    x = np.asarray(x_np, np.float32)
    y = np.asarray(y_np, np.int32)
    if len(y) % ndev:
        raise ValueError(f"Training batch {len(y)} not divisible by {ndev}")
    local = len(y) // ndev
    return x.reshape(ndev, local, *x.shape[1:]), y.reshape(ndev, local)


def _shard_eval(x_np, y_np, ndev):
    x = np.asarray(x_np, np.float32)
    y = np.asarray(y_np, np.int32)
    n = len(y)
    pad = (-n) % ndev
    valid = np.ones(n + pad, np.float32)
    if pad:
        x = np.concatenate([x, np.zeros((pad, *x.shape[1:]), np.float32)], axis=0)
        y = np.concatenate([y, np.zeros((pad,), np.int32)], axis=0)
        valid[-pad:] = 0.0
    local = len(y) // ndev
    return (x.reshape(ndev, local, *x.shape[1:]), y.reshape(ndev, local),
            valid.reshape(ndev, local))


def _rep_scalar(value, ndev):
    return np.full((ndev,), value, np.float32)


def _unrep(tree):
    return jax.tree_util.tree_map(lambda z: np.asarray(z[0]), tree)


def _confusion_and_per_class(labels, preds):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    np.add.at(cm, (labels.astype(np.int64), preds.astype(np.int64)), 1)
    denom = cm.sum(axis=1)
    acc = np.divide(np.diag(cm), np.maximum(denom, 1), dtype=np.float64)
    return cm, acc


def _save_class_metrics(outdir: Path, labels, preds, prefix: str):
    cm, per = _confusion_and_per_class(labels, preds)
    np.save(outdir / f"{prefix}_confusion.npy", cm)
    payload = {
        "per_class_accuracy": per.tolist(),
        "class_support": cm.sum(axis=1).astype(int).tolist(),
        "macro_accuracy": float(per.mean()),
    }
    (outdir / f"{prefix}_per_class.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return float(per.mean())


def main() -> int:
    a = parse_args()
    if a.frames != 16:
        raise ValueError("v3.5 is intentionally T16")
    if jax.default_backend() != "tpu":
        raise RuntimeError(f"Kaggle TPU required; backend={jax.default_backend()}")
    devices = list(jax.local_devices())
    ndev = len(devices)
    if a.expected_devices > 0 and ndev != a.expected_devices:
        raise RuntimeError(f"Expected {a.expected_devices} local TPU devices; found {ndev}: {devices}")
    if a.batch_size % ndev or a.eval_batch_size % ndev:
        raise ValueError("Global train/eval batch sizes must be divisible by TPU device count")
    if a.memory_dim % a.readout_heads:
        raise ValueError("--memory-dim must be divisible by --readout-heads")
    if a.bridge_dim % 4 or a.local_stream_dim % 4:
        raise ValueError("bridge/local-stream dimensions must be divisible by 4")

    progress_path = Path(a.progress_json) if a.progress_json else None
    _write_progress(progress_path, {
        "protocol": a.protocol, "phase": "initializing", "epoch": 0,
        "epochs": a.epochs, "n": 0, "total": 1,
        "message": f"TPU x{ndev}: building exact Attention-Lite + v3.5 memory",
    })

    mod, source = load_canonical_prefix(a.protocol)
    ns = mod.ns
    base = mod.build_model()
    model = make_wrapper_v35(
        base, bridge_dim=a.bridge_dim, local_stream_dim=a.local_stream_dim,
        memory_dim=a.memory_dim, fine_dim=a.fine_dim,
        readout_tokens=a.readout_tokens, readout_heads=a.readout_heads,
        dense_dim=a.dense_dim, dropout=a.dropout,
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
            f"CANONICAL BASE GUARD FAILED params={bp}/{EXPECTED_BASE_PARAMS} leaves={bl}/{EXPECTED_BASE_LEAVES}")

    rng, irng, drng = jax.random.split(rng, 3)
    params = model.init({"params": irng, "dropout": drng}, dummy,
                        training=True, branch_scale=0.0)["params"]
    tp, tl = tree_numel(params), tree_leaves(params)
    added = tp - bp
    if not (30_000 <= added <= 500_000):
        raise RuntimeError(f"V3.5 SIZE GUARD FAILED: added params={added:,}")

    loaded_base, loaded_path, load_meta = load_compatible_base(
        a.base_checkpoint, a.protocol, base_template)
    if loaded_base is None:
        if not a.allow_scratch:
            preview = "\n".join(load_meta[:12]) if load_meta else "no compatible best_ema.msgpack found"
            raise RuntimeError(
                "No compatible pretrained Attention-Lite checkpoint found. "
                "Attach it as a Kaggle dataset or pass --base-checkpoint PATH.\n" + preview)
        loaded_path_text = "SCRATCH"
        print("WARNING: using scratch canonical base", flush=True)
    else:
        full_base_path, _ = _find_matching_subtree(params, base_template)
        if full_base_path is None:
            raise RuntimeError("Could not locate canonical base subtree in v3.5 params")
        params = _replace_path(params, full_base_path, loaded_base)
        loaded_path_text = str(loaded_path)
        print(f"PRETRAINED BASE LOADED: {loaded_path}", flush=True)
        print(f"Checkpoint subtree: {load_meta}", flush=True)

    labels, group_counts, base_mask = make_param_labels(params)
    if min(group_counts.values()) <= 0:
        raise RuntimeError(f"Optimizer group guard failed: {group_counts}")

    dataset = find_dataset(a.dataset)
    raw = ns.load_pickle(dataset)
    train_samples, val_samples = ns.build_samples(
        raw, protocol=a.protocol, max_train=a.max_train_samples,
        max_val=a.max_val_samples, seed=a.seed)
    if a.max_train_samples == 0 and a.max_val_samples == 0:
        if (len(train_samples), len(val_samples)) != EXPECTED_COUNTS[a.protocol]:
            raise RuntimeError(
                f"Split mismatch got={(len(train_samples),len(val_samples))}, expected={EXPECTED_COUNTS[a.protocol]}")
    train_ds = ns.SkeletonDataset(train_samples)
    val_ds = ns.SkeletonDataset(val_samples)
    train_steps = len(train_ds) // a.batch_size
    val_steps = math.ceil(len(val_ds) / a.eval_batch_size)
    total_steps = max(1, train_steps * a.epochs)
    warmup_steps = max(1, int(total_steps * a.warmup_fraction))

    base_sched = warm_cosine(a.base_lr, a.base_min_lr, warmup_steps, total_steps)
    new_sched = warm_cosine(a.new_lr, a.new_min_lr, warmup_steps, total_steps)
    multi = optax.multi_transform({
        "base": optax.adamw(base_sched, b1=0.9, b2=0.999, eps=1e-8,
                             weight_decay=a.base_weight_decay),
        "new": optax.adamw(new_sched, b1=0.9, b2=0.999, eps=1e-8,
                            weight_decay=a.new_weight_decay),
    }, labels)
    tx = optax.chain(optax.clip_by_global_norm(a.grad_clip), multi)
    opt_state = tx.init(params)

    print("=" * 120, flush=True)
    print("NESTSAR v3.5 — CROSS-STREAM MULTI-RESOLUTION MEMORY — TPU", flush=True)
    print("=" * 120, flush=True)
    print(f"Protocol: {a.protocol.upper()} | backend={jax.default_backend()} | devices={ndev}", flush=True)
    print(f"Device ids: {[int(d.id) for d in devices]}", flush=True)
    print(f"Canonical source: {source}", flush=True)
    print(f"Pretrained base: {loaded_path_text}", flush=True)
    print(f"Base={bp:,}/{bl} | Added={added:,} | Total={tp:,}/{tl}", flush=True)
    print(f"Memory: coarse=320xD{a.memory_dim} + fine=320xD{a.fine_dim} | K={a.readout_tokens}", flush=True)
    print(f"Bridge D{a.bridge_dim} | local 4-stream D{a.local_stream_dim} | dense D{a.dense_dim}", flush=True)
    print(f"Global batch={a.batch_size} ({a.batch_size//ndev}/TPU) | eval={a.eval_batch_size}", flush=True)
    print(f"Optimizer groups: {group_counts}", flush=True)
    print("=" * 120, flush=True)

    audit = {}
    if a.audit_first:
        _, base_for_audit = _find_matching_subtree(params, base_template)
        base_comp = jax.jit(
            lambda p, xx: base.apply({"params": p}, xx, training=False)["logits"]
        ).lower(base_for_audit, dummy).compile()
        full_comp = jax.jit(
            lambda p, xx: model.apply({"params": p}, xx, training=False,
                                      branch_scale=1.0)["logits"]
        ).lower(params, dummy).compile()
        bf, ff = _compiled_flops(base_comp), _compiled_flops(full_comp)
        audit = {"base_gflops": bf / 1e9, "v35_gflops": ff / 1e9,
                 "added_gflops": (ff - bf) / 1e9 if bf and ff else 0.0}
        print(f"XLA base GFLOPs:     {audit['base_gflops']:.9f}", flush=True)
        print(f"XLA v3.5 GFLOPs:     {audit['v35_gflops']:.9f}", flush=True)
        print(f"XLA added GFLOPs:    {audit['added_gflops']:.9f}", flush=True)

    outdir = Path(a.outdir) / a.protocol
    outdir.mkdir(parents=True, exist_ok=True)

    # Epoch-0 canonical baseline validation before any optimization.
    _, current_base = _find_matching_subtree(params, base_template)
    base_repl = jax.device_put_replicated(current_base, devices)

    @partial(jax.pmap, axis_name="data")
    def base_eval_step(bp_, x, y, valid):
        out = base.apply({"params": bp_}, x, training=False)
        pred = jnp.argmax(out["logits"], axis=-1)
        correct = jnp.sum((pred == y).astype(jnp.float32) * valid)
        count = jnp.sum(valid)
        return correct, count, pred

    baseline_correct = baseline_count = 0.0
    baseline_labels, baseline_preds = [], []
    _write_progress(progress_path, {
        "protocol": a.protocol, "phase": "baseline", "epoch": 0,
        "epochs": a.epochs, "n": 0, "total": val_steps,
        "accuracy": 0.0, "best": -1.0, "best_epoch": 0,
    })
    for vi, (x_np, y_np) in enumerate(ns.batch_iterator(
        val_ds, batch_size=a.eval_batch_size, shuffle=False, seed=0, drop_last=False)):
        xs, ys, vs = _shard_eval(x_np, y_np, ndev)
        c, n, pr = base_eval_step(base_repl, xs, ys, vs)
        c, n = float(np.asarray(c).sum()), float(np.asarray(n).sum())
        baseline_correct += c; baseline_count += n
        pr = np.asarray(pr).reshape(-1)
        yy = ys.reshape(-1); vv = vs.reshape(-1) > 0.5
        baseline_preds.append(pr[vv]); baseline_labels.append(yy[vv])
        if (vi + 1) % max(1, a.progress_every) == 0 or vi + 1 == val_steps:
            _write_progress(progress_path, {
                "protocol": a.protocol, "phase": "baseline", "epoch": 0,
                "epochs": a.epochs, "n": vi + 1, "total": val_steps,
                "accuracy": baseline_correct / max(1.0, baseline_count),
                "best": -1.0, "best_epoch": 0,
            })
    baseline_acc = baseline_correct / max(1.0, baseline_count)
    baseline_labels = np.concatenate(baseline_labels)
    baseline_preds = np.concatenate(baseline_preds)
    baseline_macro = _save_class_metrics(outdir, baseline_labels, baseline_preds, "baseline")
    print(f"PRETRAINED BASELINE VAL: {100*baseline_acc:.5f}% | macro={100*baseline_macro:.3f}%", flush=True)

    params_repl = jax.device_put_replicated(params, devices)
    ema_repl = jax.device_put_replicated(params, devices)
    opt_repl = jax.device_put_replicated(opt_state, devices)

    @partial(jax.pmap, axis_name="data")
    def train_step(params_, ema_, opt_, x, y, step_rng,
                   branch_scale, base_grad_scale, aux_weight):
        def loss_fn(p):
            out = model.apply({"params": p}, x, training=True,
                              branch_scale=branch_scale,
                              rngs={"dropout": step_rng})
            main_ce = smoothed_ce(out["logits"], y, a.label_smoothing)
            memory_ce = smoothed_ce(out["memory_logits"], y, a.label_smoothing)
            predictive = jnp.array(0.0, jnp.float32)
            if "prediction" in out and "motion_target" in out:
                predictive = jnp.mean(jnp.square(
                    out["prediction"] - jax.lax.stop_gradient(out["motion_target"])))
            dyn = jnp.clip(out["dynamic_fusion_weights"], 1e-8, 1.0)
            bw = jnp.asarray(out["fusion_weights"])
            if bw.ndim == 1:
                bw = jnp.broadcast_to(bw[None], dyn.shape)
            bw = jnp.clip(bw, 1e-8, 1.0)
            stream_kl = jnp.mean(jnp.sum(dyn * (jnp.log(dyn) - jnp.log(bw)), axis=-1))
            div = out["query_diversity_loss"]
            loss = (main_ce + aux_weight * memory_ce
                    + a.predictive_loss_weight * predictive
                    + a.diversity_loss_weight * div
                    + a.stream_kl_weight * stream_kl)
            acc = jnp.mean(jnp.argmax(out["logits"], axis=-1) == y)
            mem_acc = jnp.mean(jnp.argmax(out["memory_logits"], axis=-1) == y)
            return loss, jnp.asarray([
                main_ce, memory_ce, predictive, div, stream_kl, acc, mem_acc,
                out["class_gate_mean"], out["class_gate_max"],
                out["query_overlap_mean"], out["query_overlap_max"],
            ], jnp.float32)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_)
        grads = jax.tree_util.tree_map(
            lambda g, m: g * (1.0 - m + m * base_grad_scale), grads, base_mask)
        grads = jax.lax.pmean(grads, "data")
        loss = jax.lax.pmean(loss, "data")
        metrics = jax.lax.pmean(metrics, "data")
        grad_norm = optax.global_norm(grads)
        updates, opt_ = tx.update(grads, opt_, params_)
        params_ = optax.apply_updates(params_, updates)
        ema_ = jax.tree_util.tree_map(
            lambda e, p: a.ema_decay * e + (1.0 - a.ema_decay) * p,
            ema_, params_)
        return params_, ema_, opt_, loss, metrics, grad_norm

    @partial(jax.pmap, axis_name="data")
    def eval_step(params_, x, y, valid, branch_scale):
        out = model.apply({"params": params_}, x, training=False,
                          branch_scale=branch_scale)
        pred = jnp.argmax(out["logits"], axis=-1)
        mem_pred = jnp.argmax(out["memory_logits"], axis=-1)
        correct = jnp.sum((pred == y).astype(jnp.float32) * valid)
        mem_correct = jnp.sum((mem_pred == y).astype(jnp.float32) * valid)
        count = jnp.sum(valid)
        weight_sum = jnp.sum(out["dynamic_fusion_weights"] * valid[:, None], axis=0)
        diagnostics = jnp.asarray([
            out["class_gate_mean"], out["class_gate_max"], out["class_gate_min"],
            out["query_overlap_mean"], out["query_overlap_max"],
        ], jnp.float32)
        return correct, mem_correct, count, weight_sum, diagnostics, pred

    best, best_epoch, stale = baseline_acc, 0, 0
    history: list[dict[str, Any]] = []
    initial_payload = {
        "model": MODEL_NAME, "protocol": a.protocol, "epoch": 0,
        "seed": a.seed, "val_accuracy": baseline_acc,
        "baseline_accuracy": baseline_acc, "base_checkpoint": loaded_path_text,
        "ema_params": params, "args": vars(a),
    }
    (outdir / "best_ema.msgpack").write_bytes(serialization.to_bytes(initial_payload))
    start = time.time(); global_step = 0

    for epoch in range(1, a.epochs + 1):
        epoch_start = time.time()
        bscale = branch_scale_for_epoch(epoch, a.freeze_branch_epochs, a.branch_ramp_epochs)
        base_scale = base_grad_scale_for_epoch(epoch, a.freeze_base_epochs,
                                               a.base_unfreeze_ramp_epochs)
        aux_weight = (a.memory_aux_warmup_weight * (1.0 - bscale)
                      + a.memory_aux_final_weight * bscale)
        bs = _rep_scalar(bscale, ndev)
        bgs = _rep_scalar(base_scale, ndev)
        aws = _rep_scalar(aux_weight, ndev)

        tr_correct = tr_count = 0
        metric_sum = np.zeros(12, np.float64)  # loss + 11 metrics
        last_grad = 0.0
        _write_progress(progress_path, {
            "protocol": a.protocol, "phase": "train", "epoch": epoch,
            "epochs": a.epochs, "n": 0, "total": train_steps,
            "accuracy": 0.0, "memory_accuracy": 0.0,
            "best": best, "best_epoch": best_epoch,
            "branch_scale": bscale, "base_grad_scale": base_scale,
        })
        for bi, (x_np, y_np) in enumerate(ns.batch_iterator(
            train_ds, batch_size=a.batch_size, shuffle=True,
            seed=a.seed + epoch, drop_last=True)):
            xs, ys = _shard_train(x_np, y_np, ndev)
            step_key = jax.random.fold_in(rng, global_step)
            keys = np.asarray(jax.random.split(step_key, ndev))
            params_repl, ema_repl, opt_repl, loss, metrics, gn = train_step(
                params_repl, ema_repl, opt_repl, xs, ys, keys, bs, bgs, aws)
            global_step += 1
            met = np.asarray(metrics)[0]
            lval = float(np.asarray(loss)[0])
            n = len(y_np)
            tr_count += n
            tr_correct += int(round(float(met[5]) * n))
            metric_sum += np.concatenate([[lval], met]) * n
            last_grad = float(np.asarray(gn)[0])
            if (bi + 1) % max(1, a.progress_every) == 0 or bi + 1 == train_steps:
                _write_progress(progress_path, {
                    "protocol": a.protocol, "phase": "train", "epoch": epoch,
                    "epochs": a.epochs, "n": bi + 1, "total": train_steps,
                    "accuracy": tr_correct / max(1, tr_count),
                    "memory_accuracy": metric_sum[7] / max(1, tr_count),
                    "loss": metric_sum[0] / max(1, tr_count),
                    "best": best, "best_epoch": best_epoch,
                    "branch_scale": bscale, "base_grad_scale": base_scale,
                    "gate": metric_sum[8] / max(1, tr_count),
                    "qoverlap": metric_sum[10] / max(1, tr_count),
                })

        val_correct = val_mem_correct = val_count = 0.0
        val_weights = np.zeros(4, np.float64)
        diag_sum = np.zeros(5, np.float64)
        val_batches = 0
        val_labels, val_preds = [], []
        _write_progress(progress_path, {
            "protocol": a.protocol, "phase": "val", "epoch": epoch,
            "epochs": a.epochs, "n": 0, "total": val_steps,
            "accuracy": 0.0, "memory_accuracy": 0.0,
            "best": best, "best_epoch": best_epoch,
            "branch_scale": bscale, "base_grad_scale": base_scale,
        })
        for vi, (x_np, y_np) in enumerate(ns.batch_iterator(
            val_ds, batch_size=a.eval_batch_size, shuffle=False, seed=0,
            drop_last=False)):
            xs, ys, vs = _shard_eval(x_np, y_np, ndev)
            c, mc, n, ws, dg, pr = eval_step(ema_repl, xs, ys, vs, bs)
            val_correct += float(np.asarray(c).sum())
            val_mem_correct += float(np.asarray(mc).sum())
            val_count += float(np.asarray(n).sum())
            val_weights += np.asarray(ws).sum(axis=0)
            diag_sum += np.asarray(dg).sum(axis=0)
            val_batches += ndev
            pr = np.asarray(pr).reshape(-1)
            yy = ys.reshape(-1); vv = vs.reshape(-1) > 0.5
            val_preds.append(pr[vv]); val_labels.append(yy[vv])
            if (vi + 1) % max(1, a.progress_every) == 0 or vi + 1 == val_steps:
                _write_progress(progress_path, {
                    "protocol": a.protocol, "phase": "val", "epoch": epoch,
                    "epochs": a.epochs, "n": vi + 1, "total": val_steps,
                    "accuracy": val_correct / max(1.0, val_count),
                    "memory_accuracy": val_mem_correct / max(1.0, val_count),
                    "best": best, "best_epoch": best_epoch,
                    "branch_scale": bscale, "base_grad_scale": base_scale,
                })

        train_acc = tr_correct / max(1, tr_count)
        val_acc = val_correct / max(1.0, val_count)
        mem_val_acc = val_mem_correct / max(1.0, val_count)
        weights_mean = val_weights / max(1.0, val_count)
        diag_mean = diag_sum / max(1, val_batches)
        val_labels = np.concatenate(val_labels)
        val_preds = np.concatenate(val_preds)
        improved = val_acc > best
        macro = _confusion_and_per_class(val_labels, val_preds)[1].mean()

        if improved:
            best, best_epoch, stale = val_acc, epoch, 0
            ema_host = _unrep(ema_repl)
            payload = {
                "model": MODEL_NAME, "protocol": a.protocol, "epoch": epoch,
                "seed": a.seed, "val_accuracy": val_acc,
                "baseline_accuracy": baseline_acc,
                "base_checkpoint": loaded_path_text,
                "ema_params": ema_host, "args": vars(a),
            }
            if a.save_online:
                payload["params"] = _unrep(params_repl)
            (outdir / "best_ema.msgpack").write_bytes(serialization.to_bytes(payload))
            _save_class_metrics(outdir, val_labels, val_preds, "best")
        else:
            stale += 1

        epoch_seconds = time.time() - epoch_start
        rec = {
            "epoch": epoch,
            "branch_scale": bscale,
            "base_grad_scale": base_scale,
            "memory_aux_weight": aux_weight,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "memory_val_accuracy": mem_val_acc,
            "macro_val_accuracy": float(macro),
            "best_val_accuracy": best,
            "best_epoch": best_epoch,
            "loss": metric_sum[0] / max(1, tr_count),
            "main_ce": metric_sum[1] / max(1, tr_count),
            "memory_ce": metric_sum[2] / max(1, tr_count),
            "predictive_loss": metric_sum[3] / max(1, tr_count),
            "diversity_loss": metric_sum[4] / max(1, tr_count),
            "stream_kl": metric_sum[5] / max(1, tr_count),
            "class_gate_mean": float(diag_mean[0]),
            "class_gate_max": float(diag_mean[1]),
            "class_gate_min": float(diag_mean[2]),
            "query_overlap_mean": float(diag_mean[3]),
            "query_overlap_max": float(diag_mean[4]),
            "dynamic_fusion_weights": weights_mean.tolist(),
            "grad_norm_last": last_grad,
            "epoch_seconds": epoch_seconds,
            "train_samples_per_sec": tr_count / max(epoch_seconds, 1e-6),
            "stale_epochs": stale,
        }
        history.append(rec)
        (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        _write_progress(progress_path, {
            "protocol": a.protocol, "phase": "epoch_done", "epoch": epoch,
            "epochs": a.epochs, "n": 1, "total": 1,
            "accuracy": val_acc, "memory_accuracy": mem_val_acc,
            "macro_accuracy": float(macro),
            "best": best, "best_epoch": best_epoch,
            "stale": stale, "patience": a.patience,
            "loss": rec["loss"], "gate": rec["class_gate_mean"],
            "qoverlap": rec["query_overlap_mean"],
            "branch_scale": bscale, "base_grad_scale": base_scale,
        })
        wtxt = "/".join(f"{w:.2f}" for w in weights_mean)
        print(
            f"{a.protocol.upper()} E{epoch:03}/{a.epochs} | train={100*train_acc:.2f}% | "
            f"val={100*val_acc:.2f}% | MEM={100*mem_val_acc:.2f}% | macro={100*macro:.2f}% | "
            f"BEST={100*best:.2f}%@E{best_epoch:03} | bscale={bscale:.2f} basegrad={base_scale:.2f} | "
            f"gate={rec['class_gate_mean']:.3f} Qov={rec['query_overlap_mean']:.3f} | "
            f"W={wtxt} | stale={stale}/{a.patience} | {epoch_seconds:.1f}s",
            flush=True)
        if a.patience > 0 and stale >= a.patience:
            break

    result = {
        "model": MODEL_NAME, "protocol": a.protocol, "seed": a.seed,
        "backend": jax.default_backend(), "tpu_devices": ndev,
        "device_ids": [int(d.id) for d in devices],
        "frames": 16, "coarse_tokens": 320, "fine_tokens": 320,
        "readout_tokens": a.readout_tokens,
        "base_checkpoint": loaded_path_text,
        "baseline_accuracy": baseline_acc,
        "baseline_macro_accuracy": baseline_macro,
        "base_params": bp, "added_params": added, "params": tp, "leaves": tl,
        "audit": audit, "optimizer_group_params": group_counts,
        "best_val_accuracy": best, "best_epoch": best_epoch,
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

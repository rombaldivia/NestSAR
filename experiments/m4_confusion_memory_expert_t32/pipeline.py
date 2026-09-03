#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training import train_state

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as raw
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_localglobal_t16 import preprocessing as lg
from experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16.model import M4PhaseUniformBiJointT16
from experiments.m4_confusion_memory_expert_t32.model import (
    ConfusionMemoryExpert,
    SELECTED_JOINTS,
    T32,
    TOKEN_FEATURES,
)

NUM_CLASSES = 120
PERSONS = 2
JOINTS = 25
XYZ = 3


def emit(event: str, **payload) -> None:
    print("@@CME@@" + json.dumps({"event": event, **payload}, separators=(",", ":")), flush=True)


def parse_int_list(text: str) -> list[int]:
    out = []
    for x in text.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    if not out:
        raise ValueError("empty integer list")
    return out


def weak_zero_based(text: str) -> list[int]:
    vals = parse_int_list(text)
    if min(vals) >= 1:
        vals = [x - 1 for x in vals]
    if min(vals) < 0 or max(vals) >= NUM_CLASSES:
        raise ValueError(f"weak classes out of range: {vals}")
    return sorted(set(vals))


def parse_pairs(text: str, weak: list[int]) -> list[tuple[int, int]]:
    if not text.strip():
        return []
    pos = {c: i for i, c in enumerate(weak)}
    out = []
    for item in text.split(","):
        a, b = item.strip().split("-")
        aa, bb = int(a), int(b)
        if aa >= 1 and bb >= 1:
            aa -= 1; bb -= 1
        if aa in pos and bb in pos:
            out.append((pos[aa], pos[bb]))
    return out


def canonical_raw(keypoints: np.ndarray) -> np.ndarray:
    x = raw.to_tmvc(keypoints).astype(np.float32)
    if x.shape[2] < JOINTS:
        pad = np.zeros((x.shape[0], x.shape[1], JOINTS - x.shape[2], XYZ), np.float32)
        x = np.concatenate([x, pad], axis=2)
    x = x[:, :, :JOINTS, :XYZ]
    energy = np.sum(np.abs(x), axis=(0, 2, 3))
    x = x[:, np.argsort(-energy)]
    if x.shape[1] < PERSONS:
        x = np.concatenate([x, np.zeros((x.shape[0], PERSONS - x.shape[1], JOINTS, XYZ), np.float32)], axis=1)
    x = x[:, :PERSONS]
    center = x[:, 0:1, 0:1, :]
    valid_center = np.any(np.abs(center) > 1e-8, axis=-1, keepdims=True)
    x = np.where(valid_center, x - center, x)
    valid_joint = np.any(np.abs(x) > 1e-8, axis=-1, keepdims=True)
    return np.where(valid_joint, x, 0.0).astype(np.float32)


def specialist_tokens(keypoints: np.ndarray) -> np.ndarray:
    x = canonical_raw(keypoints)
    n = x.shape[0]
    if n <= 0:
        return np.zeros((T32, TOKEN_FEATURES), np.float32)
    idx = np.linspace(0, max(n - 1, 0), T32, dtype=np.float32).round().astype(np.int64)
    s = x[idx][:, :, SELECTED_JOINTS, :]
    v = np.zeros_like(s)
    v[1:] = s[1:] - s[:-1]
    a = np.zeros_like(s)
    a[1:] = v[1:] - v[:-1]
    per_joint = np.concatenate([s, v, a], axis=-1).reshape(T32, -1)

    # Twelve cheap relation features: hand/hand, hand/head and hand/torso distances,
    # plus left/right wrist velocity magnitudes for both tracked people.
    def dist(j1: int, j2: int) -> np.ndarray:
        return np.linalg.norm(x[idx, :, j1] - x[idx, :, j2], axis=-1)

    rel = np.stack([
        dist(7, 11)[:, 0], dist(7, 3)[:, 0], dist(11, 3)[:, 0],
        dist(7, 0)[:, 0], dist(11, 0)[:, 0], dist(6, 10)[:, 0],
        dist(7, 11)[:, 1], dist(7, 3)[:, 1], dist(11, 3)[:, 1],
        np.linalg.norm(v[:, 0, SELECTED_JOINTS.index(6)], axis=-1),
        np.linalg.norm(v[:, 0, SELECTED_JOINTS.index(10)], axis=-1),
        np.linalg.norm(v[:, 0], axis=(-2, -1)),
    ], axis=-1).astype(np.float32)

    nz = np.abs(s) > 1e-8
    rms = float(np.sqrt(np.mean(np.square(s[nz]))) + 1e-6) if np.any(nz) else 1.0
    per_joint /= rms
    rel /= rms
    out = np.concatenate([per_joint, rel], axis=-1)
    return np.nan_to_num(out).astype(np.float32)


def load_base_model(kind: str, ckpt: Path):
    payload = serialization.msgpack_restore(ckpt.read_bytes())
    if not isinstance(payload, dict):
        raise RuntimeError("Base checkpoint is not a dict payload")
    params = payload.get("ema_params", payload.get("params"))
    if params is None:
        raise RuntimeError("Base checkpoint has no ema_params/params")
    cfg = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
    spatial_dim = int(cfg.get("spatial_dim", 24))
    model_dim = int(cfg.get("model_dim", 112))
    dropout = float(cfg.get("dropout", 0.10))
    if kind == "bijoint":
        model = M4PhaseUniformBiJointT16(spatial_dim, model_dim, dropout)
    elif kind == "localglobal":
        model = ju.M4PhaseUniformT16(spatial_dim, model_dim, dropout)
    else:
        raise ValueError(f"unknown base kind {kind}")
    return model, params, payload


def split_ids(annotations, split, protocol: str):
    tk, vk = raw.resolve_split(split, protocol)
    by_id = {raw.sample_id(a, i): a for i, a in enumerate(annotations) if isinstance(a, Mapping)}
    tr = [str(v) for v in split[tk] if str(v) in by_id]
    va = [str(v) for v in split[vk] if str(v) in by_id]
    if not tr or not va:
        raise RuntimeError(f"empty split {protocol}: train={len(tr)} val={len(va)}")
    return by_id, tr, va


def cache_complete(root: Path, ckpt_sha: str, protocol: str, base_kind: str) -> bool:
    m = root / "manifest.json"
    if not m.is_file():
        return False
    try:
        meta = json.loads(m.read_text())
    except Exception:
        return False
    required = ["train_tokens.npy", "train_logits.npy", "train_y.npy", "val_tokens.npy", "val_logits.npy", "val_y.npy"]
    return (
        meta.get("complete") is True
        and meta.get("base_ckpt_sha256") == ckpt_sha
        and meta.get("protocol") == protocol
        and meta.get("base_kind") == base_kind
        and all((root / x).is_file() for x in required)
    )


def build_cache(args) -> None:
    root = Path(args.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.base_ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if cache_complete(root, sha, args.protocol, args.base_kind):
        emit("cache_ready", protocol=args.protocol, reused=True)
        return

    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(f"cache builder requires exactly one visible GPU, got {jax.local_devices()}")

    anns, split = raw.load_ntu(Path(args.dataset))
    by_id, train_ids, val_ids = split_ids(anns, split, args.protocol)
    model, params, _payload = load_base_model(args.base_kind, ckpt)

    @jax.jit
    def infer(p, xb):
        return model.apply({"params": p}, xb, training=False)["logits"]

    def materialize(ids: Sequence[str], stem: str):
        n = len(ids)
        xt = np.lib.format.open_memmap(root / f"{stem}_tokens.npy", mode="w+", dtype=np.float32, shape=(n, T32, TOKEN_FEATURES))
        zl = np.lib.format.open_memmap(root / f"{stem}_logits.npy", mode="w+", dtype=np.float32, shape=(n, NUM_CLASSES))
        yy = np.lib.format.open_memmap(root / f"{stem}_y.npy", mode="w+", dtype=np.int32, shape=(n,))
        bs = int(args.base_batch_size)
        buf_tokens = []
        buf_rows = []
        for i, sid in enumerate(ids):
            ann = by_id[sid]
            kp = raw.annotation_keypoints(ann)
            xt[i] = specialist_tokens(kp)
            yy[i] = raw.annotation_label(ann)
            buf_tokens.append(lg.segment_phase_tokens_localglobal(kp))
            buf_rows.append(i)
            if len(buf_tokens) >= bs or i == n - 1:
                xb = np.asarray(buf_tokens, np.float32)
                pred = np.asarray(jax.device_get(infer(params, jnp.asarray(xb))), np.float32)
                zl[np.asarray(buf_rows)] = pred
                buf_tokens.clear(); buf_rows.clear()
        xt.flush(); zl.flush(); yy.flush()

    emit("cache_start", protocol=args.protocol, train=len(train_ids), val=len(val_ids))
    materialize(train_ids, "train")
    materialize(val_ids, "val")
    (root / "manifest.json").write_text(json.dumps({
        "complete": True,
        "protocol": args.protocol,
        "base_kind": args.base_kind,
        "base_ckpt": str(ckpt),
        "base_ckpt_sha256": sha,
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "t32": T32,
        "token_features": TOKEN_FEATURES,
        "dtype": "float32",
    }, indent=2))
    emit("cache_ready", protocol=args.protocol, reused=False)


class SpecialistState(train_state.TrainState):
    ema_params: object


def select_examples(y: np.ndarray, base_logits: np.ndarray, weak: list[int], margin_threshold: float):
    probs = np.exp(base_logits - base_logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    order = np.argsort(probs, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    margin = probs[np.arange(len(y)), top1] - probs[np.arange(len(y)), top2]
    wset = np.asarray(weak, np.int32)
    gt_weak = np.isin(y, wset)
    pred_weak = np.isin(top1, wset)
    top2_weak = np.isin(top2, wset)
    ambiguous = (margin < margin_threshold) & (pred_weak | top2_weak | gt_weak)
    keep = gt_weak | pred_weak | top2_weak | ambiguous
    return np.where(keep)[0], top1, margin


def combined_metrics(base_logits: np.ndarray, spec_logits: np.ndarray, y: np.ndarray, weak: list[int], alpha: float, margin_threshold: float, gate_temperature: float):
    k = len(weak)
    bp = np.exp(base_logits - base_logits.max(axis=1, keepdims=True)); bp /= bp.sum(axis=1, keepdims=True)
    order = np.argsort(bp, axis=1)
    margin = bp[np.arange(len(y)), order[:, -1]] - bp[np.arange(len(y)), order[:, -2]]
    sp = np.exp(spec_logits - spec_logits.max(axis=1, keepdims=True)); sp /= sp.sum(axis=1, keepdims=True)
    reject = sp[:, k]
    gate_unc = 1.0 / (1.0 + np.exp((margin - margin_threshold) * gate_temperature))
    gate = (1.0 - reject) * gate_unc
    delta = spec_logits[:, :k] - spec_logits[:, :k].mean(axis=1, keepdims=True)
    final = base_logits.copy()
    final[:, weak] += alpha * gate[:, None] * delta
    pb = np.argmax(base_logits, axis=1)
    pf = np.argmax(final, axis=1)
    fixes = int(np.sum((pb != y) & (pf == y)))
    harms = int(np.sum((pb == y) & (pf != y)))
    return {
        "base_acc": float(np.mean(pb == y)),
        "final_acc": float(np.mean(pf == y)),
        "fixes": fixes,
        "harms": harms,
        "net_rescue": fixes - harms,
        "trigger_mean": float(np.mean(gate)),
    }


def train_specialist(args) -> None:
    root = Path(args.cache_dir)
    out = Path(args.outdir) / args.protocol
    out.mkdir(parents=True, exist_ok=True)
    weak = weak_zero_based(args.weak_classes)
    pairs = parse_pairs(args.confusion_pairs, weak)
    reject_id = len(weak)

    Xtr = np.load(root / "train_tokens.npy", mmap_mode="r")
    Ztr = np.load(root / "train_logits.npy", mmap_mode="r")
    ytr = np.load(root / "train_y.npy", mmap_mode="r")
    Xva = np.load(root / "val_tokens.npy", mmap_mode="r")
    Zva = np.load(root / "val_logits.npy", mmap_mode="r")
    yva = np.load(root / "val_y.npy", mmap_mode="r")

    keep, base_top1, base_margin = select_examples(np.asarray(ytr), np.asarray(Ztr), weak, args.selection_margin)
    rng = np.random.default_rng(args.seed + (0 if args.protocol == "xsub" else 100000))
    rng.shuffle(keep)
    nh = max(1, int(round(len(keep) * args.holdout_fraction)))
    hold_idx = np.sort(keep[:nh])
    fit_idx = np.sort(keep[nh:])
    if len(fit_idx) < args.batch_size:
        raise RuntimeError(f"too few specialist fit examples: {len(fit_idx)}")

    weak_pos = {c: i for i, c in enumerate(weak)}
    def map_labels(ids):
        yy = np.asarray(ytr)[ids]
        return np.asarray([weak_pos.get(int(v), reject_id) for v in yy], np.int32)

    yfit = map_labels(fit_idx)
    yhold = map_labels(hold_idx)

    # Static sample weights emphasize champion mistakes and false weak predictions.
    fit_gt = np.asarray(ytr)[fit_idx]
    fit_pred = base_top1[fit_idx]
    fit_margin = base_margin[fit_idx]
    sw = np.ones(len(fit_idx), np.float32)
    sw[(np.isin(fit_gt, weak)) & (fit_pred != fit_gt)] *= args.hard_positive_weight
    sw[(np.isin(fit_gt, weak)) & (fit_margin < args.selection_margin)] *= args.ambiguous_weight
    sw[(~np.isin(fit_gt, weak)) & np.isin(fit_pred, weak)] *= args.reject_weight

    model = ConfusionMemoryExpert(len(weak) + 1, args.specialist_dim, args.topk_context, args.dropout)
    key = jax.random.PRNGKey(args.seed)
    key, ik = jax.random.split(key)
    params = model.init({"params": ik, "dropout": ik}, jnp.zeros((1, T32, TOKEN_FEATURES), jnp.float32), jnp.zeros((1, NUM_CLASSES), jnp.float32), training=False)["params"]
    nparams = sum(int(x.size) for x in jax.tree_util.tree_leaves(params))

    steps = max(1, math.ceil(len(fit_idx) / args.batch_size) * args.epochs)
    warm = max(1, int(steps * args.warmup_fraction))
    sched = optax.warmup_cosine_decay_schedule(0.0, args.learning_rate, warm, steps, end_value=args.min_learning_rate)
    tx = optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adamw(sched, weight_decay=args.weight_decay))
    state = SpecialistState.create(apply_fn=model.apply, params=params, tx=tx, ema_params=params)

    pair_arr = np.asarray(pairs, np.int32) if pairs else np.zeros((0, 2), np.int32)

    @jax.jit
    def train_step(state, rng_key, xb, zb, yb, wb):
        rng_key, dk = jax.random.split(rng_key)
        def loss_fn(p):
            logits = model.apply({"params": p}, xb, zb, training=True, rngs={"dropout": dk})["logits"]
            logp = jax.nn.log_softmax(logits, axis=-1)
            ce = -logp[jnp.arange(yb.shape[0]), yb]
            ce = jnp.sum(ce * wb) / jnp.maximum(jnp.sum(wb), 1e-6)
            pair_loss = jnp.asarray(0.0, jnp.float32)
            if pair_arr.shape[0] > 0:
                total = jnp.asarray(0.0, jnp.float32); count = jnp.asarray(0.0, jnp.float32)
                for a, b in pair_arr:
                    ma = yb == a; mb = yb == b
                    la = jnp.maximum(0.0, args.pair_margin - logits[:, a] + logits[:, b]) * ma
                    lb = jnp.maximum(0.0, args.pair_margin - logits[:, b] + logits[:, a]) * mb
                    total = total + jnp.sum(la) + jnp.sum(lb)
                    count = count + jnp.sum(ma) + jnp.sum(mb)
                pair_loss = total / jnp.maximum(count, 1.0)
            return ce + args.pair_weight * pair_loss, (ce, pair_loss)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        ema = jax.tree_util.tree_map(lambda e, p: args.ema_decay * e + (1.0 - args.ema_decay) * p, state.ema_params, state.params)
        state = state.replace(ema_params=ema)
        return state, rng_key, loss, aux

    @jax.jit
    def infer(p, xb, zb):
        return model.apply({"params": p}, xb, zb, training=False)["logits"]

    def predict(ids, X, Z, bs=512):
        outs = []
        for s in range(0, len(ids), bs):
            ii = ids[s:s+bs]
            outs.append(np.asarray(jax.device_get(infer(state.ema_params, jnp.asarray(np.asarray(X[ii]), jnp.float32), jnp.asarray(np.asarray(Z[ii]), jnp.float32)))))
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, len(weak)+1), np.float32)

    best = -1e9; best_epoch = 0; best_params = None; stale = 0
    emit("train_start", protocol=args.protocol, epochs=args.epochs, specialist_params=nparams, fit=len(fit_idx), holdout=len(hold_idx), weak_classes=[x+1 for x in weak])

    for epoch in range(1, args.epochs + 1):
        order = fit_idx.copy(); rng.shuffle(order)
        losses = []
        for s in range(0, len(order), args.batch_size):
            ii = order[s:s+args.batch_size]
            if len(ii) < 2:
                continue
            local = np.searchsorted(fit_idx, ii)
            wb = sw[local]
            yb = np.asarray([weak_pos.get(int(v), reject_id) for v in np.asarray(ytr)[ii]], np.int32)
            state, key, lv, _aux = train_step(
                state, key,
                jnp.asarray(np.asarray(Xtr[ii]), jnp.float32),
                jnp.asarray(np.asarray(Ztr[ii]), jnp.float32),
                jnp.asarray(yb), jnp.asarray(wb),
            )
            losses.append(float(lv))

        sh = predict(hold_idx, Xtr, Ztr)
        hold_comb = combined_metrics(np.asarray(Ztr[hold_idx]), sh, np.asarray(ytr[hold_idx]), weak, args.alpha, args.route_margin, args.gate_temperature)
        score = hold_comb["final_acc"] - args.harm_penalty * (hold_comb["harms"] / max(len(hold_idx), 1))
        if score > best + 1e-8:
            best = score; best_epoch = epoch; stale = 0
            best_params = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), state.ema_params)
        else:
            stale += 1
        emit("epoch", protocol=args.protocol, epoch=epoch, epochs=args.epochs, loss=float(np.mean(losses) if losses else 0.0), hold_final=hold_comb["final_acc"], hold_base=hold_comb["base_acc"], fixes=hold_comb["fixes"], harms=hold_comb["harms"], best_epoch=best_epoch)
        if stale >= args.patience:
            emit("early_stop", protocol=args.protocol, epoch=epoch)
            break

    if best_params is None:
        raise RuntimeError("specialist never produced a valid checkpoint")
    state = state.replace(ema_params=best_params)

    val_ids = np.arange(len(yva), dtype=np.int64)
    sv = []
    for s in range(0, len(val_ids), args.eval_batch_size):
        ii = val_ids[s:s+args.eval_batch_size]
        sv.append(np.asarray(jax.device_get(infer(state.ema_params, jnp.asarray(np.asarray(Xva[ii]), jnp.float32), jnp.asarray(np.asarray(Zva[ii]), jnp.float32)))))
    sv = np.concatenate(sv, axis=0)
    official = combined_metrics(np.asarray(Zva), sv, np.asarray(yva), weak, args.alpha, args.route_margin, args.gate_temperature)

    spec_payload = {
        "model": "NestSAR_ConfusionMemoryExpert_T32",
        "protocol": args.protocol,
        "best_epoch": best_epoch,
        "params": best_params,
        "weak_classes_zero_based": weak,
        "weak_classes_one_based": [x+1 for x in weak],
        "reject_id": reject_id,
        "config": vars(args),
        "specialist_params": nparams,
        "official_validation": official,
    }
    (out / "specialist_best.msgpack").write_bytes(serialization.msgpack_serialize(spec_payload))
    (out / "result.json").write_text(json.dumps({"protocol": args.protocol, "best_epoch": best_epoch, "specialist_params": nparams, **official}, indent=2))

    # One-file deployment bundle: frozen champion payload + separately trained specialist.
    base_payload = serialization.msgpack_restore(Path(args.base_ckpt).read_bytes())
    bundle = {
        "architecture": "NestSAR_BasePlusCME_T32",
        "base_kind": args.base_kind,
        "base_checkpoint": base_payload,
        "specialist": spec_payload,
        "routing": {"alpha": args.alpha, "margin": args.route_margin, "gate_temperature": args.gate_temperature},
    }
    (out / "embedded_bundle.msgpack").write_bytes(serialization.msgpack_serialize(bundle))
    emit("done", protocol=args.protocol, best_epoch=best_epoch, specialist_params=nparams, **official)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cache", "train"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", choices=["xsub", "xset"], required=True)
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--base-kind", choices=["localglobal", "bijoint"], default="bijoint")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--weak-classes", default="71,72,73,74,75,76,82,84,106,107")
    p.add_argument("--confusion-pairs", default="71-72,73-76,74-84,106-107")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--base-batch-size", type=int, default=256)
    p.add_argument("--specialist-dim", type=int, default=32)
    p.add_argument("--topk-context", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--learning-rate", type=float, default=6e-4)
    p.add_argument("--min-learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.08)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--holdout-fraction", type=float, default=0.10)
    p.add_argument("--selection-margin", type=float, default=0.20)
    p.add_argument("--hard-positive-weight", type=float, default=2.0)
    p.add_argument("--ambiguous-weight", type=float, default=1.5)
    p.add_argument("--reject-weight", type=float, default=1.5)
    p.add_argument("--pair-weight", type=float, default=0.20)
    p.add_argument("--pair-margin", type=float, default=0.20)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--route-margin", type=float, default=0.15)
    p.add_argument("--gate-temperature", type=float, default=12.0)
    p.add_argument("--harm-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=128)
    return p


def main() -> None:
    args = make_parser().parse_args()
    if args.mode == "cache":
        build_cache(args)
    else:
        train_specialist(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""Validation/activation bottleneck audit for M4-MotionPreserve-T16.

This audit complements weight_audit.py.  It loads the saved EMA checkpoint and
measures what the learned representations do on NTU120 validation data:

* full / per-stream / leave-one-stream-out accuracy
* router-off accuracy using the exact learned descriptor + classifier heads
* input-component ablations (pose, displacement, path motion)
* temporal-order sensitivity (reverse / shuffle)
* stage-wise frozen linear probes
* activation effective rank and cross-stream activation similarity
* router/fusion entropy and class-dependent stream preferences
* per-class accuracy/confusions
* segment path coverage and within-segment cancellation (phase-risk) statistics

No training weights are modified.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from experiments.m4_motionpreserve_t16.jax10_compat import install as install_jax10_compat

install_jax10_compat()

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as tr


STREAMS = ("J", "B", "JM", "BM")
NUM_STREAMS = 4
EPS = 1e-8


def log(msg: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scalar(v: Any, default: float) -> float:
    try:
        return float(np.asarray(v))
    except Exception:
        return float(default)


def int_scalar(v: Any, default: int) -> int:
    try:
        return int(np.asarray(v))
    except Exception:
        return int(default)


def entropy_np(w: np.ndarray, axis: int = -1, normalized: bool = False) -> np.ndarray:
    w = np.asarray(w, np.float64)
    h = -np.sum(w * np.log(np.maximum(w, 1e-12)), axis=axis)
    if normalized:
        h = h / math.log(w.shape[axis])
    return h


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-30)


def resolve_ids(annotations, split: Mapping[str, Any], protocol: str):
    tk, vk = tr.resolve_split(split, protocol)
    by_id = {
        tr.sample_id(a, i): a
        for i, a in enumerate(annotations)
        if isinstance(a, Mapping)
    }
    train_ids = [str(v) for v in split[tk] if str(v) in by_id]
    val_ids = [str(v) for v in split[vk] if str(v) in by_id]
    if not train_ids or not val_ids:
        raise RuntimeError(
            f"Empty resolved split for {protocol}: train={len(train_ids)} val={len(val_ids)}"
        )
    return by_id, train_ids, val_ids


def materialize(by_id, ids: Sequence[str], desc: str):
    X = np.empty((len(ids), tr.FRAMES, tr.FEATURES), np.float32)
    y = np.empty((len(ids),), np.int32)
    for i, sid in enumerate(tqdm(ids, desc=desc, mininterval=0.5)):
        a = by_id[sid]
        X[i] = tr.preprocess_keypoints(tr.annotation_keypoints(a), "segment")
        y[i] = tr.annotation_label(a)
    return X, y


def select_random_ids(ids: Sequence[str], n: int, seed: int) -> list[str]:
    if n <= 0 or n >= len(ids):
        return list(ids)
    rng = np.random.default_rng(seed)
    ii = rng.choice(len(ids), size=n, replace=False)
    return [ids[int(i)] for i in ii]


def build_model_from_payload(payload: Mapping[str, Any]):
    cfg = payload.get("config", {})
    if not isinstance(cfg, Mapping):
        cfg = {}
    spatial_dim = int_scalar(cfg.get("spatial_dim", 24), 24)
    model_dim = int_scalar(cfg.get("model_dim", 112), 112)
    dropout = scalar(cfg.get("dropout", 0.10), 0.10)
    model = tr.M4MotionPreserveT16(spatial_dim, model_dim, dropout)
    return model, spatial_dim, model_dim, dropout


def dense_apply(p: Mapping[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    y = x @ p["kernel"]
    if "bias" in p:
        y = y + p["bias"]
    return y


def make_functions(model, model_dim: int):
    @jax.jit
    def full_eval(params, xb):
        out = model.apply({"params": params}, xb, training=False)

        # Exact router-off counterfactual: feed each learned pre-router frame-memory
        # representation through the existing learned descriptor/classifier heads.
        ro_descs = []
        ro_logits = []
        for i in range(NUM_STREAMS):
            _, desc = tr.DescriptorHead(model_dim, 0.0).apply(
                {"params": params[f"descriptor_{i}"]},
                out["frame_stack"][:, :, i],
                training=False,
            )
            ro_descs.append(desc)
            ro_logits.append(dense_apply(params[f"classifier_{i}"], desc))
        ro_descs = jnp.stack(ro_descs, axis=1)
        ro_sl = jnp.stack(ro_logits, axis=1)
        ctrl = dense_apply(
            params["fusion_controller"], ro_descs.reshape(ro_descs.shape[0], -1)
        )
        ro_fusion = jax.nn.softmax(
            params["fusion_prior"][None, :] + 0.15 * jnp.tanh(ctrl), axis=-1
        )
        ro_logits_fused = jnp.einsum("bs,bsc->bc", ro_fusion, ro_sl)

        return (
            out["logits"],
            out["stream_logits"],
            out["fusion_weights"],
            out["router_weights"],
            ro_logits_fused,
        )

    @jax.jit
    def logits_only(params, xb):
        return model.apply({"params": params}, xb, training=False)["logits"]

    @jax.jit
    def feature_eval(params, xb):
        out = model.apply({"params": params}, xb, training=False)
        # Every stage is reduced to [B, 4, D] so all probe dimensions match.
        spatial = out["spatial_stack"].mean(axis=1)
        frame = out["frame_stack"].mean(axis=1)
        mixed = out["mixed_frame_stack"].mean(axis=1)
        chunk = out["chunk_states"].mean(axis=2)
        desc = out["descriptors"]
        num = jnp.linalg.norm(out["mixed_frame_stack"] - out["frame_stack"], axis=-1)
        den = jnp.maximum(jnp.linalg.norm(out["frame_stack"], axis=-1), 1e-8)
        router_delta_ratio = jnp.mean(num / den, axis=(1, 2))
        return spatial, frame, mixed, chunk, desc, router_delta_ratio

    return full_eval, logits_only, feature_eval


def run_full_eval(fn, params, X: np.ndarray, batch_size: int):
    logits_all = []
    stream_all = []
    fusion_all = []
    router_all = []
    router_off_all = []
    for s in tqdm(range(0, len(X), batch_size), desc="full validation inference", mininterval=0.5):
        xb = jnp.asarray(X[s:s + batch_size])
        out = fn(params, xb)
        logits, sl, fusion, router, ro = [np.asarray(jax.device_get(v)) for v in out]
        logits_all.append(logits)
        stream_all.append(sl)
        fusion_all.append(fusion)
        router_all.append(router)
        router_off_all.append(ro)
    return (
        np.concatenate(logits_all, axis=0),
        np.concatenate(stream_all, axis=0),
        np.concatenate(fusion_all, axis=0),
        np.concatenate(router_all, axis=0),
        np.concatenate(router_off_all, axis=0),
    )


def transform_batch(xb: np.ndarray, mode: str, shuffle_perm: np.ndarray) -> np.ndarray:
    if mode == "baseline":
        return xb
    z = xb.copy().reshape(len(xb), tr.FRAMES, tr.PERSONS, tr.JOINTS, tr.TOKEN_CHANNELS)
    if mode == "no_path":
        z[..., 6:9] = 0.0
    elif mode == "no_disp":
        z[..., 3:6] = 0.0
    elif mode == "pose_only":
        z[..., 3:9] = 0.0
    elif mode == "motion_only":
        z[..., 0:3] = 0.0
    elif mode == "disp_only":
        z[..., 0:3] = 0.0
        z[..., 6:9] = 0.0
    elif mode == "path_only":
        z[..., 0:6] = 0.0
    elif mode == "reverse_tokens":
        z = z[:, ::-1]
    elif mode == "shuffle_tokens":
        z = z[:, shuffle_perm]
    else:
        raise ValueError(mode)
    return z.reshape(len(xb), tr.FRAMES, tr.FEATURES)


def run_variant_accuracy(fn, params, X, y, batch_size: int, mode: str, seed: int) -> float:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(tr.FRAMES)
    correct = 0
    total = 0
    for s in tqdm(range(0, len(X), batch_size), desc=f"ablation {mode}", leave=False, mininterval=0.5):
        xb = transform_batch(X[s:s + batch_size], mode, perm)
        logits = np.asarray(jax.device_get(fn(params, jnp.asarray(xb))))
        pred = np.argmax(logits, axis=-1)
        yy = y[s:s + len(pred)]
        correct += int(np.sum(pred == yy))
        total += len(pred)
    return correct / max(total, 1)


def extract_features(fn, params, X: np.ndarray, batch_size: int):
    names = ("spatial", "frame_memory", "after_router", "chunk_memory", "descriptor")
    bags = {k: [] for k in names}
    delta = []
    for s in tqdm(range(0, len(X), batch_size), desc="extract activations", leave=False, mininterval=0.5):
        vals = fn(params, jnp.asarray(X[s:s + batch_size]))
        vals = [np.asarray(jax.device_get(v)) for v in vals]
        for k, v in zip(names, vals[:5]):
            bags[k].append(v.astype(np.float32, copy=False))
        delta.append(vals[5].astype(np.float32, copy=False))
    stages = {k: np.concatenate(v, axis=0) for k, v in bags.items()}
    return stages, np.concatenate(delta, axis=0)


def flatten_stage(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, np.float32).reshape(len(x), -1)


def linear_probe(train_x, train_y, val_x, val_y) -> float:
    clf = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=1.0),
    )
    clf.fit(train_x, train_y)
    return float(clf.score(val_x, val_y))


def activation_spectrum(x: np.ndarray, max_samples: int, seed: int) -> dict[str, float]:
    X = flatten_stage(x)
    if len(X) > max_samples:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), max_samples, replace=False)]
    X = X.astype(np.float64, copy=False)
    X -= X.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(X, compute_uv=False)
    except np.linalg.LinAlgError:
        return {"effective_rank": float("nan"), "rank_fraction": float("nan"), "stable_rank": float("nan")}
    lam = s * s
    p = lam / max(float(lam.sum()), 1e-30)
    eff = float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-30)))))
    stable = float(lam.sum() / max(float(lam[0]), 1e-30))
    max_rank = float(min(X.shape[0] - 1, X.shape[1]))
    return {
        "effective_rank": eff,
        "rank_fraction": eff / max(max_rank, 1.0),
        "stable_rank": stable,
    }


def activation_stream_cosines(x: np.ndarray) -> dict[str, float]:
    # x: [N,4,D]
    out = {}
    x = np.asarray(x, np.float64)
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = x[:, i], x[:, j]
            den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            cos = np.sum(a * b, axis=1) / np.maximum(den, 1e-12)
            out[f"{STREAMS[i]}-{STREAMS[j]}"] = float(np.mean(cos))
    return out


def fusion_counterfactuals(stream_logits: np.ndarray, fusion: np.ndarray, y: np.ndarray):
    result = {}
    for i, name in enumerate(STREAMS):
        result[f"{name}_only"] = accuracy(y, np.argmax(stream_logits[:, i], axis=-1))

    uniform = np.mean(stream_logits, axis=1)
    result["uniform_fusion"] = accuracy(y, np.argmax(uniform, axis=-1))

    def subset_acc(indices):
        w = fusion[:, indices]
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        lg = np.sum(w[..., None] * stream_logits[:, indices, :], axis=1)
        return accuracy(y, np.argmax(lg, axis=-1))

    result["pose_JB"] = subset_acc([0, 1])
    result["motion_JM_BM"] = subset_acc([2, 3])
    for i, name in enumerate(STREAMS):
        result[f"minus_{name}"] = subset_acc([j for j in range(4) if j != i])
    return result


def class_stats(y: np.ndarray, pred: np.ndarray, fusion: np.ndarray, router: np.ndarray):
    rows = []
    for c in range(tr.NUM_CLASSES):
        m = y == c
        if not np.any(m):
            continue
        acc = float(np.mean(pred[m] == c))
        fw = fusion[m].mean(axis=0)
        rw = router[m].mean(axis=(0, 1))
        rows.append({
            "class": c,
            "n": int(m.sum()),
            "accuracy": acc,
            "fusion_mean": fw.tolist(),
            "router_mean": rw.tolist(),
            "fusion_entropy_norm": float(entropy_np(fw[None], normalized=True)[0]),
            "router_entropy_norm": float(entropy_np(rw[None], normalized=True)[0]),
            "dominant_fusion_stream": STREAMS[int(np.argmax(fw))],
            "dominant_router_stream": STREAMS[int(np.argmax(rw))],
        })
    return rows


def top_confusions(y: np.ndarray, pred: np.ndarray, k: int = 20):
    cm = np.zeros((tr.NUM_CLASSES, tr.NUM_CLASSES), np.int64)
    np.add.at(cm, (y, pred), 1)
    rows = []
    for a in range(tr.NUM_CLASSES):
        for b in range(tr.NUM_CLASSES):
            if a != b and cm[a, b] > 0:
                rows.append((int(cm[a, b]), a, b))
    rows.sort(reverse=True)
    return [{"count": n, "true": a, "pred": b} for n, a, b in rows[:k]]


def token_phase_risk(X: np.ndarray, y: np.ndarray):
    tok = X.reshape(len(X), tr.FRAMES, tr.PERSONS, tr.JOINTS, tr.TOKEN_CHANNELS)
    disp = tok[..., 3:6]
    path = np.maximum(tok[..., 6:9], 0.0)
    disp_mag = np.sum(np.abs(disp), axis=(1, 2, 3, 4))
    path_mag = np.sum(path, axis=(1, 2, 3, 4))
    directionality = disp_mag / np.maximum(path_mag, 1e-8)
    cancellation = np.clip(1.0 - directionality, 0.0, 1.0)
    per_class = []
    for c in range(tr.NUM_CLASSES):
        m = y == c
        if np.any(m):
            per_class.append({
                "class": c,
                "n": int(m.sum()),
                "cancellation": float(cancellation[m].mean()),
                "directionality": float(directionality[m].mean()),
            })
    return {
        "global_cancellation": float(cancellation.mean()),
        "global_directionality": float(directionality.mean()),
        "per_class": per_class,
    }


def raw_path_coverage(by_id, val_ids: Sequence[str], n: int, seed: int):
    ids = select_random_ids(val_ids, min(n, len(val_ids)), seed)
    ratios = []
    by_class: dict[int, list[float]] = {}
    for sid in tqdm(ids, desc="raw segment path coverage", leave=False, mininterval=0.5):
        a = by_id[sid]
        x = tr.canonicalize_raw(tr.annotation_keypoints(a))
        if len(x) < 2:
            continue
        full = float(np.sum(np.abs(x[1:] - x[:-1])))
        if full <= 1e-12:
            continue
        covered = 0.0
        for s, e in tr.segment_bounds(len(x), tr.FRAMES):
            seg = x[s:e]
            if len(seg) >= 2:
                covered += float(np.sum(np.abs(seg[1:] - seg[:-1])))
        r = min(max(covered / full, 0.0), 1.0)
        ratios.append(r)
        by_class.setdefault(tr.annotation_label(a), []).append(r)
    class_rows = [
        {"class": c, "n": len(v), "coverage": float(np.mean(v))}
        for c, v in sorted(by_class.items()) if v
    ]
    return {
        "n": len(ratios),
        "mean_coverage": float(np.mean(ratios)) if ratios else float("nan"),
        "median_coverage": float(np.median(ratios)) if ratios else float("nan"),
        "per_class": class_rows,
    }


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def pct(x: float) -> str:
    return f"{100*x:.3f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--protocol", choices=["xsub", "xset"], required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--probe-train", type=int, default=12000)
    ap.add_argument("--probe-val", type=int, default=8000)
    ap.add_argument("--rank-samples", type=int, default=5000)
    ap.add_argument("--path-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    payload = serialization.msgpack_restore(ckpt.read_bytes())
    if "ema_params" not in payload:
        raise KeyError(f"Checkpoint lacks ema_params; keys={list(payload)}")
    params = payload["ema_params"]
    model, spatial_dim, model_dim, dropout = build_model_from_payload(payload)

    dataset = tr.find_dataset(args.dataset)
    annotations, split = tr.load_ntu(dataset)
    by_id, train_ids, val_ids = resolve_ids(annotations, split, args.protocol)

    rng = np.random.default_rng(args.seed)
    probe_train_ids = select_random_ids(train_ids, args.probe_train, args.seed + 11)
    Xtr, ytr = materialize(by_id, probe_train_ids, f"{args.protocol.upper()} probe-train preprocess")
    Xva, yva = materialize(by_id, val_ids, f"{args.protocol.upper()} full-val preprocess")

    print("=" * 122)
    print("NESTSAR M4-MOTIONPRESERVE-T16 — ACTIVATION / REPRESENTATION AUDIT")
    print("=" * 122)
    print(f"protocol              : {args.protocol}")
    print(f"checkpoint            : {ckpt}")
    print(f"checkpoint_epoch      : {int_scalar(payload.get('epoch', -1), -1)}")
    print(f"checkpoint_val_acc    : {scalar(payload.get('val_accuracy', float('nan')), float('nan')):.8f}")
    print(f"model_dim             : {model_dim}")
    print(f"spatial_dim           : {spatial_dim}")
    print(f"dropout               : {dropout}")
    print(f"validation_samples    : {len(Xva):,}")
    print(f"probe_train_samples   : {len(Xtr):,}")

    full_fn, logits_fn, feat_fn = make_functions(model, model_dim)

    logits, stream_logits, fusion, router, router_off_logits = run_full_eval(
        full_fn, params, Xva, args.batch_size
    )
    pred = np.argmax(logits, axis=-1)
    base_acc = accuracy(yva, pred)
    router_off_acc = accuracy(yva, np.argmax(router_off_logits, axis=-1))

    print("\nFULL / ROUTER COUNTERFACTUAL")
    print(f"  full EMA accuracy       : {pct(base_acc)}")
    print(f"  router-OFF accuracy     : {pct(router_off_acc)}")
    print(f"  router contribution     : {100*(base_acc-router_off_acc):+.3f} pp")

    stream_cf = fusion_counterfactuals(stream_logits, fusion, yva)
    print("\nSTREAM / FUSION COUNTERFACTUALS")
    for k, v in stream_cf.items():
        print(f"  {k:20s}: {pct(v)}  delta={100*(v-base_acc):+.3f} pp")

    variants = {}
    for mode in (
        "no_path", "no_disp", "pose_only", "motion_only",
        "disp_only", "path_only", "reverse_tokens", "shuffle_tokens",
    ):
        variants[mode] = run_variant_accuracy(
            logits_fn, params, Xva, yva, args.batch_size, mode, args.seed + 31
        )
    print("\nINPUT / TEMPORAL ABLATIONS")
    print(f"  {'baseline':18s}: {pct(base_acc)}")
    for k, v in variants.items():
        print(f"  {k:18s}: {pct(v)}  delta={100*(v-base_acc):+.3f} pp")

    # Probe validation subset chosen randomly from the already materialized full validation set.
    nprobe = min(args.probe_val, len(Xva))
    pidx = rng.choice(len(Xva), nprobe, replace=False) if nprobe < len(Xva) else np.arange(len(Xva))
    Xpv, ypv = Xva[pidx], yva[pidx]

    tr_stages, tr_delta = extract_features(feat_fn, params, Xtr, args.batch_size)
    va_stages, va_delta = extract_features(feat_fn, params, Xpv, args.batch_size)

    probes = {}
    spectra = {}
    stage_cos = {}
    print("\nFROZEN RIDGE LINEAR PROBES")
    for i, name in enumerate(("spatial", "frame_memory", "after_router", "chunk_memory", "descriptor")):
        probes[name] = linear_probe(
            flatten_stage(tr_stages[name]), ytr,
            flatten_stage(va_stages[name]), ypv,
        )
        spectra[name] = activation_spectrum(va_stages[name], args.rank_samples, args.seed + 100 + i)
        stage_cos[name] = activation_stream_cosines(va_stages[name])
        sp = spectra[name]
        print(
            f"  {name:14s}= {pct(probes[name])} | "
            f"eff_rank={sp['effective_rank']:.2f} rank_frac={sp['rank_fraction']:.3f} "
            f"stable={sp['stable_rank']:.2f}"
        )

    gains = {
        "spatial_to_frame_pp": 100 * (probes["frame_memory"] - probes["spatial"]),
        "frame_to_router_pp": 100 * (probes["after_router"] - probes["frame_memory"]),
        "router_to_chunk_pp": 100 * (probes["chunk_memory"] - probes["after_router"]),
        "chunk_to_descriptor_pp": 100 * (probes["descriptor"] - probes["chunk_memory"]),
    }
    print("\nSTAGE GAINS")
    for k, v in gains.items():
        print(f"  {k:24s}: {v:+.3f} pp")
    print(f"  router activation delta : mean={100*float(va_delta.mean()):.3f}% of frame-memory norm")

    print("\nACTIVATION CROSS-STREAM COSINE")
    for stage, pairs in stage_cos.items():
        txt = " | ".join(f"{k}={v:+.3f}" for k, v in pairs.items())
        print(f"  {stage:14s}: {txt}")

    router_mean = router.mean(axis=(0, 1))
    fusion_mean = fusion.mean(axis=0)
    router_ent = entropy_np(router, normalized=True)
    fusion_ent = entropy_np(fusion, normalized=True)
    print("\nROUTER / FUSION ACTIVATION HEALTH")
    print("  router mean : " + " ".join(f"{s}={router_mean[i]:.4f}" for i, s in enumerate(STREAMS)))
    print("  fusion mean : " + " ".join(f"{s}={fusion_mean[i]:.4f}" for i, s in enumerate(STREAMS)))
    print(f"  router normalized entropy mean : {float(router_ent.mean()):.4f}")
    print(f"  fusion normalized entropy mean : {float(fusion_ent.mean()):.4f}")

    classes = class_stats(yva, pred, fusion, router)
    hard = sorted(classes, key=lambda r: (r["accuracy"], -r["n"]))[:15]
    specialized = sorted(classes, key=lambda r: r["router_entropy_norm"])[:15]
    confusions = top_confusions(yva, pred, 20)

    print("\nHARDEST CLASSES (zero-based)")
    for r in hard:
        print(
            f"  class={r['class']:3d} acc={100*r['accuracy']:6.2f}% n={r['n']:4d} "
            f"router_dom={r['dominant_router_stream']} fusion_dom={r['dominant_fusion_stream']}"
        )

    print("\nMOST CLASS-SPECIALIZED ROUTER BEHAVIOR")
    for r in specialized[:10]:
        print(
            f"  class={r['class']:3d} H={r['router_entropy_norm']:.3f} "
            f"dom={r['dominant_router_stream']} weights={np.round(r['router_mean'], 3).tolist()}"
        )

    print("\nTOP CONFUSION PAIRS (zero-based)")
    for r in confusions[:15]:
        print(f"  true={r['true']:3d} -> pred={r['pred']:3d} count={r['count']:4d}")

    phase = token_phase_risk(Xva, yva)
    coverage = raw_path_coverage(by_id, val_ids, args.path_samples, args.seed + 71)

    class_acc_map = {r["class"]: r["accuracy"] for r in classes}
    cancel_rows = []
    for r in phase["per_class"]:
        rr = dict(r)
        rr["accuracy"] = class_acc_map.get(r["class"], float("nan"))
        cancel_rows.append(rr)
    cancel_sorted = sorted(cancel_rows, key=lambda r: r["cancellation"], reverse=True)
    corr_cancel_error = safe_corr(
        np.array([r["cancellation"] for r in cancel_rows]),
        np.array([1.0 - r["accuracy"] for r in cancel_rows]),
    )

    coverage_map = {r["class"]: r["coverage"] for r in coverage["per_class"]}
    common = [c for c in class_acc_map if c in coverage_map]
    corr_path_error = safe_corr(
        np.array([1.0 - coverage_map[c] for c in common]),
        np.array([1.0 - class_acc_map[c] for c in common]),
    )

    print("\nTEMPORAL INFORMATION AUDIT")
    print(f"  segment internal-path coverage mean   : {100*coverage['mean_coverage']:.2f}%")
    print(f"  segment internal-path coverage median : {100*coverage['median_coverage']:.2f}%")
    print(f"  within-segment cancellation mean      : {100*phase['global_cancellation']:.2f}%")
    print(f"  cancellation vs class-error corr      : {corr_cancel_error:+.4f}")
    print(f"  omitted-boundary-path vs error corr   : {corr_path_error:+.4f}")
    print("  highest cancellation / phase-risk classes:")
    for r in cancel_sorted[:12]:
        print(
            f"    class={r['class']:3d} cancel={100*r['cancellation']:5.1f}% "
            f"directionality={100*r['directionality']:5.1f}% acc={100*r['accuracy']:5.1f}%"
        )

    # Automatic, conservative summary: measurements rather than speculative fixes.
    weakest_gain = min(gains.items(), key=lambda kv: kv[1])
    strongest_gain = max(gains.items(), key=lambda kv: kv[1])
    most_damaging_input = min(variants.items(), key=lambda kv: kv[1])

    summary = {
        "full_accuracy": base_acc,
        "router_off_accuracy": router_off_acc,
        "router_contribution_pp": 100 * (base_acc - router_off_acc),
        "weakest_stage_gain": {"name": weakest_gain[0], "pp": weakest_gain[1]},
        "strongest_stage_gain": {"name": strongest_gain[0], "pp": strongest_gain[1]},
        "most_damaging_input_ablation": {
            "name": most_damaging_input[0],
            "accuracy": most_damaging_input[1],
            "delta_pp": 100 * (most_damaging_input[1] - base_acc),
        },
        "path_coverage_mean": coverage["mean_coverage"],
        "phase_cancellation_mean": phase["global_cancellation"],
        "phase_cancellation_vs_error_corr": corr_cancel_error,
    }

    print("\n" + "=" * 122)
    print("BOTTLENECK SUMMARY")
    print("=" * 122)
    print(f"FULL ACCURACY              : {pct(base_acc)}")
    print(f"ROUTER CONTRIBUTION        : {summary['router_contribution_pp']:+.3f} pp")
    print(f"WEAKEST STAGE GAIN         : {weakest_gain[0]} = {weakest_gain[1]:+.3f} pp")
    print(f"STRONGEST STAGE GAIN       : {strongest_gain[0]} = {strongest_gain[1]:+.3f} pp")
    print(
        f"MOST DAMAGING ABLATION     : {most_damaging_input[0]} -> {pct(most_damaging_input[1])} "
        f"({100*(most_damaging_input[1]-base_acc):+.3f} pp)"
    )
    print(f"SEGMENT PATH COVERAGE      : {100*coverage['mean_coverage']:.2f}%")
    print(f"PHASE CANCELLATION         : {100*phase['global_cancellation']:.2f}%")
    print(f"CANCELLATION↔ERROR CORR    : {corr_cancel_error:+.4f}")

    report = {
        "metadata": {
            "model": payload.get("model"),
            "protocol": args.protocol,
            "checkpoint": str(ckpt),
            "checkpoint_epoch": int_scalar(payload.get("epoch", -1), -1),
            "checkpoint_val_accuracy": scalar(payload.get("val_accuracy", float("nan")), float("nan")),
            "model_dim": model_dim,
            "spatial_dim": spatial_dim,
            "dropout": dropout,
            "validation_samples": len(Xva),
            "probe_train_samples": len(Xtr),
            "probe_val_samples": len(Xpv),
        },
        "accuracy": {
            "full": base_acc,
            "router_off": router_off_acc,
            "stream_and_fusion_counterfactuals": stream_cf,
            "input_temporal_ablations": variants,
        },
        "linear_probes": probes,
        "stage_gains_pp": gains,
        "activation_spectra": spectra,
        "activation_cross_stream_cosine": stage_cos,
        "router": {
            "mean_weights": router_mean.tolist(),
            "normalized_entropy_mean": float(router_ent.mean()),
            "delta_relative_norm_mean": float(va_delta.mean()),
        },
        "fusion": {
            "mean_weights": fusion_mean.tolist(),
            "normalized_entropy_mean": float(fusion_ent.mean()),
        },
        "classes": classes,
        "hardest_classes": hard,
        "top_confusions": confusions,
        "temporal_information": {
            "path_coverage": coverage,
            "phase_risk": phase,
            "cancellation_vs_class_error_corr": corr_cancel_error,
            "omitted_boundary_path_vs_class_error_corr": corr_path_error,
            "highest_cancellation_classes": cancel_sorted[:20],
        },
        "summary": summary,
    }

    out = Path(args.out) if args.out else ckpt.with_name("activation_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=True))
    print(f"REPORT: {out}")
    print("=" * 122)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""Activation/representation audit for M4-MotionPreserve-Phase-T16.

Uses a saved EMA checkpoint and the real NTU120 split. No model weights are
modified. The audit measures:

* full EMA accuracy and exact router-off counterfactual
* J/B/JM/BM single-stream and leave-one-stream-out fusion accuracy
* phase-specific token ablations
* reverse/shuffled token sensitivity
* frozen linear probes at spatial/frame/router/chunk/descriptor stages
* activation effective rank and cross-stream cosine similarity
* router/fusion entropy and mean stream preferences
* top confusion pairs and hardest classes

The phase-specific ablations are the key addition versus MotionPreserve:
  no_full_disp, no_phase_a, no_phase_b, no_phase_ab, no_path,
  pose_only, motion_only, full_disp_only, phase_only, path_only.
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

from experiments.m4_motionpreserve_phase_t16 import train_phase_t16_tpu as phase

# Importing phase installs the phase-aware overrides into the validated parent
# trainer. Use parent helpers for dataset/splits and the explicit phase model for
# model construction.
base = phase.base

STREAMS = ("J", "B", "JM", "BM")
EPS = 1e-12


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


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def entropy_np(w: np.ndarray, axis: int = -1, normalized: bool = False) -> np.ndarray:
    w = np.asarray(w, np.float64)
    h = -np.sum(w * np.log(np.maximum(w, 1e-12)), axis=axis)
    if normalized:
        h = h / math.log(w.shape[axis])
    return h


def resolve_ids(annotations, split: Mapping[str, Any], protocol: str):
    tk, vk = base.resolve_split(split, protocol)
    by_id = {
        base.sample_id(a, i): a
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


def choose_ids(ids: Sequence[str], n: int, seed: int) -> list[str]:
    ids = list(ids)
    if n <= 0 or n >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    ii = rng.choice(len(ids), size=n, replace=False)
    return [ids[int(i)] for i in ii]


def materialize(by_id, ids: Sequence[str], desc: str):
    X = np.empty((len(ids), phase.FRAMES, phase.FEATURES), np.float32)
    y = np.empty((len(ids),), np.int32)
    for i, sid in enumerate(tqdm(ids, desc=desc, mininterval=0.5)):
        a = by_id[sid]
        X[i] = phase.preprocess_keypoints(base.annotation_keypoints(a), "segment")
        y[i] = base.annotation_label(a)
    return X, y


def dense_apply(p: Mapping[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    y = x @ p["kernel"]
    if "bias" in p:
        y = y + p["bias"]
    return y


def build_model(payload: Mapping[str, Any]):
    cfg = payload.get("config", {})
    if not isinstance(cfg, Mapping):
        cfg = {}
    spatial_dim = int_scalar(cfg.get("spatial_dim", 24), 24)
    model_dim = int_scalar(cfg.get("model_dim", 112), 112)
    dropout = scalar(cfg.get("dropout", 0.10), 0.10)
    model = phase.M4MotionPreservePhaseT16(spatial_dim, model_dim, dropout)
    return model, spatial_dim, model_dim, dropout


def make_functions(model, model_dim: int):
    @jax.jit
    def full_eval(params, xb):
        out = model.apply({"params": params}, xb, training=False)

        # Router-off counterfactual: use the learned frame-memory activations but
        # bypass the cross-stream residual and run the SAME learned descriptor,
        # classifier and fusion heads.
        ro_descs = []
        ro_logits = []
        for i in range(4):
            _, desc = base.DescriptorHead(model_dim, 0.0).apply(
                {"params": params[f"descriptor_{i}"]},
                out["frame_stack"][:, :, i],
                training=False,
            )
            ro_descs.append(desc)
            ro_logits.append(dense_apply(params[f"classifier_{i}"], desc))

        ro_descs = jnp.stack(ro_descs, axis=1)
        ro_sl = jnp.stack(ro_logits, axis=1)
        ctrl = dense_apply(
            params["fusion_controller"],
            ro_descs.reshape(ro_descs.shape[0], -1),
        )
        ro_fusion = jax.nn.softmax(
            params["fusion_prior"][None, :] + 0.15 * jnp.tanh(ctrl),
            axis=-1,
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
        spatial = out["spatial_stack"].mean(axis=1)          # [B,4,D]
        frame = out["frame_stack"].mean(axis=1)              # [B,4,D]
        mixed = out["mixed_frame_stack"].mean(axis=1)        # [B,4,D]
        chunk = out["chunk_states"].mean(axis=2)             # [B,4,D]
        desc = out["descriptors"]                             # [B,4,D]
        num = jnp.linalg.norm(
            out["mixed_frame_stack"] - out["frame_stack"], axis=-1
        )
        den = jnp.maximum(jnp.linalg.norm(out["frame_stack"], axis=-1), 1e-8)
        router_delta_ratio = jnp.mean(num / den, axis=(1, 2))
        return spatial, frame, mixed, chunk, desc, router_delta_ratio

    return full_eval, logits_only, feature_eval


def run_full_eval(fn, params, X: np.ndarray, batch_size: int):
    bags = [[] for _ in range(5)]
    for s in tqdm(
        range(0, len(X), batch_size),
        desc="full validation inference",
        mininterval=0.5,
    ):
        vals = fn(params, jnp.asarray(X[s:s + batch_size]))
        vals = [np.asarray(jax.device_get(v)) for v in vals]
        for b, v in zip(bags, vals):
            b.append(v)
    return tuple(np.concatenate(b, axis=0) for b in bags)


def transform_batch(xb: np.ndarray, mode: str, shuffle_perm: np.ndarray) -> np.ndarray:
    z = xb.copy().reshape(
        len(xb), phase.FRAMES, phase.PERSONS, phase.JOINTS, phase.TOKEN_CHANNELS
    )

    # Channel layout:
    #  0:3   pose
    #  3:6   full signed displacement
    #  6:9   phase A signed displacement
    #  9:12  phase B signed displacement
    # 12:15  absolute path
    if mode == "baseline":
        pass
    elif mode == "no_pose":
        z[..., 0:3] = 0
    elif mode == "no_full_disp":
        z[..., 3:6] = 0
    elif mode == "no_phase_a":
        z[..., 6:9] = 0
    elif mode == "no_phase_b":
        z[..., 9:12] = 0
    elif mode == "no_phase_ab":
        z[..., 6:12] = 0
    elif mode == "no_path":
        z[..., 12:15] = 0
    elif mode == "pose_only":
        z[..., 3:15] = 0
    elif mode == "motion_only":
        z[..., 0:3] = 0
    elif mode == "full_disp_only":
        z[..., 0:3] = 0
        z[..., 6:15] = 0
    elif mode == "phase_only":
        z[..., 0:6] = 0
        z[..., 12:15] = 0
    elif mode == "path_only":
        z[..., 0:12] = 0
    elif mode == "pose_plus_full":
        z[..., 6:15] = 0
    elif mode == "pose_plus_phase":
        z[..., 3:6] = 0
        z[..., 12:15] = 0
    elif mode == "pose_plus_path":
        z[..., 3:12] = 0
    elif mode == "reverse_tokens":
        z = z[:, ::-1]
    elif mode == "shuffle_tokens":
        z = z[:, shuffle_perm]
    else:
        raise ValueError(mode)

    return z.reshape(len(xb), phase.FRAMES, phase.FEATURES)


def variant_accuracy(fn, params, X, y, batch_size: int, mode: str, seed: int) -> float:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(phase.FRAMES)
    correct = total = 0
    for s in tqdm(
        range(0, len(X), batch_size),
        desc=f"ablation {mode}",
        leave=False,
        mininterval=0.5,
    ):
        xb = transform_batch(X[s:s + batch_size], mode, perm)
        logits = np.asarray(jax.device_get(fn(params, jnp.asarray(xb))))
        pred = np.argmax(logits, axis=-1)
        yy = y[s:s + len(pred)]
        correct += int(np.sum(pred == yy))
        total += len(pred)
    return correct / max(total, 1)


def fusion_counterfactuals(stream_logits, fusion, y):
    out = {}
    for i, name in enumerate(STREAMS):
        out[f"{name}_only"] = accuracy(y, np.argmax(stream_logits[:, i], axis=-1))

    out["uniform_fusion"] = accuracy(
        y, np.argmax(np.mean(stream_logits, axis=1), axis=-1)
    )

    def subset(indices):
        w = fusion[:, indices]
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        lg = np.sum(w[..., None] * stream_logits[:, indices, :], axis=1)
        return accuracy(y, np.argmax(lg, axis=-1))

    out["pose_JB"] = subset([0, 1])
    out["motion_JM_BM"] = subset([2, 3])
    for i, name in enumerate(STREAMS):
        out[f"minus_{name}"] = subset([j for j in range(4) if j != i])
    return out


def extract_features(fn, params, X: np.ndarray, batch_size: int):
    names = ("spatial", "frame_memory", "after_router", "chunk_memory", "descriptor")
    bags = {k: [] for k in names}
    delta = []
    for s in tqdm(
        range(0, len(X), batch_size),
        desc="extract activations",
        leave=False,
        mininterval=0.5,
    ):
        vals = fn(params, jnp.asarray(X[s:s + batch_size]))
        vals = [np.asarray(jax.device_get(v)) for v in vals]
        for k, v in zip(names, vals[:5]):
            bags[k].append(v.astype(np.float32, copy=False))
        delta.append(vals[5].astype(np.float32, copy=False))
    stages = {k: np.concatenate(v, axis=0) for k, v in bags.items()}
    return stages, np.concatenate(delta, axis=0)


def linear_probe(train_x, train_y, val_x, val_y) -> float:
    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    clf.fit(train_x.reshape(len(train_x), -1), train_y)
    return float(clf.score(val_x.reshape(len(val_x), -1), val_y))


def activation_spectrum(x: np.ndarray, max_samples: int, seed: int):
    X = np.asarray(x, np.float32).reshape(len(x), -1)
    if len(X) > max_samples:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), max_samples, replace=False)]
    X = X.astype(np.float64, copy=False)
    X -= X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
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


def stream_cosines(x: np.ndarray):
    x = np.asarray(x, np.float64)
    out = {}
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = x[:, i], x[:, j]
            den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            c = np.sum(a * b, axis=1) / np.maximum(den, 1e-12)
            out[f"{STREAMS[i]}-{STREAMS[j]}"] = float(np.mean(c))
    return out


def class_stats(y, pred, fusion, router):
    rows = []
    for c in range(phase.NUM_CLASSES):
        m = y == c
        if not np.any(m):
            continue
        fw = fusion[m].mean(axis=0)
        rw = router[m].mean(axis=(0, 1))
        rows.append({
            "class": c,
            "n": int(m.sum()),
            "accuracy": float(np.mean(pred[m] == c)),
            "fusion_mean": fw.tolist(),
            "router_mean": rw.tolist(),
            "dominant_fusion_stream": STREAMS[int(np.argmax(fw))],
            "dominant_router_stream": STREAMS[int(np.argmax(rw))],
        })
    return rows


def top_confusions(y, pred, k: int = 20):
    cm = np.zeros((phase.NUM_CLASSES, phase.NUM_CLASSES), np.int64)
    np.add.at(cm, (y, pred), 1)
    rows = []
    for a in range(phase.NUM_CLASSES):
        for b in range(phase.NUM_CLASSES):
            if a != b and cm[a, b] > 0:
                rows.append((int(cm[a, b]), a, b))
    rows.sort(reverse=True)
    return [
        {"count": n, "true": a, "pred": b}
        for n, a, b in rows[:k]
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--protocol", choices=["xsub", "xset"], required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--probe-train", type=int, default=12000)
    ap.add_argument("--probe-val", type=int, default=8000)
    ap.add_argument("--rank-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    payload = serialization.msgpack_restore(ckpt.read_bytes())
    if "ema_params" not in payload:
        raise KeyError(f"Checkpoint has no ema_params; keys={list(payload)}")
    params = payload["ema_params"]
    model, spatial_dim, model_dim, dropout = build_model(payload)

    print("=" * 122)
    print("NESTSAR M4-MOTIONPRESERVE-PHASE-T16 — ACTIVATION AUDIT")
    print("=" * 122)
    print(f"protocol              : {args.protocol}")
    print(f"checkpoint epoch      : {int_scalar(payload.get('epoch', -1), -1)}")
    print(f"checkpoint val_acc    : {100*scalar(payload.get('val_accuracy', float('nan')), float('nan')):.3f}%")
    print(f"token channels        : {phase.TOKEN_CHANNELS}")
    print(f"features/token        : {phase.FEATURES}")
    print(f"model dims            : spatial={spatial_dim} model={model_dim} dropout={dropout}")

    dataset = base.find_dataset(args.dataset)
    print(f"dataset               : {dataset}")
    annotations, split = base.load_ntu(dataset)
    by_id, train_ids, val_ids = resolve_ids(annotations, split, args.protocol)

    # Full validation is used for exact accuracy/ablations.
    Xva, yva = materialize(
        by_id, val_ids, f"{args.protocol.upper()} phase val"
    )

    full_eval, logits_only, feature_eval = make_functions(model, model_dim)
    logits, stream_logits, fusion, router, router_off_logits = run_full_eval(
        full_eval, params, Xva, args.batch_size
    )
    pred = np.argmax(logits, axis=-1)
    full_acc = accuracy(yva, pred)
    ro_acc = accuracy(yva, np.argmax(router_off_logits, axis=-1))

    print("\nFULL / ROUTER COUNTERFACTUAL")
    print(f"  full EMA accuracy       : {100*full_acc:.3f}%")
    print(f"  router-OFF accuracy     : {100*ro_acc:.3f}%")
    print(f"  router contribution     : {100*(full_acc-ro_acc):+.3f} pp")

    stream_cf = fusion_counterfactuals(stream_logits, fusion, yva)
    print("\nSTREAM / FUSION COUNTERFACTUALS")
    for k, v in stream_cf.items():
        print(f"  {k:18s}: {100*v:7.3f}%  ({100*(v-full_acc):+7.3f} pp)")

    modes = [
        "no_pose",
        "no_full_disp",
        "no_phase_a",
        "no_phase_b",
        "no_phase_ab",
        "no_path",
        "pose_only",
        "motion_only",
        "full_disp_only",
        "phase_only",
        "path_only",
        "pose_plus_full",
        "pose_plus_phase",
        "pose_plus_path",
        "reverse_tokens",
        "shuffle_tokens",
    ]
    ablations = {}
    print("\nPHASE / INPUT ABLATIONS")
    for mode in modes:
        a = variant_accuracy(
            logits_only, params, Xva, yva, args.batch_size, mode, args.seed
        )
        ablations[mode] = a
        print(f"  {mode:18s}: {100*a:7.3f}%  ({100*(a-full_acc):+7.3f} pp)")

    print("\nROUTER / FUSION BEHAVIOR")
    mean_fusion = fusion.mean(axis=0)
    mean_router = router.mean(axis=(0, 1))
    print("  mean fusion :", {s: round(float(v), 5) for s, v in zip(STREAMS, mean_fusion)})
    print("  mean router :", {s: round(float(v), 5) for s, v in zip(STREAMS, mean_router)})
    print(f"  fusion entropy normalized : {float(entropy_np(fusion, normalized=True).mean()):.4f}")
    print(f"  router entropy normalized : {float(entropy_np(router, normalized=True).mean()):.4f}")

    # Probe subsets.
    probe_train_ids = choose_ids(train_ids, args.probe_train, args.seed + 11)
    probe_val_ids = choose_ids(val_ids, args.probe_val, args.seed + 17)
    Xtr_p, ytr_p = materialize(
        by_id, probe_train_ids, f"{args.protocol.upper()} probe train"
    )
    Xva_p, yva_p = materialize(
        by_id, probe_val_ids, f"{args.protocol.upper()} probe val"
    )
    tr_stages, _ = extract_features(feature_eval, params, Xtr_p, args.batch_size)
    va_stages, router_delta = extract_features(feature_eval, params, Xva_p, args.batch_size)

    probe_scores = {}
    print("\nFROZEN RIDGE LINEAR PROBES")
    for name in ("spatial", "frame_memory", "after_router", "chunk_memory", "descriptor"):
        sc = linear_probe(tr_stages[name], ytr_p, va_stages[name], yva_p)
        probe_scores[name] = sc
        print(f"  {name:14s}: {100*sc:7.3f}%")

    stage_gains = {
        "spatial_to_frame_pp": 100*(probe_scores["frame_memory"]-probe_scores["spatial"]),
        "frame_to_router_pp": 100*(probe_scores["after_router"]-probe_scores["frame_memory"]),
        "router_to_chunk_pp": 100*(probe_scores["chunk_memory"]-probe_scores["after_router"]),
        "chunk_to_descriptor_pp": 100*(probe_scores["descriptor"]-probe_scores["chunk_memory"]),
    }
    print("\nSTAGE GAINS")
    for k, v in stage_gains.items():
        print(f"  {k:25s}: {v:+7.3f} pp")
    print(f"  mean router activation delta ratio: {100*float(np.mean(router_delta)):.3f}%")

    spectra = {}
    stream_sim = {}
    print("\nACTIVATION RANK / STREAM SIMILARITY")
    for name in ("spatial", "frame_memory", "after_router", "chunk_memory", "descriptor"):
        spectra[name] = activation_spectrum(
            va_stages[name], args.rank_samples, args.seed
        )
        stream_sim[name] = stream_cosines(va_stages[name])
        sp = spectra[name]
        print(
            f"  {name:14s}: eff_rank={sp['effective_rank']:.2f} "
            f"rank_frac={sp['rank_fraction']:.4f} stable={sp['stable_rank']:.2f}"
        )
        print("    stream cosine:", {k: round(v, 4) for k, v in stream_sim[name].items()})

    classes = class_stats(yva, pred, fusion, router)
    hardest = sorted(classes, key=lambda r: r["accuracy"])[:15]
    conf = top_confusions(yva, pred, 20)

    print("\nHARDEST CLASSES")
    for r in hardest:
        print(
            f"  class={r['class']:3d} acc={100*r['accuracy']:6.2f}% n={r['n']:4d} "
            f"fusion={r['dominant_fusion_stream']} router={r['dominant_router_stream']}"
        )

    print("\nTOP CONFUSIONS")
    for r in conf:
        print(f"  true={r['true']:3d} -> pred={r['pred']:3d}  n={r['count']}")

    # Identify the most damaging *removal* ablation separately from the *_only
    # diagnostics; this avoids misinterpreting path_only as 'path is important'.
    removal_modes = [
        "no_pose", "no_full_disp", "no_phase_a", "no_phase_b", "no_phase_ab", "no_path"
    ]
    most_damaging_removal = min(removal_modes, key=lambda k: ablations[k])
    weakest_stage = min(stage_gains, key=lambda k: stage_gains[k])
    strongest_stage = max(stage_gains, key=lambda k: stage_gains[k])

    print("\n" + "=" * 122)
    print("BOTTLENECK SUMMARY")
    print("=" * 122)
    print(f"FULL ACCURACY              : {100*full_acc:.3f}%")
    print(f"ROUTER CONTRIBUTION        : {100*(full_acc-ro_acc):+.3f} pp")
    print(f"WEAKEST STAGE GAIN         : {weakest_stage} = {stage_gains[weakest_stage]:+.3f} pp")
    print(f"STRONGEST STAGE GAIN       : {strongest_stage} = {stage_gains[strongest_stage]:+.3f} pp")
    print(
        f"MOST DAMAGING REMOVAL      : {most_damaging_removal} -> "
        f"{100*ablations[most_damaging_removal]:.3f}% "
        f"({100*(ablations[most_damaging_removal]-full_acc):+.3f} pp)"
    )
    print(
        f"REMOVE BOTH PHASE CHANNELS : {100*ablations['no_phase_ab']:.3f}% "
        f"({100*(ablations['no_phase_ab']-full_acc):+.3f} pp)"
    )
    print(
        f"REMOVE PHASE-A             : {100*ablations['no_phase_a']:.3f}% "
        f"({100*(ablations['no_phase_a']-full_acc):+.3f} pp)"
    )
    print(
        f"REMOVE PHASE-B             : {100*ablations['no_phase_b']:.3f}% "
        f"({100*(ablations['no_phase_b']-full_acc):+.3f} pp)"
    )
    print(f"MEAN ROUTER DELTA RATIO    : {100*float(np.mean(router_delta)):.3f}%")

    report = {
        "metadata": {
            "protocol": args.protocol,
            "checkpoint": str(ckpt),
            "epoch": int_scalar(payload.get("epoch", -1), -1),
            "checkpoint_val_accuracy": scalar(payload.get("val_accuracy", float("nan")), float("nan")),
            "token_channels": phase.TOKEN_CHANNELS,
            "features_per_token": phase.FEATURES,
        },
        "full_accuracy": full_acc,
        "router_off_accuracy": ro_acc,
        "router_contribution_pp": 100*(full_acc-ro_acc),
        "stream_counterfactuals": stream_cf,
        "ablations": ablations,
        "mean_fusion": {s: float(v) for s, v in zip(STREAMS, mean_fusion)},
        "mean_router": {s: float(v) for s, v in zip(STREAMS, mean_router)},
        "fusion_entropy_norm": float(entropy_np(fusion, normalized=True).mean()),
        "router_entropy_norm": float(entropy_np(router, normalized=True).mean()),
        "probe_scores": probe_scores,
        "stage_gains_pp": stage_gains,
        "router_delta_ratio_mean": float(np.mean(router_delta)),
        "activation_spectra": spectra,
        "stream_activation_cosines": stream_sim,
        "hardest_classes": hardest,
        "top_confusions": conf,
        "summary": {
            "most_damaging_removal": most_damaging_removal,
            "weakest_stage": weakest_stage,
            "strongest_stage": strongest_stage,
        },
    }

    out = Path(args.out) if args.out else ckpt.parent / "phase_activation_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"REPORT: {out}")
    print("=" * 122)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

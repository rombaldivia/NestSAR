#!/usr/bin/env python3
from __future__ import annotations

"""Representation bottleneck audit for a trained M4-MotionLite-T16 checkpoint.

No model training is performed here. The audit evaluates a saved EMA checkpoint and
asks where discriminative information is being lost:

- full / per-stream / stream-ablation accuracy;
- temporal-order sensitivity (reverse, shuffle, static-repeat);
- multi-scale motion component ablations (Delta1 / Delta2 / Delta4 / acceleration);
- cross-stream router usefulness and router/fusion entropy;
- motion-aware selector vs uniform selector;
- linear probes at spatial -> router -> frame-memory -> final-descriptor stages;
- temporal path retention of the T16 selector;
- largest validation confusion pairs.

Designed for the same Kaggle TPU v5e-8 / JAX>=0.10 environment as the trainer.
"""

import argparse
import json
import math
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.m4_motionlite_t16.jax10_compat import install

install()

from experiments.m4_motionlite_t16 import train_m4_motionlite_t16_tpu as tr

jax = tr.jax
jnp = tr.jnp
nn = tr.nn
serialization = tr.serialization

STREAM_NAMES = ("J", "B", "JM", "BM")
MOTION_NAMES = ("D1", "D2", "D4", "ACC")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class MotionLiteAuditModel(nn.Module):
    """Same parameter tree as M4MotionLiteT16, with runtime-only audit controls."""

    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        stream_mask: jnp.ndarray,
        motion_mask: jnp.ndarray,
        router_scale: jnp.ndarray,
    ) -> Mapping[str, jnp.ndarray]:
        sk = x.reshape(x.shape[0], tr.FRAMES, 2, 25, 3)
        root = sk[:, :, :, 0:1, :]
        joint = sk - root
        parents = jnp.asarray(tr.PARENTS)
        bone = joint - jnp.take(joint, parents, axis=3)

        j_d1 = tr.lag_diff(joint, 1)
        j_d2 = tr.lag_diff(joint, 2) / 2.0
        j_d4 = tr.lag_diff(joint, 4) / 4.0
        j_acc = tr.lag_diff(j_d1, 1)
        joint_motion = jnp.concatenate(
            [
                j_d1 * motion_mask[0],
                j_d2 * motion_mask[1],
                j_d4 * motion_mask[2],
                j_acc * motion_mask[3],
            ],
            axis=-1,
        )

        b_d1 = tr.lag_diff(bone, 1)
        b_d2 = tr.lag_diff(bone, 2) / 2.0
        b_d4 = tr.lag_diff(bone, 4) / 4.0
        b_acc = tr.lag_diff(b_d1, 1)
        bone_motion = jnp.concatenate(
            [
                b_d1 * motion_mask[0],
                b_d2 * motion_mask[1],
                b_d4 * motion_mask[2],
                b_acc * motion_mask[3],
            ],
            axis=-1,
        )

        raw_streams = (joint, bone, joint_motion, bone_motion)
        encoded = []
        for i, s in enumerate(raw_streams):
            encoded.append(
                tr.SpatialEncoder(
                    self.spatial_dim,
                    self.model_dim,
                    self.dropout,
                    name=f"spatial_{i}",
                )(s, False)
            )

        streams_pre = jnp.stack(encoded, axis=2)
        sm = stream_mask[None, None, :, None]
        streams_pre = streams_pre * sm

        routed, router_weights = tr.CrossStreamRouter(
            self.model_dim, name="cross_stream"
        )(streams_pre)
        rs = jnp.asarray(router_scale, streams_pre.dtype)
        streams = (1.0 - rs) * streams_pre + rs * routed
        streams = streams * sm

        stream_logits = []
        descriptors = []
        frame_memories = []
        for i in range(tr.NUM_STREAMS):
            h, desc = tr.TemporalHierarchy(
                self.model_dim,
                self.dropout,
                name=f"temporal_{i}",
            )(streams[:, :, i], False)
            frame_memories.append(h)
            descriptors.append(desc)
            stream_logits.append(
                nn.Dense(tr.NUM_CLASSES, name=f"classifier_{i}")(desc)
            )

        frame_mem = jnp.stack(frame_memories, axis=2)  # B,T,S,D
        descs = jnp.stack(descriptors, axis=1)         # B,S,D
        sl = jnp.stack(stream_logits, axis=1)          # B,S,C

        prior = self.param("fusion_prior", nn.initializers.zeros, (tr.NUM_STREAMS,))
        controller = nn.Dense(tr.NUM_STREAMS, name="fusion_controller")(
            descs.reshape(descs.shape[0], -1)
        )
        fusion = jax.nn.softmax(
            prior[None, :] + 0.15 * jnp.tanh(controller), axis=-1
        )
        fusion = fusion * stream_mask[None, :]
        fusion = fusion / jnp.maximum(fusion.sum(axis=-1, keepdims=True), 1e-8)
        logits = jnp.einsum("bs,bsc->bc", fusion, sl)

        return {
            "logits": logits,
            "stream_logits": sl,
            "fusion_weights": fusion,
            "router_weights": router_weights,
            "spatial_repr": streams_pre.mean(axis=1).reshape(x.shape[0], -1),
            "router_repr": streams.mean(axis=1).reshape(x.shape[0], -1),
            "frame_repr": frame_mem.mean(axis=1).reshape(x.shape[0], -1),
            "descriptor_repr": descs.reshape(x.shape[0], -1),
        }


def _decode_checkpoint(path: Path):
    payload = serialization.msgpack_restore(path.read_bytes())
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint {path} is not a mapping payload")
    params = payload.get("ema_params", payload.get("params"))
    if params is None:
        raise KeyError(f"Checkpoint {path} has no ema_params/params")
    cfg = payload.get("config", {}) or {}
    return payload, params, cfg


def _by_id(annotations):
    return {
        tr.sample_id(a, i): a
        for i, a in enumerate(annotations)
        if isinstance(a, Mapping)
    }


def _split_ids(annotations, split, protocol: str, which: str):
    tk, vk = tr.resolve_split(split, protocol)
    key = tk if which == "train" else vk
    by_id = _by_id(annotations)
    ids = [str(v) for v in split[key] if str(v) in by_id]
    if not ids:
        raise RuntimeError(f"No matched {protocol}/{which} IDs")
    return by_id, ids


def materialize(
    annotations,
    split,
    protocol: str,
    which: str,
    selector: str,
    max_samples: int = 0,
    seed: int = 0,
):
    by_id, ids = _split_ids(annotations, split, protocol, which)
    if max_samples and max_samples < len(ids):
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(len(ids), size=max_samples, replace=False))
        ids = [ids[int(i)] for i in take]

    X = np.empty((len(ids), tr.FRAMES, tr.FEATURES), np.float32)
    y = np.empty((len(ids),), np.int32)
    from tqdm.auto import tqdm

    for i, sid in enumerate(tqdm(ids, desc=f"{protocol.upper()} {which} {selector}", mininterval=0.5)):
        a = by_id[sid]
        X[i] = tr.preprocess_keypoints(tr.annotation_keypoints(a), selector)
        y[i] = tr.annotation_label(a)
    return X, y, ids


def _pad_batch(x: np.ndarray, global_batch: int):
    n = len(x)
    if n == global_batch:
        return x, n
    out = np.zeros((global_batch, *x.shape[1:]), dtype=x.dtype)
    out[:n] = x
    return out, n


def _entropy(p: np.ndarray, axis=-1):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def _accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=-1) == y))


def _confusion(y: np.ndarray, pred: np.ndarray):
    cm = np.zeros((tr.NUM_CLASSES, tr.NUM_CLASSES), np.int64)
    np.add.at(cm, (y.astype(np.int64), pred.astype(np.int64)), 1)
    return cm


def _top_confusions(cm: np.ndarray, n: int = 15):
    work = cm.copy()
    np.fill_diagonal(work, 0)
    flat = np.argsort(work.ravel())[::-1]
    rows = []
    for idx in flat:
        count = int(work.ravel()[idx])
        if count <= 0 or len(rows) >= n:
            break
        a, b = np.unravel_index(idx, work.shape)
        support = int(cm[a].sum())
        rows.append(
            {
                "true": int(a),
                "pred": int(b),
                "count": count,
                "true_support": support,
                "pair_rate_within_true": float(count / max(support, 1)),
            }
        )
    return rows


def temporal_path_retention(
    annotations,
    split,
    protocol: str,
    max_samples: int,
    seed: int,
):
    by_id, ids = _split_ids(annotations, split, protocol, "val")
    if max_samples and max_samples < len(ids):
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(len(ids), size=max_samples, replace=False))
        ids = [ids[int(i)] for i in take]

    vals_motion = []
    vals_uniform = []
    per_class_motion = defaultdict(list)
    per_class_uniform = defaultdict(list)

    for sid in ids:
        a = by_id[sid]
        x = tr.canonicalize_raw(tr.annotation_keypoints(a))
        x = x - x[:, :, 0:1, :]
        if len(x) <= 1:
            continue
        full_step = np.linalg.norm(x[1:] - x[:-1], axis=-1)
        full_path = float(full_step.sum())
        if full_path <= 1e-8:
            continue

        im = tr.hybrid_motion_indices(x, tr.FRAMES)
        iu = tr.uniform_indices(len(x), tr.FRAMES)
        xm = x[im]
        xu = x[iu]
        motion_path = float(np.linalg.norm(xm[1:] - xm[:-1], axis=-1).sum())
        uniform_path = float(np.linalg.norm(xu[1:] - xu[:-1], axis=-1).sum())
        rm = min(1.0, max(0.0, motion_path / full_path))
        ru = min(1.0, max(0.0, uniform_path / full_path))
        label = int(tr.annotation_label(a))
        vals_motion.append(rm)
        vals_uniform.append(ru)
        per_class_motion[label].append(rm)
        per_class_uniform[label].append(ru)

    class_rows = []
    for c in sorted(per_class_motion):
        class_rows.append(
            {
                "class": int(c),
                "motion_retention": float(np.mean(per_class_motion[c])),
                "uniform_retention": float(np.mean(per_class_uniform[c])),
                "n": len(per_class_motion[c]),
            }
        )
    class_rows.sort(key=lambda z: z["motion_retention"])
    return {
        "n": len(vals_motion),
        "motion_mean": float(np.mean(vals_motion)) if vals_motion else float("nan"),
        "uniform_mean": float(np.mean(vals_uniform)) if vals_uniform else float("nan"),
        "worst_motion_classes": class_rows[:15],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=("xsub", "xset"), default="xsub")
    ap.add_argument("--checkpoint", default="auto")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--probe-train", type=int, default=12000)
    ap.add_argument("--probe-val", type=int, default=8000)
    ap.add_argument("--path-retention-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--out", default="/kaggle/working/NestSAR_M4_MotionLite_T16_TPU/representation_audit_xsub.json")
    args = ap.parse_args()

    if jax.default_backend() != "tpu":
        raise RuntimeError(f"TPU required for this audit; backend={jax.default_backend()}")
    devices = list(jax.local_devices())
    ndev = len(devices)
    if ndev != 8:
        raise RuntimeError(f"Expected 8 local TPU devices, got {ndev}: {devices}")
    if args.batch_size % ndev:
        raise ValueError("--batch-size must be divisible by 8")

    ckpt = (
        Path(args.checkpoint)
        if args.checkpoint != "auto"
        else Path("/kaggle/working/NestSAR_M4_MotionLite_T16_TPU") / args.protocol / "best.msgpack"
    )
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}. Run this audit after the training process has saved best.msgpack."
        )

    payload, params, cfg = _decode_checkpoint(ckpt)
    spatial_dim = int(cfg.get("spatial_dim", 24))
    model_dim = int(cfg.get("model_dim", 112))
    dropout = float(cfg.get("dropout", 0.10))

    log("=" * 118)
    log("NESTSAR M4-MOTIONLITE-T16 — REPRESENTATION BOTTLENECK AUDIT")
    log("=" * 118)
    log(f"protocol={args.protocol.upper()} checkpoint={ckpt}")
    log(f"checkpoint epoch={payload.get('epoch')} val_accuracy={payload.get('val_accuracy')}")
    log(f"JAX={jax.__version__} backend=tpu devices={ndev}")
    log(f"model D={model_dim} spatial_D={spatial_dim} params={tr.count_params(params):,}")

    dataset = tr.find_dataset(args.dataset)
    annotations, split = tr.load_ntu(dataset)
    Xv, yv, _ = materialize(
        annotations, split, args.protocol, "val", "motion", 0, args.seed
    )

    audit_model = MotionLiteAuditModel(spatial_dim, model_dim, dropout)
    params_rep = jax.device_put_replicated(params, devices)

    @partial(jax.pmap, axis_name="d")
    def p_metrics(p, xb, stream_mask, motion_mask, router_scale):
        out = audit_model.apply(
            {"params": p}, xb, stream_mask, motion_mask, router_scale
        )
        return (
            out["logits"],
            out["stream_logits"],
            out["fusion_weights"],
            out["router_weights"],
        )

    @partial(jax.pmap, axis_name="d")
    def p_features(p, xb):
        ones = jnp.ones((4,), jnp.float32)
        out = audit_model.apply({"params": p}, xb, ones, ones, jnp.float32(1.0))
        return (
            out["spatial_repr"],
            out["router_repr"],
            out["frame_repr"],
            out["descriptor_repr"],
        )

    def evaluate(
        X: np.ndarray,
        stream_mask=(1, 1, 1, 1),
        motion_mask=(1, 1, 1, 1),
        router_scale=1.0,
        transform="normal",
        collect_details=False,
    ):
        all_logits = []
        all_sl = []
        all_fw = []
        all_rw = []
        perm = np.asarray([0, 7, 2, 13, 4, 10, 6, 15, 1, 9, 3, 12, 5, 14, 8, 11], np.int64)

        sm = np.asarray(stream_mask, np.float32)
        mm = np.asarray(motion_mask, np.float32)
        rs = np.float32(router_scale)
        sm_rep = np.repeat(sm[None, :], ndev, axis=0)
        mm_rep = np.repeat(mm[None, :], ndev, axis=0)
        rs_rep = np.repeat(np.asarray([rs], np.float32), ndev, axis=0)

        for s in range(0, len(X), args.batch_size):
            xb0 = X[s : s + args.batch_size]
            if transform == "reverse":
                xb0 = xb0[:, ::-1].copy()
            elif transform == "shuffle":
                xb0 = xb0[:, perm].copy()
            elif transform == "static_repeat":
                xb0 = np.repeat(xb0[:, 7:8], tr.FRAMES, axis=1)
            elif transform != "normal":
                raise ValueError(transform)

            xb, n = _pad_batch(xb0, args.batch_size)
            xb = tr.shard(xb, ndev)
            logits, sl, fw, rw = p_metrics(params_rep, xb, sm_rep, mm_rep, rs_rep)
            logits = np.asarray(jax.device_get(logits)).reshape(-1, tr.NUM_CLASSES)[:n]
            all_logits.append(logits)
            if collect_details:
                all_sl.append(
                    np.asarray(jax.device_get(sl)).reshape(-1, tr.NUM_STREAMS, tr.NUM_CLASSES)[:n]
                )
                all_fw.append(
                    np.asarray(jax.device_get(fw)).reshape(-1, tr.NUM_STREAMS)[:n]
                )
                all_rw.append(
                    np.asarray(jax.device_get(rw)).reshape(-1, tr.FRAMES, tr.NUM_STREAMS)[:n]
                )

        result = {"logits": np.concatenate(all_logits, axis=0)}
        if collect_details:
            result.update(
                stream_logits=np.concatenate(all_sl, axis=0),
                fusion_weights=np.concatenate(all_fw, axis=0),
                router_weights=np.concatenate(all_rw, axis=0),
            )
        return result

    # ------------------------------------------------------------------
    # Baseline and representation interventions.
    # ------------------------------------------------------------------
    baseline = evaluate(Xv, collect_details=True)
    base_acc = _accuracy(baseline["logits"], yv)
    log(f"BASELINE={100*base_acc:.3f}%")

    variants = [
        ("router_off", (1, 1, 1, 1), (1, 1, 1, 1), 0.0, "normal"),
        ("pose_only_JB", (1, 1, 0, 0), (1, 1, 1, 1), 1.0, "normal"),
        ("motion_only_JMBM", (0, 0, 1, 1), (1, 1, 1, 1), 1.0, "normal"),
        ("minus_J", (0, 1, 1, 1), (1, 1, 1, 1), 1.0, "normal"),
        ("minus_B", (1, 0, 1, 1), (1, 1, 1, 1), 1.0, "normal"),
        ("minus_JM", (1, 1, 0, 1), (1, 1, 1, 1), 1.0, "normal"),
        ("minus_BM", (1, 1, 1, 0), (1, 1, 1, 1), 1.0, "normal"),
        ("J_only", (1, 0, 0, 0), (1, 1, 1, 1), 1.0, "normal"),
        ("B_only", (0, 1, 0, 0), (1, 1, 1, 1), 1.0, "normal"),
        ("JM_only", (0, 0, 1, 0), (1, 1, 1, 1), 1.0, "normal"),
        ("BM_only", (0, 0, 0, 1), (1, 1, 1, 1), 1.0, "normal"),
        ("no_D1", (1, 1, 1, 1), (0, 1, 1, 1), 1.0, "normal"),
        ("no_D2", (1, 1, 1, 1), (1, 0, 1, 1), 1.0, "normal"),
        ("no_D4", (1, 1, 1, 1), (1, 1, 0, 1), 1.0, "normal"),
        ("no_ACC", (1, 1, 1, 1), (1, 1, 1, 0), 1.0, "normal"),
        ("reverse_frames", (1, 1, 1, 1), (1, 1, 1, 1), 1.0, "reverse"),
        ("shuffle_frames", (1, 1, 1, 1), (1, 1, 1, 1), 1.0, "shuffle"),
        ("static_repeat", (1, 1, 1, 1), (1, 1, 1, 1), 1.0, "static_repeat"),
    ]

    variant_scores = {}
    for name, sm, mm, rs, tfm in variants:
        out = evaluate(Xv, sm, mm, rs, tfm, collect_details=False)
        acc = _accuracy(out["logits"], yv)
        variant_scores[name] = {
            "accuracy": acc,
            "drop_pp": 100.0 * (base_acc - acc),
        }
        log(f"{name:22s} {100*acc:7.3f}% | drop={100*(base_acc-acc):+7.3f} pp")

    # Same trained checkpoint, but use uniform frame selection at inference.
    Xvu, yvu, _ = materialize(
        annotations, split, args.protocol, "val", "uniform", 0, args.seed
    )
    if not np.array_equal(yv, yvu):
        raise RuntimeError("Motion/uniform validation label order mismatch")
    uniform_out = evaluate(Xvu, collect_details=False)
    uniform_acc = _accuracy(uniform_out["logits"], yv)
    variant_scores["uniform_selector"] = {
        "accuracy": uniform_acc,
        "drop_pp": 100.0 * (base_acc - uniform_acc),
    }
    log(f"{'uniform_selector':22s} {100*uniform_acc:7.3f}% | drop={100*(base_acc-uniform_acc):+7.3f} pp")
    del Xvu, yvu, uniform_out

    # ------------------------------------------------------------------
    # Stream heads, router/fusion statistics and confusions.
    # ------------------------------------------------------------------
    sl = baseline["stream_logits"]
    fw = baseline["fusion_weights"]
    rw = baseline["router_weights"]
    stream_head_acc = {
        STREAM_NAMES[i]: _accuracy(sl[:, i], yv) for i in range(tr.NUM_STREAMS)
    }
    fusion_mean = fw.mean(axis=0)
    router_mean = rw.mean(axis=(0, 1))
    fusion_entropy = float(_entropy(fw).mean())
    router_entropy = float(_entropy(rw).mean())
    fusion_entropy_norm = float(fusion_entropy / math.log(4.0))
    router_entropy_norm = float(router_entropy / math.log(4.0))

    log("STREAM HEAD ACCURACY (post-router): " + " | ".join(
        f"{k}={100*v:.2f}%" for k, v in stream_head_acc.items()
    ))
    log("MEAN FUSION: " + " | ".join(
        f"{STREAM_NAMES[i]}={fusion_mean[i]:.4f}" for i in range(4)
    ))
    log("MEAN ROUTER: " + " | ".join(
        f"{STREAM_NAMES[i]}={router_mean[i]:.4f}" for i in range(4)
    ))
    log(f"FUSION ENTROPY={fusion_entropy:.4f} ({100*fusion_entropy_norm:.2f}% of max)")
    log(f"ROUTER ENTROPY={router_entropy:.4f} ({100*router_entropy_norm:.2f}% of max)")

    pred = np.argmax(baseline["logits"], axis=-1)
    cm = _confusion(yv, pred)
    top_conf = _top_confusions(cm, 15)
    log("TOP CONFUSIONS (zero-based labels):")
    for r in top_conf:
        log(
            f"  true={r['true']:3d} -> pred={r['pred']:3d} | "
            f"count={r['count']:4d}/{r['true_support']:4d} | "
            f"{100*r['pair_rate_within_true']:.1f}% of true class"
        )

    # ------------------------------------------------------------------
    # Stage-wise linear probes on moderate random subsets.
    # ------------------------------------------------------------------
    Xt, yt, _ = materialize(
        annotations,
        split,
        args.protocol,
        "train",
        "motion",
        args.probe_train,
        args.seed + 11,
    )
    rng = np.random.default_rng(args.seed + 12)
    nprobe_val = min(args.probe_val, len(Xv))
    vi = np.sort(rng.choice(len(Xv), size=nprobe_val, replace=False))
    Xpv = Xv[vi]
    ypv = yv[vi]

    def extract_features(X):
        names = ("spatial", "router", "frame_memory", "descriptor")
        chunks = {k: [] for k in names}
        for s in range(0, len(X), args.batch_size):
            xb0 = X[s : s + args.batch_size]
            xb, n = _pad_batch(xb0, args.batch_size)
            xb = tr.shard(xb, ndev)
            vals = p_features(params_rep, xb)
            for k, v in zip(names, vals):
                arr = np.asarray(jax.device_get(v)).reshape(args.batch_size, -1)[:n]
                chunks[k].append(arr)
        return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}

    log(f"Extracting probe features: train={len(Xt):,} val={len(Xpv):,}")
    ftr = extract_features(Xt)
    fva = extract_features(Xpv)
    probe_scores = {}
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeClassifier

        for name in ("spatial", "router", "frame_memory", "descriptor"):
            clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
            clf.fit(ftr[name], yt)
            score = float(clf.score(fva[name], ypv))
            probe_scores[name] = score
            log(f"LINEAR PROBE {name:12s} = {100*score:.3f}%")
    except Exception as exc:
        log(f"Linear probes unavailable: {type(exc).__name__}: {exc}")
        probe_scores["error"] = f"{type(exc).__name__}: {exc}"

    del Xt, yt, Xpv, ypv, ftr, fva

    # ------------------------------------------------------------------
    # Temporal path retained by the T16 selector before the neural network.
    # ------------------------------------------------------------------
    retention = temporal_path_retention(
        annotations,
        split,
        args.protocol,
        args.path_retention_samples,
        args.seed + 99,
    )
    log(
        f"TEMPORAL PATH RETENTION n={retention['n']:,} | "
        f"motion-selector={100*retention['motion_mean']:.2f}% | "
        f"uniform={100*retention['uniform_mean']:.2f}%"
    )
    log("WORST MOTION-RETENTION CLASSES (zero-based labels):")
    for r in retention["worst_motion_classes"][:10]:
        log(
            f"  class={r['class']:3d} motion={100*r['motion_retention']:.1f}% "
            f"uniform={100*r['uniform_retention']:.1f}% n={r['n']}"
        )

    # ------------------------------------------------------------------
    # Simple evidence-based bottleneck flags. These are diagnostic hints,
    # not causal proof; the raw measurements remain the authoritative output.
    # ------------------------------------------------------------------
    flags = []
    shuf_drop = variant_scores["shuffle_frames"]["drop_pp"]
    static_drop = variant_scores["static_repeat"]["drop_pp"]
    pose_drop = variant_scores["pose_only_JB"]["drop_pp"]
    router_drop = variant_scores["router_off"]["drop_pp"]

    if shuf_drop < 3.0:
        flags.append("TEMPORAL_ORDER_WEAK: shuffling costs <3 pp; representation may be pose-dominated.")
    if static_drop < 5.0:
        flags.append("DYNAMIC_SIGNAL_WEAK: repeating one frame costs <5 pp; temporal dynamics may be underused.")
    if pose_drop < 3.0:
        flags.append("MOTION_STREAMS_WEAK: J+B alone remain within 3 pp of full model.")
    if router_drop < 1.0:
        flags.append("ROUTER_LOW_VALUE: bypassing cross-stream router costs <1 pp.")
    if retention["motion_mean"] < 0.60:
        flags.append("T16_SELECTION_LOSS: motion-aware T16 retains <60% of full temporal path length.")

    if all(k in probe_scores for k in ("spatial", "router", "frame_memory", "descriptor")):
        gains = {
            "spatial_to_router_pp": 100 * (probe_scores["router"] - probe_scores["spatial"]),
            "router_to_frame_pp": 100 * (probe_scores["frame_memory"] - probe_scores["router"]),
            "frame_to_descriptor_pp": 100 * (probe_scores["descriptor"] - probe_scores["frame_memory"]),
        }
        weakest = min(gains, key=gains.get)
        flags.append(
            f"WEAKEST_STAGE_GAIN: {weakest} = {gains[weakest]:+.2f} pp linear-probe gain."
        )
    else:
        gains = {}

    report = {
        "protocol": args.protocol,
        "checkpoint": str(ckpt),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_val_accuracy": float(payload.get("val_accuracy", float("nan"))),
        "params": tr.count_params(params),
        "baseline_accuracy": base_acc,
        "variants": variant_scores,
        "stream_head_accuracy": stream_head_acc,
        "fusion_mean": {STREAM_NAMES[i]: float(fusion_mean[i]) for i in range(4)},
        "router_mean": {STREAM_NAMES[i]: float(router_mean[i]) for i in range(4)},
        "fusion_entropy": fusion_entropy,
        "fusion_entropy_normalized": fusion_entropy_norm,
        "router_entropy": router_entropy,
        "router_entropy_normalized": router_entropy_norm,
        "linear_probes": probe_scores,
        "stage_probe_gains_pp": gains,
        "temporal_path_retention": retention,
        "top_confusions": top_conf,
        "diagnostic_flags": flags,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 118)
    print("BOTTLENECK SUMMARY")
    print("=" * 118)
    print(f"FULL ACCURACY: {100*base_acc:.3f}%")
    for f in flags:
        print("-", f)
    print(f"REPORT: {out}")
    print("=" * 118)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

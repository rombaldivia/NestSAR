#!/usr/bin/env python3
from __future__ import annotations

"""NestSAR D128-MTS class-geometry / margin-transfer audit.

Read-only targeted audit.

Core question
-------------
Does M4 create large class margins on the training split that fail to transfer
to validation, especially for the action pairs the trained model confuses?

For Spatial -> M4 -> Router -> G4 -> Descriptor this audit measures:
  * train and validation true-vs-nearest-training-centroid cosine margin;
  * positive-margin rate;
  * within-class cosine dispersion;
  * train/validation class-centroid alignment;
  * nearest-centroid separation;
  * per-class Spatial->M4 train/val margin gain and excess train gain;
  * final-model weak classes and symmetric confusion pairs;
  * pair-specific train/validation margins for the top confusion pairs.

No training and no checkpoint mutation.
"""

import argparse
import json
import math
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from experiments.nestsar_sm_all_t16.audit_generalization_localization import (
    normalize_rows,
    sequences_and_means,
)
from experiments.nestsar_sm_all_t16.model_parallel import NestSARParallelT16
from experiments.nestsar_sm_all_t16.parallel_d128_config import (
    MODEL_NAME,
    EXPECTED_PARAMS,
    M4_HALF_LIVES,
    G4_HALF_LIVES,
)
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json


STAGES = ("spatial", "m4", "router", "g4", "descriptor")
NUM_CLASSES = 120


def report(path, protocol, phase, current, total, **extra):
    if path is None:
        return
    payload = {
        "protocol": protocol,
        "phase": phase,
        "current": int(current),
        "total": int(max(total, 1)),
        "done": False,
    }
    payload.update(extra)
    atomic_json(path, payload)


def count_params(tree):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def load_checkpoint(path):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    for key in ("ema_params", "config", "model", "protocol", "epoch", "val_accuracy"):
        if key not in payload:
            raise ValueError(f"Checkpoint missing {key!r}: {path}")
    if payload["model"] != MODEL_NAME:
        raise ValueError(f"Expected {MODEL_NAME}, got {payload['model']!r}")
    params = payload["ema_params"]
    n = count_params(params)
    if n != EXPECTED_PARAMS:
        raise ValueError(f"Parameter mismatch {n:,} != {EXPECTED_PARAMS:,}")
    return payload, params, dict(payload["config"])


def make_model(config):
    return NestSARParallelT16(
        spatial_dim=config["spatial_dim"],
        model_dim=config["model_dim"],
        dropout=config["dropout"],
        controller_dim=config["controller_dim"],
        fast_rank=config["fast_rank"],
        head_rank=config["head_rank"],
        sm_residual_scale=config["sm_residual_scale"],
        head_residual_scale=config["head_residual_scale"],
        m4_half_lives=M4_HALF_LIVES,
        g4_half_lives=G4_HALF_LIVES,
    )


def stage_features(x, out):
    _, means = sequences_and_means(x, out)
    feat = {
        stage: normalize_rows(np.asarray(means[stage], np.float32))
        for stage in STAGES
    }
    dims = {stage: arr.shape[1] for stage, arr in feat.items()}
    if len(set(dims.values())) != 1:
        raise RuntimeError(f"Expected dimension-matched stages, got {dims}")
    return feat


def save_centroid_stats(path, stats):
    arrays = {}
    for stage in STAGES:
        arrays[f"counts__{stage}"] = stats[stage]["counts"]
        arrays[f"sums__{stage}"] = stats[stage]["sums"]
    np.savez_compressed(path, **arrays)


def load_centroid_stats(path):
    result = {}
    with np.load(path, allow_pickle=False) as z:
        for stage in STAGES:
            result[stage] = {
                "counts": np.asarray(z[f"counts__{stage}"], np.int64),
                "sums": np.asarray(z[f"sums__{stage}"], np.float64),
            }
    return result


def centroid_pass(
    *,
    protocol,
    split_name,
    ids,
    dataset,
    forward,
    params,
    batch_size,
    status,
    cache_path,
):
    if cache_path.is_file():
        return load_centroid_stats(cache_path)

    acc = None
    steps = math.ceil(len(ids) / batch_size)

    for bi, start in enumerate(range(0, len(ids), batch_size)):
        idx = ids[start:start + batch_size]
        x = np.asarray(dataset.canonical[idx], np.float32)
        y = np.asarray(dataset.labels[idx], np.int32)
        out = jax.device_get(forward(params, jax.device_put(x)))
        feat = stage_features(x, out)

        if acc is None:
            acc = {
                stage: {
                    "counts": np.zeros(NUM_CLASSES, np.int64),
                    "sums": np.zeros(
                        (NUM_CLASSES, feat[stage].shape[1]), np.float64
                    ),
                }
                for stage in STAGES
            }

        counts = np.bincount(y, minlength=NUM_CLASSES)
        for stage in STAGES:
            acc[stage]["counts"] += counts
            np.add.at(acc[stage]["sums"], y, feat[stage])

        if bi % 5 == 0 or bi + 1 == steps:
            report(
                status,
                protocol,
                f"{split_name} centroids",
                bi + 1,
                steps,
            )

    for stage in STAGES:
        if np.any(acc[stage]["counts"] == 0):
            missing = np.where(acc[stage]["counts"] == 0)[0].tolist()
            raise RuntimeError(
                f"{split_name} missing classes at {stage}: {missing}"
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_centroid_stats(cache_path, acc)
    return acc


def centroid_mean(stats, stage):
    return (
        stats[stage]["sums"]
        / stats[stage]["counts"][:, None]
    ).astype(np.float64)


def centroid_unit(stats, stage):
    return normalize_rows(
        centroid_mean(stats, stage).astype(np.float32)
    ).astype(np.float64)


def centroid_geometry(train_stats, val_stats, stage):
    tc = centroid_unit(train_stats, stage)
    vc = centroid_unit(val_stats, stage)

    alignment = np.sum(tc * vc, axis=1)

    train_sim = tc @ tc.T
    val_sim = vc @ vc.T
    np.fill_diagonal(train_sim, -np.inf)
    np.fill_diagonal(val_sim, -np.inf)

    train_nearest = np.argmax(train_sim, axis=1)
    val_nearest = np.argmax(val_sim, axis=1)
    train_nearest_sim = train_sim[np.arange(NUM_CLASSES), train_nearest]
    val_nearest_sim = val_sim[np.arange(NUM_CLASSES), val_nearest]

    return {
        "train_val_alignment_per_class": alignment,
        "train_nearest_class": train_nearest,
        "val_nearest_class": val_nearest,
        "train_nearest_similarity": train_nearest_sim,
        "val_nearest_similarity": val_nearest_sim,
        "train_centroid_margin": 1.0 - train_nearest_sim,
        "val_centroid_margin": 1.0 - val_nearest_sim,
    }


def margin_pass(
    *,
    protocol,
    split_name,
    ids,
    dataset,
    forward,
    params,
    train_stats,
    split_stats,
    batch_size,
    status,
    cache_path,
    include_model_predictions,
):
    if cache_path.is_file():
        return json.loads(cache_path.read_text())

    train_centroids = {
        stage: centroid_unit(train_stats, stage)
        for stage in STAGES
    }
    split_centroids = {
        stage: centroid_unit(split_stats, stage)
        for stage in STAGES
    }

    acc = {
        stage: {
            "n": 0,
            "sum_margin": 0.0,
            "sum_margin2": 0.0,
            "positive": 0,
            "sum_true_similarity": 0.0,
            "sum_rival_similarity": 0.0,
            "sum_within_cosine_distance": 0.0,
            "class_count": np.zeros(NUM_CLASSES, np.int64),
            "class_margin_sum": np.zeros(NUM_CLASSES, np.float64),
            "class_positive": np.zeros(NUM_CLASSES, np.int64),
            "class_within_sum": np.zeros(NUM_CLASSES, np.float64),
            "rival_counts": np.zeros(
                (NUM_CLASSES, NUM_CLASSES), np.int64
            ),
        }
        for stage in STAGES
    }

    model_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    model_class_count = np.zeros(NUM_CLASSES, np.int64)
    model_class_correct = np.zeros(NUM_CLASSES, np.int64)
    model_correct = 0
    seen = 0

    steps = math.ceil(len(ids) / batch_size)

    for bi, start in enumerate(range(0, len(ids), batch_size)):
        idx = ids[start:start + batch_size]
        x = np.asarray(dataset.canonical[idx], np.float32)
        y = np.asarray(dataset.labels[idx], np.int32)
        out = jax.device_get(forward(params, jax.device_put(x)))
        feat = stage_features(x, out)

        n = len(y)
        seen += n

        if include_model_predictions:
            pred = np.asarray(out["logits"]).argmax(1).astype(np.int32)
            model_correct += int(np.sum(pred == y))
            model_class_count += np.bincount(
                y, minlength=NUM_CLASSES
            )
            model_class_correct += np.bincount(
                y[pred == y], minlength=NUM_CLASSES
            )
            np.add.at(model_confusion, (y, pred), 1)

        for stage in STAGES:
            z = np.asarray(feat[stage], np.float64)
            tc = train_centroids[stage]

            sims = z @ tc.T
            true_sim = sims[np.arange(n), y].copy()
            sims[np.arange(n), y] = -np.inf
            rival = np.argmax(sims, axis=1)
            rival_sim = sims[np.arange(n), rival]
            margin = true_sim - rival_sim
            positive = margin > 0

            own_split = split_centroids[stage][y]
            within_dist = 1.0 - np.sum(z * own_split, axis=1)

            a = acc[stage]
            a["n"] += n
            a["sum_margin"] += float(margin.sum())
            a["sum_margin2"] += float(np.square(margin).sum())
            a["positive"] += int(positive.sum())
            a["sum_true_similarity"] += float(true_sim.sum())
            a["sum_rival_similarity"] += float(rival_sim.sum())
            a["sum_within_cosine_distance"] += float(
                within_dist.sum()
            )
            a["class_count"] += np.bincount(
                y, minlength=NUM_CLASSES
            )
            np.add.at(a["class_margin_sum"], y, margin)
            np.add.at(a["class_positive"], y, positive.astype(np.int64))
            np.add.at(a["class_within_sum"], y, within_dist)
            np.add.at(a["rival_counts"], (y, rival), 1)

        if bi % 5 == 0 or bi + 1 == steps:
            report(
                status,
                protocol,
                f"{split_name} class margins",
                bi + 1,
                steps,
            )

    result = {
        "split": split_name,
        "samples": int(seen),
        "stages": {},
    }

    for stage in STAGES:
        a = acc[stage]
        n = max(a["n"], 1)
        mean = a["sum_margin"] / n
        var = max(a["sum_margin2"] / n - mean * mean, 0.0)

        class_count = np.maximum(a["class_count"], 1)
        class_margin = a["class_margin_sum"] / class_count
        class_positive = a["class_positive"] / class_count
        class_within = a["class_within_sum"] / class_count

        top_rival = np.argmax(a["rival_counts"], axis=1)
        top_rival_count = a["rival_counts"][
            np.arange(NUM_CLASSES), top_rival
        ]

        result["stages"][stage] = {
            "mean_margin": float(mean),
            "std_margin": float(np.sqrt(var)),
            "positive_margin_rate": float(a["positive"] / n),
            "mean_true_similarity": float(
                a["sum_true_similarity"] / n
            ),
            "mean_nearest_rival_similarity": float(
                a["sum_rival_similarity"] / n
            ),
            "mean_within_cosine_distance": float(
                a["sum_within_cosine_distance"] / n
            ),
            "class_margin": [
                float(x) for x in class_margin
            ],
            "class_positive_margin_rate": [
                float(x) for x in class_positive
            ],
            "class_within_cosine_distance": [
                float(x) for x in class_within
            ],
            "class_top_rival_1based": [
                int(x + 1) for x in top_rival
            ],
            "class_top_rival_count": [
                int(x) for x in top_rival_count
            ],
        }

    if include_model_predictions:
        model_class_acc = (
            model_class_correct
            / np.maximum(model_class_count, 1)
        )
        result["model"] = {
            "accuracy": float(model_correct / max(seen, 1)),
            "class_count": [
                int(x) for x in model_class_count
            ],
            "class_correct": [
                int(x) for x in model_class_correct
            ],
            "class_accuracy": [
                float(x) for x in model_class_acc
            ],
            "confusion_matrix": model_confusion.tolist(),
        }

    atomic_json(cache_path, result)
    return result


def top_symmetric_confusions(confusion, k=20):
    c = np.asarray(confusion, np.int64).copy()
    np.fill_diagonal(c, 0)
    pairs = []
    for a in range(NUM_CLASSES):
        for b in range(a + 1, NUM_CLASSES):
            ab = int(c[a, b])
            ba = int(c[b, a])
            total = ab + ba
            if total > 0:
                pairs.append((total, ab, ba, a, b))
    pairs.sort(reverse=True)
    return pairs[:k]


def pair_geometry(
    *,
    a,
    b,
    train_stats,
    val_stats,
    stage,
):
    train_mean = centroid_mean(train_stats, stage)
    val_mean = centroid_mean(val_stats, stage)
    tc = centroid_unit(train_stats, stage)
    vc = centroid_unit(val_stats, stage)

    # Pair-specific margin using TRAIN centroids as the fixed reference.
    ta = float(train_mean[a] @ (tc[a] - tc[b]))
    tb = float(train_mean[b] @ (tc[b] - tc[a]))
    va = float(val_mean[a] @ (tc[a] - tc[b]))
    vb = float(val_mean[b] @ (tc[b] - tc[a]))

    return {
        "train_pair_margin_a": ta,
        "train_pair_margin_b": tb,
        "train_pair_margin_mean": 0.5 * (ta + tb),
        "val_pair_margin_a": va,
        "val_pair_margin_b": vb,
        "val_pair_margin_mean": 0.5 * (va + vb),
        "train_centroid_cosine": float(tc[a] @ tc[b]),
        "val_centroid_cosine": float(vc[a] @ vc[b]),
        "class_a_train_val_alignment": float(tc[a] @ vc[a]),
        "class_b_train_val_alignment": float(tc[b] @ vc[b]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=("xsub", "xset"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--status", default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise RuntimeError(
            f"Expected one isolated GPU; backend={jax.default_backend()} "
            f"devices={jax.local_devices()}"
        )

    payload, params, config = load_checkpoint(args.checkpoint)
    if payload["protocol"] != args.protocol:
        raise ValueError(
            f"Checkpoint protocol={payload['protocol']} != {args.protocol}"
        )

    dataset = Dataset(args.cache)
    train_ids = np.asarray(
        dataset.splits[f"{args.protocol}_train"], np.int64
    )
    val_ids = np.asarray(
        dataset.splits[f"{args.protocol}_val"], np.int64
    )

    model = make_model(config)
    forward = jax.jit(
        lambda p, x: model.apply({"params": p}, x, training=False)
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(str(out) + ".resume")
    work.mkdir(parents=True, exist_ok=True)
    status = Path(args.status) if args.status else None

    identity = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_accuracy": float(payload["val_accuracy"]),
        "params": count_params(params),
        "train_samples": int(len(train_ids)),
        "val_samples": int(len(val_ids)),
        "stages": list(STAGES),
        "feature_dim": 4 * int(config["model_dim"]),
    }
    identity_path = work / "identity.json"
    if identity_path.is_file():
        old = json.loads(identity_path.read_text())
        if old != identity:
            raise RuntimeError(
                f"Resume identity mismatch at {work}; "
                "use a fresh output path."
            )
    else:
        atomic_json(identity_path, identity)

    print("=" * 122)
    print(f"{MODEL_NAME} | {args.protocol.upper()} CLASS-GEOMETRY AUDIT")
    print("=" * 122)
    print("GPU:", jax.local_devices()[0])
    print(f"Checkpoint E{int(payload['epoch']):02d}")
    print(f"Recorded val: {100*float(payload['val_accuracy']):.4f}%")
    print(f"Train/val: {len(train_ids):,} / {len(val_ids):,}")
    print()

    train_stats = centroid_pass(
        protocol=args.protocol,
        split_name="Train",
        ids=train_ids,
        dataset=dataset,
        forward=forward,
        params=params,
        batch_size=args.batch_size,
        status=status,
        cache_path=work / "train_centroids.npz",
    )

    val_stats = centroid_pass(
        protocol=args.protocol,
        split_name="Validation",
        ids=val_ids,
        dataset=dataset,
        forward=forward,
        params=params,
        batch_size=args.batch_size,
        status=status,
        cache_path=work / "val_centroids.npz",
    )

    train_margin = margin_pass(
        protocol=args.protocol,
        split_name="Train",
        ids=train_ids,
        dataset=dataset,
        forward=forward,
        params=params,
        train_stats=train_stats,
        split_stats=train_stats,
        batch_size=args.batch_size,
        status=status,
        cache_path=work / "train_margins.json",
        include_model_predictions=False,
    )

    val_margin = margin_pass(
        protocol=args.protocol,
        split_name="Validation",
        ids=val_ids,
        dataset=dataset,
        forward=forward,
        params=params,
        train_stats=train_stats,
        split_stats=val_stats,
        batch_size=args.batch_size,
        status=status,
        cache_path=work / "val_margins.json",
        include_model_predictions=True,
    )

    verified = float(val_margin["model"]["accuracy"])
    recorded = float(payload["val_accuracy"])
    if abs(verified - recorded) > 5e-5:
        raise RuntimeError(
            f"Canonical accuracy mismatch: "
            f"{verified:.8f} vs {recorded:.8f}"
        )

    geometry = {
        stage: centroid_geometry(train_stats, val_stats, stage)
        for stage in STAGES
    }

    stage_report = {}
    for stage in STAGES:
        tr = train_margin["stages"][stage]
        va = val_margin["stages"][stage]
        g = geometry[stage]

        stage_report[stage] = {
            "train_mean_margin": tr["mean_margin"],
            "val_mean_margin": va["mean_margin"],
            "train_minus_val_margin_gap": (
                tr["mean_margin"] - va["mean_margin"]
            ),
            "val_over_train_margin_ratio": (
                va["mean_margin"]
                / max(abs(tr["mean_margin"]), 1e-12)
            ),
            "train_positive_margin_rate": tr[
                "positive_margin_rate"
            ],
            "val_positive_margin_rate": va[
                "positive_margin_rate"
            ],
            "train_within_cosine_distance": tr[
                "mean_within_cosine_distance"
            ],
            "val_within_cosine_distance": va[
                "mean_within_cosine_distance"
            ],
            "mean_train_val_centroid_alignment": float(
                np.mean(g["train_val_alignment_per_class"])
            ),
            "median_train_val_centroid_alignment": float(
                np.median(g["train_val_alignment_per_class"])
            ),
            "minimum_train_val_centroid_alignment": float(
                np.min(g["train_val_alignment_per_class"])
            ),
            "mean_train_centroid_margin": float(
                np.mean(g["train_centroid_margin"])
            ),
            "mean_val_centroid_margin": float(
                np.mean(g["val_centroid_margin"])
            ),
        }

    transitions = {}
    for left, right in zip(STAGES[:-1], STAGES[1:]):
        tr_gain = (
            stage_report[right]["train_mean_margin"]
            - stage_report[left]["train_mean_margin"]
        )
        va_gain = (
            stage_report[right]["val_mean_margin"]
            - stage_report[left]["val_mean_margin"]
        )
        transitions[f"{left}->{right}"] = {
            "train_margin_gain": tr_gain,
            "val_margin_gain": va_gain,
            "excess_train_margin_gain": tr_gain - va_gain,
        }

    # ------------------------------------------------------------------
    # Per-class Spatial->M4 margin transfer.
    # ------------------------------------------------------------------
    tr_sp = np.asarray(
        train_margin["stages"]["spatial"]["class_margin"],
        np.float64,
    )
    tr_m4 = np.asarray(
        train_margin["stages"]["m4"]["class_margin"],
        np.float64,
    )
    va_sp = np.asarray(
        val_margin["stages"]["spatial"]["class_margin"],
        np.float64,
    )
    va_m4 = np.asarray(
        val_margin["stages"]["m4"]["class_margin"],
        np.float64,
    )
    model_acc = np.asarray(
        val_margin["model"]["class_accuracy"],
        np.float64,
    )
    model_count = np.asarray(
        val_margin["model"]["class_count"],
        np.int64,
    )

    tr_gain = tr_m4 - tr_sp
    va_gain = va_m4 - va_sp
    excess = tr_gain - va_gain

    class_rows = []
    for c in range(NUM_CLASSES):
        class_rows.append({
            "ntu_action": int(c + 1),
            "n_val": int(model_count[c]),
            "model_val_accuracy": float(model_acc[c]),
            "spatial_train_margin": float(tr_sp[c]),
            "m4_train_margin": float(tr_m4[c]),
            "spatial_val_margin": float(va_sp[c]),
            "m4_val_margin": float(va_m4[c]),
            "m4_train_margin_gain": float(tr_gain[c]),
            "m4_val_margin_gain": float(va_gain[c]),
            "m4_excess_train_gain": float(excess[c]),
            "m4_train_val_centroid_alignment": float(
                geometry["m4"]["train_val_alignment_per_class"][c]
            ),
            "m4_val_top_rival_1based": int(
                val_margin["stages"]["m4"][
                    "class_top_rival_1based"
                ][c]
            ),
            "m4_val_top_rival_count": int(
                val_margin["stages"]["m4"][
                    "class_top_rival_count"
                ][c]
            ),
        })

    by_excess = sorted(
        class_rows,
        key=lambda r: r["m4_excess_train_gain"],
        reverse=True,
    )
    by_weak = sorted(
        class_rows,
        key=lambda r: r["model_val_accuracy"],
    )

    # ------------------------------------------------------------------
    # Actual final-model confusion pairs.
    # ------------------------------------------------------------------
    confusion_pairs = top_symmetric_confusions(
        val_margin["model"]["confusion_matrix"],
        k=20,
    )

    pair_rows = []
    for total, ab, ba, a, b in confusion_pairs:
        stages = {
            stage: pair_geometry(
                a=a,
                b=b,
                train_stats=train_stats,
                val_stats=val_stats,
                stage=stage,
            )
            for stage in STAGES
        }

        sp = stages["spatial"]
        m4 = stages["m4"]
        train_gain_pair = (
            m4["train_pair_margin_mean"]
            - sp["train_pair_margin_mean"]
        )
        val_gain_pair = (
            m4["val_pair_margin_mean"]
            - sp["val_pair_margin_mean"]
        )

        pair_rows.append({
            "action_a": int(a + 1),
            "action_b": int(b + 1),
            "confusions_total": int(total),
            "a_to_b": int(ab),
            "b_to_a": int(ba),
            "stages": stages,
            "m4_train_pair_margin_gain": float(train_gain_pair),
            "m4_val_pair_margin_gain": float(val_gain_pair),
            "m4_excess_train_pair_margin_gain": float(
                train_gain_pair - val_gain_pair
            ),
        })

    pair_rows.sort(
        key=lambda r: (
            r["confusions_total"],
            r["m4_excess_train_pair_margin_gain"],
        ),
        reverse=True,
    )

    automatic = {
        "largest_excess_transition": max(
            transitions,
            key=lambda k: transitions[k][
                "excess_train_margin_gain"
            ],
        ),
        "m4_margin_transfer_supported": bool(
            transitions["spatial->m4"][
                "excess_train_margin_gain"
            ] > 0
        ),
        "spatial_to_m4_excess_train_margin_gain": float(
            transitions["spatial->m4"][
                "excess_train_margin_gain"
            ]
        ),
        "interpretation": (
            "If Spatial->M4 has a large positive excess train margin gain "
            "and the same pattern appears in weak/confused classes, M4 is "
            "creating training class geometry that transfers incompletely "
            "to validation. That supports class-margin/generalization "
            "regularization rather than width/horizon/strength changes."
        ),
    }

    result = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_recorded_accuracy": recorded,
        "verified_accuracy": verified,
        "params": count_params(params),
        "train_samples": int(len(train_ids)),
        "val_samples": int(len(val_ids)),
        "feature_dimension": 4 * int(config["model_dim"]),
        "stages": stage_report,
        "transitions": transitions,
        "top_20_classes_by_m4_excess_train_gain": by_excess[:20],
        "worst_20_classes_by_model_accuracy": by_weak[:20],
        "top_20_symmetric_model_confusion_pairs": pair_rows,
        "automatic_interpretation": automatic,
    }

    atomic_json(out, result)

    if status is not None:
        atomic_json(
            status,
            {
                "protocol": args.protocol,
                "phase": "Done",
                "current": 1,
                "total": 1,
                "done": True,
                "verified_accuracy": verified,
            },
        )

    print()
    print("=" * 122)
    print("STAGE CLASS-MARGIN TRANSFER")
    print("=" * 122)
    print(
        f"{'stage':12s} "
        f"{'trainMargin':>12s} "
        f"{'valMargin':>12s} "
        f"{'gap':>10s} "
        f"{'train+':>9s} "
        f"{'val+':>9s} "
        f"{'centAlign':>10s} "
        f"{'trCentM':>10s} "
        f"{'vaCentM':>10s}"
    )
    print("-" * 100)

    for stage in STAGES:
        r = stage_report[stage]
        print(
            f"{stage:12s} "
            f"{r['train_mean_margin']:12.6f} "
            f"{r['val_mean_margin']:12.6f} "
            f"{r['train_minus_val_margin_gap']:10.6f} "
            f"{100*r['train_positive_margin_rate']:8.2f}% "
            f"{100*r['val_positive_margin_rate']:8.2f}% "
            f"{r['mean_train_val_centroid_alignment']:10.5f} "
            f"{r['mean_train_centroid_margin']:10.5f} "
            f"{r['mean_val_centroid_margin']:10.5f}"
        )

    print()
    print("TRANSITIONS")
    for name, r in transitions.items():
        print(
            f"{name:22s} | "
            f"trainGain={r['train_margin_gain']:+.6f} | "
            f"valGain={r['val_margin_gain']:+.6f} | "
            f"EXCESS={r['excess_train_margin_gain']:+.6f}"
        )

    print()
    print("=" * 122)
    print("TOP CLASSES — M4 TRAIN-GEOMETRY EXCESS")
    print("=" * 122)
    for r in by_excess[:15]:
        print(
            f"A{r['ntu_action']:03d} | "
            f"acc={100*r['model_val_accuracy']:6.2f}% | "
            f"trGain={r['m4_train_margin_gain']:+.5f} | "
            f"vaGain={r['m4_val_margin_gain']:+.5f} | "
            f"EXCESS={r['m4_excess_train_gain']:+.5f} | "
            f"align={r['m4_train_val_centroid_alignment']:.4f} | "
            f"rival=A{r['m4_val_top_rival_1based']:03d}"
        )

    print()
    print("=" * 122)
    print("TOP MODEL CONFUSION PAIRS — CLASS GEOMETRY")
    print("=" * 122)
    for r in pair_rows[:15]:
        print(
            f"A{r['action_a']:03d}<->A{r['action_b']:03d} | "
            f"conf={r['confusions_total']:4d} "
            f"({r['a_to_b']}/{r['b_to_a']}) | "
            f"M4 trainGain={r['m4_train_pair_margin_gain']:+.5f} | "
            f"valGain={r['m4_val_pair_margin_gain']:+.5f} | "
            f"EXCESS={r['m4_excess_train_pair_margin_gain']:+.5f}"
        )

    print()
    print("=" * 122)
    print("AUTOMATIC INTERPRETATION")
    print("=" * 122)
    print(
        "largest excess transition:",
        automatic["largest_excess_transition"],
    )
    print(
        "Spatial->M4 excess train margin gain:",
        f"{automatic['spatial_to_m4_excess_train_margin_gain']:+.6f}",
    )
    print(
        "M4 class-margin transfer hypothesis:",
        automatic["m4_margin_transfer_supported"],
    )

    print()
    print("Saved:", out)
    print("=" * 122)


if __name__ == "__main__":
    main()

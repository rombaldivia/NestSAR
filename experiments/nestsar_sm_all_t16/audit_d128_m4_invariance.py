#!/usr/bin/env python3
from __future__ import annotations

"""Targeted M4 representation-invariance / nuisance-leakage audit.

Read-only audit for trained NestSAR D128-MTS.

Question
--------
The previous causal audits showed:
  * M4 is essential;
  * simply changing its horizon does not help;
  * simply weakening/strengthening its transformation does not help.

This audit asks a narrower question:
  Does M4 add subject/setup/camera-specific information that is associated
  with validation errors, low-motion clips, or weak classes?

Method
------
For Spatial, M4, Router, G4, Descriptor and the transformation deltas
(M4-Spatial, Router-M4, G4-Router):
  1. L2-normalize the clip representation.
  2. Subtract the action centroid within the same split.
  3. Measure nuisance between/within scatter for subject/setup/camera.
  4. Repeat on validation subgroups:
       correct, wrong, low-motion, high-motion, weak classes, strong classes.

Because all post-spatial features are 4*D=512 dimensional, stage-to-stage
scatter comparisons are dimension matched.

No training and no checkpoint mutation.
"""

import argparse
import json
import math
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from experiments.nestsar_sm_all_t16.audit_d128_trained import parse_nuisance
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


NUM_CLASSES = 120
REP_FEATURES = ("spatial", "m4", "router", "g4", "descriptor")
DELTA_FEATURES = ("m4_delta", "router_delta", "g4_delta")
FEATURES = REP_FEATURES + DELTA_FEATURES
NUISANCES = ("subject", "setup", "camera")
SUBGROUPS = (
    "all",
    "correct",
    "wrong",
    "low_motion",
    "high_motion",
    "weak_classes",
    "strong_classes",
)


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


def feature_dict(x, out):
    _, means = sequences_and_means(x, out)
    feat = {
        "spatial": np.asarray(means["spatial"], np.float32),
        "m4": np.asarray(means["m4"], np.float32),
        "router": np.asarray(means["router"], np.float32),
        "g4": np.asarray(means["g4"], np.float32),
        "descriptor": np.asarray(means["descriptor"], np.float32),
    }
    feat["m4_delta"] = feat["m4"] - feat["spatial"]
    feat["router_delta"] = feat["router"] - feat["m4"]
    feat["g4_delta"] = feat["g4"] - feat["router"]

    dims = {k: v.shape[1] for k, v in feat.items()}
    if len(set(dims.values())) != 1:
        raise RuntimeError(f"Expected dimension-matched features, got {dims}")
    return feat


class ActionCentroids:
    def __init__(self, dim):
        self.dim = int(dim)
        self.counts = np.zeros(NUM_CLASSES, np.int64)
        self.sums = np.zeros((NUM_CLASSES, dim), np.float64)

    def add(self, x, labels):
        z = normalize_rows(x)
        self.counts += np.bincount(labels, minlength=NUM_CLASSES)
        np.add.at(self.sums, labels, z)

    def centroids(self):
        if np.any(self.counts == 0):
            missing = np.where(self.counts == 0)[0].tolist()
            raise RuntimeError(f"Missing action classes: {missing}")
        return (self.sums / self.counts[:, None]).astype(np.float32)


class FilteredGroupScatter:
    """Nuisance scatter with minimum-count filtering at report time."""

    def __init__(self, dim, max_group):
        self.dim = int(dim)
        self.counts = np.zeros(max_group + 1, np.int64)
        self.sums = np.zeros((max_group + 1, dim), np.float64)
        self.sum_norm2 = np.zeros(max_group + 1, np.float64)

    def add(self, x, groups):
        x = np.asarray(x, np.float32)
        groups = np.asarray(groups, np.int32)
        if len(x) == 0:
            return
        self.counts += np.bincount(groups, minlength=len(self.counts))
        np.add.at(self.sums, groups, x)
        np.add.at(
            self.sum_norm2,
            groups,
            np.square(x, dtype=np.float64).sum(axis=1),
        )

    def result(self, min_count=5):
        use = self.counts >= int(min_count)
        if np.sum(use) < 2:
            return {
                "groups": int(np.sum(use)),
                "samples": int(self.counts[use].sum()),
                "between": None,
                "within": None,
                "fisher_between_over_within": None,
                "between_fraction": None,
                "min_group_count": int(min_count),
            }

        counts = self.counts[use].astype(np.float64)
        sums = self.sums[use]
        sum_norm2 = self.sum_norm2[use]

        means = sums / counts[:, None]
        total_n = float(counts.sum())
        global_mean = sums.sum(axis=0) / total_n

        within = float(
            np.sum(
                sum_norm2
                - np.square(sums, dtype=np.float64).sum(axis=1) / counts
            )
        )
        between = float(
            np.sum(
                counts[:, None]
                * np.square(means - global_mean[None, :], dtype=np.float64)
            )
        )
        fisher = between / max(within, 1e-12)
        frac = between / max(between + within, 1e-12)

        return {
            "groups": int(np.sum(use)),
            "samples": int(total_n),
            "between": between,
            "within": within,
            "fisher_between_over_within": float(fisher),
            "between_fraction": float(frac),
            "min_group_count": int(min_count),
        }


def save_centroids(path, centroids):
    arrays = {f"centroid__{k}": v for k, v in centroids.items()}
    np.savez_compressed(path, **arrays)


def load_centroids(path):
    with np.load(path, allow_pickle=False) as z:
        return {
            k.replace("centroid__", ""): np.asarray(z[k], np.float32)
            for k in z.files
        }


def build_centroids(
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
        return load_centroids(cache_path)

    acc = None
    steps = math.ceil(len(ids) / batch_size)

    for bi, start in enumerate(range(0, len(ids), batch_size)):
        idx = ids[start:start + batch_size]
        x = np.asarray(dataset.canonical[idx], np.float32)
        y = np.asarray(dataset.labels[idx], np.int32)
        out = jax.device_get(forward(params, jax.device_put(x)))
        feat = feature_dict(x, out)

        if acc is None:
            acc = {
                name: ActionCentroids(arr.shape[1])
                for name, arr in feat.items()
            }

        for name in FEATURES:
            acc[name].add(feat[name], y)

        if bi % 5 == 0 or bi + 1 == steps:
            report(
                status,
                protocol,
                f"{split_name} action centroids",
                bi + 1,
                steps,
            )

    centroids = {name: acc[name].centroids() for name in FEATURES}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_centroids(cache_path, centroids)
    return centroids


def validation_metadata(
    *,
    protocol,
    ids,
    dataset,
    forward,
    params,
    batch_size,
    status,
    cache_path,
):
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as z:
            return {
                "labels": np.asarray(z["labels"], np.int32),
                "correct": np.asarray(z["correct"], bool),
                "motion": np.asarray(z["motion"], np.float32),
                "weak_classes": np.asarray(z["weak_classes"], np.int32),
                "strong_classes": np.asarray(z["strong_classes"], np.int32),
            }

    labels = np.asarray(dataset.labels[ids], np.int32)
    correct = np.zeros(len(ids), bool)
    motion = np.zeros(len(ids), np.float32)
    steps = math.ceil(len(ids) / batch_size)

    for bi, start in enumerate(range(0, len(ids), batch_size)):
        idx = ids[start:start + batch_size]
        x = np.asarray(dataset.canonical[idx], np.float32)
        y = np.asarray(dataset.labels[idx], np.int32)
        out = jax.device_get(forward(params, jax.device_put(x)))
        pred = np.asarray(out["logits"]).argmax(1)

        loc = slice(start, start + len(idx))
        correct[loc] = pred == y

        tok = x.reshape(len(x), 16, 2, 25, 15)
        motion[loc] = np.mean(
            np.abs(tok[..., 3:15]),
            axis=(1, 2, 3, 4),
        )

        if bi % 5 == 0 or bi + 1 == steps:
            report(
                status,
                protocol,
                "Validation correctness/motion",
                bi + 1,
                steps,
            )

    class_count = np.bincount(labels, minlength=NUM_CLASSES)
    class_good = np.bincount(labels[correct], minlength=NUM_CLASSES)
    class_acc = class_good / np.maximum(class_count, 1)

    order = np.argsort(class_acc)
    weak = order[:20].astype(np.int32)
    strong = order[-20:].astype(np.int32)

    np.savez_compressed(
        cache_path,
        labels=labels,
        correct=correct,
        motion=motion,
        weak_classes=weak,
        strong_classes=strong,
    )
    return {
        "labels": labels,
        "correct": correct,
        "motion": motion,
        "weak_classes": weak,
        "strong_classes": strong,
    }


def subgroup_masks(meta):
    motion = meta["motion"]
    q = np.quantile(motion, (0.25, 0.75))
    labels = meta["labels"]
    correct = meta["correct"]

    return {
        "all": np.ones(len(labels), bool),
        "correct": correct,
        "wrong": ~correct,
        "low_motion": motion <= q[0],
        "high_motion": motion >= q[1],
        "weak_classes": np.isin(labels, meta["weak_classes"]),
        "strong_classes": np.isin(labels, meta["strong_classes"]),
    }, {
        "motion_q25": float(q[0]),
        "motion_q75": float(q[1]),
    }


def scatter_pass(
    *,
    protocol,
    split_name,
    ids,
    dataset,
    forward,
    params,
    centroids,
    nuisance,
    batch_size,
    status,
    groups_by_local=None,
    output_json,
    detailed=False,
):
    if output_json.is_file():
        return json.loads(output_json.read_text())

    max_group = {name: int(nuisance[name].max()) for name in NUISANCES}

    if detailed:
        subgroup_names = SUBGROUPS
    else:
        subgroup_names = ("all",)

    # Allocate lazily after first batch determines dimensionality.
    scatters = None
    steps = math.ceil(len(ids) / batch_size)

    for bi, start in enumerate(range(0, len(ids), batch_size)):
        idx = ids[start:start + batch_size]
        x = np.asarray(dataset.canonical[idx], np.float32)
        y = np.asarray(dataset.labels[idx], np.int32)
        out = jax.device_get(forward(params, jax.device_put(x)))
        feat = feature_dict(x, out)

        if scatters is None:
            scatters = {
                subgroup: {
                    feature: {
                        n: FilteredGroupScatter(
                            feat[feature].shape[1],
                            max_group[n],
                        )
                        for n in NUISANCES
                    }
                    for feature in FEATURES
                }
                for subgroup in subgroup_names
            }

        local_n = len(idx)
        if groups_by_local is None:
            masks = {"all": np.ones(local_n, bool)}
        else:
            masks = {
                name: groups_by_local[name][start:start + local_n]
                for name in subgroup_names
            }

        for feature in FEATURES:
            z = normalize_rows(feat[feature])
            residual = z - centroids[feature][y]

            for subgroup in subgroup_names:
                mask = masks[subgroup]
                if not np.any(mask):
                    continue

                for n in NUISANCES:
                    scatters[subgroup][feature][n].add(
                        residual[mask],
                        nuisance[n][idx][mask],
                    )

        if bi % 5 == 0 or bi + 1 == steps:
            report(
                status,
                protocol,
                f"{split_name} nuisance scatter",
                bi + 1,
                steps,
            )

    result = {
        subgroup: {
            feature: {
                n: scatters[subgroup][feature][n].result(min_count=5)
                for n in NUISANCES
            }
            for feature in FEATURES
        }
        for subgroup in subgroup_names
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output_json, result)
    return result


def safe_ratio(a, b):
    if a is None or b is None or b <= 0:
        return None
    return float(a / b)


def fisher(report, subgroup, feature, nuisance):
    return report[subgroup][feature][nuisance][
        "fisher_between_over_within"
    ]


def frac(report, subgroup, feature, nuisance):
    return report[subgroup][feature][nuisance][
        "between_fraction"
    ]


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
    ids_all = json.loads((Path(args.cache) / "ids.json").read_text())
    nuisance = parse_nuisance(ids_all)

    parsed_action = nuisance["action_1based"] - 1
    if not np.array_equal(parsed_action, np.asarray(dataset.labels)):
        raise RuntimeError("Parsed NTU action IDs disagree with cache labels.")

    train_ids = np.asarray(dataset.splits[f"{args.protocol}_train"], np.int64)
    val_ids = np.asarray(dataset.splits[f"{args.protocol}_val"], np.int64)

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
        "features": list(FEATURES),
        "nuisances": list(NUISANCES),
        "subgroups": list(SUBGROUPS),
    }
    identity_path = work / "identity.json"
    if identity_path.is_file():
        old = json.loads(identity_path.read_text())
        if old != identity:
            raise RuntimeError(
                f"Resume identity mismatch at {work}; use a fresh output path."
            )
    else:
        atomic_json(identity_path, identity)

    print("=" * 122)
    print(f"{MODEL_NAME} | {args.protocol.upper()} M4 INVARIANCE / NUISANCE AUDIT")
    print("=" * 122)
    print("GPU:", jax.local_devices()[0])
    print(f"Checkpoint E{int(payload['epoch']):02d}")
    print(f"Recorded val: {100*float(payload['val_accuracy']):.4f}%")
    print(f"Train/val: {len(train_ids):,} / {len(val_ids):,}")
    print()

    train_centroids = build_centroids(
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

    val_centroids = build_centroids(
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

    meta = validation_metadata(
        protocol=args.protocol,
        ids=val_ids,
        dataset=dataset,
        forward=forward,
        params=params,
        batch_size=args.batch_size,
        status=status,
        cache_path=work / "val_meta.npz",
    )

    masks, motion_thresholds = subgroup_masks(meta)

    train_scatter = scatter_pass(
        protocol=args.protocol,
        split_name="Train",
        ids=train_ids,
        dataset=dataset,
        forward=forward,
        params=params,
        centroids=train_centroids,
        nuisance=nuisance,
        batch_size=args.batch_size,
        status=status,
        groups_by_local=None,
        output_json=work / "train_scatter.json",
        detailed=False,
    )

    val_scatter = scatter_pass(
        protocol=args.protocol,
        split_name="Validation",
        ids=val_ids,
        dataset=dataset,
        forward=forward,
        params=params,
        centroids=val_centroids,
        nuisance=nuisance,
        batch_size=args.batch_size,
        status=status,
        groups_by_local=masks,
        output_json=work / "val_scatter.json",
        detailed=True,
    )

    diagnostics = {}
    m4_spatial_all = []
    m4_wrong_correct = []
    m4_low_high = []
    m4_delta_spatial = []

    for n in NUISANCES:
        tr_sp = fisher(train_scatter, "all", "spatial", n)
        tr_m4 = fisher(train_scatter, "all", "m4", n)
        va_sp = fisher(val_scatter, "all", "spatial", n)
        va_m4 = fisher(val_scatter, "all", "m4", n)
        va_delta = fisher(val_scatter, "all", "m4_delta", n)
        wrong = fisher(val_scatter, "wrong", "m4", n)
        correct = fisher(val_scatter, "correct", "m4", n)
        low = fisher(val_scatter, "low_motion", "m4", n)
        high = fisher(val_scatter, "high_motion", "m4", n)

        row = {
            "train_spatial_fisher": tr_sp,
            "train_m4_fisher": tr_m4,
            "train_m4_over_spatial": safe_ratio(tr_m4, tr_sp),
            "val_spatial_fisher": va_sp,
            "val_m4_fisher": va_m4,
            "val_m4_over_spatial": safe_ratio(va_m4, va_sp),
            "val_m4_delta_fisher": va_delta,
            "val_m4_delta_over_spatial": safe_ratio(va_delta, va_sp),
            "val_m4_wrong_fisher": wrong,
            "val_m4_correct_fisher": correct,
            "val_m4_wrong_over_correct": safe_ratio(wrong, correct),
            "val_m4_low_motion_fisher": low,
            "val_m4_high_motion_fisher": high,
            "val_m4_low_over_high_motion": safe_ratio(low, high),
            "val_spatial_between_fraction": frac(
                val_scatter, "all", "spatial", n
            ),
            "val_m4_between_fraction": frac(
                val_scatter, "all", "m4", n
            ),
            "val_m4_delta_between_fraction": frac(
                val_scatter, "all", "m4_delta", n
            ),
        }
        diagnostics[n] = row

        for key, bucket in (
            ("val_m4_over_spatial", m4_spatial_all),
            ("val_m4_wrong_over_correct", m4_wrong_correct),
            ("val_m4_low_over_high_motion", m4_low_high),
            ("val_m4_delta_over_spatial", m4_delta_spatial),
        ):
            if row[key] is not None and np.isfinite(row[key]):
                bucket.append(row[key])

    aggregate = {
        "median_val_m4_over_spatial": float(np.median(m4_spatial_all)),
        "median_val_m4_wrong_over_correct": float(
            np.median(m4_wrong_correct)
        ),
        "median_val_m4_low_over_high_motion": float(
            np.median(m4_low_high)
        ),
        "median_val_m4_delta_over_spatial": float(
            np.median(m4_delta_spatial)
        ),
    }

    # A conservative automatic interpretation. The raw values remain the
    # primary evidence; this flag is only a compact summary.
    shortcut_supported = (
        aggregate["median_val_m4_over_spatial"] > 1.15
        and aggregate["median_val_m4_wrong_over_correct"] > 1.15
    )

    result = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_recorded_accuracy": float(payload["val_accuracy"]),
        "params": count_params(params),
        "train_samples": int(len(train_ids)),
        "val_samples": int(len(val_ids)),
        "feature_dimension": 4 * int(config["model_dim"]),
        "features": list(FEATURES),
        "nuisances": list(NUISANCES),
        "subgroups": list(SUBGROUPS),
        "motion_energy_note": (
            "Proxy is mean abs value of canonical channels 3:15 "
            "(displacement + phase + path), not pure physical velocity."
        ),
        "motion_thresholds": motion_thresholds,
        "weak_classes_1based": [
            int(x + 1) for x in meta["weak_classes"]
        ],
        "strong_classes_1based": [
            int(x + 1) for x in meta["strong_classes"]
        ],
        "train_action_residual_nuisance": train_scatter,
        "val_action_residual_nuisance": val_scatter,
        "m4_diagnostics": diagnostics,
        "aggregate": aggregate,
        "automatic_interpretation": {
            "m4_nuisance_shortcut_supported": bool(shortcut_supported),
            "rule": (
                "Support requires median M4/Spatial nuisance Fisher >1.15 "
                "and median wrong/correct M4 nuisance Fisher >1.15 across "
                "subject/setup/camera."
            ),
            "important_caveat": (
                "Within-split nuisance scatter measures representation leakage, "
                "not a causal accuracy gain. If supported, the next step should "
                "be a targeted invariance intervention, not another architecture "
                "width/horizon change."
            ),
        },
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
                "shortcut_supported": bool(shortcut_supported),
            },
        )

    print()
    print("=" * 122)
    print("M4 NUISANCE DIAGNOSTICS")
    print("=" * 122)
    print(
        f"{'nuisance':10s} "
        f"{'Tr M4/Sp':>10s} "
        f"{'Va M4/Sp':>10s} "
        f"{'M4Delta/Sp':>12s} "
        f"{'Wrong/Correct':>14s} "
        f"{'Low/High':>10s}"
    )
    print("-" * 74)

    for n in NUISANCES:
        d = diagnostics[n]

        def fmt(v):
            return "   n/a   " if v is None else f"{v:9.3f}x"

        print(
            f"{n:10s} "
            f"{fmt(d['train_m4_over_spatial'])} "
            f"{fmt(d['val_m4_over_spatial'])} "
            f"{fmt(d['val_m4_delta_over_spatial'])} "
            f"{fmt(d['val_m4_wrong_over_correct'])} "
            f"{fmt(d['val_m4_low_over_high_motion'])}"
        )

    print()
    print("AGGREGATE")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.4f}x")

    print()
    print(
        "M4 nuisance shortcut supported:",
        bool(shortcut_supported),
    )

    print()
    print("VAL STAGE LEAKAGE — ALL")
    for n in NUISANCES:
        print()
        print(n.upper())
        for feature in REP_FEATURES:
            r = val_scatter["all"][feature][n]
            print(
                f"  {feature:12s} "
                f"Fisher={r['fisher_between_over_within']:.6f} | "
                f"between%={100*r['between_fraction']:.3f}% | "
                f"groups={r['groups']} | n={r['samples']}"
            )
        for feature in DELTA_FEATURES:
            r = val_scatter["all"][feature][n]
            print(
                f"  {feature:12s} "
                f"Fisher={r['fisher_between_over_within']:.6f} | "
                f"between%={100*r['between_fraction']:.3f}%"
            )

    print()
    print("M4 SUBGROUP LEAKAGE")
    for subgroup in (
        "correct",
        "wrong",
        "low_motion",
        "high_motion",
        "weak_classes",
        "strong_classes",
    ):
        print()
        print(subgroup.upper())
        for n in NUISANCES:
            m4r = val_scatter[subgroup]["m4"][n]
            dr = val_scatter[subgroup]["m4_delta"][n]
            print(
                f"  {n:10s} "
                f"M4={m4r['fisher_between_over_within']:.6f} | "
                f"M4delta={dr['fisher_between_over_within']:.6f} | "
                f"n={m4r['samples']}"
            )

    print()
    print("Saved:", out)
    print("=" * 122)


if __name__ == "__main__":
    main()

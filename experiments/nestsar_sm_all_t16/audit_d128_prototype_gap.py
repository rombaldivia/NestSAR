#!/usr/bin/env python3
from __future__ import annotations

"""D128 class-prototype gap calibration audit.

Builds 120 class prototypes from TRAIN fused descriptors and measures:

    gap = cos(z, p_true) - max_{c != y} cos(z, p_c)

Training split uses a leave-one-out true-class prototype to avoid self-leakage.
Validation always uses fixed TRAIN prototypes.

This is a read-only calibration audit for choosing a prototype-margin loss.
No training, no gradients, no parameter/checkpoint mutation.
"""

import argparse
import json
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from experiments.nestsar_sm_all_t16.model_parallel import NestSARParallelT16
from experiments.nestsar_sm_all_t16.parallel_d128_config import (
    MODEL_NAME,
    EXPECTED_PARAMS,
    MODEL_DIM,
    M4_HALF_LIVES,
    G4_HALF_LIVES,
)
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json


NUM_CLASSES = 120
BATCH = 256
MARGINS = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20)


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
        raise RuntimeError(f"Parameter mismatch {n:,} != {EXPECTED_PARAMS:,}")
    return payload, params, dict(payload["config"])


def make_model(config):
    return NestSARParallelT16(
        spatial_dim=config["spatial_dim"],
        model_dim=MODEL_DIM,
        dropout=config["dropout"],
        controller_dim=config["controller_dim"],
        fast_rank=config["fast_rank"],
        head_rank=config["head_rank"],
        sm_residual_scale=config["sm_residual_scale"],
        head_residual_scale=config["head_residual_scale"],
        m4_half_lives=M4_HALF_LIVES,
        g4_half_lives=G4_HALF_LIVES,
    )


def normalize_rows(x):
    x = np.asarray(x, np.float32)
    n2 = np.sum(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(np.maximum(n2, 1e-12))


def fused_descriptor_np(out):
    desc = np.asarray(out["descriptors"], np.float32)
    fusion = np.asarray(out["fusion_weights"], np.float32)
    fused = np.einsum("bs,bsd->bd", fusion, desc)
    return normalize_rows(fused)


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


def atomic_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def build_train_prototypes(
    *,
    protocol,
    ids,
    dataset,
    forward,
    params,
    status,
    work,
):
    cache = work / "train_prototypes.npz"
    if cache.is_file():
        data = load_npz(cache)
        return (
            data["sums"].astype(np.float64),
            data["counts"].astype(np.int64),
            data["prototypes"].astype(np.float32),
        )

    sums = np.zeros((NUM_CLASSES, MODEL_DIM), np.float64)
    counts = np.zeros(NUM_CLASSES, np.int64)
    steps = math.ceil(len(ids) / BATCH)

    for bi, start in enumerate(range(0, len(ids), BATCH)):
        idx = ids[start:start + BATCH]
        x = np.asarray(dataset.canonical[idx], np.float32)
        y = np.asarray(dataset.labels[idx], np.int32)

        out = jax.device_get(forward(params, jax.device_put(x)))
        z = fused_descriptor_np(out)

        np.add.at(sums, y, z)
        counts += np.bincount(y, minlength=NUM_CLASSES)

        if bi % 2 == 0 or bi + 1 == steps:
            report(
                status,
                protocol,
                "Build train prototypes",
                bi + 1,
                steps,
            )

    if np.any(counts < 2):
        bad = np.where(counts < 2)[0].tolist()
        raise RuntimeError(f"Classes with <2 train samples: {bad}")

    prototypes = normalize_rows(
        (sums / counts[:, None]).astype(np.float32)
    )

    atomic_npz(
        cache,
        sums=sums,
        counts=counts,
        prototypes=prototypes,
    )

    return sums, counts, prototypes


def top_symmetric_pairs(pair_counts, k=20):
    pair_counts = np.asarray(pair_counts, np.int64)
    rows = []
    for a in range(NUM_CLASSES):
        for b in range(a + 1, NUM_CLASSES):
            count = int(pair_counts[a, b] + pair_counts[b, a])
            if count:
                rows.append((count, a, b))
    rows.sort(reverse=True)
    return [
        {
            "action_a": int(a + 1),
            "action_b": int(b + 1),
            "count": int(count),
        }
        for count, a, b in rows[:k]
    ]


def summarize_gaps(gap, true_sim, rival_sim, margins):
    gap = np.asarray(gap, np.float64)
    true_sim = np.asarray(true_sim, np.float64)
    rival_sim = np.asarray(rival_sim, np.float64)

    out = {
        "samples": int(len(gap)),
        "true_similarity": {
            "mean": float(np.mean(true_sim)),
            "std": float(np.std(true_sim)),
            "q05": float(np.quantile(true_sim, 0.05)),
            "q25": float(np.quantile(true_sim, 0.25)),
            "q50": float(np.quantile(true_sim, 0.50)),
            "q75": float(np.quantile(true_sim, 0.75)),
            "q95": float(np.quantile(true_sim, 0.95)),
        },
        "nearest_wrong_similarity": {
            "mean": float(np.mean(rival_sim)),
            "std": float(np.std(rival_sim)),
            "q05": float(np.quantile(rival_sim, 0.05)),
            "q25": float(np.quantile(rival_sim, 0.25)),
            "q50": float(np.quantile(rival_sim, 0.50)),
            "q75": float(np.quantile(rival_sim, 0.75)),
            "q95": float(np.quantile(rival_sim, 0.95)),
        },
        "prototype_gap": {
            "mean": float(np.mean(gap)),
            "std": float(np.std(gap)),
            "q01": float(np.quantile(gap, 0.01)),
            "q05": float(np.quantile(gap, 0.05)),
            "q10": float(np.quantile(gap, 0.10)),
            "q25": float(np.quantile(gap, 0.25)),
            "q50": float(np.quantile(gap, 0.50)),
            "q75": float(np.quantile(gap, 0.75)),
            "q90": float(np.quantile(gap, 0.90)),
            "positive_fraction": float(np.mean(gap > 0.0)),
        },
        "margins": {},
    }

    for m in margins:
        loss = np.maximum(0.0, float(m) - gap)
        active = loss > 1e-8
        out["margins"][f"{m:.2f}"] = {
            "active_fraction": float(np.mean(active)),
            "mean_loss": float(np.mean(loss)),
            "mean_loss_active_only": float(
                np.mean(loss[active]) if np.any(active) else 0.0
            ),
        }

    return out


def split_pass(
    *,
    protocol,
    split_name,
    ids,
    dataset,
    forward,
    params,
    train_sums,
    train_counts,
    prototypes,
    status,
    work,
    leave_one_out,
    collect_predictions,
):
    cache = work / f"{split_name.lower()}_prototype_gap.npz"

    if cache.is_file():
        data = load_npz(cache)
        return data

    steps = math.ceil(len(ids) / BATCH)

    gaps = []
    true_sims = []
    rival_sims = []
    labels_all = []
    rivals_all = []
    preds_all = []
    correct_all = []

    pair_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)

    for bi, start in enumerate(range(0, len(ids), BATCH)):
        idx = ids[start:start + BATCH]
        x = np.asarray(dataset.canonical[idx], np.float32)
        y = np.asarray(dataset.labels[idx], np.int32)

        out = jax.device_get(forward(params, jax.device_put(x)))
        z = fused_descriptor_np(out)

        sims = z @ prototypes.T

        if leave_one_out:
            cy = train_counts[y].astype(np.float64)
            sy = train_sums[y]
            loo = (sy - z.astype(np.float64)) / (cy[:, None] - 1.0)
            loo = normalize_rows(loo.astype(np.float32))
            true_sim = np.sum(z * loo, axis=1)
        else:
            true_sim = sims[np.arange(len(y)), y].copy()

        sims[np.arange(len(y)), y] = -np.inf
        rival = np.argmax(sims, axis=1).astype(np.int32)
        rival_sim = sims[np.arange(len(y)), rival]
        gap = true_sim - rival_sim

        np.add.at(pair_counts, (y, rival), 1)

        gaps.append(gap.astype(np.float32))
        true_sims.append(true_sim.astype(np.float32))
        rival_sims.append(rival_sim.astype(np.float32))
        labels_all.append(y.astype(np.int16))
        rivals_all.append(rival.astype(np.int16))

        if collect_predictions:
            pred = np.asarray(out["logits"]).argmax(1).astype(np.int32)
            np.add.at(confusion, (y, pred), 1)
            preds_all.append(pred.astype(np.int16))
            correct_all.append((pred == y).astype(np.int8))

        if bi % 2 == 0 or bi + 1 == steps:
            report(
                status,
                protocol,
                f"{split_name} prototype gaps",
                bi + 1,
                steps,
            )

    payload = {
        "gap": np.concatenate(gaps),
        "true_similarity": np.concatenate(true_sims),
        "rival_similarity": np.concatenate(rival_sims),
        "labels": np.concatenate(labels_all),
        "rivals": np.concatenate(rivals_all),
        "pair_counts": pair_counts,
    }

    if collect_predictions:
        payload.update(
            predictions=np.concatenate(preds_all),
            correct=np.concatenate(correct_all),
            confusion=confusion,
        )

    atomic_npz(cache, **payload)
    return payload


def class_summary(labels, gap, rivals, margins):
    labels = np.asarray(labels, np.int32)
    gap = np.asarray(gap, np.float64)
    rivals = np.asarray(rivals, np.int32)

    rows = []

    for c in range(NUM_CLASSES):
        sel = labels == c
        cg = gap[sel]
        cr = rivals[sel]

        if len(cg) == 0:
            continue

        counts = np.bincount(cr, minlength=NUM_CLASSES)
        counts[c] = 0
        top_rival = int(np.argmax(counts))

        row = {
            "action": int(c + 1),
            "samples": int(len(cg)),
            "mean_gap": float(np.mean(cg)),
            "median_gap": float(np.median(cg)),
            "positive_fraction": float(np.mean(cg > 0)),
            "top_rival_action": int(top_rival + 1),
            "top_rival_count": int(counts[top_rival]),
            "active_fraction": {},
        }

        for m in margins:
            row["active_fraction"][f"{m:.2f}"] = float(
                np.mean(cg < float(m))
            )

        rows.append(row)

    rows.sort(key=lambda r: r["mean_gap"])
    return rows


def pair_overlap(proto_pairs, confusion_pairs, top_k=20):
    p = {
        tuple(sorted((row["action_a"], row["action_b"])))
        for row in proto_pairs[:top_k]
    }
    c = {
        tuple(sorted((row["action_a"], row["action_b"])))
        for row in confusion_pairs[:top_k]
    }
    overlap = sorted(p & c)
    return {
        "top_k": int(top_k),
        "prototype_pairs": int(len(p)),
        "confusion_pairs": int(len(c)),
        "overlap_count": int(len(overlap)),
        "overlap_fraction_of_confusions": float(len(overlap) / max(len(c), 1)),
        "pairs": [
            {"action_a": int(a), "action_b": int(b)}
            for a, b in overlap
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=("xsub", "xset"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--status", default=None)
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
    train_ids = np.asarray(dataset.splits[f"{args.protocol}_train"], np.int64)
    val_ids = np.asarray(dataset.splits[f"{args.protocol}_val"], np.int64)

    model = make_model(config)

    @jax.jit
    def forward(p, x):
        return model.apply({"params": p}, x, training=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    work = Path(str(output) + ".resume")
    work.mkdir(parents=True, exist_ok=True)

    status = Path(args.status) if args.status else None

    identity = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_val_accuracy": float(payload["val_accuracy"]),
        "train_samples": int(len(train_ids)),
        "val_samples": int(len(val_ids)),
        "descriptor_dim": MODEL_DIM,
        "prototype_source": "train fused descriptor",
        "train_true_prototype": "leave-one-out",
        "validation_prototype": "fixed train prototype",
        "margins": list(MARGINS),
        "batch_size": BATCH,
    }

    identity_path = work / "identity.json"

    if identity_path.is_file():
        old = json.loads(identity_path.read_text())
        if old != identity:
            raise RuntimeError(
                "Resume identity mismatch; use a fresh output path."
            )
    else:
        atomic_json(identity_path, identity)

    print("=" * 124)
    print(f"{args.protocol.upper()} D128 PROTOTYPE-GAP CALIBRATION")
    print("=" * 124)
    print("GPU:", jax.local_devices()[0])
    print(
        f"Checkpoint E{int(payload['epoch']):02d} | "
        f"val={100*float(payload['val_accuracy']):.4f}%"
    )
    print(f"Train/val: {len(train_ids):,} / {len(val_ids):,}")
    print("Descriptor: normalized fused D128")
    print("Train positive prototype: leave-one-out own class")
    print("Wrong prototypes: all 119 train class prototypes")
    print("Margins:", MARGINS)
    print()

    train_sums, train_counts, prototypes = build_train_prototypes(
        protocol=args.protocol,
        ids=train_ids,
        dataset=dataset,
        forward=forward,
        params=params,
        status=status,
        work=work,
    )

    train = split_pass(
        protocol=args.protocol,
        split_name="Train",
        ids=train_ids,
        dataset=dataset,
        forward=forward,
        params=params,
        train_sums=train_sums,
        train_counts=train_counts,
        prototypes=prototypes,
        status=status,
        work=work,
        leave_one_out=True,
        collect_predictions=False,
    )

    val = split_pass(
        protocol=args.protocol,
        split_name="Validation",
        ids=val_ids,
        dataset=dataset,
        forward=forward,
        params=params,
        train_sums=train_sums,
        train_counts=train_counts,
        prototypes=prototypes,
        status=status,
        work=work,
        leave_one_out=False,
        collect_predictions=True,
    )

    val_acc = float(np.mean(np.asarray(val["correct"], np.float32)))
    recorded = float(payload["val_accuracy"])

    if abs(val_acc - recorded) > 5e-5:
        raise RuntimeError(
            f"Canonical accuracy mismatch {val_acc:.8f} vs {recorded:.8f}"
        )

    train_summary = summarize_gaps(
        train["gap"],
        train["true_similarity"],
        train["rival_similarity"],
        MARGINS,
    )

    val_summary = summarize_gaps(
        val["gap"],
        val["true_similarity"],
        val["rival_similarity"],
        MARGINS,
    )

    train_pairs = top_symmetric_pairs(train["pair_counts"], 30)
    val_pairs = top_symmetric_pairs(val["pair_counts"], 30)
    confusion_pairs = top_symmetric_pairs(val["confusion"], 30)

    overlap = pair_overlap(
        val_pairs,
        confusion_pairs,
        top_k=20,
    )

    result = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_val_accuracy": recorded,
        "verified_val_accuracy": val_acc,
        "train_samples": int(len(train_ids)),
        "val_samples": int(len(val_ids)),
        "descriptor_dim": MODEL_DIM,
        "margins": list(MARGINS),
        "train": train_summary,
        "validation": val_summary,
        "train_class_rows_weakest_first": class_summary(
            train["labels"],
            train["gap"],
            train["rivals"],
            MARGINS,
        ),
        "validation_class_rows_weakest_first": class_summary(
            val["labels"],
            val["gap"],
            val["rivals"],
            MARGINS,
        ),
        "top_train_prototype_rival_pairs": train_pairs,
        "top_validation_prototype_rival_pairs": val_pairs,
        "top_validation_model_confusion_pairs": confusion_pairs,
        "prototype_vs_model_confusion_overlap": overlap,
        "recommended_margin_rule": (
            "Choose the smallest margin that is active for roughly 10-40% "
            "of TRAIN samples on both protocols, while avoiding a margin that "
            "forces most already-correct prototype relations to remain active."
        ),
    }

    atomic_json(output, result)

    report(
        status,
        args.protocol,
        "Done",
        1,
        1,
        done=True,
    )

    print()
    print("=" * 124)
    print("PROTOTYPE MARGIN ACTIVATION")
    print("=" * 124)

    print(
        f"{'margin':>8s} "
        f"{'train active':>14s} "
        f"{'train loss':>12s} "
        f"{'val active':>12s} "
        f"{'val loss':>12s}"
    )
    print("-" * 66)

    for m in MARGINS:
        key = f"{m:.2f}"
        tr = train_summary["margins"][key]
        va = val_summary["margins"][key]
        print(
            f"{m:8.2f} "
            f"{100*tr['active_fraction']:13.3f}% "
            f"{tr['mean_loss']:12.6f} "
            f"{100*va['active_fraction']:11.3f}% "
            f"{va['mean_loss']:12.6f}"
        )

    print()
    print("PROTOTYPE GEOMETRY")
    print(
        "  train true sim:",
        f"{train_summary['true_similarity']['mean']:.6f}",
    )
    print(
        "  train wrong sim:",
        f"{train_summary['nearest_wrong_similarity']['mean']:.6f}",
    )
    print(
        "  train gap mean/median:",
        f"{train_summary['prototype_gap']['mean']:.6f}",
        "/",
        f"{train_summary['prototype_gap']['q50']:.6f}",
    )
    print(
        "  val true sim:",
        f"{val_summary['true_similarity']['mean']:.6f}",
    )
    print(
        "  val wrong sim:",
        f"{val_summary['nearest_wrong_similarity']['mean']:.6f}",
    )
    print(
        "  val gap mean/median:",
        f"{val_summary['prototype_gap']['mean']:.6f}",
        "/",
        f"{val_summary['prototype_gap']['q50']:.6f}",
    )

    print()
    print("TOP VALIDATION PROTOTYPE RIVAL PAIRS")
    for row in val_pairs[:15]:
        print(
            f"  A{row['action_a']:03d}<->A{row['action_b']:03d} "
            f"| selected={row['count']}"
        )

    print()
    print("TOP VALIDATION MODEL CONFUSION PAIRS")
    for row in confusion_pairs[:15]:
        print(
            f"  A{row['action_a']:03d}<->A{row['action_b']:03d} "
            f"| confusions={row['count']}"
        )

    print()
    print("PAIR OVERLAP")
    print(
        f"  top20 overlap: {overlap['overlap_count']}/"
        f"{overlap['confusion_pairs']} "
        f"({100*overlap['overlap_fraction_of_confusions']:.1f}%)"
    )

    print()
    print("WEAKEST VALIDATION CLASSES BY PROTOTYPE GAP")
    for row in result["validation_class_rows_weakest_first"][:15]:
        print(
            f"  A{row['action']:03d} | "
            f"mean_gap={row['mean_gap']:+.4f} | "
            f"top rival=A{row['top_rival_action']:03d} | "
            f"m=.10 active={100*row['active_fraction']['0.10']:.1f}%"
        )

    print()
    print("Saved:", output)
    print("=" * 124)


if __name__ == "__main__":
    main()

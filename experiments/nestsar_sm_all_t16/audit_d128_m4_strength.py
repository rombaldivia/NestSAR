#!/usr/bin/env python3
from __future__ import annotations

"""Causal M4 transformation-strength audit for trained NestSAR D128-MTS.

Read-only. Tests whether M4 over-specializes by scaling ONLY the learned
Spatial->M4 transformation at inference:

    M4_lambda(x) = x + lambda * (M4(x) - x)

lambda=1 is the trained network.
lambda<1 weakens the M4 transformation.
lambda>1 strengthens it.

The audit also applies stream-selective scaling and reports motion-energy
quartile effects, so we can tell whether any benefit comes specifically from
low-motion clips.
"""

import argparse
import json
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
    M4_HALF_LIVES,
    G4_HALF_LIVES,
)
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json


STREAMS = ("joint", "bone", "joint_motion", "bone_motion")
NUM_CLASSES = 120


def count_params(tree):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def load_checkpoint(path):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    for key in ("ema_params", "config", "model", "protocol", "epoch", "val_accuracy"):
        if key not in payload:
            raise ValueError(f"Checkpoint missing {key!r}")
    if payload["model"] != MODEL_NAME:
        raise ValueError(f"Expected {MODEL_NAME}, got {payload['model']!r}")
    params = payload["ema_params"]
    if count_params(params) != EXPECTED_PARAMS:
        raise ValueError(
            f"Parameter mismatch {count_params(params):,} != {EXPECTED_PARAMS:,}"
        )
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


def build_variants():
    # Same order as spatial_stack: joint, bone, joint_motion, bone_motion.
    rows = [
        ("dynamic_identity", (1.0, 1.0, 1.0, 1.0)),
        ("global_0p00", (0.0, 0.0, 0.0, 0.0)),
        ("global_0p50", (0.5, 0.5, 0.5, 0.5)),
        ("global_0p70", (0.7, 0.7, 0.7, 0.7)),
        ("global_0p80", (0.8, 0.8, 0.8, 0.8)),
        ("global_0p90", (0.9, 0.9, 0.9, 0.9)),
        ("global_0p95", (0.95, 0.95, 0.95, 0.95)),
        ("global_1p05", (1.05, 1.05, 1.05, 1.05)),
        ("global_1p10", (1.10, 1.10, 1.10, 1.10)),
        ("global_1p20", (1.20, 1.20, 1.20, 1.20)),

        ("joint_0p80", (0.8, 1.0, 1.0, 1.0)),
        ("bone_0p80", (1.0, 0.8, 1.0, 1.0)),
        ("joint_motion_0p80", (1.0, 1.0, 0.8, 1.0)),
        ("bone_motion_0p80", (1.0, 1.0, 1.0, 0.8)),

        ("pose_pair_0p80", (0.8, 0.8, 1.0, 1.0)),
        ("motion_pair_0p80", (1.0, 1.0, 0.8, 0.8)),
        ("pose_pair_0p90", (0.9, 0.9, 1.0, 1.0)),
        ("motion_pair_0p90", (1.0, 1.0, 0.9, 0.9)),
    ]
    return rows


def motion_energy_for_ids(dataset, ids, batch_size=2048):
    out = np.empty(len(ids), np.float32)
    for start in range(0, len(ids), batch_size):
        idx = ids[start:start + batch_size]
        x = np.asarray(dataset.canonical[idx], np.float32)
        tok = x.reshape(len(x), 16, 2, 25, 15)
        out[start:start + len(idx)] = np.mean(
            np.abs(tok[..., 3:15]), axis=(1, 2, 3, 4)
        )
    return out


def bucket_ids(values):
    edges = np.quantile(values, (0.25, 0.50, 0.75))
    bins = np.searchsorted(edges, values, side="right")
    return bins, edges


def per_class_accuracy(labels, correct):
    count = np.bincount(labels, minlength=NUM_CLASSES)
    good = np.bincount(labels[correct], minlength=NUM_CLASSES)
    acc = np.full(NUM_CLASSES, np.nan, np.float64)
    use = count > 0
    acc[use] = good[use] / count[use]
    return acc, count


def infer_exact(apply_exact, params, dataset, ids, batch_size):
    parts = []
    for start in range(0, len(ids), batch_size):
        idx = ids[start:start + batch_size]
        x = np.asarray(dataset.canonical[idx], np.float32)
        logits = jax.device_get(
            apply_exact(params, jax.device_put(x))
        )
        parts.append(np.asarray(logits).argmax(1))
    return np.concatenate(parts)


def infer_scaled(apply_scaled, params, dataset, ids, scales, batch_size):
    scale = jnp.asarray(scales, jnp.float32)
    parts = []
    for start in range(0, len(ids), batch_size):
        idx = ids[start:start + batch_size]
        x = np.asarray(dataset.canonical[idx], np.float32)
        logits = jax.device_get(
            apply_scaled(params, jax.device_put(x), scale)
        )
        parts.append(np.asarray(logits).argmax(1))
    return np.concatenate(parts)


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
    ids = np.asarray(dataset.splits[f"{args.protocol}_val"], np.int64)
    labels = np.asarray(dataset.labels[ids], np.int32)
    model = make_model(config)

    apply_exact = jax.jit(
        lambda p, x: model.apply({"params": p}, x, training=False)["logits"]
    )
    apply_scaled = jax.jit(
        lambda p, x, s: model.apply(
            {"params": p},
            x,
            training=False,
            m4_audit_scales=s,
        )["logits"]
    )

    variants = build_variants()

    out = Path(args.output)
    resume_dir = Path(str(out) + ".resume")
    resume_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_accuracy": float(payload["val_accuracy"]),
        "params": count_params(params),
        "val_samples": int(len(ids)),
        "variants": [{"name": n, "scales": list(s)} for n, s in variants],
    }
    meta_path = resume_dir / "meta.json"
    if meta_path.is_file():
        old = json.loads(meta_path.read_text())
        if old != meta:
            raise RuntimeError(
                f"Resume identity mismatch: {resume_dir}. "
                "Use a fresh output path."
            )
    else:
        atomic_json(meta_path, meta)

    print("=" * 122)
    print(f"{MODEL_NAME} | {args.protocol.upper()} M4 TRANSFORMATION-STRENGTH AUDIT")
    print("=" * 122)
    print("GPU:", jax.local_devices()[0])
    print(f"Checkpoint E{int(payload['epoch']):02d}")
    print(f"Recorded val: {100*float(payload['val_accuracy']):.4f}%")
    print(f"Val samples: {len(ids):,}")
    print()

    # Exact trained path: no interpolation arithmetic at all.
    exact_path = resume_dir / "00_exact_canonical.npy"
    if exact_path.is_file():
        exact_pred = np.load(exact_path, allow_pickle=False).astype(np.int32)
    else:
        exact_pred = infer_exact(
            apply_exact, params, dataset, ids, args.batch_size
        ).astype(np.int16)
        tmp = exact_path.with_suffix(".tmp.npy")
        np.save(tmp, exact_pred, allow_pickle=False)
        os.replace(tmp, exact_path)
        exact_pred = exact_pred.astype(np.int32)

    exact_acc = float(np.mean(exact_pred == labels))
    recorded = float(payload["val_accuracy"])
    if abs(exact_acc - recorded) > 5e-5:
        raise RuntimeError(
            f"Exact canonical mismatch {exact_acc:.8f} vs {recorded:.8f}"
        )

    motion = motion_energy_for_ids(dataset, ids)
    motion_bins, motion_edges = bucket_ids(motion)

    predictions = {}
    rows = {}

    for i, (name, scales) in enumerate(variants, 1):
        pred_path = resume_dir / f"{i:02d}_{name}.npy"
        reused = False

        if pred_path.is_file():
            try:
                pred = np.load(pred_path, allow_pickle=False)
                if pred.shape != (len(ids),):
                    raise ValueError(pred.shape)
                pred = pred.astype(np.int32, copy=False)
                reused = True
            except Exception:
                pred_path.unlink(missing_ok=True)
                reused = False

        if not reused:
            pred = infer_scaled(
                apply_scaled,
                params,
                dataset,
                ids,
                scales,
                args.batch_size,
            ).astype(np.int16)
            tmp = pred_path.with_suffix(".tmp.npy")
            np.save(tmp, pred, allow_pickle=False)
            os.replace(tmp, pred_path)
            pred = pred.astype(np.int32, copy=False)

        predictions[name] = pred
        acc = float(np.mean(pred == labels))
        rows[name] = {
            "accuracy": acc,
            "scales": list(scales),
            "reused": reused,
        }

        if args.status:
            atomic_json(
                args.status,
                {
                    "protocol": args.protocol,
                    "phase": "M4 strength",
                    "current": i,
                    "total": len(variants),
                    "variant": name,
                    "accuracy": acc,
                    "done": False,
                },
            )

        print(
            f"[{i:02d}/{len(variants):02d}] {name:26s} "
            f"{100*acc:8.4f}%"
            + (" [reused]" if reused else "")
        )

    identity_pred = predictions["dynamic_identity"]
    identity_acc = rows["dynamic_identity"]["accuracy"]
    identity_agreement = float(np.mean(identity_pred == exact_pred))
    identity_delta_pp = 100.0 * (identity_acc - exact_acc)

    if abs(identity_delta_pp) > 0.01 or identity_agreement < 0.9999:
        raise RuntimeError(
            "Dynamic lambda=1 path is not sufficiently equivalent to canonical: "
            f"delta={identity_delta_pp:+.5f} pp "
            f"agreement={100*identity_agreement:.5f}%"
        )

    base_pred = identity_pred
    base_correct = base_pred == labels
    base_class, class_count = per_class_accuracy(labels, base_correct)

    for name, _ in variants:
        pred = predictions[name]
        correct = pred == labels
        row = rows[name]
        row["delta_pp"] = 100.0 * (row["accuracy"] - identity_acc)
        row["prediction_agreement"] = float(np.mean(pred == base_pred))
        row["fixed_vs_identity"] = int(np.sum((~base_correct) & correct))
        row["broken_vs_identity"] = int(np.sum(base_correct & (~correct)))

        quartiles = []
        for q in range(4):
            use = motion_bins == q
            base_q = float(np.mean(base_correct[use]))
            acc_q = float(np.mean(correct[use]))
            quartiles.append({
                "quartile": q,
                "n": int(np.sum(use)),
                "motion_mean": float(np.mean(motion[use])),
                "identity_accuracy": base_q,
                "variant_accuracy": acc_q,
                "delta_pp": 100.0 * (acc_q - base_q),
            })
        row["motion_quartiles"] = quartiles

        cls, _ = per_class_accuracy(labels, correct)
        delta = cls - base_class
        finite = np.where(np.isfinite(delta))[0]
        gain = finite[np.argsort(-delta[finite])[:10]]
        loss = finite[np.argsort(delta[finite])[:10]]
        row["best_class_deltas"] = [
            {
                "ntu_action": int(c + 1),
                "n": int(class_count[c]),
                "identity_accuracy": float(base_class[c]),
                "variant_accuracy": float(cls[c]),
                "delta_pp": float(100.0 * delta[c]),
            }
            for c in gain
        ]
        row["worst_class_deltas"] = [
            {
                "ntu_action": int(c + 1),
                "n": int(class_count[c]),
                "identity_accuracy": float(base_class[c]),
                "variant_accuracy": float(cls[c]),
                "delta_pp": float(100.0 * delta[c]),
            }
            for c in loss
        ]

    ranking = sorted(
        (
            (rows[name]["delta_pp"], name)
            for name, _ in variants
            if name != "dynamic_identity"
        ),
        reverse=True,
    )

    result = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_recorded_accuracy": recorded,
        "exact_canonical_accuracy": exact_acc,
        "dynamic_identity_accuracy": identity_acc,
        "dynamic_identity_delta_pp": identity_delta_pp,
        "dynamic_identity_prediction_agreement": identity_agreement,
        "params": count_params(params),
        "val_samples": int(len(ids)),
        "stream_order": list(STREAMS),
        "motion_quartile_edges": [float(x) for x in motion_edges],
        "variants": rows,
        "ranking": [
            {"name": name, "delta_pp": float(delta)}
            for delta, name in ranking
        ],
        "decision_rules": {
            "m4_overstrength_supported": (
                "Supported when lambda<1 improves BOTH XSUB and XSET, "
                "preferably with a local optimum near 0.8-0.95, while "
                "lambda>1 is neutral or worse."
            ),
            "low_motion_specific": (
                "Supported when the lowest motion quartile gains more than "
                "the high-motion quartile under the same beneficial scaling."
            ),
            "stream_specific": (
                "Supported when shrinking one stream or one stream-pair "
                "improves both protocols more than global shrinking."
            ),
            "not_simple_strength": (
                "If lambda=1 remains optimal across both protocols, M4 is "
                "still the localization point but its failure is not simply "
                "an over-strong residual transformation."
            ),
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, result)

    best_delta, best_name = ranking[0]

    if args.status:
        atomic_json(
            args.status,
            {
                "protocol": args.protocol,
                "phase": "Done",
                "current": len(variants),
                "total": len(variants),
                "variant": best_name,
                "accuracy": exact_acc,
                "best_variant": best_name,
                "best_delta_pp": float(best_delta),
                "done": True,
            },
        )

    print()
    print("=" * 122)
    print("EXACT / DYNAMIC IDENTITY CHECK")
    print("=" * 122)
    print(f"exact canonical   : {100*exact_acc:.4f}%")
    print(f"dynamic lambda=1 : {100*identity_acc:.4f}%")
    print(f"delta            : {identity_delta_pp:+.5f} pp")
    print(f"prediction agree : {100*identity_agreement:.5f}%")

    print()
    print("=" * 122)
    print("RANKING")
    print("=" * 122)
    for delta, name in ranking:
        row = rows[name]
        q0 = row["motion_quartiles"][0]["delta_pp"]
        q3 = row["motion_quartiles"][3]["delta_pp"]
        print(
            f"{name:26s} "
            f"acc={100*row['accuracy']:8.4f}% | "
            f"delta={delta:+8.4f} pp | "
            f"lowMotion={q0:+7.3f} | "
            f"highMotion={q3:+7.3f} | "
            f"agree={100*row['prediction_agreement']:6.2f}%"
        )

    print()
    print("BEST:", best_name, f"{best_delta:+.4f} pp")
    print("Saved:", out)
    print("=" * 122)


if __name__ == "__main__":
    main()

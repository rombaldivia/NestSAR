#!/usr/bin/env python3
from __future__ import annotations

"""Direct counterfactual bottleneck audit for trained NestSAR D128-MTS.

Read-only audit.  It measures validation accuracy after controlled changes to
the already-cached T16 representation, without retraining or checkpoint edits.

The interventions are deliberately simple and interpretable:
  * pose / motion feature removal;
  * motion sub-channel removal (disp / phase / path);
  * second-person removal;
  * temporal token reversal / shuffling;
  * within-chunk and chunk-order shuffling;
  * temporal-mean repetition (removes token dynamics while preserving means);
  * alternating-token removal.

For each intervention we report:
  accuracy, delta from canonical, fixed/broken examples versus canonical,
  prediction agreement, and the classes with the largest accuracy losses/gains.
"""

import argparse
import json
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


FRAMES = 16
PERSONS = 2
JOINTS = 25
CHANNELS = 15
NUM_CLASSES = 120


def count_params(tree):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def load_checkpoint(path):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    for key in ("ema_params", "config", "model", "protocol", "epoch", "val_accuracy"):
        if key not in payload:
            raise ValueError(f"Checkpoint missing {key}: {path}")
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
        **{k: config[k] for k in (
            "spatial_dim", "model_dim", "dropout", "controller_dim", "fast_rank",
            "head_rank", "sm_residual_scale", "head_residual_scale",
        )},
        m4_half_lives=M4_HALF_LIVES,
        g4_half_lives=G4_HALF_LIVES,
    )


def reshape_tokens(x):
    return np.asarray(x, np.float32).reshape(
        len(x), FRAMES, PERSONS, JOINTS, CHANNELS
    )


def flatten_tokens(tok):
    return np.asarray(tok, np.float32).reshape(len(tok), FRAMES, -1)


def transform(x, name, rng):
    tok = reshape_tokens(x).copy()

    if name == "canonical":
        return flatten_tokens(tok)

    if name == "pose_off":
        tok[..., 0:3] = 0

    elif name == "motion_all_off":
        tok[..., 3:15] = 0

    elif name == "disp_off":
        tok[..., 3:6] = 0

    elif name == "phase_off":
        tok[..., 6:12] = 0

    elif name == "path_off":
        tok[..., 12:15] = 0

    elif name == "person2_off":
        tok[:, :, 1, :, :] = 0

    elif name == "reverse_token_order":
        tok = tok[:, ::-1]

    elif name == "shuffle_token_order":
        perm = rng.permutation(FRAMES)
        tok = tok[:, perm]

    elif name == "shuffle_within_chunks":
        # Same deterministic permutation is used for every clip so this tests
        # temporal order rather than injecting per-sample stochastic noise.
        order = []
        for base in range(0, FRAMES, 4):
            local = np.arange(base, base + 4)
            local = local[rng.permutation(4)]
            order.extend(local.tolist())
        tok = tok[:, np.asarray(order, np.int32)]

    elif name == "shuffle_chunks":
        chunks = tok.reshape(len(tok), 4, 4, PERSONS, JOINTS, CHANNELS)
        perm = rng.permutation(4)
        tok = chunks[:, perm].reshape(
            len(tok), FRAMES, PERSONS, JOINTS, CHANNELS
        )

    elif name == "temporal_mean_repeat":
        mean = tok.mean(axis=1, keepdims=True)
        tok = np.repeat(mean, FRAMES, axis=1)

    elif name == "alternate_tokens_off":
        tok[:, 1::2] = 0

    else:
        raise ValueError(name)

    return flatten_tokens(tok)


def per_class_accuracy(labels, correct):
    labels = np.asarray(labels, np.int32)
    correct = np.asarray(correct, bool)
    rows = np.full(NUM_CLASSES, np.nan, np.float64)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    good = np.bincount(labels[correct], minlength=NUM_CLASSES)
    use = counts > 0
    rows[use] = good[use] / counts[use]
    return rows, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True, choices=("xsub", "xset"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

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
    infer = jax.jit(
        lambda p, x: model.apply({"params": p}, x, training=False)["logits"]
    )

    names = (
        "canonical",
        "pose_off",
        "motion_all_off",
        "disp_off",
        "phase_off",
        "path_off",
        "person2_off",
        "reverse_token_order",
        "shuffle_token_order",
        "shuffle_within_chunks",
        "shuffle_chunks",
        "temporal_mean_repeat",
        "alternate_tokens_off",
    )

    predictions = {}
    accuracies = {}

    # Use one fixed permutation seed per intervention so the audit is exactly
    # reproducible across reruns.
    for name_i, name in enumerate(names):
        pred_parts = []
        rng = np.random.default_rng(20261004 + name_i)

        for start in range(0, len(ids), args.batch_size):
            idx = ids[start:start + args.batch_size]
            x = np.asarray(dataset.canonical[idx], np.float32)
            xx = transform(x, name, rng)
            logits = jax.device_get(infer(params, jax.device_put(xx)))
            pred_parts.append(np.asarray(logits).argmax(1))

        pred = np.concatenate(pred_parts)
        predictions[name] = pred
        accuracies[name] = float(np.mean(pred == labels))

    base_pred = predictions["canonical"]
    base_correct = base_pred == labels
    base_acc = accuracies["canonical"]
    base_class, class_counts = per_class_accuracy(labels, base_correct)

    variants = {}

    for name in names[1:]:
        pred = predictions[name]
        correct = pred == labels
        acc = accuracies[name]
        cls_acc, _ = per_class_accuracy(labels, correct)
        delta = cls_acc - base_class
        order_loss = np.argsort(np.nan_to_num(delta, nan=np.inf))
        order_gain = np.argsort(-np.nan_to_num(delta, nan=-np.inf))

        variants[name] = {
            "accuracy": acc,
            "delta_pp": 100.0 * (acc - base_acc),
            "prediction_agreement": float(np.mean(pred == base_pred)),
            "fixed_vs_canonical": int(np.sum((~base_correct) & correct)),
            "broken_vs_canonical": int(np.sum(base_correct & (~correct))),
            "worst_class_deltas": [
                {
                    "ntu_action": int(c + 1),
                    "n": int(class_counts[c]),
                    "canonical_accuracy": float(base_class[c]),
                    "variant_accuracy": float(cls_acc[c]),
                    "delta_pp": float(100.0 * delta[c]),
                }
                for c in order_loss[:10]
                if np.isfinite(delta[c])
            ],
            "best_class_deltas": [
                {
                    "ntu_action": int(c + 1),
                    "n": int(class_counts[c]),
                    "canonical_accuracy": float(base_class[c]),
                    "variant_accuracy": float(cls_acc[c]),
                    "delta_pp": float(100.0 * delta[c]),
                }
                for c in order_gain[:10]
                if np.isfinite(delta[c])
            ],
        }

    result = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_recorded_val_accuracy": float(payload["val_accuracy"]),
        "verified_canonical_accuracy": base_acc,
        "params": count_params(params),
        "val_samples": int(len(ids)),
        "variants": variants,
        "interpretation": {
            "motion_bottleneck": (
                "Large loss from motion_all_off / phase_off / path_off means those "
                "channels carry essential discriminative information. Small losses mean "
                "the model is underusing them."
            ),
            "temporal_order_bottleneck": (
                "Small losses under reverse/shuffle/mean-repeat imply the model is not "
                "exploiting temporal order strongly; very large losses imply strong "
                "temporal dependence and possible T16/order sensitivity."
            ),
            "interaction_bottleneck": (
                "Large person2_off loss, especially concentrated in weak classes, "
                "implicates two-person interaction representation."
            ),
            "t16_redundancy": (
                "Small alternate_tokens_off loss suggests 16 tokens contain substantial "
                "redundancy; a large loss suggests temporal resolution is still valuable."
            ),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, result)

    print("=" * 118)
    print(f"{MODEL_NAME} | {args.protocol.upper()} DIRECT COUNTERFACTUAL AUDIT")
    print("=" * 118)
    print(
        f"checkpoint E{int(payload['epoch']):02d} | "
        f"recorded={100*float(payload['val_accuracy']):.4f}% | "
        f"verified={100*base_acc:.4f}%"
    )
    print()
    print(f"{'intervention':24s} {'accuracy':>10s} {'delta':>10s} {'agree':>10s} {'fixed':>8s} {'broken':>8s}")
    for name in names[1:]:
        row = variants[name]
        print(
            f"{name:24s} "
            f"{100*row['accuracy']:9.3f}% "
            f"{row['delta_pp']:+9.3f} "
            f"{100*row['prediction_agreement']:9.2f}% "
            f"{row['fixed_vs_canonical']:8d} "
            f"{row['broken_vs_canonical']:8d}"
        )

    print("\nSaved:", out)
    print("=" * 118)


if __name__ == "__main__":
    main()

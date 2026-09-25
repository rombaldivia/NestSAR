#!/usr/bin/env python3
from __future__ import annotations

"""Read-only temporal causality audit for trained NestSAR R4.

The checkpoint is never modified. The audit asks whether NestSAR actually uses
temporal order/dynamics or mainly relies on pose/setup signatures.

Causal validation variants:
  clean
  reverse_raw         : reverse raw skeleton time, then recompute exact T16 features
  local_shuffle_raw   : deterministic shuffle inside four raw-time blocks, then recompute
  block_permute_raw   : permute four raw-time blocks, then recompute
  pose_only           : keep canonical pose channels 0:3, zero motion channels 3:15
  motion_only         : zero pose channels 0:3, keep motion channels 3:15
  motion_misaligned   : keep pose fixed, roll motion channels across T by 3 segments

For localization, clean TRAIN representations build nearest-centroid classifiers
using an order-aware temporal summary at:
  input -> spatial -> M4 -> router -> G4 -> descriptor

Every corrupted validation representation is evaluated against those SAME
clean-train centroids. No validation labels are used to fit a probe.
"""

import argparse
import json
import math
from contextlib import closing
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from experiments.nestsar_sm_all_t16 import preprocessing_corrected as pp
from experiments.nestsar_sm_all_t16.streaming import worker as train_worker
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json

NUM_CLASSES = 120
EXPECTED_PARAMS = 1_831_932
STAGES = ("input", "spatial", "m4", "router", "g4", "descriptor")
VARIANTS = (
    "clean",
    "reverse_raw",
    "local_shuffle_raw",
    "block_permute_raw",
    "pose_only",
    "motion_only",
    "motion_misaligned",
)


def report(path: Path, protocol: str, phase: str, current: int, total: int, **extra):
    payload = {
        "protocol": protocol,
        "phase": phase,
        "current": int(current),
        "total": int(max(total, 1)),
    }
    payload.update(extra)
    atomic_json(path, payload)


def tree_count(tree):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def load_checkpoint(path: Path):
    payload = serialization.msgpack_restore(path.read_bytes())
    if not isinstance(payload, dict) or "ema_params" not in payload or "config" not in payload:
        raise ValueError(f"Expected best checkpoint with ema_params/config: {path}")
    return payload, payload["ema_params"], dict(payload["config"])


def row_norm(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def temporal_summary(seq):
    """Order-aware but compact summary; same definition at every temporal stage."""
    seq = np.asarray(seq, np.float32)
    if seq.ndim != 3:
        raise ValueError(f"Expected [B,T,D], got {seq.shape}")

    mean = seq.mean(axis=1)
    std = seq.std(axis=1)
    endpoint = seq[:, -1] - seq[:, 0]

    if seq.shape[1] > 1:
        adjacent = np.abs(np.diff(seq, axis=1)).mean(axis=1)
    else:
        adjacent = np.zeros_like(mean)

    return np.concatenate(
        [mean, std, endpoint, adjacent],
        axis=1,
    )


def stage_features(x, out):
    b = len(x)

    seq = {
        "input": np.asarray(x, np.float32),
        "spatial": np.asarray(out["spatial_stack"], np.float32).reshape(b, 16, -1),
        "m4": np.asarray(out["frame_stack"], np.float32).reshape(b, 16, -1),
        "router": np.asarray(out["mixed_frame_stack"], np.float32).reshape(b, 16, -1),
        "g4": (
            np.asarray(out["chunk_states"], np.float32)
            .transpose(0, 2, 1, 3)
            .reshape(b, 4, -1)
        ),
    }

    features = {
        stage: temporal_summary(seq[stage])
        for stage in ("input", "spatial", "m4", "router", "g4")
    }
    features["descriptor"] = np.asarray(
        out["descriptors"], np.float32
    ).reshape(b, -1)

    temporal_energy = {}
    for stage, values in seq.items():
        if values.shape[1] > 1:
            temporal_energy[stage] = np.mean(
                np.square(np.diff(values, axis=1), dtype=np.float64),
                axis=(1, 2),
            )
        else:
            temporal_energy[stage] = np.zeros(b, np.float64)

    return features, temporal_energy


class CentroidStats:
    def __init__(self, dim):
        self.dim = int(dim)
        self.class_sum = np.zeros((NUM_CLASSES, self.dim), np.float64)
        self.counts = np.zeros(NUM_CLASSES, np.int64)

    def add(self, x, y):
        z = row_norm(x)
        self.counts += np.bincount(y, minlength=NUM_CLASSES)
        np.add.at(self.class_sum, y, z)

    def finalize(self):
        if np.any(self.counts == 0):
            missing = np.flatnonzero(self.counts == 0).tolist()
            raise RuntimeError(f"Training centroids missing classes: {missing}")
        centroids = self.class_sum / self.counts[:, None]
        return row_norm(centroids)


def ncm_predict(x, centroid_unit):
    z = row_norm(x)
    return (z @ centroid_unit.T).argmax(axis=1)


def raw_variant(raw, variant, sample_id, seed):
    raw = np.asarray(raw, np.float32)

    if variant == "reverse_raw":
        return raw[::-1].copy()

    if variant == "block_permute_raw":
        blocks = np.array_split(np.arange(len(raw), dtype=np.int64), 4)
        order = (2, 0, 3, 1)
        indices = np.concatenate([blocks[i] for i in order])
        return raw[indices].copy()

    if variant == "local_shuffle_raw":
        blocks = np.array_split(np.arange(len(raw), dtype=np.int64), 4)
        shuffled = []
        for block_idx, block in enumerate(blocks):
            block = block.copy()
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [int(seed), int(sample_id), int(block_idx)]
                )
            )
            rng.shuffle(block)
            shuffled.append(block)
        indices = np.concatenate(shuffled)
        return raw[indices].copy()

    raise ValueError(variant)


def canonical_variant(dataset, ids, variant, seed):
    ids = np.asarray(ids, np.int64)

    if variant == "clean":
        return np.asarray(dataset.canonical[ids], np.float32)

    if variant in {"reverse_raw", "local_shuffle_raw", "block_permute_raw"}:
        out = np.empty((len(ids), pp.FRAMES, pp.FEATURES), np.float32)
        for j, sample_id in enumerate(ids):
            raw = dataset.sample(int(sample_id))
            transformed = raw_variant(
                raw,
                variant,
                int(sample_id),
                seed,
            )
            out[j] = pp.features(transformed)
        return out

    x = np.asarray(dataset.canonical[ids], np.float32).copy()
    tok = x.reshape(
        len(ids),
        pp.FRAMES,
        pp.PERSONS,
        pp.JOINTS,
        pp.TOKEN_CHANNELS,
    )

    if variant == "pose_only":
        tok[..., 3:15] = 0.0
        return tok.reshape(len(ids), pp.FRAMES, pp.FEATURES)

    if variant == "motion_only":
        tok[..., 0:3] = 0.0
        return tok.reshape(len(ids), pp.FRAMES, pp.FEATURES)

    if variant == "motion_misaligned":
        motion = tok[..., 3:15].copy()
        tok[..., 3:15] = np.roll(motion, shift=3, axis=1)
        return tok.reshape(len(ids), pp.FRAMES, pp.FEATURES)

    raise ValueError(variant)


def softmax_stats(logits):
    logits = np.asarray(logits, np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-300)
    confidence = probs.max(axis=1)
    entropy = -np.sum(
        probs * np.log(np.maximum(probs, 1e-300)),
        axis=1,
    )
    return confidence, entropy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--shuffle-seed", type=int, default=20260924)
    args = p.parse_args()

    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise RuntimeError(
            f"Expected exactly one isolated GPU; "
            f"backend={jax.default_backend()}, devices={jax.local_devices()}"
        )

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    status = outdir / "status.json"

    payload, params, config = load_checkpoint(Path(args.checkpoint))

    if tree_count(params) != EXPECTED_PARAMS:
        raise RuntimeError(
            f"Parameter mismatch: {tree_count(params)} != {EXPECTED_PARAMS}"
        )
    if int(config.get("fast_rank", -1)) != 4:
        raise RuntimeError("Temporal audit expects trained Fast Rank=4 checkpoint")

    dataset = Dataset(args.cache)
    train_ids = dataset.splits[f"{args.protocol}_train"]
    val_ids = dataset.splits[f"{args.protocol}_val"]

    model = train_worker.make_model(config)
    forward = jax.jit(
        lambda p, x: model.apply(
            {"params": p},
            x,
            training=False,
        )
    )

    # ------------------------------------------------------------------
    # FIT CLEAN TRAIN CENTROIDS
    # ------------------------------------------------------------------

    centroid_stats = {}
    train_steps = math.ceil(len(train_ids) / args.batch_size)

    for bi, start in enumerate(range(0, len(train_ids), args.batch_size)):
        ids = np.asarray(
            train_ids[start:start + args.batch_size],
            np.int64,
        )
        x = np.asarray(dataset.canonical[ids], np.float32)
        y = np.asarray(dataset.labels[ids], np.int32)

        output = jax.device_get(
            forward(
                params,
                jax.device_put(x),
            )
        )

        features, _ = stage_features(x, output)

        if not centroid_stats:
            centroid_stats = {
                stage: CentroidStats(features[stage].shape[1])
                for stage in STAGES
            }

        for stage in STAGES:
            centroid_stats[stage].add(
                features[stage],
                y,
            )

        if bi % 5 == 0 or bi + 1 == train_steps:
            report(
                status,
                args.protocol,
                "Clean TRAIN temporal centroids",
                bi + 1,
                train_steps,
            )

    centroids = {
        stage: centroid_stats[stage].finalize()
        for stage in STAGES
    }

    # ------------------------------------------------------------------
    # EVALUATE CLEAN + TEMPORAL COUNTERFACTUALS
    # ------------------------------------------------------------------

    variant_results = {}
    clean_predictions = None
    clean_correct = None
    clean_stage_accuracy = None
    clean_temporal_energy = None

    for variant_idx, variant in enumerate(VARIANTS):
        steps = math.ceil(len(val_ids) / args.batch_size)

        model_correct = 0
        seen = 0
        confidence_sum = 0.0
        entropy_sum = 0.0

        stage_correct = {
            stage: 0
            for stage in STAGES
        }

        energy_sum = {
            stage: 0.0
            for stage in ("input", "spatial", "m4", "router", "g4")
        }

        variant_predictions = np.empty(len(val_ids), np.int16)
        variant_correct = np.empty(len(val_ids), bool)

        cursor = 0

        for bi, start in enumerate(range(0, len(val_ids), args.batch_size)):
            ids = np.asarray(
                val_ids[start:start + args.batch_size],
                np.int64,
            )
            y = np.asarray(dataset.labels[ids], np.int32)

            x = canonical_variant(
                dataset,
                ids,
                variant,
                args.shuffle_seed,
            )

            output = jax.device_get(
                forward(
                    params,
                    jax.device_put(x),
                )
            )

            logits = np.asarray(output["logits"])
            pred = logits.argmax(axis=1)
            correct = pred == y

            n = len(y)
            variant_predictions[cursor:cursor + n] = pred.astype(np.int16)
            variant_correct[cursor:cursor + n] = correct

            model_correct += int(correct.sum())
            seen += n

            confidence, entropy = softmax_stats(logits)
            confidence_sum += float(confidence.sum())
            entropy_sum += float(entropy.sum())

            features, temporal_energy = stage_features(
                x,
                output,
            )

            for stage in STAGES:
                ncm_pred = ncm_predict(
                    features[stage],
                    centroids[stage],
                )
                stage_correct[stage] += int(
                    np.sum(ncm_pred == y)
                )

            for stage in energy_sum:
                energy_sum[stage] += float(
                    temporal_energy[stage].sum()
                )

            cursor += n

            if bi % 5 == 0 or bi + 1 == steps:
                report(
                    status,
                    args.protocol,
                    f"Temporal test {variant}",
                    bi + 1,
                    steps,
                    variant_index=variant_idx,
                    variant_total=len(VARIANTS),
                    val_acc=model_correct / max(seen, 1),
                )

        if seen != len(val_ids):
            raise RuntimeError(
                f"{variant}: validation accounting mismatch "
                f"{seen} != {len(val_ids)}"
            )

        model_accuracy = model_correct / seen
        stage_accuracy = {
            stage: stage_correct[stage] / seen
            for stage in STAGES
        }
        energy_mean = {
            stage: energy_sum[stage] / seen
            for stage in energy_sum
        }

        if variant == "clean":
            clean_predictions = variant_predictions.copy()
            clean_correct = variant_correct.copy()
            clean_stage_accuracy = dict(stage_accuracy)
            clean_temporal_energy = dict(energy_mean)

            fixed = 0
            broken = 0
            agreement = 1.0
        else:
            fixed = int(
                np.sum(
                    (~clean_correct)
                    & variant_correct
                )
            )
            broken = int(
                np.sum(
                    clean_correct
                    & (~variant_correct)
                )
            )
            agreement = float(
                np.mean(
                    variant_predictions
                    == clean_predictions
                )
            )

        result = {
            "variant": variant,
            "samples": seen,
            "model_accuracy": model_accuracy,
            "model_delta_pp_vs_clean": (
                0.0
                if variant == "clean"
                else 100.0
                * (
                    model_accuracy
                    - variant_results["clean"]["model_accuracy"]
                )
            ),
            "prediction_agreement_with_clean": agreement,
            "clean_wrong_variant_correct_fixed": fixed,
            "clean_correct_variant_wrong_broken": broken,
            "mean_max_softmax_confidence": confidence_sum / seen,
            "mean_prediction_entropy": entropy_sum / seen,
            "stage_temporal_ncm_accuracy": stage_accuracy,
            "stage_temporal_ncm_delta_pp_vs_clean": {
                stage: (
                    0.0
                    if variant == "clean"
                    else 100.0
                    * (
                        stage_accuracy[stage]
                        - clean_stage_accuracy[stage]
                    )
                )
                for stage in STAGES
            },
            "stage_adjacent_temporal_energy": energy_mean,
            "stage_temporal_energy_ratio_vs_clean": {
                stage: (
                    1.0
                    if variant == "clean"
                    else energy_mean[stage]
                    / max(clean_temporal_energy[stage], 1e-12)
                )
                for stage in energy_mean
            },
        }

        variant_results[variant] = result
        atomic_json(
            outdir / f"{variant}.json",
            result,
        )

    # ------------------------------------------------------------------
    # AUTOMATIC TEMPORAL DIAGNOSIS
    # ------------------------------------------------------------------

    order_variants = (
        "reverse_raw",
        "local_shuffle_raw",
        "block_permute_raw",
    )

    mean_order_drop_pp = float(
        np.mean(
            [
                -variant_results[v]["model_delta_pp_vs_clean"]
                for v in order_variants
            ]
        )
    )

    stage_mean_order_drop_pp = {}
    for stage in STAGES:
        stage_mean_order_drop_pp[stage] = float(
            np.mean(
                [
                    -variant_results[v][
                        "stage_temporal_ncm_delta_pp_vs_clean"
                    ][stage]
                    for v in order_variants
                ]
            )
        )

    # Positive attenuation means order-discriminative damage gets smaller after
    # this transition, suggesting the later stage is less sensitive to order.
    transition_attenuation_pp = {}
    for left, right in zip(STAGES[:-1], STAGES[1:]):
        transition_attenuation_pp[f"{left}_to_{right}"] = (
            stage_mean_order_drop_pp[left]
            - stage_mean_order_drop_pp[right]
        )

    strongest_attenuation = max(
        transition_attenuation_pp,
        key=lambda key: transition_attenuation_pp[key],
    )

    pose_only_drop = -variant_results["pose_only"]["model_delta_pp_vs_clean"]
    motion_only_drop = -variant_results["motion_only"]["model_delta_pp_vs_clean"]
    misalignment_drop = -variant_results["motion_misaligned"]["model_delta_pp_vs_clean"]

    summary = {
        "audit": "NestSAR R4 temporal causality audit v1",
        "read_only": True,
        "protocol": args.protocol,
        "checkpoint": str(args.checkpoint),
        "checkpoint_val_accuracy": float(payload["val_accuracy"]),
        "recomputed_clean_val_accuracy": variant_results["clean"]["model_accuracy"],
        "params": tree_count(params),
        "variants": variant_results,
        "temporal_order_dependence": {
            "order_variants": list(order_variants),
            "mean_model_accuracy_drop_pp": mean_order_drop_pp,
            "stage_mean_order_ncm_drop_pp": stage_mean_order_drop_pp,
            "transition_order_sensitivity_attenuation_pp": transition_attenuation_pp,
            "strongest_order_sensitivity_attenuation_transition": strongest_attenuation,
        },
        "channel_dependence": {
            "pose_only_accuracy_drop_pp": pose_only_drop,
            "motion_only_accuracy_drop_pp": motion_only_drop,
            "pose_motion_misalignment_accuracy_drop_pp": misalignment_drop,
        },
        "interpretation_rules": [
            "Large drops for reverse/local-shuffle/block-permute mean final predictions depend on temporal ordering.",
            "If an early-stage temporal NCM drop is large but becomes much smaller after a transition, that transition attenuates order-discriminative information.",
            "Small pose-motion misalignment drop means precise alignment between pose and precomputed motion channels contributes little.",
            "Pose-only and motion-only are ablations, not matched standalone models; compare their drops rather than treating either accuracy as a fair architecture score.",
            "Raw-order variants recompute exact preprocessing after reordering, avoiding stale motion channels.",
        ],
    }

    atomic_json(
        outdir / "temporal_causality_summary.json",
        summary,
    )

    report(
        status,
        args.protocol,
        "Done",
        1,
        1,
        done=True,
        val_acc=variant_results["clean"]["model_accuracy"],
    )

    print("=" * 120)
    print(f"{args.protocol.upper()} TEMPORAL CAUSALITY AUDIT COMPLETE")
    print("=" * 120)
    print(json.dumps(summary, indent=2))
    print("REPORT:", outdir / "temporal_causality_summary.json")


if __name__ == "__main__":
    main()

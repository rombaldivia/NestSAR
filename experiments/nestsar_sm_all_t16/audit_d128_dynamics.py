#!/usr/bin/env python3
from __future__ import annotations

"""Deep bottleneck audit for trained NestSAR D128-MTS.

Diagnoses four competing bottleneck hypotheses:
  H1 input/T16 compression;
  H2 M4 temporal-memory dynamics;
  H3 G4 over-specialization;
  H4 remaining capacity/representation collapse.

Read-only: never mutates the checkpoint.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.traverse_util import flatten_dict

from experiments.nestsar_sm_all_t16 import preprocessing_corrected as pp
from experiments.nestsar_sm_all_t16.audit_generalization_localization import (
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


STREAMS = ("joint", "bone", "joint_motion", "bone_motion")
STAGES = ("input", "spatial", "m4", "router", "g4", "descriptor")


def count_params(tree):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def make_model(config):
    return NestSARParallelT16(
        **{k: config[k] for k in (
            "spatial_dim", "model_dim", "dropout", "controller_dim", "fast_rank",
            "head_rank", "sm_residual_scale", "head_residual_scale",
        )},
        m4_half_lives=M4_HALF_LIVES,
        g4_half_lives=G4_HALF_LIVES,
    )


def load_checkpoint(path):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    for key in ("ema_params", "config", "model", "protocol", "epoch", "val_accuracy"):
        if key not in payload:
            raise ValueError(f"Checkpoint missing {key}")
    if payload["model"] != MODEL_NAME:
        raise ValueError(f"Expected {MODEL_NAME}, got {payload['model']!r}")
    params = payload["ema_params"]
    if count_params(params) != EXPECTED_PARAMS:
        raise ValueError(
            f"Parameter mismatch {count_params(params):,} != {EXPECTED_PARAMS:,}"
        )
    return payload, params, dict(payload["config"])


def find_leaf(flat, path_text):
    matches = []
    for path, value in flat.items():
        text = "/".join(map(str, path))
        if text.endswith(path_text):
            matches.append((text, np.asarray(value)))
    if len(matches) != 1:
        raise ValueError(f"Expected one leaf ending {path_text!r}, found {[m[0] for m in matches]}")
    return matches[0][1]


class GateAccumulator:
    def __init__(self, stage, direction, half_lives, dim, streams=4):
        self.stage = stage
        self.direction = direction
        self.half_lives = tuple(half_lives)
        self.dim = int(dim)
        self.streams = int(streams)
        if self.dim % len(self.half_lives):
            raise ValueError("Width must divide evenly across timescales")
        self.group = self.dim // len(self.half_lives)
        self.count = np.zeros((streams, len(self.half_lives)), np.int64)
        self.sum_a = np.zeros_like(self.count, np.float64)
        self.sum_log_a = np.zeros_like(self.count, np.float64)
        self.sum_a2 = np.zeros_like(self.count, np.float64)
        self.low = np.zeros_like(self.count, np.int64)
        self.high = np.zeros_like(self.count, np.int64)

    def add(self, a):
        # a [B,T,S,D]
        a = np.asarray(a, np.float64)
        a = np.clip(a, 1e-7, 1 - 1e-7)
        if a.ndim != 4 or a.shape[2:] != (self.streams, self.dim):
            raise ValueError(f"Unexpected gate shape {a.shape}")
        for s in range(self.streams):
            for g in range(len(self.half_lives)):
                z = a[:, :, s, g*self.group:(g+1)*self.group].reshape(-1)
                self.count[s, g] += len(z)
                self.sum_a[s, g] += z.sum()
                self.sum_log_a[s, g] += np.log(z).sum()
                self.sum_a2[s, g] += np.square(z).sum()
                self.low[s, g] += int(np.sum(z < 0.10))
                self.high[s, g] += int(np.sum(z > 0.90))

    def result(self):
        rows = {}
        for s, stream in enumerate(STREAMS):
            groups = []
            for g, configured_h in enumerate(self.half_lives):
                n = max(int(self.count[s, g]), 1)
                mean = self.sum_a[s, g] / n
                geom = float(np.exp(self.sum_log_a[s, g] / n))
                var = max(self.sum_a2[s, g] / n - mean*mean, 0.0)
                half = float(np.log(0.5) / np.log(np.clip(geom, 1e-7, 1-1e-7)))
                groups.append({
                    "configured_half_life": float(configured_h),
                    "mean_retention": float(mean),
                    "std_retention": float(np.sqrt(var)),
                    "geometric_mean_retention": geom,
                    "effective_half_life_tokens": half,
                    "fraction_a_lt_0p1": float(self.low[s, g] / n),
                    "fraction_a_gt_0p9": float(self.high[s, g] / n),
                })
            rows[stream] = groups
        effective = [
            x["effective_half_life_tokens"]
            for groups in rows.values() for x in groups
        ]
        return {
            "stage": self.stage,
            "direction": self.direction,
            "streams": rows,
            "effective_half_life_min": float(np.min(effective)),
            "effective_half_life_max": float(np.max(effective)),
            "effective_half_life_ratio_max_min": float(np.max(effective) / max(np.min(effective), 1e-12)),
        }


def gates_for(x, kernel, bias):
    # x [B,T,S,D], kernel [S,D,D], bias [S,D]
    return jax.nn.sigmoid(
        jnp.einsum("btsd,sde->btse", x, kernel) + bias[None, None, :, :]
    )


def representation_rank(rows):
    x = np.asarray(rows, np.float32)
    if len(x) < 2:
        return {}
    x = x - x.mean(axis=0, keepdims=True)
    # covariance via singular values is cheaper/stabler for sampled matrices.
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False).astype(np.float64)
    eig = np.square(s) / max(len(x) - 1, 1)
    total = float(eig.sum())
    if total <= 0:
        return {"participation_ratio": 0.0, "rank90": 0, "top1_variance_fraction": 0.0}
    pr = total * total / max(float(np.square(eig).sum()), 1e-12)
    c = np.cumsum(eig) / total
    return {
        "participation_ratio": float(pr),
        "rank90": int(np.searchsorted(c, 0.90) + 1),
        "rank95": int(np.searchsorted(c, 0.95) + 1),
        "top1_variance_fraction": float(eig[0] / total),
        "dimension": int(x.shape[1]),
        "samples": int(len(x)),
    }


def bucket_accuracy(values, correct, quantiles=(0.25, 0.50, 0.75)):
    values = np.asarray(values, np.float64)
    correct = np.asarray(correct, bool)
    edges = np.quantile(values, quantiles)
    bins = np.searchsorted(edges, values, side="right")
    rows = []
    for b in range(len(edges)+1):
        use = bins == b
        if not np.any(use):
            continue
        rows.append({
            "bucket": int(b),
            "n": int(use.sum()),
            "value_mean": float(values[use].mean()),
            "value_min": float(values[use].min()),
            "value_max": float(values[use].max()),
            "accuracy": float(correct[use].mean()),
        })
    return {"edges": [float(x) for x in edges], "buckets": rows}


def yaw_rotate(x, degrees):
    valid = pp.raw_valid(x)
    theta = np.deg2rad(float(degrees))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    return np.where(valid[..., None], x @ rot.T, 0).astype(np.float32)


def center_crop(x, ratio):
    n = len(x)
    keep = max(1, int(round(n * float(ratio))))
    keep = min(keep, n)
    start = max((n - keep) // 2, 0)
    return x[start:start+keep]


def robustness_features(dataset, index, condition):
    raw = dataset.sample(int(index))
    kind, value = condition
    if kind == "rotation":
        raw = yaw_rotate(raw, value)
    elif kind == "crop":
        raw = center_crop(raw, value)
    elif kind == "crop_rot":
        ratio, degrees = value
        raw = center_crop(raw, ratio)
        raw = yaw_rotate(raw, degrees)
    else:
        raise ValueError(condition)
    return pp.features(raw)


def evaluate_robustness(model, params, dataset, ids, batch_size):
    conditions = [
        ("rotation", 10.0),
        ("rotation", 20.0),
        ("rotation", -20.0),
        ("crop", 0.90),
        ("crop", 0.80),
        ("crop_rot", (0.85, 20.0)),
    ]
    apply_logits = jax.jit(
        lambda p, x: model.apply({"params": p}, x, training=False)["logits"]
    )
    canonical = np.asarray(dataset.canonical[ids], np.float32)
    labels = np.asarray(dataset.labels[ids], np.int32)

    def infer(arr):
        pred = []
        for start in range(0, len(arr), batch_size):
            logits = jax.device_get(apply_logits(params, jax.device_put(arr[start:start+batch_size])))
            pred.append(np.asarray(logits).argmax(1))
        return np.concatenate(pred)

    base_pred = infer(canonical)
    base_acc = float(np.mean(base_pred == labels))
    result = {"canonical_accuracy": base_acc, "samples": int(len(ids)), "conditions": {}}

    for kind, value in conditions:
        aug = np.empty_like(canonical)
        for j, idx in enumerate(ids):
            aug[j] = robustness_features(dataset, int(idx), (kind, value))
        pred = infer(aug)
        key = f"{kind}:{value}"
        result["conditions"][key] = {
            "accuracy": float(np.mean(pred == labels)),
            "delta_from_canonical_pp": 100.0 * (float(np.mean(pred == labels)) - base_acc),
            "prediction_agreement": float(np.mean(pred == base_pred)),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=("xsub", "xset"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--rank-samples", type=int, default=8192)
    ap.add_argument("--robustness-samples", type=int, default=4096)
    args = ap.parse_args()

    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise RuntimeError(
            f"Expected exactly one isolated GPU, got {jax.default_backend()} {jax.local_devices()}"
        )

    payload, params, config = load_checkpoint(args.checkpoint)
    if payload["protocol"] != args.protocol:
        raise ValueError(f"Checkpoint protocol {payload['protocol']} != {args.protocol}")

    dataset = Dataset(args.cache)
    val_ids = np.asarray(dataset.splits[f"{args.protocol}_val"], np.int64)
    labels = np.asarray(dataset.labels[val_ids], np.int32)

    model = make_model(config)
    forward = jax.jit(lambda p, x: model.apply({"params": p}, x, training=False))

    flat = flatten_dict(params)
    m4 = {}
    g4 = {}
    for direction in ("fwd", "bwd"):
        m4[direction] = (
            find_leaf(flat, f"frame_memory_group/base_memory/{direction}/forget/kernel"),
            find_leaf(flat, f"frame_memory_group/base_memory/{direction}/forget/bias"),
        )
        g4[direction] = (
            find_leaf(flat, f"descriptor_group/chunk_memory/base_memory/{direction}/forget/kernel"),
            find_leaf(flat, f"descriptor_group/chunk_memory/base_memory/{direction}/forget/bias"),
        )

    accum = {
        ("m4", d): GateAccumulator("m4", d, M4_HALF_LIVES, config["model_dim"])
        for d in ("fwd", "bwd")
    }
    accum.update({
        ("g4", d): GateAccumulator("g4", d, G4_HALF_LIVES, config["model_dim"])
        for d in ("fwd", "bwd")
    })

    all_correct = np.zeros(len(val_ids), bool)
    all_pred = np.zeros(len(val_ids), np.int32)
    motion = np.zeros(len(val_ids), np.float32)
    rank_rows = {s: [] for s in STAGES}
    rank_seen = 0

    for start in range(0, len(val_ids), args.batch_size):
        idx = val_ids[start:start+args.batch_size]
        x = np.asarray(dataset.canonical[idx], np.float32)
        y = np.asarray(dataset.labels[idx], np.int32)
        out = jax.device_get(forward(params, jax.device_put(x)))
        pred = np.asarray(out["logits"]).argmax(1)

        loc = slice(start, start+len(idx))
        all_pred[loc] = pred
        all_correct[loc] = pred == y

        tok = x.reshape(len(x), 16, 2, 25, 15)
        motion[loc] = np.mean(np.abs(tok[..., 3:15]), axis=(1,2,3,4))

        spatial = jnp.asarray(out["spatial_stack"])            # [B,16,4,D]
        mixed = jnp.asarray(out["mixed_frame_stack"])         # [B,16,4,D]
        chunks = mixed.reshape(len(x), 4, 4, 4, config["model_dim"]).mean(axis=2)

        for direction in ("fwd", "bwd"):
            km, bm = m4[direction]
            kg, bg = g4[direction]
            ma = gates_for(spatial, jnp.asarray(km), jnp.asarray(bm))
            ga = gates_for(chunks, jnp.asarray(kg), jnp.asarray(bg))
            accum[("m4", direction)].add(np.asarray(ma))
            accum[("g4", direction)].add(np.asarray(ga))

        if rank_seen < args.rank_samples:
            _, means = sequences_and_means(x, out)
            take = min(len(x), args.rank_samples - rank_seen)
            for stage in STAGES:
                rank_rows[stage].append(np.asarray(means[stage][:take], np.float32))
            rank_seen += take

    verified = float(all_correct.mean())
    frame_lengths = np.asarray(dataset.shape[val_ids, 0], np.float32)
    people = np.asarray(dataset.shape[val_ids, 1], np.int32)

    class_rows = []
    confusion = Counter()
    for cls in range(120):
        use = labels == cls
        n = int(use.sum())
        if n:
            class_rows.append({
                "class_zero_based": cls,
                "ntu_action": cls + 1,
                "n": n,
                "accuracy": float(all_correct[use].mean()),
            })
    class_rows.sort(key=lambda r: (r["accuracy"], -r["n"]))

    for y, p in zip(labels, all_pred):
        if y != p:
            confusion[(int(y), int(p))] += 1

    pair_rows = [
        {
            "true_action": y + 1,
            "pred_action": p + 1,
            "count": n,
        }
        for (y, p), n in confusion.most_common(20)
    ]

    person_result = {}
    for npeople in (1, 2):
        use = people == npeople
        person_result[str(npeople)] = {
            "n": int(use.sum()),
            "accuracy": float(all_correct[use].mean()) if np.any(use) else None,
        }

    ranks = {}
    for stage in STAGES:
        rows = np.concatenate(rank_rows[stage], axis=0)
        ranks[stage] = representation_rank(rows)

    rng = np.random.default_rng(20261003)
    robustness_n = min(args.robustness_samples, len(val_ids))
    # Class-balanced-ish deterministic subset: random permutation of each class,
    # then round-robin until target size.
    per_class = [list(rng.permutation(val_ids[labels == c])) for c in range(120)]
    robust_ids = []
    cursor = 0
    while len(robust_ids) < robustness_n:
        progressed = False
        for c in range(120):
            if cursor < len(per_class[c]):
                robust_ids.append(int(per_class[c][cursor]))
                progressed = True
                if len(robust_ids) >= robustness_n:
                    break
        if not progressed:
            break
        cursor += 1
    robust_ids = np.asarray(robust_ids, np.int64)

    robustness = evaluate_robustness(
        model, params, dataset, robust_ids, min(args.batch_size, 256)
    )

    result = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_recorded_val_accuracy": float(payload["val_accuracy"]),
        "verified_val_accuracy": verified,
        "params": count_params(params),
        "gate_dynamics": {
            f"{stage}_{direction}": acc.result()
            for (stage, direction), acc in accum.items()
        },
        "failure_buckets": {
            "clip_length_frames": bucket_accuracy(frame_lengths, all_correct),
            "motion_energy": bucket_accuracy(motion, all_correct),
            "people": person_result,
        },
        "representation_effective_rank": ranks,
        "worst_20_classes": class_rows[:20],
        "top_20_confusions": pair_rows,
        "robustness": robustness,
        "interpretation_rules": {
            "input_t16_bottleneck": (
                "Supported if long-clip buckets are much worse and/or 0.8-0.9 crops cause "
                "large accuracy collapse while gate/rank diagnostics remain healthy."
            ),
            "m4_memory_bottleneck": (
                "Supported if M4 effective half-lives collapse toward a narrow range, "
                "timescale groups lose separation, or many gates saturate."
            ),
            "g4_overfit_bottleneck": (
                "Supported when trained stage audit shows Router->G4 excess train gain "
                "and G4 gate dynamics are strongly saturated/collapsed."
            ),
            "capacity_bottleneck": (
                "Supported only if D128 improves validation substantially without worsening "
                "the train-val gap and representation rank expands usefully."
            ),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, result)

    print("=" * 118)
    print(f"{MODEL_NAME} | {args.protocol.upper()} DEEP BOTTLENECK AUDIT")
    print("=" * 118)
    print(
        f"checkpoint E{int(payload['epoch']):02d} | "
        f"recorded={100*float(payload['val_accuracy']):.4f}% | verified={100*verified:.4f}%"
    )

    print("\nTRUE FORGET-GATE DYNAMICS")
    for key, row in result["gate_dynamics"].items():
        print(
            f"{key:9s} H_eff range={row['effective_half_life_min']:.3f}-"
            f"{row['effective_half_life_max']:.3f} "
            f"ratio={row['effective_half_life_ratio_max_min']:.2f}x"
        )
        for stream, groups in row["streams"].items():
            vals = ", ".join(
                f"Hcfg{g['configured_half_life']:.0f}->Heff{g['effective_half_life_tokens']:.2f}"
                for g in groups
            )
            print(f"  {stream:14s} {vals}")

    print("\nFAILURE BUCKETS")
    print("Clip length:")
    for row in result["failure_buckets"]["clip_length_frames"]["buckets"]:
        print(
            f"  B{row['bucket']} n={row['n']:5d} mean={row['value_mean']:.1f}f "
            f"acc={100*row['accuracy']:.2f}%"
        )
    print("Motion energy:")
    for row in result["failure_buckets"]["motion_energy"]["buckets"]:
        print(
            f"  B{row['bucket']} n={row['n']:5d} mean={row['value_mean']:.5f} "
            f"acc={100*row['accuracy']:.2f}%"
        )
    print("People:")
    for k, row in person_result.items():
        if row["accuracy"] is not None:
            print(f"  {k} person(s): n={row['n']} acc={100*row['accuracy']:.2f}%")

    print("\nREPRESENTATION EFFECTIVE RANK")
    for stage, row in ranks.items():
        print(
            f"  {stage:12s} PR={row['participation_ratio']:.1f} "
            f"rank90={row['rank90']:3d} rank95={row['rank95']:3d} "
            f"top1={100*row['top1_variance_fraction']:.2f}%"
        )

    print("\nROBUSTNESS SUBSET")
    print(f"  canonical: {100*robustness['canonical_accuracy']:.2f}%")
    for key, row in robustness["conditions"].items():
        print(
            f"  {key:24s} acc={100*row['accuracy']:.2f}% "
            f"delta={row['delta_from_canonical_pp']:+.2f}pp "
            f"agree={100*row['prediction_agreement']:.1f}%"
        )

    print("\nWORST CLASSES")
    for row in class_rows[:10]:
        print(
            f"  A{row['ntu_action']:03d}: {100*row['accuracy']:.2f}% "
            f"(n={row['n']})"
        )

    print("\nSaved:", out)
    print("=" * 118)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""Read-only trained-model audit for NestSAR-FULL-PARALLEL-T16-v2.

Purpose
-------
Diagnose what to change *after* a clean parallel-v2 training run, without
modifying the checkpoint.

Measures:
  1. training-history saturation and train/validation gap;
  2. full-split stage-wise frozen ridge probes:
       input -> spatial -> M4 -> router -> G4 -> descriptor;
  3. train/validation Fisher separation and class-centroid drift;
  4. temporal headroom: mean-only vs [mean,std,last-first] NCM features;
  5. validation stream dependence:
       each stream alone, leave-one-stream-out, uniform fusion;
  6. adaptive-head contribution;
  7. fusion-weight collapse/entropy;
  8. subject/setup/camera leakage after subtracting the action centroid;
  9. trained forget-bias retention/half-life proxy for the affine memories.

No training. No checkpoint mutation.
"""

import argparse
import json
import math
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.traverse_util import flatten_dict

from experiments.nestsar_sm_all_t16.audit_generalization_localization import (
    Stats,
    centroid_alignment,
    normalize_rows,
    predictions,
    sequences_and_means,
    temporal_summary,
)
from experiments.nestsar_sm_all_t16.model_parallel import NestSARParallelT16
from experiments.nestsar_sm_all_t16.parallel_config import MODEL_NAME, EXPECTED_PARAMS
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json


STAGES = ("input", "spatial", "m4", "router", "g4", "descriptor")
TEMPORAL_STAGES = ("input", "spatial", "m4", "router", "g4")
STREAMS = ("joint", "bone", "joint_motion", "bone_motion")
R4_BASELINE = {"xsub": 0.769104, "xset": 0.784605814}
SID_RE = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})", re.I)


def report(path: Path, protocol: str, phase: str, current: int, total: int, **extra):
    payload = {
        "protocol": protocol,
        "phase": phase,
        "current": int(current),
        "total": int(max(total, 1)),
    }
    payload.update(extra)
    atomic_json(path, payload)


def load_checkpoint(path: Path):
    payload = serialization.msgpack_restore(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint: {path}")
    for key in ("ema_params", "config", "model", "protocol", "epoch", "val_accuracy"):
        if key not in payload:
            raise ValueError(f"Checkpoint missing {key}: {path}")
    if payload["model"] != MODEL_NAME:
        raise ValueError(
            f"Expected {MODEL_NAME}, checkpoint identifies {payload['model']!r}"
        )
    return payload, payload["ema_params"], dict(payload["config"])


def param_count(tree):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def make_model(config):
    return NestSARParallelT16(**{k: config[k] for k in (
        "spatial_dim", "model_dim", "dropout", "controller_dim", "fast_rank",
        "head_rank", "sm_residual_scale", "head_residual_scale"
    )})


def parse_nuisance(ids):
    setup = np.zeros(len(ids), np.int32)
    camera = np.zeros(len(ids), np.int32)
    subject = np.zeros(len(ids), np.int32)
    repetition = np.zeros(len(ids), np.int32)
    action = np.zeros(len(ids), np.int32)
    for i, sid in enumerate(ids):
        m = SID_RE.search(str(sid))
        if m is None:
            raise ValueError(f"Cannot parse NTU sample id: {sid!r}")
        setup[i], camera[i], subject[i], repetition[i], action[i] = map(int, m.groups())
    return {
        "setup": setup,
        "camera": camera,
        "subject": subject,
        "repetition": repetition,
        "action_1based": action,
    }


class GroupScatter:
    """Between/within scatter for arbitrary integer nuisance groups."""

    def __init__(self, dim: int, max_group: int):
        self.dim = int(dim)
        self.counts = np.zeros(max_group + 1, np.int64)
        self.sums = np.zeros((max_group + 1, dim), np.float64)
        self.sumx = np.zeros(dim, np.float64)
        self.sum_norm2 = 0.0
        self.n = 0

    def add(self, x, groups):
        x = np.asarray(x, np.float32)
        groups = np.asarray(groups, np.int32)
        self.counts += np.bincount(groups, minlength=len(self.counts))
        np.add.at(self.sums, groups, x)
        self.sumx += np.asarray(x.sum(axis=0), np.float64)
        self.sum_norm2 += float(np.square(x, dtype=np.float64).sum())
        self.n += len(x)

    def result(self):
        use = self.counts > 0
        counts = self.counts[use]
        sums = self.sums[use]
        means = sums / counts[:, None]
        global_mean = self.sumx / max(self.n, 1)
        within = self.sum_norm2 - float(
            np.sum(np.square(sums).sum(axis=1) / counts)
        )
        between = float(
            np.sum(counts[:, None] * np.square(means - global_mean[None, :]))
        )
        return {
            "groups": int(use.sum()),
            "between": between,
            "within": within,
            "fisher_between_over_within": between / max(within, 1e-12),
        }


def history_summary(path: Path | None):
    if path is None or not path.is_file():
        return None
    rows = json.loads(path.read_text())
    if not rows:
        return None

    best = max(rows, key=lambda r: float(r["val_acc"]))
    final = rows[-1]
    best_i = rows.index(best)

    def first_epoch_at(key, threshold):
        for r in rows:
            if float(r.get(key, -1.0)) >= threshold:
                return int(r["epoch"])
        return None

    return {
        "epochs_recorded": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_val_accuracy": float(best["val_acc"]),
        "train_accuracy_at_best": float(best["train_acc"]),
        "train_val_gap_at_best_pp": 100.0 * (
            float(best["train_acc"]) - float(best["val_acc"])
        ),
        "final_epoch": int(final["epoch"]),
        "final_train_accuracy": float(final["train_acc"]),
        "final_val_accuracy": float(final["val_acc"]),
        "post_best_val_change_pp": 100.0 * (
            float(final["val_acc"]) - float(best["val_acc"])
        ),
        "post_best_train_change_pp": 100.0 * (
            float(final["train_acc"]) - float(best["train_acc"])
        ),
        "epochs_after_best": len(rows) - best_i - 1,
        "first_train_95_epoch": first_epoch_at("train_acc", 0.95),
        "first_train_98_epoch": first_epoch_at("train_acc", 0.98),
    }


def forget_bias_proxy(params):
    flat = flatten_dict(params)
    rows = []
    for path, value in flat.items():
        names = tuple(str(p) for p in path)
        if "forget" not in names or names[-1] != "bias":
            continue
        v = np.asarray(value, np.float64).reshape(-1)
        retention = 1.0 / (1.0 + np.exp(-v))
        retention = np.clip(retention, 1e-7, 1.0 - 1e-7)
        half_life = np.log(0.5) / np.log(retention)
        rows.append((names, v, retention, half_life))

    if not rows:
        return {"count": 0}

    bias = np.concatenate([r[1] for r in rows])
    retention = np.concatenate([r[2] for r in rows])
    half_life = np.concatenate([r[3] for r in rows])

    return {
        "count": int(len(bias)),
        "modules": int(len(rows)),
        "bias_mean": float(bias.mean()),
        "bias_std": float(bias.std()),
        "bias_min": float(bias.min()),
        "bias_max": float(bias.max()),
        "zero_input_retention_mean": float(retention.mean()),
        "zero_input_retention_std": float(retention.std()),
        "zero_input_half_life_mean_tokens": float(half_life.mean()),
        "zero_input_half_life_median_tokens": float(np.median(half_life)),
        "zero_input_half_life_p10_tokens": float(np.percentile(half_life, 10)),
        "zero_input_half_life_p90_tokens": float(np.percentile(half_life, 90)),
        "note": "Proxy only: real forget activations also depend on x @ W.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True, choices=("xsub", "xset"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--history", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ridge", type=float, default=1e-2)
    args = p.parse_args()

    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise RuntimeError(
            f"Expected one isolated GPU; backend={jax.default_backend()}, "
            f"devices={jax.local_devices()}"
        )

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    status = outdir / "status.json"
    result_path = outdir / "audit.json"

    checkpoint_path = Path(args.checkpoint)
    payload, params, config = load_checkpoint(checkpoint_path)

    if payload["protocol"] != args.protocol:
        raise ValueError(
            f"Checkpoint protocol={payload['protocol']} but requested {args.protocol}"
        )

    nparams = param_count(params)
    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(f"Parameter mismatch: {nparams} != {EXPECTED_PARAMS}")

    cache = Path(args.cache)
    dataset = Dataset(cache)
    ids_all = json.loads((cache / "ids.json").read_text())
    nuisance = parse_nuisance(ids_all)

    # Validate action IDs against cached zero-based labels.
    parsed_action = nuisance["action_1based"] - 1
    if not np.array_equal(parsed_action, np.asarray(dataset.labels)):
        bad = np.where(parsed_action != np.asarray(dataset.labels))[0][:5]
        raise RuntimeError(f"Sample-ID action labels disagree with cache at {bad.tolist()}")

    train_ids = np.asarray(dataset.splits[f"{args.protocol}_train"], np.int64)
    val_ids = np.asarray(dataset.splits[f"{args.protocol}_val"], np.int64)

    model = make_model(config)
    forward = jax.jit(
        lambda p, x: model.apply({"params": p}, x, training=False)
    )

    mean_train = {}
    temporal_train = {}
    train_steps = math.ceil(len(train_ids) / args.batch_size)

    # ------------------------------------------------------------------
    # PASS 1: train statistics / probe fitting.
    # ------------------------------------------------------------------
    for bi, start in enumerate(range(0, len(train_ids), args.batch_size)):
        idx = train_ids[start:start + args.batch_size]
        x = np.asarray(dataset.canonical[idx], np.float32)
        y = np.asarray(dataset.labels[idx], np.int32)
        pred = jax.device_get(forward(params, jax.device_put(x)))
        seq, means = sequences_and_means(x, pred)

        if not mean_train:
            mean_train = {
                stage: Stats(means[stage].shape[1], ridge_enabled=True)
                for stage in STAGES
            }
            temporal_train = {
                stage: Stats(
                    temporal_summary(seq[stage]).shape[1],
                    ridge_enabled=False,
                )
                for stage in TEMPORAL_STAGES
            }

        for stage in STAGES:
            mean_train[stage].add(means[stage], y)
        for stage in TEMPORAL_STAGES:
            temporal_train[stage].add(temporal_summary(seq[stage]), y)

        if bi % 5 == 0 or bi + 1 == train_steps:
            report(status, args.protocol, "Train statistics", bi + 1, train_steps)

    ridge_models = {}
    ridge_lambda = {}
    train_centroid = {}
    temporal_centroid = {}

    for stage in STAGES:
        ridge_models[stage], ridge_lambda[stage] = mean_train[stage].solve_ridge(args.ridge)
        train_centroid[stage], _ = mean_train[stage].centroids()
    for stage in TEMPORAL_STAGES:
        _, temporal_centroid[stage] = temporal_train[stage].centroids()

    # ------------------------------------------------------------------
    # PASS 2: frozen probes on train + nuisance leakage after action removal.
    # ------------------------------------------------------------------
    train_correct_ridge = {s: 0 for s in STAGES}
    train_ncm_mean = {s: 0 for s in STAGES}
    train_ncm_temporal = {s: 0 for s in TEMPORAL_STAGES}

    nuisance_train = {
        stage: {
            name: GroupScatter(
                mean_train[stage].dim,
                int(nuisance[name].max()),
            )
            for name in ("subject", "setup", "camera")
        }
        for stage in STAGES
    }

    train_seen = 0

    for bi, start in enumerate(range(0, len(train_ids), args.batch_size)):
        idx = train_ids[start:start + args.batch_size]
        x = np.asarray(dataset.canonical[idx], np.float32)
        y = np.asarray(dataset.labels[idx], np.int32)
        pred = jax.device_get(forward(params, jax.device_put(x)))
        seq, means = sequences_and_means(x, pred)
        train_seen += len(y)

        for stage in STAGES:
            pr = predictions(
                means[stage],
                ridge_w=ridge_models[stage],
                centroid_unit=normalize_rows(train_centroid[stage]),
            )
            train_correct_ridge[stage] += int(np.sum(pr["ridge"] == y))
            train_ncm_mean[stage] += int(np.sum(pr["ncm"] == y))

            z = normalize_rows(means[stage])
            residual = z - train_centroid[stage][y]
            for name in ("subject", "setup", "camera"):
                nuisance_train[stage][name].add(residual, nuisance[name][idx])

        for stage in TEMPORAL_STAGES:
            pr = predictions(
                temporal_summary(seq[stage]),
                centroid_unit=temporal_centroid[stage],
            )
            train_ncm_temporal[stage] += int(np.sum(pr["ncm"] == y))

        if bi % 5 == 0 or bi + 1 == train_steps:
            report(status, args.protocol, "Train frozen probes", bi + 1, train_steps)

    # ------------------------------------------------------------------
    # PASS 3: validation stats / probes / stream counterfactuals.
    # ------------------------------------------------------------------
    mean_val = {}
    temporal_val = {}
    val_correct_ridge = {s: 0 for s in STAGES}
    val_ncm_mean = {s: 0 for s in STAGES}
    val_ncm_temporal = {s: 0 for s in TEMPORAL_STAGES}

    model_correct = 0
    main_correct = 0
    uniform_correct = 0
    stream_correct = np.zeros(4, np.int64)
    leaveout_correct = np.zeros(4, np.int64)
    fusion_sum = np.zeros(4, np.float64)
    fusion_entropy_sum = 0.0
    val_seen = 0

    val_steps = math.ceil(len(val_ids) / args.batch_size)

    for bi, start in enumerate(range(0, len(val_ids), args.batch_size)):
        idx = val_ids[start:start + args.batch_size]
        x = np.asarray(dataset.canonical[idx], np.float32)
        y = np.asarray(dataset.labels[idx], np.int32)
        pred = jax.device_get(forward(params, jax.device_put(x)))
        seq, means = sequences_and_means(x, pred)

        if not mean_val:
            mean_val = {
                stage: Stats(means[stage].shape[1], ridge_enabled=False)
                for stage in STAGES
            }
            temporal_val = {
                stage: Stats(
                    temporal_summary(seq[stage]).shape[1],
                    ridge_enabled=False,
                )
                for stage in TEMPORAL_STAGES
            }

        logits = np.asarray(pred["logits"], np.float32)
        main_logits = np.asarray(pred["main_logits"], np.float32)
        sl = np.asarray(pred["stream_logits"], np.float32)
        fw = np.asarray(pred["fusion_weights"], np.float32)

        n = len(y)
        val_seen += n
        model_correct += int(np.sum(logits.argmax(1) == y))
        main_correct += int(np.sum(main_logits.argmax(1) == y))
        uniform_correct += int(np.sum(sl.mean(axis=1).argmax(1) == y))

        for s in range(4):
            stream_correct[s] += int(np.sum(sl[:, s].argmax(1) == y))
            w = fw.copy()
            w[:, s] = 0.0
            w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
            lo = np.einsum("bs,bsc->bc", w, sl)
            leaveout_correct[s] += int(np.sum(lo.argmax(1) == y))

        fusion_sum += fw.sum(axis=0)
        fusion_entropy_sum += float(
            (-fw * np.log(np.maximum(fw, 1e-12))).sum()
        )

        for stage in STAGES:
            mean_val[stage].add(means[stage], y)
            pr = predictions(
                means[stage],
                ridge_w=ridge_models[stage],
                centroid_unit=normalize_rows(train_centroid[stage]),
            )
            val_correct_ridge[stage] += int(np.sum(pr["ridge"] == y))
            val_ncm_mean[stage] += int(np.sum(pr["ncm"] == y))

        for stage in TEMPORAL_STAGES:
            ts = temporal_summary(seq[stage])
            temporal_val[stage].add(ts, y)
            pr = predictions(ts, centroid_unit=temporal_centroid[stage])
            val_ncm_temporal[stage] += int(np.sum(pr["ncm"] == y))

        if bi % 5 == 0 or bi + 1 == val_steps:
            report(status, args.protocol, "Validation audit", bi + 1, val_steps)

    # Validation action centroids are diagnostic-only, for action-residual nuisance leakage.
    val_centroid = {}
    for stage in STAGES:
        val_centroid[stage], _ = mean_val[stage].centroids()

    # ------------------------------------------------------------------
    # PASS 4: nuisance leakage on validation after action removal.
    # ------------------------------------------------------------------
    nuisance_val = {
        stage: {
            name: GroupScatter(
                mean_val[stage].dim,
                int(nuisance[name].max()),
            )
            for name in ("subject", "setup", "camera")
        }
        for stage in STAGES
    }

    for bi, start in enumerate(range(0, len(val_ids), args.batch_size)):
        idx = val_ids[start:start + args.batch_size]
        x = np.asarray(dataset.canonical[idx], np.float32)
        y = np.asarray(dataset.labels[idx], np.int32)
        pred = jax.device_get(forward(params, jax.device_put(x)))
        _, means = sequences_and_means(x, pred)

        for stage in STAGES:
            z = normalize_rows(means[stage])
            residual = z - val_centroid[stage][y]
            for name in ("subject", "setup", "camera"):
                nuisance_val[stage][name].add(residual, nuisance[name][idx])

        if bi % 5 == 0 or bi + 1 == val_steps:
            report(status, args.protocol, "Validation nuisance leakage", bi + 1, val_steps)

    # ------------------------------------------------------------------
    # Build report.
    # ------------------------------------------------------------------
    stage_report = {}
    for stage in STAGES:
        tr = train_correct_ridge[stage] / train_seen
        va = val_correct_ridge[stage] / val_seen
        tr_scatter = mean_train[stage].scatter()
        va_scatter = mean_val[stage].scatter()
        stage_report[stage] = {
            "dimension": mean_train[stage].dim,
            "ridge_lambda": ridge_lambda[stage],
            "train_ridge_accuracy": tr,
            "val_ridge_accuracy": va,
            "generalization_gap_pp": 100.0 * (tr - va),
            "val_over_train_accuracy_ratio": va / max(tr, 1e-12),
            "train_ncm_mean_accuracy": train_ncm_mean[stage] / train_seen,
            "val_ncm_mean_accuracy": val_ncm_mean[stage] / val_seen,
            "train_fisher": tr_scatter["fisher_between_over_within"],
            "val_fisher": va_scatter["fisher_between_over_within"],
            "val_over_train_fisher_ratio": (
                va_scatter["fisher_between_over_within"]
                / max(tr_scatter["fisher_between_over_within"], 1e-12)
            ),
            "train_val_centroid_alignment": centroid_alignment(
                mean_train[stage], mean_val[stage]
            ),
            "nuisance_action_residual_fisher_train": {
                name: nuisance_train[stage][name].result()
                for name in ("subject", "setup", "camera")
            },
            "nuisance_action_residual_fisher_val": {
                name: nuisance_val[stage][name].result()
                for name in ("subject", "setup", "camera")
            },
        }

    temporal_report = {}
    for stage in TEMPORAL_STAGES:
        temporal_report[stage] = {
            "train_mean_ncm": train_ncm_mean[stage] / train_seen,
            "train_temporal_ncm": train_ncm_temporal[stage] / train_seen,
            "train_temporal_headroom_pp": 100.0 * (
                train_ncm_temporal[stage] - train_ncm_mean[stage]
            ) / train_seen,
            "val_mean_ncm": val_ncm_mean[stage] / val_seen,
            "val_temporal_ncm": val_ncm_temporal[stage] / val_seen,
            "val_temporal_headroom_pp": 100.0 * (
                val_ncm_temporal[stage] - val_ncm_mean[stage]
            ) / val_seen,
        }

    transitions = {}
    for before, after in zip(STAGES[:-1], STAGES[1:]):
        tr_gain = 100.0 * (
            stage_report[after]["train_ridge_accuracy"]
            - stage_report[before]["train_ridge_accuracy"]
        )
        va_gain = 100.0 * (
            stage_report[after]["val_ridge_accuracy"]
            - stage_report[before]["val_ridge_accuracy"]
        )
        transitions[f"{before}->{after}"] = {
            "train_gain_pp": tr_gain,
            "val_gain_pp": va_gain,
            "excess_train_gain_pp": tr_gain - va_gain,
        }

    actual = model_correct / val_seen
    main = main_correct / val_seen

    stream_report = {
        "actual_accuracy": actual,
        "main_logits_accuracy": main,
        "adaptive_head_gain_pp": 100.0 * (actual - main),
        "uniform_fusion_accuracy": uniform_correct / val_seen,
        "learned_vs_uniform_gain_pp": 100.0 * (
            main - uniform_correct / val_seen
        ),
        "mean_fusion_weights": {
            STREAMS[s]: float(fusion_sum[s] / val_seen)
            for s in range(4)
        },
        "mean_fusion_entropy_nats": fusion_entropy_sum / val_seen,
        "max_entropy_nats": float(np.log(4.0)),
        "single_stream_accuracy": {
            STREAMS[s]: float(stream_correct[s] / val_seen)
            for s in range(4)
        },
        "leave_one_stream_out_accuracy": {
            STREAMS[s]: float(leaveout_correct[s] / val_seen)
            for s in range(4)
        },
        "leave_one_stream_out_delta_pp": {
            STREAMS[s]: 100.0 * (
                leaveout_correct[s] / val_seen - main
            )
            for s in range(4)
        },
    }

    hist = history_summary(
        Path(args.history) if args.history else checkpoint_path.parent / "history.json"
    )

    result = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_recorded_val_accuracy": float(payload["val_accuracy"]),
        "verified_full_val_accuracy": actual,
        "verified_minus_checkpoint_pp": 100.0 * (
            actual - float(payload["val_accuracy"])
        ),
        "params": nparams,
        "train_samples": int(len(train_ids)),
        "val_samples": int(len(val_ids)),
        "r4_baseline_accuracy": R4_BASELINE[args.protocol],
        "parallel_minus_r4_pp": 100.0 * (
            actual - R4_BASELINE[args.protocol]
        ),
        "history": hist,
        "stages": stage_report,
        "transitions": transitions,
        "temporal": temporal_report,
        "streams": stream_report,
        "forget_bias_timescale_proxy": forget_bias_proxy(params),
    }

    atomic_json(result_path, result)
    report(
        status, args.protocol, "Done", 1, 1,
        done=True, val_accuracy=actual,
        r4_delta_pp=result["parallel_minus_r4_pp"],
    )

    print("=" * 116)
    print(f"{MODEL_NAME} | {args.protocol.upper()} TRAINED AUDIT")
    print("=" * 116)
    print(
        f"checkpoint E{int(payload['epoch']):02d} "
        f"recorded={100*float(payload['val_accuracy']):.4f}% | "
        f"verified={100*actual:.4f}% | "
        f"R4={100*R4_BASELINE[args.protocol]:.4f}% | "
        f"delta={result['parallel_minus_r4_pp']:+.4f} pp"
    )

    if hist:
        print(
            f"history: best E{hist['best_epoch']:02d} "
            f"{100*hist['best_val_accuracy']:.4f}% | "
            f"train@best={100*hist['train_accuracy_at_best']:.2f}% | "
            f"gap={hist['train_val_gap_at_best_pp']:.2f} pp | "
            f"after-best={hist['epochs_after_best']} epochs"
        )

    print("\nSTAGE PROBES")
    print(f"{'stage':12s} {'train':>9s} {'val':>9s} {'gap':>9s} {'val/train':>10s} {'centroid':>10s}")
    for stage in STAGES:
        s = stage_report[stage]
        print(
            f"{stage:12s} "
            f"{100*s['train_ridge_accuracy']:8.2f}% "
            f"{100*s['val_ridge_accuracy']:8.2f}% "
            f"{s['generalization_gap_pp']:8.2f} "
            f"{s['val_over_train_accuracy_ratio']:10.4f} "
            f"{s['train_val_centroid_alignment']['mean_cosine']:10.4f}"
        )

    print("\nTRANSITIONS")
    for name, t in transitions.items():
        print(
            f"{name:22s} train={t['train_gain_pp']:+7.3f} pp | "
            f"val={t['val_gain_pp']:+7.3f} pp | "
            f"EXCESS={t['excess_train_gain_pp']:+7.3f} pp"
        )

    print("\nSTREAM DEPENDENCE")
    for s in STREAMS:
        print(
            f"{s:14s} alone={100*stream_report['single_stream_accuracy'][s]:7.3f}% | "
            f"leave-out={100*stream_report['leave_one_stream_out_accuracy'][s]:7.3f}% | "
            f"delta(main)={stream_report['leave_one_stream_out_delta_pp'][s]:+7.3f} pp | "
            f"weight={stream_report['mean_fusion_weights'][s]:.4f}"
        )
    print(
        f"uniform={100*stream_report['uniform_fusion_accuracy']:.4f}% | "
        f"main={100*main:.4f}% | actual={100*actual:.4f}% | "
        f"head gain={stream_report['adaptive_head_gain_pp']:+.4f} pp"
    )

    focus = "subject" if args.protocol == "xsub" else "setup"
    print(f"\nACTION-RESIDUAL NUISANCE LEAKAGE ({focus.upper()})")
    for stage in STAGES:
        trf = stage_report[stage]["nuisance_action_residual_fisher_train"][focus]["fisher_between_over_within"]
        vaf = stage_report[stage]["nuisance_action_residual_fisher_val"][focus]["fisher_between_over_within"]
        print(f"{stage:12s} train={trf:.6f} | val={vaf:.6f}")

    proxy = result["forget_bias_timescale_proxy"]
    print("\nAFFINE MEMORY TIMESCALE PROXY")
    print(
        f"forget channels={proxy.get('count')} | "
        f"retention={proxy.get('zero_input_retention_mean', float('nan')):.4f}±"
        f"{proxy.get('zero_input_retention_std', float('nan')):.4f} | "
        f"half-life median={proxy.get('zero_input_half_life_median_tokens', float('nan')):.3f} tokens | "
        f"p10-p90={proxy.get('zero_input_half_life_p10_tokens', float('nan')):.3f}-"
        f"{proxy.get('zero_input_half_life_p90_tokens', float('nan')):.3f}"
    )

    print("\nSaved:", result_path)
    print("=" * 116)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""Read-only generalization-localization audit for trained NestSAR R4.

The backbone/checkpoint never changes. The audit measures:
  1) train-vs-validation linear separability at every learned stage;
  2) train-vs-validation Fisher separation and class-centroid drift;
  3) temporal-information headroom using the same nearest-centroid rule on
     mean-only versus [mean, std, last-first] features;
  4) gate-timing counterfactuals: learned, constant-at-global-mean, reversed-time.

This localizes where generalizable information stops keeping pace with training
information without retraining NestSAR.
"""

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import serialization

from experiments.nestsar_sm_all_t16 import model as sm
from experiments.nestsar_sm_all_t16.streaming import worker as train_worker
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json

NUM_CLASSES = 120
EXPECTED_PARAMS = 1_831_932
STAGES = ("input", "spatial", "m4", "router", "g4", "descriptor")
TEMPORAL_STAGES = ("input", "spatial", "m4", "router", "g4")


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
    if not isinstance(payload, dict) or "ema_params" not in payload or "config" not in payload:
        raise ValueError(f"Expected best checkpoint with ema_params/config: {path}")
    return payload, payload["ema_params"], dict(payload["config"])


def param_count(tree):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def normalize_rows(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def sequences_and_means(x, out):
    b = len(x)
    seq = {
        "input": np.asarray(x, np.float32),
        "spatial": np.asarray(out["spatial_stack"], np.float32).reshape(b, 16, -1),
        "m4": np.asarray(out["frame_stack"], np.float32).reshape(b, 16, -1),
        "router": np.asarray(out["mixed_frame_stack"], np.float32).reshape(b, 16, -1),
        "g4": np.asarray(out["chunk_states"], np.float32).transpose(0, 2, 1, 3).reshape(b, 4, -1),
    }
    means = {name: value.mean(axis=1) for name, value in seq.items()}
    means["descriptor"] = np.asarray(out["descriptors"], np.float32).reshape(b, -1)
    return seq, means


def temporal_summary(seq):
    seq = np.asarray(seq, np.float32)
    return np.concatenate(
        [
            seq.mean(axis=1),
            seq.std(axis=1),
            seq[:, -1] - seq[:, 0],
        ],
        axis=1,
    )


class Stats:
    def __init__(self, dim, ridge_enabled=True):
        self.dim = int(dim)
        self.ridge_enabled = bool(ridge_enabled)
        self.sumx = np.zeros(self.dim, np.float64)
        self.class_sum = np.zeros((NUM_CLASSES, self.dim), np.float64)
        self.counts = np.zeros(NUM_CLASSES, np.int64)
        self.sum_norm2 = 0.0
        self.n = 0
        self.xtx = np.zeros((self.dim, self.dim), np.float64) if ridge_enabled else None

    def add(self, x, y):
        z = normalize_rows(x)
        self.sumx += np.asarray(z.sum(axis=0), np.float64)
        self.sum_norm2 += float(np.square(z, dtype=np.float64).sum())
        self.n += len(z)
        self.counts += np.bincount(y, minlength=NUM_CLASSES)
        np.add.at(self.class_sum, y, z)
        if self.ridge_enabled:
            self.xtx += np.asarray(z.T @ z, np.float64)

    def centroids(self):
        if np.any(self.counts == 0):
            missing = np.where(self.counts == 0)[0].tolist()
            raise RuntimeError(f"Split missing classes: {missing}")
        c = self.class_sum / self.counts[:, None]
        return c, normalize_rows(c)

    def scatter(self):
        c, _ = self.centroids()
        global_mean = self.sumx / self.n
        within = self.sum_norm2 - float(
            np.sum(np.square(self.class_sum).sum(axis=1) / self.counts)
        )
        between = float(
            np.sum(self.counts[:, None] * np.square(c - global_mean[None, :]))
        )
        return {
            "between": between,
            "within": within,
            "fisher_between_over_within": between / max(within, 1e-12),
        }

    def solve_ridge(self, ridge):
        if not self.ridge_enabled:
            raise RuntimeError("Ridge statistics disabled")
        d = self.dim
        a = np.zeros((d + 1, d + 1), np.float64)
        a[:d, :d] = self.xtx
        a[:d, d] = self.sumx
        a[d, :d] = self.sumx
        a[d, d] = self.n
        b = np.zeros((d + 1, NUM_CLASSES), np.float64)
        b[:d] = self.class_sum.T
        b[d] = self.counts
        scale = max(float(np.trace(self.xtx)) / max(d, 1), 1e-8)
        lam = float(ridge) * scale
        a[:d, :d] += lam * np.eye(d, dtype=np.float64)
        try:
            w = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(a, rcond=1e-8) @ b
        return w, lam


def predictions(x, ridge_w=None, centroid_unit=None):
    z = normalize_rows(x)
    result = {}
    if ridge_w is not None:
        result["ridge"] = (z @ ridge_w[:-1] + ridge_w[-1]).argmax(axis=1)
    if centroid_unit is not None:
        result["ncm"] = (z @ centroid_unit.T).argmax(axis=1)
    return result


def centroid_alignment(train_stats: Stats, val_stats: Stats):
    _, tc = train_stats.centroids()
    _, vc = val_stats.centroids()
    cos = np.sum(tc * vc, axis=1)
    order = np.argsort(cos)
    return {
        "mean_cosine": float(np.mean(cos)),
        "median_cosine": float(np.median(cos)),
        "minimum_cosine": float(np.min(cos)),
        "worst_10_classes": [
            {
                "class_zero_based": int(i),
                "ntu_action": int(i + 1),
                "train_val_centroid_cosine": float(cos[i]),
            }
            for i in order[:10]
        ],
    }


def make_gate_counterfactual_fast(mode, eta_value, alpha_value):
    if mode not in {"constant", "reverse_time"}:
        raise ValueError(mode)

    class GateCounterfactualFast(nn.Module):
        dim: int
        rank: int = 4

        @nn.compact
        def __call__(self, x, eta, alpha):
            if mode == "constant":
                eta = jnp.full_like(eta, eta_value)
                alpha = jnp.full_like(alpha, alpha_value)
            elif mode == "reverse_time":
                eta = eta[:, ::-1]
                alpha = alpha[:, ::-1]

            n = nn.LayerNorm(name="value_norm")(x)
            k = nn.Dense(
                self.rank,
                use_bias=False,
                kernel_init=nn.initializers.normal(0.02),
                name="key",
            )(n)
            q = nn.Dense(
                self.rank,
                use_bias=False,
                kernel_init=nn.initializers.normal(0.02),
                name="query",
            )(n)
            k = jnp.tanh(k)
            q = jnp.tanh(q)
            k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-6)
            q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-6)

            memory0 = self.param(
                "memory0",
                nn.initializers.normal(0.01),
                (self.rank, self.dim),
            )
            memory = jnp.broadcast_to(
                memory0[None, :, :],
                (x.shape[0], self.rank, self.dim),
            )
            kt = jnp.swapaxes(k, 0, 1)
            qt = jnp.swapaxes(q, 0, 1)
            vt = jnp.swapaxes(n, 0, 1)
            et = jnp.swapaxes(eta, 0, 1)
            at = jnp.swapaxes(alpha, 0, 1)

            def step(mem, inputs):
                key_t, query_t, value_t, eta_t, alpha_t = inputs
                pred_t = jnp.einsum("br,brd->bd", key_t, mem)
                err_t = value_t - pred_t
                delta_t = jnp.einsum("br,bd->brd", key_t, err_t)
                mem = alpha_t[..., None] * mem + eta_t[..., None] * delta_t
                read_t = jnp.einsum("br,brd->bd", query_t, mem)
                return mem, read_t

            _, reads = jax.lax.scan(step, memory, (kt, qt, vt, et, at))
            return jnp.swapaxes(reads, 0, 1)

    return GateCounterfactualFast


def counterfactual_accuracy(config, params, dataset, ids, batch_size, mode, eta_mean, alpha_mean, status, protocol):
    original = sm.FastWeightDeltaResidual
    sm.FastWeightDeltaResidual = make_gate_counterfactual_fast(mode, eta_mean, alpha_mean)
    try:
        model = train_worker.make_model(config)
        apply_logits = jax.jit(
            lambda p, x: model.apply({"params": p}, x, training=False)["logits"]
        )
        correct = 0
        seen = 0
        steps = math.ceil(len(ids) / batch_size)
        for bi, start in enumerate(range(0, len(ids), batch_size)):
            batch_ids = np.asarray(ids[start:start + batch_size], np.int64)
            x = np.asarray(dataset.canonical[batch_ids], np.float32)
            y = np.asarray(dataset.labels[batch_ids], np.int32)
            logits = np.asarray(jax.device_get(apply_logits(params, jax.device_put(x))))
            correct += int(np.sum(logits.argmax(axis=1) == y))
            seen += len(y)
            if bi % 10 == 0 or bi + 1 == steps:
                report(status, protocol, f"Gate test {mode}", bi + 1, steps)
        return correct / seen
    finally:
        sm.FastWeightDeltaResidual = original


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ridge", type=float, default=1e-2)
    args = p.parse_args()

    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise RuntimeError(
            f"Expected one isolated GPU, got backend={jax.default_backend()}, devices={jax.local_devices()}"
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.json"

    payload, params, config = load_checkpoint(Path(args.checkpoint))
    if int(config.get("fast_rank", -1)) != 4 or int(config.get("head_rank", -1)) != 2:
        raise RuntimeError("This localization audit expects the trained Fast-R4 / Head-R2 checkpoint")
    nparams = param_count(params)
    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(f"Parameter count mismatch: {nparams} != {EXPECTED_PARAMS}")

    dataset = Dataset(args.cache)
    train_ids = dataset.splits[f"{args.protocol}_train"]
    val_ids = dataset.splits[f"{args.protocol}_val"]
    model = train_worker.make_model(config)
    forward = jax.jit(lambda p, x: model.apply({"params": p}, x, training=False))

    mean_train = {}
    temporal_train = {}
    train_steps = math.ceil(len(train_ids) / args.batch_size)

    # Pass 1: fit all train-only probe/scatter statistics.
    for bi, start in enumerate(range(0, len(train_ids), args.batch_size)):
        ids = np.asarray(train_ids[start:start + args.batch_size], np.int64)
        x = np.asarray(dataset.canonical[ids], np.float32)
        y = np.asarray(dataset.labels[ids], np.int32)
        out = jax.device_get(forward(params, jax.device_put(x)))
        seq, means = sequences_and_means(x, out)

        if not mean_train:
            mean_train = {
                stage: Stats(means[stage].shape[1], ridge_enabled=True)
                for stage in STAGES
            }
            temporal_train = {
                stage: Stats(temporal_summary(seq[stage]).shape[1], ridge_enabled=False)
                for stage in TEMPORAL_STAGES
            }

        for stage in STAGES:
            mean_train[stage].add(means[stage], y)
        for stage in TEMPORAL_STAGES:
            temporal_train[stage].add(temporal_summary(seq[stage]), y)

        if bi % 5 == 0 or bi + 1 == train_steps:
            report(status, args.protocol, "Train representation statistics", bi + 1, train_steps)

    ridge_models = {}
    mean_train_centroids = {}
    temporal_train_centroids = {}
    ridge_lambda = {}
    for stage in STAGES:
        ridge_models[stage], ridge_lambda[stage] = mean_train[stage].solve_ridge(args.ridge)
        _, mean_train_centroids[stage] = mean_train[stage].centroids()
    for stage in TEMPORAL_STAGES:
        _, temporal_train_centroids[stage] = temporal_train[stage].centroids()

    # Pass 2: evaluate the exact same train representations with frozen probes.
    train_correct_ridge = {stage: 0 for stage in STAGES}
    train_correct_ncm_mean = {stage: 0 for stage in STAGES}
    train_correct_ncm_temporal = {stage: 0 for stage in TEMPORAL_STAGES}
    train_seen = 0

    for bi, start in enumerate(range(0, len(train_ids), args.batch_size)):
        ids = np.asarray(train_ids[start:start + args.batch_size], np.int64)
        x = np.asarray(dataset.canonical[ids], np.float32)
        y = np.asarray(dataset.labels[ids], np.int32)
        out = jax.device_get(forward(params, jax.device_put(x)))
        seq, means = sequences_and_means(x, out)
        train_seen += len(y)

        for stage in STAGES:
            pred = predictions(
                means[stage],
                ridge_w=ridge_models[stage],
                centroid_unit=mean_train_centroids[stage],
            )
            train_correct_ridge[stage] += int(np.sum(pred["ridge"] == y))
            train_correct_ncm_mean[stage] += int(np.sum(pred["ncm"] == y))

        for stage in TEMPORAL_STAGES:
            pred = predictions(
                temporal_summary(seq[stage]),
                centroid_unit=temporal_train_centroids[stage],
            )
            train_correct_ncm_temporal[stage] += int(np.sum(pred["ncm"] == y))

        if bi % 5 == 0 or bi + 1 == train_steps:
            report(status, args.protocol, "Frozen probes on train", bi + 1, train_steps)

    # Pass 3: validation probes + validation scatter + gate means + true model accuracy.
    mean_val = {}
    temporal_val = {}
    val_correct_ridge = {stage: 0 for stage in STAGES}
    val_correct_ncm_mean = {stage: 0 for stage in STAGES}
    val_correct_ncm_temporal = {stage: 0 for stage in TEMPORAL_STAGES}
    model_correct = 0
    val_seen = 0
    eta_sum = 0.0
    alpha_sum = 0.0
    gate_count = 0
    val_steps = math.ceil(len(val_ids) / args.batch_size)

    for bi, start in enumerate(range(0, len(val_ids), args.batch_size)):
        ids = np.asarray(val_ids[start:start + args.batch_size], np.int64)
        x = np.asarray(dataset.canonical[ids], np.float32)
        y = np.asarray(dataset.labels[ids], np.int32)
        out = jax.device_get(forward(params, jax.device_put(x)))
        seq, means = sequences_and_means(x, out)

        if not mean_val:
            mean_val = {
                stage: Stats(means[stage].shape[1], ridge_enabled=False)
                for stage in STAGES
            }
            temporal_val = {
                stage: Stats(temporal_summary(seq[stage]).shape[1], ridge_enabled=False)
                for stage in TEMPORAL_STAGES
            }

        pred_model = np.asarray(out["logits"]).argmax(axis=1)
        model_correct += int(np.sum(pred_model == y))
        val_seen += len(y)

        eta_values = np.asarray(out["sm_eta_mean"], np.float64)
        alpha_values = np.asarray(out["sm_alpha_mean"], np.float64)
        eta_sum += float(eta_values.sum())
        alpha_sum += float(alpha_values.sum())
        gate_count += len(eta_values)

        for stage in STAGES:
            mean_val[stage].add(means[stage], y)
            pred = predictions(
                means[stage],
                ridge_w=ridge_models[stage],
                centroid_unit=mean_train_centroids[stage],
            )
            val_correct_ridge[stage] += int(np.sum(pred["ridge"] == y))
            val_correct_ncm_mean[stage] += int(np.sum(pred["ncm"] == y))

        for stage in TEMPORAL_STAGES:
            ts = temporal_summary(seq[stage])
            temporal_val[stage].add(ts, y)
            pred = predictions(ts, centroid_unit=temporal_train_centroids[stage])
            val_correct_ncm_temporal[stage] += int(np.sum(pred["ncm"] == y))

        if bi % 5 == 0 or bi + 1 == val_steps:
            report(status, args.protocol, "Frozen probes on validation", bi + 1, val_steps)

    eta_mean = eta_sum / gate_count
    alpha_mean = alpha_sum / gate_count
    actual_val = model_correct / val_seen

    stage_report = {}
    for stage in STAGES:
        train_acc = train_correct_ridge[stage] / train_seen
        val_acc = val_correct_ridge[stage] / val_seen
        train_scatter = mean_train[stage].scatter()
        val_scatter = mean_val[stage].scatter()
        stage_report[stage] = {
            "dimension": mean_train[stage].dim,
            "ridge_lambda": ridge_lambda[stage],
            "train_ridge_accuracy": train_acc,
            "val_ridge_accuracy": val_acc,
            "generalization_gap_pp": 100.0 * (train_acc - val_acc),
            "val_over_train_accuracy_ratio": val_acc / max(train_acc, 1e-12),
            "train_ncm_accuracy_mean": train_correct_ncm_mean[stage] / train_seen,
            "val_ncm_accuracy_mean": val_correct_ncm_mean[stage] / val_seen,
            "train_fisher": train_scatter["fisher_between_over_within"],
            "val_fisher": val_scatter["fisher_between_over_within"],
            "val_over_train_fisher_ratio": (
                val_scatter["fisher_between_over_within"]
                / max(train_scatter["fisher_between_over_within"], 1e-12)
            ),
            "train_val_class_centroid_alignment": centroid_alignment(
                mean_train[stage], mean_val[stage]
            ),
        }

    temporal_report = {}
    for stage in TEMPORAL_STAGES:
        train_mean = train_correct_ncm_mean[stage] / train_seen
        val_mean = val_correct_ncm_mean[stage] / val_seen
        train_temporal = train_correct_ncm_temporal[stage] / train_seen
        val_temporal = val_correct_ncm_temporal[stage] / val_seen
        temporal_report[stage] = {
            "summary": "mean + std + last_minus_first",
            "dimension": temporal_train[stage].dim,
            "train_ncm_mean_accuracy": train_mean,
            "train_ncm_temporal_accuracy": train_temporal,
            "train_temporal_headroom_pp": 100.0 * (train_temporal - train_mean),
            "val_ncm_mean_accuracy": val_mean,
            "val_ncm_temporal_accuracy": val_temporal,
            "val_temporal_headroom_pp": 100.0 * (val_temporal - val_mean),
            "temporal_generalization_gap_pp": 100.0 * (train_temporal - val_temporal),
            "train_val_temporal_centroid_alignment": centroid_alignment(
                temporal_train[stage], temporal_val[stage]
            ),
        }

    transition_report = {}
    for left, right in zip(STAGES[:-1], STAGES[1:]):
        train_gain = (
            stage_report[right]["train_ridge_accuracy"]
            - stage_report[left]["train_ridge_accuracy"]
        )
        val_gain = (
            stage_report[right]["val_ridge_accuracy"]
            - stage_report[left]["val_ridge_accuracy"]
        )
        transition_report[f"{left}_to_{right}"] = {
            "train_probe_gain_pp": 100.0 * train_gain,
            "val_probe_gain_pp": 100.0 * val_gain,
            "excess_train_gain_over_val_pp": 100.0 * (train_gain - val_gain),
            "generalization_gap_change_pp": (
                stage_report[right]["generalization_gap_pp"]
                - stage_report[left]["generalization_gap_pp"]
            ),
        }

    # Cheap causal gate-timing tests on the same validation set.
    constant_acc = counterfactual_accuracy(
        config, params, dataset, val_ids, args.batch_size,
        "constant", eta_mean, alpha_mean, status, args.protocol
    )
    reverse_acc = counterfactual_accuracy(
        config, params, dataset, val_ids, args.batch_size,
        "reverse_time", eta_mean, alpha_mean, status, args.protocol
    )

    largest_gap_stage = max(
        STAGES, key=lambda s: stage_report[s]["generalization_gap_pp"]
    )
    largest_transition = max(
        transition_report,
        key=lambda k: transition_report[k]["excess_train_gain_over_val_pp"],
    )
    largest_temporal = max(
        TEMPORAL_STAGES,
        key=lambda s: temporal_report[s]["val_temporal_headroom_pp"],
    )

    final = {
        "audit": "NestSAR R4 generalization localization v1",
        "read_only": True,
        "protocol": args.protocol,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_val_accuracy": float(payload.get("val_accuracy", float("nan"))),
        "recomputed_model_val_accuracy": actual_val,
        "params": nparams,
        "split_counts": {"train": len(train_ids), "val": len(val_ids)},
        "stage_generalization": stage_report,
        "transition_localization": transition_report,
        "temporal_information_headroom": temporal_report,
        "gate_adaptivity": {
            "learned_accuracy": actual_val,
            "eta_global_validation_mean": eta_mean,
            "alpha_global_validation_mean": alpha_mean,
            "constant_gate_accuracy": constant_acc,
            "constant_gate_delta_pp": 100.0 * (constant_acc - actual_val),
            "reverse_time_gate_accuracy": reverse_acc,
            "reverse_time_gate_delta_pp": 100.0 * (reverse_acc - actual_val),
        },
        "automatic_localization": {
            "largest_absolute_probe_generalization_gap_stage": largest_gap_stage,
            "largest_excess_train_gain_transition": largest_transition,
            "largest_validation_temporal_headroom_stage": largest_temporal,
            "descriptor_probe_minus_actual_model_pp": 100.0 * (
                stage_report["descriptor"]["val_ridge_accuracy"] - actual_val
            ),
        },
        "interpretation_rules": [
            "A transition with a large positive excess_train_gain_over_val_pp is a candidate point where training separability grows faster than validation separability.",
            "A large positive val_temporal_headroom_pp means useful temporal information is present at that stage but mean-only summarization underuses it.",
            "If constant/reversed gates preserve accuracy, fast-memory gate timing is not carrying important discriminative information.",
            "These are localization diagnostics; a final causal claim still requires a matched intervention at the implicated stage.",
        ],
    }

    path = output / "generalization_localization.json"
    atomic_json(path, final)
    report(
        status, args.protocol, "Done", 1, 1, done=True,
        val_acc=actual_val, report_path=str(path)
    )

    print("=" * 120)
    print(f"{args.protocol.upper()} GENERALIZATION LOCALIZATION COMPLETE")
    print("=" * 120)
    print(json.dumps(final, indent=2))
    print(f"REPORT: {path}")


if __name__ == "__main__":
    main()

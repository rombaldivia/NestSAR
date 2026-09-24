#!/usr/bin/env python3
from __future__ import annotations

"""Read-only audit of trained NestSAR R4 checkpoints.

No training. No checkpoint mutation. The audit measures:
  * exact checkpoint/config/compute integrity;
  * full-split stage-wise frozen ridge probes + nearest-centroid accuracy;
  * Fisher-style class separability at Input(T-mean)->Spatial->M4->Router->G4->Descriptor;
  * full-validation fast-weight counterfactuals (M4 off / G4 off / all off);
  * trained R4 memory-state singular spectra and effective-rank use;
  * eta/alpha distributions and fast/base residual RMS ratios;
  * top confusion pairs and their centroid separation through the pipeline.
"""

import argparse
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import serialization

from experiments.nestsar_sm_all_t16 import model as sm
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json
from experiments.nestsar_sm_all_t16.streaming import worker as train_worker
from experiments.nestsar_sm_all_t16.streaming.audit import audit_model as compute_audit

NUM_CLASSES = 120
EXPECTED_PARAMS = 1_831_932
STAGE_ORDER = ("input_tmean", "spatial", "m4", "router", "g4", "descriptor")


def report(path: Path, protocol: str, phase: str, current: int, total: int, **extra):
    payload = {
        "protocol": protocol,
        "phase": phase,
        "current": int(current),
        "total": int(max(total, 1)),
    }
    payload.update(extra)
    atomic_json(path, payload)


def tree_param_count(tree) -> int:
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def load_checkpoint(path: Path):
    payload = serialization.msgpack_restore(path.read_bytes())
    if not isinstance(payload, dict) or "ema_params" not in payload or "config" not in payload:
        raise ValueError(f"Expected best-checkpoint payload with ema_params/config: {path}")
    return payload, payload["ema_params"], dict(payload["config"])


def l2_rows(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def stage_features(x, out):
    # All learned stages are reduced in the same way: temporal mean, preserve streams.
    # Input T-mean preserves every joint/person/channel coordinate (750-D) while
    # removing temporal order; it is a spatial-information reference, not a final model.
    return {
        "input_tmean": np.asarray(x, np.float32).mean(axis=1),
        "spatial": np.asarray(out["spatial_stack"]).mean(axis=1).reshape(len(x), -1),
        "m4": np.asarray(out["frame_stack"]).mean(axis=1).reshape(len(x), -1),
        "router": np.asarray(out["mixed_frame_stack"]).mean(axis=1).reshape(len(x), -1),
        "g4": np.asarray(out["chunk_states"]).mean(axis=2).reshape(len(x), -1),
        "descriptor": np.asarray(out["descriptors"]).reshape(len(x), -1),
    }


class ProbeStats:
    def __init__(self, dim):
        self.dim = int(dim)
        self.xtx = np.zeros((dim, dim), np.float64)
        self.sumx = np.zeros(dim, np.float64)
        self.class_sum = np.zeros((NUM_CLASSES, dim), np.float64)
        self.counts = np.zeros(NUM_CLASSES, np.int64)
        self.sum_norm2 = 0.0
        self.n = 0

    def add(self, x, y):
        z = l2_rows(x)
        self.xtx += np.asarray(z.T @ z, np.float64)
        self.sumx += np.asarray(z.sum(axis=0), np.float64)
        self.sum_norm2 += float(np.square(z, dtype=np.float64).sum())
        self.n += len(z)
        self.counts += np.bincount(y, minlength=NUM_CLASSES)
        np.add.at(self.class_sum, y, z)

    def finalize(self, ridge):
        if np.any(self.counts == 0):
            missing = np.where(self.counts == 0)[0].tolist()
            raise RuntimeError(f"Probe training split is missing classes: {missing}")
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

        centroids = self.class_sum / self.counts[:, None]
        centroid_unit = l2_rows(centroids)

        global_mean = self.sumx / self.n
        within = self.sum_norm2 - float(
            np.sum(np.square(self.class_sum).sum(axis=1) / self.counts)
        )
        between = float(
            np.sum(self.counts[:, None] * np.square(centroids - global_mean[None, :]))
        )
        fisher = between / max(within, 1e-12)
        return {
            "weights": w,
            "centroids": centroids,
            "centroid_unit": centroid_unit,
            "ridge_lambda": lam,
            "fisher_between_over_within": fisher,
            "within_scatter": within,
            "between_scatter": between,
        }


def eval_probe(z, y, solved):
    z = l2_rows(z)
    w = solved["weights"]
    logits = z @ w[:-1] + w[-1]
    ridge_pred = logits.argmax(axis=1)
    centroid_pred = (z @ solved["centroid_unit"].T).argmax(axis=1)
    return int(np.sum(ridge_pred == y)), int(np.sum(centroid_pred == y))


def make_audit_fast(mode: str):
    if mode not in {"on", "m4_off", "g4_off", "all_off"}:
        raise ValueError(mode)

    class AuditFastWeightDeltaResidual(nn.Module):
        dim: int
        rank: int = 4

        @nn.compact
        def __call__(self, x, eta, alpha):
            n = nn.LayerNorm(name="value_norm")(x)
            k = nn.Dense(
                self.rank, use_bias=False,
                kernel_init=nn.initializers.normal(0.02), name="key"
            )(n)
            q = nn.Dense(
                self.rank, use_bias=False,
                kernel_init=nn.initializers.normal(0.02), name="query"
            )(n)
            k = jnp.tanh(k)
            q = jnp.tanh(q)
            k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-6)
            q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-6)
            memory0 = self.param(
                "memory0", nn.initializers.normal(0.01), (self.rank, self.dim)
            )
            memory = jnp.broadcast_to(
                memory0[None, :, :], (x.shape[0], self.rank, self.dim)
            )
            kt, qt, vt = (jnp.swapaxes(a, 0, 1) for a in (k, q, n))
            et, at = jnp.swapaxes(eta, 0, 1), jnp.swapaxes(alpha, 0, 1)

            def step(mem, inputs):
                key_t, query_t, value_t, eta_t, alpha_t = inputs
                pred_t = jnp.einsum("br,brd->bd", key_t, mem)
                err_t = value_t - pred_t
                delta_t = jnp.einsum("br,bd->brd", key_t, err_t)
                mem = alpha_t[..., None] * mem + eta_t[..., None] * delta_t
                read_t = jnp.einsum("br,brd->bd", query_t, mem)
                return mem, read_t

            _, reads = jax.lax.scan(step, memory, (kt, qt, vt, et, at))
            reads = jnp.swapaxes(reads, 0, 1)
            is_m4 = x.shape[1] == 16
            is_g4 = x.shape[1] == 4
            disabled = (
                mode == "all_off"
                or (mode == "m4_off" and is_m4)
                or (mode == "g4_off" and is_g4)
            )
            return jnp.zeros_like(reads) if disabled else reads

    return AuditFastWeightDeltaResidual


def counterfactual_accuracy(config, params, dataset, val_ids, batch_size, mode, status, protocol):
    original = sm.FastWeightDeltaResidual
    sm.FastWeightDeltaResidual = make_audit_fast(mode)
    try:
        model = train_worker.make_model(config)
        apply_logits = jax.jit(
            lambda p, x: model.apply({"params": p}, x, training=False)["logits"]
        )
        total_steps = math.ceil(len(val_ids) / batch_size)
        correct = 0
        seen = 0
        for bi, start in enumerate(range(0, len(val_ids), batch_size)):
            ids = np.asarray(val_ids[start:start + batch_size], np.int64)
            x = np.asarray(dataset.canonical[ids], np.float32)
            y = np.asarray(dataset.labels[ids], np.int32)
            logits = np.asarray(jax.device_get(apply_logits(params, jax.device_put(x))))
            correct += int(np.sum(logits.argmax(axis=1) == y))
            seen += len(y)
            if bi % 10 == 0 or bi + 1 == total_steps:
                report(status, protocol, f"Counterfactual {mode}", bi + 1, total_steps)
        return correct / seen
    finally:
        sm.FastWeightDeltaResidual = original


def flatten_intermediates(node, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(node, Mapping):
        if "__call__" in node:
            value = node["__call__"]
            if isinstance(value, (tuple, list)):
                value = value[-1]
            out[prefix] = value
        for key, value in node.items():
            if key == "__call__":
                continue
            child = f"{prefix}/{key}" if prefix else str(key)
            flatten_intermediates(value, child, out)
    return out


def reconstruct_memory_singulars(base_y, eta, alpha, fw_params):
    x = np.asarray(base_y, np.float32)
    scale = np.asarray(fw_params["value_norm"]["scale"], np.float32)
    bias = np.asarray(fw_params["value_norm"]["bias"], np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = np.mean(np.square(x - mean), axis=-1, keepdims=True)
    n = (x - mean) / np.sqrt(var + 1e-6)
    n = n * scale + bias
    kernel = np.asarray(fw_params["key"]["kernel"], np.float32)
    k = np.tanh(n @ kernel)
    k /= np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-6)
    mem0 = np.asarray(fw_params["memory0"], np.float32)
    mem = np.broadcast_to(mem0[None, :, :], (len(x), *mem0.shape)).copy()

    spectra = []
    for t in range(x.shape[1]):
        kt = k[:, t]
        vt = n[:, t]
        pred = np.einsum("br,brd->bd", kt, mem)
        err = vt - pred
        delta = np.einsum("br,bd->brd", kt, err)
        mem = alpha[:, t, :, None] * mem + eta[:, t, :, None] * delta
        spectra.append(np.linalg.svd(mem, compute_uv=False))
    return np.concatenate(spectra, axis=0)


def summarize_spectra(s):
    s = np.asarray(s, np.float64)
    energy = np.square(s)
    total = np.maximum(energy.sum(axis=1), 1e-12)
    participation = np.square(total) / np.maximum(np.square(energy).sum(axis=1), 1e-12)
    p = energy / total[:, None]
    entropy_rank = np.exp(-np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1))
    rel = s / np.maximum(s[:, :1], 1e-12)
    result = {
        "samples_times_steps": int(len(s)),
        "mean_singular_values": s.mean(axis=0).tolist(),
        "mean_relative_to_sigma1": rel.mean(axis=0).tolist(),
        "mean_participation_rank": float(participation.mean()),
        "mean_entropy_effective_rank": float(entropy_rank.mean()),
    }
    if s.shape[1] >= 4:
        result["mean_energy_fraction_rank3_4"] = float(
            np.mean(energy[:, 2:4].sum(axis=1) / total)
        )
        result["median_energy_fraction_rank3_4"] = float(
            np.median(energy[:, 2:4].sum(axis=1) / total)
        )
    return result


def kernel_spectrum(fw_params):
    result = {}
    for name in ("key", "query"):
        w = np.asarray(fw_params[name]["kernel"], np.float64)
        s = np.linalg.svd(w, compute_uv=False)
        result[name] = {
            "singular_values": s.tolist(),
            "relative_to_first": (s / max(s[0], 1e-12)).tolist(),
        }
    return result


def memory_rank_audit(model, params, config, dataset, val_ids, sample_count, batch_size, status, protocol):
    chosen = np.asarray(val_ids[: min(sample_count, len(val_ids))], np.int64)
    predicate = lambda mdl, method: (
        method == "__call__"
        and mdl.__class__.__name__ in {
            "BiMemory", "FastWeightDeltaResidual", "SharedSMController"
        }
    )

    def captured_apply(p, x):
        return model.apply(
            {"params": p}, x, training=False,
            capture_intermediates=predicate, mutable=["intermediates"]
        )

    captured_apply = jax.jit(captured_apply)
    spectra = {f"M4_{i}": [] for i in range(4)}
    spectra.update({f"G4_{i}": [] for i in range(4)})
    residual_acc = {
        **{f"M4_{i}": [0.0, 0.0] for i in range(4)},
        **{f"G4_{i}": [0.0, 0.0] for i in range(4)},
    }
    gate_eta, gate_alpha = [], []
    total_steps = math.ceil(len(chosen) / batch_size)

    for bi, start in enumerate(range(0, len(chosen), batch_size)):
        ids = chosen[start:start + batch_size]
        x = np.asarray(dataset.canonical[ids], np.float32)
        _, mutable = jax.device_get(captured_apply(params, jax.device_put(x)))
        flat = flatten_intermediates(mutable["intermediates"])
        controller = flat.get("sm_controller")
        if controller is None:
            raise KeyError(
                "Could not capture sm_controller. Captured keys: " + ", ".join(sorted(flat))
            )
        eta = np.asarray(controller["eta"], np.float32)
        alpha = np.asarray(controller["alpha"], np.float32)
        gate_eta.append(eta.reshape(-1))
        gate_alpha.append(alpha.reshape(-1))
        eta_slow = eta.reshape(len(x), 4, 4, 1).mean(axis=2)
        alpha_slow = alpha.reshape(len(x), 4, 4, 1).mean(axis=2)

        for i in range(4):
            m4_base_key = f"frame_memory_{i}/base_memory"
            m4_fast_key = f"frame_memory_{i}/fast_weight"
            g4_base_key = f"descriptor_{i}/chunk_memory/base_memory"
            g4_fast_key = f"descriptor_{i}/chunk_memory/fast_weight"
            for key in (m4_base_key, m4_fast_key, g4_base_key, g4_fast_key):
                if key not in flat:
                    raise KeyError(
                        f"Missing captured module {key}. Available: {sorted(flat)}"
                    )

            m4_base = np.asarray(flat[m4_base_key], np.float32)
            m4_fast = np.asarray(flat[m4_fast_key], np.float32)
            g4_base = np.asarray(flat[g4_base_key], np.float32)
            g4_fast = np.asarray(flat[g4_fast_key], np.float32)

            m4_fw = params[f"frame_memory_{i}"]["fast_weight"]
            g4_fw = params[f"descriptor_{i}"]["chunk_memory"]["fast_weight"]
            spectra[f"M4_{i}"].append(
                reconstruct_memory_singulars(m4_base, eta, alpha, m4_fw)
            )
            spectra[f"G4_{i}"].append(
                reconstruct_memory_singulars(g4_base, eta_slow, alpha_slow, g4_fw)
            )

            scale = float(config["sm_residual_scale"])
            residual_acc[f"M4_{i}"][0] += float(np.square(scale * m4_fast).sum())
            residual_acc[f"M4_{i}"][1] += float(np.square(m4_base).sum())
            residual_acc[f"G4_{i}"][0] += float(np.square(scale * g4_fast).sum())
            residual_acc[f"G4_{i}"][1] += float(np.square(g4_base).sum())

        report(status, protocol, "R4 state-rank audit", bi + 1, total_steps)

    module_report = {}
    for name, pieces in spectra.items():
        merged = np.concatenate(pieces, axis=0)
        stage, idx = name.split("_")
        i = int(idx)
        fw = (
            params[f"frame_memory_{i}"]["fast_weight"]
            if stage == "M4"
            else params[f"descriptor_{i}"]["chunk_memory"]["fast_weight"]
        )
        fast_sq, base_sq = residual_acc[name]
        module_report[name] = {
            **summarize_spectra(merged),
            "effective_fast_over_base_rms": math.sqrt(fast_sq / max(base_sq, 1e-12)),
            "address_kernel_spectrum": kernel_spectrum(fw),
        }

    eta = np.concatenate(gate_eta)
    alpha = np.concatenate(gate_alpha)

    def dist(a):
        q = np.percentile(a, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "percentiles_0_1_5_25_50_75_95_99_100": q.tolist(),
        }

    return {
        "samples": int(len(chosen)),
        "modules": module_report,
        "eta": dist(eta),
        "alpha": dist(alpha),
    }


def top_confusion_pairs(confusion, solved, n=20):
    pairs = []
    for a in range(NUM_CLASSES):
        for b in range(a + 1, NUM_CLASSES):
            count = int(confusion[a, b] + confusion[b, a])
            if count:
                pairs.append((count, a, b))
    pairs.sort(reverse=True)
    result = []
    for count, a, b in pairs[:n]:
        row = {
            "symmetric_confusions": count,
            "class_a_zero_based": a,
            "class_b_zero_based": b,
            "ntu_action_a": a + 1,
            "ntu_action_b": b + 1,
            "stage_centroid_cosine_distance": {},
        }
        for stage in STAGE_ORDER:
            ca = solved[stage]["centroid_unit"][a]
            cb = solved[stage]["centroid_unit"][b]
            row["stage_centroid_cosine_distance"][stage] = float(1.0 - np.dot(ca, cb))
        result.append(row)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--rank-samples", type=int, default=512)
    p.add_argument("--rank-batch-size", type=int, default=64)
    p.add_argument("--ridge", type=float, default=1e-2)
    args = p.parse_args()

    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise RuntimeError(
            f"Expected one isolated GPU, got backend={jax.default_backend()} devices={jax.local_devices()}"
        )

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    status = outdir / "status.json"
    final_json = outdir / "trained_r4_audit.json"
    checkpoint = Path(args.checkpoint)
    cache = Path(args.cache)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (cache / "manifest.json").is_file():
        raise FileNotFoundError(cache / "manifest.json")

    report(status, args.protocol, "Load checkpoint", 0, 1)
    payload, params, config = load_checkpoint(checkpoint)
    if int(config.get("fast_rank", -1)) != 4 or int(config.get("head_rank", -1)) != 2:
        raise RuntimeError(f"Expected trained R4/head-R2 config, found {config}")
    nparams = tree_param_count(params)
    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(f"EMA param count {nparams} != {EXPECTED_PARAMS}")

    dataset = Dataset(cache)
    train_ids = dataset.splits[f"{args.protocol}_train"]
    val_ids = dataset.splits[f"{args.protocol}_val"]
    model = train_worker.make_model(config)

    # Exact static-unrolled compute audit on the trained parameter tree.
    report(status, args.protocol, "Static compute audit", 0, 1)
    compute = compute_audit(model, params)
    report(status, args.protocol, "Static compute audit", 1, 1)

    # Verify audit replacement reproduces the original fast path exactly.
    probe_ids = np.asarray(val_ids[: min(8, len(val_ids))], np.int64)
    probe_x = jnp.asarray(np.asarray(dataset.canonical[probe_ids], np.float32))
    original_logits = np.asarray(
        jax.device_get(model.apply({"params": params}, probe_x, training=False)["logits"])
    )
    original_fast = sm.FastWeightDeltaResidual
    sm.FastWeightDeltaResidual = make_audit_fast("on")
    try:
        on_model = train_worker.make_model(config)
        on_logits = np.asarray(
            jax.device_get(on_model.apply({"params": params}, probe_x, training=False)["logits"])
        )
    finally:
        sm.FastWeightDeltaResidual = original_fast
    fast_on_max_abs_diff = float(np.max(np.abs(original_logits - on_logits)))
    if fast_on_max_abs_diff > 2e-5:
        raise RuntimeError(
            f"Audit fast-weight reimplementation mismatch: max abs diff={fast_on_max_abs_diff}"
        )

    # Full training split: frozen sufficient statistics for linear ridge probes.
    forward = jax.jit(lambda p, x: model.apply({"params": p}, x, training=False))
    stage_stats = {}
    train_steps = math.ceil(len(train_ids) / args.batch_size)
    for bi, start in enumerate(range(0, len(train_ids), args.batch_size)):
        ids = np.asarray(train_ids[start:start + args.batch_size], np.int64)
        x = np.asarray(dataset.canonical[ids], np.float32)
        y = np.asarray(dataset.labels[ids], np.int32)
        out = jax.device_get(forward(params, jax.device_put(x)))
        feats = stage_features(x, out)
        if not stage_stats:
            stage_stats = {name: ProbeStats(feats[name].shape[1]) for name in STAGE_ORDER}
        for name in STAGE_ORDER:
            stage_stats[name].add(feats[name], y)
        if bi % 5 == 0 or bi + 1 == train_steps:
            report(status, args.protocol, "Frozen-probe train stats", bi + 1, train_steps)

    solved = {name: stage_stats[name].finalize(args.ridge) for name in STAGE_ORDER}

    # Full validation split: actual model accuracy + frozen probe/NCM accuracy + confusion.
    probe_correct = {name: 0 for name in STAGE_ORDER}
    ncm_correct = {name: 0 for name in STAGE_ORDER}
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    model_correct = 0
    seen = 0
    val_steps = math.ceil(len(val_ids) / args.batch_size)
    for bi, start in enumerate(range(0, len(val_ids), args.batch_size)):
        ids = np.asarray(val_ids[start:start + args.batch_size], np.int64)
        x = np.asarray(dataset.canonical[ids], np.float32)
        y = np.asarray(dataset.labels[ids], np.int32)
        out = jax.device_get(forward(params, jax.device_put(x)))
        pred = np.asarray(out["logits"]).argmax(axis=1)
        model_correct += int(np.sum(pred == y))
        seen += len(y)
        np.add.at(confusion, (y, pred), 1)
        feats = stage_features(x, out)
        for name in STAGE_ORDER:
            a, b = eval_probe(feats[name], y, solved[name])
            probe_correct[name] += a
            ncm_correct[name] += b
        if bi % 5 == 0 or bi + 1 == val_steps:
            report(status, args.protocol, "Frozen-probe validation", bi + 1, val_steps)

    stage_report = {}
    for name in STAGE_ORDER:
        stage_report[name] = {
            "dimension": stage_stats[name].dim,
            "ridge_probe_val_accuracy": probe_correct[name] / seen,
            "nearest_centroid_val_accuracy": ncm_correct[name] / seen,
            "fisher_between_over_within": solved[name]["fisher_between_over_within"],
            "ridge_lambda": solved[name]["ridge_lambda"],
        }

    # Full-validation causal fast-memory ablations.
    counterfactual = {"on": model_correct / seen}
    for mode in ("m4_off", "g4_off", "all_off"):
        counterfactual[mode] = counterfactual_accuracy(
            config, params, dataset, val_ids, args.batch_size,
            mode, status, args.protocol
        )

    # Actual trained R4 state spectrum on real held-out clips.
    rank_report = memory_rank_audit(
        model, params, config, dataset, val_ids,
        args.rank_samples, args.rank_batch_size, status, args.protocol
    )

    # Compact pairwise evidence on the classes the trained model confuses most.
    pair_report = top_confusion_pairs(confusion, solved, 20)

    # Descriptive deltas only; interpretation should be made from the full report.
    probe_deltas = {}
    for left, right in zip(STAGE_ORDER[:-1], STAGE_ORDER[1:]):
        probe_deltas[f"{left}_to_{right}"] = (
            stage_report[right]["ridge_probe_val_accuracy"]
            - stage_report[left]["ridge_probe_val_accuracy"]
        )

    final = {
        "audit": "NestSAR trained R4 bottleneck audit v1",
        "read_only": True,
        "protocol": args.protocol,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_val_accuracy": float(payload.get("val_accuracy", float("nan"))),
        "params": nparams,
        "config": {
            "frames": 16,
            "model_dim": int(config["model_dim"]),
            "fast_rank": int(config["fast_rank"]),
            "head_rank": int(config["head_rank"]),
            "sm_residual_scale": float(config["sm_residual_scale"]),
        },
        "split_counts": {"train": len(train_ids), "val": len(val_ids)},
        "compute": compute,
        "audit_fast_on_max_abs_logit_difference": fast_on_max_abs_diff,
        "model_full_val_accuracy_recomputed": model_correct / seen,
        "stage_frozen_probes": stage_report,
        "stage_probe_accuracy_deltas": probe_deltas,
        "fast_weight_counterfactual_val_accuracy": counterfactual,
        "fast_weight_counterfactual_delta_pp_vs_on": {
            k: 100.0 * (v - counterfactual["on"])
            for k, v in counterfactual.items() if k != "on"
        },
        "trained_r4_state_utilization": rank_report,
        "top_confusion_pairs": pair_report,
        "notes": [
            "input_tmean is a spatial-information reference with temporal order removed; do not compare it as a full action model.",
            "Frozen ridge probes train only a closed-form linear readout; NestSAR/EMA weights never change.",
            "Rank-state spectra are reconstructed from the trained EMA fast-weight parameters and real held-out activations.",
            "Counterfactuals retain all trained parameters and set only the selected fast residual reads to zero.",
        ],
    }
    atomic_json(final_json, final)
    report(
        status, args.protocol, "Done", 1, 1, done=True,
        model_val=counterfactual["on"],
        report_path=str(final_json)
    )
    print("=" * 120)
    print(f"{args.protocol.upper()} TRAINED R4 AUDIT COMPLETE")
    print("=" * 120)
    print(json.dumps(final, indent=2))
    print(f"\nREPORT: {final_json}")


if __name__ == "__main__":
    main()

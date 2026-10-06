#!/usr/bin/env python3
from __future__ import annotations

"""Read-only bottleneck audit for the ~30-MFLOP NestSAR R4 family.

No training. No checkpoint mutation.

For one NTU120 protocol this audit performs three complementary diagnostics on
the exact EMA best checkpoint:

A) Stage geometry transfer
   Spatial -> M4 -> Router -> G4 -> Descriptor
   Uses normalized fused representations and TRAIN prototypes only.
   TRAIN uses leave-one-out true-class prototypes to avoid self leakage.

B) Stream / fusion / adaptive-head audit
   J, B, JM, BM stream accuracies, any-stream oracle, main fusion accuracy,
   final adaptive-head accuracy, fixed/broken counts.

C) Hard-class audit
   Validation confusion pairs, descriptor prototype rivals, pair overlap,
   weakest validation class gaps, and per-class train->val gap loss.

The purpose is localization, not hyperparameter selection. Validation labels are
used only for diagnosis and must not be used to construct a training hard-pair
list without a separate training-only rule.
"""

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming import worker
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json


EXPECTED_PARAMS = 1_831_932
EXPECTED_MODEL = "NestSAR-SM-ALL-T16-R4-v1"
NUM_CLASSES = 120
STAGE_NAMES = ("Spatial", "M4", "Router", "G4", "Descriptor")
STREAM_NAMES = ("J", "B", "JM", "BM")


def normalize_np(x, eps=1e-12):
    x = np.asarray(x, np.float64)
    n = np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    return x / n


def count_params(params):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)))


def load_checkpoint(path):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    required = {
        "model", "protocol", "epoch", "val_accuracy", "ema_params", "config",
        "cache_signature",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")
    if payload["model"] != EXPECTED_MODEL:
        raise ValueError(
            f"Expected {EXPECTED_MODEL}, checkpoint says {payload['model']}"
        )
    return payload


def label_name(label):
    return f"A{int(label) + 1:03d}"


def iter_batches(dataset, ids, batch_size):
    ids = np.asarray(ids, np.int64)
    for start in range(0, len(ids), batch_size):
        take = ids[start:start + batch_size]
        n = len(take)
        x = np.zeros((batch_size, 16, 750), np.float32)
        y = np.zeros((batch_size,), np.int32)
        x[:n] = dataset.canonical[take]
        y[:n] = dataset.labels[take]
        yield x, y, n


def make_infer(model):
    @jax.jit
    def infer(params, x):
        out = model.apply({"params": params}, x, training=False)
        fusion = out["fusion_weights"]

        def temporal_fuse(v):
            # [B,T,S,D] -> [B,S,D] -> [B,D]
            stream = jnp.mean(v, axis=1)
            return jnp.einsum("bs,bsd->bd", fusion, stream)

        spatial = temporal_fuse(out["spatial_stack"])
        m4 = temporal_fuse(out["frame_stack"])
        router = temporal_fuse(out["mixed_frame_stack"])

        # [B,S,4,D] -> [B,S,D] -> [B,D]
        g4_stream = jnp.mean(out["chunk_states"], axis=2)
        g4 = jnp.einsum("bs,bsd->bd", fusion, g4_stream)

        descriptor = jnp.einsum(
            "bs,bsd->bd",
            fusion,
            out["descriptors"],
        )

        stages = jnp.stack(
            [spatial, m4, router, g4, descriptor],
            axis=1,
        )
        stages = stages * jax.lax.rsqrt(
            jnp.maximum(
                jnp.sum(jnp.square(stages), axis=-1, keepdims=True),
                1e-12,
            )
        )

        return (
            out["logits"],
            out["main_logits"],
            out["stream_logits"],
            stages,
            fusion,
        )

    return infer


def new_model_stats():
    return {
        "n": 0,
        "final_correct": 0,
        "main_correct": 0,
        "oracle_correct": 0,
        "fixed_by_adaptive": 0,
        "broken_by_adaptive": 0,
        "stream_correct": np.zeros((4,), np.int64),
        "confusion": np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64),
        "class_total": np.zeros((NUM_CLASSES,), np.int64),
        "class_correct": np.zeros((NUM_CLASSES,), np.int64),
        "fusion_sum": np.zeros((4,), np.float64),
    }


def update_model_stats(stats, logits, main_logits, stream_logits, fusion, y):
    pred = logits.argmax(-1)
    main_pred = main_logits.argmax(-1)
    stream_pred = stream_logits.argmax(-1)  # [B,4]
    final_ok = pred == y
    main_ok = main_pred == y
    stream_ok = stream_pred == y[:, None]

    stats["n"] += len(y)
    stats["final_correct"] += int(final_ok.sum())
    stats["main_correct"] += int(main_ok.sum())
    stats["oracle_correct"] += int(np.any(stream_ok, axis=1).sum())
    stats["fixed_by_adaptive"] += int((final_ok & ~main_ok).sum())
    stats["broken_by_adaptive"] += int((~final_ok & main_ok).sum())
    stats["stream_correct"] += stream_ok.sum(axis=0).astype(np.int64)
    stats["fusion_sum"] += fusion.sum(axis=0)

    np.add.at(stats["confusion"], (y, pred), 1)
    np.add.at(stats["class_total"], y, 1)
    np.add.at(stats["class_correct"], y, final_ok.astype(np.int64))


def finalize_model_stats(stats):
    n = max(int(stats["n"]), 1)
    stream = {
        name: float(stats["stream_correct"][i] / n)
        for i, name in enumerate(STREAM_NAMES)
    }
    per_class = []
    for c in range(NUM_CLASSES):
        total = int(stats["class_total"][c])
        correct = int(stats["class_correct"][c])
        per_class.append({
            "class": label_name(c),
            "support": total,
            "accuracy": None if total == 0 else float(correct / total),
        })
    return {
        "samples": int(stats["n"]),
        "final_accuracy": float(stats["final_correct"] / n),
        "main_fusion_accuracy": float(stats["main_correct"] / n),
        "stream_accuracy": stream,
        "any_stream_oracle_accuracy": float(stats["oracle_correct"] / n),
        "oracle_minus_final_pp": 100.0 * float(
            (stats["oracle_correct"] - stats["final_correct"]) / n
        ),
        "adaptive_fixed": int(stats["fixed_by_adaptive"]),
        "adaptive_broken": int(stats["broken_by_adaptive"]),
        "adaptive_net_clips": int(
            stats["fixed_by_adaptive"] - stats["broken_by_adaptive"]
        ),
        "adaptive_net_pp": 100.0 * float(
            (stats["fixed_by_adaptive"] - stats["broken_by_adaptive"]) / n
        ),
        "mean_fusion_weights": {
            name: float(stats["fusion_sum"][i] / n)
            for i, name in enumerate(STREAM_NAMES)
        },
        "per_class_accuracy": per_class,
    }


def pass_train_prototypes(dataset, ids, batch_size, params, infer, status_path, protocol):
    sums = np.zeros((len(STAGE_NAMES), NUM_CLASSES, 112), np.float64)
    counts = np.zeros((NUM_CLASSES,), np.int64)
    model_stats = new_model_stats()
    total_batches = math.ceil(len(ids) / batch_size)

    for bi, (x, y, n) in enumerate(iter_batches(dataset, ids, batch_size), 1):
        logits, main_logits, stream_logits, stages, fusion = jax.block_until_ready(
            infer(params, jax.device_put(x))
        )
        logits = np.asarray(logits)[:n]
        main_logits = np.asarray(main_logits)[:n]
        stream_logits = np.asarray(stream_logits)[:n]
        stages = np.asarray(stages)[:n]
        fusion = np.asarray(fusion)[:n]
        yy = y[:n]

        for k in range(len(STAGE_NAMES)):
            np.add.at(sums[k], yy, stages[:, k])
        counts += np.bincount(yy, minlength=NUM_CLASSES).astype(np.int64)
        update_model_stats(
            model_stats, logits, main_logits, stream_logits, fusion, yy
        )

        if bi == 1 or bi % 10 == 0 or bi == total_batches:
            atomic_json(
                status_path,
                {
                    "protocol": protocol,
                    "phase": "TRAIN prototypes",
                    "current": bi,
                    "total": total_batches,
                },
            )

    if np.any(counts < 2):
        bad = np.where(counts < 2)[0].tolist()
        raise RuntimeError(f"Need >=2 train samples/class for LOO audit, bad={bad}")

    prototypes = normalize_np(sums)
    return sums, counts, prototypes, model_stats


def new_geometry_accumulator():
    return {
        "gaps": [[] for _ in STAGE_NAMES],
        "class_gap_sum": np.zeros((len(STAGE_NAMES), NUM_CLASSES), np.float64),
        "class_count": np.zeros((NUM_CLASSES,), np.int64),
    }


def add_geometry(acc, stage_index, gaps, y):
    acc["gaps"][stage_index].append(np.asarray(gaps, np.float64))
    np.add.at(acc["class_gap_sum"][stage_index], y, gaps)


def geometry_pass(
    dataset,
    ids,
    batch_size,
    params,
    infer,
    train_sums,
    train_counts,
    prototypes,
    *,
    leave_one_out,
    status_path,
    protocol,
    phase,
    collect_model=False,
):
    acc = new_geometry_accumulator()
    model_stats = new_model_stats() if collect_model else None
    total_batches = math.ceil(len(ids) / batch_size)

    for bi, (x, y, n) in enumerate(iter_batches(dataset, ids, batch_size), 1):
        logits, main_logits, stream_logits, stages, fusion = jax.block_until_ready(
            infer(params, jax.device_put(x))
        )
        logits = np.asarray(logits)[:n]
        main_logits = np.asarray(main_logits)[:n]
        stream_logits = np.asarray(stream_logits)[:n]
        zall = np.asarray(stages)[:n].astype(np.float64)
        fusion = np.asarray(fusion)[:n]
        yy = y[:n]

        acc["class_count"] += np.bincount(yy, minlength=NUM_CLASSES).astype(
            np.int64
        )

        if collect_model:
            update_model_stats(
                model_stats, logits, main_logits, stream_logits, fusion, yy
            )

        for k in range(len(STAGE_NAMES)):
            z = zall[:, k]
            sims = z @ prototypes[k].T

            rows = np.arange(n)
            if leave_one_out:
                loo = train_sums[k, yy] - z
                loo = normalize_np(loo)
                true_sim = np.sum(z * loo, axis=1)
            else:
                true_sim = sims[rows, yy]

            sims[rows, yy] = -np.inf
            wrong_sim = np.max(sims, axis=1)
            gaps = true_sim - wrong_sim
            add_geometry(acc, k, gaps, yy)

        if bi == 1 or bi % 10 == 0 or bi == total_batches:
            atomic_json(
                status_path,
                {
                    "protocol": protocol,
                    "phase": phase,
                    "current": bi,
                    "total": total_batches,
                },
            )

    return acc, model_stats


def finalize_geometry(acc):
    rows = {}
    class_rows = {}
    counts = acc["class_count"].astype(np.float64)

    for k, name in enumerate(STAGE_NAMES):
        gaps = np.concatenate(acc["gaps"][k], axis=0)
        rows[name] = {
            "samples": int(len(gaps)),
            "gap_mean": float(np.mean(gaps)),
            "gap_median": float(np.median(gaps)),
            "gap_q05": float(np.quantile(gaps, 0.05)),
            "gap_q10": float(np.quantile(gaps, 0.10)),
            "gap_q25": float(np.quantile(gaps, 0.25)),
            "wrong_proto_closer_fraction": float(np.mean(gaps < 0)),
            "nearest_prototype_accuracy": float(np.mean(gaps > 0)),
        }

        means = np.divide(
            acc["class_gap_sum"][k],
            np.maximum(counts, 1.0),
        )
        class_rows[name] = [
            {
                "class": label_name(c),
                "support": int(acc["class_count"][c]),
                "gap_mean": float(means[c]),
            }
            for c in range(NUM_CLASSES)
        ]

    return rows, class_rows


def top_confusion_pairs(confusion, limit=20):
    items = []
    for a in range(NUM_CLASSES):
        for b in range(a + 1, NUM_CLASSES):
            n_ab = int(confusion[a, b])
            n_ba = int(confusion[b, a])
            total = n_ab + n_ba
            if total:
                items.append({
                    "a": label_name(a),
                    "b": label_name(b),
                    "a_to_b": n_ab,
                    "b_to_a": n_ba,
                    "total": total,
                })
    items.sort(key=lambda x: (-x["total"], x["a"], x["b"]))
    return items[:limit]


def prototype_rival_pairs(prototypes, stage_index=-1, limit=20):
    p = prototypes[stage_index]
    sim = p @ p.T
    np.fill_diagonal(sim, -np.inf)

    # One nearest rival per class, then collapse reciprocal duplicates.
    merged = {}
    directed = []
    for a in range(NUM_CLASSES):
        b = int(np.argmax(sim[a]))
        s = float(sim[a, b])
        directed.append({
            "class": label_name(a),
            "rival": label_name(b),
            "cosine": s,
        })
        key = tuple(sorted((a, b)))
        merged[key] = max(merged.get(key, -np.inf), s)

    pairs = [
        {
            "a": label_name(a),
            "b": label_name(b),
            "cosine": float(s),
        }
        for (a, b), s in merged.items()
    ]
    pairs.sort(key=lambda x: (-x["cosine"], x["a"], x["b"]))
    return pairs[:limit], directed


def pair_set(rows):
    return {
        tuple(sorted((row["a"], row["b"])))
        for row in rows
    }


def weakest_classes(class_rows, limit=20):
    rows = list(class_rows)
    rows.sort(key=lambda x: (x["gap_mean"], x["class"]))
    return rows[:limit]


def class_transfer_loss(train_class_rows, val_class_rows, limit=20):
    train = {r["class"]: r for r in train_class_rows}
    out = []
    for v in val_class_rows:
        t = train[v["class"]]
        out.append({
            "class": v["class"],
            "train_gap": float(t["gap_mean"]),
            "val_gap": float(v["gap_mean"]),
            "loss": float(t["gap_mean"] - v["gap_mean"]),
            "val_support": int(v["support"]),
        })
    out.sort(key=lambda x: (-x["loss"], x["class"]))
    return out[:limit]


def transition_transfer(train_rows, val_rows):
    stage_excess = {
        s: float(train_rows[s]["gap_mean"] - val_rows[s]["gap_mean"])
        for s in STAGE_NAMES
    }
    transitions = []
    for prev, cur in zip(STAGE_NAMES[:-1], STAGE_NAMES[1:]):
        growth = stage_excess[cur] - stage_excess[prev]
        transitions.append({
            "transition": f"{prev}->{cur}",
            "train_gap_delta": float(
                train_rows[cur]["gap_mean"] - train_rows[prev]["gap_mean"]
            ),
            "val_gap_delta": float(
                val_rows[cur]["gap_mean"] - val_rows[prev]["gap_mean"]
            ),
            "transfer_excess_growth": float(growth),
        })
    transitions.sort(
        key=lambda x: -x["transfer_excess_growth"]
    )
    return stage_excess, transitions


def print_report(result):
    print("=" * 126)
    print(
        f"NESTSAR R4 ~30M BOTTLENECK AUDIT | "
        f"{result['protocol'].upper()} | "
        f"E{result['checkpoint']['epoch']}"
    )
    print("=" * 126)
    print(
        f"Checkpoint recorded: {100*result['checkpoint']['recorded_val_accuracy']:.4f}% | "
        f"verified: {100*result['validation_model']['final_accuracy']:.4f}%"
    )
    print(f"Parameters: {result['checkpoint']['params']:,}")
    print()

    print("A) STAGE GEOMETRY TRANSFER")
    print("-" * 126)
    print(
        f"{'Stage':12s} {'Train gap':>11s} {'Val gap':>11s} "
        f"{'Excess':>11s} {'Train NP':>10s} {'Val NP':>10s} "
        f"{'Val wrong':>10s}"
    )
    for stage in STAGE_NAMES:
        tr = result["geometry"]["train"][stage]
        va = result["geometry"]["val"][stage]
        ex = result["geometry"]["stage_transfer_excess"][stage]
        print(
            f"{stage:12s} {tr['gap_mean']:11.5f} {va['gap_mean']:11.5f} "
            f"{ex:11.5f} {100*tr['nearest_prototype_accuracy']:9.2f}% "
            f"{100*va['nearest_prototype_accuracy']:9.2f}% "
            f"{100*va['wrong_proto_closer_fraction']:9.2f}%"
        )

    print()
    print("Transfer-excess growth by transition (largest first):")
    for row in result["geometry"]["transition_transfer"]:
        print(
            f"  {row['transition']:22s} "
            f"excess growth={row['transfer_excess_growth']:+.6f} | "
            f"train Δ={row['train_gap_delta']:+.6f} | "
            f"val Δ={row['val_gap_delta']:+.6f}"
        )

    print()
    print("B) STREAM / FUSION / HEAD")
    print("-" * 126)
    vm = result["validation_model"]
    for name in STREAM_NAMES:
        print(f"  {name:2s} stream accuracy      : {100*vm['stream_accuracy'][name]:.4f}%")
    print(f"  Any-stream oracle       : {100*vm['any_stream_oracle_accuracy']:.4f}%")
    print(f"  Main fusion             : {100*vm['main_fusion_accuracy']:.4f}%")
    print(f"  Final adaptive head     : {100*vm['final_accuracy']:.4f}%")
    print(f"  Oracle - final          : {vm['oracle_minus_final_pp']:+.4f} pp")
    print(
        f"  Adaptive fixed/broken   : "
        f"{vm['adaptive_fixed']}/{vm['adaptive_broken']} "
        f"(net {vm['adaptive_net_pp']:+.4f} pp)"
    )
    print(
        "  Mean fusion weights     : "
        + ", ".join(
            f"{k}={v:.4f}" for k, v in vm["mean_fusion_weights"].items()
        )
    )

    print()
    print("C) HARD CLASSES / PAIRS")
    print("-" * 126)
    print("Top validation confusion pairs:")
    for row in result["hard_classes"]["top_confusion_pairs"][:12]:
        print(
            f"  {row['a']} <-> {row['b']} : {row['total']} "
            f"({row['a_to_b']} / {row['b_to_a']})"
        )
    print()
    print("Weakest descriptor validation gaps:")
    for row in result["hard_classes"]["weakest_descriptor_val_classes"][:12]:
        print(
            f"  {row['class']} gap={row['gap_mean']:+.5f} "
            f"n={row['support']}"
        )
    print()
    print(
        "Top-20 prototype-rival/confusion-pair overlap: "
        f"{result['hard_classes']['top20_pair_overlap_count']}/20 "
        f"= {100*result['hard_classes']['top20_pair_overlap_fraction']:.1f}%"
    )
    print()
    print(
        "Largest transition-transfer excess: "
        f"{result['diagnosis']['largest_transfer_excess_transition']} "
        f"({result['diagnosis']['largest_transfer_excess_growth']:+.6f})"
    )
    print(
        "NOTE: diagnostic localization, not proof of causality. "
        "Use a controlled intervention to confirm the identified transition."
    )
    print("=" * 126)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()

    if args.batch < 1:
        raise ValueError("--batch must be positive")
    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise RuntimeError(
            f"Expected one isolated GPU; backend={jax.default_backend()}, "
            f"devices={jax.local_devices()}"
        )

    dataset = Dataset(args.cache)
    payload = load_checkpoint(args.checkpoint)

    if payload["protocol"] != args.protocol:
        raise ValueError(
            f"Checkpoint protocol={payload['protocol']} but requested {args.protocol}"
        )
    if payload["cache_signature"] != dataset.meta["signature"]:
        raise ValueError("Checkpoint/cache signature mismatch")

    config = payload["config"]
    model = worker.make_model(config)
    init = model.init(
        {"params": jax.random.PRNGKey(128), "dropout": jax.random.PRNGKey(128)},
        jnp.zeros((1, 16, 750), jnp.float32),
        training=False,
    )["params"]
    if count_params(init) != EXPECTED_PARAMS:
        raise RuntimeError("R4 architecture parameter drift")

    params = jax.tree_util.tree_map(jnp.asarray, payload["ema_params"])
    if count_params(params) != EXPECTED_PARAMS:
        raise RuntimeError(
            f"Checkpoint param count {count_params(params)} != {EXPECTED_PARAMS}"
        )

    infer = make_infer(model)

    train_ids = dataset.splits[f"{args.protocol}_train"]
    val_ids = dataset.splits[f"{args.protocol}_val"]

    status = Path(args.status)
    atomic_json(
        status,
        {
            "protocol": args.protocol,
            "phase": "Start",
            "current": 0,
            "total": 1,
        },
    )

    train_sums, train_counts, prototypes, train_model_raw = pass_train_prototypes(
        dataset,
        train_ids,
        args.batch,
        params,
        infer,
        status,
        args.protocol,
    )

    train_geom_raw, _ = geometry_pass(
        dataset,
        train_ids,
        args.batch,
        params,
        infer,
        train_sums,
        train_counts,
        prototypes,
        leave_one_out=True,
        status_path=status,
        protocol=args.protocol,
        phase="TRAIN LOO geometry",
        collect_model=False,
    )

    val_geom_raw, val_model_raw = geometry_pass(
        dataset,
        val_ids,
        args.batch,
        params,
        infer,
        train_sums,
        train_counts,
        prototypes,
        leave_one_out=False,
        status_path=status,
        protocol=args.protocol,
        phase="VAL geometry + streams",
        collect_model=True,
    )

    train_rows, train_class_rows = finalize_geometry(train_geom_raw)
    val_rows, val_class_rows = finalize_geometry(val_geom_raw)
    train_model = finalize_model_stats(train_model_raw)
    val_model = finalize_model_stats(val_model_raw)

    recorded = float(payload["val_accuracy"])
    verified = float(val_model["final_accuracy"])
    if abs(recorded - verified) > 5e-4:
        raise RuntimeError(
            f"Checkpoint score mismatch: recorded={recorded:.8f}, "
            f"verified={verified:.8f}"
        )

    conf_pairs = top_confusion_pairs(val_model_raw["confusion"], 20)
    proto_pairs, proto_directed = prototype_rival_pairs(prototypes, -1, 20)
    overlap = pair_set(conf_pairs) & pair_set(proto_pairs)
    stage_excess, transitions = transition_transfer(train_rows, val_rows)

    weakest = weakest_classes(val_class_rows["Descriptor"], 20)
    transfer_classes = class_transfer_loss(
        train_class_rows["Descriptor"],
        val_class_rows["Descriptor"],
        20,
    )

    result = {
        "model": EXPECTED_MODEL,
        "protocol": args.protocol,
        "device": str(jax.local_devices()[0]),
        "checkpoint": {
            "path": str(Path(args.checkpoint)),
            "epoch": int(payload["epoch"]),
            "recorded_val_accuracy": recorded,
            "params": EXPECTED_PARAMS,
        },
        "samples": {
            "train": int(len(train_ids)),
            "val": int(len(val_ids)),
        },
        "train_model": train_model,
        "validation_model": val_model,
        "geometry": {
            "train": train_rows,
            "val": val_rows,
            "stage_transfer_excess": stage_excess,
            "transition_transfer": transitions,
        },
        "hard_classes": {
            "top_confusion_pairs": conf_pairs,
            "top_descriptor_prototype_rival_pairs": proto_pairs,
            "descriptor_nearest_rival_by_class": proto_directed,
            "top20_pair_overlap": sorted([list(x) for x in overlap]),
            "top20_pair_overlap_count": int(len(overlap)),
            "top20_pair_overlap_fraction": float(len(overlap) / 20.0),
            "weakest_descriptor_val_classes": weakest,
            "largest_descriptor_train_to_val_gap_loss": transfer_classes,
        },
        "diagnosis": {
            "largest_transfer_excess_transition": transitions[0]["transition"],
            "largest_transfer_excess_growth": float(
                transitions[0]["transfer_excess_growth"]
            ),
            "oracle_minus_final_pp": float(val_model["oracle_minus_final_pp"]),
            "adaptive_head_net_pp": float(val_model["adaptive_net_pp"]),
            "note": (
                "Localization is diagnostic, not causal. Confirm with a controlled "
                "training-time or counterfactual intervention."
            ),
        },
    }

    atomic_json(args.output, result)
    atomic_json(
        status,
        {
            "protocol": args.protocol,
            "phase": "Done",
            "current": 1,
            "total": 1,
            "done": True,
            "verified_val_accuracy": verified,
        },
    )
    print_report(result)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""Read-only descriptor micro-localization audit for the ~30-MFLOP R4 model.

This audit decomposes the exact R4 descriptor path:

    Router frame states
      -> FrameMean
      -> G4 chunk memory -> G4ChunkMean
      -> Concat(FrameMean, G4ChunkMean)
      -> hier_fuse Dense
      -> GELU
      -> hier_norm LayerNorm = Descriptor

For every sub-stage it measures TRAIN leave-one-out prototype geometry versus
VAL geometry using TRAIN prototypes only.  It also reports the same transfer
excess separately for J/B/JM/BM streams.

No training. No checkpoint mutation. Validation labels are diagnostic only.
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


EXPECTED_MODEL = "NestSAR-SM-ALL-T16-R4-v1"
EXPECTED_PARAMS = 1_831_932
NUM_CLASSES = 120
STREAM_NAMES = ("J", "B", "JM", "BM")
STAGE_NAMES = (
    "FrameMean",
    "G4ChunkMean",
    "Concat",
    "HierLinear",
    "GELU",
    "DescriptorNorm",
)


def count_params(tree):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def normalize_np(x, eps=1e-12):
    x = np.asarray(x, np.float64)
    den = np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    return x / den


def load_checkpoint(path):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    required = {
        "model", "protocol", "epoch", "val_accuracy", "ema_params", "config",
        "cache_signature",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint missing {sorted(missing)}")
    if payload["model"] != EXPECTED_MODEL:
        raise ValueError(
            f"Expected {EXPECTED_MODEL}, checkpoint says {payload['model']}"
        )
    return payload


def iter_batches(dataset, ids, batch):
    ids = np.asarray(ids, np.int64)
    for start in range(0, len(ids), batch):
        take = ids[start:start + batch]
        n = len(take)
        x = np.zeros((batch, 16, 750), np.float32)
        y = np.zeros((batch,), np.int32)
        x[:n] = dataset.canonical[take]
        y[:n] = dataset.labels[take]
        yield x, y, n


def capture_descriptor_modules(module, method_name):
    return (
        method_name == "__call__"
        and module.name in ("hier_fuse", "hier_norm")
    )


def _captured_call(tree, descriptor_index, module_name):
    node = tree[f"descriptor_{descriptor_index}"][module_name]["__call__"]
    # Flax stores captured calls as a tuple/list, normally length 1.
    if isinstance(node, (tuple, list)):
        if len(node) != 1:
            raise RuntimeError(
                f"Unexpected captures for descriptor_{descriptor_index}/{module_name}: "
                f"{len(node)}"
            )
        return node[0]
    return node


def make_infer(model):
    @jax.jit
    def infer(params, x):
        out, mutable = model.apply(
            {"params": params},
            x,
            training=False,
            capture_intermediates=capture_descriptor_modules,
            mutable=["intermediates"],
        )
        inter = mutable["intermediates"]

        fusion = out["fusion_weights"]                         # [B,S]
        frame_mean = jnp.mean(out["mixed_frame_stack"], axis=1)  # [B,S,D]
        chunk_mean = jnp.mean(out["chunk_states"], axis=2)       # [B,S,D]
        concat = jnp.concatenate([frame_mean, chunk_mean], axis=-1)

        linear = jnp.stack(
            [
                _captured_call(inter, s, "hier_fuse")
                for s in range(4)
            ],
            axis=1,
        )
        norm_capture = jnp.stack(
            [
                _captured_call(inter, s, "hier_norm")
                for s in range(4)
            ],
            axis=1,
        )

        gelu = jax.nn.gelu(linear)
        descriptor = out["descriptors"]

        # In inference Dropout is identity, so hier_norm must equal descriptor.
        capture_diff = jnp.max(jnp.abs(norm_capture - descriptor))

        stream_stages = (
            frame_mean,
            chunk_mean,
            concat,
            linear,
            gelu,
            descriptor,
        )

        fused_stages = tuple(
            jnp.einsum("bs,bsd->bd", fusion, stage)
            for stage in stream_stages
        )

        return (
            out["logits"],
            tuple(stream_stages),
            tuple(fused_stages),
            capture_diff,
        )

    return infer


def init_sums(stage_dims):
    fused = [
        np.zeros((NUM_CLASSES, d), np.float64)
        for d in stage_dims
    ]
    stream = [
        np.zeros((4, NUM_CLASSES, d), np.float64)
        for d in stage_dims
    ]
    return fused, stream


def add_at_classes(dest, labels, values):
    np.add.at(dest, labels, values)


def prototype_pass(dataset, ids, batch, params, infer, status, protocol):
    fused_sums = None
    stream_sums = None
    counts = np.zeros((NUM_CLASSES,), np.int64)
    max_capture_diff = 0.0
    correct = 0
    seen = 0
    total_batches = math.ceil(len(ids) / batch)

    for bi, (x, y, n) in enumerate(iter_batches(dataset, ids, batch), 1):
        logits, stream_stages, fused_stages, capture_diff = jax.block_until_ready(
            infer(params, jax.device_put(x))
        )
        yy = y[:n]
        logits = np.asarray(logits)[:n]
        max_capture_diff = max(max_capture_diff, float(np.asarray(capture_diff)))

        ss = [normalize_np(np.asarray(v)[:n]) for v in stream_stages]
        fs = [normalize_np(np.asarray(v)[:n]) for v in fused_stages]

        if fused_sums is None:
            dims = [v.shape[-1] for v in fs]
            fused_sums, stream_sums = init_sums(dims)

        for k in range(len(STAGE_NAMES)):
            add_at_classes(fused_sums[k], yy, fs[k])
            for s in range(4):
                add_at_classes(stream_sums[k][s], yy, ss[k][:, s])

        counts += np.bincount(yy, minlength=NUM_CLASSES)
        correct += int((logits.argmax(-1) == yy).sum())
        seen += n

        if bi == 1 or bi % 10 == 0 or bi == total_batches:
            atomic_json(
                status,
                {
                    "protocol": protocol,
                    "phase": "TRAIN prototypes",
                    "current": bi,
                    "total": total_batches,
                },
            )

    if np.any(counts < 2):
        raise RuntimeError("Need >=2 TRAIN samples per class")

    fused_proto = [normalize_np(x) for x in fused_sums]
    stream_proto = [normalize_np(x) for x in stream_sums]

    return {
        "fused_sums": fused_sums,
        "stream_sums": stream_sums,
        "counts": counts,
        "fused_proto": fused_proto,
        "stream_proto": stream_proto,
        "train_accuracy": float(correct / max(seen, 1)),
        "capture_max_abs_diff": max_capture_diff,
    }


def empty_gap_acc():
    return {
        "fused": [[] for _ in STAGE_NAMES],
        "stream": [[[] for _ in STREAM_NAMES] for _ in STAGE_NAMES],
        "class_gap_sum": np.zeros(
            (len(STAGE_NAMES), NUM_CLASSES), np.float64
        ),
        "class_count": np.zeros((NUM_CLASSES,), np.int64),
    }


def gaps_against_prototypes(z, labels, proto, true_proto_override=None):
    sims = z @ proto.T
    rows = np.arange(len(labels))

    if true_proto_override is None:
        true = sims[rows, labels]
    else:
        true = np.sum(z * true_proto_override, axis=-1)

    sims[rows, labels] = -np.inf
    wrong = np.max(sims, axis=-1)
    return true - wrong


def geometry_pass(
    dataset,
    ids,
    batch,
    params,
    infer,
    proto_state,
    *,
    leave_one_out,
    status,
    protocol,
    phase,
):
    acc = empty_gap_acc()
    total_batches = math.ceil(len(ids) / batch)
    correct = 0
    seen = 0
    max_capture_diff = 0.0

    for bi, (x, y, n) in enumerate(iter_batches(dataset, ids, batch), 1):
        logits, stream_stages, fused_stages, capture_diff = jax.block_until_ready(
            infer(params, jax.device_put(x))
        )
        yy = y[:n]
        logits = np.asarray(logits)[:n]
        max_capture_diff = max(max_capture_diff, float(np.asarray(capture_diff)))
        ss = [normalize_np(np.asarray(v)[:n]) for v in stream_stages]
        fs = [normalize_np(np.asarray(v)[:n]) for v in fused_stages]

        acc["class_count"] += np.bincount(yy, minlength=NUM_CLASSES)

        for k in range(len(STAGE_NAMES)):
            if leave_one_out:
                fused_true = normalize_np(
                    proto_state["fused_sums"][k][yy] - fs[k]
                )
            else:
                fused_true = None

            fg = gaps_against_prototypes(
                fs[k],
                yy,
                proto_state["fused_proto"][k],
                fused_true,
            )
            acc["fused"][k].append(fg)
            np.add.at(acc["class_gap_sum"][k], yy, fg)

            for s in range(4):
                if leave_one_out:
                    stream_true = normalize_np(
                        proto_state["stream_sums"][k][s, yy] - ss[k][:, s]
                    )
                else:
                    stream_true = None

                sg = gaps_against_prototypes(
                    ss[k][:, s],
                    yy,
                    proto_state["stream_proto"][k][s],
                    stream_true,
                )
                acc["stream"][k][s].append(sg)

        correct += int((logits.argmax(-1) == yy).sum())
        seen += n

        if bi == 1 or bi % 10 == 0 or bi == total_batches:
            atomic_json(
                status,
                {
                    "protocol": protocol,
                    "phase": phase,
                    "current": bi,
                    "total": total_batches,
                },
            )

    return acc, {
        "accuracy": float(correct / max(seen, 1)),
        "capture_max_abs_diff": max_capture_diff,
    }


def summarize_gap(x):
    x = np.concatenate(x)
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "wrong_proto_closer_fraction": float(np.mean(x < 0)),
        "nearest_prototype_accuracy": float(np.mean(x > 0)),
    }


def summarize_geometry(acc):
    fused = {}
    streams = {}
    for k, stage in enumerate(STAGE_NAMES):
        fused[stage] = summarize_gap(acc["fused"][k])
        streams[stage] = {
            STREAM_NAMES[s]: summarize_gap(acc["stream"][k][s])
            for s in range(4)
        }

    class_means = {}
    denom = np.maximum(acc["class_count"].astype(np.float64), 1.0)
    for k, stage in enumerate(STAGE_NAMES):
        means = acc["class_gap_sum"][k] / denom
        class_means[stage] = [
            {
                "class": f"A{c+1:03d}",
                "support": int(acc["class_count"][c]),
                "gap_mean": float(means[c]),
            }
            for c in range(NUM_CLASSES)
        ]

    return fused, streams, class_means


def transition_table(train, val):
    stage_excess = {
        s: float(train[s]["mean"] - val[s]["mean"])
        for s in STAGE_NAMES
    }
    rows = []
    for a, b in zip(STAGE_NAMES[:-1], STAGE_NAMES[1:]):
        rows.append(
            {
                "transition": f"{a}->{b}",
                "train_delta": float(train[b]["mean"] - train[a]["mean"]),
                "val_delta": float(val[b]["mean"] - val[a]["mean"]),
                "transfer_excess_growth": float(
                    stage_excess[b] - stage_excess[a]
                ),
            }
        )
    rows.sort(key=lambda r: -r["transfer_excess_growth"])
    return stage_excess, rows


def stream_transition_tables(train_stream, val_stream):
    out = {}
    for stream in STREAM_NAMES:
        tr = {s: train_stream[s][stream] for s in STAGE_NAMES}
        va = {s: val_stream[s][stream] for s in STAGE_NAMES}
        excess, transitions = transition_table(tr, va)
        out[stream] = {
            "stage_transfer_excess": excess,
            "transitions": transitions,
        }
    return out


def weakest_descriptor_classes(class_rows, limit=15):
    rows = list(class_rows["DescriptorNorm"])
    rows.sort(key=lambda r: (r["gap_mean"], r["class"]))
    return rows[:limit]


def print_report(result):
    print("=" * 128)
    print(
        f"R4 DESCRIPTOR MICRO-LOCALIZATION | {result['protocol'].upper()} | "
        f"E{result['checkpoint']['epoch']}"
    )
    print("=" * 128)
    print(
        f"Recorded/verified val: "
        f"{100*result['checkpoint']['recorded_val_accuracy']:.4f}% / "
        f"{100*result['validation_accuracy']:.4f}%"
    )
    print(
        f"hier_norm capture == returned descriptor max |diff|: "
        f"{result['descriptor_capture_max_abs_diff']:.3e}"
    )
    print()
    print(
        f"{'Stage':18s} {'Train gap':>11s} {'Val gap':>11s} "
        f"{'Excess':>11s} {'Train NP':>10s} {'Val NP':>10s}"
    )
    print("-" * 80)
    for stage in STAGE_NAMES:
        tr = result["fused_geometry"]["train"][stage]
        va = result["fused_geometry"]["val"][stage]
        ex = result["fused_geometry"]["stage_transfer_excess"][stage]
        print(
            f"{stage:18s} {tr['mean']:11.5f} {va['mean']:11.5f} "
            f"{ex:11.5f} {100*tr['nearest_prototype_accuracy']:9.2f}% "
            f"{100*va['nearest_prototype_accuracy']:9.2f}%"
        )

    print()
    print("FUSED transition transfer-excess growth:")
    for row in result["fused_geometry"]["transitions"]:
        print(
            f"  {row['transition']:30s} "
            f"{row['transfer_excess_growth']:+.6f} | "
            f"train Δ={row['train_delta']:+.6f} | "
            f"val Δ={row['val_delta']:+.6f}"
        )

    print()
    print("Per-stream largest descriptor-path transfer-loss transition:")
    for stream in STREAM_NAMES:
        top = result["stream_geometry"][stream]["transitions"][0]
        print(
            f"  {stream:2s}: {top['transition']:30s} "
            f"{top['transfer_excess_growth']:+.6f}"
        )

    print()
    print("Weakest final descriptor classes:")
    for row in result["weakest_descriptor_classes"][:12]:
        print(
            f"  {row['class']} gap={row['gap_mean']:+.5f} n={row['support']}"
        )

    d = result["diagnosis"]
    print()
    print("DIAGNOSIS")
    print(
        f"  Largest fused sub-transition: "
        f"{d['largest_fused_transition']} "
        f"({d['largest_fused_transfer_excess_growth']:+.6f})"
    )
    print(
        "  This is diagnostic localization only. Confirm the selected operation "
        "with a controlled intervention before changing the paper architecture."
    )
    print("=" * 128)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()

    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise RuntimeError(
            f"Expected one isolated GPU; backend={jax.default_backend()} "
            f"devices={jax.local_devices()}"
        )

    dataset = Dataset(args.cache)
    payload = load_checkpoint(args.checkpoint)
    if payload["protocol"] != args.protocol:
        raise ValueError(
            f"Checkpoint protocol {payload['protocol']} != {args.protocol}"
        )
    if payload["cache_signature"] != dataset.meta["signature"]:
        raise ValueError("Checkpoint/cache signature mismatch")

    model = worker.make_model(payload["config"])
    init = model.init(
        {"params": jax.random.PRNGKey(128), "dropout": jax.random.PRNGKey(128)},
        jnp.zeros((1, 16, 750), jnp.float32),
        training=False,
    )["params"]
    if count_params(init) != EXPECTED_PARAMS:
        raise RuntimeError("R4 architecture parameter drift")

    params = jax.tree_util.tree_map(jnp.asarray, payload["ema_params"])
    if count_params(params) != EXPECTED_PARAMS:
        raise RuntimeError("R4 checkpoint parameter drift")

    infer = make_infer(model)
    train_ids = dataset.splits[f"{args.protocol}_train"]
    val_ids = dataset.splits[f"{args.protocol}_val"]
    status = Path(args.status)

    atomic_json(status, {
        "protocol": args.protocol, "phase": "Start", "current": 0, "total": 1
    })

    proto = prototype_pass(
        dataset, train_ids, args.batch, params, infer, status, args.protocol
    )
    if proto["capture_max_abs_diff"] > 2e-5:
        raise RuntimeError(
            f"Captured hier_norm does not match descriptor: "
            f"{proto['capture_max_abs_diff']}"
        )

    train_raw, train_meta = geometry_pass(
        dataset, train_ids, args.batch, params, infer, proto,
        leave_one_out=True,
        status=status, protocol=args.protocol, phase="TRAIN LOO geometry",
    )
    val_raw, val_meta = geometry_pass(
        dataset, val_ids, args.batch, params, infer, proto,
        leave_one_out=False,
        status=status, protocol=args.protocol, phase="VAL descriptor geometry",
    )

    train_fused, train_stream, train_classes = summarize_geometry(train_raw)
    val_fused, val_stream, val_classes = summarize_geometry(val_raw)

    recorded = float(payload["val_accuracy"])
    verified = float(val_meta["accuracy"])
    if abs(recorded - verified) > 5e-4:
        raise RuntimeError(
            f"Score mismatch recorded={recorded:.8f}, verified={verified:.8f}"
        )

    stage_excess, transitions = transition_table(train_fused, val_fused)
    per_stream = stream_transition_tables(train_stream, val_stream)
    max_capture = max(
        proto["capture_max_abs_diff"],
        train_meta["capture_max_abs_diff"],
        val_meta["capture_max_abs_diff"],
    )

    result = {
        "model": EXPECTED_MODEL,
        "protocol": args.protocol,
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
        "validation_accuracy": verified,
        "train_accuracy": float(proto["train_accuracy"]),
        "descriptor_capture_max_abs_diff": float(max_capture),
        "fused_geometry": {
            "train": train_fused,
            "val": val_fused,
            "stage_transfer_excess": stage_excess,
            "transitions": transitions,
        },
        "stream_stage_geometry": {
            "train": train_stream,
            "val": val_stream,
        },
        "stream_geometry": per_stream,
        "weakest_descriptor_classes": weakest_descriptor_classes(val_classes),
        "diagnosis": {
            "largest_fused_transition": transitions[0]["transition"],
            "largest_fused_transfer_excess_growth": float(
                transitions[0]["transfer_excess_growth"]
            ),
            "note": (
                "Diagnostic localization only; causal intervention is required "
                "before modifying the architecture."
            ),
        },
    }

    atomic_json(args.output, result)
    atomic_json(status, {
        "protocol": args.protocol,
        "phase": "Done",
        "current": 1,
        "total": 1,
        "done": True,
        "verified_val_accuracy": verified,
    })
    print_report(result)


if __name__ == "__main__":
    main()

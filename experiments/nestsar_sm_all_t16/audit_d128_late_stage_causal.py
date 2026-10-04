#!/usr/bin/env python3
from __future__ import annotations

"""Causal late-stage audit for trained NestSAR D128-MTS.

Tests the two late-stage mechanisms implicated by the class-geometry audit:

1) G4 transformation strength
   pre-G4 chunk pooling -> trained G4 chunk representation.

2) Descriptor transformation strength
   simple normalized pooled descriptor -> trained hierarchical descriptor.

The trained downstream classifier/fusion/adaptive head is reused exactly.
All interventions are inference-only and parameter-free.

The audit performs ONE canonical model forward per batch, then evaluates all
late-stage counterfactuals from the returned intermediate representations.
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
from flax.traverse_util import flatten_dict

from experiments.nestsar_sm_all_t16.model_parallel import NestSARParallelT16
from experiments.nestsar_sm_all_t16.parallel_d128_config import (
    MODEL_NAME,
    EXPECTED_PARAMS,
    M4_HALF_LIVES,
    G4_HALF_LIVES,
)
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json


NUM_CLASSES = 120
NUM_STREAMS = 4


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
        raise ValueError(f"Parameter mismatch {n:,} != {EXPECTED_PARAMS:,}")
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


def find_leaf(flat, suffix):
    hits = []
    for path, value in flat.items():
        text = "/".join(map(str, path))
        if text.endswith(suffix):
            hits.append((text, np.asarray(value)))
    if len(hits) != 1:
        raise ValueError(
            f"Expected exactly one parameter ending {suffix!r}; "
            f"found {[x[0] for x in hits]}"
        )
    return hits[0][1]


def downstream_params(params):
    flat = flatten_dict(params)
    p = {
        "g4_norm_scale": find_leaf(
            flat, "descriptor_group/chunk_memory/sm_norm/scale"
        ),
        "g4_norm_bias": find_leaf(
            flat, "descriptor_group/chunk_memory/sm_norm/bias"
        ),
        "hier_kernel": find_leaf(
            flat, "descriptor_group/hier_fuse/kernel"
        ),
        "hier_bias": find_leaf(
            flat, "descriptor_group/hier_fuse/bias"
        ),
        "hier_norm_scale": find_leaf(
            flat, "descriptor_group/hier_norm/scale"
        ),
        "hier_norm_bias": find_leaf(
            flat, "descriptor_group/hier_norm/bias"
        ),
        "classifier_kernel": find_leaf(
            flat, "classifier_group/kernel"
        ),
        "classifier_bias": find_leaf(
            flat, "classifier_group/bias"
        ),
        "head_u": find_leaf(
            flat, "adaptive_head_u/kernel"
        ),
        "head_v": find_leaf(
            flat, "adaptive_head_v/kernel"
        ),
    }

    expected_rank = {
        "g4_norm_scale": 2,
        "g4_norm_bias": 2,
        "hier_kernel": 3,
        "hier_bias": 2,
        "hier_norm_scale": 2,
        "hier_norm_bias": 2,
        "classifier_kernel": 3,
        "classifier_bias": 2,
        "head_u": 2,
        "head_v": 2,
    }
    for key, ndim in expected_rank.items():
        if p[key].ndim != ndim:
            raise ValueError(f"Unexpected {key} shape {p[key].shape}")
    return {k: jnp.asarray(v) for k, v in p.items()}


def layer_norm_affine(x, scale, bias, eps=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    y = (x - mean) * jax.lax.rsqrt(var + eps)
    return y * scale + bias


def descriptor_from_chunks(frame_mean, chunks, p):
    chunk_mean = jnp.mean(chunks, axis=2)
    pooled = jnp.concatenate([frame_mean, chunk_mean], axis=-1)
    h = jnp.einsum("bsi,sid->bsd", pooled, p["hier_kernel"])
    h = h + p["hier_bias"][None, :, :]
    h = jax.nn.gelu(h)
    return layer_norm_affine(
        h,
        p["hier_norm_scale"][None, :, :],
        p["hier_norm_bias"][None, :, :],
    )


def simple_descriptor(frame_mean, chunks, p):
    # Parameter-free pooled baseline mapped into the same per-stream affine
    # LayerNorm manifold as the learned descriptor.
    base = 0.5 * (frame_mean + jnp.mean(chunks, axis=2))
    return layer_norm_affine(
        base,
        p["hier_norm_scale"][None, :, :],
        p["hier_norm_bias"][None, :, :],
    )


def logits_from_desc(desc, fusion, head_coeff, p, head_residual_scale):
    stream_logits = (
        jnp.einsum("bsd,sdc->bsc", desc, p["classifier_kernel"])
        + p["classifier_bias"][None, :, :]
    )
    main = jnp.einsum("bs,bsc->bc", fusion, stream_logits)

    fused_desc = jnp.einsum("bs,bsd->bd", fusion, desc)
    head_u = fused_desc @ p["head_u"]
    dynamic = head_u * head_coeff
    delta = dynamic @ p["head_v"]
    logits = main + float(head_residual_scale) * delta
    return logits


def variants():
    rows = [("canonical_exact", None, None)]
    rows += [("manual_identity", 1.0, 1.0)]

    for v in (0.00, 0.50, 0.80, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10, 1.20):
        rows.append((f"g4_{v:.2f}".replace(".", "p"), v, 1.0))

    for v in (0.00, 0.50, 0.80, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10, 1.20):
        rows.append((f"desc_{v:.2f}".replace(".", "p"), 1.0, v))

    rows += [
        ("g4_0p98_desc_0p98", 0.98, 0.98),
        ("g4_0p95_desc_0p98", 0.95, 0.98),
        ("g4_0p98_desc_0p95", 0.98, 0.95),
        ("g4_0p95_desc_0p95", 0.95, 0.95),
    ]
    return rows


def downstream_variant(
    *,
    out,
    p,
    g4_scale,
    desc_scale,
    head_residual_scale,
):
    mixed = jnp.asarray(out["mixed_frame_stack"])      # [B,16,S,D]
    canonical_chunks = jnp.asarray(out["chunk_states"])  # [B,S,4,D]
    canonical_desc = jnp.asarray(out["descriptors"])     # [B,S,D]
    fusion = jnp.asarray(out["fusion_weights"])
    head_coeff = jnp.asarray(out["sm_head_coeff"])

    b, t, s, d = mixed.shape
    if s != NUM_STREAMS:
        raise ValueError(f"Unexpected stream count {s}")

    pre_chunks = mixed.reshape(b, 4, t // 4, s, d).mean(axis=2)
    pre_chunks = jnp.transpose(pre_chunks, (0, 2, 1, 3))

    g4_base = layer_norm_affine(
        pre_chunks,
        p["g4_norm_scale"][None, :, None, :],
        p["g4_norm_bias"][None, :, None, :],
    )

    chunks = g4_base + float(g4_scale) * (
        canonical_chunks - g4_base
    )

    frame_mean = jnp.mean(mixed, axis=1)
    learned_desc = descriptor_from_chunks(frame_mean, chunks, p)
    base_desc = simple_descriptor(frame_mean, chunks, p)

    desc = base_desc + float(desc_scale) * (
        learned_desc - base_desc
    )

    logits = logits_from_desc(
        desc,
        fusion,
        head_coeff,
        p,
        head_residual_scale,
    )

    return logits, learned_desc, canonical_desc


def motion_energy(x):
    tok = np.asarray(x, np.float32).reshape(
        len(x), 16, 2, 25, 15
    )
    return np.mean(
        np.abs(tok[..., 3:15]),
        axis=(1, 2, 3, 4),
    ).astype(np.float32)


def safe_load_chunk(path, n_variants, expected_n):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            pred = np.asarray(z["pred"], np.int16)
            labels = np.asarray(z["labels"], np.int16)
            motion = np.asarray(z["motion"], np.float32)
            identity_max_abs = float(z["identity_max_abs"])
            identity_agreement = float(z["identity_agreement"])

        if pred.shape != (n_variants, expected_n):
            raise ValueError(pred.shape)
        if labels.shape != (expected_n,) or motion.shape != (expected_n,):
            raise ValueError((labels.shape, motion.shape))

        return {
            "pred": pred,
            "labels": labels,
            "motion": motion,
            "identity_max_abs": identity_max_abs,
            "identity_agreement": identity_agreement,
        }
    except Exception:
        path.unlink(missing_ok=True)
        return None


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
    labels_all = np.asarray(dataset.labels[ids], np.int32)

    model = make_model(config)
    forward = jax.jit(
        lambda p, x: model.apply({"params": p}, x, training=False)
    )

    p = downstream_params(params)
    var = variants()
    names = [x[0] for x in var]

    # Compile one downstream function per intervention. These are tiny graphs
    # compared with the full model and operate only on one batch of cached
    # intermediate tensors.
    downstream_fns = {}
    for name, g4_scale, desc_scale in var:
        if name == "canonical_exact":
            continue
        downstream_fns[name] = jax.jit(
            lambda out, gs=g4_scale, ds=desc_scale: downstream_variant(
                out=out,
                p=p,
                g4_scale=gs,
                desc_scale=ds,
                head_residual_scale=config["head_residual_scale"],
            )[0]
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(str(out_path) + ".resume")
    work.mkdir(parents=True, exist_ok=True)
    status = Path(args.status) if args.status else None

    identity = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_accuracy": float(payload["val_accuracy"]),
        "params": count_params(params),
        "val_samples": int(len(ids)),
        "batch_size": int(args.batch_size),
        "variants": [
            {
                "name": n,
                "g4_scale": g,
                "descriptor_scale": d,
            }
            for n, g, d in var
        ],
    }
    identity_path = work / "identity.json"
    if identity_path.is_file():
        old = json.loads(identity_path.read_text())
        if old != identity:
            raise RuntimeError(
                f"Resume identity mismatch at {work}; "
                "use a fresh --output path."
            )
    else:
        atomic_json(identity_path, identity)

    print("=" * 122)
    print(f"{MODEL_NAME} | {args.protocol.upper()} LATE-STAGE CAUSAL AUDIT")
    print("=" * 122)
    print("GPU:", jax.local_devices()[0])
    print(f"Checkpoint E{int(payload['epoch']):02d}")
    print(f"Recorded val: {100*float(payload['val_accuracy']):.4f}%")
    print(f"Validation: {len(ids):,}")
    print(f"Variants: {len(var)}")
    print()

    n_batches = math.ceil(len(ids) / args.batch_size)
    chunks = []

    for bi, start in enumerate(range(0, len(ids), args.batch_size)):
        idx = ids[start:start + args.batch_size]
        n = len(idx)
        chunk_path = work / f"batch_{bi:04d}.npz"

        cached = safe_load_chunk(
            chunk_path,
            len(var),
            n,
        )

        if cached is not None:
            chunks.append(cached)
            if status is not None:
                atomic_json(
                    status,
                    {
                        "protocol": args.protocol,
                        "phase": "Late-stage causal batches (resume)",
                        "current": bi + 1,
                        "total": n_batches,
                        "done": False,
                    },
                )
            continue

        x = np.asarray(dataset.canonical[idx], np.float32)
        y = np.asarray(dataset.labels[idx], np.int32)

        out = jax.device_get(
            forward(params, jax.device_put(x))
        )

        canonical_logits = np.asarray(out["logits"], np.float32)
        canonical_pred = canonical_logits.argmax(1).astype(np.int16)

        batch_pred = np.empty((len(var), n), np.int16)
        batch_pred[0] = canonical_pred

        identity_max_abs = 0.0
        identity_agreement = 1.0

        for vi, (name, g4_scale, desc_scale) in enumerate(var[1:], 1):
            logits = np.asarray(
                jax.device_get(downstream_fns[name](out)),
                np.float32,
            )
            pred = logits.argmax(1).astype(np.int16)
            batch_pred[vi] = pred

            if name == "manual_identity":
                identity_max_abs = float(
                    np.max(np.abs(logits - canonical_logits))
                )
                identity_agreement = float(
                    np.mean(pred == canonical_pred)
                )

        tmp = chunk_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            pred=batch_pred,
            labels=y.astype(np.int16),
            motion=motion_energy(x),
            identity_max_abs=np.asarray(identity_max_abs, np.float64),
            identity_agreement=np.asarray(identity_agreement, np.float64),
        )
        os.replace(tmp, chunk_path)

        chunks.append({
            "pred": batch_pred,
            "labels": y.astype(np.int16),
            "motion": motion_energy(x),
            "identity_max_abs": identity_max_abs,
            "identity_agreement": identity_agreement,
        })

        if status is not None:
            atomic_json(
                status,
                {
                    "protocol": args.protocol,
                    "phase": "Late-stage causal batches",
                    "current": bi + 1,
                    "total": n_batches,
                    "done": False,
                    "identity_max_abs": identity_max_abs,
                    "identity_agreement": identity_agreement,
                },
            )

        print(
            f"[{bi+1:03d}/{n_batches:03d}] "
            f"identity maxAbs={identity_max_abs:.3e} "
            f"agree={100*identity_agreement:.5f}%"
        )

    pred = np.concatenate([c["pred"] for c in chunks], axis=1).astype(np.int32)
    labels = np.concatenate([c["labels"] for c in chunks]).astype(np.int32)
    motion = np.concatenate([c["motion"] for c in chunks]).astype(np.float32)

    if not np.array_equal(labels, labels_all):
        raise RuntimeError("Aggregated labels do not match validation split.")

    identity_max_abs = max(c["identity_max_abs"] for c in chunks)
    identity_agreement = min(c["identity_agreement"] for c in chunks)

    if identity_agreement < 1.0 or identity_max_abs > 2e-4:
        raise RuntimeError(
            "Manual downstream reconstruction failed exactness check: "
            f"maxAbs={identity_max_abs:.6e}, "
            f"agreement={100*identity_agreement:.6f}%"
        )

    canonical = pred[0]
    canonical_acc = float(np.mean(canonical == labels))
    recorded = float(payload["val_accuracy"])

    if abs(canonical_acc - recorded) > 5e-5:
        raise RuntimeError(
            f"Canonical mismatch {canonical_acc:.8f} vs {recorded:.8f}"
        )

    correct_base = canonical == labels

    class_count = np.bincount(labels, minlength=NUM_CLASSES)
    class_good = np.bincount(
        labels[correct_base], minlength=NUM_CLASSES
    )
    class_acc = class_good / np.maximum(class_count, 1)
    weak_classes = np.argsort(class_acc)[:20]

    q25, q75 = np.quantile(motion, (0.25, 0.75))
    low = motion <= q25
    high = motion >= q75
    weak_mask = np.isin(labels, weak_classes)

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    np.add.at(confusion, (labels, canonical), 1)
    np.fill_diagonal(confusion, 0)

    pairs = []
    confusion_classes = set()
    for a in range(NUM_CLASSES):
        for b in range(a + 1, NUM_CLASSES):
            total = int(confusion[a, b] + confusion[b, a])
            if total:
                pairs.append((total, a, b))
    pairs.sort(reverse=True)
    for _, a, b in pairs[:10]:
        confusion_classes.add(a)
        confusion_classes.add(b)
    confusion_mask = np.isin(
        labels,
        np.asarray(sorted(confusion_classes), np.int32),
    )

    results = {}
    ranking = []

    for vi, (name, g4_scale, desc_scale) in enumerate(var):
        pv = pred[vi]
        correct = pv == labels
        acc = float(np.mean(correct))

        per_class_good = np.bincount(
            labels[correct], minlength=NUM_CLASSES
        )
        per_class_acc = (
            per_class_good / np.maximum(class_count, 1)
        )

        row = {
            "accuracy": acc,
            "delta_pp": 100.0 * (acc - canonical_acc),
            "g4_scale": g4_scale,
            "descriptor_scale": desc_scale,
            "prediction_agreement": float(np.mean(pv == canonical)),
            "fixed_vs_canonical": int(
                np.sum((~correct_base) & correct)
            ),
            "broken_vs_canonical": int(
                np.sum(correct_base & (~correct))
            ),
            "low_motion_delta_pp": 100.0 * (
                float(np.mean(correct[low]))
                - float(np.mean(correct_base[low]))
            ),
            "high_motion_delta_pp": 100.0 * (
                float(np.mean(correct[high]))
                - float(np.mean(correct_base[high]))
            ),
            "weak20_delta_pp": 100.0 * (
                float(np.mean(correct[weak_mask]))
                - float(np.mean(correct_base[weak_mask]))
            ),
            "top_confusion_classes_delta_pp": 100.0 * (
                float(np.mean(correct[confusion_mask]))
                - float(np.mean(correct_base[confusion_mask]))
            ),
        }

        delta_class = 100.0 * (per_class_acc - class_acc)
        order_gain = np.argsort(-delta_class)
        order_loss = np.argsort(delta_class)
        row["best_class_deltas"] = [
            {
                "ntu_action": int(c + 1),
                "n": int(class_count[c]),
                "canonical_accuracy": float(class_acc[c]),
                "variant_accuracy": float(per_class_acc[c]),
                "delta_pp": float(delta_class[c]),
            }
            for c in order_gain[:10]
        ]
        row["worst_class_deltas"] = [
            {
                "ntu_action": int(c + 1),
                "n": int(class_count[c]),
                "canonical_accuracy": float(class_acc[c]),
                "variant_accuracy": float(per_class_acc[c]),
                "delta_pp": float(delta_class[c]),
            }
            for c in order_loss[:10]
        ]

        results[name] = row
        if name not in ("canonical_exact", "manual_identity"):
            ranking.append((row["delta_pp"], name))

    ranking.sort(reverse=True)

    result = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_recorded_accuracy": recorded,
        "verified_canonical_accuracy": canonical_acc,
        "params": count_params(params),
        "val_samples": int(len(labels)),
        "manual_identity_max_abs_logit_error": identity_max_abs,
        "manual_identity_prediction_agreement": identity_agreement,
        "motion_q25": float(q25),
        "motion_q75": float(q75),
        "weak20_classes_1based": [
            int(x + 1) for x in weak_classes
        ],
        "top10_confusion_pairs": [
            {
                "action_a": int(a + 1),
                "action_b": int(b + 1),
                "confusions_total": int(total),
            }
            for total, a, b in pairs[:10]
        ],
        "variants": results,
        "ranking": [
            {"name": name, "delta_pp": float(delta)}
            for delta, name in ranking
        ],
        "decision_rules": {
            "g4_overstrength_supported": (
                "Supported if a modest G4 weakening (especially 0.95-0.98) "
                "improves BOTH XSUB and XSET while G4 strengthening is worse."
            ),
            "descriptor_overstrength_supported": (
                "Supported if a modest descriptor weakening (especially "
                "0.95-0.98) improves BOTH protocols while descriptor "
                "strengthening is worse."
            ),
            "late_stage_combination_supported": (
                "Supported if a combined small weakening improves both "
                "protocols more consistently than either isolated change."
            ),
            "negative_result": (
                "If lambda≈1 remains optimal in both protocols, late-stage "
                "generalization remains diagnostically localized there, "
                "but simple transformation strength is not the causal fix."
            ),
        },
    }

    atomic_json(out_path, result)

    if status is not None:
        atomic_json(
            status,
            {
                "protocol": args.protocol,
                "phase": "Done",
                "current": 1,
                "total": 1,
                "done": True,
                "canonical_accuracy": canonical_acc,
                "best_variant": ranking[0][1],
                "best_delta_pp": float(ranking[0][0]),
            },
        )

    print()
    print("=" * 122)
    print("LATE-STAGE CAUSAL RANKING")
    print("=" * 122)
    print(
        f"canonical={100*canonical_acc:.4f}% | "
        f"manual maxAbs={identity_max_abs:.3e} | "
        f"manual agree={100*identity_agreement:.5f}%"
    )
    print(
        f"{'variant':25s} {'acc':>9s} {'delta':>9s} "
        f"{'low':>8s} {'high':>8s} {'weak20':>8s} "
        f"{'confCls':>8s} {'agree':>8s}"
    )
    print("-" * 96)

    for delta, name in ranking:
        r = results[name]
        print(
            f"{name:25s} "
            f"{100*r['accuracy']:8.4f}% "
            f"{r['delta_pp']:+8.4f} "
            f"{r['low_motion_delta_pp']:+7.3f} "
            f"{r['high_motion_delta_pp']:+7.3f} "
            f"{r['weak20_delta_pp']:+7.3f} "
            f"{r['top_confusion_classes_delta_pp']:+7.3f} "
            f"{100*r['prediction_agreement']:7.2f}%"
        )

    print()
    print("BEST:", ranking[0][1], f"{ranking[0][0]:+.4f} pp")
    print("Saved:", out_path)
    print("=" * 122)


if __name__ == "__main__":
    main()

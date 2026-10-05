#!/usr/bin/env python3
from __future__ import annotations

"""Audit whether the warm-start descriptor margin loss actually has signal.

Mirrors the implemented cross-view hard-negative descriptor margin:

    relu(m + hardest_negative - positive)

and compares two candidate pools using the SAME descriptor embeddings:
  * micro64  : exact training-time pool used inside gradient accumulation
  * batch256 : control pool showing what is missed by microbatching

The model is evaluated in training mode with dropout, and xa uses the same
training augmentation pipeline. No gradients, no parameter updates.
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

from experiments.nestsar_sm_all_t16.model_parallel import NestSARParallelT16
from experiments.nestsar_sm_all_t16.parallel_d128_config import (
    MODEL_NAME,
    EXPECTED_PARAMS,
    MODEL_DIM,
    M4_HALF_LIVES,
    G4_HALF_LIVES,
)
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json


MARGINS = (0.05, 0.10, 0.15, 0.20, 0.30)
MICRO = 64
BATCH = 256
NUM_CLASSES = 120


def count_params(tree):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def load_checkpoint(path):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    if payload.get("model") != MODEL_NAME:
        raise ValueError(
            f"Expected {MODEL_NAME}, got {payload.get('model')!r}"
        )
    if "ema_params" not in payload:
        raise ValueError("Checkpoint missing ema_params")
    params = payload["ema_params"]
    n = count_params(params)
    if n != EXPECTED_PARAMS:
        raise RuntimeError(f"Parameter mismatch {n:,} != {EXPECTED_PARAMS:,}")
    return payload, params, dict(payload["config"])


def make_model(config):
    return NestSARParallelT16(
        spatial_dim=config["spatial_dim"],
        model_dim=MODEL_DIM,
        dropout=config["dropout"],
        controller_dim=config["controller_dim"],
        fast_rank=config["fast_rank"],
        head_rank=config["head_rank"],
        sm_residual_scale=config["sm_residual_scale"],
        head_residual_scale=config["head_residual_scale"],
        m4_half_lives=M4_HALF_LIVES,
        g4_half_lives=G4_HALF_LIVES,
    )


def fused_desc(out):
    desc = out["descriptors"]
    fusion = out["fusion_weights"]
    z = jnp.einsum("bs,bsd->bd", fusion, desc)
    norm2 = jnp.sum(jnp.square(z), axis=-1, keepdims=True)
    return z * jax.lax.rsqrt(jnp.maximum(norm2, 1e-12))


def make_forward(model):
    @jax.jit
    def forward(params, x, xa, key):
        k1, k2 = jax.random.split(key)
        out = model.apply(
            {"params": params},
            x,
            training=True,
            rngs={"dropout": k1},
        )
        aug = model.apply(
            {"params": params},
            xa,
            training=True,
            rngs={"dropout": k2},
        )
        return fused_desc(out), fused_desc(aug)

    return forward


def hard_negative_stats(z, za, y, mask, same_micro):
    """Return exact directional hard negatives and selected classes."""
    z = np.asarray(z, np.float32)
    za = np.asarray(za, np.float32)
    y = np.asarray(y, np.int32)
    valid = np.asarray(mask > 0)

    sim_ab = z @ za.T
    sim_ba = za @ z.T

    different = y[:, None] != y[None, :]
    pair = different & valid[:, None] & valid[None, :]

    if same_micro:
        group = np.arange(len(y)) // MICRO
        pair &= group[:, None] == group[None, :]

    masked_ab = np.where(pair, sim_ab, -1e9)
    masked_ba = np.where(pair, sim_ba, -1e9)

    idx_ab = masked_ab.argmax(axis=1)
    idx_ba = masked_ba.argmax(axis=1)

    neg_ab = masked_ab[np.arange(len(y)), idx_ab]
    neg_ba = masked_ba[np.arange(len(y)), idx_ba]

    has_neg = pair.any(axis=1)

    pos = np.sum(z * za, axis=1)

    # Invalid/padded rows do not contribute.
    neg_ab = np.where(has_neg & valid, neg_ab, np.nan)
    neg_ba = np.where(has_neg & valid, neg_ba, np.nan)
    pos = np.where(valid, pos, np.nan)

    cls_ab = np.where(has_neg & valid, y[idx_ab], -1)
    cls_ba = np.where(has_neg & valid, y[idx_ba], -1)

    candidate_counts = pair.sum(axis=1).astype(np.int32)

    return {
        "pos": pos.astype(np.float32),
        "neg_ab": neg_ab.astype(np.float32),
        "neg_ba": neg_ba.astype(np.float32),
        "cls_ab": cls_ab.astype(np.int16),
        "cls_ba": cls_ba.astype(np.int16),
        "candidate_counts": candidate_counts,
        "valid": valid,
    }


def safe_chunk(path, expected_n):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            result = {k: np.asarray(z[k]) for k in z.files}
        if result["labels"].shape != (expected_n,):
            raise ValueError(result["labels"].shape)
        return result
    except Exception:
        path.unlink(missing_ok=True)
        return None


def summarize_pool(pos, neg_ab, neg_ba, margins):
    valid = np.isfinite(pos) & np.isfinite(neg_ab) & np.isfinite(neg_ba)
    pos = pos[valid]
    neg_ab = neg_ab[valid]
    neg_ba = neg_ba[valid]

    gap_ab = pos - neg_ab
    gap_ba = pos - neg_ba
    gap_mean = 0.5 * (gap_ab + gap_ba)

    out = {
        "samples": int(len(pos)),
        "positive_similarity": {
            "mean": float(np.mean(pos)),
            "std": float(np.std(pos)),
            "q05": float(np.quantile(pos, 0.05)),
            "q25": float(np.quantile(pos, 0.25)),
            "q50": float(np.quantile(pos, 0.50)),
            "q75": float(np.quantile(pos, 0.75)),
            "q95": float(np.quantile(pos, 0.95)),
        },
        "hard_negative_similarity": {
            "mean": float(np.mean(0.5 * (neg_ab + neg_ba))),
            "std": float(np.std(0.5 * (neg_ab + neg_ba))),
            "q05": float(np.quantile(0.5 * (neg_ab + neg_ba), 0.05)),
            "q25": float(np.quantile(0.5 * (neg_ab + neg_ba), 0.25)),
            "q50": float(np.quantile(0.5 * (neg_ab + neg_ba), 0.50)),
            "q75": float(np.quantile(0.5 * (neg_ab + neg_ba), 0.75)),
            "q95": float(np.quantile(0.5 * (neg_ab + neg_ba), 0.95)),
        },
        "positive_minus_negative_gap": {
            "mean": float(np.mean(gap_mean)),
            "std": float(np.std(gap_mean)),
            "q01": float(np.quantile(gap_mean, 0.01)),
            "q05": float(np.quantile(gap_mean, 0.05)),
            "q10": float(np.quantile(gap_mean, 0.10)),
            "q25": float(np.quantile(gap_mean, 0.25)),
            "q50": float(np.quantile(gap_mean, 0.50)),
            "q75": float(np.quantile(gap_mean, 0.75)),
            "q90": float(np.quantile(gap_mean, 0.90)),
        },
        "margins": {},
    }

    for m in margins:
        la = np.maximum(0.0, m - gap_ab)
        lb = np.maximum(0.0, m - gap_ba)
        sample_loss = 0.5 * (la + lb)

        out["margins"][f"{m:.2f}"] = {
            "sample_active_fraction": float(np.mean(sample_loss > 1e-8)),
            "direction_active_fraction": float(
                0.5 * (
                    np.mean(la > 1e-8)
                    + np.mean(lb > 1e-8)
                )
            ),
            "mean_loss": float(np.mean(sample_loss)),
            "mean_loss_active_only": float(
                np.mean(sample_loss[sample_loss > 1e-8])
                if np.any(sample_loss > 1e-8)
                else 0.0
            ),
        }

    return out


def top_pairs(labels, cls_ab, cls_ba, k=20):
    counts = {}
    labels = np.asarray(labels, np.int32)

    for hard in (cls_ab, cls_ba):
        hard = np.asarray(hard, np.int32)
        valid = hard >= 0
        for true, neg in zip(labels[valid], hard[valid]):
            if true == neg:
                continue
            a, b = sorted((int(true), int(neg)))
            counts[(a, b)] = counts.get((a, b), 0) + 1

    rows = sorted(
        ((count, a, b) for (a, b), count in counts.items()),
        reverse=True,
    )[:k]

    return [
        {
            "action_a": int(a + 1),
            "action_b": int(b + 1),
            "selected_count": int(count),
        }
        for count, a, b in rows
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=("xsub", "xset"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--status", default=None)
    ap.add_argument("--epoch", type=int, default=1)
    args = ap.parse_args()

    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise RuntimeError(
            f"Expected one isolated GPU; backend={jax.default_backend()} "
            f"devices={jax.local_devices()}"
        )

    payload, params, config = load_checkpoint(args.checkpoint)
    if payload.get("protocol") != args.protocol:
        raise ValueError(
            f"Checkpoint protocol={payload.get('protocol')} != {args.protocol}"
        )

    dataset = Dataset(args.cache)
    train_ids = np.asarray(
        dataset.splits[f"{args.protocol}_train"],
        np.int64,
    )

    # Match the original D128 training augmentation regime.
    data_config = {
        "seed": 128,
        "fresh_augmentation": True,
        "rotation_degrees": 8.0,
        "jitter_shift": 1,
        "prefetch_batches": 2,
    }

    model = make_model(config)
    forward = make_forward(model)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(str(output) + ".resume")
    work.mkdir(parents=True, exist_ok=True)
    status = Path(args.status) if args.status else None

    identity = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_val_accuracy": float(payload["val_accuracy"]),
        "train_samples": int(len(train_ids)),
        "audit_epoch": int(args.epoch),
        "batch_size": BATCH,
        "micro_batch": MICRO,
        "margins": list(MARGINS),
        "training_mode_dropout": True,
    }

    identity_path = work / "identity.json"
    if identity_path.is_file():
        old = json.loads(identity_path.read_text())
        if old != identity:
            raise RuntimeError(
                "Resume identity mismatch; use a fresh output path."
            )
    else:
        atomic_json(identity_path, identity)

    print("=" * 122)
    print(f"{args.protocol.upper()} D128 MARGIN-ACTIVATION AUDIT")
    print("=" * 122)
    print("GPU:", jax.local_devices()[0])
    print(f"Checkpoint E{int(payload['epoch']):02d}")
    print(f"Train samples: {len(train_ids):,}")
    print(f"Audit augmentation epoch: {args.epoch}")
    print("Candidate pools: exact micro64 + control batch256")
    print("Margins:", MARGINS)
    print()

    chunks = []
    n_batches = math.ceil(len(train_ids) / BATCH)

    batches = dataset.batches(
        train_ids,
        BATCH,
        data_config,
        epoch=args.epoch,
        training=True,
        protocol=args.protocol,
    )

    for bi, (batch, prep_s) in enumerate(batches):
        real_n = int(np.sum(batch["mask"]))
        chunk_path = work / f"batch_{bi:04d}.npz"

        cached = safe_chunk(chunk_path, BATCH)
        if cached is not None:
            chunks.append(cached)
            if status:
                atomic_json(
                    status,
                    {
                        "protocol": args.protocol,
                        "phase": "Margin activation (resume)",
                        "current": bi + 1,
                        "total": n_batches,
                        "done": False,
                    },
                )
            continue

        key = jax.random.PRNGKey(
            0x4D415247
            + (100000 if args.protocol == "xset" else 0)
            + args.epoch * 1009
            + bi
        )

        z, za = jax.device_get(
            forward(
                params,
                jax.device_put(batch["x"]),
                jax.device_put(batch["xa"]),
                key,
            )
        )

        micro = hard_negative_stats(
            z,
            za,
            batch["y"],
            batch["mask"],
            same_micro=True,
        )

        full = hard_negative_stats(
            z,
            za,
            batch["y"],
            batch["mask"],
            same_micro=False,
        )

        # Exact pool-quality comparison by selected class.
        same_ab = (
            (micro["cls_ab"] == full["cls_ab"])
            & micro["valid"]
            & (micro["cls_ab"] >= 0)
        )
        same_ba = (
            (micro["cls_ba"] == full["cls_ba"])
            & micro["valid"]
            & (micro["cls_ba"] >= 0)
        )

        payload_np = {
            "labels": np.asarray(batch["y"], np.int16),
            "mask": np.asarray(batch["mask"], np.float32),
            "pos": micro["pos"],
            "micro_neg_ab": micro["neg_ab"],
            "micro_neg_ba": micro["neg_ba"],
            "micro_cls_ab": micro["cls_ab"],
            "micro_cls_ba": micro["cls_ba"],
            "micro_candidate_counts": micro["candidate_counts"],
            "full_neg_ab": full["neg_ab"],
            "full_neg_ba": full["neg_ba"],
            "full_cls_ab": full["cls_ab"],
            "full_cls_ba": full["cls_ba"],
            "full_candidate_counts": full["candidate_counts"],
            "same_hard_class_ab": same_ab.astype(np.int8),
            "same_hard_class_ba": same_ba.astype(np.int8),
        }

        tmp = chunk_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload_np)
        os.replace(tmp, chunk_path)
        chunks.append(payload_np)

        if status:
            atomic_json(
                status,
                {
                    "protocol": args.protocol,
                    "phase": "Margin activation",
                    "current": bi + 1,
                    "total": n_batches,
                    "done": False,
                    "real_samples_this_batch": real_n,
                    "prepare_s": float(prep_s),
                },
            )

        print(
            f"[{bi+1:03d}/{n_batches:03d}] "
            f"real={real_n:3d} | prep={prep_s:.2f}s"
        )

    def cat(name):
        return np.concatenate([np.asarray(c[name]) for c in chunks])

    labels = cat("labels").astype(np.int32)
    mask = cat("mask") > 0
    pos = cat("pos").astype(np.float32)

    micro_neg_ab = cat("micro_neg_ab").astype(np.float32)
    micro_neg_ba = cat("micro_neg_ba").astype(np.float32)
    full_neg_ab = cat("full_neg_ab").astype(np.float32)
    full_neg_ba = cat("full_neg_ba").astype(np.float32)

    micro_cls_ab = cat("micro_cls_ab").astype(np.int32)
    micro_cls_ba = cat("micro_cls_ba").astype(np.int32)
    full_cls_ab = cat("full_cls_ab").astype(np.int32)
    full_cls_ba = cat("full_cls_ba").astype(np.int32)

    micro_candidates = cat("micro_candidate_counts").astype(np.int32)
    full_candidates = cat("full_candidate_counts").astype(np.int32)

    same_ab = cat("same_hard_class_ab").astype(bool)
    same_ba = cat("same_hard_class_ba").astype(bool)

    # Strip final-batch padding.
    labels = labels[mask]
    pos = pos[mask]
    micro_neg_ab = micro_neg_ab[mask]
    micro_neg_ba = micro_neg_ba[mask]
    full_neg_ab = full_neg_ab[mask]
    full_neg_ba = full_neg_ba[mask]
    micro_cls_ab = micro_cls_ab[mask]
    micro_cls_ba = micro_cls_ba[mask]
    full_cls_ab = full_cls_ab[mask]
    full_cls_ba = full_cls_ba[mask]
    micro_candidates = micro_candidates[mask]
    full_candidates = full_candidates[mask]
    same_ab = same_ab[mask]
    same_ba = same_ba[mask]

    micro_summary = summarize_pool(
        pos,
        micro_neg_ab,
        micro_neg_ba,
        MARGINS,
    )
    full_summary = summarize_pool(
        pos,
        full_neg_ab,
        full_neg_ba,
        MARGINS,
    )

    hard_class_match = float(
        0.5 * (
            np.mean(same_ab)
            + np.mean(same_ba)
        )
    )

    result = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_val_accuracy": float(payload["val_accuracy"]),
        "train_samples": int(len(labels)),
        "audit_epoch": int(args.epoch),
        "batch_size": BATCH,
        "micro_batch": MICRO,
        "margins": list(MARGINS),
        "micro64": micro_summary,
        "batch256": full_summary,
        "candidate_pool": {
            "micro64_mean_different_class_candidates": float(
                np.mean(micro_candidates)
            ),
            "batch256_mean_different_class_candidates": float(
                np.mean(full_candidates)
            ),
            "micro64_over_batch256_candidate_ratio": float(
                np.mean(micro_candidates)
                / max(np.mean(full_candidates), 1e-12)
            ),
            "micro64_same_hard_negative_class_as_batch256": hard_class_match,
        },
        "top_micro64_hard_negative_pairs": top_pairs(
            labels,
            micro_cls_ab,
            micro_cls_ba,
            k=20,
        ),
        "top_batch256_hard_negative_pairs": top_pairs(
            labels,
            full_cls_ab,
            full_cls_ba,
            k=20,
        ),
        "interpretation_rules": {
            "margin_inactive": (
                "If micro64 sample_active_fraction at m=0.05 is near zero, "
                "the B warm loss was effectively inactive."
            ),
            "pool_too_small": (
                "If batch256 activation is materially higher than micro64 "
                "and hard-negative class match is low, gradient accumulation "
                "is hiding relevant competitors."
            ),
            "margin_too_small": (
                "If both pools are mostly inactive at 0.05 but activate at "
                "0.10-0.30, the chosen margin was too small."
            ),
        },
    }

    atomic_json(output, result)

    if status:
        atomic_json(
            status,
            {
                "protocol": args.protocol,
                "phase": "Done",
                "current": 1,
                "total": 1,
                "done": True,
            },
        )

    print()
    print("=" * 122)
    print("MARGIN ACTIVATION")
    print("=" * 122)
    print(
        f"{'margin':>8s} "
        f"{'micro64 active':>16s} "
        f"{'micro loss':>12s} "
        f"{'batch256 active':>17s} "
        f"{'batch loss':>12s}"
    )
    print("-" * 72)

    for m in MARGINS:
        key = f"{m:.2f}"
        mi = micro_summary["margins"][key]
        fu = full_summary["margins"][key]
        print(
            f"{m:8.2f} "
            f"{100*mi['sample_active_fraction']:15.3f}% "
            f"{mi['mean_loss']:12.6f} "
            f"{100*fu['sample_active_fraction']:16.3f}% "
            f"{fu['mean_loss']:12.6f}"
        )

    print()
    print("SIMILARITY / GAP")
    print(
        "  positive similarity mean:",
        f"{micro_summary['positive_similarity']['mean']:.6f}",
    )
    print(
        "  micro hard-negative mean:",
        f"{micro_summary['hard_negative_similarity']['mean']:.6f}",
    )
    print(
        "  batch hard-negative mean:",
        f"{full_summary['hard_negative_similarity']['mean']:.6f}",
    )
    print(
        "  micro pos-neg median gap:",
        f"{micro_summary['positive_minus_negative_gap']['q50']:.6f}",
    )
    print(
        "  batch pos-neg median gap:",
        f"{full_summary['positive_minus_negative_gap']['q50']:.6f}",
    )

    print()
    print("CANDIDATE POOL")
    print(
        "  micro64 candidates:",
        f"{np.mean(micro_candidates):.2f}",
    )
    print(
        "  batch256 candidates:",
        f"{np.mean(full_candidates):.2f}",
    )
    print(
        "  pool ratio:",
        f"{result['candidate_pool']['micro64_over_batch256_candidate_ratio']:.3f}x",
    )
    print(
        "  same hard-negative class:",
        f"{100*hard_class_match:.2f}%",
    )

    print()
    print("TOP MICRO64 HARD-NEGATIVE PAIRS")
    for row in result["top_micro64_hard_negative_pairs"][:15]:
        print(
            f"  A{row['action_a']:03d}<->A{row['action_b']:03d} "
            f"| selected={row['selected_count']}"
        )

    print()
    print("Saved:", output)
    print("=" * 122)


if __name__ == "__main__":
    main()

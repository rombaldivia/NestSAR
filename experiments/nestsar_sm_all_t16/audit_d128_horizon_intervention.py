#!/usr/bin/env python3
from __future__ import annotations

"""Causal horizon intervention audit for trained NestSAR D128-MTS.

Read-only. Loads the EMA best checkpoint and perturbs ONLY forget-gate biases
at inference. This directly tests whether the observed M4/G4 horizon collapse
is causally related to validation accuracy.

Interventions:
  * M4 H15 only: negative control and positive sweep.
  * M4 H7+H15: positive sweep.
  * M4 all groups: small positive control.
  * G4 H8 only: negative control and positive sweep.
  * G4 H4+H8: positive sweep.
  * matched M4+G4 long-scale combinations.

All four semantic streams and both directions are modified identically.
No checkpoint is written or modified.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict

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


NUM_CLASSES = 120
STREAMS = 4


def count_params(tree):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def load_checkpoint(path):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    required = ("ema_params", "config", "model", "protocol", "epoch", "val_accuracy")
    for key in required:
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


def expected_bias_paths(flat, stage):
    needle = (
        "frame_memory_group/base_memory"
        if stage == "m4"
        else "descriptor_group/chunk_memory/base_memory"
    )
    found = []
    for path, value in flat.items():
        text = "/".join(map(str, path))
        if needle in text and text.endswith("/forget/bias"):
            arr = np.asarray(value)
            if arr.shape != (STREAMS, MODEL_DIM):
                raise ValueError(f"Unexpected {stage} forget bias shape {arr.shape}: {text}")
            found.append(path)
    if len(found) != 2:
        raise ValueError(
            f"Expected exactly fwd+bwd {stage} forget biases, found "
            f"{['/'.join(map(str,p)) for p in found]}"
        )
    return found


def apply_intervention(params, *, stage=None, groups=(), delta=0.0):
    if stage is None or not groups or delta == 0.0:
        return params

    flat = flatten_dict(params)
    paths = expected_bias_paths(flat, stage)

    half_lives = M4_HALF_LIVES if stage == "m4" else G4_HALF_LIVES
    group_width = MODEL_DIM // len(half_lives)

    out = dict(flat)
    for path in paths:
        arr = np.asarray(flat[path], np.float32).copy()
        for g in groups:
            if g < 0 or g >= len(half_lives):
                raise ValueError(f"Invalid group {g} for {stage}")
            lo = g * group_width
            hi = (g + 1) * group_width
            arr[:, lo:hi] += np.float32(delta)
        out[path] = jnp.asarray(arr)
    return unflatten_dict(out)


def apply_combo(params, spec):
    out = params
    for item in spec:
        out = apply_intervention(
            out,
            stage=item["stage"],
            groups=tuple(item["groups"]),
            delta=float(item["delta"]),
        )
    return out


def retention_from_bias_delta(retention, delta):
    """Equivalent retention when logit(a) receives +delta."""
    retention = np.clip(float(retention), 1e-7, 1 - 1e-7)
    logit = np.log(retention / (1.0 - retention))
    return float(1.0 / (1.0 + np.exp(-(logit + float(delta)))))


def nominal_half_life_after_shift(half_life, delta):
    a = 0.5 ** (1.0 / float(half_life))
    a2 = retention_from_bias_delta(a, delta)
    return float(np.log(0.5) / np.log(np.clip(a2, 1e-7, 1 - 1e-7)))


def per_class_accuracy(labels, correct):
    labels = np.asarray(labels, np.int32)
    correct = np.asarray(correct, bool)
    count = np.bincount(labels, minlength=NUM_CLASSES)
    good = np.bincount(labels[correct], minlength=NUM_CLASSES)
    acc = np.full(NUM_CLASSES, np.nan, np.float64)
    use = count > 0
    acc[use] = good[use] / count[use]
    return acc, count


def build_variants():
    # Group indices: M4 [H1,H3,H7,H15], G4 [H1,H2,H4,H8].
    return [
        ("canonical", []),

        # Negative controls: deliberately shorten the longest scale.
        ("m4_h15_m0p10", [{"stage": "m4", "groups": [3], "delta": -0.10}]),
        ("g4_h8_m0p10",  [{"stage": "g4", "groups": [3], "delta": -0.10}]),

        # M4 longest-scale causal sweep.
        ("m4_h15_p0p05", [{"stage": "m4", "groups": [3], "delta": 0.05}]),
        ("m4_h15_p0p10", [{"stage": "m4", "groups": [3], "delta": 0.10}]),
        ("m4_h15_p0p20", [{"stage": "m4", "groups": [3], "delta": 0.20}]),
        ("m4_h15_p0p30", [{"stage": "m4", "groups": [3], "delta": 0.30}]),

        # M4 two-longest-scale sweep.
        ("m4_h7h15_p0p05", [{"stage": "m4", "groups": [2, 3], "delta": 0.05}]),
        ("m4_h7h15_p0p10", [{"stage": "m4", "groups": [2, 3], "delta": 0.10}]),
        ("m4_h7h15_p0p20", [{"stage": "m4", "groups": [2, 3], "delta": 0.20}]),

        # Is "more retention everywhere" useful, or only on long groups?
        ("m4_all_p0p10", [{"stage": "m4", "groups": [0, 1, 2, 3], "delta": 0.10}]),

        # G4 controls / sweep.
        ("g4_h8_p0p05", [{"stage": "g4", "groups": [3], "delta": 0.05}]),
        ("g4_h8_p0p10", [{"stage": "g4", "groups": [3], "delta": 0.10}]),
        ("g4_h8_p0p20", [{"stage": "g4", "groups": [3], "delta": 0.20}]),
        ("g4_h8_p0p30", [{"stage": "g4", "groups": [3], "delta": 0.30}]),
        ("g4_h4h8_p0p10", [{"stage": "g4", "groups": [2, 3], "delta": 0.10}]),
        ("g4_h4h8_p0p20", [{"stage": "g4", "groups": [2, 3], "delta": 0.20}]),

        # Matched combined interventions.
        (
            "m4_h15_p0p10__g4_h8_p0p10",
            [
                {"stage": "m4", "groups": [3], "delta": 0.10},
                {"stage": "g4", "groups": [3], "delta": 0.10},
            ],
        ),
        (
            "m4_h15_p0p20__g4_h8_p0p10",
            [
                {"stage": "m4", "groups": [3], "delta": 0.20},
                {"stage": "g4", "groups": [3], "delta": 0.10},
            ],
        ),
    ]


def infer_predictions(apply_logits, params, dataset, val_ids, batch_size):
    parts = []
    for start in range(0, len(val_ids), batch_size):
        idx = val_ids[start:start + batch_size]
        x = np.asarray(dataset.canonical[idx], np.float32)
        logits = jax.device_get(apply_logits(params, jax.device_put(x)))
        parts.append(np.asarray(logits).argmax(axis=1))
    return np.concatenate(parts)


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
            f"Expected exactly one isolated GPU; backend={jax.default_backend()} "
            f"devices={jax.local_devices()}"
        )

    payload, params, config = load_checkpoint(args.checkpoint)
    if payload["protocol"] != args.protocol:
        raise ValueError(
            f"Checkpoint protocol={payload['protocol']} != requested {args.protocol}"
        )
    if int(config["model_dim"]) != MODEL_DIM:
        raise ValueError(f"Expected model_dim={MODEL_DIM}, got {config['model_dim']}")

    # Validate exact forget-bias paths before any evaluation.
    flat = flatten_dict(params)
    m4_paths = expected_bias_paths(flat, "m4")
    g4_paths = expected_bias_paths(flat, "g4")

    dataset = Dataset(args.cache)
    val_ids = np.asarray(dataset.splits[f"{args.protocol}_val"], np.int64)
    labels = np.asarray(dataset.labels[val_ids], np.int32)

    model = make_model(config)
    apply_logits = jax.jit(
        lambda p, x: model.apply({"params": p}, x, training=False)["logits"]
    )
    variants = build_variants()

    results = {}
    predictions = {}

    print("=" * 122)
    print(f"{MODEL_NAME} | {args.protocol.upper()} CAUSAL HORIZON INTERVENTION")
    print("=" * 122)
    print(f"GPU: {jax.local_devices()[0]}")
    print(f"Checkpoint: E{int(payload['epoch']):02d}")
    print(f"Recorded val: {100*float(payload['val_accuracy']):.4f}%")
    print(f"Val samples: {len(val_ids):,}")
    print(f"M4 forget paths: {['/'.join(map(str,p)) for p in m4_paths]}")
    print(f"G4 forget paths: {['/'.join(map(str,p)) for p in g4_paths]}")
    print()

    for i, (name, spec) in enumerate(variants, 1):
        p = apply_combo(params, spec)
        pred = infer_predictions(
            apply_logits, p, dataset, val_ids, args.batch_size
        )
        predictions[name] = pred
        acc = float(np.mean(pred == labels))
        results[name] = {
            "accuracy": acc,
            "spec": spec,
        }
        if args.status:
            atomic_json(
                args.status,
                {
                    "protocol": args.protocol,
                    "phase": "Horizon intervention",
                    "current": i,
                    "total": len(variants),
                    "variant": name,
                    "accuracy": acc,
                    "done": False,
                },
            )
        print(
            f"[{i:02d}/{len(variants):02d}] "
            f"{name:34s} acc={100*acc:8.4f}%"
        )

    base_pred = predictions["canonical"]
    base_correct = base_pred == labels
    base_acc = results["canonical"]["accuracy"]
    base_cls, cls_count = per_class_accuracy(labels, base_correct)

    # Require canonical reproduction to match the checkpoint closely.
    recorded = float(payload["val_accuracy"])
    if abs(base_acc - recorded) > 5e-5:
        raise RuntimeError(
            f"Canonical mismatch: evaluated={base_acc:.8f}, checkpoint={recorded:.8f}"
        )

    ranked = []
    for name, spec in variants:
        pred = predictions[name]
        correct = pred == labels
        acc = results[name]["accuracy"]
        cls, _ = per_class_accuracy(labels, correct)
        cls_delta = cls - base_cls
        finite = np.where(np.isfinite(cls_delta))[0]
        loss_order = finite[np.argsort(cls_delta[finite])[:10]]
        gain_order = finite[np.argsort(-cls_delta[finite])[:10]]

        row = results[name]
        row.update({
            "delta_pp": 100.0 * (acc - base_acc),
            "prediction_agreement": float(np.mean(pred == base_pred)),
            "fixed_vs_canonical": int(np.sum((~base_correct) & correct)),
            "broken_vs_canonical": int(np.sum(base_correct & (~correct))),
            "worst_class_deltas": [
                {
                    "ntu_action": int(c + 1),
                    "n": int(cls_count[c]),
                    "canonical_accuracy": float(base_cls[c]),
                    "variant_accuracy": float(cls[c]),
                    "delta_pp": float(100.0 * cls_delta[c]),
                }
                for c in loss_order
            ],
            "best_class_deltas": [
                {
                    "ntu_action": int(c + 1),
                    "n": int(cls_count[c]),
                    "canonical_accuracy": float(base_cls[c]),
                    "variant_accuracy": float(cls[c]),
                    "delta_pp": float(100.0 * cls_delta[c]),
                }
                for c in gain_order
            ],
        })
        if name != "canonical":
            ranked.append((row["delta_pp"], name))

    ranked.sort(reverse=True)

    # Nominal configured-horizon effect of each bias shift, useful for
    # interpreting magnitudes. Actual learned gates remain input-dependent.
    nominal = {}
    for stage, half_lives in (("m4", M4_HALF_LIVES), ("g4", G4_HALF_LIVES)):
        nominal[stage] = {}
        for delta in (-0.10, 0.05, 0.10, 0.20, 0.30):
            nominal[stage][f"{delta:+.2f}"] = [
                {
                    "configured_half_life": float(h),
                    "nominal_half_life_after_shift": nominal_half_life_after_shift(h, delta),
                }
                for h in half_lives
            ]

    best_delta, best_name = ranked[0]
    negative_controls = {
        "m4_h15_m0p10_delta_pp": results["m4_h15_m0p10"]["delta_pp"],
        "g4_h8_m0p10_delta_pp": results["g4_h8_m0p10"]["delta_pp"],
    }

    result = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_recorded_val_accuracy": recorded,
        "verified_canonical_accuracy": base_acc,
        "params": count_params(params),
        "val_samples": int(len(val_ids)),
        "m4_forget_bias_paths": ["/".join(map(str, p)) for p in m4_paths],
        "g4_forget_bias_paths": ["/".join(map(str, p)) for p in g4_paths],
        "variants": results,
        "ranking": [
            {"name": name, "delta_pp": float(delta)}
            for delta, name in ranked
        ],
        "best_variant": {
            "name": best_name,
            "delta_pp": float(best_delta),
            "accuracy": results[best_name]["accuracy"],
        },
        "negative_controls": negative_controls,
        "nominal_horizon_shift_reference": nominal,
        "decision_rules": {
            "horizon_causal_positive": (
                "Supported if a positive M4 long-scale intervention improves both "
                "XSUB and XSET, preferably with a dose-response region, while the "
                "negative M4 control is neutral or worse."
            ),
            "m4_bottleneck_but_not_horizon": (
                "Supported if stage localization says M4 is primary but all positive "
                "M4 horizon interventions are neutral/negative."
            ),
            "g4_horizon_secondary": (
                "Supported if G4 long-scale interventions consistently improve both "
                "protocols, especially after M4-only effects are accounted for."
            ),
            "do_not_train_on_single_protocol_only": (
                "A gain isolated to only XSUB or only XSET is not sufficient evidence "
                "for the next architecture change."
            ),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, result)
    if args.status:
        atomic_json(
            args.status,
            {
                "protocol": args.protocol,
                "phase": "Done",
                "current": len(variants),
                "total": len(variants),
                "variant": best_name,
                "accuracy": base_acc,
                "best_variant": best_name,
                "best_delta_pp": float(best_delta),
                "done": True,
            },
        )

    print()
    print("=" * 122)
    print("RANKING VS CANONICAL")
    print("=" * 122)
    print(
        f"canonical = {100*base_acc:.4f}% "
        f"(checkpoint {100*recorded:.4f}%)"
    )
    print()
    for delta, name in ranked:
        row = results[name]
        print(
            f"{name:34s} "
            f"{100*row['accuracy']:8.4f}% | "
            f"delta={row['delta_pp']:+8.4f} pp | "
            f"agree={100*row['prediction_agreement']:6.2f}% | "
            f"fixed={row['fixed_vs_canonical']:5d} | "
            f"broken={row['broken_vs_canonical']:5d}"
        )

    print()
    print("BEST:", best_name, f"{best_delta:+.4f} pp")
    print("Saved:", out)
    print("=" * 122)


if __name__ == "__main__":
    main()

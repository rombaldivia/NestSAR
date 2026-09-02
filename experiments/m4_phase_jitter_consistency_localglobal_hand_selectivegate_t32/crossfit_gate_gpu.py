#!/usr/bin/env python3
from __future__ import annotations

"""Cross-fitted diagnostic for the selective residual Hand-M4/G4 trust gate.

The frozen Hand-M4/G4 checkpoint is never updated.  For every validation sample
we cache main/hand logits and descriptors, build stratified K folds, train a fresh
2,481-parameter selective gate on K-1 folds, and evaluate only on the held-out
fold.  Concatenating held-out predictions yields leakage-free cross-fitted
predictions for mechanism diagnosis.

This is intentionally NOT a paper benchmark.
"""

import argparse
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_localglobal_t16 import preprocessing as lg
from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.model import (
    M4LocalGlobalHandM4G4T32,
)
from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.preprocessing import (
    HAND_FEATURES,
    HAND_FRAMES,
    hand_tokens_t32,
)
from experiments.m4_phase_jitter_consistency_localglobal_hand_selectivegate_t32.model import (
    SelectiveHandGate,
    GATE_EXTRA_PARAMS,
)

NUM_CLASSES = 120
RAW_BATCH = 512
TARGET_CLASSES = np.asarray(
    [
        11, 12, 16, 17, 29, 30, 34, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78,
        82, 83, 84, 91, 103, 105, 106, 107,
    ],
    dtype=np.int32,
)

EXPECTED_HAND_ACCURACY = {
    "xsub": 0.7543353168758223,
    "xset": 0.7617734586478807,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dget(d, key, default=None):
    if not isinstance(d, Mapping):
        return default
    if key in d:
        return d[key]
    b = key.encode()
    if b in d:
        return d[b]
    return default


def count_params(params) -> int:
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)))


def compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", choices=["xsub", "xset"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--gate-batch-size", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--base-alpha", type=float, default=0.20)
    p.add_argument("--delta-alpha", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=128)
    return p.parse_args()


def stratified_folds(y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    out = np.full(len(y), -1, dtype=np.int32)
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(y == c).copy()
        rng = np.random.default_rng(seed + 1009 * (c + 1))
        rng.shuffle(idx)
        out[idx] = np.arange(len(idx), dtype=np.int32) % folds
    if np.any(out < 0):
        raise RuntimeError("Stratified fold assignment failed")
    return out


def exact_mcnemar_p(fixes: int, harms: int) -> float:
    n = fixes + harms
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(fixes, n, p=0.5).pvalue)
    except Exception:
        z = (abs(fixes - harms) - 1.0) / math.sqrt(n)
        return float(math.erfc(max(0.0, z) / math.sqrt(2.0)))


def extract_validation_cache(
    dataset: Path,
    protocol: str,
    frozen_params,
    cache_path: Path,
):
    if cache_path.is_file():
        log(f"Loading existing frozen cache: {cache_path}")
        d = np.load(cache_path, allow_pickle=False)
        return (
            d["main_logits"],
            d["hand_logits"],
            d["main_desc"],
            d["hand_desc"],
            d["labels"],
        )

    annotations, split = base.load_ntu(dataset)
    by_id, _, val_ids = ju.resolve_protocol_ids(annotations, split, protocol)
    n_total = len(val_ids)

    frozen_model = M4LocalGlobalHandM4G4T32(
        spatial_dim=24,
        model_dim=112,
        dropout=0.10,
        hand_dim=32,
        hand_residual_scale=0.10,
    )

    @jax.jit
    def infer(params, x, h):
        out = frozen_model.apply({"params": params}, x, h, training=False)
        return (
            out["main_logits"],
            out["hand_logits"],
            jnp.mean(out["descriptors"], axis=1),
            out["hand_descriptor"],
        )

    main_logits = np.empty((n_total, NUM_CLASSES), np.float32)
    hand_logits = np.empty_like(main_logits)
    main_desc = np.empty((n_total, 112), np.float32)
    hand_desc = np.empty((n_total, 32), np.float32)
    labels = np.empty((n_total,), np.int32)

    log(f"{protocol.upper()} frozen validation cache START | N={n_total:,}")
    t0 = time.time()

    for start in range(0, n_total, RAW_BATCH):
        ids = val_ids[start:start + RAW_BATCH]
        n = len(ids)

        xb = np.zeros((RAW_BATCH, ju.FRAMES, ju.FEATURES), np.float32)
        hb = np.zeros((RAW_BATCH, HAND_FRAMES, HAND_FEATURES), np.float32)
        yb = np.zeros((RAW_BATCH,), np.int32)

        for i, sid in enumerate(ids):
            a = by_id[sid]
            kp = base.annotation_keypoints(a)
            xb[i] = lg.segment_phase_tokens_localglobal(kp)
            hb[i] = hand_tokens_t32(kp)
            yb[i] = base.annotation_label(a)

        ml, hl, md, hd = jax.device_get(
            infer(frozen_params, jnp.asarray(xb), jnp.asarray(hb))
        )

        main_logits[start:start+n] = np.asarray(ml[:n], np.float32)
        hand_logits[start:start+n] = np.asarray(hl[:n], np.float32)
        main_desc[start:start+n] = np.asarray(md[:n], np.float32)
        hand_desc[start:start+n] = np.asarray(hd[:n], np.float32)
        labels[start:start+n] = yb[:n]

        done = start + n
        if done % 10_000 < RAW_BATCH or done == n_total:
            log(f"{protocol.upper()} frozen cache {done:,}/{n_total:,}")

    np.savez(
        cache_path,
        main_logits=main_logits,
        hand_logits=hand_logits,
        main_desc=main_desc,
        hand_desc=hand_desc,
        labels=labels,
    )

    log(
        f"{protocol.upper()} frozen validation cache READY | "
        f"time={time.time()-t0:.1f}s"
    )

    return main_logits, hand_logits, main_desc, hand_desc, labels


def main() -> None:
    args = parse_args()

    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, got backend={jax.default_backend()} "
            f"count={jax.local_device_count()}"
        )

    dataset = Path(args.dataset)
    checkpoint = Path(args.checkpoint)
    outdir = Path(args.outdir) / args.protocol
    outdir.mkdir(parents=True, exist_ok=True)

    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    payload = serialization.msgpack_restore(checkpoint.read_bytes())
    frozen_params = dget(payload, "ema_params")
    if frozen_params is None:
        raise RuntimeError("Hand checkpoint contains no ema_params")

    if count_params(frozen_params) != 1_854_650:
        raise RuntimeError(
            f"Frozen checkpoint param mismatch: {count_params(frozen_params):,}"
        )

    lo = args.base_alpha - args.delta_alpha
    hi = args.base_alpha + args.delta_alpha

    print("=" * 120, flush=True)
    print(
        f"NESTSAR HAND-M4/G4 T32 + SELECTIVE TRUST GATE | "
        f"{args.protocol.upper()} | {args.folds}-FOLD CROSSFIT DIAGNOSTIC",
        flush=True,
    )
    print("=" * 120, flush=True)
    print("JAX:", jax.__version__, flush=True)
    print("BACKEND:", jax.default_backend(), flush=True)
    print("DEVICE:", jax.local_devices(), flush=True)
    print("FROZEN CHECKPOINT:", checkpoint, flush=True)
    print("GATE PARAMS:", GATE_EXTRA_PARAMS, flush=True)
    print("BASE ALPHA:", args.base_alpha, flush=True)
    print("DELTA ALPHA:", args.delta_alpha, flush=True)
    print(f"ALPHA RANGE: [{lo:.3f}, {hi:.3f}]", flush=True)
    print("IMPORTANT: CROSSFIT DIAGNOSTIC ONLY — NOT PAPER BENCHMARK", flush=True)
    print("=" * 120, flush=True)

    cache_path = outdir / "frozen_validation_cache.npz"
    ML, HL, MD, HD, y = extract_validation_cache(
        dataset, args.protocol, frozen_params, cache_path
    )

    fixed_010_logits = ML + 0.10 * HL
    fixed_010_pred = fixed_010_logits.argmax(axis=1)
    fixed_010_acc = float(np.mean(fixed_010_pred == y))
    expected = EXPECTED_HAND_ACCURACY[args.protocol]

    log(
        f"Frozen alpha=0.10 reproduction: {100*fixed_010_acc:.6f}% | "
        f"expected={100*expected:.6f}%"
    )

    if abs(fixed_010_acc - expected) > 1e-10:
        raise RuntimeError(
            f"Checkpoint reproduction mismatch: {fixed_010_acc} != {expected}"
        )

    fixed_scores = {}
    for alpha in (0.0, 0.10, 0.20, 0.25, 0.30, 0.35):
        pred = (ML + alpha * HL).argmax(axis=1)
        fixed_scores[f"alpha_{alpha:.2f}"] = float(np.mean(pred == y))

    folds = stratified_folds(y, args.folds, args.seed)

    gate = SelectiveHandGate(
        hidden_dim=16,
        base_alpha=args.base_alpha,
        delta_alpha=args.delta_alpha,
    )

    init_key = jax.random.PRNGKey(args.seed)
    p0 = gate.init(
        init_key,
        jnp.asarray(MD[:1]),
        jnp.asarray(HD[:1]),
        jnp.asarray(ML[:1]),
        jnp.asarray(HL[:1]),
    )["params"]
    if count_params(p0) != GATE_EXTRA_PARAMS:
        raise RuntimeError(
            f"Gate param mismatch: {count_params(p0):,} != {GATE_EXTRA_PARAMS:,}"
        )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(args.learning_rate, weight_decay=args.weight_decay),
    )

    @jax.jit
    def train_step(params, opt_state, md, hd, ml, hl, yy):
        def loss_fn(p):
            out = gate.apply({"params": p}, md, hd, ml, hl)
            ce = optax.softmax_cross_entropy_with_integer_labels(
                out["logits"], yy
            )
            loss = jnp.mean(ce)
            pred = jnp.argmax(out["logits"], axis=-1)
            acc = jnp.mean(pred == yy)
            disagreement = jnp.argmax(ml, axis=-1) != jnp.argmax(hl, axis=-1)
            disagree_alpha = jnp.sum(
                out["alpha"] * disagreement.astype(out["alpha"].dtype)
            ) / jnp.maximum(jnp.sum(disagreement), 1)
            agree_alpha = jnp.sum(
                out["alpha"] * (~disagreement).astype(out["alpha"].dtype)
            ) / jnp.maximum(jnp.sum(~disagreement), 1)
            return loss, (
                acc,
                jnp.mean(out["alpha"]),
                disagree_alpha,
                agree_alpha,
            )

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, *aux

    @jax.jit
    def gate_infer(params, md, hd, ml, hl):
        out = gate.apply({"params": params}, md, hd, ml, hl)
        return out["logits"], out["alpha"], out["same_top"]

    oof_logits = np.empty_like(ML)
    oof_alpha = np.empty((len(y),), np.float32)
    oof_same = np.empty((len(y),), np.float32)
    fold_summaries = []

    for fold in range(args.folds):
        train_idx = np.flatnonzero(folds != fold)
        test_idx = np.flatnonzero(folds == fold)

        key = jax.random.PRNGKey(args.seed + 100_000 * (fold + 1))
        params = gate.init(
            key,
            jnp.asarray(MD[:1]),
            jnp.asarray(HD[:1]),
            jnp.asarray(ML[:1]),
            jnp.asarray(HL[:1]),
        )["params"]
        opt_state = tx.init(params)

        rng = np.random.default_rng(args.seed + 7919 * (fold + 1))
        last_loss = float("nan")
        last_acc = float("nan")
        last_alpha = float("nan")
        last_disagree_alpha = float("nan")
        last_agree_alpha = float("nan")

        for epoch in range(1, args.epochs + 1):
            order = train_idx.copy()
            rng.shuffle(order)

            sum_loss = 0.0
            sum_acc = 0.0
            sum_alpha = 0.0
            sum_disagree_alpha = 0.0
            sum_agree_alpha = 0.0
            seen = 0

            for start in range(0, len(order), args.gate_batch_size):
                ii = order[start:start + args.gate_batch_size]
                (
                    params,
                    opt_state,
                    loss,
                    acc,
                    alpha_mean,
                    disagree_alpha,
                    agree_alpha,
                ) = train_step(
                    params,
                    opt_state,
                    jnp.asarray(MD[ii]),
                    jnp.asarray(HD[ii]),
                    jnp.asarray(ML[ii]),
                    jnp.asarray(HL[ii]),
                    jnp.asarray(y[ii]),
                )
                n = len(ii)
                sum_loss += float(loss) * n
                sum_acc += float(acc) * n
                sum_alpha += float(alpha_mean) * n
                sum_disagree_alpha += float(disagree_alpha) * n
                sum_agree_alpha += float(agree_alpha) * n
                seen += n

            last_loss = sum_loss / seen
            last_acc = sum_acc / seen
            last_alpha = sum_alpha / seen
            last_disagree_alpha = sum_disagree_alpha / seen
            last_agree_alpha = sum_agree_alpha / seen

            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                log(
                    f"{args.protocol.upper()} fold={fold+1}/{args.folds} "
                    f"E{epoch:02d} train_loss={last_loss:.5f} "
                    f"train_acc={100*last_acc:.3f}% alpha={last_alpha:.4f} "
                    f"disagree_alpha={last_disagree_alpha:.4f} "
                    f"agree_alpha={last_agree_alpha:.4f}"
                )

        logits, alpha, same = jax.device_get(
            gate_infer(
                params,
                jnp.asarray(MD[test_idx]),
                jnp.asarray(HD[test_idx]),
                jnp.asarray(ML[test_idx]),
                jnp.asarray(HL[test_idx]),
            )
        )

        logits = np.asarray(logits, np.float32)
        alpha = np.asarray(alpha, np.float32)
        same = np.asarray(same, np.float32)

        oof_logits[test_idx] = logits
        oof_alpha[test_idx] = alpha
        oof_same[test_idx] = same

        held_acc = float(np.mean(logits.argmax(axis=1) == y[test_idx]))
        disagree_mask = same < 0.5
        agree_mask = ~disagree_mask
        held_disagree_alpha = float(np.mean(alpha[disagree_mask])) if np.any(disagree_mask) else float("nan")
        held_agree_alpha = float(np.mean(alpha[agree_mask])) if np.any(agree_mask) else float("nan")

        log(
            f"{args.protocol.upper()} fold={fold+1}/{args.folds} HELDOUT "
            f"acc={100*held_acc:.6f}% | alpha_mean={alpha.mean():.4f} | "
            f"disagree_alpha={held_disagree_alpha:.4f} | "
            f"agree_alpha={held_agree_alpha:.4f}"
        )

        fold_summaries.append(
            {
                "fold": fold + 1,
                "heldout_accuracy": held_acc,
                "alpha_mean": float(alpha.mean()),
                "alpha_p10": float(np.percentile(alpha, 10)),
                "alpha_p50": float(np.percentile(alpha, 50)),
                "alpha_p90": float(np.percentile(alpha, 90)),
                "disagree_alpha": held_disagree_alpha,
                "agree_alpha": held_agree_alpha,
            }
        )

    oof_pred = oof_logits.argmax(axis=1)
    adaptive_acc = float(np.mean(oof_pred == y))

    fixed010_ok = fixed_010_pred == y
    adaptive_ok = oof_pred == y
    fixes = int(np.sum((~fixed010_ok) & adaptive_ok))
    harms = int(np.sum(fixed010_ok & (~adaptive_ok)))
    pvalue = exact_mcnemar_p(fixes, harms)

    target_mask = np.isin(y + 1, TARGET_CLASSES)
    fixed_target = float(np.mean(fixed010_ok[target_mask]))
    adaptive_target = float(np.mean(adaptive_ok[target_mask]))
    weak_delta = 100.0 * (adaptive_target - fixed_target)

    main_pred = ML.argmax(axis=1)
    hand_pred = HL.argmax(axis=1)
    disagree = main_pred != hand_pred

    alpha_stats = {
        "mean": float(np.mean(oof_alpha)),
        "std": float(np.std(oof_alpha)),
        "p01": float(np.percentile(oof_alpha, 1)),
        "p10": float(np.percentile(oof_alpha, 10)),
        "p25": float(np.percentile(oof_alpha, 25)),
        "p50": float(np.percentile(oof_alpha, 50)),
        "p75": float(np.percentile(oof_alpha, 75)),
        "p90": float(np.percentile(oof_alpha, 90)),
        "p99": float(np.percentile(oof_alpha, 99)),
        "agree_mean": float(np.mean(oof_alpha[~disagree])),
        "disagree_mean": float(np.mean(oof_alpha[disagree])),
        "fraction_below_015": float(np.mean(oof_alpha < 0.15)),
        "fraction_above_025": float(np.mean(oof_alpha > 0.25)),
        "fraction_above_032": float(np.mean(oof_alpha > 0.32)),
    }

    # Extra rescue diagnostic relative to the frozen main path.
    main_ok = main_pred == y
    hand_ok = hand_pred == y
    opportunities = (~main_ok) & hand_ok
    rescued = (~main_ok) & adaptive_ok
    harmed_main = main_ok & (~adaptive_ok)

    print("=" * 120, flush=True)
    print(f"{args.protocol.upper()} SELECTIVE GATE CROSSFIT RESULT", flush=True)
    print("=" * 120, flush=True)
    for key, value in fixed_scores.items():
        print(f"FIXED {key}: {100*value:.6f}%", flush=True)
    print(f"CROSSFIT SELECTIVE: {100*adaptive_acc:.6f}%", flush=True)
    print(
        f"DELTA VS FIXED 0.10: {100*(adaptive_acc-fixed_010_acc):+.6f} pp",
        flush=True,
    )
    print(f"WEAK TARGET DELTA: {weak_delta:+.6f} pp", flush=True)
    print(f"FIXES={fixes} | HARMS={harms} | McNemar p={pvalue:.10g}", flush=True)
    print(
        "ALPHA "
        f"mean={alpha_stats['mean']:.4f} std={alpha_stats['std']:.4f} "
        f"p10={alpha_stats['p10']:.4f} p50={alpha_stats['p50']:.4f} "
        f"p90={alpha_stats['p90']:.4f}",
        flush=True,
    )
    print(
        f"ALPHA agree={alpha_stats['agree_mean']:.4f} | "
        f"disagree={alpha_stats['disagree_mean']:.4f}",
        flush=True,
    )
    print(
        f"MAIN-WRONG/HAND-CORRECT opportunities={int(opportunities.sum())} | "
        f"adaptive_rescued={int(rescued.sum())} | "
        f"main_correct_harmed={int(harmed_main.sum())}",
        flush=True,
    )
    print("DIAGNOSTIC ONLY — DO NOT REPORT CROSSFIT SCORE AS PAPER BENCHMARK", flush=True)
    print("=" * 120, flush=True)

    result = {
        "protocol": args.protocol,
        "diagnostic": "selective_residual_gate_crossfit",
        "fixed_scores": fixed_scores,
        "crossfit_selective_accuracy": adaptive_acc,
        "delta_vs_fixed_010_pp": 100.0 * (adaptive_acc - fixed_010_acc),
        "weak_target_delta_pp": weak_delta,
        "fixes_vs_fixed_010": fixes,
        "harms_vs_fixed_010": harms,
        "mcnemar_p": pvalue,
        "base_alpha": args.base_alpha,
        "delta_alpha": args.delta_alpha,
        "alpha_min": lo,
        "alpha_max": hi,
        "alpha_stats": alpha_stats,
        "main_wrong_hand_correct_opportunities": int(opportunities.sum()),
        "adaptive_rescued_main_wrong": int(rescued.sum()),
        "adaptive_harmed_main_correct": int(harmed_main.sum()),
        "gate_params": GATE_EXTRA_PARAMS,
        "folds": fold_summaries,
        "not_paper_benchmark": True,
    }

    (outdir / "result.json").write_text(json.dumps(result, indent=2))
    np.savez(
        outdir / "crossfit_predictions.npz",
        labels=y,
        main_logits=ML,
        hand_logits=HL,
        adaptive_logits=oof_logits,
        alpha=oof_alpha,
        fold_id=folds,
    )


if __name__ == "__main__":
    main()

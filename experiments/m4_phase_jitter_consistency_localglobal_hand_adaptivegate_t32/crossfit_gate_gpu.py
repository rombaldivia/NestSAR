#!/usr/bin/env python3
from __future__ import annotations

"""Cross-fitted diagnostic for a sample-wise Hand-M4/G4 trust gate.

This is intentionally NOT a paper benchmark.  It answers one mechanism question:
can a tiny learned gate identify when the validated T32 hand specialist should be
trusted more than a fixed residual coefficient?

The frozen Hand-M4/G4 checkpoint is never updated.  We first cache, for every
validation sample, the frozen main/hand logits and descriptors.  We then create
stratified K folds.  For each fold, a fresh 2,401-parameter gate is trained on the
other folds and evaluated only on the held-out fold.  Concatenating all held-out
predictions gives leakage-free sample-wise cross-fitted predictions.

If this diagnostic beats fixed alpha consistently, the next experiment should
co-train the adaptive gate from scratch with the full model.
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
from experiments.m4_phase_jitter_consistency_localglobal_hand_adaptivegate_t32.model import (
    AdaptiveHandGate,
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
    return int(
        sum(
            np.asarray(x).size
            for x in jax.tree_util.tree_leaves(params)
        )
    )


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
    p.add_argument("--max-alpha", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=128)
    return p.parse_args()


def stratified_folds(y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    out = np.full(len(y), -1, dtype=np.int32)
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(y == c)
        rng = np.random.default_rng(seed + 1009 * (c + 1))
        idx = idx.copy()
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
        out = frozen_model.apply(
            {"params": params},
            x,
            h,
            training=False,
        )
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
            infer(
                frozen_params,
                jnp.asarray(xb),
                jnp.asarray(hb),
            )
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

    print("=" * 120, flush=True)
    print(
        f"NESTSAR HAND-M4/G4 T32 + ADAPTIVE TRUST GATE | "
        f"{args.protocol.upper()} | {args.folds}-FOLD CROSSFIT DIAGNOSTIC",
        flush=True,
    )
    print("=" * 120, flush=True)
    print("JAX:", jax.__version__, flush=True)
    print("BACKEND:", jax.default_backend(), flush=True)
    print("DEVICE:", jax.local_devices(), flush=True)
    print("FROZEN CHECKPOINT:", checkpoint, flush=True)
    print("GATE PARAMS:", GATE_EXTRA_PARAMS, flush=True)
    print("MAX ALPHA:", args.max_alpha, flush=True)
    print("IMPORTANT: CROSSFIT DIAGNOSTIC ONLY — NOT PAPER BENCHMARK", flush=True)
    print("=" * 120, flush=True)

    cache_path = outdir / "frozen_validation_cache.npz"
    ML, HL, MD, HD, y = extract_validation_cache(
        dataset,
        args.protocol,
        frozen_params,
        cache_path,
    )

    # Reproduce the frozen checkpoint's trained alpha=0.10 result exactly.
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
    for alpha in (0.0, 0.10, 0.20, 0.30):
        pred = (ML + alpha * HL).argmax(axis=1)
        fixed_scores[f"alpha_{alpha:.2f}"] = float(np.mean(pred == y))

    folds = stratified_folds(y, args.folds, args.seed)

    gate = AdaptiveHandGate(
        hidden_dim=16,
        max_alpha=args.max_alpha,
    )

    # Shape-only deterministic gate initialization preflight.
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
        optax.adamw(
            args.learning_rate,
            weight_decay=args.weight_decay,
        ),
    )

    @jax.jit
    def train_step(params, opt_state, md, hd, ml, hl, yy):
        def loss_fn(p):
            out = gate.apply({"params": p}, md, hd, ml, hl)
            ce = optax.softmax_cross_entropy_with_integer_labels(
                out["logits"],
                yy,
            )
            loss = jnp.mean(ce)
            acc = jnp.mean(jnp.argmax(out["logits"], axis=-1) == yy)
            return loss, (acc, jnp.mean(out["alpha"]))

        (loss, (acc, alpha_mean)), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, acc, alpha_mean

    @jax.jit
    def gate_infer(params, md, hd, ml, hl):
        out = gate.apply({"params": params}, md, hd, ml, hl)
        return out["logits"], out["alpha"]

    oof_logits = np.empty_like(ML)
    oof_alpha = np.empty((len(y),), np.float32)
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

        for epoch in range(1, args.epochs + 1):
            order = train_idx.copy()
            rng.shuffle(order)

            sum_loss = 0.0
            sum_acc = 0.0
            sum_alpha = 0.0
            seen = 0

            for start in range(0, len(order), args.gate_batch_size):
                ii = order[start:start + args.gate_batch_size]
                params, opt_state, loss, acc, alpha_mean = train_step(
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
                seen += n

            last_loss = sum_loss / seen
            last_acc = sum_acc / seen
            last_alpha = sum_alpha / seen

            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                log(
                    f"{args.protocol.upper()} fold={fold+1}/{args.folds} "
                    f"E{epoch:02d} train_loss={last_loss:.5f} "
                    f"train_acc={100*last_acc:.3f}% alpha={last_alpha:.4f}"
                )

        test_logits, test_alpha = jax.device_get(
            gate_infer(
                params,
                jnp.asarray(MD[test_idx]),
                jnp.asarray(HD[test_idx]),
                jnp.asarray(ML[test_idx]),
                jnp.asarray(HL[test_idx]),
            )
        )

        oof_logits[test_idx] = np.asarray(test_logits, np.float32)
        oof_alpha[test_idx] = np.asarray(test_alpha, np.float32)

        test_pred = np.asarray(test_logits).argmax(axis=1)
        test_acc = float(np.mean(test_pred == y[test_idx]))

        fold_payload = {
            "fold": fold,
            "train_count": int(len(train_idx)),
            "heldout_count": int(len(test_idx)),
            "heldout_accuracy": test_acc,
            "heldout_alpha_mean": float(np.mean(test_alpha)),
            "gate_params": jax.device_get(params),
            "config": vars(args),
        }
        (outdir / f"gate_fold_{fold}.msgpack").write_bytes(
            serialization.msgpack_serialize(fold_payload)
        )

        fold_summaries.append(
            {
                "fold": fold,
                "heldout_accuracy": test_acc,
                "heldout_alpha_mean": float(np.mean(test_alpha)),
                "heldout_alpha_p10": float(np.percentile(test_alpha, 10)),
                "heldout_alpha_p50": float(np.percentile(test_alpha, 50)),
                "heldout_alpha_p90": float(np.percentile(test_alpha, 90)),
                "final_train_loss": last_loss,
            }
        )

        log(
            f"{args.protocol.upper()} fold={fold+1}/{args.folds} HELDOUT "
            f"acc={100*test_acc:.6f}% | alpha_mean={np.mean(test_alpha):.4f}"
        )

    oof_pred = oof_logits.argmax(axis=1)
    oof_ok = oof_pred == y
    base_ok = fixed_010_pred == y

    fixes = int(np.sum((~base_ok) & oof_ok))
    harms = int(np.sum(base_ok & (~oof_ok)))
    pvalue = exact_mcnemar_p(fixes, harms)

    target_mask = np.isin(y + 1, TARGET_CLASSES)

    summary = {
        "protocol": args.protocol,
        "diagnostic_only": True,
        "method": f"{args.folds}-fold stratified crossfit on validation representations",
        "n": int(len(y)),
        "gate_params": GATE_EXTRA_PARAMS,
        "max_alpha": args.max_alpha,
        "fixed_scores": fixed_scores,
        "crossfit_accuracy": float(np.mean(oof_ok)),
        "crossfit_delta_vs_fixed_010_pp": float(
            100.0 * (np.mean(oof_ok) - np.mean(base_ok))
        ),
        "crossfit_target_weak_accuracy": float(np.mean(oof_ok[target_mask])),
        "fixed_010_target_weak_accuracy": float(np.mean(base_ok[target_mask])),
        "target_weak_delta_pp": float(
            100.0 * (
                np.mean(oof_ok[target_mask]) - np.mean(base_ok[target_mask])
            )
        ),
        "fixes_vs_fixed_010": fixes,
        "harms_vs_fixed_010": harms,
        "mcnemar_p": pvalue,
        "alpha_mean": float(np.mean(oof_alpha)),
        "alpha_p10": float(np.percentile(oof_alpha, 10)),
        "alpha_p50": float(np.percentile(oof_alpha, 50)),
        "alpha_p90": float(np.percentile(oof_alpha, 90)),
        "folds": fold_summaries,
    }

    np.savez(
        outdir / "crossfit_predictions.npz",
        labels=y,
        fixed_010_pred=fixed_010_pred,
        crossfit_pred=oof_pred,
        crossfit_logits=oof_logits,
        crossfit_alpha=oof_alpha,
        fold_assignment=folds,
    )
    (outdir / "result.json").write_text(json.dumps(summary, indent=2))

    print("=" * 120, flush=True)
    print(f"{args.protocol.upper()} ADAPTIVE GATE CROSSFIT RESULT", flush=True)
    print("=" * 120, flush=True)
    for k, v in fixed_scores.items():
        print(f"FIXED {k}: {100*v:.6f}%", flush=True)
    print(f"CROSSFIT ADAPTIVE: {100*summary['crossfit_accuracy']:.6f}%", flush=True)
    print(
        f"DELTA VS FIXED 0.10: {summary['crossfit_delta_vs_fixed_010_pp']:+.6f} pp",
        flush=True,
    )
    print(
        f"WEAK TARGET DELTA: {summary['target_weak_delta_pp']:+.6f} pp",
        flush=True,
    )
    print(f"FIXES={fixes:,} | HARMS={harms:,} | McNemar p={pvalue:.8g}", flush=True)
    print(
        f"ALPHA mean={summary['alpha_mean']:.4f} "
        f"p10={summary['alpha_p10']:.4f} "
        f"p50={summary['alpha_p50']:.4f} "
        f"p90={summary['alpha_p90']:.4f}",
        flush=True,
    )
    print("DIAGNOSTIC ONLY — DO NOT REPORT CROSSFIT SCORE AS PAPER BENCHMARK", flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()

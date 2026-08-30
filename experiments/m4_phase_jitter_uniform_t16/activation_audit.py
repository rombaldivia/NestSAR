#!/usr/bin/env python3
from __future__ import annotations

"""Activation/robustness audit for Phase-T16 + training-only boundary jitter + uniform fusion.

The audit is read-only. It loads the saved EMA checkpoint and reports:
  * canonical validation accuracy
  * +/-1 boundary-jitter validation accuracy and prediction agreement
  * router-off counterfactual
  * per-stream and leave-one-stream-out accuracy under fixed uniform fusion
  * phase/path/full-displacement input ablations
  * frozen stage-wise Ridge linear probes
  * router behavior and activation effective rank
  * hardest classes and jitter-sensitive classes

Validation/inference remains canonical for the reported benchmark. The jittered
validation view is diagnostic only and is NOT test-time augmentation.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from experiments.m4_motionpreserve_t16.jax10_compat import install as install_jax10_compat
install_jax10_compat()

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_motionpreserve_phase_t16 import train_phase_t16_tpu as phase
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as tr

STREAMS = ("J", "B", "JM", "BM")
EPS = 1e-12


def acc(y, pred) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def entropy_np(w, axis=-1, normalized=False):
    w = np.asarray(w, np.float64)
    h = -np.sum(w * np.log(np.maximum(w, EPS)), axis=axis)
    if normalized:
        h = h / math.log(w.shape[axis])
    return h


def resolve_ids(annotations, split, protocol: str):
    tk, vk = base.resolve_split(split, protocol)
    by_id = {
        base.sample_id(a, i): a
        for i, a in enumerate(annotations)
        if isinstance(a, Mapping)
    }
    train_ids = [str(v) for v in split[tk] if str(v) in by_id]
    val_ids = [str(v) for v in split[vk] if str(v) in by_id]
    if not train_ids or not val_ids:
        raise RuntimeError(f"Empty split for {protocol}: train={len(train_ids)} val={len(val_ids)}")
    return by_id, train_ids, val_ids


def subset_ids(ids: Sequence[str], n: int, seed: int):
    if n <= 0 or n >= len(ids):
        return list(ids)
    rng = np.random.default_rng(seed)
    ii = rng.choice(len(ids), size=n, replace=False)
    return [ids[int(i)] for i in ii]


def materialize_canonical(by_id, ids, desc):
    X = np.empty((len(ids), tr.FRAMES, tr.FEATURES), np.float32)
    y = np.empty((len(ids),), np.int32)
    for i, sid in enumerate(tqdm(ids, desc=desc, mininterval=0.5)):
        a = by_id[sid]
        X[i] = phase.segment_phase_tokens(base.annotation_keypoints(a))
        y[i] = base.annotation_label(a)
    return X, y


def materialize_jitter(by_id, ids, max_shift: int, seed: int, desc):
    X = np.empty((len(ids), tr.FRAMES, tr.FEATURES), np.float32)
    y = np.empty((len(ids),), np.int32)
    for i, sid in enumerate(tqdm(ids, desc=desc, mininterval=0.5)):
        a = by_id[sid]
        rng = np.random.default_rng(np.random.SeedSequence([seed, i, 314159]))
        X[i] = tr.jitter_phase_tokens(base.annotation_keypoints(a), max_shift, rng)
        y[i] = base.annotation_label(a)
    return X, y


def dense_apply(p, x):
    y = x @ p["kernel"]
    if "bias" in p:
        y = y + p["bias"]
    return y


def build_model(payload):
    cfg = payload.get("config", {})
    if not isinstance(cfg, Mapping):
        cfg = {}
    spatial_dim = int(cfg.get("spatial_dim", 24))
    model_dim = int(cfg.get("model_dim", 112))
    dropout = float(cfg.get("dropout", 0.10))
    return tr.M4PhaseUniformT16(spatial_dim, model_dim, dropout), spatial_dim, model_dim, dropout


def make_functions(model, model_dim: int):
    @jax.jit
    def full_eval(params, xb):
        out = model.apply({"params": params}, xb, training=False)

        # Exact router-off diagnostic: learned frame-memory states are fed directly
        # through the already-trained descriptor/classifier heads. Fixed uniform
        # stream fusion is retained.
        ro_logits = []
        for i in range(4):
            _, desc = base.DescriptorHead(model_dim, 0.0).apply(
                {"params": params[f"descriptor_{i}"]},
                out["frame_stack"][:, :, i],
                training=False,
            )
            ro_logits.append(dense_apply(params[f"classifier_{i}"], desc))
        ro_sl = jnp.stack(ro_logits, axis=1)
        ro = jnp.mean(ro_sl, axis=1)

        return (
            out["logits"],
            out["stream_logits"],
            out["router_weights"],
            ro,
        )

    @jax.jit
    def logits_only(params, xb):
        return model.apply({"params": params}, xb, training=False)["logits"]

    @jax.jit
    def feature_eval(params, xb):
        out = model.apply({"params": params}, xb, training=False)
        spatial = out["spatial_stack"].mean(axis=1)
        frame = out["frame_stack"].mean(axis=1)
        mixed = out["mixed_frame_stack"].mean(axis=1)
        chunk = out["chunk_states"].mean(axis=2)
        desc = out["descriptors"]
        num = jnp.linalg.norm(out["mixed_frame_stack"] - out["frame_stack"], axis=-1)
        den = jnp.maximum(jnp.linalg.norm(out["frame_stack"], axis=-1), 1e-8)
        delta_ratio = jnp.mean(num / den, axis=(1, 2))
        return spatial, frame, mixed, chunk, desc, delta_ratio

    return full_eval, logits_only, feature_eval


def batched_full(fn, params, X, batch_size: int, desc: str):
    L, S, R, RO = [], [], [], []
    for s in tqdm(range(0, len(X), batch_size), desc=desc, mininterval=0.5):
        vals = fn(params, jnp.asarray(X[s:s + batch_size]))
        vals = [np.asarray(jax.device_get(v)) for v in vals]
        L.append(vals[0]); S.append(vals[1]); R.append(vals[2]); RO.append(vals[3])
    return tuple(np.concatenate(v, axis=0) for v in (L, S, R, RO))


def transform(xb, mode: str):
    z = xb.copy().reshape(len(xb), tr.FRAMES, tr.PERSONS, tr.JOINTS, tr.TOKEN_CHANNELS)
    if mode == "no_pose":
        z[..., 0:3] = 0
    elif mode == "no_full_disp":
        z[..., 3:6] = 0
    elif mode == "no_phase_a":
        z[..., 6:9] = 0
    elif mode == "no_phase_b":
        z[..., 9:12] = 0
    elif mode == "no_phase_ab":
        z[..., 6:12] = 0
    elif mode == "no_path":
        z[..., 12:15] = 0
    elif mode == "pose_only":
        z[..., 3:15] = 0
    elif mode == "motion_only":
        z[..., 0:3] = 0
    else:
        raise ValueError(mode)
    return z.reshape(len(xb), tr.FRAMES, tr.FEATURES)


def ablation_accuracy(fn, params, X, y, batch_size: int, mode: str):
    correct = total = 0
    for s in tqdm(range(0, len(X), batch_size), desc=f"ablation {mode}", leave=False, mininterval=0.5):
        xb = transform(X[s:s + batch_size], mode)
        logits = np.asarray(jax.device_get(fn(params, jnp.asarray(xb))))
        pred = logits.argmax(axis=-1)
        yy = y[s:s + len(pred)]
        correct += int(np.sum(pred == yy)); total += len(pred)
    return correct / max(total, 1)


def stream_counterfactuals(stream_logits, y):
    out = {}
    for i, name in enumerate(STREAMS):
        out[f"{name}_only"] = acc(y, stream_logits[:, i].argmax(axis=-1))

    def subset(indices):
        lg = stream_logits[:, indices].mean(axis=1)
        return acc(y, lg.argmax(axis=-1))

    out["pose_JB"] = subset([0, 1])
    out["motion_JM_BM"] = subset([2, 3])
    for i, name in enumerate(STREAMS):
        out[f"minus_{name}"] = subset([j for j in range(4) if j != i])
    return out


def extract_features(fn, params, X, batch_size: int, desc: str):
    names = ("spatial", "frame_memory", "after_router", "chunk_memory", "descriptor")
    bags = {k: [] for k in names}
    delta = []
    for s in tqdm(range(0, len(X), batch_size), desc=desc, leave=False, mininterval=0.5):
        vals = fn(params, jnp.asarray(X[s:s + batch_size]))
        vals = [np.asarray(jax.device_get(v)) for v in vals]
        for k, v in zip(names, vals[:5]):
            bags[k].append(v.astype(np.float32, copy=False))
        delta.append(vals[5].astype(np.float32, copy=False))
    return {k: np.concatenate(v, axis=0) for k, v in bags.items()}, np.concatenate(delta)


def flatten_stage(x):
    return np.asarray(x, np.float32).reshape(len(x), -1)


def ridge_probe(Xtr, ytr, Xva, yva):
    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    clf.fit(flatten_stage(Xtr), ytr)
    return float(clf.score(flatten_stage(Xva), yva))


def activation_spectrum(x, max_samples: int, seed: int):
    X = flatten_stage(x)
    if len(X) > max_samples:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), max_samples, replace=False)]
    X = X.astype(np.float64, copy=False)
    X -= X.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(X, compute_uv=False)
    except np.linalg.LinAlgError:
        return {"effective_rank": float("nan"), "rank_fraction": float("nan"), "stable_rank": float("nan")}
    lam = s * s
    p = lam / max(float(lam.sum()), EPS)
    eff = float(np.exp(-np.sum(p * np.log(np.maximum(p, EPS)))))
    stable = float(lam.sum() / max(float(lam[0]), EPS))
    max_rank = float(min(X.shape[0] - 1, X.shape[1]))
    return {"effective_rank": eff, "rank_fraction": eff / max(max_rank, 1.0), "stable_rank": stable}


def class_rows(y, pred_can, pred_jit):
    rows = []
    for c in range(tr.NUM_CLASSES):
        m = y == c
        if not np.any(m):
            continue
        ac = float(np.mean(pred_can[m] == c))
        aj = float(np.mean(pred_jit[m] == c))
        agree = float(np.mean(pred_can[m] == pred_jit[m]))
        rows.append({"class": c, "n": int(m.sum()), "canonical_accuracy": ac,
                     "jitter_accuracy": aj, "jitter_drop_pp": 100.0 * (ac - aj),
                     "prediction_agreement": agree})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["xsub", "xset"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--probe-train", type=int, default=12000)
    ap.add_argument("--probe-val", type=int, default=8000)
    ap.add_argument("--rank-samples", type=int, default=5000)
    ap.add_argument("--jitter-max-shift", type=int, default=1)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    payload = serialization.msgpack_restore(ckpt.read_bytes())
    params = payload.get("ema_params")
    if params is None:
        raise KeyError("Checkpoint has no ema_params")

    model, spatial_dim, model_dim, dropout = build_model(payload)
    nparams = tr.count_params(params)

    print("=" * 122)
    print("NESTSAR M4-PHASE-JITTER-UNIFORM-T16 — ACTIVATION / ROBUSTNESS AUDIT")
    print("=" * 122)
    print(f"protocol              : {args.protocol}")
    print(f"checkpoint epoch      : {int(np.asarray(payload.get('epoch', -1)))}")
    print(f"checkpoint val_acc    : {100*float(np.asarray(payload.get('val_accuracy', float('nan')))):.3f}%")
    print(f"EMA params            : {nparams:,}")
    print(f"model dims            : spatial={spatial_dim} model={model_dim} dropout={dropout}")
    print(f"final fusion          : fixed uniform mean")

    dataset = base.find_dataset(args.dataset)
    print(f"dataset               : {dataset}")
    annotations, split = base.load_ntu(dataset)
    by_id, train_ids, val_ids = resolve_ids(annotations, split, args.protocol)

    Xcan, yva = materialize_canonical(by_id, val_ids, f"{args.protocol.upper()} val canonical")
    Xjit, yjit = materialize_jitter(by_id, val_ids, args.jitter_max_shift, args.seed,
                                    f"{args.protocol.upper()} val jitter +/-{args.jitter_max_shift}")
    if not np.array_equal(yva, yjit):
        raise RuntimeError("Canonical/jitter labels differ")
    changed = float(np.mean(np.any(np.abs(Xcan - Xjit) > 1e-7, axis=(1, 2))))

    full_fn, logits_fn, feat_fn = make_functions(model, model_dim)
    can_logits, can_sl, can_router, router_off = batched_full(
        full_fn, params, Xcan, args.batch_size, "canonical validation inference")
    jit_logits, _, jit_router, _ = batched_full(
        full_fn, params, Xjit, args.batch_size, "jitter validation inference")

    pred_can = can_logits.argmax(axis=-1)
    pred_jit = jit_logits.argmax(axis=-1)
    pred_ro = router_off.argmax(axis=-1)
    can_acc = acc(yva, pred_can)
    jit_acc = acc(yva, pred_jit)
    ro_acc = acc(yva, pred_ro)
    agreement = float(np.mean(pred_can == pred_jit))

    print("\nCANONICAL / JITTER ROBUSTNESS")
    print(f"  canonical EMA accuracy : {100*can_acc:.3f}%")
    print(f"  jittered accuracy      : {100*jit_acc:.3f}%")
    print(f"  jitter robustness drop : {100*(can_acc-jit_acc):+.3f} pp")
    print(f"  prediction agreement   : {100*agreement:.3f}%")
    print(f"  jitter view changed    : {100*changed:.2f}% of validation samples")
    print(f"  router-OFF accuracy    : {100*ro_acc:.3f}%")
    print(f"  router contribution    : {100*(can_acc-ro_acc):+.3f} pp")

    streams = stream_counterfactuals(can_sl, yva)
    print("\nSTREAM COUNTERFACTUALS — FIXED UNIFORM FUSION")
    for k, v in streams.items():
        print(f"  {k:16s}: {100*v:7.3f}% ({100*(v-can_acc):+7.3f} pp)")

    ab_modes = ["no_pose", "no_full_disp", "no_phase_a", "no_phase_b", "no_phase_ab",
                "no_path", "pose_only", "motion_only"]
    ablations = {}
    print("\nINPUT ABLATIONS")
    for mode in ab_modes:
        v = ablation_accuracy(logits_fn, params, Xcan, yva, args.batch_size, mode)
        ablations[mode] = v
        print(f"  {mode:16s}: {100*v:7.3f}% ({100*(v-can_acc):+7.3f} pp)")

    # Probe data: canonical only, because benchmark representation is canonical.
    tr_ids = subset_ids(train_ids, args.probe_train, args.seed + 11)
    va_ids = subset_ids(val_ids, args.probe_val, args.seed + 23)
    Xptr, yptr = materialize_canonical(by_id, tr_ids, f"{args.protocol.upper()} probe train")
    Xpva, ypva = materialize_canonical(by_id, va_ids, f"{args.protocol.upper()} probe val")
    Ftr, _ = extract_features(feat_fn, params, Xptr, args.batch_size, "extract probe train activations")
    Fva, delta = extract_features(feat_fn, params, Xpva, args.batch_size, "extract probe val activations")

    probes = {}
    print("\nFROZEN RIDGE LINEAR PROBES")
    for name in ("spatial", "frame_memory", "after_router", "chunk_memory", "descriptor"):
        probes[name] = ridge_probe(Ftr[name], yptr, Fva[name], ypva)
        print(f"  {name:14s}: {100*probes[name]:7.3f}%")

    stage_names = ["spatial", "frame_memory", "after_router", "chunk_memory", "descriptor"]
    stage_gains = {}
    print("\nSTAGE GAINS")
    for a, b in zip(stage_names[:-1], stage_names[1:]):
        k = f"{a}_to_{b}_pp"
        stage_gains[k] = 100.0 * (probes[b] - probes[a])
        print(f"  {k:31s}: {stage_gains[k]:+7.3f} pp")
    mean_delta = float(np.mean(delta))
    print(f"  mean router activation delta ratio: {100*mean_delta:.3f}%")

    ranks = {}
    print("\nACTIVATION EFFECTIVE RANK")
    for i, name in enumerate(stage_names):
        ranks[name] = activation_spectrum(Fva[name], args.rank_samples, args.seed + i)
        r = ranks[name]
        print(f"  {name:14s}: eff_rank={r['effective_rank']:.2f} rank_frac={r['rank_fraction']:.4f} stable={r['stable_rank']:.2f}")

    router_mean_can = can_router.mean(axis=(0, 1))
    router_mean_jit = jit_router.mean(axis=(0, 1))
    router_entropy_can = float(np.mean(entropy_np(can_router, normalized=True)))
    router_entropy_jit = float(np.mean(entropy_np(jit_router, normalized=True)))
    print("\nROUTER ROBUSTNESS")
    print("  canonical mean :", {STREAMS[i]: round(float(router_mean_can[i]), 5) for i in range(4)})
    print("  jitter mean    :", {STREAMS[i]: round(float(router_mean_jit[i]), 5) for i in range(4)})
    print(f"  canonical entropy normalized: {router_entropy_can:.4f}")
    print(f"  jitter entropy normalized   : {router_entropy_jit:.4f}")

    classes = class_rows(yva, pred_can, pred_jit)
    hard = sorted(classes, key=lambda r: r["canonical_accuracy"])[:15]
    sensitive = sorted(classes, key=lambda r: r["jitter_drop_pp"], reverse=True)[:15]
    print("\nHARDEST CANONICAL CLASSES")
    for r in hard:
        print(f"  class={r['class']:3d} acc={100*r['canonical_accuracy']:6.2f}% jitter={100*r['jitter_accuracy']:6.2f}% n={r['n']:4d}")
    print("\nMOST JITTER-SENSITIVE CLASSES")
    for r in sensitive:
        print(f"  class={r['class']:3d} canonical={100*r['canonical_accuracy']:6.2f}% jitter={100*r['jitter_accuracy']:6.2f}% drop={r['jitter_drop_pp']:+6.2f}pp agree={100*r['prediction_agreement']:6.2f}%")

    weakest = min(stage_gains, key=stage_gains.get)
    strongest = max(stage_gains, key=stage_gains.get)
    damaging = min(ablations, key=ablations.get)

    print("\n" + "=" * 122)
    print("BOTTLENECK / GENERALIZATION SUMMARY")
    print("=" * 122)
    print(f"CANONICAL ACCURACY          : {100*can_acc:.3f}%")
    print(f"JITTERED ACCURACY           : {100*jit_acc:.3f}%")
    print(f"JITTER ROBUSTNESS DROP      : {100*(can_acc-jit_acc):+.3f} pp")
    print(f"CANONICAL↔JITTER AGREEMENT  : {100*agreement:.3f}%")
    print(f"ROUTER CONTRIBUTION         : {100*(can_acc-ro_acc):+.3f} pp")
    print(f"WEAKEST STAGE GAIN          : {weakest} = {stage_gains[weakest]:+.3f} pp")
    print(f"STRONGEST STAGE GAIN        : {strongest} = {stage_gains[strongest]:+.3f} pp")
    print(f"REMOVE BOTH PHASE CHANNELS  : {100*ablations['no_phase_ab']:.3f}% ({100*(ablations['no_phase_ab']-can_acc):+.3f} pp)")
    print(f"REMOVE PATH                 : {100*ablations['no_path']:.3f}% ({100*(ablations['no_path']-can_acc):+.3f} pp)")
    print(f"MOST DAMAGING ABLATION      : {damaging} -> {100*ablations[damaging]:.3f}%")
    print(f"MEAN ROUTER DELTA RATIO     : {100*mean_delta:.3f}%")

    report = {
        "protocol": args.protocol,
        "checkpoint": str(ckpt),
        "checkpoint_epoch": int(np.asarray(payload.get("epoch", -1))),
        "checkpoint_val_accuracy": float(np.asarray(payload.get("val_accuracy", float("nan")))),
        "params": int(nparams),
        "canonical_accuracy": can_acc,
        "jitter_accuracy": jit_acc,
        "jitter_robustness_drop_pp": 100.0 * (can_acc - jit_acc),
        "prediction_agreement": agreement,
        "jitter_view_changed_fraction": changed,
        "router_off_accuracy": ro_acc,
        "router_contribution_pp": 100.0 * (can_acc - ro_acc),
        "stream_counterfactuals": streams,
        "input_ablations": ablations,
        "linear_probes": probes,
        "stage_gains_pp": stage_gains,
        "mean_router_delta_ratio": mean_delta,
        "activation_rank": ranks,
        "router_mean_canonical": router_mean_can.tolist(),
        "router_mean_jitter": router_mean_jit.tolist(),
        "router_entropy_canonical": router_entropy_can,
        "router_entropy_jitter": router_entropy_jit,
        "classes": classes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"REPORT: {out}")
    print("=" * 122)


if __name__ == "__main__":
    main()

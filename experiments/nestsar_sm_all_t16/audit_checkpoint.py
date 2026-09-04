#!/usr/bin/env python3
from __future__ import annotations

"""Post-training audit for NestSAR-SM-ALL-T16.

Answers, for XSUB and XSET:
  1) exact XLA FLOPs/GFLOPs and parameter count;
  2) learned eta distribution;
  3) learned alpha distribution;
  4) scaled self-modifying residual magnitude relative to M4/G4 base memory;
  5) adaptive fusion variability;
  6) adaptive-head magnitude, prediction flips, and accuracy contribution;
  7) full-validation accuracy;
  8) per-class improvement/regression versus Hand_M4G4_T32 champion.

Validation preprocessing is canonical only. Raw NTU clip length remains variable,
but both models consume their fixed processing views exactly as used in training.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization, traverse_util
from tqdm.auto import tqdm

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
from experiments.nestsar_sm_all_t16.model import (
    FEATURES,
    FRAMES,
    NUM_CLASSES,
    FastWeightDeltaResidual,
    SharedSMController,
    NestSARSMAllT16,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--sm-outdir",
        default="/kaggle/working/NestSAR_SM_ALL_T16_v1_DualT4",
    )
    p.add_argument(
        "--baseline-outdir",
        default="/kaggle/working/NestSAR_M4_LocalGlobal_HandM4G4Lite_T32_DualT4",
    )
    p.add_argument(
        "--audit-outdir",
        default="/kaggle/working/NestSAR_SM_ALL_T16_v1_DualT4/audit",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--internal-audit-samples",
        type=int,
        default=2048,
        help="Number of validation samples used for expensive intermediate capture.",
    )
    p.add_argument("--progress-every", type=int, default=20)
    return p.parse_args()


def load_payload(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = serialization.msgpack_restore(path.read_bytes())
    if not isinstance(obj, Mapping):
        raise TypeError(f"Checkpoint {path} did not restore to a mapping")
    if "ema_params" not in obj:
        raise KeyError(f"Checkpoint {path} has no ema_params")
    return obj


def model_from_sm_payload(payload: Mapping[str, Any]) -> NestSARSMAllT16:
    c = dict(payload.get("config", {}))
    return NestSARSMAllT16(
        spatial_dim=int(c.get("spatial_dim", 24)),
        model_dim=int(c.get("model_dim", 112)),
        dropout=float(c.get("dropout", 0.10)),
        controller_dim=int(c.get("controller_dim", 16)),
        fast_rank=int(c.get("fast_rank", 2)),
        head_rank=int(c.get("head_rank", 2)),
        sm_residual_scale=float(c.get("sm_residual_scale", 0.08)),
        head_residual_scale=float(c.get("head_residual_scale", 0.15)),
    )


def model_from_baseline_payload(payload: Mapping[str, Any]) -> M4LocalGlobalHandM4G4T32:
    c = dict(payload.get("config", {}))
    return M4LocalGlobalHandM4G4T32(
        spatial_dim=int(c.get("spatial_dim", 24)),
        model_dim=int(c.get("model_dim", 112)),
        dropout=float(c.get("dropout", 0.10)),
        hand_dim=int(c.get("hand_dim", 32)),
        hand_residual_scale=float(c.get("hand_residual_scale", 0.10)),
    )


def count_params(params) -> int:
    return int(sum(np.prod(np.asarray(x).shape) for x in jax.tree_util.tree_leaves(params)))


def xla_flops(model, params) -> float:
    dummy = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    fn = jax.jit(
        lambda p, x: model.apply({"params": p}, x, training=False)["logits"]
    )
    compiled = fn.lower(params, dummy).compile()
    ca = compiled.cost_analysis()
    if isinstance(ca, list):
        ca = ca[0] if ca else {}
    return float(ca.get("flops", float("nan")))


def resolve_val(annotations, split, protocol: str):
    by_id, _, val_ids = ju.resolve_protocol_ids(annotations, split, protocol)
    return by_id, val_ids


def iter_validation_batches(by_id, val_ids, batch_size: int):
    """Canonical validation views for SM-T16 and Hand-T32 baseline."""
    for start in range(0, len(val_ids), batch_size):
        ids = val_ids[start : start + batch_size]
        n = len(ids)

        x = np.zeros((batch_size, FRAMES, FEATURES), np.float32)
        h = np.zeros((batch_size, HAND_FRAMES, HAND_FEATURES), np.float32)
        y = np.zeros((batch_size,), np.int32)
        mask = np.zeros((batch_size,), np.float32)

        for j, sid in enumerate(ids):
            a = by_id[sid]
            kp = base.annotation_keypoints(a)
            x[j] = lg.segment_phase_tokens_localglobal(kp)
            h[j] = hand_tokens_t32(kp)
            y[j] = base.annotation_label(a)
            mask[j] = 1.0

        yield x, h, y, mask, n


def safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return num / np.maximum(den, eps)


class RunningScalar:
    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.ss = 0.0
        self.mn = float("inf")
        self.mx = -float("inf")

    def add(self, values):
        a = np.asarray(values, np.float64).reshape(-1)
        a = a[np.isfinite(a)]
        if not len(a):
            return
        self.n += len(a)
        self.s += float(np.sum(a))
        self.ss += float(np.sum(a * a))
        self.mn = min(self.mn, float(np.min(a)))
        self.mx = max(self.mx, float(np.max(a)))

    def result(self):
        if self.n == 0:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.s / self.n
        var = max(0.0, self.ss / self.n - mean * mean)
        return {
            "n": self.n,
            "mean": mean,
            "std": math.sqrt(var),
            "min": self.mn,
            "max": self.mx,
        }


def flatten_intermediates(intermediates):
    if "intermediates" not in intermediates:
        return {}
    flat = traverse_util.flatten_dict(intermediates["intermediates"], sep="/")
    out = {}
    for path, value in flat.items():
        # Flax stores captured __call__ values as a tuple/list of invocations.
        if isinstance(value, (tuple, list)) and value:
            value = value[-1]
        out[path] = value
    return out


def capture_filter(module, method_name: str) -> bool:
    return method_name == "__call__" and isinstance(
        module,
        (FastWeightDeltaResidual, SharedSMController),
    )


def extract_internal_stats(
    sm_model,
    sm_params,
    xb: np.ndarray,
    valid_n: int,
    sm_residual_scale: float,
):
    """Capture fast-weight residuals and controller eta/alpha on a small subset."""
    out, mut = sm_model.apply(
        {"params": sm_params},
        jnp.asarray(xb),
        training=False,
        capture_intermediates=capture_filter,
        mutable=["intermediates"],
    )
    del out
    flat = flatten_intermediates(mut)

    result = {
        "m4_scaled_delta_over_input": [],
        "g4_scaled_delta_over_input": [],
        "eta_full": [],
        "alpha_full": [],
    }

    # FastWeightDeltaResidual output is the raw read/delta sequence. To measure
    # its effective contribution, compare residual_scale*delta against the
    # input to that fast-weight module. The module input is the base BiMemory
    # output, which can be reconstructed by capture through its scope using the
    # nearest available final model tensors for robust reporting. For exact
    # relative magnitude we additionally derive norms from the fast read and
    # use corresponding M4/G4 state tensors returned by the model in the normal
    # forward pass in the caller.
    for path, value in flat.items():
        if path.endswith("sm_controller/__call__") and isinstance(value, Mapping):
            if "eta" in value:
                result["eta_full"].append(np.asarray(value["eta"])[:valid_n])
            if "alpha" in value:
                result["alpha_full"].append(np.asarray(value["alpha"])[:valid_n])

    return result, flat


def find_fast_reads(flat, valid_n: int):
    m4 = []
    g4 = []
    for path, value in flat.items():
        if not path.endswith("fast_weight/__call__"):
            continue
        arr = np.asarray(value)[:valid_n]
        if "frame_memory_" in path:
            m4.append(arr)
        elif "descriptor_" in path and "chunk_memory" in path:
            g4.append(arr)
    return m4, g4


def array_norm_per_sample(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return np.sqrt(np.sum(x * x, axis=tuple(range(1, x.ndim))))


def summarize_fusion(fusion_values: np.ndarray) -> dict:
    f = np.asarray(fusion_values, np.float64)
    uniform = 1.0 / f.shape[1]
    ent = -np.sum(f * np.log(np.maximum(f, 1e-12)), axis=1)
    ent_norm = ent / math.log(f.shape[1])
    return {
        "mean_per_stream": np.mean(f, axis=0).tolist(),
        "std_per_stream": np.std(f, axis=0).tolist(),
        "mean_abs_deviation_from_uniform": float(np.mean(np.abs(f - uniform))),
        "max_abs_deviation_from_uniform": float(np.max(np.abs(f - uniform))),
        "mean_normalized_entropy": float(np.mean(ent_norm)),
        "std_normalized_entropy": float(np.std(ent_norm)),
    }


def per_class_rows(
    labels: np.ndarray,
    pred_sm: np.ndarray,
    pred_base: np.ndarray,
):
    rows = []
    for c in range(NUM_CLASSES):
        mask = labels == c
        support = int(np.sum(mask))
        if support == 0:
            continue
        sm_acc = float(np.mean(pred_sm[mask] == c))
        base_acc = float(np.mean(pred_base[mask] == c))
        rows.append(
            {
                "class_index_zero_based": c,
                "action_id_one_based": c + 1,
                "support": support,
                "sm_accuracy": sm_acc,
                "baseline_accuracy": base_acc,
                "delta_pp": 100.0 * (sm_acc - base_acc),
            }
        )
    return rows


def print_top_class_deltas(rows, n: int = 15):
    improved = sorted(rows, key=lambda r: r["delta_pp"], reverse=True)
    worse = sorted(rows, key=lambda r: r["delta_pp"])

    print("\nTOP IMPROVED CLASSES (SM vs Hand champion)")
    print("action | support | baseline | SM | delta pp")
    for r in improved[:n]:
        print(
            f"A{r['action_id_one_based']:03d} | {r['support']:4d} | "
            f"{100*r['baseline_accuracy']:7.2f}% | {100*r['sm_accuracy']:7.2f}% | "
            f"{r['delta_pp']:+8.3f}"
        )

    print("\nTOP REGRESSED CLASSES (SM vs Hand champion)")
    print("action | support | baseline | SM | delta pp")
    for r in worse[:n]:
        print(
            f"A{r['action_id_one_based']:03d} | {r['support']:4d} | "
            f"{100*r['baseline_accuracy']:7.2f}% | {100*r['sm_accuracy']:7.2f}% | "
            f"{r['delta_pp']:+8.3f}"
        )


def audit_protocol(args, annotations, split, protocol: str):
    sm_ckpt = Path(args.sm_outdir) / protocol / "best.msgpack"
    base_ckpt = Path(args.baseline_outdir) / protocol / "best.msgpack"

    sm_payload = load_payload(sm_ckpt)
    base_payload = load_payload(base_ckpt)

    sm_model = model_from_sm_payload(sm_payload)
    base_model = model_from_baseline_payload(base_payload)
    sm_params = sm_payload["ema_params"]
    base_params = base_payload["ema_params"]

    sm_cfg = dict(sm_payload.get("config", {}))
    sm_residual_scale = float(sm_cfg.get("sm_residual_scale", 0.08))

    print("\n" + "=" * 118)
    print(f"{protocol.upper()} CHECKPOINT AUDIT")
    print("=" * 118)

    nparams = count_params(sm_params)
    flops = xla_flops(sm_model, sm_params)
    gflops = flops / 1e9
    print(f"1) PARAMS       : {nparams:,}")
    print(f"   XLA FLOPs    : {flops:,.0f}")
    print(f"   XLA GFLOPs   : {gflops:.9f}")
    print(f"   XLA MFLOPs   : {flops/1e6:.6f}")

    sm_apply = jax.jit(
        lambda p, x: sm_model.apply({"params": p}, x, training=False)
    )
    base_apply = jax.jit(
        lambda p, x, h: base_model.apply(
            {"params": p}, x, h, training=False
        )
    )

    by_id, val_ids = resolve_val(annotations, split, protocol)
    total_batches = (len(val_ids) + args.batch_size - 1) // args.batch_size

    labels_all = []
    sm_pred_all = []
    base_pred_all = []

    eta_stat = RunningScalar()
    alpha_stat = RunningScalar()
    head_ratio_stat = RunningScalar()
    head_abs_stat = RunningScalar()
    main_abs_stat = RunningScalar()

    fusion_chunks = []

    sm_correct = 0
    sm_main_correct = 0
    base_correct = 0
    total = 0
    head_flip = 0

    # Internal fast-weight ratios use the first N validation samples only.
    internal_target = max(0, int(args.internal_audit_samples))
    internal_seen = 0
    m4_delta_norm = RunningScalar()
    g4_delta_norm = RunningScalar()
    m4_state_norm = RunningScalar()
    g4_state_norm = RunningScalar()
    m4_ratio = RunningScalar()
    g4_ratio = RunningScalar()
    eta_full_stat = RunningScalar()
    alpha_full_stat = RunningScalar()

    bar = tqdm(
        total=total_batches,
        desc=f"{protocol.upper()} AUDIT",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5,
    )

    for batch_i, (xb, hb, yb, mask, n) in enumerate(
        iter_validation_batches(by_id, val_ids, args.batch_size),
        start=1,
    ):
        sm_out = jax.device_get(sm_apply(sm_params, jnp.asarray(xb)))
        base_out = jax.device_get(
            base_apply(base_params, jnp.asarray(xb), jnp.asarray(hb))
        )

        y = yb[:n]
        sm_logits = np.asarray(sm_out["logits"])[:n]
        sm_main = np.asarray(sm_out["main_logits"])[:n]
        head_delta = np.asarray(sm_out["adaptive_head_delta"])[:n]
        base_logits = np.asarray(base_out["logits"])[:n]

        p_sm = np.argmax(sm_logits, axis=-1)
        p_main = np.argmax(sm_main, axis=-1)
        p_base = np.argmax(base_logits, axis=-1)

        labels_all.append(y.copy())
        sm_pred_all.append(p_sm)
        base_pred_all.append(p_base)

        sm_correct += int(np.sum(p_sm == y))
        sm_main_correct += int(np.sum(p_main == y))
        base_correct += int(np.sum(p_base == y))
        total += n
        head_flip += int(np.sum(p_sm != p_main))

        eta_stat.add(np.asarray(sm_out["sm_eta_mean"])[:n])
        alpha_stat.add(np.asarray(sm_out["sm_alpha_mean"])[:n])

        head_norm = array_norm_per_sample(head_delta)
        main_norm = array_norm_per_sample(sm_main)
        # Effective correction includes configured residual scale.
        head_scale = float(sm_cfg.get("head_residual_scale", 0.15))
        effective_head_norm = head_scale * head_norm
        head_abs_stat.add(effective_head_norm)
        main_abs_stat.add(main_norm)
        head_ratio_stat.add(safe_ratio(effective_head_norm, main_norm))

        fusion_chunks.append(np.asarray(sm_out["fusion_weights"])[:n])

        if internal_seen < internal_target:
            take = min(n, internal_target - internal_seen)
            xdiag = xb.copy()
            if take < len(xdiag):
                xdiag[take:] = 0.0

            normal_diag, mut = sm_model.apply(
                {"params": sm_params},
                jnp.asarray(xdiag),
                training=False,
                capture_intermediates=capture_filter,
                mutable=["intermediates"],
            )
            normal_diag = jax.device_get(normal_diag)
            flat = flatten_intermediates(jax.device_get(mut))

            # Full eta/alpha values from the controller, not just clip means.
            for path, value in flat.items():
                if path.endswith("sm_controller/__call__") and isinstance(value, Mapping):
                    if "eta" in value:
                        eta_full_stat.add(np.asarray(value["eta"])[:take])
                    if "alpha" in value:
                        alpha_full_stat.add(np.asarray(value["alpha"])[:take])

            m4_reads, g4_reads = find_fast_reads(flat, take)

            # The model returns post-self-mod M4/G4 states. This gives a stable
            # denominator for the effective residual-to-state ratio. Because the
            # residual scale is small, post-SM state and pre-SM base state are
            # near-identical for this diagnostic purpose; we report the exact
            # definition in the JSON so it is not mislabeled.
            frame_states = np.asarray(normal_diag["frame_stack"])[:take]
            chunk_states = np.asarray(normal_diag["chunk_states"])[:take]

            for i, read in enumerate(m4_reads):
                read_norm = array_norm_per_sample(read)
                # frame_stack [B,T,S,D]
                stream_i = min(i, frame_states.shape[2] - 1)
                state_norm = array_norm_per_sample(frame_states[:, :, stream_i, :])
                effective = sm_residual_scale * read_norm
                m4_delta_norm.add(effective)
                m4_state_norm.add(state_norm)
                m4_ratio.add(safe_ratio(effective, state_norm))

            for i, read in enumerate(g4_reads):
                read_norm = array_norm_per_sample(read)
                # chunk_states [B,S,4,D]
                stream_i = min(i, chunk_states.shape[1] - 1)
                state_norm = array_norm_per_sample(chunk_states[:, stream_i, :, :])
                effective = sm_residual_scale * read_norm
                g4_delta_norm.add(effective)
                g4_state_norm.add(state_norm)
                g4_ratio.add(safe_ratio(effective, state_norm))

            internal_seen += take

        if batch_i % args.progress_every == 0 or batch_i == total_batches:
            bar.set_postfix_str(
                f"SM={100*sm_correct/max(total,1):.2f}% "
                f"BASE={100*base_correct/max(total,1):.2f}%",
                refresh=True,
            )
        bar.update(1)

    bar.close()

    labels = np.concatenate(labels_all)
    pred_sm = np.concatenate(sm_pred_all)
    pred_base = np.concatenate(base_pred_all)
    fusion = np.concatenate(fusion_chunks, axis=0)

    class_rows = per_class_rows(labels, pred_sm, pred_base)
    n_improved = sum(r["delta_pp"] > 1e-9 for r in class_rows)
    n_worse = sum(r["delta_pp"] < -1e-9 for r in class_rows)
    n_tied = len(class_rows) - n_improved - n_worse

    sm_acc = sm_correct / max(total, 1)
    sm_main_acc = sm_main_correct / max(total, 1)
    base_acc = base_correct / max(total, 1)

    eta_summary = eta_full_stat.result() if eta_full_stat.n else eta_stat.result()
    alpha_summary = alpha_full_stat.result() if alpha_full_stat.n else alpha_stat.result()
    fusion_summary = summarize_fusion(fusion)

    head_summary = {
        "effective_delta_norm": head_abs_stat.result(),
        "main_logit_norm": main_abs_stat.result(),
        "effective_delta_over_main_norm": head_ratio_stat.result(),
        "prediction_flip_fraction": head_flip / max(total, 1),
        "main_only_accuracy": sm_main_acc,
        "final_accuracy": sm_acc,
        "head_gain_pp": 100.0 * (sm_acc - sm_main_acc),
    }

    fast_summary = {
        "definition": "effective residual norm = sm_residual_scale * ||FastWeightDeltaResidual read||; denominator = ||post-SM returned M4/G4 state||",
        "internal_samples": internal_seen,
        "sm_residual_scale": sm_residual_scale,
        "m4_effective_delta_norm": m4_delta_norm.result(),
        "m4_state_norm": m4_state_norm.result(),
        "m4_effective_delta_over_state": m4_ratio.result(),
        "g4_effective_delta_norm": g4_delta_norm.result(),
        "g4_state_norm": g4_state_norm.result(),
        "g4_effective_delta_over_state": g4_ratio.result(),
    }

    print("\n2) ETA")
    print(json.dumps(eta_summary, indent=2))
    print("\n3) ALPHA")
    print(json.dumps(alpha_summary, indent=2))

    print("\n4) SELF-MODIFYING RESIDUAL SIZE")
    print(
        f"   M4 effective Δ/state mean: "
        f"{100*(fast_summary['m4_effective_delta_over_state']['mean'] or 0):.4f}%"
    )
    print(
        f"   G4 effective Δ/state mean: "
        f"{100*(fast_summary['g4_effective_delta_over_state']['mean'] or 0):.4f}%"
    )

    print("\n5) ADAPTIVE FUSION")
    print("   mean weights:", [round(v, 6) for v in fusion_summary["mean_per_stream"]])
    print("   std weights :", [round(v, 6) for v in fusion_summary["std_per_stream"]])
    print(
        f"   mean |w-0.25|={fusion_summary['mean_abs_deviation_from_uniform']:.6f} | "
        f"normalized entropy={fusion_summary['mean_normalized_entropy']:.6f}"
    )

    print("\n6) ADAPTIVE CLASSIFIER HEAD")
    print(
        f"   effective Δlogit/main norm mean: "
        f"{100*(head_summary['effective_delta_over_main_norm']['mean'] or 0):.4f}%"
    )
    print(f"   prediction flips: {100*head_summary['prediction_flip_fraction']:.4f}%")
    print(f"   main-only acc    : {100*sm_main_acc:.6f}%")
    print(f"   final SM acc     : {100*sm_acc:.6f}%")
    print(f"   head gain        : {head_summary['head_gain_pp']:+.6f} pp")

    print("\n7) FULL VALIDATION")
    print(f"   SM-ALL-T16       : {100*sm_acc:.6f}%")
    print(f"   Hand champion    : {100*base_acc:.6f}%")
    print(f"   delta            : {100*(sm_acc-base_acc):+.6f} pp")

    print("\n8) CLASS-LEVEL CHANGE")
    print(f"   improved={n_improved} | regressed={n_worse} | tied={n_tied}")
    print_top_class_deltas(class_rows, n=15)

    result = {
        "protocol": protocol,
        "checkpoint_epoch": int(sm_payload.get("epoch", 0)),
        "params": nparams,
        "flops": flops,
        "mflops": flops / 1e6,
        "gflops": gflops,
        "eta": eta_summary,
        "alpha": alpha_summary,
        "fast_weight": fast_summary,
        "fusion": fusion_summary,
        "adaptive_head": head_summary,
        "accuracy": {
            "sm_all_t16": sm_acc,
            "sm_main_only": sm_main_acc,
            "hand_champion": base_acc,
            "delta_vs_hand_champion_pp": 100.0 * (sm_acc - base_acc),
        },
        "classes": {
            "improved": n_improved,
            "regressed": n_worse,
            "tied": n_tied,
        },
    }

    outdir = Path(args.audit_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"audit_{protocol}.json").write_text(json.dumps(result, indent=2))

    with (outdir / f"class_delta_{protocol}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(class_rows[0].keys()))
        writer.writeheader()
        writer.writerows(class_rows)

    return result


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)

    print("=" * 118)
    print("NESTSAR-SM-ALL-T16 — POST-TRAINING AUDIT")
    print("Answers: compute | eta/alpha | M4/G4 ΔSM | fusion | adaptive head | per-class deltas")
    print("JAX", jax.__version__, "BACKEND", jax.default_backend(), "DEVICES", jax.devices())
    print("=" * 118)

    annotations, split = base.load_ntu(dataset)

    results = {}
    t0 = time.time()
    for protocol in ("xsub", "xset"):
        results[protocol] = audit_protocol(
            args,
            annotations,
            split,
            protocol,
        )

    mean_sm = 0.5 * (
        results["xsub"]["accuracy"]["sm_all_t16"]
        + results["xset"]["accuracy"]["sm_all_t16"]
    )
    mean_base = 0.5 * (
        results["xsub"]["accuracy"]["hand_champion"]
        + results["xset"]["accuracy"]["hand_champion"]
    )

    summary = {
        "mean_sm_accuracy": mean_sm,
        "mean_hand_champion_accuracy": mean_base,
        "mean_delta_pp": 100.0 * (mean_sm - mean_base),
        "elapsed_seconds": time.time() - t0,
        "xsub": results["xsub"],
        "xset": results["xset"],
    }

    outdir = Path(args.audit_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "audit_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 118)
    print("FINAL AUDIT SUMMARY")
    print("=" * 118)
    print(f"SM mean            : {100*mean_sm:.6f}%")
    print(f"Hand champion mean : {100*mean_base:.6f}%")
    print(f"Mean delta         : {100*(mean_sm-mean_base):+.6f} pp")
    print(f"Saved              : {outdir}")
    print("=" * 118)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from flax import serialization
from flax.traverse_util import flatten_dict

STREAM_NAMES = {0: "J", 1: "B", 2: "JM", 3: "BM"}


def arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def flatten_params(tree: Any) -> dict[str, np.ndarray]:
    flat = flatten_dict(tree, sep="/")
    return {str(k): arr(v) for k, v in flat.items()}


def group_of(path: str) -> str:
    top = path.split("/", 1)[0]
    if top.startswith("spatial_"):
        return top
    if top.startswith("frame_memory_"):
        return top
    if top.startswith("descriptor_"):
        return top
    if top.startswith("classifier_"):
        return top
    if top.startswith("cross_stream_after_frame"):
        return "cross_stream_after_frame"
    if top.startswith("fusion_controller"):
        return "fusion_controller"
    if top.startswith("fusion_prior"):
        return "fusion_prior"
    return top


def stream_family(group: str) -> tuple[str, int] | None:
    for prefix in ("spatial_", "frame_memory_", "descriptor_", "classifier_"):
        if group.startswith(prefix):
            try:
                return prefix[:-1], int(group[len(prefix):])
            except ValueError:
                return None
    return None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def tensor_stats(x: np.ndarray) -> dict[str, float | int | list[int] | None]:
    finite = np.isfinite(x)
    xf = x[finite]
    out: dict[str, float | int | list[int] | None] = {
        "shape": list(x.shape),
        "n": int(x.size),
        "nan": int(np.isnan(x).sum()),
        "inf": int(np.isinf(x).sum()),
    }
    if xf.size == 0:
        out.update({"mean": None, "std": None, "rms": None, "l2": None,
                    "max_abs": None, "near_zero_pct": None})
        return out
    out.update({
        "mean": float(xf.mean()),
        "std": float(xf.std()),
        "rms": float(np.sqrt(np.mean(xf * xf))),
        "l2": float(np.linalg.norm(xf)),
        "max_abs": float(np.max(np.abs(xf))),
        "near_zero_pct": float(100.0 * np.mean(np.abs(xf) < 1e-8)),
    })
    return out


def matrix_spectrum(x: np.ndarray) -> dict[str, float] | None:
    if x.ndim != 2 or min(x.shape) < 2:
        return None
    try:
        s = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    if s.size == 0 or not np.all(np.isfinite(s)):
        return None
    spec = float(s[0])
    fro2 = float(np.sum(s * s))
    stable_rank = fro2 / max(spec * spec, 1e-30)
    p = s / max(float(s.sum()), 1e-30)
    entropy = -float(np.sum(p * np.log(np.maximum(p, 1e-30))))
    eff_rank = float(np.exp(entropy))
    condition = float(spec / max(float(s[-1]), 1e-12))
    return {
        "spectral_norm": spec,
        "stable_rank": stable_rank,
        "effective_rank": eff_rank,
        "condition": condition,
        "rank_fraction": eff_rank / float(min(x.shape)),
    }


def aggregate_groups(flat: dict[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[np.ndarray]] = {}
    for path, x in flat.items():
        groups.setdefault(group_of(path), []).append(x)
    total_n = sum(x.size for x in flat.values())
    out: dict[str, dict[str, float | int]] = {}
    for g, xs in groups.items():
        n = int(sum(x.size for x in xs))
        l2sq = float(sum(np.sum(x * x) for x in xs))
        max_abs = float(max(np.max(np.abs(x)) for x in xs)) if xs else 0.0
        near0 = int(sum(np.sum(np.abs(x) < 1e-8) for x in xs))
        out[g] = {
            "params": n,
            "param_pct": 100.0 * n / max(total_n, 1),
            "l2": math.sqrt(l2sq),
            "rms": math.sqrt(l2sq / max(n, 1)),
            "max_abs": max_abs,
            "near_zero_pct": 100.0 * near0 / max(n, 1),
        }
    return out


def ema_drift(raw: dict[str, np.ndarray], ema: dict[str, np.ndarray]) -> dict[str, Any]:
    names = sorted(set(raw) & set(ema))
    global_raw2 = global_diff2 = dot = ema2 = 0.0
    by_group: dict[str, dict[str, float]] = {}
    tensors = []
    for name in names:
        a, b = raw[name], ema[name]
        if a.shape != b.shape:
            continue
        d = a - b
        a2 = float(np.sum(a * a)); b2 = float(np.sum(b * b)); d2 = float(np.sum(d * d))
        global_raw2 += a2; ema2 += b2; global_diff2 += d2; dot += float(np.sum(a * b))
        rel = math.sqrt(d2) / max(math.sqrt(b2), 1e-30)
        tensors.append((rel, name, math.sqrt(d2), math.sqrt(b2)))
        g = group_of(name)
        rec = by_group.setdefault(g, {"raw2": 0.0, "ema2": 0.0, "diff2": 0.0, "dot": 0.0})
        rec["raw2"] += a2; rec["ema2"] += b2; rec["diff2"] += d2; rec["dot"] += float(np.sum(a * b))
    groups = {}
    for g, r in by_group.items():
        groups[g] = {
            "relative_l2_drift": math.sqrt(r["diff2"]) / max(math.sqrt(r["ema2"]), 1e-30),
            "cosine": r["dot"] / max(math.sqrt(r["raw2"] * r["ema2"]), 1e-30),
        }
    tensors.sort(reverse=True)
    return {
        "global_relative_l2_drift": math.sqrt(global_diff2) / max(math.sqrt(ema2), 1e-30),
        "global_cosine": dot / max(math.sqrt(global_raw2 * ema2), 1e-30),
        "by_group": groups,
        "top_tensor_drifts": [
            {"path": n, "relative_l2_drift": r, "diff_l2": dl2, "ema_l2": el2}
            for r, n, dl2, el2 in tensors[:20]
        ],
    }


def stream_similarity(flat: dict[str, np.ndarray]) -> dict[str, Any]:
    # Compare corresponding tensors across streams when shapes match.
    families: dict[str, list[dict[str, Any]]] = {}
    groups = {g: {} for g in set(group_of(p) for p in flat)}
    for path, x in flat.items():
        g = group_of(path)
        suffix = path.split("/", 1)[1] if "/" in path else ""
        groups[g][suffix] = x

    for family in ("spatial", "frame_memory", "descriptor", "classifier"):
        pair_rows = []
        for i in range(4):
            for j in range(i + 1, 4):
                gi, gj = f"{family}_{i}", f"{family}_{j}"
                if gi not in groups or gj not in groups:
                    continue
                vals = []
                for suffix in sorted(set(groups[gi]) & set(groups[gj])):
                    a, b = groups[gi][suffix], groups[gj][suffix]
                    if a.shape == b.shape and a.size > 0:
                        c = cosine(a, b)
                        if np.isfinite(c):
                            vals.append(c)
                if vals:
                    pair_rows.append({
                        "pair": f"{STREAM_NAMES[i]}-{STREAM_NAMES[j]}",
                        "mean_tensor_cosine": float(np.mean(vals)),
                        "mean_abs_tensor_cosine": float(np.mean(np.abs(vals))),
                        "n_tensors": len(vals),
                    })
        families[family] = pair_rows
    return families


def gate_bias_audit(flat: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for path, x in flat.items():
        leaf = path.rsplit("/", 1)[-1]
        if leaf not in {"bz", "br"}:
            continue
        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))
        rows.append({
            "path": path,
            "gate": "update_z" if leaf == "bz" else "reset_r",
            "bias_mean": float(x.mean()),
            "bias_std": float(x.std()),
            "sigmoid_bias_mean": float(sig.mean()),
            "sigmoid_lt_0.05_pct": float(100 * np.mean(sig < 0.05)),
            "sigmoid_gt_0.95_pct": float(100 * np.mean(sig > 0.95)),
        })
    return rows


def special_heads(flat: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "fusion_prior" in flat:
        p = flat["fusion_prior"].reshape(-1)
        ex = np.exp(p - p.max()); sm = ex / ex.sum()
        out["fusion_prior"] = {
            "raw": p.tolist(),
            "softmax": {STREAM_NAMES[i]: float(sm[i]) for i in range(min(4, len(sm)))},
        }
    # Flax Dense kernels are [input, output]; column norms show output preference.
    for candidate in ("fusion_controller/kernel", "cross_stream_after_frame/score/kernel", "cross_stream_after_frame/gate/kernel"):
        if candidate in flat:
            x = flat[candidate]
            out[candidate] = {
                "shape": list(x.shape),
                "column_l2": np.linalg.norm(x, axis=0).reshape(-1).tolist() if x.ndim == 2 else None,
            }
    classifiers = {}
    for i, sn in STREAM_NAMES.items():
        k = f"classifier_{i}/kernel"
        if k in flat:
            x = flat[k]
            classifiers[sn] = {
                "kernel_l2": float(np.linalg.norm(x)),
                "kernel_rms": float(np.sqrt(np.mean(x * x))),
                "mean_class_column_l2": float(np.mean(np.linalg.norm(x, axis=0))) if x.ndim == 2 else None,
            }
    out["classifiers"] = classifiers
    return out


def print_table(title: str, rows: list[list[str]]) -> None:
    print("\n" + title)
    if not rows:
        print("  (none)")
        return
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(v.ljust(widths[i]) for i, v in enumerate(r)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    payload = serialization.msgpack_restore(ckpt.read_bytes())
    if "params" not in payload or "ema_params" not in payload:
        raise KeyError(f"Checkpoint missing params/ema_params; keys={list(payload)}")

    raw = flatten_params(payload["params"])
    ema = flatten_params(payload["ema_params"])
    raw_n = sum(x.size for x in raw.values())
    ema_n = sum(x.size for x in ema.values())

    tensor_report = {}
    spectra = {}
    bad = []
    for name, x in ema.items():
        st = tensor_stats(x)
        tensor_report[name] = st
        if st["nan"] or st["inf"]:
            bad.append(name)
        sp = matrix_spectrum(x)
        if sp is not None:
            spectra[name] = sp

    groups = aggregate_groups(ema)
    drift = ema_drift(raw, ema)
    similarities = stream_similarity(ema)
    gates = gate_bias_audit(ema)
    heads = special_heads(ema)

    low_rank = sorted(
        ((v["rank_fraction"], k, v) for k, v in spectra.items()),
        key=lambda z: z[0],
    )[:15]
    high_cond = sorted(
        ((v["condition"], k, v) for k, v in spectra.items()),
        key=lambda z: z[0], reverse=True,
    )[:15]
    high_norm = sorted(
        ((float(tensor_report[k]["rms"] or 0.0), k) for k in tensor_report),
        reverse=True,
    )[:15]

    metadata = {
        "model": payload.get("model"),
        "protocol": payload.get("protocol"),
        "selector": payload.get("selector"),
        "epoch": int(np.asarray(payload.get("epoch", -1))),
        "val_accuracy": float(np.asarray(payload.get("val_accuracy", float("nan")))),
        "step": int(np.asarray(payload.get("step", -1))),
        "raw_param_count": int(raw_n),
        "ema_param_count": int(ema_n),
    }

    print("=" * 120)
    print("NESTSAR M4-MOTIONPRESERVE-T16 — WEIGHT AUDIT")
    print("=" * 120)
    for k, v in metadata.items():
        print(f"{k:22s}: {v}")
    print(f"finite_check           : {'PASS' if not bad else 'FAIL'}")
    if bad:
        print("bad tensors:", bad)

    group_rows = [["GROUP", "PARAMS", "%", "RMS", "L2", "MAXABS", "RAW↔EMA DRIFT"]]
    for g, s in sorted(groups.items(), key=lambda kv: kv[1]["params"], reverse=True):
        d = drift["by_group"].get(g, {}).get("relative_l2_drift", float("nan"))
        group_rows.append([
            g, f"{int(s['params']):,}", f"{s['param_pct']:.2f}", f"{s['rms']:.5f}",
            f"{s['l2']:.2f}", f"{s['max_abs']:.4f}", f"{100*d:.3f}%",
        ])
    print_table("MODULE SUMMARY", group_rows)

    print("\nRAW vs EMA")
    print(f"  global relative L2 drift : {100*drift['global_relative_l2_drift']:.4f}%")
    print(f"  global cosine            : {drift['global_cosine']:.8f}")

    gate_rows = [["GATE", "SIGMOID(BIAS)", "<.05%", ">.95%", "PATH"]]
    for r in gates:
        gate_rows.append([
            r["gate"], f"{r['sigmoid_bias_mean']:.4f}", f"{r['sigmoid_lt_0.05_pct']:.1f}",
            f"{r['sigmoid_gt_0.95_pct']:.1f}", r["path"],
        ])
    print_table("GATED-SWEEP BIAS HEALTH (bias-only tendency; activations also depend on x,h)", gate_rows)

    sim_rows = [["FAMILY", "PAIR", "MEAN COS", "MEAN |COS|", "N"]]
    for fam, vals in similarities.items():
        for r in vals:
            sim_rows.append([fam, r["pair"], f"{r['mean_tensor_cosine']:+.4f}",
                             f"{r['mean_abs_tensor_cosine']:.4f}", str(r["n_tensors"])])
    print_table("CROSS-STREAM WEIGHT SIMILARITY", sim_rows)

    if "fusion_prior" in heads:
        print("\nFUSION PRIOR")
        print("  raw    :", np.round(heads["fusion_prior"]["raw"], 5).tolist())
        print("  softmax:", {k: round(v, 4) for k, v in heads["fusion_prior"]["softmax"].items()})
    print("\nCLASSIFIER HEAD NORMS")
    for sn, r in heads.get("classifiers", {}).items():
        print(f"  {sn:2s}: L2={r['kernel_l2']:.3f} RMS={r['kernel_rms']:.5f} mean-class-L2={r['mean_class_column_l2']:.4f}")

    print("\nLOWEST EFFECTIVE-RANK FRACTION MATRICES")
    for frac, name, s in low_rank[:10]:
        print(f"  {frac:7.3f}  eff_rank={s['effective_rank']:7.2f} stable={s['stable_rank']:7.2f}  {name}")

    print("\nHIGHEST CONDITION MATRICES")
    for cond, name, s in high_cond[:10]:
        print(f"  cond={cond:11.3g} rank_frac={s['rank_fraction']:.3f}  {name}")

    print("\nHIGHEST-RMS TENSORS")
    for rms, name in high_norm[:10]:
        print(f"  RMS={rms:.6f}  {name}")

    # Conservative automatic flags; these are diagnostics, not proof of a bottleneck.
    flags = []
    if bad:
        flags.append("NONFINITE_WEIGHTS")
    if drift["global_relative_l2_drift"] > 0.05:
        flags.append("HIGH_RAW_EMA_DRIFT")
    sat = [r for r in gates if r["sigmoid_lt_0.05_pct"] > 25 or r["sigmoid_gt_0.95_pct"] > 25]
    if sat:
        flags.append("GATE_BIAS_SATURATION_CANDIDATE")
    collapsed = [(k, v) for k, v in spectra.items() if v["rank_fraction"] < 0.20 and min(ema[k].shape) >= 16]
    if collapsed:
        flags.append("LOW_EFFECTIVE_RANK_CANDIDATE")

    print("\nAUDIT FLAGS")
    if flags:
        for f in flags:
            print("  -", f)
    else:
        print("  - NONE FROM STATIC WEIGHTS")

    report = {
        "metadata": metadata,
        "groups": groups,
        "ema_drift": drift,
        "gate_bias": gates,
        "stream_similarity": similarities,
        "special_heads": heads,
        "spectra": spectra,
        "tensor_stats": tensor_report,
        "flags": flags,
    }
    out = Path(args.out) if args.out else ckpt.with_name("weight_audit.json")
    out.write_text(json.dumps(report, indent=2, allow_nan=True))
    print("\nREPORT:", out)
    print("=" * 120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

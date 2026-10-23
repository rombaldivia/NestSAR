"""R9 ablation report: what each graph_mr component contributes, with seed pairing and McNemar tests.

Reads <run>/xsub/val_probs.npz (eval_heads; falls back to result.json without the paired tests) of the base
runs and of every ablation, pairs runs by seed, and prints one row per comparison:

    delta = ablation - reference, in pp (negative = removing the component hurts = it contributes)

Two noise sources are reported separately. McNemar (same seed, same 50,919 validation clips) only tells
whether two specific models differ beyond sample noise; it says nothing about seed noise, which the
reference's own seed-to-seed spread estimates. A component is called "aporta" only when the drop is
consistent across seeds, larger than that spread, and significant per seed.

    python -m experiments.nestsar_r5_t16.ablation_report --working /kaggle/working [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SEEDS = (7, 11, 13)
# Folder names per variant ({s} = seed). The first two names are the champion runs that already exist.
ALIASES = {
    "graph_mr": ["NestSAR_T16_GRAPH_MR_SEED{s}", "NestSAR_T16_GRAPH_MR_REGNOMIX_SEED{s}"],
    "graph_mr_no_fast": ["NestSAR_T16_GRAPH_MR_NO_FAST_SEED{s}", "NestSAR_T16_GRAPH_MR_NOFAST_SEED{s}"],
}
# (what is tested, variant, reference, kind)
COMPARISONS = [
    ("memoria rápida (nested fast memory)", "graph_mr_no_fast", "graph_mr", "remove"),
    ("segundo nivel temporal g4", "graph_mr_one_level", "graph_mr", "remove"),
    ("BiGRU temporal (queda solo la memoria)", "graph_mr_mem_only", "graph_mr", "remove"),
    ("todo el modelo temporal", "graph_mr_no_temporal", "graph_mr", "remove"),
    ("memoria sola vs nada temporal", "graph_mr_no_temporal", "graph_mr_mem_only", "remove"),
    ("rama de manos 4x", "graph_mr_no_hand", "graph_mr", "remove"),
    ("partes finas (14 vs 10)", "graph_mr_coarse", "graph_mr", "remove"),
    ("interacción entre personas", "graph_mr_no_inter", "graph_mr", "remove"),
    ("reconstrucción enmascarada (solo entrenamiento)", "graph_mr_no_recon", "graph_mr", "remove"),
    ("m4/g4 con pesos compartidos (-40% params)", "graph_mr_shared", "graph_mr", "efficiency"),
    ("costo de ser causal", "graph_mr_causal", "graph_mr", "efficiency"),
    ("memoria rápida en el modelo causal", "graph_mr_causal_no_fast", "graph_mr_causal", "remove"),
    ("más capacidad (model_dim 256, 61 MFLOPs)", "graph_mr_wide", "graph_mr", "efficiency"),
]


def find(working: Path, variant: str, seed: int):
    names = ALIASES.get(variant, []) + [f"NestSAR_T16_{variant.upper()}_SEED{{s}}"]
    for n in names:
        d = working / n.format(s=seed) / "xsub"
        if (d / "result.json").exists():
            return d
    return None


def load(d: Path):
    res = json.loads((d / "result.json").read_text())
    acc = res["best_val_accuracy"]
    acc = 100 * acc if acc <= 1 else acc
    probs = d / "val_probs.npz"
    correct = y = pred = None
    if probs.exists():
        z = np.load(probs)
        y, pred = z["y"], z["main"].argmax(-1)
        correct = pred == y
        acc = 100 * float(correct.mean())
    cfg_file = d / "run_config.json"
    cfg = json.loads(cfg_file.read_text()).get("config", {}) if cfg_file.exists() else {}
    # Last-epoch EMA accuracy: no max over epochs taken on the test split (unbiased, unlike best-epoch).
    last = None
    hist = d / "history.json"
    if hist.exists():
        h = json.loads(hist.read_text())
        h = h if isinstance(h, list) else h.get("epochs", [])
        if h:
            v = h[-1]["val_acc"]
            last = 100 * v if v <= 1 else v
    return dict(acc=acc, last=last, correct=correct, y=y, pred=pred, params=res.get("params"),
                mflops=res.get("strict_mflops"), cfg=cfg, dir=str(d.parent.name))


def mcnemar(a, b):
    """a, b: boolean correct vectors on the same clips. Returns (only_a, only_b, two-sided p)."""
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    if n01 + n10 == 0:
        return n01, n10, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)              # continuity-corrected, 1 dof
    return n01, n10, math.erfc(math.sqrt(chi2 / 2))


RECIPE_KEYS = ("epochs", "micro_batch", "accumulation_steps", "aug_strength", "dropout", "weight_decay",
               "label_smoothing", "recon_mask_ratio", "aug_pool_views_use", "mix_prob", "kd_weight")


def recipe_mismatch(a, b):
    return [k for k in RECIPE_KEYS if k in a["cfg"] and k in b["cfg"] and a["cfg"][k] != b["cfg"][k]]


def class_shift(ref, abl, top=4):
    if ref["y"] is None or abl["y"] is None:
        return []
    y = ref["y"]
    out = []
    for c in np.unique(y):
        m = y == c
        out.append((int(c) + 1, 100 * float(abl["correct"][m].mean() - ref["correct"][m].mean())))
    out.sort(key=lambda t: t[1])
    return out[:top]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--working", default="/kaggle/working")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    w = Path(a.working)
    runs = {}
    for _, v, r, _ in COMPARISONS:
        for name in (v, r):
            for s in SEEDS:
                d = find(w, name, s)
                if d is not None and (name, s) not in runs:
                    runs[(name, s)] = load(d)

    base = [runs[("graph_mr", s)]["acc"] for s in SEEDS if ("graph_mr", s) in runs]
    spread = (max(base) - min(base)) if len(base) > 1 else float("nan")
    print(f"Base graph_mr: {', '.join(f'{x:.2f}' for x in base)}  ->  media {np.mean(base):.2f}, "
          f"rango entre semillas {spread:.2f} pp (ruido de semilla)\n")
    rows = []
    for label, v, r, kind in COMPARISONS:
        pairs = [s for s in SEEDS if (v, s) in runs and (r, s) in runs]
        if not pairs:
            print(f"-- {label}: falta correr ({v} vs {r})")
            continue
        deltas, last_deltas, tests, warn = [], [], [], []
        for s in pairs:
            A, B = runs[(r, s)], runs[(v, s)]
            deltas.append(B["acc"] - A["acc"])
            if A["last"] is not None and B["last"] is not None:
                last_deltas.append(B["last"] - A["last"])
            warn += recipe_mismatch(A, B)
            if A["correct"] is not None and B["correct"] is not None:
                if len(A["correct"]) != len(B["correct"]) or not np.array_equal(A["y"], B["y"]):
                    raise SystemExit(f"{A['dir']} and {B['dir']} were evaluated on different clips")
                tests.append(mcnemar(A["correct"], B["correct"]))
        md = float(np.mean(deltas))
        same_sign = all(d < 0 for d in deltas) or all(d > 0 for d in deltas)
        sig = len(tests) == len(pairs) and all(p < 0.01 for _, _, p in tests)
        noise = max(0.3, spread if not math.isnan(spread) else 0.3)
        if len(pairs) < 2:
            verdict = "1 semilla: no concluyente"
        elif same_sign and sig and abs(md) >= noise:
            verdict = ("APORTA" if md < 0 else "SOBRA (quitarlo mejora)") if kind == "remove" else \
                      ("cuesta precisión" if md < 0 else "gratis o mejor")
        elif not same_sign or abs(md) < spread:
            verdict = "no se distingue del ruido" if kind == "remove" else "sin costo medible"
        else:
            verdict = "efecto pequeño, dudoso"
        v_run = runs[(v, pairs[0])]
        rows.append(dict(label=label, variant=v, reference=r, seeds=pairs, deltas=deltas, mean_delta=md,
                         last_epoch_deltas=last_deltas,
                         mcnemar=[dict(only_ref=t[0], only_abl=t[1], p=t[2]) for t in tests], verdict=verdict,
                         params=v_run["params"], mflops=v_run["mflops"], recipe_mismatch=sorted(set(warn)),
                         worst_classes=class_shift(runs[(r, pairs[0])], v_run)))
        p_txt = "  ".join(f"s{s}: p={t[2]:.1e} ({t[0]} vs {t[1]})" for s, t in zip(pairs, tests)) or "sin val_probs"
        print(f"-- {label}  [{v} vs {r}]")
        print(f"   delta por semilla: {', '.join(f'{d:+.2f}' for d in deltas)}  media {md:+.2f} pp   -> {verdict}")
        if len(last_deltas) == len(pairs):
            print(f"   última época (sin elegir el máximo): {', '.join(f'{d:+.2f}' for d in last_deltas)}  "
                  f"media {np.mean(last_deltas):+.2f} pp")
        print(f"   McNemar: {p_txt}")
        if v_run["params"] is not None:
            print(f"   {v}: {v_run['params']:,} params (incl. cabezas de entrenamiento), {v_run['mflops']:.2f} MFLOPs")
        if rows[-1]["worst_classes"]:
            print("   clases que más cambian (A###, pp): " +
                  ", ".join(f"A{c:03d} {d:+.1f}" for c, d in rows[-1]["worst_classes"]))
        if warn:
            print(f"   OJO: receta distinta en {sorted(set(warn))} - la comparación no es limpia")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(base=base, seed_range=spread, rows=rows), indent=1))
        print(f"\nGuardado en {a.json}")


if __name__ == "__main__":
    main()

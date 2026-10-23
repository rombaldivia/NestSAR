"""Where does a finished run fail on unseen subjects? Accuracy by subject, camera, setup, interaction and class.

Reads <run>/<protocol>/val_probs.npz (eval_heads / ablation_queue) and the NTU sample IDs of the cache
(SsssCcccPpppRrrrAaaa). No training, no GPU. Several runs: one column each, plus their ensemble.

    python -m experiments.nestsar_r5_t16.generalization_report --cache /tmp/NestSAR_R5_HAND_CACHE_T16 RUN [RUN ...]

The subject table also reports how much the per-subject accuracies spread beyond what sampling alone would
give (binomial noise for each subject's clip count): a large excess means the model depends on who performs
the action, not only on the action.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .data import Dataset

ID = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")
# NTU mutual (two-person) actions: A050-A060 and A106-A120.
INTERACTION = set(range(50, 61)) | set(range(106, 121))


def group_acc(keys, correct):
    out = {}
    for k in np.unique(keys):
        m = keys == k
        out[int(k)] = (100 * float(correct[m].mean()), int(m.sum()))
    return out


def subject_spread(keys, correct):
    acc = group_acc(keys, correct)
    p = float(correct.mean())
    vals = np.array([a for a, _ in acc.values()])
    expected = np.sqrt(np.mean([p * (1 - p) / n for _, n in acc.values()])) * 100
    return float(vals.std()), expected, acc


def adjusted_spread(group, other, correct):
    """Spread of `group` accuracies after removing the effect of `other` (e.g. subjects within their setups).

    Each clip's correctness is compared with the accuracy of its `other` cell; the group effect is the mean
    of those residuals. Returns (raw spread, adjusted spread, sampling-noise spread), all in pp.
    """
    p = float(correct.mean())
    other_acc = {k: correct[other == k].mean() for k in np.unique(other)}
    resid = correct - np.array([other_acc[k] for k in other])
    raw, adj, noise = [], [], []
    for g in np.unique(group):
        m = group == g
        raw.append(correct[m].mean())
        adj.append(resid[m].mean())
        noise.append(p * (1 - p) / m.sum())
    return 100 * float(np.std(raw)), 100 * float(np.std(adj)), 100 * float(np.sqrt(np.mean(noise)))


def crossed_report(subject, setup, correct, name):
    print(f"\nSUJETO x SETUP ({name})")
    print("  ¿la diferencia entre sujetos es del sujeto o de dónde lo grabaron?")
    print(f"  {'':10s}{'cruda':>9s}{'ajustada':>10s}{'azar':>8s}")
    raw, adj, noise = adjusted_spread(subject, setup, correct)
    print(f"  {'sujetos':10s}{raw:9.2f}{adj:10.2f}{noise:8.2f}   (ajustada = descontando el setup)")
    raw2, adj2, noise2 = adjusted_spread(setup, subject, correct)
    print(f"  {'setups':10s}{raw2:9.2f}{adj2:10.2f}{noise2:8.2f}   (ajustada = descontando el sujeto)")
    excess_raw, excess_adj = raw ** 2 - noise ** 2, max(adj ** 2 - noise ** 2, 0.0)   # variances beyond chance
    share = 0.0 if excess_raw <= 0 else min(1.0, max(0.0, 1 - excess_adj / excess_raw))
    print(f"  -> el setup explica ~{100 * share:.0f}% de la variación entre sujetos que excede el azar")
    per_subject = [len(np.unique(setup[subject == k])) for k in np.unique(subject)]
    print(f"  (cada sujeto aparece en {np.mean(per_subject):.1f} setups en promedio; si es ~1, sujeto y setup"
          " no se pueden separar)")
    sacc = {k: 100 * correct[setup == k].mean() for k in np.unique(setup)}
    pacc = {k: 100 * correct[subject == k].mean() for k in np.unique(subject)}
    print("\n  peores sujetos, por setup en que aparecen (acc del sujeto / acc de todo el setup, n):")
    for p_ in sorted(pacc, key=pacc.get)[:6]:
        cells = []
        for s_ in np.unique(setup[subject == p_]):
            m = (subject == p_) & (setup == s_)
            if m.sum() >= 30:
                cells.append(f"S{s_:03d} {100 * correct[m].mean():.0f}/{sacc[s_]:.0f} ({m.sum()})")
        print(f"  P{p_:03d} {pacc[p_]:5.1f}%: " + ", ".join(cells))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--protocol", default="xsub", choices=("xsub", "xset"))
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)

    ds = Dataset(a.cache)
    val_ids = np.asarray(ds.splits[f"{a.protocol}_val"])
    all_ids = json.loads((Path(ds.base_dir) / "ids.json").read_text())
    parsed = np.array([[int(g) for g in ID.search(all_ids[i]).groups()] for i in val_ids])
    setup, camera, subject, _, action = parsed.T
    labels = np.asarray(ds.labels)[val_ids]

    names, corrects, probs = [], [], []
    for run in a.runs:
        z = np.load(Path(run) / a.protocol / "val_probs.npz")
        if not np.array_equal(z["y"], labels):
            raise SystemExit(f"{run}: val_probs order/labels do not match the cache split")
        names.append(Path(run).name.replace("NestSAR_T16_", ""))
        probs.append(z["main"])
        corrects.append(z["main"].argmax(-1) == labels)
    if len(probs) > 1:
        names.append("ENSEMBLE")
        ens = np.mean(probs, 0)
        corrects.append(ens.argmax(-1) == labels)
        probs.append(ens)
    w = max(12, max(len(n) for n in names) + 1)
    head = lambda title: print(f"\n{title}\n{'':14s}" + "".join(f"{n[:w - 1]:>{w}s}" for n in names))
    row = lambda label, vals: print(f"{label:14s}" + "".join(f"{v:>{w}.2f}" for v in vals))

    head("TOTAL")
    row("top-1", [100 * c.mean() for c in corrects])

    head("POR CAMARA (C001 frente, C002/C003 laterales en NTU)")
    for c in sorted(np.unique(camera)):
        m = camera == c
        row(f"C{c:03d} n={m.sum()}", [100 * x[m].mean() for x in corrects])

    head("UNA PERSONA vs INTERACCION")
    inter = np.isin(action, list(INTERACTION))
    for label, m in ((f"una  n={(~inter).sum()}", ~inter), (f"dos  n={inter.sum()}", inter)):
        if m.any():
            row(label, [100 * x[m].mean() for x in corrects])

    head("POR SUJETO (desv. entre sujetos vs la esperada solo por muestreo)")
    spreads = [subject_spread(subject, x) for x in corrects]
    row("desv. obs.", [s[0] for s in spreads])
    row("desv. azar", [s[1] for s in spreads])
    ref = spreads[0][2]
    worst = sorted(ref, key=lambda k: ref[k][0])[:6]
    best = sorted(ref, key=lambda k: -ref[k][0])[:3]
    for k in dict.fromkeys(worst + best):
        row(f"P{k:03d} n={ref[k][1]}", [s[2][k][0] for s in spreads])

    head("POR SETUP (los 6 peores segun la primera corrida)")
    sacc = group_acc(setup, corrects[0])
    for k in sorted(sacc, key=lambda k: sacc[k][0])[:6]:
        m = setup == k
        row(f"S{k:03d} n={m.sum()}", [100 * x[m].mean() for x in corrects])

    crossed_report(subject, setup, corrects[0], names[0])

    print(f"\nCLASES MAS DEBILES ({names[0]})")
    cacc = group_acc(action, corrects[0])
    pred0 = probs[0].argmax(-1)
    for k in sorted(cacc, key=lambda k: cacc[k][0])[:10]:
        m = action == k
        wrong = pred0[m & ~corrects[0]]
        tail = f"  se confunde sobre todo con A{np.bincount(wrong).argmax() + 1:03d}" if wrong.size else ""
        print(f"  A{k:03d}  {cacc[k][0]:5.1f}%  (n={cacc[k][1]}){tail}")

    print(f"\nPARES MAS CONFUNDIDOS ({names[0]}, ambos sentidos)")
    n_cls = probs[0].shape[1]
    cm = np.zeros((n_cls, n_cls), np.int64)
    np.add.at(cm, (labels, pred0), 1)
    np.fill_diagonal(cm, 0)
    sym = cm + cm.T
    iu = np.triu_indices(n_cls, 1)
    order = np.argsort(sym[iu])[::-1][:10]
    for i, j in zip(iu[0][order], iu[1][order]):
        print(f"  A{i + 1:03d} <-> A{j + 1:03d}: {sym[i, j]} errores ({cm[i, j]} / {cm[j, i]})")


if __name__ == "__main__":
    main()

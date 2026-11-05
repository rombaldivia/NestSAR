"""Model soup: average the EMA weights of several finished runs of the SAME variant and evaluate once.

A soup costs exactly one model at inference (same graph, same FLOPs). It only helps when the runs share
their initialisation (same seed) and stay in one loss basin; the uniform soup is compared with every
ingredient and with the greedy soup (add an ingredient only if validation does not drop).

    python -m experiments.nestsar_r5_t16.soup --cache /tmp/HAND_CACHE --out /kaggle/working/SOUP RUN_DIR [RUN_DIR ...]

NOTE: the greedy soup chooses ingredients on the validation split, so its number is optimistic; report
the uniform soup as the clean result.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from .data import Dataset
from .eval_heads import acc, evaluate


def load(run, protocol):
    out = Path(run) / protocol
    payload = serialization.msgpack_restore((out / "best.msgpack").read_bytes())
    cfg = json.loads((out / "run_config.json").read_text())["config"]
    return payload, cfg


def write_soup(dst, protocol, template_run, payloads, weights):
    out = Path(dst) / protocol
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(Path(template_run) / protocol / "run_config.json", out / "run_config.json")
    w = np.asarray(weights, np.float64) / np.sum(weights)
    avg = jax.tree.map(lambda *xs: sum(wi * np.asarray(x, np.float64) for wi, x in zip(w, xs)).astype(np.float32),
                       *[p["ema_params"] for p in payloads])
    soup = dict(payloads[0], ema_params=avg)
    (out / "best.msgpack").write_bytes(serialization.to_bytes(soup))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--protocol", default="xsub", choices=("xsub", "xset"))
    ap.add_argument("--out", required=True, help="scratch dir for the soup checkpoints")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()

    loaded = [load(r, a.protocol) for r in a.runs]
    variants = {c["variant"] for _, c in loaded}
    if len(variants) != 1:
        raise SystemExit(f"All runs must share one variant (architecture); got {variants}")
    seeds = sorted({c.get("seed") for _, c in loaded})
    print(f"variant {variants.pop()}  seeds {seeds}" + ("" if len(seeds) == 1 else
          "  WARNING: different seeds -> different inits; a weight soup usually fails, use a logit ensemble"))

    dataset = Dataset(a.cache)
    singles = []
    for r in a.runs:
        y, main, _ = evaluate(r, a.protocol, dataset)
        singles.append((acc(y, main), r))
        print(f"  ingredient {acc(y, main):6.2f}  {Path(r).name}")
    payloads = [p for p, _ in loaded]

    uni = write_soup(Path(a.out) / "uniform", a.protocol, a.runs[0], payloads, [1] * len(payloads))
    y, main, _ = evaluate(uni, a.protocol, dataset)
    print(f"\nUNIFORM soup of {len(payloads)}: {acc(y, main):6.2f}   (best ingredient {max(singles)[0]:.2f})")

    order = sorted(range(len(a.runs)), key=lambda i: -singles[i][0])
    chosen, best = [order[0]], singles[order[0]][0]
    for i in order[1:]:
        trial = write_soup(Path(a.out) / f"greedy_try{i}", a.protocol, a.runs[0],
                           [payloads[j] for j in chosen + [i]], [1] * (len(chosen) + 1))
        yt, mt, _ = evaluate(trial, a.protocol, dataset)
        if acc(yt, mt) >= best:
            chosen, best = chosen + [i], acc(yt, mt)
        print(f"  greedy +{Path(a.runs[i]).name}: {acc(yt, mt):6.2f} -> {'kept' if i in chosen else 'dropped'}")
    print(f"GREEDY soup ({len(chosen)} runs, chosen on val -> optimistic): {best:6.2f}")


if __name__ == "__main__":
    main()

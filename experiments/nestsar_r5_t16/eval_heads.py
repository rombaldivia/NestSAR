"""Inference-time head averaging and cross-run ensembling for finished R5 runs (diagnostic).

For each run directory it loads the best EMA checkpoint, evaluates the official validation split once and
reports top-1 of: the main classifier, every auxiliary head, and softmax averages of main + aux heads
(the aux heads already run at inference; averaging adds only ~0.04 MFLOPs). Per-sample probabilities are
saved to ``<run>/<protocol>/val_probs.npz`` so several runs can then be ensembled without re-evaluating.

    python -m experiments.nestsar_r5_t16.eval_heads --cache /tmp/HAND_CACHE --protocol xsub RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import json
import math
from contextlib import closing
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from .config import model_kwargs, validate_config
from .data import Dataset
from .model import NestSARR5T16


def evaluate(run_dir, protocol, dataset):
    out = Path(run_dir) / protocol
    target = out / "val_probs.npz"
    if target.exists():
        z = np.load(target)
        return z["y"], z["main"], z["aux"]
    config = validate_config(json.loads((out / "run_config.json").read_text())["config"])
    model = NestSARR5T16(**model_kwargs(config))
    params = jax.device_put(serialization.msgpack_restore((out / "best.msgpack").read_bytes())["ema_params"])

    @jax.jit
    def probs(p, x):
        o = model.apply({"params": p}, x, training=False)
        return jax.nn.softmax(o["logits"], -1), jax.nn.softmax(o["stream_logits"], -1)

    val_ids = dataset.splits[f"{protocol}_val"]
    ys, mains, auxs = [], [], []
    steps = math.ceil(len(val_ids) / config["eval_batch"])
    with closing(dataset.batches(val_ids, config["eval_batch"], config, protocol=protocol)) as batches:
        for i in range(steps):
            batch, _ = next(batches)
            n = int(batch["mask"].sum())
            m, a = probs(params, jax.device_put(batch["x"]))
            ys.append(np.asarray(batch["y"][:n]))
            mains.append(np.asarray(m[:n], np.float32))
            auxs.append(np.asarray(a[:n], np.float32))
            if i % 50 == 0 or i + 1 == steps:
                print(f"  {Path(run_dir).name}: {i + 1}/{steps}", flush=True)
    y, main, aux = np.concatenate(ys), np.concatenate(mains), np.concatenate(auxs)
    if len(y) != len(val_ids):
        raise RuntimeError(f"Evaluated {len(y)} samples, expected {len(val_ids)}")
    np.savez_compressed(target, y=y, main=main, aux=aux)
    return y, main, aux


def acc(y, p):
    return 100.0 * float(np.mean(p.argmax(-1) == y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    from experiments.nestsar_sm_all_t16.streaming import dataset_spec
    ap.add_argument("--protocol", default="xsub", choices=dataset_spec.PROTOCOLS)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    dataset = Dataset(a.cache)
    mains, y0 = [], None
    for run in a.runs:
        y, main, aux = evaluate(run, a.protocol, dataset)
        if y0 is not None and not np.array_equal(y, y0):
            raise RuntimeError("Runs were evaluated on different sample orders")
        y0 = y
        names = ["m4_aux", "hand_aux"][:aux.shape[1]] + [f"aux{i}" for i in range(2, aux.shape[1])]
        print(f"\n{Path(run).name} [{a.protocol}]")
        print(f"  main                 {acc(y, main):6.2f}")
        for i, nm in enumerate(names):
            print(f"  {nm:20s} {acc(y, aux[:, i]):6.2f}")
        for w in (0.5, 0.3, 0.2):   # main weight 1, each aux head weight w
            print(f"  main + {w:.1f}*aux       {acc(y, main + w * aux.sum(1)):6.2f}")
        mains.append(main)
    if len(mains) > 1:
        print(f"\nEnsemble of {len(mains)} runs (main heads, mean prob): {acc(y0, np.mean(mains, 0)):6.2f}")
        print("  individual:", "  ".join(f"{acc(y0, m):.2f}" for m in mains))


if __name__ == "__main__":
    main()

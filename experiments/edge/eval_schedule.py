"""Full-test-set check that the edge deployment schedule (split BiGRU, experiments/edge/edge_schedules.py) gives the
same accuracy as the training schedule, with the trained EMA weights.

For every run directory it evaluates the official validation split twice — training schedule and split schedule —
and reports both top-1 values, how many test clips change their predicted class, and the largest logit difference.

    python -m experiments.edge.eval_schedule --cache /tmp/HAND_CACHE --protocol xsub RUN_DIR [RUN_DIR ...]

Same data pipeline and checkpoint loading as experiments.nestsar_r5_t16.eval_heads (run it in the same session/env).
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

from experiments.edge.edge_schedules import split_bigru
from experiments.nestsar_r5_t16.config import model_kwargs, validate_config
from experiments.nestsar_r5_t16.data import Dataset
from experiments.nestsar_r5_t16.model import NestSARR5T16


def logits_on_val(run_dir, protocol, dataset, split):
    out = Path(run_dir) / protocol
    config = validate_config(json.loads((out / "run_config.json").read_text())["config"])
    model = NestSARR5T16(**model_kwargs(config))
    params = jax.device_put(serialization.msgpack_restore((out / "best.msgpack").read_bytes())["ema_params"])
    # A fresh function object per schedule, so the jit trace cache cannot reuse the other schedule's program.
    fn = jax.jit(lambda p, x: model.apply({"params": p}, x, training=False)["logits"])
    val_ids = dataset.splits[f"{protocol}_val"]
    steps = math.ceil(len(val_ids) / config["eval_batch"])
    ys, ls = [], []
    with closing(dataset.batches(val_ids, config["eval_batch"], config, protocol=protocol)) as batches:
        for i in range(steps):
            batch, _ = next(batches)
            n = int(batch["mask"].sum())
            ys.append(np.asarray(batch["y"][:n]))
            ls.append(np.asarray(fn(params, jax.device_put(batch["x"]))[:n], np.float32))
            if i % 100 == 0 or i + 1 == steps:
                print(f"  {Path(run_dir).name} [{'split' if split else 'training'} schedule] {i + 1}/{steps}", flush=True)
    y, logits = np.concatenate(ys), np.concatenate(ls)
    if len(y) != len(val_ids):
        raise RuntimeError(f"Evaluated {len(y)} samples, expected {len(val_ids)}")
    return y, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    from experiments.nestsar_sm_all_t16.streaming import dataset_spec
    ap.add_argument("--protocol", default="xsub", choices=dataset_spec.PROTOCOLS)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    dataset = Dataset(a.cache)
    rows = []
    for run in a.runs:
        y, l0 = logits_on_val(run, a.protocol, dataset, split=False)
        with split_bigru():
            y1, l1 = logits_on_val(run, a.protocol, dataset, split=True)
        if not np.array_equal(y, y1):
            raise RuntimeError("The two passes saw different sample orders")
        p0, p1 = l0.argmax(-1), l1.argmax(-1)
        row = {"run": Path(run).name, "protocol": a.protocol, "n": int(len(y)),
               "top1_training_schedule": 100 * float(np.mean(p0 == y)),
               "top1_split_schedule": 100 * float(np.mean(p1 == y)),
               "changed_predictions": int(np.sum(p0 != p1)),
               "max_abs_logit_diff": float(np.abs(l0 - l1).max())}
        rows.append(row)
        print(f"\n{row['run']} [{a.protocol}]  n={row['n']}  training {row['top1_training_schedule']:.2f}  "
              f"split {row['top1_split_schedule']:.2f}  changed {row['changed_predictions']}  "
              f"max|dlogit| {row['max_abs_logit_diff']:.1e}", flush=True)
        (Path(run) / a.protocol / "edge_schedule_check.json").write_text(json.dumps(row, indent=1))
    print("\nrun                                    protocol   training   split   changed")
    for r in rows:
        print(f"{r['run'][:38]:38s} {r['protocol']:8s} {r['top1_training_schedule']:8.2f} "
              f"{r['top1_split_schedule']:7.2f} {r['changed_predictions']:8d}")


if __name__ == "__main__":
    main()

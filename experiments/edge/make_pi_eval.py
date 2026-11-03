"""Package one trained NestSAR run for an ON-DEVICE accuracy test (Raspberry Pi etc.).

Writes to OUT:
  model.jaxexport   the trained EMA weights baked into an XLA CPU program, edge deployment schedule (split BiGRU),
                    batch 1 - the exact program whose latency bench_xla.py measures;
  x.npy, y.npy      the official validation split of PROTOCOL, already tokenised (float32 [N,16,F], int32 [N]);
  info.json         run, protocol, sample count, GPU-side top-1 of the same weights (for comparison), jax version.

    NESTSAR_DATASET=ntu60 NESTSAR_FRAMES=16 python -m experiments.edge.make_pi_eval \
        --cache /tmp/NestSAR_R5_HAND_CACHE_T16_NTU60 --protocol xsub --out /kaggle/working/pi_eval_ntu60_xsub RUN_DIR

On the device:  python3 eval_pi.py pi_eval_ntu60_xsub   (edge kit)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--protocol", default="xsub")
    ap.add_argument("--out", required=True)
    ap.add_argument("--export-version", type=int, default=None,
                    help="jax.export calling-convention version (lower it if the device's jax is older)")
    ap.add_argument("run")
    a = ap.parse_args()
    from jax import export
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    run = Path(a.run) / a.protocol
    config = validate_config(json.loads((run / "run_config.json").read_text())["config"])
    model = NestSARR5T16(**model_kwargs(config))
    params = serialization.msgpack_restore((run / "best.msgpack").read_bytes())["ema_params"]

    # 1) tokenised validation split, in the exact order and form the training code evaluates it
    dataset = Dataset(a.cache)
    val_ids = dataset.splits[f"{a.protocol}_val"]
    steps = math.ceil(len(val_ids) / config["eval_batch"])
    xs, ys = [], []
    with closing(dataset.batches(val_ids, config["eval_batch"], config, protocol=a.protocol)) as batches:
        for _ in range(steps):
            batch, _ = next(batches)
            n = int(batch["mask"].sum())
            xs.append(np.asarray(batch["x"][:n], np.float32)); ys.append(np.asarray(batch["y"][:n], np.int32))
    x, y = np.concatenate(xs), np.concatenate(ys)
    assert len(y) == len(val_ids), (len(y), len(val_ids))
    np.save(out / "x.npy", x); np.save(out / "y.npy", y)

    # 2) reference top-1 of these weights with the training schedule on this machine (GPU if present)
    ref = jax.jit(lambda p, b: model.apply({"params": p}, b, training=False)["logits"].argmax(-1))
    pd = jax.device_put(params)
    pred = np.concatenate([np.asarray(ref(pd, x[i:i + 256])) for i in range(0, len(x), 256)])
    top1_ref = 100 * float(np.mean(pred == y))

    # 3) the device program: batch 1, split-BiGRU schedule, weights baked in, CPU platform
    if a.export_version is not None:
        jax.config.update("jax_export_calling_convention_version", a.export_version)
    with split_bigru():
        f = jax.jit(lambda b: model.apply({"params": params}, b, training=False)["logits"])
        exp = export.export(f, platforms=["cpu"])(jax.ShapeDtypeStruct((1,) + x.shape[1:], np.float32))
    (out / "model.jaxexport").write_bytes(exp.serialize())
    info = {"run": Path(a.run).name, "protocol": a.protocol, "samples": int(len(y)), "input_shape": [1] + list(x.shape[1:]),
            "top1_reference_training_schedule": top1_ref, "schedule": "split BiGRU (edge)",
            "jax_version": jax.__version__, "calling_convention_version": exp.calling_convention_version}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    size = sum(p.stat().st_size for p in out.iterdir()) / 1e9
    print(json.dumps(info, indent=1)); print(f"written {out}  ({size:.2f} GB)")


if __name__ == "__main__":
    main()

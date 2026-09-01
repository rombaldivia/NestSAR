#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Mapping, Any

import jax
import numpy as np
from tqdm.auto import tqdm

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_cdformer_logit_kd_t16.mmaction2_dataset import MMAction2KeypointDataset
from experiments.m4_cdformer_logit_kd_t16.cdformer16_jax_validated import (
    load_cdformer16_weights,
    cdformer16_forward,
)

DEFAULT_TEACHER = Path(
    "/kaggle/input/models/romelbaldivia/cdformer-jax/jax/default/1/"
    "cdformer16_teacher_jax.npz"
)
DEFAULT_OUT = Path("/kaggle/working/cdformer16_mmaction2_teacher_logits.npz")
EXPECTED_XSUB = 0.82456450


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sample_id(a: Mapping[str, Any], i: int) -> str:
    for k in ("frame_dir", "sample_name", "filename", "file_name", "name", "id", "sample_id"):
        v = a.get(k)
        if v is not None and str(v):
            return str(v)
    return f"__index__{i}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--teacher", default=str(DEFAULT_TEACHER))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if jax.default_backend() != "tpu" or jax.local_device_count() != 8:
        raise RuntimeError(
            f"Expected TPU8, got backend={jax.default_backend()} devices={jax.local_device_count()}"
        )
    if args.batch_size % 8:
        raise ValueError("--batch-size must be divisible by 8")

    dataset_path = base.find_dataset(args.dataset)
    teacher_path = Path(args.teacher).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)

    with Path(dataset_path).open("rb") as f:
        try:
            raw = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            raw = pickle.load(f, encoding="latin1")

    annotations = list(raw["annotations"])
    split = raw.get("split", raw.get("splits"))
    if not isinstance(split, Mapping):
        raise RuntimeError("NTU pickle has no split mapping")

    ids = np.asarray([sample_id(a, i) for i, a in enumerate(annotations)], dtype=np.str_)
    if len(set(ids.tolist())) != len(ids):
        raise RuntimeError("Sample IDs are not unique")

    ds = MMAction2KeypointDataset(
        raw,
        list(range(len(annotations))),
        num_frames=16,
        jitter=0,
        drop_tokens=0.0,
        is_train=False,
    )
    if len(ds) != len(annotations):
        raise RuntimeError(f"Dataset length mismatch: {len(ds)} vs {len(annotations)}")

    x0, y0 = ds[0]
    log(f"MMAction2KeypointDataset first x={x0.shape} dtype={x0.dtype} label={int(y0)}")
    if x0.shape != (16, 25, 3) or not np.all(np.isfinite(x0)):
        raise RuntimeError("Bad first teacher sample")

    weights = load_cdformer16_weights(teacher_path)
    total_params = sum(int(v.size) for v in weights.values())
    log(f"CD-Former JAX params={total_params:,}")

    devices = list(jax.local_devices())

    @partial(jax.pmap, devices=devices)
    def p_teacher(x):
        return cdformer16_forward(weights, x)

    logits_all = np.empty((len(ds), 120), np.float16)
    labels_all = np.empty((len(ds),), np.int16)

    bs = int(args.batch_size)
    ndev = len(devices)
    cursor = 0
    for start in tqdm(range(0, len(ds), bs), desc="CD-Former JAX TPU cache", mininterval=0.5):
        end = min(start + bs, len(ds))
        real_n = end - start
        xb = np.zeros((bs, 16, 25, 3), np.float32)
        yb = np.zeros((bs,), np.int16)
        for row, idx in enumerate(range(start, end)):
            x, y = ds[idx]
            xb[row] = x
            yb[row] = y
        xb_shard = xb.reshape(ndev, bs // ndev, 16, 25, 3)
        out = np.asarray(jax.device_get(p_teacher(xb_shard))).reshape(bs, 120)
        if not np.all(np.isfinite(out[:real_n])):
            raise RuntimeError(f"Non-finite teacher logits at rows {start}:{end}")
        logits_all[start:end] = out[:real_n].astype(np.float16)
        labels_all[start:end] = yb[:real_n]
        cursor = end

    if cursor != len(ds):
        raise RuntimeError(f"Teacher cache incomplete: {cursor}/{len(ds)}")

    # Verify labels against annotations.
    ann_labels = np.asarray([base.annotation_label(a) for a in annotations], np.int16)
    if not np.array_equal(labels_all, ann_labels):
        bad = np.flatnonzero(labels_all != ann_labels)[:10]
        raise RuntimeError(f"Teacher dataset label mismatch at {bad.tolist()}")

    # Strict XSUB reproduction check before any KD is allowed.
    _, vk = base.resolve_split(split, "xsub")
    index = {sid: i for i, sid in enumerate(ids.tolist())}
    val_ids = [str(v) for v in split[vk]]
    missing = [sid for sid in val_ids if sid not in index]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} XSUB val IDs from teacher cache")
    pos = np.asarray([index[sid] for sid in val_ids], np.int64)
    pred = np.argmax(logits_all[pos].astype(np.float32), axis=1)
    xsub_acc = float(np.mean(pred == labels_all[pos].astype(np.int32)))
    log(f"CD-Former XSUB cache accuracy={100*xsub_acc:.6f}% ({int(np.sum(pred == labels_all[pos]))}/{len(pos)})")
    log(f"Validated reference={100*EXPECTED_XSUB:.6f}%")

    if abs(xsub_acc - EXPECTED_XSUB) > 0.005:
        raise RuntimeError(
            "CD-Former preprocessing/checkpoint reproduction failed: "
            f"cache={100*xsub_acc:.4f}% expected≈{100*EXPECTED_XSUB:.4f}%"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ids=ids,
        logits=logits_all,
        labels=labels_all,
        teacher_frames=np.asarray([16], np.int16),
        xsub_accuracy=np.asarray([xsub_acc], np.float32),
        teacher_path=np.asarray([str(teacher_path)], dtype=np.str_),
        preprocessing=np.asarray([
            "MMAction2KeypointDataset(data,idx_list,num_frames=16,jitter=0,drop_tokens=0,is_train=False)"
        ], dtype=np.str_),
    )
    meta = {
        "samples": len(ds),
        "logits_shape": list(logits_all.shape),
        "xsub_accuracy": xsub_acc,
        "teacher_params": total_params,
        "teacher_path": str(teacher_path),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    log(f"CACHE SAVED: {output}")
    log("TEACHER CACHE PREFLIGHT=PASS")


if __name__ == "__main__":
    main()

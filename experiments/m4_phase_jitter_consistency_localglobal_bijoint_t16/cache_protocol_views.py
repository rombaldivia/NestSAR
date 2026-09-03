#!/usr/bin/env python3
from __future__ import annotations

"""Build one protocol's exact LocalGlobal canonical/jitter/validation views on disk.

The cache preserves float32 tensors exactly as produced by the verified preprocessing.
It is built one protocol at a time so Kaggle host RAM stays bounded. Training workers
later open these .npy files with mmap_mode='r', allowing XSUB and XSET to train on the
two T4s concurrently without materializing both protocols in RAM.
"""

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np

from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_t16 import train_tpu as cons
from experiments.m4_phase_jitter_consistency_localglobal_t16 import train_tpu as lg_train


class SilentProgress:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else ()
    def __iter__(self):
        return iter(self.iterable)
    def set_postfix(self, *args, **kwargs): return None
    def update(self, *args, **kwargs): return None
    def refresh(self, *args, **kwargs): return None
    def close(self): return None
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", choices=["xsub", "xset"], required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--jitter-max-shift", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)

    lg_train.install_preprocessing_override()
    ju.tqdm = SilentProgress
    cons.tqdm = SilentProgress

    out = Path(args.cache_root) / args.protocol
    files = {
        "Xcan": out / "Xcan.npy",
        "Xjit": out / "Xjit.npy",
        "ytr": out / "ytr.npy",
        "Xva": out / "Xva.npy",
        "yva": out / "yva.npy",
    }
    manifest_path = out / "manifest.json"

    if manifest_path.is_file() and all(p.is_file() for p in files.values()):
        meta = json.loads(manifest_path.read_text())
        if (
            meta.get("protocol") == args.protocol
            and meta.get("seed") == args.seed
            and meta.get("jitter_max_shift") == args.jitter_max_shift
            and meta.get("dtype") == "float32"
        ):
            print(f"CACHE READY/REUSED | {args.protocol.upper()} | {out}", flush=True)
            return 0

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 120, flush=True)
    print(f"BUILD EXACT DISK CACHE | {args.protocol.upper()}", flush=True)
    print("=" * 120, flush=True)
    print("DATASET:", dataset, flush=True)
    print("CACHE:", out, flush=True)
    print("DTYPE: float32 (NO quantization)", flush=True)

    annotations, split = ju.base.load_ntu(dataset)
    protocol_seed = args.seed + (0 if args.protocol == "xsub" else 100000)

    t0 = time.time()
    Xcan, Xjit, ytr, Xva, yva = ju.build_protocol_views(
        annotations,
        split,
        args.protocol,
        args.jitter_max_shift,
        protocol_seed,
        0,
        0,
    )
    print(f"PREPROCESS READY | time={time.time()-t0:.1f}s", flush=True)

    arrays = {"Xcan": Xcan, "Xjit": Xjit, "ytr": ytr, "Xva": Xva, "yva": yva}
    total_bytes = int(sum(a.nbytes for a in arrays.values()))
    free_bytes = shutil.disk_usage(out.parent).free
    print(f"CACHE BYTES REQUIRED: {total_bytes/2**30:.2f} GiB", flush=True)
    print(f"FREE DISK:           {free_bytes/2**30:.2f} GiB", flush=True)
    if free_bytes < total_bytes + 512 * 2**20:
        raise RuntimeError(
            f"Insufficient disk for {args.protocol}: need about {total_bytes/2**30:.2f} GiB "
            f"plus safety margin, free={free_bytes/2**30:.2f} GiB"
        )

    for name, arr in arrays.items():
        np.save(files[name], arr, allow_pickle=False)
        print(f"SAVED {name}: shape={arr.shape} dtype={arr.dtype} size={arr.nbytes/2**30:.2f} GiB", flush=True)

    meta = {
        "protocol": args.protocol,
        "dataset": str(dataset),
        "seed": args.seed,
        "protocol_seed": protocol_seed,
        "jitter_max_shift": args.jitter_max_shift,
        "preprocessing": "local_pose_global_motion_v2",
        "dtype": "float32",
        "exact_no_quantization": True,
        "arrays": {
            name: {"shape": list(arr.shape), "dtype": str(arr.dtype), "nbytes": int(arr.nbytes)}
            for name, arr in arrays.items()
        },
        "total_nbytes": total_bytes,
    }
    manifest_path.write_text(json.dumps(meta, indent=2))

    del Xcan, Xjit, ytr, Xva, yva, arrays, annotations, split
    gc.collect()
    print(f"CACHE_BUILD=PASS | {args.protocol.upper()} | {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

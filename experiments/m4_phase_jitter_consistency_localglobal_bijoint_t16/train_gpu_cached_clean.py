#!/usr/bin/env python3
from __future__ import annotations

"""Concurrent-safe BiJoint worker using exact float32 disk-mapped preprocessing views.

No raw NTU pickle is loaded in the GPU worker. Xcan/Xjit/Xval are opened with
numpy mmap_mode='r', so XSUB and XSET can train concurrently without duplicating
the multi-gigabyte preprocessed tensors in host RAM.
"""

import json
import os
from pathlib import Path

import jax
import numpy as np

from experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16 import train_gpu_memsafe as core


class SilentProgress:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else ()
    def __iter__(self): return iter(self.iterable)
    def set_postfix(self, *args, **kwargs): return None
    def update(self, *args, **kwargs): return None
    def refresh(self, *args, **kwargs): return None
    def close(self): return None
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False


def _cache_dir(protocol: str) -> Path:
    root = os.environ.get("NESTSAR_CACHE_ROOT")
    if not root:
        raise RuntimeError("NESTSAR_CACHE_ROOT is not set")
    return Path(root) / protocol


def _load_cached_views(protocol: str):
    d = _cache_dir(protocol)
    required = [d / n for n in ("Xcan.npy", "Xjit.npy", "ytr.npy", "Xva.npy", "yva.npy", "manifest.json")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(f"Missing cache files for {protocol}: {missing}")

    meta = json.loads((d / "manifest.json").read_text())
    if meta.get("dtype") != "float32" or not meta.get("exact_no_quantization", False):
        raise RuntimeError(f"Cache is not exact float32: {d}")

    Xcan = np.load(d / "Xcan.npy", mmap_mode="r", allow_pickle=False)
    Xjit = np.load(d / "Xjit.npy", mmap_mode="r", allow_pickle=False)
    ytr = np.load(d / "ytr.npy", mmap_mode="r", allow_pickle=False)
    Xva = np.load(d / "Xva.npy", mmap_mode="r", allow_pickle=False)
    yva = np.load(d / "yva.npy", mmap_mode="r", allow_pickle=False)

    print(
        f"{protocol.upper()} MMAP CACHE READY | "
        f"Xcan={Xcan.shape} Xjit={Xjit.shape} Xval={Xva.shape} "
        f"dtype={Xcan.dtype}",
        flush=True,
    )
    return Xcan, Xjit, ytr, Xva, yva


def main() -> None:
    # Keep exact model/loss/training code; only replace the source of preprocessed arrays.
    core.lg_train.install_preprocessing_override()
    core.ju.tqdm = SilentProgress
    core.cons.tqdm = SilentProgress

    args = core.cons.parse_args()
    if args.protocol not in ("xsub", "xset"):
        raise ValueError("Cached worker runs exactly one protocol")

    cache_manifest = json.loads((_cache_dir(args.protocol) / "manifest.json").read_text())
    expected_protocol_seed = args.seed + (0 if args.protocol == "xsub" else 100000)
    if cache_manifest.get("seed") != args.seed:
        raise RuntimeError("Cache seed mismatch")
    if cache_manifest.get("protocol_seed") != expected_protocol_seed:
        raise RuntimeError("Cache protocol seed mismatch")
    if cache_manifest.get("jitter_max_shift") != args.jitter_max_shift:
        raise RuntimeError("Cache jitter setting mismatch")

    def cached_build(_annotations, _split, protocol, max_shift, seed, max_train=0, max_val=0):
        if protocol != args.protocol:
            raise RuntimeError(f"Protocol mismatch: requested={protocol} worker={args.protocol}")
        if max_train or max_val:
            raise RuntimeError("Cached full-run worker does not support max sample truncation")
        if max_shift != args.jitter_max_shift or seed != expected_protocol_seed:
            raise RuntimeError("Cached preprocessing arguments do not match training request")
        return _load_cached_views(protocol)

    core.ju.build_protocol_views = cached_build

    print("=" * 120, flush=True)
    print(
        f"NESTSAR LOCALGLOBAL V2 + BI-JOINT M4/G4 — MMAP CONCURRENT | "
        f"{args.protocol.upper()} | SINGLE T4",
        flush=True,
    )
    print("=" * 120, flush=True)
    print("JAX:", jax.__version__, flush=True)
    print("BACKEND:", jax.default_backend(), flush=True)
    print("DEVICES:", jax.local_devices(), flush=True)
    print("EXPECTED PARAMS:", f"{core.EXPECTED_PARAMS:,}", flush=True)
    print("EFFECTIVE BATCH:", args.batch_size, flush=True)
    print("MICROBATCH:", os.environ.get("NESTSAR_MICROBATCH", core.DEFAULT_MICROBATCH), flush=True)
    print("CACHE MODE: mmap float32 exact", flush=True)
    print("ATTENTION: NONE", flush=True)
    print("TRAINING: FROM SCRATCH", flush=True)

    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU; backend={jax.default_backend()} "
            f"count={jax.local_device_count()}"
        )

    best, epoch, microbatch, accum_steps = core.train_protocol_memsafe(
        args,
        None,
        None,
        args.protocol,
    )

    baseline = core.BASELINE_ACCURACY[args.protocol]
    result = {
        "protocol": args.protocol,
        "best_val_accuracy": best,
        "best_epoch": epoch,
        "baseline_localglobal_v2": baseline,
        "delta_vs_baseline_pp": 100.0 * (best - baseline),
        "expected_params": core.EXPECTED_PARAMS,
        "backend": "gpu",
        "effective_batch": args.batch_size,
        "microbatch": microbatch,
        "gradient_accumulation_steps": accum_steps,
        "seed": args.seed,
        "preprocessing": "local_pose_global_motion_v2",
        "preprocessing_storage": "exact_float32_numpy_mmap",
        "spatial_joint_memory": "bidirectional_bimemory",
        "attention": False,
        "training_from_scratch": True,
        "memory_safe": True,
        "concurrent_dual_t4": True,
    }

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"result_{args.protocol}.json").write_text(json.dumps(result, separators=(",", ":")))
    print("=" * 120, flush=True)
    print("CACHED GPU WORKER DONE", json.dumps(result, separators=(",", ":")), flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""Concurrent-safe BiJoint worker using one shared exact canonical mmap.

The shared cache stores canonical tensors once globally and protocol-specific
jitter tensors separately.  Small index arrays map each protocol's train/val
split into the shared canonical tensor.  Batch indexing materializes only the
current batch, keeping host RAM bounded while both T4 workers run concurrently.
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


class IndexedCanonical:
    """Array-like view mapping protocol positions into shared canonical mmap."""
    def __init__(self, canonical: np.ndarray, indices: np.ndarray):
        self.canonical = canonical
        self.indices = np.asarray(indices, np.int64)
        self.shape = (len(self.indices),) + tuple(canonical.shape[1:])
        self.dtype = canonical.dtype
        self.nbytes = int(len(self.indices) * np.prod(canonical.shape[1:]) * canonical.dtype.itemsize)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        mapped = self.indices[key]
        return np.asarray(self.canonical[mapped])


def _root() -> Path:
    value = os.environ.get("NESTSAR_SHARED_CACHE_ROOT")
    if not value:
        raise RuntimeError("NESTSAR_SHARED_CACHE_ROOT is not set")
    return Path(value)


def _load(protocol: str):
    root = _root()
    d = root / protocol
    required = [
        root / "canonical.npy",
        root / "canonical_labels.npy",
        root / "canonical_manifest.json",
        d / "Xjit.npy",
        d / "train_idx.npy",
        d / "val_idx.npy",
        d / "ytr.npy",
        d / "yva.npy",
        d / "manifest.json",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(f"Missing shared cache files: {missing}")

    cmeta = json.loads((root / "canonical_manifest.json").read_text())
    pmeta = json.loads((d / "manifest.json").read_text())
    if cmeta.get("dtype") != "float32" or not cmeta.get("exact_no_quantization", False):
        raise RuntimeError("Canonical cache is not exact float32")
    if pmeta.get("dtype") != "float32" or not pmeta.get("exact_no_quantization", False):
        raise RuntimeError(f"{protocol} jitter cache is not exact float32")

    canonical = np.load(root / "canonical.npy", mmap_mode="r", allow_pickle=False)
    train_idx = np.load(d / "train_idx.npy", mmap_mode="r", allow_pickle=False)
    val_idx = np.load(d / "val_idx.npy", mmap_mode="r", allow_pickle=False)
    Xjit = np.load(d / "Xjit.npy", mmap_mode="r", allow_pickle=False)
    ytr = np.load(d / "ytr.npy", mmap_mode="r", allow_pickle=False)
    yva = np.load(d / "yva.npy", mmap_mode="r", allow_pickle=False)

    Xcan = IndexedCanonical(canonical, train_idx)
    Xva = IndexedCanonical(canonical, val_idx)

    print(
        f"{protocol.upper()} SHARED MMAP READY | canonical_global={canonical.shape} "
        f"train={Xcan.shape} jitter={Xjit.shape} val={Xva.shape} dtype={canonical.dtype}",
        flush=True,
    )
    return Xcan, Xjit, ytr, Xva, yva, pmeta


def main() -> None:
    core.lg_train.install_preprocessing_override()
    core.ju.tqdm = SilentProgress
    core.cons.tqdm = SilentProgress

    args = core.cons.parse_args()
    if args.protocol not in ("xsub", "xset"):
        raise ValueError("Shared-cache worker runs exactly one protocol")

    Xcan, Xjit, ytr, Xva, yva, meta = _load(args.protocol)
    expected_protocol_seed = args.seed + (0 if args.protocol == "xsub" else 100000)
    if meta.get("seed") != args.seed:
        raise RuntimeError("Cache seed mismatch")
    if meta.get("protocol_seed") != expected_protocol_seed:
        raise RuntimeError("Cache protocol seed mismatch")
    if meta.get("jitter_max_shift") != args.jitter_max_shift:
        raise RuntimeError("Cache jitter setting mismatch")

    def cached_build(_annotations, _split, protocol, max_shift, seed, max_train=0, max_val=0):
        if protocol != args.protocol:
            raise RuntimeError(f"Protocol mismatch: {protocol} vs {args.protocol}")
        if max_train or max_val:
            raise RuntimeError("Shared cached full-run worker does not support truncation")
        if max_shift != args.jitter_max_shift or seed != expected_protocol_seed:
            raise RuntimeError("Shared cache preprocessing arguments mismatch")
        return Xcan, Xjit, ytr, Xva, yva

    core.ju.build_protocol_views = cached_build

    print("=" * 120, flush=True)
    print(
        f"NESTSAR LOCALGLOBAL V2 + BI-JOINT M4/G4 — SHARED MMAP CONCURRENT | "
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
    print("CACHE: shared canonical float32 mmap + protocol jitter float32 mmap", flush=True)
    print("ATTENTION: NONE", flush=True)
    print("TRAINING: FROM SCRATCH", flush=True)

    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU; backend={jax.default_backend()} count={jax.local_device_count()}"
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
        "preprocessing_storage": "shared_exact_float32_numpy_mmap",
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
    print("SHARED-CACHE GPU WORKER DONE", json.dumps(result, separators=(",", ":")), flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()

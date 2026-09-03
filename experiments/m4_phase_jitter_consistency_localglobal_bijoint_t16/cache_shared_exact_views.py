#!/usr/bin/env python3
from __future__ import annotations

"""Build a disk-efficient exact float32 cache shared by XSUB and XSET.

Unlike the first concurrent cache (which duplicated canonical tensors separately
for each protocol), this cache stores every used sample's canonical LocalGlobal
T16 tensor exactly once, plus one protocol-specific jitter tensor for each
training split.  This reduces cache storage from roughly 15.4 GiB to about
10.4 GiB while preserving exact float32 preprocessing.

All large arrays are written incrementally with numpy open_memmap, so cache
construction itself never materializes multi-gigabyte preprocessing arrays in
host RAM.
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
from experiments.m4_phase_jitter_consistency_localglobal_t16 import preprocessing as lg

FRAMES = ju.FRAMES
FEATURES = ju.FEATURES


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--jitter-max-shift", type=int, default=1)
    return p.parse_args()


def _need_bytes(n: int) -> int:
    return int(n) * FRAMES * FEATURES * np.dtype(np.float32).itemsize


def _check_space(root: Path, need: int, label: str) -> None:
    free = shutil.disk_usage(root).free
    margin = 512 * 2**20
    print(
        f"DISK CHECK {label}: need={need/2**30:.2f} GiB "
        f"free={free/2**30:.2f} GiB margin={margin/2**30:.2f} GiB",
        flush=True,
    )
    if free < need + margin:
        raise RuntimeError(
            f"Insufficient disk for {label}: need {need/2**30:.2f} GiB + "
            f"0.50 GiB margin, free={free/2**30:.2f} GiB"
        )


def _manifest_valid(path: Path, **expected) -> bool:
    if not path.is_file():
        return False
    try:
        meta = json.loads(path.read_text())
    except Exception:
        return False
    return all(meta.get(k) == v for k, v in expected.items())


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset)
    root = Path(args.cache_root)
    root.mkdir(parents=True, exist_ok=True)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)

    lg_train.install_preprocessing_override()
    ju.tqdm = SilentProgress
    cons.tqdm = SilentProgress

    print("=" * 120, flush=True)
    print("NESTSAR BIJOINT SHARED EXACT FLOAT32 CACHE", flush=True)
    print("=" * 120, flush=True)
    print("DATASET:", dataset, flush=True)
    print("CACHE ROOT:", root, flush=True)
    print("DTYPE: float32 exact (NO quantization)", flush=True)

    annotations, split = ju.base.load_ntu(dataset)
    by_id, xs_tr, xs_va = ju.resolve_protocol_ids(annotations, split, "xsub")
    by_id2, xe_tr, xe_va = ju.resolve_protocol_ids(annotations, split, "xset")
    if set(by_id) != set(by_id2):
        raise RuntimeError("Protocol ID maps unexpectedly differ")

    used = set(xs_tr) | set(xs_va) | set(xe_tr) | set(xe_va)
    canonical_ids = [sid for sid in by_id.keys() if sid in used]
    index = {sid: i for i, sid in enumerate(canonical_ids)}

    canonical_path = root / "canonical.npy"
    labels_path = root / "canonical_labels.npy"
    canonical_manifest = root / "canonical_manifest.json"

    canonical_ok = (
        canonical_path.is_file()
        and labels_path.is_file()
        and _manifest_valid(
            canonical_manifest,
            dtype="float32",
            exact_no_quantization=True,
            count=len(canonical_ids),
            frames=FRAMES,
            features=FEATURES,
        )
    )

    if canonical_ok:
        print(f"CANONICAL CACHE REUSED | count={len(canonical_ids):,}", flush=True)
    else:
        for p in (canonical_path, labels_path, canonical_manifest):
            if p.exists(): p.unlink()
        need = _need_bytes(len(canonical_ids)) + len(canonical_ids) * 4
        _check_space(root, need, "shared canonical cache")
        X = np.lib.format.open_memmap(
            canonical_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(canonical_ids), FRAMES, FEATURES),
        )
        y = np.lib.format.open_memmap(
            labels_path,
            mode="w+",
            dtype=np.int32,
            shape=(len(canonical_ids),),
        )
        t0 = time.time()
        for i, sid in enumerate(canonical_ids):
            a = by_id[sid]
            kp = ju.base.annotation_keypoints(a)
            X[i] = lg.segment_phase_tokens_localglobal(kp)
            y[i] = ju.base.annotation_label(a)
            if (i + 1) % 5000 == 0 or i + 1 == len(canonical_ids):
                print(
                    f"CANONICAL {i+1:,}/{len(canonical_ids):,} | "
                    f"{100*(i+1)/len(canonical_ids):.1f}% | "
                    f"time={time.time()-t0:.1f}s",
                    flush=True,
                )
        X.flush(); y.flush()
        del X, y
        canonical_manifest.write_text(json.dumps({
            "dtype": "float32",
            "exact_no_quantization": True,
            "count": len(canonical_ids),
            "frames": FRAMES,
            "features": FEATURES,
            "preprocessing": "local_pose_global_motion_v2",
            "storage": "shared_canonical_open_memmap",
        }, indent=2))
        print("CANONICAL_CACHE=PASS", flush=True)

    # Small mapping files are rebuilt deterministically every time.
    for protocol, train_ids, val_ids in (
        ("xsub", xs_tr, xs_va),
        ("xset", xe_tr, xe_va),
    ):
        d = root / protocol
        d.mkdir(parents=True, exist_ok=True)
        train_idx = np.asarray([index[s] for s in train_ids], np.int32)
        val_idx = np.asarray([index[s] for s in val_ids], np.int32)
        np.save(d / "train_idx.npy", train_idx, allow_pickle=False)
        np.save(d / "val_idx.npy", val_idx, allow_pickle=False)

        canonical_labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        ytr = np.asarray(canonical_labels[train_idx], np.int32)
        yva = np.asarray(canonical_labels[val_idx], np.int32)
        np.save(d / "ytr.npy", ytr, allow_pickle=False)
        np.save(d / "yva.npy", yva, allow_pickle=False)
        del canonical_labels, ytr, yva

        protocol_seed = args.seed + (0 if protocol == "xsub" else 100000)
        jitter_path = d / "Xjit.npy"
        manifest = d / "manifest.json"
        jitter_ok = (
            jitter_path.is_file()
            and _manifest_valid(
                manifest,
                protocol=protocol,
                seed=args.seed,
                protocol_seed=protocol_seed,
                jitter_max_shift=args.jitter_max_shift,
                dtype="float32",
                exact_no_quantization=True,
                train_count=len(train_ids),
                val_count=len(val_ids),
            )
        )
        if jitter_ok:
            print(f"{protocol.upper()} JITTER CACHE REUSED | train={len(train_ids):,}", flush=True)
            continue

        if jitter_path.exists(): jitter_path.unlink()
        if manifest.exists(): manifest.unlink()
        _check_space(root, _need_bytes(len(train_ids)), f"{protocol.upper()} jitter cache")
        Xj = np.lib.format.open_memmap(
            jitter_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(train_ids), FRAMES, FEATURES),
        )
        t0 = time.time()
        for i, sid in enumerate(train_ids):
            a = by_id[sid]
            kp = ju.base.annotation_keypoints(a)
            rng = np.random.default_rng(np.random.SeedSequence([protocol_seed, i, 9173]))
            Xj[i] = ju.jitter_phase_tokens(kp, args.jitter_max_shift, rng)
            if (i + 1) % 5000 == 0 or i + 1 == len(train_ids):
                print(
                    f"{protocol.upper()} JITTER {i+1:,}/{len(train_ids):,} | "
                    f"{100*(i+1)/len(train_ids):.1f}% | time={time.time()-t0:.1f}s",
                    flush=True,
                )
        Xj.flush(); del Xj
        manifest.write_text(json.dumps({
            "protocol": protocol,
            "seed": args.seed,
            "protocol_seed": protocol_seed,
            "jitter_max_shift": args.jitter_max_shift,
            "dtype": "float32",
            "exact_no_quantization": True,
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "canonical_storage": "shared_global_exact_float32",
            "jitter_storage": "protocol_specific_exact_float32",
            "preprocessing": "local_pose_global_motion_v2",
        }, indent=2))
        print(f"{protocol.upper()}_JITTER_CACHE=PASS", flush=True)

    del annotations, split, by_id, by_id2
    gc.collect()
    print("=" * 120, flush=True)
    print("SHARED_EXACT_CACHE=PASS", flush=True)
    print("=" * 120, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pre-computed strong-augmentation pool.

Strong augmentation costs ~2 ms of CPU per view and a training step needs two views per clip, so on a
2-core Kaggle box the epoch is CPU-bound (13-17 min against ~70 s of GPU). The pool pays that cost once:
``views`` independent strongly augmented copies of every clip are written to disk as [N, 16, 942]
float32 files, and training draws two *different* views per clip and epoch from them (plus the clean
canonical view with probability ``aug_clean_prob``, exactly as the on-the-fly path does).

Trade-off: the views are finite (``views`` per clip) instead of fresh every epoch, so the regularisation is
a little weaker than live augmentation; the epoch drops to GPU speed and the pool is reused by every run
that shares the cache, augmentation strength, seed and view range.

    python -m experiments.nestsar_r5_t16.augpool build --cache C --pool P --views 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np

from experiments.nestsar_r5_t16 import preprocessing as r5pp

POOL_VERSION = 1
MANIFEST = "pool_manifest.json"
POOL_SEED = 20240


def view_file(pool_dir, view):
    return Path(pool_dir) / f"view_{view:02d}.npy"


def signature(cache_meta, views, strength, view_degrees, seed):
    return {"pool_version": POOL_VERSION, "preprocessing": r5pp.VERSION, "cache": cache_meta["signature"],
            "samples": int(cache_meta["samples"]), "views": int(views), "strength": float(strength),
            "view_degrees": float(view_degrees), "seed": int(seed)}


_DATASET = {}


def _dataset(cache):
    if cache not in _DATASET:
        from experiments.nestsar_r5_t16.data import Dataset
        _DATASET[cache] = Dataset(cache)
    return _DATASET[cache]


def _fill(args):
    """Worker: augment samples [a, b) as view ``view`` straight into the shared memmap."""
    cache, pool_dir, view, a, b, strength, view_degrees, seed = args
    ds = _dataset(cache)
    out = np.load(view_file(pool_dir, view).with_suffix(".partial.npy"), mmap_mode="r+")
    for i in range(a, b):
        out[i] = r5pp.strong_augmented_features(
            ds.base.sample(i), seed, view + 1, i, strength, stream=0, shift=1, hand_filter=ds.hand_filter,
            view_degrees=view_degrees, body_align=ds.body_align)
    out.flush()
    return b - a


def _fmt(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def _progress(view, views, finished, total, elapsed, done_view=False):
    rate = finished / elapsed if elapsed > 0 else 0.0
    eta = (total - finished) / rate if rate > 0 else 0.0
    bar = "#" * int(20 * finished / max(total, 1))
    return (f"  [{bar:<20}] {100 * finished / max(total, 1):5.1f}%  view {view + 1}/{views}"
            f"{' done' if done_view else '     '}  {rate:5.0f} views/s  elapsed {_fmt(elapsed)}  ETA {_fmt(eta)}")


def build(cache, pool_dir, views=8, strength=1.0, view_degrees=15.0, seed=POOL_SEED, workers=None,
          chunk=256, log=print, progress_every=15.0):
    from experiments.nestsar_r5_t16.data import Dataset, read_manifest
    cache, pool_dir = str(cache), Path(pool_dir)
    meta = read_manifest(Path(cache))
    ds = Dataset(cache)
    n = int(meta["samples"])
    sig = signature(meta, views, strength, view_degrees, seed)
    pool_dir.mkdir(parents=True, exist_ok=True)
    mf = pool_dir / MANIFEST
    if mf.is_file():
        old = json.loads(mf.read_text())
        if old.get("signature") != sig:
            raise ValueError(f"{pool_dir} holds a pool with another signature; use a new directory.")
        if old.get("complete"):
            log(f"pool already complete: {pool_dir}")
            return old
    need = views * n * r5pp.FRAMES * r5pp.FEATURES * 4
    free = shutil.disk_usage(pool_dir).free
    done_bytes = sum(view_file(pool_dir, v).stat().st_size for v in range(views) if view_file(pool_dir, v).is_file())
    if free + done_bytes < need + 2 ** 30:
        raise RuntimeError(f"The pool needs {need / 2**30:.1f} GiB; only {(free + done_bytes) / 2**30:.1f} GiB free "
                           f"in {pool_dir}. Use fewer --views or a bigger disk (/tmp on Kaggle).")
    workers = workers or max(1, mp.cpu_count())
    pending = [v for v in range(views) if not view_file(pool_dir, v).is_file()]
    total_views = len(pending) * n                      # views already finished do not count towards the ETA
    finished = 0
    ctx = mp.get_context("spawn")                     # no fork after JAX/threads: workers import numpy code only
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for v in range(views):
            final = view_file(pool_dir, v)
            if final.is_file():                       # finished earlier (atomic rename below)
                continue
            partial = final.with_suffix(".partial.npy")
            mm = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                           shape=(n, r5pp.FRAMES, r5pp.FEATURES))
            del mm
            jobs = [(cache, str(pool_dir), v, a, min(a + chunk, n), strength, view_degrees, seed)
                    for a in range(0, n, chunk)]
            last = time.time()
            for count in ex.map(_fill, jobs):
                finished += count
                if time.time() - last >= progress_every:
                    last = time.time()
                    log(_progress(v, views, finished, total_views, time.time() - t0))
            partial.replace(final)
            log(_progress(v, views, finished, total_views, time.time() - t0, done_view=True))
    out = {"signature": sig, "complete": True, "build_seconds": time.time() - t0}
    mf.write_text(json.dumps(out))
    return out


class Pool:
    """Read side: memory-mapped views, validated against the dataset that will consume them."""

    def __init__(self, pool_dir, dataset):
        self.dir = Path(pool_dir)
        mf = self.dir / MANIFEST
        if not mf.is_file():
            raise ValueError(f"{pool_dir} is not an augmentation pool")
        m = json.loads(mf.read_text())
        if not m.get("complete") or m["signature"].get("pool_version") != POOL_VERSION:
            raise ValueError(f"Augmentation pool {pool_dir} is incomplete or from another version")
        sig = m["signature"]
        if sig["preprocessing"] != r5pp.VERSION or sig["cache"] != dataset.meta["signature"] \
                or sig["samples"] != dataset.meta["samples"]:
            raise ValueError("The augmentation pool was built for another cache / preprocessing")
        self.signature, self.views = sig, int(sig["views"])
        self.arrays = [np.load(view_file(pool_dir, v), mmap_mode="r") for v in range(self.views)]

    def get(self, view, indices):
        return np.asarray(self.arrays[int(view)][np.asarray(indices, np.int64)], np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cache", required=True)
    b.add_argument("--pool", required=True)
    b.add_argument("--views", type=int, default=8)
    b.add_argument("--strength", type=float, default=1.0)
    b.add_argument("--view-degrees", type=float, default=15.0)
    b.add_argument("--seed", type=int, default=POOL_SEED)
    b.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    out = build(a.cache, a.pool, a.views, a.strength, a.view_degrees, a.seed, a.workers,
                log=lambda m: print(m, flush=True))
    print(json.dumps({"complete": out["complete"], "views": a.views}))


if __name__ == "__main__":
    main()

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
import os
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


DTYPES = {"float32": np.float32, "float16": np.float16}


def signature(cache_meta, views, strength, view_degrees, seed, dtype="float32"):
    sig = {"pool_version": POOL_VERSION, "preprocessing": r5pp.VERSION, "cache": cache_meta["signature"],
           "samples": int(cache_meta["samples"]), "views": int(views), "strength": float(strength),
           "view_degrees": float(view_degrees), "seed": int(seed)}
    if dtype != "float32":            # absent == float32, so pools built before keep their signature
        sig["dtype"] = dtype
    return sig


def parse_views(text, views):
    """'0,1,2,3' or '0-3' -> sorted view indices (None/'' = all views)."""
    if not text:
        return list(range(views))
    out = set()
    for part in str(text).split(","):
        a, _, b = part.strip().partition("-")
        out.update(range(int(a), int(b) + 1) if b else [int(a)])
    if not out or min(out) < 0 or max(out) >= views:
        raise ValueError(f"views {text!r} outside 0..{views - 1}")
    return sorted(out)


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
          chunk=256, log=print, progress_every=15.0, dtype="float32", only_views=None):
    """Build views ``only_views`` (default: all) of a ``views``-view pool. A pool can be split over several
    directories (e.g. two Kaggle outputs under the 20 GB limit); ``Pool`` joins them by signature."""
    from experiments.nestsar_r5_t16.data import Dataset, read_manifest
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    cache, pool_dir = str(cache), Path(pool_dir)
    meta = read_manifest(Path(cache))
    ds = Dataset(cache)
    n = int(meta["samples"])
    sig = signature(meta, views, strength, view_degrees, seed, dtype)
    wanted = list(range(views)) if only_views is None else sorted(set(only_views))
    pool_dir.mkdir(parents=True, exist_ok=True)
    mf = pool_dir / MANIFEST
    if mf.is_file():
        old = json.loads(mf.read_text())
        if old.get("signature") != sig:
            raise ValueError(f"{pool_dir} holds a pool with another signature; use a new directory.")
        if old.get("complete") and old.get("views_present", list(range(views))) == wanted:
            log(f"pool already complete: {pool_dir}")
            return old
    itemsize = np.dtype(DTYPES[dtype]).itemsize
    need = len(wanted) * n * r5pp.FRAMES * r5pp.FEATURES * itemsize
    free = shutil.disk_usage(pool_dir).free
    done_bytes = sum(view_file(pool_dir, v).stat().st_size for v in wanted if view_file(pool_dir, v).is_file())
    if free + done_bytes < need + 2 ** 30:
        raise RuntimeError(f"The pool needs {need / 2**30:.1f} GiB; only {(free + done_bytes) / 2**30:.1f} GiB free "
                           f"in {pool_dir}. Use fewer views, float16 or a bigger disk (/tmp on Kaggle).")
    workers = workers or max(1, mp.cpu_count())
    pending = [v for v in wanted if not view_file(pool_dir, v).is_file()]
    total_views = len(pending) * n                      # views already finished do not count towards the ETA
    finished = 0
    ctx = mp.get_context("spawn")                     # no fork after JAX/threads: workers import numpy code only
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for v in wanted:
            final = view_file(pool_dir, v)
            if final.is_file():                       # finished earlier (atomic rename below)
                continue
            partial = final.with_suffix(".partial.npy")
            mm = np.lib.format.open_memmap(partial, mode="w+", dtype=DTYPES[dtype],
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
    out = {"signature": sig, "complete": True, "views_present": wanted, "build_seconds": time.time() - t0}
    mf.write_text(json.dumps(out))
    return out


def pool_dirs(spec):
    """'a:b' / Path / list -> list of pool directories."""
    if isinstance(spec, (list, tuple)):
        return [Path(p) for p in spec]
    return [Path(p) for p in str(spec).split(os.pathsep) if p]


def read_part(pool_dir):
    """(signature, views present) of a finished pool directory, or None."""
    mf = Path(pool_dir) / MANIFEST
    if not mf.is_file():
        return None
    m = json.loads(mf.read_text())
    if not m.get("complete") or m.get("signature", {}).get("pool_version") != POOL_VERSION:
        return None
    sig = m["signature"]
    return sig, list(m.get("views_present", range(int(sig["views"]))))


def find_parts(roots, sig, max_depth=4):
    """Finished pool directories under ``roots`` whose signature equals ``sig``: {view: dir}."""
    found = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for mf in root.rglob(MANIFEST):
            if len(mf.relative_to(root).parts) > max_depth + 1:
                continue
            part = read_part(mf.parent)
            if part and part[0] == sig:
                for v in part[1]:
                    if view_file(mf.parent, v).is_file():
                        found.setdefault(v, mf.parent)
    return found


class Pool:
    """Read side: memory-mapped views (float32 or float16 on disk, float32 out), possibly spread over several
    directories, validated against the dataset that will consume them."""

    def __init__(self, pool_dir, dataset):
        dirs = pool_dirs(pool_dir)
        where, sig = {}, None
        for d in dirs:
            part = read_part(d)
            if part is None:
                raise ValueError(f"{d} is not a finished augmentation pool (or another pool version)")
            if sig is not None and part[0] != sig:
                raise ValueError(f"{d} belongs to another pool than {dirs[0]}")
            sig = part[0]
            for v in part[1]:
                where.setdefault(v, d)
        if sig["preprocessing"] != r5pp.VERSION or sig["cache"] != dataset.meta["signature"] \
                or sig["samples"] != dataset.meta["samples"]:
            raise ValueError("The augmentation pool was built for another cache / preprocessing")
        missing = sorted(set(range(int(sig["views"]))) - set(where))
        if missing:
            raise ValueError(f"Augmentation pool views {missing} are missing (add the other pool part)")
        self.signature, self.views = sig, int(sig["views"])
        self.dirs = dirs
        self.arrays = [np.load(view_file(where[v], v), mmap_mode="r") for v in range(self.views)]

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
    b.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    b.add_argument("--only-views", default="", help="build only these views, e.g. 0-3 (split a pool over outputs)")
    a = ap.parse_args(argv)
    Path(a.pool).mkdir(parents=True, exist_ok=True)
    logfile = Path(a.pool) / "progress.log"          # tail -f it from a terminal when the cell shows nothing

    def log(message):
        print(message, flush=True)
        with logfile.open("a") as fh:
            fh.write(time.strftime("%H:%M:%S ") + message + "\n")

    out = build(a.cache, a.pool, a.views, a.strength, a.view_degrees, a.seed, a.workers, log=log,
                dtype=a.dtype, only_views=parse_views(a.only_views, a.views))
    print(json.dumps({"complete": out["complete"], "views": a.views}))


if __name__ == "__main__":
    main()

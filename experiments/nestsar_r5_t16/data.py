"""R5 data: an R4 cache (raw + canonical tokens) plus a derived hand cache.

The derived cache stores only the 4x-rate hand block of the canonical view
([N, 16, 192] float32, ~1.4 GiB for NTU120). Raw skeletons, labels, splits and
the canonical R4 tokens are read from the R4 cache, so nothing is duplicated.
While building, every sample's recomputed R4 tokens are compared with the
stored canonical tokens; any difference stops the build (the base cache would
belong to another preprocessing).

Batches: x  = [R4 canonical tokens | hand block]            [B, 16, 942]
         xa = fresh augmented view (same seeds, yaw and boundary jitter
              as the R4 pipeline), computed from raw in one producer thread.
"""
from __future__ import annotations

import concurrent.futures
import fcntl
import glob
import json
import mmap
import os
import shutil
import time
from pathlib import Path

import numpy as np

from experiments.nestsar_sm_all_t16.streaming import data as r4_data
from experiments.nestsar_sm_all_t16.streaming.io_utils import Reporter, atomic_json

from . import preprocessing as r5pp

CACHE_VERSION = "nestsar-r5-hand-cache-v1"
HAND_FILE = "hand.npy"
FULL_FILE = "full.npy"          # [N, 16, 942] features, only for body-aligned caches (the R4 tokens change)


def read_manifest(path):
    try:
        return json.loads((Path(path) / "manifest.json").read_text())
    except (OSError, ValueError):
        return None


def is_r4_cache(path):
    meta = read_manifest(path)
    return bool(meta) and isinstance(meta.get("signature"), dict) and \
        meta["signature"].get("preprocessing") == r4_data.pp.VERSION and \
        (Path(path) / "canonical.npy").is_file()


def candidate_r4_caches(roots, preferred=()):
    """Directories under `roots` (depth <= 3) holding an R4 canonical cache."""
    seen, out = set(), []
    for root in roots:
        root = Path(root)
        names = [root / p for p in preferred]
        for pattern in ("*/manifest.json", "*/*/manifest.json", "*/*/*/manifest.json"):
            names += sorted(Path(m).parent for m in glob.glob(str(root / pattern)))
        for d in names:
            d = d.resolve() if d.exists() else d
            if d in seen or not is_r4_cache(d):
                continue
            seen.add(d)
            out.append(d)
    return out


def _signature(base, hand_filter="none", body_align="none"):
    sig = {
        "base_signature": base.meta["signature"],
        "base_files": base.meta["files"],
        "base_samples": int(base.meta["samples"]),
        "preprocessing": r5pp.VERSION,
        "cache_version": CACHE_VERSION,
    }
    if hand_filter != "none":          # absent == "none", so caches built before the filter keep loading
        sig["hand_filter"] = hand_filter
    if body_align != "none":
        sig["body_align"] = body_align
    return sig


def _existing(cache_dir, signature, base_dir):
    """Validated manifest of a finished hand cache, or None. Writes nothing unless it can."""
    meta = read_manifest(cache_dir)
    if meta is None:
        return None
    if meta.get("signature") != signature:
        raise ValueError(f"{cache_dir} holds a hand cache for another base cache/preprocessing. "
                         "Use a new hand-cache directory.")
    name = FULL_FILE if "body_align" in meta.get("signature", {}) else HAND_FILE
    path = Path(cache_dir) / name
    if not path.is_file() or path.stat().st_size != meta["files"][name]:
        raise ValueError(f"Incomplete hand cache in {cache_dir}. Use a new directory.")
    if meta.get("base_dir") != str(base_dir):
        meta["base_dir"] = str(base_dir)
        try:
            atomic_json(Path(cache_dir) / "manifest.json", meta)
        except OSError:                    # read-only (e.g. /kaggle/input): base is located at load time
            pass
    return meta


def build(base_dir, cache_dir, status=None, check_body=True, hand_filter="none", body_align="none"):
    """Create (or validate) the derived hand cache for `base_dir` in `cache_dir`."""
    base_dir, cache_dir = Path(base_dir).resolve(), Path(cache_dir)
    base = r4_data.Dataset(base_dir)          # validates the R4 manifest and file sizes
    signature = _signature(base, hand_filter, body_align)
    aligned = body_align != "none"
    out_file = FULL_FILE if aligned else HAND_FILE
    width = r5pp.FEATURES if aligned else r5pp.HAND_FEATURES
    n = int(base.meta["samples"])
    done = _existing(cache_dir, signature, base_dir)
    if done is not None:
        return done
    cache_dir.mkdir(parents=True, exist_ok=True)
    report = Reporter(status or (cache_dir / "build_status.json"))
    report(phase="R5 hand cache lock", current=0, total=1)
    with (cache_dir / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        done = _existing(cache_dir, signature, base_dir)     # another process may have finished it
        if done is not None:
            report(phase="R5 hand cache ready", current=n, total=n, done=True)
            return done

        need = n * r5pp.FRAMES * width * 4
        free = shutil.disk_usage(cache_dir).free
        if free < need + 512 * 2 ** 20:
            raise RuntimeError(f"Hand cache needs {need / 2**30:.2f} GiB (+0.5 GiB reserve); "
                               f"{free / 2**30:.2f} GiB free in {cache_dir}.")
        tmp = cache_dir / (out_file + ".partial.npy")
        hand = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                         shape=(n, r5pp.FRAMES, width))
        worst = 0.0
        t0 = time.time()
        for i in range(n):
            tokens = r5pp.features(base.sample(i), hand_filter=hand_filter, body_align=body_align)
            if check_body and not aligned:   # aligned tokens differ from the R4 cache by design
                diff = float(np.max(np.abs(tokens[:, :r5pp.R4_FEATURES] - base.canonical[i])))
                worst = max(worst, diff)
                if diff > 1e-5:
                    raise ValueError(f"Sample {i}: recomputed R4 tokens differ from {base_dir} by {diff:.3g}. "
                                     "The base cache was built with another preprocessing.")
            hand[i] = tokens if aligned else tokens[:, r5pp.R4_FEATURES:]
            if (i + 1) % 2048 == 0:
                hand.flush()
                if hasattr(hand._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                    hand._mmap.madvise(mmap.MADV_DONTNEED)
            if i % 500 == 0 or i + 1 == n:
                report(phase="R5 hand cache", current=i + 1, total=n, elapsed_s=time.time() - t0)
        hand.flush()
        del hand
        os.replace(tmp, cache_dir / out_file)
        meta = {
            "signature": signature,
            "samples": n,
            "base_dir": str(base_dir),
            "files": {out_file: (cache_dir / out_file).stat().st_size},
            "max_body_token_difference": worst,
            "build_seconds": time.time() - t0,
        }
        atomic_json(cache_dir / "manifest.json", meta)   # completion marker, written last
        report(phase="R5 hand cache ready", current=n, total=n, done=True)
        return meta


def locate_base(meta, hints=()):
    """Find the R4 cache a hand cache was derived from (it may have moved)."""
    want = meta["signature"]
    for d in [meta.get("base_dir")] + list(hints):
        if d and is_r4_cache(d):
            base_meta = read_manifest(d)
            if base_meta["signature"] == want["base_signature"] and base_meta["files"] == want["base_files"]:
                return Path(d)
    roots = [r for r in ("/kaggle/working", "/kaggle/input") if Path(r).is_dir()]
    for d in candidate_r4_caches(roots):
        base_meta = read_manifest(d)
        if base_meta["signature"] == want["base_signature"] and base_meta["files"] == want["base_files"]:
            return d
    raise FileNotFoundError("The R4 base cache of this hand cache was not found "
                            f"(was {meta.get('base_dir')}). Rebuild the hand cache from an available R4 cache.")


class Dataset:
    """Same interface as the R4 streaming Dataset, with 942-feature tokens."""

    def __init__(self, cache, base=None):
        cache = Path(cache)
        meta = read_manifest(cache)
        if meta is None or meta.get("signature", {}).get("cache_version") != CACHE_VERSION:
            raise ValueError(f"{cache} is not an R5 hand cache")
        if meta["signature"].get("preprocessing") != r5pp.VERSION:
            raise ValueError("Hand cache preprocessing/version mismatch")
        self.body_align = meta["signature"].get("body_align", "none")
        name = FULL_FILE if self.body_align != "none" else HAND_FILE
        path = cache / name
        if not path.is_file() or path.stat().st_size != meta["files"][name]:
            raise ValueError(f"Incomplete hand cache: {path}")
        self.base_dir = locate_base(meta, [base] if base else [])
        self.base = r4_data.Dataset(self.base_dir)
        self.hand = np.load(path, mmap_mode="r")
        width = r5pp.FEATURES if self.body_align != "none" else r5pp.HAND_FEATURES
        if self.hand.shape != (meta["samples"], r5pp.FRAMES, width):
            raise ValueError(f"Unexpected hand cache shape {self.hand.shape}")
        if len(self.base.labels) != meta["samples"]:
            raise ValueError("Base cache and hand cache sample counts differ")
        self.hand_filter = meta["signature"].get("hand_filter", "none")
        self.meta = {"signature": meta["signature"], "samples": meta["samples"]}
        self.splits = self.base.splits
        self.labels = self.base.labels
        self._pools = {}

    def pool(self, path):
        """Opened lazily and once per path; raises if it was built for another cache/preprocessing."""
        if path not in self._pools:
            from experiments.nestsar_r5_t16.augpool import Pool
            self._pools[path] = Pool(path, self)
        return self._pools[path]

    def canonical(self, indices):
        indices = np.asarray(indices, np.int64)
        if self.body_align != "none":                    # full 942-feature tokens are stored
            return np.array(self.hand[indices], np.float32)
        out = np.empty((len(indices), r5pp.FRAMES, r5pp.FEATURES), np.float32)
        out[..., :r5pp.R4_FEATURES] = self.base.canonical[indices]
        out[..., r5pp.R4_FEATURES:] = self.hand[indices]
        return out

    def batch(self, indices, positions, size, config, epoch, training, protocol):
        t0 = time.perf_counter()
        b = {"x": np.zeros((size, r5pp.FRAMES, r5pp.FEATURES), np.float32),
             "y": np.zeros(size, np.int32), "mask": np.zeros(size, np.float32)}
        b["x"][:len(indices)] = self.canonical(indices)
        b["y"][:len(indices)] = self.labels[indices]
        b["mask"][:len(indices)] = 1
        if training:
            b["xa"] = np.zeros_like(b["x"])
            protocol_seed = config["seed"] + (100000 if protocol == "xset" else 0)
            strength = config.get("aug_strength", 0.0)
            aug_epoch = epoch if config["fresh_augmentation"] else 1
            pool = self.pool(config["aug_pool"]) if strength > 0 and config.get("aug_pool") else None
            if pool is not None:
                self._batch_from_pool(b, pool, indices, positions, protocol_seed, aug_epoch, config)
                return b, time.perf_counter() - t0
            for j, (index, position) in enumerate(zip(indices, positions)):
                sample = self.base.sample(index)
                if strength > 0:
                    # Two independent strong views; the clean view survives with aug_clean_prob so the
                    # canonical input that validation uses stays in the training distribution.
                    b["xa"][j] = r5pp.strong_augmented_features(
                        sample, protocol_seed, aug_epoch, int(position), strength, stream=1,
                        shift=config["jitter_shift"], hand_filter=self.hand_filter, body_align=self.body_align,
                        view_degrees=config.get("aug_view_degrees", 15.0))
                    keep_clean = np.random.default_rng(np.random.SeedSequence(
                        [protocol_seed, aug_epoch, int(position), 4177])).random() < config.get("aug_clean_prob", 0.2)
                    if not keep_clean:
                        b["x"][j] = r5pp.strong_augmented_features(
                            sample, protocol_seed, aug_epoch, int(position), strength, stream=0,
                            shift=config["jitter_shift"], hand_filter=self.hand_filter, body_align=self.body_align,
                        view_degrees=config.get("aug_view_degrees", 15.0))
                else:
                    b["xa"][j] = r5pp.augmented_features(
                        sample, protocol_seed, aug_epoch, int(position),
                        config["rotation_degrees"], config["jitter_shift"], hand_filter=self.hand_filter, body_align=self.body_align)
        return b, time.perf_counter() - t0

    def _batch_from_pool(self, b, pool, indices, positions, protocol_seed, aug_epoch, config):
        """xa = a pooled strong view; x = another pooled view (the clean canonical one with aug_clean_prob).
        Views are chosen deterministically from (seed, epoch, position), so a resumed run repeats them."""
        views_a, views_x, clean = [], [], []
        for position in positions:
            rng = np.random.default_rng(np.random.SeedSequence([protocol_seed, aug_epoch, int(position), 5531]))
            va, vx = rng.choice(pool.views, 2, replace=False) if pool.views > 1 else (0, 0)
            views_a.append(int(va)), views_x.append(int(vx))
            clean.append(rng.random() < config.get("aug_clean_prob", 0.2))
        indices = np.asarray(indices, np.int64)
        views_a, views_x, clean = np.asarray(views_a), np.asarray(views_x), np.asarray(clean)
        for v in np.unique(np.concatenate([views_a, views_x[~clean]])):
            rows = np.flatnonzero(views_a == v)
            if len(rows):
                b["xa"][rows] = pool.get(v, indices[rows])
            rows = np.flatnonzero((views_x == v) & ~clean)
            if len(rows):
                b["x"][rows] = pool.get(v, indices[rows])

    def batches(self, indices, size, config, epoch=0, training=False, protocol="xsub"):
        """``prefetch_workers`` producer threads, at most max(`prefetch_batches`, workers) prepared batches; errors propagate."""
        indices = np.array(indices, np.int64)
        positions = np.arange(len(indices))
        if training:
            # Same sample order and augmentation seeds as the R4 runner.
            positions = np.random.default_rng(config["seed"] + epoch).permutation(len(indices))
        workers = int(config.get("prefetch_workers", 1))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        try:
            pending, cursor = [], 0
            while cursor < len(indices) or pending:
                while cursor < len(indices) and len(pending) < max(config.get("prefetch_batches", 2), workers):
                    pos = positions[cursor:cursor + size]
                    pending.append(pool.submit(self.batch, indices[pos], pos, size, config, epoch,
                                               training, protocol))
                    cursor += size
                yield pending.pop(0).result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Build the R5 hand cache from an R4 cache")
    p.add_argument("--base", required=True, help="R4 cache dir (manifest.json, raw.npy, canonical.npy)")
    p.add_argument("--cache", required=True, help="output hand-cache dir")
    a = p.parse_args(argv)
    meta = build(a.base, a.cache)
    print(json.dumps({k: meta[k] for k in ("samples", "base_dir", "files")}))


if __name__ == "__main__":
    main()

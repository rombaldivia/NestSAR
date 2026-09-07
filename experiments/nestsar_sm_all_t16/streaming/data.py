"""One raw/canonical cache shared by both workers; bounded batch buffers only."""
from __future__ import annotations
import argparse
import concurrent.futures
import fcntl
import gc
import hashlib
import json
import mmap
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
from .. import preprocessing_corrected as pp
from . import VERSION
from .io_utils import Reporter, atomic_json


def _value(annotation, keys):
    for key in keys:
        if key in annotation:
            return annotation[key]
    raise KeyError(f"Missing one of {keys}")


def resolve_splits(annotations, splits):
    ids = [str(_value(a, ("frame_dir", "filename", "sample_name", "name", "id", "video_id"))) for a in annotations]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate annotation sample IDs")
    by_id = {sid: i for i, sid in enumerate(ids)}
    lower = {str(k).lower(): k for k in splits}
    result = {}
    for protocol in ("xsub", "xset"):
        for part in ("train", "val"):
            aliases = (f"{protocol}_train", f"{protocol}train", f"train_{protocol}") if part == "train" else (
                f"{protocol}_val", f"{protocol}_test", f"{protocol}val", f"{protocol}test", f"val_{protocol}", f"test_{protocol}")
            found = next((lower[k] for k in aliases if k in lower), None)
            if found is None:
                raise ValueError(f"Missing {protocol}/{part} split; keys={list(splits)}")
            members = [str(s) for s in splits[found]]
            if not members or len(set(members)) != len(members):
                raise ValueError(f"Empty or duplicate {protocol}/{part} IDs")
            missing = [sid for sid in members if sid not in by_id]
            if missing:
                raise ValueError(f"{protocol}/{part}: {len(missing)} IDs absent from annotations: {missing[:5]}")
            result[f"{protocol}_{part}"] = [by_id[sid] for sid in members]
        if set(result[f"{protocol}_train"]) & set(result[f"{protocol}_val"]):
            raise ValueError(f"Train/validation leakage in {protocol}")
    return ids, result


def prepare(dataset, cache, status, layout="MTVC"):
    dataset, cache = Path(dataset), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    report = Reporter(status)
    report(phase="Cache lock", current=0, total=1)
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report(phase="Hash dataset", current=0, total=1)
        with dataset.open("rb") as f:
            source_sha = hashlib.file_digest(f, "sha256").hexdigest()
        signature = {"source_sha256": source_sha, "layout": layout,
                     "preprocessing": pp.VERSION, "cache_version": VERSION}
        existing = cache / "manifest.json"
        if existing.exists():
            meta = json.loads(existing.read_text())
            if meta["signature"] != signature:
                raise ValueError("Cache belongs to different data/preprocessing. Choose a new CACHE_DIR.")
            for name, size in meta["files"].items():
                if not (cache / name).is_file() or (cache / name).stat().st_size != size:
                    raise ValueError(f"Incomplete cache: {name}. Use a new CACHE_DIR.")
            report(phase="Cache ready", current=1, total=1, done=True, samples=meta["samples"])
            return meta
        report(phase="Load pickle once", current=0, total=1)
        with dataset.open("rb") as f:
            try:
                loaded = pickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                loaded = pickle.load(f, encoding="latin1")
        annotations = _value(loaded, ("annotations", "annotation", "samples", "data_list"))
        ids, splits = resolve_splits(annotations, _value(loaded, ("split", "splits")))
        n = len(annotations)
        shape = np.zeros((n, 2), np.int64)
        labels = np.zeros(n, np.int32)
        for i, annotation in enumerate(annotations):
            x = pp.ordered_raw(_value(annotation, ("keypoint", "keypoints", "skeleton", "skeletons", "data")), layout)
            if len(x) == 0 or not pp.raw_valid(x).any():
                raise ValueError(f"Empty skeleton: {ids[i]}")
            people = 2 if np.any(x[:, 1]) else 1
            shape[i] = len(x), people
            labels[i] = int(_value(annotation, ("label", "action_label", "class", "target")))
            if i % 200 == 0:
                report(phase="Check raw data", current=i, total=n)
        if np.any((labels < 0) | (labels >= 120)):
            raise ValueError("Expected zero-based NTU120 labels 0..119; no automatic label shifting")
        offsets = np.r_[0, np.cumsum(shape[:, 0] * shape[:, 1])]
        raw_bytes = int(offsets[-1]) * pp.JOINTS * 3 * 4
        canonical_bytes = n * pp.FRAMES * pp.FEATURES * 4
        need = raw_bytes + canonical_bytes
        free = shutil.disk_usage(cache).free
        if free < need + 1024 ** 3:
            raise RuntimeError(f"Cache needs {need / 2**30:.2f} GiB plus 1 GiB reserve; {free / 2**30:.2f} GiB free. Choose a larger CACHE_DIR.")
        raw = np.lib.format.open_memmap(cache / "raw.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]), pp.JOINTS, 3))
        canonical = np.lib.format.open_memmap(cache / "canonical.npy", mode="w+", dtype=np.float32,
                                             shape=(n, pp.FRAMES, pp.FEATURES))
        for i, annotation in enumerate(annotations):
            x = pp.ordered_raw(_value(annotation, ("keypoint", "keypoints", "skeleton", "skeletons", "data")), layout)
            raw[offsets[i]:offsets[i+1]] = x[:, :shape[i, 1]].reshape(-1, pp.JOINTS, 3)
            canonical[i] = pp.features(x)
            if (i + 1) % 512 == 0:
                # Release builder mappings after flushing; file pages remain
                # available to both workers through the OS page cache.
                for array in (raw, canonical):
                    array.flush()
                    if hasattr(array._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                        array._mmap.madvise(mmap.MADV_DONTNEED)
            if i % 200 == 0:
                report(phase="Cache raw + canonical", current=i, total=n, cache_gib=need / 2**30)
        raw.flush()
        canonical.flush()
        del raw, canonical, x, annotation, annotations, loaded
        gc.collect()
        np.save(cache / "shape.npy", shape)
        np.save(cache / "offsets.npy", offsets)
        np.save(cache / "labels.npy", labels)
        atomic_json(cache / "ids.json", ids)
        atomic_json(cache / "splits.json", splits)
        names = ("raw.npy", "canonical.npy", "shape.npy", "offsets.npy", "labels.npy", "ids.json", "splits.json")
        meta = {"signature": signature, "samples": n, "raw_bytes": raw_bytes, "canonical_bytes": canonical_bytes,
                "split_counts": {k: len(v) for k, v in splits.items()},
                "files": {name: (cache / name).stat().st_size for name in names}}
        atomic_json(existing, meta)  # Completion marker is written LAST.
        report(phase="Cache ready", current=n, total=n, done=True, samples=n)
        return meta


class Dataset:
    def __init__(self, cache):
        cache = Path(cache)
        self.meta = json.loads((cache / "manifest.json").read_text())
        if self.meta["signature"].get("cache_version") != VERSION or self.meta["signature"].get("preprocessing") != pp.VERSION:
            raise ValueError("Cache preprocessing/version mismatch")
        for name, size in self.meta["files"].items():
            if not (cache / name).is_file() or (cache / name).stat().st_size != size:
                raise ValueError(f"Incomplete cache: {name}")
        self.raw = np.load(cache / "raw.npy", mmap_mode="r")
        self.canonical = np.load(cache / "canonical.npy", mmap_mode="r")
        self.shape = np.load(cache / "shape.npy", mmap_mode="r")
        self.offsets = np.load(cache / "offsets.npy", mmap_mode="r")
        self.labels = np.load(cache / "labels.npy", mmap_mode="r")
        self.splits = json.loads((cache / "splits.json").read_text())

    def sample(self, index):
        total, people = self.shape[index]
        x = self.raw[self.offsets[index]:self.offsets[index+1]].reshape(total, people, pp.JOINTS, 3)
        if people == 2:
            return np.asarray(x)
        out = np.zeros((total, 2, pp.JOINTS, 3), np.float32)
        out[:, 0] = x[:, 0]
        return out

    def batch(self, indices, positions, size, config, epoch, training, protocol):
        t0 = time.perf_counter()
        b = {"x": np.zeros((size, pp.FRAMES, pp.FEATURES), np.float32),
             "y": np.zeros(size, np.int32), "mask": np.zeros(size, np.float32)}
        # Advanced indexing copies only this batch, never a whole split.
        b["x"][:len(indices)] = self.canonical[indices]
        b["y"][:len(indices)] = self.labels[indices]
        b["mask"][:len(indices)] = 1
        if training:
            b["xa"] = np.zeros_like(b["x"])
            protocol_seed = config["seed"] + (100000 if protocol == "xset" else 0)
            for j, (index, position) in enumerate(zip(indices, positions)):
                b["xa"][j] = pp.augmented_features(
                    self.sample(index), protocol_seed, epoch if config["fresh_augmentation"] else 1,
                    int(position), config["rotation_degrees"], config["jitter_shift"])
        return b, time.perf_counter() - t0

    def batches(self, indices, size, config, epoch=0, training=False, protocol="xsub"):
        """One producer thread, at most two prepared batches; exceptions propagate."""
        indices = np.array(indices, np.int64)
        positions = np.arange(len(indices))
        if training:
            # Match the preceding corrected SM-ALL runner's sample order/seeds.
            positions = np.random.default_rng(config["seed"] + epoch).permutation(len(indices))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            pending = []
            cursor = 0
            while cursor < len(indices) or pending:
                while cursor < len(indices) and len(pending) < config.get("prefetch_batches", 2):
                    pos = positions[cursor:cursor+size]
                    pending.append(pool.submit(self.batch, indices[pos], pos, size, config, epoch, training, protocol))
                    cursor += size
                yield pending.pop(0).result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--layout", default="MTVC", choices=("MTVC", "TMVC"))
    a = p.parse_args()
    prepare(a.dataset, a.cache, a.status, a.layout)

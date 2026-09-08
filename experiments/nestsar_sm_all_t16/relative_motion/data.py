"""A shared auxiliary memmap; original P2 raw/canonical arrays stay read-only."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import mmap
import os
import shutil
import time
from pathlib import Path

import numpy as np
from ..streaming.data import Dataset as P2Dataset
from ..streaming.io_utils import Reporter, atomic_json, read_json
from ..nonlinear_compare.data import TrainOnlyCache, grouped_plan
from . import VERSION
from . import preprocessing as pp


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_signature(cache):
    cache = Path(cache)
    meta = read_json(cache / "manifest.json")
    if meta is None:
        raise ValueError("Missing source P2 manifest")
    return dict(version=VERSION, source=meta["signature"], samples=meta["samples"],
                parents=pp.PARENTS.tolist(), preprocessing_sha256=hashlib.sha256(
                    Path(pp.__file__).read_bytes()).hexdigest(),
                metadata_sha256=hashlib.sha256(
                    (cache / "ids.json").read_bytes() + (cache / "splits.json").read_bytes()).hexdigest())


def open_paths(cache, auxiliary):
    auxiliary = Path(auxiliary)
    meta = read_json(auxiliary / "manifest.json")
    signature = source_signature(cache)
    if meta is None or meta.get("signature") != signature:
        raise ValueError("Missing/incompatible relative-path cache; run the cache builder first")
    path = auxiliary / "relative_path.npy"
    if not path.is_file() or path.stat().st_size != meta["bytes"]:
        raise ValueError("Incomplete relative-path cache")
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.shape != (signature["samples"], *pp.SHAPE) or x.dtype != np.float32:
        raise ValueError("Relative-path cache shape/dtype mismatch")
    return x, meta


def release_pages(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and hasattr(mapping, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
        mapping.madvise(mmap.MADV_DONTNEED)


def prepare(cache, auxiliary, status, chunk=256, reserve_bytes=2**30):
    """Resume at a committed chunk. Flush+fsync precede the progress commit.

    Only this new cache is written. No P2-token rebuild, CUDA dependency install,
    dataset-sized in-memory array, or deletion of another run's files occurs.
    """
    if not isinstance(chunk, int) or chunk < 1:
        raise ValueError("chunk must be positive")
    auxiliary = Path(auxiliary)
    auxiliary.mkdir(parents=True, exist_ok=True)
    report = Reporter(status)
    with (auxiliary / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        signature = source_signature(cache)
        n = signature["samples"]
        if (auxiliary / "manifest.json").exists():
            paths, meta = open_paths(cache, auxiliary)
            report(phase="Verify shared relative paths", current=0, total=n)
            previous = 0
            for entry in meta["chunks"]:
                end = entry["end"]
                if (not previous < end <= n or
                        hashlib.sha256(paths[previous:end].tobytes()).hexdigest() != entry["sha256"]):
                    raise ValueError("Corrupt relative-path cache; use a new cache directory")
                previous = end
                release_pages(paths)
                report(current=end)
            if previous != n:
                raise ValueError("Incomplete relative-path chunk manifest")
            report(phase="Reuse relative paths", current=n, total=n, done=True)
            return meta
        data = P2Dataset(cache)
        if data.canonical.shape != (n, 16, 750):
            raise ValueError("Source canonical shape mismatch")
        partial = auxiliary / "relative_path.partial.npy"
        final = auxiliary / "relative_path.npy"
        progress_file = auxiliary / "progress.json"
        progress = read_json(progress_file)
        if progress is not None and progress.get("signature") != signature:
            raise ValueError("Partial auxiliary cache belongs to different data/code; use a new directory")
        if progress is None:
            if partial.exists() or final.exists():
                raise ValueError("Unidentified auxiliary files; use a new directory")
            required = n * int(np.prod(pp.SHAPE)) * 4 + 4096
            if shutil.disk_usage(auxiliary).free < required + reserve_bytes:
                raise OSError(f"Need {(required+reserve_bytes)/2**30:.2f} GiB free for paths + output reserve")
            # Commit identity first; interrupted file allocation can be retried.
            progress = dict(signature=signature, completed=0, chunks=[])
            atomic_json(progress_file, progress)
        start = progress["completed"]
        if not isinstance(start, int) or not 0 <= start <= n:
            raise ValueError("Invalid auxiliary progress cursor")
        target = final if final.exists() and start == n else partial
        if start == 0:
            paths = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(n, *pp.SHAPE))
        else:
            paths = np.load(target, mmap_mode="r+", allow_pickle=False)
            if paths.shape != (n, *pp.SHAPE) or paths.dtype != np.float32:
                raise ValueError("Partial auxiliary shape/dtype mismatch")
        # Validate committed chunks on resume without keeping them resident.
        previous = 0
        for entry in progress["chunks"]:
            end = entry["end"]
            if not previous < end <= start or hashlib.sha256(paths[previous:end].tobytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Corrupt committed auxiliary chunk; use a new cache directory")
            previous = end
            release_pages(paths)
        if previous != start:
            raise ValueError("Auxiliary progress/chunk mismatch")
        report(phase="Build relative paths", current=start, total=n)
        for begin in range(start, n, chunk):
            end = min(n, begin + chunk)
            for i in range(begin, end):
                paths[i] = pp.relative_path(data.sample(i))
            paths.flush()
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            progress["chunks"].append(dict(end=end, sha256=hashlib.sha256(paths[begin:end].tobytes()).hexdigest()))
            progress["completed"] = end
            atomic_json(progress_file, progress)
            release_pages(paths)
            release_pages(data.raw)
            report(current=end)
        del paths
        if target != final:
            os.replace(target, final)
        meta = dict(signature=signature, bytes=final.stat().st_size, shape=[n, *pp.SHAPE],
                    dtype="float32", chunks=progress["chunks"],
                    note="Per-clip deterministic preprocessing; no split statistics fitted")
        atomic_json(auxiliary / "manifest.json", meta)
        report(phase="Relative paths ready", current=n, total=n, done=True)
        return meta


def split_plan(cache, protocol, seed, select_fraction=.15, final_fraction=.15,
               min_class_samples=(10, 3, 3)):
    source = TrainOnlyCache(cache, protocol)
    plan = grouped_plan(source.train, source.groups, source.data.labels,
                        np.arange(1, 121).reshape(60, 2), seed,
                        dict(select_fraction=select_fraction, final_fraction=final_fraction,
                             min_class_samples=min_class_samples))
    plan.update(protocol=protocol, classes=120, source_signature=source.data.meta["signature"],
                evaluation="Internal official-train groups only; not official NTU120 benchmark")
    plan.pop("sha256")
    plan["sha256"] = digest(plan)
    return plan


class Dataset(P2Dataset):
    """Batch-sized packing and one bounded producer; no feature copies per split."""
    def __init__(self, cache, auxiliary, plan):
        source = TrainOnlyCache(cache, plan["protocol"])
        self.__dict__.update(source.data.__dict__)
        self.guard = source.guard
        self.plan = plan
        body = {k: v for k, v in plan.items() if k != "sha256"}
        if digest(body) != plan["sha256"] or plan["source_signature"] != self.meta["signature"]:
            raise ValueError("Split manifest hash/source mismatch")
        seen, seen_groups = set(), set()
        for part in ("fit", "select", "final"):
            ids = source.guard(plan["indices"][part])
            groups = set(source.groups[ids].tolist())
            if (not len(ids) or len(ids) != len(set(ids.tolist())) or seen.intersection(ids)
                    or seen_groups.intersection(groups) or groups != set(plan["groups"][part])):
                raise ValueError("Invalid/overlapping sample or group partition")
            if set(np.unique(self.labels[ids])) != set(range(120)):
                raise ValueError("Each internal partition must contain all 120 classes")
            seen.update(ids.tolist())
            seen_groups.update(groups)
        if seen != set(source.train.tolist()):
            raise ValueError("Internal partitions must cover the complete official training split")
        self.paths, relative_meta = open_paths(cache, auxiliary)
        self.meta = dict(self.meta, signature=dict(p2=self.meta["signature"],
                         relative=relative_meta["signature"], split_sha256=plan["sha256"]))
        protocol = plan["protocol"]
        self.splits = {f"{protocol}_train": plan["indices"]["fit"],
                       f"{protocol}_val": plan["indices"]["select"]}

    def sample(self, index):
        self.guard([index])
        return super().sample(index)

    def batch(self, indices, positions, size, config, epoch, training, protocol):
        t0 = time.perf_counter()
        indices = self.guard(indices)
        b = dict(x=np.zeros((size, 16, 750), np.float32), y=np.zeros(size, np.int32),
                 mask=np.zeros(size, np.float32))
        b["x"][:len(indices)] = pp.pack(self.canonical[indices], self.paths[indices])
        b["y"][:len(indices)] = self.labels[indices]
        b["mask"][:len(indices)] = 1
        if training:
            b["xa"] = np.zeros_like(b["x"])
            seed = config["seed"] + (100000 if protocol == "xset" else 0)
            for j, (index, position) in enumerate(zip(indices, positions)):
                b["xa"][j] = pp.augmented_features(self.sample(index), seed,
                    epoch if config["fresh_augmentation"] else 1, int(position),
                    config["rotation_degrees"], config["jitter_shift"])
        if not np.isfinite(b["x"]).all() or (training and not np.isfinite(b["xa"]).all()):
            raise ValueError("Nonfinite packed features")
        return b, time.perf_counter() - t0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("cache", "auxiliary", "status"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    prepare(args.cache, args.auxiliary, args.status)

"""One CPU preparation pass, training-only calibration, compact pose overlays.

Both protocols share the existing read-only raw/canonical cache. Only 3/15
channels are stored per protocol; workers never load the source pickle.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import mmap
from pathlib import Path
import shutil
import time

import numpy as np

from .. import preprocessing_corrected as pp
from ..sampler_b import VERSION, POLICY, ScaleAccumulator, SamplerB, relative_motion
from .data import Dataset
from .io_utils import Reporter, atomic_json, read_json

PROTOCOLS = ("xsub", "xset")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cache_signature(cache, dataset):
    cache = Path(cache)
    return dict(version=VERSION, policy=POLICY, base_cache=dataset.meta["signature"],
                samples=int(dataset.meta["samples"]), splits_sha256=digest(dataset.splits),
                ids_sha256=hashlib.sha256((cache / "ids.json").read_bytes()).hexdigest())


def verify_splits(dataset):
    n = int(dataset.meta["samples"])
    for protocol in PROTOCOLS:
        a, b = [list(map(int, dataset.splits[f"{protocol}_{part}"])) for part in ("train", "val")]
        if not a or not b or len(set(a)) != len(a) or len(set(b)) != len(b):
            raise ValueError(f"{protocol}: empty/duplicate split indices")
        if set(a) & set(b) or set(a) | set(b) != set(range(n)):
            raise ValueError(f"{protocol}: splits must be disjoint and cover the entire cache")


def _release(array):
    array.flush()
    if hasattr(array._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
        array._mmap.madvise(mmap.MADV_DONTNEED)


def _check_files(out, manifest):
    for name, size in manifest["files"].items():
        if not (out / name).is_file() or (out / name).stat().st_size != size:
            raise ValueError(f"Incomplete sampler B cache: {name}; use a fresh sampler output directory")
    for protocol in PROTOCOLS:
        calibration = read_json(out / f"calibration_{protocol}.json")
        if digest(calibration) != manifest["calibration_sha256"][protocol]:
            raise ValueError(f"Changed sampler calibration for {protocol}; cannot resume")
        SamplerB(calibration)


def prepare_sampler_b(cache, out, status, require_full=True):
    """Fit each protocol on ALL its training samples, then freeze for all splits."""
    cache, out = Path(cache), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = Reporter(status)
    report(phase="B cache lock", current=0, total=1)
    with (out / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if require_full:
            from ..run_full_official_dual_t4 import verify_full_official_cache
            atomic_json(out / "full_official_audit.json", verify_full_official_cache(cache))
        dataset = Dataset(cache)
        verify_splits(dataset)
        signature = cache_signature(cache, dataset)
        manifest = read_json(out / "manifest.json")
        if manifest is not None:
            if manifest["signature"] != signature:
                raise ValueError("Sampler B cache/data/policy mismatch. Use a new OUT_DIR.")
            _check_files(out, manifest)
            report(phase="B cache ready", current=1, total=1, done=True)
            return manifest

        n = int(dataset.meta["samples"])
        files = [f"{kind}_{p}.npy" for p in PROTOCOLS for kind in ("pose", "indices")]
        # Float32 pose overlays: about 2.04 GiB for BOTH full protocols; indices
        # add 28 MiB. Full 15-channel tokens and raw arrays are never duplicated.
        required = 2 * n * pp.FRAMES * 2 * (25 * 3 * 4 + 4) + 4096
        reusable = sum((out / f).stat().st_size for f in files if (out / f).exists())
        free = shutil.disk_usage(out).free
        if free + reusable < required + 512 * 2**20:
            raise RuntimeError(f"B overlays need {required / 2**30:.2f} GiB + 0.5 GiB reserve; "
                               f"{free / 2**30:.2f} GiB free. Preserve checkpoints; choose a larger OUT_DIR.")

        calibration_bundle = read_json(out / "calibration.json")
        if calibration_bundle is not None and calibration_bundle["signature"] != signature:
            raise ValueError("Sampler B calibration belongs to another dataset/policy")
        if calibration_bundle is None:
            fit_start = time.perf_counter()
            accumulators = [ScaleAccumulator() for _ in PROTOCOLS]
            membership = np.zeros((n, 2), bool)
            for p, protocol in enumerate(PROTOCOLS):
                membership[dataset.splits[f"{protocol}_train"], p] = True
            fit_ids = np.flatnonzero(membership.any(1))
            for j, index in enumerate(fit_ids):
                x = dataset.sample(index)
                motion = relative_motion(x, pp.raw_valid(x))
                for p in np.flatnonzero(membership[index]):
                    accumulators[p].observe(*motion)
                if j % 100 == 0 or j + 1 == len(fit_ids):
                    report(phase="B fit train scales", current=j + 1, total=len(fit_ids))
            calibrations = {}
            for p, protocol in enumerate(PROTOCOLS):
                train_ids = dataset.splits[f"{protocol}_train"]
                calibration = accumulators[p].finish()
                if calibration["samples_observed"] != len(train_ids):
                    raise RuntimeError("Sampler training calibration sample accounting mismatch")
                calibration.update(protocol=protocol, fit_split=f"{protocol}_train",
                                   train_indices_sha256=digest(sorted(train_ids)),
                                   cache_signature=signature)
                calibrations[protocol] = calibration
            calibration_bundle = dict(signature=signature, calibrations=calibrations,
                                      fit_seconds=time.perf_counter() - fit_start)
            atomic_json(out / "calibration.json", calibration_bundle)

        calibrations = calibration_bundle["calibrations"]
        pose_start = time.perf_counter()
        samplers = {p: SamplerB(calibrations[p]) for p in PROTOCOLS}
        for p in PROTOCOLS:
            atomic_json(out / f"calibration_{p}.json", calibrations[p])
        poses = {p: np.lib.format.open_memmap(out / f"pose_{p}.npy", mode="w+", dtype=np.float32,
                                             shape=(n, 16, 2, 25, 3)) for p in PROTOCOLS}
        indices = {p: np.lib.format.open_memmap(out / f"indices_{p}.npy", mode="w+", dtype=np.int32,
                                               shape=(n, 16, 2)) for p in PROTOCOLS}
        diagnostics = {p: {part: {} for part in ("train", "val")} for p in PROTOCOLS}
        train_sets = {p: set(dataset.splits[f"{p}_train"]) for p in PROTOCOLS}
        for index in range(n):
            raw = dataset.sample(index)
            local, valid, scale = pp.canonicalize_raw(raw)
            starts, ends = pp.segment_bounds(len(raw))
            # Same centered coordinates as the fresh augmented selector. Offsets
            # are computed at every raw timestamp, once and reused by both fits.
            motion = relative_motion(local, valid)
            for p in PROTOCOLS:
                pose, chosen, stats = samplers[p].select(local, valid, starts, ends, motion=motion)
                # Match features()' float64 division followed by float32 storage.
                poses[p][index] = (pose.astype(np.float64) / scale).astype(np.float32)
                indices[p][index] = chosen
                totals = diagnostics[p]["train" if index in train_sets[p] else "val"]
                for key, value in dict(samples=1, **stats).items():
                    totals[key] = totals.get(key, 0) + value
            if (index + 1) % 512 == 0:
                for a in (*poses.values(), *indices.values()):
                    _release(a)
            if index % 100 == 0 or index + 1 == n:
                report(phase="B cache poses", current=index + 1, total=n)
        for a in (*poses.values(), *indices.values()):
            _release(a)
        atomic_json(out / "sampling_diagnostics.json", diagnostics)
        atomic_json(out / "preparation_timing.json", dict(
            fit_seconds=calibration_bundle["fit_seconds"],
            pose_cache_seconds=time.perf_counter() - pose_start,
            note="CPU cache preparation only; neural GPU execution is timed separately in history.json"))
        files += [f"calibration_{p}.json" for p in PROTOCOLS]
        manifest = dict(signature=signature, samples=n, version=VERSION,
                        files={f: (out / f).stat().st_size for f in files},
                        calibration_sha256={p: digest(calibrations[p]) for p in PROTOCOLS})
        atomic_json(out / "manifest.json", manifest)  # Completion marker LAST.
        report(phase="B cache ready", current=n, total=n, done=True)
        return manifest


def attach_sampler_b(dataset, cache, out, protocol):
    """Open read-only overlays; return the checkpoint's preprocessing identity."""
    if protocol not in PROTOCOLS:
        raise ValueError("Unknown protocol")
    out = Path(out)
    manifest = read_json(out / "manifest.json")
    if manifest is None or manifest["signature"] != cache_signature(cache, dataset):
        raise ValueError("Missing/mismatched sampler B cache; run the B launcher first")
    _check_files(out, manifest)
    calibration = read_json(out / f"calibration_{protocol}.json")
    if (calibration["fit_split"] != f"{protocol}_train"
            or calibration["train_indices_sha256"] != digest(sorted(dataset.splits[f"{protocol}_train"]))):
        raise ValueError("Sampler calibration training split mismatch")
    dataset.pose_sampler = SamplerB(calibration)
    dataset.pose_protocol = protocol
    dataset.pose_overlay = np.load(out / f"pose_{protocol}.npy", mmap_mode="r")
    if dataset.pose_overlay.shape != (dataset.meta["samples"], 16, 2, 25, 3) or dataset.pose_overlay.dtype != np.float32:
        raise ValueError("Invalid B pose overlay shape/dtype")
    return dict(version=VERSION, calibration_sha256=manifest["calibration_sha256"][protocol],
                cache_signature=manifest["signature"], calibration=calibration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--status", required=True)
    args = parser.parse_args()
    prepare_sampler_b(args.cache, args.out, args.status)

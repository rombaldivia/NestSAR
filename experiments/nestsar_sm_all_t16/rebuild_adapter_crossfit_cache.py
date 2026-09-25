#!/usr/bin/env python3
"""Build the minimal full cache needed by the R4 adapter cross-fit audit.

Reuses a previously repaired canonical.npy through a read-only symlink and
rebuilds only raw.npy from the exact NTU120 source. Small metadata files are
copied from the original cache after strict validation. The source caches are
never modified; manifest.json is written last as the completion marker.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import mmap
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from . import preprocessing_corrected as pp
from .streaming import VERSION as CACHE_VERSION

EXPECTED_SAMPLES = 113_945
EXPECTED_CANONICAL_SHAPE = (EXPECTED_SAMPLES, pp.FRAMES, pp.FEATURES)
META_FILES = ("shape.npy", "offsets.npy", "labels.npy", "ids.json", "splits.json")


def atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _value(obj, keys):
    for key in keys:
        if key in obj:
            return obj[key]
    raise KeyError(f"Missing one of {keys}")


def load_pickle(path: Path):
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def discover_dataset(explicit: str | None, expected_sha: str) -> tuple[Path, str]:
    candidates = [Path(explicit)] if explicit else sorted(
        Path("/kaggle/input").rglob("ntu120_3danno.pkl")
    )
    if not candidates:
        raise FileNotFoundError("Could not find ntu120_3danno.pkl under /kaggle/input.")

    checked = []
    for candidate in candidates:
        if not candidate.is_file():
            checked.append((str(candidate), "missing"))
            continue
        digest = sha256(candidate)
        checked.append((str(candidate), digest))
        if digest == expected_sha:
            return candidate.resolve(), digest

    detail = "\n".join(f"  {p}: {d}" for p, d in checked)
    raise ValueError(
        "No NTU120 annotation file matches the R4 cache SHA-256.\n"
        f"Expected: {expected_sha}\nChecked:\n{detail}"
    )


def validate_sources(metadata_cache: Path, canonical_cache: Path) -> dict:
    meta_manifest = metadata_cache / "manifest.json"
    canonical_manifest = canonical_cache / "manifest.json"
    if not meta_manifest.is_file():
        raise FileNotFoundError(meta_manifest)
    if not canonical_manifest.is_file():
        raise FileNotFoundError(canonical_manifest)

    source = json.loads(meta_manifest.read_text())
    repaired = json.loads(canonical_manifest.read_text())

    expected_sig = source.get("signature", {})
    if expected_sig.get("preprocessing") != pp.VERSION:
        raise ValueError("Metadata cache preprocessing mismatch.")
    if expected_sig.get("cache_version") != CACHE_VERSION:
        raise ValueError("Metadata cache version mismatch.")
    if repaired.get("signature") != expected_sig:
        raise ValueError("Canonical repair belongs to a different cache identity.")
    if int(source.get("samples", -1)) != EXPECTED_SAMPLES:
        raise ValueError("Unexpected metadata sample count.")
    if int(repaired.get("samples", -1)) != EXPECTED_SAMPLES:
        raise ValueError("Unexpected canonical sample count.")

    canonical = np.load(canonical_cache / "canonical.npy", mmap_mode="r")
    if canonical.shape != EXPECTED_CANONICAL_SHAPE or canonical.dtype != np.float32:
        raise ValueError(
            f"Canonical layout mismatch: {canonical.shape}, {canonical.dtype}"
        )
    del canonical

    for name in META_FILES:
        p = metadata_cache / name
        if not p.is_file():
            raise FileNotFoundError(p)

    shape = np.load(metadata_cache / "shape.npy", mmap_mode="r")
    offsets = np.load(metadata_cache / "offsets.npy", mmap_mode="r")
    labels = np.load(metadata_cache / "labels.npy", mmap_mode="r")
    if shape.shape != (EXPECTED_SAMPLES, 2) or shape.dtype != np.int64:
        raise ValueError(f"Unexpected shape.npy layout: {shape.shape}, {shape.dtype}")
    if offsets.shape != (EXPECTED_SAMPLES + 1,) or offsets.dtype != np.int64:
        raise ValueError(
            f"Unexpected offsets.npy layout: {offsets.shape}, {offsets.dtype}"
        )
    if labels.shape != (EXPECTED_SAMPLES,) or labels.dtype != np.int32:
        raise ValueError(
            f"Unexpected labels.npy layout: {labels.shape}, {labels.dtype}"
        )
    if int(offsets[-1]) <= 0:
        raise ValueError("Invalid raw offsets.")

    return source


def cache_ready(output: Path, signature: dict) -> bool:
    manifest = output / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        meta = json.loads(manifest.read_text())
        if meta.get("signature") != signature:
            return False
        for name, size in meta.get("files", {}).items():
            p = output / name
            if not p.is_file() or p.stat().st_size != int(size):
                return False
        raw = np.load(output / "raw.npy", mmap_mode="r")
        canonical = np.load(output / "canonical.npy", mmap_mode="r")
        if raw.dtype != np.float32 or raw.ndim != 3 or raw.shape[1:] != (pp.JOINTS, 3):
            return False
        if canonical.shape != EXPECTED_CANONICAL_SHAPE or canonical.dtype != np.float32:
            return False
    except Exception:
        return False
    return True


def build(args) -> Path:
    metadata_cache = Path(args.metadata_cache).resolve()
    canonical_cache = Path(args.canonical_cache).resolve()
    output = Path(args.output_cache).resolve()

    if output in (metadata_cache, canonical_cache):
        raise ValueError("Use a separate --output-cache directory.")

    source_meta = validate_sources(metadata_cache, canonical_cache)
    signature = source_meta["signature"]
    expected_sha = signature["source_sha256"]
    dataset, dataset_sha = discover_dataset(args.dataset, expected_sha)

    output.mkdir(parents=True, exist_ok=True)

    print("=" * 112)
    print("NESTSAR R4 — ADAPTER CROSSFIT CACHE REPAIR")
    print("=" * 112)
    print("Dataset        :", dataset)
    print("Dataset SHA    :", dataset_sha)
    print("Metadata cache :", metadata_cache)
    print("Canonical cache:", canonical_cache)
    print("Output cache   :", output)
    print("Preprocessing  :", pp.VERSION)
    print("Cache version  :", CACHE_VERSION)
    print("canonical.npy  : reuse repaired tensor (symlink)")
    print("raw.npy        : rebuild from exact NTU120 source")
    print("=" * 112, flush=True)

    lock = (output / "build.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("This cross-fit cache build is already running.")

    try:
        if cache_ready(output, signature):
            print("✅ Full adapter cross-fit cache already ready; reusing it.")
            return output

        stale_manifest = output / "manifest.json"
        had_invalid_manifest = stale_manifest.exists()
        if had_invalid_manifest:
            quarantine = output / "manifest.invalid.json"
            suffix = 1
            while quarantine.exists():
                quarantine = output / f"manifest.invalid.{suffix}.json"
                suffix += 1
            os.replace(stale_manifest, quarantine)
            print(
                "⚠️ Existing completion manifest failed validation; "
                f"quarantined as {quarantine.name}."
            )

        shape = np.load(metadata_cache / "shape.npy", mmap_mode="r")
        offsets = np.load(metadata_cache / "offsets.npy", mmap_mode="r")
        labels = np.load(metadata_cache / "labels.npy", mmap_mode="r")

        raw_shape = (int(offsets[-1]), pp.JOINTS, 3)
        expected_raw_bytes = int(offsets[-1]) * pp.JOINTS * 3 * 4

        raw_path = output / "raw.npy"
        progress_path = output / "raw_progress.json"

        existing_raw = raw_path.stat().st_size if raw_path.exists() else 0
        free = shutil.disk_usage(output).free + existing_raw
        reserve = 768 << 20
        if free < expected_raw_bytes + reserve:
            raise RuntimeError(
                f"Need about {expected_raw_bytes / 2**30:.2f} GiB for raw.npy "
                f"plus reserve; only {free / 2**30:.2f} GiB available."
            )

        recover_completed_raw = False

        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress.get("dataset_sha256") != dataset_sha:
                raise ValueError("Partial raw cache belongs to another dataset.")
            if not raw_path.is_file():
                raise ValueError("raw_progress.json exists but raw.npy is missing.")
            start = int(progress.get("next_index", 0))
            raw = np.load(raw_path, mmap_mode="r+")
            if raw.shape != raw_shape or raw.dtype != np.float32:
                raise ValueError(f"Partial raw layout mismatch: {raw.shape}, {raw.dtype}")
            if not 0 <= start <= EXPECTED_SAMPLES:
                raise ValueError(f"Invalid raw progress index: {start}")
            print(f"Resuming raw rebuild at sample {start:,}/{EXPECTED_SAMPLES:,}.")
        elif had_invalid_manifest and raw_path.is_file():
            try:
                raw = np.load(raw_path, mmap_mode="r+")
                recover_completed_raw = (
                    raw.shape == raw_shape
                    and raw.dtype == np.float32
                    and raw_path.stat().st_size == expected_raw_bytes + 128
                )
            except Exception:
                recover_completed_raw = False

            if recover_completed_raw:
                start = EXPECTED_SAMPLES
                print(
                    "✅ Existing raw.npy has the exact completed layout/size; "
                    "reusing it and regenerating metadata/manifest."
                )
            else:
                try:
                    del raw
                except Exception:
                    pass
                if raw_path.exists():
                    raw_path.unlink()
                raw = np.lib.format.open_memmap(
                    raw_path, mode="w+", dtype=np.float32, shape=raw_shape
                )
                start = 0
                atomic_json(
                    progress_path,
                    {
                        "next_index": 0,
                        "dataset_sha256": dataset_sha,
                        "preprocessing": pp.VERSION,
                        "cache_version": CACHE_VERSION,
                    },
                )
                print("⚠️ Existing raw.npy was not safely reusable; rebuilding it.")
        else:
            if raw_path.exists():
                raw_path.unlink()
            raw = np.lib.format.open_memmap(
                raw_path, mode="w+", dtype=np.float32, shape=raw_shape
            )
            start = 0
            atomic_json(
                progress_path,
                {
                    "next_index": 0,
                    "dataset_sha256": dataset_sha,
                    "preprocessing": pp.VERSION,
                    "cache_version": CACHE_VERSION,
                },
            )

        loaded = load_pickle(dataset)
        annotations = _value(
            loaded, ("annotations", "annotation", "samples", "data_list")
        )
        if len(annotations) != EXPECTED_SAMPLES:
            raise ValueError(
                f"Dataset has {len(annotations)} samples; expected {EXPECTED_SAMPLES}."
            )

        layout = signature.get("layout", "MTVC")
        flush_every = max(1, int(args.flush_every))

        bar = tqdm(
            range(start, EXPECTED_SAMPLES),
            initial=start,
            total=EXPECTED_SAMPLES,
            desc="Raw skeleton cache",
            unit="clip",
            dynamic_ncols=True,
        )

        for i in bar:
            annotation = annotations[i]
            label = int(_value(annotation, ("label", "action_label", "class", "target")))
            if label != int(labels[i]):
                raise ValueError(
                    f"Label/order mismatch at sample {i}: {label} != {int(labels[i])}"
                )

            x = pp.ordered_raw(
                _value(
                    annotation,
                    ("keypoint", "keypoints", "skeleton", "skeletons", "data"),
                ),
                layout,
            )

            total = int(shape[i, 0])
            people = int(shape[i, 1])
            if len(x) != total:
                raise ValueError(
                    f"Frame-count mismatch at sample {i}: {len(x)} != {total}"
                )
            observed_people = 2 if np.any(x[:, 1]) else 1
            if observed_people != people:
                raise ValueError(
                    f"Person-count mismatch at sample {i}: "
                    f"{observed_people} != {people}"
                )

            packed = x[:, :people].reshape(-1, pp.JOINTS, 3)
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            if len(packed) != hi - lo:
                raise ValueError(
                    f"Raw offset mismatch at sample {i}: {len(packed)} != {hi - lo}"
                )
            raw[lo:hi] = packed

            if (i + 1) % flush_every == 0 or i + 1 == EXPECTED_SAMPLES:
                raw.flush()
                if hasattr(raw, "_mmap") and raw._mmap is not None:
                    if hasattr(raw._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                        raw._mmap.madvise(mmap.MADV_DONTNEED)
                atomic_json(
                    progress_path,
                    {
                        "next_index": i + 1,
                        "dataset_sha256": dataset_sha,
                        "preprocessing": pp.VERSION,
                        "cache_version": CACHE_VERSION,
                    },
                )

        raw.flush()
        del raw, annotations, loaded

        check_raw = np.load(raw_path, mmap_mode="r")
        if check_raw.shape != raw_shape or check_raw.dtype != np.float32:
            raise RuntimeError("Final raw.npy validation failed.")
        del check_raw

        # Copy only small immutable metadata files.
        for name in META_FILES:
            shutil.copy2(metadata_cache / name, output / name)

        # Reuse the already repaired 5.09 GiB canonical tensor without duplicating it.
        canonical_link = output / "canonical.npy"
        if canonical_link.exists() or canonical_link.is_symlink():
            canonical_link.unlink()
        os.symlink(str(canonical_cache / "canonical.npy"), str(canonical_link))

        names = ("raw.npy", "canonical.npy", *META_FILES)
        manifest = {
            "signature": signature,
            "samples": EXPECTED_SAMPLES,
            "raw_bytes": expected_raw_bytes,
            "canonical_bytes": EXPECTED_SAMPLES * pp.FRAMES * pp.FEATURES * 4,
            "split_counts": source_meta.get("split_counts", {}),
            "crossfit_cache_repair": True,
            "canonical_reused_from": str(canonical_cache / "canonical.npy"),
            "files": {name: (output / name).stat().st_size for name in names},
        }
        atomic_json(output / "manifest.json", manifest)
        progress_path.unlink(missing_ok=True)

        if not cache_ready(output, signature):
            raise RuntimeError("Completed cache failed final Dataset-compatible checks.")

        print("\n✅ ADAPTER CROSSFIT CACHE READY")
        print("   ", output)
        print(
            f"   raw.npy       = {raw_path.stat().st_size:,} bytes "
            f"({raw_path.stat().st_size / 2**30:.3f} GiB)"
        )
        print(
            f"   canonical.npy = symlink -> {canonical_cache / 'canonical.npy'}"
        )
        print("   Source caches were NOT modified.")
        return output
    finally:
        lock.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metadata-cache",
        default="/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3",
    )
    p.add_argument(
        "--canonical-cache",
        default="/kaggle/working/NestSAR_R4_FROZEN_READOUT_CANONICAL_v1",
    )
    p.add_argument(
        "--output-cache",
        default="/kaggle/working/NestSAR_R4_ADAPTER_CROSSFIT_CACHE_v1",
    )
    p.add_argument("--dataset", default=None)
    p.add_argument("--flush-every", type=int, default=256)
    args = p.parse_args(argv)
    if args.flush_every < 1:
        p.error("--flush-every must be >= 1")
    return args


def main(argv=None):
    return build(parse_args(argv))


if __name__ == "__main__":
    main()

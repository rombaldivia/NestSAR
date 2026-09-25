#!/usr/bin/env python3
"""Build a minimal canonical-only cache for the frozen R4 readout audit.

This intentionally does NOT rebuild raw.npy. The frozen readout audit only needs
canonical.npy plus labels/IDs/splits. The original training cache is never
modified; a separate output directory is used and the completion manifest is
written last.

The build is resumable through canonical.npy.partial + progress.json.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
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

CACHE_VERSION = "sm-all-shared-cache-personaware-p2-r4-v1"
EXPECTED_SAMPLES = 113_945
EXPECTED_SHAPE = (EXPECTED_SAMPLES, pp.FRAMES, pp.FEATURES)
META_FILES = ("labels.npy", "ids.json", "splits.json")


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _value(annotation, keys):
    for key in keys:
        if key in annotation:
            return annotation[key]
    raise KeyError(f"Missing one of {keys}")


def load_pickle(path: Path):
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def discover_dataset(explicit: str | None, expected_sha: str) -> tuple[Path, str]:
    if explicit:
        candidates = [Path(explicit)]
    else:
        candidates = sorted(Path("/kaggle/input").rglob("ntu120_3danno.pkl"))

    if not candidates:
        raise FileNotFoundError(
            "Could not find ntu120_3danno.pkl under /kaggle/input. "
            "Pass --dataset explicitly."
        )

    checked = []
    for candidate in candidates:
        if not candidate.is_file():
            checked.append((str(candidate), "missing"))
            continue
        digest = sha256(candidate)
        checked.append((str(candidate), digest))
        if digest == expected_sha:
            return candidate.resolve(), digest

    detail = "\n".join(f"  {path}: {digest}" for path, digest in checked)
    raise ValueError(
        "No ntu120_3danno.pkl matches the source SHA-256 recorded by the R4 cache.\n"
        f"Expected: {expected_sha}\nChecked:\n{detail}"
    )


def validate_source_cache(source: Path) -> dict:
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    meta = json.loads(manifest_path.read_text())
    sig = meta.get("signature", {})

    if sig.get("preprocessing") != pp.VERSION:
        raise ValueError(
            f"Source preprocessing {sig.get('preprocessing')} != {pp.VERSION}"
        )
    if sig.get("cache_version") != CACHE_VERSION:
        raise ValueError(
            f"Source cache version {sig.get('cache_version')} != {CACHE_VERSION}"
        )
    if int(meta.get("samples", -1)) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Source sample count {meta.get('samples')} != {EXPECTED_SAMPLES}"
        )

    for name in META_FILES:
        p = source / name
        expected = meta.get("files", {}).get(name)
        if expected is None:
            raise ValueError(f"Source manifest has no size for {name}")
        if not p.is_file() or p.stat().st_size != int(expected):
            raise ValueError(f"Intact source metadata file required: {p}")

    labels = np.load(source / "labels.npy", mmap_mode="r")
    if labels.shape != (EXPECTED_SAMPLES,) or labels.dtype != np.int32:
        raise ValueError(
            f"Unexpected labels layout: {labels.shape}, {labels.dtype}"
        )

    ids = json.loads((source / "ids.json").read_text())
    if len(ids) != EXPECTED_SAMPLES or len(set(ids)) != EXPECTED_SAMPLES:
        raise ValueError("Source ids.json is incomplete or contains duplicates.")

    splits = json.loads((source / "splits.json").read_text())
    expected_counts = {
        "xsub_train": 63026,
        "xsub_val": 50919,
        "xset_train": 54468,
        "xset_val": 59477,
    }
    for key, count in expected_counts.items():
        if key not in splits or len(splits[key]) != count:
            raise ValueError(
                f"Unexpected {key} count: {len(splits.get(key, []))} != {count}"
            )

    return meta


def completed_cache_ok(output: Path, source_meta: dict) -> bool:
    manifest = output / "manifest.json"
    if not manifest.is_file():
        return False

    try:
        meta = json.loads(manifest.read_text())
        sig = meta["signature"]
        if sig != source_meta["signature"]:
            return False
        if int(meta.get("samples", -1)) != EXPECTED_SAMPLES:
            return False

        for name in ("canonical.npy", *META_FILES):
            p = output / name
            if not p.is_file() or p.stat().st_size != int(meta["files"][name]):
                return False

        canonical = np.load(output / "canonical.npy", mmap_mode="r")
        if canonical.shape != EXPECTED_SHAPE or canonical.dtype != np.float32:
            return False

        labels = np.load(output / "labels.npy", mmap_mode="r")
        if labels.shape != (EXPECTED_SAMPLES,) or labels.dtype != np.int32:
            return False

        if json.loads((output / "ids.json").read_text()) != json.loads(
            (output / "ids.json").read_text()
        ):
            return False
    except Exception:
        return False

    return True


def copy_metadata(source: Path, output: Path) -> None:
    for name in META_FILES:
        shutil.copy2(source / name, output / name)


def build(args) -> Path:
    source = Path(args.source_cache).resolve()
    output = Path(args.output_cache).resolve()

    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use a separate audit-cache directory; do not modify the source cache.")

    source_meta = validate_source_cache(source)
    expected_sha = source_meta["signature"].get("source_sha256")
    if not expected_sha:
        raise ValueError("Source manifest is missing source_sha256.")

    dataset, dataset_sha = discover_dataset(args.dataset, expected_sha)

    print("=" * 110)
    print("NESTSAR R4 — CANONICAL-ONLY AUDIT CACHE REBUILD")
    print("=" * 110)
    print("Dataset     :", dataset)
    print("Dataset SHA :", dataset_sha)
    print("Source cache:", source)
    print("Output cache:", output)
    print("Preprocess  :", pp.VERSION)
    print("Cache ver.  :", CACHE_VERSION)
    print("Target shape:", EXPECTED_SHAPE, "float32")
    print("raw.npy     : NOT rebuilt (not needed by frozen-readout audit)")
    print("=" * 110, flush=True)

    output.mkdir(parents=True, exist_ok=True)

    lock_handle = (output / "build.lock").open("a")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_handle.close()
        raise RuntimeError("This audit-cache build is already running.")

    try:
        if completed_cache_ok(output, source_meta):
            print("✅ Completed audit cache already exists; reusing it.")
            return output

        # A manifest is the completion marker. If it exists but validation failed,
        # refuse to mix identities/results in the same directory.
        if (output / "manifest.json").exists():
            raise ValueError(
                f"{output} contains an invalid completed manifest. "
                "Choose a new --output-cache directory."
            )

        target_bytes = EXPECTED_SAMPLES * pp.FRAMES * pp.FEATURES * 4
        partial = output / "canonical.npy.partial"
        progress_path = output / "progress.json"

        partial_size = partial.stat().st_size if partial.exists() else 0
        free = shutil.disk_usage(output).free + partial_size
        if free < target_bytes + (768 << 20):
            raise RuntimeError(
                f"Canonical audit cache needs about {target_bytes / 2**30:.2f} GiB "
                f"plus reserve; only {free / 2**30:.2f} GiB is available."
            )

        if progress_path.exists():
            if not partial.is_file():
                raise ValueError("progress.json exists but canonical.npy.partial is missing.")
            progress = json.loads(progress_path.read_text())
            if progress.get("dataset_sha256") != dataset_sha:
                raise ValueError("Partial build belongs to a different dataset.")
            start = int(progress.get("next_index", 0))
            canonical = np.load(partial, mmap_mode="r+")
            if canonical.shape != EXPECTED_SHAPE or canonical.dtype != np.float32:
                raise ValueError("Partial canonical cache has the wrong shape/dtype.")
            if not 0 <= start <= EXPECTED_SAMPLES:
                raise ValueError(f"Invalid resume index {start}")
            print(f"Resuming canonical build at sample {start:,}/{EXPECTED_SAMPLES:,}.")
        else:
            if partial.exists():
                partial.unlink()
            canonical = np.lib.format.open_memmap(
                partial,
                mode="w+",
                dtype=np.float32,
                shape=EXPECTED_SHAPE,
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

        # Load the official annotation source only after all identity/storage gates pass.
        loaded = load_pickle(dataset)
        annotations = _value(
            loaded, ("annotations", "annotation", "samples", "data_list")
        )
        if len(annotations) != EXPECTED_SAMPLES:
            raise ValueError(
                f"Dataset has {len(annotations)} annotations; expected {EXPECTED_SAMPLES}."
            )

        source_ids = json.loads((source / "ids.json").read_text())
        source_labels = np.load(source / "labels.npy", mmap_mode="r")

        flush_every = max(1, int(args.flush_every))
        bar = tqdm(
            range(start, EXPECTED_SAMPLES),
            initial=start,
            total=EXPECTED_SAMPLES,
            desc="Canonical T16",
            unit="clip",
            dynamic_ncols=True,
        )

        for i in bar:
            annotation = annotations[i]
            sid = str(
                _value(
                    annotation,
                    ("frame_dir", "filename", "sample_name", "name", "id", "video_id"),
                )
            )
            label = int(
                _value(annotation, ("label", "action_label", "class", "target"))
            )

            if sid != source_ids[i]:
                raise ValueError(
                    f"Dataset/cache sample order mismatch at {i}: {sid} != {source_ids[i]}"
                )
            if label != int(source_labels[i]):
                raise ValueError(
                    f"Dataset/cache label mismatch at {i}: {label} != {int(source_labels[i])}"
                )

            x = pp.ordered_raw(
                _value(
                    annotation,
                    ("keypoint", "keypoints", "skeleton", "skeletons", "data"),
                ),
                source_meta["signature"].get("layout", "MTVC"),
            )
            if len(x) == 0 or not pp.raw_valid(x).any():
                raise ValueError(f"Empty skeleton at sample {i}: {sid}")

            canonical[i] = pp.features(x)

            if (i + 1) % flush_every == 0 or i + 1 == EXPECTED_SAMPLES:
                canonical.flush()
                if hasattr(canonical, "_mmap") and canonical._mmap is not None:
                    if hasattr(canonical._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                        canonical._mmap.madvise(mmap.MADV_DONTNEED)
                atomic_json(
                    progress_path,
                    {
                        "next_index": i + 1,
                        "dataset_sha256": dataset_sha,
                        "preprocessing": pp.VERSION,
                        "cache_version": CACHE_VERSION,
                    },
                )

        canonical.flush()
        del canonical, source_labels, annotations, loaded, x, annotation
        gc.collect()

        # np.load validates the full partial file before it becomes the canonical cache.
        check = np.load(partial, mmap_mode="r")
        if check.shape != EXPECTED_SHAPE or check.dtype != np.float32:
            raise ValueError(f"Built canonical layout is invalid: {check.shape}, {check.dtype}")
        del check

        final_canonical = output / "canonical.npy"
        os.replace(partial, final_canonical)
        copy_metadata(source, output)

        files = {
            name: (output / name).stat().st_size
            for name in ("canonical.npy", *META_FILES)
        }
        manifest = {
            "signature": source_meta["signature"],
            "samples": EXPECTED_SAMPLES,
            "canonical_bytes": EXPECTED_SAMPLES * pp.FRAMES * pp.FEATURES * 4,
            "split_counts": source_meta.get("split_counts", {}),
            "audit_only": True,
            "raw_required": False,
            "source_cache": str(source),
            "source_manifest_sha256": sha256(source / "manifest.json"),
            "files": files,
        }
        atomic_json(output / "manifest.json", manifest)
        progress_path.unlink(missing_ok=True)

        # Final strict validation through the same conditions the audit needs.
        canonical = np.load(final_canonical, mmap_mode="r")
        if canonical.shape != EXPECTED_SHAPE or canonical.dtype != np.float32:
            raise RuntimeError("Final canonical validation failed.")
        del canonical

        print("\n✅ CANONICAL-ONLY AUDIT CACHE READY")
        print("   ", output)
        print(
            f"   canonical.npy = {final_canonical.stat().st_size:,} bytes "
            f"({final_canonical.stat().st_size / 2**30:.3f} GiB)"
        )
        print("   Original corrupted training cache was NOT modified.")
        return output
    finally:
        lock_handle.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-cache",
        default="/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3",
    )
    p.add_argument(
        "--output-cache",
        default="/kaggle/working/NestSAR_R4_FROZEN_READOUT_CANONICAL_v1",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Optional exact ntu120_3danno.pkl path; otherwise auto-discover under /kaggle/input.",
    )
    p.add_argument("--flush-every", type=int, default=256)
    args = p.parse_args(argv)
    if args.flush_every < 1:
        p.error("--flush-every must be >= 1")
    return args


def main(argv=None):
    return build(parse_args(argv))


if __name__ == "__main__":
    main()

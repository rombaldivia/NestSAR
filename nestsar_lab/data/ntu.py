#!/usr/bin/env python3
"""NTU RGB+D annotation discovery, download, loading, and validation.

The official/preprocessed PYSKL annotation files are not vendored in this
repository.  They are downloaded from OpenMMLab when requested, or discovered
from Kaggle inputs / an explicit user path.
"""
from __future__ import annotations

import os
import pickle
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Mapping

NTU60_3DANNO_URL = (
    "https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/"
    "ntu60_3danno.pkl"
)
NTU120_3DANNO_URL = (
    "https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/"
    "ntu120_3danno.pkl"
)

_DATASETS = {
    "ntu60": {
        "filename": "ntu60_3danno.pkl",
        "url": NTU60_3DANNO_URL,
        "required_splits": {"xsub_train", "xsub_val", "xview_train", "xview_val"},
    },
    "ntu120": {
        "filename": "ntu120_3danno.pkl",
        "url": NTU120_3DANNO_URL,
        "required_splits": {"xsub_train", "xsub_val", "xset_train", "xset_val"},
    },
}


def _normalize_variant(variant: str) -> str:
    key = str(variant).strip().lower().replace("+", "").replace("rgbd", "")
    aliases = {
        "60": "ntu60",
        "ntu60": "ntu60",
        "120": "ntu120",
        "ntu120": "ntu120",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported NTU variant: {variant!r}; use 'ntu60' or 'ntu120'.")
    return aliases[key]


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    explicit = os.environ.get("NESTSAR_DATASET")
    if explicit and explicit != "auto":
        roots.append(Path(explicit).expanduser())
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            roots.append(root)
    roots.append(Path.cwd())
    return roots


def _find_existing(filename: str) -> Path | None:
    for root in _candidate_roots():
        if root.is_file() and root.name == filename:
            return root.resolve()
        if not root.exists() or not root.is_dir():
            continue
        direct = root / filename
        if direct.is_file():
            return direct.resolve()
        try:
            match = next(root.rglob(filename), None)
        except (OSError, PermissionError):
            match = None
        if match is not None and match.is_file():
            return match.resolve()
    return None


def _default_cache_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/nestsar_data")
    return Path(os.environ.get("NESTSAR_DATA_CACHE", "~/.cache/nestsar")).expanduser()


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part", dir=destination.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        print(f"NestSAR dataset download: {url}", flush=True)
        with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
        if tmp.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded empty dataset file from {url}")
        tmp.replace(destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return destination.resolve()


def ensure_ntu_dataset(
    dataset: str | Path = "auto",
    *,
    variant: str = "ntu120",
    allow_download: bool = True,
    cache_dir: str | Path | None = None,
    validate: bool = True,
) -> Path:
    """Resolve an NTU annotation pickle, downloading it only when needed.

    Resolution order:
      1. Explicit ``dataset`` path.
      2. ``NESTSAR_DATASET`` environment variable.
      3. Kaggle input/working directories and current directory.
      4. OpenMMLab download into a local cache, if ``allow_download`` is true.
    """
    key = _normalize_variant(variant)
    spec = _DATASETS[key]
    filename = str(spec["filename"])

    if str(dataset) != "auto":
        path = Path(dataset).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")
    else:
        path = _find_existing(filename)
        if path is None:
            if not allow_download:
                raise FileNotFoundError(
                    f"Could not find {filename}; attach it to Kaggle or pass an explicit path."
                )
            cache = Path(cache_dir).expanduser() if cache_dir is not None else _default_cache_dir()
            path = cache / filename
            if not path.is_file():
                path = _download(str(spec["url"]), path)
            else:
                path = path.resolve()

    if validate:
        validate_ntu_pickle(path, variant=key)
    print(f"NestSAR dataset: {path}", flush=True)
    return path


def load_ntu_pickle(path: str | Path) -> Any:
    path = Path(path)
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def validate_ntu_pickle(path: str | Path, *, variant: str = "ntu120") -> dict[str, Any]:
    """Validate the PYSKL/OpenMMLab NTU annotation container and split names."""
    key = _normalize_variant(variant)
    required_splits = set(_DATASETS[key]["required_splits"])
    data = load_ntu_pickle(path)

    if not isinstance(data, Mapping):
        raise TypeError(f"Expected dict-like NTU pickle, got {type(data)!r}")
    if "split" not in data or "annotations" not in data:
        raise KeyError("NTU pickle must contain top-level 'split' and 'annotations' fields")
    split = data["split"]
    annotations = data["annotations"]
    if not isinstance(split, Mapping):
        raise TypeError("NTU pickle 'split' field must be a mapping")
    missing = required_splits.difference(split.keys())
    if missing:
        raise KeyError(f"NTU pickle is missing required {key} splits: {sorted(missing)}")
    if not isinstance(annotations, (list, tuple)):
        raise TypeError("NTU pickle 'annotations' field must be a sequence")
    if len(annotations) == 0:
        raise ValueError("NTU pickle contains zero annotations")

    first = annotations[0]
    if not isinstance(first, Mapping):
        raise TypeError("NTU annotation entries must be mappings")
    required_annotation_fields = {"frame_dir", "label", "keypoint", "total_frames"}
    missing_fields = required_annotation_fields.difference(first.keys())
    if missing_fields:
        raise KeyError(f"NTU annotation is missing fields: {sorted(missing_fields)}")

    summary = {
        "variant": key,
        "path": str(Path(path).resolve()),
        "annotations": len(annotations),
        "splits": {name: len(split[name]) for name in sorted(required_splits)},
    }
    print(
        "NestSAR NTU validation: "
        f"variant={key} annotations={summary['annotations']} splits={summary['splits']}",
        flush=True,
    )
    return summary

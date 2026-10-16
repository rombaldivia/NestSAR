"""Inference-time head averaging and cross-run ensembling for finished R5 runs (diagnostic).

For each run directory it loads the best EMA checkpoint, evaluates the official validation split once and
reports top-1 of: the main classifier, every auxiliary head, and softmax averages of main + aux heads
(the aux heads already run at inference; averaging adds only ~0.04 MFLOPs). Per-sample probabilities are
saved to ``<run>/<protocol>/val_probs.npz`` so several runs can then be ensembled without re-evaluating.

    python -m experiments.nestsar_r5_t16.eval_heads --cache /tmp/HAND_CACHE --protocol xsub RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from .config import model_kwargs, validate_config
from .data import Dataset
from .model import NestSARR5T16
from . import preprocessing as pp


CACHE_SCHEMA = 2


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _array_file(array):
    filename = getattr(array, "filename", None)
    if filename is None:
        raise ValueError("Prediction caching requires the source dataset memmaps")
    path = Path(filename).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                shape=list(array.shape), dtype=str(array.dtype))


def evaluation_context(run_dir, protocol, dataset):
    """Resolve the actual checkpoint, inputs and ordered sample IDs before consulting predictions."""
    if protocol not in ("xsub", "xset"):
        raise ValueError(f"Unknown protocol: {protocol}")
    out = Path(run_dir) / protocol
    config_bytes = (out / "run_config.json").read_bytes()
    run = json.loads(config_bytes)
    config = validate_config(run["config"])
    checkpoint_bytes = (out / "best.msgpack").read_bytes()
    payload = serialization.msgpack_restore(checkpoint_bytes)
    if "config" in payload and validate_config(payload["config"]) != config:
        raise ValueError("Checkpoint and run configuration disagree")
    for record in (run, payload):
        if record.get("protocol", protocol) != protocol:
            raise ValueError("Checkpoint/run belongs to a different protocol")
    signature = dataset.meta["signature"]
    for declared in (run.get("cache"), payload.get("cache_signature")):
        if declared is not None and declared != signature:
            raise ValueError("Checkpoint/run and dataset cache signatures disagree")
    if payload.get("preprocessing_version", pp.VERSION) != pp.VERSION:
        raise ValueError("Checkpoint preprocessing differs from the active input contract")
    indices = np.asarray(dataset.splits[f"{protocol}_val"], np.int64)
    names = json.loads((Path(dataset.base_dir) / "ids.json").read_text())
    if len(names) != len(dataset.labels) or len(set(names)) != len(names):
        raise ValueError("Dataset sample IDs are missing or not unique")
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices) or \
            np.any(indices < 0) or np.any(indices >= len(names)):
        raise ValueError("Invalid or empty validation sample indices")
    ids = np.asarray(names, dtype=np.str_)[indices]
    labels = np.asarray(dataset.labels[indices], np.int32)
    if np.any(labels < 0) or np.any(labels >= 120):
        raise ValueError("Validation labels are outside the 120-class contract")
    source_files = [Path(__file__), Path(__file__).with_name("model.py"),
                    Path(__file__).with_name("preprocessing.py"), Path(__file__).with_name("config.py"),
                    Path(__file__).with_name("data.py"), Path(pp.pp.__file__)]
    identity = dict(schema=CACHE_SCHEMA, protocol=protocol, checkpoint_sha256=_digest(checkpoint_bytes),
                    run_config_sha256=_digest(config_bytes), preprocessing=pp.VERSION,
                    frames=pp.FRAMES, features=pp.FEATURES, jax_version=jax.__version__,
                    backend=jax.default_backend(), dataset_signature=signature,
                    dataset_files=[_array_file(dataset.base.canonical), _array_file(dataset.hand),
                                   _array_file(dataset.labels)],
                    ids_sha256=_digest(_json(ids.tolist()).encode()), labels_sha256=_digest(labels.tobytes()),
                    source_sha256=_digest(b"\0".join(p.read_bytes() for p in source_files)))
    return config, payload, indices, ids, labels, identity


def _cached(target, identity, ids, labels, heads):
    try:
        with np.load(target, allow_pickle=False) as z:
            if json.loads(str(z["identity_json"].item())) != identity or \
                    not np.array_equal(z["ids"], ids) or not np.array_equal(z["y"], labels):
                return None
            main, aux = z["main"], z["aux"]
        if main.shape != (len(ids), 120) or aux.shape != (len(ids), heads, 120):
            return None
        for probabilities in (main, aux):
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or \
                    not np.allclose(probabilities.sum(-1), 1.0, atol=1e-4):
                return None
        return labels, main, aux
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def _save_predictions(target, identity, ids, y, main, aux):
    """Publish a whole prediction file; an interrupted write cannot replace a valid one."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".val_probs-", suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
            np.savez_compressed(f, ids=ids, y=y, main=main, aux=aux, identity_json=np.asarray(_json(identity)))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def evaluate_bundle(run_dir, protocol, dataset, *, force=False, prediction_cache_dir=None):
    config, payload, val_ids, ids, labels, identity = evaluation_context(run_dir, protocol, dataset)
    out = Path(run_dir) / protocol
    target = (out if prediction_cache_dir is None else Path(prediction_cache_dir)) / "val_probs.npz"
    heads = 1 if config["variant"] == "no_hand_branch" else 2
    saved = None if force else _cached(target, identity, ids, labels, heads)
    details = dict(ids=ids, identity=identity, config=config, checkpoint_accuracy=payload.get("val_accuracy"))
    if saved is not None:
        y, main, aux = saved
        return dict(details, y=y, main=main, aux=aux)
    if target.exists():
        print(f"  {Path(run_dir).name}: recalculating predictions (forced, legacy, stale or invalid cache)", flush=True)
    model = NestSARR5T16(**model_kwargs(config))
    params = jax.device_put(payload["ema_params"])

    @jax.jit
    def probs(p, x):
        o = model.apply({"params": p}, x, training=False)
        return jax.nn.softmax(o["logits"], -1), jax.nn.softmax(o["stream_logits"], -1)

    ys, mains, auxs = [], [], []
    steps = math.ceil(len(val_ids) / config["eval_batch"])
    with closing(dataset.batches(val_ids, config["eval_batch"], config, protocol=protocol)) as batches:
        for i in range(steps):
            batch, _ = next(batches)
            n = int(batch["mask"].sum())
            m, a = probs(params, jax.device_put(batch["x"]))
            ys.append(np.asarray(batch["y"][:n]))
            mains.append(np.asarray(m[:n], np.float32))
            auxs.append(np.asarray(a[:n], np.float32))
            if i % 50 == 0 or i + 1 == steps:
                print(f"  {Path(run_dir).name}: {i + 1}/{steps}", flush=True)
    y, main, aux = np.concatenate(ys), np.concatenate(mains), np.concatenate(auxs)
    if not np.array_equal(y, labels):
        raise RuntimeError("Evaluation sample count/order/labels disagree with the declared split")
    if main.shape != (len(ids), 120) or aux.shape != (len(ids), heads, 120) or \
            not np.isfinite(main).all() or not np.isfinite(aux).all():
        raise RuntimeError("Invalid evaluation outputs")
    target.parent.mkdir(parents=True, exist_ok=True)
    _save_predictions(target, identity, ids, y, main, aux)
    return dict(details, y=y, main=main, aux=aux)


def evaluate(run_dir, protocol, dataset, *, force=False):
    """Keep the three-array API used by soup.py; every call validates its prediction provenance."""
    result = evaluate_bundle(run_dir, protocol, dataset, force=force)
    return result["y"], result["main"], result["aux"]


def acc(y, p):
    return 100.0 * float(np.mean(p.argmax(-1) == y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--protocol", default="xsub", choices=("xsub", "xset"))
    ap.add_argument("--force", action="store_true", help="recompute even a compatible prediction cache")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    dataset = Dataset(a.cache)
    mains, ids0 = [], None
    for run in a.runs:
        result = evaluate_bundle(run, a.protocol, dataset, force=a.force)
        y, main, aux, ids = (result[k] for k in ("y", "main", "aux", "ids"))
        if ids0 is not None and not np.array_equal(ids, ids0):
            raise RuntimeError("Runs were evaluated on different sample orders")
        ids0 = ids
        names = ["m4_aux", "hand_aux"][:aux.shape[1]] + [f"aux{i}" for i in range(2, aux.shape[1])]
        print(f"\n{Path(run).name} [{a.protocol}]")
        print(f"  main                 {acc(y, main):6.2f}")
        for i, nm in enumerate(names):
            print(f"  {nm:20s} {acc(y, aux[:, i]):6.2f}")
        for w in (0.5, 0.3, 0.2):   # main weight 1, each aux head weight w
            print(f"  main + {w:.1f}*aux       {acc(y, main + w * aux.sum(1)):6.2f}")
        mains.append(main)
    if len(mains) > 1:
        print(f"\nEnsemble of {len(mains)} runs (main heads, mean prob): {acc(y, np.mean(mains, 0)):6.2f}")
        print("  individual:", "  ".join(f"{acc(y, m):.2f}" for m in mains))


if __name__ == "__main__":
    main()

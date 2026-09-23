"""Frozen, whole-clip T16 segment-boundary test-time views (no training)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import preprocessing_corrected as pp
from .streaming.io_utils import Reporter, atomic_json
from .validation_reframe import (FrozenCache, MODEL_PARAMS, cache_location,
                                 checkpoint_location, dataset_location)

PROTOCOLS = ("xsub", "xset")
VIEWS = ("original", "early", "late")
MODES = (*VIEWS, "mean3", "anchored")
DEFAULT_CHECKPOINTS = "/kaggle/working/NestSAR_SM_ALL_T16_PREPROCESS_V2"
DEFAULT_OUT = "/kaggle/working/NestSAR_T16_WHOLE_CLIP_JITTER_AUDIT"


class FixedBoundaryShift:
    """The existing segment_bounds calls integers once per internal boundary."""

    def __init__(self, direction: int):
        if direction not in (-1, 1):
            raise ValueError("Direction must be -1 or +1")
        self.direction = direction

    def integers(self, low: int, high: int) -> int:
        if not (low <= self.direction < high):
            raise ValueError("Unexpected segment-boundary jitter range")
        return self.direction


def make_views(raw: np.ndarray, canonical: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Keep all raw frames; move only interior segment boundaries by one frame."""
    original = pp.features(raw) if canonical is None else np.asarray(canonical, np.float32)
    views = {"original": original,
             "early": pp.features(raw, rng=FixedBoundaryShift(-1), shift=1),
             "late": pp.features(raw, rng=FixedBoundaryShift(+1), shift=1)}
    if any(v.shape != (16, 750) or not np.isfinite(v).all() for v in views.values()):
        raise ValueError("Expected finite [16,750] features for every view")
    return views


class RawDataset:
    """Read the original NTU pickle in one worker at a time, without disk caching."""

    def __init__(self, source: Path):
        with source.open("rb") as f:
            try:
                data = pickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                data = pickle.load(f, encoding="latin1")
        from .streaming.data import _value, resolve_splits
        self.annotations = _value(data, ("annotations", "annotation", "samples", "data_list"))
        self.ids, self.splits = resolve_splits(self.annotations, _value(data, ("split", "splits")))
        self.labels = np.asarray([int(_value(a, ("label", "action_label", "class", "target")))
                                  for a in self.annotations], dtype=np.int64)
        if np.any((self.labels < 0) | (self.labels >= 120)):
            raise ValueError("NTU120 labels must be zero-based 0..119")

    def sample(self, index: int) -> np.ndarray:
        from .streaming.data import _value
        return pp.ordered_raw(_value(self.annotations[index],
                                     ("keypoint", "keypoints", "skeleton", "skeletons", "data")))


def evaluate(protocol: str, checkpoint_root: Path, cache: Path | None,
             source: Path | None, out: Path, batch_size: int, allow_cpu: bool) -> dict:
    import jax
    import jax.numpy as jnp
    from flax import serialization

    if not allow_cpu and (jax.default_backend() != "gpu" or jax.local_device_count() != 1):
        raise RuntimeError(f"Expected one isolated GPU for {protocol}; got {jax.devices()}")
    checkpoint = checkpoint_root / protocol / "best.msgpack"
    payload = serialization.msgpack_restore(checkpoint.read_bytes())
    meta = json.loads((checkpoint_root / protocol / "best.json").read_text())
    name = payload.get("model")
    if name not in MODEL_PARAMS:
        raise ValueError(f"Unsupported model: {name!r}")
    if (payload.get("protocol") != protocol or payload.get("preprocessing_version") != pp.VERSION
            or meta.get("model") != name or meta.get("params") != MODEL_PARAMS[name]
            or meta.get("pipeline_version") != payload.get("pipeline_version")):
        raise ValueError("Checkpoint protocol, preprocessing, model, or parameters do not match")
    if cache is not None:
        dataset = FrozenCache(str(cache))
        if payload.get("cache_signature") != dataset.meta["signature"]:
            raise ValueError("Checkpoint and raw cache signatures differ")
        ids = json.loads((cache / "ids.json").read_text())
    else:
        if source is None:
            raise ValueError("Need compatible cache or original NTU120 pickle")
        dataset = RawDataset(source)
        ids = dataset.ids
    indices = dataset.splits[f"{protocol}_val"]
    if len(indices) != (50919 if protocol == "xsub" else 59477):
        raise ValueError(f"Require full official {protocol} validation split, got {len(indices)}")
    if len(ids) != len(dataset.labels):
        raise ValueError("ID and label counts differ")
    if name == "NestSAR-SM-ALL-T16-G4-MOMENTS-v1":
        from .model_g4_moments import NestSARSMAllT16
    else:
        from .model import NestSARSMAllT16
    config = payload["config"]
    model = NestSARSMAllT16(**{k: config[k] for k in (
        "spatial_dim", "model_dim", "dropout", "controller_dim", "fast_rank",
        "head_rank", "sm_residual_scale", "head_residual_scale")})
    params = dict(payload["ema_params"])
    if name == "NestSAR-SM-ALL-T16-TRAIN-ATTN-v1":
        del params["training_attention_supervisor"]
    if sum(int(a.size) for a in jax.tree.leaves(params)) != MODEL_PARAMS[name]:
        raise ValueError("Frozen parameter count differs from checkpoint metadata")
    params = jax.device_put(params)
    infer = jax.jit(lambda p, x: model.apply({"params": p}, x, training=False)["logits"])
    target = out / protocol
    target.mkdir(parents=True, exist_ok=True)
    report = Reporter(target / "status.json")
    report(phase="Frozen whole-clip T16 inference", current=0, total=len(indices))
    correct = {mode: np.zeros(120, np.int64) for mode in MODES}
    support = np.zeros(120, np.int64)
    flips = {mode: {"fixed": 0, "broken": 0} for mode in MODES[1:]}
    rows = []
    for offset in range(0, len(indices), batch_size):
        ix = indices[offset:offset + batch_size]
        n = len(ix)
        batches = {view: np.zeros((batch_size, 16, 750), np.float32) for view in VIEWS}
        for j, i in enumerate(ix):
            built = make_views(dataset.sample(i),
                               dataset.canonical[i] if cache is not None else None)
            for view in VIEWS:
                batches[view][j] = built[view]
        y = np.asarray(dataset.labels[ix], np.int64)
        np.add.at(support, y, 1)
        logits = {view: np.asarray(jax.block_until_ready(infer(params, jnp.asarray(batches[view]))))[:n]
                  for view in VIEWS}
        logits["mean3"] = (logits["original"] + logits["early"] + logits["late"]) / 3
        logits["anchored"] = (2 * logits["original"] + logits["early"] + logits["late"]) / 4
        if any(not np.isfinite(a).all() for a in logits.values()):
            raise FloatingPointError("Nonfinite model logits")
        preds = {mode: a.argmax(-1) for mode, a in logits.items()}
        for mode in MODES:
            np.add.at(correct[mode], y, preds[mode] == y)
            if mode != "original":
                flips[mode]["fixed"] += int(np.sum((preds["original"] != y) & (preds[mode] == y)))
                flips[mode]["broken"] += int(np.sum((preds["original"] == y) & (preds[mode] != y)))
        rows.extend([ids[i], int(y[j]), *(int(preds[mode][j]) for mode in MODES)]
                    for j, i in enumerate(ix))
        report(phase="Frozen whole-clip T16 inference", current=offset + n, total=len(indices),
               scores={mode: float(correct[mode].sum() / (offset + n)) for mode in MODES})
    baseline = float(correct["original"].sum() / len(indices))
    if meta.get("val_accuracy") is not None and abs(baseline - float(meta["val_accuracy"])) > 0.001:
        raise RuntimeError(f"Baseline {baseline:.6f} differs from checkpoint score "
                           f"{meta['val_accuracy']:.6f}; check data and checkpoint identity")
    with (target / "sample_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "label", *(f"pred_{mode}" for mode in MODES)])
        writer.writerows(rows)
    with (target / "class_recall.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "support", *(f"recall_{mode}" for mode in MODES)])
        for label in range(120):
            writer.writerow([label, int(support[label]), *[
                float(correct[m][label] / support[label]) if support[label] else None for m in MODES]])
    result = {"protocol": protocol, "model": name, "checkpoint": str(checkpoint),
              "sample_count": len(indices), "frozen": True, "training": False,
              "params": MODEL_PARAMS[name], "scores": {
                  mode: {"accuracy": float(correct[mode].sum() / len(indices)),
                         "delta_pp": 100 * (correct[mode].sum() / len(indices) - baseline)}
                  for mode in MODES}, "flips": flips}
    atomic_json(target / "result.json", result)
    report(phase="Done", current=len(indices), total=len(indices), done=True,
           scores={mode: result["scores"][mode]["accuracy"] for mode in MODES})
    return result


def launch(settings: dict | None = None) -> dict:
    from tqdm.auto import tqdm
    settings = {} if settings is None else dict(settings)
    unknown = set(settings) - {"checkpoint_root", "cache", "dataset", "outdir", "batch_size", "allow_cpu"}
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    checkpoint_root, pipeline = checkpoint_location(Path(settings.get("checkpoint_root", DEFAULT_CHECKPOINTS)))
    cache = cache_location(Path(settings.get("cache") or "/kaggle/working/_auto_detect_cache"), pipeline)
    source = None if cache is not None else dataset_location(settings.get("dataset"))
    out = Path(settings.get("outdir", DEFAULT_OUT))
    out.mkdir(parents=True, exist_ok=True)
    batch_size = int(settings.get("batch_size", 128))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    allow_cpu = bool(settings.get("allow_cpu", False))
    print(f"Checkpoints: {checkpoint_root}\nRaw input: {cache or source}", flush=True)
    if cache is None:
        print("No compatible cache: processing the original pickle one protocol at a time to bound RAM.", flush=True)
    root = Path(__file__).resolve().parents[2]
    bars = [tqdm(total=n, desc=f"{protocol.upper()} frozen T16", position=j, leave=True)
            for j, (protocol, n) in enumerate(zip(PROTOCOLS, (50919, 59477)))]
    processes = []
    logs = []
    try:
        for gpu, protocol in enumerate(PROTOCOLS):
            (out / protocol).mkdir(exist_ok=True)
            logfile = (out / protocol / "worker.log").open("w")
            logs.append(logfile)
            env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES=str(gpu),
                       JAX_PLATFORMS="cpu" if allow_cpu else "cuda",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1")
            cmd = [sys.executable, "-u", "-m", "experiments.nestsar_sm_all_t16.validation_jitter_t16",
                   "--worker", "--protocol", protocol, "--checkpoint-root", str(checkpoint_root),
                   "--outdir", str(out), "--batch-size", str(batch_size)]
            if cache is not None:
                cmd.extend(["--cache", str(cache)])
            else:
                cmd.extend(["--dataset", str(source)])
            if allow_cpu:
                cmd.append("--allow-cpu")
            def start():
                return subprocess.Popen(cmd, cwd=root, env=env, stdout=logfile,
                                        stderr=subprocess.STDOUT)
            child = start()
            processes.append(child)
            # Loading the full pickle twice can exhaust Kaggle RAM. Shared
            # mmap cache allows parallel protocol workers safely.
            if cache is None:
                active = [child]
            else:
                active = []
            while active and any(p.poll() is None for p in active):
                _refresh(out, bars)
                time.sleep(0.7)
        while any(p.poll() is None for p in processes):
            _refresh(out, bars)
            time.sleep(0.7)
        _refresh(out, bars)
        failures = [(protocol, p.returncode) for protocol, p in zip(PROTOCOLS, processes) if p.returncode]
        if failures:
            detail = "\n".join(f"{protocol} exit {code}:\n" +
                               "\n".join((out / protocol / "worker.log").read_text(errors="replace").splitlines()[-20:])
                               for protocol, code in failures)
            raise RuntimeError("Frozen T16 audit failed:\n" + detail)
        results = {p: json.loads((out / p / "result.json").read_text()) for p in PROTOCOLS}
        for p in PROTOCOLS:
            print(f"{p.upper()} " + " | ".join(
                f"{mode}: {100*results[p]['scores'][mode]['accuracy']:.4f}% "
                f"({results[p]['scores'][mode]['delta_pp']:+.3f} pp)" for mode in MODES), flush=True)
        atomic_json(out / "summary.json", results)
        return results
    except BaseException:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        raise
    finally:
        for bar in bars:
            bar.close()
        for log in logs:
            log.close()


def _refresh(out: Path, bars: list) -> None:
    for p, bar in zip(PROTOCOLS, bars):
        marker = out / p / "status.json"
        if not marker.is_file():
            continue
        try:
            state = json.loads(marker.read_text())
        except json.JSONDecodeError:
            continue
        bar.n = min(int(state.get("current", 0)), bar.total)
        scores = state.get("scores", {})
        bar.set_postfix_str(" | ".join(f"{k}:{v*100:.2f}%" for k, v in scores.items()
                                     if k in ("original", "anchored")), refresh=False)
        bar.refresh()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--protocol", choices=PROTOCOLS)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--cache")
    parser.add_argument("--dataset")
    parser.add_argument("--outdir", default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-cpu", action="store_true")
    a = parser.parse_args()
    if a.worker:
        if a.protocol is None:
            parser.error("--worker requires --protocol")
        evaluate(a.protocol, Path(a.checkpoint_root), Path(a.cache) if a.cache else None,
                 Path(a.dataset) if a.dataset else None, Path(a.outdir), a.batch_size, a.allow_cpu)
    else:
        launch({"checkpoint_root": a.checkpoint_root, "cache": a.cache,
                "dataset": a.dataset, "outdir": a.outdir,
                "batch_size": a.batch_size, "allow_cpu": a.allow_cpu})


if __name__ == "__main__":
    main()

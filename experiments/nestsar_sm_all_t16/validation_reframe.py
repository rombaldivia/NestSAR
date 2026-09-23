"""Paired, inference-only centered-window audit of an existing P2 checkpoint.

Run this file through runpy.run_path with NESTSAR_REFRAME_SETTINGS in Kaggle, or
use --worker for a single isolated protocol/GPU. Never initializes an optimizer.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import preprocessing_corrected as pp
from .streaming.io_utils import Reporter, atomic_json

MODES = ("original", "center64", "center32", "center16")
EXPECTED_PARAMS = 1_826_556
DEFAULT_CHECKPOINTS = "/kaggle/working/NestSAR_SM_ALL_T16_PERSON_AWARE_P2_v3"
DEFAULT_CACHE = "/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3"
DEFAULT_OUT = "/kaggle/working/NestSAR_P2_VALIDATION_REFRAME_v1"


class FrozenCache:
    """Read compatible versioned caches without rebuilding or changing their bytes."""

    def __init__(self, cache: str):
        path = Path(cache)
        self.meta = json.loads((path / "manifest.json").read_text())
        if self.meta["signature"].get("preprocessing") != pp.VERSION:
            raise ValueError("Cache was built with different preprocessing")
        for name, size in self.meta["files"].items():
            if not (path / name).is_file() or (path / name).stat().st_size != size:
                raise ValueError(f"Missing/incomplete cache file: {name}")
        self.raw = np.load(path / "raw.npy", mmap_mode="r")
        self.canonical = np.load(path / "canonical.npy", mmap_mode="r")
        self.shape = np.load(path / "shape.npy", mmap_mode="r")
        self.offsets = np.load(path / "offsets.npy", mmap_mode="r")
        self.labels = np.load(path / "labels.npy", mmap_mode="r")
        self.splits = json.loads((path / "splits.json").read_text())
        if len(self.labels) != self.meta["samples"] or self.canonical.shape != (len(self.labels), 16, 750):
            raise ValueError("Cache shape does not match P2 token contract")

    def sample(self, index: int) -> np.ndarray:
        frames, people = map(int, self.shape[index])
        raw = self.raw[self.offsets[index]:self.offsets[index + 1]].reshape(frames, people, 25, 3)
        if people == 2:
            return np.asarray(raw)
        if people != 1:
            raise ValueError(f"Invalid actor count: {people}")
        result = np.zeros((frames, 2, 25, 3), np.float32)
        result[:, 0] = raw[:, 0]
        return result


def centered(x: np.ndarray, window: int) -> np.ndarray:
    """CD-Former-style temporal crop; preserve original actor order and padding."""
    if len(x) <= window:
        return x
    start = (len(x) - window) // 2
    return x[start:start + window]


def variants(raw: np.ndarray, original: np.ndarray) -> dict[str, np.ndarray]:
    result = {"original": np.asarray(original, np.float32)}
    for window in (64, 32, 16):
        result[f"center{window}"] = pp.features(centered(raw, window))
    if any(value.shape != (16, 750) or not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Invalid reframed feature shape or nonfinite values")
    return result


def worker(protocol: str, cache: str, checkpoint_root: str, outdir: str,
           batch_size: int = 128, max_val_samples: int = 0, allow_cpu: bool = False) -> dict:
    # CUDA_VISIBLE_DEVICES and JAX_PLATFORMS are set by launch BEFORE this import.
    import jax
    import jax.numpy as jnp
    from flax import serialization

    from .model import NestSARSMAllT16
    if not allow_cpu and (jax.default_backend() != "gpu" or jax.local_device_count() != 1):
        raise RuntimeError(f"Expected one isolated GPU for {protocol}: {jax.devices()}")
    dataset = FrozenCache(cache)
    indices = dataset.splits[f"{protocol}_val"]
    if max_val_samples:
        indices = indices[:max_val_samples]
    if not indices:
        raise RuntimeError(f"No validation samples for {protocol}")
    checkpoint = Path(checkpoint_root) / protocol / "best.msgpack"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Existing P2 EMA checkpoint required: {checkpoint}")
    payload = serialization.msgpack_restore(checkpoint.read_bytes())
    if payload.get("model") not in ("NestSAR-SM-ALL-T16-v1", "NestSAR-SM-ALL-T16-TRAIN-ATTN-v1"):
        raise ValueError(f"Expected P2/P2 attention-supervised checkpoint, got {payload.get('model')!r}")
    if payload.get("protocol") != protocol or payload.get("preprocessing_version") != pp.VERSION:
        raise ValueError("Checkpoint protocol/preprocessing does not match this audit")
    if payload.get("cache_signature") != dataset.meta["signature"]:
        raise ValueError("Checkpoint was trained on a different raw/cache representation")
    config = payload["config"]
    model = NestSARSMAllT16(**{k: config[k] for k in (
        "spatial_dim", "model_dim", "dropout", "controller_dim", "fast_rank",
        "head_rank", "sm_residual_scale", "head_residual_scale")})
    params = dict(payload["ema_params"])
    # This auxiliary head was used only while training. The deployed model is
    # byte-for-byte P2 after its parameter subtree is removed.
    if payload["model"] == "NestSAR-SM-ALL-T16-TRAIN-ATTN-v1":
        if "training_attention_supervisor" not in params:
            raise ValueError("Attention-trained checkpoint lacks its training-only parameter subtree")
        del params["training_attention_supervisor"]
    if sum(int(x.size) for x in jax.tree.leaves(params)) != EXPECTED_PARAMS:
        raise ValueError("This audit accepts only the verified P2 deploy graph")

    out = Path(outdir) / protocol
    out.mkdir(parents=True, exist_ok=True)
    report = Reporter(out / "status.json")
    report(phase="Validating frozen P2", current=0, total=len(indices), checkpoint=str(checkpoint))
    infer = jax.jit(lambda p, x: model.apply({"params": p}, x, training=False)["logits"])
    params = jax.device_put(params)
    correct = {mode: np.zeros(120, np.int64) for mode in MODES}
    top5 = {mode: 0 for mode in MODES}
    support = np.zeros(120, np.int64)
    flips = {mode: {"fixed": 0, "broken": 0, "different": 0} for mode in MODES[1:]}
    shortened = {mode: 0 for mode in MODES[1:]}
    ids = json.loads((Path(cache) / "ids.json").read_text())
    if len(ids) != dataset.meta["samples"]:
        raise ValueError("Sample IDs do not match the canonical cache")
    rows = []
    for offset in range(0, len(indices), batch_size):
        ix = indices[offset:offset + batch_size]
        n = len(ix)
        data = {mode: np.zeros((batch_size, 16, 750), np.float32) for mode in MODES}
        for position, index in enumerate(ix):
            raw = dataset.sample(index)
            built = variants(raw, dataset.canonical[index])
            for mode in MODES:
                data[mode][position] = built[mode]
            for window in (64, 32, 16):
                shortened[f"center{window}"] += int(len(raw) > window)
        y = np.asarray(dataset.labels[ix], dtype=np.int64)
        np.add.at(support, y, 1)
        pred = {}
        for mode in MODES:
            logits = np.asarray(jax.block_until_ready(infer(params, jnp.asarray(data[mode]))))[:n]
            if not np.isfinite(logits).all():
                raise FloatingPointError(f"Nonfinite predictions in {mode}")
            pred[mode] = logits.argmax(-1)
            np.add.at(correct[mode], y, (pred[mode] == y).astype(np.int64))
            top5[mode] += int(np.sum(np.any(np.argpartition(logits, -5, axis=1)[:, -5:] == y[:, None], axis=1)))
        for mode in MODES[1:]:
            flips[mode]["fixed"] += int(np.sum((pred["original"] != y) & (pred[mode] == y)))
            flips[mode]["broken"] += int(np.sum((pred["original"] == y) & (pred[mode] != y)))
            flips[mode]["different"] += int(np.sum(pred["original"] != pred[mode]))
        for position, index in enumerate(ix):
            rows.append([ids[index], int(y[position]), int(dataset.shape[index, 0]),
                         *(int(pred[mode][position]) for mode in MODES)])
        report(current=offset + n, total=len(indices),
               original_acc=float(correct["original"].sum() / (offset + n)),
               scores={mode: float(correct[mode].sum() / (offset + n)) for mode in MODES})

    with (out / "sample_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "label", "raw_frames", *(f"pred_{mode}" for mode in MODES)])
        writer.writerows(rows)
    with (out / "class_recall.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "support", *(f"recall_{mode}" for mode in MODES),
                         *(f"delta_pp_{mode}" for mode in MODES[1:])])
        for label in range(120):
            recalls = [float(correct[mode][label] / support[label]) if support[label] else None for mode in MODES]
            writer.writerow([label, int(support[label]), *recalls,
                             *[100 * (v - recalls[0]) if v is not None and recalls[0] is not None else None for v in recalls[1:]]])

    baseline = Path(checkpoint_root) / protocol / "best.json"
    saved = json.loads(baseline.read_text()) if baseline.is_file() else {}
    result = {"protocol": protocol, "count": len(indices), "checkpoint": str(checkpoint),
              "epoch": int(payload["epoch"]), "frozen": True, "training": False,
              "best_json_accuracy": saved.get("val_accuracy"),
              "scores": {mode: {"accuracy": float(correct[mode].sum() / len(indices)),
                                "top5": float(top5[mode] / len(indices))} for mode in MODES},
              "delta_pp": {mode: 100 * (correct[mode].sum() - correct["original"].sum()) / len(indices)
                           for mode in MODES[1:]},
              "paired_flips": flips, "clips_cropped": shortened,
              "note": "A validation-only distribution-shift diagnostic; reframe training effect is unmeasured."}
    if not max_val_samples and saved.get("val_accuracy") is not None:
        if abs(result["scores"]["original"]["accuracy"] - float(saved["val_accuracy"])) > 1e-3:
            raise RuntimeError("Original-view score differs from saved best.json; check checkpoint/cache identity")
    atomic_json(out / "result.json", result)
    report(phase="Done", current=len(indices), total=len(indices), done=True)
    return result


def launch(settings: dict) -> dict:
    from tqdm.auto import tqdm

    allowed = {"cache", "checkpoint_root", "outdir", "batch_size", "max_val_samples", "allow_cpu"}
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    cache = Path(settings.get("cache", DEFAULT_CACHE))
    checkpoint_root = Path(settings.get("checkpoint_root", DEFAULT_CHECKPOINTS))
    out = Path(settings.get("outdir", DEFAULT_OUT))
    batch_size = int(settings.get("batch_size", 128))
    max_val_samples = int(settings.get("max_val_samples", 0))
    allow_cpu = bool(settings.get("allow_cpu", False))
    if batch_size < 1 or max_val_samples < 0:
        raise ValueError("Invalid batch size or validation cap")
    if not (cache / "manifest.json").is_file():
        raise FileNotFoundError(f"Existing P2 cache required: {cache}")
    splits = json.loads((cache / "splits.json").read_text())
    for protocol in ("xsub", "xset"):
        if not (checkpoint_root / protocol / "best.msgpack").is_file():
            raise FileNotFoundError(f"Missing {protocol} P2 checkpoint in {checkpoint_root}")
        if len(splits[f"{protocol}_val"]) not in (50919, 59477):
            raise ValueError(f"Unexpected official validation count for {protocol}")
    if len(splits["xsub_val"]) != 50919 or len(splits["xset_val"]) != 59477:
        raise ValueError("Require official NTU120 protocol splits")
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-u", "-m", "experiments.nestsar_sm_all_t16.validation_reframe",
               "--worker", "--cache", str(cache), "--checkpoint-root", str(checkpoint_root),
               "--outdir", str(out), "--batch-size", str(batch_size),
               "--max-val-samples", str(max_val_samples)]
    bars, children, logs = [], [], []
    try:
        for gpu, protocol in enumerate(("xsub", "xset")):
            (out / protocol).mkdir(exist_ok=True)
            env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES=str(gpu),
                       JAX_PLATFORMS="cpu" if allow_cpu else "cuda",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1")
            logfile = (out / protocol / "worker.log").open("w")
            logs.append(logfile)
            cmd = [*command, "--protocol", protocol] + (["--allow-cpu"] if allow_cpu else [])
            children.append(subprocess.Popen(cmd, env=env, stdout=logfile, stderr=subprocess.STDOUT))
            bars.append(tqdm(total=max_val_samples or len(splits[f"{protocol}_val"]),
                             desc=f"{protocol.upper()} GPU{gpu} frozen P2", position=gpu, leave=True))
        while any(child.poll() is None for child in children):
            for protocol, bar in zip(("xsub", "xset"), bars):
                path = out / protocol / "status.json"
                if path.is_file():
                    try:
                        progress = json.loads(path.read_text())
                    except json.JSONDecodeError:
                        continue
                    bar.n = min(int(progress.get("current", 0)), bar.total)
                    scores = progress.get("scores", {})
                    bar.set_postfix_str(" | ".join(f"{k}:{v*100:.2f}%" for k, v in scores.items()), refresh=False)
                    bar.refresh()
            time.sleep(0.5)
        failures = [(p, child.returncode) for p, child in zip(("xsub", "xset"), children) if child.returncode]
        if failures:
            tails = []
            for protocol, code in failures:
                content = (out / protocol / "worker.log").read_text(errors="replace").splitlines()
                tails.append(f"{protocol}: exit {code}\n" + "\n".join(content[-18:]))
            raise RuntimeError("Validation-only audit failed:\n" + "\n".join(tails))
        results = {p: json.loads((out / p / "result.json").read_text()) for p in ("xsub", "xset")}
        for protocol, bar in zip(("xsub", "xset"), bars):
            bar.n = bar.total
            bar.refresh()
            print(f"{protocol.upper()} original {results[protocol]['scores']['original']['accuracy']*100:.4f}% | "
                  + " | ".join(f"{m} {results[protocol]['scores'][m]['accuracy']*100:.4f}% "
                                f"({results[protocol]['delta_pp'][m]:+.3f} pp)" for m in MODES[1:]))
        return results
    except BaseException:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    finally:
        for bar in bars:
            bar.close()
        for logfile in logs:
            logfile.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--protocol", choices=("xsub", "xset"))
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--outdir", default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if not args.protocol:
            parser.error("--worker needs --protocol")
        worker(args.protocol, args.cache, args.checkpoint_root, args.outdir,
               args.batch_size, args.max_val_samples, args.allow_cpu)
    else:
        launch({k: getattr(args, k) for k in ("cache", "checkpoint_root", "outdir",
                                                "batch_size", "max_val_samples", "allow_cpu")})


if __name__ == "__main__":
    main()

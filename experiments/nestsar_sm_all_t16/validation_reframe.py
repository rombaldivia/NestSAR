"""Paired, inference-only centered-window audit of an existing P2 or G4 checkpoint.

Run this file through runpy.run_path with NESTSAR_REFRAME_SETTINGS in Kaggle, or
use --worker for a single isolated protocol/GPU. Never initializes an optimizer.
"""
from __future__ import annotations

import argparse
import concurrent.futures
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
MODEL_PARAMS = {
    "NestSAR-SM-ALL-T16-v1": 1_826_556,
    "NestSAR-SM-ALL-T16-TRAIN-ATTN-v1": 1_826_556,
    "NestSAR-SM-ALL-T16-G4-MOMENTS-v1": 1_827_452,
}
DEFAULT_CHECKPOINTS = "/kaggle/working/NestSAR_G4_TEMPORAL_MOMENTS_T16_FULL_OFFICIAL"
DEFAULT_CACHE = "/kaggle/working/NestSAR_G4_TEMPORAL_MOMENTS_T16_CACHE"
DEFAULT_OUT = "/kaggle/working/NestSAR_G4_VALIDATION_REFRAME_v1"


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
            raise ValueError("Cache shape does not match T16 token contract")

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


def _mounted_file_candidates(filename: str):
    """Search current Kaggle attachments and working files without loading them."""
    for base in (Path("/kaggle/input"), Path("/kaggle/working")):
        if base.exists():
            yield from sorted(base.rglob(filename))


def checkpoint_location(preferred: Path) -> tuple[Path, str]:
    """Find both frozen protocol weights before doing any cache work."""
    roots = [preferred]
    roots.extend(path.parent.parent for path in _mounted_file_candidates("best.msgpack")
                 if path.parent.name == "xsub")
    candidates = []
    for root in dict.fromkeys(roots):
        if not all((root / p / "best.msgpack").is_file() for p in ("xsub", "xset")):
            continue
        meta = [root / p / "best.json" for p in ("xsub", "xset")]
        if not all(path.is_file() for path in meta):
            continue
        values = [json.loads(path.read_text()) for path in meta]
        if values[0].get("model") not in MODEL_PARAMS:
            continue
        if values[0].get("model") != values[1].get("model"):
            continue
        if any(value.get("params") != MODEL_PARAMS[values[0]["model"]] for value in values):
            continue
        if any(value.get("preprocessing_version") != pp.VERSION for value in values):
            continue
        versions = [value.get("pipeline_version") for value in values]
        if versions[0] is None or versions[0] != versions[1]:
            continue
        candidates.append((root, versions[0]))
    if preferred in (root for root, _ in candidates):
        return next(pair for pair in candidates if pair[0] == preferred)
    if len(candidates) == 1:
        return candidates[0]
    found = "\n".join(f"  {root}" for root, _ in candidates) or "  none"
    raise FileNotFoundError(
        "Frozen compatible XSUB and XSET best.msgpack + best.json were not found as a pair. "
        "Attach the saved Kaggle output containing both protocol checkpoints, "
        "then set checkpoint_root to that directory. Candidate roots:\n" + found
    )


def cache_location(preferred: Path, pipeline_version: str) -> Path | None:
    roots = [preferred]
    roots.extend(path.parent for path in _mounted_file_candidates("manifest.json"))
    for root in dict.fromkeys(roots):
        marker = root / "manifest.json"
        if not marker.is_file() or not (root / "splits.json").is_file():
            continue
        try:
            signature = json.loads(marker.read_text())["signature"]
        except (ValueError, KeyError):
            continue
        if (signature.get("preprocessing") == pp.VERSION
                and signature.get("cache_version") == pipeline_version):
            return root
    return None


def dataset_location(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"NTU120 pickle not found: {path}")
        return path
    matches = list(_mounted_file_candidates("ntu120_3danno.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Compatible cache absent and found {len(matches)} NTU120 pickles. "
            "Attach the dataset or set dataset to its exact path."
        )
    return matches[0]


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
        raise FileNotFoundError(f"Existing EMA checkpoint required: {checkpoint}")
    payload = serialization.msgpack_restore(checkpoint.read_bytes())
    model_name = payload.get("model")
    if model_name not in MODEL_PARAMS:
        raise ValueError(f"Unsupported checkpoint model: {model_name!r}")
    if payload.get("protocol") != protocol or payload.get("preprocessing_version") != pp.VERSION:
        raise ValueError("Checkpoint protocol/preprocessing does not match this audit")
    if payload.get("cache_signature") != dataset.meta["signature"]:
        raise ValueError("Checkpoint was trained on a different raw/cache representation")
    saved_meta = json.loads((checkpoint.parent / "best.json").read_text())
    if (saved_meta.get("model") != model_name
            or saved_meta.get("pipeline_version") != payload.get("pipeline_version")
            or saved_meta.get("params") != MODEL_PARAMS[model_name]):
        raise ValueError("Checkpoint and best.json model/pipeline/parameter identity differ")
    if model_name == "NestSAR-SM-ALL-T16-G4-MOMENTS-v1":
        # Exact source from experiment/nestsar-g4-temporal-moments-t16; keep the
        # trained G4 chunker/parameter names, without touching the P2 model.
        from .model_g4_moments import NestSARSMAllT16
    else:
        from .model import NestSARSMAllT16
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
    if sum(int(x.size) for x in jax.tree.leaves(params)) != MODEL_PARAMS[model_name]:
        raise ValueError(f"Parameter count does not match {model_name}")

    out = Path(outdir) / protocol
    out.mkdir(parents=True, exist_ok=True)
    report = Reporter(out / "status.json")
    report(phase="Validating frozen model", current=0, total=len(indices), checkpoint=str(checkpoint))
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
    result = {"protocol": protocol, "model": model_name, "count": len(indices), "checkpoint": str(checkpoint),
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

    allowed = {"cache", "checkpoint_root", "dataset", "outdir", "batch_size", "max_val_samples", "allow_cpu"}
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    requested_cache = Path(settings.get("cache", DEFAULT_CACHE))
    requested_checkpoints = Path(settings.get("checkpoint_root", DEFAULT_CHECKPOINTS))
    out = Path(settings.get("outdir", DEFAULT_OUT))
    batch_size = int(settings.get("batch_size", 128))
    max_val_samples = int(settings.get("max_val_samples", 0))
    allow_cpu = bool(settings.get("allow_cpu", False))
    if batch_size < 1 or max_val_samples < 0:
        raise ValueError("Invalid batch size or validation cap")
    checkpoint_root, pipeline_version = checkpoint_location(requested_checkpoints)
    model_name = json.loads((checkpoint_root / "xsub" / "best.json").read_text())["model"]
    cache = cache_location(requested_cache, pipeline_version)
    if cache is None and str(requested_cache).startswith("/kaggle/input/"):
        raise ValueError("Cache is absent and /kaggle/input is read-only; use a /kaggle/working cache path")
    out.mkdir(parents=True, exist_ok=True)
    bars, children, logs = [], [], []
    try:
        if cache is None:
            source = dataset_location(settings.get("dataset"))
            cache = requested_cache
            if (cache / "manifest.json").is_file():
                # Preserve a complete cache from another experiment/version.
                cache = cache.with_name(cache.name + "__" + pipeline_version)
            print(f"Compatible cache absent; preparing it from {source} (no model training).")
            bars = [tqdm(total=1, desc=f"{p.upper()} prepare cache", position=i, leave=True)
                    for i, p in enumerate(("xsub", "xset"))]
            status_path = out / "cache_prepare_status.json"
            from .streaming import data as cache_builder
            previous_version = cache_builder.VERSION
            cache_builder.VERSION = pipeline_version
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(cache_builder.prepare, source, cache, status_path, "MTVC")
                    while not future.done():
                        if status_path.is_file():
                            try:
                                status = json.loads(status_path.read_text())
                            except json.JSONDecodeError:
                                status = {}
                            for bar in bars:
                                total = max(int(status.get("total", 1)), 1)
                                if bar.total != total:
                                    bar.reset(total=total)
                                bar.n = min(int(status.get("current", 0)), total)
                                bar.set_description_str(f"{bar.desc.split()[0]} {status.get('phase', 'Prepare cache')}",
                                                        refresh=False)
                                bar.refresh()
                        time.sleep(0.5)
                    future.result()
            finally:
                cache_builder.VERSION = previous_version
            print(f"Prepared cache: {cache}")
        else:
            print(f"Reusing compatible cache: {cache}")
        print(f"Frozen checkpoint root: {checkpoint_root}")
        splits = json.loads((cache / "splits.json").read_text())
        if len(splits["xsub_val"]) != 50919 or len(splits["xset_val"]) != 59477:
            raise ValueError("Require complete official NTU120 validation splits")
        for i, protocol in enumerate(("xsub", "xset")):
            total = max_val_samples or len(splits[f"{protocol}_val"])
            if i < len(bars):
                bars[i].reset(total=total)
                bars[i].set_description_str(f"{protocol.upper()} GPU{i} frozen {model_name}", refresh=False)
            else:
                bars.append(tqdm(total=total, desc=f"{protocol.upper()} GPU{i} frozen {model_name}",
                                 position=i, leave=True))
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "-u", "-m", "experiments.nestsar_sm_all_t16.validation_reframe",
                   "--worker", "--cache", str(cache), "--checkpoint-root", str(checkpoint_root),
                   "--outdir", str(out), "--batch-size", str(batch_size),
                   "--max-val-samples", str(max_val_samples)]
        for gpu, protocol in enumerate(("xsub", "xset")):
            (out / protocol).mkdir(exist_ok=True)
            env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES=str(gpu),
                       JAX_PLATFORMS="cpu" if allow_cpu else "cuda",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1")
            logfile = (out / protocol / "worker.log").open("w")
            logs.append(logfile)
            cmd = [*command, "--protocol", protocol] + (["--allow-cpu"] if allow_cpu else [])
            # Kaggle kernels can start in /kaggle/working, which may contain an
            # unrelated `experiments` package. Python searches the working
            # directory before PYTHONPATH, so launch from this checkout.
            children.append(subprocess.Popen(
                cmd, cwd=str(root), env=env, stdout=logfile, stderr=subprocess.STDOUT,
            ))
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
    parser.add_argument("--dataset", default=None)
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
        launch({k: getattr(args, k) for k in ("cache", "checkpoint_root", "dataset", "outdir",
                                                "batch_size", "max_val_samples", "allow_cpu")})


if __name__ == "__main__":
    main()

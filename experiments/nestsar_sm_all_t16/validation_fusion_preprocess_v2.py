"""Frozen, one-pass T16 token-fusion audit for PREPROCESS_V2 checkpoints.

Uses the exact model and preprocessing source that produced the verified
76.321216% XSUB / 78.062108% XSET run. Does not train or create a raw cache.
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

PROTOCOLS = ("xsub", "xset")
VIEWS = ("original", "motion_fused", "all_fused")
MODES = VIEWS
MODEL_NAME = "NestSAR-SM-ALL-T16-v1"
EXPECTED_PARAMS = 1_826_556
DEFAULT_CHECKPOINTS = "/kaggle/working/NestSAR_SM_ALL_T16_PREPROCESS_V2"
DEFAULT_OUT = "/kaggle/working/NestSAR_PREPROCESS_V2_T16_FUSION"


class FixedBoundaryShift:
    def __init__(self, direction: int):
        if direction not in (-1, 1):
            raise ValueError("Shift must be -1 or +1")
        self.direction = direction

    def integers(self, low: int, high: int) -> int:
        if not low <= self.direction < high:
            raise ValueError("Unexpected boundary jitter range")
        return self.direction


def make_views(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Average corresponding T16 tokens; run the frozen model once per candidate.

    15 channels per joint: pose xyz, full displacement xyz, two phase xyz
    displacements, and absolute path xyz. Keep the canonical pose untouched in
    the motion_fused candidate, so original frame selection is preserved.
    """
    original = pp.features(raw)
    early = pp.features(raw, rng=FixedBoundaryShift(-1), shift=1)
    late = pp.features(raw, rng=FixedBoundaryShift(+1), shift=1)
    pose, left, right = (x.reshape(16, 2, 25, 15) for x in (original, early, late))
    motion_fused = pose.copy()
    motion_fused[..., 3:] = ((pose[..., 3:].astype(np.float64) + left[..., 3:] +
                             right[..., 3:]) / 3).astype(np.float32)
    all_fused = ((pose.astype(np.float64) + left + right) / 3).astype(np.float32)
    views = {"original": original,
             "motion_fused": motion_fused.reshape(16, 750),
             "all_fused": all_fused.reshape(16, 750)}
    if any(x.shape != (16, 750) or not np.isfinite(x).all() for x in views.values()):
        raise ValueError("Invalid T16 feature shape or nonfinite values")
    return views


def _weight_path(root: Path, protocol: str) -> Path | None:
    p = root / protocol / "best.msgpack"
    if p.is_file():
        return p
    sidecar = root / protocol / "best.json"
    if sidecar.is_file():
        try:
            epoch = int(json.loads(sidecar.read_text())["epoch"])
        except (KeyError, TypeError, ValueError):
            return None
        p = root / protocol / f"best_epoch_{epoch:04d}.msgpack"
        if p.is_file():
            return p
    return None


def checkpoint_location(preferred: Path) -> Path:
    """Prefer an explicit pair; otherwise find exactly one mounted legacy pair."""
    roots = [preferred]
    seen = []
    for base in (Path("/kaggle/working"), Path("/kaggle/input")):
        if base.exists():
            for name in ("best.msgpack", "best_epoch_*.msgpack"):
                for p in base.rglob(name):
                    seen.append(p)
                    if p.parent.name == "xsub":
                        roots.append(p.parent.parent)
    pairs = [root for root in dict.fromkeys(roots)
             if all(_weight_path(root, p) is not None for p in PROTOCOLS)]
    if preferred in pairs:
        return preferred
    # Never silently substitute another model for the user's specified model.
    listing = "\n".join(f"  {p}" for p in seen[:30]) or "  none"
    pair_status = ", ".join(
        f"{p}={'found' if _weight_path(preferred, p) else 'missing'}" for p in PROTOCOLS)
    raise FileNotFoundError(
        f"Both XSUB and XSET checkpoints are required at {preferred}. "
        "Attach saved Kaggle output if these files came from an earlier session."
        f"\nRequested pair: {pair_status}\nVisible weights:\n{listing}"
        + (f"\nOther paired roots (not selected): {pairs}" if pairs else ""))


def evaluate(protocol: str, checkpoint_root: Path, dataset_path: Path,
             output: Path, batch_size: int, allow_cpu: bool) -> dict:
    # CUDA_VISIBLE_DEVICES was set before importing any JAX model modules.
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from .model import NestSARSMAllT16
    from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
    from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju

    if not allow_cpu and (jax.default_backend() != "gpu" or jax.local_device_count() != 1):
        raise RuntimeError(f"Expected one isolated GPU for {protocol}: {jax.devices()}")
    checkpoint = _weight_path(checkpoint_root, protocol)
    if checkpoint is None:
        raise FileNotFoundError(f"No saved best weights for {protocol} in {checkpoint_root}")
    payload = serialization.msgpack_restore(checkpoint.read_bytes())
    sidecar = checkpoint.parent / "best.json"
    meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    if payload.get("model") != MODEL_NAME or payload.get("protocol") != protocol:
        raise ValueError(f"Wrong model or protocol in {checkpoint}")
    # This is the archived train_gpu_corrected checkpoint format. The newer P2
    # pipeline stores an explicit preprocessing_version; never interpret its
    # parameters with this older model and person-selection rule.
    if payload.get("preprocessing_version") not in (None, pp.VERSION):
        raise ValueError("Checkpoint belongs to a different preprocessing pipeline")
    if payload.get("pipeline_version") is not None:
        raise ValueError("Expected archived PREPROCESS_V2 weights, got a newer pipeline")
    if meta.get("model", MODEL_NAME) != MODEL_NAME or meta.get("params", EXPECTED_PARAMS) != EXPECTED_PARAMS:
        raise ValueError("Sidecar model or parameter count disagrees with archived model")
    params = payload["ema_params"]
    if sum(int(v.size) for v in jax.tree.leaves(params)) != EXPECTED_PARAMS:
        raise ValueError("Frozen checkpoint has unexpected parameter count")
    config = payload["config"]
    model = NestSARSMAllT16(**{key: config[key] for key in (
        "spatial_dim", "model_dim", "dropout", "controller_dim", "fast_rank",
        "head_rank", "sm_residual_scale", "head_residual_scale")})
    params = jax.device_put(params)
    infer = jax.jit(lambda p, x: model.apply({"params": p}, x, training=False)["logits"])

    annotations, split = base.load_ntu(dataset_path)
    by_id, _, ids = ju.resolve_protocol_ids(annotations, split, protocol)
    if len(ids) != (50919 if protocol == "xsub" else 59477):
        raise ValueError(f"Expected complete official {protocol} split, got {len(ids)}")
    target = output / protocol
    target.mkdir(parents=True, exist_ok=True)
    progress = Reporter(target / "status.json")
    progress(phase="Frozen T16 inference", current=0, total=len(ids))
    support = np.zeros(120, np.int64)
    correct = {mode: np.zeros(120, np.int64) for mode in MODES}
    flips = {mode: {"fixed": 0, "broken": 0} for mode in MODES[1:]}
    predictions = []
    for offset in range(0, len(ids), batch_size):
        batch_ids = ids[offset:offset + batch_size]
        n = len(batch_ids)
        x = {view: np.zeros((batch_size, 16, 750), np.float32) for view in VIEWS}
        y = np.empty(n, np.int32)
        for j, sid in enumerate(batch_ids):
            a = by_id[sid]
            raw = pp.ordered_raw(base.annotation_keypoints(a), layout="MTVC")
            views = make_views(raw)
            for view in VIEWS:
                x[view][j] = views[view]
            y[j] = base.annotation_label(a)
        if np.any((y < 0) | (y >= 120)):
            raise ValueError("Labels must be in 0..119")
        np.add.at(support, y, 1)
        logits = {view: np.asarray(jax.block_until_ready(infer(params, jnp.asarray(x[view]))))[:n]
                  for view in VIEWS}
        if any(not np.isfinite(v).all() for v in logits.values()):
            raise FloatingPointError("Nonfinite inference logits")
        pred = {mode: values.argmax(axis=-1) for mode, values in logits.items()}
        for mode in MODES:
            np.add.at(correct[mode], y, (pred[mode] == y).astype(np.int64))
            if mode != "original":
                flips[mode]["fixed"] += int(np.sum((pred["original"] != y) & (pred[mode] == y)))
                flips[mode]["broken"] += int(np.sum((pred["original"] == y) & (pred[mode] != y)))
        predictions.extend((sid, int(y[j]), *(int(pred[mode][j]) for mode in MODES))
                           for j, sid in enumerate(batch_ids))
        progress(phase="Frozen T16 inference", current=offset + n, total=len(ids),
                 scores={mode: float(correct[mode].sum() / (offset + n)) for mode in MODES})

    baseline = float(correct["original"].sum() / len(ids))
    saved_score = meta.get("val_accuracy", payload.get("val_accuracy"))
    if saved_score is None:
        raise ValueError("Checkpoint has no saved validation score to verify original T16 input")
    if abs(baseline - float(saved_score)) > 0.001:
        raise RuntimeError(
            f"Original T16 {baseline*100:.6f}% differs from archived "
            f"{float(saved_score)*100:.6f}%. Stop: preprocessing/model or dataset does not match.")
    with (target / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "label", *(f"pred_{mode}" for mode in MODES)])
        writer.writerows(predictions)
    with (target / "class_recall.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "support", *(f"recall_{mode}" for mode in MODES)])
        for label in range(120):
            writer.writerow([label, int(support[label]), *(
                float(correct[mode][label] / support[label]) if support[label] else None
                for mode in MODES)])
    result = {"protocol": protocol, "model": MODEL_NAME, "source_commit":
              "490d335fc0fe6ec30a256753185c055c1ff15faf", "checkpoint": str(checkpoint),
              "params": EXPECTED_PARAMS, "frozen": True, "samples": len(ids),
              "model_passes_per_candidate": 1,
              "views_preprocessed_per_clip": 3,
              "saved_accuracy": float(saved_score),
              "scores": {mode: {"accuracy": float(correct[mode].sum() / len(ids)),
                                "delta_pp": 100 * (correct[mode].sum() / len(ids) - baseline)}
                         for mode in MODES}, "flips": flips}
    atomic_json(target / "result.json", result)
    progress(phase="Done", current=len(ids), total=len(ids), done=True,
             scores={mode: result["scores"][mode]["accuracy"] for mode in MODES})
    return result


def _refresh(output: Path, bars: list) -> None:
    for protocol, bar in zip(PROTOCOLS, bars):
        marker = output / protocol / "status.json"
        if not marker.is_file():
            continue
        try:
            state = json.loads(marker.read_text())
        except json.JSONDecodeError:
            continue
        bar.n = min(bar.total, int(state.get("current", 0)))
        scores = state.get("scores", {})
        bar.set_postfix_str(" | ".join(f"{name}:{scores[name]*100:.2f}%"
                                     for name in ("original", "motion_fused") if name in scores), refresh=False)
        bar.refresh()


def launch(settings: dict | None = None) -> dict:
    from tqdm.auto import tqdm
    options = dict(settings or {})
    unknown = set(options) - {"checkpoint_root", "dataset", "outdir", "batch_size", "allow_cpu"}
    if unknown:
        raise ValueError(f"Unexpected options: {sorted(unknown)}")
    checkpoint_root = checkpoint_location(Path(options.get("checkpoint_root", DEFAULT_CHECKPOINTS)))
    from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
    source = base.find_dataset(options.get("dataset"))
    output = Path(options.get("outdir", DEFAULT_OUT))
    output.mkdir(parents=True, exist_ok=True)
    batch_size = int(options.get("batch_size", 128))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    allow_cpu = bool(options.get("allow_cpu", False))
    root = Path(__file__).resolve().parents[2]
    print(f"Frozen checkpoint pair: {checkpoint_root}\nDataset: {source}", flush=True)
    print("Sequential pickle-reading GPU workers; 3 preprocessing views per clip, "
          "1 model pass per candidate; no cache or training.", flush=True)
    bars = [tqdm(total=n, desc=f"{p.upper()} frozen T16", position=i, leave=True)
            for i, (p, n) in enumerate(zip(PROTOCOLS, (50919, 59477)))]
    results = {}
    try:
        for gpu, protocol in enumerate(PROTOCOLS):
            (output / protocol).mkdir(exist_ok=True)
            with (output / protocol / "worker.log").open("w") as log:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(root),
                           JAX_PLATFORMS="cpu" if allow_cpu else "cuda",
                           XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1")
                command = [sys.executable, "-u", "-m",
                           "experiments.nestsar_sm_all_t16.validation_fusion_preprocess_v2",
                           "--worker", "--protocol", protocol, "--checkpoint-root", str(checkpoint_root),
                           "--dataset", str(source), "--outdir", str(output), "--batch-size", str(batch_size)]
                if allow_cpu:
                    command.append("--allow-cpu")
                child = subprocess.Popen(command, cwd=str(root), env=env, stdout=log,
                                         stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        _refresh(output, bars)
                        time.sleep(0.6)
                except BaseException:
                    child.terminate()
                    child.wait()
                    raise
            _refresh(output, bars)
            if child.returncode:
                tail = "\n".join((output / protocol / "worker.log").read_text(errors="replace").splitlines()[-22:])
                raise RuntimeError(f"{protocol.upper()} frozen audit failed (exit {child.returncode}):\n{tail}")
            results[protocol] = json.loads((output / protocol / "result.json").read_text())
            print(f"{protocol.upper()} " + " | ".join(
                f"{mode}: {100*results[protocol]['scores'][mode]['accuracy']:.4f}% "
                f"({results[protocol]['scores'][mode]['delta_pp']:+.3f} pp)" for mode in MODES), flush=True)
        atomic_json(output / "summary.json", results)
        return results
    finally:
        for bar in bars:
            bar.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--protocol", choices=PROTOCOLS)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--dataset")
    parser.add_argument("--outdir", default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-cpu", action="store_true")
    a = parser.parse_args()
    if a.worker:
        if a.protocol is None:
            parser.error("Worker needs --protocol")
        if a.dataset is None:
            parser.error("Worker needs --dataset")
        evaluate(a.protocol, Path(a.checkpoint_root), Path(a.dataset),
                 Path(a.outdir), a.batch_size, a.allow_cpu)
    else:
        launch({"checkpoint_root": a.checkpoint_root, "dataset": a.dataset,
                "outdir": a.outdir, "batch_size": a.batch_size, "allow_cpu": a.allow_cpu})


if __name__ == "__main__":
    main()

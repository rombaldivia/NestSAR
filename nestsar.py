#!/usr/bin/env python3
"""NestSAR single-file experiment runner.

Current milestone: stable CLI, dataset discovery, automatic GPU discovery and
allocation, and reproducibility metadata. The validated JAX trainer will be
ported into this same file without changing the public command line.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

VERSION: Final = "0.2.0-dev"
MODELS: Final = ("h0", "h2", "h3", "nestsar_4l")
PROTOCOLS: Final = ("xsub", "xset", "both")
GPU_STRATEGIES: Final = ("auto", "protocol_parallel", "data_parallel", "sequential")

BASE: Final[dict[str, Any]] = {
    "model": "nestsar_4l", "protocol": "both", "seed": 128, "frames": 16,
    "batch_size": 128, "eval_batch_size": 256, "epochs": 150, "patience": 40,
    "learning_rate": 2e-4, "weight_decay": 0.03, "warmup_fraction": 0.10,
    "label_smoothing": 0.05, "grad_clip": 1.0, "dropout": 0.15,
    "model_dim": 128, "memory_dim": 64, "chunk_size": 4, "clip_size": 8,
    "blocks_per_level": 2, "controller_rank": 8,
    "predictive_loss_weight": 0.10, "max_train_samples": 0,
    "max_val_samples": 0, "gpu_map": "auto", "gpu_strategy": "auto",
    "max_gpus": 0, "resume": "none",
}
PRESETS: Final = {
    "official": dict(BASE),
    "legacy_4l_seed128": dict(BASE),
    "smoke": {**BASE, "protocol": "xsub", "batch_size": 8,
              "eval_batch_size": 16, "epochs": 1, "patience": 1,
              "max_train_samples": 256, "max_val_samples": 256},
}


@dataclasses.dataclass(frozen=True)
class GpuInfo:
    logical_index: int
    physical_index: str
    name: str
    memory_mib: int | None
    source: str


@dataclasses.dataclass(frozen=True)
class RunConfig:
    model: str
    protocol: str
    dataset: str
    output_dir: str
    preset: str | None
    seed: int
    frames: int
    batch_size: int
    eval_batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    warmup_fraction: float
    label_smoothing: float
    grad_clip: float
    dropout: float
    model_dim: int
    memory_dim: int
    chunk_size: int
    clip_size: int
    blocks_per_level: int
    controller_rank: int
    predictive_loss_weight: float
    max_train_samples: int
    max_val_samples: int
    gpu_map: str
    gpu_strategy: str
    max_gpus: int
    resume: str
    dry_run: bool


def positive(value: str) -> int:
    value_i = int(value)
    if value_i <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value_i


def non_negative(value: str) -> int:
    value_i = int(value)
    if value_i < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return value_i


def probability(value: str) -> float:
    value_f = float(value)
    if not 0 <= value_f <= 1:
        raise argparse.ArgumentTypeError("must be within [0, 1]")
    return value_f


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(list(command), capture_output=True, text=True, check=False)
    except OSError:
        return None


def visible_numeric_devices() -> list[int] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return None
    if raw.strip().lower() in {"-1", "none"}:
        return []
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return [int(token) for token in tokens] if all(token.isdigit() for token in tokens) else None


def detect_gpus(max_gpus: int = 0) -> list[GpuInfo]:
    """Detect GPUs before importing JAX, preventing premature CUDA initialization."""
    detected: list[GpuInfo] = []
    result = run_command(("nvidia-smi", "--query-gpu=index,name,memory.total",
                          "--format=csv,noheader,nounits"))
    if result and result.returncode == 0:
        rows: list[tuple[int, str, int | None]] = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) != 3 or not parts[0].isdigit():
                continue
            try:
                memory = int(float(parts[2]))
            except ValueError:
                memory = None
            rows.append((int(parts[0]), parts[1], memory))
        visible = visible_numeric_devices()
        if visible is not None:
            by_id = {row[0]: row for row in rows}
            rows = [by_id[index] for index in visible if index in by_id]
        detected = [GpuInfo(i, str(physical), name, memory, "nvidia-smi")
                    for i, (physical, name, memory) in enumerate(rows)]

    if not detected:
        try:
            import jax  # type: ignore
            devices = [device for device in jax.devices() if device.platform == "gpu"]
            detected = [GpuInfo(i, str(getattr(device, "id", i)),
                                str(getattr(device, "device_kind", "GPU")), None, "jax")
                        for i, device in enumerate(devices)]
        except Exception:
            detected = []
    return detected[:max_gpus] if max_gpus > 0 else detected


def parse_gpu_map(value: str, gpu_count: int) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for item in value.split(","):
        if ":" not in item:
            raise ValueError("GPU map must look like xsub:0+1,xset:2+3")
        protocol, raw = (part.strip() for part in item.split(":", 1))
        if protocol not in ("xsub", "xset"):
            raise ValueError(f"unsupported protocol in GPU map: {protocol}")
        indices = [int(token.strip()) for token in raw.split("+")]
        if any(index < 0 or index >= gpu_count for index in indices):
            raise ValueError(f"GPU map references unavailable device; detected {gpu_count}")
        mapping[protocol] = indices
    return mapping


def automatic_gpu_plan(protocol: str, gpu_count: int, strategy: str) -> dict[str, Any]:
    protocols = ["xsub", "xset"] if protocol == "both" else [protocol]
    if gpu_count == 0:
        return {"backend": "cpu", "strategy": "sequential",
                "parallel_protocols": False,
                "assignments": {name: [] for name in protocols}, "used_gpu_count": 0}
    if strategy == "auto":
        strategy = "protocol_parallel" if protocol == "both" and gpu_count >= 2 else "data_parallel"
    if protocol != "both":
        devices = list(range(gpu_count)) if strategy == "data_parallel" else [0]
        return {"backend": "gpu", "strategy": strategy, "parallel_protocols": False,
                "assignments": {protocol: devices}, "used_gpu_count": len(devices)}
    if strategy == "protocol_parallel":
        split = (gpu_count + 1) // 2
        assignments = {"xsub": list(range(split)), "xset": list(range(split, gpu_count))}
        return {"backend": "gpu", "strategy": strategy, "parallel_protocols": True,
                "assignments": assignments, "used_gpu_count": gpu_count}
    if strategy == "data_parallel":
        devices = list(range(gpu_count))
        return {"backend": "gpu", "strategy": strategy, "parallel_protocols": False,
                "assignments": {"xsub": devices, "xset": devices}, "used_gpu_count": gpu_count}
    return {"backend": "gpu", "strategy": "sequential", "parallel_protocols": False,
            "assignments": {"xsub": [0], "xset": [0]}, "used_gpu_count": 1}


def resolve_gpu_plan(config: RunConfig, gpus: Sequence[GpuInfo]) -> dict[str, Any]:
    if config.gpu_map == "auto":
        return automatic_gpu_plan(config.protocol, len(gpus), config.gpu_strategy)
    assignments = parse_gpu_map(config.gpu_map, len(gpus))
    protocols = ["xsub", "xset"] if config.protocol == "both" else [config.protocol]
    missing = [name for name in protocols if name not in assignments]
    if missing:
        raise ValueError("missing GPU assignment for: " + ", ".join(missing))
    used = sorted({i for name in protocols for i in assignments[name]})
    overlap = config.protocol == "both" and bool(set(assignments["xsub"]) & set(assignments["xset"]))
    return {"backend": "gpu", "strategy": "explicit",
            "parallel_protocols": config.protocol == "both" and not overlap,
            "assignments": {name: assignments[name] for name in protocols},
            "used_gpu_count": len(used)}


def discover_dataset(requested: str) -> Path:
    if requested != "auto":
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"dataset not found: {path}")
        return path
    candidates: list[Path] = []
    for root in (Path("/kaggle/input"), Path.cwd(), Path.home() / "data", Path("/data")):
        if root.exists():
            for name in ("ntu120_3danno.pkl", "ntu120_3danno_clean.pkl"):
                try:
                    candidates.extend(root.rglob(name))
                except PermissionError:
                    pass
    candidates = sorted({path.resolve() for path in candidates if path.is_file()})
    if not candidates:
        raise FileNotFoundError("could not find NTU120 pickle; pass --dataset /absolute/path/file.pkl")
    return candidates[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def git_commit() -> str | None:
    result = run_command(("git", "rev-parse", "HEAD"))
    return result.stdout.strip() if result and result.returncode == 0 else None


def hash_config(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=tuple(PRESETS))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--protocol", choices=PROTOCOLS)
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--seed", type=int)
    for name in ("frames", "batch-size", "eval-batch-size", "epochs", "patience",
                 "model-dim", "memory-dim", "chunk-size", "clip-size",
                 "blocks-per-level", "controller-rank"):
        parser.add_argument(f"--{name}", type=positive)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-fraction", type=probability)
    parser.add_argument("--label-smoothing", type=probability)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--dropout", type=probability)
    parser.add_argument("--predictive-loss-weight", type=float)
    parser.add_argument("--max-train-samples", type=non_negative)
    parser.add_argument("--max-val-samples", type=non_negative)
    parser.add_argument("--gpu-map", help="auto or xsub:0+1,xset:2+3")
    parser.add_argument("--gpu-strategy", choices=GPU_STRATEGIES)
    parser.add_argument("--max-gpus", type=non_negative, help="0 uses every visible GPU")
    parser.add_argument("--resume", choices=("none", "auto"))
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NestSAR runner for NTU RGB+D 120")
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-gpus", action="store_true")
    add_arguments(parser)
    return parser


def resolve(namespace: argparse.Namespace) -> RunConfig:
    preset = namespace.preset
    values = dict(PRESETS[preset or "official"])
    for name in values:
        override = getattr(namespace, name, None)
        if override is not None:
            values[name] = override
    return RunConfig(dataset=namespace.dataset, output_dir=namespace.output_dir,
                     preset=preset, dry_run=namespace.dry_run, **values)


def validate(config: RunConfig) -> None:
    if config.frames % config.chunk_size or config.frames % config.clip_size:
        raise ValueError("frames must be divisible by chunk-size and clip-size")
    if config.learning_rate <= 0 or config.weight_decay < 0 or config.grad_clip <= 0:
        raise ValueError("invalid optimizer hyperparameters")


def print_gpus(gpus: Sequence[GpuInfo]) -> None:
    print(f"Detected visible GPUs: {len(gpus)}")
    for gpu in gpus:
        memory = "unknown" if gpu.memory_mib is None else f"{gpu.memory_mib} MiB"
        print(f"  logical={gpu.logical_index} physical={gpu.physical_index} "
              f"name={gpu.name} memory={memory} source={gpu.source}")
    if not gpus:
        print("  No visible NVIDIA GPU was detected.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        print("\n".join(MODELS))
        return 0
    if args.list_gpus:
        print_gpus(detect_gpus(args.max_gpus or 0))
        return 0

    config = resolve(args)
    validate(config)
    gpus = detect_gpus(config.max_gpus)
    plan = resolve_gpu_plan(config, gpus)
    dataset = discover_dataset(config.dataset)
    resolved = dataclasses.asdict(config) | {"dataset": str(dataset), "gpu_plan": plan}
    resolved["config_hash"] = hash_config(resolved)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{stamp}_{safe_name(config.model)}_{config.protocol}_seed{config.seed}_{resolved['config_hash']}"
    run_dir = Path(config.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "executable": sys.executable, "git_commit": git_commit(),
        "packages": {name: package_version(name) for name in ("numpy", "jax", "jaxlib", "flax", "optax")},
        "dataset": {"path": str(dataset), "size_bytes": dataset.stat().st_size,
                    "sha256": sha256_file(dataset)},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "detected_gpus": [dataclasses.asdict(gpu) for gpu in gpus], "gpu_plan": plan,
    }
    write_json(run_dir / "resolved_config.json", resolved)
    write_json(run_dir / "environment.json", environment)
    (run_dir / "command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")

    print_gpus(gpus)
    print("=" * 80)
    print(f"NestSAR {VERSION} | model={config.model} | protocol={config.protocol}")
    print(f"Dataset: {dataset}")
    print(f"Seed={config.seed} frames={config.frames} batch={config.batch_size} eval_batch={config.eval_batch_size}")
    print(f"GPU strategy: {plan['strategy']} | assignments: {plan['assignments']} | used={plan['used_gpu_count']}")
    print(f"Run directory: {run_dir}")
    print("=" * 80)
    if config.dry_run:
        print("Dry run completed. No training was started.")
        return 0
    raise RuntimeError("CLI and GPU planning are ready; the validated training engine is the next milestone. Use --dry-run for now.")


if __name__ == "__main__":
    raise SystemExit(main())

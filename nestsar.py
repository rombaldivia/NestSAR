#!/usr/bin/env python3
"""Single-file command-line entry point for NestSAR experiments.

This bootstrap version standardizes configuration resolution, dataset discovery,
GPU mapping, run-directory creation, and reproducibility metadata. The model and
training engine will be added without changing this public CLI.
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

VERSION: Final[str] = "0.1.0-dev"
MODEL_CHOICES: Final[tuple[str, ...]] = (
    "h0",
    "h2",
    "h3",
    "nestsar_4l",
)
PROTOCOL_CHOICES: Final[tuple[str, ...]] = (
    "xsub",
    "xset",
    "both",
)

PRESETS: Final[dict[str, dict[str, Any]]] = {
    "official": {
        "model": "nestsar_4l",
        "protocol": "both",
        "seed": 128,
        "frames": 16,
        "batch_size": 128,
        "eval_batch_size": 256,
        "epochs": 150,
        "patience": 40,
        "learning_rate": 2.0e-4,
        "weight_decay": 0.03,
        "warmup_fraction": 0.10,
        "label_smoothing": 0.05,
        "grad_clip": 1.0,
        "dropout": 0.15,
        "model_dim": 128,
        "memory_dim": 64,
        "chunk_size": 4,
        "clip_size": 8,
        "blocks_per_level": 2,
        "controller_rank": 8,
        "predictive_loss_weight": 0.10,
        "max_train_samples": 0,
        "max_val_samples": 0,
        "gpu_map": "xsub:0,xset:1",
        "resume": "none",
    },
    "legacy_4l_seed128": {
        "model": "nestsar_4l",
        "protocol": "both",
        "seed": 128,
        "frames": 16,
        "batch_size": 128,
        "eval_batch_size": 256,
        "epochs": 150,
        "patience": 40,
        "learning_rate": 2.0e-4,
        "weight_decay": 0.03,
        "warmup_fraction": 0.10,
        "label_smoothing": 0.05,
        "grad_clip": 1.0,
        "dropout": 0.15,
        "model_dim": 128,
        "memory_dim": 64,
        "chunk_size": 4,
        "clip_size": 8,
        "blocks_per_level": 2,
        "controller_rank": 8,
        "predictive_loss_weight": 0.10,
        "max_train_samples": 0,
        "max_val_samples": 0,
        "gpu_map": "xsub:0,xset:1",
        "resume": "none",
    },
    "smoke": {
        "model": "nestsar_4l",
        "protocol": "xsub",
        "seed": 128,
        "frames": 16,
        "batch_size": 8,
        "eval_batch_size": 16,
        "epochs": 1,
        "patience": 1,
        "learning_rate": 2.0e-4,
        "weight_decay": 0.03,
        "warmup_fraction": 0.10,
        "label_smoothing": 0.05,
        "grad_clip": 1.0,
        "dropout": 0.15,
        "model_dim": 128,
        "memory_dim": 64,
        "chunk_size": 4,
        "clip_size": 8,
        "blocks_per_level": 2,
        "controller_rank": 8,
        "predictive_loss_weight": 0.10,
        "max_train_samples": 256,
        "max_val_samples": 256,
        "gpu_map": "xsub:0,xset:1",
        "resume": "none",
    },
}


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
    resume: str
    dry_run: bool


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("The value must be greater than zero.")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("The value must be zero or greater.")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("The value must be within [0, 1].")
    return parsed


def parse_gpu_map(value: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    if not value.strip():
        return mapping

    for item in value.split(","):
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                "GPU map entries must use protocol:index, for example "
                "xsub:0,xset:1."
            )
        protocol, raw_index = (part.strip() for part in item.split(":", 1))
        if protocol not in ("xsub", "xset"):
            raise argparse.ArgumentTypeError(
                f"Unsupported protocol in GPU map: {protocol!r}."
            )
        index = int(raw_index)
        if index < 0:
            raise argparse.ArgumentTypeError("GPU indices cannot be negative.")
        mapping[protocol] = index

    return mapping


def discover_dataset(requested: str) -> Path:
    if requested != "auto":
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    roots = (
        Path("/kaggle/input"),
        Path.cwd(),
        Path.home() / "data",
        Path("/data"),
    )
    names = (
        "ntu120_3danno.pkl",
        "ntu120_3danno_clean.pkl",
    )

    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            try:
                candidates.extend(root.rglob(name))
            except PermissionError:
                continue

    candidates = sorted({path.resolve() for path in candidates if path.is_file()})
    if not candidates:
        raise FileNotFoundError(
            "Could not find ntu120_3danno.pkl automatically. "
            "Pass --dataset /absolute/path/to/ntu120_3danno.pkl."
        )
    return candidates[0]


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_config_hash(config: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-") or "run"


def detect_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def optional_package_version(package_name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return None


def environment_snapshot(dataset: Path) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_commit": detect_git_commit(),
        "packages": {
            package: optional_package_version(package)
            for package in (
                "numpy",
                "jax",
                "jaxlib",
                "flax",
                "optax",
            )
        },
        "dataset": {
            "path": str(dataset),
            "size_bytes": dataset.stat().st_size,
            "sha256": file_sha256(dataset),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=tuple(PRESETS))
    parser.add_argument("--model", choices=MODEL_CHOICES)
    parser.add_argument("--protocol", choices=PROTOCOL_CHOICES)
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--frames", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--eval-batch-size", type=positive_int)
    parser.add_argument("--epochs", type=positive_int)
    parser.add_argument("--patience", type=positive_int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-fraction", type=probability)
    parser.add_argument("--label-smoothing", type=probability)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--dropout", type=probability)
    parser.add_argument("--model-dim", type=positive_int)
    parser.add_argument("--memory-dim", type=positive_int)
    parser.add_argument("--chunk-size", type=positive_int)
    parser.add_argument("--clip-size", type=positive_int)
    parser.add_argument("--blocks-per-level", type=positive_int)
    parser.add_argument("--controller-rank", type=positive_int)
    parser.add_argument("--predictive-loss-weight", type=float)
    parser.add_argument("--max-train-samples", type=non_negative_int)
    parser.add_argument("--max-val-samples", type=non_negative_int)
    parser.add_argument(
        "--gpu-map",
        default=None,
        help="Protocol-to-device mapping, for example xsub:0,xset:1.",
    )
    parser.add_argument(
        "--resume",
        choices=("none", "auto"),
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and persist the configuration without training.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NestSAR single-file experiment runner for NTU RGB+D 120."
    )
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print supported model identifiers and exit.",
    )
    add_common_arguments(parser)
    return parser


def resolve_arguments(namespace: argparse.Namespace) -> RunConfig:
    preset_name = namespace.preset
    base = dict(PRESETS[preset_name or "official"])

    override_names = (
        "model",
        "protocol",
        "seed",
        "frames",
        "batch_size",
        "eval_batch_size",
        "epochs",
        "patience",
        "learning_rate",
        "weight_decay",
        "warmup_fraction",
        "label_smoothing",
        "grad_clip",
        "dropout",
        "model_dim",
        "memory_dim",
        "chunk_size",
        "clip_size",
        "blocks_per_level",
        "controller_rank",
        "predictive_loss_weight",
        "max_train_samples",
        "max_val_samples",
        "gpu_map",
        "resume",
    )
    for name in override_names:
        value = getattr(namespace, name)
        if value is not None:
            base[name] = value

    return RunConfig(
        model=base["model"],
        protocol=base["protocol"],
        dataset=namespace.dataset,
        output_dir=namespace.output_dir,
        preset=preset_name,
        seed=base["seed"],
        frames=base["frames"],
        batch_size=base["batch_size"],
        eval_batch_size=base["eval_batch_size"],
        epochs=base["epochs"],
        patience=base["patience"],
        learning_rate=base["learning_rate"],
        weight_decay=base["weight_decay"],
        warmup_fraction=base["warmup_fraction"],
        label_smoothing=base["label_smoothing"],
        grad_clip=base["grad_clip"],
        dropout=base["dropout"],
        model_dim=base["model_dim"],
        memory_dim=base["memory_dim"],
        chunk_size=base["chunk_size"],
        clip_size=base["clip_size"],
        blocks_per_level=base["blocks_per_level"],
        controller_rank=base["controller_rank"],
        predictive_loss_weight=base["predictive_loss_weight"],
        max_train_samples=base["max_train_samples"],
        max_val_samples=base["max_val_samples"],
        gpu_map=base["gpu_map"],
        resume=base["resume"],
        dry_run=namespace.dry_run,
    )


def validate_config(config: RunConfig) -> None:
    if config.frames % config.chunk_size != 0:
        raise ValueError("--frames must be divisible by --chunk-size.")
    if config.frames % config.clip_size != 0:
        raise ValueError("--frames must be divisible by --clip-size.")
    if config.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be greater than zero.")
    if config.weight_decay < 0.0:
        raise ValueError("--weight-decay cannot be negative.")
    if config.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be greater than zero.")
    if config.predictive_loss_weight < 0.0:
        raise ValueError("--predictive-loss-weight cannot be negative.")

    gpu_map = parse_gpu_map(config.gpu_map)
    required_protocols = (
        ("xsub", "xset") if config.protocol == "both" else (config.protocol,)
    )
    missing = [protocol for protocol in required_protocols if protocol not in gpu_map]
    if missing:
        raise ValueError(
            "Missing GPU mapping for: "
            + ", ".join(missing)
            + ". Pass --gpu-map xsub:0,xset:1."
        )


def create_run_directory(
    config: RunConfig,
    dataset: Path,
) -> tuple[Path, dict[str, Any]]:
    resolved = dataclasses.asdict(config)
    resolved["dataset"] = str(dataset)

    config_hash = normalized_config_hash(resolved)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = (
        f"{timestamp}_{safe_component(config.model)}_"
        f"{safe_component(config.protocol)}_seed{config.seed}_{config_hash}"
    )
    run_dir = Path(config.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    resolved["config_hash"] = config_hash
    return run_dir, resolved


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def print_summary(
    config: RunConfig,
    dataset: Path,
    run_dir: Path,
    resolved: Mapping[str, Any],
) -> None:
    print("=" * 80)
    print("NestSAR single-file experiment runner")
    print("=" * 80)
    print(f"Version:          {VERSION}")
    print(f"Model:            {config.model}")
    print(f"Protocol:         {config.protocol}")
    print(f"Dataset:          {dataset}")
    print(f"Seed:             {config.seed}")
    print(f"Frames:           {config.frames}")
    print(f"Batch size:       {config.batch_size}")
    print(f"Eval batch size:  {config.eval_batch_size}")
    print(f"Epochs:           {config.epochs}")
    print(f"Patience:         {config.patience}")
    print(f"GPU map:          {config.gpu_map}")
    print(f"Resume:           {config.resume}")
    print(f"Config hash:      {resolved['config_hash']}")
    print(f"Run directory:    {run_dir}")
    print("=" * 80)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)

    if namespace.list_models:
        print("\n".join(MODEL_CHOICES))
        return 0

    config = resolve_arguments(namespace)
    validate_config(config)
    dataset = discover_dataset(config.dataset)
    run_dir, resolved = create_run_directory(config, dataset)

    write_json(run_dir / "resolved_config.json", resolved)
    write_json(run_dir / "environment.json", environment_snapshot(dataset))
    (run_dir / "command.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )

    print_summary(config, dataset, run_dir, resolved)

    if config.dry_run:
        print("Dry run completed. No training was started.")
        return 0

    raise RuntimeError(
        "The standardized CLI and reproducibility metadata are ready, but the "
        "training engine is not yet included in this bootstrap commit. Run with "
        "--dry-run for now. The next milestone will port the validated legacy "
        "H0/H2/H3/NestSAR-4L implementation into this same file."
    )


if __name__ == "__main__":
    raise SystemExit(main())

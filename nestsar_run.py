#!/usr/bin/env python3
"""Stable launcher for versioned NestSAR research experiments."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

EXPERIMENTS = {
    "attention_lite": "experiments.attention_lite_v1.trainer",
    "attention_lite_v1": "experiments.attention_lite_v1.trainer",
}


def list_experiments() -> tuple[str, ...]:
    return tuple(sorted(EXPERIMENTS))


def _set_optional(env: dict[str, str], name: str, value) -> None:
    if value is not None:
        env[name] = str(value)


def train(
    experiment: str = "attention_lite",
    *,
    protocol: str = "xsub",
    epochs: int = 40,
    patience: Optional[int] = None,
    dataset: str | Path = "auto",
    seed: int = 128,
    runs_root: str | Path = "/kaggle/working",
    canonical_source: Optional[str | Path] = None,
    paper_mode: bool = True,
    run_tag: Optional[str] = None,
    dropout: Optional[float] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    warmup_fraction: Optional[float] = None,
    label_smoothing: Optional[float] = None,
    grad_clip: Optional[float] = None,
    predictive_loss_weight: Optional[float] = None,
    initial_eta: Optional[float] = None,
    initial_alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
    grad_accum_steps: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Launch Attention-Lite in a clean subprocess.

    Architecture-defining values remain frozen in the canonical trainer. Training
    hyperparameters can be overridden directly from Kaggle. If any override is
    supplied, the run is automatically tagged CUSTOM_<hash> unless ``run_tag`` is
    provided, preventing accidental overwrite of a golden seed run.

    ``patience=None`` preserves the original E40 behavior (no early stopping).
    Set a positive integer to enable generated-run early stopping.
    """
    key = experiment.strip().lower()
    if key not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment {experiment!r}. Available: {', '.join(list_experiments())}"
        )
    protocol = protocol.strip().lower()
    if protocol not in {"xsub", "xset"}:
        raise ValueError("protocol must be 'xsub' or 'xset'")
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    if patience is not None and int(patience) < 0:
        raise ValueError("patience must be >= 0 or None")

    env = os.environ.copy()
    env["NESTSAR_PROTOCOL"] = protocol
    env["NESTSAR_EPOCHS"] = str(int(epochs))
    env["NESTSAR_SEED"] = str(int(seed))
    env["NESTSAR_DATASET"] = str(dataset)
    env["NESTSAR_RUNS_ROOT"] = str(runs_root)
    env["NESTSAR_PAPER_MODE"] = "1" if paper_mode else "0"
    if canonical_source is not None:
        env["NESTSAR_CANONICAL_SOURCE"] = str(canonical_source)
    if run_tag is not None:
        env["NESTSAR_RUN_TAG"] = str(run_tag)

    _set_optional(env, "NESTSAR_PATIENCE", patience)
    _set_optional(env, "NESTSAR_DROPOUT", dropout)
    _set_optional(env, "NESTSAR_LEARNING_RATE", learning_rate)
    _set_optional(env, "NESTSAR_WEIGHT_DECAY", weight_decay)
    _set_optional(env, "NESTSAR_WARMUP_FRACTION", warmup_fraction)
    _set_optional(env, "NESTSAR_LABEL_SMOOTHING", label_smoothing)
    _set_optional(env, "NESTSAR_GRAD_CLIP", grad_clip)
    _set_optional(env, "NESTSAR_PREDICTIVE_LOSS_WEIGHT", predictive_loss_weight)
    _set_optional(env, "NESTSAR_INITIAL_ETA", initial_eta)
    _set_optional(env, "NESTSAR_INITIAL_ALPHA", initial_alpha)
    _set_optional(env, "NESTSAR_BATCH_SIZE", batch_size)
    _set_optional(env, "NESTSAR_GRAD_ACCUM_STEPS", grad_accum_steps)
    _set_optional(env, "NESTSAR_EVAL_BATCH_SIZE", eval_batch_size)

    cmd = [sys.executable, "-u", "-m", EXPERIMENTS[key]]
    print("NestSAR launch:", " ".join(cmd), flush=True)
    print(
        f"experiment={key} protocol={protocol} epochs={epochs} patience={patience} "
        f"seed={seed} dataset={dataset} runs_root={runs_root} paper_mode={paper_mode}",
        flush=True,
    )
    return subprocess.run(cmd, env=env, check=check)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run a versioned NestSAR experiment")
    parser.add_argument("experiment", nargs="?", default="attention_lite")
    parser.add_argument("--protocol", choices=("xsub", "xset"), default="xsub")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--runs-root", default="/kaggle/working")
    parser.add_argument("--canonical-source", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-fraction", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--predictive-loss-weight", type=float, default=None)
    parser.add_argument("--initial-eta", type=float, default=None)
    parser.add_argument("--initial-alpha", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--non-paper", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_only")
    args = parser.parse_args()
    if args.list_only:
        print("\n".join(list_experiments()))
        return 0
    train(
        args.experiment,
        protocol=args.protocol,
        epochs=args.epochs,
        patience=args.patience,
        dataset=args.dataset,
        seed=args.seed,
        runs_root=args.runs_root,
        canonical_source=args.canonical_source,
        paper_mode=not args.non_paper,
        run_tag=args.run_tag,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        predictive_loss_weight=args.predictive_loss_weight,
        initial_eta=args.initial_eta,
        initial_alpha=args.initial_alpha,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        eval_batch_size=args.eval_batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

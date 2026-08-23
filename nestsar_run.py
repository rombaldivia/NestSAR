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


def _resolve_attention_source(protocol: str, canonical_source: Optional[str | Path]) -> Path:
    """Resolve before spawning so Kaggle gets a useful error instead of exit status 1."""
    from experiments.attention_lite_v1.source_resolver import resolve_canonical_source

    return resolve_canonical_source(protocol, canonical_source, verbose=True)


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
    """Launch one independent Attention-Lite protocol model in a clean subprocess.

    Architecture-defining values stay frozen in the validated all-in-one source.
    Training hyperparameters may be overridden from Kaggle.  The canonical source is
    resolved and validated *before* the child process is started, so missing Kaggle
    inputs produce a precise FileNotFoundError rather than a generic CalledProcessError.
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

    resolved_source: Optional[Path] = None
    if key in {"attention_lite", "attention_lite_v1"}:
        resolved_source = _resolve_attention_source(protocol, canonical_source)

    env = os.environ.copy()
    env["NESTSAR_PROTOCOL"] = protocol
    env["NESTSAR_EPOCHS"] = str(int(epochs))
    env["NESTSAR_SEED"] = str(int(seed))
    env["NESTSAR_DATASET"] = str(dataset)
    env["NESTSAR_RUNS_ROOT"] = str(runs_root)
    env["NESTSAR_PAPER_MODE"] = "1" if paper_mode else "0"
    if resolved_source is not None:
        env["NESTSAR_CANONICAL_SOURCE"] = str(resolved_source)
    elif canonical_source is not None:
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
    print("=" * 108, flush=True)
    print("NestSAR launch:", " ".join(cmd), flush=True)
    print(
        f"experiment={key} protocol={protocol} epochs={epochs} patience={patience} "
        f"seed={seed} dataset={dataset} runs_root={runs_root} paper_mode={paper_mode}",
        flush=True,
    )
    if resolved_source is not None:
        print(f"canonical_source={resolved_source}", flush=True)
    print("=" * 108, flush=True)

    # Do not use check=True directly: it replaces the useful child failure context with
    # a bare CalledProcessError in notebooks.  The child still streams stdout/stderr.
    completed = subprocess.run(cmd, env=env, check=False)
    if completed.returncode != 0 and check:
        source_text = str(resolved_source or canonical_source or "<unresolved>")
        raise RuntimeError(
            "NestSAR Attention-Lite child process failed.\n"
            f"protocol={protocol} seed={seed} returncode={completed.returncode}\n"
            f"canonical_source={source_text}\n"
            "Read the child traceback printed immediately above this message; it now "
            "contains the real failure instead of only CalledProcessError."
        )
    return completed


def train_both(
    experiment: str = "attention_lite",
    *,
    xsub_source: Optional[str | Path] = None,
    xset_source: Optional[str | Path] = None,
    **common_kwargs,
) -> dict[str, subprocess.CompletedProcess]:
    """Train two independent models sequentially: XSUB first, then XSET.

    Each call starts a fresh Python process and initializes a fresh model with the same
    requested seed.  XSET never resumes from or reuses the XSUB weights.
    """
    if "protocol" in common_kwargs:
        raise ValueError("train_both() controls protocol; remove protocol= from common kwargs")
    if "canonical_source" in common_kwargs:
        raise ValueError(
            "Use xsub_source= and xset_source= with train_both(), not canonical_source=."
        )

    # Resolve both first.  This prevents spending TPU time on XSUB and only discovering
    # hours later that the XSET source input is absent.
    xsub_resolved = _resolve_attention_source("xsub", xsub_source)
    xset_resolved = _resolve_attention_source("xset", xset_source)

    print("=" * 108, flush=True)
    print("NESTSAR ATTENTION-LITE — BOTH OFFICIAL NTU120 PROTOCOLS", flush=True)
    print("Two independent models; same seed/config; full TPU8 used sequentially.", flush=True)
    print(f"XSUB source: {xsub_resolved}", flush=True)
    print(f"XSET source: {xset_resolved}", flush=True)
    print("=" * 108, flush=True)

    results: dict[str, subprocess.CompletedProcess] = {}
    results["xsub"] = train(
        experiment,
        protocol="xsub",
        canonical_source=xsub_resolved,
        **common_kwargs,
    )
    print("\nXSUB COMPLETE — starting a fresh XSET model.\n", flush=True)
    results["xset"] = train(
        experiment,
        protocol="xset",
        canonical_source=xset_resolved,
        **common_kwargs,
    )
    return results


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--runs-root", default="/kaggle/working")
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


def _kwargs_from_args(args) -> dict:
    return dict(
        epochs=args.epochs,
        patience=args.patience,
        dataset=args.dataset,
        seed=args.seed,
        runs_root=args.runs_root,
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


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run a versioned NestSAR experiment")
    parser.add_argument("experiment", nargs="?", default="attention_lite")
    parser.add_argument("--protocol", choices=("xsub", "xset", "both"), default="xsub")
    parser.add_argument("--canonical-source", default=None)
    parser.add_argument("--xsub-source", default=None)
    parser.add_argument("--xset-source", default=None)
    parser.add_argument("--list", action="store_true", dest="list_only")
    _add_training_args(parser)
    args = parser.parse_args()

    if args.list_only:
        print("\n".join(list_experiments()))
        return 0

    common = _kwargs_from_args(args)
    if args.protocol == "both":
        train_both(
            args.experiment,
            xsub_source=args.xsub_source,
            xset_source=args.xset_source,
            **common,
        )
    else:
        source = args.canonical_source
        if source is None:
            source = args.xsub_source if args.protocol == "xsub" else args.xset_source
        train(
            args.experiment,
            protocol=args.protocol,
            canonical_source=source,
            **common,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

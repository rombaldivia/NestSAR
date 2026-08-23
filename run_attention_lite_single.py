#!/usr/bin/env python3
"""Run one validated Attention-Lite protocol on the full TPU8 slice.

This entry point is intentionally exclusive: one protocol owns the complete Kaggle
TPU v5e-8 runtime. XSUB and XSET must therefore be run in separate TPU sessions or
sequentially in one session. A short child-process TPU8 probe runs before source
materialization/training so an occupied or non-TPU runtime fails immediately instead
of silently falling back to CPU and spending time on model initialization.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.attention_lite_v1.canonical_integrated import ensure_canonical_sources
from nestsar_run import train

RUNNER_API_VERSION = "attention-lite-single-v2-full-tpu8-preflight"


def _force_tpu_environment() -> None:
    """Make this TPU-locked production runner refuse JAX CPU fallback."""
    # JAX_PLATFORM_NAME is a legacy selector and can override/autoinfluence backend
    # choice. The production experiment is TPU-only, so remove it and explicitly
    # request the TPU backend for every descendant process.
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ["JAX_PLATFORMS"] = "tpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _assert_full_tpu8_available() -> None:
    """Probe the same Python/JAX environment used by the trainer in a child process."""
    probe = r'''
import jax
backend = jax.default_backend()
devices = list(jax.devices())
print("TPU PREFLIGHT JAX:", jax.__version__, flush=True)
print("TPU PREFLIGHT BACKEND:", backend, flush=True)
print("TPU PREFLIGHT DEVICES:", len(devices), devices, flush=True)
if backend != "tpu":
    raise RuntimeError(f"Expected TPU backend, got {backend!r}")
if len(devices) != 8:
    raise RuntimeError(f"Expected 8 TPU devices, found {len(devices)}")
print("FULL TPU8 PREFLIGHT: PASS", flush=True)
'''
    completed = subprocess.run(
        [sys.executable, "-u", "-c", probe],
        env=os.environ.copy(),
        cwd=str(Path(__file__).resolve().parent),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "FULL TPU8 PREFLIGHT FAILED. Training was NOT started.\n"
            "If another NestSAR protocol is still training in this Kaggle session, "
            "wait for it to finish before launching this one. If a previous run was "
            "interrupted and no training should be active, restart the Kaggle session "
            "to release stale TPU child processes, then run again."
        )

    # The probe owns libtpu only for a moment. Give its process a short release window
    # before the real trainer initializes the same eight devices.
    time.sleep(2.0)


def main() -> int:
    p = argparse.ArgumentParser(description="Train one Attention-Lite protocol on full TPU8")
    p.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--dataset", default="auto")
    p.add_argument("--runs-root", default="/kaggle/working")
    p.add_argument("--run-tag", default=None)
    p.add_argument("--dropout", type=float, default=0.22)
    p.add_argument("--learning-rate", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-fraction", type=float, default=0.10)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--predictive-loss-weight", type=float, default=0.10)
    p.add_argument("--initial-eta", type=float, default=0.02)
    p.add_argument("--initial-alpha", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--non-paper", action="store_true")
    args = p.parse_args()

    print("=" * 108, flush=True)
    print(f"RUNNER API: {RUNNER_API_VERSION}", flush=True)
    print(f"Protocol: {args.protocol.upper()} | exclusive full TPU8 | seed={args.seed}", flush=True)
    print("XSUB/XSET cannot share this TPU8 concurrently in one Kaggle session.", flush=True)
    print("=" * 108, flush=True)

    _force_tpu_environment()
    _assert_full_tpu8_available()

    sources = ensure_canonical_sources(verbose=True)
    source = Path(sources[args.protocol]).resolve()

    patience = None if args.patience == 0 else args.patience
    completed = train(
        "attention_lite",
        protocol=args.protocol,
        epochs=args.epochs,
        patience=patience,
        dataset=args.dataset,
        seed=args.seed,
        runs_root=args.runs_root,
        canonical_source=source,
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
        check=True,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

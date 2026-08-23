#!/usr/bin/env python3
"""Run one validated Attention-Lite protocol on the full TPU8 slice.

Use this entry point when running XSUB and XSET in two separate Kaggle TPU
sessions/notebooks at the same time.  Each session gets its own complete TPU8
runtime, preserving the validated 8-device topology instead of trying to split one
Kaggle TPU slice into unsupported independent TPU4 jobs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.attention_lite_v1.canonical_integrated import ensure_canonical_sources
from nestsar_run import train

RUNNER_API_VERSION = "attention-lite-single-v1-full-tpu8"


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
    print(f"Protocol: {args.protocol.upper()} | full TPU8 | seed={args.seed}", flush=True)
    print("=" * 108, flush=True)

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

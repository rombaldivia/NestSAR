#!/usr/bin/env python3
"""Fresh-process Kaggle entry point for Attention-Lite XSUB + XSET runs.

The exact validated XSUB source is embedded in this repository as a compressed,
SHA256-guarded payload. XSET is deterministically derived from the validated XSUB
source with protocol-only replacements and independently SHA256-guarded.

This entry point materializes and validates both canonical sources before training,
so Kaggle does not need separate Attention-Lite source inputs.
"""
from __future__ import annotations

import argparse

from experiments.attention_lite_v1.canonical_payload import ensure_canonical_sources
from nestsar_run import train_both

RUNNER_API_VERSION = "attention-lite-both-v2-embedded-canonical"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train independent XSUB and XSET Attention-Lite models")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--runs-root", default="/kaggle/working")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--xsub-source", default=None)
    parser.add_argument("--xset-source", default=None)

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
    args = parser.parse_args()

    print(f"RUNNER API: {RUNNER_API_VERSION}", flush=True)

    # Default path: build the exact canonical sources from the committed payload.
    # Explicit --xsub-source/--xset-source remain available for audit/debugging.
    embedded = ensure_canonical_sources(verbose=True)
    xsub_source = args.xsub_source or embedded["xsub"]
    xset_source = args.xset_source or embedded["xset"]

    patience = None if args.patience == 0 else args.patience
    train_both(
        "attention_lite",
        xsub_source=xsub_source,
        xset_source=xset_source,
        seed=args.seed,
        epochs=args.epochs,
        patience=patience,
        dataset=args.dataset,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

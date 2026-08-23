#!/usr/bin/env python3
"""Fresh-process Kaggle entry point for Attention-Lite XSUB + XSET.

Canonical sources are materialized from the repository-bundled artifact by
`experiments.attention_lite_v1.source_resolver`, so Kaggle needs no separate
Attention-Lite source input.  XSUB and XSET are then trained sequentially as two
independent fresh models with the same requested seed/configuration.
"""
from __future__ import annotations

import argparse

from experiments.attention_lite_v1.source_resolver import resolve_both_sources
from nestsar_run import train_both

RUNNER_API_VERSION = "attention-lite-both-v3-repo-canonical"


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

    # Resolve/materialize both sources before consuming TPU time. Explicit sources
    # remain available for audit/debugging, but are not required for the normal run.
    sources = resolve_both_sources(
        xsub=args.xsub_source,
        xset=args.xset_source,
        verbose=True,
    )

    patience = None if args.patience == 0 else args.patience
    train_both(
        "attention_lite",
        xsub_source=sources["xsub"],
        xset_source=sources["xset"],
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

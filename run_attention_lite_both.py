#!/usr/bin/env python3
"""Fresh-process Kaggle entry point for Attention-Lite XSUB + XSET.

The default path reconstructs the exact validated XSUB/XSET all-in-one trainers
entirely from GitHub-committed payload data and verifies their SHA256 hashes before
TPU training begins. No separate Kaggle Attention-Lite source input is required.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.attention_lite_v1.canonical_integrated import ensure_canonical_sources
from nestsar_run import train_both

RUNNER_API_VERSION = "attention-lite-both-v6-exact-integrated"


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
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Reconstruct and SHA-verify both GitHub canonical sources, then exit without training.",
    )

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

    if bool(args.xsub_source) != bool(args.xset_source):
        raise ValueError("Pass both --xsub-source and --xset-source, or neither")

    if args.xsub_source and args.xset_source:
        sources = {
            "xsub": Path(args.xsub_source).expanduser().resolve(),
            "xset": Path(args.xset_source).expanduser().resolve(),
        }
        for protocol, path in sources.items():
            if not path.is_file():
                raise FileNotFoundError(f"Explicit {protocol.upper()} source does not exist: {path}")
    else:
        # Default/official path: exact GitHub-integrated canonical sources.
        # The builder reconstructs exact XSUB from committed LZMA+base64 chunks,
        # derives exact XSET with guarded protocol-only edits, verifies BOTH complete
        # source SHA256 hashes, and syntax-compiles both trainers before returning.
        sources = ensure_canonical_sources(verbose=True)

    print("=" * 108, flush=True)
    print("ATTENTION-LITE BOTH-PROTOCOL CANONICAL PREFLIGHT: PASS", flush=True)
    print(f"XSUB source: {sources['xsub']}", flush=True)
    print(f"XSET source: {sources['xset']}", flush=True)
    print("=" * 108, flush=True)

    if args.preflight_only:
        print("PREFLIGHT-ONLY REQUESTED — no TPU training started.", flush=True)
        return 0

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

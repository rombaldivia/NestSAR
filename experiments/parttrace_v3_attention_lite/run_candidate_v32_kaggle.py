#!/usr/bin/env python3
"""Notebook-native runner for the current PartTrace v3.2 candidate.

Run with `%run`. Candidate defaults are injected first; any arguments explicitly
supplied by the user afterwards override them because argparse keeps the last
occurrence of an option.
"""
from __future__ import annotations

import sys

from experiments.parttrace_v3_attention_lite import run_both_t4_v32_kaggle as base

CANDIDATE_DEFAULTS = [
    "--gpu-xsub", "0",
    "--gpu-xset", "1",
    "--dataset", "auto",
    "--frames", "16",
    "--epochs", "60",
    "--patience", "10",
    "--seed", "128",
    "--batch-size", "32",
    "--eval-batch-size", "64",
    "--part-dim", "64",
    "--part-heads", "4",
    "--global-dim", "192",
    "--dense-dim", "192",
    "--branch-dropout", "0.12",
    "--base-lr", "4e-4",
    "--branch-lr", "6e-4",
    "--controller-lr", "1.5e-4",
    "--base-min-lr", "1e-5",
    "--branch-min-lr", "2e-5",
    "--controller-min-lr", "5e-6",
    "--warmup-fraction", "0.08",
    "--base-weight-decay", "0.03",
    "--branch-weight-decay", "0.04",
    "--label-smoothing", "0.05",
    "--grad-clip", "1.0",
    "--ema-decay", "0.995",
    "--predictive-loss-weight", "0.10",
    "--parttrace-aux-weight", "0.10",
    "--controller-kl-weight", "0.02",
    "--freeze-branch-epochs", "2",
    "--branch-ramp-epochs", "4",
    "--max-train-samples", "0",
    "--max-val-samples", "0",
    "--progress-every", "5",
    "--refresh-seconds", "0.30",
    "--outdir", "/kaggle/working/NestSAR_PartTrace_v32",
    "--audit-first",
]


def main() -> int:
    user_args = sys.argv[1:]
    sys.argv = [sys.argv[0], *CANDIDATE_DEFAULTS, *user_args]
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())

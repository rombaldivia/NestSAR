#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Notebook-native dual-T4 launcher for TokenPreserve v3.3.

Reuses the proven v3.2 two-bar monitor but swaps in the v3.3 T16 trainer.
Run with `%run`, not `!python`, so Kaggle keeps the tqdm widgets in place.
"""
from __future__ import annotations

import argparse
import builtins
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent


def _extract_v33(argv: list[str]):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--readout-tokens", type=int, default=8)
    known, rest = p.parse_known_args(argv)
    return known, rest


def main() -> int:
    v33, remaining = _extract_v33(sys.argv[1:])
    if v33.frames != 16:
        raise ValueError(
            f"TokenPreserve v3.3 is intentionally T16; got --frames={v33.frames}. "
            "Use --frames 16."
        )
    if not 1 <= v33.readout_tokens <= 32:
        raise ValueError("--readout-tokens must be in [1, 32]")

    os.environ["NESTSAR_READOUT_TOKENS"] = str(v33.readout_tokens)

    from experiments.parttrace_v3_attention_lite import run_both_t4_v32_kaggle as base

    base.TRAINER = HERE / "train_v33.py"

    # v3.2 launcher owns all the stable dual-GPU/tqdm behavior. It receives an
    # explicit T16; readout K travels through the environment to each worker.
    sys.argv = [sys.argv[0], "--frames", "16", *remaining]

    original_print = builtins.print

    def print_v33(*args, **kwargs):
        patched = []
        for arg in args:
            if isinstance(arg, str):
                arg = arg.replace("V3.2", "V3.3")
                arg = arg.replace("v3.2", "v3.3")
                arg = arg.replace("PartTrace", "TokenPreserve")
            patched.append(arg)
        return original_print(*patched, **kwargs)

    original_print("=" * 104)
    original_print("NESTSAR ATTENTION-LITE + TOKENPRESERVE V3.3 — DUAL T4")
    original_print(
        f"T=16 | preserved fine tokens=320 | learned readout tokens={v33.readout_tokens} | "
        "no second temporal backbone"
    )
    original_print("=" * 104)

    builtins.print = print_v33
    try:
        return int(base.main())
    finally:
        builtins.print = original_print


if __name__ == "__main__":
    raise SystemExit(main())

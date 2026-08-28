#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dedicated FROM-ZERO dual-T4 launcher for NestSAR v3.5.

This wrapper intentionally prevents pretrained checkpoint use. It reuses the
validated CUDA-only dual-T4 launcher, but forces the scratch regime:
- base checkpoint = scratch
- allow scratch = true
- canonical Attention-Lite base trains from epoch 1
- no base freeze / no unfreeze ramp
- scratch learning-rate and regularization schedule

The epoch-0 pass is only a random-initialization sanity check; it is NOT
pretraining. The notebook label is changed from PRETRAIN BASELINE to
SCRATCH E0 CHECK to make that explicit.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.parttrace_v3_attention_lite import run_both_t4_v35_kaggle_v2 as hardened

base = hardened.base

_ORIGINAL_SET_BAR = base._set_bar


def _scratch_set_bar(bar, state, previous_phase):
    phase = str(state.get("phase", "initializing"))
    out = _ORIGINAL_SET_BAR(bar, state, previous_phase)
    if phase == "baseline":
        proto = str(state.get("protocol", "?")).upper()
        bar.set_description_str(f"{proto} SCRATCH E0 CHECK", refresh=False)
        bar.refresh()
    return out


def _force_arg(argv: list[str], name: str, value: str | None = None) -> list[str]:
    out = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == name:
            i += 1
            if value is not None and i < len(argv):
                i += 1
            continue
        if value is not None and arg.startswith(name + "="):
            i += 1
            continue
        out.append(arg)
        i += 1
    out.append(name)
    if value is not None:
        out.append(value)
    return out


def main() -> int:
    argv = list(sys.argv)

    # Hard scratch guards. User CLI cannot override these with pretrained values.
    argv = _force_arg(argv, "--base-checkpoint", "scratch")
    argv = _force_arg(argv, "--allow-scratch")
    argv = _force_arg(argv, "--freeze-base-epochs", "0")
    argv = _force_arg(argv, "--base-unfreeze-ramp-epochs", "0")

    # Accuracy-first scratch schedule. These may be changed here deliberately,
    # but not accidentally inherited from the pretrained fine-tuning preset.
    argv = _force_arg(argv, "--epochs", "80")
    argv = _force_arg(argv, "--patience", "15")
    argv = _force_arg(argv, "--base-lr", "4e-4")
    argv = _force_arg(argv, "--new-lr", "6e-4")
    argv = _force_arg(argv, "--base-min-lr", "1e-5")
    argv = _force_arg(argv, "--new-min-lr", "2e-5")
    argv = _force_arg(argv, "--warmup-fraction", "0.08")
    argv = _force_arg(argv, "--base-weight-decay", "0.03")
    argv = _force_arg(argv, "--new-weight-decay", "0.04")
    argv = _force_arg(argv, "--dropout", "0.10")
    argv = _force_arg(argv, "--ema-decay", "0.995")
    argv = _force_arg(argv, "--predictive-loss-weight", "0.05")
    argv = _force_arg(argv, "--memory-aux-warmup-weight", "0.50")
    argv = _force_arg(argv, "--memory-aux-final-weight", "0.20")

    sys.argv = argv
    original = base._set_bar
    base._set_bar = _scratch_set_bar
    try:
        print("=" * 120, flush=True)
        print("NESTSAR v3.5 — STRICT FROM-ZERO MODE", flush=True)
        print("No pretrained checkpoint can be loaded by this launcher.", flush=True)
        print("Epoch-0 validation is a random-init sanity check only.", flush=True)
        print("=" * 120, flush=True)
        return int(hardened.main())
    finally:
        base._set_bar = original


if __name__ == "__main__":
    raise SystemExit(main())

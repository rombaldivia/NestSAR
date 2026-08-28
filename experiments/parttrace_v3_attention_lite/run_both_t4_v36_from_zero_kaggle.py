#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dedicated FROM-ZERO dual-T4 launcher for NestSAR v3.6.

Real frame-aware execution:
- XSUB -> physical T4 #0, XSET -> physical T4 #1 concurrently;
- canonical ns.CFG uses requested T16/T32/T64;
- full Attention-Lite sees all T frames;
- side memory is motion-aware compressed to 16 temporal anchors / 640 tokens;
- no pretrained checkpoint can be loaded;
- canonical base gradients are active from epoch 1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.parttrace_v3_attention_lite import run_both_t4_v35_kaggle_v2 as hardened

base = hardened.base
HERE = Path(__file__).resolve().parent
TRAINER_V36 = HERE / "train_v36_frames.py"
_ORIGINAL_SET_BAR = base._set_bar


def _extract_frames(argv: list[str]) -> int:
    frames = 64
    for i, arg in enumerate(argv):
        if arg == "--frames" and i + 1 < len(argv):
            frames = int(argv[i + 1])
        elif arg.startswith("--frames="):
            frames = int(arg.split("=", 1)[1])
    if frames not in (16, 32, 64):
        raise ValueError(f"v3.6 supports --frames 16, 32, or 64; got {frames}")
    return frames


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


def _scratch_set_bar(bar, state, previous_phase):
    phase = str(state.get("phase", "initializing"))
    out = _ORIGINAL_SET_BAR(bar, state, previous_phase)
    if phase == "baseline":
        proto = str(state.get("protocol", "?")).upper()
        bar.set_description_str(f"{proto} SCRATCH E0 CHECK", refresh=False)
        bar.refresh()
    return out


def main() -> int:
    requested = _extract_frames(sys.argv[1:])
    if requested % 16:
        raise ValueError(f"v3.6 requires requested frames divisible by 16; got {requested}")

    old_env_frames = os.environ.get("NESTSAR_FRAMES")
    old_trainer = base.TRAINER
    old_set_bar = base._set_bar
    original_argv = list(sys.argv)

    # The reused notebook parent is historically T16 and passes '--frames 16'
    # to children.  The real requested T is carried in NESTSAR_FRAMES and owned
    # by train_v36_frames.py, which updates ns.CFG before dataset construction.
    argv = list(sys.argv)
    argv = _force_arg(argv, "--frames", "16")

    # Strict scratch guards: no checkpoint path can override these.
    argv = _force_arg(argv, "--base-checkpoint", "scratch")
    argv = _force_arg(argv, "--allow-scratch")
    argv = _force_arg(argv, "--freeze-base-epochs", "0")
    argv = _force_arg(argv, "--base-unfreeze-ramp-epochs", "0")

    sys.argv = argv
    os.environ["NESTSAR_FRAMES"] = str(requested)
    base.TRAINER = TRAINER_V36
    base._set_bar = _scratch_set_bar

    print("=" * 120, flush=True)
    print("NESTSAR v3.6 — STRICT FROM-ZERO MULTI-FRAME MODE", flush=True)
    print(
        f"Full canonical input=T{requested} | side-memory anchors=T16 | "
        f"memory tokens=640 | reduction={requested // 16}x",
        flush=True,
    )
    print("No pretrained checkpoint can be loaded by this launcher.", flush=True)
    print("Base trains from E1; memory correction is uncertainty-gated and capped.", flush=True)
    print("=" * 120, flush=True)

    try:
        return int(hardened.main())
    finally:
        sys.argv = original_argv
        base.TRAINER = old_trainer
        base._set_bar = old_set_bar
        if old_env_frames is None:
            os.environ.pop("NESTSAR_FRAMES", None)
        else:
            os.environ["NESTSAR_FRAMES"] = old_env_frames


if __name__ == "__main__":
    raise SystemExit(main())

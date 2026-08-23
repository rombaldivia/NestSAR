#!/usr/bin/env python3
"""Notebook-safe dual-T4 launcher using corrected quiet trainer V2.

GPU0 -> XSUB
GPU1 -> XSET

This wrapper keeps the existing memory-aware dual-T4 orchestration, throttles
routine RAM console telemetry, and rewrites only the child trainer module from
``trainer_t4`` to ``trainer_t4_v2``.  No model/training math is changed.
"""
from __future__ import annotations

import builtins
import re
import time

import run_attention_lite_dual_t4 as base

RUNNER_API_VERSION = "attention-lite-dual-t4-quiet-v2-fixed-tqdm"
RAM_CONSOLE_REPORT_SECONDS = 60.0
RAM_ALWAYS_SHOW_BELOW_GIB = 3.0
_last_ram_print = 0.0


def _quiet_parent_print(*args, **kwargs):
    """Throttle only routine RAM telemetry; leave all other messages untouched."""
    global _last_ram_print

    text = " ".join(str(x) for x in args)
    if text.startswith("[RAM]"):
        now = time.time()
        match = re.search(r"available=([0-9.]+)\s+GiB", text)
        available = float(match.group(1)) if match else None
        urgent = available is not None and available < RAM_ALWAYS_SHOW_BELOW_GIB
        if not urgent and now - _last_ram_print < RAM_CONSOLE_REPORT_SECONDS:
            return
        _last_ram_print = now

    builtins.print(*args, **kwargs)


base.print = _quiet_parent_print

# Intercept only the child-training module invocation. subprocess.run also uses
# Popen internally, but all other commands (nvidia-smi, CUDA probe, etc.) pass
# through untouched.
_ORIGINAL_POPEN = base.subprocess.Popen
_OLD_MODULE = "experiments.attention_lite_v1.trainer_t4"
_NEW_MODULE = "experiments.attention_lite_v1.trainer_t4_v2"


def _popen_with_fixed_trainer(args, *pargs, **kwargs):
    if isinstance(args, (list, tuple)) and _OLD_MODULE in args:
        rewritten = list(args)
        rewritten[rewritten.index(_OLD_MODULE)] = _NEW_MODULE
        args = rewritten
    return _ORIGINAL_POPEN(args, *pargs, **kwargs)


base.subprocess.Popen = _popen_with_fixed_trainer


def main() -> int:
    builtins.print("=" * 108, flush=True)
    builtins.print(f"RUNNER API: {RUNNER_API_VERSION}", flush=True)
    builtins.print("CHILD TRAINER: trainer_t4_v2 (valid quiet tqdm patch)", flush=True)
    builtins.print("Per-batch output: OFF | epoch/validation summaries: ON", flush=True)
    builtins.print("=" * 108, flush=True)
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())

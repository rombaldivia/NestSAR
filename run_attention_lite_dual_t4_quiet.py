#!/usr/bin/env python3
"""Notebook-output-safe entry point for the dual-T4 Attention-Lite runner.

This wrapper keeps the base runner's 5-second RAM safety polling but throttles
normal `[RAM]` console reports to once per minute. Low-memory warnings are always
shown immediately. The actual per-batch tqdm streams are disabled by trainer_t4,
so Kaggle/Jupyter output does not accumulate tens of thousands of progress-bar
updates in notebook memory.

Training behavior is unchanged:
  GPU0 -> XSUB
  GPU1 -> XSET
  validation every epoch using EMA weights
"""
from __future__ import annotations

import builtins
import re
import time

import run_attention_lite_dual_t4 as base

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

        urgent = (
            available is not None
            and available < RAM_ALWAYS_SHOW_BELOW_GIB
        )
        if not urgent and now - _last_ram_print < RAM_CONSOLE_REPORT_SECONDS:
            return
        _last_ram_print = now

    builtins.print(*args, **kwargs)


# Functions defined in the imported module resolve `print` from that module's
# globals first, so assigning this throttler controls only base-runner console
# output. It does not affect Python globally and does not change RAM monitoring.
base.print = _quiet_parent_print


if __name__ == "__main__":
    raise SystemExit(base.main())

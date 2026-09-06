#!/usr/bin/env python3
from __future__ import annotations

"""Dual-T4 launcher for SM-ALL with corrected preprocessing.

Reuses the existing parent launcher unchanged except for the worker module name.
GPU0 -> XSUB and GPU1 -> XSET, with exactly the same model/training arguments
and the same two persistent progress rows.
"""

from experiments.nestsar_sm_all_t16 import run_dual_t4 as legacy

# Keep an immutable reference to the original command builder BEFORE replacing
# legacy.worker_cmd. Calling legacy.worker_cmd from the wrapper after monkey
# patching would recurse back into corrected_worker_cmd forever.
_ORIGINAL_WORKER_CMD = legacy.worker_cmd


def corrected_worker_cmd(args, protocol):
    cmd = _ORIGINAL_WORKER_CMD(args, protocol)
    old = "experiments.nestsar_sm_all_t16.train_gpu"
    new = "experiments.nestsar_sm_all_t16.train_gpu_corrected"
    try:
        i = cmd.index(old)
    except ValueError as exc:
        raise RuntimeError(f"Could not locate legacy worker module in command: {cmd}") from exc
    cmd[i] = new
    return cmd


def main():
    # Patch only while the legacy parent launcher is running. Restoring the
    # original function makes notebook reruns/imports deterministic and avoids
    # keeping mutated module state after an exception or completed run.
    previous = legacy.worker_cmd
    legacy.worker_cmd = corrected_worker_cmd
    try:
        legacy.main()
    finally:
        legacy.worker_cmd = previous


if __name__ == "__main__":
    main()

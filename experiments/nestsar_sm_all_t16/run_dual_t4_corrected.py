#!/usr/bin/env python3
from __future__ import annotations

"""Dual-T4 launcher for SM-ALL with corrected preprocessing.

Reuses the existing parent launcher unchanged except for the worker module name.
GPU0 -> XSUB and GPU1 -> XSET, with exactly the same model/training arguments
and the same two persistent progress rows.
"""

from experiments.nestsar_sm_all_t16 import run_dual_t4 as legacy


def corrected_worker_cmd(args, protocol):
    cmd = legacy.worker_cmd(args, protocol)
    old = "experiments.nestsar_sm_all_t16.train_gpu"
    new = "experiments.nestsar_sm_all_t16.train_gpu_corrected"
    try:
        i = cmd.index(old)
    except ValueError as exc:
        raise RuntimeError(f"Could not locate legacy worker module in command: {cmd}") from exc
    cmd[i] = new
    return cmd


def main():
    legacy.worker_cmd = corrected_worker_cmd
    legacy.main()


if __name__ == "__main__":
    main()

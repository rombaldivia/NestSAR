#!/usr/bin/env python3
from __future__ import annotations

"""Clean-console launcher for the memory-safe BiJoint experiment.

Reuses the verified sequential launcher and all of its training settings, but
launches train_gpu_memsafe_clean so preprocessing tqdm bars do not explode into
one notebook line per refresh when stdout is piped.
"""

from experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16 import run_memsafe_dual_t4 as base

_ORIGINAL_WORKER_COMMAND = base.worker_command


def _clean_worker_command(dataset: str, outdir: str, protocol: str) -> list[str]:
    cmd = _ORIGINAL_WORKER_COMMAND(dataset, outdir, protocol)
    target = (
        "experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16."
        "train_gpu_memsafe"
    )
    replacement = (
        "experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16."
        "train_gpu_memsafe_clean"
    )
    return [replacement if item == target else item for item in cmd]


base.worker_command = _clean_worker_command
main = base.main


if __name__ == "__main__":
    raise SystemExit(main())

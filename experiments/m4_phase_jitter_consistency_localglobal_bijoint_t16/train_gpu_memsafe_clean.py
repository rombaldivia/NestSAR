#!/usr/bin/env python3
from __future__ import annotations

"""Clean-console wrapper for the memory-safe BiJoint worker.

The underlying trainer is unchanged. This wrapper only replaces tqdm progress
objects used by preprocessing with a silent iterable so subprocess piping does
not turn carriage-return progress bars into thousands of notebook lines.
Periodic training/update logs from train_gpu_memsafe.py remain visible.
"""

from experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16 import train_gpu_memsafe as core


class SilentProgress:
    """Minimal tqdm-compatible wrapper with no console output."""

    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else ()

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return None

    def refresh(self, *args, **kwargs):
        return None

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def main() -> None:
    # ju.build_protocol_views uses ju.tqdm for canonical/jitter preprocessing.
    # Patch both modules defensively; this changes display only, not data/order.
    core.ju.tqdm = SilentProgress
    core.cons.tqdm = SilentProgress
    core.main()


if __name__ == "__main__":
    main()

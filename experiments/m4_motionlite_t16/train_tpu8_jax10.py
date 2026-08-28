#!/usr/bin/env python3
from __future__ import annotations

"""JAX-0.10-safe entry point for the M4 MotionLite TPU trainer."""

from experiments.m4_motionlite_t16.jax10_compat import install

install()

from experiments.m4_motionlite_t16 import train_m4_motionlite_t16_tpu as base


if __name__ == "__main__":
    base.main()

#!/usr/bin/env python3
from __future__ import annotations

"""JAX-0.10-safe wrapper around the existing TPU8 preflight."""

from experiments.m4_motionlite_t16.jax10_compat import install

install()

from experiments.m4_motionlite_t16 import preflight_tpu8 as base


if __name__ == "__main__":
    raise SystemExit(base.main())

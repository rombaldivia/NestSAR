#!/usr/bin/env python3
"""Short process-isolated TPU4 collective probe.

The parent launcher sets TPU visibility before this process starts. A PASS here
proves more than ``len(jax.devices()) == 4``: it forces an actual four-device psum
collective, which catches invalid physical/global device numbering before a long
Attention-Lite compilation/training run is started.
"""
from __future__ import annotations

import functools
import json
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp


def main() -> int:
    protocol = os.environ.get("NESTSAR_PROTOCOL", "unknown").upper()
    backend = jax.default_backend()
    devices = list(jax.devices())

    print("=" * 100, flush=True)
    print(f"{protocol} ISOLATED TPU4 COLLECTIVE PROBE", flush=True)
    print(f"JAX: {jax.__version__}", flush=True)
    print(f"Backend: {backend}", flush=True)
    print(f"Visible devices ({len(devices)}): {devices}", flush=True)
    print(
        "Visibility env:",
        json.dumps(
            {
                "TPU_VISIBLE_CHIPS": os.environ.get("TPU_VISIBLE_CHIPS"),
                "TPU_VISIBLE_DEVICES": os.environ.get("TPU_VISIBLE_DEVICES"),
                "TPU_CHIPS_PER_PROCESS_BOUNDS": os.environ.get(
                    "TPU_CHIPS_PER_PROCESS_BOUNDS"
                ),
                "TPU_PROCESS_BOUNDS": os.environ.get("TPU_PROCESS_BOUNDS"),
                "TPU_MESH_CONTROLLER_ADDRESS": os.environ.get(
                    "TPU_MESH_CONTROLLER_ADDRESS"
                ),
                "TPU_MESH_CONTROLLER_PORT": os.environ.get(
                    "TPU_MESH_CONTROLLER_PORT"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if backend != "tpu":
        raise RuntimeError(f"Expected TPU backend, got {backend}")
    if len(devices) != 4:
        raise RuntimeError(
            f"Expected exactly 4 process-visible TPU devices, found {len(devices)}"
        )

    # Explicit pmap+psum forces the same class of all-reduce collective that failed
    # in the one-runtime TPU[4:8] experiment.
    @functools.partial(jax.pmap, axis_name="probe", devices=devices)
    def collective(x):
        return jax.lax.psum(x, "probe")

    x = jnp.arange(4, dtype=jnp.float32)
    y = collective(x)
    y.block_until_ready()
    got = [float(v) for v in y.tolist()]
    expected = [6.0, 6.0, 6.0, 6.0]
    if got != expected:
        raise RuntimeError(f"TPU4 psum mismatch: got {got}, expected {expected}")

    print(f"Collective psum: {got}", flush=True)
    print(f"{protocol} TPU4 COLLECTIVE PROBE: PASS", flush=True)
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        print(
            f"TPU4 COLLECTIVE PROBE: FAIL | {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TPU-forced notebook launcher for NestSAR v3.5.

This wrapper fixes a subtle Kaggle/JAX failure mode: a notebook can have CUDA
visible (or simply be running a GPU accelerator session), causing JAX to choose
``gpu`` before the v3.5 worker reaches its TPU guard.

The wrapper therefore:
1. runs an isolated child-process TPU probe before training;
2. forces JAX_PLATFORMS=tpu and hides CUDA from every v3.5 worker;
3. keeps the protocol-safe checkpoint worker and the persistent two-row tqdm UI.

The parent process itself never imports JAX, so it cannot acquire TPU devices.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.parttrace_v3_attention_lite import run_both_tpu_v35_kaggle as base

HERE = Path(__file__).resolve().parent
base.TRAINER = HERE / "train_v35_tpu_safe.py"

_ORIGINAL_POPEN = subprocess.Popen


def _forced_tpu_env(env=None):
    out = dict(os.environ if env is None else env)
    # Do not allow an installed CUDA backend to win platform selection.
    out["CUDA_VISIBLE_DEVICES"] = ""
    # No fallback: if TPU is unavailable, fail loudly instead of training on GPU/CPU.
    out["JAX_PLATFORMS"] = "tpu"
    out["PYTHONUNBUFFERED"] = "1"
    out.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    out.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return out


def _expected_devices_from_argv(default=8):
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--expected-devices" and i + 2 <= len(sys.argv[1:]):
            try:
                return int(sys.argv[1:][i + 1])
            except (ValueError, IndexError):
                pass
        if arg.startswith("--expected-devices="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                pass
    return int(default)


def _probe_tpu(expected_devices: int):
    code = r'''
import jax
print("JAX_VERSION=" + str(jax.__version__))
print("BACKEND=" + str(jax.default_backend()))
print("LOCAL_DEVICE_COUNT=" + str(jax.local_device_count()))
print("DEVICES=" + repr(jax.local_devices()))
'''
    probe = subprocess.run(
        [sys.executable, "-c", code],
        env=_forced_tpu_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (probe.stdout or "") + ("\n" + probe.stderr if probe.stderr else "")
    print("=" * 120, flush=True)
    print("V3.5 TPU PREFLIGHT", flush=True)
    print("=" * 120, flush=True)
    print(combined.strip(), flush=True)

    if probe.returncode != 0:
        raise RuntimeError(
            "TPU preflight failed before training. JAX could not initialize the TPU platform.\n"
            "This normally means the current Kaggle session is still a GPU/CPU session, "
            "or the TPU runtime was not initialized. Select a TPU accelerator in Kaggle "
            "and restart the session, then rerun the notebook from the clone cell."
        )

    backend_match = re.search(r"^BACKEND=(.+)$", combined, flags=re.MULTILINE)
    count_match = re.search(r"^LOCAL_DEVICE_COUNT=(\d+)$", combined, flags=re.MULTILINE)
    backend = backend_match.group(1).strip() if backend_match else "unknown"
    count = int(count_match.group(1)) if count_match else -1
    if backend != "tpu":
        raise RuntimeError(
            f"TPU preflight selected backend={backend!r}, expected 'tpu'. "
            "Restart the Kaggle notebook with TPU accelerator enabled."
        )
    if expected_devices > 0 and count != expected_devices:
        raise RuntimeError(
            f"TPU preflight found {count} local TPU devices, expected {expected_devices}. "
            "Do not start the experiment until the topology matches the requested run."
        )
    print(f"TPU PREFLIGHT PASS | backend=tpu | local_devices={count}", flush=True)
    print("=" * 120, flush=True)


def _forced_popen(*args, **kwargs):
    kwargs["env"] = _forced_tpu_env(kwargs.get("env"))
    return _ORIGINAL_POPEN(*args, **kwargs)


def main() -> int:
    expected = _expected_devices_from_argv(default=8)
    _probe_tpu(expected)

    # base.main() owns the persistent tqdm bars/progress JSON. Only replace its
    # worker process constructor so every child starts with TPU forced before JAX
    # is imported by train_v35_tpu_safe.py.
    original = base.subprocess.Popen
    base.subprocess.Popen = _forced_popen
    try:
        return int(base.main())
    finally:
        base.subprocess.Popen = original


if __name__ == "__main__":
    raise SystemExit(main())

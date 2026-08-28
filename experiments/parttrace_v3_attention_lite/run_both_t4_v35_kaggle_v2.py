#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CUDA-hardened dual-T4 launcher for NestSAR v3.5.

This wrapper keeps the validated notebook-native dual-T4 launcher/UI, but fixes
JAX 0.7.2 platform selection.  ``JAX_PLATFORMS=gpu`` may probe both CUDA and
ROCm; this Kaggle image has CUDA T4s but no ROCm backend.  Each child is forced
to the concrete ``cuda`` backend and exactly one physical GPU.

The parent never imports JAX.  It probes GPU0/GPU1 in isolated child processes,
then delegates to run_both_t4_v35_kaggle.py with its persistent two-row tqdm UI.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.parttrace_v3_attention_lite import run_both_t4_v35_kaggle as base

_ORIGINAL_POPEN = subprocess.Popen


def _cuda_env(env=None, gpu=None):
    out = dict(os.environ if env is None else env)
    if gpu is not None:
        out["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # JAX 0.7.x concrete NVIDIA backend.  Do not use generic 'gpu', which may
    # also probe ROCm and fail before CUDA initialization.
    out["JAX_PLATFORMS"] = "cuda"
    out["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    out["MALLOC_ARENA_MAX"] = "2"
    out["TF_CPP_MIN_LOG_LEVEL"] = "2"
    out["PYTHONUNBUFFERED"] = "1"
    return out


def _probe_gpu(gpu: int) -> None:
    code = r'''
import jax
print("JAX_VERSION=" + str(jax.__version__))
print("BACKEND=" + str(jax.default_backend()))
print("LOCAL_DEVICE_COUNT=" + str(jax.local_device_count()))
print("DEVICES=" + repr(jax.local_devices()))
assert jax.default_backend() == "gpu", jax.default_backend()
assert jax.local_device_count() == 1, jax.local_devices()
'''
    p = subprocess.run(
        [sys.executable, "-c", code],
        env=_cuda_env(gpu=gpu),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    print("=" * 100, flush=True)
    print(f"V3.5 CUDA PREFLIGHT — PHYSICAL GPU{gpu}", flush=True)
    print("=" * 100, flush=True)
    print(combined.strip(), flush=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"CUDA preflight failed for physical GPU{gpu}. "
            "Do not start v3.5 until each worker sees exactly one T4."
        )
    print(f"GPU{gpu} PREFLIGHT PASS | JAX backend=gpu | local_devices=1", flush=True)


def _cuda_popen(*args, **kwargs):
    # The base launcher has already selected the physical GPU in
    # CUDA_VISIBLE_DEVICES.  Preserve that selection but force concrete CUDA.
    env = kwargs.get("env")
    kwargs["env"] = _cuda_env(env=env)
    return _ORIGINAL_POPEN(*args, **kwargs)


def main() -> int:
    # Parse only the two GPU selectors without importing JAX or duplicating the
    # full experiment CLI.  Defaults match the base launcher.
    gpu_xsub, gpu_xset = 0, 1
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--gpu-xsub" and i + 1 < len(argv):
            gpu_xsub = int(argv[i + 1])
        elif arg.startswith("--gpu-xsub="):
            gpu_xsub = int(arg.split("=", 1)[1])
        elif arg == "--gpu-xset" and i + 1 < len(argv):
            gpu_xset = int(argv[i + 1])
        elif arg.startswith("--gpu-xset="):
            gpu_xset = int(arg.split("=", 1)[1])

    if gpu_xsub == gpu_xset:
        raise ValueError("XSUB and XSET must use different physical GPUs")

    _probe_gpu(gpu_xsub)
    _probe_gpu(gpu_xset)

    original = base.subprocess.Popen
    base.subprocess.Popen = _cuda_popen
    try:
        return int(base.main())
    finally:
        base.subprocess.Popen = original


if __name__ == "__main__":
    raise SystemExit(main())

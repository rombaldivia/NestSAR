#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

EXPECTED_DEVICES = 8
TEACHER = "/kaggle/input/models/romelbaldivia/cdformer-jax/jax/default/1/cdformer16_teacher_jax.npz"
CACHE = "/kaggle/working/cdformer16_mmaction2_teacher_logits.npz"
OUTDIR = "/kaggle/working/NestSAR_M4_CDFormer_LogitKD_T16_TPU"


def forced_tpu_env():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_PLATFORMS"] = "tpu"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    return env


def probe_tpu():
    code = r'''
import jax
print("JAX_VERSION=" + str(jax.__version__))
print("BACKEND=" + str(jax.default_backend()))
print("LOCAL_DEVICE_COUNT=" + str(jax.local_device_count()))
print("DEVICES=" + repr(jax.local_devices()))
'''
    r = subprocess.run([sys.executable, "-c", code], env=forced_tpu_env(), capture_output=True, text=True)
    combined = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
    print("=" * 120, flush=True)
    print("CD-FORMER JAX LOGIT KD | ISOLATED TPU8 PROBE", flush=True)
    print("=" * 120, flush=True)
    print(combined.strip(), flush=True)
    if r.returncode != 0:
        raise RuntimeError("TPU probe failed")
    bm = re.search(r"^BACKEND=(.+)$", combined, re.MULTILINE)
    cm = re.search(r"^LOCAL_DEVICE_COUNT=(\d+)$", combined, re.MULTILINE)
    backend = bm.group(1).strip() if bm else "unknown"
    count = int(cm.group(1)) if cm else -1
    if backend != "tpu" or count != EXPECTED_DEVICES:
        raise RuntimeError(f"Expected TPU8, got {backend=} {count=}")
    print("TPU_RUNTIME_PROBE=PASS", flush=True)


def run_child(module, *args):
    cmd = [sys.executable, "-u", "-m", module, *args]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=forced_tpu_env(), check=True)


def main():
    probe_tpu()
    if not Path(TEACHER).is_file():
        raise FileNotFoundError(TEACHER)

    # Always rebuild cache in a fresh/controlled run so preprocessing is audited.
    run_child(
        "experiments.m4_cdformer_logit_kd_t16.cache_teacher_logits_tpu",
        "--teacher", TEACHER,
        "--output", CACHE,
        "--batch-size", "64",
    )

    print("=" * 120, flush=True)
    print("TEACHER REPRODUCTION PASSED — STARTING XSUB LOGIT KD", flush=True)
    print("=" * 120, flush=True)

    run_child(
        "experiments.m4_cdformer_logit_kd_t16.train_tpu",
        "--teacher-cache", CACHE,
        "--protocol", "xsub",
        "--epochs", "60",
        "--patience", "12",
        "--batch-size", "256",
        "--eval-batch-size", "512",
        "--learning-rate", "6e-4",
        "--min-learning-rate", "2e-5",
        "--warmup-fraction", "0.08",
        "--weight-decay", "0.03",
        "--label-smoothing", "0.05",
        "--grad-clip", "1.0",
        "--ema-decay", "0.995",
        "--stream-aux-weight", "0.15",
        "--spatial-dim", "24",
        "--model-dim", "112",
        "--dropout", "0.10",
        "--seed", "128",
        "--jitter-max-shift", "1",
        "--consistency-weight", "0.08",
        "--consistency-temperature", "1.0",
        "--kd-weight", "0.20",
        "--kd-temperature", "4.0",
        "--progress-every", "5",
        "--audit-first",
        "--outdir", OUTDIR,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

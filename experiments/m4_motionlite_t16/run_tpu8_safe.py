#!/usr/bin/env python3
from __future__ import annotations

"""Process-isolated launcher for one Kaggle TPU v5e-8.

Pattern intentionally mirrors the older validated NestSAR TPU launchers:
- this parent process NEVER imports JAX;
- an isolated child first proves backend=tpu and local_device_count=8;
- a second isolated child runs the real M4 MotionLite forward/backward/pmean smoke test;
- only after both pass is the full trainer started in a fresh TPU child process.
"""

import os
import re
import subprocess
import sys

EXPECTED_DEVICES = 8


def forced_tpu_env() -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_PLATFORMS"] = "tpu"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    return env


def probe_tpu() -> None:
    code = r'''
import jax
print("JAX_VERSION=" + str(jax.__version__))
print("BACKEND=" + str(jax.default_backend()))
print("LOCAL_DEVICE_COUNT=" + str(jax.local_device_count()))
print("DEVICES=" + repr(jax.local_devices()))
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=forced_tpu_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")

    print("=" * 112, flush=True)
    print("M4-MOTIONLITE-T16 | ISOLATED TPU RUNTIME PROBE", flush=True)
    print("=" * 112, flush=True)
    print(combined.strip(), flush=True)

    if result.returncode != 0:
        low = combined.lower()
        if "libtpu.so" in low and (
            "cannot open shared object file" in low or "failed to open" in low
        ):
            raise RuntimeError(
                "TPU JAX runtime is missing libtpu.so. Keep the TPU accelerator selected and "
                "repair the TPU-enabled JAX runtime before training."
            )
        raise RuntimeError("Isolated JAX TPU probe failed before model initialization")

    backend_m = re.search(r"^BACKEND=(.+)$", combined, flags=re.MULTILINE)
    count_m = re.search(r"^LOCAL_DEVICE_COUNT=(\d+)$", combined, flags=re.MULTILINE)
    backend = backend_m.group(1).strip() if backend_m else "unknown"
    count = int(count_m.group(1)) if count_m else -1

    if backend != "tpu":
        raise RuntimeError(f"Expected backend='tpu', got {backend!r}")
    if count != EXPECTED_DEVICES:
        raise RuntimeError(
            f"Expected one Kaggle v5e-8 exposing {EXPECTED_DEVICES} local TPU devices; got {count}"
        )

    print(
        f"TPU_RUNTIME_PROBE=PASS | backend=tpu | local_devices={count}",
        flush=True,
    )
    print("=" * 112, flush=True)


def run_child(module: str, *args: str) -> None:
    cmd = [sys.executable, "-u", "-m", module, *args]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=forced_tpu_env(), check=True)


def main() -> int:
    # 1) Runtime/topology only. Fresh process; parent has never imported JAX.
    probe_tpu()

    # 2) Full model smoke test: init + pmap + forward + backward + pmean + real NTU sample.
    run_child("experiments.m4_motionlite_t16.preflight_tpu8")

    # 3) Fresh TPU process for the actual long run. XSUB and XSET are sequential.
    print("PREFLIGHTS PASSED — STARTING FULL TRAINING", flush=True)
    run_child(
        "experiments.m4_motionlite_t16.train_m4_motionlite_t16_tpu",
        "--protocol", "both",
        "--selector", "motion",
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
        "--progress-every", "5",
        "--audit-first",
        "--outdir", "/kaggle/working/NestSAR_M4_MotionLite_T16_TPU",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

EXPECTED_GPUS = 2
DEFAULT_DATASET = "/kaggle/input/models/paolamaydana/ntudanno/other/default/1/ntu120_3danno.pkl"
DEFAULT_OUTDIR = "/kaggle/working/NestSAR_M4_Phase_JitterConsistency_LocalGlobal_T16_DualT4"


def gpu_env(index: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(index)
    env["JAX_PLATFORMS"] = "cuda"
    env["PYTHONUNBUFFERED"] = "1"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    return env


def visible_gpus() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def probe_gpu(index: int) -> None:
    code = r'''
import jax
print("JAX_VERSION=" + str(jax.__version__))
print("BACKEND=" + str(jax.default_backend()))
print("LOCAL_DEVICE_COUNT=" + str(jax.local_device_count()))
print("DEVICES=" + repr(jax.local_devices()))
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=gpu_env(index),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    print("=" * 120, flush=True)
    print(f"GPU{index} ISOLATED JAX PROBE", flush=True)
    print("=" * 120, flush=True)
    print(combined.strip(), flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"GPU{index} JAX probe failed")
    backend = re.search(r"^BACKEND=(.+)$", combined, re.MULTILINE)
    count = re.search(r"^LOCAL_DEVICE_COUNT=(\d+)$", combined, re.MULTILINE)
    if not backend or backend.group(1).strip() != "gpu":
        raise RuntimeError(f"GPU{index}: JAX did not select GPU backend")
    if not count or int(count.group(1)) != 1:
        raise RuntimeError(f"GPU{index}: expected exactly one visible JAX GPU")


def stream_output(proc: subprocess.Popen, prefix: str, lock: threading.Lock) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        with lock:
            print(f"[{prefix}] {line}", end="", flush=True)


def worker_command(dataset: str, outdir: str, protocol: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "experiments.m4_phase_jitter_consistency_localglobal_t16.train_gpu",
        "--dataset", dataset,
        "--protocol", protocol,
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
        "--progress-every", "5",
        "--outdir", outdir,
    ]


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET
    outdir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTDIR

    if not Path(dataset).is_file():
        raise FileNotFoundError(dataset)

    gpus = visible_gpus()
    print("=" * 120, flush=True)
    print("NESTSAR LOCALGLOBAL V2 — KAGGLE DUAL T4", flush=True)
    print("=" * 120, flush=True)
    for line in gpus:
        print("GPU:", line, flush=True)
    if len(gpus) != EXPECTED_GPUS:
        raise RuntimeError(f"Expected exactly 2 GPUs from nvidia-smi, found {len(gpus)}")

    # The launcher process never imports JAX. Each probe/worker owns only its assigned T4.
    probe_gpu(0)
    probe_gpu(1)

    print("=" * 120, flush=True)
    print("PREPROCESSING/MODEL PREFLIGHT ON GPU0", flush=True)
    print("=" * 120, flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "experiments.m4_phase_jitter_consistency_localglobal_t16.preflight_gpu",
            dataset,
        ],
        env=gpu_env(0),
        check=True,
    )

    Path(outdir).mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "M4PhaseJitterConsistencyLocalPoseGlobalMotionT16_DualT4",
        "gpu_assignment": {"xsub": 0, "xset": 1},
        "dataset": dataset,
        "outdir": outdir,
        "batch_size_per_protocol": 256,
        "optimization": "same global batch/schedule as T16 champion",
        "expected_params": 1_816_130,
        "preprocessing": "local_pose_global_motion_v2",
    }
    (Path(outdir) / "dual_t4_manifest.json").write_text(json.dumps(manifest, indent=2))

    jobs = [
        ("XSUB/GPU0", 0, "xsub"),
        ("XSET/GPU1", 1, "xset"),
    ]
    processes: list[tuple[str, subprocess.Popen]] = []
    lock = threading.Lock()
    threads: list[threading.Thread] = []

    print("=" * 120, flush=True)
    print("STARTING BOTH PROTOCOLS CONCURRENTLY", flush=True)
    print("XSUB -> physical GPU0 | XSET -> physical GPU1", flush=True)
    print("=" * 120, flush=True)

    for prefix, gpu_index, protocol in jobs:
        cmd = worker_command(dataset, outdir, protocol)
        print(f"LAUNCH {prefix}: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            env=gpu_env(gpu_index),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append((prefix, proc))
        t = threading.Thread(target=stream_output, args=(proc, prefix, lock), daemon=True)
        t.start()
        threads.append(t)

    failures = []
    for prefix, proc in processes:
        rc = proc.wait()
        if rc != 0:
            failures.append((prefix, rc))

    for t in threads:
        t.join(timeout=5.0)

    if failures:
        raise RuntimeError(f"Dual-T4 worker failure(s): {failures}")

    results = {}
    for protocol in ("xsub", "xset"):
        p = Path(outdir) / f"result_{protocol}.json"
        if not p.is_file():
            raise RuntimeError(f"Missing worker result: {p}")
        results[protocol] = json.loads(p.read_text())

    (Path(outdir) / "summary.json").write_text(json.dumps(results, indent=2))
    print("=" * 120, flush=True)
    print("DUAL T4 DONE", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print("=" * 120, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

EXPECTED_GPUS = 2
EXPECTED_PARAMS = 1_816_130
DEFAULT_DATASET = "/kaggle/input/models/paolamaydana/ntudanno/other/default/1/ntu120_3danno.pkl"
DEFAULT_OUTDIR = "/kaggle/working/NestSAR_M4_LocalGlobal_HardNeg_T16_DualT4"


def compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


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
    code = (
        "import jax; "
        "print('JAX_VERSION='+str(jax.__version__)); "
        "print('BACKEND='+str(jax.default_backend())); "
        "print('LOCAL_DEVICE_COUNT='+str(jax.local_device_count())); "
        "print('DEVICES='+repr(jax.local_devices()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=gpu_env(index),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
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


def stream_output(proc: subprocess.Popen, prefix: str, lock: threading.Lock, log_path: Path) -> None:
    assert proc.stdout is not None
    with log_path.open("a", buffering=1) as log_file:
        for line in proc.stdout:
            clean = line.rstrip("\r\n")
            log_file.write(clean + "\n")
            with lock:
                print(f"[{prefix}] {clean}", flush=True)


def worker_command(dataset: str, outdir: str, protocol: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "experiments.m4_phase_jitter_consistency_localglobal_hardneg_t16.train_gpu",
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
        "--hardneg-weight", "0.04",
        "--hardneg-margin", "0.20",
        "--progress-every", "5",
        "--outdir", outdir,
    ]


def terminate_workers(processes: list[tuple[str, subprocess.Popen]]) -> None:
    for _, proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 10.0
    while time.time() < deadline and any(proc.poll() is None for _, proc in processes):
        time.sleep(0.2)
    for _, proc in processes:
        if proc.poll() is None:
            proc.kill()


def main(dataset: str | None = None, outdir: str | None = None) -> int:
    dataset = dataset or (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET)
    outdir = outdir or (sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTDIR)

    if not Path(dataset).is_file():
        raise FileNotFoundError(dataset)

    gpus = visible_gpus()
    print("=" * 120, flush=True)
    print("NESTSAR LOCALGLOBAL V2 + HARDNEG — FROM SCRATCH — KAGGLE DUAL T4", flush=True)
    print("=" * 120, flush=True)
    for line in gpus:
        print("GPU:", line, flush=True)
    if len(gpus) != EXPECTED_GPUS:
        raise RuntimeError(f"Expected exactly 2 GPUs from nvidia-smi, found {len(gpus)}")

    probe_gpu(0)
    probe_gpu(1)

    # Architecture/preprocessing are intentionally identical to LocalGlobal V2,
    # so reuse its validated preflight. HardNeg exists only in the training loss.
    print("=" * 120, flush=True)
    print("LOCALGLOBAL PREPROCESSING/MODEL PREFLIGHT ON GPU0", flush=True)
    print("=" * 120, flush=True)
    preflight = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "experiments.m4_phase_jitter_consistency_localglobal_t16.preflight_gpu",
            dataset,
        ],
        env=gpu_env(0),
    )
    while preflight.poll() is None:
        time.sleep(1.0)
    if preflight.returncode != 0:
        raise RuntimeError(f"GPU preflight failed with exit code {preflight.returncode}")

    out_path = Path(outdir)
    if out_path.exists() and any(out_path.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite a non-empty experiment directory: {out_path}"
        )
    out_path.mkdir(parents=True, exist_ok=True)
    logs_dir = out_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "M4PhaseJitterConsistencyLocalGlobalHardNegT16_DualT4",
        "baseline": "M4PhaseJitterConsistencyLocalPoseGlobalMotionT16_DualT4",
        "baseline_accuracy": {
            "xsub": 0.7531176967340285,
            "xset": 0.7592682885821410,
        },
        "gpu_assignment": {"xsub": 0, "xset": 1},
        "dataset": dataset,
        "outdir": outdir,
        "training_from_scratch": True,
        "batch_size_per_protocol": 256,
        "epochs": 60,
        "patience": 12,
        "learning_rate": 6e-4,
        "optimization": "same schedule as LocalGlobal V2 champion",
        "expected_params": EXPECTED_PARAMS,
        "preprocessing": "local_pose_global_motion_v2",
        "architecture_change": False,
        "inference_change": False,
        "hard_negative": {
            "weight": 0.04,
            "margin": 0.20,
            "form": "softplus(hardest_wrong_logit - true_logit + margin)",
            "views": "canonical_and_jitter",
        },
    }
    (out_path / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))

    jobs = [
        ("XSUB/GPU0", 0, "xsub"),
        ("XSET/GPU1", 1, "xset"),
    ]
    processes: list[tuple[str, subprocess.Popen]] = []
    lock = threading.Lock()
    threads: list[threading.Thread] = []

    print("=" * 120, flush=True)
    print("STARTING BOTH PROTOCOLS FROM RANDOM INITIALIZATION", flush=True)
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
        thread = threading.Thread(
            target=stream_output,
            args=(proc, prefix, lock, logs_dir / f"{protocol}.log"),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    start = time.time()
    last_heartbeat = 0.0
    try:
        while True:
            alive = [(prefix, proc) for prefix, proc in processes if proc.poll() is None]
            now = time.time()
            if now - last_heartbeat >= 30.0:
                status = " | ".join(
                    f"{prefix}={'RUNNING' if proc.poll() is None else 'DONE rc='+str(proc.returncode)}"
                    for prefix, proc in processes
                )
                print(
                    f"[HARDNEG DUAL-T4 HEARTBEAT] elapsed={now-start:.0f}s | {status}",
                    flush=True,
                )
                last_heartbeat = now
            if not alive:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[HARDNEG DUAL-T4] KeyboardInterrupt; stopping both workers...", flush=True)
        terminate_workers(processes)
        print(f"[HARDNEG DUAL-T4] Partial logs are in {logs_dir}", flush=True)
        return 130

    for thread in threads:
        thread.join(timeout=5.0)

    failures = [(prefix, proc.returncode) for prefix, proc in processes if proc.returncode != 0]
    if failures:
        raise RuntimeError(f"Dual-T4 worker failure(s): {failures}; logs={logs_dir}")

    results = {}
    for protocol in ("xsub", "xset"):
        path = out_path / f"result_{protocol}.json"
        if not path.is_file():
            raise RuntimeError(f"Missing worker result: {path}")
        results[protocol] = json.loads(path.read_text())

    (out_path / "summary.json").write_text(json.dumps(results, indent=2))

    print("=" * 120, flush=True)
    print("HARDNEG DUAL T4 DONE", flush=True)
    for protocol in ("xsub", "xset"):
        r = results[protocol]
        print(
            f"{protocol.upper()} best={100*r['best_val_accuracy']:.6f}% "
            f"@ E{r['best_epoch']} | baseline={100*r['baseline_localglobal_v2']:.6f}% "
            f"| delta={r['delta_vs_baseline_pp']:+.4f} pp",
            flush=True,
        )
    print("=" * 120, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

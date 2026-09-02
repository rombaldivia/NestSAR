#!/usr/bin/env python3
from __future__ import annotations

"""Sequential Dual-T4 launcher for the memory-safe BiJoint ablation.

The original concurrent launcher can exceed Kaggle's 31-GiB host RAM because
XSUB and XSET each materialize canonical+jitter train arrays plus validation
arrays.  This launcher intentionally runs the protocols one after another:
XSUB on physical GPU0, then XSET on physical GPU1.  Each worker exits before the
next starts, releasing host and device memory.

GPU training keeps effective batch=256 via four 64-sample gradient-accumulation
microbatches in train_gpu_memsafe.py.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

EXPECTED_GPUS = 2
EXPECTED_PARAMS = 1_834_946

DEFAULT_DATASET = (
    "/kaggle/input/models/paolamaydana/"
    "ntudanno/other/default/1/ntu120_3danno.pkl"
)
DEFAULT_OUTDIR = (
    "/kaggle/working/"
    "NestSAR_M4_LocalGlobal_BiJoint_T16_MemorySafe"
)


def compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def gpu_env(index: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(index)
    env["JAX_PLATFORMS"] = "cuda"
    env["PYTHONUNBUFFERED"] = "1"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["NESTSAR_MICROBATCH"] = "64"
    env["NESTSAR_EVAL_MICROBATCH"] = "256"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    return env


def visible_gpus() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def probe_gpu(index: int) -> None:
    code = (
        "import jax; "
        "print('BACKEND='+str(jax.default_backend())); "
        "print('COUNT='+str(jax.local_device_count())); "
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
    print(f"GPU{index} JAX PROBE", flush=True)
    print("=" * 120, flush=True)
    print(combined.strip(), flush=True)

    if result.returncode != 0:
        raise RuntimeError(f"GPU{index} JAX probe failed")
    backend = re.search(r"^BACKEND=(.+)$", combined, re.MULTILINE)
    count = re.search(r"^COUNT=(\d+)$", combined, re.MULTILINE)
    if not backend or backend.group(1).strip() != "gpu":
        raise RuntimeError(f"GPU{index}: JAX backend is not GPU")
    if not count or int(count.group(1)) != 1:
        raise RuntimeError(f"GPU{index}: expected exactly one visible JAX GPU")


def worker_command(dataset: str, outdir: str, protocol: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16.train_gpu_memsafe",
        "--dataset", dataset,
        "--protocol", protocol,
        "--epochs", "60",
        "--patience", "12",
        # EFFECTIVE optimizer batch remains the champion 256.
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


def run_worker(prefix: str, gpu: int, protocol: str, dataset: str, outdir: str, logs_dir: Path) -> None:
    cmd = worker_command(dataset, outdir, protocol)
    log_path = logs_dir / f"{protocol}.log"

    print("=" * 120, flush=True)
    print(f"START {prefix} — SEQUENTIAL MEMORY-SAFE WORKER", flush=True)
    print("COMMAND:", " ".join(cmd), flush=True)
    print("LOG:", log_path, flush=True)
    print("=" * 120, flush=True)

    with log_path.open("w", buffering=1) as log_file:
        proc = subprocess.Popen(
            cmd,
            env=gpu_env(gpu),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            clean = line.rstrip("\r\n")
            log_file.write(clean + "\n")
            print(f"[{prefix}] {clean}", flush=True)
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(
            f"{prefix} failed with rc={rc}; inspect {log_path}"
        )

    print("=" * 120, flush=True)
    print(f"{prefix} DONE — worker process exited; RAM/GPU memory released", flush=True)
    print("=" * 120, flush=True)


def main(dataset: str | None = None, outdir: str | None = None) -> int:
    dataset = dataset or (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET)
    outdir = outdir or (sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTDIR)

    if not Path(dataset).is_file():
        raise FileNotFoundError(dataset)

    gpus = visible_gpus()
    print("=" * 120, flush=True)
    print("NESTSAR LOCALGLOBAL V2 + BI-JOINT M4/G4 — MEMORY-SAFE SEQUENTIAL DUAL T4", flush=True)
    print("=" * 120, flush=True)
    for line in gpus:
        print("GPU:", line, flush=True)
    if len(gpus) != EXPECTED_GPUS:
        raise RuntimeError(f"Expected exactly 2 GPUs, found {len(gpus)}")

    probe_gpu(0)
    probe_gpu(1)

    # Architecture/FLOP preflight remains the exact same one as the clean branch.
    print("=" * 120, flush=True)
    print("BIJOINT PARAM/FLOP PREFLIGHT ON GPU0", flush=True)
    print("=" * 120, flush=True)
    preflight = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16.preflight_gpu",
            dataset,
        ],
        env=gpu_env(0),
        check=False,
    )
    if preflight.returncode != 0:
        raise RuntimeError(f"GPU preflight failed with rc={preflight.returncode}")

    out_path = Path(outdir)
    if out_path.exists() and any(out_path.iterdir()):
        raise RuntimeError(
            f"Output directory is non-empty: {out_path}. "
            "Use a new directory or remove it intentionally."
        )
    out_path.mkdir(parents=True, exist_ok=True)
    logs_dir = out_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "M4LocalGlobalBiJointT16_MemorySafe",
        "dataset": dataset,
        "outdir": outdir,
        "expected_params": EXPECTED_PARAMS,
        "protocol_execution": "sequential_to_bound_host_ram",
        "gpu_assignment": {"xsub": 0, "xset": 1},
        "effective_batch": 256,
        "microbatch": 64,
        "gradient_accumulation_steps": 4,
        "optimizer_updates": "same as effective batch 256",
        "preprocessing": "local_pose_global_motion_v2",
        "spatial_joint_memory": "bidirectional_bimemory",
        "attention": False,
        "training_from_scratch": True,
        "reason": {
            "original_gpu_failure": "RESOURCE_EXHAUSTED allocating 4724652040 bytes at batch 256",
            "original_host_failure": "concurrent XSET worker received SIGKILL during preprocessing",
        },
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Important: intentionally NOT concurrent.
    run_worker("XSUB/GPU0", 0, "xsub", dataset, outdir, logs_dir)

    # Give CUDA/process teardown a small safety window.
    time.sleep(5.0)

    run_worker("XSET/GPU1", 1, "xset", dataset, outdir, logs_dir)

    results = {}
    for protocol in ("xsub", "xset"):
        path = out_path / f"result_{protocol}.json"
        if not path.is_file():
            raise RuntimeError(f"Missing result file: {path}")
        results[protocol] = json.loads(path.read_text())

    (out_path / "summary.json").write_text(json.dumps(results, indent=2))

    print("=" * 120, flush=True)
    print("BIJOINT MEMORY-SAFE RUN DONE", flush=True)
    for protocol in ("xsub", "xset"):
        r = results[protocol]
        print(
            f"{protocol.upper()} best={100*r['best_val_accuracy']:.6f}% "
            f"@ E{r['best_epoch']} | delta={r['delta_vs_baseline_pp']:+.4f} pp",
            flush=True,
        )
    print("=" * 120, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

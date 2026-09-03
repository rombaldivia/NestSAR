#!/usr/bin/env python3
from __future__ import annotations

"""True concurrent Dual-T4 launcher for BiJoint using exact float32 disk caches.

Workflow:
1) Build XSUB cache on CPU, process exits.
2) Build XSET cache on CPU, process exits.
3) Launch XSUB on physical GPU0 and XSET on physical GPU1 AT THE SAME TIME.

GPU workers mmap the cached arrays, so both protocols do not keep multi-GB copies
of Xcan/Xjit/Xval in host RAM. Neural training remains effective batch 256 via
64-sample microbatches x4 gradient accumulation.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

EXPECTED_GPUS = 2
EXPECTED_PARAMS = 1_834_946

DEFAULT_DATASET = "/kaggle/input/models/paolamaydana/ntudanno/other/default/1/ntu120_3danno.pkl"
DEFAULT_OUTDIR = "/kaggle/working/NestSAR_M4_LocalGlobal_BiJoint_T16_ConcurrentDualT4"
DEFAULT_CACHE_ROOT = "/kaggle/working/NestSAR_BiJoint_ExactFloat32_Cache"


def gpu_env(index: int, cache_root: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(index)
    env["JAX_PLATFORMS"] = "cuda"
    env["PYTHONUNBUFFERED"] = "1"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["NESTSAR_MICROBATCH"] = "64"
    env["NESTSAR_EVAL_MICROBATCH"] = "256"
    env["NESTSAR_CACHE_ROOT"] = cache_root
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    return env


def cpu_env() -> dict[str, str]:
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    return env


def visible_gpus() -> list[str]:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    return [x.strip() for x in r.stdout.splitlines() if x.strip()]


def probe_gpu(index: int, cache_root: str) -> None:
    code = (
        "import jax; "
        "print('BACKEND='+str(jax.default_backend())); "
        "print('COUNT='+str(jax.local_device_count())); "
        "print('DEVICES='+repr(jax.local_devices()))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], env=gpu_env(index, cache_root),
        capture_output=True, text=True, check=False,
    )
    combined = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
    print("=" * 120, flush=True)
    print(f"PHYSICAL GPU{index} JAX PROBE", flush=True)
    print("=" * 120, flush=True)
    print(combined.strip(), flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"GPU{index} probe failed")
    backend = re.search(r"^BACKEND=(.+)$", combined, re.MULTILINE)
    count = re.search(r"^COUNT=(\d+)$", combined, re.MULTILINE)
    if not backend or backend.group(1).strip() != "gpu" or not count or int(count.group(1)) != 1:
        raise RuntimeError(f"GPU{index} isolation failed")


def build_cache(dataset: str, cache_root: str, protocol: str) -> None:
    cmd = [
        sys.executable, "-u", "-m",
        "experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16.cache_protocol_views",
        "--dataset", dataset,
        "--protocol", protocol,
        "--cache-root", cache_root,
        "--seed", "128",
        "--jitter-max-shift", "1",
    ]
    print("=" * 120, flush=True)
    print(f"CACHE STAGE {protocol.upper()} — CPU / SEQUENTIAL", flush=True)
    print("=" * 120, flush=True)
    r = subprocess.run(cmd, env=cpu_env(), check=False)
    if r.returncode != 0:
        raise RuntimeError(f"{protocol.upper()} cache build failed rc={r.returncode}")


def worker_command(dataset: str, outdir: str, protocol: str) -> list[str]:
    return [
        sys.executable, "-u", "-m",
        "experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16.train_gpu_cached_clean",
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


def stream_worker(proc: subprocess.Popen, prefix: str, log_path: Path, lock: threading.Lock) -> None:
    assert proc.stdout is not None
    with log_path.open("w", buffering=1) as f:
        for line in proc.stdout:
            clean = line.rstrip("\r\n")
            f.write(clean + "\n")
            with lock:
                print(f"[{prefix}] {clean}", flush=True)


def terminate(processes: list[tuple[str, subprocess.Popen]]) -> None:
    for _, p in processes:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + 10
    while time.time() < deadline and any(p.poll() is None for _, p in processes):
        time.sleep(0.2)
    for _, p in processes:
        if p.poll() is None:
            p.kill()


def main(dataset: str | None = None, outdir: str | None = None, cache_root: str | None = None) -> int:
    dataset = dataset or DEFAULT_DATASET
    outdir = outdir or DEFAULT_OUTDIR
    cache_root = cache_root or DEFAULT_CACHE_ROOT

    if not Path(dataset).is_file():
        raise FileNotFoundError(dataset)

    gpus = visible_gpus()
    print("=" * 120, flush=True)
    print("NESTSAR BI-JOINT M4/G4 — TRUE CONCURRENT DUAL T4 / EXACT MMAP CACHE", flush=True)
    print("=" * 120, flush=True)
    for line in gpus:
        print("GPU:", line, flush=True)
    if len(gpus) != EXPECTED_GPUS:
        raise RuntimeError(f"Expected exactly 2 GPUs, found {len(gpus)}")

    probe_gpu(0, cache_root)
    probe_gpu(1, cache_root)

    # Same architecture/FLOP preflight as the clean ablation.
    print("=" * 120, flush=True)
    print("BIJOINT PARAM/FLOP PREFLIGHT ON GPU0", flush=True)
    print("=" * 120, flush=True)
    pre = subprocess.run(
        [sys.executable, "-u", "-m",
         "experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16.preflight_gpu", dataset],
        env=gpu_env(0, cache_root), check=False,
    )
    if pre.returncode != 0:
        raise RuntimeError(f"Preflight failed rc={pre.returncode}")

    # Cache building is sequential and CPU-only; training is concurrent.
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    build_cache(dataset, cache_root, "xsub")
    build_cache(dataset, cache_root, "xset")

    out = Path(outdir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Output directory is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "M4LocalGlobalBiJointT16_ConcurrentDualT4",
        "expected_params": EXPECTED_PARAMS,
        "dataset": dataset,
        "cache_root": cache_root,
        "cache_dtype": "float32",
        "cache_mode": "numpy_mmap_read_only",
        "exact_no_quantization": True,
        "gpu_assignment": {"xsub": 0, "xset": 1},
        "protocol_execution": "concurrent",
        "effective_batch": 256,
        "microbatch": 64,
        "gradient_accumulation_steps": 4,
        "attention": False,
        "training_from_scratch": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    jobs = [("XSUB/GPU0", 0, "xsub"), ("XSET/GPU1", 1, "xset")]
    processes: list[tuple[str, subprocess.Popen]] = []
    threads: list[threading.Thread] = []
    lock = threading.Lock()

    print("=" * 120, flush=True)
    print("STARTING BOTH TRAINING PROTOCOLS AT THE SAME TIME", flush=True)
    print("XSUB -> physical GPU0 | XSET -> physical GPU1", flush=True)
    print("HOST DATA -> exact float32 mmap caches", flush=True)
    print("=" * 120, flush=True)

    for prefix, gpu, protocol in jobs:
        cmd = worker_command(dataset, outdir, protocol)
        print(f"LAUNCH {prefix}", flush=True)
        proc = subprocess.Popen(
            cmd, env=gpu_env(gpu, cache_root), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        processes.append((prefix, proc))
        th = threading.Thread(
            target=stream_worker,
            args=(proc, prefix, logs / f"{protocol}.log", lock), daemon=True,
        )
        th.start()
        threads.append(th)

    start = time.time()
    last = 0.0
    try:
        while True:
            alive = [(name, p) for name, p in processes if p.poll() is None]
            now = time.time()
            if now - last >= 30:
                status = " | ".join(
                    f"{name}={'RUNNING' if p.poll() is None else 'DONE rc='+str(p.returncode)}"
                    for name, p in processes
                )
                print(f"[DUAL-T4] elapsed={now-start:.0f}s | {status}", flush=True)
                last = now
            if not alive:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        terminate(processes)
        return 130

    for th in threads:
        th.join(timeout=5)

    failures = [(name, p.returncode) for name, p in processes if p.returncode != 0]
    if failures:
        raise RuntimeError(f"Concurrent worker failure(s): {failures}; logs={logs}")

    results = {}
    for protocol in ("xsub", "xset"):
        path = out / f"result_{protocol}.json"
        if not path.is_file():
            raise RuntimeError(f"Missing result file: {path}")
        results[protocol] = json.loads(path.read_text())
    (out / "summary.json").write_text(json.dumps(results, indent=2))

    print("=" * 120, flush=True)
    print("CONCURRENT DUAL-T4 BIJOINT DONE", flush=True)
    for protocol in ("xsub", "xset"):
        r = results[protocol]
        print(
            f"{protocol.upper()} best={100*r['best_val_accuracy']:.6f}% @ E{r['best_epoch']} "
            f"delta={r['delta_vs_baseline_pp']:+.4f} pp",
            flush=True,
        )
    print("=" * 120, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""True concurrent Dual-T4 launcher using shared exact float32 canonical cache."""

import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

EXPECTED_GPUS = 2
EXPECTED_PARAMS = 1_834_946

DEFAULT_DATASET = (
    "/kaggle/input/models/paolamaydana/"
    "ntudanno/other/default/1/ntu120_3danno.pkl"
)
DEFAULT_OUTDIR = (
    "/kaggle/working/"
    "NestSAR_M4_LocalGlobal_BiJoint_T16_ConcurrentSharedCache"
)
DEFAULT_CACHE = (
    "/kaggle/working/"
    "NestSAR_BiJoint_SharedExactFloat32_Cache"
)


def gpu_env(index: int, cache_root: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(index)
    env["JAX_PLATFORMS"] = "cuda"
    env["PYTHONUNBUFFERED"] = "1"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["NESTSAR_MICROBATCH"] = "64"
    env["NESTSAR_EVAL_MICROBATCH"] = "256"
    env["NESTSAR_SHARED_CACHE_ROOT"] = cache_root
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    return env


def cpu_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
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
        [sys.executable, "-c", code],
        env=gpu_env(index, cache_root),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
    print("=" * 120, flush=True)
    print(f"GPU{index} JAX PROBE", flush=True)
    print("=" * 120, flush=True)
    print(combined.strip(), flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"GPU{index} probe failed")
    backend = re.search(r"^BACKEND=(.+)$", combined, re.MULTILINE)
    count = re.search(r"^COUNT=(\d+)$", combined, re.MULTILINE)
    if not backend or backend.group(1).strip() != "gpu":
        raise RuntimeError(f"GPU{index}: backend is not gpu")
    if not count or int(count.group(1)) != 1:
        raise RuntimeError(f"GPU{index}: expected one visible GPU")


def build_shared_cache(dataset: str, cache_root: str) -> None:
    cmd = [
        sys.executable, "-u", "-m",
        "experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16.cache_shared_exact_views",
        "--dataset", dataset,
        "--cache-root", cache_root,
        "--seed", "128",
        "--jitter-max-shift", "1",
    ]
    print("=" * 120, flush=True)
    print("BUILD/REUSE SHARED EXACT FLOAT32 CACHE", flush=True)
    print("COMMAND:", " ".join(cmd), flush=True)
    print("=" * 120, flush=True)
    r = subprocess.run(cmd, env=cpu_env(), check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Shared exact cache build failed rc={r.returncode}")


def worker_command(dataset: str, outdir: str, protocol: str) -> list[str]:
    return [
        sys.executable, "-u", "-m",
        "experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16.train_gpu_sharedcache_clean",
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


def _reader(prefix: str, proc: subprocess.Popen, log_path: Path, q: queue.Queue) -> None:
    with log_path.open("w", buffering=1) as logf:
        assert proc.stdout is not None
        for line in proc.stdout:
            clean = line.rstrip("\r\n")
            logf.write(clean + "\n")
            q.put((prefix, clean))
    q.put((prefix, None))


def run_both(dataset: str, outdir: str, cache_root: str, logs: Path) -> None:
    specs = [
        ("XSUB/GPU0", 0, "xsub"),
        ("XSET/GPU1", 1, "xset"),
    ]
    processes = {}
    q: queue.Queue = queue.Queue()
    threads = []

    print("=" * 120, flush=True)
    print("STARTING BOTH TRAINING PROTOCOLS AT THE SAME TIME", flush=True)
    print("XSUB -> physical GPU0 | XSET -> physical GPU1", flush=True)
    print("HOST DATA -> shared canonical exact float32 mmap + protocol jitter mmap", flush=True)
    print("=" * 120, flush=True)

    for prefix, gpu, protocol in specs:
        cmd = worker_command(dataset, outdir, protocol)
        print(f"{prefix} COMMAND: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            env=gpu_env(gpu, cache_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes[prefix] = proc
        th = threading.Thread(
            target=_reader,
            args=(prefix, proc, logs / f"{protocol}.log", q),
            daemon=True,
        )
        th.start(); threads.append(th)

    done = set()
    while len(done) < len(specs):
        prefix, line = q.get()
        if line is None:
            done.add(prefix)
        else:
            print(f"[{prefix}] {line}", flush=True)

    for th in threads:
        th.join()

    failures = []
    for prefix, proc in processes.items():
        rc = proc.wait()
        if rc != 0:
            failures.append((prefix, rc))
    if failures:
        raise RuntimeError(f"Concurrent worker failure(s): {failures}; logs={logs}")


def main(dataset: str | None = None, outdir: str | None = None, cache_root: str | None = None) -> int:
    dataset = dataset or (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET)
    outdir = outdir or (sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTDIR)
    cache_root = cache_root or (sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CACHE)

    if not Path(dataset).is_file():
        raise FileNotFoundError(dataset)

    gpus = visible_gpus()
    print("=" * 120, flush=True)
    print("NESTSAR LOCALGLOBAL V2 + BI-JOINT M4/G4 — TRUE CONCURRENT SHARED-CACHE DUAL T4", flush=True)
    print("=" * 120, flush=True)
    for x in gpus: print("GPU:", x, flush=True)
    if len(gpus) != EXPECTED_GPUS:
        raise RuntimeError(f"Expected exactly 2 GPUs, found {len(gpus)}")

    probe_gpu(0, cache_root)
    probe_gpu(1, cache_root)

    print("=" * 120, flush=True)
    print("BIJOINT PARAM/FLOP PREFLIGHT ON GPU0", flush=True)
    print("=" * 120, flush=True)
    pf = subprocess.run(
        [sys.executable, "-u", "-m",
         "experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16.preflight_gpu",
         dataset],
        env=gpu_env(0, cache_root),
        check=False,
    )
    if pf.returncode != 0:
        raise RuntimeError(f"GPU preflight failed rc={pf.returncode}")

    build_shared_cache(dataset, cache_root)

    out = Path(outdir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Output directory is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"; logs.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "M4LocalGlobalBiJointT16_ConcurrentSharedExactCache",
        "dataset": dataset,
        "outdir": outdir,
        "cache_root": cache_root,
        "expected_params": EXPECTED_PARAMS,
        "gpu_assignment": {"xsub": 0, "xset": 1},
        "protocol_execution": "concurrent",
        "effective_batch": 256,
        "microbatch": 64,
        "gradient_accumulation_steps": 4,
        "preprocessing_storage": "shared_canonical_exact_float32_plus_protocol_jitter_mmap",
        "attention": False,
        "training_from_scratch": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    run_both(dataset, outdir, cache_root, logs)

    results = {}
    for protocol in ("xsub", "xset"):
        p = out / f"result_{protocol}.json"
        if not p.is_file():
            raise RuntimeError(f"Missing result file: {p}")
        results[protocol] = json.loads(p.read_text())
    (out / "summary.json").write_text(json.dumps(results, indent=2))

    print("=" * 120, flush=True)
    print("CONCURRENT SHARED-CACHE BIJOINT RUN DONE", flush=True)
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

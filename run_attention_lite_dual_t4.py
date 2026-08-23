#!/usr/bin/env python3
"""Train Attention-Lite XSUB and XSET concurrently on Kaggle dual T4.

Topology:

    OS process A -> CUDA_VISIBLE_DEVICES=0 -> XSUB -> one T4
    OS process B -> CUDA_VISIBLE_DEVICES=1 -> XSET -> one T4

Unlike the failed TPU 4+4 experiment, CUDA device visibility remaps each worker's
assigned physical GPU to a process-local device 0 before JAX initializes. The two
protocols therefore have independent CUDA/JAX runtimes and no cross-device
collectives.

The launcher is also deliberately aware of Kaggle's ~30 GB host-RAM budget:

- the parent never unpickles NTU120;
- canonical sources are materialized once before workers start;
- each worker receives MALLOC_ARENA_MAX=2 and no JAX VRAM preallocation;
- worker startup is staggered slightly to avoid simultaneous pickle/JIT peaks;
- host MemAvailable is monitored and both workers are stopped before a severe
  system-RAM exhaustion event.

The dataset implementation keeps split lists as references to annotations and
preprocesses samples lazily per batch, so there is no extra full preprocessed
[T,N,...] copy created by this launcher.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from experiments.attention_lite_v1.canonical_integrated import ensure_canonical_sources
from experiments.attention_lite_v1.paths import validate_seed

RUNNER_API_VERSION = "attention-lite-dual-t4-v1-memory-aware-30gb"
REPO = Path(__file__).resolve().parent
GIB = 1024 ** 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train XSUB and XSET concurrently, one protocol per Kaggle T4"
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--dataset", default="auto")
    p.add_argument("--runs-root", default="/kaggle/working")
    p.add_argument("--run-tag", default="paper_dual_t4_p12")
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--non-paper", action="store_true")

    p.add_argument("--dropout", type=float, default=0.22)
    p.add_argument("--learning-rate", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-fraction", type=float, default=0.10)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--predictive-loss-weight", type=float, default=0.10)
    p.add_argument("--initial-eta", type=float, default=0.02)
    p.add_argument("--initial-alpha", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=32)

    # 30-GB host-RAM safety controls.
    p.add_argument("--startup-stagger-seconds", type=float, default=15.0)
    p.add_argument("--min-start-ram-gb", type=float, default=20.0)
    p.add_argument("--critical-free-ram-gb", type=float, default=1.5)
    p.add_argument("--ram-poll-seconds", type=float, default=5.0)
    return p.parse_args()


def _read_meminfo() -> tuple[float, float]:
    """Return (total_GiB, available_GiB) from /proc/meminfo."""
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, rest = line.split(":", 1)
            raw = rest.strip().split()[0]
            values[key] = int(raw) * 1024
    except Exception:
        return 0.0, 0.0

    total = values.get("MemTotal", 0) / GIB
    available = values.get("MemAvailable", 0) / GIB
    return total, available


def _ram_preflight(args: argparse.Namespace) -> None:
    total, available = _read_meminfo()
    print("=" * 108, flush=True)
    print("30-GB HOST-RAM PREFLIGHT", flush=True)
    print(f"MemTotal:     {total:.2f} GiB", flush=True)
    print(f"MemAvailable: {available:.2f} GiB", flush=True)
    print(
        "Design: parent keeps no dataset copy; XSUB/XSET each unpickle independently; "
        f"startup stagger={args.startup_stagger_seconds:.1f}s",
        flush=True,
    )
    print("=" * 108, flush=True)

    if total and total < 26.0:
        raise RuntimeError(
            f"Dual-T4 30-GB mode expects roughly >=26 GiB host RAM; found {total:.2f} GiB"
        )
    if available and available < args.min_start_ram_gb:
        raise RuntimeError(
            f"Not enough free/reclaimable host RAM to start two workers: "
            f"{available:.2f} GiB < {args.min_start_ram_gb:.2f} GiB"
        )


def _gpu_inventory() -> list[str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "nvidia-smi failed; Kaggle dual-T4 accelerator is not available.\n"
            + completed.stdout
        )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) < 2:
        raise RuntimeError(
            "Expected two Kaggle GPUs, but nvidia-smi reported:\n" + completed.stdout
        )
    return rows


def _worker_env(protocol: str, gpu_index: int) -> dict[str, str]:
    env = os.environ.copy()

    # Remove TPU/distributed settings from previous experiments.
    for name in (
        "TPU_VISIBLE_CHIPS",
        "TPU_VISIBLE_DEVICES",
        "TPU_CHIPS_PER_PROCESS_BOUNDS",
        "TPU_PROCESS_BOUNDS",
        "TPU_MESH_CONTROLLER_ADDRESS",
        "TPU_MESH_CONTROLLER_PORT",
        "TPU_PROCESS_ADDRESSES",
        "TPU_PROCESS_PORT",
        "CLOUD_TPU_TASK_ID",
        "JAX_COORDINATOR_ADDRESS",
        "JAX_PROCESS_ID",
        "JAX_NUM_PROCESSES",
        "JAX_PLATFORM_NAME",
    ):
        env.pop(name, None)

    # CRITICAL: set GPU visibility before JAX is imported. Each worker then sees
    # one local GPU with id 0 even though the physical assignments are 0 and 1.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["NESTSAR_PHYSICAL_GPU"] = str(gpu_index)
    env["NESTSAR_PROTOCOL"] = protocol

    # Keep GPU and host memory behavior conservative on two concurrent workers.
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    env["MALLOC_ARENA_MAX"] = "2"
    env["MALLOC_TRIM_THRESHOLD_"] = "131072"
    env["OMP_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env["PYTHONUNBUFFERED"] = "1"

    # Do not hard-code JAX_PLATFORMS here: Kaggle JAX builds can expose the CUDA
    # backend under slightly different plugin names. The probe below requires
    # jax.default_backend() == 'gpu', so a CPU fallback cannot slip through.
    env.pop("JAX_PLATFORMS", None)
    return env


def _training_env(
    protocol: str,
    gpu_index: int,
    args: argparse.Namespace,
    canonical_source: Path,
) -> dict[str, str]:
    env = _worker_env(protocol, gpu_index)
    env.update(
        {
            "NESTSAR_PROTOCOL": protocol,
            "NESTSAR_EPOCHS": str(args.epochs),
            "NESTSAR_PATIENCE": str(args.patience),
            "NESTSAR_SEED": str(args.seed),
            "NESTSAR_DATASET": str(args.dataset),
            "NESTSAR_RUNS_ROOT": str(args.runs_root),
            "NESTSAR_RUN_TAG": str(args.run_tag),
            "NESTSAR_PAPER_MODE": "0" if args.non_paper else "1",
            "NESTSAR_CANONICAL_SOURCE": str(canonical_source),
            "NESTSAR_DROPOUT": str(args.dropout),
            "NESTSAR_LEARNING_RATE": str(args.learning_rate),
            "NESTSAR_WEIGHT_DECAY": str(args.weight_decay),
            "NESTSAR_WARMUP_FRACTION": str(args.warmup_fraction),
            "NESTSAR_LABEL_SMOOTHING": str(args.label_smoothing),
            "NESTSAR_GRAD_CLIP": str(args.grad_clip),
            "NESTSAR_PREDICTIVE_LOSS_WEIGHT": str(args.predictive_loss_weight),
            "NESTSAR_INITIAL_ETA": str(args.initial_eta),
            "NESTSAR_INITIAL_ALPHA": str(args.initial_alpha),
            "NESTSAR_BATCH_SIZE": str(args.batch_size),
            "NESTSAR_GRAD_ACCUM_STEPS": str(args.grad_accum_steps),
            "NESTSAR_EVAL_BATCH_SIZE": str(args.eval_batch_size),
        }
    )
    return env


def _probe_gpu(protocol: str, gpu_index: int) -> tuple[int, str]:
    probe = r'''
import jax
import jax.numpy as jnp

backend = jax.default_backend()
devices = list(jax.devices())
print("JAX:", jax.__version__, flush=True)
print("Backend:", backend, flush=True)
print("Visible devices:", devices, flush=True)
if backend != "gpu":
    raise RuntimeError(f"Expected GPU backend, got {backend!r}")
if len(devices) != 1:
    raise RuntimeError(f"Expected exactly one process-visible GPU, found {len(devices)}")

# Force a real CUDA compilation/execution, not just device enumeration.
x = jnp.arange(1024 * 1024, dtype=jnp.float32).reshape(1024, 1024) / 1024.0
y = jnp.tanh(x @ x.T)
y.block_until_ready()
print("CUDA COMPUTE PROBE: PASS", float(y[0, 0]), flush=True)
'''
    completed = subprocess.run(
        [sys.executable, "-u", "-c", probe],
        cwd=str(REPO),
        env=_worker_env(protocol, gpu_index),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=180,
    )
    return completed.returncode, completed.stdout


def _parallel_gpu_probe() -> None:
    results: dict[str, tuple[int, str]] = {}
    mapping = {"xsub": 0, "xset": 1}

    def worker(protocol: str) -> None:
        try:
            results[protocol] = _probe_gpu(protocol, mapping[protocol])
        except BaseException as exc:
            results[protocol] = (
                99,
                f"GPU probe launcher exception: {type(exc).__name__}: {exc}\n",
            )

    threads = [
        threading.Thread(target=worker, args=("xsub",), daemon=True),
        threading.Thread(target=worker, args=("xset",), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failures: list[str] = []
    for protocol in ("xsub", "xset"):
        code, output = results.get(protocol, (98, "Probe produced no result\n"))
        print("\n" + "=" * 108, flush=True)
        print(
            f"{protocol.upper()} | physical T4 {mapping[protocol]} | "
            f"GPU PROBE | returncode={code}",
            flush=True,
        )
        print("=" * 108, flush=True)
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        if code != 0 or "CUDA COMPUTE PROBE: PASS" not in output:
            failures.append(f"{protocol.upper()} GPU{mapping[protocol]} probe failed")

    if failures:
        raise RuntimeError(
            "Dual-T4 GPU preflight failed. Training was NOT started.\n"
            + "\n".join(failures)
        )

    print("=" * 108, flush=True)
    print("BOTH ISOLATED T4 CUDA PROBES: PASS", flush=True)
    print("XSUB -> physical GPU0 | XSET -> physical GPU1", flush=True)
    print("=" * 108, flush=True)


def _tee_output(proc: subprocess.Popen, protocol: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"[{protocol.upper()}] "
    with log_path.open("w", encoding="utf-8") as log:
        assert proc.stdout is not None
        for line in proc.stdout:
            text = prefix + line
            print(text, end="", flush=True)
            log.write(text)
            log.flush()


def _terminate(proc: subprocess.Popen, label: str) -> None:
    if proc.poll() is not None:
        return
    print(f"TERMINATING {label}...", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _run_training_pair(
    args: argparse.Namespace,
    sources: dict[str, Path],
) -> int:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.attention_lite_v1.trainer_t4",
    ]

    mapping = {"xsub": 0, "xset": 1}
    processes: dict[str, subprocess.Popen] = {}
    readers: list[threading.Thread] = []

    for index, protocol in enumerate(("xsub", "xset")):
        if index and args.startup_stagger_seconds > 0:
            print(
                f"30-GB RAM safeguard: waiting {args.startup_stagger_seconds:.1f}s "
                "before starting the second worker...",
                flush=True,
            )
            time.sleep(args.startup_stagger_seconds)

        gpu_index = mapping[protocol]
        env = _training_env(protocol, gpu_index, args, sources[protocol])
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes[protocol] = proc

        reader = threading.Thread(
            target=_tee_output,
            args=(
                proc,
                protocol,
                Path(args.runs_root)
                / f"attention_lite_{protocol}_dual_t4_parent.log",
            ),
            daemon=True,
        )
        reader.start()
        readers.append(reader)

    print("=" * 108, flush=True)
    print("TRUE DUAL-T4 PARALLEL TRAINING STARTED", flush=True)
    print("XSUB -> physical T4 GPU0 -> process-local GPU0", flush=True)
    print("XSET -> physical T4 GPU1 -> process-local GPU0", flush=True)
    print(f"Global batch/protocol: {args.batch_size}", flush=True)
    print(f"Grad accumulation:     {args.grad_accum_steps}", flush=True)
    print(
        f"Effective batch:       {args.batch_size * args.grad_accum_steps}",
        flush=True,
    )
    total, available = _read_meminfo()
    print(f"Host RAM at launch:    {available:.2f}/{total:.2f} GiB available", flush=True)
    print("=" * 108, flush=True)

    failure: tuple[str, int] | None = None
    low_ram_checks = 0
    last_ram_report = 0.0

    while True:
        statuses = {name: proc.poll() for name, proc in processes.items()}

        for name, code in statuses.items():
            if code is not None and code != 0:
                failure = (name, code)
                other = "xset" if name == "xsub" else "xsub"
                _terminate(processes[other], other.upper())
                break
        if failure is not None:
            break

        if all(code is not None for code in statuses.values()):
            break

        now = time.time()
        if now - last_ram_report >= max(1.0, args.ram_poll_seconds):
            total, available = _read_meminfo()
            if total:
                print(
                    f"[RAM] available={available:.2f} GiB / {total:.2f} GiB",
                    flush=True,
                )
                if available < args.critical_free_ram_gb:
                    low_ram_checks += 1
                else:
                    low_ram_checks = 0

                # Require three consecutive low-memory samples to avoid reacting to
                # a short transient allocation/JIT spike.
                if low_ram_checks >= 3:
                    for name, proc in processes.items():
                        _terminate(proc, name.upper())
                    raise RuntimeError(
                        "Stopped both dual-T4 workers before host OOM: "
                        f"MemAvailable stayed below {args.critical_free_ram_gb:.2f} GiB "
                        "for three consecutive checks."
                    )
            last_ram_report = now

        time.sleep(1.0)

    for proc in processes.values():
        if proc.poll() is None:
            proc.wait()
    for reader in readers:
        reader.join(timeout=30)

    final = {name: proc.returncode for name, proc in processes.items()}
    if failure is not None or any(code != 0 for code in final.values()):
        raise RuntimeError(
            "Dual-T4 parallel Attention-Lite failed.\n"
            f"Return codes: {final}\n"
            "The sibling worker was stopped on first failure; read the prefixed "
            "traceback above and the dual_t4_parent.log files."
        )

    print("=" * 108, flush=True)
    print("XSUB + XSET DUAL-T4 PARALLEL TRAINING COMPLETE", flush=True)
    print(f"Return codes: {final}", flush=True)
    print("=" * 108, flush=True)
    return 0


def main() -> int:
    args = parse_args()
    validate_seed(args.seed, paper_mode=not args.non_paper)

    if args.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if args.patience < 0:
        raise ValueError("patience must be >= 0")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch sizes must be > 0")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad-accum-steps must be > 0")
    if args.startup_stagger_seconds < 0:
        raise ValueError("startup-stagger-seconds must be >= 0")
    if args.min_start_ram_gb < 0 or args.critical_free_ram_gb < 0:
        raise ValueError("RAM thresholds must be >= 0")

    print("=" * 108, flush=True)
    print(f"RUNNER API: {RUNNER_API_VERSION}", flush=True)
    print("Parent process intentionally does NOT import JAX or unpickle NTU120.", flush=True)
    print("XSUB -> GPU0 | XSET -> GPU1 | independent CUDA/JAX processes", flush=True)
    print("=" * 108, flush=True)

    _ram_preflight(args)

    inventory = _gpu_inventory()
    print("DUAL-T4 INVENTORY", flush=True)
    for row in inventory:
        print("  " + row, flush=True)

    # Reconstruct/verify canonical source once in the lightweight parent before
    # either worker starts. This avoids concurrent source-materialization writes.
    sources_raw = ensure_canonical_sources(verbose=True)
    sources = {name: Path(path).resolve() for name, path in sources_raw.items()}

    _parallel_gpu_probe()
    if args.probe_only:
        print("PROBE-ONLY requested — no training started.", flush=True)
        return 0

    # Let the short probe CUDA contexts release before real workers start.
    time.sleep(3.0)
    return _run_training_pair(args, sources)


if __name__ == "__main__":
    raise SystemExit(main())

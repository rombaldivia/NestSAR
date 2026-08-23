#!/usr/bin/env python3
"""True parallel Attention-Lite XSUB+XSET using two isolated TPU4 processes.

This replaces the one-JAX-runtime 4+4 submesh experiment.  On Kaggle TPU v5e-8,
XLA collective lowering for the second in-process submesh used global physical IDs
4..7 against a four-device compilation target and failed with:

    Unexpected device_id 4 ... target has 4 device_id

The safe design is process isolation: each protocol owns a non-overlapping 2x2
four-chip slice before JAX initializes.  Each process therefore builds its own
four-device runtime and collective replica group.

Before expensive model compilation, this launcher runs BOTH four-chip collective
probes concurrently.  Training begins only if both probes successfully execute an
actual psum all-reduce.  If either training process fails, the other is terminated
so TPU time is not wasted.
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

RUNNER_API_VERSION = "attention-lite-parallel-v4-two-isolated-tpu4-processes"
REPO = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train XSUB and XSET concurrently in isolated TPU4 processes"
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--dataset", default="auto")
    p.add_argument("--runs-root", default="/kaggle/working")
    p.add_argument("--run-tag", default="parallel_proc4x4_v4")
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
    return p.parse_args()


def _partition_env(protocol: str) -> dict[str, str]:
    """Return a JAX/libtpu environment for one independent 2x2 TPU slice."""
    p = protocol.lower()
    if p == "xsub":
        visible = "0,1,2,3"
        controller_port = 8476
    elif p == "xset":
        visible = "4,5,6,7"
        controller_port = 8477
    else:
        raise ValueError(protocol)

    env = os.environ.copy()

    # Clear distributed-process settings that could make these two independent
    # jobs accidentally join one JAX/libtpu process group.
    for name in (
        "TPU_PROCESS_ADDRESSES",
        "TPU_PROCESS_PORT",
        "CLOUD_TPU_TASK_ID",
        "JAX_COORDINATOR_ADDRESS",
        "JAX_PROCESS_ID",
        "JAX_NUM_PROCESSES",
    ):
        env.pop(name, None)

    # One independent process owns one contiguous 2x2 block of v5e chips.
    env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "2,2,1"
    env["TPU_PROCESS_BOUNDS"] = "1,1,1"

    # libtpu/JAX releases have used both spellings.  Keeping the two equal makes
    # the intent explicit while the collective probe verifies the actual behavior.
    env["TPU_VISIBLE_DEVICES"] = visible
    env["TPU_VISIBLE_CHIPS"] = visible

    # Unique controller endpoint per non-communicating process.
    env["TPU_MESH_CONTROLLER_ADDRESS"] = f"localhost:{controller_port}"
    env["TPU_MESH_CONTROLLER_PORT"] = str(controller_port)

    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["MALLOC_ARENA_MAX"] = "2"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env["PYTHONUNBUFFERED"] = "1"
    env["NESTSAR_PROTOCOL"] = p
    return env


def _training_env(
    protocol: str,
    args: argparse.Namespace,
    canonical_source: Path,
) -> dict[str, str]:
    env = _partition_env(protocol)
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


def _run_probe(protocol: str) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.attention_lite_v1.tpu4_collective_probe",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=_partition_env(protocol),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=180,
    )
    return proc.returncode, proc.stdout


def _parallel_probe() -> None:
    results: dict[str, tuple[int, str]] = {}

    def worker(protocol: str) -> None:
        try:
            results[protocol] = _run_probe(protocol)
        except BaseException as exc:
            results[protocol] = (99, f"Probe launcher exception: {type(exc).__name__}: {exc}\n")

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
        print(f"{protocol.upper()} TPU4 PROBE OUTPUT | returncode={code}", flush=True)
        print("=" * 108, flush=True)
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
        if code != 0 or "TPU4 COLLECTIVE PROBE: PASS" not in output:
            failures.append(f"{protocol.upper()} probe failed (returncode={code})")

    if failures:
        raise RuntimeError(
            "TPU4+TPU4 process isolation preflight failed. Training was NOT started.\n"
            + "\n".join(failures)
        )

    print("=" * 108, flush=True)
    print("BOTH ISOLATED TPU4 COLLECTIVE PROBES: PASS", flush=True)
    print("The invalid global device-id 4 collective path was not reproduced.", flush=True)
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
    print(f"TERMINATING {label} because the sibling protocol failed.", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _run_training_pair(args: argparse.Namespace, sources: dict[str, Path]) -> int:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.attention_lite_v1.trainer_tpu4",
    ]

    processes: dict[str, subprocess.Popen] = {}
    readers: list[threading.Thread] = []

    for protocol in ("xsub", "xset"):
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            env=_training_env(protocol, args, sources[protocol]),
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
                Path(args.runs_root) / f"attention_lite_{protocol}_parallel_v4_parent.log",
            ),
            daemon=True,
        )
        reader.start()
        readers.append(reader)

    print("=" * 108, flush=True)
    print("TRUE PROCESS-ISOLATED PARALLEL TRAINING STARTED", flush=True)
    print("XSUB process -> physical chips 0,1,2,3", flush=True)
    print("XSET process -> physical chips 4,5,6,7", flush=True)
    print("Each process must see exactly four devices and owns its own collectives.", flush=True)
    print("=" * 108, flush=True)

    failure: tuple[str, int] | None = None
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
        time.sleep(1.0)

    # Ensure both processes have exited and all streamed output is drained.
    for proc in processes.values():
        if proc.poll() is None:
            proc.wait()
    for reader in readers:
        reader.join(timeout=30)

    final = {name: proc.returncode for name, proc in processes.items()}
    if failure is not None or any(code != 0 for code in final.values()):
        raise RuntimeError(
            "Process-isolated parallel Attention-Lite failed.\n"
            f"Return codes: {final}\n"
            "The sibling process was stopped on first failure; read the prefixed traceback above."
        )

    print("=" * 108, flush=True)
    print("XSUB + XSET PROCESS-ISOLATED PARALLEL TRAINING COMPLETE", flush=True)
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
    if args.batch_size <= 0 or args.batch_size % 4:
        raise ValueError("batch-size must be positive and divisible by 4")
    if args.eval_batch_size <= 0 or args.eval_batch_size % 4:
        raise ValueError("eval-batch-size must be positive and divisible by 4")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad-accum-steps must be > 0")

    print("=" * 108, flush=True)
    print(f"RUNNER API: {RUNNER_API_VERSION}", flush=True)
    print("No JAX runtime is initialized in the parent process.", flush=True)
    print("XSUB -> isolated TPU4 process: chips 0,1,2,3", flush=True)
    print("XSET -> isolated TPU4 process: chips 4,5,6,7", flush=True)
    print("=" * 108, flush=True)

    sources_raw = ensure_canonical_sources(verbose=True)
    sources = {name: Path(path).resolve() for name, path in sources_raw.items()}

    # Critical gate: if the Kaggle/libtpu environment does not support two isolated
    # 4-chip processes, stop here instead of discovering it after model compilation.
    _parallel_probe()
    if args.probe_only:
        print("PROBE-ONLY requested — no training started.", flush=True)
        return 0

    # Give libtpu a short moment to release probe processes' runtime resources.
    time.sleep(3.0)
    return _run_training_pair(args, sources)


if __name__ == "__main__":
    raise SystemExit(main())

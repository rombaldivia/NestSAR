#!/usr/bin/env python3
from __future__ import annotations

"""Dual-T4 launcher for NestSAR-FULL-PARALLEL-T16-D192-MTS-v1."""

import fcntl
import subprocess
import sys
import time
from pathlib import Path

from experiments.nestsar_sm_all_t16.streaming import launch as base
from experiments.nestsar_sm_all_t16.parallel_d192_config import (
    MODEL_NAME,
    EXPECTED_PARAMS,
    MODEL_DIM,
    implementation_identity,
    validate_d192_config,
)
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json, read_json


WORKER_MODULE = "experiments.nestsar_sm_all_t16.streaming.worker_parallel_d192"
AUDIT_MODULE = "experiments.nestsar_sm_all_t16.audit_parallel_d192"
DATA_MODULE = "experiments.nestsar_sm_all_t16.streaming.data"


def run(
    dataset=None,
    outdir="/kaggle/working/NestSAR_FULL_PARALLEL_D192_MTS_T16_v1",
    cache_dir="/kaggle/working/NestSAR_FULL_PARALLEL_CACHE_v1",
    config=None,
    raw_layout="MTVC",
    audit_first=True,
):
    requested = dict(config or {})
    requested["model_dim"] = MODEL_DIM
    c = validate_d192_config(requested)

    dataset = base.find_dataset(dataset)
    out = Path(outdir)
    cache = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)

    lock = (out / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(
            "This OUT_DIR already has an active launcher. Stop its cell before restarting."
        )

    bars = []
    processes = []
    streams = []

    try:
        identity = {
            "model": MODEL_NAME,
            "parameters": EXPECTED_PARAMS,
            "implementation": implementation_identity(),
            "config": c,
        }
        identity_path = out / "model_config.json"
        previous_identity = read_json(identity_path)
        if previous_identity is not None and previous_identity != identity:
            raise ValueError(
                "OUT_DIR belongs to another implementation/config; use a fresh OUT_DIR"
            )
        if previous_identity is None and any(
            (out / p / "run_config.json").exists() for p in ("xsub", "xset")
        ):
            raise ValueError("Legacy output has no model identity; use a fresh OUT_DIR")
        atomic_json(identity_path, identity)

        gpus, jax_devices = base.discover_gpus(
            sys.executable, out / "gpu_discovery.log"
        )
        atomic_json(
            out / "hardware.json",
            {
                "gpu_discovery": "jax_subprocess",
                "jax_devices": jax_devices,
                "nvidia_smi": base.optional_nvidia_smi(),
                "assignment": dict(zip(("xsub", "xset"), gpus)),
                "model": MODEL_NAME,
            },
        )

        bars = base.make_bars()
        python = base.ensure_runtime(out, bars, gpus[0])

        cfg_path = out / "config.json"
        old = read_json(cfg_path)
        if old is not None and old != c:
            raise ValueError(
                "OUT_DIR has a different config. Use a new OUT_DIR for this experiment."
            )
        atomic_json(cfg_path, c)

        source = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[2]),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
        )
        atomic_json(
            out / "source.json",
            {
                "commit": source.stdout.strip() if source.returncode == 0 else None,
                "model": MODEL_NAME,
                "parameters": EXPECTED_PARAMS,
                "implementation": implementation_identity(),
            },
        )

        for i, gpu in enumerate(gpus):
            base.quiet_run(
                [
                    python,
                    "-c",
                    (
                        "import jax; "
                        "assert jax.default_backend()=='gpu' "
                        "and jax.local_device_count()==1; "
                        "print(jax.devices())"
                    ),
                ],
                out / f"gpu{i}_probe.log",
                base.isolated_env(gpu),
                lambda i=i: base.update_bar(
                    bars[i],
                    ("xsub", "xset")[i],
                    i,
                    dict(phase="GPU probe", current=0, total=1),
                ),
            )

        tests = Path(__file__).resolve().parent / "test_preprocessing_corrected.py"
        base.quiet_run(
            [python, "-m", "pytest", "-q", str(tests)],
            out / "preprocessing_tests.log",
            base.isolated_env(),
            lambda: base.update_bar(
                bars[0],
                "xsub",
                0,
                dict(phase="Regression checks", current=0, total=1),
            ),
        )

        status_path = out / "prepare_status.json"
        base.quiet_run(
            [
                python,
                "-m",
                DATA_MODULE,
                "--dataset",
                str(dataset),
                "--cache",
                str(cache),
                "--status",
                str(status_path),
                "--layout",
                raw_layout,
            ],
            out / "prepare.log",
            base.isolated_env(),
            lambda: [
                base.update_bar(
                    bar,
                    protocol,
                    i,
                    read_json(
                        status_path,
                        dict(phase="Prepare cache", current=0, total=1),
                    ),
                )
                for i, (bar, protocol) in enumerate(zip(bars, ("xsub", "xset")))
            ],
        )

        if audit_first:
            base.quiet_run(
                [
                    python,
                    "-u",
                    "-m",
                    AUDIT_MODULE,
                    "--output",
                    str(out / "d192_audit.json"),
                ],
                out / "d192_audit.log",
                base.isolated_env(gpus[0]),
                lambda: base.update_bar(
                    bars[0],
                    "xsub",
                    0,
                    dict(phase="D192 hard audit", current=0, total=1),
                ),
            )

        for protocol, gpu in zip(("xsub", "xset"), gpus):
            child_out = out / protocol
            child_out.mkdir(exist_ok=True)

            atomic_json(
                child_out / "status.json",
                dict(phase="Starting", current=0, total=1),
            )

            stream = (out / f"{protocol}.log").open("a", buffering=1)
            streams.append(stream)

            cmd = [
                python,
                "-u",
                "-m",
                WORKER_MODULE,
                "--config",
                str(cfg_path),
                "--protocol",
                protocol,
                "--cache",
                str(cache),
                "--outdir",
                str(out),
            ]

            processes.append(
                subprocess.Popen(
                    cmd,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=base.isolated_env(gpu),
                    start_new_session=True,
                )
            )

        last_scores = None
        while True:
            done = 0
            statuses = {}

            for i, (proc, protocol) in enumerate(
                zip(processes, ("xsub", "xset"))
            ):
                status = read_json(out / protocol / "status.json", {})
                finished, status = base.check_worker_finished(
                    proc, protocol, out, status
                )
                done += int(finished)
                statuses[protocol] = status
                base.update_bar(bars[i], protocol, i, status)

            scores = base.best_score_snapshot(statuses)
            if scores != last_scores:
                atomic_json(out / "best_scores.json", scores)
                last_scores = scores

            if done == 2:
                break
            time.sleep(0.5)

        results = {
            protocol: read_json(out / protocol / "result.json")
            for protocol in ("xsub", "xset")
        }
        atomic_json(out / "results.json", results)
        return results

    finally:
        base.stop_processes(processes)
        for stream in streams:
            stream.close()
        for bar in bars:
            bar.close()
        lock.close()


if __name__ == "__main__":
    print("Import run() from this module in Kaggle.")

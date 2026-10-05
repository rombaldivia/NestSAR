#!/usr/bin/env python3
from __future__ import annotations

"""Dual-T4 launcher for D128 from-zero prototype-margin training.

The launcher deliberately reuses the proven streaming/TQDM infrastructure:
GPU0 -> XSUB, GPU1 -> XSET.  The child workers own exact-resume state in
last.msgpack, including optimizer/EMA/RNG/prototype-bank state.
"""

import fcntl
import subprocess
import sys
import time
from pathlib import Path

from experiments.nestsar_sm_all_t16.parallel_d128_config import (
    EXPECTED_PARAMS,
    MODEL_DIM,
    validate_d128_config,
)
from experiments.nestsar_sm_all_t16.streaming import launch as base
from experiments.nestsar_sm_all_t16.streaming.io_utils import (
    atomic_json,
    read_json,
)


MODEL_NAME = "NestSAR-FULL-PARALLEL-T16-D128-MTS-PROTOMARGIN-v1"

PROTOTYPE_DEFAULTS = {
    "prototype_margin": 0.20,
    "prototype_weight": 0.10,
    "prototype_momentum": 0.99,
    "prototype_delay_epochs": 4,
    "prototype_ramp_epochs": 10,
}

WORKER_MODULE = (
    "experiments.nestsar_sm_all_t16.streaming."
    "worker_parallel_d128_prototype"
)
DATA_MODULE = "experiments.nestsar_sm_all_t16.streaming.data"


def validate_config(config):
    config = dict(config or {})
    allowed = set(base.DEFAULTS) | set(PROTOTYPE_DEFAULTS)
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    d128_input = {
        k: v
        for k, v in config.items()
        if k in base.DEFAULTS
    }
    c = validate_d128_config(d128_input)

    p = dict(PROTOTYPE_DEFAULTS)
    p.update(
        {
            k: config[k]
            for k in PROTOTYPE_DEFAULTS
            if k in config
        }
    )

    if not 0 <= float(p["prototype_margin"]) <= 1:
        raise ValueError("prototype_margin must be in [0,1]")
    if float(p["prototype_weight"]) < 0:
        raise ValueError("prototype_weight must be nonnegative")
    if not 0 <= float(p["prototype_momentum"]) < 1:
        raise ValueError("prototype_momentum must be in [0,1)")
    for key in ("prototype_delay_epochs", "prototype_ramp_epochs"):
        if not isinstance(p[key], int) or p[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")

    c.update(p)
    return c


def run(
    dataset=None,
    outdir="/kaggle/working/NestSAR_D128_PROTOTYPE_MARGIN_SCRATCH_v1",
    cache_dir="/kaggle/working/NestSAR_R4_EMA_REP_CONSISTENCY_CACHE_REBUILT_v1",
    config=None,
    raw_layout="MTVC",
):
    c = validate_config(config)

    dataset = base.find_dataset(dataset)
    out = Path(outdir)
    cache = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)

    lock = (out / "run.lock").open("a")
    try:
        fcntl.flock(
            lock,
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError:
        lock.close()
        raise RuntimeError(
            "This OUT_DIR already has an active launcher. "
            "Stop its previous cell before restarting."
        )

    bars = []
    processes = []
    streams = []

    try:
        identity = {
            "model": MODEL_NAME,
            "parameters": EXPECTED_PARAMS,
            "model_dim": MODEL_DIM,
            "inference_architecture": (
                "NestSAR-FULL-PARALLEL-T16-D128-MTS-v1"
            ),
            "training_change": (
                "training-only EMA 120-class prototype hinge margin"
            ),
            "prototype_bank_in_inference": False,
            "config": c,
        }

        identity_path = out / "model_config.json"
        previous_identity = read_json(identity_path)

        if (
            previous_identity is not None
            and previous_identity != identity
        ):
            raise ValueError(
                "OUT_DIR belongs to another implementation/config. "
                "Use a fresh OUT_DIR."
            )

        if (
            previous_identity is None
            and any(
                (out / p / "run_config.json").exists()
                for p in ("xsub", "xset")
            )
        ):
            raise ValueError(
                "Legacy output has no prototype model identity. "
                "Use a fresh OUT_DIR."
            )

        atomic_json(
            identity_path,
            identity,
        )

        gpus, jax_devices = base.discover_gpus(
            sys.executable,
            out / "gpu_discovery.log",
        )

        atomic_json(
            out / "hardware.json",
            {
                "gpu_discovery": "jax_subprocess",
                "jax_devices": jax_devices,
                "nvidia_smi": base.optional_nvidia_smi(),
                "assignment": dict(
                    zip(
                        ("xsub", "xset"),
                        gpus,
                    )
                ),
                "model": MODEL_NAME,
            },
        )

        bars = base.make_bars()

        python = base.ensure_runtime(
            out,
            bars,
            gpus[0],
        )

        cfg_path = out / "config.json"
        old = read_json(cfg_path)

        if (
            old is not None
            and old != c
        ):
            raise ValueError(
                "OUT_DIR has a different config. "
                "Keep this cell unchanged when resuming."
            )

        atomic_json(
            cfg_path,
            c,
        )

        source = subprocess.run(
            [
                "git",
                "-C",
                str(
                    Path(__file__).resolve().parents[2]
                ),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
        )

        atomic_json(
            out / "source.json",
            {
                "commit": (
                    source.stdout.strip()
                    if source.returncode == 0
                    else None
                ),
                "model": MODEL_NAME,
                "parameters": EXPECTED_PARAMS,
                "prototype_config": {
                    k: c[k]
                    for k in PROTOTYPE_DEFAULTS
                },
            },
        )

        # Each worker must see exactly one GPU.
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
                    dict(
                        phase="GPU probe",
                        current=0,
                        total=1,
                    ),
                ),
            )

        tests = (
            Path(__file__).resolve().parent
            / "test_preprocessing_corrected.py"
        )

        base.quiet_run(
            [
                python,
                "-m",
                "pytest",
                "-q",
                str(tests),
            ],
            out / "preprocessing_tests.log",
            base.isolated_env(),
            lambda: base.update_bar(
                bars[0],
                "xsub",
                0,
                dict(
                    phase="Regression checks",
                    current=0,
                    total=1,
                ),
            ),
        )

        # Reuse the healthy canonical cache when already present; data.py
        # verifies its signature before returning.
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
                        dict(
                            phase="Prepare cache",
                            current=0,
                            total=1,
                        ),
                    ),
                )
                for i, (bar, protocol)
                in enumerate(
                    zip(
                        bars,
                        ("xsub", "xset"),
                    )
                )
            ],
        )

        # Launch the two protocols concurrently.
        for protocol, gpu in zip(
            ("xsub", "xset"),
            gpus,
        ):
            child_out = out / protocol
            child_out.mkdir(exist_ok=True)

            atomic_json(
                child_out / "status.json",
                dict(
                    phase="Starting / resume",
                    current=0,
                    total=1,
                ),
            )

            stream = (
                out / f"{protocol}.log"
            ).open(
                "a",
                buffering=1,
            )
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
                zip(
                    processes,
                    ("xsub", "xset"),
                )
            ):
                status = read_json(
                    out / protocol / "status.json",
                    {},
                )

                finished, status = base.check_worker_finished(
                    proc,
                    protocol,
                    out,
                    status,
                )

                done += int(finished)
                statuses[protocol] = status

                base.update_bar(
                    bars[i],
                    protocol,
                    i,
                    status,
                )

            scores = base.best_score_snapshot(
                statuses
            )

            if scores != last_scores:
                atomic_json(
                    out / "best_scores.json",
                    scores,
                )
                last_scores = scores

            if done == 2:
                break

            time.sleep(0.5)

        results = {
            protocol: read_json(
                out
                / protocol
                / "result.json"
            )
            for protocol in (
                "xsub",
                "xset",
            )
        }

        atomic_json(
            out / "results.json",
            results,
        )

        return results

    finally:
        base.stop_processes(processes)

        for stream in streams:
            try:
                stream.close()
            except Exception:
                pass

        for bar in bars:
            try:
                bar.close()
            except Exception:
                pass

        lock.close()


if __name__ == "__main__":
    print(
        "Import run() from this module in Kaggle."
    )

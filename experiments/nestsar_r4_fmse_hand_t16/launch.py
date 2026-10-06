from __future__ import annotations

"""Dual-T4 launcher for NestSAR R4-FMSE + Hand Relations."""

import fcntl
import subprocess
import sys
import time
from pathlib import Path

from experiments.nestsar_sm_all_t16.streaming import launch as base
from experiments.nestsar_sm_all_t16.streaming.io_utils import (
    atomic_json,
    read_json,
)

from . import MODEL_IDENTITY, MODEL_NAME, VERSION


MODULE = "experiments.nestsar_r4_fmse_hand_t16"


def run(
    *,
    outdir="/kaggle/working/NestSAR_R4_FMSE_HAND_T16_v1",
    cache_dir="/kaggle/working/NestSAR_R4_EMA_REP_CONSISTENCY_CACHE_REBUILT_v1",
    config=None,
):
    c = base.validate_config(config or {})

    out = Path(outdir)
    cache = Path(cache_dir)

    if not cache.is_dir():
        raise FileNotFoundError(
            f"Expected existing canonical cache: {cache}"
        )

    for name in (
        "manifest.json",
        "canonical.npy",
        "labels.npy",
        "ids.json",
        "splits.json",
    ):
        if not (cache / name).is_file():
            raise FileNotFoundError(
                f"Incomplete canonical cache: {cache / name}"
            )

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
            "Stop that cell before restarting."
        )

    bars = []
    processes = []
    streams = []

    try:
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
                "experiment_version": VERSION,
                "model": MODEL_NAME,
                "model_identity": MODEL_IDENTITY,
            },
        )

        bars = base.make_bars()

        python = base.ensure_runtime(
            out,
            bars,
            gpus[0],
        )

        cfg_path = out / "config.json"

        old = read_json(
            cfg_path
        )
        if old is not None and old != c:
            raise ValueError(
                "OUT_DIR has a different config. "
                "Use the same config to resume or a new OUT_DIR."
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
                "experiment_version": VERSION,
                "model": MODEL_NAME,
                "model_identity": MODEL_IDENTITY,
            },
        )

        for i, gpu in enumerate(gpus):
            base.quiet_run(
                [
                    python,
                    "-c",
                    (
                        "import jax; "
                        "assert jax.default_backend()=='gpu'; "
                        "assert jax.local_device_count()==1; "
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

        # Architecture preflight: exact parameter count + finite forward pass.
        preflight = (
            "import json,jax,jax.numpy as jnp; "
            "from experiments.nestsar_sm_all_t16.streaming.launch import DEFAULTS; "
            "from experiments.nestsar_r4_fmse_hand_t16.worker import "
            "make_model,EXPECTED_PARAMS; "
            "m=make_model(DEFAULTS); "
            "k=jax.random.PRNGKey(1); "
            "p=m.init({'params':k,'dropout':k},"
            "jnp.zeros((1,16,750),jnp.float32),training=False)['params']; "
            "n=sum(x.size for x in jax.tree.leaves(p)); "
            "assert n==EXPECTED_PARAMS,(n,EXPECTED_PARAMS); "
            "o=m.apply({'params':p},jnp.zeros((1,16,750),jnp.float32),training=False); "
            "assert o['logits'].shape==(1,120); "
            "assert bool(jnp.all(jnp.isfinite(o['logits']))); "
            "print('HAND_PREFLIGHT='+json.dumps({'params':n,"
            "'logits':list(o['logits'].shape)}))"
        )

        base.quiet_run(
            [
                python,
                "-c",
                preflight,
            ],
            out / "hand_preflight.log",
            base.isolated_env(gpus[0]),
            lambda: base.update_bar(
                bars[0],
                "xsub",
                0,
                dict(
                    phase="Hand preflight",
                    current=0,
                    total=1,
                ),
            ),
        )

        for protocol, gpu in zip(
            ("xsub", "xset"),
            gpus,
        ):
            child_out = out / protocol
            child_out.mkdir(exist_ok=True)

            if not (child_out / "status.json").exists():
                atomic_json(
                    child_out / "status.json",
                    dict(
                        phase="Starting",
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
                MODULE + ".worker",
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
            p: read_json(
                out / p / "result.json"
            )
            for p in (
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
            stream.close()

        for bar in bars:
            bar.close()

        lock.close()

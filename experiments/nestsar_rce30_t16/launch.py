from __future__ import annotations

"""Dual-T4 launcher for NestSAR-RCE30.

GPU0 -> XSUB specialist using protected XSUB FMSE checkpoint.
GPU1 -> XSET specialist using protected XSET FMSE checkpoint.

The launcher performs an exact-baseline preflight before training.
"""

import fcntl
import json
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
from .worker import validate_config


MODULE = "experiments.nestsar_rce30_t16"


def run(
    *,
    outdir="/kaggle/working/NestSAR_RCE30_T16_v1",
    cache_dir="/kaggle/working/NestSAR_R4_EMA_REP_CONSISTENCY_CACHE_REBUILT_v1",
    base_xsub_checkpoint="/kaggle/working/NestSAR_R4_FMSE_LOCAL_GEOMETRY_T16_v1/xsub/best.msgpack",
    base_xset_checkpoint="/kaggle/working/NestSAR_R4_FMSE_LOCAL_GEOMETRY_T16_v1/xset/best.msgpack",
    config=None,
):
    c = validate_config(config or {})
    out = Path(outdir)
    cache = Path(cache_dir)
    base_ckpts = {
        "xsub": Path(base_xsub_checkpoint),
        "xset": Path(base_xset_checkpoint),
    }

    if not cache.is_dir():
        raise FileNotFoundError(cache)
    for name in (
        "manifest.json",
        "canonical.npy",
        "labels.npy",
        "ids.json",
        "splits.json",
    ):
        if not (cache / name).is_file():
            raise FileNotFoundError(cache / name)

    for protocol, path in base_ckpts.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{protocol.upper()} protected checkpoint missing: {path}"
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
            "This OUT_DIR already has an active RCE launcher."
        )

    bars = []
    processes = []
    streams = []

    try:
        gpus, jax_devices = base.discover_gpus(
            sys.executable,
            out / "gpu_discovery.log",
        )

        bars = base.make_bars()
        python = base.ensure_runtime(
            out,
            bars,
            gpus[0],
        )

        cfg_path = out / "config.json"
        old = read_json(cfg_path)
        if old is not None and old != c:
            raise ValueError(
                "OUT_DIR has a different RCE config. "
                "Use the same config to resume or choose a new OUT_DIR."
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
        source_commit = (
            source.stdout.strip()
            if source.returncode == 0
            else None
        )

        atomic_json(
            out / "hardware.json",
            {
                "jax_devices": jax_devices,
                "nvidia_smi": base.optional_nvidia_smi(),
                "assignment": dict(
                    zip(("xsub", "xset"), gpus)
                ),
                "model": MODEL_NAME,
                "identity": MODEL_IDENTITY,
                "version": VERSION,
                "source_commit": source_commit,
                "base_checkpoints": {
                    k: str(v)
                    for k, v in base_ckpts.items()
                },
            },
        )

        # GPU isolation probes.
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

        # Static design FLOP audit.
        base.quiet_run(
            [
                python,
                "-m",
                MODULE + ".audit",
            ],
            out / "compute_audit.log",
            base.isolated_env(gpus[0]),
            lambda: base.update_bar(
                bars[0],
                "xsub",
                0,
                dict(
                    phase="RCE compute audit",
                    current=0,
                    total=1,
                ),
            ),
        )

        # Exact-baseline synthetic checks for BOTH prototypes.
        preflight = (
            "import json,jax,jax.numpy as jnp;"
            "from flax import serialization;"
            "from pathlib import Path;"
            "from experiments.nestsar_rce30_t16.worker import "
            "load_base_checkpoint,specialist_parameter_count;"
            "from experiments.nestsar_rce30_t16.model import NestSARRCE30T16;"
            f"ckpt={str(base_ckpts['xsub'])!r};"
            "bm,bp,payload=load_base_checkpoint(ckpt);"
            "x=jnp.zeros((2,16,750),jnp.float32);"
            "base=bm.apply({'params':bp},x,training=False)['logits'];"
            "rivals=jnp.tile(jnp.array([[1,2]],jnp.int32),(120,1));"
            "rivals=rivals.at[:,0].set((jnp.arange(120)+1)%120);"
            "rivals=rivals.at[:,1].set((jnp.arange(120)+2)%120);"
            "result={};"
            "
for variant in ('fixed','rival'):
"
            " m=NestSARRCE30T16(variant=variant,dim=40,blocks=2,rank=4,dropout=0.05)
"
            " k=jax.random.PRNGKey(11)
"
            " p=m.init({'params':k,'dropout':k},x,base,rivals,training=False)['params']
"
            " o=m.apply({'params':p},x,base,rivals,training=False)
"
            " err=float(jnp.max(jnp.abs(o['logits']-base)))
"
            " n=int(sum(v.size for v in jax.tree.leaves(p)))
"
            " assert err<=1e-7,(variant,err)
"
            " assert o['carrier'].shape==(2,16,25,40)
"
            " assert o['evidence_bank'].shape==(2,25,40)
"
            " result[variant]={'params':n,'baseline_max_abs_error':err,"
            "'carrier':list(o['carrier'].shape),'bank':list(o['evidence_bank'].shape)}
"
            "print('RCE_PREFLIGHT='+json.dumps(result),flush=True)"
        )

        base.quiet_run(
            [python, "-c", preflight],
            out / "rce_preflight.log",
            base.isolated_env(gpus[0]),
            lambda: base.update_bar(
                bars[0],
                "xsub",
                0,
                dict(
                    phase="RCE exact-baseline preflight",
                    current=0,
                    total=1,
                ),
            ),
        )

        # Protocol workers.
        for protocol, gpu in zip(
            ("xsub", "xset"),
            gpus,
        ):
            child = out / protocol
            child.mkdir(exist_ok=True)
            if not (child / "status.json").exists():
                atomic_json(
                    child / "status.json",
                    dict(
                        phase="Starting",
                        current=0,
                        total=1,
                    ),
                )

            stream = (
                out / f"{protocol}.log"
            ).open("a", buffering=1)
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
                "--base-checkpoint",
                str(base_ckpts[protocol]),
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

            scores = base.best_score_snapshot(statuses)
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
            for p in ("xsub", "xset")
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

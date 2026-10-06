from __future__ import annotations

"""Dual-T4 launcher for FMSE + training-only local geometry."""

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
from .config import validate_config


MODULE = "experiments.nestsar_r4_fmse_geometry_t16"


def update_bar(bar, protocol, gpu, status):
    phase = status.get("phase", "Starting")
    total = max(int(status.get("total", 1)), 1)
    tag = (phase, status.get("epoch", 0), total)

    if getattr(bar, "_nestsar_tag", None) != tag:
        bar.reset(total=total)
        bar._nestsar_tag = tag

    epoch = status.get("epoch", 0)
    bar.set_description_str(
        f"{protocol.upper()} G{gpu} E{epoch:02d} {phase}",
        refresh=False,
    )
    bar.n = min(int(status.get("current", 0)), total)

    best = status.get("best")
    best_epoch = int(status.get("best_epoch", 0))
    stats = {
        "BEST": (
            f"{100*best:.4f}%@E{best_epoch:02d}"
            if best is not None and best_epoch
            else "--"
        )
    }

    if status.get("val_acc") is not None:
        stats["val"] = f"{100*status['val_acc']:.2f}%"
    if status.get("train_acc") is not None:
        stats["tr"] = f"{100*status['train_acc']:.2f}%"
    if status.get("loss") is not None:
        stats["loss"] = f"{status['loss']:.3f}"
    if status.get("geometry") is not None:
        stats["geo"] = f"{status['geometry']:.4f}"
    if status.get("geo_desc_active") is not None:
        stats["Dact"] = f"{100*status['geo_desc_active']:.1f}%"
    if status.get("geo_g4_active") is not None:
        stats["Gact"] = f"{100*status['geo_g4_active']:.1f}%"
    if status.get("geometry_scale") is not None:
        stats["lambda"] = f"{status['geometry_scale']:.2f}"
    if status.get("rss_gib") is not None:
        stats["RAM"] = f"{status['rss_gib']:.1f}G"
    if status.get("wait_s") is not None:
        stats["wait"] = f"{status['wait_s']:.0f}s"
    if status.get("gpu_s") is not None:
        stats["GPU"] = f"{status['gpu_s']:.0f}s"

    bar.set_postfix(stats, refresh=False)
    bar.refresh()


def run(
    *,
    outdir="/kaggle/working/NestSAR_R4_FMSE_LOCAL_GEOMETRY_T16_v1",
    cache_dir="/kaggle/working/NestSAR_R4_EMA_REP_CONSISTENCY_CACHE_REBUILT_v1",
    config=None,
):
    c = validate_config(config or {})

    out = Path(outdir)
    cache = Path(cache_dir)

    if not cache.is_dir():
        raise FileNotFoundError(f"Expected canonical cache: {cache}")

    for name in (
        "manifest.json",
        "canonical.npy",
        "labels.npy",
        "ids.json",
        "splits.json",
    ):
        if not (cache / name).is_file():
            raise FileNotFoundError(cache / name)

    out.mkdir(parents=True, exist_ok=True)

    lock = (out / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(
            "This OUT_DIR already has an active launcher. "
            "Stop its cell before restarting."
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
                "assignment": dict(zip(("xsub", "xset"), gpus)),
                "experiment_version": VERSION,
                "model": MODEL_NAME,
                "model_identity": MODEL_IDENTITY,
            },
        )

        bars = base.make_bars()
        python = base.ensure_runtime(out, bars, gpus[0])

        cfg_path = out / "config.json"
        old = read_json(cfg_path)

        if old is not None and old != c:
            raise ValueError(
                "OUT_DIR has a different config. "
                "Use a new OUT_DIR or restore the original config."
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
                lambda i=i: update_bar(
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

        # Use one REAL canonical training sample for the compile/run preflight.
        # An all-zero synthetic skeleton can exercise artificial masked/normalization
        # edge cases that never occur in the actual NTU cache and made the previous
        # preflight assertion opaque.  This preflight prints every checked value and
        # raises an explicit error if anything is wrong.
        preflight = (
            "import json,numpy as np,jax,jax.numpy as jnp; "
            "from pathlib import Path; "
            "from experiments.nestsar_r4_fmse_geometry_t16.config import validate_config; "
            "from experiments.nestsar_r4_fmse_geometry_t16.worker import "
            "create_state,build_steps,EXPECTED_PARAMS; "
            f"c=validate_config(json.load(open({str(cfg_path)!r}))); "
            "pc=dict(c); pc['micro_batch']=1; pc['accumulation_steps']=1; "
            "m,s,k,sch,w=create_state(pc,10); "
            "n=sum(x.size for x in jax.tree.leaves(s.params)); "
            f"cache=Path({str(cache)!r}); "
            "splits=json.load(open(cache/'splits.json')); "
            "idx=int(splits['xsub_train'][0]); "
            "canonical=np.load(cache/'canonical.npy',mmap_mode='r'); "
            "labels=np.load(cache/'labels.npy',mmap_mode='r'); "
            "x=jnp.asarray(np.asarray(canonical[idx:idx+1],dtype=np.float32)); "
            "y=jnp.asarray(np.asarray(labels[idx:idx+1],dtype=np.int32)); "
            "b={'x':x,'xa':x,'y':y,'mask':jnp.ones((1,),jnp.float32)}; "
            "train_step,eval_step=build_steps(m,pc); "
            "exe=train_step.lower(s,k,b,jnp.asarray(0.5,jnp.float32)).compile(); "
            "s2,k2,met=jax.block_until_ready(exe(s,k,b,jnp.asarray(0.5,jnp.float32))); "
            "finite=bool(np.asarray(jnp.all(jnp.isfinite(met)))); "
            "dcount=float(np.asarray(jnp.sum(s2.proto_desc_count))); "
            "gcount=float(np.asarray(jnp.sum(s2.proto_g4_count))); "
            "diag={'params':int(n),'expected_params':int(EXPECTED_PARAMS),"
            "'proto_desc':list(s2.proto_desc.shape),'proto_g4':list(s2.proto_g4.shape),"
            "'metrics':list(met.shape),'finite':finite,'desc_count_sum':dcount,"
            "'g4_count_sum':gcount,'sample_index':idx,'label':int(np.asarray(y)[0]),"
            "'metric_min':float(np.asarray(met).min()),"
            "'metric_max':float(np.asarray(met).max())}; "
            "print('GEOMETRY_PREFLIGHT_DIAG='+json.dumps(diag),flush=True); "
            "assert n==EXPECTED_PARAMS,(n,EXPECTED_PARAMS); "
            "assert s2.proto_desc.shape==(120,2,112),s2.proto_desc.shape; "
            "assert s2.proto_g4.shape==(120,2,112),s2.proto_g4.shape; "
            "assert met.shape==(20,),met.shape; "
            "assert finite,diag; "
            "assert dcount>0.0,diag; "
            "assert gcount>0.0,diag; "
            "print('GEOMETRY_PREFLIGHT=PASS',flush=True)"
        )

        base.quiet_run(
            [python, "-c", preflight],
            out / "geometry_preflight.log",
            base.isolated_env(gpus[0]),
            lambda: update_bar(
                bars[0],
                "xsub",
                0,
                dict(
                    phase="Geometry preflight",
                    current=0,
                    total=1,
                ),
            ),
        )

        for protocol, gpu in zip(("xsub", "xset"), gpus):
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

            stream = (out / f"{protocol}.log").open("a", buffering=1)
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

                update_bar(
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
            protocol: read_json(
                out / protocol / "result.json"
            )
            for protocol in ("xsub", "xset")
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

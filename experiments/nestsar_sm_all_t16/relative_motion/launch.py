"""Two persistent notebook bars; XSUB/GPU0 and XSET/GPU1 concurrently.

Proxy/relative arms and repeated grouped partitions run sequentially. This is
a full 120-class internal ablation, not a binary task or official test score.
"""
from __future__ import annotations
import csv
import fcntl
import hashlib
import io
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ..streaming.io_utils import atomic_bytes, atomic_json, read_json
from ..streaming.launch import discover_gpus, isolated_env, make_bars, quiet_run, tail, stop_processes
from ..nonlinear_compare.launch import locate_or_prepare, update_bar
from . import VERSION
from .config import validate_config
from .data import open_paths, split_plan

MODULE = "experiments.nestsar_sm_all_t16.relative_motion"
DEFAULT_OUT = "/kaggle/working/NestSAR_TrueRelative_T16_v1"
DEFAULT_AUX = "/kaggle/working/NestSAR_TrueRelative_PathCache_v1"
MODES = ("proxy", "relative")
PROTOCOLS = ("xsub", "xset")


def code_signature():
    experiment = Path(__file__).resolve().parents[1]
    paths = list(experiment.rglob("*.py"))
    for folder in ("m4_motionpreserve_t16", "m4_phase_jitter_uniform_t16",
                   "m4_phase_jitter_consistency_localglobal_t16"):
        paths.extend((experiment.parent / folder).glob("*.py"))
    h = hashlib.sha256()
    for path in sorted(set(paths)):
        h.update(str(path.relative_to(experiment.parent)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def paired_results(proxy_path, relative_path):
    with np.load(proxy_path, allow_pickle=False) as p, np.load(relative_path, allow_pickle=False) as r:
        for field in ("indices", "labels"):
            if not np.array_equal(p[field], r[field]):
                raise ValueError("Paired predictions have different IDs/labels")
        labels = p["labels"]
        a, b = p["predictions"] == labels, r["predictions"] == labels
        return dict(samples=len(labels), proxy_accuracy=float(a.mean()), relative_accuracy=float(b.mean()),
                    delta_pp=float(100*(b.mean()-a.mean())),
                    corrected=int(np.sum(~a & b)), damaged=int(np.sum(a & ~b)),
                    per_class=[dict(action=c+1, support=int(np.sum(labels == c)),
                        corrected=int(np.sum((labels == c) & ~a & b)),
                        damaged=int(np.sum((labels == c) & a & ~b))) for c in range(120)])


def export_results(out, config):
    records = []
    for seed in config["seeds"]:
        for protocol in PROTOCOLS:
            root = out / f"seed_{seed}"
            results = {m: read_json(root / m / protocol / "result.json") for m in MODES}
            if any(r is None for r in results.values()):
                raise RuntimeError("A worker exited without its result")
            pair = paired_results(root / "proxy" / protocol / "final_predictions.npz",
                                  root / "relative" / protocol / "final_predictions.npz")
            records.append(dict(protocol=protocol, seed=seed, final=pair,
                arms={m: {k: results[m][k] for k in ("best_val_accuracy", "best_epoch", "last_epoch",
                                                     "internal_final_accuracy", "params", "config_hash")}
                      for m in MODES}))
    summary = {}
    for protocol in PROTOCOLS:
        selected = [r for r in records if r["protocol"] == protocol]
        def stats(values):
            return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if len(values) > 1 else None)
        summary[protocol] = {m: stats([100*r["arms"][m]["internal_final_accuracy"] for r in selected]) for m in MODES}
        summary[protocol]["relative_minus_proxy_pp"] = stats([r["final"]["delta_pp"] for r in selected])
    report = dict(version=VERSION, smoke_test=config["smoke_test"], classes=120, records=records, summary=summary,
        scope="Internal official-train groups. Select groups choose best EMA; final groups evaluated after selection.",
        uncertainty="Repeats can overlap in subjects/setups; SD is descriptive, not an independent-sample confidence interval.")
    atomic_json(out / "results.json", report)
    rows = [dict(protocol=r["protocol"], seed=r["seed"],
                 proxy_best_select_pct=100*r["arms"]["proxy"]["best_val_accuracy"],
                 relative_best_select_pct=100*r["arms"]["relative"]["best_val_accuracy"],
                 proxy_final_pct=100*r["final"]["proxy_accuracy"],
                 relative_final_pct=100*r["final"]["relative_accuracy"],
                 delta_pp=r["final"]["delta_pp"], corrected=r["final"]["corrected"], damaged=r["final"]["damaged"])
            for r in records]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(out / "scores.csv", buffer.getvalue().encode())
    return report


def run(cache_dir=None, auxiliary_dir=DEFAULT_AUX, outdir=DEFAULT_OUT, config=None,
        dataset=None, *, _allow_cpu=False):
    config = validate_config(config)
    out = Path(str(outdir) + ("_smoke" if config["smoke_test"] else "")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("This output directory already has a running launcher")
    bars, processes, logs = [], [], []
    root = Path(__file__).resolve().parents[3]
    try:
        for path in out.glob("seed_*/*/*/status.json"):
            status = read_json(path, {})
            proc = Path(f"/proc/{status.get('pid', -1)}/cmdline")
            if proc.exists():
                line = proc.read_bytes().replace(b"\0", b" ").decode(errors="replace")
                if MODULE + ".worker" in line and str(out) in line:
                    raise RuntimeError(f"Earlier worker still running: {path}, PID {status['pid']}")
        bars = make_bars()
        if _allow_cpu:
            gpus, devices = ["0", "1"], ["CPU synthetic integration only"]
        else:
            gpus, devices = discover_gpus(sys.executable, out / "gpu_discovery.log")
        environments = []
        for i, protocol in enumerate(PROTOCOLS):
            env = isolated_env(None if _allow_cpu else gpus[i])
            environments.append(env)
            probe = ("import jax,jax.numpy as jnp,flax,optax,psutil,numpy,tqdm; "
                     "jax.jit(lambda x:x@x)(jnp.ones((16,16))).block_until_ready(); "
                     "print(jax.__version__,flax.__version__,optax.__version__,numpy.__version__,jax.devices())")
            quiet_run([sys.executable, "-c", probe], out / f"{protocol}_runtime.log", env,
                      lambda i=i, p=protocol: update_bar(bars[i], p, gpus[i], dict(phase="Runtime probe")))
        # Reuse the Kaggle GPU runtime. No venv, ensurepip, pip, CUDA download,
        # or removal of an earlier runtime directory in this launcher.
        cache = locate_or_prepare(cache_dir, dataset, out, bars)
        auxiliary = Path(auxiliary_dir).resolve()
        cache_status = out / "relative_cache_status.json"
        def refresh_cache():
            status = read_json(cache_status, dict(phase="Relative path cache", total=1, current=0))
            for i, p in enumerate(PROTOCOLS):
                update_bar(bars[i], p, gpus[i], status)
        quiet_run([sys.executable, "-m", MODULE + ".data", "--cache", str(cache),
                   "--auxiliary", str(auxiliary), "--status", str(cache_status)],
                  out / "relative_cache.log", isolated_env(), refresh_cache)
        _, meta = open_paths(cache, auxiliary)
        signature = dict(version=VERSION, code_sha256=code_signature(), config=config, cache=meta["signature"])
        old = read_json(out / "signature.json")
        if old is not None and old != signature:
            raise ValueError("Different code/config/data in this output; choose a new outdir. Existing results preserved.")
        atomic_json(out / "signature.json", signature)
        atomic_json(out / "config.json", config)
        git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        atomic_json(out / "source.json", dict(commit=git.stdout.strip() if git.returncode == 0 else None, **signature))
        atomic_json(out / "hardware.json", dict(devices=devices, assignment=dict(zip(PROTOCOLS, gpus)),
            runtime_logs={p: (out / f"{p}_runtime.log").read_text() for p in PROTOCOLS}))
        audit_key = dict(code=signature["code_sha256"], runtime=(out / "xsub_runtime.log").read_text(),
                         model={k: config["training"][k] for k in ("spatial_dim", "model_dim", "controller_dim",
                             "fast_rank", "head_rank", "dropout", "sm_residual_scale", "head_residual_scale")})
        if config["audit_first"] and (read_json(out / "audit_key.json") != audit_key or not (out / "audit.json").exists()):
            atomic_json(out / "audit_config.json", config["training"])
            quiet_run([sys.executable, "-m", MODULE + ".audit", "--config", str(out / "audit_config.json"),
                       "--output", str(out / "audit.json")], out / "audit.log", environments[0],
                      lambda: [update_bar(b, p, gpus[i], dict(phase="Verify model compute"))
                               for i, (b, p) in enumerate(zip(bars, PROTOCOLS))])
            atomic_json(out / "audit_key.json", audit_key)
        for repeat, seed in enumerate(config["seeds"], 1):
            stage = out / f"seed_{seed}"
            stage.mkdir(exist_ok=True)
            training_config = dict(config["training"], seed=seed)
            atomic_json(stage / "config.json", training_config)
            plans = {}
            for protocol in PROTOCOLS:
                plans[protocol] = split_plan(cache, protocol, seed, config["select_fraction"],
                                             config["final_fraction"], config["min_class_samples"])
                path = stage / f"{protocol}_split.json"
                old_plan = read_json(path)
                if old_plan is not None and old_plan != plans[protocol]:
                    raise ValueError("Stored internal partition does not match current data/settings")
                atomic_json(path, plans[protocol])
            for mode in MODES:
                processes, logs = [], []
                for i, protocol in enumerate(PROTOCOLS):
                    dest = stage / mode / protocol
                    dest.mkdir(parents=True, exist_ok=True)
                    experiment = dict(version=VERSION, mode=mode, code_sha256=signature["code_sha256"],
                        split_sha256=plans[protocol]["sha256"], classes=120, internal=True,
                        smoke_test=config["smoke_test"], seed=seed)
                    atomic_json(dest / "experiment.json", experiment)
                    stream = (dest / "worker.log").open("a", buffering=1)
                    logs.append(stream)
                    command = [sys.executable, "-m", MODULE + ".worker", "--cache", str(cache),
                        "--auxiliary", str(auxiliary), "--outdir", str(stage / mode),
                        "--protocol", protocol, "--mode", mode, "--config", str(stage / "config.json"),
                        "--plan", str(stage / f"{protocol}_split.json"), "--experiment", str(dest / "experiment.json")]
                    if _allow_cpu:
                        command.append("--allow-cpu")
                    processes.append(subprocess.Popen(command, cwd=root, env=environments[i], stdout=stream,
                                                       stderr=subprocess.STDOUT, start_new_session=True))
                while True:
                    for i, (p, proc) in enumerate(zip(PROTOCOLS, processes)):
                        dest = stage / mode / p
                        status = read_json(dest / "status.json", {})
                        update_bar(bars[i], p, gpus[i], dict(status, arm=mode, repeat=repeat))
                        if proc.poll() not in (None, 0):
                            raise RuntimeError(f"{p.upper()} {mode} failed: {dest/'worker.log'}\n{tail(dest/'worker.log')}")
                    if all(p.poll() == 0 for p in processes):
                        break
                    time.sleep(.4)
                for stream in logs:
                    stream.close()
                processes, logs = [], []
                for protocol in PROTOCOLS:
                    if not read_json(stage / mode / protocol / "status.json", {}).get("done"):
                        raise RuntimeError("Worker exited without completing checkpoint evaluation")
        result = export_results(out, config)
        for i, protocol in enumerate(PROTOCOLS):
            update_bar(bars[i], protocol, gpus[i], dict(phase="Final internal means", current=1, total=1,
                summaries={m: result["summary"][protocol][m]["mean"] for m in MODES}))
        for b in bars:
            b.close()
        bars = []
        print("SMOKE ONLY" if config["smoke_test"] else "120-CLASS INTERNAL GROUP COMPARISON (not official NTU test scores)")
        for p, values in result["summary"].items():
            print(f"{p.upper()}: proxy {values['proxy']['mean']:.2f}% | relative {values['relative']['mean']:.2f}% "
                  f"| delta {values['relative_minus_proxy_pp']['mean']:+.2f} pp")
        print(f"Best selection scores, final scores and per-class corrections: {out}")
        return result
    finally:
        stop_processes(processes)
        for stream in logs:
            stream.close()
        for b in bars:
            b.close()
        lock.close()

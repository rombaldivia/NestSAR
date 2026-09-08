"""One notebook-owned TQDM bar per protocol; one subprocess per T4."""
from __future__ import annotations
import csv
import fcntl
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..streaming.io_utils import atomic_bytes, atomic_json, read_json
from ..streaming.launch import discover_gpus, find_dataset, isolated_env, make_bars, quiet_run, tail
from . import VERSION
from .config import ARMS, validate_config
from .data import EXPECTED_PREPROCESSING

MODULE = "experiments.nestsar_sm_all_t16.nonlinear_compare.worker"
DEFAULT_CACHE = "/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3"
DEFAULT_OUT = "/kaggle/working/NestSAR_P2_Nonlinear_Grouped_v1"


def locate_cache(cache_dir):
    if cache_dir is not None:
        candidates = [Path(cache_dir)]
    elif (Path(DEFAULT_CACHE)/"manifest.json").exists():
        candidates = [Path(DEFAULT_CACHE)]
    else:
        candidates = [p.parent for p in Path("/kaggle/input").rglob("manifest.json")]
    matches = []
    for path in candidates:
        meta = read_json(path/"manifest.json", {})
        if (meta.get("signature", {}).get("preprocessing") == EXPECTED_PREPROCESSING
                and "internal_split" not in meta.get("signature", {})):
            matches.append(path.resolve())
    if len(matches) != 1:
        raise FileNotFoundError(
            "Set cache_dir to your existing person-aware P2 v3 cache containing manifest.json, "
            "raw.npy, canonical.npy, ids.json and splits.json. Attach that cache as Kaggle input "
            "when starting a new session. This diagnostic does not reload the full NTU pickle. "
            f"Matching caches: {matches}")
    return matches[0]


def locate_or_prepare(cache_dir, dataset, out, bars):
    try:
        return locate_cache(cache_dir)
    except FileNotFoundError:
        target = Path(cache_dir or DEFAULT_CACHE).resolve()
        if (target/"manifest.json").exists():
            raise ValueError("Requested cache has incompatible metadata; use the original P2 cache or a new cache_dir")
        dataset = find_dataset(dataset)
        status_file = out/"cache_prepare.json"

        def refresh():
            status = read_json(status_file, dict(phase="Build shared P2 cache", current=0, total=1))
            for i, (bar, protocol) in enumerate(zip(bars, ("xsub", "xset"))):
                update_bar(bar, protocol, i, status)

        quiet_run([sys.executable, "-m", "experiments.nestsar_sm_all_t16.streaming.data",
                   "--dataset", str(dataset), "--cache", str(target), "--status", str(status_file)],
                  out/"cache_prepare.log", isolated_env(), refresh)
        return locate_cache(target)


def update_bar(bar, protocol, gpu, status):
    total = max(1, int(status.get("total", 1)))
    tag = tuple(status.get(k) for k in ("pair", "repeat", "arm", "trial", "epoch", "phase"))+(total,)
    if getattr(bar, "_comparison_tag", None) != tag:
        bar.reset(total=total)
        bar._comparison_tag = tag
    info = " ".join(str(status[k]) for k in ("pair", "arm") if status.get(k))
    bar.set_description_str(f"{protocol.upper()} G{gpu} R{status.get('repeat', 0)} "
                            f"E{status.get('epoch', 0):02d} {info} {status.get('phase', 'Starting')}", refresh=False)
    bar.n = min(total, int(status.get("current", 0)))
    best, epoch = status.get("best"), status.get("best_epoch", 0)
    stats = {"BEST_INNER": f"{100*best:.2f}%@E{epoch:02d}" if best is not None and epoch else "--"}
    for key, label in (("train_acc", "train"), ("val_acc", "inner")):
        if status.get(key) is not None:
            stats[label] = f"{100*status[key]:.2f}%"
    if status.get("trial"):
        stats["trial"] = f"{status['trial']}/{status.get('trials', 1)}"
    if status.get("loss") is not None:
        stats["loss"] = f"{status['loss']:.3f}"
    if status.get("summaries"):
        stats = {arm: f"{value:.2f}%" for arm, value in status["summaries"].items()}
    bar.set_postfix(stats, refresh=False)
    bar.refresh()


def _source_signature(cache, config):
    folder = Path(__file__).resolve().parent
    source_hash = hashlib.sha256()
    for path in sorted(folder.glob("*.py")) + [folder.parent/"preprocessing_corrected.py", folder.parent/"streaming"/"data.py"]:
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    return dict(version=VERSION, code_sha256=source_hash.hexdigest(),
                cache_signature=read_json(cache/"manifest.json")["signature"], config=config)


def _export(out, summaries):
    rows = []
    for protocol in ("xsub", "xset"):
        for record in read_json(out/protocol/"comparisons.json"):
            row = dict(protocol=protocol, pair="/".join(f"A{a:03d}" for a in record["pair"]), seed=record["seed"])
            for arm in ARMS:
                info = record["arms"][arm]
                row[arm+"_final_bacc_pct"] = 100*info["final"]["balanced_accuracy"]
                row[arm+"_best_inner_bacc_pct"] = 100*info["selection"]["balanced_accuracy"]
                row[arm+"_best_epoch"] = info["best_epoch"]
                row[arm+"_params"] = info["params"]
            for name, diff in record["paired_differences"].items():
                row[name+"_pp"] = diff["delta_pp"]
                row[name+"_ci95_low_pp"] = diff["ci95_pp"][0] if diff["ci95_pp"] else None
                row[name+"_ci95_high_pp"] = diff["ci95_pp"][1] if diff["ci95_pp"] else None
            rows.append(row)
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(out/"pair_comparisons.csv", buf.getvalue().encode())
    atomic_json(out/"results.json", summaries)


def run(cache_dir=None, outdir=DEFAULT_OUT, config=None, dataset=None, *, _allow_cpu=False):
    c = validate_config(config)
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/"comparison.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("This comparison is already running in this output directory")
    processes, logs, bars = [], [], []
    try:
        bars = make_bars()
        cache = locate_or_prepare(cache_dir, dataset, out, bars)
        signature = _source_signature(cache, c)
        old = read_json(out/"signature.json")
        if old is not None and old != signature:
            raise ValueError("Output has different code/config/data. Choose a new outdir; results were preserved.")
        for protocol in ("xsub", "xset"):
            status = read_json(out/protocol/"status.json", {})
            pid = status.get("pid")
            if pid and not status.get("done"):
                cmdline = Path(f"/proc/{pid}/cmdline")
                if cmdline.exists():
                    value = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
                    if MODULE in value and str(out) in value:
                        raise RuntimeError(f"An earlier {protocol} worker is still running (PID {pid})")
        atomic_json(out/"signature.json", signature)
        atomic_json(out/"config.json", c)
        root = Path(__file__).resolve().parents[3]
        source = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        atomic_json(out/"source.json", dict(commit=source.stdout.strip() if source.returncode == 0 else None, **signature))
        if _allow_cpu:
            gpus, devices = ["0", "1"], ["CPU synthetic integration only"]
        else:
            gpus, devices = discover_gpus(sys.executable, out/"gpu_discovery.log")
        atomic_json(out/"hardware.json", dict(devices=devices, assignment=dict(zip(("xsub", "xset"), gpus))))
        for i, protocol in enumerate(("xsub", "xset")):
            env = isolated_env(gpus[i])
            env["PYTHONPATH"] = str(root) + (os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            env["PYTHONUNBUFFERED"] = "1"
            if _allow_cpu:
                env["JAX_PLATFORMS"] = "cpu"
            probe = "import jax,jax.numpy as jnp,flax,optax,psutil,numpy; "
            if not _allow_cpu:
                probe += "assert jax.default_backend()=='gpu' and jax.local_device_count()==1; "
            probe += "jax.jit(lambda x:x@x)(jnp.ones((16,16))).block_until_ready(); print(jax.devices())"
            quiet_run([sys.executable, "-c", probe], out/f"{protocol}_runtime_probe.log", env,
                      lambda i=i, protocol=protocol: update_bar(bars[i], protocol, gpus[i], dict(phase="Runtime probe")))
            log = (out/f"{protocol}.log").open("a", buffering=1)
            logs.append(log)
            args = [sys.executable, "-m", MODULE, "--cache", str(cache), "--outdir", str(out),
                    "--protocol", protocol, "--config", str(out/"config.json")]
            if _allow_cpu:
                args.append("--allow-cpu")
            processes.append(subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=root, env=env))
        while True:
            for i, (protocol, process) in enumerate(zip(("xsub", "xset"), processes)):
                status = read_json(out/protocol/"status.json", {})
                update_bar(bars[i], protocol, gpus[i], status)
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"{protocol.upper()} failed. Log: {out/f'{protocol}.log'}\n{tail(out/f'{protocol}.log')}")
            if all(p.poll() == 0 for p in processes):
                break
            time.sleep(.4)
        summaries = {p: read_json(out/p/"summary.json") for p in ("xsub", "xset")}
        if any(value is None for value in summaries.values()):
            raise RuntimeError("A worker exited without a complete summary")
        _export(out, summaries)
        for bar in bars:
            bar.close()
        bars = []
        print("\nINTERNAL BINARY DIAGNOSTIC — mean ± SD across grouped repeats; NOT NTU120 top-1")
        print(f"{'Protocol':<10} {'T16 MLP':>18} {'Seq16 GRU':>18} {'Seq64 GRU':>18} {'64−T16 pp':>13}")
        for protocol, summary in summaries.items():
            def fmt(arm):
                d = summary[arm]
                return f"{d['mean']:.2f}%" if d["sd"] is None else f"{d['mean']:.2f} ± {d['sd']:.2f}%"
            print(f"{protocol.upper():<10} {fmt(ARMS[0]):>18} {fmt(ARMS[1]):>18} {fmt(ARMS[2]):>18} "
                  f"{summary['sequence64_minus_t16_pp']['mean']:>+13.2f}")
        print(f"Saved final predictions, split IDs, best inner scores, histories and paired intervals: {out}")
        return summaries
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for log in logs:
            log.close()
        for bar in bars:
            bar.close()
        lock.close()

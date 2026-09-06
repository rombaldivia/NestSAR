"""Notebook owns two persistent bars. Workers write only files, never bars."""
from __future__ import annotations
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import venv
from pathlib import Path

from .io_utils import atomic_json, read_json

DEFAULTS = dict(
    epochs=60, patience=5, micro_batch=64, accumulation_steps=4, eval_batch=256,
    seed=128, learning_rate=6e-4, min_learning_rate=2e-5, warmup_fraction=0.08,
    weight_decay=0.03, label_smoothing=0.05, grad_clip=1.0, ema_decay=0.995,
    dropout=0.10, stream_aux_weight=0.15, hand_aux_weight=0.05,
    consistency_weight=0.08, consistency_temperature=1.0, hand_residual_scale=0.10,
    fresh_augmentation=True, rotation_degrees=8.0, jitter_shift=1,
    min_delta=0.0, progress_every=5, max_train_samples=0, max_val_samples=0,
)

RUNTIME_REQUIREMENTS = (
    "jax[cuda12]==0.7.2", "flax==0.11.2", "optax==0.2.5",
    "numpy==2.2.6", "psutil==7.0.0", "pytest==8.4.2",
)


def validate_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    c = dict(DEFAULTS, **config)
    for k in ("epochs", "patience", "micro_batch", "accumulation_steps", "eval_batch", "progress_every"):
        if not isinstance(c[k], int) or c[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    for k in ("seed", "jitter_shift", "max_train_samples", "max_val_samples"):
        if not isinstance(c[k], int) or c[k] < 0:
            raise ValueError(f"{k} must be a nonnegative integer")
    for k in ("dropout", "label_smoothing", "ema_decay"):
        if not 0 <= c[k] < 1:
            raise ValueError(f"Invalid {k}")
    if not 0 < c["warmup_fraction"] < 1 or c["consistency_temperature"] <= 0:
        raise ValueError("Invalid warmup/temperature")
    if not 0 < c["min_learning_rate"] <= c["learning_rate"] or c["grad_clip"] <= 0:
        raise ValueError("Invalid learning rate/gradient clipping")
    if not 0 <= c["rotation_degrees"] <= 20:
        raise ValueError("Keep yaw augmentation within 0..20 degrees")
    if any(c[k] < 0 for k in ("weight_decay", "stream_aux_weight", "hand_aux_weight", "consistency_weight", "min_delta")):
        raise ValueError("Loss weights/weight decay/min_delta must be nonnegative")
    return c


def isolated_env(gpu=None):
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", XLA_PYTHON_CLIENT_PREALLOCATE="false", MALLOC_ARENA_MAX="2",
               TF_CPP_MIN_LOG_LEVEL="2")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    env["JAX_PLATFORMS"] = "cpu" if gpu is None else "cuda"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    return env


def tail(path, n=25):
    from collections import deque
    try:
        with Path(path).open(errors="replace") as f:
            return "".join(deque(f, maxlen=n))
    except FileNotFoundError:
        return "No log file was created."


def stop_processes(processes):
    for proc in processes:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 8
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(0.1)
    for proc in processes:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait()


def quiet_run(cmd, log, env, refresh=None):
    with Path(log).open("w") as stream:
        proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            while proc.poll() is None:
                if refresh:
                    refresh()
                time.sleep(0.3)
        except BaseException:
            stop_processes([proc])
            raise
        if proc.returncode:
            raise RuntimeError(f"Stage failed (exit {proc.returncode}). Log: {log}\n{tail(log)}")


def find_dataset(explicit):
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    found = list(Path("/kaggle/input").rglob("ntu120_3danno.pkl"))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one ntu120_3danno.pkl under /kaggle/input, found {len(found)}. Set DATASET explicitly.")
    return found[0]


def make_bars():
    # Select notebook widgets explicitly. Never forward carriage returns through
    # subprocess pipes; that is what converted updates to new lines before.
    try:
        from IPython import get_ipython
        notebook = get_ipython() is not None and getattr(get_ipython(), "kernel", None) is not None
    except ImportError:
        notebook = False
    if notebook:
        try:
            import ipywidgets  # noqa: F401
            from tqdm.notebook import tqdm
        except ImportError as exc:
            raise RuntimeError("Notebook bars need tqdm and ipywidgets in the notebook kernel.") from exc
    else:
        from tqdm import tqdm
    return [tqdm(total=1, desc=f"{p} GPU{i} setup", position=i, leave=True,
                 mininterval=0.5, dynamic_ncols=True) for i, p in enumerate(("XSUB", "XSET"))]


def update_bar(bar, protocol, gpu, status):
    phase = status.get("phase", "Starting")
    total = max(int(status.get("total", 1)), 1)
    tag = (phase, status.get("epoch", 0), total)
    if getattr(bar, "_nestsar_tag", None) != tag:
        bar.reset(total=total)
        bar._nestsar_tag = tag
    epoch = status.get("epoch", 0)
    bar.set_description_str(f"{protocol.upper()} G{gpu} E{epoch:02d} {phase}", refresh=False)
    bar.n = min(int(status.get("current", 0)), total)
    best, best_epoch = status.get("best"), int(status.get("best_epoch", 0))
    # BEST comes first so it remains visible on narrower notebook outputs.
    stats = {"BEST": f"{100*best:.4f}%@E{best_epoch:02d}" if best is not None and best_epoch else "--"}
    for k, label in (("val_acc", "val"), ("train_acc", "tr")):
        if status.get(k) is not None:
            stats[label] = f"{100*status[k]:.2f}%"
    if "loss" in status:
        stats["loss"] = f"{status['loss']:.3f}"
    if "rss_gib" in status:
        stats["RAM"] = f"{status['rss_gib']:.1f}G"
    if "wait_s" in status:
        stats["wait"] = f"{status['wait_s']:.0f}s"
    if "gpu_s" in status:
        stats["GPU"] = f"{status['gpu_s']:.0f}s"
    bar.set_postfix(stats, refresh=False)
    bar.refresh()


def create_runtime_without_pip(runtime):
    """Complete/recover an interrupted venv without ever calling ensurepip."""
    runtime = Path(runtime)
    # clear=False preserves any installed packages from an interrupted setup.
    # Also finish activation scripts if the former with_pip=True path failed.
    venv.EnvBuilder(with_pip=False, clear=False, system_site_packages=False).create(runtime)
    python = runtime / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"Runtime Python was not created: {python}")
    return python


def install_into_runtime(python, requirements, log, refresh=None, extra_options=()):
    """The host's pip --python works even when the target has no pip installed.

    https://pip.pypa.io/en/stable/topics/python-option/
    """
    command = [sys.executable, "-m", "pip", "--python", str(python), "install",
               "--disable-pip-version-check", "--no-cache-dir", *extra_options, *requirements]
    quiet_run(command, log, isolated_env(), refresh)


def runtime_probe(python, code, env, log):
    try:
        result = subprocess.run([str(python), "-c", code], env=env,
                                capture_output=True, text=True, timeout=45)
    except (OSError, subprocess.TimeoutExpired) as exc:
        Path(log).write_text(str(exc))
        return False
    Path(log).write_text((result.stdout or "") + (result.stderr or ""))
    return result.returncode == 0


def ensure_runtime(out, bars, gpu):
    out = Path(out)
    probe = ("import jax,flax,optax,psutil,pytest; "
             "assert (jax.__version__,flax.__version__,optax.__version__)==('0.7.2','0.11.2','0.2.5'); "
             "assert jax.default_backend()=='gpu' and jax.local_device_count()==1; "
             "print(jax.devices())")
    env = isolated_env(gpu)
    if runtime_probe(sys.executable, probe, env, out / "notebook_runtime_probe.log"):
        return sys.executable
    runtime = out / "runtime"
    python = runtime / "bin" / "python"
    if python.exists() and runtime_probe(python, probe, env, out / "cached_runtime_probe.log"):
        return str(python)
    refresh = lambda: [update_bar(b, p, i, dict(phase="Install runtime", current=0, total=1)) for i, (b, p) in enumerate(zip(bars, ("xsub", "xset")))]
    refresh()
    python = create_runtime_without_pip(runtime)
    install_into_runtime(python, RUNTIME_REQUIREMENTS, out / "install.log", refresh)
    quiet_run([str(python), "-c", probe], out / "runtime_probe.log", env, refresh)
    return str(python)


def best_score_snapshot(statuses):
    """Only completed validation can supply a best score; live val_acc cannot."""
    scores = {}
    for protocol in ("xsub", "xset"):
        status = statuses.get(protocol, {})
        epoch = int(status.get("best_epoch", 0))
        best = status.get("best") if epoch > 0 else None
        scores[protocol] = {"best_val_accuracy": best,
                            "best_val_percent": None if best is None else 100 * best,
                            "best_epoch": epoch, "completed_epoch": int(status.get("completed_epoch", 0))}
    return scores


def check_worker_finished(proc, protocol, out, status):
    rc = proc.poll()
    if rc is None:
        return False, status
    if rc:
        raise RuntimeError(f"{protocol.upper()} failed, exit {rc}. Log: {out / (protocol + '.log')}\n{tail(out / (protocol + '.log'))}")
    # A worker may finish between the first status read and poll(). Re-read its
    # atomic completion status instead of failing on the previous progress row.
    status = read_json(out / protocol / "status.json", {})
    if not status.get("done"):
        raise RuntimeError(f"{protocol} exited without a completion status")
    return True, status


def run(dataset=None, outdir="/kaggle/working/NestSAR_Fixed_T16_2xT4",
        cache_dir="/kaggle/working/NestSAR_RawCache_v1", config=None,
        raw_layout="MTVC", audit_first=True):
    c = validate_config(config or {})
    dataset = find_dataset(dataset)
    out, cache = Path(outdir), Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("This OUT_DIR already has an active launcher. Stop its cell before restarting.")
    bars, processes, streams = [], [], []
    try:
        nv = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True, timeout=20).stdout
        gpus = [line.split(",")[0].strip() for line in nv.splitlines() if "T4" in line]
        if len(gpus) < 2:
            raise RuntimeError(f"Select Kaggle accelerator GPU T4 x2. Detected:\n{nv}")
        gpus = gpus[:2]
        atomic_json(out / "hardware.json", {"nvidia_smi": nv, "assignment": dict(zip(("xsub", "xset"), gpus))})
        bars = make_bars()
        python = ensure_runtime(out, bars, gpus[0])
        cfg_path = out / "config.json"
        old = read_json(cfg_path)
        if old is not None and old != c:
            raise ValueError("OUT_DIR has a different config. Use a new OUT_DIR for this ablation.")
        atomic_json(cfg_path, c)
        # One probe per GPU before spending time preparing the real dataset.
        for i, gpu in enumerate(gpus):
            quiet_run([python, "-c", "import jax; assert jax.default_backend()=='gpu' and jax.local_device_count()==1; print(jax.devices())"],
                      out / f"gpu{i}_probe.log", isolated_env(gpu), lambda: update_bar(bars[i], ("xsub", "xset")[i], i, dict(phase="GPU probe")))
        tests = Path(__file__).parent / "tests" / "test_preprocessing.py"
        quiet_run([python, "-m", "pytest", "-q", str(tests)], out / "preprocessing_tests.log", isolated_env(),
                  lambda: update_bar(bars[0], "xsub", 0, dict(phase="Regression checks")))
        status_path = out / "prepare_status.json"
        quiet_run([python, "-m", "nestsar_fixed.data", "--dataset", str(dataset), "--cache", str(cache),
                   "--status", str(status_path), "--layout", raw_layout], out / "prepare.log", isolated_env(),
                  lambda: [update_bar(b, p, i, read_json(status_path, dict(phase="Prepare cache")))
                           for i, (b, p) in enumerate(zip(bars, ("xsub", "xset")))])
        if audit_first:
            quiet_run([python, "-m", "nestsar_fixed.audit", "--output", str(out / "compute_audit.json")],
                      out / "compute_audit.log", isolated_env(gpus[0]),
                      lambda: update_bar(bars[0], "xsub", 0, dict(phase="Full FLOP audit")))
        for protocol, gpu in zip(("xsub", "xset"), gpus):
            child_out = out / protocol
            child_out.mkdir(exist_ok=True)
            atomic_json(child_out / "status.json", dict(phase="Starting", current=0, total=1))
            stream = (out / f"{protocol}.log").open("a", buffering=1)
            streams.append(stream)
            cmd = [python, "-u", "-m", "nestsar_fixed.worker", "--config", str(cfg_path), "--protocol", protocol,
                   "--cache", str(cache), "--outdir", str(out)]
            processes.append(subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                              env=isolated_env(gpu), start_new_session=True))
        last_scores = None
        while True:
            done = 0
            statuses = {}
            for i, (proc, protocol) in enumerate(zip(processes, ("xsub", "xset"))):
                status = read_json(out / protocol / "status.json", {})
                finished, status = check_worker_finished(proc, protocol, out, status)
                done += int(finished)
                statuses[protocol] = status
                update_bar(bars[i], protocol, i, status)
            scores = best_score_snapshot(statuses)
            if scores != last_scores:
                atomic_json(out / "best_scores.json", scores)
                last_scores = scores
            if done == 2:
                break
            time.sleep(0.5)
        results = {p: read_json(out / p / "result.json") for p in ("xsub", "xset")}
        atomic_json(out / "results.json", results)
        return results
    finally:
        stop_processes(processes)
        for stream in streams:
            stream.close()
        for bar in bars:
            bar.close()
        lock.close()

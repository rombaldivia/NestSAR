"""Notebook owns two persistent bars. Workers write only files, never bars."""
from __future__ import annotations
import fcntl, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

from .io_utils import atomic_json, read_json
from . import VERSION

MODULE = "experiments.nestsar_sm_all_t16.streaming"

DEFAULTS = dict(
    epochs=60, patience=5, micro_batch=64, accumulation_steps=4, eval_batch=256,
    seed=128, learning_rate=6e-4, min_learning_rate=2e-5, warmup_fraction=0.08,
    weight_decay=0.03, label_smoothing=0.05, grad_clip=1.0, ema_decay=0.995,
    dropout=0.10, stream_aux_weight=0.15,
    consistency_weight=0.08, consistency_temperature=1.0,
    spatial_dim=24, model_dim=112, controller_dim=16, fast_rank=2, head_rank=2,
    sm_residual_scale=0.08, head_residual_scale=0.15,
    fresh_augmentation=True, rotation_degrees=8.0, jitter_shift=1,
    min_delta=1e-6, progress_every=5, max_train_samples=0, max_val_samples=0,
    prefetch_batches=2,
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
    if any(c[k] < 0 for k in ("weight_decay", "stream_aux_weight", "consistency_weight",
                              "min_delta", "sm_residual_scale", "head_residual_scale")):
        raise ValueError("Loss weights/weight decay/min_delta must be nonnegative")
    if c["prefetch_batches"] not in (1, 2):
        raise ValueError("prefetch_batches must be 1 or 2 to bound host memory")
    for k in ("spatial_dim", "model_dim", "controller_dim", "fast_rank", "head_rank"):
        if c[k] != DEFAULTS[k]:
            raise ValueError(f"Keep {k}={DEFAULTS[k]} for the unchanged SM-ALL experiment")
    return c


def _common_env():
    env = dict(os.environ)
    env.update(
        PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", MALLOC_ARENA_MAX="2",
        TF_CPP_MIN_LOG_LEVEL="2",
    )
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    return env


def isolated_env(gpu=None):
    env = _common_env()
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    env["JAX_PLATFORMS"] = "cpu" if gpu is None else "cuda"
    return env


def discovery_env():
    env = _common_env()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("JAX_PLATFORMS", None)
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
        proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
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
        raise FileNotFoundError(
            f"Expected one ntu120_3danno.pkl under /kaggle/input, found {len(found)}. Set DATASET explicitly."
        )
    return found[0]


def make_bars():
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
                 mininterval=0.5, dynamic_ncols=True)
            for i, p in enumerate(("XSUB", "XSET"))]


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


def runtime_probe(python, code, env, log):
    try:
        result = subprocess.run([str(python), "-c", code], env=env,
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        Path(log).write_text(str(exc))
        return False
    Path(log).write_text((result.stdout or "") + (result.stderr or ""))
    return result.returncode == 0


def ensure_runtime(out, bars, gpu):
    """Use Kaggle's existing runtime. Never download/install JAX or CUDA wheels."""
    out = Path(out)
    refresh = lambda: [update_bar(b, p, i, dict(phase="Runtime probe", current=0, total=1))
                       for i, (b, p) in enumerate(zip(bars, ("xsub", "xset")))]
    refresh()
    probe = (
        "import json,jax,flax,optax,psutil,tqdm,numpy; "
        "assert jax.default_backend()=='gpu' and jax.local_device_count()==1; "
        "print('NESTSAR_RUNTIME='+json.dumps({"
        "'python':__import__('sys').version.split()[0],"
        "'jax':jax.__version__,'flax':flax.__version__,'optax':optax.__version__,"
        "'numpy':numpy.__version__,'backend':jax.default_backend(),"
        "'devices':[str(d) for d in jax.devices()]}))"
    )
    log = out / "notebook_runtime_probe.log"
    if not runtime_probe(sys.executable, probe, isolated_env(gpu), log):
        raise RuntimeError(
            "Kaggle's existing Python/JAX runtime is not usable on the selected GPU. "
            "No packages were installed automatically. Ensure accelerator GPU T4 x2 is selected. "
            f"Inspect {log}:\n{tail(log)}"
        )
    stale = out / "runtime"
    if stale.exists():
        shutil.rmtree(stale, ignore_errors=True)
    return sys.executable


def discover_gpus(python, log):
    """Discover GPUs through a fresh JAX subprocess; nvidia-smi is optional."""
    code = (
        "import json,jax; "
        "ds=[d for d in jax.devices() if getattr(d,'platform','') in ('gpu','cuda')]; "
        "print('NESTSAR_GPU_DISCOVERY='+json.dumps(["
        "{'id':int(getattr(d,'id',i)),'platform':str(getattr(d,'platform','')),'kind':str(getattr(d,'device_kind',d))} "
        "for i,d in enumerate(ds)]))"
    )
    try:
        result = subprocess.run([str(python), "-c", code], env=discovery_env(),
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        Path(log).write_text(str(exc))
        raise RuntimeError(f"GPU discovery failed: {exc}") from exc
    Path(log).write_text((result.stdout or "") + (result.stderr or ""))
    if result.returncode:
        raise RuntimeError(
            "Could not initialize JAX GPU discovery. Select Kaggle accelerator GPU T4 x2. "
            f"Log: {log}\n{tail(log)}"
        )
    prefix = "NESTSAR_GPU_DISCOVERY="
    payload = next((line[len(prefix):] for line in (result.stdout or "").splitlines()
                    if line.startswith(prefix)), None)
    if payload is None:
        raise RuntimeError(f"GPU discovery returned no parseable device list. Log: {log}\n{tail(log)}")
    devices = json.loads(payload)
    if len(devices) < 2:
        raise RuntimeError(
            f"NestSAR dual-protocol training needs two visible GPUs. JAX detected {devices}. "
            "Select Kaggle accelerator GPU T4 x2."
        )
    return [str(d["id"]) for d in devices[:2]], devices


def optional_nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def best_score_snapshot(statuses):
    scores = {}
    for protocol in ("xsub", "xset"):
        status = statuses.get(protocol, {})
        epoch = int(status.get("best_epoch", 0))
        best = status.get("best") if epoch > 0 else None
        scores[protocol] = {
            "best_val_accuracy": best,
            "best_val_percent": None if best is None else 100 * best,
            "best_epoch": epoch,
            "completed_epoch": int(status.get("completed_epoch", 0)),
        }
    return scores


def check_worker_finished(proc, protocol, out, status):
    rc = proc.poll()
    if rc is None:
        return False, status
    if rc:
        raise RuntimeError(
            f"{protocol.upper()} failed, exit {rc}. Log: {out / (protocol + '.log')}\n"
            f"{tail(out / (protocol + '.log'))}"
        )
    status = read_json(out / protocol / "status.json", {})
    if not status.get("done"):
        raise RuntimeError(f"{protocol} exited without a completion status")
    return True, status


def run(dataset=None, outdir="/kaggle/working/NestSAR_SM_ALL_T16_SharedCache_v2",
        cache_dir="/kaggle/working/NestSAR_SM_ALL_SharedCache_v2", config=None,
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
        gpus, jax_devices = discover_gpus(sys.executable, out / "gpu_discovery.log")
        atomic_json(out / "hardware.json", {
            "gpu_discovery": "jax_subprocess",
            "jax_devices": jax_devices,
            "nvidia_smi": optional_nvidia_smi(),
            "assignment": dict(zip(("xsub", "xset"), gpus)),
            "pipeline_version": VERSION,
        })

        bars = make_bars()
        python = ensure_runtime(out, bars, gpus[0])

        cfg_path = out / "config.json"
        old = read_json(cfg_path)
        if old is not None and old != c:
            raise ValueError("OUT_DIR has a different config. Use a new OUT_DIR for this ablation.")
        atomic_json(cfg_path, c)

        source = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        atomic_json(out / "source.json", {
            "commit": source.stdout.strip() if source.returncode == 0 else None,
            "pipeline_version": VERSION,
        })

        for i, gpu in enumerate(gpus):
            quiet_run(
                [python, "-c",
                 "import jax; assert jax.default_backend()=='gpu' and jax.local_device_count()==1; print(jax.devices())"],
                out / f"gpu{i}_probe.log", isolated_env(gpu),
                lambda i=i: update_bar(bars[i], ("xsub", "xset")[i], i,
                                       dict(phase="GPU probe", current=0, total=1)),
            )

        tests = Path(__file__).resolve().parent.parent / "test_preprocessing_corrected.py"
        quiet_run(
            [python, "-m", "pytest", "-q", str(tests)],
            out / "preprocessing_tests.log", isolated_env(),
            lambda: update_bar(bars[0], "xsub", 0,
                               dict(phase="Regression checks", current=0, total=1)),
        )

        status_path = out / "prepare_status.json"
        quiet_run(
            [python, "-m", MODULE + ".data", "--dataset", str(dataset), "--cache", str(cache),
             "--status", str(status_path), "--layout", raw_layout],
            out / "prepare.log", isolated_env(),
            lambda: [update_bar(
                b, p, i, read_json(status_path, dict(phase="Prepare cache", current=0, total=1))
            ) for i, (b, p) in enumerate(zip(bars, ("xsub", "xset")))],
        )

        if audit_first:
            quiet_run(
                [python, "-m", MODULE + ".audit", "--config", str(cfg_path),
                 "--output", str(out / "compute_audit.json")],
                out / "compute_audit.log", isolated_env(gpus[0]),
                lambda: update_bar(bars[0], "xsub", 0,
                                   dict(phase="Full FLOP audit", current=0, total=1)),
            )

        for protocol, gpu in zip(("xsub", "xset"), gpus):
            child_out = out / protocol
            child_out.mkdir(exist_ok=True)
            atomic_json(child_out / "status.json", dict(phase="Starting", current=0, total=1))
            stream = (out / f"{protocol}.log").open("a", buffering=1)
            streams.append(stream)
            cmd = [python, "-u", "-m", MODULE + ".worker", "--config", str(cfg_path),
                   "--protocol", protocol, "--cache", str(cache), "--outdir", str(out)]
            processes.append(subprocess.Popen(
                cmd, stdout=stream, stderr=subprocess.STDOUT,
                env=isolated_env(gpu), start_new_session=True,
            ))

        last_scores = None
        while True:
            done, statuses = 0, {}
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

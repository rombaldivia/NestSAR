"""Single-cell Kaggle entry point for NestSAR-R5-T16.

In order:
  1. checks that two GPUs are visible (Accelerator: GPU T4 x2);
  2. builds the model on CPU: exact parameter count, finite forward pass and
     finite gradients on a zero-padded batch (catches JAX/Flax differences
     before any GPU time is spent);
  3. finds an R4 canonical cache (or builds one from ntu120*.pkl);
  4. builds or validates the R5 hand cache derived from it (~1.4 GiB, a few
     minutes; every sample's R4 tokens are re-checked bit-for-bit);
  5. starts the dual-GPU launcher DETACHED from the notebook.

Stopping the cell never stops training; re-running it re-attaches, finished or
killed runs are reported, and interrupted runs resume from last.msgpack.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.nestsar_r5_t16 import DEFAULT_HAND_CACHE, DEFAULT_OUTDIR, MODEL_NAME, launch

REPO = Path(__file__).resolve().parents[2]
PREFERRED_R4_CACHES = ("NestSAR_R4_EMA_REP_CONSISTENCY_CACHE_REBUILT_v1", "NestSAR_canonical_cache")

PREFLIGHT = """
import sys, json
import jax, jax.numpy as jnp, numpy as np
from experiments.nestsar_r5_t16.config import validate_config
from experiments.nestsar_r5_t16.worker import EXPECTED_PARAMS, STRICT_MFLOPS, make_model
from experiments.nestsar_r5_t16.preprocessing import FRAMES, FEATURES
variant = sys.argv[1]
cfg = validate_config({"variant": variant})
m = make_model(cfg)
x = jnp.zeros((2, FRAMES, FEATURES))
p = m.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x, training=False)["params"]
n = int(sum(np.prod(a.shape) for a in jax.tree_util.tree_leaves(p)))
assert n == EXPECTED_PARAMS[variant], f"params {n} != expected {EXPECTED_PARAMS[variant]}"
xr = x.at[0].set(jax.random.normal(jax.random.PRNGKey(2), x.shape[1:]) * 0.3)
def loss(q):
    out = m.apply({"params": q}, xr, training=True, rngs={"dropout": jax.random.PRNGKey(3)})
    return jnp.mean(jax.nn.logsumexp(out["logits"], -1)) + jnp.mean(out["stream_logits"] ** 2)
v, g = jax.value_and_grad(loss)(p)
assert bool(jnp.isfinite(v)), "non-finite loss"
assert all(bool(jnp.isfinite(a).all()) for a in jax.tree_util.tree_leaves(g)), "non-finite gradients"
out = m.apply({"params": p}, xr, training=False)
assert bool(jnp.isfinite(out["logits"]).all()), "non-finite logits"
print(json.dumps({"params": n, "mflops": STRICT_MFLOPS[variant], "jax": jax.__version__}))
"""

R4_CACHE_CHECK = """
import sys
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
Dataset(sys.argv[1])
"""

R4_CACHE_BUILD = """
import sys
from experiments.nestsar_sm_all_t16.streaming.data import prepare
meta = prepare(sys.argv[1], sys.argv[2], sys.argv[2] + "_status.json")
print("R4 cache samples:", meta.get("samples"), meta.get("split_counts"))
"""

HAND_CACHE_BUILD = """
import sys, json
from experiments.nestsar_r5_t16.data import build
meta = build(sys.argv[1], sys.argv[2], hand_filter=sys.argv[3] if len(sys.argv) > 3 else "none",
             body_align=sys.argv[4] if len(sys.argv) > 4 else "none")
print(json.dumps({"samples": meta["samples"], "base_dir": meta["base_dir"],
                  "max_body_token_difference": meta.get("max_body_token_difference")}))
"""

HAND_CACHE_CHECK = """
import sys
from experiments.nestsar_r5_t16.data import Dataset
d = Dataset(sys.argv[1])
print(d.base_dir)
"""


def cpu_env():
    env = dict(os.environ)
    env.update(PYTHONPATH=str(REPO), JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    return env


def cpu_python(code, *args, capture=True, timeout=3600):
    return subprocess.run([sys.executable, "-c", code, *map(str, args)], cwd=REPO, env=cpu_env(),
                          capture_output=capture, text=True, timeout=timeout)


def last_line(text):
    lines = [x for x in (text or "").strip().splitlines() if x.strip()]
    return lines[-1] if lines else "?"


def visible_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in out.splitlines() if line.startswith("GPU ")]


GPU_PROBE = """
import json, jax, jax.numpy as jnp, flax, optax, psutil, numpy
assert jax.default_backend() == "gpu", f"JAX backend is {jax.default_backend()}, not gpu"
assert jax.local_device_count() == 1, jax.devices()
float(jnp.ones((256, 256)).sum().block_until_ready())
print(json.dumps({"jax": jax.__version__, "flax": flax.__version__, "optax": optax.__version__,
                  "device": str(jax.devices()[0])}))
"""


def gpu_runtime_ok(n_gpus):
    """JAX must see exactly one GPU per isolated worker; fail before building caches."""
    for gpu in range(n_gpus):
        env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS="cuda",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        r = subprocess.run([sys.executable, "-c", GPU_PROBE], cwd=REPO, env=env, capture_output=True,
                           text=True, timeout=600)
        if r.returncode != 0:
            raise SystemExit(f"JAX cannot use GPU{gpu} (no packages are installed automatically):\n"
                             + "\n".join((r.stderr or r.stdout or "").splitlines()[-15:]))
        print(f"  GPU{gpu} runtime OK: {last_line(r.stdout)}", flush=True)


def r4_cache_usable(path):
    r = cpu_python(R4_CACHE_CHECK, path, timeout=900)
    return r.returncode == 0, last_line(r.stderr)


def find_r4_cache(explicit, working, inputs):
    if explicit:
        ok, why = r4_cache_usable(explicit)
        if not ok:
            raise SystemExit(f"--r4-cache {explicit} is not usable: {why}")
        return explicit
    from experiments.nestsar_r5_t16.data import candidate_r4_caches
    roots = [r for r in (working, inputs) if Path(r).is_dir()]
    for d in candidate_r4_caches(roots, PREFERRED_R4_CACHES):
        ok, why = r4_cache_usable(str(d))
        print(f"  R4 cache {'OK' if ok else 'NO'}  {d}" + ("" if ok else f"  ({why})"), flush=True)
        if ok:
            return str(d)
    pkls = sorted(p for p in glob.glob(str(Path(inputs) / "**" / "*.pkl"), recursive=True)
                  if "120" in Path(p).name)
    if not pkls:
        raise SystemExit(f"No R4 cache under {working} or {inputs} and no ntu120*.pkl under {inputs}. "
                         "Add the NTU120 dataset as a notebook input.")
    target = Path(working) / "NestSAR_canonical_cache"
    print(f"No R4 cache found. Building it from {pkls[0]} -> {target} (several minutes)...", flush=True)
    r = cpu_python(R4_CACHE_BUILD, pkls[0], target, capture=False, timeout=6 * 3600)
    if r.returncode != 0:
        raise SystemExit("R4 cache build failed (see the error above).")
    ok, why = r4_cache_usable(str(target))
    if not ok:
        raise SystemExit(f"Built R4 cache is not usable: {why}")
    return str(target)


def default_hand_cache_name(r4_cache, hand_filter="none", body_align="none"):
    """One hand cache per R4 cache: the name carries a hash of the R4 manifest."""
    import hashlib
    manifest = json.loads((Path(r4_cache) / "manifest.json").read_text())
    key = json.dumps({"signature": manifest["signature"], "files": manifest["files"]}, sort_keys=True)
    suffix = ("" if hand_filter == "none" else f"_{hand_filter}") + ("" if body_align == "none" else f"_{body_align}")
    return f"{DEFAULT_HAND_CACHE}_{hashlib.sha256(key.encode()).hexdigest()[:8]}{suffix}"


def _dir_gib(path):
    total = 0
    for p in Path(path).rglob("*"):
        try:
            total += p.stat().st_size if p.is_file() else 0
        except OSError:
            pass
    return total / 2 ** 30


def check_disk_for_cache(hand_cache, body_align, working, r4_cache=None):
    """Fail early (with the exact cleanup command) when the new cache cannot fit."""
    import shutil
    from experiments.nestsar_r5_t16 import preprocessing as pp
    if (Path(hand_cache) / "manifest.json").is_file():
        return
    n = 128_000                                    # NTU-120 upper bound unless the R4 manifest says otherwise
    if r4_cache:
        try:
            n = int(json.loads((Path(r4_cache) / "manifest.json").read_text())["samples"])
        except (OSError, KeyError, ValueError):
            pass
    width = pp.FEATURES if body_align != "none" else pp.HAND_FEATURES
    need = n * pp.FRAMES * width * 4 / 2 ** 30 + 0.5
    Path(hand_cache).parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(Path(hand_cache).parent).free / 2 ** 30
    print(f"Disk: {free:.1f} GiB free, new cache needs ~{need:.1f} GiB", flush=True)
    if free >= need:
        return
    olds = [d for d in sorted(Path(working).iterdir()) if d.is_dir()
            and (d.name.startswith("NestSAR_R5") or (d.name.startswith("NestSAR_") and not (d / "manifest.json").is_file()))]
    partial = Path(hand_cache)
    if partial.is_dir() and not (partial / "manifest.json").is_file() and partial not in olds:
        olds.append(partial)                       # leftover of an interrupted build: usually the big one
    olds = [d for d in olds if d.name not in ("NestSAR_R5_ALIGNMIX_branch",)]
    lines = [f"  {_dir_gib(d):5.1f} GiB  rm -rf {d}" for d in olds]
    raise SystemExit(f"Not enough disk for the new cache ({free:.1f} GiB free, ~{need:.1f} GiB needed).\n"
                     "Delete caches of finished/stopped runs, e.g.:\n" + ("\n".join(lines) or "  (none found)"))


def ensure_hand_cache(r4_cache, hand_cache, hand_filter="none", body_align="none"):
    print(f"R5 hand cache: {hand_cache} (from {r4_cache})", flush=True)
    t0 = time.time()
    r = cpu_python(HAND_CACHE_BUILD, r4_cache, hand_cache, hand_filter, body_align, capture=True, timeout=6 * 3600)
    if r.returncode != 0:
        raise SystemExit("Hand cache build failed:\n" + "\n".join((r.stderr or "").splitlines()[-20:]))
    info = json.loads(last_line(r.stdout))
    print(f"  hand cache OK: {info['samples']:,} samples, R4 tokens re-checked "
          f"(max diff {info.get('max_body_token_difference', 0):.2g}), {time.time() - t0:.0f} s", flush=True)
    r = cpu_python(HAND_CACHE_CHECK, hand_cache, timeout=900)
    if r.returncode != 0:
        raise SystemExit("Hand cache check failed:\n" + "\n".join((r.stderr or "").splitlines()[-20:]))


def follow(log, offset, pid, proc):
    with open(log, "rb") as fh:
        fh.seek(offset)
        while True:
            chunk = fh.read()
            if chunk:
                sys.stdout.write(chunk.decode(errors="replace"))
                sys.stdout.flush()
            finished = (proc.poll() is not None) if proc is not None else not launch.alive(pid, launch.LAUNCHER)
            if finished:
                rest = fh.read()
                if rest:
                    sys.stdout.write(rest.decode(errors="replace"))
                    sys.stdout.flush()
                return proc.returncode if proc is not None else 0
            time.sleep(2)


def start_or_attach(launch_argv, outdir, follow_log=True):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    log, pid_file = out / "launcher.log", out / "launcher.pid"
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        pid = None
    if pid and launch.alive(pid, launch.LAUNCHER):
        print(f"\nLauncher already running (pid {pid}). Last lines of its log:", flush=True)
        print(launch.tail(log, 15), flush=True)
        if not follow_log:
            return 0
        print("--- following ---", flush=True)
        return follow(log, log.stat().st_size if log.exists() else 0, pid, None)
    offset = log.stat().st_size if log.exists() else 0
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONUNBUFFERED="1")
    with open(log, "ab") as fh:
        proc = subprocess.Popen([sys.executable, "-u", "-m", "experiments.nestsar_r5_t16.launch", *launch_argv],
                                cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    pid_file.write_text(str(proc.pid))
    print(f"\nLauncher started (pid {proc.pid}); log: {log}", flush=True)
    print("Stopping the cell does NOT stop training; re-running it re-attaches.\n", flush=True)
    if not follow_log:
        deadline = time.time() + 20
        while time.time() < deadline and proc.poll() is None:
            if log.exists() and "started pid" in log.read_text(errors="replace")[offset:]:
                break
            time.sleep(1)
        with open(log, "rb") as fh:
            fh.seek(offset)
            sys.stdout.write(fh.read().decode(errors="replace"))
            sys.stdout.flush()
        if proc.poll() is not None and proc.returncode != 0:
            return proc.returncode
        return 0
    return follow(log, offset, proc.pid, proc)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full")
    ap.add_argument("--protocols", default="xsub,xset")
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--r4-cache", default=None, help="R4 canonical cache dir (auto-detected if omitted)")
    ap.add_argument("--hand-cache", default=None, help=f"default: <working>/{DEFAULT_HAND_CACHE}")
    ap.add_argument("--working", default="/kaggle/working")
    ap.add_argument("--inputs", default="/kaggle/input")
    ap.add_argument("--reference-dir", default=None)
    ap.add_argument("--kill-epoch", type=int, default=10)
    ap.add_argument("--kill-margin-pp", type=float, default=1.0)
    ap.add_argument("--ignore-kill", action="store_true")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--heartbeat-minutes", type=float, default=10.0)
    ap.add_argument("--no-follow", action="store_true",
                    help="return after starting/attaching (the notebook monitor shows progress)")
    ap.add_argument("--cpu-smoke", action="store_true", help="local test only (no GPU)")
    ap.add_argument("--hand-filter", default="none", choices=["none", "hampel", "smooth", "sun", "sun_smooth"],
                    help="denoise the hand joints (builds/uses a separate hand cache)")
    ap.add_argument("--body-align", default="none", choices=["none", "yaw"],
                    help="rotate each clip so the torso faces +x (builds a separate, larger cache)")
    ap.add_argument("--extra-config", default="{}", help='JSON merged into config, e.g. \'{"aug_strength": 1.0, "prefetch_workers": 2}\'')
    a = ap.parse_args(argv)

    print(f"{MODEL_NAME}  variant={a.variant}  micro_batch={a.micro_batch}  out={a.outdir}", flush=True)
    if not a.cpu_smoke:
        gpus = visible_gpus()
        print("GPUs:", gpus, flush=True)
        n_needed = len([p for p in a.protocols.split(",") if p.strip()])
        if len(gpus) < n_needed:
            raise SystemExit(f"Need {n_needed} GPUs: Notebook settings -> Accelerator -> GPU T4 x2.")
        gpu_runtime_ok(n_needed)

    print("Checking the model on CPU (params, forward, gradients)...", flush=True)
    r = cpu_python(PREFLIGHT, a.variant, timeout=1800)
    if r.returncode != 0:
        raise SystemExit("Model check failed:\n" + "\n".join((r.stderr or "").splitlines()[-25:]))
    info = json.loads(last_line(r.stdout))
    print(f"Model OK: {info['params']:,} params, {info['mflops']:.2f} strict MFLOPs per clip "
          f"(R4-FMSE: 1,831,932 params, 60.87 MFLOPs), JAX {info['jax']}", flush=True)

    r4_cache = find_r4_cache(a.r4_cache, a.working, a.inputs)
    print("R4 CACHE:", r4_cache, flush=True)
    hand_cache = a.hand_cache or str(Path(a.working) / default_hand_cache_name(r4_cache, a.hand_filter, a.body_align))
    check_disk_for_cache(hand_cache, a.body_align, a.working, r4_cache)
    ensure_hand_cache(r4_cache, hand_cache, a.hand_filter, a.body_align)

    launch_argv = ["--cache", hand_cache, "--outdir", a.outdir, "--variant", a.variant,
                   "--protocols", a.protocols, "--micro-batch", str(a.micro_batch),
                   "--epochs", str(a.epochs), "--kill-epoch", str(a.kill_epoch),
                   "--kill-margin-pp", str(a.kill_margin_pp), "--working-root", a.working,
                   "--poll-seconds", str(a.poll_seconds), "--heartbeat-minutes", str(a.heartbeat_minutes)]
    if a.reference_dir:
        launch_argv += ["--reference-dir", a.reference_dir]
    if a.ignore_kill:
        launch_argv.append("--ignore-kill")
    if a.cpu_smoke:
        launch_argv.append("--cpu-smoke")
    extra = json.loads(a.extra_config)
    if a.hand_filter != "none":
        extra["hand_filter"] = a.hand_filter
    if a.body_align != "none":
        extra["body_align"] = a.body_align
    if extra:
        launch_argv += ["--extra-config", json.dumps(extra)]
    code = start_or_attach(launch_argv, a.outdir, follow_log=not a.no_follow)
    decision = launch.read_json(Path(a.outdir) / "kill_decision.json")
    if decision:
        print("kill_decision.json:", json.dumps(decision.get("delta_pp", {})), decision.get("decision"),
              flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main() or 0)

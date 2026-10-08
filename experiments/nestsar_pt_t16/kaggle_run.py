"""Single-cell Kaggle entry point for NestSAR-PT-T16.

Runs everything the notebook cell needs, in order:
  1. checks that two GPUs are visible (Accelerator: GPU T4 x2);
  2. builds the model on CPU and checks its exact parameter count;
  3. finds a compatible canonical NTU120 cache (or builds one from ntu120*.pkl);
  4. starts the dual-GPU launcher DETACHED from the notebook and streams its log.

Because the launcher is detached, stopping the cell (or losing the browser tab)
never stops training or the epoch-10 kill rule. Re-running is safe: a live
launcher is re-attached, finished or killed runs are reported, and interrupted
runs resume from their last checkpoint.
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

from experiments.nestsar_pt_t16 import MODEL_NAME, launch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = "NestSAR_R4_EMA_REP_CONSISTENCY_CACHE_REBUILT_v1"

PREFLIGHT = """
import sys
import jax, jax.numpy as jnp, numpy as np
from experiments.nestsar_pt_t16.model import NestSARPTT16
from experiments.nestsar_pt_t16.worker import EXPECTED_PARAMS, STRICT_MFLOPS
from experiments.nestsar_sm_all_t16.model import FRAMES, FEATURES
d = int(sys.argv[1])
if d not in EXPECTED_PARAMS:
    raise SystemExit(f"part_dim must be one of {sorted(EXPECTED_PARAMS)}")
m = NestSARPTT16(part_dim=d)
x = jnp.zeros((1, FRAMES, FEATURES))
p = m.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x, training=False)["params"]
n = int(sum(np.prod(a.shape) for a in jax.tree_util.tree_leaves(p)))
assert n == EXPECTED_PARAMS[d], f"params {n} != expected {EXPECTED_PARAMS[d]}"
print(n, STRICT_MFLOPS[d])
"""

CACHE_CHECK = """
import sys
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
Dataset(sys.argv[1])
"""

CACHE_BUILD = """
import sys
from experiments.nestsar_sm_all_t16.streaming.data import prepare
meta = prepare(sys.argv[1], sys.argv[2], sys.argv[2] + "_status.json")
print("cache samples:", meta.get("samples"), meta.get("split_counts"))
"""


def cpu_env():
    env = dict(os.environ)
    env.update(PYTHONPATH=str(REPO), JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
               PYTHONUNBUFFERED="1")
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


def cache_usable(path):
    r = cpu_python(CACHE_CHECK, path, timeout=900)
    return r.returncode == 0, last_line(r.stderr)


def find_cache(explicit, working, inputs):
    if explicit:
        ok, why = cache_usable(explicit)
        if not ok:
            raise SystemExit(f"--cache {explicit} is not usable: {why}")
        return explicit
    working = Path(working)
    cands = [working / DEFAULT_CACHE]
    for pattern in ("*/manifest.json", "*/*/manifest.json"):
        cands += sorted(Path(m).parent for m in glob.glob(str(working / pattern)))
    seen = set()
    for d in cands:
        if d in seen or not (d / "manifest.json").is_file() or not (d / "canonical.npy").is_file():
            continue
        seen.add(d)
        ok, why = cache_usable(str(d))
        print(f"  cache {'OK' if ok else 'NO'}  {d}" + ("" if ok else f"  ({why})"), flush=True)
        if ok:
            return str(d)
    pkls = sorted(p for p in glob.glob(str(Path(inputs) / "**" / "*.pkl"), recursive=True)
                  if "120" in Path(p).name)
    if not pkls:
        raise SystemExit(f"No compatible cache under {working} and no ntu120*.pkl under {inputs}. "
                         "Add the NTU120 dataset as a notebook input.")
    target = working / "NestSAR_PT_cache"
    print(f"No cache found. Building it from {pkls[0]} -> {target} (several minutes)...", flush=True)
    r = cpu_python(CACHE_BUILD, pkls[0], target, capture=False, timeout=6 * 3600)
    if r.returncode != 0:
        raise SystemExit("Cache build failed (see the error above).")
    ok, why = cache_usable(str(target))
    if not ok:
        raise SystemExit(f"Built cache is not usable: {why}")
    return str(target)


def follow(log, offset, pid, proc):
    """Stream the launcher log from `offset` until the launcher exits."""
    with open(log, "rb") as fh:
        fh.seek(offset)
        while True:
            chunk = fh.read()
            if chunk:
                sys.stdout.write(chunk.decode(errors="replace"))
                sys.stdout.flush()
            finished = (proc.poll() is not None) if proc is not None else not launch.alive(pid)
            if finished:
                rest = fh.read()
                if rest:
                    sys.stdout.write(rest.decode(errors="replace"))
                    sys.stdout.flush()
                return proc.returncode if proc is not None else 0
            time.sleep(2)


def start_or_attach(launch_argv, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    log, pid_file = out / "launcher.log", out / "launcher.pid"
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        pid = None
    if pid and launch.alive(pid):
        print(f"\nLauncher already running (pid {pid}). Last lines of its log:", flush=True)
        print(launch.tail(log, 15), flush=True)
        print("--- following ---", flush=True)
        return follow(log, log.stat().st_size if log.exists() else 0, pid, None)
    offset = log.stat().st_size if log.exists() else 0
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONUNBUFFERED="1")
    with open(log, "ab") as fh:
        proc = subprocess.Popen([sys.executable, "-u", "-m", "experiments.nestsar_pt_t16.launch",
                                 *launch_argv],
                                cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    pid_file.write_text(str(proc.pid))
    print(f"\nLauncher started (pid {proc.pid}); log: {log}", flush=True)
    print("Stopping this cell does NOT stop training. For progress bars, stop it and run the "
          "monitor cell (experiments/nestsar_pt_t16/README.md).\n", flush=True)
    return follow(log, offset, proc.pid, proc)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part-dim", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--outdir", default="/kaggle/working/NestSAR_PT_T16_v1")
    ap.add_argument("--cache", default=None, help="canonical cache dir (auto-detected if omitted)")
    ap.add_argument("--working", default="/kaggle/working")
    ap.add_argument("--inputs", default="/kaggle/input")
    ap.add_argument("--reference-dir", default=None)
    ap.add_argument("--kill-epoch", type=int, default=10)
    ap.add_argument("--ignore-kill", action="store_true")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--heartbeat-minutes", type=float, default=10.0)
    ap.add_argument("--cpu-smoke", action="store_true", help="local test only (no GPU)")
    ap.add_argument("--extra-config", default="{}", help="JSON merged into config (smoke tests)")
    a = ap.parse_args(argv)

    print(f"{MODEL_NAME}  part_dim={a.part_dim}  micro_batch={a.micro_batch}  out={a.outdir}", flush=True)

    # 1) GPUs
    if not a.cpu_smoke:
        gpus = visible_gpus()
        print("GPUs:", gpus, flush=True)
        if len(gpus) < 2:
            raise SystemExit("Need 2 GPUs: Notebook settings -> Accelerator -> GPU T4 x2.")

    # 2) Exact parameter check on CPU (does not touch the GPUs)
    print("Checking the model on CPU...", flush=True)
    r = cpu_python(PREFLIGHT, a.part_dim, timeout=1800)
    if r.returncode != 0:
        raise SystemExit("Model check failed:\n" + "\n".join((r.stderr or "").splitlines()[-25:]))
    n, mflops = r.stdout.split()[-2:]
    print(f"Model OK: {int(n):,} params, {float(mflops):.2f} strict MFLOPs per clip", flush=True)

    # 3) Canonical NTU120 cache
    cache = find_cache(a.cache, a.working, a.inputs)
    print("CACHE:", cache, flush=True)

    # 4) Detached dual-GPU launcher + live log
    launch_argv = ["--cache", cache, "--outdir", a.outdir, "--part-dim", str(a.part_dim),
                   "--micro-batch", str(a.micro_batch), "--epochs", str(a.epochs),
                   "--kill-epoch", str(a.kill_epoch), "--working-root", a.working,
                   "--poll-seconds", str(a.poll_seconds),
                   "--heartbeat-minutes", str(a.heartbeat_minutes)]
    if a.reference_dir:
        launch_argv += ["--reference-dir", a.reference_dir]
    if a.ignore_kill:
        launch_argv.append("--ignore-kill")
    if a.cpu_smoke:
        launch_argv += ["--cpu-smoke", "--extra-config", a.extra_config]
    code = start_or_attach(launch_argv, a.outdir)

    decision = launch.read_json(Path(a.outdir) / "kill_decision.json")
    if decision:
        print("kill_decision.json:", json.dumps(decision.get("delta_pp", {})),
              decision.get("decision"), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main() or 0)

#!/usr/bin/env python3
"""Two isolated GPU workers with notebook-safe persistent progress bars."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# Load the display helper by file path so an older notebook import of
# "experiments" cannot redirect this launch into a different checkout.
_helper = Path(__file__).resolve().parent / "streaming" / "notebook_progress.py"
_spec = importlib.util.spec_from_file_location("_nestsar_frozen_progress", _helper)
_progress = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_progress)
make_bars, update_bar = _progress.make_bars, _progress.update_bar


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def tail(path):
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-70:])
    except OSError:
        return "No worker log available."


def stop_workers(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(0.1)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", default="/kaggle/working/NestSAR_SM_ALL_T16_P2_R4")
    p.add_argument("--cache", default="/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3")
    p.add_argument("--output", default="/kaggle/working/NestSAR_R4_FROZEN_READOUT_v1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[128, 28, 42])
    p.add_argument("--split-seed", type=int, default=20260925)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--extract-batch", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.output).resolve()
    checkpoints = {p: Path(args.run_root).resolve() / p / "best.msgpack"
                   for p in ("xsub", "xset")}
    for checkpoint in checkpoints.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing R4 checkpoint: {checkpoint}")
    if not (Path(args.cache) / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing existing P2 v3 canonical cache: {args.cache}")
    protected = [Path(args.run_root).resolve(), Path(args.cache).resolve()]
    if any(root == p or root in p.parents or p in root.parents for p in protected):
        raise ValueError("Use a separate audit output directory.")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "launcher.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("This diagnostic is already running in this output directory.")
    processes, streams, bars = [], [], []
    repo = Path(__file__).resolve().parents[2]
    print("=" * 100)
    print("NESTSAR R4 — FROZEN READOUT / TEMPORAL AGGREGATION DIAGNOSTIC")
    print("A: refit original head | B: nonlinear descriptor readout | C: temporal contrasts")
    print(f"Frozen backbone | T16 | {args.epochs} probe epochs | probe seeds {args.seeds}")
    print("Selection: development SUBJECTS/SETUPS inside official TRAIN only")
    print("Official evaluation: scored after every probe has finished selection")
    print("XSUB -> GPU0 | XSET -> GPU1 | existing JAX/Flax/Optax; no installation")
    print("Exploratory: the frozen backbone already saw the probe development examples.")
    print("OUTPUT:", root)
    print("=" * 100, flush=True)
    try:
        # Keep the notebook kernel free of JAX/CUDA imports and allocations.
        bars = make_bars()
        for gpu, protocol in enumerate(("xsub", "xset")):
            out = root / protocol
            out.mkdir(exist_ok=True)
            status = out / "status.json"
            if status.exists():
                status.unlink()
            cmd = [sys.executable, "-u", "-m",
                   "experiments.nestsar_sm_all_t16.audit_frozen_readout",
                   "--protocol", protocol, "--checkpoint", str(checkpoints[protocol]),
                   "--cache", str(Path(args.cache).resolve()), "--output", str(out),
                   "--epochs", str(args.epochs), "--seeds", *map(str, args.seeds),
                   "--split-seed", str(args.split_seed),
                   "--batch-size", str(args.batch_size),
                   "--extract-batch", str(args.extract_batch),
                   "--learning-rate", str(args.learning_rate)]
            env = dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONPATH=str(repo) + os.pathsep + env.get("PYTHONPATH", ""))
            # Do not silently inherit a CPU backend from unrelated notebook cells.
            env.pop("JAX_PLATFORMS", None)
            env.pop("JAX_PLATFORM_NAME", None)
            stream = (root / f"{protocol}.log").open("a", buffering=1)
            streams.append(stream)
            processes.append(subprocess.Popen(cmd, cwd=repo, env=env, stdout=stream,
                                              stderr=subprocess.STDOUT,
                                              start_new_session=True, text=True))
        while True:
            for gpu, protocol in enumerate(("xsub", "xset")):
                status = read_json(root / protocol / "status.json")
                if status:
                    update_bar(bars[gpu], protocol, gpu, status)
            failed = [(p, process.returncode) for p, process in zip(("xsub", "xset"), processes)
                      if process.poll() is not None and process.returncode != 0]
            if failed:
                raise RuntimeError("\n\n".join(
                    f"{p.upper()} failed ({code}):\n{tail(root / (p + '.log'))}" for p, code in failed))
            if all(process.poll() is not None for process in processes):
                break
            time.sleep(0.5)
    finally:
        stop_workers(processes)
        for stream in streams:
            stream.close()
        for bar in bars:
            bar.close()
        lock.close()
    reports = {p: read_json(root / p / "summary.json") for p in ("xsub", "xset")}
    if any(report is None for report in reports.values()):
        raise RuntimeError("A worker exited without a completed summary.")
    summary = dict(protocols=reports, interpretation={})
    for name in ("B_vs_A", "C_vs_B", "C_vs_base"):
        summary["interpretation"][name] = dict(
            positive_for_all_probe_seeds_in_both_protocols=all(
                r["diagnosis"]["comparisons"][name]["all_seeds_positive"] for r in reports.values()),
            protocol_mean_gains_pp={p: r["diagnosis"]["comparisons"][name]["mean_gain_pp"]
                                    for p, r in reports.items()})
    summary["interpretation"]["proven_single_bottleneck"] = False
    summary["interpretation"]["note"] = (
        "Consistent gains identify an intervention worth confirming. Negative probes "
        "are inconclusive; probe seeds do not replace independent backbone seeds.")
    tmp = root / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2, allow_nan=False))
    tmp.replace(root / "summary.json")
    print("\n" + "=" * 100)
    print("COMPACT RESULTS (accuracy and gains in percentage points)")
    for protocol, result in reports.items():
        print(f"\n{protocol.upper()} baseline: {100 * result['baseline_accuracy']:.4f}%")
        for seed, record in result["seeds"].items():
            parts = [f"seed {seed}"]
            for variant, metrics in record["variants"].items():
                acc = 100 * metrics["official_evaluation"]["accuracy"]
                parts.append(f"{variant}={acc:.4f}% @E{metrics['selected_epoch']:02d}")
            print(" | ".join(parts))
        for name, stats in result["diagnosis"]["comparisons"].items():
            print(f"  {name}: mean {stats['mean_gain_pp']:+.4f} pp; "
                  f"all seeds positive={stats['all_seeds_positive']}")
        for message in result["diagnosis"]["interpretation"]:
            print(" ", message)
    print("\nSEND BACK:", root / "summary.json")
    print("Rerunning the same configuration resumes extraction/epochs or reuses completed probes.")


if __name__ == "__main__":
    main()

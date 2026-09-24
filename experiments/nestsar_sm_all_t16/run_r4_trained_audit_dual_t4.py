#!/usr/bin/env python3
from __future__ import annotations

"""Dual-T4 launcher for the read-only trained R4 bottleneck audit."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-root",
        default="/kaggle/working/NestSAR_SM_ALL_T16_P2_R4",
        help="Directory containing xsub/best.msgpack and xset/best.msgpack",
    )
    p.add_argument(
        "--cache",
        default="/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3",
    )
    p.add_argument(
        "--output",
        default="/kaggle/working/NestSAR_R4_TRAINED_AUDIT",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--rank-samples", type=int, default=512)
    p.add_argument("--rank-batch-size", type=int, default=64)
    p.add_argument("--ridge", type=float, default=1e-2)
    return p.parse_args()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def tail(path, n=50):
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "No log available."


def launch_worker(args, protocol, gpu, outdir, log):
    checkpoint = Path(args.run_root) / protocol / "best.msgpack"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.nestsar_sm_all_t16.audit_r4_trained",
        "--protocol", protocol,
        "--checkpoint", str(checkpoint),
        "--cache", str(args.cache),
        "--output", str(outdir),
        "--batch-size", str(args.batch_size),
        "--rank-samples", str(args.rank_samples),
        "--rank-batch-size", str(args.rank_batch_size),
        "--ridge", str(args.ridge),
    ]
    env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    stream = Path(log).open("w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return proc, stream


def update_bar(bar, protocol, gpu, status):
    if not status:
        bar.set_description_str(f"{protocol.upper()} G{gpu} waiting")
        bar.refresh()
        return
    phase = status.get("phase", "working")
    total = max(int(status.get("total", 1)), 1)
    tag = (phase, total)
    if getattr(bar, "_audit_tag", None) != tag:
        bar.reset(total=total)
        bar._audit_tag = tag
    bar.n = min(int(status.get("current", 0)), total)
    bar.set_description_str(f"{protocol.upper()} G{gpu} {phase}", refresh=False)
    postfix = {}
    if status.get("model_val") is not None:
        postfix["val"] = f"{100*float(status['model_val']):.4f}%"
    bar.set_postfix(postfix, refresh=False)
    bar.refresh()


def compact_summary(report):
    c = report["fast_weight_counterfactual_val_accuracy"]
    stages = report["stage_frozen_probes"]
    modules = report["trained_r4_state_utilization"]["modules"]
    return {
        "protocol": report["protocol"],
        "checkpoint_val_accuracy": report["checkpoint_val_accuracy"],
        "recomputed_val_accuracy": report["model_full_val_accuracy_recomputed"],
        "params": report["params"],
        "gflops": report["compute"]["gflops"],
        "counterfactual_accuracy": c,
        "counterfactual_delta_pp": report["fast_weight_counterfactual_delta_pp_vs_on"],
        "ridge_probe_accuracy": {
            k: stages[k]["ridge_probe_val_accuracy"] for k in stages
        },
        "fisher_between_over_within": {
            k: stages[k]["fisher_between_over_within"] for k in stages
        },
        "r4_extra_energy_fraction_rank3_4": {
            k: modules[k].get("mean_energy_fraction_rank3_4") for k in modules
        },
        "effective_rank": {
            k: modules[k]["mean_entropy_effective_rank"] for k in modules
        },
        "effective_fast_over_base_rms": {
            k: modules[k]["effective_fast_over_base_rms"] for k in modules
        },
        "eta": report["trained_r4_state_utilization"]["eta"],
        "alpha": report["trained_r4_state_utilization"]["alpha"],
    }


def main():
    args = parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    if not (cache / "manifest.json").is_file():
        raise FileNotFoundError(
            f"Expected existing P2 cache at {cache}. "
            "This audit never rebuilds preprocessing automatically."
        )

    print("=" * 120)
    print("NESTSAR R4 — TRAINED CHECKPOINT BOTTLENECK AUDIT")
    print("=" * 120)
    print(f"Run root : {args.run_root}")
    print(f"Cache    : {cache}")
    print(f"Output   : {root}")
    print("Read-only: YES")
    print("Training : NONE")
    print("GPU map  : XSUB -> GPU0 | XSET -> GPU1")
    print("=" * 120)

    bars = [
        tqdm(total=1, desc="XSUB G0 setup", position=0, leave=True, dynamic_ncols=True),
        tqdm(total=1, desc="XSET G1 setup", position=1, leave=True, dynamic_ncols=True),
    ]
    processes = []
    streams = []
    protocols = ("xsub", "xset")
    try:
        for gpu, protocol in enumerate(protocols):
            out = root / protocol
            out.mkdir(parents=True, exist_ok=True)
            proc, stream = launch_worker(
                args, protocol, gpu, out, root / f"{protocol}.log"
            )
            processes.append(proc)
            streams.append(stream)

        while any(p.poll() is None for p in processes):
            for i, protocol in enumerate(protocols):
                status = read_json(root / protocol / "status.json")
                update_bar(bars[i], protocol, i, status)
            time.sleep(0.5)

        for i, protocol in enumerate(protocols):
            status = read_json(root / protocol / "status.json")
            update_bar(bars[i], protocol, i, status)

        failures = []
        for protocol, proc in zip(protocols, processes):
            if proc.returncode:
                failures.append(
                    f"{protocol.upper()} exit={proc.returncode}\n"
                    + tail(root / f"{protocol}.log")
                )
        if failures:
            raise RuntimeError("\n\n".join(failures))

    finally:
        for stream in streams:
            stream.close()
        for bar in bars:
            bar.close()

    reports = {}
    for protocol in protocols:
        path = root / protocol / "trained_r4_audit.json"
        reports[protocol] = json.loads(path.read_text())

    summary = {p: compact_summary(r) for p, r in reports.items()}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 120)
    print("R4 TRAINED AUDIT — COMPACT SUMMARY")
    print("=" * 120)
    print(json.dumps(summary, indent=2))
    print("\nSEND BACK:")
    print(f"  {root / 'summary.json'}")
    print("and, if possible, the final printed compact summary.")


if __name__ == "__main__":
    main()

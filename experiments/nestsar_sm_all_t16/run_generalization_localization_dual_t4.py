#!/usr/bin/env python3
from __future__ import annotations

"""Dual-T4 notebook-safe launcher for the read-only R4 localization audit."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.nestsar_sm_all_t16.streaming.notebook_progress import make_bars, update_bar


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", default="/kaggle/working/NestSAR_SM_ALL_T16_P2_R4")
    p.add_argument("--cache", default="/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3")
    p.add_argument("--output", default="/kaggle/working/NestSAR_R4_GENERALIZATION_LOCALIZATION")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ridge", type=float, default=1e-2)
    return p.parse_args()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def tail(path, n=60):
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "No log available."


def launch_worker(args, protocol, gpu, outdir, logfile):
    checkpoint = Path(args.run_root) / protocol / "best.msgpack"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.nestsar_sm_all_t16.audit_generalization_localization",
        "--protocol", protocol,
        "--checkpoint", str(checkpoint),
        "--cache", str(args.cache),
        "--output", str(outdir),
        "--batch-size", str(args.batch_size),
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
    stream = Path(logfile).open("w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return proc, stream


def compact(report):
    stages = report["stage_generalization"]
    trans = report["transition_localization"]
    temporal = report["temporal_information_headroom"]
    return {
        "protocol": report["protocol"],
        "checkpoint_val_accuracy": report["checkpoint_val_accuracy"],
        "recomputed_model_val_accuracy": report["recomputed_model_val_accuracy"],
        "stage_train_ridge_accuracy": {
            k: stages[k]["train_ridge_accuracy"] for k in stages
        },
        "stage_val_ridge_accuracy": {
            k: stages[k]["val_ridge_accuracy"] for k in stages
        },
        "stage_generalization_gap_pp": {
            k: stages[k]["generalization_gap_pp"] for k in stages
        },
        "stage_val_over_train_fisher_ratio": {
            k: stages[k]["val_over_train_fisher_ratio"] for k in stages
        },
        "stage_train_val_centroid_cosine": {
            k: stages[k]["train_val_class_centroid_alignment"]["mean_cosine"]
            for k in stages
        },
        "transition_excess_train_gain_over_val_pp": {
            k: trans[k]["excess_train_gain_over_val_pp"] for k in trans
        },
        "validation_temporal_headroom_pp": {
            k: temporal[k]["val_temporal_headroom_pp"] for k in temporal
        },
        "gate_adaptivity": report["gate_adaptivity"],
        "automatic_localization": report["automatic_localization"],
    }


def main():
    args = parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    if not (Path(args.cache) / "manifest.json").is_file():
        raise FileNotFoundError(Path(args.cache) / "manifest.json")

    print("=" * 120)
    print("NESTSAR R4 — GENERALIZATION LOCALIZATION AUDIT")
    print("=" * 120)
    print(f"Run root : {args.run_root}")
    print(f"Cache    : {args.cache}")
    print(f"Output   : {root}")
    print("Read-only: YES")
    print("Training : NONE")
    print("GPU map  : XSUB -> GPU0 | XSET -> GPU1")
    print("=" * 120)

    bars = make_bars()
    protocols = ("xsub", "xset")
    processes, streams = [], []

    try:
        for gpu, protocol in enumerate(protocols):
            outdir = root / protocol
            outdir.mkdir(parents=True, exist_ok=True)
            proc, stream = launch_worker(
                args, protocol, gpu, outdir, root / f"{protocol}.log"
            )
            processes.append(proc)
            streams.append(stream)

        while any(p.poll() is None for p in processes):
            for i, protocol in enumerate(protocols):
                status = read_json(root / protocol / "status.json")
                if status is not None:
                    update_bar(bars[i], protocol, i, status)
            time.sleep(0.5)

        for i, protocol in enumerate(protocols):
            status = read_json(root / protocol / "status.json")
            if status is not None:
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
        reports[protocol] = json.loads(
            (root / protocol / "generalization_localization.json").read_text()
        )

    summary = {protocol: compact(report) for protocol, report in reports.items()}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 120)
    print("GENERALIZATION LOCALIZATION — COMPACT SUMMARY")
    print("=" * 120)
    print(json.dumps(summary, indent=2))
    print("\nSEND BACK:")
    print(root / "summary.json")


if __name__ == "__main__":
    main()

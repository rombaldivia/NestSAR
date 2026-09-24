#!/usr/bin/env python3
from __future__ import annotations

"""Dual-T4 notebook-safe launcher for cross-fitted causal adapter audit."""

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
    p.add_argument("--output", default="/kaggle/working/NestSAR_R4_ADAPTER_CROSSFIT")
    p.add_argument("--adapter-rank", type=int, default=8)
    p.add_argument("--adapter-seed", type=int, default=20260924)
    p.add_argument("--fold-seed", type=int, default=20260924)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--micro-batch", type=int, default=64)
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    return p.parse_args()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def tail(path, n=100):
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
        "experiments.nestsar_sm_all_t16.audit_adapter_crossfit",
        "--protocol", protocol,
        "--checkpoint", str(checkpoint),
        "--cache", str(args.cache),
        "--output", str(outdir),
        "--adapter-rank", str(args.adapter_rank),
        "--adapter-seed", str(args.adapter_seed),
        "--fold-seed", str(args.fold_seed),
        "--epochs", str(args.epochs),
        "--micro-batch", str(args.micro_batch),
        "--accumulation-steps", str(args.accumulation_steps),
        "--eval-batch", str(args.eval_batch),
        "--learning-rate", str(args.learning_rate),
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
    results = report["results"]
    return {
        "protocol": report["protocol"],
        "base_checkpoint_val_accuracy": report["base_checkpoint_val_accuracy"],
        "adapter_rank": report["adapter_rank"],
        "adapter_params_each": report["adapter_params_each"],
        "same_parameter_count": report["same_parameter_count"],
        "crossfit_accuracy": {
            loc: results[loc]["crossfit_accuracy"]
            for loc in results
        },
        "crossfit_gain_pp": {
            loc: results[loc]["crossfit_gain_pp"]
            for loc in results
        },
        "selected_epoch_on_A": {
            loc: results[loc]["selected_on_fold_a"]["best_epoch"]
            for loc in results
        },
        "selected_epoch_on_B": {
            loc: results[loc]["selected_on_fold_b"]["best_epoch"]
            for loc in results
        },
        "test_B_gain_when_selected_on_A_pp": {
            loc: results[loc]["selected_on_fold_a"][
                "heldout_gain_pp_vs_baseline_fold_b"
            ]
            for loc in results
        },
        "test_A_gain_when_selected_on_B_pp": {
            loc: results[loc]["selected_on_fold_b"][
                "heldout_gain_pp_vs_baseline_fold_a"
            ]
            for loc in results
        },
        "zero_init_equivalence_max_abs_logits": {
            loc: results[loc]["zero_init_equivalence_max_abs_logits"]
            for loc in results
        },
        "ranking_by_crossfit_gain": report["ranking_by_crossfit_gain"],
        "tied_winners": report["tied_winners"],
        "unique_winner": report["unique_winner"],
        "best_crossfit_gain_pp": report["best_crossfit_gain_pp"],
        "margin_over_runner_up_pp": report["margin_over_runner_up_pp"],
    }


def main():
    args = parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    if not (Path(args.cache) / "manifest.json").is_file():
        raise FileNotFoundError(Path(args.cache) / "manifest.json")

    print("=" * 120)
    print("NESTSAR R4 — CROSS-FITTED CAUSAL ADAPTER RECOVERY AUDIT")
    print("=" * 120)
    print(f"Run root   : {args.run_root}")
    print(f"Cache      : {args.cache}")
    print(f"Output     : {root}")
    print("Base model : FROZEN")
    print("Trainable  : one shared rank-8 adapter only")
    print("Locations  : pre-M4 | post-M4 | pre-G4 | post-G4")
    print("Training   : original NTU TRAIN split only")
    print("Selection  : validation fold A/B")
    print("Testing    : opposite validation fold")
    print("Cross-fit  : A selects -> test B | B selects -> test A")
    print("Early stop : NONE (fixed epochs)")
    print("GPU map    : XSUB -> GPU0 | XSET -> GPU1")
    print("=" * 120)

    bars = make_bars()
    protocols = ("xsub", "xset")
    processes = []
    streams = []

    try:
        for gpu, protocol in enumerate(protocols):
            outdir = root / protocol
            outdir.mkdir(parents=True, exist_ok=True)

            proc, stream = launch_worker(
                args,
                protocol,
                gpu,
                outdir,
                root / f"{protocol}.log",
            )
            processes.append(proc)
            streams.append(stream)

        while any(proc.poll() is None for proc in processes):
            for i, protocol in enumerate(protocols):
                status = read_json(root / protocol / "status.json")
                if status is not None:
                    update_bar(
                        bars[i],
                        protocol,
                        i,
                        status,
                    )
            time.sleep(0.5)

        for i, protocol in enumerate(protocols):
            status = read_json(root / protocol / "status.json")
            if status is not None:
                update_bar(
                    bars[i],
                    protocol,
                    i,
                    status,
                )

        failures = []

        for protocol, proc in zip(protocols, processes):
            if proc.returncode:
                failures.append(
                    f"{protocol.upper()} exit={proc.returncode}\n"
                    + tail(root / f"{protocol}.log")
                )

        if failures:
            raise RuntimeError(
                "\n\n".join(failures)
            )

    finally:
        for stream in streams:
            stream.close()

        for bar in bars:
            bar.close()

    reports = {}

    for protocol in protocols:
        path = (
            root
            / protocol
            / "adapter_crossfit_summary.json"
        )
        reports[protocol] = json.loads(
            path.read_text()
        )

    summary = {
        protocol: compact(report)
        for protocol, report in reports.items()
    }

    locations = (
        "pre_m4",
        "post_m4",
        "pre_g4",
        "post_g4",
    )

    mean_gain = {
        loc: (
            summary["xsub"]["crossfit_gain_pp"][loc]
            + summary["xset"]["crossfit_gain_pp"][loc]
        ) / 2.0
        for loc in locations
    }

    ordered = sorted(
        locations,
        key=lambda loc: mean_gain[loc],
        reverse=True,
    )

    best_mean = mean_gain[ordered[0]]
    tied = [
        loc
        for loc in locations
        if abs(mean_gain[loc] - best_mean) <= 1e-12
    ]

    summary["cross_protocol"] = {
        "xsub_unique_winner": summary["xsub"]["unique_winner"],
        "xset_unique_winner": summary["xset"]["unique_winner"],
        "same_unique_winner": (
            summary["xsub"]["unique_winner"] is not None
            and summary["xsub"]["unique_winner"]
            == summary["xset"]["unique_winner"]
        ),
        "mean_crossfit_gain_pp": mean_gain,
        "ranking_by_mean_crossfit_gain": ordered,
        "tied_mean_winners": tied,
        "unique_mean_winner": (
            tied[0]
            if len(tied) == 1
            else None
        ),
    }

    (root / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print("\n" + "=" * 120)
    print("CROSS-FITTED CAUSAL ADAPTER — COMPACT SUMMARY")
    print("=" * 120)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print("\nSEND BACK:")
    print(root / "summary.json")


if __name__ == "__main__":
    main()

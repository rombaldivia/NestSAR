#!/usr/bin/env python3
from __future__ import annotations

"""Dual-T4 notebook-safe launcher for the R4 temporal causality audit."""

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
    p.add_argument("--output", default="/kaggle/working/NestSAR_R4_TEMPORAL_CAUSALITY")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--shuffle-seed", type=int, default=20260924)
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
        "experiments.nestsar_sm_all_t16.audit_temporal_causality",
        "--protocol", protocol,
        "--checkpoint", str(checkpoint),
        "--cache", str(args.cache),
        "--output", str(outdir),
        "--batch-size", str(args.batch_size),
        "--shuffle-seed", str(args.shuffle_seed),
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
    variants = report["variants"]

    return {
        "protocol": report["protocol"],
        "checkpoint_val_accuracy": report["checkpoint_val_accuracy"],
        "recomputed_clean_val_accuracy": report["recomputed_clean_val_accuracy"],
        "model_accuracy": {
            name: variants[name]["model_accuracy"]
            for name in variants
        },
        "model_delta_pp_vs_clean": {
            name: variants[name]["model_delta_pp_vs_clean"]
            for name in variants
        },
        "prediction_agreement_with_clean": {
            name: variants[name]["prediction_agreement_with_clean"]
            for name in variants
        },
        "fixed_vs_broken": {
            name: {
                "fixed": variants[name]["clean_wrong_variant_correct_fixed"],
                "broken": variants[name]["clean_correct_variant_wrong_broken"],
            }
            for name in variants
        },
        "stage_temporal_ncm_delta_pp_vs_clean": {
            name: variants[name]["stage_temporal_ncm_delta_pp_vs_clean"]
            for name in variants
        },
        "stage_temporal_energy_ratio_vs_clean": {
            name: variants[name]["stage_temporal_energy_ratio_vs_clean"]
            for name in variants
        },
        "temporal_order_dependence": report["temporal_order_dependence"],
        "channel_dependence": report["channel_dependence"],
    }


def main():
    args = parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    if not (Path(args.cache) / "manifest.json").is_file():
        raise FileNotFoundError(Path(args.cache) / "manifest.json")

    print("=" * 120)
    print("NESTSAR R4 — TEMPORAL CAUSALITY AUDIT")
    print("=" * 120)
    print(f"Run root : {args.run_root}")
    print(f"Cache    : {args.cache}")
    print(f"Output   : {root}")
    print("Mode     : READ ONLY")
    print("Training : NONE")
    print("Raw reorder tests recompute exact T16 preprocessing")
    print("Variants : clean | reverse_raw | local_shuffle_raw | block_permute_raw")
    print("           pose_only | motion_only | motion_misaligned")
    print("GPU map  : XSUB -> GPU0 | XSET -> GPU1")
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
            / "temporal_causality_summary.json"
        )
        reports[protocol] = json.loads(
            path.read_text()
        )

    summary = {
        protocol: compact(report)
        for protocol, report in reports.items()
    }

    order_variants = (
        "reverse_raw",
        "local_shuffle_raw",
        "block_permute_raw",
    )

    summary["cross_protocol"] = {
        "mean_model_delta_pp_vs_clean": {
            variant: (
                summary["xsub"]["model_delta_pp_vs_clean"][variant]
                + summary["xset"]["model_delta_pp_vs_clean"][variant]
            ) / 2.0
            for variant in (
                "clean",
                "reverse_raw",
                "local_shuffle_raw",
                "block_permute_raw",
                "pose_only",
                "motion_only",
                "motion_misaligned",
            )
        },
        "mean_order_drop_pp": (
            summary["xsub"]["temporal_order_dependence"][
                "mean_model_accuracy_drop_pp"
            ]
            + summary["xset"]["temporal_order_dependence"][
                "mean_model_accuracy_drop_pp"
            ]
        ) / 2.0,
        "strongest_attenuation_transition": {
            "xsub": summary["xsub"]["temporal_order_dependence"][
                "strongest_order_sensitivity_attenuation_transition"
            ],
            "xset": summary["xset"]["temporal_order_dependence"][
                "strongest_order_sensitivity_attenuation_transition"
            ],
        },
        "same_strongest_attenuation_transition": (
            summary["xsub"]["temporal_order_dependence"][
                "strongest_order_sensitivity_attenuation_transition"
            ]
            == summary["xset"]["temporal_order_dependence"][
                "strongest_order_sensitivity_attenuation_transition"
            ]
        ),
        "mean_channel_drop_pp": {
            key: (
                summary["xsub"]["channel_dependence"][key]
                + summary["xset"]["channel_dependence"][key]
            ) / 2.0
            for key in (
                "pose_only_accuracy_drop_pp",
                "motion_only_accuracy_drop_pp",
                "pose_motion_misalignment_accuracy_drop_pp",
            )
        },
    }

    (root / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print("\n" + "=" * 120)
    print("TEMPORAL CAUSALITY — COMPACT SUMMARY")
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

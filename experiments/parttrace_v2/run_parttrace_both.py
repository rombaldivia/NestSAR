#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel launcher for NestSAR-HOPE Attention-Lite v2 PartTrace.

Mirrors the proven multi-GPU protocol strategy in nestsar.py:
- --protocol both
- auto-detect NVIDIA GPUs
- XSUB -> first visible GPU
- XSET -> second visible GPU when available
- parallel workers when protocols map to different GPUs
- sequential fallback when only one GPU is available

The model/training implementation stays in:
    nestsar_hope_attention_lite_parttrace_v2.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


HERE = Path(__file__).resolve().parent
TRAINER = HERE / "nestsar_hope_attention_lite_parttrace_v2.py"


def detect_nvidia_gpus(max_gpus: int = 0) -> List[Dict[str, object]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []

    if result.returncode != 0:
        return []

    gpus: List[Dict[str, object]] = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",", 2)]
        if len(parts) != 3:
            continue
        try:
            index = int(parts[0])
            memory_mib = int(float(parts[2]))
        except ValueError:
            continue
        gpus.append({
            "index": index,
            "name": parts[1],
            "memory_mib": memory_mib,
        })

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() not in ("", "-1"):
        tokens = [t.strip() for t in visible.split(",")]
        if all(t.isdigit() for t in tokens):
            allowed = {int(t) for t in tokens}
            gpus = [g for g in gpus if int(g["index"]) in allowed]

    if max_gpus > 0:
        gpus = gpus[:max_gpus]
    return gpus


def parse_gpu_map(value: str, gpus: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    if value == "auto":
        if not gpus:
            return {}
        if len(gpus) == 1:
            idx = int(gpus[0]["index"])
            return {"xsub": idx, "xset": idx}
        return {
            "xsub": int(gpus[0]["index"]),
            "xset": int(gpus[1]["index"]),
        }

    mapping: Dict[str, int] = {}
    for item in value.split(","):
        protocol, raw_index = item.split(":", 1)
        protocol = protocol.strip().lower()
        if protocol not in ("xsub", "xset"):
            raise ValueError(f"Invalid protocol in --gpu-map: {protocol}")
        mapping[protocol] = int(raw_index.strip())

    if set(mapping) != {"xsub", "xset"}:
        raise ValueError("--gpu-map must define both xsub and xset")

    available = {int(g["index"]) for g in gpus}
    missing = [idx for idx in mapping.values() if idx not in available]
    if missing:
        raise ValueError(
            f"--gpu-map references invisible GPUs {missing}; visible={sorted(available)}"
        )
    return mapping


def trainer_args_from_cli(argv: Sequence[str]) -> List[str]:
    """Remove launcher-only arguments before forwarding to the trainer."""
    launcher_with_value = {"--gpu-map", "--max-gpus", "--protocol"}
    launcher_flags = {"--allow-cpu", "--dry-run"}
    result: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in launcher_flags:
            i += 1
            continue
        if token in launcher_with_value:
            i += 2
            continue
        if any(token.startswith(name + "=") for name in launcher_with_value):
            i += 1
            continue
        result.append(token)
        i += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PartTrace v2 XSUB and XSET on separate GPUs when available.",
        add_help=False,
    )
    parser.add_argument("--protocol", choices=("xsub", "xset", "both"), default="both")
    parser.add_argument("--gpu-map", default="auto", help="auto or xsub:0,xset:1")
    parser.add_argument("--max-gpus", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, _ = parser.parse_known_args()

    if args.protocol in ("xsub", "xset"):
        forwarded = trainer_args_from_cli(sys.argv[1:])
        cmd = [sys.executable, "-u", str(TRAINER), "--protocol", args.protocol] + forwarded
        return subprocess.call(cmd)

    gpus = detect_nvidia_gpus(args.max_gpus)
    print(f"Visible NVIDIA GPUs: {len(gpus)}", flush=True)
    for gpu in gpus:
        print(
            f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory_mib']} MiB)",
            flush=True,
        )

    if not gpus and not args.allow_cpu:
        raise RuntimeError(
            "No NVIDIA GPU detected. Enable a Kaggle GPU accelerator or use --allow-cpu for a tiny test."
        )

    mapping = parse_gpu_map(args.gpu_map, gpus)
    parallel = bool(gpus) and mapping.get("xsub") != mapping.get("xset")

    print(
        f"Plan: XSUB->{mapping.get('xsub', 'CPU')} | "
        f"XSET->{mapping.get('xset', 'CPU')} | "
        f"{'PARALLEL' if parallel else 'SEQUENTIAL'}",
        flush=True,
    )

    forwarded = trainer_args_from_cli(sys.argv[1:])

    def command(protocol: str) -> List[str]:
        # Give each protocol its own output root automatically.
        return [
            sys.executable,
            "-u",
            str(TRAINER),
            "--protocol",
            protocol,
        ] + forwarded

    def environment(protocol: str) -> Dict[str, str]:
        env = os.environ.copy()
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(mapping[protocol])
        else:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env.setdefault("MALLOC_ARENA_MAX", "2")
        env["NESTSAR_PROTOCOL"] = protocol.upper()
        # Reserved for tqdm positioning in the trainer.
        env["NESTSAR_TQDM_POSITION"] = "0" if protocol == "xsub" else "1"
        return env

    if args.dry_run:
        for protocol in ("xsub", "xset"):
            print(
                f"{protocol.upper()} GPU={mapping.get(protocol, 'CPU')} :: "
                + " ".join(command(protocol)),
                flush=True,
            )
        return 0

    if parallel:
        # Inherit stdout/stderr directly so terminal control sequences and tqdm
        # can render in place instead of being destroyed by line-prefix piping.
        processes = {
            protocol: subprocess.Popen(command(protocol), env=environment(protocol))
            for protocol in ("xsub", "xset")
        }
        codes = {protocol: process.wait() for protocol, process in processes.items()}
    else:
        codes = {}
        for protocol in ("xsub", "xset"):
            codes[protocol] = subprocess.call(command(protocol), env=environment(protocol))

    failures = {p: code for p, code in codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"PartTrace workers failed: {failures}")

    print("XSUB and XSET completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

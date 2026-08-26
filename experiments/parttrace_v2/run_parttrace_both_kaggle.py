#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kaggle notebook launcher for parallel PartTrace XSUB/XSET training.

IMPORTANT: run this file with IPython `%run`, not `!python`.

Why: Kaggle/Jupyter captures terminal carriage returns from subprocesses as
separate output records. This launcher keeps worker stdout/stderr in log files
and renders exactly two notebook display rows (XSUB and XSET) using IPython
DisplayHandle updates.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from IPython.display import display


HERE = Path(__file__).resolve().parent
TRAINER = HERE / "nestsar_hope_attention_lite_parttrace_v2_patience.py"
LOG_ROOT = Path("/kaggle/working/NestSAR_PartTrace_v2_monitor")


def detect_nvidia_gpus(max_gpus: int = 0) -> List[Dict[str, object]]:
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
        gpus.append({"index": index, "name": parts[1], "memory_mib": memory_mib})

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
        if len(gpus) < 2:
            raise RuntimeError(
                "Parallel Kaggle runner requires two visible GPUs. "
                f"Detected {len(gpus)}."
            )
        return {"xsub": int(gpus[0]["index"]), "xset": int(gpus[1]["index"])}

    mapping: Dict[str, int] = {}
    for item in value.split(","):
        protocol, raw_index = item.split(":", 1)
        protocol = protocol.strip().lower()
        if protocol not in ("xsub", "xset"):
            raise ValueError(f"Invalid protocol in --gpu-map: {protocol}")
        mapping[protocol] = int(raw_index.strip())

    if set(mapping) != {"xsub", "xset"}:
        raise ValueError("--gpu-map must define xsub and xset")
    if mapping["xsub"] == mapping["xset"]:
        raise ValueError("Kaggle parallel runner requires different GPUs for XSUB and XSET")
    return mapping


def forwarded_args(argv: Sequence[str]) -> List[str]:
    launcher_with_value = {"--gpu-map", "--max-gpus", "--refresh-seconds"}
    result: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in launcher_with_value:
            i += 2
            continue
        if any(token.startswith(name + "=") for name in launcher_with_value):
            i += 1
            continue
        result.append(token)
        i += 1
    return result


def worker_env(gpu_index: int, protocol: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["MALLOC_ARENA_MAX"] = "2"
    env["NESTSAR_PROTOCOL"] = protocol.upper()
    # Worker progress is redirected to a file, never rendered directly.
    env["NESTSAR_TQDM_POSITION"] = "0"
    return env


def latest_status(path: Path, protocol: str) -> str:
    if not path.exists():
        return f"{protocol.upper()} INITIALIZING..."
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return f"{protocol.upper()} INITIALIZING..."

    # tqdm uses carriage returns; normal logs use newlines. Treat both equally.
    chunks = [x.strip() for x in re.split(r"[\r\n]+", data) if x.strip()]
    wanted = protocol.upper()

    for line in reversed(chunks):
        if line.startswith(f"{wanted} TRAIN") or line.startswith(f"{wanted} VAL"):
            return line
        if line.startswith(f"{wanted} COMPLETE"):
            return line

    # During dataset loading/JIT compilation, keep a compact state instead of
    # replaying verbose worker logs into the notebook.
    if any("split keys:" in line for line in chunks[-30:]):
        return f"{wanted} PREPARING BATCHES..."
    if any("XLA GFLOPs:" in line for line in chunks[-50:]):
        return f"{wanted} LOADING DATASET..."
    return f"{wanted} INITIALIZING..."


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Kaggle two-GPU PartTrace runner with exactly two live notebook rows."
    )
    ap.add_argument("--gpu-map", default="auto", help="auto or xsub:0,xset:1")
    ap.add_argument("--max-gpus", type=int, default=0)
    ap.add_argument("--refresh-seconds", type=float, default=0.5)
    args, _ = ap.parse_known_args()

    gpus = detect_nvidia_gpus(args.max_gpus)
    mapping = parse_gpu_map(args.gpu_map, gpus)

    print(
        f"PartTrace parallel | XSUB→GPU {mapping['xsub']} | "
        f"XSET→GPU {mapping['xset']} | workers=2",
        flush=True,
    )

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    logs = {
        "xsub": LOG_ROOT / "xsub_worker.log",
        "xset": LOG_ROOT / "xset_worker.log",
    }
    for path in logs.values():
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    common = forwarded_args(sys.argv[1:])
    # Ensure protocol is controlled only by this parent.
    filtered: List[str] = []
    i = 0
    while i < len(common):
        if common[i] == "--protocol":
            i += 2
            continue
        if common[i].startswith("--protocol="):
            i += 1
            continue
        filtered.append(common[i])
        i += 1
    common = filtered

    processes = {}
    handles = {}
    files = {}

    try:
        for protocol in ("xsub", "xset"):
            files[protocol] = logs[protocol].open("w", encoding="utf-8", buffering=1)
            cmd = [
                sys.executable,
                "-u",
                str(TRAINER),
                "--protocol",
                protocol,
            ] + common
            processes[protocol] = subprocess.Popen(
                cmd,
                env=worker_env(mapping[protocol], protocol),
                stdout=files[protocol],
                stderr=subprocess.STDOUT,
                text=True,
            )

        # Exactly two notebook output objects. Each update replaces its object.
        handles["xsub"] = display("XSUB INITIALIZING...", display_id=True)
        handles["xset"] = display("XSET INITIALIZING...", display_id=True)

        while True:
            alive = False
            for protocol in ("xsub", "xset"):
                status = latest_status(logs[protocol], protocol)
                handles[protocol].update(status)
                if processes[protocol].poll() is None:
                    alive = True
            if not alive:
                break
            time.sleep(max(0.1, args.refresh_seconds))

        codes = {p: proc.wait() for p, proc in processes.items()}
        for protocol in ("xsub", "xset"):
            final_status = latest_status(logs[protocol], protocol)
            if codes[protocol] != 0:
                final_status = f"{protocol.upper()} FAILED (exit={codes[protocol]}) — see {logs[protocol]}"
            handles[protocol].update(final_status)

        failures = {p: code for p, code in codes.items() if code != 0}
        if failures:
            raise RuntimeError(f"PartTrace workers failed: {failures}")
        return 0

    except KeyboardInterrupt:
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in processes.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if "xsub" in handles:
            handles["xsub"].update("XSUB INTERRUPTED")
        if "xset" in handles:
            handles["xset"].update("XSET INTERRUPTED")
        raise
    finally:
        for f in files.values():
            try:
                f.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

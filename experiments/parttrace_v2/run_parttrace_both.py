#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel launcher for NestSAR-HOPE Attention-Lite v2 PartTrace.

Kaggle/Jupyter-safe behavior:
- --protocol both
- auto-detect NVIDIA GPUs
- XSUB -> first visible GPU
- XSET -> second visible GPU when available
- parallel workers when protocols map to different GPUs
- sequential fallback when only one GPU is available
- worker stdout/stderr goes to log files
- parent renders ONLY TWO persistent status lines (XSUB + XSET)
  instead of letting two tqdm instances spam the notebook output

The launcher uses the patience-enabled trainer by default.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, TextIO


HERE = Path(__file__).resolve().parent
TRAINER = HERE / "nestsar_hope_attention_lite_parttrace_v2_patience.py"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


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
    launcher_with_value = {"--gpu-map", "--max-gpus", "--protocol", "--refresh-seconds"}
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


def _clean_text(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace("\x00", "")
    return text.strip()


def _latest_worker_status(log_path: Path, protocol: str, returncode) -> str:
    label = protocol.upper()
    if not log_path.exists():
        return f"{label} STARTING..."

    try:
        # Logs are small enough to read during training; tqdm refreshes are separated
        # by carriage returns, while ordinary prints use newlines.
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return f"{label} STARTING..."

    chunks = re.split(r"[\r\n]+", text)
    cleaned = [_clean_text(c) for c in chunks if _clean_text(c)]

    # Prefer the newest tqdm phase line.
    for line in reversed(cleaned):
        if (f"{label} TRAIN" in line) or (f"{label} VAL" in line):
            # Keep each dashboard row narrow enough that Kaggle does not wrap it.
            return line[:150]

    # Then prefer completion/error information.
    for line in reversed(cleaned):
        if "COMPLETE" in line or "Traceback" in line or "Error" in line or "Exception" in line:
            return f"{label} {line}"[:150]

    if returncode is None:
        return f"{label} INITIALIZING..."
    if returncode == 0:
        return f"{label} COMPLETE"
    return f"{label} FAILED (exit={returncode})"


class TwoLineDashboard:
    """Render exactly two mutable terminal rows without accumulating tqdm lines."""

    def __init__(self) -> None:
        self.started = False

    def render(self, xsub: str, xset: str) -> None:
        xsub = xsub[:150]
        xset = xset[:150]
        if not self.started:
            sys.stdout.write("\033[2K\r" + xsub + "\n")
            sys.stdout.write("\033[2K\r" + xset)
            sys.stdout.flush()
            self.started = True
            return

        # Cursor is at the end of the XSET row. Clear XSET, move to XSUB,
        # overwrite XSUB, then overwrite XSET. No new persistent lines.
        sys.stdout.write("\r\033[2K")
        sys.stdout.write("\033[1A\r\033[2K" + xsub)
        sys.stdout.write("\n\033[2K\r" + xset)
        sys.stdout.flush()

    def finish(self) -> None:
        if self.started:
            sys.stdout.write("\n")
            sys.stdout.flush()


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
    parser.add_argument("--refresh-seconds", type=float, default=0.35)
    args, _ = parser.parse_known_args()

    if args.refresh_seconds <= 0:
        raise ValueError("--refresh-seconds must be > 0")

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
        # Worker tqdm may write as much as it wants, but only into its private log.
        env["NESTSAR_TQDM_POSITION"] = "0"
        return env

    if args.dry_run:
        for protocol in ("xsub", "xset"):
            print(
                f"{protocol.upper()} GPU={mapping.get(protocol, 'CPU')} :: "
                + " ".join(command(protocol)),
                flush=True,
            )
        return 0

    log_root = Path(os.environ.get("NESTSAR_PARALLEL_LOGDIR", "/kaggle/working/parttrace_parallel_logs"))
    log_root.mkdir(parents=True, exist_ok=True)
    log_paths = {p: log_root / f"{p}.log" for p in ("xsub", "xset")}
    for path in log_paths.values():
        path.write_text("", encoding="utf-8")

    handles: Dict[str, TextIO] = {}
    processes: Dict[str, subprocess.Popen] = {}
    dashboard = TwoLineDashboard()

    try:
        if parallel:
            for protocol in ("xsub", "xset"):
                handle = log_paths[protocol].open("w", encoding="utf-8", buffering=1)
                handles[protocol] = handle
                processes[protocol] = subprocess.Popen(
                    command(protocol),
                    env=environment(protocol),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            while any(process.poll() is None for process in processes.values()):
                dashboard.render(
                    _latest_worker_status(log_paths["xsub"], "xsub", processes["xsub"].poll()),
                    _latest_worker_status(log_paths["xset"], "xset", processes["xset"].poll()),
                )
                time.sleep(args.refresh_seconds)

            codes = {protocol: process.wait() for protocol, process in processes.items()}
            dashboard.render(
                _latest_worker_status(log_paths["xsub"], "xsub", codes["xsub"]),
                _latest_worker_status(log_paths["xset"], "xset", codes["xset"]),
            )
            dashboard.finish()

        else:
            # One GPU: run each protocol sequentially. Keep the same clean dashboard.
            codes = {}
            for protocol in ("xsub", "xset"):
                handle = log_paths[protocol].open("w", encoding="utf-8", buffering=1)
                handles[protocol] = handle
                process = subprocess.Popen(
                    command(protocol),
                    env=environment(protocol),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                processes[protocol] = process
                while process.poll() is None:
                    other = "xset" if protocol == "xsub" else "xsub"
                    other_status = f"{other.upper()} WAITING..."
                    current_status = _latest_worker_status(log_paths[protocol], protocol, None)
                    dashboard.render(
                        current_status if protocol == "xsub" else other_status,
                        current_status if protocol == "xset" else other_status,
                    )
                    time.sleep(args.refresh_seconds)
                codes[protocol] = process.wait()
            dashboard.finish()

    except KeyboardInterrupt:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        dashboard.finish()
        print("Interrupted. Worker logs kept in:", log_root, flush=True)
        raise
    finally:
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass

    failures = {p: code for p, code in codes.items() if code != 0}
    if failures:
        print(f"Worker logs: {log_root}", flush=True)
        raise RuntimeError(f"PartTrace workers failed: {failures}")

    print(f"XSUB and XSET completed successfully. Logs: {log_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

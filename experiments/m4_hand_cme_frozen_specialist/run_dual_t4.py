#!/usr/bin/env python3
from __future__ import annotations

"""Dual-T4 launcher for the frozen Hand-M4/G4 CME specialist.

GPU0 -> XSUB
GPU1 -> XSET

The child workers write their tqdm output to per-protocol log files. Only this
parent process renders progress, so Kaggle/Jupyter sees exactly two persistent
progress rows instead of interleaved subprocess carriage-return/newline noise.

The default patience is intentionally 10 epochs. With the 40-epoch schedule and
10% LR warm-up (~4 epochs), a no-improvement specialist therefore receives
about six post-warmup epochs instead of stopping almost immediately after
warm-up.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from tqdm.auto import tqdm

ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
STATUS_RE = re.compile(r"(E(?P<epoch>\d+)/(?P<total>\d+)\s+[^\r\n]*)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--xsub-checkpoint",
        default="/kaggle/working/NestSAR_M4_LocalGlobal_HandM4G4Lite_T32_DualT4/xsub/best.msgpack",
    )
    p.add_argument(
        "--xset-checkpoint",
        default="/kaggle/working/NestSAR_M4_LocalGlobal_HandM4G4Lite_T32_DualT4/xset/best.msgpack",
    )
    p.add_argument(
        "--outdir",
        default="/kaggle/working/NestSAR_HandM4G4_CME_FrozenSpecialist_DualT4",
    )
    p.add_argument("--pairs", default="71-72,73-76,74-84,16-17,106-107,11-12,12-30,10-34")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--cache-batch-size", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--residual-scale", type=float, default=0.15)
    p.add_argument("--route-margin", type=float, default=0.20)
    p.add_argument("--target-weight", type=float, default=3.0)
    p.add_argument("--delta-l2", type=float, default=1e-4)
    p.add_argument("--preserve-kl-weight", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--min-learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.10)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--startup-stagger", type=float, default=15.0)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    return p.parse_args()


def gpu_inventory() -> str:
    r = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    return (r.stdout or r.stderr).strip()


def worker_cmd(args: argparse.Namespace, protocol: str, checkpoint: Path):
    return [
        sys.executable,
        "-u",
        "-m",
        "experiments.m4_hand_cme_frozen_specialist.train_gpu",
        "--dataset", args.dataset,
        "--protocol", protocol,
        "--checkpoint", str(checkpoint),
        "--outdir", args.outdir,
        "--pairs", args.pairs,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--cache-batch-size", str(args.cache_batch_size),
        "--hidden-dim", str(args.hidden_dim),
        "--dropout", str(args.dropout),
        "--residual-scale", str(args.residual_scale),
        "--route-margin", str(args.route_margin),
        "--target-weight", str(args.target_weight),
        "--delta-l2", str(args.delta_l2),
        "--preserve-kl-weight", str(args.preserve_kl_weight),
        "--learning-rate", str(args.learning_rate),
        "--min-learning-rate", str(args.min_learning_rate),
        "--warmup-fraction", str(args.warmup_fraction),
        "--weight-decay", str(args.weight_decay),
        "--grad-clip", str(args.grad_clip),
        "--ema-decay", str(args.ema_decay),
        "--seed", str(args.seed),
        # The worker bar is redirected to a file, so position zero keeps that
        # log free from cross-line cursor-control sequences.
        "--tqdm-position", "0",
        "--max-train-samples", str(args.max_train_samples),
        "--max-val-samples", str(args.max_val_samples),
    ]


def launch(args, protocol: str, physical_gpu: int, checkpoint: Path, log_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        worker_cmd(args, protocol, checkpoint),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, log_handle


def read_latest_status(log_path: Path):
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    text = ANSI_RE.sub("", text).replace("\r", "\n")
    matches = list(STATUS_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    status = " ".join(m.group(1).strip().split())
    return int(m.group("epoch")), int(m.group("total")), status


def display_value(epoch: int, status: str) -> float:
    # Intra-epoch placement is only visual; the worker remains the source of
    # truth. The final epoch summary contains ' VAL=' and lands on the integer.
    if " VAL=" in status:
        return float(epoch)
    if " VAL routed=" in status or " VAL routed" in status:
        return max(0.0, epoch - 0.10)
    if " TRAIN " in f" {status} ":
        return max(0.0, epoch - 0.55)
    return max(0.0, epoch - 0.75)


def tail_for_error(log_path: Path, lines: int = 40) -> str:
    if not log_path.is_file():
        return "<log file missing>"
    text = ANSI_RE.sub(
        "",
        log_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n"),
    )
    clean = [line for line in text.splitlines() if line.strip()]
    return "\n".join(clean[-lines:])


def main() -> None:
    args = parse_args()
    if args.patience < 7:
        raise ValueError(
            "For this rerun use patience >= 7. Recommended: 10, so the "
            "specialist receives a real post-warmup training window."
        )

    xsub_ckpt = Path(args.xsub_checkpoint)
    xset_ckpt = Path(args.xset_checkpoint)
    if not xsub_ckpt.is_file():
        raise FileNotFoundError(xsub_ckpt)
    if not xset_ckpt.is_file():
        raise FileNotFoundError(xset_ckpt)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    xsub_log = logdir / "xsub_worker.log"
    xset_log = logdir / "xset_worker.log"

    print("=" * 118)
    print("NESTSAR HAND-M4/G4 T32 -> FROZEN CME SPECIALIST | DUAL T4")
    print("GPU0 -> XSUB | GPU1 -> XSET | frozen champion; specialist parameters only")
    print("Console: parent-owned tqdm -> exactly two persistent rows, no child line breaks")
    print("=" * 118)
    print(gpu_inventory())
    print("-" * 118)
    print("XSUB checkpoint:", xsub_ckpt)
    print("XSET checkpoint:", xset_ckpt)
    print("Pairs (NTU one-based):", args.pairs)
    print(
        f"EPOCHS={args.epochs} PATIENCE={args.patience} BATCH={args.batch_size} "
        f"EVAL_BATCH={args.eval_batch_size} LR={args.learning_rate:g} "
        f"WARMUP={100*args.warmup_fraction:.1f}% HIDDEN={args.hidden_dim} "
        f"RESIDUAL={args.residual_scale:g}"
    )
    print("Worker logs:", logdir)
    print("=" * 118, flush=True)

    bars = {
        "xsub": tqdm(
            total=args.epochs,
            desc="XSUB CME",
            position=0,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.25,
            smoothing=0.05,
        ),
        "xset": tqdm(
            total=args.epochs,
            desc="XSET CME",
            position=1,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.25,
            smoothing=0.05,
        ),
    }
    bars["xsub"].set_postfix_str("STARTING GPU0", refresh=True)
    bars["xset"].set_postfix_str("WAITING GPU1", refresh=True)

    xsub, xsub_handle = launch(args, "xsub", 0, xsub_ckpt, xsub_log)
    stagger_until = time.time() + max(0.0, args.startup_stagger)
    while time.time() < stagger_until and xsub.poll() is None:
        status = read_latest_status(xsub_log)
        if status is not None:
            epoch, total, text = status
            bars["xsub"].total = total
            bars["xsub"].n = min(float(total), display_value(epoch, text))
            bars["xsub"].set_postfix_str(text, refresh=True)
        time.sleep(0.35)

    xset, xset_handle = launch(args, "xset", 1, xset_ckpt, xset_log)
    bars["xset"].set_postfix_str("STARTING GPU1", refresh=True)

    procs = {"xsub": xsub, "xset": xset}
    logs = {"xsub": xsub_log, "xset": xset_log}
    last_epoch = {"xsub": 0, "xset": 0}
    last_text = {"xsub": None, "xset": None}

    try:
        while any(proc.poll() is None for proc in procs.values()):
            for protocol in ("xsub", "xset"):
                status = read_latest_status(logs[protocol])
                if status is None:
                    continue
                epoch, total, text = status
                last_epoch[protocol] = max(last_epoch[protocol], epoch)
                if text == last_text[protocol]:
                    continue
                last_text[protocol] = text
                bar = bars[protocol]
                bar.total = total
                bar.n = min(float(total), display_value(epoch, text))
                bar.set_postfix_str(text, refresh=True)
            time.sleep(0.35)
    finally:
        rc_xsub = xsub.wait()
        rc_xset = xset.wait()
        xsub_handle.close()
        xset_handle.close()

        for protocol, rc in (("xsub", rc_xsub), ("xset", rc_xset)):
            bar = bars[protocol]
            status = read_latest_status(logs[protocol])
            if status is not None:
                epoch, _, text = status
                last_epoch[protocol] = max(last_epoch[protocol], epoch)
                if rc == 0:
                    # If early stopping occurred, finish the visual bar at the
                    # actual last epoch instead of falsely showing 40/40.
                    bar.total = max(last_epoch[protocol], 1)
                    bar.n = bar.total
                    bar.set_postfix_str(f"DONE | last={text}", refresh=True)
                else:
                    bar.set_postfix_str(f"FAILED rc={rc}", refresh=True)
            else:
                bar.set_postfix_str("DONE" if rc == 0 else f"FAILED rc={rc}", refresh=True)
            bar.close()

    if rc_xsub != 0 or rc_xset != 0:
        print("\nXSUB worker log tail:\n", tail_for_error(xsub_log), sep="")
        print("\nXSET worker log tail:\n", tail_for_error(xset_log), sep="")
        raise RuntimeError(f"CME workers failed: XSUB={rc_xsub}, XSET={rc_xset}")

    print()
    print("=" * 118)
    print("CME FROZEN-SPECIALIST RESULTS")
    print("=" * 118)
    for protocol in ("xsub", "xset"):
        path = outdir / f"result_{protocol}.json"
        if not path.is_file():
            print(protocol.upper(), "result missing:", path)
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"{protocol.upper()} | BASE={100*d['base_accuracy']:.6f}% | "
            f"BEST={100*d['best_accuracy']:.6f}% | GAIN={d['gain_pp']:+.6f} pp | "
            f"BEST_E={d['best_epoch']} | LAST_E={last_epoch[protocol]} | "
            f"SPECIALIST_PARAMS={d['specialist_params']:,} | "
            f"SPECIALIST_FLOPs={d['specialist_flops']}"
        )
    print("=" * 118)


if __name__ == "__main__":
    main()

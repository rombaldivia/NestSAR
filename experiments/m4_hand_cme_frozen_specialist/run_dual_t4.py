#!/usr/bin/env python3
from __future__ import annotations

"""Dual-T4 launcher for the frozen Hand-M4/G4 CME specialist.

GPU0 -> XSUB
GPU1 -> XSET

No parent JAX import and no parent dataset copy. Each worker owns one persistent
tqdm bar (position 0/1), so the notebook shows exactly two live rows total.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


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
    p.add_argument("--patience", type=int, default=5)
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


def worker_cmd(args: argparse.Namespace, protocol: str, checkpoint: Path, position: int):
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
        "--tqdm-position", str(position),
        "--max-train-samples", str(args.max_train_samples),
        "--max-val-samples", str(args.max_val_samples),
    ]


def launch(args, protocol: str, physical_gpu: int, checkpoint: Path, position: int):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    return subprocess.Popen(
        worker_cmd(args, protocol, checkpoint, position),
        env=env,
    )


def main() -> None:
    args = parse_args()
    xsub_ckpt = Path(args.xsub_checkpoint)
    xset_ckpt = Path(args.xset_checkpoint)
    if not xsub_ckpt.is_file():
        raise FileNotFoundError(xsub_ckpt)
    if not xset_ckpt.is_file():
        raise FileNotFoundError(xset_ckpt)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("NESTSAR HAND-M4/G4 T32 -> FROZEN CME SPECIALIST | DUAL T4")
    print("GPU0 -> XSUB | GPU1 -> XSET | frozen champion; specialist parameters only")
    print("Console: exactly two persistent tqdm rows; TRAIN/VAL reuse the same row")
    print("=" * 118)
    print(gpu_inventory())
    print("-" * 118)
    print("XSUB checkpoint:", xsub_ckpt)
    print("XSET checkpoint:", xset_ckpt)
    print("Pairs (NTU one-based):", args.pairs)
    print(
        f"EPOCHS={args.epochs} PATIENCE={args.patience} BATCH={args.batch_size} "
        f"EVAL_BATCH={args.eval_batch_size} LR={args.learning_rate:g} "
        f"HIDDEN={args.hidden_dim} RESIDUAL={args.residual_scale:g}"
    )
    print("=" * 118, flush=True)

    xsub = launch(args, "xsub", 0, xsub_ckpt, 0)
    time.sleep(max(0.0, args.startup_stagger))
    xset = launch(args, "xset", 1, xset_ckpt, 1)

    rc_xsub = xsub.wait()
    rc_xset = xset.wait()
    if rc_xsub != 0 or rc_xset != 0:
        raise RuntimeError(f"CME workers failed: XSUB={rc_xsub}, XSET={rc_xset}")

    print()
    print("=" * 118)
    print("CME FROZEN-SPECIALIST RESULTS")
    print("=" * 118)
    for protocol in ("xsub", "xset"):
        path = Path(args.outdir) / f"result_{protocol}.json"
        if not path.is_file():
            print(protocol.upper(), "result missing:", path)
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"{protocol.upper()} | BASE={100*d['base_accuracy']:.6f}% | "
            f"BEST={100*d['best_accuracy']:.6f}% | GAIN={d['gain_pp']:+.6f} pp | "
            f"E={d['best_epoch']} | SPECIALIST_PARAMS={d['specialist_params']:,} | "
            f"SPECIALIST_FLOPs={d['specialist_flops']}"
        )
    print("=" * 118)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER = HERE / "train_v31.py"


def parse_args():
    p = argparse.ArgumentParser(description="Run PartTrace v3.1 XSUB+XSET concurrently on two T4 GPUs")
    p.add_argument("--gpu-xsub", type=int, default=0)
    p.add_argument("--gpu-xset", type=int, default=1)
    p.add_argument("--dataset", default="auto")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)

    p.add_argument("--base-lr", type=float, default=4e-4)
    p.add_argument("--branch-lr", type=float, default=6e-4)
    p.add_argument("--controller-lr", type=float, default=1.5e-4)
    p.add_argument("--base-min-lr", type=float, default=1e-5)
    p.add_argument("--branch-min-lr", type=float, default=2e-5)
    p.add_argument("--controller-min-lr", type=float, default=5e-6)
    p.add_argument("--warmup-fraction", type=float, default=0.08)

    p.add_argument("--base-weight-decay", type=float, default=0.03)
    p.add_argument("--branch-weight-decay", type=float, default=0.04)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)

    p.add_argument("--predictive-loss-weight", type=float, default=0.10)
    p.add_argument("--parttrace-aux-weight", type=float, default=0.10)
    p.add_argument("--controller-kl-weight", type=float, default=0.02)
    p.add_argument("--freeze-branch-epochs", type=int, default=2)
    p.add_argument("--branch-ramp-epochs", type=int, default=4)

    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--outdir", default="/kaggle/working/NestSAR_PartTrace_v31")
    p.add_argument("--save-online", action="store_true")
    p.add_argument("--audit-first", action="store_true")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(probe.stdout.strip(), flush=True)

    common = [
        sys.executable, "-u", str(TRAINER),
        "--dataset", a.dataset,
        "--epochs", str(a.epochs),
        "--patience", str(a.patience),
        "--seed", str(a.seed),
        "--batch-size", str(a.batch_size),
        "--eval-batch-size", str(a.eval_batch_size),
        "--base-lr", str(a.base_lr),
        "--branch-lr", str(a.branch_lr),
        "--controller-lr", str(a.controller_lr),
        "--base-min-lr", str(a.base_min_lr),
        "--branch-min-lr", str(a.branch_min_lr),
        "--controller-min-lr", str(a.controller_min_lr),
        "--warmup-fraction", str(a.warmup_fraction),
        "--base-weight-decay", str(a.base_weight_decay),
        "--branch-weight-decay", str(a.branch_weight_decay),
        "--label-smoothing", str(a.label_smoothing),
        "--grad-clip", str(a.grad_clip),
        "--ema-decay", str(a.ema_decay),
        "--predictive-loss-weight", str(a.predictive_loss_weight),
        "--parttrace-aux-weight", str(a.parttrace_aux_weight),
        "--controller-kl-weight", str(a.controller_kl_weight),
        "--freeze-branch-epochs", str(a.freeze_branch_epochs),
        "--branch-ramp-epochs", str(a.branch_ramp_epochs),
        "--max-train-samples", str(a.max_train_samples),
        "--max-val-samples", str(a.max_val_samples),
        "--outdir", a.outdir,
    ]
    if a.save_online:
        common += ["--save-online"]
    if a.audit_first:
        common += ["--audit-first"]

    processes = {}
    for protocol, gpu in (("xsub", a.gpu_xsub), ("xset", a.gpu_xset)):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["MALLOC_ARENA_MAX"] = "2"
        cmd = common + ["--protocol", protocol]
        print(f"Launching {protocol.upper()} v3.1 on physical GPU {gpu}", flush=True)
        processes[protocol] = subprocess.Popen(cmd, env=env)

    codes = {p: proc.wait() for p, proc in processes.items()}
    bad = {p: c for p, c in codes.items() if c != 0}
    if bad:
        raise RuntimeError(f"PartTrace v3.1 workers failed: {bad}")
    print("XSUB + XSET V3.1 COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

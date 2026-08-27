#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kaggle/Jupyter dual-T4 launcher for PartTrace v3.2.

IMPORTANT: run with `%run`, not `!python`.

Child workers write stdout/stderr to private log files and atomic progress JSON.
The notebook parent owns exactly two tqdm widgets (XSUB/XSET), each reused for
TRAIN -> VAL -> next epoch. This avoids captured-carriage-return line spam.

Frame support:
- canonical chunk_size=4 / clip_size=8 => --frames must be divisible by 8
- T16 remains the reference configuration
- T32/T64/etc. are real runtime sequence lengths, not aliases/resampling flags
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER = HERE / "train_v32_frames.py"
DEFAULT_FRAMES = 16

try:
    from tqdm.notebook import tqdm
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="PartTrace v3.2 dual T4 with two no-break notebook tqdm bars")
    p.add_argument("--gpu-xsub", type=int, default=0)
    p.add_argument("--gpu-xset", type=int, default=1)
    p.add_argument("--dataset", default="auto")
    p.add_argument(
        "--frames", type=int, default=DEFAULT_FRAMES,
        help="Real input sequence length. Must be divisible by 8 (e.g. 16, 32, 64).",
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)

    p.add_argument("--part-dim", type=int, default=64)
    p.add_argument("--part-heads", type=int, default=4)
    p.add_argument("--global-dim", type=int, default=128)
    p.add_argument("--dense-dim", type=int, default=192,
                   help="PartTrace fusion Dense width (D_dense)")
    p.add_argument("--branch-dropout", type=float, default=0.12)

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
    p.add_argument("--outdir", default="/kaggle/working/NestSAR_PartTrace_v32")
    p.add_argument("--save-online", action="store_true")
    p.add_argument("--audit-first", action="store_true")
    p.add_argument("--refresh-seconds", type=float, default=0.30)
    p.add_argument("--progress-every", type=int, default=5)
    return p.parse_args()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _set_bar(bar, state, previous_phase):
    phase = str(state.get("phase", "initializing"))
    epoch = int(state.get("epoch", 0))
    epochs = int(state.get("epochs", 0))
    total = max(1, int(state.get("total", 1)))
    n = min(total, max(0, int(state.get("n", 0))))
    proto = str(state.get("protocol", "?")).upper()

    if phase in ("train", "val"):
        desc = f"{proto} {'TRAIN' if phase == 'train' else 'VAL'} E{epoch:03d}/{epochs}"
    elif phase == "initializing":
        desc = f"{proto} INITIALIZING"
    elif phase == "epoch_done":
        desc = f"{proto} E{epoch:03d}/{epochs} DONE"
    elif phase == "done":
        desc = f"{proto} COMPLETE"
    else:
        desc = f"{proto} {phase.upper()}"

    if phase != previous_phase or bar.total != total:
        bar.reset(total=total)
    bar.set_description_str(desc, refresh=False)
    bar.n = n

    postfix = {}
    if "accuracy" in state:
        postfix["acc"] = f"{100.0*float(state['accuracy']):.2f}%"

    best = float(state.get("best", -1.0))
    be = int(state.get("best_epoch", 0))
    postfix["BEST"] = f"{100.0*best:.2f}%@E{be:03d}" if best >= 0.0 and be > 0 else "--"

    if "loss" in state:
        postfix["loss"] = f"{float(state['loss']):.3f}"
    if "stale" in state:
        postfix["stale"] = f"{int(state['stale'])}/{int(state.get('patience', 0))}"
    bar.set_postfix(postfix, refresh=False)
    bar.refresh()
    return phase


def main() -> int:
    a = parse_args()
    if a.gpu_xsub == a.gpu_xset:
        raise ValueError("XSUB and XSET must use different physical GPUs")
    if a.frames < 8 or a.frames % 8:
        raise ValueError(
            f"--frames must be >=8 and divisible by 8 for chunk_size=4 / clip_size=8; got {a.frames}"
        )

    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    print("Visible NVIDIA GPUs:")
    print(probe.stdout.strip())
    print(
        f"Plan: XSUB->GPU{a.gpu_xsub} | XSET->GPU{a.gpu_xset} | "
        f"T={a.frames} Dpart={a.part_dim} Dglobal={a.global_dim} Ddense={a.dense_dim}"
    )

    root = Path(a.outdir)
    monitor = root / "_monitor"
    monitor.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable, "-u", str(TRAINER),
        "--dataset", a.dataset,
        "--frames", str(a.frames),
        "--epochs", str(a.epochs),
        "--patience", str(a.patience),
        "--seed", str(a.seed),
        "--batch-size", str(a.batch_size),
        "--eval-batch-size", str(a.eval_batch_size),
        "--part-dim", str(a.part_dim),
        "--part-heads", str(a.part_heads),
        "--global-dim", str(a.global_dim),
        "--dense-dim", str(a.dense_dim),
        "--branch-dropout", str(a.branch_dropout),
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
        "--progress-every", str(a.progress_every),
    ]
    if a.save_online:
        common.append("--save-online")
    if a.audit_first:
        common.append("--audit-first")

    workers = {}
    logs = {}
    progress_paths = {}
    for protocol, gpu in (("xsub", a.gpu_xsub), ("xset", a.gpu_xset)):
        progress = monitor / f"{protocol}.json"
        log_path = monitor / f"{protocol}.log"
        progress.unlink(missing_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["MALLOC_ARENA_MAX"] = "2"
        env["NESTSAR_FRAMES"] = str(a.frames)
        cmd = common + [
            "--protocol", protocol,
            "--progress-json", str(progress),
        ]
        workers[protocol] = subprocess.Popen(
            cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True
        )
        logs[protocol] = log_handle
        progress_paths[protocol] = progress

    bars = {
        "xsub": tqdm(total=1, desc="XSUB INITIALIZING", position=0, leave=True, dynamic_ncols=True),
        "xset": tqdm(total=1, desc="XSET INITIALIZING", position=1, leave=True, dynamic_ncols=True),
    }
    phases = {"xsub": None, "xset": None}

    try:
        while True:
            all_done = True
            for protocol in ("xsub", "xset"):
                proc = workers[protocol]
                state = _read_json(progress_paths[protocol])
                if state is not None:
                    phases[protocol] = _set_bar(bars[protocol], state, phases[protocol])
                if proc.poll() is None:
                    all_done = False
                elif state is None or state.get("phase") != "done":
                    bars[protocol].set_description_str(
                        f"{protocol.upper()} {'FAILED' if proc.returncode else 'COMPLETE'}"
                    )
                    bars[protocol].refresh()
            if all_done:
                break
            time.sleep(max(0.10, a.refresh_seconds))
    except KeyboardInterrupt:
        for proc in workers.values():
            if proc.poll() is None:
                proc.terminate()
        raise
    finally:
        for handle in logs.values():
            handle.close()
        for bar in bars.values():
            bar.close()

    codes = {p: proc.returncode for p, proc in workers.items()}
    bad = {p: c for p, c in codes.items() if c != 0}
    if bad:
        print(f"Worker failure: {bad}")
        print(f"Logs: {monitor}")
        raise RuntimeError(f"PartTrace v3.2 workers failed: {bad}")

    print(f"XSUB + XSET V3.2 COMPLETE | T={a.frames} | results: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

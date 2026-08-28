#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Notebook-native dual-T4 launcher for NestSAR v3.5.

Matches the latest validated Kaggle method:
- XSUB -> physical GPU0
- XSET -> physical GPU1
- both workers run concurrently
- one process-visible GPU per worker
- workers write private logs/progress JSON
- notebook parent owns two persistent tqdm rows
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
TRAINER = HERE / "train_v35_t4.py"

try:
    from tqdm.notebook import tqdm
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="NestSAR v3.5 dual-T4 Kaggle launcher")
    p.add_argument("--gpu-xsub", type=int, default=0)
    p.add_argument("--gpu-xset", type=int, default=1)
    p.add_argument("--dataset", default="auto")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--base-checkpoint", default="auto",
                   help="auto, exact path, or path containing {protocol}")
    p.add_argument("--allow-scratch", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)

    p.add_argument("--bridge-dim", type=int, default=32)
    p.add_argument("--local-stream-dim", type=int, default=16)
    p.add_argument("--memory-dim", type=int, default=32)
    p.add_argument("--fine-dim", type=int, default=24)
    p.add_argument("--readout-tokens", type=int, default=8)
    p.add_argument("--readout-heads", type=int, default=4)
    p.add_argument("--dense-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.08)
    p.add_argument("--stream-reweight-strength", type=float, default=0.08)

    p.add_argument("--base-lr", type=float, default=7.5e-5)
    p.add_argument("--new-lr", type=float, default=5e-4)
    p.add_argument("--base-min-lr", type=float, default=3e-6)
    p.add_argument("--new-min-lr", type=float, default=2e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.06)
    p.add_argument("--base-weight-decay", type=float, default=0.015)
    p.add_argument("--new-weight-decay", type=float, default=0.03)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997)

    p.add_argument("--predictive-loss-weight", type=float, default=0.03)
    p.add_argument("--memory-aux-warmup-weight", type=float, default=0.45)
    p.add_argument("--memory-aux-final-weight", type=float, default=0.15)
    p.add_argument("--diversity-loss-weight", type=float, default=0.03)
    p.add_argument("--stream-kl-weight", type=float, default=0.01)
    p.add_argument("--freeze-base-epochs", type=int, default=3)
    p.add_argument("--base-unfreeze-ramp-epochs", type=int, default=3)
    p.add_argument("--freeze-branch-epochs", type=int, default=2)
    p.add_argument("--branch-ramp-epochs", type=int, default=4)

    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--outdir",
                   default="/kaggle/working/NestSAR_CrossStreamMemory_v35_DualT4")
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

    if phase == "baseline":
        desc = f"{proto} PRETRAIN BASELINE"
    elif phase == "train":
        desc = f"{proto} TRAIN E{epoch:03d}/{epochs}"
    elif phase == "val":
        desc = f"{proto} VAL E{epoch:03d}/{epochs}"
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
        postfix["acc"] = f"{100*float(state['accuracy']):.2f}%"
    if "memory_accuracy" in state:
        postfix["MEM"] = f"{100*float(state['memory_accuracy']):.2f}%"
    if "macro_accuracy" in state:
        postfix["macro"] = f"{100*float(state['macro_accuracy']):.2f}%"
    best = float(state.get("best", -1.0))
    best_epoch = int(state.get("best_epoch", 0))
    postfix["BEST"] = (
        f"{100*best:.2f}%@E{best_epoch:03d}" if best >= 0 else "--"
    )
    if "loss" in state:
        postfix["loss"] = f"{float(state['loss']):.3f}"
    if "gate" in state:
        postfix["gate"] = f"{float(state['gate']):.3f}"
    if "qoverlap" in state:
        postfix["Qov"] = f"{float(state['qoverlap']):.3f}"
    if "branch_scale" in state:
        postfix["bs"] = f"{float(state['branch_scale']):.2f}"
    if "base_grad_scale" in state:
        postfix["bg"] = f"{float(state['base_grad_scale']):.2f}"
    if "stale" in state:
        postfix["stale"] = (
            f"{int(state['stale'])}/{int(state.get('patience', 0))}"
        )
    bar.set_postfix(postfix, refresh=False)
    bar.refresh()
    return phase


def _tail(path: Path, n=120):
    try:
        lines = path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:
        return f"<could not read {path}: {exc}>"


def main() -> int:
    a = parse_args()
    if a.gpu_xsub == a.gpu_xset:
        raise ValueError("XSUB and XSET must use different physical GPUs")
    if a.frames != 16:
        raise ValueError("v3.5 is intentionally T16")

    probe = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "nvidia-smi failed. Start this notebook with Kaggle GPU accelerator."
        )
    visible_lines = [x for x in probe.stdout.splitlines() if x.strip()]
    if len(visible_lines) < 2:
        raise RuntimeError(
            f"Dual-T4 run requires two GPUs; nvidia-smi reported:\n{probe.stdout}"
        )

    print("=" * 120)
    print("NESTSAR v3.5 — CROSS-STREAM MULTI-RESOLUTION MEMORY — DUAL T4")
    print("=" * 120)
    print("Accuracy-first: exact Attention-Lite + 4-stream bridge + 640-token memory")
    print("Visible NVIDIA GPUs:")
    print(probe.stdout.strip())
    print(
        f"Plan: XSUB->GPU{a.gpu_xsub} | XSET->GPU{a.gpu_xset} | "
        f"T16 bridge D{a.bridge_dim} local D{a.local_stream_dim} "
        f"memory D{a.memory_dim}/D{a.fine_dim} K={a.readout_tokens} Ddense={a.dense_dim}"
    )
    print(
        f"Per-worker batch={a.batch_size} eval={a.eval_batch_size} | "
        f"seed={a.seed} | pretrained required={not a.allow_scratch}"
    )
    print("=" * 120)

    root = Path(a.outdir)
    monitor = root / "_monitor"
    monitor.mkdir(parents=True, exist_ok=True)

    def worker_cmd(protocol: str, progress: Path):
        checkpoint = a.base_checkpoint.format(protocol=protocol)
        cmd = [
            sys.executable, "-u", str(TRAINER),
            "--protocol", protocol,
            "--dataset", a.dataset,
            "--frames", "16",
            "--base-checkpoint", checkpoint,
            "--epochs", str(a.epochs),
            "--patience", str(a.patience),
            "--seed", str(a.seed),
            "--batch-size", str(a.batch_size),
            "--eval-batch-size", str(a.eval_batch_size),
            "--bridge-dim", str(a.bridge_dim),
            "--local-stream-dim", str(a.local_stream_dim),
            "--memory-dim", str(a.memory_dim),
            "--fine-dim", str(a.fine_dim),
            "--readout-tokens", str(a.readout_tokens),
            "--readout-heads", str(a.readout_heads),
            "--dense-dim", str(a.dense_dim),
            "--dropout", str(a.dropout),
            "--stream-reweight-strength", str(a.stream_reweight_strength),
            "--base-lr", str(a.base_lr),
            "--new-lr", str(a.new_lr),
            "--base-min-lr", str(a.base_min_lr),
            "--new-min-lr", str(a.new_min_lr),
            "--warmup-fraction", str(a.warmup_fraction),
            "--base-weight-decay", str(a.base_weight_decay),
            "--new-weight-decay", str(a.new_weight_decay),
            "--label-smoothing", str(a.label_smoothing),
            "--grad-clip", str(a.grad_clip),
            "--ema-decay", str(a.ema_decay),
            "--predictive-loss-weight", str(a.predictive_loss_weight),
            "--memory-aux-warmup-weight", str(a.memory_aux_warmup_weight),
            "--memory-aux-final-weight", str(a.memory_aux_final_weight),
            "--diversity-loss-weight", str(a.diversity_loss_weight),
            "--stream-kl-weight", str(a.stream_kl_weight),
            "--freeze-base-epochs", str(a.freeze_base_epochs),
            "--base-unfreeze-ramp-epochs", str(a.base_unfreeze_ramp_epochs),
            "--freeze-branch-epochs", str(a.freeze_branch_epochs),
            "--branch-ramp-epochs", str(a.branch_ramp_epochs),
            "--max-train-samples", str(a.max_train_samples),
            "--max-val-samples", str(a.max_val_samples),
            "--outdir", a.outdir,
            "--progress-json", str(progress),
            "--progress-every", str(a.progress_every),
        ]
        if a.allow_scratch:
            cmd.append("--allow-scratch")
        if a.save_online:
            cmd.append("--save-online")
        if a.audit_first:
            cmd.append("--audit-first")
        return cmd

    workers = {}
    logs = {}
    log_paths = {}
    progress_paths = {}
    for protocol, gpu in (("xsub", a.gpu_xsub), ("xset", a.gpu_xset)):
        progress = monitor / f"{protocol}.json"
        log_path = monitor / f"{protocol}.log"
        progress.unlink(missing_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["JAX_PLATFORMS"] = "gpu"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["MALLOC_ARENA_MAX"] = "2"
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        env["PYTHONUNBUFFERED"] = "1"
        workers[protocol] = subprocess.Popen(
            worker_cmd(protocol, progress),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logs[protocol] = log_handle
        log_paths[protocol] = log_path
        progress_paths[protocol] = progress

    bars = {
        "xsub": tqdm(
            total=1, desc="XSUB INITIALIZING", position=0,
            leave=True, dynamic_ncols=True
        ),
        "xset": tqdm(
            total=1, desc="XSET INITIALIZING", position=1,
            leave=True, dynamic_ncols=True
        ),
    }
    phases = {"xsub": None, "xset": None}

    try:
        while True:
            all_done = True
            for protocol in ("xsub", "xset"):
                proc = workers[protocol]
                state = _read_json(progress_paths[protocol])
                if state is not None:
                    phases[protocol] = _set_bar(
                        bars[protocol], state, phases[protocol]
                    )
                if proc.poll() is None:
                    all_done = False
                elif state is None or state.get("phase") != "done":
                    bars[protocol].set_description_str(
                        f"{protocol.upper()} "
                        f"{'FAILED' if proc.returncode else 'COMPLETE'}"
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
        for protocol in bad:
            print(
                "\n" + "=" * 48 +
                f" {protocol.upper()} LOG TAIL " +
                "=" * 48
            )
            print(_tail(log_paths[protocol]))
        raise RuntimeError(f"NestSAR v3.5 dual-T4 workers failed: {bad}")

    print(f"XSUB + XSET V3.5 DUAL-T4 COMPLETE | results: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

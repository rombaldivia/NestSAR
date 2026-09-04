#!/usr/bin/env python3
from __future__ import annotations

"""Dual-T4 launcher: GPU0 XSUB, GPU1 XSET.

Child workers never render tqdm. They write STATUS lines to log files. This
parent process owns exactly two persistent notebook progress rows and updates
them in place through preprocessing, training and validation.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--outdir", default="/kaggle/working/NestSAR_Distal_SM_Specialist_T16_DualT4")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=8e-4)
    p.add_argument("--min-learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.08)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--consistency-weight", type=float, default=0.05)
    p.add_argument("--consistency-temperature", type=float, default=1.0)
    p.add_argument("--spatial-dim", type=int, default=16)
    p.add_argument("--model-dim", type=int, default=64)
    p.add_argument("--controller-dim", type=int, default=16)
    p.add_argument("--fast-rank", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--sm-residual-scale", type=float, default=0.08)
    p.add_argument("--jitter-max-shift", type=int, default=1)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--eval-progress-every", type=int, default=10)
    p.add_argument("--preprocess-progress-every", type=int, default=2500)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--startup-stagger", type=float, default=10.0)
    return p.parse_args()


def gpu_inventory():
    r = subprocess.run([
        "nvidia-smi", "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()


def worker_cmd(args, protocol):
    return [
        sys.executable, "-u", "-m", "experiments.distal_sm_specialist_t16.train_gpu",
        "--dataset", args.dataset,
        "--protocol", protocol,
        "--outdir", args.outdir,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--learning-rate", str(args.learning_rate),
        "--min-learning-rate", str(args.min_learning_rate),
        "--warmup-fraction", str(args.warmup_fraction),
        "--weight-decay", str(args.weight_decay),
        "--label-smoothing", str(args.label_smoothing),
        "--grad-clip", str(args.grad_clip),
        "--ema-decay", str(args.ema_decay),
        "--consistency-weight", str(args.consistency_weight),
        "--consistency-temperature", str(args.consistency_temperature),
        "--spatial-dim", str(args.spatial_dim),
        "--model-dim", str(args.model_dim),
        "--controller-dim", str(args.controller_dim),
        "--fast-rank", str(args.fast_rank),
        "--dropout", str(args.dropout),
        "--sm-residual-scale", str(args.sm_residual_scale),
        "--jitter-max-shift", str(args.jitter_max_shift),
        "--seed", str(args.seed),
        "--progress-every", str(args.progress_every),
        "--eval-progress-every", str(args.eval_progress_every),
        "--preprocess-progress-every", str(args.preprocess_progress_every),
        "--max-train-samples", str(args.max_train_samples),
        "--max-val-samples", str(args.max_val_samples),
        "--audit-first",
    ]


def launch(args, protocol, gpu, log_path):
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", ""),
    })
    handle = log_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        worker_cmd(args, protocol), env=env,
        stdout=handle, stderr=subprocess.STDOUT, text=True,
    )
    return proc, handle


def parse_status_line(line):
    line = line.strip()
    if not line.startswith("STATUS|"):
        return None
    d = {}
    for part in line.split("|")[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
    try:
        for k in ("epoch", "epochs", "done", "total"):
            d[k] = int(d.get(k, 0))
    except ValueError:
        return None
    return d


def read_latest_status(path):
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 131072), os.SEEK_SET)
            text = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        x = parse_status_line(line)
        if x is not None:
            return x
    return None


def bar_position(s):
    epoch = s.get("epoch", 0)
    phase = s.get("phase", "")
    done = s.get("done", 0)
    total = max(s.get("total", 0), 1)
    frac = min(max(done / total, 0.0), 1.0)
    if epoch <= 0:
        return 0.0
    base = float(epoch - 1)
    if phase == "TRAIN":
        return base + 0.72 * frac
    if phase == "VAL":
        return base + 0.72 + 0.27 * frac
    if phase == "EPOCH":
        return float(epoch)
    return base


def compact_status(s):
    phase = s.get("phase", "WAIT")
    done, total = s.get("done", 0), s.get("total", 0)
    if phase == "PREP_TRAIN":
        return f"PREP TRAIN {done}/{total}"
    if phase == "PREP_VAL":
        return f"PREP VAL {done}/{total}"
    if phase == "TRAIN":
        return (
            f"TRAIN {done}/{total} acc={s.get('acc','?')}% "
            f"loss={s.get('loss','?')} best={s.get('best','?')}%"
        )
    if phase == "VAL":
        return f"VAL {done}/{total} acc={s.get('acc','?')}% best={s.get('best','?')}%"
    if phase == "EPOCH":
        return (
            f"E{s.get('epoch',0)} val={s.get('val','?')}% "
            f"best={s.get('best','?')}%@E{s.get('best_e','?')} "
            f"eta={s.get('eta','?')} a={s.get('alpha','?')}"
        )
    return phase


def tail(path, lines=50):
    if not path.is_file():
        return "<missing log>"
    clean = [x for x in path.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
    return "\n".join(clean[-lines:])


def main():
    args = parse_args()
    dataset = Path(args.dataset)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if args.fast_rank > 4:
        raise ValueError("Keep fast-rank <=4; recommended=2")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logs = {p: logdir / f"{p}_worker.log" for p in ("xsub", "xset")}

    print("="*118)
    print("DISTAL-SM SPECIALIST T16 | FROM SCRATCH | DUAL T4")
    print("GPU0 -> XSUB | GPU1 -> XSET")
    print("WRISTS/HANDS/HAND-TIPS/THUMBS + ANKLES/FEET ONLY")
    print("RAW VARIABLE LENGTH -> FIXED T16 | NO ATTENTION | 120-WAY SPECIALIST")
    print("Exactly two parent-owned tqdm rows; child workers are notebook-silent")
    print("="*118)
    print(gpu_inventory())
    print("-"*118)
    print(
        f"EPOCHS={args.epochs} PATIENCE={args.patience} BATCH={args.batch_size} "
        f"EVAL={args.eval_batch_size} D={args.model_dim} RANK={args.fast_rank} LR={args.learning_rate:g}"
    )
    print("Worker logs:", logdir)
    print("="*118, flush=True)

    bars = {
        "xsub": tqdm(total=args.epochs, desc="XSUB DISTAL", position=0, leave=True,
                     dynamic_ncols=True, mininterval=0.25, smoothing=0.05),
        "xset": tqdm(total=args.epochs, desc="XSET DISTAL", position=1, leave=True,
                     dynamic_ncols=True, mininterval=0.25, smoothing=0.05),
    }
    bars["xsub"].set_postfix_str("STARTING GPU0", refresh=True)
    bars["xset"].set_postfix_str("WAITING GPU1", refresh=True)

    xsub, hx = launch(args, "xsub", 0, logs["xsub"])
    until = time.time() + max(0.0, args.startup_stagger)
    while time.time() < until and xsub.poll() is None:
        s = read_latest_status(logs["xsub"])
        if s:
            bars["xsub"].n = min(float(args.epochs), bar_position(s))
            bars["xsub"].set_postfix_str(compact_status(s), refresh=True)
        time.sleep(0.35)

    xset, hs = launch(args, "xset", 1, logs["xset"])
    bars["xset"].set_postfix_str("STARTING GPU1", refresh=True)
    procs = {"xsub": xsub, "xset": xset}
    last = {"xsub": None, "xset": None}

    try:
        while any(p.poll() is None for p in procs.values()):
            for protocol in ("xsub", "xset"):
                s = read_latest_status(logs[protocol])
                if not s:
                    continue
                fp = tuple(sorted(s.items()))
                if fp == last[protocol]:
                    continue
                last[protocol] = fp
                bars[protocol].n = min(float(args.epochs), bar_position(s))
                bars[protocol].set_postfix_str(compact_status(s), refresh=True)
            time.sleep(0.35)
    finally:
        rcx, rcs = xsub.wait(), xset.wait()
        hx.close(); hs.close()
        for protocol, rc in (("xsub", rcx), ("xset", rcs)):
            rp = outdir / f"result_{protocol}.json"
            bar = bars[protocol]
            if rc == 0 and rp.is_file():
                d = json.loads(rp.read_text())
                actual = max(int(d.get("last_epoch", 1)), 1)
                bar.total = actual
                bar.n = actual
                bar.set_postfix_str(
                    f"DONE best={100*d['best_val_accuracy']:.3f}% @E{d['best_epoch']}",
                    refresh=True,
                )
            else:
                bar.set_postfix_str(f"FAILED rc={rc}", refresh=True)
            bar.close()

    if rcx != 0 or rcs != 0:
        print("\nXSUB log tail:\n" + tail(logs["xsub"]))
        print("\nXSET log tail:\n" + tail(logs["xset"]))
        raise RuntimeError(f"Distal workers failed: XSUB={rcx}, XSET={rcs}")

    print("\n" + "="*118)
    print("DISTAL-SM SPECIALIST RESULTS")
    print("="*118)
    for protocol in ("xsub", "xset"):
        d = json.loads((outdir / f"result_{protocol}.json").read_text())
        flops = d.get("raw_xla_flops")
        ft = "NA" if flops is None else f"{flops/1e6:.6f} raw-XLA MFLOPs"
        print(
            f"{protocol.upper()} | BEST={100*d['best_val_accuracy']:.6f}% | "
            f"BEST_E={d['best_epoch']} | LAST_E={d['last_epoch']} | "
            f"PARAMS={d['params']:,} | {ft}"
        )
    print("="*118)


if __name__ == "__main__":
    main()

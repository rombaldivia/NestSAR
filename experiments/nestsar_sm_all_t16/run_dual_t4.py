#!/usr/bin/env python3
from __future__ import annotations

"""Dual-T4 launcher for NestSAR-SM-ALL-T16.

GPU0 -> XSUB
GPU1 -> XSET

Only this parent process renders tqdm. Child workers write ordinary STATUS lines
to log files, so Kaggle/Jupyter never receives competing carriage-return output.
The result is exactly two persistent progress rows which update in place through
preprocessing, training and validation.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--outdir",
        default="/kaggle/working/NestSAR_SM_ALL_T16_DualT4",
    )

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)

    p.add_argument("--learning-rate", type=float, default=6e-4)
    p.add_argument("--min-learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.08)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)

    p.add_argument("--stream-aux-weight", type=float, default=0.15)
    p.add_argument("--consistency-weight", type=float, default=0.08)
    p.add_argument("--consistency-temperature", type=float, default=1.0)

    p.add_argument("--spatial-dim", type=int, default=24)
    p.add_argument("--model-dim", type=int, default=112)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--controller-dim", type=int, default=16)
    p.add_argument("--fast-rank", type=int, default=2)
    p.add_argument("--head-rank", type=int, default=2)
    p.add_argument("--sm-residual-scale", type=float, default=0.08)
    p.add_argument("--head-residual-scale", type=float, default=0.15)

    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--jitter-max-shift", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--eval-progress-every", type=int, default=10)
    p.add_argument("--preprocess-progress-every", type=int, default=2500)

    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-gflops", type=float, default=0.025)
    p.add_argument("--startup-stagger", type=float, default=15.0)
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


def worker_cmd(args: argparse.Namespace, protocol: str):
    return [
        sys.executable,
        "-u",
        "-m",
        "experiments.nestsar_sm_all_t16.train_gpu",
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
        "--stream-aux-weight", str(args.stream_aux_weight),
        "--consistency-weight", str(args.consistency_weight),
        "--consistency-temperature", str(args.consistency_temperature),
        "--spatial-dim", str(args.spatial_dim),
        "--model-dim", str(args.model_dim),
        "--dropout", str(args.dropout),
        "--controller-dim", str(args.controller_dim),
        "--fast-rank", str(args.fast_rank),
        "--head-rank", str(args.head_rank),
        "--sm-residual-scale", str(args.sm_residual_scale),
        "--head-residual-scale", str(args.head_residual_scale),
        "--seed", str(args.seed),
        "--jitter-max-shift", str(args.jitter_max_shift),
        "--progress-every", str(args.progress_every),
        "--eval-progress-every", str(args.eval_progress_every),
        "--preprocess-progress-every", str(args.preprocess_progress_every),
        "--max-train-samples", str(args.max_train_samples),
        "--max-val-samples", str(args.max_val_samples),
        "--max-gflops", str(args.max_gflops),
        "--audit-first",
    ]


def launch(
    args: argparse.Namespace,
    protocol: str,
    physical_gpu: int,
    log_path: Path,
):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    handle = log_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        worker_cmd(args, protocol),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, handle


def parse_status_line(line: str):
    line = line.strip()
    if not line.startswith("STATUS|"):
        return None
    fields = {}
    for part in line.split("|")[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    try:
        fields["epoch"] = int(fields.get("epoch", 0))
        fields["epochs"] = int(fields.get("epochs", 0))
        fields["done"] = int(fields.get("done", 0))
        fields["total"] = int(fields.get("total", 0))
    except ValueError:
        return None
    return fields


def read_latest_status(log_path: Path):
    if not log_path.is_file():
        return None
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 131072), os.SEEK_SET)
            text = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return None

    for line in reversed(text.splitlines()):
        status = parse_status_line(line)
        if status is not None:
            return status
    return None


def bar_position(status) -> float:
    epoch = status["epoch"]
    phase = status.get("phase", "")
    done = status.get("done", 0)
    total = max(status.get("total", 0), 1)
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


def compact_status(status) -> str:
    phase = status.get("phase", "WAIT")
    done = status.get("done", 0)
    total = status.get("total", 0)

    if phase.startswith("PREP_"):
        label = "PREP TRAIN" if phase == "PREP_TRAIN" else "PREP VAL"
        return f"{label} {done}/{total}"

    if phase == "TRAIN":
        return (
            f"TRAIN {done}/{total} loss={status.get('loss','?')} "
            f"acc={status.get('acc','?')}% best={status.get('best','?')}%"
        )
    if phase == "VAL":
        return (
            f"VAL {done}/{total} acc={status.get('acc','?')}% "
            f"best={status.get('best','?')}%"
        )
    if phase == "EPOCH":
        return (
            f"E{status.get('epoch',0)} val={status.get('val','?')}% "
            f"best={status.get('best','?')}% @E{status.get('best_e','?')} "
            f"eta={status.get('eta','?')} a={status.get('alpha','?')}"
        )
    return phase


def tail_for_error(log_path: Path, lines: int = 50) -> str:
    if not log_path.is_file():
        return "<log missing>"
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    clean = [line for line in text.splitlines() if line.strip()]
    return "\n".join(clean[-lines:])


def main() -> None:
    args = parse_args()
    if args.fast_rank > 4:
        raise ValueError("Keep fast-rank <= 4 for the low-compute experiment; recommended=2")
    if args.head_rank > 4:
        raise ValueError("Keep head-rank <= 4 for the low-compute experiment; recommended=2")
    if args.max_gflops > 0.025:
        raise ValueError("This branch intentionally enforces a <=0.025 GFLOP design budget")

    dataset = Path(args.dataset)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    xsub_log = logdir / "xsub_worker.log"
    xset_log = logdir / "xset_worker.log"

    print("=" * 118)
    print("NESTSAR-SM-ALL-T16 v1 | FROM SCRATCH | DUAL T4")
    print("GPU0 -> XSUB | GPU1 -> XSET")
    print("RAW VARIABLE-LENGTH CLIP -> FIXED 16 PROCESSING TOKENS")
    print("NO ATTENTION | NO GCN | NO TCN | NO TRANSFORMER")
    print("Notebook progress is parent-owned: exactly two persistent rows")
    print("=" * 118)
    print(gpu_inventory())
    print("-" * 118)
    print(
        f"EPOCHS={args.epochs} PATIENCE={args.patience} BATCH={args.batch_size} "
        f"EVAL={args.eval_batch_size} LR={args.learning_rate:g} "
        f"FAST_RANK={args.fast_rank} HEAD_RANK={args.head_rank} "
        f"CTRL={args.controller_dim} GFLOP_LIMIT={args.max_gflops:.6f}"
    )
    print("Worker logs:", logdir)
    print("=" * 118, flush=True)

    bars = {
        "xsub": tqdm(
            total=args.epochs,
            desc="XSUB SM-T16",
            position=0,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.25,
            smoothing=0.05,
        ),
        "xset": tqdm(
            total=args.epochs,
            desc="XSET SM-T16",
            position=1,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.25,
            smoothing=0.05,
        ),
    }
    bars["xsub"].set_postfix_str("STARTING GPU0", refresh=True)
    bars["xset"].set_postfix_str("WAITING GPU1", refresh=True)

    xsub, xsub_handle = launch(args, "xsub", 0, xsub_log)

    stagger_until = time.time() + max(0.0, args.startup_stagger)
    while time.time() < stagger_until and xsub.poll() is None:
        status = read_latest_status(xsub_log)
        if status is not None:
            bar = bars["xsub"]
            bar.n = min(float(args.epochs), bar_position(status))
            bar.set_postfix_str(compact_status(status), refresh=True)
        time.sleep(0.35)

    xset, xset_handle = launch(args, "xset", 1, xset_log)
    bars["xset"].set_postfix_str("STARTING GPU1", refresh=True)

    procs = {"xsub": xsub, "xset": xset}
    logs = {"xsub": xsub_log, "xset": xset_log}
    last_status = {"xsub": None, "xset": None}
    last_epoch = {"xsub": 0, "xset": 0}

    try:
        while any(proc.poll() is None for proc in procs.values()):
            for protocol in ("xsub", "xset"):
                status = read_latest_status(logs[protocol])
                if status is None:
                    continue
                fingerprint = tuple(sorted(status.items()))
                if fingerprint == last_status[protocol]:
                    continue
                last_status[protocol] = fingerprint
                last_epoch[protocol] = max(
                    last_epoch[protocol], status.get("epoch", 0)
                )
                bar = bars[protocol]
                bar.n = min(float(args.epochs), bar_position(status))
                bar.set_postfix_str(compact_status(status), refresh=True)
            time.sleep(0.35)
    finally:
        rc_xsub = xsub.wait()
        rc_xset = xset.wait()
        xsub_handle.close()
        xset_handle.close()

        for protocol, rc in (("xsub", rc_xsub), ("xset", rc_xset)):
            result_path = outdir / f"result_{protocol}.json"
            bar = bars[protocol]
            if rc == 0 and result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                actual_last = max(int(result.get("last_epoch", 0)), 1)
                bar.total = actual_last
                bar.n = actual_last
                bar.set_postfix_str(
                    f"DONE best={100*result['best_val_accuracy']:.3f}% @E{result['best_epoch']}",
                    refresh=True,
                )
            else:
                bar.set_postfix_str(f"FAILED rc={rc}", refresh=True)
            bar.close()

    if rc_xsub != 0 or rc_xset != 0:
        print("\nXSUB worker log tail:\n" + tail_for_error(xsub_log))
        print("\nXSET worker log tail:\n" + tail_for_error(xset_log))
        raise RuntimeError(
            f"SM-ALL workers failed: XSUB={rc_xsub}, XSET={rc_xset}"
        )

    print()
    print("=" * 118)
    print("NESTSAR-SM-ALL-T16 RESULTS")
    print("=" * 118)
    results = {}
    for protocol in ("xsub", "xset"):
        path = outdir / f"result_{protocol}.json"
        d = json.loads(path.read_text(encoding="utf-8"))
        results[protocol] = d
        gflops = d.get("gflops")
        gflops_text = "NA" if gflops is None else f"{gflops:.9f}"
        print(
            f"{protocol.upper()} | BEST={100*d['best_val_accuracy']:.6f}% | "
            f"BEST_E={d['best_epoch']} | LAST_E={d['last_epoch']} | "
            f"PARAMS={d['params']:,} | GFLOPs={gflops_text} | "
            f"RAW_T=variable -> PROC_T={d['processing_frames']}"
        )

    mean_acc = 0.5 * (
        results["xsub"]["best_val_accuracy"]
        + results["xset"]["best_val_accuracy"]
    )
    print(f"MEAN BEST ACCURACY={100*mean_acc:.6f}%")
    print("=" * 118)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from tqdm.auto import tqdm

MARK = "@@CME@@"
MODULE = "experiments.m4_confusion_memory_expert_t32.pipeline"


def gpu_env(index: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(index)
    env["JAX_PLATFORMS"] = "cuda"
    env["PYTHONUNBUFFERED"] = "1"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    return env


def common_args(args, protocol: str, mode: str) -> list[str]:
    ckpt = args.base_ckpt_xsub if protocol == "xsub" else args.base_ckpt_xset
    cache = str(Path(args.cache_root) / protocol)
    return [
        sys.executable, "-u", "-m", MODULE,
        "--mode", mode,
        "--dataset", args.dataset,
        "--protocol", protocol,
        "--base-ckpt", ckpt,
        "--base-kind", args.base_kind,
        "--cache-dir", cache,
        "--outdir", args.outdir,
        "--weak-classes", args.weak_classes,
        "--confusion-pairs", args.confusion_pairs,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--base-batch-size", str(args.base_batch_size),
        "--specialist-dim", str(args.specialist_dim),
        "--topk-context", str(args.topk_context),
        "--dropout", str(args.dropout),
        "--learning-rate", str(args.learning_rate),
        "--min-learning-rate", str(args.min_learning_rate),
        "--warmup-fraction", str(args.warmup_fraction),
        "--weight-decay", str(args.weight_decay),
        "--grad-clip", str(args.grad_clip),
        "--ema-decay", str(args.ema_decay),
        "--holdout-fraction", str(args.holdout_fraction),
        "--selection-margin", str(args.selection_margin),
        "--hard-positive-weight", str(args.hard_positive_weight),
        "--ambiguous-weight", str(args.ambiguous_weight),
        "--reject-weight", str(args.reject_weight),
        "--pair-weight", str(args.pair_weight),
        "--pair-margin", str(args.pair_margin),
        "--alpha", str(args.alpha),
        "--route-margin", str(args.route_margin),
        "--gate-temperature", str(args.gate_temperature),
        "--harm-penalty", str(args.harm_penalty),
        "--seed", str(args.seed),
    ]


def run_cache(args, protocol: str, gpu: int, logdir: Path) -> None:
    cmd = common_args(args, protocol, "cache")
    r = subprocess.run(cmd, env=gpu_env(gpu), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (logdir / f"{protocol}_cache.log").write_text(r.stdout or "")
    if r.returncode != 0:
        tail = "\n".join((r.stdout or "").splitlines()[-30:])
        raise RuntimeError(f"{protocol.upper()} cache failed rc={r.returncode}\n{tail}")


def reader(prefix: str, proc: subprocess.Popen, q: queue.Queue, logfile: Path) -> None:
    with logfile.open("w", buffering=1) as f:
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            line = line.rstrip("\r\n")
            if line.startswith(MARK):
                try:
                    q.put((prefix, json.loads(line[len(MARK):])))
                except Exception:
                    pass
    q.put((prefix, {"event": "process_end"}))


def run_training(args, logdir: Path) -> None:
    specs = [("xsub", 0), ("xset", 1)]
    bars = {
        "xsub": tqdm(total=args.epochs, desc="XSUB CME", position=0, leave=True, dynamic_ncols=True, mininterval=0.5),
        "xset": tqdm(total=args.epochs, desc="XSET CME", position=1, leave=True, dynamic_ncols=True, mininterval=0.5),
    }
    processes = {}
    threads = []
    q: queue.Queue = queue.Queue()

    try:
        for protocol, gpu in specs:
            cmd = common_args(args, protocol, "train")
            proc = subprocess.Popen(cmd, env=gpu_env(gpu), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            processes[protocol] = proc
            th = threading.Thread(target=reader, args=(protocol, proc, q, logdir / f"{protocol}_train.log"), daemon=True)
            th.start(); threads.append(th)

        ended = set()
        while len(ended) < 2:
            protocol, ev = q.get()
            kind = ev.get("event")
            bar = bars[protocol]
            if kind == "epoch":
                epoch = int(ev.get("epoch", 0))
                if epoch > bar.n:
                    bar.update(epoch - bar.n)
                bar.set_postfix(
                    val=f"{100*float(ev.get('hold_final', 0.0)):.2f}%",
                    fix=int(ev.get("fixes", 0)),
                    harm=int(ev.get("harms", 0)),
                    loss=f"{float(ev.get('loss', 0.0)):.3f}",
                    refresh=True,
                )
            elif kind == "train_start":
                bar.set_postfix(stage="train", n=int(ev.get("fit", 0)), p=int(ev.get("specialist_params", 0)), refresh=True)
            elif kind == "done":
                bar.set_postfix(
                    final=f"{100*float(ev.get('final_acc', 0.0)):.3f}%",
                    delta=f"{100*(float(ev.get('final_acc', 0.0))-float(ev.get('base_acc', 0.0))):+.3f}pp",
                    fix=int(ev.get("fixes", 0)),
                    harm=int(ev.get("harms", 0)),
                    refresh=True,
                )
            elif kind == "process_end":
                ended.add(protocol)

        for th in threads:
            th.join()
    finally:
        for bar in bars.values():
            bar.close()

    failures = []
    for protocol, proc in processes.items():
        rc = proc.wait()
        if rc != 0:
            failures.append((protocol, rc))
    if failures:
        details = []
        for protocol, rc in failures:
            p = logdir / f"{protocol}_train.log"
            tail = "\n".join(p.read_text().splitlines()[-35:]) if p.is_file() else "missing log"
            details.append(f"{protocol.upper()} rc={rc}\n{tail}")
        raise RuntimeError("\n\n".join(details))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NestSAR CME T32 Dual-T4 trainer with exactly two tqdm bars")
    p.add_argument("--dataset", required=True)
    p.add_argument("--base-ckpt-xsub", required=True)
    p.add_argument("--base-ckpt-xset", required=True)
    p.add_argument("--base-kind", choices=["localglobal", "bijoint"], default="bijoint")
    p.add_argument("--cache-root", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--weak-classes", default="71,72,73,74,75,76,82,84,106,107")
    p.add_argument("--confusion-pairs", default="71-72,73-76,74-84,106-107")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--base-batch-size", type=int, default=256)
    p.add_argument("--specialist-dim", type=int, default=32)
    p.add_argument("--topk-context", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--learning-rate", type=float, default=6e-4)
    p.add_argument("--min-learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.08)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--holdout-fraction", type=float, default=0.10)
    p.add_argument("--selection-margin", type=float, default=0.20)
    p.add_argument("--hard-positive-weight", type=float, default=2.0)
    p.add_argument("--ambiguous-weight", type=float, default=1.5)
    p.add_argument("--reject-weight", type=float, default=1.5)
    p.add_argument("--pair-weight", type=float, default=0.20)
    p.add_argument("--pair-margin", type=float, default=0.20)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--route-margin", type=float, default=0.15)
    p.add_argument("--gate-temperature", type=float, default=12.0)
    p.add_argument("--harm-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=128)
    return p


def main() -> int:
    args = parser().parse_args()
    if not Path(args.dataset).is_file():
        raise FileNotFoundError(args.dataset)
    for p in (args.base_ckpt_xsub, args.base_ckpt_xset):
        if not Path(p).is_file():
            raise FileNotFoundError(p)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    logdir = out / "logs"; logdir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_root).mkdir(parents=True, exist_ok=True)

    # Cache sequentially to avoid loading the large NTU pickle twice in host RAM.
    # Training itself is concurrent on both T4s.
    run_cache(args, "xsub", 0, logdir)
    run_cache(args, "xset", 1, logdir)
    run_training(args, logdir)

    summary = {}
    for protocol in ("xsub", "xset"):
        rp = out / protocol / "result.json"
        if not rp.is_file():
            raise RuntimeError(f"missing result {rp}")
        summary[protocol] = json.loads(rp.read_text())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

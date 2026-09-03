#!/usr/bin/env python3
from __future__ import annotations

"""Compact-console concurrent Dual-T4 launcher for BiJoint shared-cache training.

Both protocol workers still run concurrently and every worker line is preserved in
logs/xsub.log and logs/xset.log. The notebook console suppresses per-update Uxxxx
training lines and prints only startup/important lines plus one summary per epoch.
This changes display only; model, data, optimizer, seed, losses and GPU assignment
are identical to run_concurrent_sharedcache_dual_t4.py.
"""

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

from experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16 import (
    run_concurrent_sharedcache_dual_t4 as base,
)


def _is_update_line(line: str) -> bool:
    # Examples:
    # [00:41:31] XSUB E002 U0020/246 loss=...
    # Keep these in the protocol log file, but do not spam the Kaggle cell.
    return bool(base.re.search(r"\b(?:XSUB|XSET)\s+E\d{3}\s+U\d{4}/\d+\b", line))


def _is_epoch_summary(line: str) -> bool:
    # Example:
    # [00:40:49] XSET E001 train_can=... val=... time=...
    return bool(base.re.search(r"\b(?:XSUB|XSET)\s+E\d{3}\s+train_can=", line))


def _is_important_startup(line: str) -> bool:
    keys = (
        "MMAP SHARED CACHE",
        "SHARED CACHE",
        "BACKEND:",
        "DEVICES:",
        "EXPECTED PARAMS:",
        "EFFECTIVE BATCH:",
        "MICROBATCH:",
        "CACHE:",
        "ATTENTION:",
        "TRAINING:",
        "preprocessing START",
        "preprocessing READY",
        "params=",
        "MEMORY-SAFE TRAINING",
        "early stop:",
        "CACHED GPU WORKER DONE",
        "SHARED-CACHE GPU WORKER DONE",
    )
    return any(k in line for k in keys)


def run_both_compact(dataset: str, outdir: str, cache_root: str, logs: Path) -> None:
    specs = [
        ("XSUB/GPU0", 0, "xsub"),
        ("XSET/GPU1", 1, "xset"),
    ]
    processes: dict[str, subprocess.Popen] = {}
    q: queue.Queue = queue.Queue()
    threads: list[threading.Thread] = []

    print("=" * 120, flush=True)
    print("STARTING BOTH TRAINING PROTOCOLS AT THE SAME TIME — COMPACT CONSOLE", flush=True)
    print("XSUB -> physical GPU0 | XSET -> physical GPU1", flush=True)
    print("Detailed Uxxxx progress is saved to logs; console prints epoch summaries only.", flush=True)
    print("=" * 120, flush=True)

    for prefix, gpu, protocol in specs:
        cmd = base.worker_command(dataset, outdir, protocol)
        proc = subprocess.Popen(
            cmd,
            env=base.gpu_env(gpu, cache_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes[prefix] = proc
        th = threading.Thread(
            target=base._reader,
            args=(prefix, proc, logs / f"{protocol}.log", q),
            daemon=True,
        )
        th.start()
        threads.append(th)

    done: set[str] = set()
    while len(done) < len(specs):
        prefix, line = q.get()
        if line is None:
            done.add(prefix)
            continue

        # Full output is already written by _reader() to the protocol log.
        # Suppress only the frequent training update lines in the notebook.
        if _is_update_line(line):
            continue

        if _is_epoch_summary(line) or _is_important_startup(line):
            print(f"[{prefix}] {line}", flush=True)

    for th in threads:
        th.join()

    failures = []
    for prefix, proc in processes.items():
        rc = proc.wait()
        if rc != 0:
            failures.append((prefix, rc))
    if failures:
        raise RuntimeError(f"Concurrent worker failure(s): {failures}; logs={logs}")


def main(dataset: str | None = None, outdir: str | None = None, cache_root: str | None = None) -> int:
    dataset = dataset or (sys.argv[1] if len(sys.argv) > 1 else base.DEFAULT_DATASET)
    outdir = outdir or (sys.argv[2] if len(sys.argv) > 2 else base.DEFAULT_OUTDIR)
    cache_root = cache_root or (sys.argv[3] if len(sys.argv) > 3 else base.DEFAULT_CACHE)

    if not Path(dataset).is_file():
        raise FileNotFoundError(dataset)

    gpus = base.visible_gpus()
    print("=" * 120, flush=True)
    print("NESTSAR LOCALGLOBAL V2 + BI-JOINT M4/G4 — TRUE CONCURRENT DUAL T4 — COMPACT", flush=True)
    print("=" * 120, flush=True)
    for x in gpus:
        print("GPU:", x, flush=True)
    if len(gpus) != base.EXPECTED_GPUS:
        raise RuntimeError(f"Expected exactly 2 GPUs, found {len(gpus)}")

    base.probe_gpu(0, cache_root)
    base.probe_gpu(1, cache_root)

    print("=" * 120, flush=True)
    print("BIJOINT PARAM/FLOP PREFLIGHT ON GPU0", flush=True)
    print("=" * 120, flush=True)
    pf = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16.preflight_gpu",
            dataset,
        ],
        env=base.gpu_env(0, cache_root),
        check=False,
    )
    if pf.returncode != 0:
        raise RuntimeError(f"GPU preflight failed rc={pf.returncode}")

    base.build_shared_cache(dataset, cache_root)

    out = Path(outdir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Output directory is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "M4LocalGlobalBiJointT16_ConcurrentSharedExactCache_CompactConsole",
        "dataset": dataset,
        "outdir": outdir,
        "cache_root": cache_root,
        "expected_params": base.EXPECTED_PARAMS,
        "gpu_assignment": {"xsub": 0, "xset": 1},
        "protocol_execution": "concurrent",
        "effective_batch": 256,
        "microbatch": 64,
        "gradient_accumulation_steps": 4,
        "preprocessing_storage": "shared_canonical_exact_float32_plus_protocol_jitter_mmap",
        "console": "epoch_summary_only_full_worker_logs_on_disk",
        "attention": False,
        "training_from_scratch": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    run_both_compact(dataset, outdir, cache_root, logs)

    results = {}
    for protocol in ("xsub", "xset"):
        p = out / f"result_{protocol}.json"
        if not p.is_file():
            raise RuntimeError(f"Missing result file: {p}")
        results[protocol] = json.loads(p.read_text())
    (out / "summary.json").write_text(json.dumps(results, indent=2))

    print("=" * 120, flush=True)
    print("CONCURRENT SHARED-CACHE BIJOINT RUN DONE", flush=True)
    for protocol in ("xsub", "xset"):
        r = results[protocol]
        print(
            f"{protocol.upper()} best={100*r['best_val_accuracy']:.6f}% "
            f"@ E{r['best_epoch']} | delta={r['delta_vs_baseline_pp']:+.4f} pp",
            flush=True,
        )
    print("=" * 120, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

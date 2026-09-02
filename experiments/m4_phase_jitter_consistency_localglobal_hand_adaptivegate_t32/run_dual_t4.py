#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

EXPECTED_GPUS = 2

DEFAULT_DATASET = (
    "/kaggle/input/models/paolamaydana/"
    "ntudanno/other/default/1/ntu120_3danno.pkl"
)
DEFAULT_HAND_ROOT = (
    "/kaggle/working/"
    "NestSAR_M4_LocalGlobal_HandM4G4Lite_T32_DualT4"
)
DEFAULT_OUTDIR = (
    "/kaggle/working/"
    "NestSAR_HandM4G4_AdaptiveGate_Crossfit_DualT4"
)


def gpu_env(index: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(index)
    env["JAX_PLATFORMS"] = "cuda"
    env["PYTHONUNBUFFERED"] = "1"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    return env


def visible_gpus() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [x.strip() for x in result.stdout.splitlines() if x.strip()]


def probe_gpu(index: int) -> None:
    code = (
        "import jax; "
        "print('BACKEND='+str(jax.default_backend())); "
        "print('COUNT='+str(jax.local_device_count())); "
        "print('DEVICES='+repr(jax.local_devices()))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        env=gpu_env(index),
        capture_output=True,
        text=True,
        check=False,
    )
    out = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
    print("=" * 120, flush=True)
    print(f"GPU{index} JAX PROBE", flush=True)
    print("=" * 120, flush=True)
    print(out.strip(), flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"GPU{index} JAX probe failed")
    backend = re.search(r"^BACKEND=(.+)$", out, re.MULTILINE)
    count = re.search(r"^COUNT=(\d+)$", out, re.MULTILINE)
    if not backend or backend.group(1).strip() != "gpu":
        raise RuntimeError(f"GPU{index}: GPU backend unavailable")
    if not count or int(count.group(1)) != 1:
        raise RuntimeError(f"GPU{index}: expected one visible JAX GPU")


def stream_output(proc, prefix, lock, log_path: Path):
    assert proc.stdout is not None
    with log_path.open("a", buffering=1) as f:
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            f.write(line + "\n")
            with lock:
                print(f"[{prefix}] {line}", flush=True)


def worker_command(dataset: str, hand_root: str, outdir: str, protocol: str):
    checkpoint = str(Path(hand_root) / protocol / "best.msgpack")
    return [
        sys.executable,
        "-u",
        "-m",
        (
            "experiments."
            "m4_phase_jitter_consistency_localglobal_hand_adaptivegate_t32."
            "crossfit_gate_gpu"
        ),
        "--dataset", dataset,
        "--protocol", protocol,
        "--checkpoint", checkpoint,
        "--outdir", outdir,
        "--folds", "5",
        "--epochs", "25",
        "--gate-batch-size", "4096",
        "--learning-rate", "2e-3",
        "--weight-decay", "1e-4",
        "--max-alpha", "0.30",
        "--seed", "128",
    ]


def main(
    dataset: str | None = None,
    hand_root: str | None = None,
    outdir: str | None = None,
) -> int:
    dataset = dataset or DEFAULT_DATASET
    hand_root = hand_root or DEFAULT_HAND_ROOT
    outdir = outdir or DEFAULT_OUTDIR

    if not Path(dataset).is_file():
        raise FileNotFoundError(dataset)

    for protocol in ("xsub", "xset"):
        ckpt = Path(hand_root) / protocol / "best.msgpack"
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"Missing trained Hand-M4/G4 checkpoint: {ckpt}"
            )

    out_path = Path(outdir)
    if out_path.exists() and any(out_path.iterdir()):
        raise RuntimeError(
            f"Output directory is non-empty: {out_path}\n"
            "Move/delete it if you intentionally want to rerun the diagnostic."
        )
    out_path.mkdir(parents=True, exist_ok=True)
    logs_dir = out_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    gpus = visible_gpus()
    print("=" * 120, flush=True)
    print("NESTSAR HAND-M4/G4 T32 + ADAPTIVE TRUST GATE — 5-FOLD CROSSFIT — DUAL T4", flush=True)
    print("=" * 120, flush=True)
    for g in gpus:
        print("GPU:", g, flush=True)
    if len(gpus) != EXPECTED_GPUS:
        raise RuntimeError(f"Expected exactly 2 GPUs, found {len(gpus)}")

    probe_gpu(0)
    probe_gpu(1)

    print("=" * 120, flush=True)
    print("ADAPTIVE GATE PARAM/FLOP PREFLIGHT ON GPU0", flush=True)
    print("=" * 120, flush=True)
    r = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            (
                "experiments."
                "m4_phase_jitter_consistency_localglobal_hand_adaptivegate_t32."
                "preflight_gpu"
            ),
        ],
        env=gpu_env(0),
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Adaptive gate preflight failed rc={r.returncode}")

    manifest = {
        "experiment": "HandM4G4T32_AdaptiveGate_Crossfit_Diagnostic",
        "diagnostic_only": True,
        "paper_benchmark": False,
        "frozen_hand_root": hand_root,
        "dataset": dataset,
        "gpu_assignment": {"xsub": 0, "xset": 1},
        "folds": 5,
        "gate_epochs_per_fold": 25,
        "gate_params": 2401,
        "gate_input": "main_desc112+hand_desc32+main/hand_margin+main/hand_entropy",
        "alpha": "0.30*sigmoid(gate)",
        "attention": False,
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    jobs = [
        ("XSUB/GPU0", 0, "xsub"),
        ("XSET/GPU1", 1, "xset"),
    ]
    processes = []
    threads = []
    lock = threading.Lock()

    for prefix, gpu, protocol in jobs:
        cmd = worker_command(dataset, hand_root, outdir, protocol)
        print(f"LAUNCH {prefix}: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            env=gpu_env(gpu),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append((prefix, proc))
        th = threading.Thread(
            target=stream_output,
            args=(proc, prefix, lock, logs_dir / f"{protocol}.log"),
            daemon=True,
        )
        th.start()
        threads.append(th)

    start = time.time()
    last = 0.0
    while True:
        now = time.time()
        if now - last >= 30.0:
            status = " | ".join(
                f"{prefix}=" + (
                    "RUNNING" if proc.poll() is None else f"DONE rc={proc.returncode}"
                )
                for prefix, proc in processes
            )
            print(
                f"[ADAPTIVE-GATE HEARTBEAT] elapsed={now-start:.0f}s | {status}",
                flush=True,
            )
            last = now
        if all(proc.poll() is not None for _, proc in processes):
            break
        time.sleep(1.0)

    for th in threads:
        th.join(timeout=5.0)

    failures = [(p, pr.returncode) for p, pr in processes if pr.returncode != 0]
    if failures:
        raise RuntimeError(f"Adaptive gate worker failures: {failures}; logs={logs_dir}")

    results = {}
    for protocol in ("xsub", "xset"):
        p = out_path / protocol / "result.json"
        if not p.is_file():
            raise RuntimeError(f"Missing result: {p}")
        results[protocol] = json.loads(p.read_text())

    (out_path / "summary.json").write_text(json.dumps(results, indent=2))

    print("=" * 120, flush=True)
    print("ADAPTIVE GATE CROSSFIT DIAGNOSTIC DONE", flush=True)
    print("=" * 120, flush=True)
    for protocol in ("xsub", "xset"):
        r = results[protocol]
        print(
            f"{protocol.upper()} fixed0.10={100*r['fixed_scores']['alpha_0.10']:.6f}% | "
            f"crossfit={100*r['crossfit_accuracy']:.6f}% | "
            f"delta={r['crossfit_delta_vs_fixed_010_pp']:+.4f} pp | "
            f"weak_delta={r['target_weak_delta_pp']:+.4f} pp | "
            f"p={r['mcnemar_p']:.6g}",
            flush=True,
        )
    print("DIAGNOSTIC ONLY — DO NOT REPORT CROSSFIT ACCURACY AS PAPER BENCHMARK", flush=True)
    print("=" * 120, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
EXPECTED_GATE_PARAMS = 2_481

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
    "NestSAR_HandM4G4_SelectiveGate_Crossfit_DualT4"
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
    r = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [x.strip() for x in r.stdout.splitlines() if x.strip()]


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
    combined = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
    print("=" * 120, flush=True)
    print(f"GPU{index} JAX PROBE", flush=True)
    print("=" * 120, flush=True)
    print(combined.strip(), flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"GPU{index} probe failed")
    backend = re.search(r"^BACKEND=(.+)$", combined, re.MULTILINE)
    count = re.search(r"^COUNT=(\d+)$", combined, re.MULTILINE)
    if not backend or backend.group(1).strip() != "gpu":
        raise RuntimeError(f"GPU{index}: JAX backend is not GPU")
    if not count or int(count.group(1)) != 1:
        raise RuntimeError(f"GPU{index}: expected one visible JAX GPU")


def stream_output(proc, prefix: str, lock: threading.Lock, log_path: Path):
    assert proc.stdout is not None
    with log_path.open("a", buffering=1) as f:
        for line in proc.stdout:
            clean = line.rstrip("\r\n")
            f.write(clean + "\n")
            with lock:
                print(f"[{prefix}] {clean}", flush=True)


def worker_command(
    dataset: str,
    checkpoint: str,
    outdir: str,
    protocol: str,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        (
            "experiments."
            "m4_phase_jitter_consistency_localglobal_hand_selectivegate_t32."
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
        "--base-alpha", "0.20",
        "--delta-alpha", "0.15",
        "--seed", "128",
    ]


def terminate_workers(processes):
    for _, proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 10
    while time.time() < deadline and any(p.poll() is None for _, p in processes):
        time.sleep(0.2)
    for _, proc in processes:
        if proc.poll() is None:
            proc.kill()


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

    checkpoints = {
        "xsub": str(Path(hand_root) / "xsub" / "best.msgpack"),
        "xset": str(Path(hand_root) / "xset" / "best.msgpack"),
    }
    for p in checkpoints.values():
        if not Path(p).is_file():
            raise FileNotFoundError(p)

    gpus = visible_gpus()
    print("=" * 120, flush=True)
    print(
        "NESTSAR HAND-M4/G4 T32 + SELECTIVE RESIDUAL TRUST GATE "
        "— 5-FOLD CROSSFIT — DUAL T4",
        flush=True,
    )
    print("=" * 120, flush=True)
    for line in gpus:
        print("GPU:", line, flush=True)
    if len(gpus) != EXPECTED_GPUS:
        raise RuntimeError(f"Expected exactly 2 GPUs, found {len(gpus)}")

    probe_gpu(0)
    probe_gpu(1)

    print("=" * 120, flush=True)
    print("SELECTIVE GATE PARAM/FLOP PREFLIGHT ON GPU0", flush=True)
    print("=" * 120, flush=True)
    pre = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            (
                "experiments."
                "m4_phase_jitter_consistency_localglobal_hand_selectivegate_t32."
                "preflight_gpu"
            ),
            checkpoints["xsub"],
        ],
        env=gpu_env(0),
        check=False,
    )
    if pre.returncode != 0:
        raise RuntimeError(f"Preflight failed rc={pre.returncode}")

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    logs_dir = out_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "HandM4G4T32SelectiveResidualGateCrossfit",
        "dataset": dataset,
        "hand_root": hand_root,
        "gpu_assignment": {"xsub": 0, "xset": 1},
        "folds": 5,
        "epochs_per_fold": 25,
        "gate_params": EXPECTED_GATE_PARAMS,
        "base_alpha": 0.20,
        "delta_alpha": 0.15,
        "alpha_range": [0.05, 0.35],
        "attention": False,
        "frozen_hand_model": True,
        "diagnostic_only": True,
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    jobs = [
        ("XSUB/GPU0", 0, "xsub"),
        ("XSET/GPU1", 1, "xset"),
    ]

    processes = []
    threads = []
    lock = threading.Lock()

    for prefix, gpu_index, protocol in jobs:
        cmd = worker_command(dataset, checkpoints[protocol], outdir, protocol)
        print(f"LAUNCH {prefix}: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            env=gpu_env(gpu_index),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append((prefix, proc))
        t = threading.Thread(
            target=stream_output,
            args=(proc, prefix, lock, logs_dir / f"{protocol}.log"),
            daemon=True,
        )
        t.start()
        threads.append(t)

    start = time.time()
    last = 0.0
    try:
        while True:
            alive = [(n, p) for n, p in processes if p.poll() is None]
            now = time.time()
            if now - last >= 30:
                status = " | ".join(
                    f"{n}={'RUNNING' if p.poll() is None else 'DONE rc='+str(p.returncode)}"
                    for n, p in processes
                )
                print(
                    f"[SELECTIVE-GATE HEARTBEAT] elapsed={now-start:.0f}s | {status}",
                    flush=True,
                )
                last = now
            if not alive:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        terminate_workers(processes)
        return 130

    for t in threads:
        t.join(timeout=5)

    failures = [(n, p.returncode) for n, p in processes if p.returncode != 0]
    if failures:
        raise RuntimeError(f"Worker failures: {failures}; logs={logs_dir}")

    results = {}
    for protocol in ("xsub", "xset"):
        path = out_path / protocol / "result.json"
        if not path.is_file():
            raise RuntimeError(f"Missing result {path}")
        results[protocol] = json.loads(path.read_text())

    (out_path / "summary.json").write_text(json.dumps(results, indent=2))

    print("=" * 120, flush=True)
    print("SELECTIVE GATE CROSSFIT DIAGNOSTIC DONE", flush=True)
    print("=" * 120, flush=True)
    for protocol in ("xsub", "xset"):
        r = results[protocol]
        print(
            f"{protocol.upper()} fixed0.10="
            f"{100*r['fixed_scores']['alpha_0.10']:.6f}% | "
            f"selective={100*r['crossfit_selective_accuracy']:.6f}% | "
            f"delta={r['delta_vs_fixed_010_pp']:+.4f} pp | "
            f"weak_delta={r['weak_target_delta_pp']:+.4f} pp | "
            f"p={r['mcnemar_p']:.6g}",
            flush=True,
        )
        a = r["alpha_stats"]
        print(
            f"{protocol.upper()} alpha mean={a['mean']:.4f} std={a['std']:.4f} "
            f"p10={a['p10']:.4f} p50={a['p50']:.4f} p90={a['p90']:.4f} | "
            f"agree={a['agree_mean']:.4f} disagree={a['disagree_mean']:.4f}",
            flush=True,
        )
    print("DIAGNOSTIC ONLY — DO NOT REPORT CROSSFIT ACCURACY AS PAPER BENCHMARK", flush=True)
    print("=" * 120, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

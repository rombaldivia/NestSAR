#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BUNDLE = "NestSAR_HOPE_v4_1_SHORTL3FIX_Kaggle_16F_ONE_CELL.py"
REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "experiments" / "hope_fidelity_d128_v1" / "run_universal_v3.py"


def first_hit(name: str):
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            hits = sorted(root.rglob(name))
            if hits:
                return hits[0].resolve()
    return None


def main() -> int:
    bundle = first_hit(BUNDLE)
    if bundle is None:
        raise FileNotFoundError(
            f"Missing {BUNDLE}. Attach it as Kaggle input or upload it to /kaggle/working."
        )

    dataset = first_hit("ntu120_3danno.pkl") or first_hit("ntu120_3danno_clean.pkl")
    if dataset is None:
        raise FileNotFoundError("Missing ntu120_3danno.pkl in /kaggle/input or /kaggle/working")

    cmd = [
        sys.executable, "-u", str(RUNNER),
        "--preset", "tpu-v5e8",
        "--bundle-cell", str(bundle),
        "--dataset", str(dataset),
        "--protocol", "xsub",
        "--frames", "16",
        "--model-dim", "128",
        "--memory-dim", "64",
        "--controller-rank", "32",
        "--frame-blocks", "2",
        "--chunk-blocks", "2",
        "--clip-blocks", "2",
        "--controller-blocks", "2",
        "--chunk-size", "4",
        "--clip-size", "8",
        "--cms-bottleneck", "32",
        "--batch-size", "128",
        "--grad-accum-steps", "1",
        "--eval-batch-size", "256",
        "--epochs", "3",
        "--patience", "3",
        "--learning-rate", "1e-3",
        "--weight-decay", "0.05",
        "--warmup-fraction", "0.10",
        "--dropout", "0.22",
        "--label-smoothing", "0.05",
        "--grad-clip", "1.0",
        "--predictive-loss-weight", "0.10",
        "--memory-residual-scale", "0.25",
        "--initial-eta", "0.02",
        "--initial-alpha", "0.95",
        "--ema-decay", "0.995",
        "--frame-mask-prob", "0.08",
        "--joint-mask-prob", "0.08",
        "--part-mask-prob", "0.03",
        "--cms-period-l1", "1",
        "--cms-period-l2", "2",
        "--cms-period-l3", "4",
        "--cms-period-l4", "8",
        "--dmgd-momentum", "0.90",
        "--dmgd-memory-lr", "0.01",
        "--dmgd-mix", "0.10",
        "--dmgd-projection-cap", "2.0",
        "--fresh",
    ]

    print("Bundle:", bundle)
    print("Dataset:", dataset)
    print("Runner:", RUNNER)
    print("Command:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(REPO))


if __name__ == "__main__":
    raise SystemExit(main())

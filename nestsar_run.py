#!/usr/bin/env python3
"""Stable launcher for versioned NestSAR research experiments."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

EXPERIMENTS = {
    "attention_lite": "experiments.attention_lite_v1.trainer",
    "attention_lite_v1": "experiments.attention_lite_v1.trainer",
}


def list_experiments() -> tuple[str, ...]:
    return tuple(sorted(EXPERIMENTS))


def train(
    experiment: str = "attention_lite",
    *,
    protocol: str = "xsub",
    epochs: int = 40,
    dataset: str | Path = "auto",
    seed: int = 128,
    runs_root: str | Path = "/kaggle/working",
    canonical_source: Optional[str | Path] = None,
    paper_mode: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Launch a versioned experiment in a clean Python subprocess.

    For Attention-Lite paper runs, the trainer generates a deterministic output
    folder containing both protocol and seed, for example:

        /kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_28

    The subprocess boundary is deliberate: TPU/JAX environment configuration must
    happen before JAX import and notebook kernels may contain stale compiled state.
    """
    key = experiment.strip().lower()
    if key not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment {experiment!r}. Available: {', '.join(list_experiments())}"
        )
    protocol = protocol.strip().lower()
    if protocol not in {"xsub", "xset"}:
        raise ValueError("protocol must be 'xsub' or 'xset'")
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")

    env = os.environ.copy()
    env["NESTSAR_PROTOCOL"] = protocol
    env["NESTSAR_EPOCHS"] = str(int(epochs))
    env["NESTSAR_SEED"] = str(int(seed))
    env["NESTSAR_DATASET"] = str(dataset)
    env["NESTSAR_RUNS_ROOT"] = str(runs_root)
    env["NESTSAR_PAPER_MODE"] = "1" if paper_mode else "0"
    if canonical_source is not None:
        env["NESTSAR_CANONICAL_SOURCE"] = str(canonical_source)

    cmd = [sys.executable, "-u", "-m", EXPERIMENTS[key]]
    print("NestSAR launch:", " ".join(cmd), flush=True)
    print(
        f"experiment={key} protocol={protocol} epochs={epochs} seed={seed} "
        f"dataset={dataset} runs_root={runs_root} paper_mode={paper_mode}",
        flush=True,
    )
    return subprocess.run(cmd, env=env, check=check)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run a versioned NestSAR experiment")
    parser.add_argument("experiment", nargs="?", default="attention_lite")
    parser.add_argument("--protocol", choices=("xsub", "xset"), default="xsub")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--runs-root", default="/kaggle/working")
    parser.add_argument("--canonical-source", default=None)
    parser.add_argument("--non-paper", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_only")
    args = parser.parse_args()
    if args.list_only:
        print("\n".join(list_experiments()))
        return 0
    train(
        args.experiment,
        protocol=args.protocol,
        epochs=args.epochs,
        dataset=args.dataset,
        seed=args.seed,
        runs_root=args.runs_root,
        canonical_source=args.canonical_source,
        paper_mode=not args.non_paper,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

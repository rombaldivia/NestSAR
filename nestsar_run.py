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
    epochs: int = 40,
    dataset: str | Path = "auto",
    seed: int = 128,
    output: Optional[str | Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Launch a versioned experiment in a clean Python subprocess.

    The subprocess boundary is deliberate: TPU/JAX environment configuration must
    happen before JAX import and notebook kernels may contain stale compiled state.
    """
    key = experiment.strip().lower()
    if key not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment {experiment!r}. Available: {', '.join(list_experiments())}"
        )
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")

    env = os.environ.copy()
    env["NESTSAR_EPOCHS"] = str(int(epochs))
    env["NESTSAR_SEED"] = str(int(seed))
    env["NESTSAR_DATASET"] = str(dataset)
    if output is not None:
        env["NESTSAR_OUTPUT"] = str(output)

    cmd = [sys.executable, "-u", "-m", EXPERIMENTS[key]]
    print("NestSAR launch:", " ".join(cmd), flush=True)
    print(f"experiment={key} epochs={epochs} seed={seed} dataset={dataset}", flush=True)
    return subprocess.run(cmd, env=env, check=check)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run a versioned NestSAR experiment")
    parser.add_argument("experiment", nargs="?", default="attention_lite")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--output", default=None)
    parser.add_argument("--list", action="store_true", dest="list_only")
    args = parser.parse_args()
    if args.list_only:
        print("\n".join(list_experiments()))
        return 0
    train(
        args.experiment,
        epochs=args.epochs,
        dataset=args.dataset,
        seed=args.seed,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

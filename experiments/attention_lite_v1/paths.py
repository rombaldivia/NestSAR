#!/usr/bin/env python3
"""Deterministic output layout for Attention-Lite paper/reproducibility runs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

MODEL_SLUG = "NestSAR_HOPE_Attention_Lite_D128"
DEFAULT_ROOT = Path("/kaggle/working")
ALLOWED_PROTOCOLS = ("xsub", "xset")
PAPER_SEEDS = (28, 42, 128, 2026)


@dataclass(frozen=True)
class RunPaths:
    protocol: str
    seed: int
    root: Path
    source_dir: Path
    logs_dir: Path
    metadata_dir: Path

    def as_dict(self) -> dict[str, str | int]:
        data = asdict(self)
        return {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}


def validate_protocol(protocol: str) -> str:
    p = str(protocol).strip().lower()
    if p not in ALLOWED_PROTOCOLS:
        raise ValueError(f"protocol must be one of {ALLOWED_PROTOCOLS}, got {protocol!r}")
    return p


def validate_seed(seed: int, *, paper_mode: bool = True) -> int:
    s = int(seed)
    if s < 0:
        raise ValueError("seed must be >= 0")
    if paper_mode and s not in PAPER_SEEDS:
        raise ValueError(
            f"paper-mode seed must be one of {PAPER_SEEDS}; got {s}. "
            "Set NESTSAR_PAPER_MODE=0 only for non-paper experiments."
        )
    return s


def run_folder_name(protocol: str, seed: int) -> str:
    p = validate_protocol(protocol)
    s = int(seed)
    return f"{MODEL_SLUG}_{p.upper()}_SEED_{s}"


def make_run_paths(
    protocol: str,
    seed: int,
    *,
    base_dir: str | Path | None = None,
    paper_mode: bool = True,
    create: bool = True,
) -> RunPaths:
    p = validate_protocol(protocol)
    s = validate_seed(seed, paper_mode=paper_mode)
    base = Path(base_dir or os.environ.get("NESTSAR_RUNS_ROOT", DEFAULT_ROOT)).expanduser()
    root = base / run_folder_name(p, s)
    paths = RunPaths(
        protocol=p,
        seed=s,
        root=root,
        source_dir=root / "generated_source",
        logs_dir=root / "logs",
        metadata_dir=root / "metadata",
    )
    if create:
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.source_dir.mkdir(parents=True, exist_ok=True)
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        paths.metadata_dir.mkdir(parents=True, exist_ok=True)
    return paths


def write_path_manifest(paths: RunPaths) -> Path:
    target = paths.metadata_dir / "paths.json"
    target.write_text(json.dumps(paths.as_dict(), indent=2), encoding="utf-8")
    return target

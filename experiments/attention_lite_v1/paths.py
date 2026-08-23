#!/usr/bin/env python3
"""Deterministic output layout for Attention-Lite paper/reproducibility runs."""
from __future__ import annotations

import json
import os
import re
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
    tag: str | None
    root: Path
    source_dir: Path
    logs_dir: Path
    metadata_dir: Path

    def as_dict(self) -> dict[str, str | int | None]:
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


def sanitize_tag(tag: str | None) -> str | None:
    if tag is None:
        return None
    raw = str(tag).strip()
    if not raw:
        return None
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    if not clean:
        raise ValueError(f"run tag contains no usable characters: {tag!r}")
    return clean[:80]


def run_folder_name(protocol: str, seed: int, tag: str | None = None) -> str:
    p = validate_protocol(protocol)
    s = int(seed)
    base = f"{MODEL_SLUG}_{p.upper()}_SEED_{s}"
    clean_tag = sanitize_tag(tag)
    return f"{base}_{clean_tag}" if clean_tag else base


def make_run_paths(
    protocol: str,
    seed: int,
    *,
    base_dir: str | Path | None = None,
    paper_mode: bool = True,
    tag: str | None = None,
    create: bool = True,
) -> RunPaths:
    p = validate_protocol(protocol)
    s = validate_seed(seed, paper_mode=paper_mode)
    clean_tag = sanitize_tag(tag)
    base = Path(base_dir or os.environ.get("NESTSAR_RUNS_ROOT", DEFAULT_ROOT)).expanduser()
    root = base / run_folder_name(p, s, clean_tag)
    paths = RunPaths(
        protocol=p,
        seed=s,
        tag=clean_tag,
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protocol-safe notebook launcher for v3.5 TPU.

Reuses the persistent two-row tqdm UI from run_both_tpu_v35_kaggle.py and points
workers at train_v35_tpu_safe.py.
"""
from __future__ import annotations

from pathlib import Path

from experiments.parttrace_v3_attention_lite import run_both_tpu_v35_kaggle as base

base.TRAINER = Path(__file__).resolve().parent / "train_v35_tpu_safe.py"


if __name__ == "__main__":
    raise SystemExit(base.main())

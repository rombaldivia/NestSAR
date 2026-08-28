#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility entry point for the hardened v3.5 TPU launcher.

Kept so existing Kaggle cells that call ``run_both_tpu_v35_kaggle_v2.py``
automatically receive the TPU preflight/forcing fix implemented in v3.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.parttrace_v3_attention_lite import run_both_tpu_v35_kaggle_v3 as hardened


if __name__ == "__main__":
    raise SystemExit(hardened.main())

#!/usr/bin/env python3
from __future__ import annotations

"""Dual-T4 launcher for CME T32 on the frozen Hand-M4/G4-Lite champion.

Reuses the tested two-bar launcher and changes only:
  - base kind -> hand
  - worker module -> pipeline_hand

Notebook output remains exactly two tqdm bars.
"""

from experiments.m4_confusion_memory_expert_t32 import run_dual_t4 as base


_ORIGINAL_PARSER = base.parser


def hand_parser():
    p = _ORIGINAL_PARSER()
    for action in p._actions:
        if action.dest == "base_kind":
            action.choices = ["hand"]
            action.default = "hand"
    return p


def main() -> int:
    base.MODULE = "experiments.m4_confusion_memory_expert_t32.pipeline_hand"
    base.parser = hand_parser
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())

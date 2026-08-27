#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frame-aware wrapper around the validated PartTrace v3.2 trainer.

The canonical Attention-Lite source was originally authored/configured at T16,
but its actual top-level model is sequence-length agnostic apart from requiring
T to be divisible by chunk_size=4 and clip_size=8. This wrapper changes the
runtime frame count *after* loading the exact canonical source and *before*
model/dataset construction, preserving the canonical architecture/parameter
code while allowing real T32/T64/etc. runs.

It also updates the PartTrace SharedPartTemporal relative-bias horizon by
changing audit_model.FRAMES before model initialization.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

from experiments.parttrace_v3_attention_lite import audit_model as audit
from experiments.parttrace_v3_attention_lite import train_v32 as base


def _extract_frames(argv: list[str]) -> tuple[int, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--frames",
        type=int,
        default=int(os.environ.get("NESTSAR_FRAMES", "16")),
    )
    known, rest = parser.parse_known_args(argv)
    return int(known.frames), rest


def main() -> int:
    frames, remaining = _extract_frames(sys.argv[1:])

    if frames < 8:
        raise ValueError(f"--frames must be >= 8; got {frames}")
    if frames % 8:
        raise ValueError(
            f"--frames must be divisible by 8 for canonical chunk_size=4 / clip_size=8; got {frames}"
        )

    os.environ["NESTSAR_FRAMES"] = str(frames)

    # train_v32 imported FRAMES by value from audit_model, while
    # SharedPartTemporal resolves audit_model.FRAMES from its defining module at
    # apply/init time. Update both so dummy inputs and relative-bias buckets use
    # the requested sequence length.
    audit.FRAMES = frames
    base.FRAMES = frames

    original_load = base.load_canonical_prefix
    original_parse = base.parse_args

    def load_with_frames(protocol: str):
        mod, source = original_load(protocol)
        mod.ns.CFG = dataclasses.replace(mod.ns.CFG, frames=frames)
        if int(mod.ns.CFG.frames) != frames:
            raise RuntimeError(
                f"Failed to set canonical runtime frames: {mod.ns.CFG.frames} != {frames}"
            )
        return mod, source

    def parse_with_frames():
        args = original_parse()
        args.frames = frames
        return args

    base.load_canonical_prefix = load_with_frames
    base.parse_args = parse_with_frames

    # The underlying trainer should not see --frames because this wrapper owns
    # that argument; all other v3.2 arguments remain unchanged.
    sys.argv = [sys.argv[0], *remaining]

    print(
        f"FRAME-AWARE V3.2: T={frames} | canonical chunk=4 clip=8 | "
        f"relative-bias horizon={frames}",
        flush=True,
    )
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())

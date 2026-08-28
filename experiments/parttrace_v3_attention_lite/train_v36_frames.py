#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frame-aware single-T4 worker for NestSAR v3.6.

Reuses the validated v3.5 optimization/evaluation loop, while changing only the
runtime pieces required for real T16/T32/T64 training:
- canonical ns.CFG.frames is replaced before dataset construction;
- the canonical base audit dummy uses the requested T;
- v3.6 lets the base see full T while side memory stays at 16 anchors;
- checkpoint/result metadata record the real runtime frame count.

This is intentionally a wrapper around train_v35_t4.py so tqdm metrics, EMA,
confusion matrices, early stopping and optimizer behavior stay identical.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.parttrace_v3_attention_lite import train_v35_t4 as base
from experiments.parttrace_v3_attention_lite.model_v36_multiframe import make_wrapper_v36

MODEL_NAME_V36 = "AttentionLiteCrossStreamMultiResolutionMemoryV36"
MEMORY_FRAMES = 16
GATE_SCALE = 0.50
UNCERTAINTY_FLOOR = 0.35


def _runtime_args(argv: list[str]):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    p.add_argument("--outdir", default="/kaggle/working/NestSAR_CrossStreamMemory_v36_FROM_ZERO_DualT4")
    known, _ = p.parse_known_args(argv)
    frames = int(os.environ.get("NESTSAR_FRAMES", "16"))
    return known, frames


def main() -> int:
    runtime, frames = _runtime_args(sys.argv[1:])
    if frames not in (16, 32, 64):
        raise ValueError(f"v3.6 supports --frames 16, 32, or 64; got {frames}")
    if frames % MEMORY_FRAMES:
        raise ValueError(f"T={frames} must be divisible by memory_frames={MEMORY_FRAMES}")

    # train_v35_t4 imported FRAMES by value.  Its dummy/audit must use the real
    # full-resolution sequence length, while its CLI guard still sees 16 below.
    base.FRAMES = frames
    base.MODEL_NAME = MODEL_NAME_V36

    original_load = base.load_canonical_prefix
    original_parse = base.parse_args
    original_factory = base.make_wrapper_v35
    original_payload = base._checkpoint_payload

    def load_with_frames(protocol: str):
        mod, source = original_load(protocol)
        if not hasattr(mod.ns, "CFG"):
            raise RuntimeError("Canonical runtime namespace has no CFG; cannot set frames safely")
        mod.ns.CFG = dataclasses.replace(mod.ns.CFG, frames=frames)
        actual = int(mod.ns.CFG.frames)
        if actual != frames:
            raise RuntimeError(f"Failed to set canonical runtime frames: {actual} != {frames}")
        print(
            f"V3.6 CANONICAL RUNTIME: protocol={protocol.upper()} "
            f"frames={actual} chunk=4 clip=8",
            flush=True,
        )
        return mod, source

    def parse_for_reused_loop():
        args = original_parse()
        # train_v35_t4 has a historical T16 guard.  The actual tensor length is
        # owned by this wrapper via base.FRAMES + ns.CFG.frames.
        args.frames = 16
        args.runtime_frames = frames
        args.memory_frames = MEMORY_FRAMES
        args.gate_scale = GATE_SCALE
        args.uncertainty_floor = UNCERTAINTY_FLOOR
        return args

    def factory(base_model, **kwargs):
        return make_wrapper_v36(
            base_model,
            input_frames=frames,
            memory_frames=MEMORY_FRAMES,
            gate_scale=GATE_SCALE,
            uncertainty_floor=UNCERTAINTY_FLOOR,
            **kwargs,
        )

    def payload_with_runtime(*args, **kwargs):
        payload = original_payload(*args, **kwargs)
        payload["model"] = MODEL_NAME_V36
        payload["runtime_frames"] = frames
        payload["memory_frames"] = MEMORY_FRAMES
        payload["memory_tokens"] = MEMORY_FRAMES * 2 * (10 + 10)
        payload["gate_scale"] = GATE_SCALE
        payload["uncertainty_floor"] = UNCERTAINTY_FLOOR
        saved_args = dict(payload.get("args", {}))
        saved_args["frames"] = frames
        saved_args["runtime_frames"] = frames
        saved_args["memory_frames"] = MEMORY_FRAMES
        saved_args["gate_scale"] = GATE_SCALE
        saved_args["uncertainty_floor"] = UNCERTAINTY_FLOOR
        payload["args"] = saved_args
        return payload

    base.load_canonical_prefix = load_with_frames
    base.parse_args = parse_for_reused_loop
    base.make_wrapper_v35 = factory
    base._checkpoint_payload = payload_with_runtime

    print("=" * 120, flush=True)
    print("NESTSAR v3.6 — FRAME-AWARE CROSS-STREAM MEMORY", flush=True)
    print(
        f"Full Attention-Lite input: T{frames} | side memory anchors: T{MEMORY_FRAMES} "
        f"| memory tokens=640 | temporal reduction={frames // MEMORY_FRAMES}x",
        flush=True,
    )
    print(
        f"Correction gate: v3.5 gate x {GATE_SCALE:.2f}, uncertainty floor={UNCERTAINTY_FLOOR:.2f}",
        flush=True,
    )
    print("=" * 120, flush=True)

    try:
        rc = int(base.main())
    finally:
        base.load_canonical_prefix = original_load
        base.parse_args = original_parse
        base.make_wrapper_v35 = original_factory
        base._checkpoint_payload = original_payload

    # train_v35_t4's result writer contains historical literal T16 metadata.
    # Correct it after successful completion so paper/audit artifacts are exact.
    result_path = Path(runtime.outdir) / runtime.protocol / "result.json"
    if rc == 0 and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["model"] = MODEL_NAME_V36
        result["frames"] = frames
        result["runtime_frames"] = frames
        result["memory_frames"] = MEMORY_FRAMES
        result["coarse_tokens"] = MEMORY_FRAMES * 2 * 10
        result["fine_tokens"] = MEMORY_FRAMES * 2 * 10
        result["memory_tokens"] = MEMORY_FRAMES * 2 * 20
        result["temporal_reduction_factor"] = frames // MEMORY_FRAMES
        result["gate_scale"] = GATE_SCALE
        result["uncertainty_floor"] = UNCERTAINTY_FLOOR
        if isinstance(result.get("args"), dict):
            result["args"]["frames"] = frames
            result["args"]["runtime_frames"] = frames
            result["args"]["memory_frames"] = MEMORY_FRAMES
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())

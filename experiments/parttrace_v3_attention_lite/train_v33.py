#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train TokenPreserve v3.3 using the validated v3.2 training engine.

The training engine is reused deliberately so the optimizer/EMA/data path stays
matched to the v3.2 ablation. Only the residual architecture changes.

v3.3 invariants:
- T16 only
- exact 2,381,028-param Attention-Lite base
- no dynamic stream controller; dynamic_fusion_weights == base fusion weights
- 320 anatomical tokens preserved to K learned readout queries
"""
from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.parttrace_v3_attention_lite import train_v32 as base
from experiments.parttrace_v3_attention_lite.model_v33_tokenpreserve import make_wrapper_v33

MODEL_NAME = "AttentionLiteTokenPreserveV33"


def _extract_v33_args(argv: list[str]):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument(
        "--readout-tokens",
        type=int,
        default=int(os.environ.get("NESTSAR_READOUT_TOKENS", "8")),
    )
    known, rest = p.parse_known_args(argv)
    return known, rest


def _locate_output_args(argv: list[str]):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    p.add_argument("--outdir", default="/kaggle/working/NestSAR_TokenPreserve_v33")
    known, _ = p.parse_known_args(argv)
    return known


def main() -> int:
    v33, remaining = _extract_v33_args(sys.argv[1:])
    if v33.frames != 16:
        raise ValueError(
            f"TokenPreserve v3.3 is the T16 Pareto experiment; got --frames={v33.frames}. "
            "Use --frames 16."
        )
    if not 1 <= v33.readout_tokens <= 32:
        raise ValueError("--readout-tokens must be in [1, 32]")

    output_args = _locate_output_args(remaining)
    os.environ["NESTSAR_READOUT_TOKENS"] = str(v33.readout_tokens)

    # train_v32 owns the stable data/optimizer/EMA loop. Remove v3.3-only args
    # before handing argparse control to that engine.
    sys.argv = [sys.argv[0], *remaining]

    original_parse = base.parse_args
    original_to_bytes = base.serialization.to_bytes
    original_print = builtins.print

    def parse_v33():
        a = original_parse()
        a.frames = 16
        a.readout_tokens = int(v33.readout_tokens)
        return a

    def build_v33(
        base_model,
        *,
        part_dim,
        part_heads,
        global_dim,
        dense_dim,
        branch_dropout,
    ):
        return make_wrapper_v33(
            base_model,
            part_dim=part_dim,
            part_heads=part_heads,
            global_dim=global_dim,
            dense_dim=dense_dim,
            branch_dropout=branch_dropout,
            readout_tokens=v33.readout_tokens,
        )

    def to_bytes_v33(payload):
        if isinstance(payload, dict) and payload.get("model") == "AttentionLitePartTraceV32":
            payload = dict(payload)
            payload["model"] = MODEL_NAME
            payload["readout_tokens"] = int(v33.readout_tokens)
        return original_to_bytes(payload)

    def print_v33(*args, **kwargs):
        patched = []
        for arg in args:
            if isinstance(arg, str):
                arg = arg.replace(
                    "NESTSAR ATTENTION-LITE + PARTTRACE V3.2 — CONFIGURABLE DENSE TRAINER",
                    "NESTSAR ATTENTION-LITE + TOKENPRESERVE V3.3 — T16 PARETO TRAINER",
                )
                arg = arg.replace("XLA v3.2 GFLOPs", "XLA v3.3 GFLOPs")
                arg = arg.replace("V3.2 SIZE GUARD FAILED", "V3.3 SIZE GUARD FAILED")
            patched.append(arg)
        return original_print(*patched, **kwargs)

    base.parse_args = parse_v33
    base.make_wrapper_v32 = build_v33
    base.serialization.to_bytes = to_bytes_v33
    builtins.print = print_v33

    original_print("=" * 118, flush=True)
    original_print("TOKENPRESERVE V3.3 PRE-FLIGHT", flush=True)
    original_print(
        f"T=16 | fine tokens=16x2x10=320 | readout K={v33.readout_tokens} | "
        "exact Attention-Lite base retained",
        flush=True,
    )
    original_print("=" * 118, flush=True)

    try:
        code = int(base.main())
    finally:
        builtins.print = original_print
        base.parse_args = original_parse
        base.serialization.to_bytes = original_to_bytes

    # Normalize result metadata after the shared engine finishes.
    result_path = Path(output_args.outdir) / output_args.protocol / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["model"] = MODEL_NAME
        result["frames"] = 16
        result["fine_tokens"] = 16 * 2 * 10
        result["readout_tokens"] = int(v33.readout_tokens)
        widths = dict(result.get("widths", {}))
        widths["token_dim"] = widths.get("part_dim")
        widths["readout_heads"] = widths.get("part_heads")
        widths["mixer_hidden_dim"] = widths.get("global_dim")
        widths["readout_tokens"] = int(v33.readout_tokens)
        result["widths"] = widths
        audit = dict(result.get("audit", {}))
        if "v32_gflops" in audit:
            audit["v33_gflops"] = audit.pop("v32_gflops")
        result["audit"] = audit
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return code


if __name__ == "__main__":
    raise SystemExit(main())

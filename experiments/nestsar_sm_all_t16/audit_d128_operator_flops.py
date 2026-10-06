#!/usr/bin/env python3
from __future__ import annotations

"""Compiler-independent operator FLOP audit for current NestSAR D128-MTS.

This audit intentionally does NOT use XLA cost_analysis for the paper number.

It reports two independent counts:
  1) closed-form MAC count from the exact source equations/tensor shapes;
  2) raw JAXPR dot_general MAC count before XLA compilation.

The two counts must agree.  Paper FLOPs use 1 MAC = 2 FLOPs.

Elementwise/reduction operations are reported separately because paper FLOP
conventions vary for activations, normalization, comparisons, and reductions.
"""

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.nestsar_sm_all_t16.model_parallel import NestSARParallelT16
from experiments.nestsar_sm_all_t16.parallel_d128_config import (
    EXPECTED_PARAMS,
    MODEL_DIM,
    M4_HALF_LIVES,
    G4_HALF_LIVES,
)


# Exact architecture constants.
B = 1
T = 16
M = 2
V = 25
TOKEN_C = 15
S = 4
DSP = 24
D = MODEL_DIM
R = 4
PARTS = 10
CLASSES = 120
HEAD_R = 2
G = 4
FEATURES = 750


def prod(xs):
    out = 1
    for x in xs:
        out *= int(x)
    return int(out)


def count_params(params):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)))


def assoc_combine_count(n: int) -> int:
    """Number of element-wise combine applications used by JAX associative_scan.

    Mirrors the recursive algorithm in jax.lax.associative_scan.  This is useful
    for the explicit matrix products inside associative fast-weight composition.
    """
    if n < 2:
        return 0
    m = n // 2
    reduced = m
    recursive = assoc_combine_count(m)
    even = (m - 1) if n % 2 == 0 else m
    return reduced + recursive + even


def closed_form_dot_macs(dim: int):
    """Count matrix/contraction MACs from the exact current equations."""
    d = int(dim)

    rows = {}

    # Shared controller Dense layers.
    rows["controller_dense"] = (
        T * 15 * 16
        + 2 * T * 16 * 15
        + T * 16 * S
        + T * 16 * 2
        + 16 * S
        + 16 * HEAD_R
    )

    # Four spatial streams: J, B, JM, BM.
    # Input channels are 3,3,12,12.
    rows["spatial_input_projection"] = T * M * V * DSP * (3 + 3 + 12 + 12)

    # ParallelAffineSweep at the joint level:
    # forget Dsp->Dsp
    # candidate_in Dsp->2Dsp
    # candidate_out 2Dsp->Dsp
    # skip Dsp->Dsp
    # = 6*Dsp^2 MAC/token.
    rows["spatial_joint_memory_projections"] = S * T * M * V * 6 * DSP * DSP

    # Part pooling einsum btmvd,pv->btmpd.
    rows["spatial_part_pool_einsum"] = S * T * M * PARTS * DSP * V

    # [M*10*Dsp] -> D per frame/stream.
    rows["spatial_part_fuse"] = S * T * (M * PARTS * DSP) * d

    # M4 ParallelBiMemory:
    # two directions, each 6D^2 => 12D^2
    # merge [2D]->D => 2D^2
    rows["m4_affine_memory_projections"] = S * T * 12 * d * d
    rows["m4_bidirectional_merge"] = S * T * 2 * d * d

    # M4 fast-weight projections and actual associative matrix products.
    rows["m4_fast_key_query"] = S * T * 2 * d * R
    rows["m4_fast_assoc_matmuls"] = (
        S
        * assoc_combine_count(T)
        * (R**3 + R * R * d)
    )
    rows["m4_fast_prefix_memory"] = S * T * R * R * d
    rows["m4_fast_read"] = S * T * R * d

    # Cross-stream router Dense operators.
    rows["router_score"] = T * S * d
    rows["router_context_projection"] = T * d * d
    rows["router_gate"] = T * S * d

    # G4 at 4 chunks/stream.
    rows["g4_affine_memory_projections"] = S * G * 12 * d * d
    rows["g4_bidirectional_merge"] = S * G * 2 * d * d
    rows["g4_fast_key_query"] = S * G * 2 * d * R
    rows["g4_fast_assoc_matmuls"] = (
        S
        * assoc_combine_count(G)
        * (R**3 + R * R * d)
    )
    rows["g4_fast_prefix_memory"] = S * G * R * R * d
    rows["g4_fast_read"] = S * G * R * d

    # Descriptor and classifiers.
    rows["descriptor_hier_fuse"] = S * 2 * d * d
    rows["stream_classifiers"] = S * d * CLASSES

    # Weighted final contractions implemented as einsums.
    rows["stream_logit_fusion"] = S * CLASSES
    rows["descriptor_fusion"] = S * d

    # Adaptive low-rank head.
    rows["adaptive_head_u"] = d * HEAD_R
    rows["adaptive_head_v"] = HEAD_R * CLASSES

    total = int(sum(rows.values()))
    return rows, total


def numel_var(v):
    aval = getattr(v, "aval", None)
    shape = getattr(aval, "shape", ())
    return prod(shape)


def nested_jaxpr_from_eqn(eqn):
    """Return only nested forward jaxprs that are actually executed."""
    found = []

    # Common call primitives.
    for key in ("jaxpr", "call_jaxpr"):
        value = eqn.params.get(key)
        if value is not None:
            found.append(value)

    # We deliberately reject control-flow branches below rather than summing
    # both branches, because that would not be an execution FLOP count.
    return found


def dot_macs_from_jaxpr(closed):
    """Count all raw-JAXPR dot_general MACs before any compiler optimization."""
    by_shape = {}
    total = 0
    primitives = {}

    def walk(obj):
        nonlocal total

        j = getattr(obj, "jaxpr", obj)
        for eqn in j.eqns:
            name = eqn.primitive.name
            primitives[name] = primitives.get(name, 0) + 1

            if name in ("scan", "while", "cond"):
                raise RuntimeError(
                    f"Unexpected control-flow primitive in current parallel inference graph: {name}"
                )

            nested = nested_jaxpr_from_eqn(eqn)
            if nested:
                for child in nested:
                    walk(child)
                continue

            if name != "dot_general":
                continue

            lhs = eqn.invars[0]
            rhs = eqn.invars[1]
            lhs_shape = tuple(int(x) for x in lhs.aval.shape)
            rhs_shape = tuple(int(x) for x in rhs.aval.shape)
            out_shape = tuple(int(x) for x in eqn.outvars[0].aval.shape)

            dims = eqn.params["dimension_numbers"]
            lhs_contract = tuple(int(x) for x in dims[0][0])

            contraction = prod(lhs_shape[i] for i in lhs_contract)
            macs = prod(out_shape) * contraction

            key = f"{lhs_shape} x {rhs_shape} -> {out_shape} K={contraction}"
            by_shape[key] = by_shape.get(key, 0) + int(macs)
            total += int(macs)

    walk(closed)
    return int(total), by_shape, primitives


def arithmetic_ops_from_jaxpr(closed):
    """Approximate non-MAC floating arithmetic from raw JAXPR.

    This is reported separately and is NOT used as the primary paper number.
    It counts standard scalar floating arithmetic and reductions. Structural
    operations, comparisons, gathers, reshapes, etc. are zero-cost here.
    """
    counts = {}
    unknown = {}

    unary = {
        "abs", "neg", "exp", "expm1", "log", "log1p", "sqrt", "rsqrt",
        "tanh", "logistic", "erf", "sin", "cos",
    }
    binary = {
        "add", "add_any", "sub", "mul", "div", "pow",
    }
    zero = {
        "broadcast_in_dim", "reshape", "transpose", "slice", "dynamic_slice",
        "concatenate", "gather", "scatter", "iota", "rev", "copy",
        "device_put", "stop_gradient", "convert_element_type",
        "bitcast_convert_type", "squeeze", "pad", "select_n",
        "eq", "ne", "lt", "le", "gt", "ge", "and", "or", "xor", "not",
        "reduce_and", "reduce_or", "reduce_max", "reduce_min",
        "max", "min", "clamp", "sort", "top_k",
        "dot_general",
    }

    total = 0

    def add(name, n):
        nonlocal total
        n = int(n)
        counts[name] = counts.get(name, 0) + n
        total += n

    def walk(obj):
        j = getattr(obj, "jaxpr", obj)

        for eqn in j.eqns:
            name = eqn.primitive.name

            if name in ("scan", "while", "cond"):
                raise RuntimeError(f"Unexpected control flow in arithmetic audit: {name}")

            nested = nested_jaxpr_from_eqn(eqn)
            if nested:
                for child in nested:
                    walk(child)
                continue

            out_n = 0
            for v in eqn.outvars:
                out_n += numel_var(v)

            if name in unary:
                add(name, out_n)
            elif name in binary:
                add(name, out_n)
            elif name == "integer_pow":
                exponent = int(eqn.params.get("y", 1))
                # x^0 and x^1 need no multiply; x^k roughly k-1 multiplies.
                add(name, out_n * max(abs(exponent) - 1, 0))
            elif name == "reduce_sum":
                in_n = numel_var(eqn.invars[0])
                add(name, max(in_n - out_n, 0))
            elif name == "reduce_prod":
                in_n = numel_var(eqn.invars[0])
                add(name, max(in_n - out_n, 0))
            elif name in zero:
                pass
            else:
                # Keep the audit transparent: report anything not classified.
                unknown[name] = unknown.get(name, 0) + 1

    walk(closed)
    return int(total), counts, unknown


def make_model(dim, mts):
    return NestSARParallelT16(
        spatial_dim=DSP,
        model_dim=dim,
        dropout=0.10,
        controller_dim=16,
        fast_rank=R,
        head_rank=HEAD_R,
        sm_residual_scale=0.08,
        head_residual_scale=0.15,
        m4_half_lives=M4_HALF_LIVES if mts else None,
        g4_half_lives=G4_HALF_LIVES if mts else None,
    )


def init(model, seed=128):
    key = jax.random.PRNGKey(seed)
    x = jnp.zeros((B, T, FEATURES), jnp.float32)
    params = model.init({"params": key, "dropout": key}, x, training=False)["params"]
    return params, x


def audit_one(dim, mts):
    model = make_model(dim, mts)
    params, x = init(model)
    fn = lambda p, xx: model.apply({"params": p}, xx, training=False)["logits"]

    closed = jax.make_jaxpr(fn)(params, x)
    jaxpr_macs, dot_shapes, primitives = dot_macs_from_jaxpr(closed)
    nonmac_ops, nonmac_breakdown, unknown = arithmetic_ops_from_jaxpr(closed)
    closed_rows, closed_macs = closed_form_dot_macs(dim)

    if closed_macs != jaxpr_macs:
        raise AssertionError(
            "Independent MAC counts disagree:\n"
            f"  closed-form = {closed_macs:,}\n"
            f"  raw JAXPR   = {jaxpr_macs:,}\n"
            "Inspect the architecture before quoting a paper FLOP number."
        )

    return {
        "params": count_params(params),
        "closed_form_mac_breakdown": closed_rows,
        "closed_form_macs": closed_macs,
        "raw_jaxpr_dot_macs": jaxpr_macs,
        "mac_counts_match": True,
        "paper_flops_1mac_eq_2": 2 * closed_macs,
        "paper_mflops_1mac_eq_2": 2 * closed_macs / 1e6,
        "paper_gflops_1mac_eq_2": 2 * closed_macs / 1e9,
        "approx_nonmac_arithmetic_ops": nonmac_ops,
        "approx_total_arithmetic_including_nonmac": 2 * closed_macs + nonmac_ops,
        "approx_total_mflops_including_nonmac": (2 * closed_macs + nonmac_ops) / 1e6,
        "unclassified_raw_jaxpr_primitives": unknown,
        "dot_shape_macs": dot_shapes,
        "primitive_counts": primitives,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise RuntimeError(
            f"Expected one isolated GPU; backend={jax.default_backend()} "
            f"devices={jax.local_devices()}"
        )

    print("=" * 124)
    print("NESTSAR — COMPILER-INDEPENDENT OPERATOR FLOP AUDIT")
    print("=" * 124)
    print("Device:", jax.local_devices()[0])
    print("Primary convention: raw mathematical contractions, 1 MAC = 2 FLOPs")
    print("XLA cost_analysis is NOT used for the paper number.")
    print()

    d112 = audit_one(112, False)
    d128 = audit_one(D, True)

    if d112["params"] != 1_831_932:
        raise AssertionError(f"D112 params drift: {d112['params']:,}")
    if d128["params"] != EXPECTED_PARAMS:
        raise AssertionError(f"D128 params drift: {d128['params']:,}")

    # Hard hand-derived lower bound: just spatial joint memories + M4 + G4
    # base affine memories, before input/fusion/router/fast-weight/classifier.
    spatial_joint_floor = S * T * M * V * 6 * DSP * DSP * 2
    m4_base_floor = S * T * (12 * D * D + 2 * D * D) * 2
    g4_base_floor = S * G * (12 * D * D + 2 * D * D) * 2
    hard_floor = spatial_joint_floor + m4_base_floor + g4_base_floor

    if d128["paper_flops_1mac_eq_2"] < hard_floor:
        raise AssertionError(
            f"D128 count {d128['paper_flops_1mac_eq_2']:,} is below "
            f"the hand-derived hard floor {hard_floor:,}"
        )

    result = {
        "convention": {
            "primary": "compiler-independent raw contraction count; 1 MAC = 2 FLOPs",
            "batch": B,
            "frames": T,
            "preprocessing_included": False,
            "xla_cost_analysis_used_for_primary_number": False,
            "elementwise_reported_separately": True,
        },
        "d112_parallel": d112,
        "d128_mts": d128,
        "hard_lower_bound": {
            "spatial_joint_memory_mflops": spatial_joint_floor / 1e6,
            "m4_base_memory_mflops": m4_base_floor / 1e6,
            "g4_base_memory_mflops": g4_base_floor / 1e6,
            "combined_mflops": hard_floor / 1e6,
        },
        "comparison": {
            "d128_vs_d112_ratio": (
                d128["paper_flops_1mac_eq_2"] / d112["paper_flops_1mac_eq_2"]
            ),
            "d128_minus_d112_mflops": (
                d128["paper_flops_1mac_eq_2"] - d112["paper_flops_1mac_eq_2"]
            ) / 1e6,
        },
    }

    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))

    print("HARD LOWER BOUND — D128")
    print("-" * 76)
    print(f"Spatial joint memories : {spatial_joint_floor/1e6:12.6f} MFLOPs")
    print(f"M4 base memories       : {m4_base_floor/1e6:12.6f} MFLOPs")
    print(f"G4 base memories       : {g4_base_floor/1e6:12.6f} MFLOPs")
    print(f"Combined floor         : {hard_floor/1e6:12.6f} MFLOPs")
    print()

    print("PRIMARY PAPER COUNT — 1 MAC = 2 FLOPs")
    print("-" * 96)
    print(f"{'MODEL':24s} {'PARAMS':>12s} {'MACs':>14s} {'MFLOPs':>14s} {'GFLOPs':>14s}")
    print("-" * 96)
    for name, row in (("Parallel D112", d112), ("Parallel D128-MTS", d128)):
        print(
            f"{name:24s} "
            f"{row['params']:12,d} "
            f"{row['closed_form_macs']:14,d} "
            f"{row['paper_mflops_1mac_eq_2']:14.6f} "
            f"{row['paper_gflops_1mac_eq_2']:14.9f}"
        )

    print()
    print("INDEPENDENT CROSS-CHECK")
    print("-" * 76)
    print(
        f"D112 closed-form MACs = {d112['closed_form_macs']:,} | "
        f"raw JAXPR = {d112['raw_jaxpr_dot_macs']:,}"
    )
    print(
        f"D128 closed-form MACs = {d128['closed_form_macs']:,} | "
        f"raw JAXPR = {d128['raw_jaxpr_dot_macs']:,}"
    )
    print("Counts match            :", d128["mac_counts_match"] and d112["mac_counts_match"])
    print()

    print("NON-MAC ARITHMETIC — REPORTED SEPARATELY")
    print("-" * 76)
    print(
        f"D112 approx scalar/reduction ops : "
        f"{d112['approx_nonmac_arithmetic_ops']:,}"
    )
    print(
        f"D128 approx scalar/reduction ops : "
        f"{d128['approx_nonmac_arithmetic_ops']:,}"
    )
    print(
        f"D128 MAC FLOPs + these ops       : "
        f"{d128['approx_total_mflops_including_nonmac']:.6f} MFLOPs"
    )
    print(
        "Unclassified D128 primitives       :",
        d128["unclassified_raw_jaxpr_primitives"],
    )
    print()

    print("D128 CLOSED-FORM MAC BREAKDOWN")
    print("-" * 76)
    for key, value in d128["closed_form_mac_breakdown"].items():
        print(f"{key:38s} {value:12,d} MACs  {2*value/1e6:10.6f} MFLOPs")

    print()
    print("COMPARISON")
    print("-" * 76)
    print(
        f"D128 / D112 paper FLOPs : "
        f"{result['comparison']['d128_vs_d112_ratio']:.6f}x"
    )
    print(
        f"D128 - D112             : "
        f"{result['comparison']['d128_minus_d112_mflops']:+.6f} MFLOPs"
    )
    print()
    print("=" * 124)
    print("PAPER-CANDIDATE NUMBER")
    print("=" * 124)
    print(
        f"NestSAR D128-MTS = "
        f"{d128['paper_mflops_1mac_eq_2']:.6f} MFLOPs "
        f"= {d128['paper_gflops_1mac_eq_2']:.9f} GFLOPs "
        f"(1 MAC = 2 FLOPs)"
    )
    print(f"Parameters = {d128['params']:,}")
    print("Independent closed-form/JAXPR MAC agreement = PASS")
    print("=" * 124)


if __name__ == "__main__":
    main()

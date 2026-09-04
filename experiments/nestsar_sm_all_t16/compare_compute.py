#!/usr/bin/env python3
from __future__ import annotations

"""Apples-to-apples GPU-XLA compute audit for NestSAR low-compute variants.

Measures on one visible GPU with the exact same JAX/XLA cost_analysis path:
  1) LocalGlobal V2 T16
  2) Hand_M4G4_T32 champion architecture
  3) NestSAR-SM-ALL-T16

Also calibrates the SM-ALL value back to the historical NestSAR paper convention
using both known historical anchors:
  LocalGlobal V2 = 0.020181640 GFLOPs/clip
  Hand_M4G4_T32  = 0.020630000 GFLOPs/clip

The calibrated values are estimates unless the historical anchors were themselves
measured with exactly the same FLOP definition. The raw same-runtime GPU-XLA
numbers and their ratios are the authoritative comparison within this audit.
"""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.model import (
    M4LocalGlobalHandM4G4T32,
)
from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.preprocessing import (
    HAND_FEATURES,
    HAND_FRAMES,
)
from experiments.nestsar_sm_all_t16.model import (
    FEATURES,
    FRAMES,
    NestSARSMAllT16,
)

HIST_LOCALGLOBAL_GFLOPS = 0.020181640
HIST_HAND_GFLOPS = 0.020630000


def xla_cost(fn, *args):
    compiled = jax.jit(fn).lower(*args).compile()
    ca = compiled.cost_analysis()
    if isinstance(ca, list):
        ca = ca[0] if ca else {}
    return float(ca.get("flops", float("nan")))


def count_params(params):
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))


def main():
    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"Run with exactly one visible GPU; backend={jax.default_backend()} "
            f"devices={jax.local_devices()}"
        )

    print("=" * 118)
    print("NESTSAR SAME-RUNTIME COMPUTE CALIBRATION | GPU-XLA | BATCH=1")
    print("=" * 118)
    print("JAX:", jax.__version__)
    print("DEVICE:", jax.local_devices()[0])

    key = jax.random.PRNGKey(128)
    k1, k2, k3 = jax.random.split(key, 3)

    # 1) LocalGlobal V2 / fixed-uniform T16 architecture.
    local = ju.M4PhaseUniformT16(spatial_dim=24, model_dim=112, dropout=0.10)
    x = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    p_local = local.init({"params": k1, "dropout": k1}, x, training=False)["params"]
    f_local = xla_cost(
        lambda p, xx: local.apply({"params": p}, xx, training=False)["logits"],
        p_local,
        x,
    )

    # 2) Hand_M4G4_T32 champion architecture.
    hand = M4LocalGlobalHandM4G4T32(
        spatial_dim=24,
        model_dim=112,
        dropout=0.10,
        hand_dim=32,
        hand_residual_scale=0.10,
    )
    h = jnp.zeros((1, HAND_FRAMES, HAND_FEATURES), jnp.float32)
    p_hand = hand.init({"params": k2, "dropout": k2}, x, h, training=False)["params"]
    f_hand = xla_cost(
        lambda p, xx, hh: hand.apply({"params": p}, xx, hh, training=False)["logits"],
        p_hand,
        x,
        h,
    )

    # 3) New SM-ALL-T16 architecture.
    sm = NestSARSMAllT16(
        spatial_dim=24,
        model_dim=112,
        dropout=0.10,
        controller_dim=16,
        fast_rank=2,
        head_rank=2,
        sm_residual_scale=0.08,
        head_residual_scale=0.15,
    )
    p_sm = sm.init({"params": k3, "dropout": k3}, x, training=False)["params"]
    f_sm = xla_cost(
        lambda p, xx: sm.apply({"params": p}, xx, training=False)["logits"],
        p_sm,
        x,
    )

    rows = [
        ("LocalGlobal_V2", p_local, f_local),
        ("Hand_M4G4_T32", p_hand, f_hand),
        ("SM_ALL_T16", p_sm, f_sm),
    ]

    print("\nRAW SAME-RUNTIME GPU-XLA")
    print("model                params        MFLOPs      GFLOPs")
    print("-" * 72)
    for name, params, flops in rows:
        print(
            f"{name:20s} {count_params(params):10,d} "
            f"{flops/1e6:12.6f} {flops/1e9:12.9f}"
        )

    print("\nRELATIVE COMPUTE")
    print(f"SM / LocalGlobal = {f_sm/f_local:.9f}  ({100*(f_sm/f_local-1):+.3f}%)")
    print(f"SM / Hand        = {f_sm/f_hand:.9f}  ({100*(f_sm/f_hand-1):+.3f}%)")

    # Map the same-runtime ratio onto the user's historical reporting convention.
    sm_from_local = HIST_LOCALGLOBAL_GFLOPS * (f_sm / f_local)
    sm_from_hand = HIST_HAND_GFLOPS * (f_sm / f_hand)
    sm_cal = 0.5 * (sm_from_local + sm_from_hand)
    spread = abs(sm_from_local - sm_from_hand) / 2.0

    print("\nHISTORICAL-CONVENTION CALIBRATION")
    print(f"From LocalGlobal anchor : {sm_from_local:.9f} GFLOPs")
    print(f"From Hand anchor        : {sm_from_hand:.9f} GFLOPs")
    print(f"Calibrated midpoint     : {sm_cal:.9f} GFLOPs")
    print(f"Half-spread             : {spread:.9f} GFLOPs")

    print("\nREPORTING")
    print(f"Exact GPU-XLA counter   : {f_sm/1e9:.9f} GFLOPs/clip")
    print(f"Paper-comparable est.   : {sm_cal:.9f} GFLOPs/clip")
    print("Use the exact GPU-XLA value only when comparing models audited with this same runtime/path.")
    print("Use the calibrated historical estimate only to maintain continuity with the older NestSAR table.")
    print("=" * 118)


if __name__ == "__main__":
    main()

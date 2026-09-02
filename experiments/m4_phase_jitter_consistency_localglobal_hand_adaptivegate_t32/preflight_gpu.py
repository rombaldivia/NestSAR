#!/usr/bin/env python3
from __future__ import annotations

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
from experiments.m4_phase_jitter_consistency_localglobal_hand_adaptivegate_t32.model import (
    EXPECTED_TOTAL_PARAMS,
    GATE_EXTRA_PARAMS,
    M4LocalGlobalHandAdaptiveGateT32,
)

BASE_PARAMS = 1_854_650


def count_params(params) -> int:
    return int(
        sum(
            np.asarray(x).size
            for x in jax.tree_util.tree_leaves(params)
        )
    )


def xla_flops(model, params):
    x = jnp.zeros((1, ju.FRAMES, ju.FEATURES), jnp.float32)
    h = jnp.zeros((1, HAND_FRAMES, HAND_FEATURES), jnp.float32)
    fn = jax.jit(
        lambda p, xx, hh: model.apply(
            {"params": p}, xx, hh, training=False
        )["logits"]
    )
    compiled = fn.lower(params, x, h).compile()
    ca = compiled.cost_analysis()
    if isinstance(ca, list):
        ca = ca[0] if ca else {}
    return float(ca.get("flops", float("nan")))


def main() -> None:
    print("=" * 120)
    print("NESTSAR HAND-M4/G4 T32 + ADAPTIVE TRUST GATE — GPU PREFLIGHT")
    print("=" * 120)
    print("JAX:", jax.__version__)
    print("BACKEND:", jax.default_backend())
    print("LOCAL DEVICES:", jax.local_device_count())
    print("DEVICES:", jax.local_devices())

    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, got backend={jax.default_backend()} "
            f"count={jax.local_device_count()}"
        )

    x = jnp.zeros((1, ju.FRAMES, ju.FEATURES), jnp.float32)
    h = jnp.zeros((1, HAND_FRAMES, HAND_FEATURES), jnp.float32)
    key = jax.random.PRNGKey(128)

    base = M4LocalGlobalHandM4G4T32(
        spatial_dim=24,
        model_dim=112,
        dropout=0.10,
        hand_dim=32,
        hand_residual_scale=0.10,
    )
    base_params = base.init(
        {"params": key, "dropout": key}, x, h, training=False
    )["params"]

    gated = M4LocalGlobalHandAdaptiveGateT32(
        spatial_dim=24,
        model_dim=112,
        dropout=0.10,
        hand_dim=32,
        gate_hidden_dim=16,
        max_alpha=0.30,
    )
    gate_params = gated.init(
        {"params": key, "dropout": key}, x, h, training=False
    )["params"]

    nb = count_params(base_params)
    ng = count_params(gate_params)

    if nb != BASE_PARAMS:
        raise RuntimeError(f"Base params mismatch: {nb:,} != {BASE_PARAMS:,}")
    if ng != EXPECTED_TOTAL_PARAMS:
        raise RuntimeError(
            f"Adaptive params mismatch: {ng:,} != {EXPECTED_TOTAL_PARAMS:,}"
        )

    fb = xla_flops(base, base_params)
    fg = xla_flops(gated, gate_params)

    print(f"BASE HAND MODEL PARAMS: {nb:,}")
    print(f"ADAPTIVE MODEL PARAMS: {ng:,}")
    print(f"GATE ADDED PARAMS: {ng-nb:,}")
    print(f"EXPECTED GATE PARAMS: {GATE_EXTRA_PARAMS:,}")
    print(f"PARAM INCREASE: {100.0*(ng/nb-1.0):.4f}%")
    print(f"BASE GPU-XLA FLOPS/CLIP: {fb:,.0f}")
    print(f"BASE GPU-XLA GFLOPS/CLIP: {fb/1e9:.9f}")
    print(f"ADAPTIVE GPU-XLA FLOPS/CLIP: {fg:,.0f}")
    print(f"ADAPTIVE GPU-XLA GFLOPS/CLIP: {fg/1e9:.9f}")
    print(f"INCREMENTAL GPU-XLA MFLOPS: {(fg-fb)/1e6:.6f}")
    print(f"GPU-XLA COMPUTE INCREASE: {100.0*(fg/fb-1.0):.4f}%")
    print("MAX ALPHA: 0.30")
    print("ATTENTION: NONE")
    print("PREFLIGHT=PASS")
    print("=" * 120)


if __name__ == "__main__":
    main()

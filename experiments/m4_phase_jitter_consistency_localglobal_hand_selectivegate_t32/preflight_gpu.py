#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.model import (
    M4LocalGlobalHandM4G4T32,
)
from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.preprocessing import (
    HAND_FEATURES,
    HAND_FRAMES,
)
from experiments.m4_phase_jitter_consistency_localglobal_hand_selectivegate_t32.model import (
    M4LocalGlobalHandSelectiveGateT32,
    EXPECTED_TOTAL_PARAMS,
    GATE_EXTRA_PARAMS,
)

BASE_HAND_PARAMS = 1_854_650


def count_params(tree) -> int:
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def xla_flops(fn, *args) -> float:
    compiled = jax.jit(fn).lower(*args).compile()
    ca = compiled.cost_analysis()
    if isinstance(ca, list):
        ca = ca[0] if ca else {}
    return float(ca.get("flops", float("nan")))


def main() -> None:
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"Expected one GPU; backend={jax.default_backend()} count={jax.local_device_count()}"
        )

    base_model = M4LocalGlobalHandM4G4T32(
        spatial_dim=24,
        model_dim=112,
        dropout=0.10,
        hand_dim=32,
        hand_residual_scale=0.10,
    )
    selective_model = M4LocalGlobalHandSelectiveGateT32(
        spatial_dim=24,
        model_dim=112,
        dropout=0.10,
        hand_dim=32,
        gate_hidden_dim=16,
        base_alpha=0.20,
        delta_alpha=0.15,
    )

    dm = jnp.zeros((1, ju.FRAMES, ju.FEATURES), jnp.float32)
    dh = jnp.zeros((1, HAND_FRAMES, HAND_FEATURES), jnp.float32)
    key = jax.random.PRNGKey(128)

    bp = base_model.init({"params": key, "dropout": key}, dm, dh, training=False)["params"]
    sp = selective_model.init({"params": key, "dropout": key}, dm, dh, training=False)["params"]

    bp_count = count_params(bp)
    sp_count = count_params(sp)

    if bp_count != BASE_HAND_PARAMS:
        raise RuntimeError(f"Base params mismatch {bp_count:,} != {BASE_HAND_PARAMS:,}")
    if sp_count != EXPECTED_TOTAL_PARAMS:
        raise RuntimeError(
            f"Selective params mismatch {sp_count:,} != {EXPECTED_TOTAL_PARAMS:,}"
        )
    if sp_count - bp_count != GATE_EXTRA_PARAMS:
        raise RuntimeError("Gate parameter delta mismatch")

    base_flops = xla_flops(
        lambda p, x, h: base_model.apply({"params": p}, x, h, training=False)["logits"],
        bp,
        dm,
        dh,
    )
    selective_flops = xla_flops(
        lambda p, x, h: selective_model.apply({"params": p}, x, h, training=False)["logits"],
        sp,
        dm,
        dh,
    )

    print("=" * 120)
    print("NESTSAR HAND-M4/G4 T32 + SELECTIVE RESIDUAL TRUST GATE — GPU PREFLIGHT")
    print("=" * 120)
    print("JAX:", jax.__version__)
    print("BACKEND:", jax.default_backend())
    print("LOCAL DEVICES:", jax.local_device_count())
    print("DEVICES:", jax.local_devices())
    print(f"BASE HAND MODEL PARAMS: {bp_count:,}")
    print(f"SELECTIVE MODEL PARAMS: {sp_count:,}")
    print(f"GATE ADDED PARAMS: {sp_count-bp_count:,}")
    print(f"EXPECTED GATE PARAMS: {GATE_EXTRA_PARAMS:,}")
    print(f"PARAM INCREASE: {100*(sp_count-bp_count)/bp_count:.4f}%")
    print(f"BASE GPU-XLA FLOPS/CLIP: {base_flops:,.0f}")
    print(f"BASE GPU-XLA GFLOPS/CLIP: {base_flops/1e9:.9f}")
    print(f"SELECTIVE GPU-XLA FLOPS/CLIP: {selective_flops:,.0f}")
    print(f"SELECTIVE GPU-XLA GFLOPS/CLIP: {selective_flops/1e9:.9f}")
    print(f"INCREMENTAL GPU-XLA MFLOPS: {(selective_flops-base_flops)/1e6:.6f}")
    print(f"GPU-XLA COMPUTE INCREASE: {100*(selective_flops-base_flops)/base_flops:.4f}%")
    print("BASE ALPHA: 0.20")
    print("DELTA ALPHA: 0.15")
    print("ALPHA RANGE: [0.05, 0.35]")
    print("ATTENTION: NONE")

    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = serialization.msgpack_restore(checkpoint.read_bytes())
        params = payload.get("ema_params", payload.get(b"ema_params"))
        if params is None or count_params(params) != BASE_HAND_PARAMS:
            raise RuntimeError("Checkpoint EMA parameter count mismatch")
        print("FROZEN CHECKPOINT: VERIFIED")

    print("PREFLIGHT=PASS")
    print("=" * 120)


if __name__ == "__main__":
    main()

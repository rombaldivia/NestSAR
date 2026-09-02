#!/usr/bin/env python3
from __future__ import annotations

"""Single-T4 preflight for LocalGlobal V2 + Hand-M4/G4-Lite T32.

Measures baseline and new model with the SAME GPU/XLA audit method so the
incremental cost is directly comparable even if absolute GPU cost_analysis
differs from older TPU audit conventions.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_localglobal_t16 import preprocessing as lg

from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.model import (
    EXPECTED_PARAMS_D32,
    M4LocalGlobalHandM4G4T32,
)
from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.preprocessing import (
    HAND_FEATURES,
    HAND_FRAMES,
    HAND_JOINT_IDS,
    hand_tokens_t32,
)

FRAMES = ju.FRAMES
FEATURES = ju.FEATURES
BASELINE_PARAMS = 1_816_130
EXPECTED_PARAMS = EXPECTED_PARAMS_D32


def xla_flops_baseline(model, params):
    x = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    fn = jax.jit(
        lambda p, xx: model.apply(
            {"params": p},
            xx,
            training=False,
        )["logits"]
    )
    compiled = fn.lower(params, x).compile()
    ca = compiled.cost_analysis()
    if isinstance(ca, list):
        ca = ca[0] if ca else {}
    return float(ca.get("flops", float("nan")))


def xla_flops_new(model, params):
    x = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    h = jnp.zeros((1, HAND_FRAMES, HAND_FEATURES), jnp.float32)
    fn = jax.jit(
        lambda p, xx, hh: model.apply(
            {"params": p},
            xx,
            hh,
            training=False,
        )["logits"]
    )
    compiled = fn.lower(params, x, h).compile()
    ca = compiled.cost_analysis()
    if isinstance(ca, list):
        ca = ca[0] if ca else {}
    return float(ca.get("flops", float("nan")))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m "
            "experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32."
            "preflight_gpu /path/to/ntu120_3danno.pkl"
        )

    dataset = Path(sys.argv[1])
    if not dataset.is_file():
        raise FileNotFoundError(dataset)

    print("=" * 120, flush=True)
    print(
        "NESTSAR LOCALGLOBAL V2 + HAND-M4/G4-LITE T32 — SINGLE T4 PREFLIGHT",
        flush=True,
    )
    print("=" * 120, flush=True)
    print("JAX:", jax.__version__, flush=True)
    print("BACKEND:", jax.default_backend(), flush=True)
    print("LOCAL DEVICES:", jax.local_device_count(), flush=True)
    print("DEVICES:", jax.local_devices(), flush=True)
    print("DATASET:", dataset, flush=True)

    if (
        jax.default_backend() != "gpu"
        or jax.local_device_count() != 1
    ):
        raise RuntimeError(
            f"Expected one visible GPU; "
            f"backend={jax.default_backend()} "
            f"count={jax.local_device_count()}"
        )

    annotations, _ = base.load_ntu(dataset)

    main_shapes = set()
    hand_shapes = set()
    hand_nonzero = 0
    hand_velocity_nonzero = 0
    max_abs = 0.0

    n = min(128, len(annotations))

    for i in range(n):
        a = annotations[i]
        kp = base.annotation_keypoints(a)

        main = lg.segment_phase_tokens_localglobal(kp)
        hand = hand_tokens_t32(kp)

        main_shapes.add(main.shape)
        hand_shapes.add(hand.shape)

        if not np.all(np.isfinite(main)):
            raise RuntimeError(f"Nonfinite main tokens at sample {i}")
        if not np.all(np.isfinite(hand)):
            raise RuntimeError(f"Nonfinite hand tokens at sample {i}")

        hand_nonzero += int(np.any(np.abs(hand) > 1e-7))
        hand_velocity_nonzero += int(
            np.any(np.abs(hand[:, HAND_FEATURES // 2:]) > 1e-7)
        )
        max_abs = max(max_abs, float(np.max(np.abs(hand))))

    if main_shapes != {(FRAMES, FEATURES)}:
        raise RuntimeError(f"Unexpected main shapes: {main_shapes}")
    if hand_shapes != {(HAND_FRAMES, HAND_FEATURES)}:
        raise RuntimeError(f"Unexpected hand shapes: {hand_shapes}")

    print("PREFLIGHT SAMPLES:", n, flush=True)
    print("MAIN TOKEN SHAPE:", (FRAMES, FEATURES), flush=True)
    print("HAND TOKEN SHAPE:", (HAND_FRAMES, HAND_FEATURES), flush=True)
    print("HAND JOINT IDS ZERO-BASED:", HAND_JOINT_IDS.tolist(), flush=True)
    print(
        "HAND NONZERO SAMPLES:",
        f"{100*hand_nonzero/max(n,1):.2f}%",
        flush=True,
    )
    print(
        "HAND VELOCITY NONZERO SAMPLES:",
        f"{100*hand_velocity_nonzero/max(n,1):.2f}%",
        flush=True,
    )
    print("HAND MAX ABS:", max_abs, flush=True)

    key = jax.random.PRNGKey(128)
    key_base, key_new = jax.random.split(key)

    baseline = ju.M4PhaseUniformT16(
        spatial_dim=24,
        model_dim=112,
        dropout=0.10,
    )
    base_main = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    base_params = baseline.init(
        {"params": key_base, "dropout": key_base},
        base_main,
        training=False,
    )["params"]
    base_nparams = ju.count_params(base_params)

    if base_nparams != BASELINE_PARAMS:
        raise RuntimeError(
            f"Baseline params mismatch: {base_nparams:,} != {BASELINE_PARAMS:,}"
        )

    model = M4LocalGlobalHandM4G4T32(
        spatial_dim=24,
        model_dim=112,
        dropout=0.10,
        hand_dim=32,
        hand_residual_scale=0.10,
    )
    new_main = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    new_hand = jnp.zeros((1, HAND_FRAMES, HAND_FEATURES), jnp.float32)
    params = model.init(
        {"params": key_new, "dropout": key_new},
        new_main,
        new_hand,
        training=False,
    )["params"]
    nparams = ju.count_params(params)

    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(
            f"New params mismatch: {nparams:,} != {EXPECTED_PARAMS:,}"
        )

    print("BASELINE PARAMS:", f"{base_nparams:,}", flush=True)
    print("NEW MODEL PARAMS:", f"{nparams:,}", flush=True)
    print("ADDED PARAMS:", f"{nparams-base_nparams:,}", flush=True)
    print(
        "PARAM INCREASE:",
        f"{100*(nparams/base_nparams-1):.3f}%",
        flush=True,
    )

    baseline_flops = xla_flops_baseline(
        baseline,
        base_params,
    )
    new_flops = xla_flops_new(
        model,
        params,
    )

    print(
        "BASELINE GPU-XLA FLOPS/CLIP:",
        f"{baseline_flops:,.0f}",
        flush=True,
    )
    print(
        "BASELINE GPU-XLA GFLOPS/CLIP:",
        f"{baseline_flops/1e9:.9f}",
        flush=True,
    )
    print(
        "NEW GPU-XLA FLOPS/CLIP:",
        f"{new_flops:,.0f}",
        flush=True,
    )
    print(
        "NEW GPU-XLA GFLOPS/CLIP:",
        f"{new_flops/1e9:.9f}",
        flush=True,
    )
    print(
        "INCREMENTAL GPU-XLA MFLOPS:",
        f"{(new_flops-baseline_flops)/1e6:.6f}",
        flush=True,
    )
    print(
        "GPU-XLA COMPUTE INCREASE:",
        f"{100*(new_flops/baseline_flops-1):.3f}%",
        flush=True,
    )

    print(
        "NOTE: compare baseline/new GPU-XLA numbers only to each other; "
        "older TPU 0.020181636-GFLOP reference used a different audit path.",
        flush=True,
    )

    print("ATTENTION: NONE", flush=True)
    print("TRAINING INIT: RANDOM / FROM SCRATCH", flush=True)
    print("GPU_PREFLIGHT=PASS", flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()

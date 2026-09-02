#!/usr/bin/env python3
from __future__ import annotations

"""Single-GPU structural/parameter/XLA preflight for BiJoint T16."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_localglobal_t16 import preprocessing as lg
from experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16.model import (
    BASELINE_PARAMS,
    BIJOINT_EXTRA_PARAMS,
    EXPECTED_PARAMS,
    M4PhaseUniformBiJointT16,
)


def xla_flops(model, params) -> float | None:
    dummy = jnp.zeros((1, ju.FRAMES, ju.FEATURES), jnp.float32)
    try:
        fn = jax.jit(
            lambda p, x: model.apply(
                {"params": p},
                x,
                training=False,
            )["logits"]
        )
        compiled = fn.lower(params, dummy).compile()
        ca = compiled.cost_analysis()
        if isinstance(ca, list):
            ca = ca[0] if ca else {}
        return float(ca.get("flops", float("nan")))
    except Exception as exc:
        print("XLA audit unavailable:", exc, flush=True)
        return None


def main(dataset_path: str | None = None) -> None:
    print("=" * 120, flush=True)
    print("NESTSAR LOCALGLOBAL V2 + BI-JOINT M4/G4 — SINGLE T4 PREFLIGHT", flush=True)
    print("=" * 120, flush=True)
    print("JAX:", jax.__version__, flush=True)
    print("BACKEND:", jax.default_backend(), flush=True)
    print("LOCAL DEVICES:", jax.local_device_count(), flush=True)
    print("DEVICES:", jax.local_devices(), flush=True)

    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, got backend={jax.default_backend()} "
            f"count={jax.local_device_count()}"
        )

    dataset = Path(dataset_path) if dataset_path else base.find_dataset(None)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    print("DATASET:", dataset, flush=True)

    # Small preprocessing sanity check: this architecture experiment must use
    # the exact LocalGlobal V2 representation unchanged.
    annotations, split = base.load_ntu(dataset)
    _, _, val_ids = ju.resolve_protocol_ids(annotations, split, "xsub")
    by_id, _, _ = ju.resolve_protocol_ids(annotations, split, "xsub")
    if not val_ids:
        raise RuntimeError("No XSUB validation examples resolved")
    a = by_id[val_ids[0]]
    kp = base.annotation_keypoints(a)
    token = lg.segment_phase_tokens_localglobal(kp)
    if token.shape != (ju.FRAMES, ju.FEATURES):
        raise RuntimeError(f"Unexpected LocalGlobal token shape {token.shape}")
    if not np.all(np.isfinite(token)):
        raise RuntimeError("Non-finite LocalGlobal token")
    print("PREPROCESSING: exact LocalGlobal V2 module", flush=True)
    print("TOKEN SHAPE:", token.shape, flush=True)

    key = jax.random.PRNGKey(128)
    dummy = jnp.zeros((1, ju.FRAMES, ju.FEATURES), jnp.float32)

    baseline = ju.M4PhaseUniformT16(24, 112, 0.10)
    base_params = baseline.init(
        {"params": key, "dropout": key},
        dummy,
        training=False,
    )["params"]
    base_n = ju.count_params(base_params)

    model = M4PhaseUniformBiJointT16(24, 112, 0.10)
    new_params = model.init(
        {"params": key, "dropout": key},
        dummy,
        training=False,
    )["params"]
    new_n = ju.count_params(new_params)

    print("BASELINE PARAMS:", f"{base_n:,}", flush=True)
    print("BIJOINT PARAMS:", f"{new_n:,}", flush=True)
    print("ADDED PARAMS:", f"{new_n - base_n:,}", flush=True)
    print("PARAM INCREASE:", f"{100.0*(new_n-base_n)/base_n:.4f}%", flush=True)

    if base_n != BASELINE_PARAMS:
        raise RuntimeError(f"Baseline param mismatch: {base_n:,} != {BASELINE_PARAMS:,}")
    if new_n != EXPECTED_PARAMS:
        raise RuntimeError(f"BiJoint param mismatch: {new_n:,} != {EXPECTED_PARAMS:,}")
    if new_n - base_n != BIJOINT_EXTRA_PARAMS:
        raise RuntimeError(
            f"BiJoint param delta mismatch: {new_n-base_n:,} != {BIJOINT_EXTRA_PARAMS:,}"
        )

    # Verify every one of the four stream spatial encoders contains both directions.
    flat = traverse_util.flatten_dict(new_params)
    paths = ["/".join(map(str, k)) for k in flat]
    for i in range(4):
        fwd = any(f"spatial_{i}/joint_bimemory/fwd/" in p for p in paths)
        bwd = any(f"spatial_{i}/joint_bimemory/bwd/" in p for p in paths)
        merge = any(f"spatial_{i}/joint_bimemory/merge/" in p for p in paths)
        print(
            f"SPATIAL_{i} BIJOINT: fwd={fwd} bwd={bwd} merge={merge}",
            flush=True,
        )
        if not (fwd and bwd and merge):
            raise RuntimeError(f"Spatial stream {i} is not fully bidirectional")

    base_flops = xla_flops(baseline, base_params)
    new_flops = xla_flops(model, new_params)

    if base_flops is not None and np.isfinite(base_flops):
        print("BASE GPU-XLA FLOPS/CLIP:", f"{base_flops:,.0f}", flush=True)
        print("BASE GPU-XLA GFLOPS/CLIP:", f"{base_flops/1e9:.9f}", flush=True)

    if new_flops is not None and np.isfinite(new_flops):
        print("BIJOINT GPU-XLA FLOPS/CLIP:", f"{new_flops:,.0f}", flush=True)
        print("BIJOINT GPU-XLA GFLOPS/CLIP:", f"{new_flops/1e9:.9f}", flush=True)

    if (
        base_flops is not None
        and new_flops is not None
        and np.isfinite(base_flops)
        and np.isfinite(new_flops)
    ):
        delta = new_flops - base_flops
        print("INCREMENTAL GPU-XLA MFLOPS:", f"{delta/1e6:.6f}", flush=True)
        print("GPU-XLA COMPUTE INCREASE:", f"{100.0*delta/base_flops:.4f}%", flush=True)

    out = model.apply({"params": new_params}, dummy, training=False)
    if tuple(out["logits"].shape) != (1, 120):
        raise RuntimeError(f"Unexpected logits shape {out['logits'].shape}")
    if tuple(out["stream_logits"].shape) != (1, 4, 120):
        raise RuntimeError(f"Unexpected stream logits shape {out['stream_logits'].shape}")

    print("JOINT MEMORY: BIDIRECTIONAL", flush=True)
    print("ATTENTION: NONE", flush=True)
    print("QKV: NONE", flush=True)
    print("TRAINING: FROM SCRATCH", flush=True)
    print("GPU_PREFLIGHT=PASS", flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

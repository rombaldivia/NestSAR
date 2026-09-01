#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_motionpreserve_phase_t16 import train_phase_t16_tpu as phase
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_localglobal_t16 import preprocessing as lg
from experiments.m4_phase_jitter_consistency_localglobal_t16 import preflight_tpu8 as tpu_pf

EXPECTED_PARAMS = 1_816_130
BASELINE_GFLOPS = 0.020181636


def main(dataset_path: str | None = None) -> None:
    print("=" * 120, flush=True)
    print("NESTSAR T16 LOCAL-POSE + GLOBAL-MOTION V2 — SINGLE T4 PREFLIGHT", flush=True)
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

    annotations, split = base.load_ntu(dataset)
    examples = tpu_pf._resolve_examples(annotations, split, "xsub", 128)
    if not examples:
        raise RuntimeError("Could not resolve XSUB examples for preprocessing audit")

    changed = 0
    local_coord_max = 0.0
    pose_max = 0.0
    motion_mae_sum = 0.0
    phase_identity_max = 0.0
    root_motion_local = []
    root_motion_global = []

    for a in examples:
        kp = base.annotation_keypoints(a)
        baseline_local = base.canonicalize_raw(kp)
        new_local, global_motion = lg.canonicalize_local_and_global(kp)

        local_coord_max = max(
            local_coord_max,
            float(np.max(np.abs(baseline_local - new_local))),
        )

        old = phase.segment_phase_tokens(kp).reshape(
            phase.FRAMES, phase.PERSONS, phase.JOINTS, phase.TOKEN_CHANNELS
        )
        new = lg.segment_phase_tokens_localglobal(kp).reshape(
            phase.FRAMES, phase.PERSONS, phase.JOINTS, phase.TOKEN_CHANNELS
        )

        if not np.all(np.isfinite(new)):
            raise RuntimeError("Non-finite LocalPose+GlobalMotion token")

        pose_max = max(
            pose_max,
            float(np.max(np.abs(old[..., 0:3] - new[..., 0:3]))),
        )
        motion_mae = float(np.mean(np.abs(old[..., 3:15] - new[..., 3:15])))
        motion_mae_sum += motion_mae
        changed += int(motion_mae > 1e-7)

        identity = new[..., 3:6] - (new[..., 6:9] + new[..., 9:12])
        phase_identity_max = max(
            phase_identity_max,
            float(np.max(np.abs(identity))),
        )

        if new_local.shape[0] >= 2:
            root_motion_local.append(
                float(np.mean(np.abs(np.diff(new_local[:, 0, 0, :], axis=0))))
            )
            root_motion_global.append(
                float(np.mean(np.abs(np.diff(global_motion[:, 0, 0, :], axis=0))))
            )

    changed_fraction = changed / len(examples)
    motion_mae = motion_mae_sum / len(examples)

    print("PREPROCESS EXAMPLES:", len(examples), flush=True)
    print("LOCAL COORD MAX DELTA VS base.canonicalize_raw:", local_coord_max, flush=True)
    print("POSE CHANNEL MAX DELTA VS CHAMPION:", pose_max, flush=True)
    print("MOTION CHANNEL MEAN ABS DELTA VS CHAMPION:", motion_mae, flush=True)
    print("SAMPLES WITH CHANGED MOTION CHANNELS:", f"{100*changed_fraction:.2f}%", flush=True)
    print("MAX |full_disp - (phase_a + phase_b)|:", phase_identity_max, flush=True)

    if root_motion_local:
        print(
            "MEAN PERSON0 ROOT MOTION | LOCAL FRAMEWISE CENTERED:",
            float(np.mean(root_motion_local)),
            flush=True,
        )
        print(
            "MEAN PERSON0 ROOT MOTION | GLOBAL FIRST-ROOT FRAME:",
            float(np.mean(root_motion_global)),
            flush=True,
        )

    if local_coord_max > 1e-7:
        raise RuntimeError(f"Local coordinates are not champion-exact: {local_coord_max}")
    if pose_max > 1e-5:
        raise RuntimeError(f"Pose channels changed unexpectedly: {pose_max}")
    if changed_fraction <= 0.0:
        raise RuntimeError("Global-motion preprocessing did not change motion channels")
    if phase_identity_max > 1e-4:
        raise RuntimeError("Phase identity full_disp == phase_a + phase_b failed")

    model = ju.M4PhaseUniformT16(24, 112, 0.10)
    key = jax.random.PRNGKey(128)
    dummy = jnp.zeros((1, ju.FRAMES, ju.FEATURES), jnp.float32)
    params = model.init({"params": key, "dropout": key}, dummy, training=False)["params"]
    nparams = ju.count_params(params)
    print("MODEL PARAMS:", f"{nparams:,}", flush=True)
    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(f"Expected {EXPECTED_PARAMS:,} params, got {nparams:,}")

    flops = ju.audit_flops(model, params)
    if flops is not None and np.isfinite(flops):
        print("XLA FLOPS/CLIP:", int(flops), flush=True)
        print("XLA GFLOPS/CLIP:", float(flops) / 1e9, flush=True)
        print("CHAMPION XLA REFERENCE:", BASELINE_GFLOPS, flush=True)

    out = model.apply({"params": params}, dummy, training=False)
    if tuple(out["logits"].shape) != (1, 120):
        raise RuntimeError(f"Unexpected logits shape: {out['logits'].shape}")

    print("GPU_PREFLIGHT=PASS", flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)

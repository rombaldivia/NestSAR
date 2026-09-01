#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_motionpreserve_phase_t16 import train_phase_t16_tpu as phase
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_localglobal_t16 import preprocessing as lg

EXPECTED_PARAMS = 1_816_130
BASELINE_GFLOPS = 0.020181636


def _resolve_examples(annotations, split, protocol: str = "xsub", n: int = 128):
    tk, _ = base.resolve_split(split, protocol)
    by_id = {
        base.sample_id(a, i): a
        for i, a in enumerate(annotations)
        if isinstance(a, Mapping)
    }
    ids = [str(v) for v in split[tk] if str(v) in by_id]
    return [by_id[sid] for sid in ids[:n]]


def main(dataset_path: str | None = None) -> None:
    print("=" * 120)
    print("NESTSAR T16 LOCAL-POSE + GLOBAL-MOTION — TPU8 PREFLIGHT")
    print("=" * 120)
    print("JAX:", jax.__version__)
    print("BACKEND:", jax.default_backend())
    print("LOCAL DEVICES:", jax.local_device_count())
    print("DEVICES:", jax.local_devices())

    if jax.default_backend() != "tpu" or jax.local_device_count() != 8:
        raise RuntimeError(
            f"Expected TPU8, got backend={jax.default_backend()} count={jax.local_device_count()}"
        )

    dataset = Path(dataset_path) if dataset_path else base.find_dataset(None)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    print("DATASET:", dataset)

    annotations, split = base.load_ntu(dataset)
    examples = _resolve_examples(annotations, split, "xsub", 128)
    if not examples:
        raise RuntimeError("Could not resolve XSUB examples for preprocessing audit")

    changed = 0
    pose_max = 0.0
    motion_mae_sum = 0.0
    phase_identity_max = 0.0
    root_motion_local = []
    root_motion_global = []

    for a in examples:
        kp = base.annotation_keypoints(a)
        old = phase.segment_phase_tokens(kp).reshape(
            phase.FRAMES, phase.PERSONS, phase.JOINTS, phase.TOKEN_CHANNELS
        )
        new = lg.segment_phase_tokens_localglobal(kp).reshape(
            phase.FRAMES, phase.PERSONS, phase.JOINTS, phase.TOKEN_CHANNELS
        )

        if not np.all(np.isfinite(new)):
            raise RuntimeError("Non-finite LocalPose+GlobalMotion token")

        pose_max = max(pose_max, float(np.max(np.abs(old[..., 0:3] - new[..., 0:3]))))
        motion_mae = float(np.mean(np.abs(old[..., 3:15] - new[..., 3:15])))
        motion_mae_sum += motion_mae
        changed += int(motion_mae > 1e-7)

        identity = new[..., 3:6] - (new[..., 6:9] + new[..., 9:12])
        phase_identity_max = max(phase_identity_max, float(np.max(np.abs(identity))))

        local, global_motion = lg.canonicalize_local_and_global(kp)
        if local.shape[0] >= 2:
            root_motion_local.append(float(np.mean(np.abs(np.diff(local[:, 0, 0, :], axis=0)))))
            root_motion_global.append(float(np.mean(np.abs(np.diff(global_motion[:, 0, 0, :], axis=0)))))

    changed_fraction = changed / len(examples)
    motion_mae = motion_mae_sum / len(examples)

    print("PREPROCESS EXAMPLES:", len(examples))
    print("POSE CHANNEL MAX DELTA VS CHAMPION:", pose_max)
    print("MOTION CHANNEL MEAN ABS DELTA VS CHAMPION:", motion_mae)
    print("SAMPLES WITH CHANGED MOTION CHANNELS:", f"{100*changed_fraction:.2f}%")
    print("MAX |full_disp - (phase_a + phase_b)|:", phase_identity_max)
    if root_motion_local:
        print("MEAN PERSON0 ROOT MOTION | LOCAL FRAMEWISE CENTERED:", float(np.mean(root_motion_local)))
        print("MEAN PERSON0 ROOT MOTION | GLOBAL FIRST-ROOT FRAME:", float(np.mean(root_motion_global)))

    if pose_max > 1e-5:
        raise RuntimeError(f"Pose channels changed unexpectedly: max_delta={pose_max}")
    if changed_fraction <= 0.0:
        raise RuntimeError("Global-motion preprocessing did not change any audited motion channels")
    if phase_identity_max > 1e-4:
        raise RuntimeError("Phase identity full_disp == phase_a + phase_b failed")

    model = ju.M4PhaseUniformT16(24, 112, 0.10)
    key = jax.random.PRNGKey(128)
    dummy = jnp.zeros((1, ju.FRAMES, ju.FEATURES), jnp.float32)
    params = model.init({"params": key, "dropout": key}, dummy, training=False)["params"]
    nparams = ju.count_params(params)

    print("MODEL PARAMS:", f"{nparams:,}")
    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(f"Expected {EXPECTED_PARAMS:,} params, got {nparams:,}")

    flops = ju.audit_flops(model, params)
    if flops is not None and np.isfinite(flops):
        gflops = float(flops) / 1e9
        print("XLA FLOPS/CLIP:", int(flops))
        print("XLA GFLOPS/CLIP:", gflops)
        print("CHAMPION GFLOPS REFERENCE:", BASELINE_GFLOPS)

    out = model.apply({"params": params}, dummy, training=False)
    if tuple(out["logits"].shape) != (1, 120):
        raise RuntimeError(f"Unexpected logits shape: {out['logits'].shape}")

    print("PREFLIGHT=PASS")
    print("=" * 120)


if __name__ == "__main__":
    main()

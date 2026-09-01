#!/usr/bin/env python3
from __future__ import annotations

"""Preprocessing-only ablation: local pose + global motion.

Pose channels use the established NestSAR frame-wise person-0 root centering.
Motion channels are computed from coordinates referenced to the FIRST VALID
person-0 root for the whole sequence, so whole-body translation is preserved.

Token layout stays exactly Phase15:
  0:3   local pose xyz
  3:6   global full signed displacement xyz
  6:9   global first-half signed displacement xyz
  9:12  global second-half signed displacement xyz
  12:15 global accumulated absolute path xyz

The geometric normalization remains the baseline local-pose RMS so model input
size, architecture, parameter count, and neural FLOPs are unchanged.
"""

import numpy as np

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_motionpreserve_phase_t16 import train_phase_t16_tpu as phase

FRAMES = phase.FRAMES
PERSONS = phase.PERSONS
JOINTS = phase.JOINTS
XYZ = phase.XYZ
TOKEN_CHANNELS = phase.TOKEN_CHANNELS
FEATURES = phase.FEATURES


def canonicalize_local_and_global(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (local_pose_coords, global_motion_coords) with identical ordering.

    local_pose_coords:
        established NestSAR convention, person-0 joint-0 centered every frame.

    global_motion_coords:
        same persons/joints, but all frames share one constant reference: the
        first valid person-0 joint-0 position. Temporal differences therefore
        retain whole-body translation that frame-wise centering would remove.
    """
    x = base.to_tmvc(keypoints)

    if x.shape[2] < JOINTS:
        pad = np.zeros(
            (x.shape[0], x.shape[1], JOINTS - x.shape[2], XYZ),
            dtype=np.float32,
        )
        x = np.concatenate([x, pad], axis=2)

    x = x[:, :, :JOINTS, :XYZ].astype(np.float32, copy=False)

    # Keep the exact established person ordering: descending sequence energy.
    person_energy = np.sum(np.abs(x), axis=(0, 2, 3))
    x = x[:, np.argsort(-person_energy)]

    if x.shape[1] < PERSONS:
        x = np.concatenate(
            [
                x,
                np.zeros(
                    (x.shape[0], PERSONS - x.shape[1], JOINTS, XYZ),
                    dtype=np.float32,
                ),
            ],
            axis=1,
        )

    x = x[:, :PERSONS]
    valid_joint = np.any(np.abs(x) > 1e-8, axis=-1, keepdims=True)

    # Local pose coordinates: EXACT baseline frame-wise centering semantics.
    center = x[:, 0:1, 0:1, :]
    valid_center = np.any(np.abs(center) > 1e-8, axis=-1, keepdims=True)
    local = np.where(valid_center, x - center, x)
    local = np.where(valid_joint, local, 0.0).astype(np.float32)

    # Global-motion coordinates: one constant reference for the entire sequence.
    # Use the first valid person-0 root; for ordinary NTU clips this is frame 0.
    roots = x[:, 0, 0, :]
    valid_roots = np.any(np.abs(roots) > 1e-8, axis=-1)
    if np.any(valid_roots):
        ref_idx = int(np.flatnonzero(valid_roots)[0])
        root_ref = roots[ref_idx].reshape(1, 1, 1, XYZ)
    else:
        root_ref = np.zeros((1, 1, 1, XYZ), dtype=np.float32)

    global_motion = x - root_ref
    global_motion = np.where(valid_joint, global_motion, 0.0).astype(np.float32)

    return local, global_motion


def phase_tokens_from_bounds_localglobal(keypoints: np.ndarray, bounds) -> np.ndarray:
    """Build Phase15 tokens using local pose and global-reference motion."""
    local, global_motion = canonicalize_local_and_global(keypoints)

    if local.shape[0] <= 0:
        return np.zeros((FRAMES, FEATURES), dtype=np.float32)

    tokens = np.zeros(
        (FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS),
        dtype=np.float32,
    )

    for i, (s, e) in enumerate(bounds):
        local_seg = local[s:e]
        motion_seg = global_motion[s:e]

        pose = local_seg[(len(local_seg) - 1) // 2]

        if len(motion_seg) >= 2:
            d = motion_seg[1:] - motion_seg[:-1]
            full_disp = np.sum(d, axis=0)

            # Preserve the baseline Phase15 split exactly: split transitions.
            cut = max(1, len(d) // 2)
            phase_a = np.sum(d[:cut], axis=0)
            if cut < len(d):
                phase_b = np.sum(d[cut:], axis=0)
            else:
                phase_b = np.zeros_like(full_disp)

            path = np.sum(np.abs(d), axis=0)
        else:
            full_disp = np.zeros_like(pose)
            phase_a = np.zeros_like(pose)
            phase_b = np.zeros_like(pose)
            path = np.zeros_like(pose)

        tokens[i, ..., 0:3] = pose
        tokens[i, ..., 3:6] = full_disp
        tokens[i, ..., 6:9] = phase_a
        tokens[i, ..., 9:12] = phase_b
        tokens[i, ..., 12:15] = path

    # Keep the champion's normalization convention: one RMS from LOCAL centered
    # poses for every channel. This isolates only the coordinate frame used to
    # derive motion; it does not introduce a new scale normalization.
    nz = np.abs(local) > 1e-8
    if np.any(nz):
        rms = float(np.sqrt(np.mean(np.square(local[nz]))) + 1e-6)
        tokens /= rms

    return np.nan_to_num(tokens).reshape(FRAMES, FEATURES).astype(np.float32)


def segment_phase_tokens_localglobal(keypoints: np.ndarray) -> np.ndarray:
    local, _ = canonicalize_local_and_global(keypoints)
    total = int(local.shape[0])
    if total <= 0:
        return np.zeros((FRAMES, FEATURES), dtype=np.float32)
    return phase_tokens_from_bounds_localglobal(
        keypoints,
        base.segment_bounds(total, FRAMES),
    )

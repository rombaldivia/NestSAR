#!/usr/bin/env python3
from __future__ import annotations

"""Preprocessing-only ablation: exact champion local pose + global motion.

Pose channels are produced by calling the champion canonicalizer directly:
    local = base.canonicalize_raw(keypoints)
This guarantees the pose path and its RMS normalization are exactly the same as
M4PhaseJitterConsistencyT16.

Motion channels are computed from a second raw-coordinate view with the SAME
person-energy ordering, but referenced to one constant first-valid person-0 root.
Because that reference is constant over time, temporal differences retain the
whole-body/root translation that frame-wise centering removes.

Token layout remains exactly Phase15:
  0:3   local pose xyz                         (champion exact)
  3:6   global full signed displacement xyz
  6:9   global first-half signed displacement xyz
  9:12  global second-half signed displacement xyz
  12:15 global accumulated absolute path xyz

Input shape, model architecture, parameters and neural FLOPs are unchanged.
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


def _raw_ordered_two_persons(keypoints: np.ndarray) -> np.ndarray:
    """Raw T,M,V,3 coordinates with the champion's person-energy ordering."""
    x = base.to_tmvc(keypoints)

    if x.shape[2] < JOINTS:
        pad = np.zeros(
            (x.shape[0], x.shape[1], JOINTS - x.shape[2], XYZ),
            dtype=np.float32,
        )
        x = np.concatenate([x, pad], axis=2)

    x = x[:, :, :JOINTS, :XYZ].astype(np.float32, copy=False)

    # EXACT ordering rule used by base.canonicalize_raw().
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

    return x[:, :PERSONS].astype(np.float32, copy=False)


def canonicalize_local_and_global(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact champion local pose coordinates and global-motion coordinates."""

    # CRITICAL CONTROL: do not reimplement the champion local preprocessing.
    # Call it directly, so pose values and RMS normalization are guaranteed to
    # follow exactly the same code path as the verified baseline.
    local = base.canonicalize_raw(keypoints).astype(np.float32, copy=False)

    x = _raw_ordered_two_persons(keypoints)

    if x.shape[0] != local.shape[0]:
        raise RuntimeError(
            f"Local/raw temporal mismatch: local={local.shape}, raw={x.shape}"
        )

    valid_joint_raw = np.any(np.abs(x) > 1e-8, axis=-1, keepdims=True)

    # One constant reference for the WHOLE sequence. For normal NTU clips this
    # is person-0 joint-0 at frame 0; first-valid fallback handles leading zeros.
    roots = x[:, 0, 0, :]
    valid_roots = np.any(np.abs(roots) > 1e-8, axis=-1)

    if np.any(valid_roots):
        ref_idx = int(np.flatnonzero(valid_roots)[0])
        root_ref = roots[ref_idx].reshape(1, 1, 1, XYZ)
    else:
        root_ref = np.zeros((1, 1, 1, XYZ), dtype=np.float32)

    global_motion = x - root_ref
    global_motion = np.where(
        valid_joint_raw,
        global_motion,
        0.0,
    ).astype(np.float32)

    return local, global_motion


def phase_tokens_from_bounds_localglobal(keypoints: np.ndarray, bounds) -> np.ndarray:
    """Build Phase15 tokens: champion-local pose + global-reference motion."""
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

            # Split transitions exactly as Phase15 baseline.
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

    # EXACT champion scale source: RMS from base.canonicalize_raw() output.
    nz = np.abs(local) > 1e-8
    if np.any(nz):
        rms = float(np.sqrt(np.mean(np.square(local[nz]))) + 1e-6)
        tokens /= rms

    return np.nan_to_num(tokens).reshape(FRAMES, FEATURES).astype(np.float32)


def segment_phase_tokens_localglobal(keypoints: np.ndarray) -> np.ndarray:
    local = base.canonicalize_raw(keypoints)
    total = int(local.shape[0])
    if total <= 0:
        return np.zeros((FRAMES, FEATURES), dtype=np.float32)

    return phase_tokens_from_bounds_localglobal(
        keypoints,
        base.segment_bounds(total, FRAMES),
    )

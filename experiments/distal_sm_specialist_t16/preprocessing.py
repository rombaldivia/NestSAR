#!/usr/bin/env python3
from __future__ import annotations

"""Distal-motion preprocessing for hands/fingers/feet specialist.

Raw NTU clip length is arbitrary. Every clip is summarized into exactly 16
whole-action temporal segments, so neural processing length is fixed at T16.

Selected NTU25 joints, zero-based:
  left wrist/hand/hand-tip/thumb : 6,7,21,22
  right wrist/hand/hand-tip/thumb: 10,11,23,24
  left ankle/foot                : 14,15
  right ankle/foot               : 18,19

NTU25 does not contain individual finger phalanges; hand-tip/thumb are the
finest finger-related joints available.

Per selected joint and temporal token (15 channels):
  local pose xyz
  global full displacement xyz
  global first-half displacement xyz
  global second-half displacement xyz
  global accumulated absolute path xyz
"""

import numpy as np

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_localglobal_t16 import preprocessing as lg

FRAMES = 16
PERSONS = 2
XYZ = 3
TOKEN_CHANNELS = 15
DISTAL_JOINT_IDS = np.asarray([6,7,21,22,10,11,23,24,14,15,18,19], dtype=np.int32)
DISTAL_JOINTS = int(DISTAL_JOINT_IDS.size)
FEATURES = PERSONS * DISTAL_JOINTS * TOKEN_CHANNELS  # 360


def distal_tokens_from_bounds(keypoints: np.ndarray, bounds) -> np.ndarray:
    local, global_motion = lg.canonicalize_local_and_global(keypoints)
    if local.shape[0] <= 0:
        return np.zeros((FRAMES, FEATURES), np.float32)

    local = local[:, :, DISTAL_JOINT_IDS, :]
    global_motion = global_motion[:, :, DISTAL_JOINT_IDS, :]

    tokens = np.zeros(
        (FRAMES, PERSONS, DISTAL_JOINTS, TOKEN_CHANNELS),
        np.float32,
    )

    for i, (s, e) in enumerate(bounds):
        lseg = local[s:e]
        gseg = global_motion[s:e]
        pose = lseg[(len(lseg) - 1) // 2]

        if len(gseg) >= 2:
            d = gseg[1:] - gseg[:-1]
            full_disp = np.sum(d, axis=0)
            cut = max(1, len(d) // 2)
            phase_a = np.sum(d[:cut], axis=0)
            phase_b = np.sum(d[cut:], axis=0) if cut < len(d) else np.zeros_like(full_disp)
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

    # Keep the same geometric scale source as the LocalGlobal champion.
    full_local = base.canonicalize_raw(keypoints)
    nz = np.abs(full_local) > 1e-8
    if np.any(nz):
        rms = float(np.sqrt(np.mean(np.square(full_local[nz]))) + 1e-6)
        tokens /= rms

    return np.nan_to_num(tokens).reshape(FRAMES, FEATURES).astype(np.float32)


def distal_tokens_t16(keypoints: np.ndarray) -> np.ndarray:
    total = int(base.canonicalize_raw(keypoints).shape[0])
    if total <= 0:
        return np.zeros((FRAMES, FEATURES), np.float32)
    return distal_tokens_from_bounds(
        keypoints,
        base.segment_bounds(total, FRAMES),
    )


def jittered_distal_tokens_t16(
    keypoints: np.ndarray,
    max_shift: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total = int(base.canonicalize_raw(keypoints).shape[0])
    bounds = ju.jittered_segment_bounds(total, FRAMES, max_shift, rng)
    return distal_tokens_from_bounds(keypoints, bounds)

#!/usr/bin/env python3
from __future__ import annotations

"""High-rate hand micro-motion preprocessing for LocalGlobal V2."""

import numpy as np

from experiments.m4_phase_jitter_consistency_localglobal_t16 import preprocessing as lg

PERSONS = lg.PERSONS
XYZ = lg.XYZ

HAND_FRAMES = 32
HAND_JOINT_IDS = np.asarray(
    [6, 7, 21, 22, 10, 11, 23, 24],
    dtype=np.int32,
)
HAND_JOINTS = int(HAND_JOINT_IDS.size)
HAND_FEATURE_GROUPS = 2
HAND_FEATURES = PERSONS * HAND_JOINTS * XYZ * HAND_FEATURE_GROUPS


def uniform_indices(total: int, n: int = HAND_FRAMES) -> np.ndarray:
    if total <= 1:
        return np.zeros((n,), dtype=np.int64)
    return np.linspace(0, total - 1, n, dtype=np.float64).round().astype(np.int64)


def hand_tokens_t32(keypoints: np.ndarray) -> np.ndarray:
    local, global_motion = lg.canonicalize_local_and_global(keypoints)
    total = int(local.shape[0])

    if total <= 0:
        return np.zeros((HAND_FRAMES, HAND_FEATURES), dtype=np.float32)

    idx = uniform_indices(total, HAND_FRAMES)

    local_hand = local[:, :, HAND_JOINT_IDS, :]
    global_hand = global_motion[:, :, HAND_JOINT_IDS, :]

    local_sampled = local_hand[idx]
    global_sampled = global_hand[idx]

    vel = np.zeros_like(global_sampled, dtype=np.float32)
    valid = np.any(np.abs(global_sampled) > 1e-8, axis=-1, keepdims=True)

    for t in range(1, HAND_FRAMES):
        dt = int(idx[t] - idx[t - 1])
        if dt <= 0:
            continue
        pair_valid = valid[t] & valid[t - 1]
        step = (global_sampled[t] - global_sampled[t - 1]) / float(dt)
        vel[t] = np.where(pair_valid, step, 0.0)

    nz = np.abs(local) > 1e-8
    if np.any(nz):
        rms = float(np.sqrt(np.mean(np.square(local[nz]))) + 1e-6)
        local_sampled = local_sampled / rms
        vel = vel / rms

    feat = np.concatenate(
        [
            local_sampled.reshape(HAND_FRAMES, -1),
            vel.reshape(HAND_FRAMES, -1),
        ],
        axis=-1,
    )

    if feat.shape != (HAND_FRAMES, HAND_FEATURES):
        raise RuntimeError(
            f"Unexpected hand feature shape {feat.shape}; expected {(HAND_FRAMES, HAND_FEATURES)}"
        )

    return np.nan_to_num(feat).astype(np.float32)

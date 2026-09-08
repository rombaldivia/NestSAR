"""Fixed-width packing with native-frame, mask-safe relative path summaries.

Channels: pose, full displacement, early displacement, relative path, joint path.
The model reconstructs late = full - early BEFORE its learned modulation.
Paths are per-axis L1 travel, not Euclidean length or a signed trajectory.
"""
from __future__ import annotations

import numpy as np
from .. import preprocessing_corrected as p2

# Keep this NumPy module free of JAX/CUDA imports in cache builders/notebooks.
# A regression test checks equality with the model's NTU parent map.
PARENTS = np.asarray([0, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0,
                      12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11], np.int32)
SHAPE = (p2.FRAMES, p2.PERSONS, p2.JOINTS, 3)


def _features(x, valid=None, rng=None, shift=0, *, path_only=False):
    x = np.asarray(x, np.float32)
    if x.ndim != 4 or x.shape[1:] != (2, 25, 3):
        raise ValueError(f"Expected [T,2,25,3], got {x.shape}")
    local, valid, scale = p2.canonicalize_raw(x, valid)
    if not len(x):
        return np.zeros(SHAPE if path_only else (16, 750), np.float32)
    starts, ends = p2.segment_bounds(len(x), rng=rng, shift=shift)
    owners = p2.transition_owners(len(x), ends)
    lo = np.searchsorted(owners, np.arange(16), side="left")
    hi = np.searchsorted(owners, np.arange(16), side="right")

    # Four valid endpoints per bone; no bridging gaps or padding artifacts.
    joint_ok = valid[1:] & valid[:-1]
    bone_ok = joint_ok & joint_ok[:, :, PARENTS]
    delta = np.diff(x, axis=0)
    relative = np.where(bone_ok[..., None],
                        delta - delta[:, :, PARENTS], 0)

    def prefix(a):
        return np.concatenate([np.zeros_like(x[:1]),
                               np.cumsum(a, axis=0, dtype=np.float64)])

    rp = prefix(np.abs(relative))
    relative_path = (rp[hi] - rp[lo]) / scale
    if path_only:
        return relative_path.astype(np.float32)

    delta = np.where(joint_ok[..., None], delta, 0)
    dp, jp = prefix(delta), prefix(np.abs(delta))
    length = hi - lo
    mid = lo + np.where(length > 0, np.maximum(1, length // 2), 0)
    # Normalize each block identically to P2; no ratios or extra clipping.
    # P2 promotes pose to float64 when concatenating it with prefix sums,
    # before normalization. Match that order for cache/direct-view identity.
    blocks = [p2.representative_pose(local, valid, starts, ends).astype(np.float64) / scale,
              (dp[hi] - dp[lo]) / scale, (dp[mid] - dp[lo]) / scale,
              relative_path, (jp[hi] - jp[lo]) / scale]
    return np.concatenate(blocks, axis=-1).reshape(16, 750).astype(np.float32)


def features(x, valid=None, rng=None, shift=0):
    """Compute all packed features together, sharing masks/deltas/normalization."""
    return _features(x, valid, rng, shift)


def relative_path(x, valid=None):
    """Only the 3 new channels, for an auxiliary cache beside existing P2 tokens."""
    return _features(x, valid, path_only=True)


def pack(p2_tokens, paths):
    """Copy just the requested batch; never mutate the original P2 cache."""
    x = np.array(p2_tokens, dtype=np.float32, copy=True)
    if x.shape[-2:] != (16, 750):
        raise ValueError("Expected [...,16,750] P2 tokens")
    expected = x.shape[:-2] + SHAPE
    if np.shape(paths) != expected:
        raise ValueError(f"Expected relative paths {expected}, got {np.shape(paths)}")
    x.reshape(*x.shape[:-2], 16, 2, 25, 15)[..., 9:12] = paths
    return x


def augmented_features(x, seed, epoch, sample_index, rotation_degrees=8.0, shift=1):
    """Rebuild signed motion AND paths from the same freshly rotated raw clip."""
    x = np.asarray(x, np.float32)
    valid = p2.raw_valid(x)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, sample_index]))
    theta = np.deg2rad(rng.uniform(-rotation_degrees, rotation_degrees)) if rotation_degrees else 0.0
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    return features(np.where(valid[..., None], x @ rotation.T, 0), valid, rng, shift)

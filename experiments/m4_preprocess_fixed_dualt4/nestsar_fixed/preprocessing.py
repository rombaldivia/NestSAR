"""Mask-preserving raw-skeleton preprocessing. NumPy only; no GPU imports."""
from __future__ import annotations

import numpy as np

FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS = 16, 2, 25, 15
FEATURES = PERSONS * JOINTS * TOKEN_CHANNELS
HAND_FRAMES, HAND_FEATURES = 32, 96
HAND_IDS = np.asarray([6, 7, 21, 22, 10, 11, 23, 24])
VERSION = "masked-complete-transitions-v1"


def ordered_raw(keypoints, layout="MTVC"):
    """NTU pickle layout is M,T,V,C; TMVC must be selected explicitly.

    Preserve the baseline's clip-level person-energy ordering. The cache stores
    this order before augmentation, so augmentation cannot swap person slots.
    """
    x = np.asarray(keypoints, np.float32)
    if x.ndim == 3:
        x = x[:, None]
    elif x.ndim == 4 and layout == "MTVC":
        x = x.transpose(1, 0, 2, 3)
    elif x.ndim != 4 or layout != "TMVC":
        raise ValueError(f"Expected MTVC/TMVC or TVC skeleton, got {x.shape}, {layout}")
    if x.shape[-1] != 3 or x.shape[2] != JOINTS or not 1 <= x.shape[1] <= 4:
        raise ValueError(f"Expected NTU 25-joint xyz skeleton, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite raw coordinates: repair the source instead of hiding NaNs")
    order = np.argsort(-np.abs(x).sum(axis=(0, 2, 3)), kind="stable")[:PERSONS]
    out = np.zeros((len(x), PERSONS, JOINTS, 3), np.float32)
    out[:, :len(order)] = x[:, order]
    return out


def raw_valid(x):
    return np.any(np.abs(x) > 1e-8, axis=-1)


def canonicalize_raw(x, valid=None):
    """Return local coordinates, original joint mask and one valid-joint RMS.

    Missing primary roots use interpolation of valid primary roots; if none
    exist, use the primary person's valid-joint centroid. Normal NTU clips use
    the same person-0/joint-0 frame reference as the baseline.
    """
    x = np.asarray(x, np.float32)
    valid = raw_valid(x) if valid is None else np.asarray(valid, bool)
    if valid.shape != x.shape[:-1] or not np.isfinite(x).all():
        raise ValueError("Invalid mask or nonfinite coordinates")
    if len(x) == 0:
        return x.copy(), valid, 1.0
    roots = x[:, 0, 0].copy()
    good = valid[:, 0, 0].copy()
    if not good.any():
        counts = valid[:, 0].sum(axis=-1)
        roots = (x[:, 0] * valid[:, 0, :, None]).sum(axis=1) / np.maximum(counts[:, None], 1)
        good = counts > 0
    if good.any():
        positions = np.arange(len(x))
        for c in range(3):
            roots[:, c] = np.interp(positions, positions[good], roots[good, c])
    else:
        roots.fill(0)
    local = np.where(valid[..., None], x - roots[:, None, None, :], 0).astype(np.float32)
    # Include all xyz coordinates of valid joints, including legitimate zeros.
    denom = int(valid.sum()) * 3
    rms = float(np.sqrt(np.square(local, dtype=np.float64).sum() / max(denom, 1)))
    scale = rms + 1e-6 if rms > 1e-6 else 1.0
    return local, valid, scale


def segment_bounds(total, n=FRAMES, rng=None, shift=0):
    if total < 1:
        return np.zeros(n, np.int64), np.zeros(n, np.int64)
    if total < n:
        s = np.linspace(0, total - 1, n).round().astype(np.int64)
        return s, s + 1
    edges = np.linspace(0, total, n + 1).astype(np.int64)
    if rng is not None and shift:
        reference = edges.copy()
        for i in range(1, n):
            edges[i] = np.clip(reference[i] + rng.integers(-shift, shift + 1),
                               edges[i - 1] + 1, total - (n - i))
    return edges[:-1], edges[1:]


def transition_owners(total, ends):
    """Destination-frame ownership. Repeated poses never duplicate transitions."""
    owners = np.searchsorted(ends, np.arange(1, total), side="right")
    if len(owners) and (owners.min() < 0 or owners.max() >= FRAMES):
        raise ValueError("Segmentation does not cover the full clip")
    return owners


def features(x, valid=None, rng=None, shift=0):
    """Return main [16,750] and hand [32,96] from ONE raw coordinate view."""
    local, valid, scale = canonicalize_raw(x, valid)
    total = len(x)
    if total == 0:
        return np.zeros((FRAMES, FEATURES), np.float32), np.zeros((HAND_FRAMES, HAND_FEATURES), np.float32)
    starts, ends = segment_bounds(total, rng=rng, shift=shift)
    pose_indices = (starts + ends - 1) // 2
    # Constant-reference global differences equal raw-coordinate differences.
    # Mask both endpoints BEFORE summing; never bridge a missing-data gap.
    d = np.where((valid[1:] & valid[:-1])[..., None], x[1:] - x[:-1], 0)
    prefix = np.concatenate([np.zeros_like(x[:1]), np.cumsum(d, axis=0, dtype=np.float64)])
    path_prefix = np.concatenate([np.zeros_like(x[:1]), np.cumsum(np.abs(d), axis=0, dtype=np.float64)])
    owners = transition_owners(total, ends)
    lo = np.searchsorted(owners, np.arange(FRAMES), side="left")
    hi = np.searchsorted(owners, np.arange(FRAMES), side="right")
    length = hi - lo
    mid = lo + np.where(length > 0, np.maximum(1, length // 2), 0)
    tokens = np.concatenate([
        local[pose_indices], prefix[hi] - prefix[lo],
        prefix[mid] - prefix[lo], prefix[hi] - prefix[mid],
        path_prefix[hi] - path_prefix[lo],
    ], axis=-1) / scale
    idx = np.linspace(0, total - 1, HAND_FRAMES).round().astype(np.int64)
    hand_pose = local[idx][:, :, HAND_IDS]
    # Integrate valid consecutive transitions between sampled hand frames;
    # divide by elapsed raw frames. A gap cannot become an artificial jump.
    vel = np.zeros_like(hand_pose)
    dt = np.diff(idx)
    sums = prefix[idx[1:]] - prefix[idx[:-1]]
    vel[1:] = (sums / np.maximum(dt[:, None, None, None], 1))[:, :, HAND_IDS]
    hand = np.concatenate([hand_pose.reshape(HAND_FRAMES, -1), vel.reshape(HAND_FRAMES, -1)], axis=-1) / scale
    return tokens.reshape(FRAMES, FEATURES).astype(np.float32), hand.astype(np.float32)


def training_views(x, seed, epoch, sample_index, fresh=True, rotation_degrees=8.0, shift=1):
    """Mild whole-clip yaw plus fresh segmentation; no label-dependent transforms.

    Replay is independent of batch order, worker scheduling and resume points.
    Both persons receive the same yaw; main and hand features are rebuilt from
    the transformed skeleton using the original validity mask.
    """
    valid = raw_valid(x)
    main, hand = features(x, valid)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch if fresh else 0, sample_index]))
    theta = np.deg2rad(rng.uniform(-rotation_degrees, rotation_degrees)) if rotation_degrees else 0.0
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    augmented = np.where(valid[..., None], x @ rotation.T, 0)
    aug_main, aug_hand = features(augmented, valid, rng, shift)
    return main, aug_main, hand, aug_hand

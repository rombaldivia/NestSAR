"""Corrected T16 preprocessing for NestSAR-SM-ALL.

Fixes three data-path defects without changing model input dimensions:
1) preserve missing-joint/person masks through centering and normalization,
2) account for every valid consecutive-frame transition exactly once,
3) support deterministic fresh per-epoch label-preserving augmentation.
"""
from __future__ import annotations

import numpy as np

FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS = 16, 2, 25, 15
FEATURES = PERSONS * JOINTS * TOKEN_CHANNELS
VERSION = "sm-all-masked-complete-transitions-freshaug-v1"


def ordered_raw(keypoints, layout="MTVC"):
    """Convert NTU skeletons to [T,M,V,C] while preserving person-energy order."""
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
        raise ValueError("Nonfinite raw coordinates")
    order = np.argsort(-np.abs(x).sum(axis=(0, 2, 3)), kind="stable")[:PERSONS]
    out = np.zeros((len(x), PERSONS, JOINTS, 3), np.float32)
    out[:, :len(order)] = x[:, order]
    return out


def raw_valid(x):
    return np.any(np.abs(x) > 1e-8, axis=-1)


def canonicalize_raw(x, valid=None):
    """Center only valid joints and compute scale only from valid coordinates."""
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
            edges[i] = np.clip(
                reference[i] + rng.integers(-shift, shift + 1),
                edges[i - 1] + 1,
                total - (n - i),
            )
    return edges[:-1], edges[1:]


def transition_owners(total, ends):
    """Assign transition t-1->t to the segment owning destination frame t."""
    owners = np.searchsorted(ends, np.arange(1, total), side="right")
    if len(owners) and (owners.min() < 0 or owners.max() >= FRAMES):
        raise ValueError("Segmentation does not cover the full clip")
    return owners


def features(x, valid=None, rng=None, shift=0):
    """Build corrected fixed [16,750] LocalGlobal pose/motion tokens."""
    local, valid, scale = canonicalize_raw(x, valid)
    total = len(x)
    if total == 0:
        return np.zeros((FRAMES, FEATURES), np.float32)

    starts, ends = segment_bounds(total, rng=rng, shift=shift)
    pose_indices = (starts + ends - 1) // 2

    # Compute ALL adjacent transitions first. Both endpoints must be valid.
    d = np.where((valid[1:] & valid[:-1])[..., None], x[1:] - x[:-1], 0)
    prefix = np.concatenate([np.zeros_like(x[:1]), np.cumsum(d, axis=0, dtype=np.float64)])
    path_prefix = np.concatenate([np.zeros_like(x[:1]), np.cumsum(np.abs(d), axis=0, dtype=np.float64)])

    owners = transition_owners(total, ends)
    lo = np.searchsorted(owners, np.arange(FRAMES), side="left")
    hi = np.searchsorted(owners, np.arange(FRAMES), side="right")
    length = hi - lo
    mid = lo + np.where(length > 0, np.maximum(1, length // 2), 0)

    tokens = np.concatenate(
        [
            local[pose_indices],
            prefix[hi] - prefix[lo],
            prefix[mid] - prefix[lo],
            prefix[hi] - prefix[mid],
            path_prefix[hi] - path_prefix[lo],
        ],
        axis=-1,
    ) / scale
    return tokens.reshape(FRAMES, FEATURES).astype(np.float32)


def training_views(x, seed, epoch, sample_index, rotation_degrees=8.0, shift=1):
    """Return canonical and fresh augmented views, rebuilt from raw skeletons."""
    valid = raw_valid(x)
    canonical = features(x, valid)

    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, sample_index]))
    theta = np.deg2rad(rng.uniform(-rotation_degrees, rotation_degrees)) if rotation_degrees else 0.0
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    augmented = np.where(valid[..., None], x @ rotation.T, 0)
    aug = features(augmented, valid, rng=rng, shift=shift)
    return canonical, aug

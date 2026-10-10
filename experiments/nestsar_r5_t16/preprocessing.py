"""R5 input = the unchanged R4 T16 tokens + a 4x-rate hand block.

Per clip the model receives [16, 942]:
  * [:, :750]  the exact R4 person-aware tokens (2 persons x 25 joints x 15
    channels: pose, net displacement, two phase displacements, path), produced
    by the same code path as ``preprocessing_corrected.features``;
  * [:, 750:]  4 sub-segments per segment x 2 persons x 24 hand channels.

Hand channels per person and sub-segment (both hands, L then R):
    tip - hand, thumb - hand, hand - wrist   (mean over the sub-segment frames
                                              where both joints are valid)
    wrist displacement                        (sum of valid frame-to-frame steps
                                              whose destination frame lies in the
                                              sub-segment)
All values are divided by the same clip scale as the R4 tokens. Sub-segments
split every R4 segment in four, so the +-1 frame boundary jitter of the
augmented view moves both representations consistently, and the same yaw
rotation is applied to both. A sub-segment shorter than one frame reuses the
nearest frame of its segment for the relative vectors and has zero
displacement. Displacements follow the R4 ownership rule (a transition t-1 -> t
belongs to the first segment whose end is after t), so every valid transition
is counted exactly once, also for clips shorter than 16 frames where segments
repeat frames.

Why: the R4 part pooling averaged thumb and hand tip with wrist and hand, and a
16-token clip summary keeps only net displacement and path length, which erases
the rhythm of small repeated hand motions (counting money, writing, cutting
nails, OK vs victory sign).
"""
from __future__ import annotations

import numpy as np

from experiments.nestsar_sm_all_t16 import preprocessing_corrected as pp

FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS = pp.FRAMES, pp.PERSONS, pp.JOINTS, pp.TOKEN_CHANNELS
R4_FEATURES = pp.FEATURES                     # 750
SUB = 4                                       # hand sub-segments per segment
HAND_FRAMES = FRAMES * SUB                    # 64
HAND_SIDES = ((6, 7, 21, 22), (10, 11, 23, 24))   # (wrist, hand, tip, thumb), left then right
HAND_CHANNELS = 24                            # per person per sub-segment
HAND_FEATURES = SUB * PERSONS * HAND_CHANNELS  # 192 per segment
FEATURES = R4_FEATURES + HAND_FEATURES        # 942
VERSION = f"r5-hand4x-v2+{pp.VERSION}"

if (FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS) != (16, 2, 25, 15):
    raise RuntimeError("R5 expects the R4 T16/M2/V25/C15 token contract")


def representative_pose(local, valid, starts, ends):
    """Vectorized ``pp.representative_pose`` (identical choice, tested bit-exact).

    Per segment and person: the frame with most valid joints, ties broken by
    distance to the segment centre, then by the earlier frame.
    """
    starts = np.asarray(starts, np.int64)
    ends = np.asarray(ends, np.int64)
    pose = np.zeros((FRAMES, PERSONS, JOINTS, 3), np.float32)
    total = len(local)
    if total == 0:
        return pose
    width = int(max(np.max(ends - starts), 1))
    frames = starts[:, None] + np.arange(width, dtype=np.int64)[None, :]        # [F, W]
    inside = frames < ends[:, None]
    safe = np.clip(frames, 0, total - 1)
    counts = valid.sum(axis=-1)                                                 # [T, P]
    seg_counts = np.where(inside[..., None], counts[safe], -1)                  # [F, W, P]
    best = seg_counts.max(axis=1)                                               # [F, P]
    center = 0.5 * (starts.astype(np.float64) + (ends - 1).astype(np.float64))
    distance = np.abs(frames.astype(np.float64) - center[:, None])              # [F, W]
    candidate = inside[..., None] & (seg_counts == best[:, None, :])
    score = np.where(candidate, distance[..., None], np.inf)
    chosen = np.take_along_axis(safe, np.argmin(score, axis=1), axis=1)         # [F, P]
    use = (best > 0) & (ends > starts)[:, None]
    picked = local[chosen, np.arange(PERSONS)[None, :]]                         # [F, P, V, 3]
    pose[use] = picked[use]
    return pose


def r4_tokens(x, local, valid, scale, starts, ends):
    """Body of ``preprocessing_corrected.features`` for given segment bounds."""
    total = len(x)
    pose = representative_pose(local, valid, starts, ends)
    d = np.where((valid[1:] & valid[:-1])[..., None], x[1:] - x[:-1], 0)
    prefix = np.concatenate([np.zeros_like(x[:1]), np.cumsum(d, axis=0, dtype=np.float64)])
    path_prefix = np.concatenate([np.zeros_like(x[:1]), np.cumsum(np.abs(d), axis=0, dtype=np.float64)])
    owners = pp.transition_owners(total, ends)
    lo = np.searchsorted(owners, np.arange(FRAMES), side="left")
    hi = np.searchsorted(owners, np.arange(FRAMES), side="right")
    length = hi - lo
    mid = lo + np.where(length > 0, np.maximum(1, length // 2), 0)
    tokens = np.concatenate(
        [
            pose,
            prefix[hi] - prefix[lo],
            prefix[mid] - prefix[lo],
            prefix[hi] - prefix[mid],
            path_prefix[hi] - path_prefix[lo],
        ],
        axis=-1,
    ) / scale
    return tokens.reshape(FRAMES, R4_FEATURES).astype(np.float32)


def subsegment_windows(total, starts, ends):
    """Sub-segment frame windows.

    Returns (pos_lo, pos_hi, disp_lo, disp_hi), each [FRAMES, SUB] int64.
    ``pos`` windows are never empty and stay inside their segment (inside the
    clip for degenerate segments); ``disp`` windows partition the frames of each
    segment and may be empty.
    """
    starts = np.asarray(starts, np.int64)
    ends = np.asarray(ends, np.int64)
    length = np.maximum(ends - starts, 0)
    k = np.arange(SUB + 1, dtype=np.int64)
    edges = starts[:, None] + (k[None, :] * length[:, None]) // SUB      # [FRAMES, SUB+1]
    disp_lo, disp_hi = edges[:, :-1], edges[:, 1:]
    # R4 ownership: destination frame t belongs to segment searchsorted(ends, t, "right").
    # Contiguous segments own all their frames; a repeated single-frame segment
    # (clips < 16 frames) owns nothing, so its transition is not counted twice.
    owns = np.searchsorted(ends, starts, side="right") == np.arange(len(starts))
    disp_hi = np.where(owns[:, None], disp_hi, disp_lo)
    last = np.clip(np.maximum(ends - 1, starts), 0, max(total - 1, 0))[:, None]
    pos_lo = np.minimum(disp_lo, last)
    pos_lo = np.clip(pos_lo, 0, max(total - 1, 0))
    pos_hi = np.maximum(disp_hi, pos_lo + 1)
    pos_hi = np.minimum(pos_hi, total)
    pos_hi = np.maximum(pos_hi, pos_lo + 1)
    return pos_lo, pos_hi, disp_lo, disp_hi


def hand_tokens(x, valid, starts, ends, scale):
    """4x-rate hand block [FRAMES, HAND_FEATURES] (see module docstring)."""
    x = np.asarray(x, np.float32)
    total = len(x)
    out = np.zeros((FRAMES, SUB, PERSONS, HAND_CHANNELS), np.float32)
    if total == 0:
        return out.reshape(FRAMES, HAND_FEATURES)

    vec, cnt, disp = [], [], []
    for wrist, hand, tip, thumb in HAND_SIDES:
        for a, b in ((tip, hand), (thumb, hand), (hand, wrist)):
            ok = valid[:, :, a] & valid[:, :, b]                       # [T, P]
            vec.append(np.where(ok[..., None], x[:, :, a] - x[:, :, b], 0))
            cnt.append(ok.astype(np.float64))
        step_ok = valid[1:, :, wrist] & valid[:-1, :, wrist]
        step = np.where(step_ok[..., None], x[1:, :, wrist] - x[:-1, :, wrist], 0)
        first = np.zeros((1,) + step.shape[1:], step.dtype)
        disp.append(np.concatenate([first, step], axis=0))   # destination-indexed, [T, P, 3]
    vec = np.stack(vec, axis=2)        # [T, P, 6, 3]
    cnt = np.stack(cnt, axis=2)        # [T, P, 6]
    disp = np.stack(disp, axis=2)      # [T, P, 2, 3]

    zero = lambda a: np.zeros_like(a[:1], dtype=np.float64)
    pv = np.concatenate([zero(vec), np.cumsum(vec, axis=0, dtype=np.float64)])
    pc = np.concatenate([zero(cnt), np.cumsum(cnt, axis=0, dtype=np.float64)])
    pd = np.concatenate([zero(disp), np.cumsum(disp, axis=0, dtype=np.float64)])

    pos_lo, pos_hi, disp_lo, disp_hi = subsegment_windows(total, starts, ends)
    sums = pv[pos_hi] - pv[pos_lo]                                   # [F, S, P, 6, 3]
    counts = pc[pos_hi] - pc[pos_lo]                                 # [F, S, P, 6]
    means = sums / np.maximum(counts, 1.0)[..., None]
    steps = pd[disp_hi] - pd[disp_lo]                                # [F, S, P, 2, 3]

    left = [means[..., 0, :], means[..., 1, :], means[..., 2, :], steps[..., 0, :]]
    right = [means[..., 3, :], means[..., 4, :], means[..., 5, :], steps[..., 1, :]]
    out = np.concatenate(left + right, axis=-1) / scale              # [F, S, P, 24]
    return out.astype(np.float32).reshape(FRAMES, HAND_FEATURES)


HAND_FILTERS = ("none", "hampel", "smooth", "sun", "sun_smooth")
HAMPEL_FLOOR = 0.002          # metres; keeps static joints (MAD ~ 0) from being flagged by sensor jitter
_HAND_JOINTS = tuple(sorted({j for side in HAND_SIDES for j in side}))


def _hampel(run, window, k):
    """Median/MAD outlier replacement of a [n, 3] run (edge-padded window, per coordinate)."""
    half = window // 2
    padded = np.pad(run, ((half, half), (0, 0)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)     # [n, 3, window]
    med = np.median(win, axis=-1)
    mad = 1.4826 * np.median(np.abs(win - med[..., None]), axis=-1)
    bad = np.abs(run - med) > k * np.maximum(mad, HAMPEL_FLOOR)
    return np.where(bad, med, run)


def _hold_noisy_frames(x, valid, k=4.0, rel=0.25, jump=6.0):
    """Frame-level denoising in the spirit of Sun et al. (replace a noisy frame by the previous one).

    Per person and hand side, a frame is noisy when a hand bone (wrist-hand, hand-tip, hand-thumb) is
    more than max(k MAD, rel * median) away from its clip median length, or the wrist jumps more than
    ``jump`` robust standard deviations in one step. All four joints of that side are replaced by the last
    clean frame (the next clean one at the start). Invalid (missing) frames are never filled.
    """
    out = np.array(x, np.float32, copy=True)
    for p in range(x.shape[1]):
        for wrist, hand, tip, thumb in HAND_SIDES:
            ok = valid[:, p, wrist] & valid[:, p, hand] & valid[:, p, tip] & valid[:, p, thumb]
            if ok.sum() < 8:
                continue
            idx = np.flatnonzero(ok)
            bad = np.zeros(len(idx), bool)
            for a, b in ((wrist, hand), (hand, tip), (hand, thumb)):
                length = np.linalg.norm(x[idx, p, a] - x[idx, p, b], axis=-1)
                med = np.median(length)
                mad = 1.4826 * np.median(np.abs(length - med))
                bad |= np.abs(length - med) > np.maximum(k * mad, rel * med)
            step = np.linalg.norm(np.diff(x[idx, p, wrist], axis=0), axis=-1)
            smed = np.median(step)
            sstd = 1.4826 * np.median(np.abs(step - smed))
            bad[1:] |= step > smed + jump * np.maximum(sstd, HAMPEL_FLOOR)
            if not bad.any() or bad.all():
                continue
            good = np.flatnonzero(~bad)
            src = good[np.maximum(np.searchsorted(good, np.arange(len(idx)), side="right") - 1, 0)]
            for j in (wrist, hand, tip, thumb):
                out[idx[bad], p, j] = x[idx[src[bad]], p, j]
    return out


def denoise_hand_joints(x, valid, mode):
    """Zero-phase denoising of the wrist/hand/tip/thumb joints only (body joints are returned untouched).

    ``hampel``: Hampel identifier (window 5, 3 MAD) replaces isolated spikes by the local median.
    ``smooth``: Hampel followed by a Savitzky-Golay filter (window 7, order 2, symmetric so no lag).
    ``sun``: frame-level (see ``_hold_noisy_frames``); ``sun_smooth``: that, then ``smooth``.
    It runs per person and per joint on each run of consecutive valid frames and never fills gaps.
    """
    if mode == "none" or len(x) == 0:
        return x
    if mode not in HAND_FILTERS:
        raise ValueError(f"hand filter must be one of {HAND_FILTERS}, got {mode!r}")
    from scipy.signal import savgol_filter
    if mode.startswith("sun"):
        x = _hold_noisy_frames(x, valid)
        if mode == "sun":
            return x
    out = np.array(x, np.float32, copy=True)
    for p in range(x.shape[1]):
        for j in _HAND_JOINTS:
            ok = valid[:, p, j]
            if not ok.any():
                continue
            edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(np.int8), [0]])))
            for a, b in zip(edges[0::2], edges[1::2]):
                run = x[a:b, p, j].astype(np.float64)
                if len(run) >= 5:
                    run = _hampel(run, 5, 3.0)
                if mode in ("smooth", "sun_smooth") and len(run) >= 7:
                    run = savgol_filter(run, 7, 2, axis=0, mode="interp")
                out[a:b, p, j] = run
    return out


def features(x, valid=None, rng=None, shift=0, hand_filter="none"):
    """R5 tokens [FRAMES, FEATURES]; the first 750 equal ``pp.features`` exactly."""
    x = np.asarray(x, np.float32)
    local, valid, scale = pp.canonicalize_raw(x, valid)
    total = len(x)
    if total == 0:
        return np.zeros((FRAMES, FEATURES), np.float32)
    starts, ends = pp.segment_bounds(total, rng=rng, shift=shift)
    body = r4_tokens(x, local, valid, scale, starts, ends)
    hands = hand_tokens(denoise_hand_joints(x, valid, hand_filter), valid, starts, ends, scale)
    return np.concatenate([body, hands], axis=-1)


def augmented_features(x, seed, epoch, sample_index, rotation_degrees=8.0, shift=1, hand_filter="none"):
    """Same RNG stream, yaw rotation and boundary jitter as ``pp.augmented_features``."""
    valid = pp.raw_valid(x)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, sample_index]))
    theta = np.deg2rad(rng.uniform(-rotation_degrees, rotation_degrees)) if rotation_degrees else 0.0
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    augmented = np.where(valid[..., None], x @ rotation.T, 0)
    return features(augmented, valid, rng=rng, shift=shift, hand_filter=hand_filter)


# --------------------------------------------------------------------------- strong augmentation
# Training-only, label-preserving. The R5 audit showed 99.7-99.9 % clean-train accuracy against
# 76-77 % validation (a 22-23 pp gap) with the mild yaw +-8 deg / +-1 frame augmentation, and XSub
# splits by subject, so the transformations below mostly vary what differs between subjects
# (body proportions) and recording conditions (viewpoint, speed, noise, missing joints).
# ``strength`` scales every range; 0 would be the identity.
NTU_PARENTS = np.asarray([0, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0,
                          12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11], np.int32)


def _tree_order(parents):
    order, seen = [0], {0}
    while len(order) < len(parents):
        for j in range(len(parents)):
            if j not in seen and parents[j] in seen:
                order.append(j)
                seen.add(j)
    return tuple(order)


TREE_ORDER = _tree_order(NTU_PARENTS)           # root first, parents before children


def _random_affine(rng, s):
    """Rotation (yaw +-15, pitch/roll +-5 deg) x shear x per-axis scale, all scaled by ``s``."""
    yaw, pitch, roll = np.deg2rad(rng.uniform(-1, 1, 3) * np.array([15.0, 5.0, 5.0]) * s)
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.asarray([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.asarray([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    shear = np.eye(3) + (rng.uniform(-1, 1, (3, 3)) * 0.05 * s) * (1 - np.eye(3))
    scale = np.diag(1.0 + rng.uniform(-1, 1, 3) * 0.06 * s)
    return (ry @ rx @ rz @ shear @ scale).astype(np.float32)


def _perturb_bones(x, valid, rng, s):
    """Rescale every bone of each actor by a clip-constant factor, keeping the kinematic tree
    connected (children follow their parent). Imitates a different body size or limb ratio."""
    out = x.copy()
    amount = 0.12 * s
    for person in range(x.shape[1]):
        if not valid[:, person].any():
            continue
        factor = 1.0 + rng.uniform(-amount, amount, JOINTS)
        for j in TREE_ORDER[1:]:
            p = NTU_PARENTS[j]
            ok = valid[:, person, j] & valid[:, person, p]
            moved = out[:, person, p] + factor[j] * (x[:, person, j] - x[:, person, p])
            out[:, person, j] = np.where(ok[:, None], moved, out[:, person, j])
    return out


def _resample_time(x, valid, factor):
    """Speed change by linear interpolation; a joint is valid only if it was in both frames."""
    total = len(x)
    length = max(2, int(round(total / factor)))
    position = np.linspace(0, total - 1, length)
    i0 = np.floor(position).astype(np.int64)
    i1 = np.minimum(i0 + 1, total - 1)
    w = (position - i0).astype(np.float32)[:, None, None, None]
    # Validity follows the frames that actually carry weight, so integer positions copy a frame as it is.
    ok = np.where(w[..., 0] < 1e-6, valid[i0], np.where(w[..., 0] > 1 - 1e-6, valid[i1], valid[i0] & valid[i1]))
    out = (1 - w) * x[i0] + w * x[i1]
    return np.where(ok[..., None], out, 0).astype(np.float32), ok


def strong_augmented_features(x, seed, epoch, sample_index, strength=1.0, stream=0, shift=1, hand_filter="none"):
    """R5 tokens of a strongly augmented view of the raw clip ``x`` [T, 2, 25, 3].

    Deterministic in (seed, epoch, sample_index, stream), so a resumed run repeats the same views.
    Order: affine -> bone lengths -> joint noise -> speed -> temporal crop -> joint cut-out -> tokens.
    """
    x = np.asarray(x, np.float32)
    valid = pp.raw_valid(x)
    if len(x) == 0 or not valid.any():
        return features(x, valid, hand_filter=hand_filter)
    s = float(strength)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, sample_index, 7919, stream]))

    x = np.where(valid[..., None], x @ _random_affine(rng, s).T, 0).astype(np.float32)
    if rng.random() < 0.8:
        x = _perturb_bones(x, valid, rng, s)
    if rng.random() < 0.8:
        x = x + np.where(valid[..., None], rng.normal(0.0, 0.004 * s, x.shape), 0).astype(np.float32)
    if len(x) >= 8 and rng.random() < 0.7:
        x, valid = _resample_time(x, valid, 1.0 + rng.uniform(-0.25, 0.25) * s)
    if len(x) >= 16 and rng.random() < 0.5:
        keep = max(8, int(round(len(x) * (1.0 - rng.uniform(0.0, 0.25) * s))))
        start = int(rng.integers(0, len(x) - keep + 1))
        x, valid = x[start:start + keep], valid[start:start + keep]
    if len(x) >= 8 and rng.random() < 0.35 * min(s, 1.0):
        span = max(1, int(round(len(x) * rng.uniform(0.15, 0.40))))
        start = int(rng.integers(0, len(x) - span + 1))
        cut = rng.choice(JOINTS, int(rng.integers(1, 4)), replace=False)
        person = int(rng.integers(0, PERSONS))
        x = x.copy()
        x[start:start + span, person, cut] = 0
    valid = pp.raw_valid(x)
    return features(np.where(valid[..., None], x, 0).astype(np.float32), valid, rng=rng, shift=shift,
                    hand_filter=hand_filter)


def split(tokens):
    """[..., FRAMES, FEATURES] -> R4 tokens [..., 16, 2, 25, 15], hands [..., 16, 4, 2, 24]."""
    tokens = np.asarray(tokens)
    lead = tokens.shape[:-2]
    body = tokens[..., :R4_FEATURES].reshape(*lead, FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS)
    hands = tokens[..., R4_FEATURES:].reshape(*lead, FRAMES, SUB, PERSONS, HAND_CHANNELS)
    return body, hands

import numpy as np
import pytest

from experiments.nestsar_sm_all_t16 import preprocessing_corrected as pp
from experiments.nestsar_r5_t16 import preprocessing as r5


def make_clip(total, people=2, seed=0, drop=0.0):
    rng = np.random.default_rng(seed)
    x = np.zeros((total, 2, 25, 3), np.float32)
    base = rng.normal(0, 0.3, (people, 25, 3)).astype(np.float32) + np.array([0.2, 0.5, 3.0], np.float32)
    t = np.arange(total, dtype=np.float32)[:, None, None, None]
    wobble = 0.05 * np.sin(0.7 * t + rng.normal(0, 1, (1, people, 25, 1)))
    x[:, :people] = base[None] + wobble + rng.normal(0, 0.01, (total, people, 25, 3))
    if drop:
        mask = rng.random((total, 2, 25)) < drop
        x[mask] = 0
    if people == 2 and total > 6:
        x[: total // 3, 1] = 0          # intermittent second person
    return x


def reference_hand(x, valid, starts, ends, scale):
    """Loop implementation of the hand block, written independently."""
    total = len(x)
    out = np.zeros((16, 4, 2, 24), np.float64)
    for f in range(16):
        s, e = int(starts[f]), int(ends[f])
        length = max(e - s, 0)
        for k in range(4):
            a = s + (k * length) // 4
            b = s + ((k + 1) * length) // 4
            if b > a:
                frames = range(a, b)
            else:
                c = min(a, max(e - 1, s))
                c = min(max(c, 0), total - 1)
                frames = range(c, c + 1)
            for p in range(2):
                col = 0
                for wrist, hand, tip, thumb in r5.HAND_SIDES:
                    for j1, j2 in ((tip, hand), (thumb, hand), (hand, wrist)):
                        acc, n = np.zeros(3), 0
                        for t in frames:
                            if valid[t, p, j1] and valid[t, p, j2]:
                                acc += x[t, p, j1] - x[t, p, j2]
                                n += 1
                        out[f, k, p, col:col + 3] = acc / max(n, 1)
                        col += 3
                    acc = np.zeros(3)
                    for t in range(a, b):
                        # R4 ownership of transition t-1 -> t: first segment whose end is after t.
                        owner = int(np.sum(np.asarray(ends) <= t))
                        if owner == f and t >= 1 and valid[t, p, wrist] and valid[t - 1, p, wrist]:
                            acc += x[t, p, wrist] - x[t - 1, p, wrist]
                    out[f, k, p, col:col + 3] = acc
                    col += 3
    return (out / scale).astype(np.float32).reshape(16, 192)


@pytest.mark.parametrize("total", [1, 5, 15, 16, 17, 41, 103, 300])
@pytest.mark.parametrize("people", [1, 2])
def test_body_tokens_match_r4_exactly(total, people):
    x = make_clip(total, people, seed=total + people, drop=0.05)
    got = r5.features(x)
    assert got.shape == (16, r5.FEATURES) and got.dtype == np.float32
    np.testing.assert_array_equal(got[:, :750], pp.features(x))
    assert np.isfinite(got).all()


@pytest.mark.parametrize("total", [5, 16, 33, 120])
def test_augmented_body_tokens_match_r4_exactly(total):
    x = make_clip(total, 2, seed=7 * total, drop=0.03)
    for epoch in (1, 2):
        got = r5.augmented_features(x, 128, epoch, 11, 8.0, 1)
        ref = pp.augmented_features(x, 128, epoch, 11, 8.0, 1)
        np.testing.assert_array_equal(got[:, :750], ref)


@pytest.mark.parametrize("total", [1, 3, 15, 16, 19, 64, 77, 250])
@pytest.mark.parametrize("shift", [0, 1])
def test_hand_block_matches_loop_reference(total, shift):
    x = make_clip(total, 2, seed=total, drop=0.1)
    local, valid, scale = pp.canonicalize_raw(x)
    rng = np.random.default_rng(3) if shift else None
    starts, ends = pp.segment_bounds(total, rng=rng, shift=shift)
    got = r5.hand_tokens(x, valid, starts, ends, scale)
    want = reference_hand(x, valid, starts, ends, scale)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_absent_second_person_gives_zero_hand_block():
    x = make_clip(60, 1, seed=1)
    hands = r5.split(r5.features(x))[1]
    assert np.abs(hands[:, :, 1]).max() == 0
    assert np.abs(hands[:, :, 0]).max() > 0


def test_translation_invariance_of_hand_block():
    x = make_clip(70, 2, seed=5)
    valid = pp.raw_valid(x)
    shifted = np.where(valid[..., None], x + np.array([0.3, -0.2, 0.5], np.float32), 0)
    a = r5.features(x)[:, 750:]
    b = r5.features(shifted)[:, 750:]
    np.testing.assert_allclose(a, b, atol=2e-5)


@pytest.mark.parametrize("total", [2, 5, 12, 15, 16, 17, 97])
def test_displacements_count_every_transition_once(total):
    x = make_clip(total, 2, seed=9)
    valid = pp.raw_valid(x)
    local, valid, scale = pp.canonicalize_raw(x)
    starts, ends = pp.segment_bounds(total)
    hands = r5.hand_tokens(x, valid, starts, ends, scale).reshape(16, 4, 2, 24)
    for p in range(2):
        for side, (wrist, *_rest) in enumerate(r5.HAND_SIDES):
            ok = valid[1:, p, wrist] & valid[:-1, p, wrist]
            total_disp = np.where(ok[:, None], x[1:, p, wrist] - x[:-1, p, wrist], 0).sum(0) / scale
            col = side * 12 + 9
            np.testing.assert_allclose(hands[:, :, p, col:col + 3].sum((0, 1)), total_disp, atol=1e-5)


def test_yaw_rotation_preserves_hand_vector_norms():
    x = make_clip(50, 2, seed=2)
    a = r5.split(r5.augmented_features(x, 1, 1, 0, rotation_degrees=0.0, shift=0))[1]
    b = r5.split(r5.augmented_features(x, 1, 1, 0, rotation_degrees=15.0, shift=0))[1]
    na = np.linalg.norm(a.reshape(*a.shape[:-1], 8, 3), axis=-1)
    nb = np.linalg.norm(b.reshape(*b.shape[:-1], 8, 3), axis=-1)
    np.testing.assert_allclose(na, nb, rtol=1e-4, atol=1e-5)


def test_empty_clip_is_all_zero():
    assert np.abs(r5.features(np.zeros((0, 2, 25, 3), np.float32))).max() == 0


def test_vectorized_representative_pose_is_bit_exact():
    rng = np.random.default_rng(0)
    for trial in range(300):
        total = int(rng.integers(1, 120))
        x = rng.normal(0, 1, (total, 2, 25, 3)).astype(np.float32)
        # Coarse random validity creates many count ties inside segments.
        drop_frames = rng.random((total, 2)) < 0.3
        x[drop_frames] = 0
        drop_joints = rng.random((total, 2, 25)) < rng.choice([0.0, 0.1, 0.6])
        x[drop_joints] = 0
        local, valid, scale = pp.canonicalize_raw(x)
        shift = int(rng.integers(0, 3))
        bounds_rng = np.random.default_rng(trial) if shift else None
        starts, ends = pp.segment_bounds(total, rng=bounds_rng, shift=shift)
        np.testing.assert_array_equal(
            r5.representative_pose(local, valid, starts, ends),
            pp.representative_pose(local, valid, starts, ends),
        )

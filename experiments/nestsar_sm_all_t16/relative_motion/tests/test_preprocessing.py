import numpy as np
import pytest
from experiments.nestsar_sm_all_t16 import preprocessing_corrected as p2
from experiments.nestsar_sm_all_t16.relative_motion import preprocessing as pp


def clip(n=64):
    x = np.zeros((n, 2, 25, 3), np.float32)
    x[:, 0] = np.array([2, 2, 4], np.float32) + np.arange(25, dtype=np.float32)[:, None]*[1/32, 1/64, 1/128]
    return x


def ambiguous_pair():
    together = clip()
    opposite = together.copy()
    wave = np.array([1, 0, -1, 0], np.float32)/8
    for t in range(4, 64):
        v = wave[(t-4) % 4]
        together[t, 0, 7, 0] += v
        together[t, 0, 6, 0] += v
        opposite[t, 0, 7, 0] += v
        opposite[t, 0, 6, 0] -= v
    return together, opposite


def reference(x, valid, ends):
    """Independent scalar accumulation, with explicit four-endpoint checks."""
    result = np.zeros(pp.SHAPE, np.float64)
    for t in range(1, len(x)):
        segment = next(s for s, end in enumerate(ends) if t < end)
        for person in range(2):
            for joint, parent in enumerate(pp.PARENTS):
                if valid[t-1, person, joint] and valid[t, person, joint] and valid[t-1, person, parent] and valid[t, person, parent]:
                    dj = x[t, person, joint] - x[t-1, person, joint]
                    dp = x[t, person, parent] - x[t-1, person, parent]
                    result[segment, person, joint] += abs(dj - dp)
    return result / p2.canonicalize_raw(x, valid)[2]


def test_preserves_information_that_is_absent_from_entire_old_tensor():
    together, opposite = ambiguous_pair()
    np.testing.assert_array_equal(p2.features(together), p2.features(opposite))
    a, b = pp.relative_path(together), pp.relative_path(opposite)
    assert not a[:, 0, 7].any()
    assert b[:, 0, 7, 0].sum() > 0
    assert not np.array_equal(pp.features(together), pp.features(opposite))


@pytest.mark.parametrize("n", [0, 1, 2, 7, 15, 16, 17, 32, 64, 301])
def test_all_native_transitions_exactly_once_and_packing(n):
    x = clip(n)
    if n:
        x[:, 0, 7, 0] += np.arange(n, dtype=np.float32)/16
    old = p2.features(x).reshape(16, 2, 25, 15)
    new = pp.features(x).reshape(16, 2, 25, 15)
    _, ends = p2.segment_bounds(n)
    expected = reference(x, p2.raw_valid(x), ends)
    np.testing.assert_allclose(new[..., 9:12], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(new[..., :9], old[..., :9])
    np.testing.assert_array_equal(new[..., 12:], old[..., 12:])
    np.testing.assert_allclose(new[..., 3:6]-new[..., 6:9], old[..., 9:12], rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(pp.pack(old.reshape(16, 750), pp.relative_path(x)), new.reshape(16, 750))
    assert new.dtype == np.float32 and np.isfinite(new).all()
    assert not new[:, 1].any() and not new[:, :, 0, 9:12].any()


def test_intermittent_tracking_masks_and_translation():
    x = clip(32)
    x[:, 0, 7, 0] += np.arange(32, dtype=np.float32)/32
    x[4:12, 0, 6] = 0
    x[14:17, 0, 7] = 0
    x[10:22, 1] = x[10:22, 0] + [1, 0, 0]
    valid = p2.raw_valid(x)
    translated = np.where(valid[..., None], x + np.array([4, -2, 8], np.float32), 0)
    np.testing.assert_allclose(pp.features(x), pp.features(translated), atol=3e-6, rtol=3e-6)
    _, ends = p2.segment_bounds(len(x))
    np.testing.assert_allclose(pp.relative_path(x), reference(x, valid, ends), atol=1e-6, rtol=1e-6)
    # A large change while the parent is missing must not become relative travel.
    y = clip(16)
    y[5:10, 0, 6] = 0
    y[5:10, 0, 7, 0] += 100
    assert not pp.relative_path(y)[:, 0, 7].any()


def test_rotation_and_phase_jitter_rebuild_motion_consistently_and_freshly():
    x = ambiguous_pair()[1]
    seed, epoch, index = 128, 3, 17
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, index]))
    theta = np.deg2rad(rng.uniform(-8, 8))
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    valid = p2.raw_valid(x)
    rotated = np.where(valid[..., None], x @ rotation.T, 0)
    bounds_rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, index]))
    bounds_rng.uniform(-8, 8)
    _, ends = p2.segment_bounds(len(x), rng=bounds_rng, shift=1)
    actual = pp.augmented_features(x, seed, epoch, index)
    np.testing.assert_array_equal(actual, pp.features(rotated, valid, rng, shift=1))
    np.testing.assert_allclose(actual.reshape(16, 2, 25, 15)[..., 9:12], reference(rotated, valid, ends), atol=1e-6)
    np.testing.assert_array_equal(actual, pp.augmented_features(x, seed, epoch, index))
    assert not np.array_equal(actual, pp.augmented_features(x, seed, epoch+1, index))
    assert not actual.reshape(16, 2, 25, 15)[:, 1].any()


def test_rejects_bad_shapes_and_nonfinite_input():
    with pytest.raises(ValueError):
        pp.features(np.zeros((16, 25, 3)))
    x = clip()
    x[3, 0, 5, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        pp.features(x)

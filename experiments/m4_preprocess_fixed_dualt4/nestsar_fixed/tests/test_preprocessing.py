import numpy as np
import pytest
from nestsar_fixed import preprocessing as p


def clip(total=32):
    x = np.zeros((total, 2, 25, 3), np.float32)
    if total:
        pose = np.random.default_rng(7).normal(0, 0.3, (25, 3)).astype(np.float32)
        pose[0] = 0
        x[:, 0] = pose + np.asarray([1, 2, 3], np.float32)
        x[:, 0, :, 0] += np.arange(total, dtype=np.float32)[:, None] * 0.125
    return x


def test_padding_mask_is_preserved():
    x = clip()
    local, valid, rms = p.canonicalize_raw(x)
    assert np.count_nonzero(local[:, 1]) == 0
    assert not valid[:, 1].any()
    main, hand = p.features(x)
    assert np.count_nonzero(main.reshape(16, 2, 25, 15)[:, 1]) == 0
    assert not hand[:, :48].reshape(32, 2, 8, 3)[:, 1].any()
    assert not hand[:, 48:].reshape(32, 2, 8, 3)[:, 1].any()


def test_translation_does_not_change_the_representation():
    x = clip()
    translated = np.where(p.raw_valid(x)[..., None], x + [5, -3, 7], 0).astype(np.float32)
    for a, b in zip(p.features(x), p.features(translated)):
        np.testing.assert_allclose(a, b, atol=8e-6, rtol=5e-5)


def test_rms_counts_valid_zero_coordinates():
    x = clip(1)
    local, valid, rms = p.canonicalize_raw(x)
    assert valid[0, 0, 0] and np.all(local[0, 0, 0] == 0)
    np.testing.assert_allclose(rms, np.sqrt(np.mean(local[valid].astype(np.float64)**2)) + 1e-6)


@pytest.mark.parametrize("total", [1, 2, 3, 7, 15, 16, 17, 32, 100, 300])
@pytest.mark.parametrize("jitter", [False, True])
def test_every_transition_is_counted_once(total, jitter):
    x = clip(total)
    rng = np.random.default_rng(123) if jitter else None
    main, _ = p.features(x, rng=rng, shift=1)
    tok = main.reshape(16, 2, 25, 15)
    _, valid, scale = p.canonicalize_raw(x)
    delta = np.where((valid[1:] & valid[:-1])[..., None], np.diff(x, axis=0), 0)
    np.testing.assert_allclose(tok[..., 3:6].sum(0), delta.sum(0) / scale, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(tok[..., 12:15].sum(0), np.abs(delta).sum(0) / scale, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(tok[..., 3:6], tok[..., 6:9] + tok[..., 9:12], rtol=1e-5, atol=2e-5)
    assert np.all(tok[..., 12:15] >= -1e-6)


def test_16_frame_motion_is_not_zero():
    main, _ = p.features(clip(16))
    tok = main.reshape(16, 2, 25, 15)
    assert np.count_nonzero(tok[:, 0, 0, 3]) == 15


def test_no_motion_across_missing_joints():
    x = clip(3)
    x[1, 0, 21] = 0
    main, hand = p.features(x)
    assert not main.reshape(16, 2, 25, 15)[:, 0, 21, 3:].any()
    assert not hand[:, 48:].reshape(32, 2, 8, 3)[:, 0, 2].any()


def test_hand_at_reference_origin_still_has_valid_velocity():
    x = clip(32)
    x[:, 0, 6] = x[:, 0, 0]
    _, hand = p.features(x)
    velocity = hand[:, 48:].reshape(32, 2, 8, 3)
    assert np.all(velocity[1:, 0, 0, 0] > 0)


def test_missing_root_fallback_preserves_translation_invariance():
    x = clip(17)
    x[3:6, 0, 0] = 0
    mask = p.raw_valid(x)
    y = np.where(mask[..., None], x + [2, 3, -1], 0).astype(np.float32)
    np.testing.assert_allclose(p.features(x)[0], p.features(y)[0], atol=1e-5)


def test_augmentation_is_fresh_reproducible_and_does_not_mutate_input():
    x = clip(100)
    saved = x.copy()
    a = p.training_views(x, 128, 1, 13)
    b = p.training_views(x, 128, 1, 13)
    c = p.training_views(x, 128, 2, 13)
    for first, replay in zip(a, b):
        np.testing.assert_array_equal(first, replay)
    np.testing.assert_array_equal(a[0], c[0])
    assert not np.array_equal(a[1], c[1])
    assert not np.array_equal(a[3], c[3])
    np.testing.assert_array_equal(x, saved)
    assert not a[1].reshape(16, 2, 25, 15)[:, 1].any()


@pytest.mark.parametrize("total", [0, 1, 16])
def test_empty_and_padded_inputs_are_finite(total):
    for a in p.features(np.zeros((total, 2, 25, 3), np.float32)):
        assert np.isfinite(a).all() and not a.any()


def test_short_clip_layout_is_explicit():
    x = clip(2)
    raw = x.transpose(1, 0, 2, 3)
    np.testing.assert_array_equal(p.ordered_raw(raw), x)
    np.testing.assert_array_equal(p.ordered_raw(x, "TMVC"), x)
    with pytest.raises(ValueError):
        p.ordered_raw(np.zeros((2, 16, 24, 3)))

import numpy as np

from experiments.nestsar_sm_all_t16 import preprocessing_corrected as p


def clip(total=32):
    x = np.zeros((total, 2, 25, 3), np.float32)
    if total:
        pose = np.random.default_rng(7).normal(0, 0.3, (25, 3)).astype(np.float32)
        pose[0] = 0
        x[:, 0] = pose + np.asarray([1, 2, 3], np.float32)
        x[:, 0, :, 0] += np.arange(total, dtype=np.float32)[:, None] * 0.125
    return x


def test_absent_person_stays_zero():
    x = clip(32)
    local, valid, _ = p.canonicalize_raw(x)
    assert not valid[:, 1].any()
    assert np.count_nonzero(local[:, 1]) == 0
    tok = p.features(x).reshape(16, 2, 25, 15)
    assert np.count_nonzero(tok[:, 1]) == 0


def test_ordered_raw_preserves_two_occupied_source_tracks_not_energy_order():
    # P0 is deliberately low-energy and P1 high-energy. Their identities must
    # remain in source order rather than being swapped by clip energy.
    raw = np.zeros((2, 20, 25, 3), np.float32)  # M,T,V,C
    raw[0, :, :, 0] = 1.0
    raw[1, :, :, 0] = 9.0
    out = p.ordered_raw(raw, "MTVC")
    assert np.allclose(out[:, 0, :, 0], 1.0)
    assert np.allclose(out[:, 1, :, 0], 9.0)


def test_ordered_raw_compacts_empty_leading_track_without_losing_real_actor():
    raw = np.zeros((2, 20, 25, 3), np.float32)
    raw[1, :, :, 0] = 4.0
    out = p.ordered_raw(raw, "MTVC")
    assert np.allclose(out[:, 0, :, 0], 4.0)
    assert not out[:, 1].any()


def test_intermittent_p2_is_preserved_when_common_midpoint_is_empty():
    x = clip(32)
    # Segment 0 is frames [0,2) for T=32 -> 16 segments. Put P2 only at frame
    # 1; the historical common midpoint selected frame 0 and erased P2 pose.
    p2_pose = np.random.default_rng(9).normal(0.7, 0.05, (25, 3)).astype(np.float32)
    x[1, 1] = p2_pose + np.asarray([1.5, 2.5, 3.5], np.float32)
    tok = p.features(x).reshape(16, 2, 25, 15)
    assert np.count_nonzero(tok[0, 1, :, 0:3]) > 0
    # P2 is absent from the next segment and must not leak there.
    assert np.count_nonzero(tok[1, 1, :, 0:3]) == 0


def test_intermittent_p2_does_not_create_false_motion_jump():
    x = clip(32)
    p2_pose = np.random.default_rng(11).normal(0.5, 0.1, (25, 3)).astype(np.float32)
    x[1, 1] = p2_pose + np.asarray([2, 1, 3], np.float32)
    tok = p.features(x).reshape(16, 2, 25, 15)
    # One isolated P2 frame has no valid consecutive P2 transition.
    assert not tok[:, 1, :, 3:].any()


def test_translation_invariance_with_padding():
    x = clip(32)
    valid = p.raw_valid(x)
    moved = np.where(valid[..., None], x + [5, -3, 7], 0).astype(np.float32)
    np.testing.assert_allclose(p.features(x), p.features(moved), atol=8e-6, rtol=5e-5)


def test_every_transition_is_counted_once_32_frames():
    x = clip(32)
    tok = p.features(x).reshape(16, 2, 25, 15)
    _, valid, scale = p.canonicalize_raw(x)
    delta = np.where((valid[1:] & valid[:-1])[..., None], np.diff(x, axis=0), 0)
    np.testing.assert_allclose(tok[..., 3:6].sum(0), delta.sum(0) / scale, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(tok[..., 12:15].sum(0), np.abs(delta).sum(0) / scale, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(tok[..., 3:6], tok[..., 6:9] + tok[..., 9:12], rtol=1e-5, atol=2e-5)


def test_16_frame_motion_is_not_zero():
    tok = p.features(clip(16)).reshape(16, 2, 25, 15)
    assert np.count_nonzero(tok[:, 0, 0, 3]) == 15


def test_missing_joint_does_not_create_motion_jump():
    x = clip(3)
    x[1, 0, 21] = 0
    tok = p.features(x).reshape(16, 2, 25, 15)
    assert not tok[:, 0, 21, 3:].any()


def test_fresh_augmentation_is_reproducible_per_epoch():
    x = clip(100)
    can1, aug1 = p.training_views(x, seed=128, epoch=1, sample_index=13)
    can1b, aug1b = p.training_views(x, seed=128, epoch=1, sample_index=13)
    can2, aug2 = p.training_views(x, seed=128, epoch=2, sample_index=13)
    np.testing.assert_array_equal(can1, can1b)
    np.testing.assert_array_equal(aug1, aug1b)
    np.testing.assert_array_equal(can1, can2)
    assert not np.array_equal(aug1, aug2)
    assert not aug1.reshape(16, 2, 25, 15)[:, 1].any()


def test_shapes_are_architecture_compatible():
    assert p.features(clip(32)).shape == (16, 750)

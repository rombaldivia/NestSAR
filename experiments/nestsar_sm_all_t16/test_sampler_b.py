"""CPU regression checks for the sampling-only intervention."""
import numpy as np
import pytest

from . import preprocessing_corrected as pp
from .sampler_b import SamplerB, ScaleAccumulator, relative_motion, CHILD, PARENT, POLICY, VERSION, LINK_NAMES


def calibration(scale=0.01):
    return dict(version=VERSION, policy=POLICY, links=list(LINK_NAMES),
                scale=[scale] * 8, enabled=[True] * 8)


def clip(total=160):
    rng = np.random.default_rng(42)
    x = np.zeros((total, 2, 25, 3), np.float32)
    x[:, 0] = rng.normal(0, .1, (1, 25, 3)) + [1, 2, 3]
    return x


@pytest.mark.parametrize("link", range(8))
@pytest.mark.parametrize("endpoint", [0, 1])
@pytest.mark.parametrize("joint", ["child", "parent"])
def test_motion_requires_all_four_endpoints(link, endpoint, joint):
    x = clip(2)
    x[1, 0, CHILD[link], 0] += .5
    mask = pp.raw_valid(x)
    mask[endpoint, 0, (CHILD if joint == "child" else PARENT)[link]] = False
    d, eligible = relative_motion(x, mask)
    assert not eligible[0, 0, link]
    assert not d[0, 0, link].any()


def test_opposed_equal_paths_have_nonzero_relative_motion():
    x = clip(40)
    x[:, 0, 21, 0] += np.arange(40) * .02
    x[:, 0, 6, 0] -= np.arange(40) * .02
    d, eligible = relative_motion(x, pp.raw_valid(x))
    np.testing.assert_allclose(d[:, 0, 0, 0], .04, atol=3e-7)
    assert eligible[:, 0, 0].all()


@pytest.mark.parametrize("total", [0, 1, 2, 7, 16, 32, 160])
def test_static_and_short_clips_match_midpoint_and_preserve_padding(total):
    x = clip(total)
    b = pp.features(x, pose_selector=SamplerB(calibration()))
    np.testing.assert_array_equal(b, pp.features(x))
    assert b.shape == (16, 750) and np.isfinite(b).all()
    assert not b.reshape(16, 2, 25, 15)[:, 1].any()


def test_rigid_translation_does_not_create_local_motion():
    x = clip()
    x[:, 0] += (np.arange(len(x)) * .01)[:, None, None]
    sampler = SamplerB(calibration())
    np.testing.assert_array_equal(pp.features(x, pose_selector=sampler), pp.features(x))


def test_reliable_local_event_changes_pose_only_and_selects_whole_skeleton():
    x = clip()
    # A non-scoring joint identifies the selected timestamp in the full pose.
    x[:, 0, 2, 0] += np.arange(len(x)) * .001
    # Sustained event at frames 1..3, away from first bin midpoint (4.5).
    x[1:, 0, 21, 0] += .01
    x[2:, 0, 21, 0] += .02
    x[3:, 0, 21, 0] += .02
    sampler = SamplerB(calibration())
    local, valid, scale = pp.canonicalize_raw(x)
    starts, ends = pp.segment_bounds(len(x))
    pose, selected, stats = sampler.select(local, valid, starts, ends)
    assert selected[0, 0] in (1, 2, 3) and stats["changed_from_midpoint"] > 0
    for seg in range(16):
        np.testing.assert_array_equal(pose[seg, 0], local[selected[seg, 0], 0])
    base = pp.features(x).reshape(16, 2, 25, 15)
    b = pp.features(x, pose_selector=sampler).reshape(base.shape)
    np.testing.assert_array_equal(b[..., 3:], base[..., 3:])
    assert not np.array_equal(b[..., :3], base[..., :3])


def test_large_reversal_spike_cannot_win_over_supported_movement():
    x = clip()
    x[1:, 0, 21, 0] += .01
    x[2:, 0, 21, 0] += .02
    x[3:, 0, 21, 0] += .02
    x[7, 0, 21, 0] += 50.0  # one-frame tracking error
    sampler = SamplerB(calibration())
    local, valid, _ = pp.canonicalize_raw(x)
    scores, suspect = sampler.frame_scores(*relative_motion(local, valid))
    assert suspect[7, 0] and scores[7, 0] == 0
    _, selected, stats = sampler.select(local, valid, *pp.segment_bounds(len(x)))
    assert selected[0, 0] in (1, 2, 3)
    assert stats["suspect_frames"] >= 1


def test_unsupported_single_transition_falls_back():
    x = clip()
    x[2:, 0, 21, 0] += .1
    sampler = SamplerB(calibration())
    np.testing.assert_array_equal(pp.features(x, pose_selector=sampler), pp.features(x))


def test_missing_interval_cannot_support_a_spike_across_the_gap():
    x = clip(8)
    x[1, 0, 21, 0] += .1
    x[2:6, 0, 21] = 0
    x[6:, 0, 21, 0] += 2
    scores, _ = SamplerB(calibration()).frame_scores(*relative_motion(x, pp.raw_valid(x)))
    assert not scores.any()


def test_maximum_valid_joints_and_intermittent_p2_policy():
    x = clip()
    x[2:4, 1] = x[2:4, 0] + .5
    x[2, 1, 24] = 0
    x[3, 1, 21, 0] += .5
    sampler = SamplerB(calibration())
    local, valid, _ = pp.canonicalize_raw(x)
    pose, selected, _ = sampler.select(local, valid, *pp.segment_bounds(len(x)))
    assert selected[0, 1] == 3 and (selected[1:, 1] == -1).all()
    np.testing.assert_array_equal(pose[0, 1], local[3, 1])
    assert not pose[1:, 1].any()


@pytest.mark.parametrize("total", [7, 16, 32, 161])
@pytest.mark.parametrize("epoch", [1, 2])
def test_fresh_augmented_motion_is_bitwise_unchanged(total, epoch):
    x = clip(total)
    x[:, 0, 21, 0] += .03 * np.sin(np.arange(total))
    x[::5, 0, 22] = 0
    sampler = SamplerB(calibration())
    args = (x, 128, epoch, 19, 8.0, 1)
    a = pp.augmented_features(*args).reshape(16, 2, 25, 15)
    b = pp.augmented_features(*args, pose_selector=sampler).reshape(a.shape)
    np.testing.assert_array_equal(a[..., 3:], b[..., 3:])
    np.testing.assert_array_equal(b, pp.augmented_features(*args, pose_selector=sampler).reshape(a.shape))


def test_link_scales_balance_equal_relative_hand_and_leg_activity():
    d = np.zeros((4, 2, 8, 3), np.float64)
    d[:, 0, :4, 0], d[:, 1, 4:, 0] = .01, .1
    c = calibration()
    c["scale"] = [.01] * 4 + [.1] * 4
    score, _ = SamplerB(c).frame_scores(d, np.ones((4, 2, 8), bool))
    np.testing.assert_allclose(score[:, 0], score[:, 1])


def test_calibration_is_bounded_and_resists_rare_extreme_values():
    a = ScaleAccumulator()
    d = np.full((200, 2, 8, 3), .01, np.float64)
    d[-1] = 100.0
    a.observe(d, np.ones(d.shape[:-1], bool))
    c = a.finish()
    assert a.histogram.nbytes <= 32768
    assert all(c["enabled"])
    np.testing.assert_allclose(c["scale"], np.sqrt(3) * .01, rtol=.05)
    assert c["samples_observed"] == 1
    assert c["valid_transitions"] == [400] * 8


def test_empty_training_signal_disables_sampler():
    a = ScaleAccumulator()
    x = clip()
    a.observe(*relative_motion(x, pp.raw_valid(x)))
    c = a.finish()
    assert not any(c["enabled"])
    x[:, 0, 21, 0] += np.arange(len(x)) * .1
    np.testing.assert_array_equal(pp.features(x), pp.features(x, pose_selector=SamplerB(c)))

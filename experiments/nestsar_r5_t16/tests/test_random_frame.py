"""aug_random_frame (SGN-style random frame per segment): only the pose channels change, every other draw and
channel is the same, the pose is a real frame of its segment, and the option is off by default."""
import numpy as np
import pytest

from experiments.nestsar_r5_t16 import preprocessing as pp
from experiments.nestsar_r5_t16.config import validate_config
from experiments.nestsar_sm_all_t16 import preprocessing_corrected as base_pp


def _clip(seed=0, frames=90):
    x = np.random.default_rng(seed).normal(0, 0.3, (frames, 2, 25, 3)).astype(np.float32)
    x[:, 1] = 0                                          # one actor
    return x


def test_only_pose_channels_change():
    x = _clip()
    a = pp.strong_augmented_features(x, 7, 3, 11, 1.0, stream=1, shift=1)
    b = pp.strong_augmented_features(x, 7, 3, 11, 1.0, stream=1, shift=1, random_frame=True)
    body_a, hand_a = pp.split(a)
    body_b, hand_b = pp.split(b)
    np.testing.assert_array_equal(body_a[..., 3:], body_b[..., 3:])     # displacements and path
    np.testing.assert_array_equal(hand_a, hand_b)
    assert np.abs(body_a[..., :3] - body_b[..., :3]).max() > 0


def test_pose_is_a_frame_of_its_segment():
    x = _clip(1)
    valid = base_pp.raw_valid(x)
    local, valid, _ = base_pp.canonicalize_raw(x, valid)
    starts, ends = base_pp.segment_bounds(len(x))
    pose = pp.representative_pose(local, valid, starts, ends, np.random.default_rng(0))
    for f in range(pp.FRAMES):
        seg = local[starts[f]:ends[f], 0]
        assert any(np.array_equal(pose[f, 0], frame) for frame in seg)
    np.testing.assert_array_equal(pp.representative_pose(local, valid, starts, ends),
                                  base_pp.representative_pose(local, valid, starts, ends))


def test_config():
    assert validate_config({})["aug_random_frame"] is False
    validate_config({"aug_strength": 1.0, "aug_random_frame": True})
    for bad in ({"aug_random_frame": True}, {"aug_random_frame": 1},
                {"aug_strength": 1.0, "aug_pool": "/tmp/p", "aug_random_frame": True}):
        with pytest.raises(ValueError):
            validate_config(bad)

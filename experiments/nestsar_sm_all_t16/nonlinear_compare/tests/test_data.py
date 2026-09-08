import numpy as np
import pytest

from experiments.nestsar_sm_all_t16.nonlinear_compare.config import validate_config
from experiments.nestsar_sm_all_t16.nonlinear_compare.data import (
    TrainOnlyCache, fit_scale, grouped_plan, make_batch, sequence_features)
from experiments.nestsar_sm_all_t16.nonlinear_compare.metrics import paired_group_bootstrap, scores
from experiments.nestsar_sm_all_t16 import preprocessing_corrected as pp


def clip():
    x = np.zeros((32, 2, 25, 3), np.float32)
    x[:, 0] = np.random.default_rng(5).normal(0, .2, (25, 3)) + [1, 2, 3]
    x[:, 0, :, 0] += np.arange(32)[:, None]*.05
    return x


def test_group_plan_is_disjoint_reproducible_and_class_supported():
    indices = np.arange(192)
    groups = np.repeat(np.arange(12), 16)
    labels = np.tile([70, 71], 96)
    c = validate_config(dict(pairs=[[71, 72]], min_class_samples=[8, 4, 4]))
    plan = grouped_plan(indices, groups, labels, c["pairs"], 128, c)
    assert plan == grouped_plan(indices, groups, labels, c["pairs"], 128, c)
    parts = [set(plan["groups"][k]) for k in ("fit", "select", "final")]
    assert all(not parts[i] & parts[j] for i in range(3) for j in range(i))
    for name, minimum in zip(("fit", "select", "final"), c["min_class_samples"]):
        assert min(np.bincount(labels[plan["indices"][name]]-70)) >= minimum
    assert plan != grouped_plan(indices, groups, labels, c["pairs"], 42, c)


def test_guard_rejects_heldout_before_accessing_arrays():
    cache = object.__new__(TrainOnlyCache)
    cache.allowed = np.array([True, False, True])
    assert cache.guard([0, 2]).tolist() == [0, 2]
    with pytest.raises(ValueError, match="held-out access blocked"):
        cache.guard([0, 1])
    with pytest.raises(ValueError):
        cache.guard([-1])


@pytest.mark.parametrize("frames", [16, 64])
def test_sequence_masks_and_translation_invariance(frames):
    x = clip()
    valid = pp.raw_valid(x)
    moved = np.where(valid[..., None], x+[5, -4, 7], 0).astype(np.float32)
    actual = sequence_features(x, frames)
    assert actual.shape == (frames, 2, 25, 8)
    assert not actual[:, 1].any()
    np.testing.assert_allclose(actual, sequence_features(moved, frames), atol=1e-5, rtol=5e-5)
    assert actual[..., 3:6].any()


def test_sequence_interpolation_never_bridges_missing_joint():
    x = clip()[:3]
    x[1, 0, 21] = 0
    result = sequence_features(x, 5)
    assert not result[:, 0, 21, 3:6].any()
    assert not result[1:4, 0, 21, :].any()
    assert result[0, 0, 21, 6] == result[-1, 0, 21, 6] == 1


def test_normalization_never_uses_select_or_final_and_preserves_padding():
    x = np.stack([pp.features(clip()) for _ in range(4)])
    scale = fit_scale(x, np.array([0, 1]))
    x[2:] = 100000
    np.testing.assert_array_equal(scale, fit_scale(x, np.array([0, 1])))
    b = make_batch(x, np.array([0, 1, 0, 1]), [0, 1], 4, scale)
    assert b["mask"].tolist() == [1, 1, 0, 0]
    assert not b["x"][2:].any()
    assert not b["x"][:2].reshape(2, 16, 2, 25, 15)[:, :, 1].any()


def test_paired_group_bootstrap_is_paired_and_has_known_difference():
    y = np.tile([0, 1], 12)
    groups = np.repeat(np.arange(6), 4)
    equal = paired_group_bootstrap(y, y, y, groups, 5, 200)
    assert equal["delta_pp"] == 0 and equal["ci95_pp"] == [0, 0]
    opposite = paired_group_bootstrap(y, y, 1-y, groups, 5, 200)
    assert opposite["delta_pp"] == -100 and opposite["ci95_pp"] == [-100, -100]
    assert scores(np.array([0, 0, 0, 1]), np.tile([.9, .1], (4, 1)))["balanced_accuracy"] == .5


def test_invalid_config_is_rejected():
    for config in ({"pairs": [[71, 71]]}, {"seeds": [42, 42]}, {"batch_size": 0},
                   {"trials": [{"learning_rate": float("nan"), "weight_decay": .1, "dropout": .1}]}):
        with pytest.raises(ValueError):
            validate_config(config)

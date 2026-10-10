import numpy as np
import pytest

from experiments.nestsar_r5_t16 import data as r5data
from experiments.nestsar_r5_t16 import model as r5model
from experiments.nestsar_r5_t16 import preprocessing as pp
from experiments.nestsar_r5_t16.config import DEFAULTS, validate_config
from experiments.nestsar_r5_t16.smoke_cpu import synthetic_pickle
from experiments.nestsar_sm_all_t16 import preprocessing_corrected as r4pp
from experiments.nestsar_sm_all_t16.streaming.data import prepare


def clip(total=90, persons=2, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((total, 2, 25, 3), np.float32)
    base = rng.normal(0, 0.3, (1, persons, 25, 3)) + np.array([0, 0.4, 3.0])
    t = np.arange(total)[:, None, None, None]
    x[:, :persons] = base + 0.05 * np.sin(0.3 * t) + rng.normal(0, 0.01, (total, persons, 25, 3))
    return x


def test_parents_match_the_model_tree_and_order_is_topological():
    np.testing.assert_array_equal(pp.NTU_PARENTS, r5model.PARENTS)
    seen = set()
    for j in pp.TREE_ORDER:
        assert j == 0 or pp.NTU_PARENTS[j] in seen
        seen.add(j)
    assert len(seen) == 25


def test_deterministic_and_stream_dependent():
    x = clip()
    a = pp.strong_augmented_features(x, 1, 3, 7, 1.0, stream=1)
    b = pp.strong_augmented_features(x, 1, 3, 7, 1.0, stream=1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, pp.strong_augmented_features(x, 1, 3, 7, 1.0, stream=0))
    assert not np.array_equal(a, pp.strong_augmented_features(x, 1, 4, 7, 1.0, stream=1))
    assert not np.array_equal(a, pp.strong_augmented_features(x, 1, 3, 8, 1.0, stream=1))


@pytest.mark.parametrize("total", [1, 2, 7, 8, 15, 16, 17, 40, 300])
@pytest.mark.parametrize("persons", [1, 2])
def test_finite_and_correct_shape_for_every_clip_length(total, persons):
    x = clip(total, persons)
    for seed in range(6):
        out = pp.strong_augmented_features(x, seed, 1, seed, 1.5)
        assert out.shape == (pp.FRAMES, pp.FEATURES) and out.dtype == np.float32
        assert np.isfinite(out).all()


def test_empty_and_all_invalid_clips_do_not_crash():
    assert pp.strong_augmented_features(np.zeros((0, 2, 25, 3), np.float32), 0, 1, 0).shape == (16, 942)
    assert np.isfinite(pp.strong_augmented_features(np.zeros((30, 2, 25, 3), np.float32), 0, 1, 0)).all()


def test_absent_second_actor_stays_absent():
    x = clip(60, persons=1)
    for seed in range(10):
        body, hands = pp.split(pp.strong_augmented_features(x, seed, 1, seed, 2.0))
        assert np.abs(body[:, 1]).max() == 0 and np.abs(hands[:, :, 1]).max() == 0


def test_bone_perturbation_keeps_the_tree_connected_and_scales_bones():
    x = clip(30)
    valid = r4pp.raw_valid(x)
    out = pp._perturb_bones(x, valid, np.random.default_rng(3), 1.0)
    ratios = []
    for j in range(1, 25):
        p = pp.NTU_PARENTS[j]
        a = np.linalg.norm(x[:, 0, j] - x[:, 0, p], axis=-1)
        b = np.linalg.norm(out[:, 0, j] - out[:, 0, p], axis=-1)
        r = b / np.maximum(a, 1e-9)
        assert np.allclose(r, r[0], rtol=1e-4)            # one factor per bone for the whole clip
        ratios.append(r[0])
    ratios = np.asarray(ratios)
    assert ratios.min() >= 1 - 0.12 - 1e-4 and ratios.max() <= 1 + 0.12 + 1e-4
    assert ratios.std() > 0.01                              # bones really differ
    np.testing.assert_array_equal(out[:, :, 0], x[:, :, 0])  # root untouched


def test_bone_perturbation_ignores_missing_joints():
    x = clip(30)
    x[:, 0, 7] = 0                                          # left hand missing
    out = pp._perturb_bones(x, r4pp.raw_valid(x), np.random.default_rng(0), 1.0)
    assert np.abs(out[:, 0, 7]).max() == 0 and np.isfinite(out).all()


def test_resample_time_length_and_validity():
    x = clip(40)
    x[10, 0, 5] = 0                                         # joint invalid in one frame
    valid = r4pp.raw_valid(x)
    fast, vf = pp._resample_time(x, valid, 1.25)
    slow, vs = pp._resample_time(x, valid, 0.8)
    assert len(fast) == 32 and len(slow) == 50
    assert np.abs(fast[~vf]).max(initial=0) == 0 and np.abs(slow[~vs]).max(initial=0) == 0
    assert vs[:, 0, 5].sum() < len(vs)                      # the gap is not filled in
    ident, vi = pp._resample_time(x, valid, 1.0)
    np.testing.assert_allclose(ident, x, atol=1e-6)


def test_strength_scales_how_far_the_view_moves():
    x = clip(80)
    canonical = pp.features(x)
    d = lambda s: np.mean([np.abs(pp.strong_augmented_features(x, 0, 1, i, s) - canonical).mean()
                           for i in range(12)])
    assert d(0.25) < d(1.0) < d(2.0)


def test_default_config_keeps_the_r4_augmentation_and_validates_ranges():
    assert DEFAULTS["aug_strength"] == 0.0
    c = validate_config({"aug_strength": 1.0, "prefetch_workers": 2, "aug_clean_prob": 0.3})
    assert c["aug_strength"] == 1.0
    for bad in ({"aug_strength": -0.1}, {"aug_strength": 2.5}, {"aug_clean_prob": 1.5}, {"prefetch_workers": 0},
                {"prefetch_workers": 9}):
        with pytest.raises(ValueError):
            validate_config(bad)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("strong_aug")
    synthetic_pickle(root / "ntu120_3danno.pkl")
    prepare(root / "ntu120_3danno.pkl", root / "r4", root / "r4_status.json")
    r5data.build(root / "r4", root / "hand")
    return r5data.Dataset(root / "hand")


def test_dataset_default_path_is_unchanged_and_strong_path_differs(dataset):
    ids = dataset.splits["xsub_train"]
    base_cfg = validate_config({"micro_batch": 2, "accumulation_steps": 2})
    strong_cfg = validate_config({"micro_batch": 2, "accumulation_steps": 2, "aug_strength": 1.0,
                                  "aug_clean_prob": 0.0, "prefetch_workers": 2})
    base = list(dataset.batches(ids, 4, base_cfg, epoch=3, training=True, protocol="xsub"))
    strong = list(dataset.batches(ids, 4, strong_cfg, epoch=3, training=True, protocol="xsub"))
    again = list(dataset.batches(ids, 4, strong_cfg, epoch=3, training=True, protocol="xsub"))
    assert len(base) == len(strong)
    total = 0
    for (b, _), (s, _), (a, _) in zip(base, strong, again):
        n = int(b["mask"].sum())
        total += n
        np.testing.assert_array_equal(s["xa"], a["xa"])         # reproducible with several workers
        np.testing.assert_array_equal(s["x"], a["x"])
        assert not np.array_equal(b["xa"][:n], s["xa"][:n])
        assert not np.array_equal(b["x"][:n], s["x"][:n])       # clean view is augmented when prob = 0
        assert np.isfinite(s["x"]).all() and np.isfinite(s["xa"]).all()
        assert np.abs(s["x"][n:]).max(initial=0) == 0 and np.abs(s["xa"][n:]).max(initial=0) == 0
    assert total == len(ids)


def test_aug_clean_prob_one_keeps_the_canonical_view(dataset):
    ids = dataset.splits["xsub_train"]
    cfg = validate_config({"micro_batch": 2, "accumulation_steps": 2, "aug_strength": 1.0, "aug_clean_prob": 1.0})
    for batch, _ in dataset.batches(ids, 4, cfg, epoch=2, training=True, protocol="xset"):
        n = int(batch["mask"].sum())
        assert np.isfinite(batch["x"]).all()
    # every clean view equals some canonical clip
    canon = {dataset.canonical([i])[0].tobytes() for i in ids}
    for batch, _ in dataset.batches(ids, 4, cfg, epoch=2, training=True, protocol="xset"):
        for j in range(int(batch["mask"].sum())):
            assert batch["x"][j].tobytes() in canon


# ------------------------------------------------------------------ hand-joint denoising
def noisy_hand_clip(total=90, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(total)[:, None, None, None]
    base = np.random.default_rng(seed).normal(0, 0.3, (1, 2, 25, 3)) + np.array([0, 0.4, 3.0])
    clean = (base + 0.05 * np.sin(0.3 * t)).astype(np.float32)   # smooth ground truth
    clean = np.broadcast_to(clean, (total, 2, 25, 3)).copy()
    x = clean.copy()
    for j in (6, 7, 21, 22, 10, 11, 23, 24):
        x[:, :, j] += rng.normal(0, 0.006, x[:, :, j].shape)
        spike = rng.integers(3, total - 3, 3)
        x[spike, 0, j] += 0.25
    return clean, x


def test_filter_touches_only_hand_joints_and_none_is_identity():
    _, x = noisy_hand_clip()
    valid = r4pp.raw_valid(x)
    assert pp.denoise_hand_joints(x, valid, "none") is x
    for mode in ("hampel", "smooth"):
        y = pp.denoise_hand_joints(x, valid, mode)
        other = [j for j in range(25) if j not in pp._HAND_JOINTS]
        np.testing.assert_array_equal(y[:, :, other], x[:, :, other])
        assert not np.array_equal(y[:, :, list(pp._HAND_JOINTS)], x[:, :, list(pp._HAND_JOINTS)])
        assert y.shape == x.shape and y.dtype == np.float32


def test_filter_reduces_error_against_the_clean_signal():
    clean, x = noisy_hand_clip()
    valid = r4pp.raw_valid(x)
    h = list(pp._HAND_JOINTS)
    err = lambda a: np.abs(a[:, :, h] - clean[:, :, h]).mean()
    assert err(pp.denoise_hand_joints(x, valid, "hampel")) < err(x)
    assert err(pp.denoise_hand_joints(x, valid, "smooth")) < err(pp.denoise_hand_joints(x, valid, "hampel"))


def test_filter_keeps_gaps_and_handles_short_runs():
    _, x = noisy_hand_clip(40)
    x[10:14, 0, 7] = 0                                       # gap
    x[20:23, 1, 11] = x[20:23, 1, 11]                        # 3-frame run between two gaps below
    x[19, 1, 11] = 0
    x[23, 1, 11] = 0
    valid = r4pp.raw_valid(x)
    for mode in ("hampel", "smooth"):
        y = pp.denoise_hand_joints(x, valid, mode)
        assert np.abs(y[10:14, 0, 7]).max() == 0 and np.isfinite(y).all()
        np.testing.assert_array_equal(y[20:23, 1, 11], x[20:23, 1, 11])    # too short to filter
    assert np.isfinite(pp.features(x[:3], hand_filter="smooth")).all()


def test_filter_leaves_r4_tokens_unchanged_and_changes_hand_tokens():
    _, x = noisy_hand_clip()
    plain, filt = pp.features(x), pp.features(x, hand_filter="smooth")
    np.testing.assert_array_equal(plain[:, :pp.R4_FEATURES], filt[:, :pp.R4_FEATURES])
    assert not np.array_equal(plain[:, pp.R4_FEATURES:], filt[:, pp.R4_FEATURES:])
    with pytest.raises(ValueError):
        pp.features(x, hand_filter="bogus")
    for fn in (lambda f: pp.augmented_features(x, 0, 1, 2, hand_filter=f),
               lambda f: pp.strong_augmented_features(x, 0, 1, 2, 1.0, hand_filter=f)):
        np.testing.assert_array_equal(fn("none")[:, :pp.R4_FEATURES], fn("smooth")[:, :pp.R4_FEATURES])


def test_filtered_cache_has_its_own_signature_and_dataset_uses_it(tmp_path):
    synthetic_pickle(tmp_path / "ntu120_3danno.pkl")
    prepare(tmp_path / "ntu120_3danno.pkl", tmp_path / "r4", tmp_path / "s.json")
    plain = r5data.build(tmp_path / "r4", tmp_path / "plain")
    filt = r5data.build(tmp_path / "r4", tmp_path / "filt", hand_filter="smooth")
    assert "hand_filter" not in plain["signature"] and filt["signature"]["hand_filter"] == "smooth"
    dp, df = r5data.Dataset(tmp_path / "plain"), r5data.Dataset(tmp_path / "filt")
    assert dp.hand_filter == "none" and df.hand_filter == "smooth"
    np.testing.assert_array_equal(dp.base.canonical[:2], df.base.canonical[:2])
    with pytest.raises(ValueError):
        r5data.build(tmp_path / "r4", tmp_path / "plain", hand_filter="hampel")
    assert validate_config({"hand_filter": "smooth"})["hand_filter"] == "smooth"
    with pytest.raises(ValueError):
        validate_config({"hand_filter": "x"})


def test_sun_frame_filter_replaces_noisy_frames_with_the_previous_one():
    clean, x = noisy_hand_clip()
    valid = r4pp.raw_valid(x)
    h = list(pp._HAND_JOINTS)
    bad = x.copy()
    bad[40, 0, [6, 7, 21, 22]] += 1.0                        # whole left hand teleports for one frame
    y = pp.denoise_hand_joints(bad, valid, "sun")
    np.testing.assert_array_equal(y[40, 0, [6, 7, 21, 22]], bad[39, 0, [6, 7, 21, 22]])
    other = [j for j in range(25) if j not in pp._HAND_JOINTS]
    np.testing.assert_array_equal(y[:, :, other], bad[:, :, other])
    err = lambda a: np.abs(a[:, :, h] - clean[:, :, h]).mean()
    assert err(y) < err(bad)
    assert err(pp.denoise_hand_joints(bad, valid, "sun_smooth")) < err(y)


def test_sun_filter_keeps_clean_clips_and_gaps():
    clean = noisy_hand_clip()[0]
    valid = r4pp.raw_valid(clean)
    np.testing.assert_array_equal(pp.denoise_hand_joints(clean, valid, "sun"), clean)
    x = clean.copy()
    x[10:15, 0, 7] = 0
    y = pp.denoise_hand_joints(x, r4pp.raw_valid(x), "sun")
    assert np.abs(y[10:15, 0, 7]).max() == 0 and np.isfinite(y).all()
    assert np.isfinite(pp.features(x[:5], hand_filter="sun_smooth")).all()


def test_view_degrees_widens_the_viewpoint_range_and_default_is_unchanged():
    x = clip(60)
    base = pp.strong_augmented_features(x, 0, 1, 3, 1.0)
    np.testing.assert_array_equal(base, pp.strong_augmented_features(x, 0, 1, 3, 1.0, view_degrees=15.0))
    d = lambda deg: np.mean([np.abs(pp.strong_augmented_features(x, 0, 1, i, 1.0, view_degrees=deg)
                                    - pp.features(x)).mean() for i in range(12)])
    assert d(45.0) > d(15.0) > d(0.0)
    assert validate_config({"aug_view_degrees": 45})["aug_view_degrees"] == 45
    with pytest.raises(ValueError):
        validate_config({"aug_view_degrees": 120})


# ------------------------------------------------------------------ body alignment
def posed_clip(total=40, yaw_deg=0.0, seed=0):
    """A person whose shoulders/hips lie along +x, then rotated about the vertical axis."""
    rng = np.random.default_rng(seed)
    x = np.zeros((total, 2, 25, 3), np.float32)
    pose = rng.normal(0, 0.1, (25, 3)).astype(np.float32)
    pose[4], pose[8], pose[12], pose[16] = (-0.2, 0.4, 0), (0.2, 0.4, 0), (-0.1, 0, 0), (0.1, 0, 0)
    pose[0] = 0
    t = np.arange(total)[:, None, None]
    x[:, 0] = pose + 0.03 * np.sin(0.2 * t)
    th = np.deg2rad(yaw_deg)
    r = np.asarray([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]], np.float32)
    x = x @ r.T + np.array([0.3, 0.0, 3.0], np.float32)
    return np.where(r4pp.raw_valid(x)[..., None], x, 0).astype(np.float32)


def test_alignment_makes_features_invariant_to_yaw():
    base = pp.features(posed_clip(yaw_deg=0), body_align="yaw")
    for yaw in (-60, -25, 17, 45, 90, 170):
        np.testing.assert_allclose(pp.features(posed_clip(yaw_deg=yaw), body_align="yaw"), base, atol=2e-4)
    assert not np.allclose(pp.features(posed_clip(yaw_deg=60)), pp.features(posed_clip(yaw_deg=0)), atol=1e-3)


def test_alignment_keeps_zeros_gaps_and_the_second_actor():
    x = posed_clip(40, 30)
    x[5:9, 0, 7] = 0
    x[:, 1] = x[:, 0] + 0.5
    y = pp.align_to_body(x)
    assert np.abs(y[5:9, 0, 7]).max() == 0 and np.isfinite(y).all()
    np.testing.assert_allclose(np.linalg.norm(y[:, 1, 5] - y[:, 0, 5], axis=-1),
                               np.linalg.norm(x[:, 1, 5] - x[:, 0, 5], axis=-1), atol=1e-4)
    z = pp.align_to_body(np.zeros((10, 2, 25, 3), np.float32))
    assert np.abs(z).max() == 0
    assert pp.align_to_body(np.zeros((0, 2, 25, 3), np.float32)).shape == (0, 2, 25, 3)


def test_alignment_leaves_clips_without_torso_unchanged():
    x = posed_clip(20, 30)
    x[:, 0, 4] = 0                                           # no left shoulder anywhere
    np.testing.assert_array_equal(pp.align_to_body(x), x)


def test_alignment_through_the_augmentation_paths_is_finite_and_shaped():
    x = posed_clip(60, 40)
    for fn in (lambda: pp.augmented_features(x, 0, 1, 2, body_align="yaw"),
               lambda: pp.strong_augmented_features(x, 0, 1, 2, 1.0, view_degrees=5.0, body_align="yaw")):
        out = fn()
        assert out.shape == (16, 942) and np.isfinite(out).all()
    with pytest.raises(ValueError):
        pp.features(x, body_align="roll")
    # small rotation aug around the aligned pose: far smaller change than the same aug without alignment
    a = pp.augmented_features(posed_clip(60, 0), 0, 1, 2, 8.0, body_align="yaw")
    b = pp.augmented_features(posed_clip(60, 90), 0, 1, 2, 8.0, body_align="yaw")
    np.testing.assert_allclose(a, b, atol=2e-4)


def test_aligned_cache_roundtrip_and_config(tmp_path):
    synthetic_pickle(tmp_path / "ntu120_3danno.pkl")
    prepare(tmp_path / "ntu120_3danno.pkl", tmp_path / "r4", tmp_path / "s.json")
    meta = r5data.build(tmp_path / "r4", tmp_path / "al", body_align="yaw")
    assert meta["signature"]["body_align"] == "yaw" and r5data.FULL_FILE in meta["files"]
    d = r5data.Dataset(tmp_path / "al")
    assert d.body_align == "yaw"
    got = d.canonical([0, 1])
    want = np.stack([pp.features(d.base.sample(i), body_align="yaw") for i in (0, 1)])
    np.testing.assert_allclose(got, want, atol=1e-6)
    with pytest.raises(ValueError):
        r5data.build(tmp_path / "r4", tmp_path / "al", body_align="none")      # other signature, same dir
    assert validate_config({"body_align": "yaw"})["body_align"] == "yaw"
    with pytest.raises(ValueError):
        validate_config({"body_align": "pitch"})
    cfg = validate_config({"micro_batch": 2, "accumulation_steps": 2, "aug_strength": 1.0, "aug_view_degrees": 5,
                           "body_align": "yaw"})
    for batch, _ in d.batches(d.splits["xsub_train"], 4, cfg, epoch=1, training=True, protocol="xsub"):
        assert np.isfinite(batch["x"]).all() and np.isfinite(batch["xa"]).all()
        break

import json

import numpy as np
import pytest

from experiments.nestsar_r5_t16 import augpool
from experiments.nestsar_r5_t16 import data as r5data
from experiments.nestsar_r5_t16 import preprocessing as pp
from experiments.nestsar_r5_t16.config import validate_config
from experiments.nestsar_r5_t16.smoke_cpu import synthetic_pickle
from experiments.nestsar_sm_all_t16.streaming.data import prepare


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("augpool")
    synthetic_pickle(root / "ntu120_3danno.pkl")
    prepare(root / "ntu120_3danno.pkl", root / "r4", root / "s.json")
    r5data.build(root / "r4", root / "hand")
    ds = r5data.Dataset(root / "hand")
    augpool.build(root / "hand", root / "pool", views=3, strength=1.0, workers=2, chunk=7, log=lambda m: None)
    return root, ds


def test_pool_views_equal_the_live_strong_augmentation(env):
    root, ds = env
    pool = augpool.Pool(root / "pool", ds)
    assert pool.views == 3
    for v in range(3):
        for i in (0, 5, len(ds.labels) - 1):
            want = pp.strong_augmented_features(ds.base.sample(i), augpool.POOL_SEED, v + 1, i, 1.0, stream=0,
                                                shift=1, hand_filter=ds.hand_filter, view_degrees=15.0)
            np.testing.assert_array_equal(pool.get(v, [i])[0], want)
    assert not np.array_equal(pool.get(0, [3]), pool.get(1, [3]))


def test_batches_draw_two_different_pool_views(env):
    root, ds = env
    ids = ds.splits["xsub_train"]
    cfg = validate_config({"micro_batch": 2, "accumulation_steps": 2, "aug_strength": 1.0, "aug_clean_prob": 0.0,
                           "aug_pool": str(root / "pool")})
    pool = ds.pool(str(root / "pool"))
    got = list(ds.batches(ids, 4, cfg, epoch=2, training=True, protocol="xsub"))
    again = list(ds.batches(ids, 4, cfg, epoch=2, training=True, protocol="xsub"))
    seen = 0
    for (b, _), (a, _) in zip(got, again):
        n = int(b["mask"].sum())
        np.testing.assert_array_equal(b["x"], a["x"])
        np.testing.assert_array_equal(b["xa"], a["xa"])
        assert not np.array_equal(b["x"][:n], b["xa"][:n])
        assert np.abs(b["x"][n:]).max(initial=0) == 0
        seen += n
    assert seen == len(ids)
    ids = np.asarray(ids)
    pos = np.random.default_rng(cfg["seed"] + 2).permutation(len(ids))[:4]     # the first batch's clips
    b = got[0][0]
    for j in range(int(b["mask"].sum())):
        assert any(np.array_equal(b["xa"][j], pool.arrays[v][ids[pos[j]]]) for v in range(3)), "xa is not a pooled view"
        assert any(np.array_equal(b["x"][j], pool.arrays[v][ids[pos[j]]]) for v in range(3)), "x is not a pooled view"


def test_clean_prob_one_keeps_canonical_x_and_pool_is_validated(env, tmp_path):
    root, ds = env
    ids = np.asarray(ds.splits["xsub_train"])
    cfg = validate_config({"micro_batch": 2, "accumulation_steps": 2, "aug_strength": 1.0, "aug_clean_prob": 1.0,
                           "aug_pool": str(root / "pool")})
    (b, _) = next(iter(ds.batches(ids[:4], 4, cfg, epoch=1, training=False, protocol="xsub")))
    assert "xa" not in b                                    # evaluation never touches the pool
    (b, _) = next(iter(ds.batches(ids, 4, cfg, epoch=1, training=True, protocol="xsub")))
    n = int(b["mask"].sum())
    pos = np.random.default_rng(cfg["seed"] + 1).permutation(len(ids))[:4]
    np.testing.assert_array_equal(b["x"][:n], ds.canonical(ids[pos][:n]))
    with pytest.raises(ValueError):
        augpool.build(root / "hand", root / "pool", views=4, workers=1, log=lambda m: None)   # other signature
    (tmp_path / augpool.MANIFEST).write_text(json.dumps({"complete": False, "signature": {}}))
    with pytest.raises(ValueError):
        augpool.Pool(tmp_path, ds)
    with pytest.raises(ValueError):
        validate_config({"aug_pool": "/x"})                  # needs aug_strength > 0


def test_float16_pool_split_in_two_parts_is_found_and_joined(env, tmp_path):
    """A pool saved as two Kaggle outputs (views 0-1 and 2, float16) is discovered and read as one pool."""
    root, ds = env
    kw = dict(views=3, strength=1.0, workers=2, chunk=7, log=lambda m: None, dtype="float16")
    augpool.build(root / "hand", tmp_path / "input" / "out_a" / "pool", only_views=[0, 1], **kw)
    augpool.build(root / "hand", tmp_path / "input" / "out_b" / "pool", only_views=[2], **kw)
    sig = augpool.signature(r5data.read_manifest(root / "hand"), 3, 1.0, 15.0, augpool.POOL_SEED, "float16")
    found = augpool.find_parts([tmp_path / "input"], sig)
    assert sorted(found) == [0, 1, 2] and found[0] != found[2]
    spec = ":".join(sorted({str(d) for d in found.values()}))
    pool = augpool.Pool(spec, ds)
    full = augpool.Pool(root / "pool", ds)                                   # float32 reference pool
    for v in range(3):
        a, b = pool.get(v, [0, 4]), full.get(v, [0, 4])
        assert a.dtype == np.float32
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-3)              # float16 rounding only
    assert augpool.find_parts([tmp_path / "input"], augpool.signature(
        r5data.read_manifest(root / "hand"), 3, 1.0, 15.0, augpool.POOL_SEED)) == {}   # float32 signature differs
    with pytest.raises(ValueError, match="missing"):
        augpool.Pool(str(found[0]), ds)                                       # one part alone is incomplete
    assert augpool.parse_views("0-3", 8) == [0, 1, 2, 3] and augpool.parse_views("", 2) == [0, 1]


def test_ensure_aug_pool_reuses_saved_parts(env, tmp_path, monkeypatch):
    from experiments.nestsar_r5_t16 import kaggle_run
    root, ds = env
    kw = dict(views=3, strength=1.0, workers=2, chunk=7, log=lambda m: None, dtype="float16")
    augpool.build(root / "hand", tmp_path / "input" / "a", only_views=[0, 1, 2], **kw)
    monkeypatch.setattr(kaggle_run.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rebuilt")))
    spec = kaggle_run.ensure_aug_pool(str(root / "hand"), str(tmp_path / "tmp_pool"), 3, 1.0, 15.0, "float16",
                                      "", [str(tmp_path / "input")])
    assert spec == str(tmp_path / "input" / "a")

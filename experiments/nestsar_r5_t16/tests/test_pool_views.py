"""aug_pool_views_use: draw the pooled views from the first K views only (view-count ablation).
0 (default) keeps the draws exactly as before."""
import numpy as np
import pytest

from experiments.nestsar_r5_t16 import data
from experiments.nestsar_r5_t16 import preprocessing as pp
from experiments.nestsar_r5_t16.config import DEFAULTS, validate_config


class _Pool:
    views = 6

    def get(self, view, indices):
        return np.full((len(indices), pp.FRAMES, pp.FEATURES), float(view), np.float32)


class _Self:
    def canonical(self, indices):
        return np.full((len(indices), pp.FRAMES, pp.FEATURES), -1.0, np.float32)


def _draw(use, n=512, clean_prob=0.0):
    b = {"x": np.zeros((n, pp.FRAMES, pp.FEATURES), np.float32), "xa": np.zeros((n, pp.FRAMES, pp.FEATURES), np.float32)}
    config = {"aug_clean_prob": clean_prob}
    if use is not None:
        config["aug_pool_views_use"] = use
    data.Dataset._batch_from_pool(_Self(), b, _Pool(), np.arange(n), np.arange(n), 7, 3, config)
    return np.unique(b["xa"][:, 0, 0]), np.unique(b["x"][:, 0, 0])


def test_default_uses_every_view():
    xa, x = _draw(None)
    assert set(xa) == set(range(6)) and set(x) == set(range(6))
    assert _draw(0)[0].tolist() == xa.tolist()


@pytest.mark.parametrize("use", [2, 3])
def test_first_k_views_only(use):
    xa, x = _draw(use)
    assert set(xa) == set(range(use)) and set(x) == set(range(use))


def test_one_view_and_oversized_request():
    xa, x = _draw(1)
    assert xa.tolist() == [0.0] and x.tolist() == [0.0]
    assert set(_draw(99)[0]) == set(range(6))              # more than the pool has = all of it


def test_views_use_is_validated():
    assert DEFAULTS["aug_pool_views_use"] == 0
    assert validate_config({"aug_pool_views_use": 3})["aug_pool_views_use"] == 3
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            validate_config({"aug_pool_views_use": bad})

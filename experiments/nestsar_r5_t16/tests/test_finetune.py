import numpy as np

from experiments.nestsar_r5_t16 import finetune as ft


def test_epoch_ids_mix_weak_and_replay():
    labels = np.repeat(np.arange(10), 100)
    train = np.arange(1000)
    ids = ft.epoch_ids(train, labels, weak=[2, 7], fraction=0.4, rng=np.random.default_rng(0))
    assert len(ids) == 1000
    share = np.isin(labels[ids], [2, 7]).mean()
    assert 0.47 < share < 0.57                         # 40% forced + ~20% of the 60% replay = 52%
    assert set(ft.epoch_ids(train, labels, [], 0.4, np.random.default_rng(0))) == set(train)


def test_summary_separates_weak_gain_from_forgetting():
    base = np.full(ft.NUM_CLASSES, 0.8); base[[0, 1]] = 0.3
    now = base.copy(); now[[0, 1]] = 0.5; now[5] = 0.7; now[6] = 0.9
    support = np.ones(ft.NUM_CLASSES, int)
    s = ft.summarize(now, base, [0, 1], support)
    assert abs(s["weak_delta_pp"] - 20.0) < 1e-6
    assert abs(s["forgetting_pp"] - 10.0 / 118) < 1e-6        # only class 5 lost recall (10 pp / 118 others)
    assert s["classes_worse_by_2pp"] == 1
    assert ft.summarize(now, base, [0], np.zeros(ft.NUM_CLASSES, int))["weak_mean_recall"] is None

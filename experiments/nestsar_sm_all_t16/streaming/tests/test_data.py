import json
import pickle
from contextlib import closing
import numpy as np
import pytest

from experiments.nestsar_sm_all_t16 import preprocessing_corrected as pp
from experiments.nestsar_sm_all_t16.streaming.data import Dataset, prepare, resolve_splits
from experiments.nestsar_sm_all_t16.streaming.launch import validate_config


@pytest.fixture
def cached(tmp_path):
    annotations = []
    for i in range(5):
        x = np.zeros((2, 16+i, 25, 3), np.float32)
        x[0] = np.random.default_rng(i).normal(0, .1, x[0].shape) + [1, 2, 3]
        x[0, :, :, 0] += np.arange(16+i)[:, None] * .05
        if i == 3:
            x[1] = x[0] + 1
        annotations.append(dict(frame_dir=f'id{i}', keypoint=x, label=i))
    splits = dict(xsub_train=['id0', 'id1', 'id2'], xsub_val=['id3', 'id4'],
                  xset_train=['id1', 'id3', 'id4'], xset_val=['id0', 'id2'])
    path = tmp_path / 'tiny.pkl'
    path.write_bytes(pickle.dumps(dict(annotations=annotations, split=splits)))
    cache = tmp_path / 'cache'
    prepare(path, cache, tmp_path / 'status.json')
    return path, cache, annotations, splits


def test_two_readers_share_read_only_raw_and_canonical_files(cached, monkeypatch):
    _, cache, annotations, _ = cached
    monkeypatch.setattr(pickle, 'load', lambda *a, **k: pytest.fail('Workers must not unpickle annotations'))
    a, b = Dataset(cache), Dataset(cache)
    for key in ('raw', 'canonical'):
        first, second = getattr(a, key), getattr(b, key)
        assert isinstance(first, np.memmap) and isinstance(second, np.memmap)
        assert first.filename == second.filename
        assert not first.flags.writeable and not second.flags.writeable
    for i, annotation in enumerate(annotations):
        raw = pp.ordered_raw(annotation['keypoint'])
        np.testing.assert_array_equal(a.sample(i), raw)
        np.testing.assert_array_equal(a.canonical[i], pp.features(raw))
    assert a.meta['canonical_bytes'] == 5 * 16 * 750 * 4


def test_validation_uses_cached_tokens_without_raw_or_preprocessing(cached, monkeypatch):
    _, cache, _, _ = cached
    ds = Dataset(cache)
    expected = ds.canonical[[3, 4]].copy()
    def forbidden(*a, **kw):
        pytest.fail('Validation must not rebuild tokens or read raw skeletons')
    monkeypatch.setattr(ds, 'sample', forbidden)
    monkeypatch.setattr(pp, 'features', forbidden)
    with closing(ds.batches([3, 4], 4, validate_config({}))) as iterator:
        batch, _ = next(iterator)
    np.testing.assert_array_equal(batch['x'][:2], expected)
    assert batch['mask'].tolist() == [1, 1, 0, 0]
    assert not batch['x'][2:].any()


@pytest.mark.parametrize('protocol', ['xsub', 'xset'])
def test_fresh_batches_rebuild_only_one_view_and_keep_tail(cached, monkeypatch, protocol):
    _, cache, _, _ = cached
    ds = Dataset(cache)
    config = validate_config(dict(micro_batch=2, accumulation_steps=1))
    ids = ds.splits[f'{protocol}_train']
    expected_positions = np.random.default_rng(130).permutation(len(ids))
    original = pp.features
    calls = []
    def counted(*a, **kw):
        calls.append(1)
        return original(*a, **kw)
    monkeypatch.setattr(pp, 'features', counted)
    batches = list(ds.batches(ids, 2, config, epoch=2, training=True, protocol=protocol))
    assert len(calls) == len(ids)  # Exactly one augmented feature build per real sample.
    assert sum(b['mask'].sum() for b, _ in batches) == len(ids)
    joined = np.concatenate([b['xa'][b['mask'] > 0] for b, _ in batches])
    monkeypatch.setattr(pp, 'features', original)
    for row, pos in zip(joined, expected_positions):
        _, expected = pp.training_views(ds.sample(ids[pos]), 128 + (100000 if protocol == 'xset' else 0), 2, int(pos))
        np.testing.assert_array_equal(row, expected)
    replay = list(ds.batches(ids, 2, config, epoch=2, training=True, protocol=protocol))
    next_epoch = list(ds.batches(ids, 2, config, epoch=3, training=True, protocol=protocol))
    np.testing.assert_array_equal(batches[0][0]['xa'], replay[0][0]['xa'])
    assert not np.array_equal(batches[0][0]['xa'], next_epoch[0][0]['xa'])


def test_cache_reuse_and_corruption_detection(cached, tmp_path):
    path, cache, _, _ = cached
    stamp = (cache/'canonical.npy').stat().st_mtime_ns
    prepare(path, cache, tmp_path/'status.json')
    assert stamp == (cache/'canonical.npy').stat().st_mtime_ns
    with (cache/'canonical.npy').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='Incomplete cache'):
        Dataset(cache)


def test_split_errors_are_not_silently_dropped(cached):
    _, _, annotations, splits = cached
    bad = dict(splits, xsub_val=['missing'])
    with pytest.raises(ValueError, match='absent'):
        resolve_splits(annotations, bad)
    bad = dict(splits, xsub_val=['id0'])
    with pytest.raises(ValueError, match='leakage'):
        resolve_splits(annotations, bad)
    with pytest.raises(ValueError, match='Duplicate annotation'):
        resolve_splits(annotations + annotations[:1], splits)


def test_prefetch_worker_exceptions_propagate(cached, monkeypatch):
    _, cache, _, _ = cached
    ds = Dataset(cache)
    def fail(*a, **kw):
        raise RuntimeError('augmentation failed')
    monkeypatch.setattr(ds, 'batch', fail)
    with pytest.raises(RuntimeError, match='augmentation failed'):
        list(ds.batches([0, 1], 2, validate_config({})))


def test_cache_builds_canonical_once_per_annotation(cached, tmp_path, monkeypatch):
    path, _, annotations, _ = cached
    original, calls = pp.features, []
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(pp, 'features', count)
    prepare(path, tmp_path/'new_cache', tmp_path/'new_status.json')
    assert len(calls) == len(annotations)

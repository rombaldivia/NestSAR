import json
from contextlib import closing

import numpy as np
import pytest

from experiments.nestsar_sm_all_t16.streaming.tests.test_data import cached
from experiments.nestsar_sm_all_t16.streaming import launch
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json
from experiments.nestsar_sm_all_t16.streaming.sampler_b_data import prepare_sampler_b, attach_sampler_b
from experiments.nestsar_sm_all_t16 import preprocessing_corrected as pp
from experiments.nestsar_sm_all_t16.sampler_b import VERSION, ScaleAccumulator, relative_motion


def test_full_launcher_rejects_internal_tiny_cache(cached, tmp_path):
    _, cache, _, _ = cached
    with pytest.raises(RuntimeError, match="subset/internal"):
        prepare_sampler_b(cache, tmp_path/'b', tmp_path/'status.json')


def test_each_protocol_calibrates_only_its_complete_training_members(cached, tmp_path):
    _, cache, _, _ = cached
    out = tmp_path/'b'
    prepare_sampler_b(cache, out, tmp_path/'status.json', require_full=False)
    ds = Dataset(cache)
    for protocol in ('xsub', 'xset'):
        a = ScaleAccumulator()
        for index in sorted(ds.splits[f'{protocol}_train']):
            x = ds.sample(index)
            a.observe(*relative_motion(x, pp.raw_valid(x)))
        actual = json.loads((out/f'calibration_{protocol}.json').read_text())
        for key, value in a.finish().items():
            assert actual[key] == value
        assert actual['fit_split'] == f'{protocol}_train'
        assert actual['samples_observed'] == len(ds.splits[f'{protocol}_train'])


def test_overlay_and_augmented_batches_match_direct_sampler_and_keep_all_samples(cached, tmp_path, monkeypatch):
    _, cache, _, _ = cached
    out = tmp_path/'b'
    prepare_sampler_b(cache, out, tmp_path/'status.json', require_full=False)
    c = launch.validate_config(dict(pose_sampler=VERSION, micro_batch=2, accumulation_steps=1))
    for protocol in ('xsub', 'xset'):
        ds = Dataset(cache)
        identity = attach_sampler_b(ds, cache, out, protocol)
        assert identity['calibration']['protocol'] == protocol
        assert isinstance(ds.pose_overlay, np.memmap) and not ds.pose_overlay.flags.writeable
        original_sample = ds.sample
        for part in ('train', 'val'):
            ids = ds.splits[f'{protocol}_{part}']
            batches = list(ds.batches(ids, 2, c, epoch=2, training=part=='train', protocol=protocol))
            assert sum(b['mask'].sum() for b, _ in batches) == len(ids)
            positions = (np.random.default_rng(c['seed']+2).permutation(len(ids))
                         if part == 'train' else np.arange(len(ids)))
            joined = np.concatenate([b['x'][b['mask'] > 0] for b, _ in batches])
            for row, position in zip(joined, positions):
                expected = pp.features(ds.sample(ids[position]), pose_selector=ds.pose_sampler)
                np.testing.assert_array_equal(row, expected)
                np.testing.assert_array_equal(row.reshape(16,2,25,15)[...,3:],
                                              ds.canonical[ids[position]].reshape(16,2,25,15)[...,3:])
            if part == 'train':
                augmented = np.concatenate([b['xa'][b['mask'] > 0] for b, _ in batches])
                for row, pos in zip(augmented, positions):
                    expected = pp.augmented_features(ds.sample(ids[pos]),
                        c['seed'] + (100000 if protocol == 'xset' else 0), 2, int(pos),
                        c['rotation_degrees'], c['jitter_shift'], pose_selector=ds.pose_sampler)
                    np.testing.assert_array_equal(row, expected)
        def no_raw(*a, **kw):
            pytest.fail('Validation must not reopen raw or rebuild motion features')
        monkeypatch.setattr(ds, 'sample', no_raw)
        with closing(ds.batches(ds.splits[f'{protocol}_val'], 2, c, protocol=protocol)) as batches:
            next(batches)
        monkeypatch.setattr(ds, 'sample', original_sample)
        with pytest.raises(ValueError, match='protocol mismatch'):
            ds.batch([0], [0], 1, c, 1, False, 'xset' if protocol == 'xsub' else 'xsub')


def test_reuse_does_not_refit_or_rebuild_and_detects_calibration_changes(cached, tmp_path, monkeypatch):
    _, cache, _, _ = cached
    out = tmp_path/'b'
    prepare_sampler_b(cache, out, tmp_path/'status.json', require_full=False)
    before = (out/'pose_xsub.npy').stat().st_mtime_ns
    def forbidden(*a, **k):
        pytest.fail('Completed B cache must be reused')
    monkeypatch.setattr(Dataset, 'sample', forbidden)
    prepare_sampler_b(cache, out, tmp_path/'status.json', require_full=False)
    assert (out/'pose_xsub.npy').stat().st_mtime_ns == before
    path = out/'calibration_xsub.json'
    data = json.loads(path.read_text())
    data['scale'][0] *= 2
    atomic_json(path, data)
    with pytest.raises(ValueError, match='calibration|Incomplete'):
        attach_sampler_b(Dataset(cache), cache, out, 'xsub')


def test_b_refuses_subset_caps_and_preserves_original_default_config():
    assert 'pose_sampler' not in launch.validate_config({})
    for key in ('max_train_samples', 'max_val_samples'):
        with pytest.raises(ValueError, match='subset caps'):
            launch.validate_config({key: 100, 'pose_sampler': VERSION})


def test_b_entry_uses_latest_defaults_and_restores_progress_backend(monkeypatch):
    from experiments.nestsar_sm_all_t16.run_sampler_b_dual_t4 import run_sampler_b
    from experiments.nestsar_sm_all_t16.streaming import notebook_progress
    before = launch.make_bars, launch.update_bar
    def run(**kw):
        c = launch.validate_config(kw['config'])
        assert c['pose_sampler'] == VERSION
        for key, value in launch.DEFAULTS.items():
            assert c[key] == value
        assert launch.make_bars is notebook_progress.make_bars
        assert launch.update_bar is notebook_progress.update_bar
        return {p: dict(best_val_accuracy=.5, best_epoch=2) for p in ('xsub','xset')}
    monkeypatch.setattr(launch, 'run', run)
    run_sampler_b(dataset='fixture.pkl')
    assert (launch.make_bars, launch.update_bar) == before

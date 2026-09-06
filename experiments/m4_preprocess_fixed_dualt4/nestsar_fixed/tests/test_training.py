import json
import pickle
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from nestsar_fixed.worker import State, build_steps, stopping_update, save_checkpoint, restore_checkpoint
from nestsar_fixed.data import prepare, Dataset, resolve_splits
from nestsar_fixed.launch import validate_config


class TinyModel(nn.Module):
    @nn.compact
    def __call__(self, x, hand, training=False):
        logits = nn.Dense(120)(x[:, 0])
        return dict(logits=logits, hand_logits=logits, main_logits=logits,
                    stream_logits=jnp.repeat(logits[:, None], 4, axis=1))


def test_gradient_accumulation_and_padded_tail(tmp_path):
    model = TinyModel()
    batch = dict(x=jnp.arange(12, dtype=jnp.float32).reshape(4, 1, 3) / 10,
                 xa=jnp.arange(12, dtype=jnp.float32).reshape(4, 1, 3) / 9,
                 h=jnp.zeros((4, 1, 3)), ha=jnp.zeros((4, 1, 3)),
                 y=jnp.array([0, 1, 2, 0]), mask=jnp.array([1., 1., 1., 0.]))
    params = model.init(jax.random.PRNGKey(0), batch['x'], batch['h'])['params']
    state = State.create(apply_fn=model.apply, params=params, ema_params=params, tx=optax.adam(0.01))
    a, eva = build_steps(model, validate_config(dict(micro_batch=1, accumulation_steps=4)))
    b, _ = build_steps(model, validate_config(dict(micro_batch=4, accumulation_steps=1)))
    first, key, metrics = a(state, jax.random.PRNGKey(1), batch)
    second, _, expected = b(state, jax.random.PRNGKey(1), batch)
    for x, y in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)
    assert int(first.step) == 1 and float(metrics[6]) == 3
    assert float(eva(first.ema_params, {k: batch[k] for k in ('x', 'h', 'y', 'mask')})[-1]) == 3
    path = tmp_path / 'last.msgpack'
    save_checkpoint(path, first, key, {'config_hash': 'valid', 'epoch': 1})
    restored, restored_key, metadata = restore_checkpoint(path, state, 'valid')
    for x, y in zip(jax.tree.leaves(first), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(key, restored_key)
    with pytest.raises(ValueError, match='mismatch'):
        restore_checkpoint(path, state, 'wrong')


def test_patience_starts_after_warmup():
    best, bad = -1, 0
    for epoch in range(1, 11):
        best, bad, _ = stopping_update(best, bad, 0.5, epoch, 5, 0)
        assert bad == max(0, epoch - 5)
    assert bad == 5
    assert stopping_update(best, bad, .6, 11, 5, 0) == (.6, 0, True)


def test_cache_split_integrity_and_raw_replay(tmp_path):
    anns = []
    for i in range(4):
        x = np.ones((1, 17+i, 25, 3), np.float32)
        x[0, :, :, 0] += np.arange(17+i)[:, None] * 0.01
        x[0, :, :, 1] += np.arange(25)[None] * .1
        anns.append(dict(frame_dir=f'id{i}', keypoint=x, label=i))
    split = dict(xsub_train=['id0', 'id1'], xsub_val=['id2', 'id3'],
                 xset_train=['id0', 'id2'], xset_val=['id1', 'id3'])
    file = tmp_path / 'tiny.pkl'
    file.write_bytes(pickle.dumps(dict(annotations=anns, split=split)))
    cache = tmp_path / 'cache'
    prepare(file, cache, tmp_path / 'status.json')
    ds = Dataset(cache)
    assert ds.raw.dtype == np.float32
    for i in range(4):
        np.testing.assert_array_equal(ds.sample(i)[:, 0], anns[i]['keypoint'][0])
        assert not ds.sample(i)[:, 1].any()
    config = validate_config(dict(micro_batch=2))
    batch, _ = next(ds.batches(ds.splits['xsub_train'], 3, config, 1, True))
    assert batch['mask'].sum() == 2
    assert batch['x'].shape == (3, 16, 750) and batch['h'].shape == (3, 32, 96)
    prepare(file, cache, tmp_path / 'status.json')  # Valid completed cache reuse.
    split['xsub_val'].append('id0')
    with pytest.raises(ValueError, match='leakage'):
        resolve_splits(anns, split)
    split['xsub_val'] = ['does-not-exist']
    with pytest.raises(ValueError, match='absent'):
        resolve_splits(anns, split)

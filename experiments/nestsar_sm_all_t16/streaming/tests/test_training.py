import numpy as np
import pytest
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from experiments.nestsar_sm_all_t16.streaming.worker import (
    State, build_steps, stopping_update, save_checkpoint, restore_checkpoint, publish_best)
from experiments.nestsar_sm_all_t16.streaming.launch import validate_config


class TinySM(nn.Module):
    @nn.compact
    def __call__(self, x, training=False):
        logits = nn.Dense(120)(x[:, 0])
        return dict(logits=logits, main_logits=logits,
                    sm_eta_mean=jnp.ones(len(x))*.2, sm_alpha_mean=jnp.ones(len(x))*.99,
                    stream_logits=jnp.repeat(logits[:, None], 4, axis=1))


def test_accumulation_matches_real_samples_only_and_checkpoint_resume(tmp_path):
    model = TinySM()
    batch = dict(x=jnp.arange(12, dtype=jnp.float32).reshape(4, 1, 3)/10,
                 xa=jnp.arange(12, dtype=jnp.float32).reshape(4, 1, 3)/9,
                 y=jnp.array([0, 1, 2, 0]), mask=jnp.array([1., 1., 1., 0.]))
    params = model.init(jax.random.PRNGKey(0), batch['x'])['params']
    state = State.create(apply_fn=model.apply, params=params, ema_params=params, tx=optax.adam(.01))
    a, evaluate = build_steps(model, validate_config(dict(micro_batch=1, accumulation_steps=4)))
    b, _ = build_steps(model, validate_config(dict(micro_batch=3, accumulation_steps=1)))
    first, key, metrics = a(state, jax.random.PRNGKey(1), batch)
    real = {k: v[:3] for k, v in batch.items()}
    second, _, expected = b(state, jax.random.PRNGKey(1), real)
    for x, y in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(metrics[:10], expected[:10], rtol=1e-5, atol=1e-6)
    assert int(first.step) == 1 and float(metrics[9]) == 3
    assert float(evaluate(first.ema_params, {k: batch[k] for k in ('x', 'y', 'mask')})[-1]) == 3
    path = tmp_path/'last.msgpack'
    save_checkpoint(path, first, key, {'config_hash': 'valid', 'epoch': 1})
    restored, restored_key, metadata = restore_checkpoint(path, state, 'valid')
    for x, y in zip(jax.tree.leaves(first), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(key, restored_key)
    with pytest.raises(ValueError, match='mismatch'):
        restore_checkpoint(path, state, 'wrong')


def test_patience_five_starts_after_warmup():
    best, bad = -1, 0
    for epoch in range(1, 11):
        best, bad, _ = stopping_update(best, bad, .5, epoch, 5, 1e-6)
        assert bad == max(0, epoch-5)
    assert bad == 5
    assert stopping_update(best, bad, .6, 11, 5, 1e-6) == (.6, 0, True)


def test_zero_weight_padding_does_not_poison_real_fast_weight_gradients():
    from experiments.nestsar_sm_all_t16.model import FastWeightDeltaResidual
    from experiments.nestsar_sm_all_t16.streaming.worker import safe_training_padding
    model = FastWeightDeltaResidual(dim=8, rank=2)
    real = jax.random.normal(jax.random.key(4), (1, 4, 8))
    padded = jnp.concatenate([real, jnp.zeros_like(real)])
    batch = dict(x=padded, xa=padded, mask=jnp.array([1., 0.]), y=jnp.array([1, 0]))
    safe = safe_training_padding(batch)
    np.testing.assert_array_equal(safe['x'][0], real[0])
    np.testing.assert_array_equal(safe['mask'], batch['mask'])
    eta, alpha = jnp.ones((2,4,1))*.1, jnp.ones((2,4,1))*.99
    params = model.init(jax.random.key(3), padded, eta, alpha)['params']
    def masked_loss(p, x):
        return (model.apply({'params':p}, x, eta, alpha) * batch['mask'][:,None,None]).sum()
    safe_grad = jax.jit(jax.grad(masked_loss))(params, safe['x'])
    real_grad = jax.jit(jax.grad(lambda p: model.apply({'params':p}, real, eta[:1], alpha[:1]).sum()))(params)
    for a, b in zip(jax.tree.leaves(safe_grad), jax.tree.leaves(real_grad)):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=1e-6)


def test_best_alias_recovered_from_committed_checkpoint_reference(tmp_path):
    metadata = dict(best_checkpoint='best_epoch_0004.msgpack', best_epoch=4, best=.76, config_hash='run')
    (tmp_path/'best_epoch_0004.msgpack').write_bytes(b'committed best')
    (tmp_path/'best.msgpack').write_bytes(b'stale alias from interrupted write')
    publish_best(tmp_path, metadata)
    assert (tmp_path/'best.msgpack').read_bytes() == b'committed best'


def test_sampler_calibration_can_be_restored_from_best_checkpoint():
    from flax import serialization
    from experiments.nestsar_sm_all_t16.sampler_b import SamplerB
    from experiments.nestsar_sm_all_t16.test_sampler_b import calibration, clip
    from experiments.nestsar_sm_all_t16 import preprocessing_corrected as pp
    c = calibration()
    restored = serialization.msgpack_restore(serialization.to_bytes({'sampler_b': {'calibration': c}}))
    sampler = SamplerB(restored['sampler_b']['calibration'])
    x = clip(100)
    x[:,0,21,0] += np.sin(np.arange(100))*.03
    np.testing.assert_array_equal(pp.features(x, pose_selector=sampler),
                                  pp.features(x, pose_selector=SamplerB(c)))


def test_checkpoint_audit_uses_distinct_corrected_and_legacy_inputs():
    from experiments.nestsar_sm_all_t16.audit_checkpoint import iter_validation_batches
    from experiments.nestsar_sm_all_t16.test_preprocessing_corrected import clip
    from experiments.nestsar_sm_all_t16 import preprocessing_corrected as pp
    x = clip(32)
    by_id = {'sample': dict(keypoint=x.transpose(1,0,2,3), label=2)}
    sm, baseline, hand, labels, mask, count = next(iter_validation_batches(by_id, ['sample'], 2))
    np.testing.assert_array_equal(sm[0], pp.features(x))
    assert not sm[0].reshape(16,2,25,15)[:,1].any()
    assert baseline[0].reshape(16,2,25,15)[:,1].any()
    assert count == 1 and mask.tolist() == [1, 0]
    assert not sm[1].any() and not baseline[1].any()

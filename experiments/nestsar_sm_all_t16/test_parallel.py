"""Regression checks for zero padding, associative scans, and worker isolation."""
import importlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.nestsar_sm_all_t16 import model as r4
from experiments.nestsar_sm_all_t16 import model_parallel as par
from experiments.nestsar_sm_all_t16 import audit_parallel as audit


def assert_close(a, b, rtol=3e-4, atol=3e-5):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol, equal_nan=False)


def test_normalization_preserves_forward_and_has_finite_zero_gradient():
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 4))
    x = x.at[0].set(0).at[1].set(1e-9)
    old = x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-6)
    np.testing.assert_allclose(r4.safe_unit_normalize(x), old, rtol=1e-6, atol=1e-7)
    grad = jax.grad(lambda z: r4.safe_unit_normalize(z).sum())(x)
    assert np.isfinite(grad).all()


@pytest.mark.parametrize('length,reverse', [(1, False), (4, True), (16, False), (25, True)])
def test_affine_sweep_values_and_gradients_match_serial(length, reverse):
    x = jax.random.normal(jax.random.PRNGKey(2), (2, length, 7))
    parallel = par.ParallelAffineSweep(7, reverse=reverse)
    serial = par.ParallelAffineSweep(7, reverse=reverse, parallel=False)
    p = parallel.init(jax.random.PRNGKey(3), x)['params']
    assert sum(a.size for a in jax.tree.leaves(p)) == 6*7*7+3*7
    def loss(p, x, model):
        y = model.apply({'params': p}, x)
        return jnp.sum(y*y), y
    f = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
    assert_close(f(p, x, parallel), f(p, x, serial))


def test_fast_memory_values_gradients_and_zero_inputs():
    x = jax.random.normal(jax.random.PRNGKey(4), (2, 16, 7))
    eta = jnp.linspace(.01, .2, 32).reshape(2, 16, 1)
    alpha = jnp.linspace(.90, .999, 32).reshape(2, 16, 1)
    original = r4.FastWeightDeltaResidual(7, 4)
    parallel = par.ParallelFastWeightDeltaResidual(7, 4)
    serial = par.ParallelFastWeightDeltaResidual(7, 4, parallel=False)
    p = original.init(jax.random.PRNGKey(5), x, eta, alpha)['params']
    def loss(p, x, e, a, model):
        y = model.apply({'params': p}, x, e, a)
        return jnp.sum(y*y), y
    f = jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True)
    reference = f(p, x, eta, alpha, original)
    for model in (parallel, serial):
        assert_close(reference, f(p, x, eta, alpha, model))
    for model in (original, parallel, serial):
        result = f(p, jnp.zeros_like(x), eta, alpha, model)
        assert all(np.isfinite(a).all() for a in jax.tree.leaves(result))


def test_full_model_graph_and_identical_equation_control():
    result = audit.check_parallel_graph_and_outputs()
    assert result['params'] == 1_831_932
    assert result['scan_count'] == result['while_count'] == 0


def test_real_masked_training_updates_optimizer_and_ema():
    assert audit.check_padded_training()['padded_training_steps'] == 2


def test_import_does_not_mutate_r4_worker_and_resume_is_model_specific():
    from experiments.nestsar_sm_all_t16.streaming import worker
    from experiments.nestsar_sm_all_t16.parallel_config import MODEL_NAME, implementation_identity
    before = (worker.make_model, worker.publish_best, worker.write_result)
    parallel = importlib.import_module('experiments.nestsar_sm_all_t16.streaming.worker_parallel')
    importlib.reload(parallel)
    assert before == (worker.make_model, worker.publish_best, worker.write_result)
    old = worker.run_signature({}, 'xsub', 'same-cache')
    new = worker.run_signature({}, 'xsub', 'same-cache', MODEL_NAME, implementation_identity())
    assert 'model' not in old  # Preserve the old R4 resume signature.
    assert old != new and new['model'] == MODEL_NAME
    other = dict(new, model_identity={'version': 'other'})
    assert other != new


def test_parallel_checkpoint_alias_and_result_keep_identity(tmp_path):
    from experiments.nestsar_sm_all_t16.streaming import worker
    from experiments.nestsar_sm_all_t16.parallel_config import MODEL_NAME, implementation_identity
    meta = dict(model=MODEL_NAME, model_identity=implementation_identity(),
                best_checkpoint='best_epoch_0001.msgpack', best_epoch=1, best=.75,
                config_hash='test', epoch=2, stopped_early=False)
    (tmp_path/meta['best_checkpoint']).write_bytes(b'checkpoint')
    worker.publish_best(tmp_path, meta)
    result = worker.write_result(tmp_path, 'xsub', meta, 'test')
    best = json.loads((tmp_path/'best.json').read_text())
    assert best['model'] == result['model'] == MODEL_NAME
    assert best['model_identity'] == result['model_identity'] == implementation_identity()

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from experiments.nestsar_sm_all_t16.nonlinear_compare.config import validate_config
from experiments.nestsar_sm_all_t16.nonlinear_compare.data import fit_scale, make_batch
from experiments.nestsar_sm_all_t16.nonlinear_compare.models import SkeletonGRU
from experiments.nestsar_sm_all_t16.nonlinear_compare.training import Trainer, train_candidate


def test_sequence_controls_have_identical_parameters_and_finite_masked_output():
    model = SkeletonGRU(width=8, dropout=0)
    params = []
    for frames in (16, 64):
        x = jnp.zeros((2, frames, 2, 25, 8))
        p = model.init(jax.random.PRNGKey(4), x)["params"]
        y = jax.jit(model.apply)({"params": p}, x)
        assert y.shape == (2, 2) and bool(jnp.isfinite(y).all())
        params.append(jax.tree.leaves(p))
    assert sum(v.size for v in params[0]) == sum(v.size for v in params[1])
    for first, second in zip(*params):
        np.testing.assert_array_equal(first, second)


def test_training_padding_and_resume_do_not_change_optimization(tmp_path):
    c = validate_config(dict(epochs=2, warmup_epochs=1, batch_size=4, mlp_width=8,
                             trials=[dict(learning_rate=.01, weight_decay=0, dropout=0)]))
    x = np.zeros((12, 16, 750), np.float32)
    y = np.arange(12, dtype=np.int32) % 2
    x[:, 0, 0] = 2*y-1
    fit, select = np.arange(8), np.arange(8, 12)
    scale = fit_scale(x, fit)
    trainer = Trainer("t16_mlp", c, c["trials"][0])
    state, key = trainer.initialize(x.shape[1:], 4)
    b = make_batch(x, y, [0, 1], 4, scale)
    b2 = {k: v.copy() for k, v in b.items()}
    b2["x"][2:] = 999
    b2["y"][2:] = 1
    trainer.prepare(state, key, jax.device_put(b))
    first = trainer.compiled_step(state, key, jax.device_put(b), jnp.asarray(.01))
    second = trainer.compiled_step(state, key, jax.device_put(b2), jnp.asarray(.01))
    for a, z in zip(jax.tree.leaves(first[0].params), jax.tree.leaves(second[0].params)):
        np.testing.assert_allclose(a, z, rtol=1e-6, atol=1e-6)

    report = lambda **kwargs: None
    full = train_candidate(trainer, x, y, fit, select, scale, 7, tmp_path/"full", report)
    # Stop immediately after epoch 1's atomic last checkpoint is committed.
    def interrupt(**kwargs):
        if "val_acc" in kwargs and kwargs.get("current") == 1 and "best" in kwargs:
            raise KeyboardInterrupt
    try:
        train_candidate(trainer, x, y, fit, select, scale, 7, tmp_path/"interrupted", interrupt)
    except KeyboardInterrupt:
        pass
    assert (tmp_path/"interrupted"/"last.msgpack").exists()
    resumed = train_candidate(trainer, x, y, fit, select, scale, 7, tmp_path/"interrupted", report)
    assert (full["best"], full["best_epoch"], full["epoch"]) == (resumed["best"], resumed["best_epoch"], resumed["epoch"])
    a = serialization.msgpack_restore((tmp_path/"full"/"best.msgpack").read_bytes())
    b = serialization.msgpack_restore((tmp_path/"interrupted"/"best.msgpack").read_bytes())
    for a, b in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(a, b)

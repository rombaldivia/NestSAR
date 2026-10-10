import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.nestsar_r5_t16 import mixing, worker
from experiments.nestsar_r5_t16 import preprocessing as pp
from experiments.nestsar_r5_t16.config import DEFAULTS, validate_config
from experiments.nestsar_r5_t16.model import NUM_CLASSES


def batch(n=8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, pp.FRAMES, pp.FEATURES)).astype(np.float32)
    xa = rng.normal(0, 1, x.shape).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, n).astype(np.int32)
    return jnp.asarray(x), jnp.asarray(xa), jnp.asarray(y), jnp.ones(n, jnp.float32)


def test_prob_zero_and_defaults_leave_the_batch_unchanged():
    assert DEFAULTS["mix_prob"] == 0.0
    x, xa, y, mask = batch()
    mx, mxa, y2, lam = mixing.mix_batch(jax.random.PRNGKey(0), x, xa, y, mask, 0.0)
    np.testing.assert_array_equal(mx, x)
    np.testing.assert_array_equal(mxa, xa)
    np.testing.assert_array_equal(y2, y)
    np.testing.assert_array_equal(lam, 1.0)


def test_mixed_spans_come_from_the_partner_and_labels_follow_the_span():
    x, xa, y, mask = batch(32)
    mx, mxa, y2, lam = mixing.mix_batch(jax.random.PRNGKey(1), x, xa, y, mask, 1.0)
    mx, mxa, lam, y2 = map(np.asarray, (mx, mxa, lam, y2))
    x, xa, y = map(np.asarray, (x, xa, y))
    changed = 0
    for i in range(len(x)):
        same = np.all(mx[i] == x[i], axis=-1)                                  # [frames]
        replaced = (~same).sum()
        assert np.isclose(lam[i], 1 - replaced / pp.FRAMES, atol=1e-6) or replaced == 0
        if replaced:
            changed += 1
            assert 1 <= replaced <= pp.FRAMES - 1
            idx = np.flatnonzero(~same)
            assert np.all(np.diff(idx) == 1)                                   # one contiguous span
            partner = [j for j in range(len(x)) if np.all(mx[i, idx[0]] == x[j, idx[0]])]
            assert partner and y2[i] == y[partner[0]]
            assert np.all(mxa[i, idx] == xa[partner[0], idx])                  # same partner/span in the 2nd view
            assert np.all(mxa[i, ~(~same)] == xa[i, ~(~same)])
    assert changed >= 20
    assert (lam > 0).all() and (lam < 1.0 + 1e-6).all()


def test_padded_rows_are_never_mixed_in_or_out():
    x, xa, y, mask = batch(8)
    mask = mask.at[5:].set(0.0)
    for seed in range(8):
        mx, mxa, y2, lam = mixing.mix_batch(jax.random.PRNGKey(seed), x, xa, y, mask, 1.0)
        np.testing.assert_array_equal(np.asarray(mx)[5:], np.asarray(x)[5:])
        np.testing.assert_array_equal(np.asarray(lam)[5:], 1.0)
        valid = np.asarray(mx)[:5]
        for i in range(5):
            for t in range(pp.FRAMES):
                assert np.any(np.all(valid[i, t] == np.asarray(x)[:5, t], axis=-1))


def test_soft_cross_entropy_matches_the_manual_formula_and_lam_one_is_plain():
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(0, 1, (4, NUM_CLASSES)).astype(np.float32))
    y, y2 = jnp.asarray([1, 2, 3, 4]), jnp.asarray([5, 6, 7, 8])
    lam = jnp.asarray([1.0, 0.75, 0.5, 0.25])
    got = worker.ce(logits, y, 0.1, y2, lam)
    logp = np.asarray(jax.nn.log_softmax(logits))
    for i in range(4):
        t = np.zeros(NUM_CLASSES)
        t[int(y[i])] += float(lam[i])
        t[int(y2[i])] += 1 - float(lam[i])
        t = t * 0.9 + 0.1 / NUM_CLASSES
        assert np.isclose(float(got[i]), -(t * logp[i]).sum(), atol=1e-5)
    np.testing.assert_allclose(worker.ce(logits, y, 0.1, y, jnp.ones(4)), worker.ce(logits, y, 0.1), atol=1e-6)


@pytest.fixture(scope="module")
def tiny():
    config = validate_config({"micro_batch": 2, "accumulation_steps": 2, "epochs": 2})
    model, state, key, _, _ = worker.create_state(config, 2)
    x, xa, y, mask = batch(4, 3)
    return model, state, key, {"x": x, "xa": xa, "y": y, "mask": mask}, config


def test_train_step_runs_with_and_without_mixing(tiny):
    model, state, key, b, config = tiny
    for prob in (0.0, 1.0):
        cfg = dict(config, mix_prob=prob)
        train_step, _, _ = worker.build_steps(model, cfg)
        new_state, new_key, metrics = train_step(state, key, b)
        assert np.isfinite(np.asarray(metrics)).all()
        assert float(np.asarray(metrics)[-1]) > 0                        # gradient norm (step 0 has lr 0)
    assert validate_config({"mix_prob": 0.5})["mix_prob"] == 0.5
    with pytest.raises(ValueError):
        validate_config({"mix_prob": 1.5})


def test_mixing_changes_the_loss_but_prob_zero_matches_the_plain_path(tiny):
    model, state, key, b, config = tiny
    losses = {}
    for prob in (0.0, 1.0):
        train_step, _, _ = worker.build_steps(model, dict(config, mix_prob=prob))
        losses[prob] = float(train_step(state, key, b)[2][0])
    assert losses[0.0] != losses[1.0]
    train_step, _, _ = worker.build_steps(model, config)                       # key absent == prob 0
    assert float(train_step(state, key, b)[2][0]) == losses[0.0]

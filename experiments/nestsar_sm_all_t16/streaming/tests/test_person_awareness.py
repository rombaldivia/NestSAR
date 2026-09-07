import jax
import jax.numpy as jnp

from experiments.nestsar_sm_all_t16 import model as sm


def _token_pair():
    tok = jnp.zeros((1, 1, 2, 25, 15), jnp.float32)
    tok = tok.at[:, :, 0, :, 0:3].set(jnp.asarray([1.0, 2.0, 3.0]))
    tok = tok.at[:, :, 1, :, 0:3].set(jnp.asarray([4.0, 6.0, 8.0]))
    tok = tok.at[:, :, 0, :, 3:6].set(jnp.asarray([0.1, 0.2, 0.3]))
    tok = tok.at[:, :, 1, :, 3:6].set(jnp.asarray([0.5, 0.7, 0.9]))
    return tok


def test_controller_summary_keeps_person_identity_and_explicit_relation():
    tok = _token_pair()
    summary, present, _ = sm.person_aware_controller_summary(tok)
    assert summary.shape == (1, 1, 15)
    assert jnp.allclose(summary[0, 0, 0:3], jnp.asarray([1.0, 2.0, 3.0]))
    assert jnp.allclose(summary[0, 0, 3:6], jnp.asarray([4.0, 6.0, 8.0]))
    assert jnp.allclose(summary[0, 0, 6:9], jnp.asarray([3.0, 4.0, 5.0]))
    assert jnp.allclose(summary[0, 0, 9:12], jnp.asarray([0.4, 0.5, 0.6]))
    assert jnp.allclose(summary[0, 0, 12:15], jnp.asarray([1.0, 1.0, 1.0]))
    assert jnp.all(present == 1)

    swapped = tok[:, :, ::-1]
    swapped_summary, _, _ = sm.person_aware_controller_summary(swapped)
    assert not jnp.allclose(summary, swapped_summary)
    assert jnp.allclose(swapped_summary[0, 0, 6:9], -summary[0, 0, 6:9])


def test_absent_p2_has_zero_p2_and_pair_relation_but_explicit_presence_bit():
    tok = _token_pair().at[:, :, 1].set(0)
    summary, present, _ = sm.person_aware_controller_summary(tok)
    assert jnp.allclose(summary[0, 0, 3:12], 0)
    assert jnp.allclose(summary[0, 0, 12:15], jnp.asarray([1.0, 0.0, 0.0]))
    assert jnp.allclose(present[0, 0], jnp.asarray([1.0, 0.0]))


def test_person_aware_revision_keeps_parameter_budget():
    model = sm.NestSARSMAllT16()
    params = model.init(
        {"params": jax.random.PRNGKey(3), "dropout": jax.random.PRNGKey(4)},
        jnp.zeros((1, 16, 750), jnp.float32),
        training=False,
    )["params"]
    count = sum(x.size for x in jax.tree.leaves(params))
    assert count == 1_826_556

    # Learned identity is still explicit in every stream's spatial encoder.
    for i in range(4):
        pe = params[f"spatial_{i}"]["person_embed"]
        assert pe.shape[2] == 2

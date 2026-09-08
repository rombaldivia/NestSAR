import numpy as np
import jax
import jax.numpy as jnp
from experiments.nestsar_sm_all_t16.model import (
    NestSARSMAllT16, FastWeightDeltaResidual, skeleton_streams, unpack_relative_tokens, base)
from experiments.nestsar_sm_all_t16.relative_motion import preprocessing as pp
from experiments.nestsar_sm_all_t16 import preprocessing_corrected as p2
from .test_preprocessing import ambiguous_pair


def test_parent_map_and_stream_change_is_confined_to_bone_path():
    np.testing.assert_array_equal(pp.PARENTS, base.PARENTS)
    raw = ambiguous_pair()[1]
    tok, path = unpack_relative_tokens(jnp.asarray(pp.features(raw)[None].reshape(1, 16, 2, 25, 15)))
    valid = jnp.any(jnp.abs(tok) > 1e-8, -1)
    valid = valid.at[..., 0].set(jnp.any(valid, -1))
    gain = jnp.linspace(.91, 1.09, 15).reshape(1, 1, 1, 1, 15)
    beta = jnp.linspace(-.049, .049, 15).reshape(1, 1, 1, 1, 15)
    modulated = (tok * gain + beta) * valid[..., None]
    old, _ = skeleton_streams(modulated, valid)
    new, _ = skeleton_streams(modulated, valid, path, gain[..., 12:15])
    for i in range(3):
        np.testing.assert_array_equal(old[i], new[i])
    np.testing.assert_array_equal(old[3][..., :9], new[3][..., :9])
    np.testing.assert_allclose(new[3][..., 9:12], path * gain[..., 12:15], atol=1e-6)
    assert float(new[3][0, :, 0, 7, 9].sum()) > 0
    static, _ = skeleton_streams(modulated, valid, jnp.zeros_like(path), gain[..., 12:15])
    assert not np.asarray(static[3][..., 9:12]).any()  # beta must not invent travel
    assert not np.asarray(new[3][:, :, 1]).any()


def test_same_parameter_tree_initialization_and_legacy_proxy_equivalence():
    raw = ambiguous_pair()[1]
    x, packed = jnp.asarray(p2.features(raw)[None]), jnp.asarray(pp.features(raw)[None])
    key = jax.random.PRNGKey(128)
    old = NestSARSMAllT16()
    params = old.init(key, x, training=False)["params"]
    assert sum(a.size for a in jax.tree.leaves(params)) == 1_826_556
    forward = jax.jit(lambda p, v: old.apply({"params": p}, v, training=False)["logits"])
    expected = forward(params, x)
    for mode in ("proxy", "relative"):
        model = old.clone(motion_path=mode)
        actual = model.init(key, packed, training=False)["params"]
        assert jax.tree.structure(actual) == jax.tree.structure(params)
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(params)):
            np.testing.assert_array_equal(a, b)
        result = jax.jit(lambda p, v: model.apply({"params": p}, v, training=False)["logits"])(params, packed)
        assert np.isfinite(result).all()
        if mode == "proxy":
            np.testing.assert_allclose(result, expected, atol=3e-5, rtol=2e-4)
        else:
            assert not np.allclose(result, expected)
    jax.clear_caches()


def test_fast_memory_zero_padding_has_finite_input_and_parameter_gradients():
    model = FastWeightDeltaResidual(dim=8, rank=2)
    x = jnp.zeros((2, 4, 8))
    eta, alpha = jnp.full((2, 4, 1), .1), jnp.full((2, 4, 1), .95)
    params = model.init(jax.random.PRNGKey(4), x, eta, alpha)["params"]
    def loss(p, value):
        return jnp.sum(model.apply({"params": p}, value, eta, alpha))
    gradients = jax.jit(jax.grad(loss, argnums=(0, 1)))(params, x)
    assert all(np.isfinite(a).all() for a in jax.tree.leaves(gradients))
    # Also check an ordinary mixed real/padded batch, not only all-zero loss.
    mixed = x.at[0].set(jax.random.normal(jax.random.PRNGKey(9), (4, 8)))
    gradients = jax.jit(jax.grad(loss, argnums=(0, 1)))(params, mixed)
    assert all(np.isfinite(a).all() for a in jax.tree.leaves(gradients))

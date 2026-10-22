"""R9 leave-one-out ablations of graph_mr: structure, causality and parameter bookkeeping."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.nestsar_r5_t16.config import model_kwargs, validate_config
from experiments.nestsar_r5_t16.model import NestedMemory, NestSARR5T16
from experiments.nestsar_r5_t16.preprocessing import FEATURES, FRAMES
from experiments.nestsar_r5_t16.worker import EXPECTED_PARAMS

R9 = ("graph_mr_no_recon", "graph_mr_no_hand", "graph_mr_coarse", "graph_mr_no_inter", "graph_mr_one_level",
      "graph_mr_mem_only", "graph_mr_no_temporal", "graph_mr_shared", "graph_mr_causal", "graph_mr_causal_no_fast")


def params_of(variant):
    model = NestSARR5T16(**model_kwargs(validate_config({"variant": variant})))
    return jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), jnp.zeros((1, FRAMES, FEATURES))))["params"]


def count(tree):
    return sum(int(np.prod(a.shape)) for a in jax.tree.leaves(tree))


def test_every_r9_variant_is_recorded():
    for v in R9:
        assert v in EXPECTED_PARAMS


def test_no_recon_changes_only_the_training_head():
    base, ablated = params_of("graph_mr"), params_of("graph_mr_no_recon")
    assert set(base) - set(ablated) == {"recon_w", "recon_b"}
    assert count(base) - count(ablated) == 176 * 942 + 942


def test_shared_levels_have_one_memory():
    p = params_of("graph_mr_shared")
    assert "g4" not in p and "m4" in p and p["g4_level_embed"].shape == (176,)
    assert count(params_of("graph_mr")) - count(p) == count(params_of("graph_mr")["g4"]) - 176


def test_one_level_and_mem_only_structure():
    assert "g4" not in params_of("graph_mr_one_level")
    p = params_of("graph_mr_mem_only")
    assert "base" not in p["m4"] and "base" not in p["g4"] and "fast" in p["m4"]
    p = params_of("graph_mr_no_temporal")
    assert "m4" not in p and "g4" not in p          # no temporal parameters left at all


def test_causal_memory_never_sees_later_segments():
    mem = NestedMemory(16, rank=4, bidirectional=False, hidden=24)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 12, 16))
    variables = mem.init(jax.random.PRNGKey(1), x)
    y0 = mem.apply(variables, x)[0]
    t = 5
    y1 = mem.apply(variables, x.at[:, t + 1:].add(3.0))[0]
    np.testing.assert_allclose(np.asarray(y0[:, :t + 1]), np.asarray(y1[:, :t + 1]), atol=1e-6)
    assert float(jnp.max(jnp.abs(y0[:, t + 1:] - y1[:, t + 1:]))) > 1e-3


def test_bidirectional_memory_does_see_later_segments():
    mem = NestedMemory(16, rank=4)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 12, 16))
    variables = mem.init(jax.random.PRNGKey(1), x)
    y0, y1 = mem.apply(variables, x)[0], mem.apply(variables, x.at[:, 6:].add(3.0))[0]
    assert float(jnp.max(jnp.abs(y0[:, :6] - y1[:, :6]))) > 1e-3


@pytest.mark.parametrize("variant", ("graph_mr_shared", "graph_mr_one_level", "graph_mr_mem_only",
                                     "graph_mr_causal"))
def test_forward_and_gradients_finite(variant):
    model = NestSARR5T16(**model_kwargs(validate_config({"variant": variant})))
    x = jax.random.normal(jax.random.PRNGKey(2), (2, FRAMES, FEATURES)).at[1].set(0.0)
    variables = model.init(jax.random.PRNGKey(0), x)

    def loss(p):
        out = model.apply({"params": p}, x, training=True, rngs={"dropout": jax.random.PRNGKey(3)})
        return jnp.sum(out["logits"] ** 2) + jnp.sum(out["frame_recon"] ** 2)

    value, grads = jax.value_and_grad(loss)(variables["params"])
    assert np.isfinite(float(value))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads))

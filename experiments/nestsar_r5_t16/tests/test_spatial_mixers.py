"""R8 spatial mixers (graph / single-head skeleton attention) and the second temporal BiGRU."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.nestsar_r5_t16.config import model_kwargs, validate_config
from experiments.nestsar_r5_t16.model import (SKELETON_ADJ, SKELETON_HOPS, GraphMix, NestSARR5T16,
                                              SkeletonAttention)
from experiments.nestsar_r5_t16.preprocessing import FEATURES, FRAMES
from experiments.nestsar_r5_t16.worker import EXPECTED_PARAMS, STRICT_MFLOPS, make_loss


def test_skeleton_graph_is_a_tree():
    assert SKELETON_HOPS.shape == (25, 25) and (SKELETON_HOPS == SKELETON_HOPS.T).all()
    assert (np.diag(SKELETON_HOPS) == 0).all()
    assert int((SKELETON_HOPS == 1).sum()) == 2 * 24          # 24 bones, both directions
    np.testing.assert_allclose(SKELETON_ADJ, SKELETON_ADJ.T)


@pytest.mark.parametrize("mixer", [GraphMix(), SkeletonAttention()])
def test_missing_joints_do_not_leak(mixer):
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(3, 25, 48)).astype(np.float32))
    valid = np.ones((3, 25), bool); valid[:, 21:] = False; valid[2] = False   # one empty frame
    valid = jnp.asarray(valid)
    p = mixer.init(jax.random.PRNGKey(0), x, valid)
    y1 = mixer.apply(p, x, valid)
    x2 = x.at[:, 21:].set(123.0)                                             # garbage on missing joints
    y2 = mixer.apply(p, x2, valid)
    np.testing.assert_allclose(np.asarray(y1[:, :21]), np.asarray(y2[:, :21]), atol=1e-5)
    assert float(jnp.abs(y1[:, 21:]).max()) == 0 and float(jnp.abs(y1[2]).max()) == 0
    assert bool(jnp.all(jnp.isfinite(y1)))


@pytest.mark.parametrize("variant", ["graph2_mr", "attn2_mr"])
def test_variant_budget_and_finite_training(variant):
    assert STRICT_MFLOPS[variant] < STRICT_MFLOPS["full"]
    cfg = validate_config({"variant": variant})
    model = NestSARR5T16(**model_kwargs(cfg))
    rng = np.random.default_rng(1)
    for zero in (False, True):
        x = np.zeros((2, FRAMES, FEATURES), np.float32) if zero else \
            rng.normal(0, 0.3, (2, FRAMES, FEATURES)).astype(np.float32)
        batch = {"x": jnp.asarray(x), "xa": jnp.asarray(x), "y": jnp.asarray([3, 7]),
                 "y2": jnp.asarray([3, 7]), "lam": jnp.ones((2,), jnp.float32)}
        params = model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
                            batch["x"], training=False)["params"]
        assert sum(a.size for a in jax.tree.leaves(params)) == EXPECTED_PARAMS[variant]
        fn = make_loss(model, cfg)
        g = jax.grad(lambda p: fn(p, jax.random.PRNGKey(2), batch)[:, 0].mean())(params)
        assert all(bool(jnp.all(jnp.isfinite(a))) for a in jax.tree.leaves(g))
        out = model.apply({"params": params}, batch["x"], training=False)
        assert out["logits"].shape == (2, 120) and bool(jnp.all(jnp.isfinite(out["logits"])))

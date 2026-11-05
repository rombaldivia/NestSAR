"""R7 masked reconstruction: training-only head, unchanged inference graph, finite and effective loss."""
import jax
import jax.numpy as jnp
import numpy as np

from experiments.nestsar_r5_t16.config import model_kwargs, validate_config
from experiments.nestsar_r5_t16.model import NestSARR5T16
from experiments.nestsar_r5_t16.preprocessing import FEATURES, FRAMES
from experiments.nestsar_r5_t16.worker import EXPECTED_PARAMS, STRICT_MFLOPS, make_loss


def _model(variant, **extra):
    cfg = validate_config(dict(variant=variant, **extra))
    return NestSARR5T16(**model_kwargs(cfg)), cfg


def _init(model, x):
    return model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x,
                      training=False)["params"]


def _batch(b=4, seed=0, zero=False):
    rng = np.random.default_rng(seed)
    x = np.zeros((b, FRAMES, FEATURES), np.float32) if zero else \
        rng.normal(0, 0.3, (b, FRAMES, FEATURES)).astype(np.float32)
    xa = x if zero else x + rng.normal(0, 0.02, x.shape).astype(np.float32)
    y = rng.integers(0, 120, b)
    return {"x": jnp.asarray(x), "xa": jnp.asarray(xa), "y": jnp.asarray(y), "y2": jnp.asarray(y),
            "lam": jnp.ones((b,), jnp.float32)}


def test_inference_graph_unchanged():
    x = _batch(2)["x"]
    base, _ = _model("full")
    mr, _ = _model("full_mr")
    pb, pm = _init(base, x), _init(mr, x)
    assert sum(a.size for a in jax.tree.leaves(pm)) == EXPECTED_PARAMS["full_mr"]
    assert STRICT_MFLOPS["full_mr"] == STRICT_MFLOPS["full"]
    assert set(pm) - set(pb) == {"recon_w", "recon_b"}
    # Same shared weights -> identical inference logits, and no reconstruction output at eval time.
    ob = base.apply({"params": pb}, x, training=False)
    om = mr.apply({"params": dict(pb, recon_w=pm["recon_w"], recon_b=pm["recon_b"])}, x, training=False)
    assert "frame_recon" not in om
    np.testing.assert_array_equal(np.asarray(ob["logits"]), np.asarray(om["logits"]))
    ot = mr.apply({"params": pm}, x, training=True, rngs={"dropout": jax.random.PRNGKey(2)})
    assert ot["frame_recon"].shape == (2, FRAMES, FEATURES)


def test_base_loss_unchanged_by_recon_keys():
    """A non-MR variant ignores recon_* config (same key chain, same loss)."""
    model, cfg = _model("full")
    batch = _batch()
    p = _init(model, batch["x"])
    a = make_loss(model, cfg)(p, jax.random.PRNGKey(4), batch)
    b = make_loss(model, dict(cfg, recon_weight=2.0, recon_mask_ratio=0.7))(p, jax.random.PRNGKey(4), batch)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_recon_loss_adds_and_trains_head():
    model, cfg = _model("full_mr")
    batch = _batch()
    params = _init(model, batch["x"])
    loss = lambda p, w: make_loss(model, dict(cfg, recon_weight=w))(p, jax.random.PRNGKey(5), batch)[:, 0].mean()
    l0, l1 = float(loss(params, 0.0)), float(loss(params, 0.5))
    assert np.isfinite(l0) and np.isfinite(l1) and l1 > l0
    g = jax.grad(lambda p: loss(p, 0.5))(params)
    assert float(jnp.abs(g["recon_w"]).sum()) > 0
    assert all(bool(jnp.all(jnp.isfinite(a))) for a in jax.tree.leaves(g))
    g0 = jax.grad(lambda p: loss(p, 0.0))(params)
    assert float(jnp.abs(g0["recon_w"]).sum()) == 0


def test_empty_clip_gradients_finite():
    model, cfg = _model("full_mr")
    batch = _batch(zero=True)
    params = _init(model, batch["x"])
    fn = make_loss(model, cfg)
    g = jax.grad(lambda p: fn(p, jax.random.PRNGKey(1), batch)[:, 0].mean())(params)
    assert all(bool(jnp.all(jnp.isfinite(a))) for a in jax.tree.leaves(g))

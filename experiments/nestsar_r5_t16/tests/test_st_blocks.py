"""graph_mr_st2: joint-level spatio-temporal blocks. Identity at init (= graph_mr), missing joints never leak,
the blocks really mix across segments and joints, budget <= 60 strict MFLOPs, finite training step."""
import jax
import jax.numpy as jnp
import numpy as np

from experiments.nestsar_r5_t16 import preprocessing as pp
from experiments.nestsar_r5_t16 import worker
from experiments.nestsar_r5_t16.config import model_kwargs, validate_config
from experiments.nestsar_r5_t16.model import NestSARR5T16, STBlock


def _x(seed=0, b=2, t=16, p=2, v=25, c=48):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(b, t, p, v, c)).astype(np.float32)
    valid = np.ones((b, t, p, v), bool)
    valid[:, :, :, 21:] = False          # hand tips missing
    valid[:, :, 1] = False               # no second person
    valid[1, 10:] = False                # short clip: last segments empty
    x = x * valid[..., None]
    return jnp.asarray(x), jnp.asarray(valid)


def _trained_block(x, valid, dilation=1):
    blk = STBlock(5, dilation, 8)
    p = blk.init(jax.random.PRNGKey(0), x, valid)
    leaves, tree = jax.tree_util.tree_flatten(p)       # make every zero-init parameter non-zero
    keys = jax.random.split(jax.random.PRNGKey(1), len(leaves))
    leaves = [a + 0.1 * jax.random.normal(k, a.shape) for a, k in zip(leaves, keys)]
    return blk, jax.tree_util.tree_unflatten(tree, leaves)


def test_identity_at_init():
    x, valid = _x()
    blk = STBlock()
    p = blk.init(jax.random.PRNGKey(0), x, valid)
    np.testing.assert_array_equal(np.asarray(blk.apply(p, x, valid)), np.asarray(x))


def test_missing_joints_and_segments_do_not_leak():
    x, valid = _x()
    blk, p = _trained_block(x, valid)
    y1 = blk.apply(p, x, valid)
    garbage = jnp.where(valid[..., None], x, 77.0)                 # garbage everywhere that is missing
    y2 = blk.apply(p, garbage, valid)
    np.testing.assert_allclose(np.asarray(y1), np.asarray(y2), atol=1e-5)
    assert float(jnp.abs(jnp.where(valid[..., None], 0.0, y1)).max()) == 0.0
    assert bool(jnp.all(jnp.isfinite(y1)))


def test_block_mixes_across_segments_and_joints():
    x, valid = _x()
    blk, p = _trained_block(x, valid, dilation=2)
    y = blk.apply(p, x, valid)
    bump = jax.random.normal(jax.random.PRNGKey(5), (x.shape[-1],))   # not a constant: LayerNorm removes those
    x2 = x.at[0, 5, 0, 3].add(bump)                                 # change one joint in one segment
    d = np.abs(np.asarray(blk.apply(p, x2, valid) - y))[0, :, 0].max(-1)    # [T,V]
    assert d[5 + 2, 3] > 1e-4 and d[5 - 4, 3] > 1e-4                # other segments of the same joint
    assert d[5, 2] > 1e-4                                           # a skeleton neighbour, same segment
    # outside the +-4 temporal field only the clip-level (time-averaged) topology refinement can reach
    p0 = jax.tree_util.tree_map(lambda a: a, p)
    p0["params"]["alpha"] = jnp.zeros(())
    d0 = np.abs(np.asarray(blk.apply(p0, x2, valid) - blk.apply(p0, x, valid)))[0, :, 0].max(-1)
    assert d0[5 + 6, 3] < 1e-7 and d0[5 + 4, 3] > 1e-4


def test_model_equals_graph_mr_at_init_and_budget():
    assert worker.STRICT_MFLOPS["graph_mr_st2"] <= 60.0
    m = NestSARR5T16(**model_kwargs(validate_config({"variant": "graph_mr_st2"})))
    base = NestSARR5T16(**model_kwargs(validate_config({"variant": "graph_mr"})))
    k = jax.random.PRNGKey(0)
    x = jax.random.normal(k, (3, pp.FRAMES, pp.FEATURES)) * 0.3
    p = m.init({"params": k, "dropout": k}, x, training=False)["params"]
    assert sum(a.size for a in jax.tree.leaves(p)) == worker.EXPECTED_PARAMS["graph_mr_st2"]
    pb = {kk: vv for kk, vv in p.items() if not kk.startswith("st_block")}
    np.testing.assert_array_equal(np.asarray(m.apply({"params": p}, x)["logits"]),
                                  np.asarray(base.apply({"params": pb}, x)["logits"]))


def test_training_step_is_finite_and_moves_the_blocks():
    cfg = validate_config({"variant": "graph_mr_st2", "micro_batch": 2, "accumulation_steps": 2, "epochs": 2})
    model, state, key, _, _ = worker.create_state(cfg, 2)
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.3, (4, pp.FRAMES, pp.FEATURES)).astype(np.float32)
    batch = {"x": jnp.asarray(x), "xa": jnp.asarray(x + rng.normal(0, 0.05, x.shape).astype(np.float32)),
             "y": jnp.asarray(rng.integers(0, 120, 4).astype(np.int32)), "mask": jnp.asarray([1, 1, 1, 0], jnp.float32)}
    new, _, metrics = worker.build_steps(model, cfg)[0](state, key, batch)
    assert np.isfinite(np.asarray(metrics)).all()
    # the zero-init residual projection gets a gradient, so the blocks start learning on the first real step
    def loss(params):
        out = model.apply({"params": params}, batch["x"], training=True, rngs={"dropout": key})
        return -jnp.mean(jax.nn.log_softmax(out["logits"])[jnp.arange(4), batch["y"]])
    g = jax.grad(loss)(state.params)
    for i in (0, 1):
        assert float(jnp.abs(g[f"st_block_{i}"]["out"]["kernel"]).max()) > 0
    assert all(bool(jnp.all(jnp.isfinite(a))) for a in jax.tree.leaves(g))

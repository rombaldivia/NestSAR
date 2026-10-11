"""R6-HOPE: self-referential memory, CMS levels, outer multi-frequency updates, DMGD-L2."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from experiments.nestsar_r5_t16 import model as m5
from experiments.nestsar_r5_t16.config import OUTER_CMS_PERIODS, validate_config
from experiments.nestsar_r5_t16.worker import dmgd_l2, make_optimizer, tier_labels
from experiments.nestsar_r5_t16.tests.test_model import build, real_batch


def selfref(seed=0, length=16, dim=24, inner=8):
    mem = m5.SelfRefMemory(dim, inner)
    x = jax.random.normal(jax.random.PRNGKey(seed), (3, length, dim))
    params = mem.init(jax.random.PRNGKey(1), x)["params"]
    return mem, params, x


def test_selfref_is_causal():
    mem, params, x = selfref()
    out, eta, alpha = mem.apply({"params": params}, x)
    x2 = x.at[:, 10:].set(jax.random.normal(jax.random.PRNGKey(9), x[:, 10:].shape) * 5)
    out2, _, _ = mem.apply({"params": params}, x2)
    np.testing.assert_allclose(out[:, :10], out2[:, :10], atol=1e-5)      # past never sees the future
    assert not np.allclose(out[:, 10:], out2[:, 10:])
    assert np.all((eta > 0) & (eta < 1)) and np.all((alpha > 0.5) & (alpha <= 1))


def test_selfref_components_actually_self_modify():
    """With the component write switched off (rate 0) the output changes: K/V/Q/eta/alpha are memories."""
    mem, params, x = selfref()
    out, _, _ = mem.apply({"params": params}, x)
    rate = m5.SELFREF_COMPONENT_RATE
    try:
        m5.SELFREF_COMPONENT_RATE = 0.0
        frozen, _, _ = mem.apply({"params": params}, x)
    finally:
        m5.SELFREF_COMPONENT_RATE = rate
    np.testing.assert_allclose(out[:, 0], frozen[:, 0], atol=1e-6)       # first step: nothing written yet
    assert float(jnp.max(jnp.abs(out[:, 1:] - frozen[:, 1:]))) > 1e-6


def test_selfref_stays_bounded_on_long_large_inputs():
    mem, params, _ = selfref(length=256)
    x = 50.0 * jax.random.normal(jax.random.PRNGKey(3), (2, 256, 24))
    out, _, _ = mem.apply({"params": params}, x)
    assert np.isfinite(np.asarray(out)).all()
    grad = jax.grad(lambda p: jnp.sum(mem.apply({"params": p}, x)[0] ** 2))(params)
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grad))


def test_hope_levels_and_r5_default_unchanged():
    _, hope = build("hope")
    assert set(OUTER_CMS_PERIODS) <= set(hope)                 # m4, l2, g4, l8 all present
    for name in OUTER_CMS_PERIODS:
        assert "selfref" in hope[name] and "cms_mlp" in hope[name]
    _, full = build("full")
    assert not ({"l2", "l8"} & set(full)) and "fast" in full["m4"]


def test_auto_switches_follow_the_variant():
    assert validate_config({"variant": "hope"})["outer_cms"] is True
    assert validate_config({"variant": "hope"})["dmgd"] is True
    assert validate_config({"variant": "full"})["outer_cms"] is False
    assert validate_config({"variant": "full"})["dmgd"] is False
    assert validate_config({"variant": "hope", "dmgd": False})["dmgd"] is False
    with pytest.raises(ValueError):
        validate_config({"variant": "hope", "outer_cms": "yes"})


def test_outer_cms_updates_each_level_at_its_period():
    _, params = build("hope")
    config = validate_config({"variant": "hope", "dmgd": False})
    tx = make_optimizer(config, optax.constant_schedule(1e-2), params)
    state = tx.init(params)
    grads = jax.tree.map(jnp.ones_like, params)
    labels = tier_labels(params)
    assert labels["l8"]["cms_mlp"]["up"]["kernel"] == "p8" and labels["hand_in"]["kernel"] == "p1"
    moved = {name: [] for name in ("m4", "l2", "g4", "l8", "classifier")}
    p = params
    for _ in range(8):
        updates, state = tx.update(grads, state, p)
        for name in moved:
            moved[name].append(any(float(jnp.max(jnp.abs(u))) > 0 for u in jax.tree.leaves(updates[name])))
        p = optax.apply_updates(p, updates)
    assert moved["m4"] == [True] * 8 and moved["classifier"] == [True] * 8
    assert moved["l2"] == [False, True] * 4
    assert moved["g4"] == [False, False, False, True] * 2
    assert moved["l8"] == [False] * 7 + [True]


def test_r5_optimizer_is_the_plain_r5_chain():
    _, params = build("full")
    config = validate_config({"variant": "full"})
    sched = optax.constant_schedule(1e-3)
    tx = make_optimizer(config, sched, params)
    ref = optax.chain(optax.clip_by_global_norm(config["grad_clip"]),
                      optax.adamw(sched, weight_decay=config["weight_decay"]))
    g = jax.tree.map(lambda a: jnp.full_like(a, 0.3), params)
    u1, _ = tx.update(g, tx.init(params), params)
    u2, _ = ref.update(g, ref.init(params), params)
    for a, b in zip(jax.tree.leaves(u1), jax.tree.leaves(u2)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_dmgd_learns_to_predict_momentum():
    tx = dmgd_l2(momentum=0.9, memory_lr=0.05, mix=0.1)
    p = {"w": jnp.zeros(4)}
    state = tx.init(p)
    for _ in range(200):
        _, state = tx.update({"w": jnp.full(4, 0.5)}, state, p)
    # constant gradient g: momentum -> g, the predictor tanh(p) -> 1 (pred -> g)
    assert float(jnp.min(jnp.tanh(state.projection["w"]))) > 0.9


def test_hope_forward_backward_on_real_features():
    model, params = build("hope")
    x = real_batch(4)
    out = model.apply({"params": params}, x, training=False)
    assert out["logits"].shape == (4, 120) and np.isfinite(np.asarray(out["logits"])).all()
    grad = jax.grad(lambda p: jnp.mean(model.apply({"params": p}, x, training=False)["logits"] ** 2))(params)
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grad))
    assert float(jnp.max(jnp.abs(grad["l8"]["selfref"]["components0"]))) > 0   # gradient reaches the slowest level


def test_hope_train_state_checkpoint_roundtrip(tmp_path):
    """Outer-CMS (MultiSteps) + DMGD optimizer state survives save/restore mid-window."""
    from experiments.nestsar_r5_t16 import worker
    config = validate_config({"variant": "hope", "micro_batch": 2, "accumulation_steps": 1, "epochs": 2})
    model, state, key, _, _ = worker.create_state(config, steps_per_epoch=5)
    grads = jax.tree.map(lambda a: jnp.full_like(a, 1e-3), state.params)
    for _ in range(3):                      # stop inside the period-4 and period-8 windows
        state = state.apply_gradients(grads=grads)
    worker.save_checkpoint(tmp_path / "last.msgpack", state, key, {"config_hash": "h"})
    _, template, _, _, _ = worker.create_state(config, steps_per_epoch=5)
    restored, _, _ = worker.restore_checkpoint(tmp_path / "last.msgpack", template, "h")
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    s1 = state.apply_gradients(grads=grads)        # step 4: the period-4 tier fires in both
    s2 = restored.apply_gradients(grads=grads)
    for a, b in zip(jax.tree.leaves(s1.params), jax.tree.leaves(s2.params)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)


def test_hope_gradients_finite_on_an_all_zero_clip_without_mask():
    """Empty clip with a nonzero loss (the Kaggle preflight case): ~12 chained LayerNorms at exactly 0."""
    model, params = build("hope")
    x = jnp.zeros((1,) + real_batch(1).shape[1:])
    g = jax.grad(lambda p: jnp.mean(jax.nn.logsumexp(model.apply({"params": p}, x, training=False)["logits"])))(params)
    assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(g))


# ----------------------------------------------------------------------------- hope_core (full HOPE backbone)
def test_deep_memory_error_matches_autodiff():
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(5), 4)
    w1 = jax.random.normal(k1, (2, 6, 4)); w2 = jax.random.normal(k2, (2, 4, 6))
    k = jax.random.normal(k3, (2, 4)); v = jax.random.normal(k4, (2, 4))
    err, act, back = m5.deep_memory_error(w1, w2, k, v)
    loss = lambda a, b: 0.5 * jnp.sum(jnp.square(jnp.einsum("bdh,bh->bd", b, jnp.tanh(jnp.einsum("bhd,bd->bh", a, k))) - v))
    g1, g2 = jax.grad(loss, argnums=(0, 1))(w1, w2)
    np.testing.assert_allclose(-np.asarray(g2), np.asarray(jnp.einsum("bd,bh->bdh", err, act)), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(-np.asarray(g1), np.asarray(jnp.einsum("bh,bd->bhd", back, k)), rtol=1e-5, atol=1e-5)


def test_deep_selfref_is_causal_bounded_and_finite():
    mem = m5.SelfRefMemory(24, 8, deep=True, mem_hidden=8)
    x = 20.0 * jax.random.normal(jax.random.PRNGKey(0), (2, 64, 24))
    params = mem.init(jax.random.PRNGKey(1), x)["params"]
    out, eta, alpha = mem.apply({"params": params}, x)
    x2 = x.at[:, 30:].set(0.0)
    out2, _, _ = mem.apply({"params": params}, x2)
    np.testing.assert_allclose(out[:, :30], out2[:, :30], atol=1e-5)
    assert np.isfinite(np.asarray(out)).all()
    grad = jax.grad(lambda p: jnp.sum(mem.apply({"params": p}, x)[0] ** 2))(params)
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grad))


def test_hope_core_has_no_bigru_and_a_cms_chain_per_level():
    _, p = build("hope_core")
    for name in OUTER_CMS_PERIODS:
        assert "base" not in p[name] and "mixer" in p[name]                       # Titans short conv, no BiGRU
        assert set(p[name]["cms_chain"]) == {"mlp_p1", "mlp_p2", "mlp_p4", "mlp_p8"}
        assert {"memory_w1", "memory_w2"} <= set(p[name]["selfref"])              # deep memory
    labels = tier_labels(p)
    assert labels["l2"]["cms_chain"]["mlp_p8"]["up"]["kernel"] == "p8"          # chain period wins
    assert labels["l8"]["cms_chain"]["mlp_p1"]["up"]["kernel"] == "p1"
    assert labels["l8"]["selfref"]["memory_w1"] == "p8"                          # level period otherwise


def test_hope_core_gradients_finite_on_real_and_all_zero_clips():
    model, params = build("hope_core")
    for x in (real_batch(3), jnp.zeros((1,) + real_batch(1).shape[1:])):
        g = jax.grad(lambda p: jnp.mean(jax.nn.logsumexp(model.apply({"params": p}, x, training=False)["logits"])))(params)
        assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(g))

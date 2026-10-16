"""FR-Head (graph_mr_fr): training-only, inference unchanged, memories behave as the official RenovateNet."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.nestsar_r5_t16 import fr_head, worker
from experiments.nestsar_r5_t16 import preprocessing as pp
from experiments.nestsar_r5_t16.config import model_kwargs, validate_config
from experiments.nestsar_r5_t16.model import NUM_CLASSES, NestSARR5T16


def _batch(n=4, seed=0):
    rng = np.random.default_rng(seed)
    # (no all-zero clip: the base graph_mr already has an infinite gradient there, with or without FR-Head)
    x = rng.normal(0, 0.3, (n, pp.FRAMES, pp.FEATURES)).astype(np.float32)
    xa = x + rng.normal(0, 0.05, x.shape).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, n).astype(np.int32)
    return {"x": jnp.asarray(x), "xa": jnp.asarray(xa), "y": jnp.asarray(y), "mask": jnp.ones(n, jnp.float32)}


def test_inference_is_the_base_model():
    base = NestSARR5T16(**model_kwargs(validate_config({"variant": "graph_mr"})))
    fr = NestSARR5T16(**model_kwargs(validate_config({"variant": "graph_mr_fr"})))
    x = _batch()["x"]
    p_fr = fr.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, x, training=False)["params"]
    p_base = {k: v for k, v in p_fr.items() if k != "fr_head"}
    out_fr, out_base = fr.apply({"params": p_fr}, x), base.apply({"params": p_base}, x)
    np.testing.assert_array_equal(out_fr["logits"], out_base["logits"])
    assert "fr_feats" not in out_fr
    assert sum(v.size for v in jax.tree.leaves(p_fr["fr_head"])) == \
        worker.EXPECTED_PARAMS["graph_mr_fr"] - worker.EXPECTED_PARAMS["graph_mr"]
    assert worker.STRICT_MFLOPS["graph_mr_fr"] == worker.STRICT_MFLOPS["graph_mr"]
    train = fr.apply({"params": p_fr}, x, training=True, rngs={"dropout": jax.random.PRNGKey(1)})
    assert set(train["fr_feats"]) == set(fr_head.LEVELS)
    assert all(v.shape == (4, 256) and np.isfinite(np.asarray(v)).all() for v in train["fr_feats"].values())


def test_memory_moves_only_for_true_positive_classes_and_ignores_padding():
    rng = np.random.default_rng(1)
    k, c = 6, 8
    z = jnp.asarray(rng.normal(size=(5, c)).astype(np.float32))
    y = jnp.asarray([0, 0, 1, 2, 3])
    logits = np.full((5, k), -5.0, np.float32)
    logits[[0, 1, 2, 3, 4], [0, 4, 1, 5, 3]] = 5.0                 # TP: rows 0, 2, 4; FN/FP: rows 1, 3
    mask = jnp.asarray([1, 1, 1, 1, 0], jnp.float32)               # row 4 (class 3) is padding
    bank = jnp.asarray(rng.normal(size=(c, k)).astype(np.float32))
    loss, new = fr_head.renovate(z, y, jnp.asarray(logits), mask, bank)
    moved = np.any(np.abs(np.asarray(new - bank)) > 1e-7, axis=0)
    assert moved.tolist() == [True, True, False, False, False, False]  # classes with a real TP only
    np.testing.assert_allclose(new[:, 0], 0.9 * bank[:, 0] + 0.1 * z[0], atol=1e-6)
    assert float(loss[4]) == 0.0 and np.isfinite(np.asarray(loss)).all() and float(loss[:4].min()) > 0
    grad = jax.grad(lambda zz: fr_head.renovate(zz, y, jnp.asarray(logits), mask, bank)[0].sum())(z)
    assert np.isfinite(np.asarray(grad)).all() and float(jnp.abs(grad[4]).max()) == 0.0


def test_zero_features_stay_finite():
    z = jnp.zeros((3, 8))
    loss, new = fr_head.renovate(z, jnp.asarray([0, 1, 2]), jnp.zeros((3, 4)), jnp.ones(3), jnp.ones((8, 4)))
    assert np.isfinite(np.asarray(loss)).all() and np.isfinite(np.asarray(new)).all()


@pytest.fixture(scope="module")
def tiny():
    config = validate_config({"variant": "graph_mr_fr", "micro_batch": 2, "accumulation_steps": 2, "epochs": 2})
    model, state, key, _, _ = worker.create_state(config, 2)
    return model, state, key, config


def test_train_step_updates_the_memories_and_weight_zero_is_the_base_path(tiny):
    model, state, key, config = tiny
    b = _batch()
    # make two clips true positives of the untrained model, so their classes get a memory update
    pred = model.apply({"params": state.params}, b["x"], training=False)["logits"].argmax(-1)
    b["y"] = b["y"].at[:2].set(pred[:2])
    assert set(state.fr_banks) == set(fr_head.LEVELS)
    new_state, _, metrics = worker.build_steps(model, config)[0](state, key, b)
    m = np.asarray(metrics)
    assert np.isfinite(m).all() and m[worker.TRAIN_METRICS.index("fr")] > 0
    assert any(float(jnp.abs(new_state.fr_banks[n] - state.fr_banks[n]).max()) > 0 for n in fr_head.LEVELS)
    off = dict(config, fr_weight=0.0)
    off_state, _, off_metrics = worker.build_steps(model, off)[0](state, key, b)
    assert float(np.asarray(off_metrics)[worker.TRAIN_METRICS.index("fr")]) == 0.0
    for n in fr_head.LEVELS:
        np.testing.assert_array_equal(off_state.fr_banks[n], state.fr_banks[n])


def test_config_checks():
    assert validate_config({})["fr_level_weights"] == (0.1, 0.2, 0.5, 1.0)
    for bad in ({"fr_level_weights": [1, 2]}, {"fr_weight": -1}, {"fr_temperature": 0},
                {"variant": "graph_mr_fr", "mix_prob": 0.5}):
        with pytest.raises(ValueError):
            validate_config(bad)

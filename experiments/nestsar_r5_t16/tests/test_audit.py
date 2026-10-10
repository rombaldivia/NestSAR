import jax
import numpy as np
import pytest

from experiments.nestsar_r5_t16 import audit_r5_checkpoint as audit
from experiments.nestsar_r5_t16 import preprocessing as pp
from experiments.nestsar_r5_t16.config import validate_config
from experiments.nestsar_r5_t16.worker import make_model


@pytest.fixture(scope="module")
def trained_like():
    """Random-init R5 with non-trivial weights (zero-init layers would hide the counterfactuals)."""
    model = make_model(validate_config({}))
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.5, (4, pp.FRAMES, pp.FEATURES)).astype(np.float32)
    params = model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
                        x, training=False)["params"]
    # Make the zero-initialised parts of the real trained network non-zero.
    leaves, tree = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(2), len(leaves))
    leaves = [leaf + 0.05 * jax.random.normal(k, leaf.shape, leaf.dtype) for leaf, k in zip(leaves, keys)]
    params = jax.tree_util.tree_unflatten(tree, leaves)
    return model, jax.device_get(params), x


def logits(model, params, x):
    return np.asarray(model.apply({"params": params}, x, training=False)["logits"])


def test_trained_mode_is_identity(trained_like):
    model, params, x = trained_like
    same = audit.counterfactual_params(params, "trained")
    np.testing.assert_array_equal(logits(model, params, x), logits(model, same, x))


@pytest.mark.parametrize("mode", ["fast_scale_zero", "fast_frozen", "pair_message_off"])
def test_parameter_counterfactuals_change_the_output_and_not_the_original(trained_like, mode):
    model, params, x = trained_like
    snapshot = jax.tree_util.tree_map(np.array, params)
    changed = audit.counterfactual_params(params, mode)
    assert audit.applicable(params, mode)
    assert not np.allclose(logits(model, params, x), logits(model, changed, x), atol=1e-6)
    for a, b in zip(jax.tree_util.tree_leaves(snapshot), jax.tree_util.tree_leaves(params)):
        np.testing.assert_array_equal(a, b)               # the input params were not modified


def test_fast_frozen_really_freezes_the_memory(trained_like):
    model, params, x = trained_like
    out = model.apply({"params": audit.counterfactual_params(params, "fast_frozen")}, x, training=False)
    assert float(np.max(np.abs(np.asarray(out["sm_eta_mean"])))) < 1e-6
    assert float(np.min(np.asarray(out["sm_alpha_mean"]))) > 1 - 1e-6
    assert float(np.max(np.abs(np.asarray(out["g4_eta_mean"])))) < 1e-6
    base = model.apply({"params": params}, x, training=False)
    assert float(np.mean(np.asarray(base["sm_eta_mean"]))) > 1e-3     # the trained memory does write


def test_fast_scale_zero_removes_the_residual(trained_like):
    model, params, x = trained_like
    p = audit.counterfactual_params(params, "fast_scale_zero")
    assert not np.any(np.asarray(p["m4"]["fast_scale"]))
    assert not np.any(np.asarray(p["g4"]["fast_scale"]))
    out = model.apply({"params": p}, x, training=False)
    assert float(np.max(np.asarray(out["fast_scale_m4"]))) == 0.0


def test_variants_without_the_component_are_skipped():
    model = make_model(validate_config({"variant": "no_fast_memory"}))
    x = np.zeros((1, pp.FRAMES, pp.FEATURES), np.float32)
    params = model.init({"params": jax.random.PRNGKey(0)}, x, training=False)["params"]
    assert not audit.applicable(params, "fast_scale_zero")
    assert not audit.applicable(params, "fast_frozen")
    model = make_model(validate_config({"variant": "no_interaction"}))
    params = model.init({"params": jax.random.PRNGKey(0)}, x, training=False)["params"]
    assert not audit.applicable(params, "pair_message_off")


def test_calibration_and_class_report_on_a_known_case():
    labels = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 1, 0])
    report = audit.class_report(labels, pred, {})
    assert report["recall"]["A001"] == 0.5 and report["recall"]["A002"] == 1.0 and report["recall"]["A003"] == 0.0
    assert {"true": "A001", "pred": "A002", "count": 1} in report["top_confusions"]
    prob = np.full((5, audit.NUM_CLASSES), 1e-4)
    prob[np.arange(5), pred] = 0.9
    cal = audit.calibration(prob / prob.sum(1, keepdims=True), labels)
    assert cal["accuracy"] == 0.6 and cal["confidence_when_wrong"] is not None


def test_actor_count_from_tokens():
    class Fake:
        def canonical(self, ids):
            out = np.zeros((len(ids), pp.FRAMES, pp.FEATURES), np.float32)
            body = out[..., :pp.R4_FEATURES].reshape(len(ids), pp.FRAMES, pp.PERSONS, pp.JOINTS, pp.TOKEN_CHANNELS)
            body[:, :, 0] = 1.0
            body[1, :, 1] = 1.0                                   # second sample has two actors
            return out
    two = audit.actor_count(Fake(), [0, 1, 2])
    assert two.tolist() == [False, True, False]

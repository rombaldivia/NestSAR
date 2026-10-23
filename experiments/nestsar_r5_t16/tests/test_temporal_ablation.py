"""Temporal-core ablation variants: base untouched, every variant has its tabled size/cost and trains finitely
(zero-padded rows and a clip without a second person included)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.nestsar_r5_t16 import preprocessing as pp
from experiments.nestsar_r5_t16 import worker
from experiments.nestsar_r5_t16.config import model_kwargs, validate_config
from experiments.nestsar_r5_t16.model import NestSARR5T16

ABLATION = ("graph_mr_flat", "graph_mr_flat_no_fast", "graph_mr_flat_stack", "graph_mr_tattn")


def test_base_is_untouched():
    m = NestSARR5T16(**model_kwargs(validate_config({"variant": "graph_mr"})))
    k = jax.random.PRNGKey(0)
    p = m.init({"params": k, "dropout": k}, jnp.zeros((1, pp.FRAMES, pp.FEATURES)), training=False)["params"]
    assert sum(a.size for a in jax.tree.leaves(p)) == 1_300_768 == worker.EXPECTED_PARAMS["graph_mr"]
    assert "g4" in p and "m4" in p


@pytest.mark.parametrize("variant", ABLATION)
def test_variant_trains_finitely(variant):
    cfg = validate_config({"variant": variant, "micro_batch": 2, "accumulation_steps": 2, "epochs": 2})
    model, state, key, _, _ = worker.create_state(cfg, 2)       # checks EXPECTED_PARAMS
    assert ("g4" in state.params) == (variant == "graph_mr_tattn")
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.3, (4, pp.FRAMES, pp.FEATURES)).astype(np.float32)
    x[1, :, pp.R4_FEATURES // 2:pp.R4_FEATURES] = 0                    # no second person in clip 1
    x[3] = 0                                                           # padding row
    batch = {"x": jnp.asarray(x), "xa": jnp.asarray(x), "y": jnp.asarray([0, 5, 9, 0], jnp.int32),
             "mask": jnp.asarray([1, 1, 1, 0], jnp.float32)}
    _, _, metrics = worker.build_steps(model, cfg)[0](state, key, batch)
    assert np.isfinite(np.asarray(metrics)).all()
    out = model.apply({"params": state.params}, batch["x"][:3], training=False)
    assert out["logits"].shape == (3, 120) and bool(jnp.all(jnp.isfinite(out["logits"])))

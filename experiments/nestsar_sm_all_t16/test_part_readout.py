import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict

from .part_readout import NonlinearPartReadout
from .part_readout_config import VERSION, extra_parameters
from .model import NestSARSMAllT16
from .streaming.launch import validate_config
from .streaming.worker import expected_parameters, model_metadata, publish_best


def active_params(model, h, valid):
    p = model.init(jax.random.key(17), h, valid)["params"]
    p["out_proj"]["kernel"] = jnp.ones_like(p["out_proj"]["kernel"]) * .3
    return p


def test_can_distinguish_opposing_joint_features_with_identical_part_means():
    h = jnp.zeros((1, 1, 2, 25, 24))
    h = h.at[0, 0, 0, 14, 0].set(1).at[0, 0, 0, 15, 0].set(-1)
    valid = jnp.zeros(h.shape[:-1], bool).at[0, 0, 0, 14:16].set(True)
    model = NonlinearPartReadout()
    p = active_params(model, h, valid)
    p["feature_map"]["kernel"] = jnp.ones_like(p["feature_map"]["kernel"])
    p["joint_mixtures"] = p["joint_mixtures"].at[14].set(-1).at[15].set(1)
    np.testing.assert_array_equal(h[..., 14:16, :].mean(-2), 0)
    out = model.apply({"params": p}, h, valid)
    zero = model.apply({"params": p}, jnp.zeros_like(h), valid)
    assert float(jnp.max(jnp.abs(out - zero))) > 1e-3
    np.testing.assert_array_equal(out[:, :, 1], 0)  # Absent second person.


def test_masked_garbage_single_joints_and_gradients_are_safe():
    h = jax.random.normal(jax.random.key(5), (1, 2, 2, 25, 24))
    valid = jnp.zeros(h.shape[:-1], bool).at[:, :, 0, 6:8].set(True)
    valid = valid.at[:, :, 0, 14].set(True)  # Only ankle; foot absent.
    clean = jnp.where(valid[..., None], h, 0)
    dirty = jnp.where(valid[..., None], h, jnp.nan)
    model = NonlinearPartReadout()
    p = active_params(model, clean, valid)
    expected = model.apply({"params": p}, clean, valid)
    result = model.apply({"params": p}, dirty, valid)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(result[:, :, 1], 0)
    np.testing.assert_array_equal(result[:, :, 0, 7], 0)  # Singleton leg part.
    grad = jax.grad(lambda x: model.apply({"params": p}, x, valid).sum())(dirty)
    assert np.isfinite(grad).all()
    np.testing.assert_array_equal(np.asarray(grad)[~np.asarray(valid)], 0)


def test_all_branch_parameters_receive_gradients_after_zero_output_warm_start():
    h = jax.random.normal(jax.random.key(9), (2, 2, 2, 25, 24))
    valid = jnp.ones(h.shape[:-1], bool)
    model = NonlinearPartReadout()
    p = model.init(jax.random.key(7), h, valid)["params"]
    def loss(p):
        return jnp.square(model.apply({"params": p}, h, valid) - .1).mean()
    grad = jax.jit(jax.grad(loss))(p)
    assert float(jnp.linalg.norm(grad["out_proj"]["kernel"])) > 0
    p = jax.tree.map(lambda a, g: a - .1*g, p, grad)
    grad = jax.jit(jax.grad(loss))(p)
    for value in jax.tree.leaves(grad):
        assert np.isfinite(value).all() and float(jnp.linalg.norm(value)) > 0


def test_full_model_starts_as_baseline_and_preserves_all_existing_parameters():
    x = jax.random.normal(jax.random.key(51), (1, 16, 2, 25, 15)) * .1
    x = x.at[:, :, 1].set(0).reshape(1, 16, 750)
    reference, proposed = NestSARSMAllT16(), NestSARSMAllT16(part_readout=True)
    key = jax.random.key(52)
    old = reference.init(key, x)["params"]
    new = proposed.init(key, x)["params"]
    a, b = flatten_dict(old), flatten_dict(new)
    for name, value in a.items():
        np.testing.assert_array_equal(value, b[name])
    assert all("part_readout" in name for name in b.keys() - a.keys())
    assert sum(v.size for v in a.values()) == 1_826_556
    assert sum(v.size for v in b.values()) == 1_826_556 + extra_parameters() == 1_829_060
    forward = jax.jit(lambda model, p, x: model.apply({"params": p}, x)["logits"], static_argnums=0)
    np.testing.assert_allclose(forward(reference, old, x), forward(proposed, new, x), atol=2e-6, rtol=2e-6)


def test_config_and_best_checkpoint_describe_actual_architecture(tmp_path):
    assert "part_readout" not in validate_config({})  # Historical resume hash stable.
    config = validate_config({"part_readout": VERSION})
    assert expected_parameters(config) == 1_829_060
    meta = dict(model_metadata(config), best_checkpoint="best_epoch_0002.msgpack",
                best_epoch=2, best=.7, config_hash="readout")
    (tmp_path / meta["best_checkpoint"]).write_bytes(b"checkpoint")
    publish_best(tmp_path, meta)
    summary = json.loads((tmp_path / "best.json").read_text())
    assert summary["params"] == 1_829_060 and summary["part_readout"] == VERSION
    for invalid in ({"part_readout": "other"}, {"part_readout": VERSION, "max_train_samples": 5},
                    {"sampler_cache_dir": "/tmp/cache"}):
        with pytest.raises(ValueError):
            validate_config(invalid)


def test_launcher_keeps_last_run_settings_and_reuses_b_cache(monkeypatch, tmp_path):
    from . import run_part_readout_dual_t4 as entry
    from .sampler_b import VERSION as B_VERSION
    captured = {}
    def fake_run(**kwargs):
        captured.update(kwargs)
        return "started"
    monkeypatch.setattr(entry, "run_sampler_b", fake_run)
    assert entry.run_part_readout(sampler_cache_dir=tmp_path, dataset="data.pkl") == "started"
    c = validate_config(dict(captured["config"], pose_sampler=B_VERSION))
    assert c["sampler_cache_dir"] == str(tmp_path)
    assert (c["epochs"], c["patience"], c["seed"], c["micro_batch"], c["accumulation_steps"]) == (60, 5, 128, 64, 4)
    assert c["max_train_samples"] == c["max_val_samples"] == 0
    assert c["fresh_augmentation"] and captured["audit_first"]

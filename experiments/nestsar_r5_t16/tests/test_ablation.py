import contextlib
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from experiments.nestsar_r4_fmse_geometry_t16.config import validate_config as geo_config
from experiments.nestsar_r4_fmse_geometry_t16.worker import make_model
from experiments.nestsar_r5_t16 import ablate_r4_fastweights as ab
from experiments.nestsar_r5_t16.smoke_cpu import synthetic_pickle
from experiments.nestsar_sm_all_t16.streaming.data import Dataset, prepare


def perturbed_r4(seed=0):
    config = geo_config({})
    model = make_model(config)
    params = model.init({"params": jax.random.PRNGKey(seed), "dropout": jax.random.PRNGKey(1)},
                        jnp.zeros((1, 16, 750)), training=False)["params"]
    leaves, tree = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(seed + 7), len(leaves))
    # Break the zero-initialised controller heads so every counterfactual has an effect.
    leaves = [p + 0.3 * jax.random.normal(k, p.shape) for p, k in zip(leaves, keys)]
    return config, model, jax.tree_util.tree_unflatten(tree, leaves)


def logits(model, params, x, mode):
    ctx = ab.interceptor(mode)
    with (ctx if ctx is not None else contextlib.nullcontext()):
        return np.asarray(model.apply({"params": params}, x, training=False)["logits"])


def test_every_counterfactual_changes_the_logits():
    config, model, params = perturbed_r4()
    x = jax.random.normal(jax.random.PRNGKey(3), (3, 16, 750)) * 0.3
    base = logits(model, params, x, "trained")
    for mode in ab.MODES[1:]:
        diff = float(np.abs(logits(model, params, x, mode) - base).max())
        assert diff > 1e-5, mode
    # Freezing both levels is the composition of freezing each level.
    a = logits(model, params, x, "fast_frozen")
    assert not np.allclose(a, logits(model, params, x, "fast_frozen_m4"))
    assert not np.allclose(a, logits(model, params, x, "fast_frozen_g4"))


def test_end_to_end_on_tiny_cache(tmp_path):
    synthetic_pickle(tmp_path / "ntu120_3danno.pkl")
    prepare(tmp_path / "ntu120_3danno.pkl", tmp_path / "cache", tmp_path / "status.json")
    config, model, params = perturbed_r4()
    dataset = Dataset(tmp_path / "cache")
    root = tmp_path / "run"
    for protocol in ("xsub", "xset"):
        ids = dataset.splits[f"{protocol}_val"]
        pred, *_ = ab.evaluate(model, params, dataset, ids, 4, "trained")
        acc = float(np.mean(pred == np.asarray(dataset.labels[np.asarray(ids)])))
        (root / protocol).mkdir(parents=True)
        payload = {"model": ab.REFERENCE_MODEL, "epoch": 3, "val_accuracy": acc,
                   "ema_params": jax.device_get(params), "config": config}
        (root / protocol / "best.msgpack").write_bytes(serialization.to_bytes(payload))
        (root / protocol / "best.json").write_text(json.dumps({"model": ab.REFERENCE_MODEL}))
    report = ab.main(["--checkpoint-root", str(root), "--r4-cache", str(tmp_path / "cache"), "--batch", "4"])
    assert ab.find_checkpoint_root([tmp_path]) == root
    for protocol in ("xsub", "xset"):
        rows = report["protocols"][protocol]["modes"]
        assert set(rows) == set(ab.MODES)
        assert rows["trained"]["reproduced"]
        # The R4 cap: logits in [-0.15, 0.15] keep every weight within [0.198, 0.311].
        assert rows["trained"]["fusion_weight_max_abs_from_uniform"] <= 0.0604
        for mode in ab.MODES[1:]:
            assert rows[mode]["fixed"] >= 0 and rows[mode]["broken"] >= 0
    saved = json.loads((root / "r4_fastweight_ablation.json").read_text())
    assert saved["protocols"].keys() == report["protocols"].keys()

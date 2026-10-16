"""Regressions for stale checkpoint predictions and same-label sample reorderings."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from experiments.nestsar_r5_t16 import eval_heads as evaluation
from experiments.nestsar_r5_t16.config import validate_config
from experiments.nestsar_r5_t16 import preprocessing as pp


class ToyModel:
    calls = 0

    def apply(self, variables, x, training=False):
        type(self).calls += 1
        logits = jnp.broadcast_to(variables["params"]["bias"], (len(x), 120))
        return dict(logits=logits, stream_logits=jnp.stack([logits, logits], axis=1))


class Dataset:
    def __init__(self, root):
        root.mkdir()
        self.base_dir = root
        (root / "ids.json").write_text(json.dumps(["sample-a", "sample-b", "sample-c"]))
        np.save(root / "labels.npy", np.array([1, 1, 2], np.int32))
        np.save(root / "canonical.npy", np.zeros((3, pp.FRAMES, pp.R4_FEATURES), np.float32))
        np.save(root / "hand.npy", np.zeros((3, pp.FRAMES, pp.FEATURES - pp.R4_FEATURES), np.float32))
        self.labels = np.load(root / "labels.npy", mmap_mode="r+")
        self.base = SimpleNamespace(canonical=np.load(root / "canonical.npy", mmap_mode="r+"))
        self.hand = np.load(root / "hand.npy", mmap_mode="r+")
        self.meta = {"signature": {"preprocessing": pp.VERSION, "source": "toy"}}
        self.splits = {"xsub_val": [0, 1, 2]}

    def batches(self, ids, size, config, protocol):
        for start in range(0, len(ids), size):
            index = ids[start:start + size]
            x = np.zeros((size, pp.FRAMES, pp.FEATURES), np.float32)
            y, mask = np.zeros(size, np.int32), np.zeros(size, np.float32)
            x[:len(index)] = np.concatenate([self.base.canonical[index], self.hand[index]], axis=-1)
            y[:len(index)], mask[:len(index)] = self.labels[index], 1
            yield dict(x=x, y=y, mask=mask), 0.0


@pytest.fixture
def run(tmp_path, monkeypatch):
    ToyModel.calls = 0
    monkeypatch.setattr(evaluation, "NestSARR5T16", lambda **kwargs: ToyModel())
    dataset = Dataset(tmp_path / "data")
    root = tmp_path / "run"
    (root / "xsub").mkdir(parents=True)
    config = validate_config({"eval_batch": 2})
    _write_run(root, dataset, config, winner=1)
    return root, dataset, config


def _write_run(root, dataset, config, winner):
    out = root / "xsub"
    (out / "run_config.json").write_text(json.dumps(
        dict(config=config, protocol="xsub", cache=dataset.meta["signature"])))
    bias = np.zeros(120, np.float32)
    bias[winner] = 10.0
    (out / "best.msgpack").write_bytes(serialization.to_bytes(dict(
        config=config, protocol="xsub", cache_signature=dataset.meta["signature"],
        preprocessing_version=pp.VERSION, ema_params={"bias": bias})))


def test_compatible_cache_avoids_another_inference(run):
    root, dataset, _ = run
    first = evaluation.evaluate_bundle(root, "xsub", dataset)
    calls = ToyModel.calls
    second = evaluation.evaluate_bundle(root, "xsub", dataset)
    assert ToyModel.calls == calls and calls > 0
    np.testing.assert_array_equal(first["main"], second["main"])
    np.testing.assert_array_equal(second["ids"], ["sample-a", "sample-b", "sample-c"])


def test_changed_checkpoint_recomputes_instead_of_returning_old_predictions(run):
    root, dataset, config = run
    first = evaluation.evaluate_bundle(root, "xsub", dataset)
    _write_run(root, dataset, config, winner=2)
    second = evaluation.evaluate_bundle(root, "xsub", dataset)
    assert np.all(first["main"].argmax(1) == 1)
    assert np.all(second["main"].argmax(1) == 2)
    assert first["identity"]["checkpoint_sha256"] != second["identity"]["checkpoint_sha256"]


def test_existing_predictions_cannot_bypass_a_missing_checkpoint(run):
    root, dataset, _ = run
    evaluation.evaluate(root, "xsub", dataset)
    (root / "xsub" / "best.msgpack").unlink()
    with pytest.raises(FileNotFoundError):
        evaluation.evaluate(root, "xsub", dataset)


def test_reordered_samples_with_identical_labels_invalidate_predictions(run):
    root, dataset, _ = run
    first = evaluation.evaluate_bundle(root, "xsub", dataset)
    calls = ToyModel.calls
    (dataset.base_dir / "ids.json").write_text(json.dumps(["sample-b", "sample-a", "sample-c"]))
    second = evaluation.evaluate_bundle(root, "xsub", dataset)
    assert ToyModel.calls > calls
    np.testing.assert_array_equal(first["y"], second["y"])
    assert first["identity"]["ids_sha256"] != second["identity"]["ids_sha256"]


def test_changed_input_file_invalidates_predictions(run):
    root, dataset, _ = run
    first = evaluation.evaluate_bundle(root, "xsub", dataset)
    path = Path(dataset.hand.filename)
    stat = path.stat()
    dataset.hand[0, 0, 0] = 1.0
    dataset.hand.flush()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = evaluation.evaluate_bundle(root, "xsub", dataset)
    assert first["identity"]["dataset_files"] != second["identity"]["dataset_files"]


@pytest.mark.parametrize("kind", ["legacy", "corrupt", "wrong_shape"])
def test_invalid_prediction_file_is_rebuilt(run, kind):
    root, dataset, _ = run
    first = evaluation.evaluate_bundle(root, "xsub", dataset)
    target = root / "xsub" / "val_probs.npz"
    if kind == "corrupt":
        target.write_bytes(b"not a numpy archive")
    else:
        arrays = dict(y=first["y"], main=first["main"][:1], aux=first["aux"], ids=first["ids"])
        if kind == "wrong_shape":
            arrays["identity_json"] = np.asarray(evaluation._json(first["identity"]))
        np.savez_compressed(target, **arrays)
    calls = ToyModel.calls
    result = evaluation.evaluate_bundle(root, "xsub", dataset)
    assert ToyModel.calls > calls and result["main"].shape == (3, 120)


def test_checkpoint_with_another_dataset_is_rejected(run):
    root, dataset, _ = run
    dataset.meta["signature"] = dict(dataset.meta["signature"], source="different")
    with pytest.raises(ValueError, match="cache signatures disagree"):
        evaluation.evaluate(root, "xsub", dataset)


def test_output_cache_can_live_outside_the_checkpoint_directory(run, tmp_path):
    root, dataset, _ = run
    destination = tmp_path / "writable-output"
    evaluation.evaluate_bundle(root, "xsub", dataset, prediction_cache_dir=destination)
    assert (destination / "val_probs.npz").is_file()
    assert not (root / "xsub" / "val_probs.npz").exists()

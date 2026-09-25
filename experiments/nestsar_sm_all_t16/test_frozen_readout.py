"""CPU checks for numerical equivalence, isolation, and resumable probe fitting."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from . import audit_frozen_readout as audit
from .model import NestSARSMAllT16


def zero_core():
    core = {f"classifier_{i}": dict(kernel=np.zeros((112, 120), np.float32),
                                   bias=np.zeros(120, np.float32)) for i in range(4)}
    core["adaptive_head_u"] = {"kernel": np.zeros((112, 2), np.float32)}
    core["adaptive_head_v"] = {"kernel": np.zeros((2, 120), np.float32)}
    return core


def synthetic_bank(root, n=48):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    y = (np.arange(n) % 2).astype(np.int32)
    rng = np.random.default_rng(918)
    desc = np.zeros((n, 4, 112), np.float32)
    desc[:, :, 0] = (2 * y - 1)[:, None]
    context = np.zeros((n, 6), np.float32)
    context[:, :4] = 0.25
    temporal = rng.normal(size=(n, 2688)).astype(np.float32) * 0.05
    temporal[:, 0] = 2 * y - 1
    for name, values in dict(desc=desc, context=context, temporal=temporal,
                             logits=np.zeros((n, 120), np.float32)).items():
        np.save(root / (name + ".npy"), values)
    return audit.FeatureBank(root), y


def args_for(root):
    return SimpleNamespace(output=str(root), protocol="xsub", epochs=3,
                           batch_size=16, learning_rate=0.03, weight_decay=0.0,
                           smoothing=0.0, dropout=0.10)


class FrozenReadoutTests(unittest.TestCase):
    def test_actual_backbone_head_equivalence_and_zero_residual(self):
        model = NestSARSMAllT16(dropout=0.10, fast_rank=4)
        x = jax.random.normal(jax.random.PRNGKey(3), (2, 16, 750)) * 0.1
        params = model.init(jax.random.PRNGKey(4), x, training=False)["params"]
        self.assertEqual(audit.tree_count(params), audit.EXPECTED_PARAMS)
        out = jax.device_get(model.apply({"params": params}, x, training=False))
        core = audit.core_params(params)
        context = np.concatenate([out["fusion_weights"], out["sm_head_coeff"]], -1)
        logits = audit.core_logits(core, out["descriptors"], context, 0.15, np)
        np.testing.assert_allclose(logits, out["logits"], rtol=2e-5, atol=3e-5)
        contrasts = audit.temporal_contrasts(out["mixed_frame_stack"], out["chunk_states"])
        self.assertEqual(contrasts.shape, (2, 2688))
        for variant, hidden in audit.hidden_sizes().items():
            dim = 3142 if variant == "C_temporal" else 454
            p = audit.init_probe(core, dim, hidden, 128)
            _, _, predict = audit.make_steps(0.15, args_for("."), 2)
            batch = dict(desc=jnp.array(out["descriptors"]), context=jnp.array(context),
                         z=jnp.ones((2, dim)))
            actual = np.asarray(predict(jax.tree.map(jnp.asarray, p), batch))
            np.testing.assert_allclose(actual, out["logits"], rtol=2e-5, atol=3e-5)

    def test_temporal_contrasts_exclude_static_means_and_keep_order(self):
        m = audit.contrast_matrix()
        np.testing.assert_allclose(m.T @ m, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(m.sum(0), 0, atol=1e-6)
        rng = np.random.default_rng(4)
        mixed = rng.normal(size=(2, 16, 4, 112)).astype(np.float32)
        chunks = rng.normal(size=(2, 4, 4, 112)).astype(np.float32)
        before = audit.temporal_contrasts(mixed, chunks)
        np.testing.assert_allclose(before, audit.temporal_contrasts(mixed + 2, chunks - 3), atol=2e-6)
        self.assertGreater(float(np.linalg.norm(before - audit.temporal_contrasts(mixed[:, ::-1], chunks[:, :, ::-1]))), 1)

    def test_group_split_is_disjoint_and_training_only(self):
        groups = np.repeat(np.arange(10), 240)
        labels = np.tile(np.repeat(np.arange(120), 2), 10)
        train = np.flatnonzero(groups < 8)
        fit, dev, chosen = audit.group_dev_split(train, labels, groups, 128)
        self.assertFalse(set(fit) & set(dev))
        self.assertEqual(set(fit) | set(dev), set(train))
        self.assertFalse(set(groups[fit]) & set(groups[dev]))
        self.assertEqual(set(chosen), set(groups[dev]))
        np.testing.assert_array_equal(audit.group_dev_split(train, labels, groups, 128)[0], fit)

    def test_normalization_ignores_unselected_rows_and_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            bank, labels = synthetic_bank(folder)
            fit = np.arange(16)
            mean, std = bank.normalization(fit, "C_temporal")
            expected = bank.inputs(fit, "C_temporal")[2].mean(0)
            np.testing.assert_allclose(mean, expected, atol=1e-6)
            values = np.load(Path(folder) / "temporal.npy", mmap_mode="r+")
            values[16:] = 1e8
            values.flush()
            changed = bank.normalization(fit, "C_temporal")
            np.testing.assert_array_equal(mean, changed[0])
            np.testing.assert_array_equal(std, changed[1])
            batch = bank.batch(np.arange(3), "C_temporal", labels, mean, std, 8)
            np.testing.assert_array_equal(batch["mask"], [1, 1, 1, 0, 0, 0, 0, 0])
            self.assertTrue(np.isfinite(batch["z"]).all())

    def test_parameter_budget_and_paired_direction(self):
        sizes = audit.hidden_sizes()
        core = zero_core()
        b = audit.tree_count(audit.init_probe(core, 454, sizes["B_descriptor"], 1))
        c = audit.tree_count(audit.init_probe(core, 3142, sizes["C_temporal"], 1))
        self.assertLess(abs(b - c) / b, 0.02)
        result = audit.paired_metrics(np.array([1, 0, 1, 0]), np.array([0, 0, 1, 1]),
                                     np.array([0, 0, 1, 1]), np.array([1, 1, 2, 2]), replicates=50)
        self.assertEqual(result["gain_pp"], 50)
        self.assertEqual(result["fixed"], 2)
        self.assertEqual(result["broken"], 0)

    def test_fitting_all_variants_and_exact_epoch_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bank, labels = synthetic_bank(root / "features")
            fit, dev = np.arange(32), np.arange(32, 48)
            store = SimpleNamespace(y=labels)
            core = zero_core()
            original = serialization.msgpack_serialize(core)
            args = args_for(root / "full")
            Path(args.output).mkdir()
            outputs = {}
            for variant in audit.VARIANTS:
                norm = bank.normalization(fit, variant)
                output = audit.train_probe(args, bank, store, fit, dev, core, variant, 128, norm, 0.15)
                result = json.loads((output / "fit_complete.json").read_text())
                history = json.loads((output / "history.json").read_text())
                self.assertLess(result["selected_dev_nll"], history[0]["dev"]["nll"])
                self.assertFalse(result["used_official_evaluation_for_selection"])
                outputs[variant] = output
            self.assertEqual(serialization.msgpack_serialize(core), original)
            args.output = str(root / "resume")
            Path(args.output).mkdir()
            norm = bank.normalization(fit, "B_descriptor")
            write = audit.atomic_bytes
            def interrupt_after_epoch(path, data):
                write(path, data)
                if Path(path).name == "last.msgpack":
                    raise RuntimeError("simulated interruption")
            with patch.object(audit, "atomic_bytes", side_effect=interrupt_after_epoch):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    audit.train_probe(args, bank, store, fit, dev, core, "B_descriptor", 128, norm, 0.15)
            output = audit.train_probe(args, bank, store, fit, dev, core, "B_descriptor", 128, norm, 0.15)
            a = serialization.msgpack_restore((outputs["B_descriptor"] / "selected.msgpack").read_bytes())
            b = serialization.msgpack_restore((output / "selected.msgpack").read_bytes())
            for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
                np.testing.assert_array_equal(left, right)

    def test_identity_conflict_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "identity.json"
            audit.validate_identity(path, {"seed": 128})
            audit.validate_identity(path, {"seed": 128})
            with self.assertRaises(ValueError):
                audit.validate_identity(path, {"seed": 42})


if __name__ == "__main__":
    unittest.main()

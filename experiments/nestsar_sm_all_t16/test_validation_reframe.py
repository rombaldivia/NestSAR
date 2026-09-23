"""CPU checks for the validation-only framing intervention."""
import unittest
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import preprocessing_corrected as pp
from .validation_reframe import cache_location, centered, checkpoint_location, variants


class CenteredWindowTest(unittest.TestCase):
    def test_worker_import_ignores_shadow_package_in_notebook_directory(self):
        checkout = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "experiments"
            fake.mkdir()
            (fake / "__init__.py").write_text("")
            env = dict(os.environ, PYTHONPATH=str(checkout))
            command = [sys.executable, "-m", "experiments.nestsar_sm_all_t16.validation_reframe", "--help"]
            shadowed = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True)
            self.assertNotEqual(shadowed.returncode, 0)
            resolved = subprocess.run(command, cwd=checkout, env=env, capture_output=True, text=True)
            self.assertEqual(resolved.returncode, 0, resolved.stderr)

    def test_checkpoint_pair_is_checked_before_any_missing_cache_is_built(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("experiments.nestsar_sm_all_t16.validation_reframe._mounted_file_candidates",
                       return_value=iter(())):
                with self.assertRaisesRegex(FileNotFoundError, "Attach the saved Kaggle output"):
                    checkpoint_location(root)
            for protocol in ("xsub", "xset"):
                target = root / protocol
                target.mkdir()
                (target / "best.msgpack").write_bytes(b"frozen-checkpoint")
                (target / "best.json").write_text(json.dumps({
                    "model": "NestSAR-SM-ALL-T16-v1", "pipeline_version": "p2-v3",
                    "params": 1_826_556, "preprocessing_version": pp.VERSION}))
            with patch("experiments.nestsar_sm_all_t16.validation_reframe._mounted_file_candidates",
                       return_value=iter(())):
                self.assertEqual(checkpoint_location(root), (root, "p2-v3"))

    def test_existing_g4_pair_is_accepted_with_matching_model_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for protocol in ("xsub", "xset"):
                target = root / protocol
                target.mkdir()
                (target / "best.msgpack").write_bytes(b"frozen-g4-checkpoint")
                (target / "best.json").write_text(json.dumps({
                    "model": "NestSAR-SM-ALL-T16-G4-MOMENTS-v1",
                    "params": 1_827_452, "preprocessing_version": pp.VERSION,
                    "pipeline_version": "sm-all-shared-cache-personaware-p2-g4-moments-v1",
                }))
            with patch("experiments.nestsar_sm_all_t16.validation_reframe._mounted_file_candidates",
                       return_value=iter(())):
                self.assertEqual(checkpoint_location(root), (
                    root, "sm-all-shared-cache-personaware-p2-g4-moments-v1"))
                (root / "xset" / "best.json").write_text(json.dumps({
                    "model": "NestSAR-SM-ALL-T16-v1", "params": 1_826_556,
                    "preprocessing_version": pp.VERSION,
                    "pipeline_version": "sm-all-shared-cache-personaware-p2-g4-moments-v1",
                }))
                with self.assertRaises(FileNotFoundError):
                    checkpoint_location(root)

    def test_only_matching_preprocessing_and_pipeline_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "splits.json").write_text("{}")
            (root / "manifest.json").write_text(json.dumps({"signature": {
                "preprocessing": pp.VERSION, "cache_version": "other-pipeline"}}))
            with patch("experiments.nestsar_sm_all_t16.validation_reframe._mounted_file_candidates",
                       return_value=iter(())):
                self.assertIsNone(cache_location(root, "p2-v3"))
            (root / "manifest.json").write_text(json.dumps({"signature": {
                "preprocessing": pp.VERSION, "cache_version": "p2-v3"}}))
            with patch("experiments.nestsar_sm_all_t16.validation_reframe._mounted_file_candidates",
                       return_value=iter(())):
                self.assertEqual(cache_location(root, "p2-v3"), root)

    def test_short_clip_reuses_original_tokens_and_keeps_missing_actor_zero(self):
        raw = np.zeros((12, 2, 25, 3), np.float32)
        raw[:, 0, :, :] = [1.0, 2.0, 3.0]
        raw[:, 0, 4, 0] += np.arange(len(raw), dtype=np.float32) * 0.03
        canonical = pp.features(raw)
        result = variants(raw, canonical)
        for mode in result:
            np.testing.assert_array_equal(result[mode], canonical)
            np.testing.assert_array_equal(result[mode].reshape(16, 2, 25, 15)[:, 1], 0)

    def test_center_window_excludes_edges_and_changes_tokens(self):
        raw = np.zeros((80, 2, 25, 3), np.float32)
        raw[:, 0, :, :] = [1.0, 2.0, 3.0]
        raw[:, 0, 5, 0] += np.arange(len(raw), dtype=np.float32) * .01
        raw[:8, 0, 5, 1] += np.arange(8, dtype=np.float32)
        self.assertEqual(centered(raw, 32).shape[0], 32)
        np.testing.assert_array_equal(centered(raw, 32)[0], raw[24])
        result = variants(raw, pp.features(raw))
        self.assertFalse(np.array_equal(result["original"], result["center32"]))
        np.testing.assert_array_equal(result["center64"], pp.features(raw[8:72]))


if __name__ == "__main__":
    unittest.main()

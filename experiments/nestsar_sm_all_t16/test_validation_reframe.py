"""CPU checks for the validation-only framing intervention."""
import unittest

import numpy as np

from . import preprocessing_corrected as pp
from .validation_reframe import centered, variants


class CenteredWindowTest(unittest.TestCase):
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

"""Body DCT block (NESTSAR_BODY_DCT). Runs with the switch off: checks the block itself through an explicit k,
and that the default contract (942 features) is untouched."""
import numpy as np

from experiments.nestsar_r5_t16 import preprocessing as pp
from experiments.nestsar_sm_all_t16 import preprocessing_corrected as base_pp


def _clip(T=96, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((T, 2, 25, 3), np.float32)
    x[:, 0] = rng.normal(0, 0.3, (25, 3))
    t = np.arange(T)
    x[:, 0, 7, 0] += 0.2 * np.sin(2 * np.pi * t / 6)        # fast oscillation (hand)
    x[:, 0, 11, 0] += 0.2 * t / T                           # slow drift (other hand)
    return x


def _block(x, k=3):
    local, valid, scale = base_pp.canonicalize_raw(x)
    starts, ends = base_pp.segment_bounds(len(x))
    return pp.body_dct_tokens(local, valid, starts, ends, scale, k=k).reshape(pp.FRAMES, 2, 25, k, 3)


def test_default_contract_unchanged():
    assert pp.BODY_DCT == 0 and pp.FEATURES == pp.BASE_FEATURES == 942
    assert pp.features(_clip()).shape == (pp.FRAMES, 942)


def test_dct_matrix_orthonormal():
    c = pp.dct_matrix()
    np.testing.assert_allclose(c @ c.T, np.eye(pp.DCT_POINTS), atol=1e-12)


def test_oscillation_vs_drift():
    d = _block(_clip())
    osc, drift = np.abs(d[5, 0, 7, :, 0]), np.abs(d[5, 0, 11, :, 0])
    assert osc[1:].sum() > 0.5 * osc[0]           # oscillation puts energy above k = 1
    assert drift[1:].sum() < 0.2 * drift[0]       # a ramp lives almost only in k = 1


def test_static_absent_and_short():
    d = _block(_clip())
    assert np.abs(d[:, 0, 3]).max() < 1e-6       # a still joint has no within-segment motion
    assert np.abs(d[:, 1]).max() == 0            # absent second person
    assert np.abs(_block(_clip(T=5))).max() < 1e-6   # one-frame segments are constant


def test_invalid_frames_do_not_leak_zeros():
    x = _clip()
    x[40:44, 0, 3] = 0                              # joint 3 missing for 4 frames: filled, not a jump to 0
    assert np.abs(_block(x)[:, 0, 3]).max() < 1e-6

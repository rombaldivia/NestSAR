"""Configurable segment count (NESTSAR_FRAMES): T24/T32 keep the token contract per segment and the params."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PROBE = r"""
import json, numpy as np, jax, jax.numpy as jnp
from experiments.nestsar_r5_t16 import preprocessing as r5pp
from experiments.nestsar_sm_all_t16 import preprocessing_corrected as pp
from experiments.nestsar_r5_t16.config import model_kwargs, validate_config
from experiments.nestsar_r5_t16.model import NestSARR5T16
from experiments.nestsar_r5_t16.worker import expected_params
rng = np.random.default_rng(0)
raw = rng.normal(0, .2, (2, 90, 25, 3)).astype(np.float32)                    # M,T,V,C, 90 raw frames
tok = r5pp.features(pp.ordered_raw(raw))
m = NestSARR5T16(**model_kwargs(validate_config({"variant": "graph_mr"})))
x = jnp.asarray(tok)[None]
p = m.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x, training=False)["params"]
out = m.apply({"params": p}, x, training=False)
print(json.dumps({"frames": pp.FRAMES, "shape": list(tok.shape), "finite": bool(np.isfinite(tok).all()),
                  "params": int(sum(a.size for a in jax.tree.leaves(p))), "expected": expected_params("graph_mr"),
                  "logits": list(out["logits"].shape), "version": pp.VERSION}))
"""


@pytest.mark.parametrize("frames", [16, 24, 32])
def test_segment_count(frames):
    env = dict(os.environ, NESTSAR_FRAMES=str(frames), JAX_PLATFORMS="cpu", PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-c", PROBE], env=env, cwd=REPO, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["frames"] == frames and info["shape"] == [frames, 942] and info["finite"]
    assert info["params"] == info["expected"] and info["logits"] == [1, 120]
    assert info["version"].endswith("v2" if frames == 16 else f"-T{frames}")


def test_invalid_segment_count_rejected():
    env = dict(os.environ, NESTSAR_FRAMES="20", JAX_PLATFORMS="cpu", PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-c", "import experiments.nestsar_sm_all_t16.preprocessing_corrected"],
                       env=env, cwd=REPO, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "NESTSAR_FRAMES" in r.stderr

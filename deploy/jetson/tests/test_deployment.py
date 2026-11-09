"""Host-side regression tests. Real CUDA validation also runs on the Jetson."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import helper as H

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "deploy/jetson"))
from prepare_models import Lowerer, session
from runner import load_manifest, verify, worker


@pytest.mark.parametrize("equation,shapes", [
    ("uv,nvc->nuc", [(5, 5), (2, 5, 3)]),
    ("btpv,qv->btpq", [(1, 2, 2, 5), (4, 5)]),
    ("btpvd,qv->btpqd", [(1, 2, 2, 5, 3), (4, 5)]),
    ("dnh,dhk->dnk", [(2, 3, 4), (2, 4, 5)]),
    ("br,brd->bd", [(1, 3), (1, 3, 5)]),
    ("br,bd->brd", [(1, 3), (1, 5)]),
    ("...qhd,...khd->...hqk", [(1, 3, 2, 4), (1, 5, 2, 4)]),
    ("...hqk,...khd->...qhd", [(1, 2, 3, 5), (1, 5, 2, 4)]),
    ("ncuv,nctv->nctu", [(1, 2, 3, 3), (1, 2, 4, 3)]),
])
def test_lowering_preserves_einsum_axis_order(equation, shapes):
    arrays = [np.random.default_rng(i).normal(size=s).astype(np.float32) for i, s in enumerate(shapes)]
    expected = np.einsum(equation, *arrays)
    inputs = [H.make_tensor_value_info(n, onnx.TensorProto.FLOAT, s) for n, s in zip(["a", "b"], shapes)]
    graph = H.make_graph([H.make_node("Einsum", ["a", "b"], ["y"], equation=equation)], "test", inputs,
                         [H.make_tensor_value_info("y", onnx.TensorProto.FLOAT, expected.shape)])
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 23)], ir_version=10)
    lowered = Lowerer(model).run()
    actual = session(lowered.SerializeToString()).run(None, dict(zip(["a", "b"], arrays)))[0]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_cuda_reference_check_rejects_wrong_output(tmp_path):
    np.savez(tmp_path / "model.reference.npz", x=np.zeros((1, 1, 2, 3), np.float32), y=np.ones((1, 1, 2), np.float32))
    class WrongEngine:
        def run(self, x):
            return np.zeros((1, 2), np.float32)
    with pytest.raises(AssertionError):
        verify(WrongEngine(), tmp_path, "model", {}, "fp32")


def test_static_batch_one_required(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"model": {"input_shape": [2, 16, 942]}}))
    with pytest.raises(ValueError, match="batch 1"):
        load_manifest(tmp_path)


def test_untrained_predictions_are_blocked_before_cuda(tmp_path):
    from types import SimpleNamespace
    payload = b"not an engine"
    (tmp_path / "model.onnx").write_bytes(payload)
    (tmp_path / "manifest.json").write_text(json.dumps({"model": {
        "input_shape": [1, 16, 942], "trained": False,
        "onnx_sha256": hashlib.sha256(payload).hexdigest()}}))
    with pytest.raises(ValueError, match="pesos sin entrenar"):
        worker(SimpleNamespace(models=tmp_path, only=["model"], command="predict"))


@pytest.mark.parametrize("release,image", [
    ("# R32 (release), REVISION: 7.6, GCID: 123", "l4t-tensorrt:r8.2.1-runtime"),
    ("# R36 (release), REVISION: 4.3, GCID: 123", "l4t-jetpack:r36.4.0"),
])
def test_detects_matching_cuda_base(tmp_path, release, image):
    import os
    path = tmp_path / "nv_tegra_release"
    path.write_text(release)
    env = dict(os.environ, NESTSAR_L4T_RELEASE_FILE=str(path))
    result = subprocess.run([str(ROOT / "jetson.sh"), "--dry-run", "benchmark", "--only", "nestsar"],
                            env=env, text=True, capture_output=True, check=True)
    assert image in result.stdout
    assert "--runtime nvidia" in result.stdout


def test_unknown_jetpack_fails_instead_of_guessing(tmp_path):
    import os
    path = tmp_path / "release"
    path.write_text("# R35 (release), REVISION: 4.1")
    result = subprocess.run([str(ROOT / "jetson.sh"), "--dry-run"],
                            env=dict(os.environ, NESTSAR_L4T_RELEASE_FILE=str(path)), capture_output=True)
    assert result.returncode == 2

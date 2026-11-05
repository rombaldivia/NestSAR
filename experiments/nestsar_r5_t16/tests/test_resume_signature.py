"""A run must resume after its run_config.json round-trips through JSON (tuples in the config become lists)."""
import json

from experiments.nestsar_r5_t16.config import validate_config


def test_signature_survives_json_roundtrip():
    cfg = validate_config({"variant": "graph_mr"})
    assert isinstance(cfg["fr_level_weights"], tuple)          # the case that broke resuming
    signature = {"config": cfg, "protocol": "xset"}
    stored = json.loads(json.dumps(signature))                 # what read_json returns
    assert stored != signature                                  # the old comparison
    assert stored == json.loads(json.dumps(signature))          # the comparison worker.run now makes


def test_worker_uses_json_form_comparison():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "worker.py").read_text()
    assert "previous != json.loads(json.dumps(signature))" in src


def test_per_class_report_skips_ntu120_references_on_ntu60():
    """NTU-60 xsub crashed after the last epoch: the NTU-120 weak-class table indexes classes >= 60."""
    import subprocess, sys, os
    from pathlib import Path
    code = ("from experiments.nestsar_r5_t16 import worker as w\n"
            "import inspect\n"
            "src = inspect.getsource(w.per_class_report)\n"
            "assert 'spec.DATASET == \"ntu120\"' in src\n"
            "assert w.NUM_CLASSES == 60\n"
            "assert max(int(k[1:]) for k in w.R4_WEAK_CLASS_RECALL['xsub']) > 60  # the table that broke it\n")
    repo = Path(__file__).resolve().parents[3]
    env = dict(os.environ, NESTSAR_DATASET="ntu60", JAX_PLATFORMS="cpu", PYTHONPATH=str(repo))
    r = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]

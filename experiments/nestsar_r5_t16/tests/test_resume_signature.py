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

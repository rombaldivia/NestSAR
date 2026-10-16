import json

import numpy as np
import pytest

from experiments.nestsar_r5_t16.review_weak_classes import class_report, discover_runs, find_cache, input_contract


def test_oracle_complementarity_and_per_class_errors():
    def probabilities(pred):
        return np.eye(120, dtype=np.float32)[pred]
    result = dict(y=np.array([0, 0, 1, 1, 2, 2]),
                  main=probabilities([0, 1, 1, 2, 1, 2]),
                  aux=np.stack([probabilities([1, 0, 1, 2, 1, 2]),
                                probabilities([0, 1, 0, 1, 0, 1])], axis=1),
                  identity={"protocol": "xsub"}, config={"variant": "graph_mr_pt", "seed": 128},
                  checkpoint_accuracy=0.5)
    report = class_report(result)
    assert report["top1"] == 0.5 and report["reproduced"] is True
    assert report["main_errors_correct_in_any_aux"] == 2
    assert report["errors_wrong_in_every_head"] == 1
    assert report["existing_heads_oracle"] == pytest.approx(5 / 6)
    assert sum(c["errors"] for c in report["classes"]) == 3
    by_class = {r["class_id"]: r for r in report["classes"]}
    assert by_class["A001"]["aux"]["m4"]["main_errors_corrected"] == 1
    assert by_class["A002"]["aux"]["hand"]["main_errors_corrected"] == 1
    assert by_class["A003"]["errors_wrong_in_every_head"] == 1
    json.dumps(report, allow_nan=False)


def test_input_contract_and_discovery_do_not_select_a_higher_frame_run(tmp_path):
    for name, prep in [("t16", "r5-hand-v2"), ("t24", "r5-hand-v2-T24"),
                       ("dct", "r5-hand-v2+bodydct2x8-v1")]:
        out = tmp_path / name / "xsub"
        out.mkdir(parents=True)
        record = dict(config={"variant": "graph_mr_pt"}, cache={"preprocessing": prep})
        (out / "run_config.json").write_text(json.dumps(record))
        (out / "best.msgpack").write_bytes(b"saved checkpoint")
    assert {p.name for p in discover_runs([tmp_path])} == {"t16", "dct"}
    assert input_contract({"cache": {"preprocessing": "r5-v2-T32+bodydct3x8-v1"}}) == (32, 3)


def test_cache_discovery_uses_the_recorded_signature(tmp_path):
    signature = {"preprocessing": "current", "source": "wanted"}
    for name, source in [("wrong", "other"), ("matching", "wanted")]:
        root = tmp_path / name
        root.mkdir()
        (root / "manifest.json").write_text(json.dumps({"signature": dict(signature, source=source)}))
    assert find_cache(signature, None, [tmp_path]).name == "matching"
    with pytest.raises(FileNotFoundError):
        find_cache({"source": "missing"}, None, [tmp_path])

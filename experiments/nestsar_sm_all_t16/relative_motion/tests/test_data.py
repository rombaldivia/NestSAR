import hashlib
import json
import pickle
import shutil
from pathlib import Path
import numpy as np
import pytest
from experiments.nestsar_sm_all_t16.streaming.data import prepare as prepare_p2
from experiments.nestsar_sm_all_t16.streaming.launch import validate_config as training_config
from experiments.nestsar_sm_all_t16.streaming.io_utils import read_json, atomic_json
from experiments.nestsar_sm_all_t16.relative_motion import data, preprocessing as pp
from experiments.nestsar_sm_all_t16.relative_motion.config import validate_config
from experiments.nestsar_sm_all_t16.relative_motion.launch import paired_results
from experiments.nestsar_sm_all_t16.relative_motion.smoke_cpu import synthetic_cache
from .test_preprocessing import clip


def small_cache(tmp_path):
    samples = [dict(frame_dir=f"id{i}", label=i, keypoint=clip(16+i).transpose(1, 0, 2, 3)) for i in range(5)]
    splits = {f"{p}_{part}": ids for p in ("xsub", "xset") for part, ids in
              (("train", ["id0", "id1", "id2"]), ("val", ["id3", "id4"]))}
    path = tmp_path / "source.pkl"
    path.write_bytes(pickle.dumps(dict(annotations=samples, split=splits)))
    prepare_p2(path, tmp_path / "p2", tmp_path / "prepare.json")
    return tmp_path / "p2"


def test_auxiliary_resume_integrity_batch_equivalence_and_no_source_writes(tmp_path, monkeypatch):
    cache = small_cache(tmp_path)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.iterdir() if p.is_file()}
    aux = tmp_path / "relative"
    function = pp.relative_path
    calls = []
    def interrupted(x):
        calls.append(len(x))
        if len(calls) == 4:
            raise KeyboardInterrupt("synthetic interruption")
        return function(x)
    with monkeypatch.context() as patch:
        patch.setattr(pp, "relative_path", interrupted)
        with pytest.raises(KeyboardInterrupt):
            data.prepare(cache, aux, tmp_path / "status.json", chunk=2, reserve_bytes=0)
    assert read_json(aux / "progress.json")["completed"] == 2
    data.prepare(cache, aux, tmp_path / "status.json", chunk=2, reserve_bytes=0)
    paths, meta = data.open_paths(cache, aux)
    source = data.P2Dataset(cache)
    for i in range(5):
        np.testing.assert_array_equal(pp.pack(source.canonical[i], paths[i]), pp.features(source.sample(i)))
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.iterdir() if p.is_file()}
    assert isinstance(paths, np.memmap) and not paths.flags.writeable
    data.prepare(cache, aux, tmp_path / "status.json", reserve_bytes=0)
    # Even finite on-disk corruption is rejected at next launcher reuse.
    changed = np.load(aux / "relative_path.npy", mmap_mode="r+")
    changed[2, 0, 0, 7, 0] += 1
    changed.flush()
    with pytest.raises(ValueError, match="Corrupt"):
        data.prepare(cache, aux, tmp_path / "status.json", reserve_bytes=0)


def test_foreign_cache_and_insufficient_disk_are_rejected(tmp_path, monkeypatch):
    cache = small_cache(tmp_path)
    aux = tmp_path / "relative"
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "disk_usage", lambda path: type("Disk", (), {"free": 1})())
        with pytest.raises(OSError, match="free"):
            data.prepare(cache, aux, tmp_path / "status.json")
    assert not (aux / "relative_path.partial.npy").exists()
    atomic_json(aux / "progress.json", {"signature": {"foreign": True}})
    with pytest.raises(ValueError, match="different data"):
        data.prepare(cache, aux, tmp_path / "status.json", reserve_bytes=0)


def test_grouped_full_class_plans_and_official_heldout_guard(tmp_path):
    cache = synthetic_cache(tmp_path)
    aux = tmp_path / "relative"
    data.prepare(cache, aux, tmp_path / "relative_status.json", reserve_bytes=0)
    for protocol in ("xsub", "xset"):
        first = data.split_plan(cache, protocol, 128, min_class_samples=[1, 1, 1])
        assert first == data.split_plan(cache, protocol, 128, min_class_samples=[1, 1, 1])
        second = data.split_plan(cache, protocol, 42, min_class_samples=[1, 1, 1])
        assert first["groups"] != second["groups"]
        source = data.Dataset(cache, aux, first)
        ids = first["indices"]["fit"][:3]
        cfg = training_config({})
        a, _ = source.batch(ids, range(3), 4, cfg, 2, True, protocol)
        b, _ = source.batch(ids, range(3), 4, cfg, 3, True, protocol)
        assert a["mask"].tolist() == [1, 1, 1, 0]
        assert not a["x"][3].any() and not a["xa"][3].any()
        np.testing.assert_array_equal(a["x"], b["x"])
        assert not np.array_equal(a["xa"][:3], b["xa"][:3])
        held = read_json(cache / "splits.json")[protocol + "_val"][0]
        with pytest.raises(ValueError, match="held-out"):
            source.sample(held)
        with pytest.raises(ValueError, match="held-out"):
            source.batch([held], [0], 1, cfg, 0, False, protocol)
        damaged = json.loads(json.dumps(first))
        damaged["indices"]["fit"][0] = held
        damaged["sha256"] = data.digest({k: v for k, v in damaged.items() if k != "sha256"})
        with pytest.raises(ValueError, match="held-out"):
            data.Dataset(cache, aux, damaged)


def test_scores_count_both_corrected_and_damaged_predictions(tmp_path):
    ids, labels = np.arange(4), np.array([0, 0, 1, 1])
    np.savez(tmp_path / "a.npz", indices=ids, labels=labels, predictions=[0, 1, 0, 1])
    np.savez(tmp_path / "b.npz", indices=ids, labels=labels, predictions=[1, 0, 1, 1])
    r = paired_results(tmp_path / "a.npz", tmp_path / "b.npz")
    assert r["corrected"] == 2 and r["damaged"] == 1 and r["delta_pp"] == 25
    np.savez(tmp_path / "b.npz", indices=ids[::-1], labels=labels, predictions=[1, 0, 1, 1])
    with pytest.raises(ValueError, match="different IDs"):
        paired_results(tmp_path / "a.npz", tmp_path / "b.npz")


def test_configuration_prevents_unlabelled_partial_runs():
    with pytest.raises(ValueError, match="caps"):
        validate_config({"training": {"max_train_samples": 10}})
    with pytest.raises(ValueError, match="patience"):
        validate_config({"training": {"patience": 10}})
    c = validate_config({"smoke_test": True})
    assert c["training"]["max_train_samples"] == 3 and c["seeds"] == [128]

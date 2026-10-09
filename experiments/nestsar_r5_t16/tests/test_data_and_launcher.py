import json
import shutil

import numpy as np
import pytest

from experiments.nestsar_r5_t16 import data as r5data
from experiments.nestsar_r5_t16 import launch
from experiments.nestsar_r5_t16 import preprocessing as r5
from experiments.nestsar_r5_t16.config import DEFAULTS, VARIANTS, model_kwargs, validate_config
from experiments.nestsar_r5_t16.smoke_cpu import synthetic_pickle
from experiments.nestsar_sm_all_t16.streaming.data import prepare


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    root = tmp_path_factory.mktemp("r5data")
    synthetic_pickle(root / "ntu120_3danno.pkl")
    prepare(root / "ntu120_3danno.pkl", root / "r4", root / "r4_status.json")
    meta = r5data.build(root / "r4", root / "hand")
    return root, meta


def test_hand_cache_matches_preprocessing(caches):
    root, meta = caches
    assert meta["max_body_token_difference"] == 0.0
    ds = r5data.Dataset(root / "hand")
    idx = np.arange(len(ds.labels))
    x = ds.canonical(idx)
    assert x.shape == (len(idx), 16, r5.FEATURES)
    for i in idx:
        np.testing.assert_array_equal(x[i], r5.features(ds.base.sample(i)))


def test_batches_shapes_padding_and_augmentation(caches):
    root, _ = caches
    ds = r5data.Dataset(root / "hand")
    cfg = validate_config({"micro_batch": 2, "accumulation_steps": 2})
    ids = ds.splits["xsub_train"]
    batches = list(ds.batches(ids, 4, cfg, epoch=3, training=True, protocol="xsub"))
    assert len(batches) == 3
    total = 0
    for b, _ in batches:
        assert b["x"].shape == (4, 16, r5.FEATURES) and b["xa"].shape == b["x"].shape
        n = int(b["mask"].sum())
        total += n
        assert np.abs(b["x"][n:]).max(initial=0) == 0 and np.abs(b["xa"][n:]).max(initial=0) == 0
        assert np.isfinite(b["xa"]).all()
        assert not np.array_equal(b["x"][:n], b["xa"][:n])
    assert total == len(ids)
    # Same seed/epoch -> identical augmented view (reproducible resume).
    again = list(ds.batches(ids, 4, cfg, epoch=3, training=True, protocol="xsub"))
    for (a, _), (b, _) in zip(batches, again):
        np.testing.assert_array_equal(a["xa"], b["xa"])


def test_rebuild_is_a_validation_and_mismatch_is_refused(caches, tmp_path):
    root, meta = caches
    assert r5data.build(root / "r4", root / "hand")["files"] == meta["files"]
    other = tmp_path / "r4_other"
    shutil.copytree(root / "r4", other)
    m = json.loads((other / "manifest.json").read_text())
    m["signature"]["source_sha256"] = "0" * 64
    (other / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="another base cache"):
        r5data.build(other, root / "hand")


def test_corrupted_base_tokens_stop_the_build(caches, tmp_path):
    root, _ = caches
    bad = tmp_path / "r4_bad"
    shutil.copytree(root / "r4", bad)
    canonical = np.load(bad / "canonical.npy", mmap_mode="r+")
    canonical[2, 5, 100] += 0.5
    canonical.flush()
    del canonical
    with pytest.raises(ValueError, match="differ"):
        r5data.build(bad, tmp_path / "hand_bad")


def test_moved_base_cache_is_found_by_hint(caches, tmp_path):
    root, _ = caches
    moved = tmp_path / "moved_r4"
    shutil.copytree(root / "r4", moved)
    hand = tmp_path / "hand_copy"
    shutil.copytree(root / "hand", hand)
    m = json.loads((hand / "manifest.json").read_text())
    m["base_dir"] = str(tmp_path / "does_not_exist")
    (hand / "manifest.json").write_text(json.dumps(m))
    ds = r5data.Dataset(hand, base=moved)
    assert ds.base_dir == moved
    assert r5data.candidate_r4_caches([tmp_path]) and moved in r5data.candidate_r4_caches([tmp_path])


def test_kill_rule():
    assert launch.kill_decision({"xsub": 0.70, "xset": 0.71}, {"xsub": None, "xset": None}, 1.0) == (None, {})
    assert launch.kill_decision({"xsub": None, "xset": 0.7}, {"xsub": 0.7, "xset": 0.7}, 1.0) == (None, {})
    d, deltas = launch.kill_decision({"xsub": 0.680, "xset": 0.689}, {"xsub": 0.70, "xset": 0.70}, 1.0)
    assert d == "KILL" and deltas["xsub"] == pytest.approx(-2.0)
    d, _ = launch.kill_decision({"xsub": 0.680, "xset": 0.695}, {"xsub": 0.70, "xset": 0.70}, 1.0)
    assert d == "CONTINUE"        # XSET only 0.5 pp behind
    d, _ = launch.kill_decision({"xsub": 0.68}, {"xsub": 0.70, "xset": 0.71}, 1.0)
    assert d == "KILL"            # single-protocol run


def test_reference_discovery(tmp_path):
    meta = launch.meta()
    good = tmp_path / "a" / meta.PREFERRED_REFERENCE
    other = tmp_path / "b" / "SomethingElse"
    for d, model in ((good, meta.REFERENCE_MODEL), (other, "Other")):
        (d / "xsub").mkdir(parents=True)
        (d / "xsub" / "run_config.json").write_text(json.dumps(
            {"model": model, "parameters": meta.REFERENCE_PARAMS, "config": {"epochs": 60, "micro_batch": 64,
                                                                             "accumulation_steps": 4}}))
        (d / "xsub" / "history.json").write_text(json.dumps([{"epoch": 10, "val_acc": 0.66}]))
    found = launch.find_reference(None, [tmp_path])
    assert found == str(good)
    cfg = validate_config({})
    assert launch.reference_config_issues(found, cfg) == []
    cfg2 = validate_config({"micro_batch": 32, "accumulation_steps": 4})
    assert any("effective batch" in s for s in launch.reference_config_issues(found, cfg2))


def test_config_validation():
    c = validate_config({})
    assert c["variant"] == "full" and c["epochs"] == 60 and c["micro_batch"] * c["accumulation_steps"] == 256
    for v in VARIANTS:
        kw = model_kwargs(validate_config({"variant": v}))
        assert kw["model_dim"] == 176
    with pytest.raises(ValueError):
        validate_config({"variant": "nope"})
    with pytest.raises(ValueError):
        validate_config({"model_dim": 192})
    with pytest.raises(ValueError):
        validate_config({"geometry_desc_weight": 0.1})
    assert "controller_dim" not in DEFAULTS and "head_rank" not in DEFAULTS


import fcntl
import os
import subprocess
import sys
import time


def test_alive_ignores_zombies_and_foreign_pids():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.time() + 10
    while time.time() < deadline:
        state = open(f"/proc/{child.pid}/stat").read().rsplit(")", 1)[1].split()[0]
        if state == "Z":
            break
        time.sleep(0.05)
    assert not launch.alive(child.pid)            # exited but not yet reaped
    child.wait()
    assert launch.alive(os.getpid())
    assert not launch.alive(os.getpid(), "nestsar_r5_t16.worker")
    assert not launch.alive(None) and not launch.alive("x")


def _launch_with_fake_worker(tmp_path, mode, monkeypatch, kill=False):
    out = tmp_path / f"run_{mode}"
    out.mkdir()
    (out / "fake_plan.json").write_text(json.dumps({"mode": mode}))
    meta = launch.meta()
    ref = tmp_path / f"ref_{mode}" / meta.PREFERRED_REFERENCE
    for p in ("xsub", "xset"):
        (ref / p).mkdir(parents=True)
        (ref / p / "run_config.json").write_text(json.dumps(
            {"model": meta.REFERENCE_MODEL, "parameters": meta.REFERENCE_PARAMS, "config": {}}))
        (ref / p / "history.json").write_text(json.dumps([{"epoch": 1, "val_acc": 0.90}]))
    monkeypatch.setattr(launch, "WORKER", "experiments.nestsar_r5_t16.tests.fake_worker")
    args = ["--cache", str(tmp_path), "--outdir", str(out), "--cpu-smoke", "--micro-batch", "64",
            "--poll-seconds", "1", "--kill-epoch", "1", "--working-root", str(tmp_path / f"ref_{mode}")]
    if not kill:
        args += ["--reference-dir", str(tmp_path / "nowhere")]
    launch.main(args)
    return out


def test_transient_worker_crash_is_restarted(tmp_path, monkeypatch, capsys):
    out = _launch_with_fake_worker(tmp_path, "crash_once", monkeypatch)
    log = capsys.readouterr().out
    for p in ("xsub", "xset"):
        assert (out / p / "result.json").exists()
        assert (out / p / "attempts").read_text() == "2"
    assert "restarted (attempt 1/2)" in log


def test_deterministic_failure_is_not_retried(tmp_path, monkeypatch, capsys):
    out = _launch_with_fake_worker(tmp_path, "nan", monkeypatch)
    log = capsys.readouterr().out
    for p in ("xsub", "xset"):
        assert not (out / p / "result.json").exists()
        assert (out / p / "attempts").read_text() == "1"
    assert "not restarted" in log


def test_kill_rule_stops_workers(tmp_path, monkeypatch):
    out = _launch_with_fake_worker(tmp_path, "kill", monkeypatch, kill=True)
    decision = json.loads((out / "kill_decision.json").read_text())
    assert decision["decision"] == "KILL" and decision["delta_pp"]["xsub"] == pytest.approx(-80.0)
    pids = json.loads((out / "pids.json").read_text())
    deadline = time.time() + 15
    while time.time() < deadline and any(launch.alive(pid) for pid in pids.values()):
        time.sleep(0.2)
    assert not any(launch.alive(pid) for pid in pids.values())
    # Re-launching a killed run starts nothing.
    launch.main(["--cache", str(tmp_path), "--outdir", str(out), "--cpu-smoke", "--micro-batch", "64"])
    assert json.loads((out / "pids.json").read_text()) == pids


def test_second_launcher_is_refused(tmp_path, capsys):
    out = tmp_path / "locked"
    out.mkdir()
    with (out / "launch.lock").open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch.main(["--cache", str(tmp_path), "--outdir", str(out), "--cpu-smoke"])
    assert "Another launcher" in capsys.readouterr().out
    assert not (out / "pids.json").exists()


def test_read_only_complete_hand_cache_is_accepted(caches, tmp_path, monkeypatch):
    root, meta = caches
    ro = tmp_path / "ro_hand"
    shutil.copytree(root / "hand", ro)
    for name in ("build.lock", "build_status.json"):
        (ro / name).unlink(missing_ok=True)
    m = json.loads((ro / "manifest.json").read_text())
    m["base_dir"] = "/somewhere/else"
    (ro / "manifest.json").write_text(json.dumps(m))

    def read_only(*_args, **_kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(r5data, "atomic_json", read_only)
    before = sorted(p.name for p in ro.iterdir())
    got = r5data.build(root / "r4", ro)
    assert got["files"] == meta["files"]
    assert sorted(p.name for p in ro.iterdir()) == before          # nothing created
    ds = r5data.Dataset(ro, base=root / "r4")                      # base found by hint
    assert ds.base_dir == (root / "r4")

"""Regression coverage for Kaggle GPU bootstrap, reporting and dual-worker launch."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json
from experiments.nestsar_sm_all_t16.streaming import launch
from experiments.nestsar_sm_all_t16.streaming.launch import (
    best_score_snapshot,
    check_worker_finished,
    discover_gpus,
    ensure_runtime,
    update_bar,
)


class Bar:
    def reset(self, total):
        self.total = total

    def set_description_str(self, value, **kw):
        self.description = value

    def set_postfix(self, values, **kw):
        self.values = values

    def refresh(self):
        pass

    def close(self):
        self.closed = True


def test_discover_gpus_uses_jax_child_and_not_nvidia_smi(tmp_path, monkeypatch):
    payload = [
        {"id": 0, "platform": "gpu", "kind": "Tesla T4"},
        {"id": 1, "platform": "gpu", "kind": "Tesla T4"},
    ]

    def run_command(cmd, **kwargs):
        assert cmd[0] == sys.executable
        assert cmd[1] == "-c"
        assert "nvidia-smi" not in cmd
        env = kwargs["env"]
        assert "CUDA_VISIBLE_DEVICES" not in env
        assert "JAX_PLATFORMS" not in env
        return SimpleNamespace(
            returncode=0,
            stdout="NESTSAR_GPU_DISCOVERY=" + json.dumps(payload) + "\n",
            stderr="",
        )

    monkeypatch.setattr(launch.subprocess, "run", run_command)
    gpus, devices = discover_gpus(sys.executable, tmp_path / "gpu.log")
    assert gpus == ["0", "1"]
    assert devices == payload


def test_discover_gpus_rejects_single_gpu(tmp_path, monkeypatch):
    payload = [{"id": 0, "platform": "gpu", "kind": "Tesla T4"}]

    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout="NESTSAR_GPU_DISCOVERY=" + json.dumps(payload) + "\n",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="needs two visible GPUs"):
        discover_gpus(sys.executable, tmp_path / "gpu.log")


def test_runtime_uses_host_python_and_removes_legacy_runtime(tmp_path, monkeypatch):
    stale = tmp_path / "runtime"
    stale.mkdir()
    (stale / "huge_cuda_wheel").write_text("old")
    (stale / "pyvenv.cfg").write_text("home = /usr/bin\n")

    monkeypatch.setattr(launch, "runtime_probe", lambda *a, **k: True)

    bars = [Bar(), Bar()]
    python = ensure_runtime(tmp_path, bars, "0")

    assert python == sys.executable
    assert not stale.exists()
    assert bars[0].description.startswith("XSUB")
    assert bars[1].description.startswith("XSET")


def test_runtime_failure_never_installs_or_creates_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "runtime_probe", lambda *a, **k: False)

    with pytest.raises(RuntimeError, match="No packages were installed automatically"):
        ensure_runtime(tmp_path, [Bar(), Bar()], "0")

    assert not (tmp_path / "runtime").exists()
    assert not (tmp_path / "install.log").exists()


@pytest.mark.parametrize("ready_venv", [False, True])
def test_host_reuse_preserves_unknown_directories_and_ready_venvs(tmp_path, monkeypatch, ready_venv):
    stale = tmp_path / "runtime"
    stale.mkdir()
    (stale / "keep.txt").write_text("keep")
    if ready_venv:
        (stale / "pyvenv.cfg").write_text("home = /usr/bin\n")
        atomic_json(stale / "nestsar_runtime.json", {"state": "ready"})
    monkeypatch.setattr(launch, "runtime_probe", lambda *a, **k: True)
    assert ensure_runtime(tmp_path, [], "0") == sys.executable
    assert (stale / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize(
    "protocol,score,epoch",
    [("xsub", .754335, 40), ("xset", .761773, 31)],
)
def test_best_is_visible_for_each_protocol_and_all_phases(protocol, score, epoch):
    bar = Bar()
    for phase in ("Train", "Validate EMA", "Save checkpoint", "Done"):
        update_bar(
            bar,
            protocol,
            0,
            dict(phase=phase, epoch=45, best=score, best_epoch=epoch),
        )
        assert list(bar.values)[0] == "BEST"
        assert bar.values["BEST"] == f"{100*score:.4f}%@E{epoch:02d}"
        assert protocol.upper() in bar.description


def test_no_invented_best_before_first_completed_validation():
    bar = Bar()
    update_bar(
        bar,
        "xsub",
        0,
        dict(phase="Validate EMA", val_acc=1.0, best=None, best_epoch=0),
    )
    assert bar.values["BEST"] == "--"
    assert best_score_snapshot({"xsub": {"val_acc": 1.0}})["xsub"]["best_val_accuracy"] is None


def test_snapshot_keeps_protocols_separate_and_ignores_live_accuracy():
    statuses = {
        "xsub": dict(best=.75, best_epoch=4, completed_epoch=7, val_acc=.99),
        "xset": dict(best=.77, best_epoch=6, completed_epoch=6, val_acc=.98),
    }
    scores = best_score_snapshot(statuses)
    assert scores["xsub"]["best_val_percent"] == 75.0
    assert scores["xset"]["best_val_percent"] == 77.0
    assert scores["xsub"]["best_epoch"] == 4
    assert scores["xset"]["best_epoch"] == 6


def test_worker_completion_race_reads_final_score(tmp_path):
    stale = dict(phase="Validate EMA", done=False, best=.7, best_epoch=1)
    final = dict(phase="Done", done=True, best=.76, best_epoch=2, completed_epoch=3)

    class Process:
        def poll(self):
            atomic_json(tmp_path / "xsub" / "status.json", final)
            return 0

    done, status = check_worker_finished(Process(), "xsub", tmp_path, stale)
    assert done and status == final


def test_worker_failure_is_reported_with_log(tmp_path):
    (tmp_path / "xset.log").write_text("test failure detail\n")

    class Process:
        def poll(self):
            return 9

    with pytest.raises(RuntimeError, match="test failure detail"):
        check_worker_finished(Process(), "xset", tmp_path, {})


def test_parent_launches_both_isolated_protocols_and_keeps_two_bars(tmp_path, monkeypatch):
    dataset = tmp_path / "ntu.pkl"
    dataset.write_bytes(b"fixture; preparation tested separately")
    out = tmp_path / "out"

    bars = [Bar(), Bar()]
    started = []

    monkeypatch.setattr(launch, "make_bars", lambda: bars)
    monkeypatch.setattr(
        launch,
        "discover_gpus",
        lambda *a: (
            ["0", "1"],
            [
                {"id": 0, "platform": "gpu", "kind": "Tesla T4"},
                {"id": 1, "platform": "gpu", "kind": "Tesla T4"},
            ],
        ),
    )
    monkeypatch.setattr(launch, "optional_nvidia_smi", lambda: None)
    monkeypatch.setattr(launch, "ensure_runtime", lambda *a: sys.executable)
    monkeypatch.setattr(launch, "quiet_run", lambda *a, **k: None)

    def run_command(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout="test-sha\n", stderr="")

    monkeypatch.setattr(launch.subprocess, "run", run_command)

    class Process:
        def poll(self):
            return 0

        def wait(self):
            return 0

    def spawn(cmd, **kwargs):
        protocol = cmd[cmd.index("--protocol") + 1]
        started.append((protocol, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        assert kwargs["env"]["JAX_PLATFORMS"] == "cuda"
        assert Path(kwargs["env"]["PYTHONPATH"]).joinpath(
            "experiments/nestsar_sm_all_t16/model.py"
        ).is_file()
        assert cmd[cmd.index("-m") + 1] == "experiments.nestsar_sm_all_t16.streaming.worker"

        best = .76 if protocol == "xsub" else .78
        atomic_json(
            out / protocol / "status.json",
            dict(done=True, phase="Done", best=best, best_epoch=2, completed_epoch=3),
        )
        atomic_json(
            out / protocol / "result.json",
            dict(best_val_accuracy=best, best_epoch=2),
        )
        return Process()

    monkeypatch.setattr(launch.subprocess, "Popen", spawn)

    results = launch.run(dataset=dataset, outdir=out, cache_dir=tmp_path / "cache")

    assert started == [("xsub", "0"), ("xset", "1")]
    assert results["xsub"]["best_val_accuracy"] == .76
    assert results["xset"]["best_val_accuracy"] == .78
    assert bars[0].values["BEST"] == "76.0000%@E02"
    assert bars[1].values["BEST"] == "78.0000%@E02"
    assert all(bar.closed for bar in bars)
    assert json.loads((out / "best_scores.json").read_text())["xset"]["best_val_percent"] == 78

    hardware = json.loads((out / "hardware.json").read_text())
    assert hardware["gpu_discovery"] == "jax_subprocess"
    assert hardware["assignment"] == {"xsub": "0", "xset": "1"}
    assert hardware["nvidia_smi"] is None

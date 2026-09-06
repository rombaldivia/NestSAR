"""Regression coverage for the reported Kaggle ensurepip failure and best scores."""
import json
import subprocess
import venv
import zipfile

import pytest

from nestsar_fixed.io_utils import atomic_json
from nestsar_fixed.launch import (
    best_score_snapshot, check_worker_finished, create_runtime_without_pip,
    install_into_runtime, update_bar,
)


def test_recover_partial_runtime_and_install_without_ensurepip(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("ensurepip must never run")
    monkeypatch.setattr(venv.EnvBuilder, "_setup_pip", forbidden)
    runtime = tmp_path / "runtime"
    python = create_runtime_without_pip(runtime)
    assert subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True).returncode != 0
    sentinel = runtime / "already_installed.txt"
    sentinel.write_text("keep")
    (runtime / "bin" / "activate").unlink()  # Mimic interrupted EnvBuilder setup.
    python = create_runtime_without_pip(runtime)
    assert sentinel.read_text() == "keep" and (runtime / "bin" / "activate").is_file()
    wheel = tmp_path / "nestsar_bootstrap_probe-0.0.1-py3-none-any.whl"
    info = "nestsar_bootstrap_probe-0.0.1.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("nestsar_bootstrap_probe.py", "VALUE = 42\n")
        archive.writestr(f"{info}/METADATA", "Metadata-Version: 2.1\nName: nestsar-bootstrap-probe\nVersion: 0.0.1\n")
        archive.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nGenerator: regression\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{info}/RECORD", "")
    install_into_runtime(python, [str(wheel)], tmp_path / "install.log",
                         extra_options=("--no-index", "--no-deps"))
    result = subprocess.run([str(python), "-c",
        "import sys,nestsar_bootstrap_probe as p; assert p.VALUE==42; assert p.__file__.startswith(sys.prefix); print('PASS')"],
        capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "PASS"


class Bar:
    def reset(self, total):
        self.total = total
    def set_description_str(self, value, **kw):
        self.description = value
    def set_postfix(self, values, **kw):
        self.values = values
    def refresh(self):
        pass


@pytest.mark.parametrize("protocol,score,epoch", [("xsub", .754335, 40), ("xset", .761773, 31)])
def test_best_is_visible_for_each_protocol_and_all_phases(protocol, score, epoch):
    bar = Bar()
    for phase in ("Train", "Validate EMA", "Save checkpoint", "Done"):
        update_bar(bar, protocol, 0, dict(phase=phase, epoch=45, best=score, best_epoch=epoch))
        assert list(bar.values)[0] == "BEST"
        assert bar.values["BEST"] == f"{100*score:.4f}%@E{epoch:02d}"
        assert protocol.upper() in bar.description


def test_no_invented_best_before_first_completed_validation():
    bar = Bar()
    update_bar(bar, "xsub", 0, dict(phase="Validate EMA", val_acc=1.0, best=None, best_epoch=0))
    assert bar.values["BEST"] == "--"
    assert best_score_snapshot({"xsub": {"val_acc": 1.0}})["xsub"]["best_val_accuracy"] is None


def test_snapshot_keeps_protocols_separate_and_ignores_live_accuracy():
    statuses = {"xsub": dict(best=.75, best_epoch=4, completed_epoch=7, val_acc=.99),
                "xset": dict(best=.77, best_epoch=6, completed_epoch=6, val_acc=.98)}
    scores = best_score_snapshot(statuses)
    assert scores["xsub"]["best_val_percent"] == 75.0
    assert scores["xset"]["best_val_percent"] == 77.0
    assert scores["xsub"]["best_epoch"] == 4
    assert scores["xset"]["best_epoch"] == 6


def test_worker_completion_race_reads_final_score(tmp_path):
    stale = dict(phase="Validate EMA", done=False, best=.7, best_epoch=1)
    final = dict(phase="Done", done=True, best=.76, best_epoch=2)
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

"""Regressions for Kaggle's full disk during the former CUDA-stack install."""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.nestsar_sm_all_t16.streaming import launch, runtime


def fake_cuda(tmp_path, version="12.6"):
    host = tmp_path / "host"
    nvcc = host / "nvidia/cuda_nvcc"
    (nvcc / "bin").mkdir(parents=True)
    (nvcc / "nvvm/libdevice").mkdir(parents=True)
    (nvcc / "nvvm/libdevice/libdevice.10.bc").write_bytes(b"fixture")
    (nvcc / "__init__.py").write_text("")
    ptxas = nvcc / "bin/ptxas"
    ptxas.write_text(f"#!/bin/sh\nprintf 'Cuda compilation tools, release {version}\\n'\n")
    ptxas.chmod(0o755)
    # A host Python package must never leak through alongside CUDA links.
    (host / "jax.py").write_text("raise AssertionError('host JAX leaked')")
    (host / "nvidia/cu13").mkdir()
    return host, nvcc


def tiny_wheel(path):
    wheel = path / "nestsar_disk_probe-0.0.1-py3-none-any.whl"
    info = "nestsar_disk_probe-0.0.1.dist-info"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("nestsar_disk_probe.py", "VALUE = 73\n")
        archive.writestr(f"{info}/METADATA", "Metadata-Version: 2.1\nName: nestsar-disk-probe\nVersion: 0.0.1\n")
        archive.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nGenerator: regression\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{info}/RECORD", "")
    return wheel


def test_local_extra_and_frozen_plan_exclude_cuda_payloads():
    assert runtime.RUNTIME_REQUIREMENTS[0] == "jax[cuda12-local]==0.7.2"
    item = {"metadata": {"name": "jax-cuda12-pjrt", "version": "0.7.2"},
            "download_info": {"archive_info": {"hashes": {"sha256": "a" * 64}}}}
    lock, versions = runtime.resolved_lock({"install": [item]})
    assert "jax-cuda12-pjrt==0.7.2 --hash=sha256:" in lock
    assert versions == {"jax-cuda12-pjrt": "0.7.2"}
    for name in ("nvidia_cudnn_cu12", "NVIDIA-CUBLAS-CU12", "jax-cuda13-plugin"):
        item["metadata"]["name"] = name
        with pytest.raises(RuntimeError, match="CUDA download"):
            runtime.resolved_lock({"install": [item]})


def test_pip_free_caller_uses_base_python_without_ensurepip(monkeypatch):
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(executable="/venv/python", _base_executable="/base/python"))
    calls = []
    def probe(command, **kwargs):
        calls.append(command)
        ok = command[0] == "/base/python"
        return SimpleNamespace(returncode=0 if ok else 1, stdout="pip 25.0 from host" if ok else "")
    monkeypatch.setattr(runtime.subprocess, "run", probe)
    assert runtime.host_pip_python() == "/base/python"
    assert calls == [["/venv/python", "-m", "pip", "--version"], ["/base/python", "-m", "pip", "--version"]]


def test_disk_guard_reports_space_before_installing(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda p: SimpleNamespace(free=runtime.GIB // 2))
    def forbidden(*a, **kw):
        raise AssertionError("pip must not start with insufficient disk")
    with pytest.raises(RuntimeError, match="0.50 GiB free"):
        runtime.install_checked("python", tmp_path, {}, forbidden)
    assert list(tmp_path.iterdir()) == []


def test_expanded_wheel_bytes_not_only_download_bytes(tmp_path):
    with zipfile.ZipFile(tmp_path / "compressed.whl", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("big.py", b"a" * 500000)
    compressed, expanded = runtime.wheel_bytes(tmp_path)
    assert expanded == 500000 and compressed < expanded / 100


def test_install_offline_with_owned_temp_and_cleanup(tmp_path, monkeypatch):
    wheel = tiny_wheel(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    python = launch.create_runtime_without_pip(out / "runtime")
    monkeypatch.setattr(runtime, "RUNTIME_REQUIREMENTS", (str(wheel),))
    env = dict(launch.isolated_env(), PIP_NO_INDEX="1", PIP_FIND_LINKS=str(tmp_path))
    calls = []
    def execute(command, log, child_env, tick):
        assert Path(child_env["TMPDIR"]).parent == out
        assert child_env["TEMP"] == child_env["TMP"] == child_env["TMPDIR"]
        calls.append(command)
        launch.quiet_run(command, log, child_env, tick)
    info = runtime.install_checked(python, out, env, execute)
    assert len(calls) == 3 and "--dry-run" in calls[0]
    assert "download" in calls[1] and "--require-hashes" in calls[1]
    assert "--no-index" in calls[2] and "--no-compile" in calls[2]
    assert info["cuda_packages_downloaded"] == [] and info["expanded_wheel_bytes"] > 0
    assert not list(out.glob(".runtime-install-*"))
    result = subprocess.run([str(python), "-c", "import nestsar_disk_probe as p; assert p.VALUE==73"], capture_output=True)
    assert result.returncode == 0, result.stderr


def test_failure_cleans_temp_and_rejects_cuda_before_download(tmp_path):
    calls = []
    def execute(command, log, env, tick):
        calls.append(command)
        Path(env["TMPDIR"], "partial.tmp").write_text("fixture")
        report = {"install": [{"metadata": {"name": "nvidia-cudnn-cu12", "version": "9.8"}}]}
        Path(command[command.index("--report") + 1]).write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="CUDA download"):
        runtime.install_checked("python", tmp_path, {}, execute)
    assert len(calls) == 1 and not list(tmp_path.glob(".runtime-install-*"))


def test_only_cuda_namespaces_shared_and_host_libraries_survive_cleanup(tmp_path):
    host, nvcc = fake_cuda(tmp_path)
    cuda = runtime.discover_cuda(search_paths=[host], system_roots=[])
    assert cuda["cuda_root"] == str(nvcc)
    out = tmp_path / "out"
    python = launch.create_runtime_without_pip(out / "runtime")
    runtime.link_cuda(out / "runtime", cuda)
    code = "import importlib.util, nvidia.cuda_nvcc as n; assert importlib.util.find_spec('jax') is None; assert importlib.util.find_spec('nvidia.cu13') is None; print(n.__file__)"
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == nvcc / "__init__.py"
    (out / "best.msgpack").write_bytes(b"saved-checkpoint")
    (out / "raw-cache.npy").write_bytes(b"saved-cache")
    assert runtime.remove_failed_runtime(out)
    assert (nvcc / "bin/ptxas").is_file()
    assert (out / "best.msgpack").read_bytes() == b"saved-checkpoint"
    assert (out / "raw-cache.npy").read_bytes() == b"saved-cache"


def test_cuda13_compiler_is_rejected_before_download(tmp_path):
    host, _ = fake_cuda(tmp_path, version="13.0")
    with pytest.raises(RuntimeError, match="No installed CUDA 12"):
        runtime.discover_cuda(search_paths=[host], system_roots=[])


def test_recovery_refuses_unrecognized_or_linked_runtime(tmp_path):
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "important.txt").write_text("keep")
    with pytest.raises(RuntimeError, match="unrecognized"):
        runtime.remove_failed_runtime(tmp_path)
    assert (target / "important.txt").read_text() == "keep"
    out = tmp_path / "out"
    out.mkdir()
    (out / "runtime").symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="unrecognized"):
        runtime.remove_failed_runtime(out)
    assert target.is_dir()


def test_working_host_skips_all_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "_RUNTIME_ENV", {})
    monkeypatch.setattr(launch, "runtime_probe", lambda *a, **kw: True)
    def forbidden(*a, **kw):
        raise AssertionError("working runtime must not be replaced")
    monkeypatch.setattr(launch, "install_checked", forbidden)
    monkeypatch.setattr(launch, "remove_failed_runtime", forbidden)
    assert launch.ensure_local_runtime(tmp_path, [], "0") == sys.executable


def test_failed_install_is_recovered_with_local_cuda_and_same_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "_RUNTIME_ENV", {})
    monkeypatch.setenv("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "1")
    host, _ = fake_cuda(tmp_path)
    cuda = runtime.discover_cuda([host], [])
    out = tmp_path / "out"
    launch.create_runtime_without_pip(out / "runtime")
    (out / "runtime/partial-CUDA.whl").write_bytes(b"old incomplete installer")
    (out / "best_scores.json").write_text('{"xsub":76.3,"xset":78.1}')
    monkeypatch.setattr(launch, "discover_cuda", lambda: cuda)
    probes = []
    def probe(python, code, env, log, refresh):
        assert "JAX_SKIP_CUDA_CONSTRAINTS_CHECK" not in env
        assert "jax.block_until_ready" in code and "conv_general_dilated" in code
        probes.append(env)
        return len(probes) == 3
    monkeypatch.setattr(launch, "runtime_probe", probe)
    def install(python, folder, env, quiet, refresh):
        assert not (out / "runtime/partial-CUDA.whl").exists()
        assert env["CUDA_ROOT"] == cuda["cuda_root"]
        return {"cuda_packages_downloaded": []}
    monkeypatch.setattr(launch, "install_checked", install)
    python = launch.ensure_local_runtime(out, [], "0")
    assert python == str(out / "runtime/bin/python")
    assert probes[-1]["CUDA_ROOT"] == cuda["cuda_root"]
    assert launch.isolated_env("1")["CUDA_VISIBLE_DEVICES"] == "1"
    assert json.loads((out / "runtime.json").read_text())["state"] == "ready"
    assert json.loads((out / "best_scores.json").read_text()) == {"xsub": 76.3, "xset": 78.1}


def test_cached_runtime_restores_cuda_paths_without_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "_RUNTIME_ENV", {})
    host, _ = fake_cuda(tmp_path)
    cuda = runtime.discover_cuda([host], [])
    out = tmp_path / "out"
    python = launch.create_runtime_without_pip(out / "runtime")
    launch.atomic_json(out / "runtime/nestsar_runtime.json", {"schema": runtime.RUNTIME_SCHEMA, "cuda": cuda, "state": "ready"})
    calls = []
    def probe(python, code, env, log, refresh):
        calls.append(env)
        return len(calls) == 2
    monkeypatch.setattr(launch, "runtime_probe", probe)
    def forbidden(*a, **kw):
        raise AssertionError("cached GPU runtime must not be reinstalled")
    monkeypatch.setattr(launch, "install_checked", forbidden)
    assert launch.ensure_local_runtime(out, [], "0") == str(python)
    assert calls[-1]["CUDA_ROOT"] == cuda["cuda_root"]


def test_full_disk_bootstrap_removes_only_failed_venv(tmp_path):
    from experiments.nestsar_sm_all_t16.kaggle_bootstrap import recover_disk_failure
    out = tmp_path / "out"
    launch.create_runtime_without_pip(out / "runtime")
    (out / "install.log").write_text("ERROR: [Errno 28] No space left on device\n")
    (out / "best.msgpack").write_bytes(b"best")
    (out / "raw.npy").write_bytes(b"cached")
    assert recover_disk_failure(out)
    assert not (out / "runtime").exists()
    assert (out / "best.msgpack").read_bytes() == b"best"
    assert (out / "raw.npy").read_bytes() == b"cached"
    assert not recover_disk_failure(out)


def test_bootstrap_keeps_ready_or_active_runtime(tmp_path):
    import fcntl
    from experiments.nestsar_sm_all_t16.kaggle_bootstrap import recover_disk_failure
    launch.create_runtime_without_pip(tmp_path / "runtime")
    (tmp_path / "install.log").write_text("ERROR: No space left on device\n")
    launch.atomic_json(tmp_path / "runtime/nestsar_runtime.json", {"state": "ready"})
    assert not recover_disk_failure(tmp_path)
    (tmp_path / "runtime/nestsar_runtime.json").unlink()
    with (tmp_path / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active launcher"):
            recover_disk_failure(tmp_path)
    assert (tmp_path / "runtime/bin/python").is_file()

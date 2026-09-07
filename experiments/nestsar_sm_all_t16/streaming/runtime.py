"""Disk-bounded bootstrap; reuse CUDA without exposing the kernel's Python packages."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from .io_utils import read_json

GIB = 1024 ** 3
RUNTIME_SCHEMA = "cuda12-local-v1"
RUNTIME_REQUIREMENTS = (
    "jax[cuda12-local]==0.7.2", "flax==0.11.2", "optax==0.2.5",
    "numpy==2.2.6", "psutil==7.0.0", "pytest==8.4.2", "tqdm==4.67.1",
)
# CUDA 13's namespace is deliberately excluded. Only these read-only CUDA 12
# package directories are shared, never host jax/jaxlib/flax or site-packages.
CUDA_PACKAGES = (
    "cuda_runtime", "cuda_nvcc", "cuda_cupti", "cuda_nvrtc", "cublas",
    "cudnn", "cufft", "cusolver", "cusparse", "nvjitlink", "nccl", "nvshmem",
)


def host_pip_python():
    """Also support launching/checking from a pip-free venv itself."""
    candidates = dict.fromkeys(p for p in (sys.executable, getattr(sys, "_base_executable", None)) if p)
    for python in candidates:
        try:
            result = subprocess.run([python, "-m", "pip", "--version"],
                                    capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            continue
        version = re.search(r"pip (\d+)\.(\d+)", result.stdout)
        if result.returncode == 0 and version and tuple(map(int, version.groups())) >= (22, 3):
            return python
    raise RuntimeError("The notebook/base Python must provide pip >=22.3 for pip --python; ensurepip is not used.")


def require_space(path, needed, stage):
    free = shutil.disk_usage(path).free
    if free < needed:
        raise RuntimeError(
            f"Insufficient disk for {stage}: {free/GIB:.2f} GiB free at {path}; "
            f"need {needed/GIB:.2f} GiB including a 1 GiB reserve. "
            "Cache/checkpoints were not deleted. Free space or choose an OUT_DIR "
            "on a larger writable volume, then rerun the same cell."
        )
    return free


def remove_failed_runtime(out):
    """Called only under run.lock, after its interpreter failed the GPU probe.

    Only the launcher's venv is disposable. Refuse links, arbitrary directories,
    the current interpreter, and any path other than OUT_DIR/runtime.
    """
    runtime = Path(out) / "runtime"
    if not runtime.exists() and not runtime.is_symlink():
        return False
    if runtime.is_symlink() or not (runtime / "pyvenv.cfg").is_file():
        raise RuntimeError(f"Refusing to remove an unrecognized runtime: {runtime}")
    if runtime.resolve() in (Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()):
        raise RuntimeError("Cannot replace the notebook's active Python environment.")
    # rmtree unlinks CUDA symlinks without following them into the host install.
    shutil.rmtree(runtime)
    return True


def discover_cuda(search_paths=None, system_roots=None):
    """Locate existing libraries and a CUDA 12 ptxas/libdevice pair; no imports."""
    packages = {}
    for entry in (sys.path if search_paths is None else search_paths):
        if not entry:
            continue
        base = Path(entry) / "nvidia"
        for name in CUDA_PACKAGES:
            path = base / name
            if name not in packages and path.is_dir():
                packages[name] = str(path.resolve())
    if system_roots is None:
        system_roots = [os.environ.get("CUDA_ROOT"), os.environ.get("CUDA_HOME"),
                        "/usr/local/cuda", *sorted(Path("/usr/local").glob("cuda-12*"), reverse=True)]
    roots = [packages.get("cuda_nvcc"), *system_roots]
    cuda_root = None
    compiler_version = None
    for entry in dict.fromkeys(str(x) for x in roots if x):
        root = Path(entry)
        ptxas = root / "bin" / "ptxas"
        if not ptxas.is_file() or not (root / "nvvm/libdevice/libdevice.10.bc").is_file():
            continue
        try:
            result = subprocess.run([str(ptxas), "--version"], capture_output=True,
                                    text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and re.search(r"release\s+12\.", result.stdout + result.stderr):
            cuda_root, compiler_version = str(root.resolve()), result.stdout.strip()
            break
    if cuda_root is None:
        raise RuntimeError(
            "No installed CUDA 12 ptxas + libdevice.10.bc pair found. "
            "Use a Kaggle GPU T4 x2 image with CUDA 12 and compatible cuDNN "
            "(JAX 0.7.2 uses cuDNN 9.8 or newer). No CUDA packages were downloaded. "
            f"Searched CUDA roots: {[str(x) for x in roots if x]}"
        )
    lib_dirs = [str(Path(p) / "lib") for p in packages.values() if (Path(p) / "lib").is_dir()]
    for entry in (Path(cuda_root) / "lib64", Path(cuda_root) / "targets/x86_64-linux/lib",
                  Path("/usr/local/cuda/lib64"), Path("/usr/local/cuda/targets/x86_64-linux/lib")):
        if entry.is_dir() and entry.name != "stubs":
            lib_dirs.append(str(entry.resolve()))
    # Explicit paths lead; retain the image's driver/library search locations.
    lib_dirs.extend(x for x in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if x)
    return {
        "packages": packages, "cuda_root": cuda_root, "compiler_version": compiler_version,
        "environment": {
            "CUDA_ROOT": cuda_root,
            "PATH": str(Path(cuda_root) / "bin") + os.pathsep + os.environ.get("PATH", ""),
            "LD_LIBRARY_PATH": os.pathsep.join(dict.fromkeys(lib_dirs)),
        },
    }


def link_cuda(runtime, inventory):
    """Expose just host CUDA namespaces to a clean, isolated, pip-free venv."""
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    target = Path(runtime) / "lib" / version / "site-packages" / "nvidia"
    target.mkdir(parents=True, exist_ok=True)
    (target / "__init__.py").write_text("# NestSAR: read-only links to the host CUDA 12 libraries.\n")
    for name, path in inventory["packages"].items():
        if name not in CUDA_PACKAGES:
            raise ValueError(f"Unexpected CUDA package: {name}")
        (target / name).symlink_to(path, target_is_directory=True)


def resolved_lock(report):
    """Freeze the resolver output and disallow downloading any NVIDIA payload."""
    lines, versions = [], {}
    for item in report.get("install", []):
        meta = item["metadata"]
        name, version = meta["name"], meta["version"]
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized.startswith("nvidia-") or "cuda13" in normalized:
            raise RuntimeError(f"Refusing a duplicate/incompatible CUDA download: {name}=={version}")
        digest = item["download_info"].get("archive_info", {}).get("hashes", {}).get("sha256")
        if not digest or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"Missing wheel SHA256 in pip's plan: {name}")
        lines.append(f"{name}=={version} --hash=sha256:{digest}")
        versions[name] = version
    if not lines:
        raise RuntimeError("Empty installation plan for a new runtime")
    return "\n".join(lines) + "\n", versions


def wheel_bytes(wheelhouse):
    wheels = list(Path(wheelhouse).glob("*.whl"))
    if not wheels:
        raise RuntimeError("No wheels downloaded for the runtime")
    installed = 0
    for path in wheels:
        try:
            with zipfile.ZipFile(path) as archive:
                installed += sum(entry.file_size for entry in archive.infolist())
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Downloaded wheel is incomplete or invalid: {path.name}; rerun setup.") from exc
    return sum(p.stat().st_size for p in wheels), installed


def install_checked(python, out, env, quiet_run, refresh=None):
    """Resolve, download once, check exact expanded bytes, install offline.

    pip's temporary files and wheels use OUT_DIR's filesystem, not a possibly
    full /tmp. Both are removed even on failure. No ensurepip or kernel mutation.
    """
    out = Path(out)
    before = require_space(out, 2 * GIB, "runtime download")
    def tick():
        if refresh:
            refresh()
        require_space(out, GIB, "runtime setup")
    pip = [host_pip_python(), "-m", "pip", "--python", str(python)]
    flags = ["--disable-pip-version-check", "--no-cache-dir", "--progress-bar", "off",
             "--only-binary=:all:"]
    with tempfile.TemporaryDirectory(prefix=".runtime-install-", dir=out) as temp:
        temp = Path(temp)
        temp_env = dict(env, TMPDIR=str(temp), TEMP=str(temp), TMP=str(temp))
        plan_path = out / "runtime_install_plan.json"
        quiet_run([*pip, "install", *flags, "--dry-run", "--ignore-installed", "--report", str(plan_path),
                   *RUNTIME_REQUIREMENTS], out / "runtime_resolve.log", temp_env, tick)
        lock, versions = resolved_lock(read_json(plan_path, {}))
        lock_path = out / "runtime_requirements.lock"
        lock_path.write_text(lock)
        wheelhouse = temp / "wheels"
        wheelhouse.mkdir()
        quiet_run([*pip, "download", *flags, "--no-deps", "--require-hashes", "-r", str(lock_path),
                   "--dest", str(wheelhouse)], out / "runtime_download.log", temp_env, tick)
        compressed, installed = wheel_bytes(wheelhouse)
        # No bytecode compilation; allow another 15% for wheel metadata, files
        # rounded to filesystem blocks, and installer bookkeeping.
        require_space(out, int(installed * 1.15) + GIB, "expanded runtime installation")
        quiet_run([*pip, "install", *flags, "--no-index", "--find-links", str(wheelhouse),
                   "--no-deps", "--no-compile", "--require-hashes", "-r", str(lock_path)],
                  out / "install.log", temp_env, tick)
    return {"packages": versions, "wheel_bytes": compressed, "expanded_wheel_bytes": installed,
            "free_bytes_before": before, "free_bytes_after": shutil.disk_usage(out).free,
            "cuda_packages_downloaded": [], "schema": RUNTIME_SCHEMA}


# Runs in an isolated child. A GPU enumeration alone does not test ptxas,
# cuBLAS/cuDNN loading, or compilation. Exercise all three and synchronize.
GPU_PROBE = '''
import json
import importlib.metadata as metadata
import jax, jax.numpy as jnp, flax, optax, numpy, psutil, pytest, tqdm
assert jax.default_backend() == 'gpu' and jax.local_device_count() == 1
@jax.jit
def exercise(x, w, a):
    conv = jax.lax.conv_general_dilated(x, w, (1, 1), 'VALID', dimension_numbers=('NHWC','HWIO','NHWC'))
    return conv, jnp.tanh(a) @ a.T, jax.grad(lambda z: jnp.sum(jnp.sin(z)))(a)
result = jax.block_until_ready(exercise(jnp.ones((1, 8, 8, 4), jnp.float32), jnp.ones((3, 3, 4, 8), jnp.float32), jnp.ones((32, 32), jnp.float32)))
assert all(bool(jnp.all(jnp.isfinite(x))) for x in result)
numpy.testing.assert_allclose(numpy.asarray(result[0]), 36.0, rtol=1e-5)
numpy.testing.assert_allclose(numpy.asarray(result[1]), 32*numpy.tanh(1.0), rtol=1e-5)
numpy.testing.assert_allclose(numpy.asarray(result[2]), numpy.cos(1.0), rtol=1e-5)
print(json.dumps({'devices':[str(d) for d in jax.devices()], 'packages': {n:metadata.version(n) for n in ('jax','jaxlib','flax','optax','numpy','psutil','pytest','tqdm')}, 'gpu_execution_verified':jax.default_backend() == 'gpu'}))
'''

PINNED_GPU_PROBE = '''
import importlib.metadata as metadata
assert tuple(metadata.version(n) for n in ('jax','flax','optax','numpy')) == ('0.7.2','0.11.2','0.2.5','2.2.6')
''' + GPU_PROBE

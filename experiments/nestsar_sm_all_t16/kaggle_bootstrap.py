"""Small stdlib-only bootstrap, safe to fetch into memory on a full Kaggle disk."""
from __future__ import annotations

import fcntl
import json
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = "https://github.com/rombaldivia/NestSAR.git"
DEFAULT_OUT = "/kaggle/working/NestSAR_SM_ALL_T16_SharedCache_v2"


def recover_disk_failure(out):
    """Recover the reported legacy Errno 28 before git needs writable space.

    An old error log alone is insufficient: a ready runtime must stay intact.
    Acquiring the same launcher lock also prevents deleting a running venv.
    """
    out = Path(out)
    log, runtime = out / "install.log", out / "runtime"
    if not log.is_file() or not runtime.is_dir() or runtime.is_symlink():
        return False
    with (out / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This OUT_DIR has an active launcher; stop its cell before restarting.") from exc
        from collections import deque
        with log.open(errors="replace") as stream:
            failed = "No space left on device" in "".join(deque(stream, maxlen=100))
        if not failed or not (runtime / "pyvenv.cfg").is_file():
            return False
        try:
            manifest = json.loads((runtime / "nestsar_runtime.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = {}
        if manifest.get("state") == "ready":
            return False
        if runtime.resolve() in (Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()):
            raise RuntimeError("Cannot replace the notebook's active Python environment.")
        shutil.rmtree(runtime)
    return True


def command(*args):
    result = subprocess.run([str(arg) for arg in args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stdout + "\n" + result.stderr)[-5000:])
    return result.stdout.strip()


def run_pinned(commit, settings=None, source_parent="/kaggle/working"):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use the full, pinned 40-character GitHub commit SHA")
    settings = dict(settings or {})
    out = str(settings.get("outdir", DEFAULT_OUT))
    if settings.get("smoke_test", False):
        out += "_smoke"
    recover_disk_failure(out)
    parent = Path(source_parent)
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / ("NestSAR_SM_ALL_" + commit[:12])
    if root.exists() and not (root / ".git").is_dir():
        root = Path(tempfile.mkdtemp(prefix="NestSAR_SM_ALL_", dir=parent))
    if not (root / ".git").is_dir():
        root.mkdir(exist_ok=True)
        command("git", "init", "--quiet", root)
        command("git", "-C", root, "remote", "add", "origin", REPOSITORY)
    if command("git", "-C", root, "status", "--porcelain"):
        raise RuntimeError(f"Source has local edits: {root}. Use a fresh source_parent.")
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
                          capture_output=True, text=True)
    if head.returncode or head.stdout.strip() != commit:
        command("git", "-C", root, "fetch", "--quiet", "--depth", "1", REPOSITORY, commit)
        command("git", "-C", root, "checkout", "--quiet", "--detach", commit)
    if command("git", "-C", root, "rev-parse", "HEAD") != commit:
        raise RuntimeError("Source commit does not match the requested revision")
    return runpy.run_path(str(root / "experiments/nestsar_sm_all_t16/kaggle_cell.py"),
                         init_globals={"NESTSAR_SETTINGS": settings})["nestsar_results"]

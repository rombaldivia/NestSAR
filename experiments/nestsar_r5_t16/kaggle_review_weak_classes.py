"""Paste this complete file into ONE Kaggle cell with the saved runs and their caches attached.

This reviews existing T16 graph runs. It neither starts training nor changes the saved weights.
Outputs: /kaggle/working/NestSAR_WEAK_REVIEW/<run>/<protocol>/weak_class_report.json
"""
from pathlib import Path
import os
import subprocess
import sys


def main():
    repository = Path("/kaggle/working/NestSAR_WEAK_REVIEW_CODE")
    branch = "fix/r7-evaluation-provenance-weak-classes"
    url = "https://github.com/rombaldivia/NestSAR.git"
    if not (repository / ".git").is_dir():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", branch, url, str(repository)], check=True)
    else:
        remote = subprocess.check_output(["git", "-C", str(repository), "remote", "get-url", "origin"], text=True).strip()
        if remote != url:
            raise RuntimeError(f"{repository} belongs to another repository; choose a fresh code directory")
        if subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip():
            raise RuntimeError(f"{repository} has local edits; preserve them and choose a fresh code directory")
        subprocess.run(["git", "-C", str(repository), "fetch", "--depth", "1", "origin", branch], check=True)
        subprocess.run(["git", "-C", str(repository), "checkout", "--detach", "FETCH_HEAD"], check=True)
    subprocess.run(["git", "-C", str(repository), "log", "-1", "--oneline"], check=True)
    env = dict(os.environ, PYTHONPATH=str(repository), CUDA_VISIBLE_DEVICES="0",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([sys.executable, "-u", "-m", "experiments.nestsar_r5_t16.review_weak_classes", "--discover"],
                   cwd=repository, env=env, check=True)


if __name__ == "__main__":
    main()

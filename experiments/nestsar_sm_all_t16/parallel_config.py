"""Parallel experiment identity; safe to import before GPU isolation."""
import hashlib
from pathlib import Path

MODEL_NAME = "NestSAR-FULL-PARALLEL-T16-v2"
EXPECTED_PARAMS = 1_831_932


def implementation_identity():
    root = Path(__file__).resolve().parent
    names = ("model.py", "model_parallel.py", "streaming/worker.py",
             "streaming/worker_parallel.py", "streaming/launch.py")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode() + b"\0" + (root/name).read_bytes() + b"\0")
    return {"version": MODEL_NAME, "source_sha256": digest.hexdigest()}

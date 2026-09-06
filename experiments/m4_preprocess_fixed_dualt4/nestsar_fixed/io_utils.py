from __future__ import annotations
import json
import os
import time
from pathlib import Path


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path, data):
    atomic_bytes(path, json.dumps(data, allow_nan=False, separators=(",", ":")).encode())


class Reporter:
    def __init__(self, path):
        self.path = Path(path)
        self.state = {}

    def __call__(self, **changes):
        self.state.update(changes)
        self.state.update(timestamp=time.time(), pid=os.getpid())
        atomic_json(self.path, self.state)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default

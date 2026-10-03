"""Identity for the controlled D128 + multi-timescale parallel experiment."""
import hashlib
from pathlib import Path

MODEL_NAME = "NestSAR-FULL-PARALLEL-T16-D128-MTS-v1"
EXPECTED_PARAMS = 2_338_668
MODEL_DIM = 128
M4_HALF_LIVES = (1.0, 3.0, 7.0, 15.0)
G4_HALF_LIVES = (1.0, 2.0, 4.0, 8.0)


def implementation_identity():
    root = Path(__file__).resolve().parent
    names = (
        "model.py",
        "model_parallel.py",
        "parallel_d128_config.py",
        "streaming/worker.py",
        "streaming/worker_parallel_d128.py",
        "streaming/launch.py",
    )
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    return {
        "version": MODEL_NAME,
        "source_sha256": digest.hexdigest(),
        "model_dim": MODEL_DIM,
        "m4_half_lives": list(M4_HALF_LIVES),
        "g4_half_lives": list(G4_HALF_LIVES),
    }

"""NestSAR-SA inside the proven R4-FMSE + LocalGeometry training pipeline.

Same cache, preprocessing, augmentation, losses (main + stream aux +
consistency + training-only local geometry), optimizer, EMA, checkpointing and
resume as the R4-FMSE geometry run. Only the model factory, its identity and the
expected parameter count change, and only inside this worker process (the
geometry worker module is not modified on disk).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.nestsar_r4_fmse_geometry_t16 import worker as geo
from experiments.nestsar_sa_t16 import MODEL_IDENTITY, MODEL_NAME, VERSION
from experiments.nestsar_sa_t16.model import NestSARSAT16

# Measured with the exact model (jax.make_jaxpr + parameter count), batch 1, T=16.
EXPECTED_PARAMS = 1_841_604
STRICT_MFLOPS = 90.98          # 1 MAC = 2 FLOPs (R4-FMSE: 60.87)

MODEL_KEYS = ("spatial_dim", "model_dim", "dropout", "controller_dim", "fast_rank",
              "head_rank", "sm_residual_scale", "head_residual_scale")


def make_model(config):
    return NestSARSAT16(**{k: config[k] for k in MODEL_KEYS})


def implementation_identity():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("model.py", "worker.py", "__init__.py"):
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    return dict(MODEL_IDENTITY, version=VERSION, source_sha256=digest.hexdigest())


def install():
    """Point the geometry pipeline at NestSAR-SA (this process only)."""
    geo.make_model = make_model
    geo.EXPECTED_PARAMS = EXPECTED_PARAMS
    geo.MODEL_NAME = MODEL_NAME
    geo.MODEL_IDENTITY = implementation_identity()


def run(config, protocol, cache, outdir, allow_cpu=False):
    install()
    return geo.run(config, protocol, cache, outdir, allow_cpu)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--protocol", required=True, choices=("xsub", "xset"))
    p.add_argument("--cache", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    print(MODEL_NAME, flush=True)
    run(json.loads(Path(a.config).read_text()), a.protocol, a.cache, a.outdir, a.allow_cpu)


if __name__ == "__main__":
    main()

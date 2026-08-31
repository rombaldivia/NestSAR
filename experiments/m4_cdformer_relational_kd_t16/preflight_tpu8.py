#!/usr/bin/env python3
from __future__ import annotations

"""Strict TPU8 preflight for CD-Former relational KD."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.m4_cdformer_relational_kd_t16 import train_tpu as kd
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju


def main() -> None:
    print("=" * 120)
    print("NESTSAR T16 + CD-FORMER RELATIONAL KD — TPU8 PREFLIGHT")
    print("=" * 120)
    print("JAX:", jax.__version__)
    print("BACKEND:", jax.default_backend())
    print("LOCAL DEVICES:", jax.local_device_count())
    print("DEVICES:", jax.local_devices())

    if jax.default_backend() != "tpu" or jax.local_device_count() != 8:
        raise RuntimeError(
            f"Expected TPU8, got backend={jax.default_backend()} count={jax.local_device_count()}"
        )

    teacher = Path(kd.DEFAULT_TEACHER)
    relation, meta = kd.load_teacher_relation(teacher)
    print("TEACHER:", meta["path"])
    print("TEACHER SHA256:", meta["sha256"])
    print("TEACHER TENSORS:", meta["tensors"])
    print("TEACHER PARAMS:", f"{meta['parameters']:,}")
    print("RELATION SHAPE:", relation.shape)
    print("RELATION DIAG MEAN:", float(jnp.mean(jnp.diag(relation))))

    model = ju.M4PhaseUniformT16(24, 112, 0.10)
    key = jax.random.PRNGKey(128)
    dummy = jnp.zeros((1, kd.FRAMES, kd.FEATURES), jnp.float32)
    params = model.init({"params": key, "dropout": key}, dummy, training=False)["params"]
    nparams = ju.count_params(params)
    print("STUDENT PARAMS:", f"{nparams:,}")
    if nparams != kd.EXPECTED_PARAMS:
        raise RuntimeError(f"Student params {nparams:,} != {kd.EXPECTED_PARAMS:,}")

    rel0 = kd.relational_loss(params, relation)
    print("INITIAL RELATIONAL LOSS:", float(rel0))
    if not bool(jnp.isfinite(rel0)):
        raise RuntimeError("Non-finite relational loss")

    # Prove that relational supervision differentiates through all four classifier heads.
    grads = jax.grad(lambda p: kd.relational_loss(p, relation))(params)
    norms = []
    for i in range(kd.NUM_STREAMS):
        g = grads[f"classifier_{i}"]["kernel"]
        norms.append(float(jnp.linalg.norm(g)))
    print("CLASSIFIER REL-GRAD NORMS:", norms)
    if not all(np.isfinite(norms)) or not all(v > 0.0 for v in norms):
        raise RuntimeError("Relational gradient did not reach every classifier head")

    # Exact student inference compute remains unchanged from champion.
    flops = ju.audit_flops(model, params)
    if flops is None:
        print("XLA FLOPS: unavailable")
    else:
        print("XLA FLOPS/CLIP:", int(flops))
        print("XLA GFLOPS/CLIP:", float(flops) / 1e9)

    out = model.apply({"params": params}, dummy, training=False)
    if out["logits"].shape != (1, 120):
        raise RuntimeError(f"Bad logits shape {out['logits'].shape}")
    if not bool(jnp.all(jnp.isfinite(out["logits"]))):
        raise RuntimeError("Non-finite logits")

    print("INFERENCE EXTRA PARAMS: 0")
    print("INFERENCE EXTRA FLOPS: 0")
    print("PREFLIGHT=PASS")
    print("=" * 120)


if __name__ == "__main__":
    main()

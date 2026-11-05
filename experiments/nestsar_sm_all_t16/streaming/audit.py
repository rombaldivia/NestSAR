"""SM-ALL static-unrolled compute audit with forward-equivalence verification."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from .. import model as sm
from ..compute_unrolled_audit import UnrolledGatedSweep, UnrolledFastWeightDeltaResidual
from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from .io_utils import atomic_json
from .launch import validate_config
from .worker import make_model, EXPECTED_PARAMS


def audit_model(model, params):
    x = jax.random.normal(jax.random.PRNGKey(31), (1, 16, 750)) * 0.1
    original = jax.block_until_ready(jax.jit(lambda p, a: model.apply({"params": p}, a, training=False)["logits"])(params, x))
    old_gated, old_fast = base.GatedSweep, sm.FastWeightDeltaResidual
    try:
        base.GatedSweep, sm.FastWeightDeltaResidual = UnrolledGatedSweep, UnrolledFastWeightDeltaResidual
        compiled = jax.jit(lambda p, a: model.apply({"params": p}, a, training=False)["logits"]).lower(params, x).compile()
        unrolled = jax.block_until_ready(compiled(params, x))
        np.testing.assert_allclose(unrolled, original, rtol=2e-4, atol=2e-5)
        cost = compiled.cost_analysis()
        if isinstance(cost, list):
            cost = cost[0]
        flops = float(cost["flops"])
        if not np.isfinite(flops) or flops <= 0:
            raise RuntimeError(f"Invalid FLOP cost: {cost}")
        return {"params": sum(a.size for a in jax.tree.leaves(params)),
                "flops": flops, "gflops": flops / 1e9,
                "scan_unrolled_outputs_match": True,
                "max_abs_difference": float(np.max(np.abs(np.asarray(original-unrolled)))),
                "backend": jax.default_backend(), "jax": jax.__version__, "batch": 1,
                "convention": "Static-unrolled XLA forward; 1 MAC=2 FLOPs; preprocessing excluded"}
    finally:
        base.GatedSweep, sm.FastWeightDeltaResidual = old_gated, old_fast


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    config = validate_config(json.loads(Path(args.config).read_text()))
    model = make_model(config)
    params = model.init({"params": jax.random.PRNGKey(128)}, jnp.zeros((1,16,750)), training=False)["params"]
    result = audit_model(model, params)
    if result["params"] != EXPECTED_PARAMS:
        raise RuntimeError("SM-ALL parameter count changed")
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()

"""Full-forward audit with loop iteration costs exposed by static unrolling."""
import argparse
import json
import jax
import jax.numpy as jnp
import numpy as np
from . import model as m
from .preprocessing import FRAMES, FEATURES, HAND_FRAMES, HAND_FEATURES
from .io_utils import atomic_json


def audit():
    key = jax.random.PRNGKey(31)
    x = jax.random.normal(key, (1, FRAMES, FEATURES)) * 0.1
    h = jax.random.normal(jax.random.fold_in(key, 1), (1, HAND_FRAMES, HAND_FEATURES)) * 0.1
    model = m.M4LocalGlobalHandM4G4T32()
    params = model.init({"params": key}, x, h, training=False)["params"]
    count = sum(a.size for a in jax.tree.leaves(params))
    assert count == m.EXPECTED_PARAMS, count
    def forward(p, a, b):
        return model.apply({"params": p}, a, b, training=False)["logits"]
    original = jax.jit(forward)(params, x, h)
    jax.block_until_ready(original)
    try:
        m.AUDIT_UNROLL = True
        # New function identity forces tracing with unrolling enabled.
        compiled = jax.jit(lambda p, a, b: forward(p, a, b)).lower(params, x, h).compile()
        expanded = compiled(params, x, h)
        np.testing.assert_allclose(np.asarray(expanded), np.asarray(original), rtol=2e-4, atol=2e-5)
        cost = compiled.cost_analysis()
        if isinstance(cost, list):
            cost = cost[0]
        flops = float(cost["flops"])
        if not np.isfinite(flops) or flops <= 0:
            raise RuntimeError(f"Invalid XLA FLOP cost: {cost}")
        return {"parameters": count, "full_unrolled_flops": flops, "full_unrolled_gflops": flops / 1e9,
                "scan_unrolled_outputs_match": True, "max_abs_difference": float(np.max(np.abs(np.asarray(expanded-original)))),
                "jax": jax.__version__, "backend": jax.default_backend(), "batch": 1,
                "main_shape": [FRAMES, FEATURES], "hand_shape": [HAND_FRAMES, HAND_FEATURES],
                "convention": "XLA full-forward static unrolling, 1 MAC=2 FLOPs; raw preprocessing excluded"}
    finally:
        m.AUDIT_UNROLL = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit()
    atomic_json(args.output, result)
    print(json.dumps(result))

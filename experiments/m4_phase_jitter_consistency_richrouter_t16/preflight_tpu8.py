#!/usr/bin/env python3
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_richrouter_t16.model import (
    M4PhaseRichRouterT16,
    EXPECTED_PARAMS,
)

BASELINE_PARAMS = 1_816_130
EXPECTED_DELTA = 62_943


def main() -> None:
    print("=" * 120)
    print("NESTSAR T16 RICH-ROUTER — TPU8 PREFLIGHT")
    print("=" * 120)
    print("JAX:", jax.__version__)
    print("BACKEND:", jax.default_backend())
    print("LOCAL DEVICES:", jax.local_device_count())
    print("DEVICES:", jax.local_devices())

    if jax.default_backend() != "tpu" or jax.local_device_count() != 8:
        raise RuntimeError(
            f"Expected TPU8, got backend={jax.default_backend()} count={jax.local_device_count()}"
        )

    model = M4PhaseRichRouterT16(24, 112, 0.10)
    key = jax.random.PRNGKey(128)
    dummy = jnp.zeros((1, ju.FRAMES, ju.FEATURES), jnp.float32)
    params = model.init({"params": key, "dropout": key}, dummy, training=False)["params"]

    nparams = ju.count_params(params)
    print("BASELINE PARAMS:", f"{BASELINE_PARAMS:,}")
    print("RICH ROUTER PARAMS TOTAL MODEL:", f"{nparams:,}")
    print("PARAM DELTA:", f"{nparams - BASELINE_PARAMS:+,}")

    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(f"Param mismatch: {nparams:,} != {EXPECTED_PARAMS:,}")
    if nparams - BASELINE_PARAMS != EXPECTED_DELTA:
        raise RuntimeError("Unexpected router parameter delta")

    out = model.apply({"params": params}, dummy, training=False)

    expected_shapes = {
        "logits": (1, 120),
        "stream_logits": (1, 4, 120),
        "router_weights": (1, 16, 4),
        "frame_stack": (1, 16, 4, 112),
        "mixed_frame_stack": (1, 16, 4, 112),
    }
    for name, shape in expected_shapes.items():
        got = tuple(out[name].shape)
        print(f"{name}: {got}")
        if got != shape:
            raise RuntimeError(f"{name}: expected {shape}, got {got}")
        if not bool(jnp.all(jnp.isfinite(out[name]))):
            raise RuntimeError(f"Non-finite {name}")

    weights_sum = np.asarray(jax.device_get(jnp.sum(out["router_weights"], axis=2)))
    if not np.allclose(weights_sum, 1.0, atol=1e-5):
        raise RuntimeError("Router weights do not sum to one")

    # Confirm gradients reach every learnable component of the richer router.
    def router_objective(p):
        o = model.apply({"params": p}, dummy, training=False)
        return jnp.sum(jnp.square(o["mixed_frame_stack"]))

    grads = jax.grad(router_objective)(params)["cross_stream_after_frame"]
    grad_norms = {}
    for name, tree in grads.items():
        leaves = jax.tree_util.tree_leaves(tree)
        norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))
        grad_norms[name] = float(norm)
    print("ROUTER GRAD NORMS:", grad_norms)
    if not all(np.isfinite(v) for v in grad_norms.values()):
        raise RuntimeError("Non-finite router gradient")

    flops = ju.audit_flops(model, params)
    if flops is not None and np.isfinite(flops):
        print("XLA FLOPS/CLIP:", int(flops))
        print("XLA GFLOPS/CLIP:", float(flops) / 1e9)

    print("PREFLIGHT=PASS")
    print("=" * 120)


if __name__ == "__main__":
    main()

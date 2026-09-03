#!/usr/bin/env python3
from __future__ import annotations

import argparse
import jax
import jax.numpy as jnp

from experiments.m4_confusion_memory_expert_t32.model import ConfusionMemoryExpert, T32, TOKEN_FEATURES, count_params


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--specialist-dim", type=int, default=32)
    p.add_argument("--topk-context", type=int, default=8)
    p.add_argument("--weak-count", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.10)
    args = p.parse_args()

    model = ConfusionMemoryExpert(args.weak_count + 1, args.specialist_dim, args.topk_context, args.dropout)
    key = jax.random.PRNGKey(0)
    xt = jnp.zeros((1, T32, TOKEN_FEATURES), jnp.float32)
    zb = jnp.zeros((1, 120), jnp.float32)
    params = model.init({"params": key, "dropout": key}, xt, zb, training=False)["params"]
    nparams = count_params(params)
    fn = jax.jit(lambda p, x, z: model.apply({"params": p}, x, z, training=False)["logits"])
    compiled = fn.lower(params, xt, zb).compile()
    ca = compiled.cost_analysis()
    if isinstance(ca, list):
        ca = ca[0] if ca else {}
    flops = float(ca.get("flops", float("nan")))

    print("=" * 96)
    print("NESTSAR CONFUSION-MEMORY EXPERT T32 — PREFLIGHT")
    print("=" * 96)
    print("JAX:", jax.__version__)
    print("BACKEND:", jax.default_backend())
    print("DEVICES:", jax.local_devices())
    print("T32 FEATURES:", TOKEN_FEATURES)
    print("SPECIALIST DIM:", args.specialist_dim)
    print("WEAK OUTPUTS + REJECT:", args.weak_count + 1)
    print("PARAMS:", f"{nparams:,}")
    print("GPU-XLA FLOPs/clip:", f"{flops:,.0f}")
    print("GPU-XLA GFLOPs/clip:", f"{flops/1e9:.9f}")
    print("ATTENTION: NONE")
    print("QKV: NONE")
    print("GCN: NONE")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

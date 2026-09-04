#!/usr/bin/env python3
from __future__ import annotations

"""Scan-corrected static FLOP audit for NestSAR.

JAX/XLA cost_analysis on a graph containing lax.scan/while may report loop-body
cost without charging every recurrent iteration. For paper FLOPs we therefore
replace ONLY the recurrent scan implementations with mathematically identical
static Python-unrolled loops before lowering. T is fixed by the tensor shape
(T16 main, T4 chunks, T32 hand), so XLA sees every executed operation.

No model dimensions, equations, branches, inputs, or outputs are changed.
This is an inference-counting transform only; do not use these classes to train.

Audits with one identical GPU/XLA path:
  LocalGlobal V2
  Hand_M4G4_T32
  NestSAR-SM-ALL-T16
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.model import (
    M4LocalGlobalHandM4G4T32,
)
from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.preprocessing import (
    HAND_FEATURES,
    HAND_FRAMES,
)
import experiments.nestsar_sm_all_t16.model as sm_mod

FRAMES = sm_mod.FRAMES
FEATURES = sm_mod.FEATURES


class UnrolledGatedSweep(nn.Module):
    """Exact base.GatedSweep equations, statically unrolled for FLOP counting."""
    dim: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d = self.dim
        init = nn.initializers.xavier_uniform()
        wz_x = self.param("wz_x", init, (d, d))
        wz_h = self.param("wz_h", init, (d, d))
        bz = self.param("bz", nn.initializers.zeros, (d,))
        wr_x = self.param("wr_x", init, (d, d))
        wr_h = self.param("wr_h", init, (d, d))
        br = self.param("br", nn.initializers.zeros, (d,))
        wc_x = self.param("wc_x", init, (d, d))
        wc_h = self.param("wc_h", init, (d, d))
        bc = self.param("bc", nn.initializers.zeros, (d,))

        xt = jnp.swapaxes(x, 0, 1)
        if self.reverse:
            xt = xt[::-1]

        h = jnp.zeros((x.shape[0], d), x.dtype)
        ys = []
        # x.shape[1] is static (25 joints, 16 frames, 4 chunks, or 32 hand frames).
        for t in range(int(xt.shape[0])):
            token = xt[t]
            z = jax.nn.sigmoid(token @ wz_x + h @ wz_h + bz)
            r = jax.nn.sigmoid(token @ wr_x + h @ wr_h + br)
            cand = jnp.tanh(token @ wc_x + (r * h) @ wc_h + bc)
            h = (1.0 - z) * h + z * cand
            ys.append(h)

        yt = jnp.stack(ys, axis=0)
        if self.reverse:
            yt = yt[::-1]
        return jnp.swapaxes(yt, 0, 1)


class UnrolledFastWeightDeltaResidual(nn.Module):
    """Exact SM fast-weight equations, statically unrolled for FLOP counting."""
    dim: int
    rank: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, eta: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
        n = nn.LayerNorm(name="value_norm")(x)

        k = nn.Dense(
            self.rank,
            use_bias=False,
            kernel_init=nn.initializers.normal(0.02),
            name="key",
        )(n)
        q = nn.Dense(
            self.rank,
            use_bias=False,
            kernel_init=nn.initializers.normal(0.02),
            name="query",
        )(n)

        k = jnp.tanh(k)
        q = jnp.tanh(q)
        k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-6)
        q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-6)

        memory0 = self.param(
            "memory0",
            nn.initializers.normal(0.01),
            (self.rank, self.dim),
        )
        mem = jnp.broadcast_to(
            memory0[None, :, :],
            (x.shape[0], self.rank, self.dim),
        )

        reads = []
        for t in range(int(x.shape[1])):
            key_t = k[:, t]
            query_t = q[:, t]
            value_t = n[:, t]
            eta_t = eta[:, t]
            alpha_t = alpha[:, t]

            pred_t = jnp.einsum("br,brd->bd", key_t, mem)
            err_t = value_t - pred_t
            delta_t = jnp.einsum("br,bd->brd", key_t, err_t)
            mem = alpha_t[..., None] * mem + eta_t[..., None] * delta_t
            read_t = jnp.einsum("br,brd->bd", query_t, mem)
            reads.append(read_t)

        return jnp.stack(reads, axis=1)


def count_params(params):
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))


def xla_flops(fn, *args):
    compiled = jax.jit(fn).lower(*args).compile()
    ca = compiled.cost_analysis()
    if isinstance(ca, list):
        ca = ca[0] if ca else {}
    return float(ca.get("flops", float("nan")))


def audit_models():
    # Monkey-patch only the scan implementations before model construction.
    original_gated = base.GatedSweep
    original_fast = sm_mod.FastWeightDeltaResidual
    base.GatedSweep = UnrolledGatedSweep
    sm_mod.FastWeightDeltaResidual = UnrolledFastWeightDeltaResidual

    try:
        key = jax.random.PRNGKey(128)
        k1, k2, k3 = jax.random.split(key, 3)
        x = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)

        local = ju.M4PhaseUniformT16(
            spatial_dim=24,
            model_dim=112,
            dropout=0.10,
        )
        p_local = local.init({"params": k1, "dropout": k1}, x, training=False)["params"]
        f_local = xla_flops(
            lambda p, xx: local.apply({"params": p}, xx, training=False)["logits"],
            p_local,
            x,
        )

        hand = M4LocalGlobalHandM4G4T32(
            spatial_dim=24,
            model_dim=112,
            dropout=0.10,
            hand_dim=32,
            hand_residual_scale=0.10,
        )
        hx = jnp.zeros((1, HAND_FRAMES, HAND_FEATURES), jnp.float32)
        p_hand = hand.init({"params": k2, "dropout": k2}, x, hx, training=False)["params"]
        f_hand = xla_flops(
            lambda p, xx, hh: hand.apply({"params": p}, xx, hh, training=False)["logits"],
            p_hand,
            x,
            hx,
        )

        sm = sm_mod.NestSARSMAllT16(
            spatial_dim=24,
            model_dim=112,
            dropout=0.10,
            controller_dim=16,
            fast_rank=2,
            head_rank=2,
            sm_residual_scale=0.08,
            head_residual_scale=0.15,
        )
        p_sm = sm.init({"params": k3, "dropout": k3}, x, training=False)["params"]
        f_sm = xla_flops(
            lambda p, xx: sm.apply({"params": p}, xx, training=False)["logits"],
            p_sm,
            x,
        )

        return [
            ("LocalGlobal_V2", count_params(p_local), f_local),
            ("Hand_M4G4_T32", count_params(p_hand), f_hand),
            ("SM_ALL_T16", count_params(p_sm), f_sm),
        ]
    finally:
        base.GatedSweep = original_gated
        sm_mod.FastWeightDeltaResidual = original_fast


def main():
    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU; backend={jax.default_backend()}, "
            f"devices={jax.local_devices()}"
        )

    print("=" * 118)
    print("NESTSAR SCAN-CORRECTED STATIC FLOP AUDIT | BATCH=1")
    print("Every recurrent iteration is explicitly unrolled before XLA cost analysis.")
    print("=" * 118)
    print("JAX:", jax.__version__)
    print("DEVICE:", jax.local_devices()[0])

    rows = audit_models()

    print("\nmodel                params          FLOPs       MFLOPs        GFLOPs")
    print("-" * 86)
    for name, params, flops in rows:
        print(
            f"{name:20s} {params:10,d} {flops:14,.0f} "
            f"{flops/1e6:12.6f} {flops/1e9:13.9f}"
        )

    d = {name: (params, flops) for name, params, flops in rows}
    f_local = d["LocalGlobal_V2"][1]
    f_hand = d["Hand_M4G4_T32"][1]
    f_sm = d["SM_ALL_T16"][1]

    print("\nINCREMENTAL COMPUTE")
    print(f"Hand vs LocalGlobal : {(f_hand-f_local)/1e6:+.6f} MFLOPs ({100*(f_hand/f_local-1):+.3f}%)")
    print(f"SM vs LocalGlobal   : {(f_sm-f_local)/1e6:+.6f} MFLOPs ({100*(f_sm/f_local-1):+.3f}%)")
    print(f"SM vs Hand          : {(f_sm-f_hand)/1e6:+.6f} MFLOPs ({100*(f_sm/f_hand-1):+.3f}%)")

    print("\nPAPER NUMBER")
    print(f"NestSAR-SM-ALL-T16 = {f_sm/1e9:.9f} GFLOPs/clip = {f_sm/1e6:.6f} MFLOPs/clip")
    print("=" * 118)


if __name__ == "__main__":
    main()

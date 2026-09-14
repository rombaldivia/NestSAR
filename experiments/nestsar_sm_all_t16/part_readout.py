"""Attention-free nonlinear part readout inspired by Ta-CNN's learned subspaces.

This is a NestSAR experiment, not a reproduction of CAG/VAG. It operates on
existing joint-memory features. All weights are ordinary learned parameters;
there are no input-dependent attention weights, graph/temporal convolutions,
new tokens, or additional temporal memories.
"""
import jax.numpy as jnp
from flax import linen as nn

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from .part_readout_config import VERSION, RANK, MIXTURES, RESIDUAL_SCALE, extra_parameters


class NonlinearPartReadout(nn.Module):
    """Return a small residual for the ten existing part descriptors.

    For each person/frame/part, form two learned signed combinations of joint
    features after subtracting their valid-joint mean. A nonlinearity follows
    each combination before projecting back to the existing 24-D part width.
    Opposing joint features can therefore survive even if their mean is zero.

    The output starts at zero, preserving the baseline function at initialization.
    Missing joints contribute neither features nor normalization weight. A part
    with fewer than two valid joints contributes no relational residual.
    """

    dim: int = 24

    @nn.compact
    def __call__(self, h, valid):
        if h.ndim != 5 or h.shape[-2:] != (25, self.dim) or valid.shape != h.shape[:-1]:
            raise ValueError(f"Expected h=[B,T,M,25,{self.dim}] and matching validity; got {h.shape}, {valid.shape}")
        # where also excludes nonfinite garbage in explicitly invalid positions.
        safe = jnp.where(valid[..., None], h, 0)
        z = nn.Dense(RANK, use_bias=False, name="feature_map")(safe)
        membership = jnp.asarray(base.PART_MASK_NP, h.dtype)
        weights = self.param("joint_mixtures", nn.initializers.normal(0.2), (25, MIXTURES))
        weights = membership[..., None] * jnp.tanh(weights)[None, ...]
        vf = valid.astype(h.dtype)
        counts = jnp.einsum("btmv,pv->btmp", vf, membership)
        denom = jnp.maximum(counts, 1)[..., None]
        means = jnp.einsum("btmvr,pv->btmpr", z, membership) / denom
        weight_sums = jnp.einsum("btmv,pvq->btmpq", vf, weights)
        mixtures = jnp.einsum("btmvr,pvq->btmpqr", z, weights)
        centered = (mixtures - weight_sums[..., None] * means[..., None, :]) / denom[..., None]
        centered = jnp.where((counts >= 2)[..., None, None], centered, 0)
        hidden = nn.gelu(centered).reshape(*centered.shape[:-2], MIXTURES * RANK)
        delta = nn.Dense(self.dim, use_bias=False, kernel_init=nn.initializers.zeros,
                         name="out_proj")(hidden)
        return RESIDUAL_SCALE * delta

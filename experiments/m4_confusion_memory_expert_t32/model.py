#!/usr/bin/env python3
from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base

T32 = 32
SELECTED_JOINTS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24)
PERSONS = 2
PER_JOINT_CHANNELS = 9  # xyz + velocity xyz + acceleration xyz
REL_FEATURES = 12
TOKEN_FEATURES = PERSONS * len(SELECTED_JOINTS) * PER_JOINT_CHANNELS + REL_FEATURES


class ConfusionMemoryExpert(nn.Module):
    """Tiny NestSAR-native T32 specialist.

    No attention / no QKV / no GCN. The expert consumes T32 local-detail tokens
    plus frozen base logits, then predicts only weak classes + one reject class.
    """

    num_outputs: int
    dim: int = 32
    topk_context: int = 8
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, base_logits: jnp.ndarray, training: bool = False):
        h = nn.Dense(self.dim, name="token_proj")(x)
        h = nn.gelu(h)
        h = nn.LayerNorm(name="token_norm")(h)
        h = base.BiMemory(self.dim, name="temporal_bimemory")(h)

        mean = jnp.mean(h, axis=1)
        peak = jnp.max(h, axis=1)
        delta = h[:, -1] - h[:, 0]
        motion_desc = nn.Dense(self.dim, name="motion_fuse")(
            jnp.concatenate([mean, peak, delta], axis=-1)
        )
        motion_desc = nn.gelu(motion_desc)
        motion_desc = nn.LayerNorm(name="motion_norm")(motion_desc)

        probs = nn.softmax(base_logits, axis=-1)
        topv = jnp.sort(probs, axis=-1)[:, -self.topk_context:]
        topv = topv[:, ::-1]
        margin = topv[:, :1] - topv[:, 1:2]
        entropy = -jnp.sum(probs * jnp.log(jnp.clip(probs, 1e-8, 1.0)), axis=-1, keepdims=True)
        ctx = jnp.concatenate([topv, margin, entropy], axis=-1)
        ctx = nn.Dense(self.dim, name="context_proj")(ctx)
        ctx = nn.gelu(ctx)

        z = jnp.concatenate([motion_desc, ctx], axis=-1)
        z = nn.Dense(self.dim, name="fuse")(z)
        z = nn.gelu(z)
        z = nn.Dropout(self.dropout)(z, deterministic=not training)
        logits = nn.Dense(self.num_outputs, name="classifier")(z)
        return {"logits": logits, "descriptor": motion_desc}


def count_params(params) -> int:
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))

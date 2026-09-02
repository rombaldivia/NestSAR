#!/usr/bin/env python3
from __future__ import annotations

"""Tiny adaptive trust gate for LocalGlobal V2 + Hand-M4/G4-Lite T32.

The validated hand specialist already adds complementary information, but the
fixed residual coefficient underuses it.  This module learns only a bounded,
sample-wise trust coefficient:

    alpha(x) = max_alpha * sigmoid(g(x))
    logits   = main_logits + alpha(x) * hand_logits

Gate inputs:
  * mean main descriptor: 112
  * hand descriptor:       32
  * main top1-top2 margin:   1
  * hand top1-top2 margin:   1
  * main entropy:            1
  * hand entropy:            1
  --------------------------------
  total:                   148

Default MLP: 148 -> 16 -> 1 = 2,401 parameters.
No attention and no change to the validated T16/T32 feature extractors.
"""

from typing import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn

from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.model import (
    M4LocalGlobalHandM4G4T32,
)

MAIN_DESC_DIM = 112
HAND_DESC_DIM = 32
CONF_FEATURES = 4
GATE_INPUT_DIM = MAIN_DESC_DIM + HAND_DESC_DIM + CONF_FEATURES  # 148
GATE_HIDDEN_DIM = 16
GATE_EXTRA_PARAMS = GATE_INPUT_DIM * GATE_HIDDEN_DIM + GATE_HIDDEN_DIM
GATE_EXTRA_PARAMS += GATE_HIDDEN_DIM * 1 + 1  # 2,401
BASE_HAND_PARAMS = 1_854_650
EXPECTED_TOTAL_PARAMS = BASE_HAND_PARAMS + GATE_EXTRA_PARAMS  # 1,857,051


def _margin(logits: jnp.ndarray) -> jnp.ndarray:
    top2, _ = jax.lax.top_k(logits, 2)
    return top2[:, 0] - top2[:, 1]


def _entropy(logits: jnp.ndarray) -> jnp.ndarray:
    logp = jax.nn.log_softmax(logits, axis=-1)
    p = jnp.exp(logp)
    return -jnp.sum(p * logp, axis=-1)


class AdaptiveHandGate(nn.Module):
    hidden_dim: int = GATE_HIDDEN_DIM
    max_alpha: float = 0.30

    @nn.compact
    def __call__(
        self,
        main_desc: jnp.ndarray,
        hand_desc: jnp.ndarray,
        main_logits: jnp.ndarray,
        hand_logits: jnp.ndarray,
    ) -> Mapping[str, jnp.ndarray]:
        if main_desc.shape[-1] != MAIN_DESC_DIM:
            raise ValueError(
                f"Expected main descriptor dim {MAIN_DESC_DIM}, got {main_desc.shape}"
            )
        if hand_desc.shape[-1] != HAND_DESC_DIM:
            raise ValueError(
                f"Expected hand descriptor dim {HAND_DESC_DIM}, got {hand_desc.shape}"
            )

        main_margin = _margin(main_logits)
        hand_margin = _margin(hand_logits)
        main_entropy = _entropy(main_logits)
        hand_entropy = _entropy(hand_logits)

        z = jnp.concatenate(
            [
                main_desc,
                hand_desc,
                main_margin[:, None],
                hand_margin[:, None],
                main_entropy[:, None],
                hand_entropy[:, None],
            ],
            axis=-1,
        )

        if z.shape[-1] != GATE_INPUT_DIM:
            raise RuntimeError(
                f"Gate input mismatch: got {z.shape[-1]}, expected {GATE_INPUT_DIM}"
            )

        h = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="fc1",
        )(z)
        h = nn.gelu(h)

        # Small final initialization keeps alpha near max_alpha/2 (~0.15),
        # between the trained fixed 0.10 and the observed useful 0.20-0.30 range.
        raw = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(0.01),
            bias_init=nn.initializers.zeros,
            name="fc2",
        )(h)[..., 0]

        alpha = self.max_alpha * jax.nn.sigmoid(raw)
        logits = main_logits + alpha[:, None] * hand_logits

        return {
            "logits": logits,
            "alpha": alpha,
            "gate_features": z,
            "main_margin": main_margin,
            "hand_margin": hand_margin,
            "main_entropy": main_entropy,
            "hand_entropy": hand_entropy,
        }


class M4LocalGlobalHandAdaptiveGateT32(nn.Module):
    """Deployment wrapper: validated base model plus adaptive trust gate."""

    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10
    hand_dim: int = 32
    gate_hidden_dim: int = GATE_HIDDEN_DIM
    max_alpha: float = 0.30

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        hand_x: jnp.ndarray,
        training: bool = False,
    ) -> Mapping[str, jnp.ndarray]:
        base_out = M4LocalGlobalHandM4G4T32(
            spatial_dim=self.spatial_dim,
            model_dim=self.model_dim,
            dropout=self.dropout,
            hand_dim=self.hand_dim,
            hand_residual_scale=0.0,
            name="base_hand_model",
        )(
            x,
            hand_x,
            training=training,
        )

        main_desc = jnp.mean(
            base_out["descriptors"],
            axis=1,
        )

        gate = AdaptiveHandGate(
            hidden_dim=self.gate_hidden_dim,
            max_alpha=self.max_alpha,
            name="adaptive_hand_gate",
        )(
            main_desc,
            base_out["hand_descriptor"],
            base_out["main_logits"],
            base_out["hand_logits"],
        )

        return {
            "logits": gate["logits"],
            "alpha": gate["alpha"],
            "main_logits": base_out["main_logits"],
            "hand_logits": base_out["hand_logits"],
            "main_descriptor": main_desc,
            "hand_descriptor": base_out["hand_descriptor"],
            "stream_logits": base_out["stream_logits"],
            "router_weights": base_out["router_weights"],
        }

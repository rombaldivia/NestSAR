#!/usr/bin/env python3
from __future__ import annotations

"""Selective residual trust gate for LocalGlobal V2 + Hand-M4/G4-Lite T32.

This is a diagnostic mechanism, not a paper benchmark.  The validated Hand-M4/G4
T32 model is frozen.  Only a tiny sample-wise gate is trained.

Instead of the previous bounded sigmoid gate, which saturated near its upper
limit, this gate is centered on the empirically safe common residual alpha=0.20:

    alpha(x) = base_alpha + delta_alpha * tanh(g(x))
    logits   = main_logits + alpha(x) * hand_logits

Defaults:
    base_alpha  = 0.20
    delta_alpha = 0.15
    alpha range = [0.05, 0.35]

Gate inputs:
  * mean main descriptor:          112
  * hand descriptor:                32
  * main logit top1-top2 margin:     1
  * hand logit top1-top2 margin:     1
  * main entropy:                    1
  * hand entropy:                    1
  * main top1 confidence:            1
  * hand top1 confidence:            1
  * same top1 prediction:            1
  * main prob of hand top1 class:    1
  * hand prob of main top1 class:    1
  ------------------------------------------------
  total:                            153

Default MLP: 153 -> 16 -> 1 = 2,481 parameters.
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
TRUST_FEATURES = 9
GATE_INPUT_DIM = MAIN_DESC_DIM + HAND_DESC_DIM + TRUST_FEATURES  # 153
GATE_HIDDEN_DIM = 16
GATE_EXTRA_PARAMS = (
    GATE_INPUT_DIM * GATE_HIDDEN_DIM
    + GATE_HIDDEN_DIM
    + GATE_HIDDEN_DIM
    + 1
)  # 2,481
BASE_HAND_PARAMS = 1_854_650
EXPECTED_TOTAL_PARAMS = BASE_HAND_PARAMS + GATE_EXTRA_PARAMS  # 1,857,131


def _margin(logits: jnp.ndarray) -> jnp.ndarray:
    top2, _ = jax.lax.top_k(logits, 2)
    return top2[:, 0] - top2[:, 1]


def _probs_entropy(logits: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    logp = jax.nn.log_softmax(logits, axis=-1)
    p = jnp.exp(logp)
    entropy = -jnp.sum(p * logp, axis=-1)
    return p, entropy


class SelectiveHandGate(nn.Module):
    hidden_dim: int = GATE_HIDDEN_DIM
    base_alpha: float = 0.20
    delta_alpha: float = 0.15

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

        main_p, main_entropy = _probs_entropy(main_logits)
        hand_p, hand_entropy = _probs_entropy(hand_logits)

        main_top = jnp.argmax(main_logits, axis=-1)
        hand_top = jnp.argmax(hand_logits, axis=-1)

        main_top_conf = jnp.max(main_p, axis=-1)
        hand_top_conf = jnp.max(hand_p, axis=-1)
        same_top = (main_top == hand_top).astype(main_logits.dtype)

        main_prob_hand_top = jnp.take_along_axis(
            main_p, hand_top[:, None], axis=-1
        )[:, 0]
        hand_prob_main_top = jnp.take_along_axis(
            hand_p, main_top[:, None], axis=-1
        )[:, 0]

        main_margin = _margin(main_logits)
        hand_margin = _margin(hand_logits)

        z = jnp.concatenate(
            [
                main_desc,
                hand_desc,
                main_margin[:, None],
                hand_margin[:, None],
                main_entropy[:, None],
                hand_entropy[:, None],
                main_top_conf[:, None],
                hand_top_conf[:, None],
                same_top[:, None],
                main_prob_hand_top[:, None],
                hand_prob_main_top[:, None],
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

        # Near-zero initialization starts the gate almost exactly at alpha=0.20
        # while still allowing gradients to flow into fc1 from the first update.
        raw = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(1e-3),
            bias_init=nn.initializers.zeros,
            name="fc2",
        )(h)[..., 0]

        delta = self.delta_alpha * jnp.tanh(raw)
        alpha = self.base_alpha + delta
        logits = main_logits + alpha[:, None] * hand_logits

        return {
            "logits": logits,
            "alpha": alpha,
            "delta_alpha": delta,
            "gate_features": z,
            "main_margin": main_margin,
            "hand_margin": hand_margin,
            "main_entropy": main_entropy,
            "hand_entropy": hand_entropy,
            "main_top_conf": main_top_conf,
            "hand_top_conf": hand_top_conf,
            "same_top": same_top,
            "main_prob_hand_top": main_prob_hand_top,
            "hand_prob_main_top": hand_prob_main_top,
        }


class M4LocalGlobalHandSelectiveGateT32(nn.Module):
    """Deployment wrapper: validated Hand-M4/G4 base plus selective trust gate."""

    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10
    hand_dim: int = 32
    gate_hidden_dim: int = GATE_HIDDEN_DIM
    base_alpha: float = 0.20
    delta_alpha: float = 0.15

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

        main_desc = jnp.mean(base_out["descriptors"], axis=1)

        gate = SelectiveHandGate(
            hidden_dim=self.gate_hidden_dim,
            base_alpha=self.base_alpha,
            delta_alpha=self.delta_alpha,
            name="selective_hand_gate",
        )(
            main_desc,
            base_out["hand_descriptor"],
            base_out["main_logits"],
            base_out["hand_logits"],
        )

        return {
            "logits": gate["logits"],
            "alpha": gate["alpha"],
            "delta_alpha": gate["delta_alpha"],
            "same_top": gate["same_top"],
            "main_logits": base_out["main_logits"],
            "hand_logits": base_out["hand_logits"],
            "main_descriptor": main_desc,
            "hand_descriptor": base_out["hand_descriptor"],
            "stream_logits": base_out["stream_logits"],
            "router_weights": base_out["router_weights"],
        }

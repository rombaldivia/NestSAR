#!/usr/bin/env python3
from __future__ import annotations

"""Standalone attention-free distal-motion specialist.

Only hands/finger-related joints and feet are processed. The specialist is
trained from random initialization and predicts all 120 NTU classes, making its
logits usable later as a routed/fused expert without retraining its head.
"""

from typing import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn

from experiments.nestsar_sm_all_t16.model import SelfModBiMemory
from experiments.distal_sm_specialist_t16.preprocessing import (
    DISTAL_JOINTS,
    FEATURES,
    FRAMES,
    PERSONS,
    TOKEN_CHANNELS,
)

NUM_CLASSES = 120


class DistalController(nn.Module):
    controller_dim: int = 16
    eta_max: float = 0.20
    alpha_min: float = 0.90
    alpha_max: float = 0.999
    input_gain: float = 0.10
    input_shift: float = 0.05

    @nn.compact
    def __call__(self, tok: jnp.ndarray) -> Mapping[str, jnp.ndarray]:
        # [B,T,M,J,C] -> cheap [B,T,C] summary.
        pooled = jnp.mean(tok, axis=(2, 3))
        h = nn.Dense(self.controller_dim, name="in_proj")(pooled)
        h = nn.LayerNorm(name="norm")(nn.gelu(h))

        zero = nn.initializers.zeros
        gamma_raw = nn.Dense(
            TOKEN_CHANNELS, kernel_init=zero, bias_init=zero, name="gamma"
        )(h)
        beta_raw = nn.Dense(
            TOKEN_CHANNELS, kernel_init=zero, bias_init=zero, name="beta"
        )(h)
        lr_raw = nn.Dense(2, kernel_init=zero, bias_init=zero, name="eta_alpha")(h)

        gamma = 1.0 + self.input_gain * jnp.tanh(gamma_raw)
        beta = self.input_shift * jnp.tanh(beta_raw)
        eta = self.eta_max * jax.nn.sigmoid(lr_raw[..., 0:1])
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * jax.nn.sigmoid(
            lr_raw[..., 1:2]
        )
        return {"gamma": gamma, "beta": beta, "eta": eta, "alpha": alpha}


class DistalSpatialEncoder(nn.Module):
    spatial_dim: int = 16
    model_dim: int = 64
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        b, t, m, j, _ = x.shape
        h = nn.Dense(self.spatial_dim, name="joint_proj")(x)
        je = self.param(
            "joint_embed",
            nn.initializers.normal(0.02),
            (1, 1, 1, DISTAL_JOINTS, self.spatial_dim),
        )
        pe = self.param(
            "person_embed",
            nn.initializers.normal(0.02),
            (1, 1, PERSONS, 1, self.spatial_dim),
        )
        h = nn.gelu(h + je + pe)
        h = h.reshape(b, t, m * j * self.spatial_dim)
        h = nn.Dense(self.model_dim, name="fuse")(h)
        h = nn.LayerNorm(name="norm")(nn.gelu(h))
        return nn.Dropout(self.dropout)(h, deterministic=not training)


class DistalSMSpecialistT16(nn.Module):
    spatial_dim: int = 16
    model_dim: int = 64
    controller_dim: int = 16
    fast_rank: int = 2
    dropout: float = 0.10
    sm_residual_scale: float = 0.08

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Mapping[str, jnp.ndarray]:
        if x.shape[1:] != (FRAMES, FEATURES):
            raise ValueError(f"Expected [B,{FRAMES},{FEATURES}], got {x.shape}")

        tok = x.reshape(
            x.shape[0], FRAMES, PERSONS, DISTAL_JOINTS, TOKEN_CHANNELS
        )
        ctrl = DistalController(
            controller_dim=self.controller_dim,
            name="controller",
        )(tok)

        valid = jnp.any(jnp.abs(tok[..., 0:3]) > 1e-8, axis=-1, keepdims=True).astype(tok.dtype)
        gamma = ctrl["gamma"][:, :, None, None, :]
        beta = ctrl["beta"][:, :, None, None, :]
        tok = tok * gamma + valid * beta

        h = DistalSpatialEncoder(
            spatial_dim=self.spatial_dim,
            model_dim=self.model_dim,
            dropout=self.dropout,
            name="spatial",
        )(tok, training)

        frame_h = SelfModBiMemory(
            dim=self.model_dim,
            rank=self.fast_rank,
            residual_scale=self.sm_residual_scale,
            name="m4_fast",
        )(h, ctrl["eta"], ctrl["alpha"])

        chunks = frame_h.reshape(
            frame_h.shape[0], 4, FRAMES // 4, self.model_dim
        ).mean(axis=2)
        eta_slow = ctrl["eta"].reshape(
            x.shape[0], 4, FRAMES // 4, 1
        ).mean(axis=2)
        alpha_slow = ctrl["alpha"].reshape(
            x.shape[0], 4, FRAMES // 4, 1
        ).mean(axis=2)

        chunk_h = SelfModBiMemory(
            dim=self.model_dim,
            rank=self.fast_rank,
            residual_scale=self.sm_residual_scale,
            name="g4_slow",
        )(chunks, eta_slow, alpha_slow)

        desc = jnp.concatenate(
            [frame_h.mean(axis=1), chunk_h.mean(axis=1)], axis=-1
        )
        desc = nn.Dense(self.model_dim, name="hier_fuse")(desc)
        desc = nn.LayerNorm(name="hier_norm")(nn.gelu(desc))
        desc = nn.Dropout(self.dropout)(desc, deterministic=not training)
        logits = nn.Dense(NUM_CLASSES, name="classifier")(desc)

        return {
            "logits": logits,
            "descriptor": desc,
            "frame_states": frame_h,
            "chunk_states": chunk_h,
            "eta_mean": jnp.mean(ctrl["eta"], axis=(1, 2)),
            "alpha_mean": jnp.mean(ctrl["alpha"], axis=(1, 2)),
        }

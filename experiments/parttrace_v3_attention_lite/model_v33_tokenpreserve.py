#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from experiments.parttrace_v3_attention_lite.audit_model import (
    FRAMES, PERSONS, JOINTS, COORDS, NUM_CLASSES, PART_MASK,
)

# v3.3 is intentionally the T16 Pareto experiment.
if FRAMES != 16:
    raise RuntimeError(f"TokenPreserve v3.3 expects the canonical T16 audit module; got FRAMES={FRAMES}")

PARENTS = jnp.asarray([
    0, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0,
    12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11,
], dtype=jnp.int32)

HAND_LEFT_PART = 3
HAND_RIGHT_PART = 5
LEFT_HAND = 7
LEFT_HAND_TIP = 21
LEFT_THUMB = 22
RIGHT_HAND = 11
RIGHT_HAND_TIP = 23
RIGHT_THUMB = 24
MAX_RESIDUAL_GATE = 0.15


def _safe_scale(xyz: jnp.ndarray, present: jnp.ndarray) -> jnp.ndarray:
    shoulder = jnp.linalg.norm(xyz[:, :, :, 4] - xyz[:, :, :, 8], axis=-1)
    hip = jnp.linalg.norm(xyz[:, :, :, 12] - xyz[:, :, :, 16], axis=-1)
    scale = 0.5 * (shoulder + hip)
    scale = jnp.where((present & (scale > 1e-4)), scale, 1.0)
    return scale[..., None, None]


def _hand_geometry(
    centered_n: jnp.ndarray,
    hand_idx: int,
    tip_idx: int,
    thumb_idx: int,
) -> jnp.ndarray:
    hand = centered_n[:, :, :, hand_idx, :]
    tip = centered_n[:, :, :, tip_idx, :]
    thumb = centered_n[:, :, :, thumb_idx, :]
    tip_hand = tip - hand
    thumb_hand = thumb - hand
    tip_thumb = tip - thumb
    distances = jnp.stack(
        [
            jnp.linalg.norm(tip_hand, axis=-1),
            jnp.linalg.norm(thumb_hand, axis=-1),
            jnp.linalg.norm(tip_thumb, axis=-1),
        ],
        axis=-1,
    )
    return jnp.concatenate([tip_hand, thumb_hand, tip_thumb, distances], axis=-1)


def _masked_entropy(prob: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Mean entropy of prob [B,Q,N] restricted to mask [B,N]."""
    masked = prob * mask[:, None, :].astype(prob.dtype)
    z = jnp.sum(masked, axis=-1, keepdims=True)
    normalized = jnp.where(z > 1e-8, masked / jnp.maximum(z, 1e-8), 0.0)
    entropy = -jnp.sum(
        jnp.where(normalized > 0.0, normalized * jnp.log(normalized + 1e-8), 0.0),
        axis=-1,
    )
    valid = (z[..., 0] > 1e-8).astype(prob.dtype)
    return jnp.sum(entropy * valid) / jnp.maximum(jnp.sum(valid), 1.0)


class TinyReadoutMixer(nn.Module):
    dim: int = 64
    heads: int = 4
    hidden_dim: int = 128
    dropout: float = 0.12

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        b, k, d = x.shape
        if d != self.dim or self.dim % self.heads:
            raise ValueError(f"Unexpected readout tensor {x.shape}")
        dh = d // self.heads

        z = nn.LayerNorm(name="attn_norm")(x)
        q = nn.Dense(d, use_bias=False, name="q")(z)
        key = nn.Dense(d, use_bias=False, name="k")(z)
        value = nn.Dense(d, use_bias=False, name="v")(z)
        q = q.reshape(b, k, self.heads, dh).transpose(0, 2, 1, 3)
        key = key.reshape(b, k, self.heads, dh).transpose(0, 2, 1, 3)
        value = value.reshape(b, k, self.heads, dh).transpose(0, 2, 1, 3)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, key) / math.sqrt(dh)
        weights = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.einsum("bhqk,bhkd->bhqd", weights, value)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(b, k, d)
        ctx = nn.Dense(d, name="out")(ctx)
        ctx = nn.Dropout(rate=self.dropout, name="attn_dropout")(
            ctx, deterministic=not training
        )
        attn_gate = jax.nn.sigmoid(
            self.param("attn_gate_logit", nn.initializers.constant(-2.1972246), (1,))
        )[0]
        x = x + attn_gate * ctx

        h = nn.LayerNorm(name="ffn_norm")(x)
        h = nn.Dense(self.hidden_dim, name="ffn_in")(h)
        h = jax.nn.gelu(h)
        h = nn.Dropout(rate=self.dropout, name="ffn_dropout")(h, deterministic=not training)
        h = nn.Dense(d, name="ffn_out")(h)
        ffn_gate = jax.nn.sigmoid(
            self.param("ffn_gate_logit", nn.initializers.constant(-2.1972246), (1,))
        )[0]
        return nn.LayerNorm(name="out_norm")(x + ffn_gate * h)


class TokenPreserveV33Branch(nn.Module):
    token_dim: int = 64
    heads: int = 4
    readout_tokens: int = 8
    mixer_hidden_dim: int = 128
    dense_dim: int = 192
    dropout: float = 0.12
    frame_mask_rate: float = 0.08
    joint_mask_rate: float = 0.08
    part_mask_rate: float = 0.03

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False):
        if x.ndim != 3 or x.shape[1] != FRAMES:
            raise ValueError(f"TokenPreserve v3.3 expects [B,{FRAMES},150], got {x.shape}")
        if self.token_dim <= 0 or self.token_dim % self.heads:
            raise ValueError("token_dim must be positive and divisible by heads")
        if self.readout_tokens <= 0:
            raise ValueError("readout_tokens must be > 0")

        b, t, _ = x.shape
        xyz = x.reshape(b, t, PERSONS, JOINTS, COORDS)
        person_present = jnp.any(jnp.abs(xyz) > 1e-6, axis=(3, 4))
        present_joint = person_present[..., None, None].astype(x.dtype)

        root = xyz[:, :, :, 0:1, :]
        centered = (xyz - root) * present_joint
        centered_n = centered / _safe_scale(xyz, person_present)

        pair_valid = (
            person_present[:, 1:] & person_present[:, :-1]
        )[..., None, None].astype(x.dtype)
        velocity = jnp.concatenate(
            [
                jnp.zeros_like(centered_n[:, :1]),
                (centered_n[:, 1:] - centered_n[:, :-1]) * pair_valid,
            ],
            axis=1,
        )
        acceleration = jnp.concatenate(
            [
                jnp.zeros_like(velocity[:, :1]),
                (velocity[:, 1:] - velocity[:, :-1]) * pair_valid,
            ],
            axis=1,
        )

        parent_xyz = jnp.take(centered_n, PARENTS, axis=3)
        bone = centered_n - parent_xyz
        torso = centered_n[:, :, :, 20:21, :]
        torso_dist = jnp.linalg.norm(centered_n - torso, axis=-1, keepdims=True)
        features = jnp.concatenate(
            [centered_n, velocity, acceleration, bone, torso_dist], axis=-1
        )

        h = nn.Dense(self.token_dim, name="joint_projection")(features)
        joint_emb = self.param(
            "joint_embedding",
            nn.initializers.normal(0.02),
            (1, 1, 1, JOINTS, self.token_dim),
        )
        person_emb = self.param(
            "person_embedding",
            nn.initializers.normal(0.02),
            (1, 1, PERSONS, 1, self.token_dim),
        )
        h = jax.nn.gelu(nn.LayerNorm(name="joint_norm")(h + joint_emb + person_emb))
        joint_valid = jnp.broadcast_to(
            person_present[..., None], (b, t, PERSONS, JOINTS)
        )

        part_keep = None
        if training:
            kf, kj, kp = jax.random.split(self.make_rng("dropout"), 3)
            frame_keep = jax.random.bernoulli(
                kf, 1.0 - self.frame_mask_rate, (b, t, PERSONS, 1, 1)
            ).astype(h.dtype)
            joint_keep = jax.random.bernoulli(
                kj, 1.0 - self.joint_mask_rate, (b, t, PERSONS, JOINTS, 1)
            ).astype(h.dtype)
            part_keep = jax.random.bernoulli(
                kp, 1.0 - self.part_mask_rate, (b, t, PERSONS, 10, 1)
            ).astype(h.dtype)
            h = h * frame_keep * joint_keep
            joint_valid = joint_valid & (joint_keep[..., 0] > 0.5)
            joint_valid = joint_valid & jnp.broadcast_to(
                (frame_keep[..., 0, 0] > 0.5)[..., None], joint_valid.shape
            )

        h = nn.Dropout(rate=self.dropout, name="joint_dropout")(
            h, deterministic=not training
        )
        h = h * present_joint

        # Learned joint -> semantic-part pooling. No uniform mean.
        part_queries = self.param(
            "joint_to_part_queries",
            nn.initializers.normal(0.02),
            (10, self.token_dim),
        )
        joint_part_logits = jnp.einsum(
            "btmjd,pd->btmpj", h, part_queries
        ) / math.sqrt(self.token_dim)
        anatomical_mask = PART_MASK.astype(bool)[None, None, None, :, :]
        valid_mask = anatomical_mask & joint_valid[..., None, :]
        joint_part_logits = jnp.where(valid_mask, joint_part_logits, -1e9)
        joint_part_w = jax.nn.softmax(joint_part_logits, axis=-1)
        part_has_joint = jnp.any(valid_mask, axis=-1, keepdims=True)
        joint_part_w = jnp.where(part_has_joint, joint_part_w, 0.0)
        parts = jnp.einsum("btmpj,btmjd->btmpd", joint_part_w, h)

        part_emb = self.param(
            "part_embedding",
            nn.initializers.normal(0.02),
            (1, 1, 1, 10, self.token_dim),
        )
        time_emb = self.param(
            "time_embedding",
            nn.initializers.normal(0.02),
            (1, FRAMES, 1, 1, self.token_dim),
        )

        # Explicit hand geometry is injected into its own anatomical tokens.
        left_geom = nn.Dense(self.token_dim, name="left_hand_geometry_projection")(
            _hand_geometry(centered_n, LEFT_HAND, LEFT_HAND_TIP, LEFT_THUMB)
        )
        right_geom = nn.Dense(self.token_dim, name="right_hand_geometry_projection")(
            _hand_geometry(centered_n, RIGHT_HAND, RIGHT_HAND_TIP, RIGHT_THUMB)
        )
        left_selector = jax.nn.one_hot(HAND_LEFT_PART, 10, dtype=parts.dtype)
        right_selector = jax.nn.one_hot(HAND_RIGHT_PART, 10, dtype=parts.dtype)
        hand_delta = (
            left_geom[..., None, :] * left_selector[None, None, None, :, None]
            + right_geom[..., None, :] * right_selector[None, None, None, :, None]
        )

        effective_part_mask = part_has_joint.astype(parts.dtype)
        if part_keep is not None:
            effective_part_mask = effective_part_mask * part_keep
        parts = nn.LayerNorm(name="part_norm")(
            parts + part_emb + time_emb + hand_delta
        )
        parts = nn.Dropout(rate=self.dropout, name="part_dropout")(
            parts, deterministic=not training
        ) * effective_part_mask

        # Preserve every T x person x part token until the learned readout.
        # T16 x 2 x 10 = 320 tokens.
        tokens = parts.reshape(b, FRAMES * PERSONS * 10, self.token_dim)
        token_valid = (effective_part_mask[..., 0] > 0.5).reshape(
            b, FRAMES * PERSONS * 10
        )
        n_tokens = tokens.shape[1]

        # K learned evidence queries cross-attend to all 320 preserved tokens.
        query_seed = self.param(
            "readout_queries",
            nn.initializers.normal(0.02),
            (self.readout_tokens, self.token_dim),
        )
        query = jnp.broadcast_to(
            query_seed[None, :, :], (b, self.readout_tokens, self.token_dim)
        )
        query_n = nn.LayerNorm(name="query_norm")(query)
        token_n = nn.LayerNorm(name="token_norm")(tokens)

        q = nn.Dense(self.token_dim, use_bias=False, name="readout_q")(query_n)
        key = nn.Dense(self.token_dim, use_bias=False, name="readout_k")(token_n)
        value = nn.Dense(self.token_dim, use_bias=False, name="readout_v")(token_n)
        dh = self.token_dim // self.heads
        q = q.reshape(b, self.readout_tokens, self.heads, dh).transpose(0, 2, 1, 3)
        key = key.reshape(b, n_tokens, self.heads, dh).transpose(0, 2, 1, 3)
        value = value.reshape(b, n_tokens, self.heads, dh).transpose(0, 2, 1, 3)
        scores = jnp.einsum("bhqd,bhnd->bhqn", q, key) / math.sqrt(dh)

        # Cheap learned anatomical prior per query; tokens still carry full identity.
        part_ids = jnp.tile(jnp.arange(10, dtype=jnp.int32), FRAMES * PERSONS)
        query_part_bias = self.param(
            "query_part_bias",
            nn.initializers.zeros,
            (self.readout_tokens, 10),
        )
        scores = scores + query_part_bias[:, part_ids][None, None, :, :]
        scores = jnp.where(token_valid[:, None, None, :], scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        any_valid = jnp.any(token_valid, axis=-1)[:, None, None, None]
        attn = jnp.where(any_valid, attn, 0.0)

        ctx = jnp.einsum("bhqn,bhnd->bhqd", attn, value)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(
            b, self.readout_tokens, self.token_dim
        )
        ctx = nn.Dense(self.token_dim, name="readout_out")(ctx)
        ctx = nn.Dropout(rate=self.dropout, name="readout_dropout")(
            ctx, deterministic=not training
        )
        readout = nn.LayerNorm(name="readout_residual_norm")(query + ctx)

        readout = TinyReadoutMixer(
            dim=self.token_dim,
            heads=self.heads,
            hidden_dim=self.mixer_hidden_dim,
            dropout=self.dropout,
            name="tiny_readout_mixer",
        )(readout, training=training)

        # Keep K identities distinct until the penultimate dense layer.
        flat = readout.reshape(b, self.readout_tokens * self.token_dim)
        feature = nn.Dense(self.dense_dim, name="fusion_hidden")(flat)
        feature = jax.nn.gelu(feature)
        feature = nn.Dropout(rate=self.dropout, name="fusion_dropout")(
            feature, deterministic=not training
        )
        branch_logits = nn.Dense(
            NUM_CLASSES,
            kernel_init=nn.initializers.normal(1e-3),
            bias_init=nn.initializers.zeros,
            name="branch_classifier",
        )(feature)

        attn_mean = jnp.mean(attn, axis=1)  # [B,Q,N]
        left_mask = token_valid & (part_ids[None, :] == HAND_LEFT_PART)
        right_mask = token_valid & (part_ids[None, :] == HAND_RIGHT_PART)
        all_entropy = _masked_entropy(attn_mean, token_valid)
        left_entropy = _masked_entropy(attn_mean, left_mask)
        right_entropy = _masked_entropy(attn_mean, right_mask)

        return {
            "branch_logits": branch_logits,
            "feature": feature,
            "joint_to_part_entropy": -jnp.mean(
                jnp.sum(
                    jnp.where(
                        joint_part_w > 0,
                        joint_part_w * jnp.log(joint_part_w + 1e-8),
                        0.0,
                    ),
                    axis=-1,
                )
            ),
            "temporal_entropy_global": all_entropy,
            "temporal_entropy_left": left_entropy,
            "temporal_entropy_right": right_entropy,
            "readout_attention": attn_mean,
        }


def make_wrapper_v33(
    base_model: nn.Module,
    *,
    part_dim: int = 64,
    part_heads: int = 4,
    global_dim: int = 128,
    dense_dim: int = 192,
    branch_dropout: float = 0.12,
    readout_tokens: int = 8,
):
    """Wrap exact Attention-Lite with a cheap T16 token-preserving residual head.

    `global_dim` is retained as a CLI-compatible name and is used as the tiny
    readout-mixer FFN hidden width. There is no second global temporal backbone.
    """

    class AttentionLiteTokenPreserveV33(nn.Module):
        base: nn.Module

        @nn.compact
        def __call__(
            self,
            x: jnp.ndarray,
            training: bool = False,
            branch_scale: float = 1.0,
        ):
            base_out = self.base(x, training=training)
            branch = TokenPreserveV33Branch(
                token_dim=part_dim,
                heads=part_heads,
                readout_tokens=readout_tokens,
                mixer_hidden_dim=global_dim,
                dense_dim=dense_dim,
                dropout=branch_dropout,
                name="parttrace_branch",
            )(x, training=training)

            raw_gate = self.param(
                "parttrace_residual_gate_logit",
                nn.initializers.constant(-1.38629436112),
                (1,),
            )
            gate = MAX_RESIDUAL_GATE * jax.nn.sigmoid(raw_gate)[0]
            effective_gate = jnp.asarray(branch_scale, jnp.float32) * gate
            logits = base_out["logits"] + effective_gate * branch["branch_logits"]

            base_w = jnp.asarray(base_out["fusion_weights"])
            if base_w.ndim == 1:
                dynamic_w = jnp.broadcast_to(base_w[None, :], (x.shape[0], base_w.shape[0]))
            else:
                dynamic_w = base_w

            result = dict(base_out)
            result.update(
                {
                    "base_logits": base_out["logits"],
                    "parttrace_logits": branch["branch_logits"],
                    "parttrace_feature": branch["feature"],
                    "dynamic_fusion_weights": dynamic_w,
                    "parttrace_gate": gate,
                    "effective_parttrace_gate": effective_gate,
                    "joint_to_part_entropy": branch["joint_to_part_entropy"],
                    "temporal_entropy_global": branch["temporal_entropy_global"],
                    "temporal_entropy_left": branch["temporal_entropy_left"],
                    "temporal_entropy_right": branch["temporal_entropy_right"],
                    "readout_attention": branch["readout_attention"],
                    "logits": logits,
                }
            )
            return result

    return AttentionLiteTokenPreserveV33(base=base_model)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import jax
import jax.numpy as jnp
from flax import linen as nn

from experiments.parttrace_v3_attention_lite.audit_model import (
    FRAMES, PERSONS, JOINTS, COORDS, NUM_CLASSES,
    PART_DIM, PART_HEADS, PART_MASK,
    PartMixer10, SharedPartTemporal,
)

# NTU-120 25-joint parent map (zero-based).
PARENTS = jnp.asarray([
    0, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0,
    12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11,
], dtype=jnp.int32)

HAND_LEFT = 3
HAND_RIGHT = 5
MAX_RESIDUAL_GATE = 0.15
CONTROLLER_STRENGTH = 0.20


def _safe_scale(xyz: jnp.ndarray, present: jnp.ndarray) -> jnp.ndarray:
    """Body-scale normalization using shoulder+hip width with safe fallback."""
    shoulder = jnp.linalg.norm(xyz[:, :, :, 4] - xyz[:, :, :, 8], axis=-1)
    hip = jnp.linalg.norm(xyz[:, :, :, 12] - xyz[:, :, :, 16], axis=-1)
    scale = 0.5 * (shoulder + hip)
    scale = jnp.where((present & (scale > 1e-4)), scale, 1.0)
    return scale[..., None, None]


class PartTraceV31Branch(nn.Module):
    dim: int = PART_DIM
    heads: int = PART_HEADS
    dropout: float = 0.12
    frame_mask_rate: float = 0.08
    joint_mask_rate: float = 0.08
    part_mask_rate: float = 0.03

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False):
        b, t, _ = x.shape
        xyz = x.reshape(b, t, PERSONS, JOINTS, COORDS)
        person_present = jnp.any(jnp.abs(xyz) > 1e-6, axis=(3, 4))
        present_joint = person_present[..., None, None].astype(x.dtype)

        root = xyz[:, :, :, 0:1, :]
        centered = (xyz - root) * present_joint
        scale = _safe_scale(xyz, person_present)
        centered_n = centered / scale

        pair_valid = (person_present[:, 1:] & person_present[:, :-1])[..., None, None].astype(x.dtype)
        velocity = jnp.concatenate([
            jnp.zeros_like(centered_n[:, :1]),
            (centered_n[:, 1:] - centered_n[:, :-1]) * pair_valid,
        ], axis=1)
        acceleration = jnp.concatenate([
            jnp.zeros_like(velocity[:, :1]),
            (velocity[:, 1:] - velocity[:, :-1]) * pair_valid,
        ], axis=1)

        parent_xyz = jnp.take(centered_n, PARENTS, axis=3)
        bone = centered_n - parent_xyz
        torso = centered_n[:, :, :, 20:21, :]
        torso_dist = jnp.linalg.norm(centered_n - torso, axis=-1, keepdims=True)

        # 13 channels: xyz(3) + vel(3) + acc(3) + bone(3) + torso distance(1)
        features = jnp.concatenate([
            centered_n, velocity, acceleration, bone, torso_dist
        ], axis=-1)

        h = nn.Dense(self.dim, name="joint_projection")(features)
        joint_emb = self.param(
            "joint_embedding", nn.initializers.normal(0.02),
            (1, 1, 1, JOINTS, self.dim),
        )
        person_emb = self.param(
            "person_embedding", nn.initializers.normal(0.02),
            (1, 1, PERSONS, 1, self.dim),
        )
        h = jax.nn.gelu(nn.LayerNorm(name="joint_norm")(h + joint_emb + person_emb))

        if training:
            kf = self.make_rng("dropout")
            kj = self.make_rng("dropout")
            frame_keep = jax.random.bernoulli(
                kf, 1.0 - self.frame_mask_rate, (b, t, PERSONS, 1, 1)
            ).astype(h.dtype)
            joint_keep = jax.random.bernoulli(
                kj, 1.0 - self.joint_mask_rate, (b, t, PERSONS, JOINTS, 1)
            ).astype(h.dtype)
            h = h * frame_keep * joint_keep

        h = nn.Dropout(rate=self.dropout, name="joint_dropout")(h, deterministic=not training)
        h = h * present_joint

        mask = PART_MASK.astype(h.dtype)
        numerator = jnp.einsum("btmjd,pj->btmpd", h, mask)
        counts = jnp.sum(mask, axis=1)[None, None, None, :, None]
        parts = numerator / jnp.maximum(counts, 1.0)

        part_emb = self.param(
            "part_embedding", nn.initializers.normal(0.02),
            (1, 1, 1, 10, self.dim),
        )
        part_present = person_present[..., None, None].astype(parts.dtype)
        parts = nn.LayerNorm(name="part_norm")(parts + part_emb) * part_present

        if training:
            kp = self.make_rng("dropout")
            part_keep = jax.random.bernoulli(
                kp, 1.0 - self.part_mask_rate, (b, t, PERSONS, 10, 1)
            ).astype(parts.dtype)
            parts = parts * part_keep

        parts = PartMixer10(self.dim, self.heads, name="part_mixer")(parts)
        parts = nn.Dropout(rate=self.dropout, name="part_dropout")(parts, deterministic=not training)
        parts = parts * part_present

        tracks = parts.transpose(0, 2, 3, 1, 4).reshape(b * PERSONS * 10, t, self.dim)
        tracks = SharedPartTemporal(self.dim, self.heads, name="shared_part_temporal")(tracks)
        parts = tracks.reshape(b, PERSONS, 10, t, self.dim).transpose(0, 3, 1, 2, 4)
        parts = parts * part_present

        # Explicit fine-motion taps: left/right forearm+hand survive part pooling.
        left = parts[:, :, :, HAND_LEFT, :]
        right = parts[:, :, :, HAND_RIGHT, :]
        pw = person_present[..., None].astype(parts.dtype)
        denom = jnp.maximum(jnp.sum(pw, axis=2), 1.0)
        left = jnp.sum(left * pw, axis=2) / denom
        right = jnp.sum(right * pw, axis=2) / denom

        gate_logits = nn.Dense(1, name="part_pool_gate")(parts)[..., 0]
        gate_logits = jnp.where(person_present[..., None], gate_logits, -1e9)
        part_w = jax.nn.softmax(gate_logits, axis=3)
        person_desc = jnp.sum(part_w[..., None] * parts, axis=3)

        first, second = person_desc[:, :, 0], person_desc[:, :, 1]
        pair = jnp.concatenate([
            first + second,
            jnp.abs(first - second),
            first * second,
        ], axis=-1)

        global_trace = nn.Dense(128, name="global_projection")(pair)
        any_person = jnp.any(person_present, axis=2)
        global_trace = global_trace * any_person[..., None].astype(global_trace.dtype)

        temporal_scores = nn.Dense(1, name="temporal_gate")(global_trace)[..., 0]
        temporal_scores = jnp.where(any_person, temporal_scores, -1e9)
        tw = jax.nn.softmax(temporal_scores, axis=1)

        global_pooled = jnp.sum(tw[..., None] * global_trace, axis=1)
        left_pooled = jnp.sum(tw[..., None] * left, axis=1)
        right_pooled = jnp.sum(tw[..., None] * right, axis=1)

        fused = jnp.concatenate([global_pooled, left_pooled, right_pooled], axis=-1)
        fused = jax.nn.gelu(nn.Dense(192, name="fusion_hidden")(fused))
        fused = nn.Dropout(rate=self.dropout, name="fusion_dropout")(fused, deterministic=not training)

        branch_logits = nn.Dense(
            NUM_CLASSES,
            kernel_init=nn.initializers.normal(1e-3),
            bias_init=nn.initializers.zeros,
            name="branch_classifier",
        )(fused)

        # Dynamic controller is zero-initialized -> exact base fusion at initialization.
        controller_delta = nn.Dense(
            4,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="stream_controller",
        )(fused)

        return {
            "branch_logits": branch_logits,
            "controller_delta": controller_delta,
            "feature": fused,
        }


def make_wrapper_v31(base_model: nn.Module):
    class AttentionLitePartTraceV31(nn.Module):
        base: nn.Module

        @nn.compact
        def __call__(self, x, training: bool = False, branch_scale: float = 1.0):
            base_out = self.base(x, training=training)
            if "stream_logits" not in base_out or "fusion_weights" not in base_out:
                raise RuntimeError("Canonical Attention-Lite output lacks stream_logits/fusion_weights")

            branch = PartTraceV31Branch(name="parttrace_branch")(x, training=training)

            base_w = jnp.asarray(base_out["fusion_weights"])
            if base_w.ndim == 1:
                base_logw = jnp.log(jnp.clip(base_w, 1e-8, 1.0))[None, :]
            else:
                base_logw = jnp.log(jnp.clip(base_w, 1e-8, 1.0))

            controller = CONTROLLER_STRENGTH * jnp.asarray(branch_scale) * jnp.tanh(branch["controller_delta"])
            dynamic_w = jax.nn.softmax(base_logw + controller, axis=-1)
            controlled_logits = jnp.einsum("bs,bsc->bc", dynamic_w, base_out["stream_logits"])

            # Max contribution 15%; init=3% (= 0.15 * sigmoid(-1.386294)).
            raw_gate = self.param(
                "parttrace_residual_gate_logit",
                nn.initializers.constant(-1.38629436112),
                (1,),
            )
            gate = MAX_RESIDUAL_GATE * jax.nn.sigmoid(raw_gate)[0]
            effective_gate = jnp.asarray(branch_scale) * gate
            logits = controlled_logits + effective_gate * branch["branch_logits"]

            result = dict(base_out)
            result.update({
                "base_logits": base_out["logits"],
                "controlled_logits": controlled_logits,
                "parttrace_logits": branch["branch_logits"],
                "parttrace_feature": branch["feature"],
                "dynamic_fusion_weights": dynamic_w,
                "parttrace_gate": gate,
                "effective_parttrace_gate": effective_gate,
                "logits": logits,
            })
            return result

    return AttentionLitePartTraceV31(base=base_model)

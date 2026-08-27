#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import jax
import jax.numpy as jnp
from flax import linen as nn

from experiments.parttrace_v3_attention_lite.audit_model import (
    FRAMES, PERSONS, JOINTS, COORDS, NUM_CLASSES,
    PART_MASK, PartMixer10, SharedPartTemporal,
)

PARENTS = jnp.asarray([
    0, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0,
    12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11,
], dtype=jnp.int32)

HAND_LEFT = 3
HAND_RIGHT = 5
LEFT_HAND = 7
LEFT_HAND_TIP = 21
LEFT_THUMB = 22
RIGHT_HAND = 11
RIGHT_HAND_TIP = 23
RIGHT_THUMB = 24
MAX_RESIDUAL_GATE = 0.15
CONTROLLER_STRENGTH = 0.20


def _safe_scale(xyz: jnp.ndarray, present: jnp.ndarray) -> jnp.ndarray:
    shoulder = jnp.linalg.norm(xyz[:, :, :, 4] - xyz[:, :, :, 8], axis=-1)
    hip = jnp.linalg.norm(xyz[:, :, :, 12] - xyz[:, :, :, 16], axis=-1)
    scale = 0.5 * (shoulder + hip)
    scale = jnp.where((present & (scale > 1e-4)), scale, 1.0)
    return scale[..., None, None]


def _hand_geometry(centered_n: jnp.ndarray, hand_idx: int, tip_idx: int, thumb_idx: int) -> jnp.ndarray:
    hand = centered_n[:, :, :, hand_idx, :]
    tip = centered_n[:, :, :, tip_idx, :]
    thumb = centered_n[:, :, :, thumb_idx, :]
    tip_hand = tip - hand
    thumb_hand = thumb - hand
    tip_thumb = tip - thumb
    distances = jnp.stack([
        jnp.linalg.norm(tip_hand, axis=-1),
        jnp.linalg.norm(thumb_hand, axis=-1),
        jnp.linalg.norm(tip_thumb, axis=-1),
    ], axis=-1)
    return jnp.concatenate([tip_hand, thumb_hand, tip_thumb, distances], axis=-1)


class PartTraceV32Branch(nn.Module):
    part_dim: int = 64
    part_heads: int = 4
    global_dim: int = 128
    dense_dim: int = 192
    dropout: float = 0.12
    frame_mask_rate: float = 0.08
    joint_mask_rate: float = 0.08
    part_mask_rate: float = 0.03

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False):
        if self.part_dim % self.part_heads:
            raise ValueError(
                f"part_dim={self.part_dim} must be divisible by part_heads={self.part_heads}"
            )
        b, t, _ = x.shape
        xyz = x.reshape(b, t, PERSONS, JOINTS, COORDS)
        person_present = jnp.any(jnp.abs(xyz) > 1e-6, axis=(3, 4))
        present_joint = person_present[..., None, None].astype(x.dtype)

        root = xyz[:, :, :, 0:1, :]
        centered = (xyz - root) * present_joint
        centered_n = centered / _safe_scale(xyz, person_present)

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
        features = jnp.concatenate([centered_n, velocity, acceleration, bone, torso_dist], axis=-1)

        h = nn.Dense(self.part_dim, name="joint_projection")(features)
        joint_emb = self.param(
            "joint_embedding", nn.initializers.normal(0.02),
            (1, 1, 1, JOINTS, self.part_dim),
        )
        person_emb = self.param(
            "person_embedding", nn.initializers.normal(0.02),
            (1, 1, PERSONS, 1, self.part_dim),
        )
        h = jax.nn.gelu(nn.LayerNorm(name="joint_norm")(h + joint_emb + person_emb))
        joint_valid = jnp.broadcast_to(person_present[..., None], (b, t, PERSONS, JOINTS))

        if training:
            kf, kj = jax.random.split(self.make_rng("dropout"))
            frame_keep = jax.random.bernoulli(
                kf, 1.0 - self.frame_mask_rate, (b, t, PERSONS, 1, 1)
            ).astype(h.dtype)
            joint_keep = jax.random.bernoulli(
                kj, 1.0 - self.joint_mask_rate, (b, t, PERSONS, JOINTS, 1)
            ).astype(h.dtype)
            h = h * frame_keep * joint_keep
            joint_valid = joint_valid & (joint_keep[..., 0] > 0.5)
            joint_valid = joint_valid & jnp.broadcast_to(
                (frame_keep[..., 0, 0] > 0.5)[..., None], joint_valid.shape
            )

        h = nn.Dropout(rate=self.dropout, name="joint_dropout")(h, deterministic=not training)
        h = h * present_joint

        # Learned within-part joint pooling.
        part_queries = self.param(
            "joint_to_part_queries", nn.initializers.normal(0.02), (10, self.part_dim)
        )
        joint_part_logits = jnp.einsum(
            "btmjd,pd->btmpj", h, part_queries
        ) / math.sqrt(self.part_dim)
        anatomical_mask = PART_MASK.astype(bool)[None, None, None, :, :]
        valid_mask = anatomical_mask & joint_valid[..., None, :]
        joint_part_logits = jnp.where(valid_mask, joint_part_logits, -1e9)
        joint_part_w = jax.nn.softmax(joint_part_logits, axis=-1)
        part_has_joint = jnp.any(valid_mask, axis=-1, keepdims=True)
        joint_part_w = jnp.where(part_has_joint, joint_part_w, 0.0)
        parts = jnp.einsum("btmpj,btmjd->btmpd", joint_part_w, h)

        part_emb = self.param(
            "part_embedding", nn.initializers.normal(0.02),
            (1, 1, 1, 10, self.part_dim),
        )
        effective_part_mask = part_has_joint.astype(parts.dtype)
        parts = nn.LayerNorm(name="part_norm")(parts + part_emb) * effective_part_mask

        if training:
            kp = self.make_rng("dropout")
            part_keep = jax.random.bernoulli(
                kp, 1.0 - self.part_mask_rate, (b, t, PERSONS, 10, 1)
            ).astype(parts.dtype)
            effective_part_mask = effective_part_mask * part_keep
            parts = parts * effective_part_mask

        parts = PartMixer10(self.part_dim, self.part_heads, name="part_mixer")(parts)
        parts = nn.Dropout(rate=self.dropout, name="part_dropout")(parts, deterministic=not training)
        parts = parts * effective_part_mask

        tracks = parts.transpose(0, 2, 3, 1, 4).reshape(b * PERSONS * 10, t, self.part_dim)
        tracks = SharedPartTemporal(self.part_dim, self.part_heads, name="shared_part_temporal")(tracks)
        parts = tracks.reshape(b, PERSONS, 10, t, self.part_dim).transpose(0, 3, 1, 2, 4)
        parts = parts * effective_part_mask

        left = parts[:, :, :, HAND_LEFT, :]
        right = parts[:, :, :, HAND_RIGHT, :]
        pw = person_present[..., None].astype(parts.dtype)
        denom = jnp.maximum(jnp.sum(pw, axis=2), 1.0)
        left = jnp.sum(left * pw, axis=2) / denom
        right = jnp.sum(right * pw, axis=2) / denom

        left_geom = _hand_geometry(centered_n, LEFT_HAND, LEFT_HAND_TIP, LEFT_THUMB)
        right_geom = _hand_geometry(centered_n, RIGHT_HAND, RIGHT_HAND_TIP, RIGHT_THUMB)
        left_geom = jnp.sum(left_geom * pw, axis=2) / denom
        right_geom = jnp.sum(right_geom * pw, axis=2) / denom
        left_geom = nn.Dense(self.part_dim, name="left_hand_geometry_projection")(left_geom)
        right_geom = nn.Dense(self.part_dim, name="right_hand_geometry_projection")(right_geom)
        left = nn.LayerNorm(name="left_hand_geometry_norm")(left + left_geom)
        right = nn.LayerNorm(name="right_hand_geometry_norm")(right + right_geom)

        gate_logits = nn.Dense(1, name="part_pool_gate")(parts)[..., 0]
        part_available = effective_part_mask[..., 0] > 0.5
        gate_logits = jnp.where(part_available, gate_logits, -1e9)
        part_w = jax.nn.softmax(gate_logits, axis=3)
        any_part = jnp.any(part_available, axis=3, keepdims=True)
        part_w = jnp.where(any_part, part_w, 0.0)
        person_desc = jnp.sum(part_w[..., None] * parts, axis=3)

        first, second = person_desc[:, :, 0], person_desc[:, :, 1]
        pair = jnp.concatenate([first + second, jnp.abs(first - second), first * second], axis=-1)
        global_trace = nn.Dense(self.global_dim, name="global_projection")(pair)
        any_person = jnp.any(person_present, axis=2)
        valid_time = any_person[..., None]
        global_trace = global_trace * valid_time.astype(global_trace.dtype)
        left = left * valid_time.astype(left.dtype)
        right = right * valid_time.astype(right.dtype)

        # Independent temporal gates: body, left hand, right hand.
        global_scores = nn.Dense(1, name="temporal_gate_global")(global_trace)[..., 0]
        left_scores = nn.Dense(1, name="temporal_gate_left")(left)[..., 0]
        right_scores = nn.Dense(1, name="temporal_gate_right")(right)[..., 0]
        global_scores = jnp.where(any_person, global_scores, -1e9)
        left_scores = jnp.where(any_person, left_scores, -1e9)
        right_scores = jnp.where(any_person, right_scores, -1e9)
        wg = jax.nn.softmax(global_scores, axis=1)
        wl = jax.nn.softmax(left_scores, axis=1)
        wr = jax.nn.softmax(right_scores, axis=1)

        global_pooled = jnp.sum(wg[..., None] * global_trace, axis=1)
        left_pooled = jnp.sum(wl[..., None] * left, axis=1)
        right_pooled = jnp.sum(wr[..., None] * right, axis=1)

        fused = jnp.concatenate([global_pooled, left_pooled, right_pooled], axis=-1)
        fused = jax.nn.gelu(nn.Dense(self.dense_dim, name="fusion_hidden")(fused))
        fused = nn.Dropout(rate=self.dropout, name="fusion_dropout")(fused, deterministic=not training)

        branch_logits = nn.Dense(
            NUM_CLASSES,
            kernel_init=nn.initializers.normal(1e-3),
            bias_init=nn.initializers.zeros,
            name="branch_classifier",
        )(fused)
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
            "joint_to_part_entropy": -jnp.mean(jnp.sum(
                jnp.where(joint_part_w > 0, joint_part_w * jnp.log(joint_part_w + 1e-8), 0.0),
                axis=-1,
            )),
            "temporal_entropy_global": -jnp.mean(jnp.sum(wg * jnp.log(wg + 1e-8), axis=-1)),
            "temporal_entropy_left": -jnp.mean(jnp.sum(wl * jnp.log(wl + 1e-8), axis=-1)),
            "temporal_entropy_right": -jnp.mean(jnp.sum(wr * jnp.log(wr + 1e-8), axis=-1)),
        }


def make_wrapper_v32(
    base_model: nn.Module,
    *,
    part_dim: int = 64,
    part_heads: int = 4,
    global_dim: int = 128,
    dense_dim: int = 192,
    branch_dropout: float = 0.12,
):
    class AttentionLitePartTraceV32(nn.Module):
        base: nn.Module

        @nn.compact
        def __call__(self, x, training: bool = False, branch_scale: float = 1.0):
            base_out = self.base(x, training=training)
            if "stream_logits" not in base_out or "fusion_weights" not in base_out:
                raise RuntimeError("Canonical Attention-Lite output lacks stream_logits/fusion_weights")

            branch = PartTraceV32Branch(
                part_dim=part_dim,
                part_heads=part_heads,
                global_dim=global_dim,
                dense_dim=dense_dim,
                dropout=branch_dropout,
                name="parttrace_branch",
            )(x, training=training)

            base_w = jnp.asarray(base_out["fusion_weights"])
            if base_w.ndim == 1:
                base_logw = jnp.log(jnp.clip(base_w, 1e-8, 1.0))[None, :]
            else:
                base_logw = jnp.log(jnp.clip(base_w, 1e-8, 1.0))

            controller = CONTROLLER_STRENGTH * jnp.asarray(branch_scale) * jnp.tanh(branch["controller_delta"])
            dynamic_w = jax.nn.softmax(base_logw + controller, axis=-1)
            controlled_logits = jnp.einsum("bs,bsc->bc", dynamic_w, base_out["stream_logits"])

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
                "joint_to_part_entropy": branch["joint_to_part_entropy"],
                "temporal_entropy_global": branch["temporal_entropy_global"],
                "temporal_entropy_left": branch["temporal_entropy_left"],
                "temporal_entropy_right": branch["temporal_entropy_right"],
                "logits": logits,
            })
            return result

    return AttentionLitePartTraceV32(base=base_model)

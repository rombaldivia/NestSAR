#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NestSAR v3.5 Cross-Stream Multi-Resolution Memory.

Accuracy-first, low-compute extension of the exact canonical Attention-Lite T16 model.

The canonical base is not modified. Its four stream logits (joint, bone,
joint-motion, bone-motion) are treated as semantic stream tokens. A tiny
cross-stream bridge conditions a multi-resolution skeleton memory:

  1) four raw geometric modalities remain separate;
  2) a shared 4-token local stream mixer interacts them at every joint/time/person;
  3) coarse 10-part D32 memory + fine arm/hand D24 memory are preserved;
  4) cheap depthwise temporal enrichment adds local motion context;
  5) K=8 conditioned queries read the concatenated memory;
  6) a low-rank class-conditioned gate applies bounded residual corrections.

The design avoids a second full temporal backbone and avoids N x N memory attention.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from experiments.parttrace_v3_attention_lite.audit_model import (
    FRAMES, PERSONS, JOINTS, COORDS, NUM_CLASSES, PART_MASK,
)

if FRAMES != 16:
    raise RuntimeError(f"CrossStreamMemory v3.5 expects T16; audit FRAMES={FRAMES}")

NUM_STREAMS = 4
PARENTS = jnp.asarray([
    0, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0,
    12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11,
], dtype=jnp.int32)

# elbow, wrist, hand, hand-tip, thumb for left and right arms.
FINE_JOINTS = jnp.asarray([5, 6, 7, 21, 22, 9, 10, 11, 23, 24], dtype=jnp.int32)
FINE_COUNT = 10
MAX_CLASS_GATE = 0.20
DEFAULT_STREAM_REWEIGHT = 0.08


def _safe_scale(xyz: jnp.ndarray, present: jnp.ndarray) -> jnp.ndarray:
    shoulder = jnp.linalg.norm(xyz[:, :, :, 4] - xyz[:, :, :, 8], axis=-1)
    hip = jnp.linalg.norm(xyz[:, :, :, 12] - xyz[:, :, :, 16], axis=-1)
    scale = 0.5 * (shoulder + hip)
    scale = jnp.where((present & (scale > 1e-4)), scale, 1.0)
    return scale[..., None, None]


def _query_diversity(attn: jnp.ndarray, valid: jnp.ndarray):
    """Anti-collapse loss for attention [B,Q,N] with valid [B,N]."""
    a = attn * valid[:, None, :].astype(attn.dtype)
    a = a / jnp.sqrt(jnp.sum(jnp.square(a), axis=-1, keepdims=True) + 1e-8)
    gram = jnp.einsum("bqn,bkn->bqk", a, a)
    q = attn.shape[1]
    off = 1.0 - jnp.eye(q, dtype=attn.dtype)[None]
    denom = jnp.maximum(jnp.asarray(q * max(q - 1, 1), attn.dtype), 1.0)
    overlap = gram * off
    loss = jnp.mean(jnp.sum(jnp.square(overlap), axis=(1, 2)) / denom)
    mean = jnp.mean(jnp.sum(overlap, axis=(1, 2)) / denom)
    max_overlap = jnp.max(overlap)
    return loss, mean, max_overlap


class TinySelfMixer(nn.Module):
    dim: int
    heads: int
    hidden_dim: int
    dropout: float = 0.08

    @nn.compact
    def __call__(self, x, training: bool):
        b, n, d = x.shape
        if d != self.dim or d % self.heads:
            raise ValueError(f"TinySelfMixer got shape={x.shape}, dim={self.dim}, heads={self.heads}")
        dh = d // self.heads
        z = nn.LayerNorm(name="attn_norm")(x)
        q = nn.Dense(d, use_bias=False, name="q")(z)
        k = nn.Dense(d, use_bias=False, name="k")(z)
        v = nn.Dense(d, use_bias=False, name="v")(z)
        q = q.reshape(b, n, self.heads, dh).transpose(0, 2, 1, 3)
        k = k.reshape(b, n, self.heads, dh).transpose(0, 2, 1, 3)
        v = v.reshape(b, n, self.heads, dh).transpose(0, 2, 1, 3)
        a = jax.nn.softmax(jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(dh), axis=-1)
        ctx = jnp.einsum("bhqk,bhkd->bhqd", a, v)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(b, n, d)
        ctx = nn.Dense(d, name="out")(ctx)
        ctx = nn.Dropout(rate=self.dropout, name="attn_dropout")(ctx, deterministic=not training)
        x = x + 0.20 * ctx
        h = nn.LayerNorm(name="ffn_norm")(x)
        h = nn.Dense(self.hidden_dim, name="ffn_in")(h)
        h = jax.nn.gelu(h)
        h = nn.Dropout(rate=self.dropout, name="ffn_dropout")(h, deterministic=not training)
        h = nn.Dense(d, name="ffn_out")(h)
        return nn.LayerNorm(name="out_norm")(x + 0.20 * h)


class SemanticStreamBridge(nn.Module):
    dim: int = 32
    heads: int = 4
    dropout: float = 0.08

    @nn.compact
    def __call__(self, stream_logits, training: bool):
        if stream_logits.ndim != 3 or stream_logits.shape[1] != NUM_STREAMS:
            raise ValueError(f"Expected [B,4,C] stream_logits, got {stream_logits.shape}")
        h = nn.Dense(self.dim, name="stream_projection")(stream_logits)
        emb = self.param("stream_embedding", nn.initializers.normal(0.02),
                         (1, NUM_STREAMS, self.dim))
        h = jax.nn.gelu(nn.LayerNorm(name="stream_norm")(h + emb))
        h = TinySelfMixer(dim=self.dim, heads=self.heads,
                          hidden_dim=2 * self.dim, dropout=self.dropout,
                          name="bridge_mixer")(h, training=training)
        score = nn.Dense(1, name="pool_score")(h)[..., 0]
        w = jax.nn.softmax(score, axis=1)
        global_ctx = jnp.sum(w[..., None] * h, axis=1)
        delta = nn.Dense(NUM_STREAMS, kernel_init=nn.initializers.zeros,
                         bias_init=nn.initializers.zeros,
                         name="fusion_delta")(global_ctx)
        return {"stream_tokens": h, "global_context": global_ctx,
                "stream_pool_weights": w, "fusion_delta": delta}


class LocalFourStreamMixer(nn.Module):
    dim: int = 16
    heads: int = 4

    @nn.compact
    def __call__(self, x):
        shape = x.shape
        if shape[-2] != NUM_STREAMS or shape[-1] != self.dim:
            raise ValueError(f"Expected [...,4,{self.dim}], got {shape}")
        flat = x.reshape((-1, NUM_STREAMS, self.dim))
        mixed = TinySelfMixer(dim=self.dim, heads=self.heads,
                              hidden_dim=2 * self.dim, dropout=0.0,
                              name="local_stream_mixer")(flat, training=False)
        return mixed.reshape(shape)


class TemporalDepthwiseContext(nn.Module):
    dim: int
    dropout: float = 0.08

    @nn.compact
    def __call__(self, x, valid, training: bool):
        # x [B,T,M,N,D], valid [B,T,M,N]
        b, t, m, n, d = x.shape
        tracks = x.transpose(0, 2, 3, 1, 4).reshape(b * m * n, t, d)
        z = nn.LayerNorm(name="norm")(tracks)
        z1 = jnp.pad(z, ((0, 0), (2, 0), (0, 0)))
        c1 = nn.Conv(features=d, kernel_size=(3,), padding="VALID",
                     feature_group_count=d, use_bias=False,
                     name="dw_d1")(z1)
        z2 = jnp.pad(z, ((0, 0), (4, 0), (0, 0)))
        c2 = nn.Conv(features=d, kernel_size=(3,), padding="VALID",
                     kernel_dilation=(2,), feature_group_count=d,
                     use_bias=False, name="dw_d2")(z2)
        gates = self.param("context_gates", nn.initializers.constant(-1.38629436112), (2,))
        g = jax.nn.sigmoid(gates)
        ctx = g[0] * jax.nn.silu(c1) + g[1] * jax.nn.silu(c2)
        ctx = nn.Dropout(rate=self.dropout, name="dropout")(ctx, deterministic=not training)
        tracks = tracks + 0.5 * ctx
        out = tracks.reshape(b, m, n, t, d).transpose(0, 3, 1, 2, 4)
        return out * valid[..., None].astype(out.dtype)


class CrossStreamMultiResolutionMemory(nn.Module):
    bridge_dim: int = 32
    local_stream_dim: int = 16
    memory_dim: int = 32
    fine_dim: int = 24
    readout_tokens: int = 8
    readout_heads: int = 4
    dense_dim: int = 128
    dropout: float = 0.08

    @nn.compact
    def __call__(self, x, bridge, training: bool):
        b, t, _ = x.shape
        xyz = x.reshape(b, t, PERSONS, JOINTS, COORDS)
        present = jnp.any(jnp.abs(xyz) > 1e-6, axis=(3, 4))
        pj = present[..., None, None].astype(x.dtype)
        root = xyz[:, :, :, 0:1]
        joint = (xyz - root) * pj
        joint = joint / _safe_scale(xyz, present)
        pair_valid = (present[:, 1:] & present[:, :-1])[..., None, None].astype(x.dtype)
        joint_motion = jnp.concatenate([
            jnp.zeros_like(joint[:, :1]),
            (joint[:, 1:] - joint[:, :-1]) * pair_valid,
        ], axis=1)
        parent_joint = jnp.take(joint, PARENTS, axis=3)
        bone = joint - parent_joint
        parent_motion = jnp.take(joint_motion, PARENTS, axis=3)
        bone_motion = joint_motion - parent_motion

        semantic = nn.Dense(self.local_stream_dim, name="semantic_to_local")(
            bridge["stream_tokens"])
        stream_tokens = []
        for s, mod in enumerate((joint, bone, joint_motion, bone_motion)):
            hs = nn.Dense(self.local_stream_dim,
                          name=f"modality_projection_{s}")(mod)
            hs = hs + semantic[:, None, None, None, s, :]
            stream_tokens.append(hs)
        h4 = jnp.stack(stream_tokens, axis=-2)
        stream_emb = self.param("local_stream_embedding", nn.initializers.normal(0.02),
                                (1, 1, 1, 1, NUM_STREAMS, self.local_stream_dim))
        h4 = jax.nn.gelu(nn.LayerNorm(name="local_stream_norm")(h4 + stream_emb))
        h4 = LocalFourStreamMixer(dim=self.local_stream_dim, heads=4,
                                  name="four_stream_interaction")(h4)
        h4 = nn.Dropout(rate=self.dropout, name="stream_dropout")(
            h4, deterministic=not training)
        fused_joint = nn.Dense(self.memory_dim, name="four_stream_fusion")(
            h4.reshape(b, t, PERSONS, JOINTS,
                       NUM_STREAMS * self.local_stream_dim))
        fused_joint = jax.nn.gelu(nn.LayerNorm(name="fused_joint_norm")(fused_joint))
        fused_joint = fused_joint * pj

        # Coarse learned anatomical pooling: 10 parts x T16 x 2 persons = 320 tokens.
        part_q = self.param("part_queries", nn.initializers.normal(0.02),
                            (10, self.memory_dim))
        logits = jnp.einsum("btmjd,pd->btmpj", fused_joint, part_q) / math.sqrt(self.memory_dim)
        anatomical = PART_MASK.astype(bool)[None, None, None]
        valid_joint = jnp.broadcast_to(present[..., None], (b, t, PERSONS, JOINTS))
        mask = anatomical & valid_joint[..., None, :]
        logits = jnp.where(mask, logits, -1e9)
        pw = jax.nn.softmax(logits, axis=-1)
        has = jnp.any(mask, axis=-1)
        pw = jnp.where(has[..., None], pw, 0.0)
        coarse = jnp.einsum("btmpj,btmjd->btmpd", pw, fused_joint)
        coarse_valid = has
        part_emb = self.param("part_embedding", nn.initializers.normal(0.02),
                              (1, 1, 1, 10, self.memory_dim))
        time_emb_c = self.param("coarse_time_embedding", nn.initializers.normal(0.02),
                                (1, FRAMES, 1, 1, self.memory_dim))
        person_emb_c = self.param("coarse_person_embedding", nn.initializers.normal(0.02),
                                  (1, 1, PERSONS, 1, self.memory_dim))
        coarse = nn.LayerNorm(name="coarse_norm")(
            coarse + part_emb + time_emb_c + person_emb_c)
        coarse = TemporalDepthwiseContext(dim=self.memory_dim, dropout=self.dropout,
                                          name="coarse_temporal")(
            coarse, coarse_valid, training=training)

        # Fine selected arm/hand memory: another 320 tokens, kept at D24 until readout.
        fine = jnp.take(fused_joint, FINE_JOINTS, axis=3)
        fine = nn.Dense(self.fine_dim, name="fine_projection")(fine)
        fine_emb = self.param("fine_joint_embedding", nn.initializers.normal(0.02),
                              (1, 1, 1, FINE_COUNT, self.fine_dim))
        time_emb_f = self.param("fine_time_embedding", nn.initializers.normal(0.02),
                                (1, FRAMES, 1, 1, self.fine_dim))
        person_emb_f = self.param("fine_person_embedding", nn.initializers.normal(0.02),
                                  (1, 1, PERSONS, 1, self.fine_dim))
        fine_valid = jnp.broadcast_to(present[..., None],
                                      (b, t, PERSONS, FINE_COUNT))
        fine = nn.LayerNorm(name="fine_norm")(
            fine + fine_emb + time_emb_f + person_emb_f)
        fine = TemporalDepthwiseContext(dim=self.fine_dim, dropout=self.dropout,
                                        name="fine_temporal")(
            fine, fine_valid, training=training)
        fine = nn.Dense(self.memory_dim, name="fine_to_memory")(fine)

        coarse_tokens = coarse.reshape(b, FRAMES * PERSONS * 10, self.memory_dim)
        fine_tokens = fine.reshape(b, FRAMES * PERSONS * FINE_COUNT, self.memory_dim)
        coarse_valid_flat = coarse_valid.reshape(b, FRAMES * PERSONS * 10)
        fine_valid_flat = fine_valid.reshape(b, FRAMES * PERSONS * FINE_COUNT)
        type_emb = self.param("memory_type_embedding", nn.initializers.normal(0.02),
                              (2, self.memory_dim))
        memory = jnp.concatenate([coarse_tokens + type_emb[0],
                                  fine_tokens + type_emb[1]], axis=1)
        memory_valid = jnp.concatenate([coarse_valid_flat, fine_valid_flat], axis=1)
        memory = nn.LayerNorm(name="memory_norm")(memory)
        nmem = memory.shape[1]

        # Eight sample-conditioned evidence queries read 640 memory tokens.
        q_seed = self.param("readout_queries", nn.initializers.normal(0.02),
                            (self.readout_tokens, self.memory_dim))
        q_delta = nn.Dense(self.readout_tokens * self.memory_dim,
                           name="query_condition")(
            bridge["global_context"]).reshape(b, self.readout_tokens, self.memory_dim)
        query = q_seed[None] + 0.25 * q_delta
        qn = nn.LayerNorm(name="query_norm")(query)
        mn = nn.LayerNorm(name="memory_key_norm")(memory)
        q = nn.Dense(self.memory_dim, use_bias=False, name="readout_q")(qn)
        k = nn.Dense(self.memory_dim, use_bias=False, name="readout_k")(mn)
        v = nn.Dense(self.memory_dim, use_bias=False, name="readout_v")(mn)
        dh = self.memory_dim // self.readout_heads
        q = q.reshape(b, self.readout_tokens, self.readout_heads, dh).transpose(0, 2, 1, 3)
        k = k.reshape(b, nmem, self.readout_heads, dh).transpose(0, 2, 1, 3)
        v = v.reshape(b, nmem, self.readout_heads, dh).transpose(0, 2, 1, 3)
        score = jnp.einsum("bhqd,bhnd->bhqn", q, k) / math.sqrt(dh)

        q_time = self.param("query_time_bias", nn.initializers.zeros,
                            (self.readout_tokens, FRAMES))
        q_person = self.param("query_person_bias", nn.initializers.zeros,
                              (self.readout_tokens, PERSONS))
        q_type = self.param("query_type_bias", nn.initializers.zeros,
                            (self.readout_tokens, 2))
        q_coarse = self.param("query_part_bias", nn.initializers.zeros,
                              (self.readout_tokens, 10))
        q_fine = self.param("query_fine_joint_bias", nn.initializers.zeros,
                            (self.readout_tokens, FINE_COUNT))
        ct = jnp.repeat(jnp.arange(FRAMES), PERSONS * 10)
        cp = jnp.tile(jnp.repeat(jnp.arange(PERSONS), 10), FRAMES)
        ci = jnp.tile(jnp.arange(10), FRAMES * PERSONS)
        ft = jnp.repeat(jnp.arange(FRAMES), PERSONS * FINE_COUNT)
        fp = jnp.tile(jnp.repeat(jnp.arange(PERSONS), FINE_COUNT), FRAMES)
        fi = jnp.tile(jnp.arange(FINE_COUNT), FRAMES * PERSONS)
        coarse_bias = q_time[:, ct] + q_person[:, cp] + q_type[:, 0:1] + q_coarse[:, ci]
        fine_bias = q_time[:, ft] + q_person[:, fp] + q_type[:, 1:2] + q_fine[:, fi]
        identity_bias = jnp.concatenate([coarse_bias, fine_bias], axis=-1)
        score = score + identity_bias[None, None]
        score = jnp.where(memory_valid[:, None, None], score, -1e9)
        attn = jax.nn.softmax(score, axis=-1)
        any_valid = jnp.any(memory_valid, axis=-1)[:, None, None, None]
        attn = jnp.where(any_valid, attn, 0.0)
        ctx = jnp.einsum("bhqn,bhnd->bhqd", attn, v)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(b, self.readout_tokens, self.memory_dim)
        ctx = nn.Dense(self.memory_dim, name="readout_out")(ctx)
        readout = nn.LayerNorm(name="readout_residual_norm")(query + ctx)
        readout = TinySelfMixer(dim=self.memory_dim, heads=self.readout_heads,
                                hidden_dim=2 * self.memory_dim, dropout=self.dropout,
                                name="readout_mixer")(readout, training=training)

        flat = readout.reshape(b, self.readout_tokens * self.memory_dim)
        feature = jnp.concatenate([flat, bridge["global_context"]], axis=-1)
        feature = nn.Dense(self.dense_dim, name="memory_hidden")(feature)
        feature = jax.nn.gelu(feature)
        feature = nn.Dropout(rate=self.dropout, name="memory_dropout")(
            feature, deterministic=not training)
        memory_logits = nn.Dense(NUM_CLASSES, kernel_init=nn.initializers.normal(1e-3),
                                 bias_init=nn.initializers.zeros,
                                 name="memory_classifier")(feature)

        # Low-rank class-aware residual gate. Initial full-scale gate is ~0.10.
        gate_in = jnp.concatenate([feature, bridge["global_context"]], axis=-1)
        gate_rank = jax.nn.tanh(nn.Dense(8, name="class_gate_rank")(gate_in))
        gate_logits = nn.Dense(NUM_CLASSES, kernel_init=nn.initializers.normal(1e-3),
                               bias_init=nn.initializers.zeros,
                               name="class_gate_out")(gate_rank)
        class_gate = MAX_CLASS_GATE * jax.nn.sigmoid(gate_logits)

        attn_mean = jnp.mean(attn, axis=1)
        div_loss, overlap_mean, overlap_max = _query_diversity(attn_mean, memory_valid)
        return {
            "memory_logits": memory_logits,
            "memory_feature": feature,
            "class_gate": class_gate,
            "class_gate_mean": jnp.mean(class_gate),
            "class_gate_max": jnp.max(class_gate),
            "class_gate_min": jnp.min(class_gate),
            "query_diversity_loss": div_loss,
            "query_overlap_mean": overlap_mean,
            "query_overlap_max": overlap_max,
            "readout_attention": attn_mean,
        }


def make_wrapper_v35(base_model: nn.Module, *, bridge_dim: int = 32,
                     local_stream_dim: int = 16, memory_dim: int = 32,
                     fine_dim: int = 24, readout_tokens: int = 8,
                     readout_heads: int = 4, dense_dim: int = 128,
                     dropout: float = 0.08,
                     stream_reweight_strength: float = DEFAULT_STREAM_REWEIGHT):
    class AttentionLiteCrossStreamMemoryV35(nn.Module):
        base: nn.Module

        @nn.compact
        def __call__(self, x, training: bool = False, branch_scale: float = 1.0):
            base_out = self.base(x, training=training)
            if "stream_logits" not in base_out or "fusion_weights" not in base_out:
                raise RuntimeError("Canonical Attention-Lite output lacks stream_logits/fusion_weights")
            bridge = SemanticStreamBridge(dim=bridge_dim, heads=4, dropout=dropout,
                                          name="cross_stream_bridge")(
                base_out["stream_logits"], training=training)
            memory = CrossStreamMultiResolutionMemory(
                bridge_dim=bridge_dim, local_stream_dim=local_stream_dim,
                memory_dim=memory_dim, fine_dim=fine_dim,
                readout_tokens=readout_tokens, readout_heads=readout_heads,
                dense_dim=dense_dim, dropout=dropout,
                name="cross_stream_memory")(x, bridge, training=training)

            base_w = jnp.asarray(base_out["fusion_weights"])
            if base_w.ndim == 1:
                base_logw = jnp.broadcast_to(
                    jnp.log(jnp.clip(base_w, 1e-8, 1.0))[None],
                    (x.shape[0], NUM_STREAMS))
            else:
                base_logw = jnp.log(jnp.clip(base_w, 1e-8, 1.0))
            scale = jnp.asarray(branch_scale, jnp.float32)
            dynamic_w = jax.nn.softmax(
                base_logw + stream_reweight_strength * scale *
                jnp.tanh(bridge["fusion_delta"]), axis=-1)
            controlled_logits = jnp.einsum("bs,bsc->bc", dynamic_w,
                                           base_out["stream_logits"])
            effective_class_gate = scale * memory["class_gate"]
            logits = controlled_logits + effective_class_gate * memory["memory_logits"]

            result = dict(base_out)
            result.update({
                "base_logits": base_out["logits"],
                "controlled_logits": controlled_logits,
                "memory_logits": memory["memory_logits"],
                "memory_feature": memory["memory_feature"],
                "dynamic_fusion_weights": dynamic_w,
                "stream_pool_weights": bridge["stream_pool_weights"],
                "class_gate": memory["class_gate"],
                "effective_class_gate": effective_class_gate,
                "class_gate_mean": memory["class_gate_mean"],
                "class_gate_max": memory["class_gate_max"],
                "class_gate_min": memory["class_gate_min"],
                "query_diversity_loss": memory["query_diversity_loss"],
                "query_overlap_mean": memory["query_overlap_mean"],
                "query_overlap_max": memory["query_overlap_max"],
                "readout_attention": memory["readout_attention"],
                "logits": logits,
            })
            return result

    return AttentionLiteCrossStreamMemoryV35(base=base_model)

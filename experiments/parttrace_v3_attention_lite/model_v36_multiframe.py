#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NestSAR v3.6 Multi-Frame Cross-Stream Memory.

Accuracy-first follow-up to v3.5.

The canonical Attention-Lite backbone receives the full T16/T32/T64 sequence.
The side memory does NOT grow linearly with T: a motion-aware temporal reducer
compresses the full sequence to 16 representative temporal anchors before the
v3.5 coarse/fine memory. This keeps the memory readout at 640 tokens while still
letting the canonical four-stream backbone exploit the longer sequence.

v3.6 also fixes the v3.5 correction-gate saturation observed in scratch runs.
The memory correction is capped at half of the original v3.5 class-gate range
and is modulated by stop-gradient base uncertainty. Confident base predictions
therefore receive smaller corrections; uncertain examples can use more memory.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from experiments.parttrace_v3_attention_lite.audit_model import NUM_CLASSES
from experiments.parttrace_v3_attention_lite.model_v35_crossstream_memory import (
    NUM_STREAMS,
    SemanticStreamBridge,
    CrossStreamMultiResolutionMemory,
)

DEFAULT_MEMORY_FRAMES = 16
DEFAULT_GATE_SCALE = 0.50
DEFAULT_UNCERTAINTY_FLOOR = 0.35


def motion_aware_temporal_reduce(x: jnp.ndarray, target_frames: int = 16):
    """Reduce [B,T,F] to [B,target_frames,F] using local motion salience.

    T must be an integer multiple of target_frames.  Each temporal segment keeps
    a differentiable weighted representative.  Empty/padded frames are masked.
    At T==target_frames this is exactly the identity.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected [B,T,F], got {x.shape}")
    b, t, f = x.shape
    target_frames = int(target_frames)
    if target_frames <= 0 or t < target_frames or t % target_frames:
        raise ValueError(
            f"Temporal reduction requires T divisible by target_frames; "
            f"got T={t}, target={target_frames}"
        )
    if t == target_frames:
        weights = jnp.ones((b, target_frames, 1), x.dtype)
        return x, weights

    stride = t // target_frames
    seg = x.reshape(b, target_frames, stride, f)

    prev = jnp.concatenate([x[:, :1], x[:, :-1]], axis=1)
    motion = jnp.mean(jnp.abs(x - prev), axis=-1)
    motion = motion.reshape(b, target_frames, stride)
    valid = jnp.any(jnp.abs(seg) > 1e-7, axis=-1)

    # Normalize salience inside each segment.  This avoids scale differences
    # between clips and makes the reducer prefer informative local motion rather
    # than simply selecting the largest-coordinate frame.
    mean = jnp.sum(motion * valid, axis=2, keepdims=True) / jnp.maximum(
        jnp.sum(valid, axis=2, keepdims=True), 1
    )
    var = jnp.sum(jnp.square(motion - mean) * valid, axis=2, keepdims=True) / jnp.maximum(
        jnp.sum(valid, axis=2, keepdims=True), 1
    )
    score = 1.5 * (motion - mean) / jnp.sqrt(var + 1e-6)
    score = jnp.where(valid, score, -1e9)
    weights = jax.nn.softmax(score, axis=2)
    any_valid = jnp.any(valid, axis=2, keepdims=True)
    weights = jnp.where(any_valid, weights, 0.0)
    reduced = jnp.sum(weights[..., None] * seg, axis=2)
    return reduced, weights


def _normalized_entropy(logits: jnp.ndarray) -> jnp.ndarray:
    p = jax.nn.softmax(logits, axis=-1)
    entropy = -jnp.sum(p * jnp.log(jnp.clip(p, 1e-8, 1.0)), axis=-1)
    return entropy / math.log(NUM_CLASSES)


def make_wrapper_v36(
    base_model: nn.Module,
    *,
    input_frames: int = 64,
    memory_frames: int = DEFAULT_MEMORY_FRAMES,
    bridge_dim: int = 32,
    local_stream_dim: int = 16,
    memory_dim: int = 32,
    fine_dim: int = 24,
    readout_tokens: int = 8,
    readout_heads: int = 4,
    dense_dim: int = 128,
    dropout: float = 0.10,
    stream_reweight_strength: float = 0.08,
    gate_scale: float = DEFAULT_GATE_SCALE,
    uncertainty_floor: float = DEFAULT_UNCERTAINTY_FLOOR,
):
    input_frames = int(input_frames)
    memory_frames = int(memory_frames)
    if input_frames not in (16, 32, 64):
        raise ValueError(f"v3.6 supports T16/T32/T64; got T={input_frames}")
    if input_frames % memory_frames:
        raise ValueError(
            f"input_frames={input_frames} must be divisible by memory_frames={memory_frames}"
        )
    if memory_frames != 16:
        # The reused v3.5 memory module has a committed 16-position temporal
        # parameterization.  Keep this explicit rather than silently changing it.
        raise ValueError("v3.6 currently fixes side-memory temporal anchors at 16")
    if not 0.0 < gate_scale <= 1.0:
        raise ValueError("gate_scale must be in (0,1]")
    if not 0.0 <= uncertainty_floor <= 1.0:
        raise ValueError("uncertainty_floor must be in [0,1]")

    class AttentionLiteCrossStreamMemoryV36(nn.Module):
        base: nn.Module

        @nn.compact
        def __call__(self, x, training: bool = False, branch_scale: float = 1.0):
            if x.ndim != 3 or x.shape[1] != input_frames:
                raise ValueError(
                    f"v3.6 expected [B,{input_frames},150], got {x.shape}"
                )

            # Full-resolution canonical path: all requested frames reach the
            # original four Attention-Lite specialists.
            base_out = self.base(x, training=training)
            if "stream_logits" not in base_out or "fusion_weights" not in base_out:
                raise RuntimeError(
                    "Canonical Attention-Lite output lacks stream_logits/fusion_weights"
                )

            bridge = SemanticStreamBridge(
                dim=bridge_dim,
                heads=4,
                dropout=dropout,
                name="cross_stream_bridge",
            )(base_out["stream_logits"], training=training)

            # Side memory stays fixed at 16 temporal anchors.  T64 therefore
            # contributes information from all 64 input frames without growing
            # the K-query readout from 640 to 2,560 tokens.
            memory_x, reduction_weights = motion_aware_temporal_reduce(
                x, target_frames=memory_frames
            )
            memory = CrossStreamMultiResolutionMemory(
                bridge_dim=bridge_dim,
                local_stream_dim=local_stream_dim,
                memory_dim=memory_dim,
                fine_dim=fine_dim,
                readout_tokens=readout_tokens,
                readout_heads=readout_heads,
                dense_dim=dense_dim,
                dropout=dropout,
                name="cross_stream_memory",
            )(memory_x, bridge, training=training)

            base_w = jnp.asarray(base_out["fusion_weights"])
            if base_w.ndim == 1:
                base_logw = jnp.broadcast_to(
                    jnp.log(jnp.clip(base_w, 1e-8, 1.0))[None],
                    (x.shape[0], NUM_STREAMS),
                )
            else:
                base_logw = jnp.log(jnp.clip(base_w, 1e-8, 1.0))

            scale = jnp.asarray(branch_scale, jnp.float32)
            dynamic_w = jax.nn.softmax(
                base_logw
                + stream_reweight_strength
                * scale
                * jnp.tanh(bridge["fusion_delta"]),
                axis=-1,
            )
            controlled_logits = jnp.einsum(
                "bs,bsc->bc", dynamic_w, base_out["stream_logits"]
            )

            # v3.5 scratch runs drove the class gate almost to its 0.20 ceiling.
            # v3.6 turns it into an actual correction gate: maximum correction is
            # halved, and confident base predictions receive less branch weight.
            uncertainty = jax.lax.stop_gradient(
                _normalized_entropy(controlled_logits)
            )
            trust = uncertainty_floor + (1.0 - uncertainty_floor) * uncertainty
            effective_class_gate = (
                scale
                * gate_scale
                * memory["class_gate"]
                * trust[:, None]
            )

            # Centering is class-softmax invariant but prevents an unnecessary
            # common-mode shift from the residual classifier.
            correction_logits = memory["memory_logits"] - jnp.mean(
                memory["memory_logits"], axis=-1, keepdims=True
            )
            logits = controlled_logits + effective_class_gate * correction_logits

            reduction_peak = jnp.mean(jnp.max(reduction_weights, axis=-1))
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
                "base_uncertainty": jnp.mean(uncertainty),
                "temporal_reduction_peak": reduction_peak,
                "query_diversity_loss": memory["query_diversity_loss"],
                "query_overlap_mean": memory["query_overlap_mean"],
                "query_overlap_max": memory["query_overlap_max"],
                "readout_attention": memory["readout_attention"],
                "logits": logits,
            })
            return result

    return AttentionLiteCrossStreamMemoryV36(base=base_model)

#!/usr/bin/env python3
from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn

from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju

FRAMES = ju.FRAMES
PERSONS = ju.PERSONS
JOINTS = ju.JOINTS
TOKEN_CHANNELS = ju.TOKEN_CHANNELS
FEATURES = ju.FEATURES
NUM_CLASSES = ju.NUM_CLASSES
NUM_STREAMS = ju.NUM_STREAMS

# Baseline champion params = 1,816,130.
# Old router params = 13,106.
# Rich router params = 76,049.
# New total = 1,879,073.
EXPECTED_PARAMS = 1_879_073


class RichCrossStreamRouter(nn.Module):
    """Post-frame cross-stream router with nonlinear context and vector gating.

    Input:  [B,T,S,D]
    Output: [B,T,S,D]

    Compared with the baseline router:
      - keeps learned scalar stream attention weights;
      - replaces one linear context projection with D -> 2D -> D MLP;
      - replaces scalar gate with a D-dimensional gate conditioned on both
        the local stream token and the global routed context;
      - keeps the same conservative residual scale (0.15).
    """

    dim: int = 112
    expansion: int = 2
    residual_scale: float = 0.15

    @nn.compact
    def __call__(self, streams: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if streams.ndim != 4:
            raise ValueError(f"Expected [B,T,S,D], got {streams.shape}")

        n = nn.LayerNorm(name="router_norm")(streams)

        # Learned stream importance at every time step.
        scores = nn.Dense(1, name="score")(n)[..., 0]
        weights = jax.nn.softmax(scores, axis=2)

        # Global cross-stream context.
        context = jnp.sum(weights[..., None] * n, axis=2)  # [B,T,D]
        hidden = nn.Dense(self.dim * self.expansion, name="context_up")(context)
        hidden = nn.gelu(hidden)
        delta = nn.Dense(self.dim, name="context_down")(hidden)

        # Per-stream, per-channel gate conditioned on local + routed context.
        context_s = jnp.broadcast_to(
            context[:, :, None, :],
            (streams.shape[0], streams.shape[1], streams.shape[2], self.dim),
        )
        gate_in = jnp.concatenate([n, context_s], axis=-1)
        gate = jax.nn.sigmoid(nn.Dense(self.dim, name="vector_gate")(gate_in))

        out = streams + self.residual_scale * gate * delta[:, :, None, :]
        return out, weights


class M4PhaseRichRouterT16(nn.Module):
    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Mapping[str, jnp.ndarray]:
        tok = x.reshape(x.shape[0], FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS)
        pose = tok[..., 0:3]
        full_disp = tok[..., 3:6]
        phase_a = tok[..., 6:9]
        phase_b = tok[..., 9:12]
        path = tok[..., 12:15]

        joint = pose
        parents = jnp.asarray(ju.base.PARENTS)
        bone = joint - jnp.take(joint, parents, axis=3)

        joint_motion = jnp.concatenate([full_disp, phase_a, phase_b, path], axis=-1)
        parent_full = jnp.take(full_disp, parents, axis=3)
        parent_a = jnp.take(phase_a, parents, axis=3)
        parent_b = jnp.take(phase_b, parents, axis=3)
        parent_path = jnp.take(path, parents, axis=3)
        bone_motion = jnp.concatenate([
            full_disp - parent_full,
            phase_a - parent_a,
            phase_b - parent_b,
            jnp.abs(path - parent_path),
        ], axis=-1)

        raw_streams = (joint, bone, joint_motion, bone_motion)
        spatial = []
        for i, s in enumerate(raw_streams):
            spatial.append(ju.base.SpatialEncoder(
                self.spatial_dim,
                self.model_dim,
                self.dropout,
                name=f"spatial_{i}",
            )(s, training))

        frame_streams = []
        for i, s in enumerate(spatial):
            frame_streams.append(ju.base.BiMemory(
                self.model_dim,
                name=f"frame_memory_{i}",
            )(s))

        frame_stack = jnp.stack(frame_streams, axis=2)

        mixed, router_weights = RichCrossStreamRouter(
            self.model_dim,
            expansion=2,
            residual_scale=0.15,
            name="cross_stream_after_frame",
        )(frame_stack)

        descriptors = []
        stream_logits = []
        chunk_states = []

        for i in range(NUM_STREAMS):
            chunks, desc = ju.base.DescriptorHead(
                self.model_dim,
                self.dropout,
                name=f"descriptor_{i}",
            )(mixed[:, :, i], training)
            descriptors.append(desc)
            chunk_states.append(chunks)
            stream_logits.append(
                nn.Dense(NUM_CLASSES, name=f"classifier_{i}")(desc)
            )

        descs = jnp.stack(descriptors, axis=1)
        sl = jnp.stack(stream_logits, axis=1)

        # Keep the champion's fixed uniform final fusion unchanged.
        fusion = jnp.full(
            (x.shape[0], NUM_STREAMS),
            1.0 / NUM_STREAMS,
            dtype=sl.dtype,
        )
        logits = jnp.mean(sl, axis=1)

        return {
            "logits": logits,
            "stream_logits": sl,
            "fusion_weights": fusion,
            "router_weights": router_weights,
            "spatial_stack": jnp.stack(spatial, axis=2),
            "frame_stack": frame_stack,
            "mixed_frame_stack": mixed,
            "descriptors": descs,
            "chunk_states": jnp.stack(chunk_states, axis=1),
        }

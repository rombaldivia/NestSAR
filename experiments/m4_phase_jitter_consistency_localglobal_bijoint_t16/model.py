#!/usr/bin/env python3
from __future__ import annotations

"""LocalGlobal V2 with bidirectional joint-memory spatial encoding.

The verified LocalGlobal V2 architecture uses one forward GatedSweep over the
25 joints (after kinematic reordering) before 10-part pooling. This ablation
replaces only that one-way joint sweep with the existing NestSAR BiMemory:

    ordered joints -> forward GatedSweep
                   -> backward GatedSweep
                   -> linear merge + residual LayerNorm
                   -> restore NTU order -> existing 10-part pooling

Everything after spatial encoding is unchanged: frame BiMemory, post-frame
CrossStreamRouter, descriptor hierarchy, four classifier heads, and fixed
uniform final fusion. No attention / no QKV.
"""

from typing import Mapping

import jax.numpy as jnp
from flax import linen as nn

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju

FRAMES = ju.FRAMES
PERSONS = ju.PERSONS
JOINTS = ju.JOINTS
TOKEN_CHANNELS = ju.TOKEN_CHANNELS
NUM_CLASSES = ju.NUM_CLASSES
NUM_STREAMS = ju.NUM_STREAMS

BASELINE_PARAMS = 1_816_130
BIJOINT_EXTRA_PARAMS = 18_816
EXPECTED_PARAMS = BASELINE_PARAMS + BIJOINT_EXTRA_PARAMS  # 1,834,946


class SpatialEncoderBiJoint(nn.Module):
    """Exact SpatialEncoder except joint memory is bidirectional."""

    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        b, t, m, _, _ = x.shape

        h = nn.Dense(self.spatial_dim, name="in_proj")(x)

        je = self.param(
            "joint_embed",
            nn.initializers.normal(0.02),
            (1, 1, 1, JOINTS, self.spatial_dim),
        )
        pe = self.param(
            "person_embed",
            nn.initializers.normal(0.02),
            (1, 1, PERSONS, 1, self.spatial_dim),
        )
        h = nn.gelu(h + je + pe)

        # Preserve the exact champion kinematic joint ordering.
        order = jnp.asarray(base.JOINT_ORDER)
        inv = jnp.argsort(order)
        h = jnp.take(h, order, axis=3)
        h = h.reshape(b * t * m, JOINTS, self.spatial_dim)

        # ONLY architecture change in this experiment.
        # Same BiMemory primitive already used by NestSAR's temporal hierarchy.
        h = base.BiMemory(
            self.spatial_dim,
            name="joint_bimemory",
        )(h)

        h = h.reshape(b, t, m, JOINTS, self.spatial_dim)
        h = jnp.take(h, inv, axis=3)

        # Exact existing 10-part pooling/fusion.
        mask = jnp.asarray(base.PART_MASK_NP, h.dtype)
        counts = jnp.asarray(base.PART_COUNTS_NP, h.dtype)
        parts = jnp.einsum("btmvd,pv->btmpd", h, mask)
        parts = parts / counts[None, None, None, :, None]

        flat = parts.reshape(
            b,
            t,
            m * 10 * self.spatial_dim,
        )
        y = nn.Dense(self.model_dim, name="part_fuse")(flat)
        y = nn.LayerNorm(name="out_norm")(nn.gelu(y))
        return nn.Dropout(self.dropout)(
            y,
            deterministic=not training,
        )


class M4PhaseUniformBiJointT16(nn.Module):
    """Exact Phase/LocalGlobal main network with BiJoint spatial encoders."""

    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> Mapping[str, jnp.ndarray]:

        tok = x.reshape(
            x.shape[0],
            FRAMES,
            PERSONS,
            JOINTS,
            TOKEN_CHANNELS,
        )

        pose = tok[..., 0:3]
        full_disp = tok[..., 3:6]
        phase_a = tok[..., 6:9]
        phase_b = tok[..., 9:12]
        path = tok[..., 12:15]

        joint = pose
        parents = jnp.asarray(base.PARENTS)
        bone = joint - jnp.take(joint, parents, axis=3)

        joint_motion = jnp.concatenate(
            [full_disp, phase_a, phase_b, path],
            axis=-1,
        )

        parent_full = jnp.take(full_disp, parents, axis=3)
        parent_a = jnp.take(phase_a, parents, axis=3)
        parent_b = jnp.take(phase_b, parents, axis=3)
        parent_path = jnp.take(path, parents, axis=3)

        bone_motion = jnp.concatenate(
            [
                full_disp - parent_full,
                phase_a - parent_a,
                phase_b - parent_b,
                jnp.abs(path - parent_path),
            ],
            axis=-1,
        )

        raw_streams = (
            joint,
            bone,
            joint_motion,
            bone_motion,
        )

        spatial = []
        for i, stream in enumerate(raw_streams):
            spatial.append(
                SpatialEncoderBiJoint(
                    self.spatial_dim,
                    self.model_dim,
                    self.dropout,
                    name=f"spatial_{i}",
                )(stream, training)
            )

        # Exact champion temporal hierarchy.
        frame_streams = []
        for i, stream in enumerate(spatial):
            frame_streams.append(
                base.BiMemory(
                    self.model_dim,
                    name=f"frame_memory_{i}",
                )(stream)
            )

        frame_stack = jnp.stack(frame_streams, axis=2)

        mixed, router_weights = base.CrossStreamRouter(
            self.model_dim,
            name="cross_stream_after_frame",
        )(frame_stack)

        descriptors = []
        stream_logits = []
        chunk_states = []

        for i in range(NUM_STREAMS):
            chunks, desc = base.DescriptorHead(
                self.model_dim,
                self.dropout,
                name=f"descriptor_{i}",
            )(mixed[:, :, i], training)

            descriptors.append(desc)
            chunk_states.append(chunks)
            stream_logits.append(
                nn.Dense(
                    NUM_CLASSES,
                    name=f"classifier_{i}",
                )(desc)
            )

        descs = jnp.stack(descriptors, axis=1)
        sl = jnp.stack(stream_logits, axis=1)

        # Exact champion fixed uniform fusion.
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

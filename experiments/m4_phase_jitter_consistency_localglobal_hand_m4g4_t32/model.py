#!/usr/bin/env python3
from __future__ import annotations

"""LocalGlobal V2 main model + tiny high-rate Hand-M4/G4-Lite T32 branch.

No attention is used.  The auxiliary branch reuses the exact GatedSweep/BiMemory
operator already used by the NestSAR core, but at dim=32 and only on hand-region
features.  It has two memory timescales:
  T32 frame memory -> 8 chunks of 4 -> chunk memory.

The proven LocalGlobal V2 four-stream path is copied exactly.  Its uniform
four-stream mean remains the main prediction.  The hand specialist contributes
a conservative fixed residual:
    logits = main_logits + hand_residual_scale * hand_logits
Default hand_residual_scale = 0.10.
"""

from typing import Mapping

import jax.numpy as jnp
from flax import linen as nn

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju

from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.preprocessing import (
    HAND_FEATURES,
    HAND_FRAMES,
)

FRAMES = ju.FRAMES
PERSONS = ju.PERSONS
JOINTS = ju.JOINTS
TOKEN_CHANNELS = ju.TOKEN_CHANNELS
NUM_CLASSES = ju.NUM_CLASSES
NUM_STREAMS = ju.NUM_STREAMS

BASELINE_PARAMS = 1_816_130
HAND_DIM_DEFAULT = 32
HAND_EXTRA_PARAMS_D32 = 38_520
EXPECTED_PARAMS_D32 = BASELINE_PARAMS + HAND_EXTRA_PARAMS_D32  # 1,854,650


class HandM4G4LiteT32(nn.Module):
    """Tiny hierarchical hand memory using the same NestSAR BiMemory core."""
    dim: int = HAND_DIM_DEFAULT
    dropout: float = 0.10

    @nn.compact
    def __call__(
        self,
        hand_x: jnp.ndarray,
        training: bool = False,
    ) -> Mapping[str, jnp.ndarray]:
        if hand_x.shape[1] != HAND_FRAMES or hand_x.shape[2] != HAND_FEATURES:
            raise ValueError(
                f"Expected hand input [B,{HAND_FRAMES},{HAND_FEATURES}], "
                f"got {hand_x.shape}"
            )

        h = nn.Dense(self.dim, name="in_proj")(hand_x)
        h = nn.LayerNorm(name="in_norm")(nn.gelu(h))

        # Same recurrent/gated memory primitive as the main NestSAR core.
        frame_h = base.BiMemory(self.dim, name="frame_memory")(h)

        # T32 -> 8 chunks x 4 frames: a second, slower memory level.
        chunks = frame_h.reshape(
            frame_h.shape[0],
            HAND_FRAMES // 4,
            4,
            self.dim,
        ).mean(axis=2)
        chunks = base.BiMemory(self.dim, name="chunk_memory")(chunks)

        pooled = jnp.concatenate(
            [
                frame_h.mean(axis=1),
                chunks.mean(axis=1),
            ],
            axis=-1,
        )
        desc = nn.Dense(self.dim, name="hier_fuse")(pooled)
        desc = nn.LayerNorm(name="hier_norm")(nn.gelu(desc))
        desc = nn.Dropout(self.dropout)(
            desc,
            deterministic=not training,
        )
        logits = nn.Dense(NUM_CLASSES, name="classifier")(desc)

        return {
            "descriptor": desc,
            "logits": logits,
            "frame_states": frame_h,
            "chunk_states": chunks,
        }


class M4LocalGlobalHandM4G4T32(nn.Module):
    """Exact LocalGlobal V2 main path plus the T32 hand specialist."""
    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10
    hand_dim: int = HAND_DIM_DEFAULT
    hand_residual_scale: float = 0.10

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        hand_x: jnp.ndarray,
        training: bool = False,
    ) -> Mapping[str, jnp.ndarray]:
        # ------------------------------------------------------------------------------------------
        # EXACT LocalGlobal V2 / Phase-T16 four-stream network.
        # ------------------------------------------------------------------------------------------
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
                base.SpatialEncoder(
                    self.spatial_dim,
                    self.model_dim,
                    self.dropout,
                    name=f"spatial_{i}",
                )(stream, training)
            )

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

        # Keep the exact champion fixed uniform four-stream fusion.
        fusion = jnp.full(
            (x.shape[0], NUM_STREAMS),
            1.0 / NUM_STREAMS,
            dtype=sl.dtype,
        )
        main_logits = jnp.mean(sl, axis=1)

        # ------------------------------------------------------------------------------------------
        # NEW: high-rate hand-only M4/G4-Lite specialist.  No attention.
        # ------------------------------------------------------------------------------------------
        hand = HandM4G4LiteT32(
            dim=self.hand_dim,
            dropout=self.dropout,
            name="hand_m4g4_t32",
        )(hand_x, training)

        logits = (
            main_logits
            + self.hand_residual_scale
            * hand["logits"]
        )

        return {
            "logits": logits,
            "main_logits": main_logits,
            "hand_logits": hand["logits"],
            "hand_descriptor": hand["descriptor"],
            "hand_frame_states": hand["frame_states"],
            "hand_chunk_states": hand["chunk_states"],
            "stream_logits": sl,
            "fusion_weights": fusion,
            "router_weights": router_weights,
            "spatial_stack": jnp.stack(spatial, axis=2),
            "frame_stack": frame_stack,
            "mixed_frame_stack": mixed,
            "descriptors": descs,
            "chunk_states": jnp.stack(chunk_states, axis=1),
        }

"""Exact Hand-M4/G4-T32 equations vendored from NestSAR commit 1a570d5.
Only imports were detached; AUDIT_UNROLL expands scan for counting only.
Original source: experiments/m4_phase_jitter_consistency_localglobal_hand_m4g4_t32/model.py
"""
from typing import Mapping
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from .preprocessing import FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS, FEATURES, HAND_FRAMES, HAND_FEATURES
NUM_CLASSES = 120
NUM_STREAMS = 4
HAND_DIM_DEFAULT = 32
EXPECTED_PARAMS = 1_854_650
AUDIT_UNROLL = False

PARENTS = np.asarray([
    0, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0,
    12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11,
], dtype=np.int32)

JOINT_ORDER = np.asarray([
    0, 1, 20, 2, 3,
    4, 5, 6, 7, 21, 22,
    8, 9, 10, 11, 23, 24,
    12, 13, 14, 15,
    16, 17, 18, 19,
], dtype=np.int32)

TEN_PARTS = (
    (0, 1, 20),
    (2, 3),
    (4, 5),
    (6, 7, 21, 22),
    (8, 9),
    (10, 11, 23, 24),
    (12, 13),
    (14, 15),
    (16, 17),
    (18, 19),
)

PART_MASK_NP = np.zeros((10, 25), np.float32)
for p, joints in enumerate(TEN_PARTS):
    PART_MASK_NP[p, list(joints)] = 1.0
PART_COUNTS_NP = np.maximum(PART_MASK_NP.sum(axis=1), 1.0)

class GatedSweep(nn.Module):
    dim: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d = self.dim
        init = nn.initializers.xavier_uniform()
        wz_x = self.param("wz_x", init, (d, d))
        wz_h = self.param("wz_h", init, (d, d))
        bz = self.param("bz", nn.initializers.zeros, (d,))
        wr_x = self.param("wr_x", init, (d, d))
        wr_h = self.param("wr_h", init, (d, d))
        br = self.param("br", nn.initializers.zeros, (d,))
        wc_x = self.param("wc_x", init, (d, d))
        wc_h = self.param("wc_h", init, (d, d))
        bc = self.param("bc", nn.initializers.zeros, (d,))
        xt = jnp.swapaxes(x, 0, 1)
        if self.reverse:
            xt = xt[::-1]
        h0 = jnp.zeros((x.shape[0], d), x.dtype)

        def step(h, token):
            z = jax.nn.sigmoid(token @ wz_x + h @ wz_h + bz)
            r = jax.nn.sigmoid(token @ wr_x + h @ wr_h + br)
            cand = jnp.tanh(token @ wc_x + (r * h) @ wc_h + bc)
            h = (1.0 - z) * h + z * cand
            return h, h

        if AUDIT_UNROLL:
            h = h0
            states = []
            for t in range(xt.shape[0]):
                h, y = step(h, xt[t])
                states.append(y)
            yt = jnp.stack(states)
        else:
            _, yt = jax.lax.scan(step, h0, xt)
        if self.reverse:
            yt = yt[::-1]
        return jnp.swapaxes(yt, 0, 1)


class BiMemory(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        f = GatedSweep(self.dim, reverse=False, name="fwd")(x)
        b = GatedSweep(self.dim, reverse=True, name="bwd")(x)
        y = nn.Dense(self.dim, name="merge")(jnp.concatenate([f, b], axis=-1))
        return nn.LayerNorm(name="norm")(x + y)


class SpatialEncoder(nn.Module):
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

        order = jnp.asarray(JOINT_ORDER)
        inv = jnp.argsort(order)
        h = jnp.take(h, order, axis=3)
        h = h.reshape(b * t * m, JOINTS, self.spatial_dim)
        mem = GatedSweep(self.spatial_dim, reverse=False, name="joint_memory")(h)
        h = nn.LayerNorm(name="joint_memory_norm")(h + mem)
        h = h.reshape(b, t, m, JOINTS, self.spatial_dim)
        h = jnp.take(h, inv, axis=3)

        mask = jnp.asarray(PART_MASK_NP, h.dtype)
        counts = jnp.asarray(PART_COUNTS_NP, h.dtype)
        parts = jnp.einsum("btmvd,pv->btmpd", h, mask)
        parts = parts / counts[None, None, None, :, None]
        flat = parts.reshape(b, t, m * 10 * self.spatial_dim)
        y = nn.Dense(self.model_dim, name="part_fuse")(flat)
        y = nn.LayerNorm(name="out_norm")(nn.gelu(y))
        return nn.Dropout(self.dropout)(y, deterministic=not training)


class CrossStreamRouter(nn.Module):
    dim: int = 112
    residual_scale: float = 0.15

    @nn.compact
    def __call__(self, streams: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # streams: [B,T,S,D], deliberately AFTER each stream's frame memory.
        n = nn.LayerNorm(name="router_norm")(streams)
        scores = nn.Dense(1, name="score")(n)[..., 0]
        weights = jax.nn.softmax(scores, axis=2)
        context = jnp.sum(weights[..., None] * n, axis=2)
        delta = nn.Dense(self.dim, name="context_proj")(context)
        gate = jax.nn.sigmoid(nn.Dense(1, name="gate")(n))
        out = streams + self.residual_scale * gate * delta[:, :, None, :]
        return out, weights


class DescriptorHead(nn.Module):
    dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, frame_h: jnp.ndarray, training: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
        chunks = frame_h.reshape(frame_h.shape[0], 4, 4, self.dim).mean(axis=2)
        chunks = BiMemory(self.dim, name="chunk_memory")(chunks)
        pooled = jnp.concatenate(
            [frame_h.mean(axis=1), chunks.mean(axis=1)], axis=-1
        )
        pooled = nn.Dense(self.dim, name="hier_fuse")(pooled)
        pooled = nn.LayerNorm(name="hier_norm")(nn.gelu(pooled))
        pooled = nn.Dropout(self.dropout)(pooled, deterministic=not training)
        return chunks, pooled


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
        frame_h = BiMemory(self.dim, name="frame_memory")(h)

        # T32 -> 8 chunks x 4 frames: a second, slower memory level.
        chunks = frame_h.reshape(
            frame_h.shape[0],
            HAND_FRAMES // 4,
            4,
            self.dim,
        ).mean(axis=2)
        chunks = BiMemory(self.dim, name="chunk_memory")(chunks)

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
        parents = jnp.asarray(PARENTS)
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
                SpatialEncoder(
                    self.spatial_dim,
                    self.model_dim,
                    self.dropout,
                    name=f"spatial_{i}",
                )(stream, training)
            )

        frame_streams = []
        for i, stream in enumerate(spatial):
            frame_streams.append(
                BiMemory(
                    self.model_dim,
                    name=f"frame_memory_{i}",
                )(stream)
            )

        frame_stack = jnp.stack(frame_streams, axis=2)

        mixed, router_weights = CrossStreamRouter(
            self.model_dim,
            name="cross_stream_after_frame",
        )(frame_stack)

        descriptors = []
        stream_logits = []
        chunk_states = []

        for i in range(NUM_STREAMS):
            chunks, desc = DescriptorHead(
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

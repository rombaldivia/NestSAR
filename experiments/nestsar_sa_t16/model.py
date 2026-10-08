"""NestSAR-SA-T16: R4-FMSE + per-frame spatial joint attention.

Titans/HOPE-style division of labour for skeletons:
  * short-term, exact memory over SPACE: softmax attention over the 50 joint
    tokens of the current frame (2 persons x 25 joints), with a learned
    skeleton-hop bias and a separate cross-person bucket, masked so absent
    joints/persons are never attended;
  * long-term memory over TIME: the unchanged R4-FMSE self-modifying M4/G4
    nested memory, router, descriptors and heads.

The only change from R4-FMSE is one pre-norm attention block (layer-scaled
residual) inside every stream's spatial encoder, after the joint sweep and
before part pooling. No temporal attention, no graph convolution, no CNN/TCN.
"""
from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from experiments.nestsar_sm_all_t16 import model as r4


FRAMES = r4.FRAMES
PERSONS = r4.PERSONS
JOINTS = r4.JOINTS
TOKEN_CHANNELS = r4.TOKEN_CHANNELS
FEATURES = r4.FEATURES
NUM_CLASSES = r4.NUM_CLASSES
NUM_STREAMS = r4.NUM_STREAMS


ATTN_HEADS = 2
HOP_BUCKETS = 8          # hop distance 0..7 (>=7 clipped); bucket 8 = joint of the other person


def _hop_matrix() -> np.ndarray:
    parents = np.asarray(r4.base.PARENTS)
    v = len(parents)
    adj = [[] for _ in range(v)]
    for j, p in enumerate(parents):
        if p != j:
            adj[j].append(int(p))
            adj[int(p)].append(j)
    dist = np.full((v, v), 10 ** 6, np.int64)
    for s in range(v):
        dist[s, s] = 0
        frontier = [s]
        while frontier:
            nxt = []
            for a in frontier:
                for b in adj[a]:
                    if dist[s, b] > dist[s, a] + 1:
                        dist[s, b] = dist[s, a] + 1
                        nxt.append(b)
            frontier = nxt
    if (dist >= 10 ** 6).any():
        raise ValueError("Skeleton graph is not connected")
    return dist


def _bias_index() -> np.ndarray:
    hop = np.minimum(_hop_matrix(), HOP_BUCKETS - 1)
    v = hop.shape[0]
    idx = np.full((PERSONS * v, PERSONS * v), HOP_BUCKETS, np.int32)
    for m in range(PERSONS):
        idx[m * v:(m + 1) * v, m * v:(m + 1) * v] = hop
    return idx


BIAS_INDEX = _bias_index()   # [50, 50]


class SpatialJointAttention(nn.Module):
    """Pre-norm multi-head attention over the joints of BOTH persons in a frame."""

    dim: int = 24
    heads: int = ATTN_HEADS
    layer_scale_init: float = 0.1

    @nn.compact
    def __call__(self, h: jnp.ndarray, valid: jnp.ndarray) -> jnp.ndarray:
        b, t, m, v, d = h.shape
        n = m * v
        if (m, v) != (PERSONS, JOINTS) or d != self.dim or d % self.heads:
            raise ValueError(f"SpatialJointAttention got {h.shape}")
        hd = d // self.heads
        x = h.reshape(b * t, n, d)
        vm = valid.reshape(b * t, n).astype(bool)
        z = nn.LayerNorm(name="norm")(x)
        qkv = nn.Dense(3 * d, use_bias=False, name="qkv")(z).reshape(b * t, n, 3, self.heads, hd)
        q, k, val = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        scores = jnp.einsum("nqhd,nkhd->nhqk", q, k) * (hd ** -0.5)
        table = self.param("hop_bias", nn.initializers.zeros, (self.heads, HOP_BUCKETS + 1))
        scores = scores + table[:, jnp.asarray(BIAS_INDEX)][None]
        scores = jnp.where(vm[:, None, None, :], scores, -1e9)
        weights = jax.nn.softmax(scores, axis=-1)
        o = jnp.einsum("nhqk,nkhd->nqhd", weights, val).reshape(b * t, n, d)
        o = nn.Dense(d, name="proj")(o)
        gamma = self.param("layer_scale", nn.initializers.constant(self.layer_scale_init), (d,))
        y = (x + gamma * o) * vm[..., None].astype(x.dtype)
        return y.reshape(b, t, m, v, d)


class SAMaskSafeSpatialEncoder(nn.Module):
    """R4 MaskSafeSpatialEncoder + SpatialJointAttention before part pooling."""

    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, valid: jnp.ndarray, training: bool) -> jnp.ndarray:
        b, t, m, _, _ = x.shape
        if valid.shape != x.shape[:-1]:
            raise ValueError(f"Spatial validity mismatch: x={x.shape}, valid={valid.shape}")
        valid_f = valid[..., None].astype(x.dtype)

        h = nn.Dense(self.spatial_dim, name="in_proj")(x)
        je = self.param("joint_embed", nn.initializers.normal(0.02), (1, 1, 1, JOINTS, self.spatial_dim))
        pe = self.param("person_embed", nn.initializers.normal(0.02), (1, 1, PERSONS, 1, self.spatial_dim))
        h = nn.gelu(h + je + pe) * valid_f

        order = jnp.asarray(r4.base.JOINT_ORDER)
        inv = jnp.argsort(order)
        h = jnp.take(h, order, axis=3)
        vm = jnp.take(valid_f, order, axis=3)
        h = h.reshape(b * t * m, JOINTS, self.spatial_dim)
        vm = vm.reshape(b * t * m, JOINTS, 1)
        mem = r4.base.GatedSweep(self.spatial_dim, reverse=False, name="joint_memory")(h)
        h = nn.LayerNorm(name="joint_memory_norm")(h + mem) * vm
        h = h.reshape(b, t, m, JOINTS, self.spatial_dim)
        h = jnp.take(h, inv, axis=3)

        h = SpatialJointAttention(self.spatial_dim, name="spatial_attention")(h, valid)

        mask = jnp.asarray(r4.base.PART_MASK_NP, h.dtype)
        counts = jnp.asarray(r4.base.PART_COUNTS_NP, h.dtype)
        parts = jnp.einsum("btmvd,pv->btmpd", h, mask)
        parts = parts / counts[None, None, None, :, None]
        flat = parts.reshape(b, t, m * 10 * self.spatial_dim)
        y = nn.Dense(self.model_dim, name="part_fuse")(flat)
        y = nn.LayerNorm(name="out_norm")(nn.gelu(y))
        return nn.Dropout(self.dropout)(y, deterministic=not training)


class FactorizedMotionSpatialEncoder(nn.Module):
    """Parameter-matched replacement for R4 Spatial-2 only."""

    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10
    mixer_rank: int = 4

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        valid: jnp.ndarray,
        training: bool,
    ) -> jnp.ndarray:
        b, t, m, _, c = x.shape

        if valid.shape != x.shape[:-1]:
            raise ValueError(
                f"Spatial validity mismatch: x={x.shape}, valid={valid.shape}"
            )
        if c != 12:
            raise ValueError(
                f"FMSE expects the 12-D joint-motion stream, got C={c}"
            )
        if self.spatial_dim % 4:
            raise ValueError("FMSE spatial_dim must be divisible by four")
        if self.spatial_dim != 24 or self.mixer_rank != 4:
            raise ValueError("FMSE-v1 is fixed at spatial_dim=24, mixer_rank=4")

        valid_f = valid[..., None].astype(x.dtype)
        branch_dim = self.spatial_dim // 4  # 6

        # Four motion components remain separate until the 24-D latent.
        pieces = []
        for i in range(4):
            xi = x[..., 3 * i : 3 * (i + 1)]
            hi = nn.Dense(
                branch_dim,
                name=f"motion_branch_{i}",
            )(xi)
            pieces.append(hi)

        h = jnp.concatenate(pieces, axis=-1)

        # Equal-parameter rank-4 cross-component mixer.
        mix = nn.Dense(
            self.mixer_rank,
            use_bias=False,
            name="motion_mix_down",
        )(h)
        mix = nn.gelu(mix)
        mix = nn.Dense(
            self.spatial_dim,
            use_bias=False,
            name="motion_mix_up",
        )(mix)
        mix_bias = self.param(
            "motion_mix_bias",
            nn.initializers.zeros,
            (self.spatial_dim,),
        )

        h = h + mix + mix_bias

        # Everything below matches MaskSafeSpatialEncoder exactly.
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
        h = nn.gelu(h + je + pe) * valid_f

        order = jnp.asarray(r4.base.JOINT_ORDER)
        inv = jnp.argsort(order)
        h = jnp.take(h, order, axis=3)
        vm = jnp.take(valid_f, order, axis=3)

        h = h.reshape(
            b * t * m,
            JOINTS,
            self.spatial_dim,
        )
        vm = vm.reshape(
            b * t * m,
            JOINTS,
            1,
        )

        mem = r4.base.GatedSweep(
            self.spatial_dim,
            reverse=False,
            name="joint_memory",
        )(h)

        h = nn.LayerNorm(
            name="joint_memory_norm",
        )(h + mem) * vm

        h = h.reshape(
            b,
            t,
            m,
            JOINTS,
            self.spatial_dim,
        )
        h = jnp.take(
            h,
            inv,
            axis=3,
        )

        h = SpatialJointAttention(self.spatial_dim, name="spatial_attention")(h, valid)

        mask = jnp.asarray(
            r4.base.PART_MASK_NP,
            h.dtype,
        )
        counts = jnp.asarray(
            r4.base.PART_COUNTS_NP,
            h.dtype,
        )

        parts = jnp.einsum(
            "btmvd,pv->btmpd",
            h,
            mask,
        )
        parts = parts / counts[
            None,
            None,
            None,
            :,
            None,
        ]

        flat = parts.reshape(
            b,
            t,
            m * 10 * self.spatial_dim,
        )

        y = nn.Dense(
            self.model_dim,
            name="part_fuse",
        )(flat)

        y = nn.LayerNorm(
            name="out_norm",
        )(nn.gelu(y))

        return nn.Dropout(
            self.dropout,
        )(
            y,
            deterministic=not training,
        )


class NestSARSAT16(nn.Module):
    """R4-FMSE with per-frame spatial joint attention in every stream."""

    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    controller_dim: int = 16
    fast_rank: int = 4
    head_rank: int = 2
    sm_residual_scale: float = 0.08
    head_residual_scale: float = 0.15

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> Mapping[str, jnp.ndarray]:

        if x.shape[1] != FRAMES or x.shape[2] != FEATURES:
            raise ValueError(
                f"Expected [B,{FRAMES},{FEATURES}], got {x.shape}"
            )

        tok = x.reshape(
            x.shape[0],
            FRAMES,
            PERSONS,
            JOINTS,
            TOKEN_CHANNELS,
        )

        controller = r4.SharedSMController(
            controller_dim=self.controller_dim,
            head_rank=self.head_rank,
            name="sm_controller",
        )(tok)

        # Exact R4 validity / self-modulation.
        joint_valid = controller["joint_valid"]
        person_present = controller["person_present"].astype(bool)
        joint_valid = joint_valid.at[..., 0].set(person_present)
        valid_f = joint_valid[..., None].astype(tok.dtype)

        gamma = controller["gamma"][:, :, None, None, :]
        beta = controller["beta"][:, :, None, None, :]
        tok = (tok * gamma + valid_f * beta) * valid_f

        pose = tok[..., 0:3]
        full_disp = tok[..., 3:6]
        phase_a = tok[..., 6:9]
        phase_b = tok[..., 9:12]
        path = tok[..., 12:15]

        joint = pose

        parents = jnp.asarray(r4.base.PARENTS)
        parent_valid = jnp.take(
            joint_valid,
            parents,
            axis=3,
        )
        bone_valid = joint_valid & parent_valid

        bone = (
            joint
            - jnp.take(
                joint,
                parents,
                axis=3,
            )
        ) * bone_valid[..., None]

        joint_motion = jnp.concatenate(
            [
                full_disp,
                phase_a,
                phase_b,
                path,
            ],
            axis=-1,
        )

        parent_full = jnp.take(
            full_disp,
            parents,
            axis=3,
        )
        parent_a = jnp.take(
            phase_a,
            parents,
            axis=3,
        )
        parent_b = jnp.take(
            phase_b,
            parents,
            axis=3,
        )
        parent_path = jnp.take(
            path,
            parents,
            axis=3,
        )

        bone_motion = jnp.concatenate(
            [
                full_disp - parent_full,
                phase_a - parent_a,
                phase_b - parent_b,
                jnp.abs(path - parent_path),
            ],
            axis=-1,
        ) * bone_valid[..., None]

        raw_streams = (
            joint,
            bone,
            joint_motion,
            bone_motion,
        )
        stream_valid = (
            joint_valid,
            bone_valid,
            joint_valid,
            bone_valid,
        )

        # Only stream 2 changes.
        spatial = []
        for i, (stream, valid) in enumerate(
            zip(raw_streams, stream_valid)
        ):
            if i == 2:
                s = FactorizedMotionSpatialEncoder(
                    spatial_dim=self.spatial_dim,
                    model_dim=self.model_dim,
                    dropout=self.dropout,
                    mixer_rank=4,
                    name="spatial_2",
                )(
                    stream,
                    valid,
                    training,
                )
            else:
                s = SAMaskSafeSpatialEncoder(
                    self.spatial_dim,
                    self.model_dim,
                    self.dropout,
                    name=f"spatial_{i}",
                )(
                    stream,
                    valid,
                    training,
                )

            gate = controller[
                "stream_gate"
            ][:, :, i:i + 1]
            spatial.append(
                s * gate
            )

        # Exact R4 M4.
        frame_streams = []
        for i, stream in enumerate(spatial):
            frame_streams.append(
                r4.SelfModBiMemory(
                    dim=self.model_dim,
                    rank=self.fast_rank,
                    residual_scale=self.sm_residual_scale,
                    name=f"frame_memory_{i}",
                )(
                    stream,
                    controller["eta"],
                    controller["alpha"],
                )
            )

        frame_stack = jnp.stack(
            frame_streams,
            axis=2,
        )

        # Exact R4 router.
        mixed, router_weights = r4.base.CrossStreamRouter(
            self.model_dim,
            name="cross_stream_after_frame",
        )(
            frame_stack
        )

        # Exact R4 slow controls.
        eta_slow = controller["eta"].reshape(
            x.shape[0],
            4,
            FRAMES // 4,
            1,
        ).mean(axis=2)

        alpha_slow = controller["alpha"].reshape(
            x.shape[0],
            4,
            FRAMES // 4,
            1,
        ).mean(axis=2)

        # Exact R4 G4 / descriptor / classifiers.
        descriptors = []
        stream_logits = []
        chunk_states = []

        for i in range(NUM_STREAMS):
            chunks, desc = r4.SelfModDescriptorHead(
                dim=self.model_dim,
                dropout=self.dropout,
                rank=self.fast_rank,
                residual_scale=self.sm_residual_scale,
                name=f"descriptor_{i}",
            )(
                mixed[:, :, i],
                eta_slow,
                alpha_slow,
                training,
            )
            descriptors.append(desc)
            chunk_states.append(chunks)
            stream_logits.append(
                nn.Dense(
                    NUM_CLASSES,
                    name=f"classifier_{i}",
                )(desc)
            )

        descs = jnp.stack(
            descriptors,
            axis=1,
        )
        sl = jnp.stack(
            stream_logits,
            axis=1,
        )

        # Exact R4 fusion.
        fusion = jax.nn.softmax(
            controller["fusion_logits"],
            axis=-1,
        )
        main_logits = jnp.einsum(
            "bs,bsc->bc",
            fusion,
            sl,
        )

        fused_desc = jnp.einsum(
            "bs,bsd->bd",
            fusion,
            descs,
        )

        # Exact R4 adaptive head.
        head_u = nn.Dense(
            self.head_rank,
            use_bias=False,
            name="adaptive_head_u",
        )(fused_desc)

        dynamic_low_rank = (
            head_u
            * controller["head_coeff"]
        )

        delta_logits = nn.Dense(
            NUM_CLASSES,
            use_bias=False,
            kernel_init=nn.initializers.normal(0.01),
            name="adaptive_head_v",
        )(dynamic_low_rank)

        logits = (
            main_logits
            + self.head_residual_scale
            * delta_logits
        )

        return {
            "logits": logits,
            "main_logits": main_logits,
            "adaptive_head_delta": delta_logits,
            "stream_logits": sl,
            "fusion_weights": fusion,
            "router_weights": router_weights,
            "spatial_stack": jnp.stack(spatial, axis=2),
            "frame_stack": frame_stack,
            "mixed_frame_stack": mixed,
            "descriptors": descs,
            "chunk_states": jnp.stack(chunk_states, axis=1),
            "sm_eta_mean": jnp.mean(
                controller["eta"],
                axis=(1, 2),
            ),
            "sm_alpha_mean": jnp.mean(
                controller["alpha"],
                axis=(1, 2),
            ),
            "sm_head_coeff": controller["head_coeff"],
            "person_presence_rate": jnp.mean(
                controller["person_present"],
                axis=1,
            ),
            "pair_presence_rate": jnp.mean(
                controller["person_present"][..., 0]
                * controller["person_present"][..., 1],
                axis=1,
            ),
        }

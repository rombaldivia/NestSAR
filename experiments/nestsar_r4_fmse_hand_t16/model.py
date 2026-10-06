from __future__ import annotations

"""NestSAR R4-FMSE + pre-pooling distal hand-relation motion residual.

The successful FMSE architecture is preserved.  The only new mechanism is a
small relation projector inside Spatial-2 before joint memory and part pooling.

For each person/frame, use six 12-D distal-to-wrist motion relations:
  left hand      - left wrist
  left hand tip  - left wrist
  left thumb     - left wrist
  right hand     - right wrist
  right hand tip - right wrist
  right thumb    - right wrist

6 * 12 = 72 dimensions.

The 72-D relation vector is projected:
    72 -> 8 -> 24
and injected only into distal/wrist joints before the existing joint-memory and
part-pooling operations.

Extra parameters:
    Dense(72,8) = 72*8 + 8 = 584
    Dense(8,24) = 8*24 + 24 = 216
    total = 800

Full model:
    1,831,932 + 800 = 1,832,732 parameters.
"""

from typing import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn

from experiments.nestsar_sm_all_t16 import model as r4


FRAMES = r4.FRAMES
PERSONS = r4.PERSONS
JOINTS = r4.JOINTS
TOKEN_CHANNELS = r4.TOKEN_CHANNELS
FEATURES = r4.FEATURES
NUM_CLASSES = r4.NUM_CLASSES
NUM_STREAMS = r4.NUM_STREAMS

# NTU120 0-based joint ids.
_REL_CHILD = (7, 21, 22, 11, 23, 24)
_REL_WRIST = (6, 6, 6, 10, 10, 10)
_DISTAL_JOINTS = (6, 7, 21, 22, 10, 11, 23, 24)


class HandRelationFMSESpatialEncoder(nn.Module):
    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10
    mixer_rank: int = 4
    relation_hidden: int = 8
    relation_scale: float = 0.10

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
                f"Hand-FMSE expects 12-D joint-motion input, got C={c}"
            )
        if self.spatial_dim != 24 or self.mixer_rank != 4:
            raise ValueError("Hand-FMSE-v1 is fixed at spatial_dim=24, mixer_rank=4")

        valid_f = valid[..., None].astype(x.dtype)
        branch_dim = self.spatial_dim // 4

        # Exact successful FMSE factorization.
        pieces = []
        for i in range(4):
            hi = nn.Dense(
                branch_dim,
                name=f"motion_branch_{i}",
            )(x[..., 3 * i : 3 * (i + 1)])
            pieces.append(hi)

        h = jnp.concatenate(pieces, axis=-1)

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

        # ------------------------------------------------------------------
        # New: pre-pooling distal relation residual.
        # ------------------------------------------------------------------
        child = jnp.take(
            x,
            jnp.asarray(_REL_CHILD),
            axis=3,
        )
        wrist = jnp.take(
            x,
            jnp.asarray(_REL_WRIST),
            axis=3,
        )
        child_valid = jnp.take(
            valid,
            jnp.asarray(_REL_CHILD),
            axis=3,
        )
        wrist_valid = jnp.take(
            valid,
            jnp.asarray(_REL_WRIST),
            axis=3,
        )
        pair_valid = (
            child_valid
            &
            wrist_valid
        )[..., None].astype(x.dtype)

        rel = (
            child
            -
            wrist
        ) * pair_valid

        rel = rel.reshape(
            b,
            t,
            m,
            72,
        )

        rel_h = nn.Dense(
            self.relation_hidden,
            name="hand_relation_down",
        )(rel)
        rel_h = nn.gelu(rel_h)
        rel_h = nn.Dense(
            self.spatial_dim,
            kernel_init=nn.initializers.normal(0.01),
            bias_init=nn.initializers.zeros,
            name="hand_relation_up",
        )(rel_h)

        # Never create a learned relation signal when none of the required
        # distal-to-wrist pairs is actually present in this person/frame.
        rel_present = jnp.any(
            pair_valid[..., 0] > 0,
            axis=-1,
        ).astype(x.dtype)[..., None]
        rel_h = rel_h * rel_present

        distal_mask = jnp.zeros(
            (JOINTS,),
            dtype=x.dtype,
        ).at[
            jnp.asarray(_DISTAL_JOINTS)
        ].set(1.0)

        h = (
            h
            +
            self.relation_scale
            *
            rel_h[:, :, :, None, :]
            *
            distal_mask[None, None, None, :, None]
        )

        # Exact FMSE/R4 path below.
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


class NestSARR4FMSEHandT16(nn.Module):
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
            -
            jnp.take(
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

        spatial = []
        for i, (stream, valid) in enumerate(
            zip(raw_streams, stream_valid)
        ):
            if i == 2:
                s = HandRelationFMSESpatialEncoder(
                    spatial_dim=self.spatial_dim,
                    model_dim=self.model_dim,
                    dropout=self.dropout,
                    mixer_rank=4,
                    relation_hidden=8,
                    relation_scale=0.10,
                    name="spatial_2",
                )(
                    stream,
                    valid,
                    training,
                )
            else:
                s = r4.MaskSafeSpatialEncoder(
                    self.spatial_dim,
                    self.model_dim,
                    self.dropout,
                    name=f"spatial_{i}",
                )(
                    stream,
                    valid,
                    training,
                )

            gate = controller["stream_gate"][:, :, i:i + 1]
            spatial.append(s * gate)

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

        mixed, router_weights = r4.base.CrossStreamRouter(
            self.model_dim,
            name="cross_stream_after_frame",
        )(frame_stack)

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

        head_u = nn.Dense(
            self.head_rank,
            use_bias=False,
            name="adaptive_head_u",
        )(fused_desc)

        dynamic_low_rank = (
            head_u
            *
            controller["head_coeff"]
        )

        delta_logits = nn.Dense(
            NUM_CLASSES,
            use_bias=False,
            kernel_init=nn.initializers.normal(0.01),
            name="adaptive_head_v",
        )(dynamic_low_rank)

        logits = (
            main_logits
            +
            self.head_residual_scale
            *
            delta_logits
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
                *
                controller["person_present"][..., 1],
                axis=1,
            ),
        }

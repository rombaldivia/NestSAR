"""NestSAR-PT-T16: Part-Token nested memory.

Diagnosis (R4 code, MaskSafeSpatialEncoder): every frame is flattened
(2 persons x 10 parts x 24) -> ONE 112-d vector *before* any temporal memory.
The temporal hierarchy therefore never sees the trajectory of an individual
body part; hands (parts 3 and 5) are mixed with the whole body first.

NestSAR-PT moves the collapse AFTER the fast temporal memory:

  joints -> mask-safe joint sweep -> 10 part tokens (persons fused per part)
         -> bidirectional cross-part GatedSweep (spatial, every frame)
         -> per-part self-modifying M4 memory (weights shared across parts)
         -> part read-out (10 x Dp -> 112) -> R4 router -> G4 -> heads

Unchanged from R4: SharedSMController, mask-safe streams (J/B/JM/BM),
CrossStreamRouter, G4 SelfModDescriptorHead, classifiers, fusion, adaptive
head and every output key used by the streaming worker.

No softmax attention, no graph/GCN, no CNN/TCN, no TxT operation.
"""
from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from experiments.nestsar_sm_all_t16 import model as r4
from experiments.nestsar_sm_all_t16.model import (
    FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS, FEATURES, NUM_CLASSES, NUM_STREAMS,
    SharedSMController, SelfModBiMemory, SelfModDescriptorHead,
)

base = r4.base  # m4_motionpreserve_t16 trainer module (GatedSweep, PARENTS, parts...)
NUM_PARTS = int(base.PART_MASK_NP.shape[0])  # 10


class PartTokenEncoder(nn.Module):
    """Mask-safe joint encoder that KEEPS the 10 anatomical part tokens.

    The joint stage is the R4 MaskSafeSpatialEncoder up to part pooling.
    Differences: (1) part pooling divides by the number of *valid* joints, so a
    partially missing part is not diluted by zeros; (2) the two persons are
    fused per part (concat -> Dense) instead of flattening the whole frame.
    """

    spatial_dim: int = 24
    part_dim: int = 40
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x, valid, training: bool):
        b, t, m, _, _ = x.shape
        valid_f = valid[..., None].astype(x.dtype)

        h = nn.Dense(self.spatial_dim, name="in_proj")(x)
        je = self.param("joint_embed", nn.initializers.normal(0.02),
                        (1, 1, 1, JOINTS, self.spatial_dim))
        pe = self.param("person_embed", nn.initializers.normal(0.02),
                        (1, 1, PERSONS, 1, self.spatial_dim))
        h = nn.gelu(h + je + pe) * valid_f

        order = jnp.asarray(base.JOINT_ORDER)
        inv = jnp.argsort(order)
        h = jnp.take(h, order, axis=3)
        vm = jnp.take(valid_f, order, axis=3)
        h = h.reshape(b * t * m, JOINTS, self.spatial_dim)
        vm = vm.reshape(b * t * m, JOINTS, 1)
        mem = base.GatedSweep(self.spatial_dim, reverse=False, name="joint_memory")(h)
        h = nn.LayerNorm(name="joint_memory_norm")(h + mem) * vm
        h = h.reshape(b, t, m, JOINTS, self.spatial_dim)
        h = jnp.take(h, inv, axis=3)

        mask = jnp.asarray(base.PART_MASK_NP, h.dtype)              # [P, V]
        part_sum = jnp.einsum("btmvd,pv->btmpd", h, mask)
        part_cnt = jnp.einsum("btmv,pv->btmp", valid.astype(h.dtype), mask)
        parts = part_sum / jnp.maximum(part_cnt, 1.0)[..., None]      # [B,T,M,P,Ds]
        part_valid = (part_cnt.sum(axis=2) > 0).astype(h.dtype)       # [B,T,P]

        # Fuse persons inside each part: [B,T,P, M*Ds] -> [B,T,P,Dp]
        parts = jnp.transpose(parts, (0, 1, 3, 2, 4)).reshape(
            b, t, NUM_PARTS, m * self.spatial_dim)
        pe2 = self.param("part_embed", nn.initializers.normal(0.02),
                         (1, 1, NUM_PARTS, self.part_dim))
        tok = nn.Dense(self.part_dim, name="person_fuse")(parts) + pe2
        tok = nn.LayerNorm(name="part_norm")(nn.gelu(tok)) * part_valid[..., None]
        tok = nn.Dropout(self.dropout)(tok, deterministic=not training)
        return tok, part_valid


class CrossPartSweep(nn.Module):
    """Bidirectional recurrent sweep over the 10 part tokens of each frame.

    Same GatedSweep operator NestSAR already uses over joints: no graph,
    no attention. Part order follows TEN_PARTS (torso, head, L arm, L hand,
    R arm, R hand, L thigh, L crus, R thigh, R crus); both directions are used.
    """

    dim: int

    @nn.compact
    def __call__(self, tok, part_valid):
        b, t, p, d = tok.shape
        flat = tok.reshape(b * t, p, d)
        f = base.GatedSweep(d, reverse=False, name="fwd")(flat)
        r = base.GatedSweep(d, reverse=True, name="bwd")(flat)
        y = nn.Dense(d, name="merge")(jnp.concatenate([f, r], axis=-1))
        out = nn.LayerNorm(name="norm")(flat + y).reshape(b, t, p, d)
        return out * part_valid[..., None]


class PartTemporalMemory(nn.Module):
    """R4 self-modifying M4 memory applied to EVERY part token in parallel.

    Weights are shared across the 10 parts (part identity comes from the part
    embedding), so parameters stay small while each part keeps its own
    temporal trajectory and fast-weight state.
    """

    dim: int
    rank: int = 4
    residual_scale: float = 0.08

    @nn.compact
    def __call__(self, tok, eta, alpha):
        b, t, p, d = tok.shape
        seq = jnp.transpose(tok, (0, 2, 1, 3)).reshape(b * p, t, d)
        eta_p = jnp.repeat(eta, p, axis=0)        # [B*P, T, 1], same order as reshape
        alpha_p = jnp.repeat(alpha, p, axis=0)
        y = SelfModBiMemory(dim=d, rank=self.rank, residual_scale=self.residual_scale,
                            name="part_memory")(seq, eta_p, alpha_p)
        return jnp.transpose(y.reshape(b, p, t, d), (0, 2, 1, 3))


class PartReadout(nn.Module):
    """Collapse parts only AFTER temporal memory: 10 x Dp -> model_dim per frame."""

    model_dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, tok, training: bool):
        b, t, p, d = tok.shape
        y = nn.Dense(self.model_dim, name="part_fuse")(tok.reshape(b, t, p * d))
        y = nn.LayerNorm(name="out_norm")(nn.gelu(y))
        return nn.Dropout(self.dropout)(y, deterministic=not training)


class NestSARPTT16(nn.Module):
    spatial_dim: int = 24
    part_dim: int = 40
    model_dim: int = 112
    dropout: float = 0.10
    controller_dim: int = 16
    fast_rank: int = 4
    head_rank: int = 2
    sm_residual_scale: float = 0.08
    head_residual_scale: float = 0.15

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Mapping[str, jnp.ndarray]:
        if x.shape[1] != FRAMES or x.shape[2] != FEATURES:
            raise ValueError(f"Expected [B,{FRAMES},{FEATURES}], got {x.shape}")
        tok = x.reshape(x.shape[0], FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS)

        # ---- identical to R4: controller + mask-safe self-modulation -------
        controller = SharedSMController(controller_dim=self.controller_dim,
                                        head_rank=self.head_rank, name="sm_controller")(tok)
        joint_valid = controller["joint_valid"]
        person_present = controller["person_present"].astype(bool)
        joint_valid = joint_valid.at[..., 0].set(person_present)
        valid_f = joint_valid[..., None].astype(tok.dtype)
        gamma = controller["gamma"][:, :, None, None, :]
        beta = controller["beta"][:, :, None, None, :]
        tok = (tok * gamma + valid_f * beta) * valid_f

        pose, full_disp = tok[..., 0:3], tok[..., 3:6]
        phase_a, phase_b, path = tok[..., 6:9], tok[..., 9:12], tok[..., 12:15]
        parents = jnp.asarray(base.PARENTS)
        parent_valid = jnp.take(joint_valid, parents, axis=3)
        bone_valid = joint_valid & parent_valid
        joint = pose
        bone = (joint - jnp.take(joint, parents, axis=3)) * bone_valid[..., None]
        joint_motion = jnp.concatenate([full_disp, phase_a, phase_b, path], axis=-1)
        bone_motion = jnp.concatenate([
            full_disp - jnp.take(full_disp, parents, axis=3),
            phase_a - jnp.take(phase_a, parents, axis=3),
            phase_b - jnp.take(phase_b, parents, axis=3),
            jnp.abs(path - jnp.take(path, parents, axis=3)),
        ], axis=-1) * bone_valid[..., None]
        raw_streams = (joint, bone, joint_motion, bone_motion)
        stream_valid = (joint_valid, bone_valid, joint_valid, bone_valid)

        # ---- NEW: part tokens survive through the fast temporal memory -----
        part_stack, frame_streams = [], []
        for i, (stream, valid) in enumerate(zip(raw_streams, stream_valid)):
            ptok, pvalid = PartTokenEncoder(self.spatial_dim, self.part_dim, self.dropout,
                                            name=f"part_encoder_{i}")(stream, valid, training)
            gate = controller["stream_gate"][:, :, i:i + 1][..., None]   # [B,T,1,1]
            ptok = ptok * gate
            ptok = CrossPartSweep(self.part_dim, name=f"cross_part_{i}")(ptok, pvalid)
            ptok = PartTemporalMemory(self.part_dim, self.fast_rank, self.sm_residual_scale,
                                      name=f"part_temporal_{i}")(ptok, controller["eta"],
                                                                 controller["alpha"])
            ptok = ptok * pvalid[..., None]
            part_stack.append(ptok)
            frame_streams.append(PartReadout(self.model_dim, self.dropout,
                                             name=f"part_readout_{i}")(ptok, training))

        # ---- identical to R4 from here: router -> G4 -> heads --------------
        frame_stack = jnp.stack(frame_streams, axis=2)
        mixed, router_weights = base.CrossStreamRouter(
            self.model_dim, name="cross_stream_after_frame")(frame_stack)
        eta_slow = controller["eta"].reshape(x.shape[0], 4, FRAMES // 4, 1).mean(axis=2)
        alpha_slow = controller["alpha"].reshape(x.shape[0], 4, FRAMES // 4, 1).mean(axis=2)

        descriptors, stream_logits, chunk_states = [], [], []
        for i in range(NUM_STREAMS):
            chunks, desc = SelfModDescriptorHead(
                dim=self.model_dim, dropout=self.dropout, rank=self.fast_rank,
                residual_scale=self.sm_residual_scale, name=f"descriptor_{i}",
            )(mixed[:, :, i], eta_slow, alpha_slow, training)
            descriptors.append(desc)
            chunk_states.append(chunks)
            stream_logits.append(nn.Dense(NUM_CLASSES, name=f"classifier_{i}")(desc))

        descs = jnp.stack(descriptors, axis=1)
        sl = jnp.stack(stream_logits, axis=1)
        fusion = jax.nn.softmax(controller["fusion_logits"], axis=-1)
        main_logits = jnp.einsum("bs,bsc->bc", fusion, sl)
        fused_desc = jnp.einsum("bs,bsd->bd", fusion, descs)
        head_u = nn.Dense(self.head_rank, use_bias=False, name="adaptive_head_u")(fused_desc)
        delta_logits = nn.Dense(NUM_CLASSES, use_bias=False,
                                kernel_init=nn.initializers.normal(0.01),
                                name="adaptive_head_v")(head_u * controller["head_coeff"])
        logits = main_logits + self.head_residual_scale * delta_logits

        return {
            "logits": logits,
            "main_logits": main_logits,
            "adaptive_head_delta": delta_logits,
            "stream_logits": sl,
            "fusion_weights": fusion,
            "router_weights": router_weights,
            "part_stack": jnp.stack(part_stack, axis=2),          # [B,T,S,P,Dp]
            "frame_stack": frame_stack,
            "mixed_frame_stack": mixed,
            "descriptors": descs,
            "chunk_states": jnp.stack(chunk_states, axis=1),
            "sm_eta_mean": jnp.mean(controller["eta"], axis=(1, 2)),
            "sm_alpha_mean": jnp.mean(controller["alpha"], axis=(1, 2)),
            "sm_head_coeff": controller["head_coeff"],
            "person_presence_rate": jnp.mean(controller["person_present"], axis=1),
            "pair_presence_rate": jnp.mean(
                controller["person_present"][..., 0] * controller["person_present"][..., 1], axis=1),
        }

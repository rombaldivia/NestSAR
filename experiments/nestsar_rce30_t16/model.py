from __future__ import annotations

"""NestSAR-RCEX: stream-oracle-aware Rival-Conditioned Evidence.

RCEX keeps the successful FMSE/R4 model frozen and changes the specialist
problem substantially relative to RCE30:

1. no teacher-forced class insertion at training time;
2. candidates come from four complementary sources:
   final FMSE Top-3, four stream Top-1 predictions, learned evidence retrieval
   Top-3, and two training-only rivals of the base Top-1;
3. a global class-conditioned retrieval head is trained to increase candidate
   recall before local reranking;
4. local rival queries consume high-resolution joint evidence AND frozen FMSE
   descriptor/stream evidence;
5. gating sees cross-stream disagreement and is trained explicitly in worker.py;
6. zero-initialized masked corrections still make initial RCEX == FMSE exactly.

The goal is to exploit the previously measured large any-stream oracle gap
without allowing a new global model to destroy already-correct FMSE decisions.
"""

from typing import Mapping
import math

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

from experiments.nestsar_sm_all_t16 import model as r4


FRAMES = 16
PERSONS = 2
JOINTS = 25
TOKEN_CHANNELS = 15
FEATURES = 750
NUM_CLASSES = 120
NUM_STREAMS = 4
BASE_DESCRIPTOR_DIM = 112

EVIDENCE_DIM = 40
LOW_RANK = 4
QUERY_MODES = 4

BASE_TOPK = 3
STREAM_TOP1_SLOTS = 4
RETRIEVAL_TOPK = 3
RIVALS_FOR_BASE_TOP1 = 2
CANDIDATE_SLOTS = (
    BASE_TOPK
    + STREAM_TOP1_SLOTS
    + RETRIEVAL_TOPK
    + RIVALS_FOR_BASE_TOP1
)  # 12


PARENTS_NP = np.asarray(r4.base.PARENTS, np.int32)

DISTAL_ANCHOR_NP = np.arange(JOINTS, dtype=np.int32)
for child, anchor in (
    (6, 5),
    (7, 6),
    (21, 6),
    (22, 6),
    (10, 9),
    (11, 10),
    (23, 10),
    (24, 10),
):
    DISTAL_ANCHOR_NP[child] = anchor

DISTAL_MASK_NP = np.zeros((JOINTS,), np.float32)
DISTAL_MASK_NP[[6, 7, 10, 11, 21, 22, 23, 24]] = 1.0


def _dct_basis(k_count=3):
    t = np.arange(FRAMES, dtype=np.float32)
    rows = []
    for k in range(1, k_count + 1):
        rows.append(
            np.sqrt(2.0 / FRAMES)
            * np.cos(np.pi * (t + 0.5) * k / FRAMES)
        )
    return np.asarray(rows, np.float32)


DCT_BASIS_NP = _dct_basis(3)


def _entropy_from_logits(logits: jnp.ndarray) -> jnp.ndarray:
    p = jax.nn.softmax(logits, axis=-1)
    return -jnp.sum(
        p * jax.nn.log_softmax(logits, axis=-1),
        axis=-1,
    )


class _FamilyViewProjector(nn.Module):
    """Keep P1, P2 and P2-P1 distinct instead of compressing 9 -> 4."""

    out_each: int = 2

    @nn.compact
    def __call__(
        self,
        family: jnp.ndarray,
        pair_valid: jnp.ndarray,
    ) -> jnp.ndarray:
        p1 = family[:, :, 0]
        p2 = family[:, :, 1]
        rel = (
            p2 - p1
        ) * pair_valid[..., None].astype(family.dtype)

        pieces = []
        for name, z in (
            ("p1", p1),
            ("p2", p2),
            ("rel", rel),
        ):
            pieces.append(
                nn.gelu(
                    nn.Dense(
                        self.out_each,
                        name=name,
                    )(z)
                )
            )
        return jnp.concatenate(pieces, axis=-1)


class RichEvidenceEncoder(nn.Module):
    """Complete pre-pooling joint-time evidence -> persistent D40 carrier."""

    dim: int = EVIDENCE_DIM
    dropout: float = 0.05

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool,
    ) -> jnp.ndarray:
        if x.shape[1:] != (FRAMES, FEATURES):
            raise ValueError(
                f"Expected [B,16,750], got {x.shape}"
            )

        tok = x.reshape(
            x.shape[0],
            FRAMES,
            PERSONS,
            JOINTS,
            TOKEN_CHANNELS,
        )

        joint_valid = jnp.any(
            jnp.abs(tok) > 1e-8,
            axis=-1,
        )
        person_present = jnp.any(
            joint_valid,
            axis=3,
        )
        joint_valid = joint_valid.at[..., 0].set(
            person_present
        )
        tok = tok * joint_valid[..., None].astype(tok.dtype)

        p1v = joint_valid[:, :, 0]
        p2v = joint_valid[:, :, 1]
        pair_valid = p1v & p2v

        pose = tok[..., 0:3]
        full = tok[..., 3:6]
        phase_a = tok[..., 6:9]
        phase_b = tok[..., 9:12]
        path = tok[..., 12:15]

        parents = jnp.asarray(PARENTS_NP)
        parent_valid = jnp.take(
            joint_valid,
            parents,
            axis=3,
        )
        bone_valid = joint_valid & parent_valid

        def parent_relative(z, absolute=False):
            parent = jnp.take(
                z,
                parents,
                axis=3,
            )
            rel = z - parent
            if absolute:
                rel = jnp.abs(rel)
            return rel * bone_valid[..., None].astype(z.dtype)

        bone_pose = parent_relative(pose)
        bone_full = parent_relative(full)
        bone_phase_a = parent_relative(phase_a)
        bone_phase_b = parent_relative(phase_b)
        bone_path = parent_relative(path, absolute=True)

        families = (
            pose,
            bone_pose,
            full,
            phase_a,
            phase_b,
            path,
            bone_full,
            bone_phase_a,
            bone_phase_b,
            bone_path,
        )

        projected = []
        for i, family in enumerate(families):
            projected.append(
                _FamilyViewProjector(
                    out_each=2,
                    name=f"family_{i}",
                )(
                    family,
                    pair_valid,
                )
            )

        presence = jnp.stack(
            [
                p1v.astype(tok.dtype),
                p2v.astype(tok.dtype),
                pair_valid.astype(tok.dtype),
            ],
            axis=-1,
        )
        presence = nn.gelu(
            nn.Dense(
                4,
                name="presence_proj",
            )(presence)
        )

        motion12 = jnp.concatenate(
            [full, phase_a, phase_b, path],
            axis=-1,
        )
        distal_anchor = jnp.asarray(DISTAL_ANCHOR_NP)
        anchor_motion = jnp.take(
            motion12,
            distal_anchor,
            axis=3,
        )
        anchor_valid = jnp.take(
            joint_valid,
            distal_anchor,
            axis=3,
        )
        distal_valid = joint_valid & anchor_valid
        distal = (
            motion12 - anchor_motion
        ) * distal_valid[..., None].astype(tok.dtype)
        distal = distal * jnp.asarray(
            DISTAL_MASK_NP,
            tok.dtype,
        )[None, None, None, :, None]

        d1 = distal[:, :, 0]
        d2 = distal[:, :, 1]
        drel = (
            d2 - d1
        ) * pair_valid[..., None].astype(tok.dtype)

        distal_pieces = []
        for name, z in (
            ("p1", d1),
            ("p2", d2),
            ("rel", drel),
        ):
            distal_pieces.append(
                nn.gelu(
                    nn.Dense(
                        2,
                        name=f"distal_{name}",
                    )(z)
                )
            )
        distal_feat = jnp.concatenate(
            distal_pieces,
            axis=-1,
        )

        h = jnp.concatenate(
            projected + [presence, distal_feat],
            axis=-1,
        )
        if h.shape[-1] != 70:
            raise ValueError(
                f"Expected 70-D evidence, got {h.shape}"
            )

        h = nn.Dense(
            self.dim,
            name="evidence_fuse",
        )(h)

        joint_embed = self.param(
            "joint_embed",
            nn.initializers.normal(0.02),
            (1, 1, JOINTS, self.dim),
        )
        time_embed = self.param(
            "time_embed",
            nn.initializers.normal(0.02),
            (1, FRAMES, 1, self.dim),
        )

        h = nn.LayerNorm(
            name="evidence_norm",
        )(
            h + joint_embed + time_embed
        )

        return nn.Dropout(
            self.dropout,
            name="evidence_dropout",
        )(
            h,
            deterministic=not training,
        )


class EvidenceRefineBlock(nn.Module):
    """Parallel low-rank channel, temporal and anatomical evidence updates."""

    dim: int = EVIDENCE_DIM
    rank: int = LOW_RANK
    dropout: float = 0.05

    def branch(
        self,
        x,
        name,
        training,
    ):
        h = nn.Dense(
            self.rank,
            use_bias=False,
            name=f"{name}_down",
        )(x)
        h = nn.gelu(h)
        h = nn.Dense(
            self.dim,
            use_bias=False,
            name=f"{name}_up",
        )(h)
        return nn.Dropout(
            self.dropout,
            name=f"{name}_drop",
        )(
            h,
            deterministic=not training,
        )

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool,
    ) -> jnp.ndarray:
        h = nn.LayerNorm(
            name="norm",
        )(x)

        temporal_prev = jnp.concatenate(
            [h[:, 0:1], h[:, :-1]],
            axis=1,
        )
        temporal_delta = h - temporal_prev

        parents = jnp.asarray(PARENTS_NP)
        parent_h = jnp.take(
            h,
            parents,
            axis=2,
        )
        spatial_delta = h - parent_h

        channel = self.branch(
            h,
            "channel",
            training,
        )
        temporal = self.branch(
            temporal_delta,
            "temporal",
            training,
        )
        spatial = self.branch(
            spatial_delta,
            "spatial",
            training,
        )

        g_channel = 0.25 * jax.nn.sigmoid(
            self.param(
                "gate_channel",
                nn.initializers.zeros,
                (),
            )
        )
        g_temporal = 0.25 * jax.nn.sigmoid(
            self.param(
                "gate_temporal",
                nn.initializers.zeros,
                (),
            )
        )
        g_spatial = 0.25 * jax.nn.sigmoid(
            self.param(
                "gate_spatial",
                nn.initializers.zeros,
                (),
            )
        )

        return (
            x
            + g_channel * channel
            + g_temporal * temporal
            + g_spatial * spatial
        )


class TemporalEvidenceBank(nn.Module):
    """Order-sensitive per-joint evidence; no global joint mean."""

    dim: int = EVIDENCE_DIM

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        mean = jnp.mean(x, axis=1)
        direction = x[:, -1] - x[:, 0]
        half = (
            jnp.mean(
                x[:, FRAMES // 2 :],
                axis=1,
            )
            - jnp.mean(
                x[:, : FRAMES // 2],
                axis=1,
            )
        )

        basis = jnp.asarray(
            DCT_BASIS_NP,
            x.dtype,
        )
        dct = jnp.einsum(
            "kt,btjd->bkjd",
            basis,
            x,
        )
        dct = jnp.transpose(
            dct,
            (0, 2, 1, 3),
        ).reshape(
            x.shape[0],
            JOINTS,
            3 * self.dim,
        )

        summary = jnp.concatenate(
            [
                mean,
                direction,
                half,
                dct,
            ],
            axis=-1,
        )

        z = nn.gelu(
            nn.Dense(
                self.dim,
                name="temporal_summary",
            )(summary)
        )

        parents = jnp.asarray(PARENTS_NP)
        parent = jnp.take(
            z,
            parents,
            axis=1,
        )

        local = jnp.concatenate(
            [z, parent, z - parent],
            axis=-1,
        )

        return nn.LayerNorm(
            name="bank_norm",
        )(
            z
            + nn.Dense(
                self.dim,
                name="anatomical_mix",
            )(local)
        )


class GlobalEvidenceRetrieval(nn.Module):
    """Class-conditioned retrieval over all 25 joint evidence tokens."""

    dim: int = EVIDENCE_DIM

    @nn.compact
    def __call__(
        self,
        bank: jnp.ndarray,
        class_embed: jnp.ndarray,
        base_logits: jnp.ndarray,
    ) -> Mapping[str, jnp.ndarray]:
        q = nn.Dense(
            self.dim,
            use_bias=False,
            name="retrieval_query",
        )(class_embed)

        k = nn.Dense(
            self.dim,
            use_bias=False,
            name="retrieval_key",
        )(bank)

        v = nn.Dense(
            self.dim,
            use_bias=False,
            name="retrieval_value",
        )(bank)

        score = jnp.einsum(
            "cd,bjd->bcj",
            q,
            k,
        ) / math.sqrt(self.dim)

        attn = jax.nn.softmax(
            score,
            axis=-1,
        )

        context = jnp.einsum(
            "bcj,bjd->bcd",
            attn,
            v,
        )

        class_feature = jnp.broadcast_to(
            class_embed[None, :, :],
            context.shape,
        )

        hidden = nn.gelu(
            nn.Dense(
                self.dim,
                name="retrieval_hidden",
            )(
                jnp.concatenate(
                    [
                        context,
                        class_feature,
                        context * class_feature,
                    ],
                    axis=-1,
                )
            )
        )

        evidence_logits = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(0.01),
            bias_init=nn.initializers.zeros,
            name="retrieval_score",
        )(hidden)[..., 0]

        # Base prior makes early retrieval sensible while the evidence head learns.
        logits = (
            evidence_logits
            + 0.20 * jax.lax.stop_gradient(base_logits)
        )

        return {
            "logits": logits,
            "attention": attn,
            "context": context,
        }


class CandidateBuilder:
    """Final Top3 + 4 stream winners + retrieval Top3 + 2 base-Top1 rivals."""

    @staticmethod
    def build(
        base_logits: jnp.ndarray,
        stream_logits: jnp.ndarray,
        retrieval_logits: jnp.ndarray,
        rival_table: jnp.ndarray,
    ):
        base_values, base_idx = jax.lax.top_k(
            base_logits,
            BASE_TOPK,
        )

        stream_top1 = jnp.argmax(
            stream_logits,
            axis=-1,
        )

        _, retrieval_idx = jax.lax.top_k(
            retrieval_logits,
            RETRIEVAL_TOPK,
        )

        base_top1 = base_idx[:, 0]
        rivals = rival_table[base_top1]

        candidate_idx = jnp.concatenate(
            [
                base_idx,
                stream_top1,
                retrieval_idx,
                rivals,
            ],
            axis=-1,
        )

        if candidate_idx.shape[-1] != CANDIDATE_SLOTS:
            raise ValueError(
                f"Expected {CANDIDATE_SLOTS} candidate slots, "
                f"got {candidate_idx.shape}"
            )

        b = candidate_idx.shape[0]
        rows = jnp.arange(b)[:, None]
        mask = jnp.zeros(
            (b, NUM_CLASSES),
            dtype=jnp.float32,
        )
        mask = mask.at[
            rows,
            candidate_idx,
        ].set(1.0)

        return {
            "candidate_idx": candidate_idx,
            "candidate_mask": mask,
            "base_top_values": base_values,
            "base_top_idx": base_idx,
            "stream_top1": stream_top1,
            "retrieval_top_idx": retrieval_idx,
        }


class RivalConditionedReadout(nn.Module):
    """Local scorer using joint evidence plus frozen FMSE context."""

    dim: int = EVIDENCE_DIM
    query_modes: int = QUERY_MODES
    delta_scale: float = 3.0

    @nn.compact
    def __call__(
        self,
        bank: jnp.ndarray,
        class_embed: jnp.ndarray,
        base_logits: jnp.ndarray,
        stream_logits: jnp.ndarray,
        base_context: jnp.ndarray,
        retrieval_logits: jnp.ndarray,
        candidates: Mapping[str, jnp.ndarray],
    ) -> Mapping[str, jnp.ndarray]:
        candidate_idx = candidates["candidate_idx"]
        candidate_mask = candidates["candidate_mask"]
        base_top_idx = candidates["base_top_idx"]
        base_top_values = candidates["base_top_values"]

        ref = class_embed[base_top_idx[:, 0]]
        cand = class_embed[candidate_idx]

        mode_embed = self.param(
            "mode_embed",
            nn.initializers.normal(0.02),
            (self.query_modes, self.dim),
        )

        q0 = (
            cand - ref[:, None, :]
        )[:, :, None, :]
        q0 = (
            q0
            + mode_embed[None, None, :, :]
            + 0.25
            * base_context[:, None, None, :]
        )

        q = nn.Dense(
            self.dim,
            use_bias=False,
            name="query_proj",
        )(q0)

        k = nn.Dense(
            self.dim,
            use_bias=False,
            name="key_proj",
        )(bank)

        v = nn.Dense(
            self.dim,
            use_bias=False,
            name="value_proj",
        )(bank)

        score = jnp.einsum(
            "bcmd,bjd->bcmj",
            q,
            k,
        ) / math.sqrt(self.dim)

        attn = jax.nn.softmax(
            score,
            axis=-1,
        )

        context = jnp.einsum(
            "bcmj,bjd->bcmd",
            attn,
            v,
        ).reshape(
            bank.shape[0],
            CANDIDATE_SLOTS,
            self.query_modes * self.dim,
        )

        rows = jnp.arange(
            base_logits.shape[0]
        )[:, None]

        cand_base = base_logits[
            rows,
            candidate_idx,
        ]

        cand_retrieval = retrieval_logits[
            rows,
            candidate_idx,
        ]

        stream_support = jnp.take_along_axis(
            stream_logits,
            candidate_idx[:, None, :],
            axis=2,
        )
        stream_support = jnp.transpose(
            stream_support,
            (0, 2, 1),
        )

        top1 = base_top_values[:, 0:1]
        base_gap = cand_base - top1

        ctx = jnp.broadcast_to(
            base_context[:, None, :],
            (
                base_context.shape[0],
                CANDIDATE_SLOTS,
                self.dim,
            ),
        )

        feat = jnp.concatenate(
            [
                context,
                cand,
                ctx,
                stream_support,
                cand_base[..., None],
                cand_retrieval[..., None],
                base_gap[..., None],
            ],
            axis=-1,
        )

        hidden = nn.gelu(
            nn.Dense(
                2 * self.dim,
                name="candidate_hidden_1",
            )(feat)
        )
        hidden = nn.gelu(
            nn.Dense(
                self.dim,
                name="candidate_hidden_2",
            )(hidden)
        )

        raw_delta = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="candidate_delta",
        )(hidden)[..., 0]

        candidate_delta = (
            self.delta_scale
            * jnp.tanh(raw_delta)
        )

        delta = jnp.zeros_like(
            base_logits
        )
        counts = jnp.zeros_like(
            base_logits
        )

        delta = delta.at[
            rows,
            candidate_idx,
        ].add(candidate_delta)

        counts = counts.at[
            rows,
            candidate_idx,
        ].add(1.0)

        delta = jnp.where(
            counts > 0,
            delta / jnp.maximum(counts, 1.0),
            0.0,
        )

        delta = delta * candidate_mask

        return {
            "delta_logits": delta,
            "attention": attn,
        }


class NestSARRCEXT16(nn.Module):
    """Complete RCEX specialist over a frozen FMSE/R4 base."""

    dim: int = EVIDENCE_DIM
    blocks: int = 2
    rank: int = LOW_RANK
    dropout: float = 0.05

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        base_logits: jnp.ndarray,
        stream_logits: jnp.ndarray,
        descriptors: jnp.ndarray,
        fusion_weights: jnp.ndarray,
        rival_table: jnp.ndarray,
        training: bool = False,
    ) -> Mapping[str, jnp.ndarray]:

        if base_logits.shape[-1] != NUM_CLASSES:
            raise ValueError(
                f"base_logits must end in {NUM_CLASSES}"
            )

        if stream_logits.shape[-2:] != (
            NUM_STREAMS,
            NUM_CLASSES,
        ):
            raise ValueError(
                f"stream_logits expected [B,4,120], "
                f"got {stream_logits.shape}"
            )

        if descriptors.shape[-2:] != (
            NUM_STREAMS,
            BASE_DESCRIPTOR_DIM,
        ):
            raise ValueError(
                f"descriptors expected [B,4,112], "
                f"got {descriptors.shape}"
            )

        class_embed = self.param(
            "class_embed",
            nn.initializers.normal(0.02),
            (NUM_CLASSES, self.dim),
        )

        carrier = RichEvidenceEncoder(
            self.dim,
            self.dropout,
            name="evidence_encoder",
        )(
            x,
            training,
        )

        block_states = []
        for i in range(self.blocks):
            carrier = EvidenceRefineBlock(
                self.dim,
                self.rank,
                self.dropout,
                name=f"evidence_block_{i}",
            )(
                carrier,
                training,
            )
            block_states.append(carrier)

        bank = TemporalEvidenceBank(
            self.dim,
            name="temporal_bank",
        )(carrier)

        fused_descriptor = jnp.einsum(
            "bs,bsd->bd",
            fusion_weights,
            descriptors,
        )

        base_context = nn.LayerNorm(
            name="base_context_norm",
        )(
            nn.gelu(
                nn.Dense(
                    self.dim,
                    name="base_context_proj",
                )(fused_descriptor)
            )
        )

        retrieval = GlobalEvidenceRetrieval(
            self.dim,
            name="global_retrieval",
        )(
            bank,
            class_embed,
            base_logits,
        )

        candidates = CandidateBuilder.build(
            base_logits,
            stream_logits,
            retrieval["logits"],
            rival_table,
        )

        readout = RivalConditionedReadout(
            self.dim,
            QUERY_MODES,
            delta_scale=3.0,
            name="rival_readout",
        )(
            bank,
            class_embed,
            base_logits,
            stream_logits,
            base_context,
            retrieval["logits"],
            candidates,
        )

        delta = readout["delta_logits"]
        candidate_mask = candidates["candidate_mask"]

        top2 = jax.lax.top_k(
            base_logits,
            2,
        )[0]
        margin = top2[:, 0] - top2[:, 1]

        masked_base = jnp.where(
            candidate_mask > 0,
            base_logits,
            -1e9,
        )
        local_entropy = _entropy_from_logits(
            masked_base
        )

        base_prob = jax.nn.softmax(
            base_logits,
            axis=-1,
        )
        candidate_mass = jnp.sum(
            base_prob * candidate_mask,
            axis=-1,
        )

        retrieval_top2 = jax.lax.top_k(
            retrieval["logits"],
            2,
        )[0]
        retrieval_margin = (
            retrieval_top2[:, 0]
            - retrieval_top2[:, 1]
        )

        base_top1 = candidates[
            "base_top_idx"
        ][:, 0]

        stream_top1 = candidates[
            "stream_top1"
        ]

        stream_agreement = jnp.mean(
            (
                stream_top1
                == base_top1[:, None]
            ).astype(jnp.float32),
            axis=-1,
        )

        stream_disagreement = (
            1.0 - stream_agreement
        )

        fusion_entropy = -jnp.sum(
            fusion_weights
            * jnp.log(
                jnp.maximum(
                    fusion_weights,
                    1e-8,
                )
            ),
            axis=-1,
        ) / math.log(NUM_STREAMS)

        correction_strength = jnp.max(
            jnp.abs(delta),
            axis=-1,
        )

        gate_features = jnp.stack(
            [
                margin,
                local_entropy,
                candidate_mass,
                retrieval_margin,
                stream_disagreement,
                fusion_entropy,
                correction_strength,
            ],
            axis=-1,
        )

        gate_hidden = nn.gelu(
            nn.Dense(
                16,
                name="gate_hidden_1",
            )(gate_features)
        )
        gate_hidden = jnp.tanh(
            nn.Dense(
                8,
                name="gate_hidden_2",
            )(gate_hidden)
        )

        # Bias -1.5 => ~0.18 initial gate.  Final logits are STILL exactly
        # FMSE because candidate_delta is exactly zero at initialization.
        gate_logit = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(-1.5),
            name="gate_out",
        )(gate_hidden)[..., 0]

        gate = jax.nn.sigmoid(
            gate_logit
        )

        final_logits = (
            base_logits
            + gate[:, None] * delta
        )

        return {
            "logits": final_logits,
            "base_logits": base_logits,
            "delta_logits": delta,
            "gate": gate,
            "gate_logit": gate_logit,
            "gate_features": gate_features,
            "candidate_idx": candidates["candidate_idx"],
            "candidate_mask": candidate_mask,
            "base_top_idx": candidates["base_top_idx"],
            "stream_top1": stream_top1,
            "retrieval_top_idx": candidates["retrieval_top_idx"],
            "retrieval_logits": retrieval["logits"],
            "retrieval_attention": retrieval["attention"],
            "stream_disagreement": stream_disagreement,
            "fusion_entropy": fusion_entropy,
            "base_context": base_context,
            "evidence_bank": bank,
            "carrier": carrier,
            "block_states": jnp.stack(
                block_states,
                axis=1,
            ),
            "attention": readout["attention"],
        }


# Compatibility alias for older RCE30 import paths in shared utilities.
NestSARRCE30T16 = NestSARRCEXT16

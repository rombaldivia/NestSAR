from __future__ import annotations

"""Compact Rival-Conditioned Evidence (RCE30) specialist.

The FMSE/R4 base remains frozen and supplies base logits.  This module consumes
the same [B,16,750] input and learns only a residual correction.

Design constraints from the diagnostic work:
- preserve individual joint-time evidence until the specialist readout;
- retain complete pose/motion/parent-relative motion families;
- include distal and inter-person evidence;
- avoid JT32's destructive mean-pooling readout;
- build candidate sets from FMSE Top-3 plus training-derived rivals;
- use learned rival-conditioned queries;
- zero-initialize the correction so initial predictions exactly equal FMSE;
- mask corrections outside the active candidate set.

The implementation also includes a fixed-query ablation so the ordinary
learned-query and rival-conditioned prototypes can be compared under the same
training schedule.
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

EVIDENCE_DIM = 40
LOW_RANK = 4
QUERY_MODES = 4
TOPK = 3
RIVALS_PER_TOP = 2
CANDIDATE_SLOTS = TOPK * (1 + RIVALS_PER_TOP)  # 9


PARENTS_NP = np.asarray(r4.base.PARENTS, np.int32)

# NTU 0-based distal chain anchors.
# wrist -> elbow, hand -> wrist, hand-tip/thumb -> wrist.
DISTAL_ANCHOR_NP = np.arange(JOINTS, dtype=np.int32)
for child, anchor in (
    (6, 5),   # left wrist -> left elbow
    (7, 6),   # left hand -> left wrist
    (21, 6),  # left hand tip -> left wrist
    (22, 6),  # left thumb -> left wrist
    (10, 9),  # right wrist -> right elbow
    (11, 10), # right hand -> right wrist
    (23, 10), # right hand tip -> right wrist
    (24, 10), # right thumb -> right wrist
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


def _safe_entropy_from_logits(logits: jnp.ndarray) -> jnp.ndarray:
    p = jax.nn.softmax(logits, axis=-1)
    return -jnp.sum(p * jax.nn.log_softmax(logits, axis=-1), axis=-1)


class _FamilyViewProjector(nn.Module):
    """Project P1, P2 and relative 3-D evidence separately (never 9 -> 4)."""

    out_each: int = 2

    @nn.compact
    def __call__(
        self,
        family: jnp.ndarray,   # [B,T,P,J,3]
        pair_valid: jnp.ndarray,
    ) -> jnp.ndarray:
        p1 = family[:, :, 0]
        p2 = family[:, :, 1]
        rel = (p2 - p1) * pair_valid[..., None].astype(family.dtype)

        pieces = []
        for name, x in (("p1", p1), ("p2", p2), ("rel", rel)):
            pieces.append(
                nn.gelu(
                    nn.Dense(
                        self.out_each,
                        name=name,
                    )(x)
                )
            )
        return jnp.concatenate(pieces, axis=-1)  # 6-D


class RichEvidenceEncoder(nn.Module):
    """Preserve full pre-pooling joint-time evidence in a D40 carrier."""

    dim: int = EVIDENCE_DIM
    dropout: float = 0.05

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        if x.shape[1:] != (FRAMES, FEATURES):
            raise ValueError(f"Expected [B,16,750], got {x.shape}")

        tok = x.reshape(
            x.shape[0],
            FRAMES,
            PERSONS,
            JOINTS,
            TOKEN_CHANNELS,
        )

        joint_valid = jnp.any(jnp.abs(tok) > 1e-8, axis=-1)
        person_present = jnp.any(joint_valid, axis=3)
        joint_valid = joint_valid.at[..., 0].set(person_present)
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

        parent_valid = jnp.take(joint_valid, parents, axis=3)
        bone_valid = joint_valid & parent_valid

        def parent_relative(z, absolute=False):
            parent = jnp.take(z, parents, axis=3)
            rel = z - parent
            if absolute:
                rel = jnp.abs(rel)
            return rel * bone_valid[..., None].astype(z.dtype)

        bone_pose = parent_relative(pose)
        bone_full = parent_relative(full)
        bone_phase_a = parent_relative(phase_a)
        bone_phase_b = parent_relative(phase_b)
        bone_path = parent_relative(path, absolute=True)

        # Ten complete evidence families.  R4/FMSE information is not dropped.
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

        # Validity/inter-person presence evidence.
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

        # Explicit distal evidence on the complete 12-D motion descriptor.
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
        distal = (motion12 - anchor_motion) * distal_valid[..., None].astype(tok.dtype)
        distal = distal * jnp.asarray(DISTAL_MASK_NP, tok.dtype)[None, None, None, :, None]

        d1 = distal[:, :, 0]
        d2 = distal[:, :, 1]
        drel = (d2 - d1) * pair_valid[..., None].astype(tok.dtype)

        distal_pieces = []
        for name, z in (("p1", d1), ("p2", d2), ("rel", drel)):
            distal_pieces.append(
                nn.gelu(
                    nn.Dense(
                        2,
                        name=f"distal_{name}",
                    )(z)
                )
            )
        distal_feat = jnp.concatenate(distal_pieces, axis=-1)  # 6-D

        h = jnp.concatenate(
            projected + [presence, distal_feat],
            axis=-1,
        )
        if h.shape[-1] != 70:
            raise ValueError(f"Expected 70-D preserved evidence, got {h.shape}")

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

        h = nn.LayerNorm(name="evidence_norm")(
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
    """Cheap joint-time refinement with channel, temporal and anatomical deltas.

    Three D40 -> rank4 -> D40 residual branches preserve the ~5 MFLOP specialist
    budget while still changing the architecture substantially.
    """

    dim: int = EVIDENCE_DIM
    rank: int = LOW_RANK
    dropout: float = 0.05

    def branch(self, x, name, training):
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
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        h = nn.LayerNorm(name="norm")(x)

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

        channel = self.branch(h, "channel", training)
        temporal = self.branch(temporal_delta, "temporal", training)
        spatial = self.branch(spatial_delta, "spatial", training)

        # Small bounded gates.  These branches augment rather than replace the
        # carrier.
        g_channel = 0.20 * jax.nn.sigmoid(
            self.param("gate_channel", nn.initializers.zeros, ())
        )
        g_temporal = 0.20 * jax.nn.sigmoid(
            self.param("gate_temporal", nn.initializers.zeros, ())
        )
        g_spatial = 0.20 * jax.nn.sigmoid(
            self.param("gate_spatial", nn.initializers.zeros, ())
        )

        return (
            x
            + g_channel * channel
            + g_temporal * temporal
            + g_spatial * spatial
        )


class TemporalEvidenceBank(nn.Module):
    """Order-sensitive temporal summaries without destructive backbone pooling."""

    dim: int = EVIDENCE_DIM

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(x, axis=1)
        direction = x[:, -1] - x[:, 0]
        half = (
            jnp.mean(x[:, FRAMES // 2 :], axis=1)
            - jnp.mean(x[:, : FRAMES // 2], axis=1)
        )

        basis = jnp.asarray(DCT_BASIS_NP, x.dtype)
        dct = jnp.einsum(
            "kt,btjd->bkjd",
            basis,
            x,
        )
        dct = jnp.transpose(dct, (0, 2, 1, 3)).reshape(
            x.shape[0],
            JOINTS,
            3 * self.dim,
        )

        summary = jnp.concatenate(
            [mean, direction, half, dct],
            axis=-1,
        )  # 6 * D

        z = nn.gelu(
            nn.Dense(
                self.dim,
                name="temporal_summary",
            )(summary)
        )

        parents = jnp.asarray(PARENTS_NP)
        parent = jnp.take(z, parents, axis=1)
        local = jnp.concatenate(
            [z, parent, z - parent],
            axis=-1,
        )
        return nn.LayerNorm(name="bank_norm")(
            z
            + nn.Dense(
                self.dim,
                name="anatomical_mix",
            )(local)
        )


class CandidateBuilder:
    """Pure helper for Top-3 + two training-derived rivals per top candidate."""

    @staticmethod
    def build(
        base_logits: jnp.ndarray,
        rival_table: jnp.ndarray,
        force_class: jnp.ndarray | None = None,
    ):
        top_values, top_idx = jax.lax.top_k(base_logits, TOPK)
        rivals = rival_table[top_idx]  # [B,3,2]
        candidate_idx = jnp.concatenate(
            [
                top_idx[:, :, None],
                rivals,
            ],
            axis=-1,
        ).reshape(base_logits.shape[0], CANDIDATE_SLOTS)

        natural_idx = candidate_idx
        if force_class is not None:
            candidate_idx = candidate_idx.at[:, -1].set(force_class)

        def mask_from_idx(idx):
            b = idx.shape[0]
            rows = jnp.arange(b)[:, None]
            mask = jnp.zeros(
                (b, NUM_CLASSES),
                dtype=jnp.float32,
            )
            return mask.at[rows, idx].max(1.0)

        return (
            candidate_idx,
            mask_from_idx(candidate_idx),
            natural_idx,
            mask_from_idx(natural_idx),
            top_values,
            top_idx,
        )


class RivalConditionedReadout(nn.Module):
    dim: int = EVIDENCE_DIM
    query_modes: int = QUERY_MODES

    @nn.compact
    def __call__(
        self,
        bank: jnp.ndarray,          # [B,25,D]
        base_logits: jnp.ndarray,   # [B,120]
        rival_table: jnp.ndarray,   # [120,2]
        force_class: jnp.ndarray | None,
    ) -> Mapping[str, jnp.ndarray]:

        (
            candidate_idx,
            candidate_mask,
            natural_idx,
            natural_mask,
            top_values,
            top_idx,
        ) = CandidateBuilder.build(
            base_logits,
            rival_table,
            force_class,
        )

        class_embed = self.param(
            "class_embed",
            nn.initializers.normal(0.02),
            (NUM_CLASSES, self.dim),
        )
        mode_embed = self.param(
            "mode_embed",
            nn.initializers.normal(0.02),
            (self.query_modes, self.dim),
        )

        ref = class_embed[top_idx[:, 0]]
        cand = class_embed[candidate_idx]

        q0 = cand - ref[:, None, :]
        q0 = q0[:, :, None, :] + mode_embed[None, None, :, :]
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
        attn = jax.nn.softmax(score, axis=-1)

        context = jnp.einsum(
            "bcmj,bjd->bcmd",
            attn,
            v,
        ).reshape(
            bank.shape[0],
            CANDIDATE_SLOTS,
            self.query_modes * self.dim,
        )

        rows = jnp.arange(base_logits.shape[0])[:, None]
        cand_base = base_logits[rows, candidate_idx]
        top1 = top_values[:, 0:1]
        gap = cand_base - top1

        feat = jnp.concatenate(
            [
                context,
                cand,
                cand_base[..., None],
                gap[..., None],
            ],
            axis=-1,
        )

        hidden = nn.gelu(
            nn.Dense(
                self.dim,
                name="candidate_hidden",
            )(feat)
        )

        # Exact-baseline guarantee at initialization.
        candidate_delta = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="candidate_delta",
        )(hidden)[..., 0]

        # Average duplicate candidate slots rather than double-counting them.
        delta = jnp.zeros_like(base_logits)
        counts = jnp.zeros_like(base_logits)
        delta = delta.at[rows, candidate_idx].add(candidate_delta)
        counts = counts.at[rows, candidate_idx].add(1.0)
        delta = jnp.where(counts > 0, delta / jnp.maximum(counts, 1.0), 0.0)
        delta = delta * candidate_mask

        return {
            "delta_logits": delta,
            "candidate_idx": candidate_idx,
            "candidate_mask": candidate_mask,
            "natural_candidate_idx": natural_idx,
            "natural_candidate_mask": natural_mask,
            "attention": attn,
        }


class FixedQueryReadout(nn.Module):
    """Ordinary learned-query ablation under the same evidence encoder."""

    dim: int = EVIDENCE_DIM
    query_modes: int = QUERY_MODES

    @nn.compact
    def __call__(
        self,
        bank: jnp.ndarray,
        base_logits: jnp.ndarray,
        rival_table: jnp.ndarray,
        force_class: jnp.ndarray | None,
    ) -> Mapping[str, jnp.ndarray]:

        (
            candidate_idx,
            candidate_mask,
            natural_idx,
            natural_mask,
            _,
            _,
        ) = CandidateBuilder.build(
            base_logits,
            rival_table,
            force_class,
        )

        queries = self.param(
            "queries",
            nn.initializers.normal(0.02),
            (self.query_modes, self.dim),
        )
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
            "md,bjd->bmj",
            queries,
            k,
        ) / math.sqrt(self.dim)
        attn = jax.nn.softmax(score, axis=-1)
        context = jnp.einsum(
            "bmj,bjd->bmd",
            attn,
            v,
        ).reshape(bank.shape[0], self.query_modes * self.dim)

        hidden = nn.gelu(
            nn.Dense(
                80,
                name="fixed_hidden",
            )(context)
        )
        delta = nn.Dense(
            NUM_CLASSES,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="fixed_delta",
        )(hidden)
        delta = delta * candidate_mask

        return {
            "delta_logits": delta,
            "candidate_idx": candidate_idx,
            "candidate_mask": candidate_mask,
            "natural_candidate_idx": natural_idx,
            "natural_candidate_mask": natural_mask,
            "attention": attn,
        }


class NestSARRCE30T16(nn.Module):
    """Full RCE specialist.  Base FMSE logits are provided externally/frozen."""

    variant: str = "rival"
    dim: int = EVIDENCE_DIM
    blocks: int = 2
    rank: int = LOW_RANK
    dropout: float = 0.05

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        base_logits: jnp.ndarray,
        rival_table: jnp.ndarray,
        training: bool = False,
        force_class: jnp.ndarray | None = None,
    ) -> Mapping[str, jnp.ndarray]:

        if self.variant not in ("rival", "fixed"):
            raise ValueError(f"variant must be rival or fixed, got {self.variant}")

        carrier = RichEvidenceEncoder(
            self.dim,
            self.dropout,
            name="evidence_encoder",
        )(x, training)

        block_states = []
        for i in range(self.blocks):
            carrier = EvidenceRefineBlock(
                self.dim,
                self.rank,
                self.dropout,
                name=f"evidence_block_{i}",
            )(carrier, training)
            block_states.append(carrier)

        bank = TemporalEvidenceBank(
            self.dim,
            name="temporal_bank",
        )(carrier)

        if self.variant == "rival":
            readout = RivalConditionedReadout(
                self.dim,
                QUERY_MODES,
                name="rival_readout",
            )(
                bank,
                base_logits,
                rival_table,
                force_class,
            )
        else:
            readout = FixedQueryReadout(
                self.dim,
                QUERY_MODES,
                name="fixed_readout",
            )(
                bank,
                base_logits,
                rival_table,
                force_class,
            )

        delta = readout["delta_logits"]

        top2 = jax.lax.top_k(base_logits, 2)[0]
        margin = top2[:, 0] - top2[:, 1]

        natural_mask = readout["natural_candidate_mask"]
        masked_base = jnp.where(
            natural_mask > 0,
            base_logits,
            -1e9,
        )
        local_entropy = _safe_entropy_from_logits(masked_base)

        base_prob = jax.nn.softmax(base_logits, axis=-1)
        candidate_mass = jnp.sum(base_prob * natural_mask, axis=-1)
        correction_strength = jnp.max(jnp.abs(delta), axis=-1)

        gate_features = jnp.stack(
            [
                margin,
                local_entropy,
                candidate_mass,
                correction_strength,
            ],
            axis=-1,
        )

        gate_hidden = jnp.tanh(
            nn.Dense(
                8,
                name="gate_hidden",
            )(gate_features)
        )
        gate_logit = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(-4.0),
            name="gate_out",
        )(gate_hidden)[..., 0]
        gate = jax.nn.sigmoid(gate_logit)

        final_logits = base_logits + gate[:, None] * delta

        return {
            "logits": final_logits,
            "base_logits": base_logits,
            "delta_logits": delta,
            "gate": gate,
            "candidate_idx": readout["candidate_idx"],
            "candidate_mask": readout["candidate_mask"],
            "natural_candidate_idx": readout["natural_candidate_idx"],
            "natural_candidate_mask": natural_mask,
            "evidence_bank": bank,
            "carrier": carrier,
            "block_states": jnp.stack(block_states, axis=1),
            "attention": readout["attention"],
        }

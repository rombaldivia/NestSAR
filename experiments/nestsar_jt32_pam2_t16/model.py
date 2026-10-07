from __future__ import annotations

"""NestSAR-JT32-PAM2-T16.

A deep redesign focused on preserving joint x time information until the final
readout.  The old M4 -> scalar stream router -> G4 pipeline is removed.

Core representation:
    [B,16,2,25,15]
        -> factorized P1/P2/relative joint features
        -> [B,16,25,32] high-resolution carrier

Each JointTimeBlock computes IN PARALLEL from the same carrier:
    temporal axial attention
    spatial axial attention with anatomical-hop bias
    parallel associative delta memory
    fixed spectral temporal branch
    pointwise FFN

The branch residuals are summed back into the carrier.  No joint pooling and no
temporal downsampling occur inside the two core blocks.

Only at readout are T16/T8/T4 and hand-region summaries created in parallel.
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

DIM = 32
HEADS = 4
HEAD_DIM = DIM // HEADS
MEM_RANK = 4


# --------------------------------------------------------------------------------------
# Static anatomy / temporal bases.
# --------------------------------------------------------------------------------------

PARENTS_NP = np.asarray(r4.base.PARENTS, np.int32)


def _hop_buckets():
    adjacency = np.zeros((JOINTS, JOINTS), np.int32)
    for j, p in enumerate(PARENTS_NP.tolist()):
        adjacency[j, p] = 1
        adjacency[p, j] = 1

    out = np.full((JOINTS, JOINTS), 4, np.int32)

    for src in range(JOINTS):
        dist = np.full((JOINTS,), 999, np.int32)
        dist[src] = 0
        frontier = [src]

        while frontier:
            u = frontier.pop(0)
            for v in np.flatnonzero(adjacency[u]):
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    frontier.append(int(v))

        out[src] = np.minimum(dist, 4)

    return out


HOP_BUCKETS_NP = _hop_buckets()


def _dct_basis(k_count=4):
    t = np.arange(FRAMES, dtype=np.float32)
    basis = []
    for k in range(1, k_count + 1):
        v = np.sqrt(2.0 / FRAMES) * np.cos(
            np.pi * (t + 0.5) * k / FRAMES
        )
        basis.append(v.astype(np.float32))
    return np.stack(basis, axis=0)


DCT_BASIS_NP = _dct_basis(4)

HAND_JOINTS = np.asarray(
    [6, 7, 10, 11, 21, 22, 23, 24],
    np.int32,
)


def safe_unit(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    sq = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
    return x / jnp.sqrt(jnp.maximum(sq, eps * eps))


class FactorizedJointEncoder(nn.Module):
    """Build one 32-D token for every (time, joint) without pooling joints."""

    dim: int = DIM

    @nn.compact
    def __call__(self, tok: jnp.ndarray) -> jnp.ndarray:
        if tok.shape[1:] != (FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS):
            raise ValueError(
                f"Expected [B,16,2,25,15], got {tok.shape}"
            )

        valid = jnp.any(
            jnp.abs(tok) > 1e-8,
            axis=-1,
        )
        person_present = jnp.any(
            valid,
            axis=3,
        )
        valid = valid.at[:, :, :, 0].set(person_present)

        valid_f = valid[..., None].astype(tok.dtype)
        tok = tok * valid_f

        p1 = tok[:, :, 0]
        p2 = tok[:, :, 1]
        v1 = valid[:, :, 0]
        v2 = valid[:, :, 1]
        pair = (v1 & v2)[..., None].astype(tok.dtype)

        parents = jnp.asarray(PARENTS_NP)

        def actor_group(start, end, *, absolute_rel=False):
            a = p1[..., start:end]
            b = p2[..., start:end]
            rel = (b - a) * pair
            if absolute_rel:
                rel = jnp.abs(rel)
            return jnp.concatenate([a, b, rel], axis=-1)

        # Pose and five motion families.
        pose = actor_group(0, 3)
        full = actor_group(3, 6)
        phase_a = actor_group(6, 9)
        phase_b = actor_group(9, 12)
        path = actor_group(12, 15, absolute_rel=True)

        # Bone geometry and bone displacement retain local anatomical changes.
        #
        # p1/p2 are [B,T,J,C], so the joint axis is 2 (not 3).  Keep the
        # parent/child validity mask explicit as in historical R4 so a missing
        # parent cannot create a false bone vector against padded zeroes.
        p1_parent_valid = jnp.take(v1, parents, axis=2)
        p2_parent_valid = jnp.take(v2, parents, axis=2)
        p1_bone_valid = (v1 & p1_parent_valid)[..., None].astype(tok.dtype)
        p2_bone_valid = (v2 & p2_parent_valid)[..., None].astype(tok.dtype)

        p1_parent_pose = jnp.take(p1[..., 0:3], parents, axis=2)
        p2_parent_pose = jnp.take(p2[..., 0:3], parents, axis=2)
        p1_bone = (p1[..., 0:3] - p1_parent_pose) * p1_bone_valid
        p2_bone = (p2[..., 0:3] - p2_parent_pose) * p2_bone_valid
        bone_pair = (
            (v1 & p1_parent_valid & v2 & p2_parent_valid)[..., None]
            .astype(tok.dtype)
        )
        bone_rel = (p2_bone - p1_bone) * bone_pair
        bone_pose = jnp.concatenate(
            [p1_bone, p2_bone, bone_rel],
            axis=-1,
        )

        p1_parent_full = jnp.take(p1[..., 3:6], parents, axis=2)
        p2_parent_full = jnp.take(p2[..., 3:6], parents, axis=2)
        p1_bm = (p1[..., 3:6] - p1_parent_full) * p1_bone_valid
        p2_bm = (p2[..., 3:6] - p2_parent_full) * p2_bone_valid
        bm_rel = (p2_bm - p1_bm) * bone_pair
        bone_motion = jnp.concatenate(
            [p1_bm, p2_bm, bm_rel],
            axis=-1,
        )

        families = (
            pose,
            bone_pose,
            full,
            bone_motion,
            phase_a,
            phase_b,
            path,
        )

        pieces = []
        for i, f in enumerate(families):
            z = nn.Dense(
                4,
                name=f"family_{i}",
            )(f)
            pieces.append(nn.gelu(z))

        presence = jnp.stack(
            [
                v1.astype(tok.dtype),
                v2.astype(tok.dtype),
                (v1 & v2).astype(tok.dtype),
            ],
            axis=-1,
        )
        presence = nn.gelu(
            nn.Dense(
                4,
                name="presence_proj",
            )(presence)
        )
        pieces.append(presence)

        x = jnp.concatenate(
            pieces,
            axis=-1,
        )

        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Factorized encoder produced {x.shape[-1]} dims, expected {self.dim}"
            )

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

        x = x + joint_embed + time_embed
        return nn.LayerNorm(name="out_norm")(x)


class TemporalAxialAttention(nn.Module):
    dim: int = DIM
    heads: int = HEADS
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x, training: bool):
        b, t, j, d = x.shape
        if d != self.dim or t != FRAMES:
            raise ValueError(f"Unexpected temporal attention input {x.shape}")

        h = nn.LayerNorm(name="norm")(x)
        h = jnp.transpose(h, (0, 2, 1, 3)).reshape(
            b * j,
            t,
            d,
        )

        qkv = nn.Dense(
            3 * self.dim,
            name="qkv",
        )(h)
        qkv = qkv.reshape(
            b * j,
            t,
            3,
            self.heads,
            self.dim // self.heads,
        )

        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        scores = jnp.einsum(
            "nlhd,nshd->nhls",
            q,
            k,
        ) / math.sqrt(self.dim // self.heads)

        rel_bias = self.param(
            "relative_bias",
            nn.initializers.zeros,
            (self.heads, 6),
        )
        idx = jnp.arange(t)
        distance = jnp.abs(
            idx[:, None] - idx[None, :]
        )
        buckets = jnp.where(
            distance == 0,
            0,
            jnp.where(
                distance == 1,
                1,
                jnp.where(
                    distance == 2,
                    2,
                    jnp.where(
                        distance <= 4,
                        3,
                        jnp.where(distance <= 8, 4, 5),
                    ),
                ),
            ),
        )
        scores = scores + jnp.take(
            rel_bias,
            buckets,
            axis=1,
        )[None, ...]

        attn = jax.nn.softmax(
            scores,
            axis=-1,
        )
        attn = nn.Dropout(
            self.dropout,
            name="attn_dropout",
        )(
            attn,
            deterministic=not training,
        )

        y = jnp.einsum(
            "nhls,nshd->nlhd",
            attn,
            v,
        ).reshape(
            b * j,
            t,
            d,
        )

        y = nn.Dense(
            self.dim,
            name="out",
        )(y)

        y = nn.Dropout(
            self.dropout,
            name="out_dropout",
        )(
            y,
            deterministic=not training,
        )

        return jnp.transpose(
            y.reshape(b, j, t, d),
            (0, 2, 1, 3),
        )


class SpatialAxialAttention(nn.Module):
    dim: int = DIM
    heads: int = HEADS
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x, training: bool):
        b, t, j, d = x.shape
        if d != self.dim or j != JOINTS:
            raise ValueError(f"Unexpected spatial attention input {x.shape}")

        h = nn.LayerNorm(name="norm")(x).reshape(
            b * t,
            j,
            d,
        )

        qkv = nn.Dense(
            3 * self.dim,
            name="qkv",
        )(h)
        qkv = qkv.reshape(
            b * t,
            j,
            3,
            self.heads,
            self.dim // self.heads,
        )

        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        scores = jnp.einsum(
            "nlhd,nshd->nhls",
            q,
            k,
        ) / math.sqrt(self.dim // self.heads)

        hop_bias = self.param(
            "hop_bias",
            nn.initializers.zeros,
            (self.heads, 5),
        )
        hop_bucket = jnp.asarray(
            HOP_BUCKETS_NP,
        )
        scores = scores + jnp.take(
            hop_bias,
            hop_bucket,
            axis=1,
        )[None, ...]

        attn = jax.nn.softmax(
            scores,
            axis=-1,
        )
        attn = nn.Dropout(
            self.dropout,
            name="attn_dropout",
        )(
            attn,
            deterministic=not training,
        )

        y = jnp.einsum(
            "nhls,nshd->nlhd",
            attn,
            v,
        ).reshape(
            b * t,
            j,
            d,
        )

        y = nn.Dense(
            self.dim,
            name="out",
        )(y)

        y = nn.Dropout(
            self.dropout,
            name="out_dropout",
        )(
            y,
            deterministic=not training,
        )

        return y.reshape(
            b,
            t,
            j,
            d,
        )


class ParallelAssociativeMemory(nn.Module):
    """Low-rank delta memory computed with an associative parallel scan."""

    dim: int = DIM
    rank: int = MEM_RANK
    alpha: float = 0.99
    eta: float = 0.10

    @nn.compact
    def __call__(self, x):
        b, t, j, d = x.shape
        h = nn.LayerNorm(name="norm")(x)

        # Scan each joint independently across T=16.
        h = jnp.transpose(
            h,
            (0, 2, 1, 3),
        )  # [B,J,T,D]

        k = nn.Dense(
            self.rank,
            use_bias=False,
            name="key",
        )(h)
        q = nn.Dense(
            self.rank,
            use_bias=False,
            name="query",
        )(h)

        k = safe_unit(jnp.tanh(k))
        q = safe_unit(jnp.tanh(q))

        eye = jnp.eye(
            self.rank,
            dtype=h.dtype,
        )

        kk = jnp.einsum(
            "...r,...s->...rs",
            k,
            k,
        )

        A = (
            self.alpha
            * eye[None, None, None, :, :]
            -
            self.eta
            * kk
        )

        Bmat = self.eta * jnp.einsum(
            "...r,...d->...rd",
            k,
            h,
        )

        def compose(left, right):
            A_l, B_l = left
            A_r, B_r = right

            A_out = jnp.einsum(
                "...ij,...jk->...ik",
                A_r,
                A_l,
            )
            B_out = (
                jnp.einsum(
                    "...ij,...jd->...id",
                    A_r,
                    B_l,
                )
                +
                B_r
            )
            return A_out, B_out

        A_pref, B_pref = jax.lax.associative_scan(
            compose,
            (A, Bmat),
            axis=2,
        )

        memory0 = self.param(
            "memory0",
            nn.initializers.normal(0.01),
            (self.rank, self.dim),
        )

        memory = (
            jnp.einsum(
                "...ij,jd->...id",
                A_pref,
                memory0,
            )
            +
            B_pref
        )

        reads = jnp.einsum(
            "...r,...rd->...d",
            q,
            memory,
        )

        return jnp.transpose(
            reads,
            (0, 2, 1, 3),
        )


class SpectralFineMotion(nn.Module):
    dim: int = DIM

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm(name="norm")(x)

        basis = jnp.asarray(
            DCT_BASIS_NP,
            dtype=x.dtype,
        )

        coeff = jnp.einsum(
            "kt,btjd->bjkd",
            basis,
            h,
        )
        recon = jnp.einsum(
            "kt,bjkd->btjd",
            basis,
            coeff,
        )

        return nn.Dense(
            self.dim,
            use_bias=False,
            name="proj",
        )(recon)


class PointwiseFFN(nn.Module):
    dim: int = DIM
    hidden: int = 64
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x, training: bool):
        h = nn.LayerNorm(name="norm")(x)
        h = nn.Dense(
            self.hidden,
            name="in_proj",
        )(h)
        h = nn.gelu(h)
        h = nn.Dropout(
            self.dropout,
            name="drop1",
        )(
            h,
            deterministic=not training,
        )
        h = nn.Dense(
            self.dim,
            name="out_proj",
        )(h)
        return nn.Dropout(
            self.dropout,
            name="drop2",
        )(
            h,
            deterministic=not training,
        )


class JointTimeBlock(nn.Module):
    dim: int = DIM
    heads: int = HEADS
    memory_rank: int = MEM_RANK
    dropout: float = 0.10

    def _gate(self, name: str, maximum: float):
        raw = self.param(
            name,
            nn.initializers.zeros,
            (),
        )
        return maximum * jax.nn.sigmoid(raw)

    @nn.compact
    def __call__(self, x, training: bool):
        # Every branch consumes the SAME input carrier.  No branch depends on
        # another branch's output.
        temporal = TemporalAxialAttention(
            self.dim,
            self.heads,
            self.dropout,
            name="temporal",
        )(x, training)

        spatial = SpatialAxialAttention(
            self.dim,
            self.heads,
            self.dropout,
            name="spatial",
        )(x, training)

        memory = ParallelAssociativeMemory(
            self.dim,
            self.memory_rank,
            name="memory",
        )(x)

        spectral = SpectralFineMotion(
            self.dim,
            name="spectral",
        )(x)

        ffn = PointwiseFFN(
            self.dim,
            2 * self.dim,
            self.dropout,
            name="ffn",
        )(x, training)

        return (
            x
            + self._gate("gate_temporal", 0.20) * temporal
            + self._gate("gate_spatial", 0.20) * spatial
            + self._gate("gate_memory", 0.10) * memory
            + self._gate("gate_spectral", 0.20) * spectral
            + self._gate("gate_ffn", 0.20) * ffn
        )


class ReadoutProjection(nn.Module):
    out_dim: int = 64
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x, training: bool):
        h = nn.Dense(
            self.out_dim,
            name="proj",
        )(x)
        h = nn.LayerNorm(
            name="norm",
        )(nn.gelu(h))
        return nn.Dropout(
            self.dropout,
            name="drop",
        )(
            h,
            deterministic=not training,
        )


class MultiScaleReadout(nn.Module):
    dim: int = DIM
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x, training: bool):
        b, t, j, d = x.shape

        # All scales are derived directly from the same T16 carrier.
        fine_seq = jnp.mean(
            x,
            axis=2,
        )  # [B,16,D]

        medium = x.reshape(
            b,
            8,
            2,
            j,
            d,
        ).mean(axis=2)
        medium_seq = jnp.mean(
            medium,
            axis=2,
        )  # [B,8,D]

        coarse = x.reshape(
            b,
            4,
            4,
            j,
            d,
        ).mean(axis=2)
        coarse_seq = jnp.mean(
            coarse,
            axis=2,
        )  # [B,4,D]

        hands = jnp.take(
            x,
            jnp.asarray(HAND_JOINTS),
            axis=2,
        ).mean(axis=2)  # [B,16,D]

        raw = (
            fine_seq.reshape(b, -1),
            medium_seq.reshape(b, -1),
            coarse_seq.reshape(b, -1),
            hands.reshape(b, -1),
        )

        descs = []
        for i, z in enumerate(raw):
            descs.append(
                ReadoutProjection(
                    64,
                    self.dropout,
                    name=f"readout_{i}",
                )(
                    z,
                    training,
                )
            )

        desc = jnp.stack(
            descs,
            axis=1,
        )

        stream_logits = []
        for i in range(4):
            stream_logits.append(
                nn.Dense(
                    NUM_CLASSES,
                    name=f"aux_classifier_{i}",
                )(desc[:, i])
            )

        stream_logits = jnp.stack(
            stream_logits,
            axis=1,
        )

        fused = jnp.concatenate(
            descs,
            axis=-1,
        )

        fused = nn.Dense(
            64,
            name="fuse",
        )(fused)
        fused = nn.LayerNorm(
            name="fuse_norm",
        )(nn.gelu(fused))
        fused = nn.Dropout(
            self.dropout,
            name="fuse_drop",
        )(
            fused,
            deterministic=not training,
        )

        main_logits = nn.Dense(
            NUM_CLASSES,
            name="main_classifier",
        )(fused)

        # Regional/scale heads are a small residual vote rather than replacing
        # the fused classifier.
        logits = (
            main_logits
            +
            0.10
            *
            jnp.mean(
                stream_logits,
                axis=1,
            )
        )

        return logits, main_logits, stream_logits, desc


class NestSARJT32PAM2T16(nn.Module):
    dim: int = DIM
    heads: int = HEADS
    blocks: int = 2
    memory_rank: int = MEM_RANK
    dropout: float = 0.10

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> Mapping[str, jnp.ndarray]:

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

        carrier = FactorizedJointEncoder(
            self.dim,
            name="joint_encoder",
        )(tok)

        block_states = []

        for i in range(self.blocks):
            carrier = JointTimeBlock(
                self.dim,
                self.heads,
                self.memory_rank,
                self.dropout,
                name=f"jt_block_{i}",
            )(
                carrier,
                training,
            )
            block_states.append(carrier)

        carrier = nn.LayerNorm(
            name="carrier_out_norm",
        )(carrier)

        (
            logits,
            main_logits,
            stream_logits,
            readout_desc,
        ) = MultiScaleReadout(
            self.dim,
            self.dropout,
            name="readout",
        )(
            carrier,
            training,
        )

        b = x.shape[0]

        return {
            "logits": logits,
            "main_logits": main_logits,
            "stream_logits": stream_logits,
            "carrier": carrier,
            "block_states": jnp.stack(block_states, axis=1),
            "readout_descriptors": readout_desc,
            # Compatibility metrics for the historical training harness.
            "sm_eta_mean": jnp.full((b,), 0.10, dtype=x.dtype),
            "sm_alpha_mean": jnp.full((b,), 0.99, dtype=x.dtype),
        }

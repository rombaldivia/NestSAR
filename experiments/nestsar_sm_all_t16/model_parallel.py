#!/usr/bin/env python3
from __future__ import annotations

"""Fully-parallel NestSAR T16 prototype.

Goals
-----
1. Remove the token-serial GRU-like GatedSweep recurrence.
2. Preserve the exact FastWeightDeltaResidual update law while evaluating it with
   an associative prefix scan instead of jax.lax.scan.
3. Vectorize the four semantic streams wherever their tensor shapes match.
4. Preserve the R4 hierarchy, controller, router, descriptor layout, head, and
   parameter budget as closely as possible.
5. Keep attention/GCN/TCN/T x T operators out of the model.

Important
---------
ParallelFastWeightDeltaResidual is algebraically equivalent to the old serial
fast-weight recurrence (up to floating-point reduction order).

ParallelAffineSweep is NOT algebraically identical to the old nonlinear
GatedSweep, because GatedSweep computes gates from h_{t-1}.  It is a
parameter-count-matched affine state-space replacement:

    h_t = a_t * h_{t-1} + b_t

where a_t,b_t depend only on x_t.  Affine transforms compose associatively, so
all prefix states are evaluated with jax.lax.associative_scan.

The per-direction parameter count of ParallelAffineSweep exactly matches the
historical GatedSweep:
    6 * D^2 + 3 * D
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


def _half_life_bias_init(half_lives):
    """Create a channel-wise forget-bias initializer from target half-lives.

    The same channel schedule is used independently in every vmapped stream.
    For h_t = a*h_(t-1)+..., a**H=0.5 gives a target half-life H.
    """
    half_lives = tuple(float(h) for h in half_lives)
    if not half_lives or any(h <= 0 for h in half_lives):
        raise ValueError(f"Invalid half-lives: {half_lives}")

    def init(key, shape, dtype=jnp.float32):
        del key
        if len(shape) != 1:
            raise ValueError(f"Forget bias must be rank-1, got {shape}")
        dim = int(shape[0])
        if dim % len(half_lives):
            raise ValueError(
                f"dim={dim} must be divisible by {len(half_lives)} timescales"
            )
        per = dim // len(half_lives)
        values = []
        for h in half_lives:
            retention = 0.5 ** (1.0 / h)
            bias = jnp.log(retention / (1.0 - retention))
            values.append(jnp.full((per,), bias, dtype=dtype))
        return jnp.concatenate(values, axis=0)

    return init


def _affine_compose(left, right):
    """Compose diagonal affine maps in temporal order.

    left : h -> a_l * h + b_l
    right: h -> a_r * h + b_r

    right(left(h)) = (a_r*a_l) * h + (a_r*b_l + b_r)
    """
    a_l, b_l = left
    a_r, b_r = right
    return a_r * a_l, a_r * b_l + b_r


class ParallelAffineSweep(nn.Module):
    """Parameter-count-matched, fully parallel replacement for GatedSweep.

    Historical GatedSweep per direction:
        z: xW + hU + b
        r: xW + hU + b
        c: xW + (r*h)U + b

    Parameters = 6 D^2 + 3 D.

    This module uses the same count:
        forget      D -> D      : D^2 + D
        candidate_1 D -> 2D     : 2D^2 + 2D
        candidate_2 2D -> D     : 2D^2
        skip        D -> D      : D^2
        --------------------------------
                                  6D^2 + 3D
    """

    dim: int
    reverse: bool = False
    parallel: bool = True
    half_lives: tuple[float, ...] | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(f"ParallelAffineSweep expects [B,T,D], got {x.shape}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected D={self.dim}, got {x.shape[-1]}")

        seq = jnp.flip(x, axis=1) if self.reverse else x

        init = nn.initializers.xavier_uniform()

        # v2 used a scalar +1.5 bias for every channel.  D128-MTS optionally
        # assigns short/medium/long half-lives across CHANNELS inside EACH
        # semantic stream, without adding parameters or operators.
        forget_bias_init = (
            nn.initializers.constant(1.5)
            if self.half_lives is None
            else _half_life_bias_init(self.half_lives)
        )
        a = jax.nn.sigmoid(
            nn.Dense(
                self.dim,
                kernel_init=init,
                bias_init=forget_bias_init,
                name="forget",
            )(seq)
        )

        hidden = nn.gelu(
            nn.Dense(
                2 * self.dim,
                kernel_init=init,
                name="candidate_in",
            )(seq)
        )
        candidate = nn.Dense(
            self.dim,
            use_bias=False,
            kernel_init=init,
            name="candidate_out",
        )(hidden)
        candidate = candidate + nn.Dense(
            self.dim,
            use_bias=False,
            kernel_init=init,
            name="skip",
        )(seq)
        candidate = jnp.tanh(candidate)

        # Convex gated update: h_t = a_t h_{t-1} + (1-a_t)c_t.
        b = (1.0 - a) * candidate

        if self.parallel:
            _, h = jax.lax.associative_scan(_affine_compose, (a, b), axis=1)
        else:
            # Numerical/performance control: identical parameters and equations.
            def step(state, ab):
                state = ab[0] * state + ab[1]
                return state, state
            _, h = jax.lax.scan(step, jnp.zeros_like(x[:, 0]),
                               (jnp.swapaxes(a, 0, 1), jnp.swapaxes(b, 0, 1)))
            h = jnp.swapaxes(h, 0, 1)

        return jnp.flip(h, axis=1) if self.reverse else h


class ParallelBiMemory(nn.Module):
    """Bidirectional parallel memory with the same topology as R4 BiMemory."""

    dim: int
    parallel: bool = True
    half_lives: tuple[float, ...] | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        f = ParallelAffineSweep(
            self.dim,
            reverse=False,
            parallel=self.parallel,
            half_lives=self.half_lives,
            name="fwd",
        )(x)
        b = ParallelAffineSweep(
            self.dim,
            reverse=True,
            parallel=self.parallel,
            half_lives=self.half_lives,
            name="bwd",
        )(x)
        y = nn.Dense(
            self.dim,
            name="merge",
        )(jnp.concatenate([f, b], axis=-1))
        return nn.LayerNorm(name="norm")(x + y)


def associative_fast_weight_reads(
    k: jnp.ndarray,
    q: jnp.ndarray,
    v: jnp.ndarray,
    eta: jnp.ndarray,
    alpha: jnp.ndarray,
    memory0: jnp.ndarray,
    *,
    parallel: bool = True,
) -> jnp.ndarray:
    """Exact associative form of the R4 fast-weight recurrence.

    Original serial update:
        pred_t  = k_t^T M_{t-1}
        err_t   = v_t - pred_t
        M_t     = alpha_t M_{t-1} + eta_t k_t err_t^T
        read_t  = q_t^T M_t

    Rearranging:
        M_t = A_t M_{t-1} + B_t

        A_t = alpha_t I - eta_t k_t k_t^T
        B_t = eta_t k_t v_t^T

    Since affine maps compose associatively:
        (A2,B2) o (A1,B1)
          = (A2 A1, A2 B1 + B2)

    the whole sequence can be evaluated with associative_scan.
    """

    if k.ndim != 3 or q.shape != k.shape:
        raise ValueError(f"k/q must be [B,T,R], got {k.shape}, {q.shape}")
    if v.ndim != 3 or v.shape[:2] != k.shape[:2]:
        raise ValueError(f"v must be [B,T,D], got {v.shape}")
    if eta.shape != k.shape[:2] + (1,) or alpha.shape != eta.shape:
        raise ValueError(
            f"eta/alpha must be [B,T,1], got {eta.shape}, {alpha.shape}"
        )

    rank = k.shape[-1]
    eye = jnp.eye(rank, dtype=k.dtype)

    kk = k[..., :, None] * k[..., None, :]
    a = alpha[..., None] * eye - eta[..., None] * kk

    b = (
        eta[..., None]
        * k[..., :, None]
        * v[..., None, :]
    )

    def compose(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return (
            jnp.matmul(a_r, a_l),
            jnp.matmul(a_r, b_l) + b_r,
        )

    if parallel:
        a_prefix, b_prefix = jax.lax.associative_scan(compose, (a, b), axis=1)
        memory = jnp.matmul(a_prefix, memory0) + b_prefix
    else:
        def step(state, ab):
            state = ab[0] @ state + ab[1]
            return state, state
        initial = jnp.broadcast_to(memory0, (k.shape[0], *memory0.shape))
        _, memory = jax.lax.scan(step, initial,
                                (jnp.swapaxes(a, 0, 1), jnp.swapaxes(b, 0, 1)))
        memory = jnp.swapaxes(memory, 0, 1)

    return jnp.einsum(
        "btr,btrd->btd",
        q,
        memory,
    )


class ParallelFastWeightDeltaResidual(nn.Module):
    """Exact fast-weight update law, evaluated with an associative scan."""

    dim: int
    rank: int = 4
    parallel: bool = True

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        eta: jnp.ndarray,
        alpha: jnp.ndarray,
    ) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(
                f"ParallelFastWeightDeltaResidual expects [B,T,D], got {x.shape}"
            )
        if eta.shape[:2] != x.shape[:2] or alpha.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"eta/alpha temporal mismatch: x={x.shape}, "
                f"eta={eta.shape}, alpha={alpha.shape}"
            )

        n = nn.LayerNorm(name="value_norm")(x)

        k = nn.Dense(
            self.rank,
            use_bias=False,
            kernel_init=nn.initializers.normal(0.02),
            name="key",
        )(n)
        q = nn.Dense(
            self.rank,
            use_bias=False,
            kernel_init=nn.initializers.normal(0.02),
            name="query",
        )(n)

        k = jnp.tanh(k)
        q = jnp.tanh(q)
        k = r4.safe_unit_normalize(k)
        q = r4.safe_unit_normalize(q)

        memory0 = self.param(
            "memory0",
            nn.initializers.normal(0.01),
            (self.rank, self.dim),
        )

        return associative_fast_weight_reads(
            k,
            q,
            n,
            eta,
            alpha,
            memory0,
            parallel=self.parallel,
        )


class ParallelSelfModBiMemory(nn.Module):
    """Parallel base memory + exact associative rank-4 fast-weight residual."""

    dim: int
    rank: int = 4
    residual_scale: float = 0.08
    parallel: bool = True
    half_lives: tuple[float, ...] | None = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        eta: jnp.ndarray,
        alpha: jnp.ndarray,
    ) -> jnp.ndarray:
        base_y = ParallelBiMemory(
            self.dim,
            parallel=self.parallel,
            half_lives=self.half_lives,
            name="base_memory",
        )(x)
        delta = ParallelFastWeightDeltaResidual(
            self.dim,
            self.rank,
            parallel=self.parallel,
            name="fast_weight",
        )(base_y, eta, alpha)
        return nn.LayerNorm(name="sm_norm")(
            base_y + self.residual_scale * delta
        )


class ParallelMaskSafeSpatialEncoder(nn.Module):
    """R4 spatial encoder with a parallel joint-memory sweep."""

    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10
    parallel: bool = True

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        valid: jnp.ndarray,
        training: bool,
    ) -> jnp.ndarray:
        b, t, m, _, _ = x.shape

        if valid.shape != x.shape[:-1]:
            raise ValueError(
                f"Spatial validity mismatch: x={x.shape}, valid={valid.shape}"
            )

        valid_f = valid[..., None].astype(x.dtype)

        h = nn.Dense(
            self.spatial_dim,
            name="in_proj",
        )(x)

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

        mem = ParallelAffineSweep(
            self.spatial_dim,
            reverse=False,
            parallel=self.parallel,
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
        h = jnp.take(h, inv, axis=3)

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
        )(y, deterministic=not training)


# Two pose streams share a vectorized call shape C=3, and the two motion
# streams share C=12.  Parameters remain independent through variable_axes=0.
VmapSpatialPair = nn.vmap(
    ParallelMaskSafeSpatialEncoder,
    variable_axes={"params": 0},
    split_rngs={"params": True, "dropout": True},
    in_axes=(2, 2, None),
    out_axes=2,
    axis_size=2,
)

VmapM4 = nn.vmap(
    ParallelSelfModBiMemory,
    variable_axes={"params": 0},
    split_rngs={"params": True},
    in_axes=(2, None, None),
    out_axes=2,
    axis_size=NUM_STREAMS,
)


class ParallelSelfModDescriptorHead(nn.Module):
    dim: int = 112
    dropout: float = 0.10
    rank: int = 4
    residual_scale: float = 0.08
    parallel: bool = True
    half_lives: tuple[float, ...] | None = None

    @nn.compact
    def __call__(
        self,
        frame_h: jnp.ndarray,
        eta_slow: jnp.ndarray,
        alpha_slow: jnp.ndarray,
        training: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if frame_h.shape[1] != FRAMES:
            raise ValueError(
                f"Expected T={FRAMES}, got {frame_h.shape}"
            )

        chunks = frame_h.reshape(
            frame_h.shape[0],
            4,
            FRAMES // 4,
            self.dim,
        ).mean(axis=2)

        chunks = ParallelSelfModBiMemory(
            self.dim,
            self.rank,
            self.residual_scale,
            parallel=self.parallel,
            half_lives=self.half_lives,
            name="chunk_memory",
        )(
            chunks,
            eta_slow,
            alpha_slow,
        )

        pooled = jnp.concatenate(
            [
                frame_h.mean(axis=1),
                chunks.mean(axis=1),
            ],
            axis=-1,
        )
        pooled = nn.Dense(
            self.dim,
            name="hier_fuse",
        )(pooled)
        pooled = nn.LayerNorm(
            name="hier_norm",
        )(nn.gelu(pooled))
        pooled = nn.Dropout(
            self.dropout,
        )(pooled, deterministic=not training)

        return chunks, pooled


VmapDescriptor = nn.vmap(
    ParallelSelfModDescriptorHead,
    variable_axes={"params": 0},
    split_rngs={"params": True, "dropout": True},
    in_axes=(2, None, None, None),
    out_axes=(1, 1),
    axis_size=NUM_STREAMS,
)


class GroupedStreamClassifier(nn.Module):
    """Four independent Dense(D->120) heads in one batched contraction."""

    streams: int = NUM_STREAMS
    dim: int = 112
    classes: int = NUM_CLASSES

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 3 or x.shape[1:] != (self.streams, self.dim):
            raise ValueError(
                f"Expected [B,{self.streams},{self.dim}], got {x.shape}"
            )

        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (self.streams, self.dim, self.classes),
        )
        bias = self.param(
            "bias",
            nn.initializers.zeros,
            (self.streams, self.classes),
        )

        return jnp.einsum(
            "bsd,sdc->bsc",
            x,
            kernel,
        ) + bias[None, :, :]


class NestSARParallelT16(nn.Module):
    """R4 hierarchy with all token recurrences converted to parallel scans."""

    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    controller_dim: int = 16
    fast_rank: int = 4
    head_rank: int = 2
    sm_residual_scale: float = 0.08
    head_residual_scale: float = 0.15
    parallel: bool = True
    m4_half_lives: tuple[float, ...] | None = None
    g4_half_lives: tuple[float, ...] | None = None

    # Audit-only M4 interpolation controls. Defaults preserve the exact
    # trained architecture and parameter tree.
    m4_mix_scale: float = 1.0
    m4_stream_scales: tuple[float, float, float, float] | None = None

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
        person_present = controller[
            "person_present"
        ].astype(bool)
        joint_valid = joint_valid.at[
            ...,
            0,
        ].set(person_present)
        valid_f = joint_valid[
            ...,
            None,
        ].astype(tok.dtype)

        gamma = controller[
            "gamma"
        ][:, :, None, None, :]
        beta = controller[
            "beta"
        ][:, :, None, None, :]

        tok = (
            tok * gamma
            + valid_f * beta
        ) * valid_f

        pose = tok[..., 0:3]
        full_disp = tok[..., 3:6]
        phase_a = tok[..., 6:9]
        phase_b = tok[..., 9:12]
        path = tok[..., 12:15]

        joint = pose
        parents = jnp.asarray(
            r4.base.PARENTS
        )
        parent_valid = jnp.take(
            joint_valid,
            parents,
            axis=3,
        )
        bone_valid = (
            joint_valid
            & parent_valid
        )
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

        # ------------------------------------------------------------------
        # Grouped/vectorized Spatial:
        #   pose pair   = Joint, Bone      (C=3)
        #   motion pair = JointM, BoneM    (C=12)
        # ------------------------------------------------------------------

        pose_pair = jnp.stack(
            [
                joint,
                bone,
            ],
            axis=2,
        )
        pose_valid = jnp.stack(
            [
                joint_valid,
                bone_valid,
            ],
            axis=2,
        )

        motion_pair = jnp.stack(
            [
                joint_motion,
                bone_motion,
            ],
            axis=2,
        )
        motion_valid = jnp.stack(
            [
                joint_valid,
                bone_valid,
            ],
            axis=2,
        )

        pose_spatial = VmapSpatialPair(
            self.spatial_dim,
            self.model_dim,
            self.dropout,
            parallel=self.parallel,
            name="spatial_pose_pair",
        )(
            pose_pair,
            pose_valid,
            training,
        )

        motion_spatial = VmapSpatialPair(
            self.spatial_dim,
            self.model_dim,
            self.dropout,
            parallel=self.parallel,
            name="spatial_motion_pair",
        )(
            motion_pair,
            motion_valid,
            training,
        )

        spatial_stack = jnp.concatenate(
            [
                pose_spatial,
                motion_spatial,
            ],
            axis=2,
        )

        spatial_stack = (
            spatial_stack
            * controller[
                "stream_gate"
            ][..., None]
        )

        # ------------------------------------------------------------------
        # M4: one vectorized call over S=4.
        # ------------------------------------------------------------------

        frame_stack = VmapM4(
            dim=self.model_dim,
            rank=self.fast_rank,
            residual_scale=self.sm_residual_scale,
            parallel=self.parallel,
            half_lives=self.m4_half_lives,
            name="frame_memory_group",
        )(
            spatial_stack,
            controller["eta"],
            controller["alpha"],
        )

        # --------------------------------------------------------------
        # AUDIT-ONLY CAUSAL INTERPOLATION
        #
        # lambda = 0 -> Spatial bypass
        # lambda = 1 -> exact trained D128-MTS
        # lambda > 1 -> strengthen M4 transformation
        #
        # Skip the arithmetic entirely at the canonical setting so exact
        # checkpoint reproduction is preserved.
        # --------------------------------------------------------------
        if self.m4_stream_scales is not None:
            if len(self.m4_stream_scales) != NUM_STREAMS:
                raise ValueError(
                    f"m4_stream_scales must have {NUM_STREAMS} values, "
                    f"got {self.m4_stream_scales}"
                )
            scales = jnp.asarray(
                self.m4_stream_scales,
                dtype=frame_stack.dtype,
            )[None, None, :, None]
            if tuple(float(v) for v in self.m4_stream_scales) != (1.0, 1.0, 1.0, 1.0):
                frame_stack = spatial_stack + scales * (
                    frame_stack - spatial_stack
                )
        elif float(self.m4_mix_scale) != 1.0:
            frame_stack = spatial_stack + float(self.m4_mix_scale) * (
                frame_stack - spatial_stack
            )

        mixed, router_weights = r4.base.CrossStreamRouter(
            self.model_dim,
            name="cross_stream_after_frame",
        )(frame_stack)

        # ------------------------------------------------------------------
        # G4: one vectorized call over S=4.
        # ------------------------------------------------------------------

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

        chunk_states, descs = VmapDescriptor(
            dim=self.model_dim,
            dropout=self.dropout,
            rank=self.fast_rank,
            residual_scale=self.sm_residual_scale,
            parallel=self.parallel,
            half_lives=self.g4_half_lives,
            name="descriptor_group",
        )(
            mixed,
            eta_slow,
            alpha_slow,
            training,
        )

        sl = GroupedStreamClassifier(
            streams=NUM_STREAMS,
            dim=self.model_dim,
            classes=NUM_CLASSES,
            name="classifier_group",
        )(descs)

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
            "spatial_stack": spatial_stack,
            "frame_stack": frame_stack,
            "mixed_frame_stack": mixed,
            "descriptors": descs,
            "chunk_states": chunk_states,
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

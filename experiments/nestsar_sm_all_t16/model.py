#!/usr/bin/env python3
from __future__ import annotations

"""NestSAR-SM-ALL-T16.

From-scratch, attention-free LocalGlobal M4/G4 model with whole-network
self-modulation and low-rank self-modifying fast weights.

Hard temporal contract:
  raw NTU clip length is arbitrary at preprocessing time;
  the neural network always receives exactly 16 temporal tokens.

The existing LocalGlobal V2 topology is preserved:
  J / B / JM / BM streams
  -> SpatialEncoder
  -> frame BiMemory (M4)
  -> post-frame CrossStreamRouter
  -> chunk BiMemory (G4)
  -> stream classifiers.

Self-modification is deliberately cheap:
  * one shared controller modulates input channels and per-stream features;
  * M4 and G4 keep the original BiMemory and receive a rank-r fast-weight
    residual updated by a learned eta/alpha delta rule;
  * fusion is clip-adaptive but zero-initialized to uniform;
  * final classifier correction is a rank-r dynamic head, evaluated once/clip.

No attention, GCN, TCN, Transformer, or T x T operation is introduced.
"""

from typing import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju

FRAMES = ju.FRAMES
PERSONS = ju.PERSONS
JOINTS = ju.JOINTS
TOKEN_CHANNELS = ju.TOKEN_CHANNELS
FEATURES = ju.FEATURES
NUM_CLASSES = ju.NUM_CLASSES
NUM_STREAMS = ju.NUM_STREAMS

if FRAMES != 16:
    raise RuntimeError(f"NestSAR-SM-ALL-T16 requires FRAMES=16, got {FRAMES}")


class SharedSMController(nn.Module):
    """Tiny controller shared by every adaptive stage."""

    controller_dim: int = 16
    head_rank: int = 2
    eta_max: float = 0.20
    alpha_min: float = 0.90
    alpha_max: float = 0.999
    input_gain: float = 0.10
    input_shift: float = 0.05
    stream_gain: float = 0.10
    fusion_scale: float = 0.15

    @nn.compact
    def __call__(self, tok: jnp.ndarray) -> Mapping[str, jnp.ndarray]:
        # tok: [B,T,M,V,C]. The controller sees one cheap per-frame summary.
        pooled = jnp.mean(tok, axis=(2, 3))  # [B,T,C]
        h = nn.Dense(self.controller_dim, name="in_proj")(pooled)
        h = nn.LayerNorm(name="norm")(nn.gelu(h))

        zero = nn.initializers.zeros

        gamma_raw = nn.Dense(
            TOKEN_CHANNELS,
            kernel_init=zero,
            bias_init=zero,
            name="gamma",
        )(h)
        beta_raw = nn.Dense(
            TOKEN_CHANNELS,
            kernel_init=zero,
            bias_init=zero,
            name="beta",
        )(h)
        stream_raw = nn.Dense(
            NUM_STREAMS,
            kernel_init=zero,
            bias_init=zero,
            name="stream_gate",
        )(h)
        lr_raw = nn.Dense(
            2,
            kernel_init=zero,
            bias_init=zero,
            name="eta_alpha",
        )(h)

        gamma = 1.0 + self.input_gain * jnp.tanh(gamma_raw)
        beta = self.input_shift * jnp.tanh(beta_raw)
        stream_gate = 1.0 + self.stream_gain * jnp.tanh(stream_raw)

        # Initial values: eta=0.10, alpha~=0.9495.
        eta = self.eta_max * jax.nn.sigmoid(lr_raw[..., 0:1])
        alpha = (
            self.alpha_min
            + (self.alpha_max - self.alpha_min)
            * jax.nn.sigmoid(lr_raw[..., 1:2])
        )

        clip_context = jnp.mean(h, axis=1)

        fusion_raw = nn.Dense(
            NUM_STREAMS,
            kernel_init=zero,
            bias_init=zero,
            name="fusion",
        )(clip_context)
        fusion_logits = self.fusion_scale * jnp.tanh(fusion_raw)

        head_coeff = jnp.tanh(
            nn.Dense(
                self.head_rank,
                kernel_init=zero,
                bias_init=zero,
                name="head_coeff",
            )(clip_context)
        )

        return {
            "context": h,
            "clip_context": clip_context,
            "gamma": gamma,
            "beta": beta,
            "stream_gate": stream_gate,
            "eta": eta,
            "alpha": alpha,
            "fusion_logits": fusion_logits,
            "head_coeff": head_coeff,
        }


class FastWeightDeltaResidual(nn.Module):
    """Low-rank self-modifying fast-weight memory.

    Memory state S_t has shape [rank, dim] per sample and changes inside the
    sequence using a learned eta/alpha delta rule:

      pred_t = k_t^T S_{t-1}
      err_t  = v_t - pred_t
      S_t    = alpha_t S_{t-1} + eta_t k_t err_t^T
      r_t    = q_t^T S_t

    S_0 is meta-learned by the outer training loop. S_t is reset to S_0 for
    every new clip. This keeps the adaptive state O(rank * dim), not O(dim^2).
    """

    dim: int
    rank: int = 2

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        eta: jnp.ndarray,
        alpha: jnp.ndarray,
    ) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(f"FastWeightDeltaResidual expects [B,T,D], got {x.shape}")
        if eta.shape[:2] != x.shape[:2] or alpha.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"eta/alpha temporal mismatch: x={x.shape}, eta={eta.shape}, alpha={alpha.shape}"
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
        k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-6)
        q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-6)

        memory0 = self.param(
            "memory0",
            nn.initializers.normal(0.01),
            (self.rank, self.dim),
        )
        memory = jnp.broadcast_to(
            memory0[None, :, :],
            (x.shape[0], self.rank, self.dim),
        )

        kt = jnp.swapaxes(k, 0, 1)
        qt = jnp.swapaxes(q, 0, 1)
        vt = jnp.swapaxes(n, 0, 1)
        et = jnp.swapaxes(eta, 0, 1)
        at = jnp.swapaxes(alpha, 0, 1)

        def step(mem, inputs):
            key_t, query_t, value_t, eta_t, alpha_t = inputs
            pred_t = jnp.einsum("br,brd->bd", key_t, mem)
            err_t = value_t - pred_t
            delta_t = jnp.einsum("br,bd->brd", key_t, err_t)
            mem = alpha_t[..., None] * mem + eta_t[..., None] * delta_t
            read_t = jnp.einsum("br,brd->bd", query_t, mem)
            return mem, read_t

        _, reads = jax.lax.scan(step, memory, (kt, qt, vt, et, at))
        return jnp.swapaxes(reads, 0, 1)


class SelfModBiMemory(nn.Module):
    """Original NestSAR BiMemory plus a cheap self-modifying residual."""

    dim: int
    rank: int = 2
    residual_scale: float = 0.08

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        eta: jnp.ndarray,
        alpha: jnp.ndarray,
    ) -> jnp.ndarray:
        base_y = base.BiMemory(self.dim, name="base_memory")(x)
        delta = FastWeightDeltaResidual(
            self.dim,
            self.rank,
            name="fast_weight",
        )(base_y, eta, alpha)
        return nn.LayerNorm(name="sm_norm")(
            base_y + self.residual_scale * delta
        )


class SelfModDescriptorHead(nn.Module):
    dim: int = 112
    dropout: float = 0.10
    rank: int = 2
    residual_scale: float = 0.08

    @nn.compact
    def __call__(
        self,
        frame_h: jnp.ndarray,
        eta_slow: jnp.ndarray,
        alpha_slow: jnp.ndarray,
        training: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if frame_h.shape[1] != FRAMES:
            raise ValueError(f"Expected T={FRAMES}, got {frame_h.shape}")

        chunks = frame_h.reshape(
            frame_h.shape[0],
            4,
            FRAMES // 4,
            self.dim,
        ).mean(axis=2)

        chunks = SelfModBiMemory(
            self.dim,
            self.rank,
            self.residual_scale,
            name="chunk_memory",
        )(chunks, eta_slow, alpha_slow)

        pooled = jnp.concatenate(
            [frame_h.mean(axis=1), chunks.mean(axis=1)],
            axis=-1,
        )
        pooled = nn.Dense(self.dim, name="hier_fuse")(pooled)
        pooled = nn.LayerNorm(name="hier_norm")(nn.gelu(pooled))
        pooled = nn.Dropout(self.dropout)(
            pooled,
            deterministic=not training,
        )
        return chunks, pooled


class NestSARSMAllT16(nn.Module):
    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    controller_dim: int = 16
    fast_rank: int = 2
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

        controller = SharedSMController(
            controller_dim=self.controller_dim,
            head_rank=self.head_rank,
            name="sm_controller",
        )(tok)

        # Whole-input self-modulation. Preserve zero/padded joints.
        valid = jnp.any(
            jnp.abs(tok[..., 0:3]) > 1e-8,
            axis=-1,
            keepdims=True,
        ).astype(tok.dtype)
        gamma = controller["gamma"][:, :, None, None, :]
        beta = controller["beta"][:, :, None, None, :]
        tok = tok * gamma + valid * beta

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

        # Spatial/local-global stage remains proven, but is dynamically gated.
        spatial = []
        for i, stream in enumerate(raw_streams):
            s = base.SpatialEncoder(
                self.spatial_dim,
                self.model_dim,
                self.dropout,
                name=f"spatial_{i}",
            )(stream, training)
            gate = controller["stream_gate"][:, :, i:i + 1]
            spatial.append(s * gate)

        # M4: original BiMemory + shared eta/alpha fast-weight residual.
        frame_streams = []
        for i, stream in enumerate(spatial):
            frame_streams.append(
                SelfModBiMemory(
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

        frame_stack = jnp.stack(frame_streams, axis=2)

        mixed, router_weights = base.CrossStreamRouter(
            self.model_dim,
            name="cross_stream_after_frame",
        )(frame_stack)

        # G4: same update law at four-times slower temporal resolution.
        eta_slow = controller["eta"].reshape(
            x.shape[0], 4, FRAMES // 4, 1
        ).mean(axis=2)
        alpha_slow = controller["alpha"].reshape(
            x.shape[0], 4, FRAMES // 4, 1
        ).mean(axis=2)

        descriptors = []
        stream_logits = []
        chunk_states = []

        for i in range(NUM_STREAMS):
            chunks, desc = SelfModDescriptorHead(
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

        descs = jnp.stack(descriptors, axis=1)
        sl = jnp.stack(stream_logits, axis=1)

        # Zero-initialized controller => exact uniform fusion at initialization.
        fusion = jax.nn.softmax(
            controller["fusion_logits"],
            axis=-1,
        )
        main_logits = jnp.einsum("bs,bsc->bc", fusion, sl)

        # Clip-level low-rank adaptive classifier correction.
        fused_desc = jnp.einsum("bs,bsd->bd", fusion, descs)
        head_u = nn.Dense(
            self.head_rank,
            use_bias=False,
            name="adaptive_head_u",
        )(fused_desc)
        dynamic_low_rank = head_u * controller["head_coeff"]
        delta_logits = nn.Dense(
            NUM_CLASSES,
            use_bias=False,
            kernel_init=nn.initializers.normal(0.01),
            name="adaptive_head_v",
        )(dynamic_low_rank)

        logits = main_logits + self.head_residual_scale * delta_logits

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
            "sm_eta_mean": jnp.mean(controller["eta"], axis=(1, 2)),
            "sm_alpha_mean": jnp.mean(controller["alpha"], axis=(1, 2)),
            "sm_head_coeff": controller["head_coeff"],
        }

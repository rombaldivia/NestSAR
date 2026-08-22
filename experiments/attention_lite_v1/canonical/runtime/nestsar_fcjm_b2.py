#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NestSAR-4L + FCJM-B2
====================

Rama espacial MLP-only con fusión residual tardía.

Propiedades:
- No modifica las coordenadas XYZ.
- Sin convoluciones.
- Sin GCN o GNN.
- Sin matriz de adyacencia.
- Sin atención ni softmax.
- Mezcla joints solamente dentro del mismo frame.
- La rama temporal NestSAR permanece sin cambios.
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
from flax import linen as nn

import nestsar as ns


FCJM_B2_ID = "nestsar_4l_fcjm_b2"
FCJM_B2_MODE = "NestSAR_4L_FCJM_B2"


class LateFusionJointMixer(nn.Module):
    """
    Rama espacial frame-wise.

    Entrada:
        [B, T, 150]

    Salida:
        [B, T, spatial_dim]

    No mezcla frames entre sí.
    """

    persons: int = 2
    joints: int = 25
    coords: int = 3

    joint_dim: int = 24
    mixer_rank: int = 8
    spatial_dim: int = 32
    dropout: float = 0.15

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool,
    ) -> jnp.ndarray:
        expected_dim = self.persons * self.joints * self.coords

        if x.ndim != 3:
            raise ValueError(
                f"FCJM-B2 esperaba [B,T,D], recibido {x.shape}"
            )

        if x.shape[-1] != expected_dim:
            raise ValueError(
                f"FCJM-B2 esperaba D={expected_dim}, "
                f"recibido D={x.shape[-1]}"
            )

        batch, time, _ = x.shape

        coordinates = x.reshape(
            batch,
            time,
            self.persons,
            self.joints,
            self.coords,
        )

        h = nn.Dense(
            self.joint_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="coordinate_embedding",
        )(coordinates)

        joint_embedding = self.param(
            "joint_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, 1, 1, self.joints, self.joint_dim),
        )

        person_embedding = self.param(
            "person_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.persons, 1, self.joint_dim),
        )

        h = h + joint_embedding + person_embedding
        h = nn.LayerNorm(name="embedding_norm")(h)

        channel_gate = nn.Dense(
            self.joint_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="channel_gate",
        )(h)

        channel_gate = jax.nn.sigmoid(channel_gate)
        conditioned = h * channel_gate

        mixed = jnp.transpose(
            conditioned,
            (0, 1, 2, 4, 3),
        )

        mixed = nn.Dense(
            self.mixer_rank,
            kernel_init=nn.initializers.xavier_uniform(),
            name="joint_reduce",
        )(mixed)

        mixed = nn.gelu(mixed)

        mixed = nn.Dense(
            self.joints,
            kernel_init=nn.initializers.xavier_uniform(),
            name="joint_expand",
        )(mixed)

        mixed = jnp.transpose(
            mixed,
            (0, 1, 2, 4, 3),
        )

        mixed = nn.Dropout(
            rate=self.dropout,
            name="joint_mixer_dropout",
        )(
            mixed,
            deterministic=not training,
        )

        h = nn.LayerNorm(
            name="spatial_output_norm"
        )(
            h + mixed
        )

        spatial = jnp.mean(
            h,
            axis=(2, 3),
        )

        spatial = nn.Dense(
            self.spatial_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="spatial_summary",
        )(spatial)

        spatial = nn.gelu(spatial)

        spatial = nn.Dropout(
            rate=self.dropout,
            name="spatial_summary_dropout",
        )(
            spatial,
            deterministic=not training,
        )

        return spatial


class NestSAR4LFCJMB2Model(nn.Module):
    mode: str
    num_classes: int
    model_dim: int
    memory_dim: int
    dropout: float
    memory_residual_scale: float
    initial_eta: float
    initial_alpha: float

    frame_blocks: int = 2
    chunk_blocks: int = 2
    clip_blocks: int = 2
    controller_blocks: int = 2

    chunk_size: int = 4
    clip_size: int = 8
    controller_rank: int = 32

    joint_dim: int = 24
    mixer_rank: int = 8
    spatial_dim: int = 32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool,
    ) -> Dict[str, jnp.ndarray]:
        if self.mode != FCJM_B2_MODE:
            raise ValueError(
                f"Modo FCJM-B2 no soportado: {self.mode}"
            )

        if x.shape[1] != 16:
            raise ValueError(
                "NestSAR-4L-FCJM-B2 requiere 16 frames; "
                f"recibido {x.shape[1]}"
            )

        pose = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="pose_projection",
        )(x)

        pose = nn.LayerNorm(
            name="pose_norm"
        )(pose)

        pose = nn.gelu(pose)

        spatial_summary = LateFusionJointMixer(
            persons=ns.CFG.persons,
            joints=ns.CFG.joints,
            coords=ns.CFG.coords,
            joint_dim=self.joint_dim,
            mixer_rank=self.mixer_rank,
            spatial_dim=self.spatial_dim,
            dropout=self.dropout,
            name="fcjm_b2",
        )(
            x,
            training,
        )

        late_fusion_input = jnp.concatenate(
            [
                pose,
                spatial_summary,
            ],
            axis=-1,
        )

        spatial_delta = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="fcjm_late_fusion",
        )(
            late_fusion_input
        )

        late_gate_logit = self.param(
            "fcjm_late_gate_logit",
            lambda key, shape: jnp.full(
                shape,
                -3.0,
                dtype=jnp.float32,
            ),
            (1,),
        )

        spatial_gate = jax.nn.sigmoid(
            late_gate_logit
        )[0]

        pose = pose + spatial_gate * spatial_delta

        zero = jnp.zeros_like(pose[:, :1])

        motion = jnp.concatenate(
            [
                zero,
                pose[:, 1:] - pose[:, :-1],
            ],
            axis=1,
        )

        motion_projected = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="motion_projection",
        )(motion)

        motion_projected = nn.gelu(
            motion_projected
        )

        motion_gate = jax.nn.sigmoid(
            self.param(
                "motion_gate_logit",
                nn.initializers.zeros,
                (1,),
            )
        )

        direct = nn.LayerNorm(
            name="direct_norm"
        )(
            pose + motion_gate * motion_projected
        )

        all_deltas = []
        all_gates = []

        l1 = direct
        l1_contexts = []

        for index in range(self.frame_blocks):
            l1, context, delta, gate = ns.MemoryBlock(
                model_dim=self.model_dim,
                memory_dim=self.memory_dim,
                dropout=self.dropout,
                residual_scale=self.memory_residual_scale,
                initial_eta=self.initial_eta,
                initial_alpha=self.initial_alpha,
                name=f"l1_frame_memory_{index + 1}",
            )(
                l1,
                training,
            )

            l1_contexts.append(context)
            all_deltas.append(delta)
            all_gates.append(gate)

        l1_context = sum(l1_contexts)

        l2 = ns.pool_tokens(
            l1,
            self.chunk_size,
        )

        l2_contexts = []
        l2_deltas = []

        for index in range(self.chunk_blocks):
            l2, context, delta, gate = ns.MemoryBlock(
                model_dim=self.model_dim,
                memory_dim=self.memory_dim,
                dropout=self.dropout,
                residual_scale=self.memory_residual_scale,
                initial_eta=self.initial_eta,
                initial_alpha=self.initial_alpha,
                name=f"l2_chunk_memory_{index + 1}",
            )(
                l2,
                training,
            )

            l2_contexts.append(context)
            l2_deltas.append(delta)
            all_gates.append(gate)

        l2_context = ns.delay_tokens(
            ns.repeat_tokens(
                sum(l2_contexts),
                self.chunk_size,
                x.shape[1],
            ),
            self.chunk_size,
        )

        all_deltas.extend(
            ns.repeat_tokens(
                delta,
                self.chunk_size,
                x.shape[1],
            )
            for delta in l2_deltas
        )

        l3 = ns.pool_tokens(
            l2,
            self.clip_size // self.chunk_size,
        )

        l3_contexts = []
        l3_deltas = []

        for index in range(self.clip_blocks):
            l3, context, delta, gate = ns.MemoryBlock(
                model_dim=self.model_dim,
                memory_dim=self.memory_dim,
                dropout=self.dropout,
                residual_scale=self.memory_residual_scale,
                initial_eta=self.initial_eta,
                initial_alpha=self.initial_alpha,
                name=f"l3_clip_memory_{index + 1}",
            )(
                l3,
                training,
            )

            l3_contexts.append(context)
            l3_deltas.append(delta)
            all_gates.append(gate)

        l3_context = ns.delay_tokens(
            ns.repeat_tokens(
                sum(l3_contexts),
                self.clip_size,
                x.shape[1],
            ),
            self.clip_size,
        )

        all_deltas.extend(
            ns.repeat_tokens(
                delta,
                self.clip_size,
                x.shape[1],
            )
            for delta in l3_deltas
        )

        multiscale = jnp.concatenate(
            [
                direct,
                l1_context,
                l2_context,
                l3_context,
            ],
            axis=-1,
        )

        fusion = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="multilevel_fusion",
        )(multiscale)

        fusion = nn.gelu(fusion)

        fusion = nn.Dropout(
            rate=self.dropout,
            name="fusion_dropout",
        )(
            fusion,
            deterministic=not training,
        )

        fusion_gate = jax.nn.sigmoid(
            self.param(
                "fusion_gate_logit",
                nn.initializers.zeros,
                (1,),
            )
        )

        features = nn.LayerNorm(
            name="fusion_norm"
        )(
            direct + fusion_gate * fusion
        )

        controller_gates = []

        for index in range(self.controller_blocks):
            features, gate = ns.SlowControllerBlock(
                model_dim=self.model_dim,
                rank=self.controller_rank,
                dropout=self.dropout,
                name=f"l4_slow_controller_{index + 1}",
            )(
                features,
                training,
            )

            controller_gates.append(gate)

        features = nn.Dropout(
            rate=self.dropout,
            name="feature_dropout",
        )(
            features,
            deterministic=not training,
        )

        prediction = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="next_motion_predictor",
        )(features)

        pooled = jnp.mean(
            features,
            axis=1,
        )

        pooled = nn.LayerNorm(
            name="classifier_norm"
        )(pooled)

        logits = nn.Dense(
            self.num_classes,
            kernel_init=nn.initializers.xavier_uniform(),
            name="classifier",
        )(pooled)

        memory_delta = (
            sum(all_deltas) / float(len(all_deltas))
        )

        memory_gate = (
            sum(all_gates) / float(len(all_gates))
        )

        controller_gate = (
            sum(controller_gates)
            / float(len(controller_gates))
        )

        return {
            "logits": logits,
            "prediction": prediction,
            "motion_target": motion,
            "memory_delta": memory_delta,
            "motion_gate": motion_gate,
            "memory_gate": memory_gate,
            "controller_gate": controller_gate,
            "fusion_gate": fusion_gate,
            "spatial_gate": spatial_gate,
            "spatial_delta": jnp.mean(
                jnp.abs(spatial_delta),
                axis=-1,
            ),
        }


_original_build_model = ns.build_model

ns.MODEL_ALIASES[FCJM_B2_ID] = FCJM_B2_MODE


def build_model(model_id: str) -> nn.Module:
    if model_id != FCJM_B2_ID:
        return _original_build_model(model_id)

    return NestSAR4LFCJMB2Model(
        mode=FCJM_B2_MODE,
        num_classes=ns.CFG.num_classes,
        model_dim=ns.CFG.model_dim,
        memory_dim=ns.CFG.memory_dim,
        dropout=ns.CFG.dropout,
        memory_residual_scale=ns.CFG.memory_residual_scale,
        initial_eta=ns.CFG.initial_eta,
        initial_alpha=ns.CFG.initial_alpha,
        frame_blocks=ns.CFG.frame_blocks,
        chunk_blocks=ns.CFG.chunk_blocks,
        clip_blocks=ns.CFG.clip_blocks,
        controller_blocks=ns.CFG.controller_blocks,
        chunk_size=ns.CFG.chunk_size,
        clip_size=ns.CFG.clip_size,
        controller_rank=ns.CFG.controller_rank,
        joint_dim=24,
        mixer_rank=8,
        spatial_dim=32,
    )


ns.build_model = build_model
ns.__file__ = __file__


if __name__ == "__main__":
    raise SystemExit(ns.main())

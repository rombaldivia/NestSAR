#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NestSAR Self-Weight Clean Ablations
===================================

S1C:
    Dynamic Write Clean
    - beta_t dinámico por token
    - proyección sin bias
    - modulación centrada temporalmente
    - alpha fijo por bloque

S2C:
    Dynamic Write + Dynamic Decay Clean
    - beta_t dinámico limpio
    - alpha_t dinámico limpio
    - ambas modulaciones centradas temporalmente

Se conserva:
- FCJM-B2 con fusión tardía.
- Read-before-write.
- Memoria reiniciada por secuencia.
- Una escritura rank-1 por token.
- Sin Conv, GCN, GNN ni atención softmax.
"""

from __future__ import annotations

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

import nestsar as ns
import nestsar_fcjm_b2 as b2

MODEL_VARIANTS = {
    "nestsar_b2_s1c": "s1c",
    "nestsar_b2_s2c": "s2c",
}

MODEL_MODES = {
    "nestsar_b2_s1c":
        "NestSAR_B2_S1C_DynamicWriteClean",

    "nestsar_b2_s2c":
        "NestSAR_B2_S2C_DynamicWriteDecayClean",
}


def temporal_center(
    values: jnp.ndarray,
) -> jnp.ndarray:
    return (
        values
        - jnp.mean(
            values,
            axis=1,
            keepdims=True,
        )
    )


def clean_associative_scan(
    keys: jnp.ndarray,
    queries: jnp.ndarray,
    values: jnp.ndarray,
    beta: jnp.ndarray,
    alpha_rows: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    scan_inputs = (
        jnp.swapaxes(keys, 0, 1),
        jnp.swapaxes(queries, 0, 1),
        jnp.swapaxes(values, 0, 1),
        jnp.swapaxes(beta, 0, 1),
        jnp.swapaxes(alpha_rows, 0, 1),
    )

    batch_size = keys.shape[0]
    memory_dim = keys.shape[-1]

    initial_memory = jnp.zeros(
        (
            batch_size,
            memory_dim,
            memory_dim,
        ),
        dtype=keys.dtype,
    )

    def step(memory, inputs):
        (
            key_t,
            query_t,
            value_t,
            beta_t,
            alpha_rows_t,
        ) = inputs

        read_t = jnp.einsum(
            "bij,bj->bi",
            memory,
            query_t,
        )

        reconstructed_t = jnp.einsum(
            "bij,bj->bi",
            memory,
            key_t,
        )

        error_t = reconstructed_t - value_t

        delta_update = jnp.einsum(
            "bi,bj->bij",
            error_t,
            key_t,
        )

        decayed_memory = (
            alpha_rows_t[:, :, None]
            * memory
        )

        new_memory = (
            decayed_memory
            - beta_t[:, None, None]
            * delta_update
        )

        memory_delta_t = jnp.sqrt(
            jnp.mean(
                jnp.square(
                    new_memory - memory
                ),
                axis=(1, 2),
            )
            + 1e-12
        )

        return new_memory, (
            read_t,
            memory_delta_t,
        )

    _, (reads_t, deltas_t) = jax.lax.scan(
        step,
        initial_memory,
        scan_inputs,
    )

    return (
        jnp.swapaxes(reads_t, 0, 1),
        jnp.swapaxes(deltas_t, 0, 1),
    )


class CleanAdaptiveMemoryBlock(nn.Module):
    variant: str

    model_dim: int
    memory_dim: int
    dropout: float
    residual_scale: float
    initial_eta: float
    initial_alpha: float

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool,
    ):
        if self.variant not in {"s1c", "s2c"}:
            raise ValueError(
                f"Variante no válida: {self.variant}"
            )

        h = nn.LayerNorm(
            name="input_norm"
        )(x)

        keys = nn.Dense(
            self.memory_dim,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            name="key",
        )(h)

        queries = nn.Dense(
            self.memory_dim,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            name="query",
        )(h)

        values = nn.Dense(
            self.memory_dim,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            name="value",
        )(h)

        keys = ns.four_normalize_vectors(keys)
        queries = ns.four_normalize_vectors(queries)

        eta_logit = self.param(
            "eta_logit",
            lambda key, shape: jnp.full(
                shape,
                ns.four_logit(
                    self.initial_eta
                ),
            ),
            (1,),
        )

        alpha_logit = self.param(
            "alpha_logit",
            lambda key, shape: jnp.full(
                shape,
                ns.four_logit(
                    self.initial_alpha
                ),
            ),
            (1,),
        )

        memory_gate_logit = self.param(
            "memory_gate_logit",
            nn.initializers.zeros,
            (1,),
        )

        memory_gate = jax.nn.sigmoid(
            memory_gate_logit
        )

        raw_beta_delta = nn.Dense(
            1,
            use_bias=False,
            kernel_init=nn.initializers.zeros,
            name="beta_delta",
        )(h)[..., 0]

        beta_delta = temporal_center(
            raw_beta_delta
        )

        beta = jax.nn.sigmoid(
            eta_logit[0]
            + beta_delta
        )

        batch_size, time_steps, _ = h.shape

        if self.variant == "s1c":
            alpha_scalar = jax.nn.sigmoid(
                alpha_logit[0]
            )

            alpha_rows = jnp.broadcast_to(
                alpha_scalar,
                (
                    batch_size,
                    time_steps,
                    self.memory_dim,
                ),
            )

        else:
            raw_alpha_delta = nn.Dense(
                1,
                use_bias=False,
                kernel_init=nn.initializers.zeros,
                name="alpha_delta",
            )(h)[..., 0]

            alpha_delta = temporal_center(
                raw_alpha_delta
            )

            alpha_scalar = jax.nn.sigmoid(
                alpha_logit[0]
                + alpha_delta
            )

            alpha_rows = jnp.broadcast_to(
                alpha_scalar[..., None],
                (
                    batch_size,
                    time_steps,
                    self.memory_dim,
                ),
            )

        self.sow(
            "intermediates",
            "beta",
            beta,
        )

        self.sow(
            "intermediates",
            "beta_delta",
            beta_delta,
        )

        self.sow(
            "intermediates",
            "alpha_mean",
            jnp.mean(
                alpha_rows,
                axis=-1,
            ),
        )

        reads, memory_delta = clean_associative_scan(
            keys,
            queries,
            values,
            beta,
            alpha_rows,
        )

        context = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="readout",
        )(reads)

        context = nn.Dropout(
            rate=self.dropout,
            name="context_dropout",
        )(
            context,
            deterministic=not training,
        )

        scaled_context = (
            self.residual_scale
            * memory_gate
            * context
        )

        x = nn.LayerNorm(
            name="memory_residual_norm"
        )(
            x + scaled_context
        )

        ff = nn.LayerNorm(
            name="ff_norm"
        )(x)

        ff = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="ff_in",
        )(ff)

        ff = nn.gelu(ff)

        ff = nn.Dropout(
            rate=self.dropout,
            name="ff_dropout",
        )(
            ff,
            deterministic=not training,
        )

        ff = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.zeros,
            name="ff_out",
        )(ff)

        ff_gate = jax.nn.sigmoid(
            self.param(
                "ff_gate_logit",
                nn.initializers.zeros,
                (1,),
            )
        )

        output = nn.LayerNorm(
            name="output_norm"
        )(
            x + ff_gate * ff
        )

        return (
            output,
            scaled_context,
            memory_delta,
            memory_gate,
        )


_ORIGINAL_MEMORY_BLOCK = ns.MemoryBlock
_PREVIOUS_BUILD_MODEL = ns.build_model

for model_id, mode in MODEL_MODES.items():
    ns.MODEL_ALIASES[model_id] = mode


def build_model(
    model_id: str,
) -> nn.Module:
    if model_id not in MODEL_VARIANTS:
        ns.MemoryBlock = _ORIGINAL_MEMORY_BLOCK
        return _PREVIOUS_BUILD_MODEL(
            model_id
        )

    variant = MODEL_VARIANTS[model_id]
    mode = MODEL_MODES[model_id]

    ns.MemoryBlock = functools.partial(
        CleanAdaptiveMemoryBlock,
        variant=variant,
    )

    b2.FCJM_B2_MODE = mode

    return b2.NestSAR4LFCJMB2Model(
        mode=mode,
        num_classes=ns.CFG.num_classes,
        model_dim=ns.CFG.model_dim,
        memory_dim=ns.CFG.memory_dim,
        dropout=ns.CFG.dropout,
        memory_residual_scale=(
            ns.CFG.memory_residual_scale
        ),
        initial_eta=ns.CFG.initial_eta,
        initial_alpha=ns.CFG.initial_alpha,
        frame_blocks=ns.CFG.frame_blocks,
        chunk_blocks=ns.CFG.chunk_blocks,
        clip_blocks=ns.CFG.clip_blocks,
        controller_blocks=(
            ns.CFG.controller_blocks
        ),
        chunk_size=ns.CFG.chunk_size,
        clip_size=ns.CFG.clip_size,
        controller_rank=(
            ns.CFG.controller_rank
        ),
        joint_dim=24,
        mixer_rank=8,
        spatial_dim=32,
    )


ns.build_model = build_model
ns.__file__ = __file__


if __name__ == "__main__":
    raise SystemExit(ns.main())

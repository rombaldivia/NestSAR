#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HOPE-SAR Nested V1
==================

Primera reconstrucción limpia:

H0_direct_motion
    Pose stem + movimiento directo + mean pooling.

H2_predictive_memory
    Mismo backbone que H0 +
    memoria asociativa causal +
    predicción auxiliar del siguiente movimiento.

Propiedades:
- Sin convolución.
- Read-before-write.
- Estado rápido independiente por muestra.
- Estado reiniciado en cada secuencia.
- Memoria activa únicamente dentro de los 16 frames.
- Comparación H0/H2 bajo protocolo idéntico.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import hashlib
import math
import os
import pickle
import random
import sys
import time
import subprocess
import threading
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Debe definirse antes de importar JAX cuando el script se lanza por GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from flax import serialization
    from flax.training import train_state
    import optax
except ImportError as exc:
    raise RuntimeError(
        "Faltan JAX/Flax/Optax. En Kaggle suelen estar preinstalados. "
        "Activa un acelerador GPU y reinicia la sesión."
    ) from exc


# ============================================================
# CONFIGURACIÓN
# ============================================================

@dataclasses.dataclass(frozen=True)
class Config:
    frames: int = 16
    persons: int = 2
    joints: int = 25
    coords: int = 3
    num_classes: int = 120

    model_dim: int = 128
    memory_dim: int = 64
    dropout: float = 0.15

    batch_size: int = 128
    eval_batch_size: int = 256
    epochs: int = 150
    patience: int = 40

    learning_rate: float = 2.0e-4
    weight_decay: float = 3.0e-2
    warmup_fraction: float = 0.10
    label_smoothing: float = 0.05
    grad_clip: float = 1.0

    memory_residual_scale: float = 0.25
    initial_eta: float = 0.10
    initial_alpha: float = 0.95
    predictive_loss_weight: float = 0.10

    adapter_rank: int = 8
    nested_qkv_scale: float = 0.25

    frame_blocks: int = 2
    chunk_blocks: int = 2
    clip_blocks: int = 2
    controller_blocks: int = 2
    chunk_size: int = 4
    clip_size: int = 8
    controller_rank: int = 32

    max_train_samples: int = 0
    max_val_samples: int = 0

    seed: int = 128
    log_every_batches: int = 20


CFG = Config()


# ============================================================
# UTILIDADES
# ============================================================

def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def tree_all_finite(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return all(bool(np.asarray(jnp.all(jnp.isfinite(x)))) for x in leaves)


def count_parameters(params: Any) -> int:
    return int(
        sum(np.prod(np.asarray(x).shape) for x in jax.tree_util.tree_leaves(params))
    )


def find_dataset(explicit_path: Optional[str]) -> Path:
    candidates: List[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    common_names = (
        "ntu120_3danno.pkl",
        "ntu120_3danno_clean.pkl",
        "ntu120.pkl",
    )

    roots = (
        Path("/kaggle/input"),
        Path("/kaggle/working"),
    )

    for root in roots:
        if not root.exists():
            continue

        for name in common_names:
            candidates.extend(root.rglob(name))

    checked: List[str] = []

    for path in candidates:
        checked.append(str(path))
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        "No se encontró ntu120_3danno.pkl.\n"
        "Rutas revisadas:\n- " + "\n- ".join(checked[:100])
    )


def load_pickle(path: Path) -> Any:
    log(f"Cargando dataset: {path}")

    with path.open("rb") as file:
        try:
            return pickle.load(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pickle.load(file, encoding="latin1")


# ============================================================
# LECTURA ROBUSTA DEL PKL NTU
# ============================================================

def extract_annotations_and_split(data: Any) -> Tuple[List[Mapping[str, Any]], Mapping[str, Any]]:
    if not isinstance(data, Mapping):
        raise TypeError(
            f"Se esperaba un diccionario en el PKL; recibido: {type(data)!r}"
        )

    annotation_keys = (
        "annotations",
        "annotation",
        "samples",
        "data_list",
    )

    split_keys = (
        "split",
        "splits",
    )

    annotations = None
    split_map = None

    for key in annotation_keys:
        value = data.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            annotations = list(value)
            break

    for key in split_keys:
        value = data.get(key)
        if isinstance(value, Mapping):
            split_map = value
            break

    if annotations is None:
        # Algunos archivos contienen directamente frame_dir -> anotación.
        possible = []
        for value in data.values():
            if isinstance(value, Mapping) and (
                "keypoint" in value or "skeleton" in value or "data" in value
            ):
                possible.append(value)

        if possible:
            annotations = possible

    if annotations is None:
        raise KeyError(
            "No se encontró una lista de anotaciones. "
            f"Claves disponibles: {list(data.keys())[:50]}"
        )

    if split_map is None:
        raise KeyError(
            "No se encontró el mapa 'split' dentro del PKL. "
            f"Claves disponibles: {list(data.keys())[:50]}"
        )

    return annotations, split_map


def sample_identifier(annotation: Mapping[str, Any], index: int) -> str:
    for key in (
        "frame_dir",
        "filename",
        "sample_name",
        "name",
        "id",
        "video_id",
    ):
        value = annotation.get(key)
        if value is not None:
            return str(value)

    return str(index)


def annotation_label(annotation: Mapping[str, Any]) -> int:
    for key in ("label", "action_label", "class", "target"):
        if key in annotation:
            label = int(annotation[key])
            if not 0 <= label < CFG.num_classes:
                raise ValueError(f"Etiqueta fuera de rango: {label}")
            return label

    raise KeyError(
        f"No se encontró etiqueta en la anotación. Claves: {list(annotation.keys())}"
    )


def annotation_keypoints(annotation: Mapping[str, Any]) -> np.ndarray:
    for key in (
        "keypoint",
        "keypoints",
        "skeleton",
        "skeletons",
        "data",
    ):
        if key in annotation:
            array = np.asarray(annotation[key], dtype=np.float32)
            return array

    raise KeyError(
        "No se encontraron keypoints. "
        f"Claves de la anotación: {list(annotation.keys())}"
    )


def resolve_split_keys(split_map: Mapping[str, Any], protocol: str) -> Tuple[str, str]:
    keys = list(split_map.keys())
    lower_to_original = {str(key).lower(): str(key) for key in keys}

    train_candidates = [
        f"{protocol}_train",
        f"{protocol}train",
        f"{protocol}-train",
        f"train_{protocol}",
    ]

    val_candidates = [
        f"{protocol}_val",
        f"{protocol}_test",
        f"{protocol}val",
        f"{protocol}test",
        f"{protocol}-val",
        f"{protocol}-test",
        f"val_{protocol}",
        f"test_{protocol}",
    ]

    train_key = next(
        (
            lower_to_original[candidate]
            for candidate in train_candidates
            if candidate in lower_to_original
        ),
        None,
    )

    val_key = next(
        (
            lower_to_original[candidate]
            for candidate in val_candidates
            if candidate in lower_to_original
        ),
        None,
    )

    if train_key is None or val_key is None:
        raise KeyError(
            f"No pude resolver los splits de {protocol}.\n"
            f"Claves disponibles: {keys}"
        )

    return train_key, val_key


def normalize_split_members(members: Any) -> List[str]:
    if isinstance(members, np.ndarray):
        members = members.tolist()

    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
        raise TypeError(f"Formato de split inesperado: {type(members)!r}")

    return [str(value) for value in members]


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def to_tmvc(keypoints: np.ndarray) -> np.ndarray:
    """
    Convierte formatos comunes a [T, M, V, C].

    Formatos admitidos principalmente:
    - [M, T, V, C]
    - [T, M, V, C]
    - [T, V, C]
    """

    x = np.asarray(keypoints, dtype=np.float32)

    if x.ndim == 3:
        # [T, V, C] -> [T, 1, V, C]
        if x.shape[-1] not in (2, 3):
            raise ValueError(f"Formato 3D no reconocido: {x.shape}")
        x = x[:, None, :, :]

    elif x.ndim == 4:
        if x.shape[-1] not in (2, 3):
            raise ValueError(f"Última dimensión inválida: {x.shape}")

        # MMACTION usa normalmente [M, T, V, C].
        if x.shape[0] <= 4 and x.shape[1] > 4:
            x = np.transpose(x, (1, 0, 2, 3))

        # Si shape[1] parece representar personas, ya es [T, M, V, C].
        elif x.shape[1] <= 4 and x.shape[0] > 4:
            pass

        else:
            # Heurística conservadora.
            if x.shape[0] < x.shape[1]:
                x = np.transpose(x, (1, 0, 2, 3))
    else:
        raise ValueError(f"Número de dimensiones no admitido: {x.shape}")

    return x


def temporal_indices(total_frames: int, target_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros((target_frames,), dtype=np.int64)

    if total_frames == 1:
        return np.zeros((target_frames,), dtype=np.int64)

    return np.linspace(
        0,
        total_frames - 1,
        num=target_frames,
        dtype=np.float32,
    ).round().astype(np.int64)


def preprocess_keypoints(keypoints: np.ndarray) -> np.ndarray:
    x = to_tmvc(keypoints)

    # Coordenadas 2D -> añade Z=0.
    if x.shape[-1] == 2:
        zeros = np.zeros((*x.shape[:-1], 1), dtype=np.float32)
        x = np.concatenate([x, zeros], axis=-1)

    # Ajusta número de joints.
    if x.shape[2] < CFG.joints:
        pad = np.zeros(
            (
                x.shape[0],
                x.shape[1],
                CFG.joints - x.shape[2],
                CFG.coords,
            ),
            dtype=np.float32,
        )
        x = np.concatenate([x, pad], axis=2)
    else:
        x = x[:, :, : CFG.joints, : CFG.coords]

    # Ordena personas por energía de movimiento.
    person_energy = np.sum(np.abs(x), axis=(0, 2, 3))
    order = np.argsort(-person_energy)
    x = x[:, order]

    if x.shape[1] < CFG.persons:
        pad = np.zeros(
            (
                x.shape[0],
                CFG.persons - x.shape[1],
                CFG.joints,
                CFG.coords,
            ),
            dtype=np.float32,
        )
        x = np.concatenate([x, pad], axis=1)
    else:
        x = x[:, : CFG.persons]

    # Muestreo uniforme a 16 frames.
    indices = temporal_indices(x.shape[0], CFG.frames)
    x = x[indices]

    # Centro corporal: joint 0 de la primera persona.
    center = x[:, 0:1, 0:1, :]
    valid_center = np.any(np.abs(center) > 1e-8, axis=-1, keepdims=True)
    centered = x - center
    x = np.where(valid_center, centered, x)

    # Preserva joints totalmente ausentes como cero.
    valid_joint = np.any(np.abs(x) > 1e-8, axis=-1, keepdims=True)
    x = np.where(valid_joint, x, 0.0)

    # Normalización RMS por secuencia.
    nonzero = np.abs(x) > 1e-8
    if np.any(nonzero):
        rms = float(np.sqrt(np.mean(np.square(x[nonzero]))) + 1e-6)
        x = x / rms

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.reshape(CFG.frames, -1).astype(np.float32)


def build_samples(
    data: Any,
    protocol: str,
    max_train: int,
    max_val: int,
    seed: int,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    annotations, split_map = extract_annotations_and_split(data)
    train_key, val_key = resolve_split_keys(split_map, protocol)

    train_ids = set(normalize_split_members(split_map[train_key]))
    val_ids = set(normalize_split_members(split_map[val_key]))

    indexed: Dict[str, Mapping[str, Any]] = {}

    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            continue
        indexed[sample_identifier(annotation, index)] = annotation

    train = [indexed[name] for name in train_ids if name in indexed]
    val = [indexed[name] for name in val_ids if name in indexed]

    missing_train = len(train_ids) - len(train)
    missing_val = len(val_ids) - len(val)

    if missing_train or missing_val:
        log(
            f"Advertencia: omitidos por falta de anotación: "
            f"train={missing_train}, val={missing_val}"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(train)

    # Validación en orden fijo para comparabilidad.
    val = sorted(
        val,
        key=lambda annotation: sample_identifier(annotation, 0),
    )

    if max_train > 0:
        train = train[:max_train]

    if max_val > 0:
        val = val[:max_val]

    if not train or not val:
        raise RuntimeError(
            f"Split vacío. train={len(train)}, val={len(val)}, "
            f"keys=({train_key}, {val_key})"
        )

    log(
        f"{protocol.upper()} | split keys: {train_key} / {val_key} | "
        f"train={len(train)} | val={len(val)}"
    )

    return train, val


class SkeletonDataset:
    def __init__(self, samples: Sequence[Mapping[str, Any]]) -> None:
        self.samples = list(samples)
        self.cache: Dict[int, Tuple[np.ndarray, np.int32]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.int32]:
        cached = self.cache.get(index)
        if cached is not None:
            return cached

        annotation = self.samples[index]
        x = preprocess_keypoints(annotation_keypoints(annotation))
        y = np.int32(annotation_label(annotation))

        self.cache[index] = (x, y)
        return x, y


def batch_iterator(
    dataset: SkeletonDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(dataset))

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        subset = indices[start : start + batch_size]

        if drop_last and len(subset) < batch_size:
            continue

        xs: List[np.ndarray] = []
        ys: List[np.int32] = []

        for index in subset:
            x, y = dataset[int(index)]
            xs.append(x)
            ys.append(y)

        yield (
            np.stack(xs).astype(np.float32),
            np.asarray(ys, dtype=np.int32),
        )


# ============================================================
# MODELO
# ============================================================

def stable_l2_normalize(
    x: jnp.ndarray,
    epsilon: float = 1e-12,
) -> jnp.ndarray:
    """L2 normalization with finite gradients for zero vectors."""
    squared_norm = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(squared_norm + epsilon)


class HOPEModel(nn.Module):
    mode: str
    num_classes: int
    model_dim: int
    memory_dim: int
    dropout: float
    memory_residual_scale: float
    initial_eta: float
    initial_alpha: float

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool,
    ) -> Dict[str, jnp.ndarray]:
        """
        x: [B, T, 150]
        """

        if self.mode not in (
            "H0_direct_motion",
            "H2_predictive_memory",
        ):
            raise ValueError(f"Modo no soportado: {self.mode}")

        # Stem idéntico para H0 y H2.
        pose = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="pose_projection",
        )(x)
        pose = nn.LayerNorm(name="pose_norm")(pose)
        pose = nn.gelu(pose)

        # Movimiento latente directo.
        zero = jnp.zeros_like(pose[:, :1])
        motion = jnp.concatenate(
            [zero, pose[:, 1:] - pose[:, :-1]],
            axis=1,
        )

        motion_projected = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="motion_projection",
        )(motion)
        motion_projected = nn.gelu(motion_projected)

        motion_gate = self.param(
            "motion_gate_logit",
            nn.initializers.zeros,
            (1,),
        )
        motion_gate = jax.nn.sigmoid(motion_gate)

        direct_features = pose + motion_gate * motion_projected
        direct_features = nn.LayerNorm(name="direct_norm")(direct_features)

        prediction = jnp.zeros_like(motion)
        memory_delta = jnp.zeros((x.shape[0], x.shape[1]), dtype=x.dtype)
        memory_gate_value = jnp.zeros((1,), dtype=x.dtype)

        if self.mode == "H0_direct_motion":
            features = direct_features

        else:
            # Proyecciones lentas. La memoria rápida no es un parámetro
            # persistente: se crea y reinicia para cada secuencia.
            keys = nn.Dense(
                self.memory_dim,
                use_bias=False,
                kernel_init=nn.initializers.xavier_uniform(),
                name="memory_key",
            )(direct_features)

            queries = nn.Dense(
                self.memory_dim,
                use_bias=False,
                kernel_init=nn.initializers.xavier_uniform(),
                name="memory_query",
            )(direct_features)

            values = nn.Dense(
                self.memory_dim,
                use_bias=False,
                kernel_init=nn.initializers.xavier_uniform(),
                name="memory_value",
            )(direct_features)

            keys = stable_l2_normalize(keys)
            queries = stable_l2_normalize(queries)

            eta_logit = self.param(
                "eta_logit",
                lambda key, shape: jnp.full(
                    shape,
                    jnp.log(
                        self.initial_eta /
                        max(1.0 - self.initial_eta, 1e-6)
                    ),
                ),
                (1,),
            )

            alpha_logit = self.param(
                "alpha_logit",
                lambda key, shape: jnp.full(
                    shape,
                    jnp.log(
                        self.initial_alpha /
                        max(1.0 - self.initial_alpha, 1e-6)
                    ),
                ),
                (1,),
            )

            memory_gate_logit = self.param(
                "memory_gate_logit",
                nn.initializers.zeros,
                (1,),
            )

            eta = jax.nn.sigmoid(eta_logit)[0]
            alpha = jax.nn.sigmoid(alpha_logit)[0]
            memory_gate_value = jax.nn.sigmoid(memory_gate_logit)

            # [B,T,D] -> [T,B,D] para lax.scan.
            scan_inputs = (
                jnp.swapaxes(keys, 0, 1),
                jnp.swapaxes(queries, 0, 1),
                jnp.swapaxes(values, 0, 1),
            )

            batch_size = x.shape[0]
            initial_memory = jnp.zeros(
                (
                    batch_size,
                    self.memory_dim,
                    self.memory_dim,
                ),
                dtype=x.dtype,
            )

            def memory_step(
                memory: jnp.ndarray,
                inputs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
                key_t, query_t, value_t = inputs

                # --------------------------------------------------------
                # READ BEFORE WRITE
                # --------------------------------------------------------
                read_t = jnp.einsum(
                    "bij,bj->bi",
                    memory,
                    query_t,
                )

                # Predicción causada únicamente por la memoria previa.
                predicted_t = read_t

                # Error asociativo actual.
                reconstructed_t = jnp.einsum(
                    "bij,bj->bi",
                    memory,
                    key_t,
                )
                error_t = reconstructed_t - value_t

                # Delta rule:
                # M_t = alpha*M_(t-1) - eta*(M*k-v)k^T
                update = jnp.einsum(
                    "bi,bj->bij",
                    error_t,
                    key_t,
                )

                new_memory = alpha * memory - eta * update

                delta = jnp.sqrt(
                    jnp.mean(jnp.square(new_memory - memory), axis=(1, 2))
                    + 1e-12
                )

                return new_memory, (predicted_t, delta)

            final_memory, (memory_reads, memory_delta_t) = jax.lax.scan(
                memory_step,
                initial_memory,
                scan_inputs,
            )

            del final_memory

            # [T,B,D] -> [B,T,D]
            memory_reads = jnp.swapaxes(memory_reads, 0, 1)
            memory_delta = jnp.swapaxes(memory_delta_t, 0, 1)

            memory_features = nn.Dense(
                self.model_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                name="memory_readout",
            )(memory_reads)

            prediction = nn.Dense(
                self.model_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                name="next_motion_predictor",
            )(memory_reads)

            features = (
                direct_features
                + self.memory_residual_scale
                * memory_gate_value
                * memory_features
            )
            features = nn.LayerNorm(name="memory_fusion_norm")(features)

        features = nn.Dropout(
            rate=self.dropout,
            name="feature_dropout",
        )(features, deterministic=not training)

        pooled = jnp.mean(features, axis=1)

        logits = nn.Dense(
            self.num_classes,
            kernel_init=nn.initializers.xavier_uniform(),
            name="classifier",
        )(pooled)

        return {
            "logits": logits,
            "prediction": prediction,
            "motion_target": motion,
            "memory_delta": memory_delta,
            "motion_gate": motion_gate,
            "memory_gate": memory_gate_value,
        }




# ============================================================
# MODELO H3
# ============================================================

H3_MODE = "H3_nestsar"
H3_ADAPTER_RANK = 8
H3_NESTED_QKV_SCALE = 0.25

def h3_logit(value: float) -> float:
    clipped = min(max(float(value), 1e-5), 1.0 - 1e-5)
    return float(np.log(clipped / (1.0 - clipped)))


def h3_associative_scan(
    keys: jnp.ndarray,
    queries: jnp.ndarray,
    values: jnp.ndarray,
    *,
    eta: jnp.ndarray,
    alpha: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Memoria asociativa causal por secuencia.

    En cada frame se lee M_(t-1) con q_t y después se escribe k_t -> v_t.
    La memoria rápida no es parámetro persistente y se reinicia por muestra.
    """

    scan_inputs = (
        jnp.swapaxes(keys, 0, 1),
        jnp.swapaxes(queries, 0, 1),
        jnp.swapaxes(values, 0, 1),
    )

    batch_size = keys.shape[0]
    memory_dim = keys.shape[-1]

    initial_memory = jnp.zeros(
        (batch_size, memory_dim, memory_dim),
        dtype=keys.dtype,
    )

    def memory_step(memory, inputs):
        key_t, query_t, value_t = inputs

        # Read-before-write: la lectura nunca contiene el frame actual.
        read_t = jnp.einsum("bij,bj->bi", memory, query_t)

        reconstructed_t = jnp.einsum("bij,bj->bi", memory, key_t)
        error_t = reconstructed_t - value_t
        update_t = jnp.einsum("bi,bj->bij", error_t, key_t)

        new_memory = alpha * memory - eta * update_t

        delta_t = jnp.sqrt(
            jnp.mean(jnp.square(new_memory - memory), axis=(1, 2))
            + 1e-12
        )

        return new_memory, (read_t, delta_t)

    _, (reads_t, deltas_t) = jax.lax.scan(
        memory_step,
        initial_memory,
        scan_inputs,
    )

    return (
        jnp.swapaxes(reads_t, 0, 1),
        jnp.swapaxes(deltas_t, 0, 1),
    )


class NestSARH3Model(nn.Module):
    mode: str
    num_classes: int
    model_dim: int
    memory_dim: int
    dropout: float
    memory_residual_scale: float
    initial_eta: float
    initial_alpha: float
    adapter_rank: int = H3_ADAPTER_RANK
    nested_qkv_scale: float = H3_NESTED_QKV_SCALE

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool,
    ) -> Dict[str, jnp.ndarray]:
        """x: [B, T, 150]."""

        if self.mode != H3_MODE:
            raise ValueError(f"Modo no soportado: {self.mode}")

        # --------------------------------------------------------
        # 1. Backbone idéntico a H0/H2
        # --------------------------------------------------------
        pose = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="pose_projection",
        )(x)
        pose = nn.LayerNorm(name="pose_norm")(pose)
        pose = nn.gelu(pose)

        zero = jnp.zeros_like(pose[:, :1])
        motion = jnp.concatenate(
            [zero, pose[:, 1:] - pose[:, :-1]],
            axis=1,
        )

        motion_projected = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="motion_projection",
        )(motion)
        motion_projected = nn.gelu(motion_projected)

        motion_gate_logit = self.param(
            "motion_gate_logit",
            nn.initializers.zeros,
            (1,),
        )
        motion_gate = jax.nn.sigmoid(motion_gate_logit)

        direct_features = pose + motion_gate * motion_projected
        direct_features = nn.LayerNorm(name="direct_norm")(direct_features)

        # --------------------------------------------------------
        # 2. Nivel interno 1: memoria predictiva H2
        # --------------------------------------------------------
        keys_1 = nn.Dense(
            self.memory_dim,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            name="memory_key",
        )(direct_features)
        queries_1 = nn.Dense(
            self.memory_dim,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            name="memory_query",
        )(direct_features)
        values_1 = nn.Dense(
            self.memory_dim,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            name="memory_value",
        )(direct_features)

        keys_1 = stable_l2_normalize(keys_1)
        queries_1 = stable_l2_normalize(queries_1)

        eta_1_logit = self.param(
            "eta_logit",
            lambda key, shape: jnp.full(
                shape,
                h3_logit(self.initial_eta),
            ),
            (1,),
        )
        alpha_1_logit = self.param(
            "alpha_logit",
            lambda key, shape: jnp.full(
                shape,
                h3_logit(self.initial_alpha),
            ),
            (1,),
        )
        memory_gate_logit = self.param(
            "memory_gate_logit",
            nn.initializers.zeros,
            (1,),
        )

        eta_1 = jax.nn.sigmoid(eta_1_logit)[0]
        alpha_1 = jax.nn.sigmoid(alpha_1_logit)[0]
        memory_gate = jax.nn.sigmoid(memory_gate_logit)

        reads_1, delta_1 = h3_associative_scan(
            keys_1,
            queries_1,
            values_1,
            eta=eta_1,
            alpha=alpha_1,
        )

        # --------------------------------------------------------
        # 3. Nivel interno 2: Q/K/V se auto-modifican en bajo rango
        # --------------------------------------------------------
        controller_input = nn.LayerNorm(
            name="nested_controller_norm"
        )(reads_1)

        controller = nn.Dense(
            3 * self.adapter_rank,
            kernel_init=nn.initializers.xavier_uniform(),
            name="nested_controller",
        )(controller_input)
        controller = jnp.tanh(controller)
        control_q, control_k, control_v = jnp.split(controller, 3, axis=-1)

        q_low = nn.Dense(
            self.adapter_rank,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            name="nested_q_down",
        )(direct_features)
        q_delta = nn.Dense(
            self.memory_dim,
            use_bias=False,
            kernel_init=nn.initializers.zeros,
            name="nested_q_up",
        )(q_low * control_q)

        k_low = nn.Dense(
            self.adapter_rank,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            name="nested_k_down",
        )(direct_features)
        k_delta = nn.Dense(
            self.memory_dim,
            use_bias=False,
            kernel_init=nn.initializers.zeros,
            name="nested_k_up",
        )(k_low * control_k)

        v_low = nn.Dense(
            self.adapter_rank,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
            name="nested_v_down",
        )(direct_features)
        v_delta = nn.Dense(
            self.memory_dim,
            use_bias=False,
            kernel_init=nn.initializers.zeros,
            name="nested_v_up",
        )(v_low * control_v)

        nested_gate_logit = self.param(
            "nested_gate_logit",
            nn.initializers.zeros,
            (1,),
        )
        nested_gate = jax.nn.sigmoid(nested_gate_logit)
        nested_scale = self.nested_qkv_scale * nested_gate

        queries_2 = queries_1 + nested_scale * q_delta
        keys_2 = keys_1 + nested_scale * k_delta
        values_2 = values_1 + nested_scale * v_delta

        queries_2 = stable_l2_normalize(queries_2)
        keys_2 = stable_l2_normalize(keys_2)

        eta_2_logit = self.param(
            "nested_eta_logit",
            lambda key, shape: jnp.full(
                shape,
                h3_logit(self.initial_eta),
            ),
            (1,),
        )
        alpha_2_logit = self.param(
            "nested_alpha_logit",
            lambda key, shape: jnp.full(
                shape,
                h3_logit(self.initial_alpha),
            ),
            (1,),
        )

        eta_2 = jax.nn.sigmoid(eta_2_logit)[0]
        alpha_2 = jax.nn.sigmoid(alpha_2_logit)[0]

        reads_2, delta_2 = h3_associative_scan(
            keys_2,
            queries_2,
            values_2,
            eta=eta_2,
            alpha=alpha_2,
        )

        # --------------------------------------------------------
        # 4. Fusión y predicción auxiliar
        # --------------------------------------------------------
        memory_features = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="nested_memory_readout",
        )(reads_2)

        prediction = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="next_motion_predictor",
        )(reads_2)

        features = (
            direct_features
            + self.memory_residual_scale
            * memory_gate
            * memory_features
        )
        features = nn.LayerNorm(name="memory_fusion_norm")(features)

        features = nn.Dropout(
            rate=self.dropout,
            name="feature_dropout",
        )(features, deterministic=not training)

        # Se conserva mean pooling para que H3 sea comparable con H2.
        pooled = jnp.mean(features, axis=1)

        logits = nn.Dense(
            self.num_classes,
            kernel_init=nn.initializers.xavier_uniform(),
            name="classifier",
        )(pooled)

        return {
            "logits": logits,
            "prediction": prediction,
            "motion_target": motion,
            "memory_delta": 0.5 * (delta_1 + delta_2),
            "motion_gate": motion_gate,
            "memory_gate": memory_gate,
            "nested_gate": nested_gate,
        }




# ============================================================
# MODELO 4L CAUSAL
# ============================================================

NESTSAR4L_MODE = "NestSAR_4L"

def four_logit(value: float) -> float:
    value = min(max(float(value), 1e-5), 1.0 - 1e-5)
    return float(np.log(value / (1.0 - value)))


def four_normalize_vectors(x: jnp.ndarray) -> jnp.ndarray:
    return stable_l2_normalize(x)


def four_associative_scan(
    keys: jnp.ndarray,
    queries: jnp.ndarray,
    values: jnp.ndarray,
    *,
    eta: jnp.ndarray,
    alpha: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Causal associative memory: read M_(t-1), then write token t."""
    scan_inputs = (
        jnp.swapaxes(keys, 0, 1),
        jnp.swapaxes(queries, 0, 1),
        jnp.swapaxes(values, 0, 1),
    )
    batch_size = keys.shape[0]
    memory_dim = keys.shape[-1]
    initial_memory = jnp.zeros(
        (batch_size, memory_dim, memory_dim),
        dtype=keys.dtype,
    )

    def step(memory, inputs):
        key_t, query_t, value_t = inputs
        read_t = jnp.einsum("bij,bj->bi", memory, query_t)
        reconstructed_t = jnp.einsum("bij,bj->bi", memory, key_t)
        error_t = reconstructed_t - value_t
        update_t = jnp.einsum("bi,bj->bij", error_t, key_t)
        new_memory = alpha * memory - eta * update_t
        delta_t = jnp.sqrt(
            jnp.mean(jnp.square(new_memory - memory), axis=(1, 2)) + 1e-12
        )
        return new_memory, (read_t, delta_t)

    _, (reads_t, deltas_t) = jax.lax.scan(
        step,
        initial_memory,
        scan_inputs,
    )
    return jnp.swapaxes(reads_t, 0, 1), jnp.swapaxes(deltas_t, 0, 1)


class MemoryBlock(nn.Module):
    model_dim: int
    memory_dim: int
    dropout: float
    residual_scale: float
    initial_eta: float
    initial_alpha: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool):
        h = nn.LayerNorm(name="input_norm")(x)

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

        keys = four_normalize_vectors(keys)
        queries = four_normalize_vectors(queries)

        eta_logit = self.param(
            "eta_logit",
            lambda key, shape: jnp.full(shape, four_logit(self.initial_eta)),
            (1,),
        )
        alpha_logit = self.param(
            "alpha_logit",
            lambda key, shape: jnp.full(shape, four_logit(self.initial_alpha)),
            (1,),
        )
        gate_logit = self.param("memory_gate_logit", nn.initializers.zeros, (1,))

        eta = jax.nn.sigmoid(eta_logit)[0]
        alpha = jax.nn.sigmoid(alpha_logit)[0]
        gate = jax.nn.sigmoid(gate_logit)

        reads, delta = four_associative_scan(
            keys,
            queries,
            values,
            eta=eta,
            alpha=alpha,
        )

        context = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="readout",
        )(reads)
        context = nn.Dropout(rate=self.dropout, name="context_dropout")(
            context,
            deterministic=not training,
        )
        scaled_context = self.residual_scale * gate * context
        x = nn.LayerNorm(name="memory_residual_norm")(x + scaled_context)

        ff = nn.LayerNorm(name="ff_norm")(x)
        ff = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="ff_in",
        )(ff)
        ff = nn.gelu(ff)
        ff = nn.Dropout(rate=self.dropout, name="ff_dropout")(ff, deterministic=not training)
        ff = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.zeros,
            name="ff_out",
        )(ff)
        ff_gate = jax.nn.sigmoid(
            self.param("ff_gate_logit", nn.initializers.zeros, (1,))
        )
        out = nn.LayerNorm(name="output_norm")(x + ff_gate * ff)
        return out, scaled_context, delta, gate


class SlowControllerBlock(nn.Module):
    model_dim: int
    rank: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool):
        h = nn.LayerNorm(name="controller_norm")(x)
        update = nn.Dense(
            self.rank,
            kernel_init=nn.initializers.xavier_uniform(),
            name="down",
        )(h)
        update = nn.gelu(update)
        update = nn.Dropout(rate=self.dropout, name="dropout")(
            update,
            deterministic=not training,
        )
        update = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.zeros,
            name="up",
        )(update)
        gate = jax.nn.sigmoid(
            self.param("controller_gate_logit", nn.initializers.zeros, (1,))
        )
        return nn.LayerNorm(name="controller_output_norm")(x + gate * update), gate


def pool_tokens(x: jnp.ndarray, group_size: int) -> jnp.ndarray:
    batch, time, dim = x.shape
    if time % group_size != 0:
        raise ValueError(f"T={time} no es divisible por group_size={group_size}")
    return x.reshape(batch, time // group_size, group_size, dim).mean(axis=2)


def repeat_tokens(x: jnp.ndarray, repeat: int, target_time: int) -> jnp.ndarray:
    return jnp.repeat(x, repeats=repeat, axis=1)[:, :target_time]


def delay_tokens(x: jnp.ndarray, delay: int) -> jnp.ndarray:
    """Expose multiscale context only after its source group is complete."""
    if delay <= 0:
        return x
    zeros = jnp.zeros_like(x[:, :delay])
    return jnp.concatenate([zeros, x[:, :-delay]], axis=1)


class NestSAR4LModel(nn.Module):
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

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> Dict[str, jnp.ndarray]:
        if self.mode != NESTSAR4L_MODE:
            raise ValueError(f"Modo no soportado: {self.mode}")
        if x.shape[1] != 16:
            raise ValueError(f"NestSAR-4L inicial requiere 16 frames; recibido {x.shape[1]}")

        # Stem compartido con H0/H2/H3.
        pose = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="pose_projection",
        )(x)
        pose = nn.LayerNorm(name="pose_norm")(pose)
        pose = nn.gelu(pose)

        zero = jnp.zeros_like(pose[:, :1])
        motion = jnp.concatenate([zero, pose[:, 1:] - pose[:, :-1]], axis=1)
        motion_projected = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="motion_projection",
        )(motion)
        motion_projected = nn.gelu(motion_projected)
        motion_gate = jax.nn.sigmoid(
            self.param("motion_gate_logit", nn.initializers.zeros, (1,))
        )
        direct = nn.LayerNorm(name="direct_norm")(pose + motion_gate * motion_projected)

        all_deltas = []
        all_gates = []

        # L1: 2 bloques de memoria por frame, 16 pasos cada uno.
        l1 = direct
        l1_contexts = []
        for index in range(self.frame_blocks):
            l1, context, delta, gate = MemoryBlock(
                model_dim=self.model_dim,
                memory_dim=self.memory_dim,
                dropout=self.dropout,
                residual_scale=self.memory_residual_scale,
                initial_eta=self.initial_eta,
                initial_alpha=self.initial_alpha,
                name=f"l1_frame_memory_{index + 1}",
            )(l1, training)
            l1_contexts.append(context)
            all_deltas.append(delta)
            all_gates.append(gate)
        l1_context = sum(l1_contexts)

        # L2: 2 bloques de memoria sobre 4 chunks de 4 frames.
        l2 = pool_tokens(l1, self.chunk_size)
        l2_contexts = []
        l2_deltas = []
        for index in range(self.chunk_blocks):
            l2, context, delta, gate = MemoryBlock(
                model_dim=self.model_dim,
                memory_dim=self.memory_dim,
                dropout=self.dropout,
                residual_scale=self.memory_residual_scale,
                initial_eta=self.initial_eta,
                initial_alpha=self.initial_alpha,
                name=f"l2_chunk_memory_{index + 1}",
            )(l2, training)
            l2_contexts.append(context)
            l2_deltas.append(delta)
            all_gates.append(gate)
        l2_context = delay_tokens(
            repeat_tokens(sum(l2_contexts), self.chunk_size, x.shape[1]),
            self.chunk_size,
        )
        all_deltas.extend(repeat_tokens(d, self.chunk_size, x.shape[1]) for d in l2_deltas)

        # L3: 2 bloques de memoria sobre 2 segmentos de 8 frames.
        # l2 tiene 4 tokens; cada segmento agrupa 2 tokens de chunk.
        l3 = pool_tokens(l2, self.clip_size // self.chunk_size)
        l3_contexts = []
        l3_deltas = []
        for index in range(self.clip_blocks):
            l3, context, delta, gate = MemoryBlock(
                model_dim=self.model_dim,
                memory_dim=self.memory_dim,
                dropout=self.dropout,
                residual_scale=self.memory_residual_scale,
                initial_eta=self.initial_eta,
                initial_alpha=self.initial_alpha,
                name=f"l3_clip_memory_{index + 1}",
            )(l3, training)
            l3_contexts.append(context)
            l3_deltas.append(delta)
            all_gates.append(gate)
        l3_context = delay_tokens(
            repeat_tokens(sum(l3_contexts), self.clip_size, x.shape[1]),
            self.clip_size,
        )
        all_deltas.extend(repeat_tokens(d, self.clip_size, x.shape[1]) for d in l3_deltas)

        # Fusion multinivel y L4: 2 controladores lentos entrenados por AdamW.
        multiscale = jnp.concatenate([direct, l1_context, l2_context, l3_context], axis=-1)
        fusion = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="multilevel_fusion",
        )(multiscale)
        fusion = nn.gelu(fusion)
        fusion = nn.Dropout(rate=self.dropout, name="fusion_dropout")(
            fusion,
            deterministic=not training,
        )
        fusion_gate = jax.nn.sigmoid(
            self.param("fusion_gate_logit", nn.initializers.zeros, (1,))
        )
        features = nn.LayerNorm(name="fusion_norm")(direct + fusion_gate * fusion)

        controller_gates = []
        for index in range(self.controller_blocks):
            features, gate = SlowControllerBlock(
                model_dim=self.model_dim,
                rank=self.controller_rank,
                dropout=self.dropout,
                name=f"l4_slow_controller_{index + 1}",
            )(features, training)
            controller_gates.append(gate)

        features = nn.Dropout(rate=self.dropout, name="feature_dropout")(
            features,
            deterministic=not training,
        )

        prediction = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="next_motion_predictor",
        )(features)

        pooled = jnp.mean(features, axis=1)
        pooled = nn.LayerNorm(name="classifier_norm")(pooled)
        logits = nn.Dense(
            self.num_classes,
            kernel_init=nn.initializers.xavier_uniform(),
            name="classifier",
        )(pooled)

        memory_delta = sum(all_deltas) / float(len(all_deltas))
        memory_gate = sum(all_gates) / float(len(all_gates))
        controller_gate = sum(controller_gates) / float(len(controller_gates))

        return {
            "logits": logits,
            "prediction": prediction,
            "motion_target": motion,
            "memory_delta": memory_delta,
            "motion_gate": motion_gate,
            "memory_gate": memory_gate,
            "controller_gate": controller_gate,
            "fusion_gate": fusion_gate,
        }





# ============================================================
# ENTRENAMIENTO UNIFICADO, CHECKPOINTS Y CLI
# ============================================================

MODEL_ALIASES = {
    "h0": "H0_direct_motion",
    "h2": "H2_predictive_memory",
    "h3": H3_MODE,
    "nestsar_4l": NESTSAR4L_MODE,
}
LOG_PATH: Optional[Path] = None


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    if LOG_PATH is not None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class TrainState(train_state.TrainState):
    pass


def make_schedule(total_steps: int) -> optax.Schedule:
    warmup_steps = max(1, int(total_steps * CFG.warmup_fraction))
    decay_steps = max(1, total_steps - warmup_steps)
    warmup = optax.linear_schedule(
        init_value=CFG.learning_rate * 0.05,
        end_value=CFG.learning_rate,
        transition_steps=warmup_steps,
    )
    cosine = optax.cosine_decay_schedule(
        init_value=CFG.learning_rate,
        decay_steps=decay_steps,
        alpha=0.05,
    )
    return optax.join_schedules([warmup, cosine], [warmup_steps])


def build_model(model_id: str) -> nn.Module:
    mode = MODEL_ALIASES[model_id]
    common = dict(
        mode=mode,
        num_classes=CFG.num_classes,
        model_dim=CFG.model_dim,
        memory_dim=CFG.memory_dim,
        dropout=CFG.dropout,
        memory_residual_scale=CFG.memory_residual_scale,
        initial_eta=CFG.initial_eta,
        initial_alpha=CFG.initial_alpha,
    )
    if model_id in ("h0", "h2"):
        return HOPEModel(**common)
    if model_id == "h3":
        return NestSARH3Model(
            **common,
            adapter_rank=CFG.adapter_rank,
            nested_qkv_scale=CFG.nested_qkv_scale,
        )
    if model_id == "nestsar_4l":
        return NestSAR4LModel(
            **common,
            frame_blocks=CFG.frame_blocks,
            chunk_blocks=CFG.chunk_blocks,
            clip_blocks=CFG.clip_blocks,
            controller_blocks=CFG.controller_blocks,
            chunk_size=CFG.chunk_size,
            clip_size=CFG.clip_size,
            controller_rank=CFG.controller_rank,
        )
    raise ValueError(f"Modelo no soportado: {model_id}")


def create_state(rng: jax.Array, model: nn.Module, total_steps: int) -> TrainState:
    dummy = jnp.zeros(
        (2, CFG.frames, CFG.persons * CFG.joints * CFG.coords),
        dtype=jnp.float32,
    )
    variables = model.init(
        {"params": rng, "dropout": rng},
        dummy,
        training=True,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(CFG.grad_clip),
        optax.adamw(
            learning_rate=make_schedule(total_steps),
            weight_decay=CFG.weight_decay,
        ),
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
    )


def classification_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    one_hot = jax.nn.one_hot(labels, CFG.num_classes)
    smooth = (
        one_hot * (1.0 - CFG.label_smoothing)
        + CFG.label_smoothing / CFG.num_classes
    )
    return optax.softmax_cross_entropy(logits, smooth).mean()


def predictive_loss(
    prediction: jnp.ndarray,
    motion_target: jnp.ndarray,
) -> jnp.ndarray:
    predicted_next = prediction[:, :-1]
    target_next = jax.lax.stop_gradient(motion_target[:, 1:])
    return jnp.mean(jnp.square(predicted_next - target_next))


def build_steps(model: nn.Module, model_id: str):
    use_predictive = model_id != "h0"

    @jax.jit
    def train_step(
        state: TrainState,
        batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
        dropout_rng: jax.Array,
    ):
        def loss_fn(params):
            output = model.apply(
                {"params": params},
                batch_x,
                training=True,
                rngs={"dropout": dropout_rng},
            )
            ce = classification_loss(output["logits"], batch_y)
            pred = (
                predictive_loss(output["prediction"], output["motion_target"])
                if use_predictive
                else jnp.asarray(0.0, dtype=ce.dtype)
            )
            total = ce + CFG.predictive_loss_weight * pred
            accuracy = jnp.mean(jnp.argmax(output["logits"], axis=-1) == batch_y)
            metrics = {
                "loss": total,
                "ce": ce,
                "predictive": pred,
                "accuracy": accuracy,
                "memory_delta": jnp.mean(output["memory_delta"]),
                "motion_gate": jnp.mean(output["motion_gate"]),
                "memory_gate": jnp.mean(output["memory_gate"]),
            }
            return total, metrics

        (_, metrics), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(state.params)
        new_state = state.apply_gradients(grads=grads)
        metrics = dict(metrics)
        metrics["grad_norm"] = optax.global_norm(grads)
        return new_state, metrics

    @jax.jit
    def eval_step(
        params: Any,
        batch_x: jnp.ndarray,
        batch_y: jnp.ndarray,
    ):
        output = model.apply(
            {"params": params},
            batch_x,
            training=False,
        )
        ce = classification_loss(output["logits"], batch_y)
        pred = (
            predictive_loss(output["prediction"], output["motion_target"])
            if use_predictive
            else jnp.asarray(0.0, dtype=ce.dtype)
        )
        total = ce + CFG.predictive_loss_weight * pred
        predictions = jnp.argmax(output["logits"], axis=-1)
        correct = jnp.sum(predictions == batch_y)
        count = batch_y.shape[0]
        return {
            "loss_sum": total * count,
            "ce_sum": ce * count,
            "predictive_sum": pred * count,
            "correct": correct,
            "count": count,
            "memory_delta_sum": jnp.mean(output["memory_delta"]) * count,
            "motion_gate_sum": jnp.mean(output["motion_gate"]) * count,
            "memory_gate_sum": jnp.mean(output["memory_gate"]) * count,
        }

    return train_step, eval_step


def evaluate(
    state: TrainState,
    dataset: SkeletonDataset,
    eval_step,
) -> Dict[str, float]:
    totals = {
        "loss_sum": 0.0,
        "ce_sum": 0.0,
        "predictive_sum": 0.0,
        "correct": 0.0,
        "count": 0.0,
        "memory_delta_sum": 0.0,
        "motion_gate_sum": 0.0,
        "memory_gate_sum": 0.0,
    }
    for batch_x, batch_y in batch_iterator(
        dataset,
        CFG.eval_batch_size,
        shuffle=False,
        seed=0,
        drop_last=False,
    ):
        metrics = eval_step(
            state.params,
            jnp.asarray(batch_x),
            jnp.asarray(batch_y),
        )
        for key in totals:
            totals[key] += float(np.asarray(metrics[key]))
    count = max(1, int(totals["count"]))
    return {
        "loss": totals["loss_sum"] / count,
        "ce": totals["ce_sum"] / count,
        "predictive": totals["predictive_sum"] / count,
        "accuracy": totals["correct"] / count,
        "correct": int(totals["correct"]),
        "count": count,
        "memory_delta": totals["memory_delta_sum"] / count,
        "motion_gate": totals["motion_gate_sum"] / count,
        "memory_gate": totals["memory_gate_sum"] / count,
    }


def smoke_test(
    model: nn.Module,
    state: TrainState,
    train_step,
) -> None:
    mode = getattr(model, "mode", "unknown")
    log(f"Smoke test: {mode}")
    rng = jax.random.PRNGKey(12345)
    x = jax.random.normal(
        rng,
        (4, CFG.frames, CFG.persons * CFG.joints * CFG.coords),
    )
    y = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)
    output = model.apply({"params": state.params}, x, training=False)

    assert output["logits"].shape == (4, CFG.num_classes)
    assert output["prediction"].shape == (4, CFG.frames, CFG.model_dim)
    assert output["memory_delta"].shape == (4, CFG.frames)
    if not tree_all_finite(output):
        raise FloatingPointError("El smoke test produjo valores no finitos.")

    output_2 = model.apply({"params": state.params}, x, training=False)
    repeat_error = float(
        np.asarray(jnp.max(jnp.abs(output["logits"] - output_2["logits"])))
    )
    if repeat_error > 1e-6:
        raise AssertionError(
            f"El estado parece persistir entre secuencias: {repeat_error}"
        )

    x_changed = x.at[:, -1].set(x[:, -1] + 100.0)
    changed = model.apply({"params": state.params}, x_changed, training=False)
    causal_error = float(
        np.asarray(
            jnp.max(
                jnp.abs(
                    output["prediction"][:, :-1]
                    - changed["prediction"][:, :-1]
                )
            )
        )
    )
    if causal_error > 1e-5:
        raise AssertionError(
            f"Fallo de causalidad temporal: error={causal_error}"
        )

    batch_x = jnp.zeros(
        (
            CFG.batch_size,
            CFG.frames,
            CFG.persons * CFG.joints * CFG.coords,
        ),
        dtype=jnp.float32,
    )
    batch_y = jnp.arange(CFG.batch_size, dtype=jnp.int32) % CFG.num_classes
    _, metrics = train_step(
        state,
        batch_x,
        batch_y,
        jax.random.PRNGKey(CFG.seed + 999),
    )
    jax.block_until_ready(metrics["loss"])
    if not tree_all_finite(metrics):
        raise FloatingPointError("El paso físico produjo métricas no finitas.")

    log(
        f"Smoke test superado | repeat={repeat_error:.3e} | "
        f"causal={causal_error:.3e} | batch={CFG.batch_size} | "
        f"loss={float(np.asarray(metrics['loss'])):.6f}"
    )


def save_full_checkpoint(
    prefix: Path,
    state: TrainState,
    rng: jax.Array,
    metadata: Mapping[str, Any],
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": state.params,
        "opt_state": state.opt_state,
        "step": state.step,
        "rng": rng,
    }
    prefix.with_suffix(".msgpack").write_bytes(serialization.to_bytes(payload))
    prefix.with_suffix(".json").write_text(
        json.dumps(dict(metadata), indent=2),
        encoding="utf-8",
    )


def restore_full_checkpoint(
    prefix: Path,
    state: TrainState,
    rng: jax.Array,
) -> Tuple[TrainState, jax.Array, Dict[str, Any]]:
    metadata_path = prefix.with_suffix(".json")
    payload_path = prefix.with_suffix(".msgpack")
    if not metadata_path.is_file() or not payload_path.is_file():
        raise FileNotFoundError(f"Checkpoint incompleto: {prefix}")
    template = {
        "params": state.params,
        "opt_state": state.opt_state,
        "step": state.step,
        "rng": rng,
    }
    restored = serialization.from_bytes(template, payload_path.read_bytes())
    state = state.replace(
        params=restored["params"],
        opt_state=restored["opt_state"],
        step=restored["step"],
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return state, restored["rng"], metadata


def archive_outputs(output_dir: Path, protocol: str, mode: str) -> None:
    matching = list(output_dir.glob(f"{protocol}_{mode}_*"))
    if not matching:
        return
    archive = output_dir / "archive" / time.strftime("%Y%m%d_%H%M%S")
    archive.mkdir(parents=True, exist_ok=True)
    for path in matching:
        shutil.move(str(path), str(archive / path.name))


def run_experiment(
    protocol: str,
    model_id: str,
    dataset_path: Path,
    output_dir: Path,
    config_hash: str,
    resume: str,
    force_rerun: bool,
    smoke_only: bool,
) -> Dict[str, Any]:
    global LOG_PATH
    seed_everything(CFG.seed)
    mode = MODEL_ALIASES[model_id]
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_PATH = output_dir / f"{protocol}_{mode}.log"

    result_path = output_dir / f"{protocol}_{mode}_result.json"
    if force_rerun:
        archive_outputs(output_dir, protocol, mode)
    elif result_path.is_file() and resume == "none" and not smoke_only:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        log(
            f"Resultado existente reutilizado: "
            f"{100.0 * result['best_accuracy']:.6f}%"
        )
        return result

    log("=" * 76)
    log(f"PROTOCOLO={protocol.upper()} | MODELO={mode} | HASH={config_hash}")
    log("=" * 76)
    log(f"JAX={jax.__version__} | backend={jax.default_backend()}")
    log(f"Devices visibles al worker: {jax.devices()}")
    if jax.default_backend() != "gpu":
        log("ADVERTENCIA: este worker no está usando GPU.")

    raw_data = load_pickle(dataset_path)
    train_samples, val_samples = build_samples(
        raw_data,
        protocol=protocol,
        max_train=CFG.max_train_samples,
        max_val=CFG.max_val_samples,
        seed=CFG.seed,
    )
    train_dataset = SkeletonDataset(train_samples)
    val_dataset = SkeletonDataset(val_samples)
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / CFG.batch_size))
    total_steps = CFG.epochs * steps_per_epoch

    model = build_model(model_id)
    rng = jax.random.PRNGKey(CFG.seed)
    state = create_state(rng, model, total_steps)
    parameter_count = count_parameters(state.params)
    log(f"Parámetros: {parameter_count:,}")

    train_step, eval_step = build_steps(model, model_id)
    smoke_test(model, state, train_step)
    if smoke_only:
        return {
            "protocol": protocol,
            "mode": mode,
            "parameters": parameter_count,
            "smoke_only": True,
        }

    last_prefix = output_dir / f"{protocol}_{mode}_checkpoint_last"
    best_prefix = output_dir / f"{protocol}_{mode}_checkpoint_best"
    history_path = output_dir / f"{protocol}_{mode}_history.json"

    start_epoch = 1
    best_accuracy = -1.0
    best_correct = -1
    best_epoch = 0
    best_loss = float("inf")
    epochs_without_improvement = 0
    best_state = state
    history: List[Dict[str, Any]] = []

    if resume == "auto" and last_prefix.with_suffix(".msgpack").is_file():
        state, rng, metadata = restore_full_checkpoint(last_prefix, state, rng)
        if metadata.get("config_hash") != config_hash:
            raise RuntimeError(
                "El checkpoint no coincide con la configuración actual. "
                "Usa los mismos hiperparámetros o --force-rerun."
            )
        start_epoch = int(metadata["epoch"]) + 1
        best_accuracy = float(metadata["best_accuracy"])
        best_correct = int(metadata["best_correct"])
        best_epoch = int(metadata["best_epoch"])
        best_loss = float(metadata["best_loss"])
        epochs_without_improvement = int(metadata["epochs_without_improvement"])
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        if best_prefix.with_suffix(".msgpack").is_file():
            best_state, _, _ = restore_full_checkpoint(best_prefix, state, rng)
        log(
            f"Reanudación exacta desde época {start_epoch - 1}; "
            f"siguiente época={start_epoch}; step={int(np.asarray(state.step))}"
        )

    run_start = time.time()
    for epoch in range(start_epoch, CFG.epochs + 1):
        epoch_start = time.time()
        train_totals = {
            "loss": 0.0,
            "ce": 0.0,
            "predictive": 0.0,
            "accuracy": 0.0,
            "memory_delta": 0.0,
            "motion_gate": 0.0,
            "memory_gate": 0.0,
            "grad_norm": 0.0,
        }
        batch_count = 0

        for batch_index, (batch_x, batch_y) in enumerate(
            batch_iterator(
                train_dataset,
                CFG.batch_size,
                shuffle=True,
                seed=CFG.seed + epoch,
                drop_last=False,
            ),
            start=1,
        ):
            rng, dropout_rng = jax.random.split(rng)
            state, metrics = train_step(
                state,
                jnp.asarray(batch_x),
                jnp.asarray(batch_y),
                dropout_rng,
            )
            host = {key: float(np.asarray(value)) for key, value in metrics.items()}
            for key in train_totals:
                train_totals[key] += host[key]
            batch_count += 1

            if (
                batch_index == 1
                or batch_index % CFG.log_every_batches == 0
                or batch_index == steps_per_epoch
            ):
                elapsed = time.time() - epoch_start
                rate = batch_index / max(elapsed, 1e-6)
                remaining = (steps_per_epoch - batch_index) / max(rate, 1e-6)
                log(
                    f"{protocol.upper()} {mode} | Ep {epoch:03d}/{CFG.epochs} | "
                    f"batch {batch_index:04d}/{steps_per_epoch:04d} | "
                    f"loss={host['loss']:.4f} | "
                    f"acc={100.0 * host['accuracy']:.2f}% | "
                    f"pred={host['predictive']:.5f} | "
                    f"memΔ={host['memory_delta']:.6f} | ETA={remaining:.0f}s"
                )

        train_mean = {
            key: value / max(batch_count, 1)
            for key, value in train_totals.items()
        }
        val_metrics = evaluate(state, val_dataset, eval_step)
        epoch_seconds = time.time() - epoch_start
        record = {
            "epoch": epoch,
            "train": train_mean,
            "validation": val_metrics,
            "epoch_seconds": epoch_seconds,
        }
        history.append(record)

        improved = (
            val_metrics["correct"] > best_correct
            or (
                val_metrics["correct"] == best_correct
                and val_metrics["loss"] < best_loss
            )
        )
        if improved:
            best_accuracy = val_metrics["accuracy"]
            best_correct = val_metrics["correct"]
            best_epoch = epoch
            best_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            best_state = state
            save_full_checkpoint(
                best_prefix,
                best_state,
                rng,
                {
                    "config_hash": config_hash,
                    "protocol": protocol,
                    "mode": mode,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_accuracy": best_accuracy,
                    "best_correct": best_correct,
                    "best_loss": best_loss,
                    "validation_count": val_metrics["count"],
                    "parameters": parameter_count,
                    "config": dataclasses.asdict(CFG),
                },
            )
        else:
            epochs_without_improvement += 1

        history_path.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        save_full_checkpoint(
            last_prefix,
            state,
            rng,
            {
                "config_hash": config_hash,
                "protocol": protocol,
                "mode": mode,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_accuracy": best_accuracy,
                "best_correct": best_correct,
                "best_loss": best_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "validation_count": val_metrics["count"],
                "parameters": parameter_count,
                "config": dataclasses.asdict(CFG),
            },
        )

        log(
            f"{protocol.upper()} {mode} | Ep {epoch:03d} FINAL | "
            f"train_acc={100.0 * train_mean['accuracy']:.3f}% | "
            f"val_acc={100.0 * val_metrics['accuracy']:.6f}% | "
            f"{val_metrics['correct']}/{val_metrics['count']} | "
            f"val_loss={val_metrics['loss']:.5f} | "
            f"pred={val_metrics['predictive']:.6f} | "
            f"memΔ={val_metrics['memory_delta']:.7f} | "
            f"motion_gate={val_metrics['motion_gate']:.4f} | "
            f"memory_gate={val_metrics['memory_gate']:.4f} | "
            f"best={100.0 * best_accuracy:.6f}%@{best_epoch} | "
            f"patience={epochs_without_improvement}/{CFG.patience} | "
            f"time={epoch_seconds:.1f}s"
        )
        if epochs_without_improvement >= CFG.patience:
            log(f"Early stopping: {CFG.patience} épocas sin mejora.")
            break

    total_seconds = time.time() - run_start
    if best_prefix.with_suffix(".msgpack").is_file():
        best_state, _, _ = restore_full_checkpoint(best_prefix, state, rng)
    best_metrics = evaluate(best_state, val_dataset, eval_step)
    result = {
        "protocol": protocol,
        "model": model_id,
        "mode": mode,
        "best_epoch": best_epoch,
        "best_accuracy": best_metrics["accuracy"],
        "best_correct": best_metrics["correct"],
        "validation_count": best_metrics["count"],
        "best_loss": best_metrics["loss"],
        "best_ce": best_metrics["ce"],
        "best_predictive": best_metrics["predictive"],
        "memory_delta": best_metrics["memory_delta"],
        "motion_gate": best_metrics["motion_gate"],
        "memory_gate": best_metrics["memory_gate"],
        "parameters": parameter_count,
        "runtime_seconds": total_seconds,
        "config_hash": config_hash,
        "config": dataclasses.asdict(CFG),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log("=" * 76)
    log(
        f"RESULTADO {protocol.upper()} {mode}: "
        f"{100.0 * result['best_accuracy']:.6f}% | "
        f"{result['best_correct']}/{result['validation_count']} | "
        f"época {result['best_epoch']} | loss={result['best_loss']:.6f}"
    )
    log("=" * 76)
    print("FINAL_JSON=" + json.dumps(result), flush=True)
    return result


def detect_nvidia_gpus(max_gpus: int = 0) -> List[Dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        try:
            index = int(parts[0])
            memory = int(float(parts[2]))
        except ValueError:
            continue
        gpus.append({"index": index, "name": parts[1], "memory_mib": memory})
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() not in ("", "-1"):
        tokens = [token.strip() for token in visible.split(",")]
        if all(token.isdigit() for token in tokens):
            allowed = {int(token) for token in tokens}
            gpus = [gpu for gpu in gpus if gpu["index"] in allowed]
    if max_gpus > 0:
        gpus = gpus[:max_gpus]
    return gpus


def parse_gpu_map(value: str, gpus: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    if value == "auto":
        if not gpus:
            return {}
        if len(gpus) == 1:
            return {"xsub": int(gpus[0]["index"]), "xset": int(gpus[0]["index"])}
        return {"xsub": int(gpus[0]["index"]), "xset": int(gpus[1]["index"])}
    mapping: Dict[str, int] = {}
    for item in value.split(","):
        protocol, raw_index = item.split(":", 1)
        protocol = protocol.strip().lower()
        if protocol not in ("xsub", "xset"):
            raise ValueError(f"Protocolo inválido en --gpu-map: {protocol}")
        mapping[protocol] = int(raw_index.strip())
    available = {int(gpu["index"]) for gpu in gpus}
    missing = [index for index in mapping.values() if index not in available]
    if missing:
        raise ValueError(
            f"--gpu-map usa GPU no visible: {missing}; visibles={sorted(available)}"
        )
    return mapping


def replace_protocol_argument(argv: Sequence[str], protocol: str) -> List[str]:
    result: List[str] = []
    skip = False
    for index, token in enumerate(argv):
        if skip:
            skip = False
            continue
        if token == "--protocol":
            skip = True
            continue
        if token.startswith("--protocol="):
            continue
        if token == "--worker":
            continue
        result.append(token)
    result.extend(["--protocol", protocol, "--worker"])
    return result


def stream_process(process: subprocess.Popen, label: str) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{label}] {line}", end="", flush=True)


def launch_protocols(args: argparse.Namespace) -> int:
    gpus = detect_nvidia_gpus(args.max_gpus)
    print(f"GPU visibles detectadas: {len(gpus)}", flush=True)
    for gpu in gpus:
        print(
            f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory_mib']} MiB)",
            flush=True,
        )
    if not gpus and not args.allow_cpu:
        raise RuntimeError(
            "No se detectó GPU NVIDIA. En Kaggle activa GPU antes de entrenar "
            "o usa --allow-cpu solamente para pruebas."
        )

    mapping = parse_gpu_map(args.gpu_map, gpus)
    protocols = ("xsub", "xset")
    parallel = bool(gpus) and mapping.get("xsub") != mapping.get("xset")
    print(
        f"Plan: XSUB->{mapping.get('xsub', 'CPU')} | "
        f"XSET->{mapping.get('xset', 'CPU')} | "
        f"{'paralelo' if parallel else 'secuencial'}",
        flush=True,
    )
    if args.dry_run:
        return 0

    def start(protocol: str) -> subprocess.Popen:
        env = os.environ.copy()
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(mapping[protocol])
        else:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        command = [sys.executable, "-u", str(Path(__file__).resolve())]
        command += replace_protocol_argument(sys.argv[1:], protocol)
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

    if parallel:
        processes = {protocol: start(protocol) for protocol in protocols}
        threads = [
            threading.Thread(
                target=stream_process,
                args=(process, protocol.upper()),
                daemon=True,
            )
            for protocol, process in processes.items()
        ]
        for thread in threads:
            thread.start()
        codes = {protocol: process.wait() for protocol, process in processes.items()}
        for thread in threads:
            thread.join()
    else:
        codes = {}
        for protocol in protocols:
            process = start(protocol)
            stream_process(process, protocol.upper())
            codes[protocol] = process.wait()

    failures = {protocol: code for protocol, code in codes.items() if code != 0}
    if failures:
        raise RuntimeError(f"Fallaron workers: {failures}")
    print("XSUB y XSET terminaron correctamente.", flush=True)
    return 0


def config_hash(model_id: str, dataset_path: Path) -> str:
    payload = {
        "model": model_id,
        "dataset_name": dataset_path.name,
        "dataset_size": dataset_path.stat().st_size,
        "config": dataclasses.asdict(CFG),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def configure_from_args(args: argparse.Namespace) -> None:
    global CFG
    CFG = dataclasses.replace(
        CFG,
        frames=args.frames,
        num_classes=args.num_classes,
        model_dim=args.model_dim,
        memory_dim=args.memory_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        memory_residual_scale=args.memory_residual_scale,
        initial_eta=args.initial_eta,
        initial_alpha=args.initial_alpha,
        predictive_loss_weight=args.predictive_loss_weight,
        adapter_rank=args.adapter_rank,
        nested_qkv_scale=args.nested_qkv_scale,
        frame_blocks=args.frame_blocks,
        chunk_blocks=args.chunk_blocks,
        clip_blocks=args.clip_blocks,
        controller_blocks=args.controller_blocks,
        chunk_size=args.chunk_size,
        clip_size=args.clip_size,
        controller_rank=args.controller_rank,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        seed=args.seed,
        log_every_batches=args.log_every_batches,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrenador único NestSAR para NTU RGB+D 120. "
            "Detecta las GPU y lanza XSUB/XSET como procesos independientes."
        )
    )
    parser.add_argument("--version", action="version", version="NestSAR 0.3.1")
    parser.add_argument("--model", choices=tuple(MODEL_ALIASES), default="nestsar_4l")
    parser.add_argument("--protocol", choices=("xsub", "xset", "both"), default="both")
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=120)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--warmup-fraction", type=float, default=0.10)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--memory-residual-scale", type=float, default=0.25)
    parser.add_argument("--initial-eta", type=float, default=0.10)
    parser.add_argument("--initial-alpha", type=float, default=0.95)
    parser.add_argument("--predictive-loss-weight", type=float, default=0.10)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--nested-qkv-scale", type=float, default=0.25)
    parser.add_argument("--frame-blocks", type=int, default=2)
    parser.add_argument("--chunk-blocks", type=int, default=2)
    parser.add_argument("--clip-blocks", type=int, default=2)
    parser.add_argument("--controller-blocks", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--clip-size", type=int, default=8)
    parser.add_argument("--controller-rank", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument("--gpu-map", default="auto", help="auto o xsub:0,xset:1")
    parser.add_argument("--max-gpus", type=int, default=0, help="0 usa todas las visibles")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--resume", choices=("none", "auto"), default="none")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-gpus", action="store_true")
    parser.add_argument("--hash-dataset", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "frames", "num_classes", "model_dim", "memory_dim", "batch_size",
        "eval_batch_size", "epochs", "patience", "adapter_rank",
        "frame_blocks", "chunk_blocks", "clip_blocks", "controller_blocks",
        "chunk_size", "clip_size", "controller_rank", "log_every_batches",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} debe ser mayor que cero")
    if args.max_train_samples < 0 or args.max_val_samples < 0:
        raise ValueError("Los límites de muestras no pueden ser negativos.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout debe estar en [0,1).")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("--label-smoothing debe estar en [0,1).")
    if not 0 <= args.warmup_fraction <= 1:
        raise ValueError("--warmup-fraction debe estar en [0,1].")
    if args.model == "nestsar_4l":
        if args.frames != 16:
            raise ValueError("La arquitectura 4L validada requiere --frames 16.")
        if args.frames % args.chunk_size or args.frames % args.clip_size:
            raise ValueError("frames debe ser divisible por chunk-size y clip-size.")


def write_run_metadata(
    output_dir: Path,
    protocol: str,
    args: argparse.Namespace,
    dataset_path: Path,
    digest: str,
) -> None:
    environment = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "jax": jax.__version__,
        "jaxlib": getattr(jax.lib, "__version__", None),
        "flax": optional_package_version("flax"),
        "optax": optional_package_version("optax"),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dataset": {
            "path": str(dataset_path),
            "size_bytes": dataset_path.stat().st_size,
            "sha256": sha256_file(dataset_path) if args.hash_dataset else None,
        },
        "config_hash": digest,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"resolved_config_{protocol}.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "config": dataclasses.asdict(CFG),
                "config_hash": digest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / f"environment_{protocol}.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )


def optional_package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    validate_args(args)

    if args.list_gpus:
        gpus = detect_nvidia_gpus(args.max_gpus)
        print(json.dumps(gpus, indent=2))
        return 0

    if args.protocol == "both" and not args.worker:
        return launch_protocols(args)

    configure_from_args(args)
    dataset_path = find_dataset(None if args.dataset == "auto" else args.dataset)
    digest = config_hash(args.model, dataset_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        / args.model
        / f"seed_{args.seed}"
        / digest
    )
    write_run_metadata(output_dir, args.protocol, args, dataset_path, digest)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "protocol": args.protocol,
                    "dataset": str(dataset_path),
                    "output_dir": str(output_dir),
                    "config_hash": digest,
                    "config": dataclasses.asdict(CFG),
                },
                indent=2,
            )
        )
        return 0

    if not args.allow_cpu and jax.default_backend() != "gpu":
        raise RuntimeError(
            "JAX no está usando GPU. Activa GPU o usa --allow-cpu para pruebas."
        )

    run_experiment(
        protocol=args.protocol,
        model_id=args.model,
        dataset_path=dataset_path,
        output_dir=output_dir,
        config_hash=digest,
        resume=args.resume,
        force_rerun=args.force_rerun,
        smoke_only=args.smoke_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

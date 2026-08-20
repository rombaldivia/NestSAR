#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NestSAR Native-Reframe v1
=========================

Pure nested-learning skeleton action recognition:
- NO softmax self-attention
- NO Transformer block
- NO GCN / adjacency matrix
- NO CNN / TCN
- Spatial + temporal interaction are associative fast memories only.
- Each memory block performs two nested read-before-write scans; the first
  memory read controls low-rank Q/K/V deltas for the second memory.
- Reframe-window native mechanism: one parameter tree is independent of T.
  A checkpoint trained at T=32 can be evaluated at T=16/24/32 with exactly
  the same parameters and no temporal-embedding resize/interpolation.

Designed for Kaggle JAX GPU/TPU and uses all visible local JAX devices via pmap.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import os
import pickle
import random
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization
from flax.training import train_state
import optax


# =============================================================================
# CONFIG
# =============================================================================

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))

def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")

def _env_frames(name: str, default: str) -> Tuple[int, ...]:
    raw = os.environ.get(name, default)
    vals = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not vals:
        raise ValueError(f"{name} must contain at least one frame count")
    return vals

@dataclasses.dataclass(frozen=True)
class Config:
    protocol: str = os.environ.get("NESTSAR_PROTOCOL", "xsub").lower()
    train_frames: int = _env_int("NESTSAR_TRAIN_FRAMES", 32)
    eval_frames: Tuple[int, ...] = _env_frames("NESTSAR_EVAL_FRAMES", "16,24,32")

    persons: int = 2
    joints: int = 25
    coords: int = 3
    num_classes: int = 120

    model_dim: int = _env_int("NESTSAR_MODEL_DIM", 128)
    memory_dim: int = _env_int("NESTSAR_MEMORY_DIM", 64)
    nested_rank: int = _env_int("NESTSAR_NESTED_RANK", 16)
    controller_rank: int = _env_int("NESTSAR_CONTROLLER_RANK", 32)

    spatial_blocks: int = _env_int("NESTSAR_SPATIAL_BLOCKS", 1)
    frame_blocks: int = _env_int("NESTSAR_FRAME_BLOCKS", 1)
    chunk_blocks: int = _env_int("NESTSAR_CHUNK_BLOCKS", 1)
    clip_blocks: int = _env_int("NESTSAR_CLIP_BLOCKS", 1)
    controller_blocks: int = _env_int("NESTSAR_CONTROLLER_BLOCKS", 2)

    chunk_size: int = _env_int("NESTSAR_CHUNK_SIZE", 4)
    clip_size: int = _env_int("NESTSAR_CLIP_SIZE", 8)

    dropout: float = _env_float("NESTSAR_DROPOUT", 0.15)
    memory_residual_scale: float = _env_float("NESTSAR_MEMORY_RESIDUAL_SCALE", 0.25)
    nested_qkv_scale: float = _env_float("NESTSAR_NESTED_QKV_SCALE", 0.25)
    initial_eta: float = _env_float("NESTSAR_INITIAL_ETA", 0.10)
    initial_alpha: float = _env_float("NESTSAR_INITIAL_ALPHA", 0.95)

    epochs: int = _env_int("NESTSAR_EPOCHS", 40)
    patience: int = _env_int("NESTSAR_PATIENCE", 15)
    global_batch: int = _env_int("NESTSAR_GLOBAL_BATCH", 16)
    eval_batch: int = _env_int("NESTSAR_EVAL_BATCH", 32)
    learning_rate: float = _env_float("NESTSAR_LR", 4.0e-4)
    min_learning_rate: float = _env_float("NESTSAR_MIN_LR", 2.0e-5)
    warmup_fraction: float = _env_float("NESTSAR_WARMUP", 0.08)
    weight_decay: float = _env_float("NESTSAR_WEIGHT_DECAY", 0.03)
    label_smoothing: float = _env_float("NESTSAR_LABEL_SMOOTHING", 0.05)
    grad_clip: float = _env_float("NESTSAR_GRAD_CLIP", 1.0)

    seed: int = _env_int("NESTSAR_SEED", 128)
    max_train_samples: int = _env_int("NESTSAR_MAX_TRAIN", 0)
    max_val_samples: int = _env_int("NESTSAR_MAX_VAL", 0)
    resume: bool = _env_bool("NESTSAR_RESUME", False)
    cache_train: bool = _env_bool("NESTSAR_CACHE_TRAIN", True)

    output_dir: str = os.environ.get(
        "NESTSAR_OUTPUT",
        "/kaggle/working/nestsar_native_reframe_v1",
    )
    explicit_dataset: str = os.environ.get("NESTSAR_DATASET", "")

CFG = Config()


def validate_config() -> None:
    if CFG.protocol not in ("xsub", "xset"):
        raise ValueError("NESTSAR_PROTOCOL must be xsub or xset")
    if CFG.model_dim < 1 or CFG.memory_dim < 1 or CFG.nested_rank < 1:
        raise ValueError("model/memory/nested dimensions must be positive")
    if CFG.clip_size % CFG.chunk_size:
        raise ValueError("clip_size must be divisible by chunk_size")
    if CFG.train_frames % CFG.clip_size:
        raise ValueError(
            f"train_frames={CFG.train_frames} must be divisible by clip_size={CFG.clip_size}"
        )
    for t in CFG.eval_frames:
        if t % CFG.clip_size:
            raise ValueError(
                f"eval frame count {t} must be divisible by clip_size={CFG.clip_size}"
            )
    if not (0.0 < CFG.initial_eta < 1.0):
        raise ValueError("initial_eta must be in (0,1)")
    if not (0.0 < CFG.initial_alpha < 1.0):
        raise ValueError("initial_alpha must be in (0,1)")
    if CFG.global_batch < 1 or CFG.eval_batch < 1:
        raise ValueError("batch sizes must be positive")


# =============================================================================
# UTIL
# =============================================================================

def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def tree_param_count(tree: Any) -> int:
    return int(sum(np.prod(np.asarray(x).shape) for x in jax.tree_util.tree_leaves(tree)))

def tree_shape_signature(tree: Any):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return treedef, tuple(tuple(np.asarray(x).shape) for x in leaves)

def all_finite(tree: Any) -> bool:
    return all(bool(np.asarray(jnp.all(jnp.isfinite(x)))) for x in jax.tree_util.tree_leaves(tree))

def logit(p: float) -> float:
    p = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return float(np.log(p / (1.0 - p)))

def stable_l2(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.lax.rsqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + 1e-8)


# =============================================================================
# DATA
# =============================================================================

def find_dataset() -> Path:
    if CFG.explicit_dataset:
        p = Path(CFG.explicit_dataset)
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"NESTSAR_DATASET does not exist: {p}")

    names = ("ntu120_3danno.pkl", "ntu120_3danno_clean.pkl", "ntu120.pkl")
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path("/content")):
        if not root.exists():
            continue
        for name in names:
            try:
                hits = list(root.rglob(name))
            except Exception:
                hits = []
            if hits:
                return hits[0].resolve()
    raise FileNotFoundError(
        "Could not find NTU120 PKL. Set NESTSAR_DATASET=/path/to/ntu120_3danno.pkl"
    )

def load_pickle(path: Path):
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")

def sample_id(annotation: Mapping[str, Any], index: int) -> str:
    for k in ("frame_dir", "filename", "sample_name", "name", "id", "video_id"):
        if k in annotation:
            return str(annotation[k])
    return str(index)

def annotation_label(annotation: Mapping[str, Any]) -> int:
    for k in ("label", "action_label", "class", "target"):
        if k in annotation:
            y = int(annotation[k])
            if 0 <= y < CFG.num_classes:
                return y
    raise KeyError("No valid label found")

def annotation_keypoints(annotation: Mapping[str, Any]) -> np.ndarray:
    for k in ("keypoint", "keypoints", "skeleton", "skeletons", "data"):
        if k in annotation:
            return np.asarray(annotation[k], dtype=np.float32)
    raise KeyError("No keypoints found")

def to_tmvc(keypoints: np.ndarray) -> np.ndarray:
    """Convert common formats to [T,M,V,C]."""
    x = np.asarray(keypoints, dtype=np.float32)
    if x.ndim == 3:
        if x.shape[-1] not in (2, 3):
            raise ValueError(f"Unsupported keypoint shape: {x.shape}")
        x = x[:, None, :, :]
    elif x.ndim == 4:
        if x.shape[-1] not in (2, 3):
            raise ValueError(f"Unsupported keypoint shape: {x.shape}")
        if x.shape[0] <= 4 and x.shape[1] > 4:
            x = np.transpose(x, (1, 0, 2, 3))
        elif x.shape[1] <= 4 and x.shape[0] > 4:
            pass
        elif x.shape[0] < x.shape[1]:
            x = np.transpose(x, (1, 0, 2, 3))
    else:
        raise ValueError(f"Unsupported keypoint rank: {x.shape}")
    return x

def temporal_reframe_indices(total_frames: int, target_frames: int) -> np.ndarray:
    """
    Native reframe window: uniformly re-sample the complete original clip.

    target_frames is a runtime/data choice only. It creates no trainable
    parameters and does not alter the NestSAR parameter tree.
    """
    if total_frames <= 1:
        return np.zeros((target_frames,), dtype=np.int64)
    return np.linspace(
        0,
        total_frames - 1,
        num=target_frames,
        dtype=np.float32,
    ).round().astype(np.int64)

def preprocess_keypoints(keypoints: np.ndarray, target_frames: int) -> np.ndarray:
    x = to_tmvc(keypoints)

    if x.shape[-1] == 2:
        x = np.concatenate(
            [x, np.zeros((*x.shape[:-1], 1), dtype=np.float32)],
            axis=-1,
        )

    if x.shape[2] < CFG.joints:
        pad = np.zeros(
            (x.shape[0], x.shape[1], CFG.joints - x.shape[2], CFG.coords),
            dtype=np.float32,
        )
        x = np.concatenate([x, pad], axis=2)
    x = x[:, :, :CFG.joints, :CFG.coords]

    person_energy = np.sum(np.abs(x), axis=(0, 2, 3))
    x = x[:, np.argsort(-person_energy)]
    if x.shape[1] < CFG.persons:
        pad = np.zeros(
            (x.shape[0], CFG.persons - x.shape[1], CFG.joints, CFG.coords),
            dtype=np.float32,
        )
        x = np.concatenate([x, pad], axis=1)
    x = x[:, :CFG.persons]

    x = x[temporal_reframe_indices(x.shape[0], target_frames)]

    center = x[:, 0:1, 0:1, :]
    valid_center = np.any(np.abs(center) > 1e-8, axis=-1, keepdims=True)
    x = np.where(valid_center, x - center, x)

    valid_joint = np.any(np.abs(x) > 1e-8, axis=-1, keepdims=True)
    x = np.where(valid_joint, x, 0.0)

    nz = np.abs(x) > 1e-8
    if np.any(nz):
        rms = float(np.sqrt(np.mean(np.square(x[nz]))) + 1e-6)
        x = x / rms

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    coords = np.transpose(x, (0, 2, 1, 3)).reshape(
        target_frames, CFG.joints, CFG.persons * CFG.coords
    )
    motion = np.zeros_like(coords)
    motion[1:] = coords[1:] - coords[:-1]
    return np.concatenate([coords, motion], axis=-1).astype(np.float32)

def extract_splits(data: Mapping[str, Any], protocol: str):
    annotations = list(data["annotations"])
    split = data["split"]
    train_key = f"{protocol}_train"
    val_key = f"{protocol}_val"
    if train_key not in split:
        raise KeyError(f"Missing split {train_key}")
    if val_key not in split:
        val_key = f"{protocol}_test"
        if val_key not in split:
            raise KeyError(f"Missing split {protocol}_val/{protocol}_test")

    indexed = {sample_id(a, i): a for i, a in enumerate(annotations)}
    train = [indexed[str(k)] for k in split[train_key] if str(k) in indexed]
    val = [indexed[str(k)] for k in split[val_key] if str(k) in indexed]

    rng = np.random.default_rng(CFG.seed)
    rng.shuffle(train)
    val = sorted(val, key=lambda a: sample_id(a, 0))

    if CFG.max_train_samples > 0:
        train = train[:CFG.max_train_samples]
    if CFG.max_val_samples > 0:
        val = val[:CFG.max_val_samples]
    return train, val, train_key, val_key

class SkeletonDataset:
    def __init__(self, samples: Sequence[Mapping[str, Any]], frames: int, cache: bool):
        self.samples = list(samples)
        self.frames = int(frames)
        self.cache_enabled = bool(cache)
        self.cache: Dict[int, Tuple[np.ndarray, np.int32]] = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.cache_enabled and idx in self.cache:
            return self.cache[idx]
        ann = self.samples[idx]
        item = (
            preprocess_keypoints(annotation_keypoints(ann), self.frames),
            np.int32(annotation_label(ann)),
        )
        if self.cache_enabled:
            self.cache[idx] = item
        return item

def batch_iterator(
    dataset: SkeletonDataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    drop_last: bool,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    ids = np.arange(len(dataset))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
    for start in range(0, len(ids), batch_size):
        sub = ids[start:start + batch_size]
        if drop_last and len(sub) < batch_size:
            continue
        xs, ys = [], []
        for i in sub:
            x, y = dataset[int(i)]
            xs.append(x)
            ys.append(y)
        yield np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int32)


# =============================================================================
# PURE NESTED ASSOCIATIVE MEMORY
# =============================================================================

def associative_scan(
    keys: jnp.ndarray,
    queries: jnp.ndarray,
    values: jnp.ndarray,
    eta: jnp.ndarray,
    alpha: jnp.ndarray,
    *,
    bidirectional: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Read-before-write fast associative memory. No QK^T token matrix/softmax."""

    def one_direction(k, q, v):
        scan_inputs = (
            jnp.swapaxes(k, 0, 1),
            jnp.swapaxes(q, 0, 1),
            jnp.swapaxes(v, 0, 1),
        )
        b = k.shape[0]
        dm = k.shape[-1]
        memory0 = jnp.zeros((b, dm, dm), dtype=k.dtype)

        def step(memory, inputs):
            key_t, query_t, value_t = inputs
            read_t = jnp.einsum("bij,bj->bi", memory, query_t)
            reconstructed = jnp.einsum("bij,bj->bi", memory, key_t)
            error = reconstructed - value_t
            update = jnp.einsum("bi,bj->bij", error, key_t)
            memory_new = alpha * memory - eta * update
            delta = jnp.sqrt(
                jnp.mean(jnp.square(memory_new - memory), axis=(1, 2)) + 1e-12
            )
            return memory_new, (read_t, delta)

        _, (reads_t, deltas_t) = jax.lax.scan(step, memory0, scan_inputs)
        return jnp.swapaxes(reads_t, 0, 1), jnp.swapaxes(deltas_t, 0, 1)

    reads_f, delta_f = one_direction(keys, queries, values)
    if not bidirectional:
        return reads_f, delta_f

    reads_b, delta_b = one_direction(
        keys[:, ::-1], queries[:, ::-1], values[:, ::-1]
    )
    reads_b = reads_b[:, ::-1]
    delta_b = delta_b[:, ::-1]
    return 0.5 * (reads_f + reads_b), 0.5 * (delta_f + delta_b)


class NestedAssociativeBlock(nn.Module):
    model_dim: int
    memory_dim: int
    nested_rank: int
    dropout: float
    residual_scale: float
    nested_qkv_scale: float
    initial_eta: float
    initial_alpha: float
    bidirectional: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool):
        h = nn.LayerNorm(name="input_norm")(x)

        q1 = nn.Dense(
            self.memory_dim, use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(), name="q1"
        )(h)
        k1 = nn.Dense(
            self.memory_dim, use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(), name="k1"
        )(h)
        v1 = nn.Dense(
            self.memory_dim, use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(), name="v1"
        )(h)
        q1, k1 = stable_l2(q1), stable_l2(k1)

        eta1 = jax.nn.sigmoid(self.param(
            "eta1_logit",
            lambda key, shape: jnp.full(shape, logit(self.initial_eta)),
            (1,),
        ))[0]
        alpha1 = jax.nn.sigmoid(self.param(
            "alpha1_logit",
            lambda key, shape: jnp.full(shape, logit(self.initial_alpha)),
            (1,),
        ))[0]

        reads1, delta1 = associative_scan(
            k1, q1, v1, eta1, alpha1, bidirectional=self.bidirectional
        )

        control = nn.LayerNorm(name="nested_controller_norm")(reads1)
        control = nn.Dense(
            3 * self.nested_rank,
            kernel_init=nn.initializers.xavier_uniform(),
            name="nested_controller",
        )(control)
        control = jnp.tanh(control)
        cq, ck, cv = jnp.split(control, 3, axis=-1)

        def low_rank_delta(prefix: str, ctrl: jnp.ndarray):
            low = nn.Dense(
                self.nested_rank,
                use_bias=False,
                kernel_init=nn.initializers.xavier_uniform(),
                name=f"{prefix}_down",
            )(h)
            return nn.Dense(
                self.memory_dim,
                use_bias=False,
                kernel_init=nn.initializers.zeros,
                name=f"{prefix}_up",
            )(low * ctrl)

        q_delta = low_rank_delta("q", cq)
        k_delta = low_rank_delta("k", ck)
        v_delta = low_rank_delta("v", cv)

        nested_gate = jax.nn.sigmoid(
            self.param("nested_gate_logit", nn.initializers.zeros, (1,))
        )
        nested_scale = self.nested_qkv_scale * nested_gate

        q2 = stable_l2(q1 + nested_scale * q_delta)
        k2 = stable_l2(k1 + nested_scale * k_delta)
        v2 = v1 + nested_scale * v_delta

        eta2 = jax.nn.sigmoid(self.param(
            "eta2_logit",
            lambda key, shape: jnp.full(shape, logit(self.initial_eta)),
            (1,),
        ))[0]
        alpha2 = jax.nn.sigmoid(self.param(
            "alpha2_logit",
            lambda key, shape: jnp.full(shape, logit(self.initial_alpha)),
            (1,),
        ))[0]

        reads2, delta2 = associative_scan(
            k2, q2, v2, eta2, alpha2, bidirectional=self.bidirectional
        )

        context = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="readout",
        )(reads2)
        context = nn.Dropout(rate=self.dropout, name="context_dropout")(
            context, deterministic=not training
        )

        memory_gate = jax.nn.sigmoid(
            self.param("memory_gate_logit", nn.initializers.zeros, (1,))
        )
        y = nn.LayerNorm(name="residual_norm")(
            x + self.residual_scale * memory_gate * context
        )

        ff = nn.LayerNorm(name="ff_norm")(y)
        ff = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="ff_in",
        )(ff)
        ff = nn.gelu(ff)
        ff = nn.Dropout(rate=self.dropout, name="ff_dropout")(
            ff, deterministic=not training
        )
        ff = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.zeros,
            name="ff_out",
        )(ff)
        ff_gate = jax.nn.sigmoid(
            self.param("ff_gate_logit", nn.initializers.zeros, (1,))
        )
        out = nn.LayerNorm(name="output_norm")(y + ff_gate * ff)
        return out, 0.5 * (delta1 + delta2), memory_gate, nested_gate


class SlowControllerBlock(nn.Module):
    model_dim: int
    rank: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool):
        h = nn.LayerNorm(name="norm")(x)
        u = nn.Dense(
            self.rank,
            kernel_init=nn.initializers.xavier_uniform(),
            name="down",
        )(h)
        u = nn.gelu(u)
        u = nn.Dropout(rate=self.dropout, name="dropout")(
            u, deterministic=not training
        )
        u = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.zeros,
            name="up",
        )(u)
        gate = jax.nn.sigmoid(
            self.param("gate_logit", nn.initializers.zeros, (1,))
        )
        return nn.LayerNorm(name="out_norm")(x + gate * u), gate


def temporal_pool(x: jnp.ndarray, group: int) -> jnp.ndarray:
    b, t, j, d = x.shape
    if t % group:
        raise ValueError(f"T={t} not divisible by group={group}")
    return x.reshape(b, t // group, group, j, d).mean(axis=2)


class NativeNestSARReframe(nn.Module):
    num_classes: int
    joints: int
    descriptor_dim: int
    model_dim: int
    memory_dim: int
    nested_rank: int
    controller_rank: int
    spatial_blocks: int
    frame_blocks: int
    chunk_blocks: int
    clip_blocks: int
    controller_blocks: int
    chunk_size: int
    clip_size: int
    dropout: float
    residual_scale: float
    nested_qkv_scale: float
    initial_eta: float
    initial_alpha: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool):
        if x.ndim != 4 or x.shape[2] != self.joints:
            raise ValueError(f"Expected [B,T,{self.joints},F], got {x.shape}")

        h = nn.Dense(
            self.model_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="joint_stem",
        )(x)
        joint_embed = self.param(
            "joint_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.joints, self.model_dim),
        )
        h = nn.LayerNorm(name="stem_norm")(h + joint_embed)
        h = nn.gelu(h)

        deltas = []
        mem_gates = []
        nested_gates = []

        def spatial_stage(z, count: int, prefix: str):
            for i in range(count):
                b, t, j, d = z.shape
                flat = z.reshape(b * t, j, d)
                flat, delta, mg, ng = NestedAssociativeBlock(
                    model_dim=self.model_dim,
                    memory_dim=self.memory_dim,
                    nested_rank=self.nested_rank,
                    dropout=self.dropout,
                    residual_scale=self.residual_scale,
                    nested_qkv_scale=self.nested_qkv_scale,
                    initial_eta=self.initial_eta,
                    initial_alpha=self.initial_alpha,
                    bidirectional=True,
                    name=f"{prefix}_spatial_{i+1}",
                )(flat, training)
                z = flat.reshape(b, t, j, d)
                deltas.append(jnp.mean(delta))
                mem_gates.append(jnp.mean(mg))
                nested_gates.append(jnp.mean(ng))
            return z

        def temporal_stage(z, count: int, prefix: str):
            for i in range(count):
                b, t, j, d = z.shape
                flat = jnp.transpose(z, (0, 2, 1, 3)).reshape(b * j, t, d)
                flat, delta, mg, ng = NestedAssociativeBlock(
                    model_dim=self.model_dim,
                    memory_dim=self.memory_dim,
                    nested_rank=self.nested_rank,
                    dropout=self.dropout,
                    residual_scale=self.residual_scale,
                    nested_qkv_scale=self.nested_qkv_scale,
                    initial_eta=self.initial_eta,
                    initial_alpha=self.initial_alpha,
                    bidirectional=False,
                    name=f"{prefix}_temporal_{i+1}",
                )(flat, training)
                z = jnp.transpose(flat.reshape(b, j, t, d), (0, 2, 1, 3))
                deltas.append(jnp.mean(delta))
                mem_gates.append(jnp.mean(mg))
                nested_gates.append(jnp.mean(ng))
            return z

        h = spatial_stage(h, self.spatial_blocks, "l1")
        h = temporal_stage(h, self.frame_blocks, "l1")

        h = temporal_pool(h, self.chunk_size)
        h = spatial_stage(h, self.spatial_blocks, "l2")
        h = temporal_stage(h, self.chunk_blocks, "l2")

        clip_group = self.clip_size // self.chunk_size
        h = temporal_pool(h, clip_group)
        h = spatial_stage(h, self.spatial_blocks, "l3")
        h = temporal_stage(h, self.clip_blocks, "l3")

        for i in range(self.controller_blocks):
            h, gate = SlowControllerBlock(
                model_dim=self.model_dim,
                rank=self.controller_rank,
                dropout=self.dropout,
                name=f"l4_controller_{i+1}",
            )(h, training)
            mem_gates.append(jnp.mean(gate))

        h = nn.Dropout(rate=self.dropout, name="final_dropout")(
            h, deterministic=not training
        )
        pooled = jnp.mean(h, axis=(1, 2))
        pooled = nn.LayerNorm(name="classifier_norm")(pooled)
        logits = nn.Dense(
            self.num_classes,
            kernel_init=nn.initializers.xavier_uniform(),
            name="classifier",
        )(pooled)

        return {
            "logits": logits,
            "memory_delta": sum(deltas) / max(1, len(deltas)),
            "memory_gate": sum(mem_gates) / max(1, len(mem_gates)),
            "nested_gate": sum(nested_gates) / max(1, len(nested_gates)),
        }


# =============================================================================
# TRAINING / ALL-DEVICE DATA PARALLEL
# =============================================================================

class TrainState(train_state.TrainState):
    pass


def build_model() -> NativeNestSARReframe:
    return NativeNestSARReframe(
        num_classes=CFG.num_classes,
        joints=CFG.joints,
        descriptor_dim=CFG.persons * CFG.coords * 2,
        model_dim=CFG.model_dim,
        memory_dim=CFG.memory_dim,
        nested_rank=CFG.nested_rank,
        controller_rank=CFG.controller_rank,
        spatial_blocks=CFG.spatial_blocks,
        frame_blocks=CFG.frame_blocks,
        chunk_blocks=CFG.chunk_blocks,
        clip_blocks=CFG.clip_blocks,
        controller_blocks=CFG.controller_blocks,
        chunk_size=CFG.chunk_size,
        clip_size=CFG.clip_size,
        dropout=CFG.dropout,
        residual_scale=CFG.memory_residual_scale,
        nested_qkv_scale=CFG.nested_qkv_scale,
        initial_eta=CFG.initial_eta,
        initial_alpha=CFG.initial_alpha,
    )


def make_schedule(total_steps: int):
    warm = max(1, int(total_steps * CFG.warmup_fraction))
    decay = max(1, total_steps - warm)
    s1 = optax.linear_schedule(
        init_value=CFG.learning_rate * 0.05,
        end_value=CFG.learning_rate,
        transition_steps=warm,
    )
    alpha = CFG.min_learning_rate / max(CFG.learning_rate, 1e-12)
    s2 = optax.cosine_decay_schedule(
        init_value=CFG.learning_rate,
        decay_steps=decay,
        alpha=alpha,
    )
    return optax.join_schedules([s1, s2], [warm])


def classification_loss(logits: jnp.ndarray, labels: jnp.ndarray):
    onehot = jax.nn.one_hot(labels, CFG.num_classes)
    smooth = onehot * (1.0 - CFG.label_smoothing) + CFG.label_smoothing / CFG.num_classes
    return optax.softmax_cross_entropy(logits, smooth).mean()


def create_state(model, rng, total_steps: int):
    dummy = jnp.zeros(
        (2, CFG.train_frames, CFG.joints, CFG.persons * CFG.coords * 2),
        dtype=jnp.float32,
    )
    variables = model.init({"params": rng, "dropout": rng}, dummy, training=True)
    tx = optax.chain(
        optax.clip_by_global_norm(CFG.grad_clip),
        optax.adamw(
            learning_rate=make_schedule(total_steps),
            weight_decay=CFG.weight_decay,
        ),
    )
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)


def audit_reframe_parameter_invariance(model, params) -> None:
    sig_ref = tree_shape_signature(params)
    count_ref = tree_param_count(params)
    for t in sorted(set((CFG.train_frames,) + CFG.eval_frames)):
        dummy = jnp.zeros(
            (1, t, CFG.joints, CFG.persons * CFG.coords * 2),
            dtype=jnp.float32,
        )
        key = jax.random.PRNGKey(CFG.seed)
        p_t = model.init(
            {"params": key, "dropout": key}, dummy, training=False
        )["params"]
        if tree_shape_signature(p_t) != sig_ref:
            raise RuntimeError(
                f"Reframe parameter-invariance FAILED at T={t}: parameter tree depends on frame count."
            )
        if tree_param_count(p_t) != count_ref:
            raise RuntimeError(f"Parameter count changed at T={t}")
    log(
        "REFRAME PARAMETER INVARIANCE: PASS | "
        f"same {count_ref:,} params for T={sorted(set((CFG.train_frames,) + CFG.eval_frames))}"
    )


def make_pmapped_steps(model):
    @partial(jax.pmap, axis_name="devices", donate_argnums=(0,))
    def train_step(state, x, y, dropout_rng):
        def loss_fn(params):
            out = model.apply(
                {"params": params}, x, training=True,
                rngs={"dropout": dropout_rng},
            )
            ce = classification_loss(out["logits"], y)
            acc = jnp.mean(jnp.argmax(out["logits"], axis=-1) == y)
            return ce, {
                "loss": ce,
                "accuracy": acc,
                "memory_delta": out["memory_delta"],
                "memory_gate": out["memory_gate"],
                "nested_gate": out["nested_gate"],
            }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name="devices")
        metrics = jax.lax.pmean(metrics, axis_name="devices")
        state = state.apply_gradients(grads=grads)
        metrics["grad_norm"] = optax.global_norm(grads)
        return state, metrics

    @partial(jax.pmap, axis_name="devices")
    def eval_step(params, x, y, mask):
        out = model.apply({"params": params}, x, training=False)
        pred = jnp.argmax(out["logits"], axis=-1)
        correct_local = jnp.sum((pred == y) * mask)
        count_local = jnp.sum(mask)
        loss_each = optax.softmax_cross_entropy_with_integer_labels(out["logits"], y)
        loss_local = jnp.sum(loss_each * mask)
        return {
            "correct": jax.lax.psum(correct_local, "devices"),
            "count": jax.lax.psum(count_local, "devices"),
            "loss_sum": jax.lax.psum(loss_local, "devices"),
            "memory_delta": jax.lax.pmean(out["memory_delta"], "devices"),
            "memory_gate": jax.lax.pmean(out["memory_gate"], "devices"),
            "nested_gate": jax.lax.pmean(out["nested_gate"], "devices"),
        }

    return train_step, eval_step


def shard_batch(x: np.ndarray, y: np.ndarray, ndev: int):
    if len(x) % ndev:
        raise ValueError("Global batch must be divisible by visible device count")
    local = len(x) // ndev
    return x.reshape(ndev, local, *x.shape[1:]), y.reshape(ndev, local)


def pad_and_shard_eval(x: np.ndarray, y: np.ndarray, eval_batch: int, ndev: int):
    n = len(x)
    if n > eval_batch:
        raise ValueError("Internal eval batch larger than configured eval_batch")
    if eval_batch % ndev:
        raise ValueError("eval_batch must be divisible by visible device count")
    if n < eval_batch:
        pad_n = eval_batch - n
        x = np.concatenate([x, np.zeros((pad_n, *x.shape[1:]), dtype=x.dtype)], axis=0)
        y = np.concatenate([y, np.zeros((pad_n,), dtype=y.dtype)], axis=0)
    mask = np.zeros((eval_batch,), dtype=np.float32)
    mask[:n] = 1.0
    local = eval_batch // ndev
    return (
        x.reshape(ndev, local, *x.shape[1:]),
        y.reshape(ndev, local),
        mask.reshape(ndev, local),
    )


def evaluate(params_repl, eval_step, samples, frames: int, ndev: int) -> Dict[str, float]:
    ds = SkeletonDataset(samples, frames=frames, cache=False)
    total_correct = total_count = total_loss = 0.0
    md = mg = ng = 0.0
    batches = 0
    for x, y in batch_iterator(
        ds, CFG.eval_batch, shuffle=False, seed=0, drop_last=False
    ):
        xs, ys, mask = pad_and_shard_eval(x, y, CFG.eval_batch, ndev)
        out = eval_step(
            params_repl, jnp.asarray(xs), jnp.asarray(ys), jnp.asarray(mask)
        )
        total_correct += float(np.asarray(out["correct"][0]))
        total_count += float(np.asarray(out["count"][0]))
        total_loss += float(np.asarray(out["loss_sum"][0]))
        md += float(np.asarray(out["memory_delta"][0]))
        mg += float(np.asarray(out["memory_gate"][0]))
        ng += float(np.asarray(out["nested_gate"][0]))
        batches += 1
    return {
        "frames": frames,
        "accuracy": 100.0 * total_correct / max(1.0, total_count),
        "loss": total_loss / max(1.0, total_count),
        "correct": int(total_correct),
        "count": int(total_count),
        "memory_delta": md / max(1, batches),
        "memory_gate": mg / max(1, batches),
        "nested_gate": ng / max(1, batches),
    }


def save_params(path: Path, params) -> None:
    path.write_bytes(serialization.to_bytes(params))

def load_params(path: Path, params_template):
    return serialization.from_bytes(params_template, path.read_bytes())


# =============================================================================
# MAIN
# =============================================================================

def main():
    validate_config()
    seed_everything(CFG.seed)

    devices = jax.local_devices()
    ndev = len(devices)
    if ndev < 1:
        raise RuntimeError("No JAX devices visible")
    if CFG.global_batch % ndev:
        raise ValueError(
            f"NESTSAR_GLOBAL_BATCH={CFG.global_batch} must be divisible by {ndev} devices"
        )
    if CFG.eval_batch % ndev:
        raise ValueError(
            f"NESTSAR_EVAL_BATCH={CFG.eval_batch} must be divisible by {ndev} devices"
        )

    outdir = Path(CFG.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    best_path = outdir / "best_params.msgpack"
    last_path = outdir / "last_state.msgpack"
    meta_path = outdir / "run_config.json"
    curve_path = outdir / "curve.jsonl"

    dataset_path = find_dataset()
    data = load_pickle(dataset_path)
    train_samples, val_samples, train_key, val_key = extract_splits(data, CFG.protocol)
    train_ds = SkeletonDataset(
        train_samples, frames=CFG.train_frames, cache=CFG.cache_train
    )

    steps_per_epoch = len(train_ds) // CFG.global_batch
    if steps_per_epoch < 1:
        raise RuntimeError("Training split smaller than global batch")
    total_steps = CFG.epochs * steps_per_epoch

    model = build_model()
    rng = jax.random.PRNGKey(CFG.seed)
    state = create_state(model, rng, total_steps)
    params_count = tree_param_count(state.params)

    audit_reframe_parameter_invariance(model, state.params)

    log("=" * 108)
    log("NESTSAR NATIVE-REFRAME v1")
    log("=" * 108)
    log(f"Backend:               {jax.default_backend()}")
    log(f"Visible JAX devices:   {ndev} | {[str(d) for d in devices]}")
    log(f"Active devices:        {ndev}/{ndev} via pmap + gradient pmean")
    log(f"Protocol:              {CFG.protocol.upper()} ({train_key}/{val_key})")
    log(f"Dataset:               {dataset_path}")
    log(f"Train/val:             {len(train_samples):,}/{len(val_samples):,}")
    log(f"Train frames:          {CFG.train_frames}")
    log(f"Reframe eval windows:  {CFG.eval_frames}")
    log("Temporal embedding:    NONE")
    log("Softmax attention:     NONE")
    log("Transformer:           NONE")
    log("GCN/CNN/TCN:           NONE")
    log("Spatial interaction:   bidirectional nested associative memory")
    log("Temporal interaction:  causal nested associative memory")
    log("QKV mechanism:         H3-style low-rank adaptation between fast memories")
    log(f"Hierarchy:             T -> T/{CFG.chunk_size} -> T/{CFG.clip_size} -> L4")
    log(f"Parameters:            {params_count:,}")
    log(f"Global/local batch:    {CFG.global_batch}/{CFG.global_batch // ndev}")
    log(f"Eval global/local:     {CFG.eval_batch}/{CFG.eval_batch // ndev}")
    log(f"Epochs/steps:          {CFG.epochs}/{total_steps:,}")
    log("=" * 108)

    meta_path.write_text(json.dumps(dataclasses.asdict(CFG), indent=2), encoding="utf-8")

    start_epoch = 1
    best_acc = -1.0
    patience = 0
    if CFG.resume and last_path.is_file():
        template = {
            "state": state,
            "epoch": np.int32(0),
            "best_acc": np.float32(-1.0),
            "patience": np.int32(0),
        }
        restored = serialization.from_bytes(template, last_path.read_bytes())
        state = restored["state"]
        start_epoch = int(restored["epoch"]) + 1
        best_acc = float(restored["best_acc"])
        patience = int(restored["patience"])
        log(f"Resume: epoch={start_epoch}, best={best_acc:.4f}% patience={patience}")

    train_step, eval_step = make_pmapped_steps(model)
    state_repl = jax.device_put_replicated(state, devices)

    for epoch in range(start_epoch, CFG.epochs + 1):
        t0 = time.time()
        losses, accs, memd, memg, nestg, gradn = [], [], [], [], [], []

        for step, (x, y) in enumerate(
            batch_iterator(
                train_ds,
                CFG.global_batch,
                shuffle=True,
                seed=CFG.seed + epoch,
                drop_last=True,
            ),
            start=1,
        ):
            xs, ys = shard_batch(x, y, ndev)
            keys = jax.random.split(
                jax.random.fold_in(rng, epoch * 1_000_000 + step), ndev
            )
            state_repl, metrics = train_step(
                state_repl, jnp.asarray(xs), jnp.asarray(ys), keys
            )
            losses.append(float(np.asarray(metrics["loss"][0])))
            accs.append(float(np.asarray(metrics["accuracy"][0])))
            memd.append(float(np.asarray(metrics["memory_delta"][0])))
            memg.append(float(np.asarray(metrics["memory_gate"][0])))
            nestg.append(float(np.asarray(metrics["nested_gate"][0])))
            gradn.append(float(np.asarray(metrics["grad_norm"][0])))

        val = evaluate(state_repl.params, eval_step, val_samples, CFG.train_frames, ndev)
        train_acc = 100.0 * float(np.mean(accs))
        log(
            f"E{epoch:03d} | train loss={np.mean(losses):.5f} acc={train_acc:.3f}% | "
            f"val T{CFG.train_frames}={val['accuracy']:.5f}% loss={val['loss']:.5f} | "
            f"memD={np.mean(memd):.5f} mem_gate={np.mean(memg):.4f} "
            f"nested_gate={np.mean(nestg):.4f} grad={np.mean(gradn):.4f} | "
            f"{(time.time()-t0)/60.0:.2f} min"
        )

        host_state = jax.tree_util.tree_map(lambda z: z[0], state_repl)
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            patience = 0
            save_params(best_path, host_state.params)
            log(f"BEST -> {best_acc:.5f}% | saved {best_path}")
        else:
            patience += 1

        last_payload = {
            "state": host_state,
            "epoch": np.int32(epoch),
            "best_acc": np.float32(best_acc),
            "patience": np.int32(patience),
        }
        last_path.write_bytes(serialization.to_bytes(last_payload))

        with curve_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_accuracy": train_acc,
                "val": val,
                "best_accuracy": best_acc,
                "patience": patience,
            }) + "\n")

        if patience >= CFG.patience:
            log(f"Early stop: patience {patience}/{CFG.patience}")
            break

    host_state = jax.tree_util.tree_map(lambda z: z[0], state_repl)
    best_params = (
        load_params(best_path, host_state.params) if best_path.is_file() else host_state.params
    )
    best_repl = jax.device_put_replicated(best_params, devices)

    log("=" * 108)
    log("FINAL NATIVE REFRAME EVALUATION — SAME PARAMETER TREE, NO RETRAINING")
    log("=" * 108)
    final = {}
    for t in CFG.eval_frames:
        gc.collect()
        metrics = evaluate(best_repl, eval_step, val_samples, t, ndev)
        final[str(t)] = metrics
        log(
            f"T{t:02d}: acc={metrics['accuracy']:.5f}% "
            f"({metrics['correct']}/{metrics['count']}) loss={metrics['loss']:.5f}"
        )

    summary = {
        "model": "NestSAR-Native-Reframe-v1",
        "protocol": CFG.protocol,
        "parameters": params_count,
        "train_frames": CFG.train_frames,
        "eval_frames": list(CFG.eval_frames),
        "same_checkpoint_all_windows": True,
        "temporal_embedding": False,
        "softmax_attention": False,
        "transformer": False,
        "spatial_memory": "bidirectional nested associative",
        "temporal_memory": "causal nested associative",
        "best_train_window_val_accuracy": best_acc,
        "reframe_results": final,
    }
    (outdir / "final_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log(f"Done. Summary: {outdir / 'final_summary.json'}")


if __name__ == "__main__":
    main()

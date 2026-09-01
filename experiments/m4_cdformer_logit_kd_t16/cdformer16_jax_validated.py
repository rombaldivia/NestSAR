#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

D_MODEL = 192
NUM_HEADS = 8
HEAD_DIM = D_MODEL // NUM_HEADS
NUM_LAYERS = 12
FFN_DIM = 2048
NUM_CLASSES = 120
NUM_JOINTS = 25
NUM_FRAMES = 16
IN_CHANNELS = 3


def load_cdformer16_weights(npz_path: str | Path, *, dtype=jnp.float32) -> Dict[str, jax.Array]:
    npz_path = Path(npz_path).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    z = np.load(npz_path, allow_pickle=False)
    weights = {key: jnp.asarray(z[key], dtype=dtype) for key in z.files}
    if "cls_token" not in weights and "cls" in weights:
        weights["cls_token"] = weights["cls"]

    required = [
        "embedding.weight", "embedding.bias", "cls_token", "temb", "jemb",
        "norm.weight", "norm.bias", "head.weight", "head.bias",
    ]
    for i in range(NUM_LAYERS):
        p = f"enc.layers.{i}."
        required.extend([
            p + "self_attn.in_proj_weight", p + "self_attn.in_proj_bias",
            p + "self_attn.out_proj.weight", p + "self_attn.out_proj.bias",
            p + "linear1.weight", p + "linear1.bias",
            p + "linear2.weight", p + "linear2.bias",
            p + "norm1.weight", p + "norm1.bias",
            p + "norm2.weight", p + "norm2.bias",
        ])
    missing = [k for k in required if k not in weights]
    if missing:
        raise KeyError("Missing CD-Former tensors:\n  " + "\n  ".join(missing))

    expected_shapes = {
        "embedding.weight": (192, 3), "embedding.bias": (192,),
        "cls_token": (1, 1, 192), "temb": (1, 16, 192), "jemb": (1, 25, 192),
        "norm.weight": (192,), "norm.bias": (192,),
        "head.weight": (120, 192), "head.bias": (120,),
    }
    for key, shape in expected_shapes.items():
        actual = tuple(weights[key].shape)
        if actual != shape:
            raise RuntimeError(f"{key}: expected {shape}, got {actual}")

    total = sum(int(x.size) for x in weights.values())
    if total != 11_284_344:
        raise RuntimeError(f"Expected 11,284,344 parameters, got {total:,}")
    return weights


def linear(x, weight_out_in, bias):
    return jnp.matmul(x, weight_out_in.T) + bias


def layer_norm(x, weight, bias, eps: float = 1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normalized = (x - mean) * jax.lax.rsqrt(variance + eps)
    return normalized * weight + bias


def multihead_self_attention(x, weights, layer_index: int):
    p = f"enc.layers.{layer_index}."
    qkv = linear(x, weights[p + "self_attn.in_proj_weight"], weights[p + "self_attn.in_proj_bias"])
    q, k, v = jnp.split(qkv, 3, axis=-1)
    batch, length, _ = q.shape

    def split_heads(t):
        return t.reshape(batch, length, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    scale = jnp.asarray(1.0 / math.sqrt(HEAD_DIM), dtype=x.dtype)
    scores = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
    attention = jax.nn.softmax(scores, axis=-1)
    context = jnp.einsum("bhij,bhjd->bhid", attention, v)
    context = context.transpose(0, 2, 1, 3).reshape(batch, length, D_MODEL)
    return linear(context, weights[p + "self_attn.out_proj.weight"], weights[p + "self_attn.out_proj.bias"])


def cdformer16_forward(weights: Dict[str, jax.Array], x: jax.Array) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected [B,T,V,C], got {x.shape}")
    batch, frames, joints, channels = x.shape
    if frames != NUM_FRAMES or joints != NUM_JOINTS or channels != IN_CHANNELS:
        raise ValueError(f"Validated configuration requires [B,16,25,3], got {x.shape}")

    x = linear(x, weights["embedding.weight"], weights["embedding.bias"])
    x = x + weights["temb"][:, :, None, :] + weights["jemb"][:, None, :, :]
    x = x.reshape(batch, frames * joints, D_MODEL)
    cls = jnp.broadcast_to(weights["cls_token"], (batch, 1, D_MODEL))
    x = jnp.concatenate([cls, x], axis=1)
    x = layer_norm(x, weights["norm.weight"], weights["norm.bias"])

    for i in range(NUM_LAYERS):
        p = f"enc.layers.{i}."
        attn_out = multihead_self_attention(x, weights, i)
        x = layer_norm(x + attn_out, weights[p + "norm1.weight"], weights[p + "norm1.bias"])
        ff = linear(x, weights[p + "linear1.weight"], weights[p + "linear1.bias"])
        ff = jax.nn.relu(ff)
        ff = linear(ff, weights[p + "linear2.weight"], weights[p + "linear2.bias"])
        x = layer_norm(x + ff, weights[p + "norm2.weight"], weights[p + "norm2.bias"])

    cls_feature = x[:, 0]
    token_mean = jnp.mean(x[:, 1:], axis=1)
    feature = 0.5 * (cls_feature + token_mean)
    return linear(feature, weights["head.weight"], weights["head.bias"])


def make_cdformer16_jit(weights):
    return jax.jit(lambda x: cdformer16_forward(weights, x))

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization
from flax.training import train_state
import optax
from tqdm.auto import tqdm

from experiments.m4_motionpreserve_t16.jax10_compat import install as install_jax10_compat

install_jax10_compat()

PARENTS = np.asarray([
    0, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0,
    12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11,
], dtype=np.int32)

JOINT_ORDER = np.asarray([
    0, 1, 20, 2, 3,
    4, 5, 6, 7, 21, 22,
    8, 9, 10, 11, 23, 24,
    12, 13, 14, 15,
    16, 17, 18, 19,
], dtype=np.int32)

TEN_PARTS = (
    (0, 1, 20),
    (2, 3),
    (4, 5),
    (6, 7, 21, 22),
    (8, 9),
    (10, 11, 23, 24),
    (12, 13),
    (14, 15),
    (16, 17),
    (18, 19),
)

PART_MASK_NP = np.zeros((10, 25), np.float32)
for p, joints in enumerate(TEN_PARTS):
    PART_MASK_NP[p, list(joints)] = 1.0
PART_COUNTS_NP = np.maximum(PART_MASK_NP.sum(axis=1), 1.0)

NUM_CLASSES = 120
FRAMES = 16
PERSONS = 2
JOINTS = 25
XYZ = 3
TOKEN_CHANNELS = 9  # pose xyz + net displacement xyz + path-motion xyz
FEATURES = PERSONS * JOINTS * TOKEN_CHANNELS  # 450
NUM_STREAMS = 4


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_tmvc(keypoints: np.ndarray) -> np.ndarray:
    x = np.asarray(keypoints, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, None, :, :]
    elif x.ndim == 4:
        if x.shape[0] <= 4 and x.shape[1] > 4:
            x = np.transpose(x, (1, 0, 2, 3))
        elif x.shape[1] <= 4 and x.shape[0] > 4:
            pass
        elif x.shape[0] < x.shape[1]:
            x = np.transpose(x, (1, 0, 2, 3))
    else:
        raise ValueError(f"Unsupported keypoint shape {x.shape}")
    if x.shape[-1] == 2:
        x = np.concatenate(
            [x, np.zeros((*x.shape[:-1], 1), np.float32)], axis=-1
        )
    return x[..., :3]


def canonicalize_raw(keypoints: np.ndarray) -> np.ndarray:
    x = to_tmvc(keypoints)
    if x.shape[2] < JOINTS:
        pad = np.zeros(
            (x.shape[0], x.shape[1], JOINTS - x.shape[2], XYZ), np.float32
        )
        x = np.concatenate([x, pad], axis=2)
    x = x[:, :, :JOINTS, :XYZ]
    person_energy = np.sum(np.abs(x), axis=(0, 2, 3))
    x = x[:, np.argsort(-person_energy)]
    if x.shape[1] < PERSONS:
        x = np.concatenate(
            [
                x,
                np.zeros(
                    (x.shape[0], PERSONS - x.shape[1], JOINTS, XYZ),
                    np.float32,
                ),
            ],
            axis=1,
        )
    x = x[:, :PERSONS]

    # Match the established NestSAR preprocessing convention: person-0 joint-0
    # is the sequence reference, applied frame-wise to both persons.
    center = x[:, 0:1, 0:1, :]
    valid_center = np.any(np.abs(center) > 1e-8, axis=-1, keepdims=True)
    x = np.where(valid_center, x - center, x)
    valid_joint = np.any(np.abs(x) > 1e-8, axis=-1, keepdims=True)
    return np.where(valid_joint, x, 0.0).astype(np.float32)


def uniform_indices(total: int, n: int = FRAMES) -> np.ndarray:
    if total <= 1:
        return np.zeros((n,), np.int64)
    return np.linspace(0, total - 1, n, dtype=np.float32).round().astype(np.int64)


def segment_bounds(total: int, n: int = FRAMES) -> list[tuple[int, int]]:
    """Return n non-empty [start,end) segments, repeating frames when total<n."""
    if total <= 0:
        return [(0, 1)] * n
    if total < n:
        idx = uniform_indices(total, n)
        return [(int(i), int(i) + 1) for i in idx]

    edges = np.linspace(0, total, n + 1, dtype=np.float64)
    out: list[tuple[int, int]] = []
    for i in range(n):
        s = int(np.floor(edges[i]))
        e = int(np.floor(edges[i + 1]))
        s = min(max(s, 0), total - 1)
        e = min(max(e, s + 1), total)
        out.append((s, e))
    return out


def segment_motion_tokens(keypoints: np.ndarray) -> np.ndarray:
    """Compress the full sequence into 16 motion-preserving temporal tokens.

    Each token stores, per person/joint:
      0:3  representative center-frame pose
      3:6  net displacement across the segment
      6:9  accumulated absolute path motion across the segment

    The NN still processes exactly 16 temporal tokens, but each token summarizes
    all raw frames inside its segment instead of discarding the intermediate path.
    """
    x = canonicalize_raw(keypoints)
    total = x.shape[0]
    if total <= 0:
        return np.zeros((FRAMES, FEATURES), np.float32)

    tokens = np.zeros((FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS), np.float32)
    for i, (s, e) in enumerate(segment_bounds(total, FRAMES)):
        seg = x[s:e]
        mid = (len(seg) - 1) // 2
        pose = seg[mid]
        if len(seg) >= 2:
            d = seg[1:] - seg[:-1]
            net = seg[-1] - seg[0]
            path = np.sum(np.abs(d), axis=0)
        else:
            net = np.zeros_like(pose)
            path = np.zeros_like(pose)
        tokens[i, ..., 0:3] = pose
        tokens[i, ..., 3:6] = net
        tokens[i, ..., 6:9] = path

    # One geometric scale from the raw centered poses is used for all channels so
    # the relative magnitude of displacement/path versus pose is preserved.
    nz = np.abs(x) > 1e-8
    if np.any(nz):
        rms = float(np.sqrt(np.mean(np.square(x[nz]))) + 1e-6)
        tokens = tokens / rms
    return np.nan_to_num(tokens).reshape(FRAMES, FEATURES).astype(np.float32)


def preprocess_keypoints(keypoints: np.ndarray, selector: str = "segment") -> np.ndarray:
    if selector == "segment":
        return segment_motion_tokens(keypoints)
    if selector == "uniform":
        # Uniform fallback encoded in the same 9-channel token layout. Motion
        # summary channels are intentionally zero for a clean ablation/control.
        x = canonicalize_raw(keypoints)
        idx = uniform_indices(x.shape[0], FRAMES)
        pose = x[idx]
        tokens = np.zeros((FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS), np.float32)
        tokens[..., :3] = pose
        nz = np.abs(x) > 1e-8
        if np.any(nz):
            rms = float(np.sqrt(np.mean(np.square(x[nz]))) + 1e-6)
            tokens = tokens / rms
        return np.nan_to_num(tokens).reshape(FRAMES, FEATURES).astype(np.float32)
    raise ValueError(selector)


def annotation_keypoints(a: Mapping[str, Any]) -> np.ndarray:
    for k in ("keypoint", "keypoints", "skeleton", "skeletons", "data"):
        if k in a:
            return np.asarray(a[k], np.float32)
    raise KeyError(f"No keypoints in annotation keys={list(a)[:20]}")


def annotation_label(a: Mapping[str, Any]) -> int:
    for k in ("label", "action_label", "class", "target"):
        if k in a:
            return int(a[k])
    raise KeyError("No label")


def sample_id(a: Mapping[str, Any], i: int) -> str:
    for k in ("frame_dir", "filename", "sample_name", "name", "id", "video_id"):
        if k in a:
            return str(a[k])
    return str(i)


def load_ntu(path: Path):
    with path.open("rb") as f:
        try:
            data = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            data = pickle.load(f, encoding="latin1")
    anns = None
    for k in ("annotations", "annotation", "samples", "data_list"):
        if isinstance(data.get(k), Sequence):
            anns = list(data[k])
            break
    if anns is None:
        raise KeyError("Could not find annotation list")
    split = data.get("split", data.get("splits"))
    if not isinstance(split, Mapping):
        raise KeyError("Could not find split map")
    return anns, split


def resolve_split(split: Mapping[str, Any], protocol: str):
    lower = {str(k).lower(): k for k in split}
    train_names = [f"{protocol}_train", f"{protocol}train", f"train_{protocol}"]
    val_names = [
        f"{protocol}_val", f"{protocol}_test", f"{protocol}val",
        f"{protocol}test", f"val_{protocol}", f"test_{protocol}",
    ]
    tk = next((lower[k] for k in train_names if k in lower), None)
    vk = next((lower[k] for k in val_names if k in lower), None)
    if tk is None or vk is None:
        raise KeyError(f"Cannot resolve {protocol}; split keys={list(split)}")
    return tk, vk


def build_protocol_arrays(
    annotations,
    split,
    protocol: str,
    selector: str,
    max_train: int = 0,
    max_val: int = 0,
):
    tk, vk = resolve_split(split, protocol)
    by_id = {
        sample_id(a, i): a
        for i, a in enumerate(annotations)
        if isinstance(a, Mapping)
    }
    train_ids = [str(v) for v in split[tk] if str(v) in by_id]
    val_ids = [str(v) for v in split[vk] if str(v) in by_id]
    if max_train:
        train_ids = train_ids[:max_train]
    if max_val:
        val_ids = val_ids[:max_val]
    if not train_ids or not val_ids:
        raise RuntimeError(
            f"Resolved empty {protocol} arrays: train={len(train_ids)} val={len(val_ids)}"
        )

    def materialize(ids, desc):
        X = np.empty((len(ids), FRAMES, FEATURES), np.float32)
        y = np.empty((len(ids),), np.int32)
        for i, sid in enumerate(tqdm(ids, desc=desc, mininterval=0.5)):
            a = by_id[sid]
            X[i] = preprocess_keypoints(annotation_keypoints(a), selector)
            y[i] = annotation_label(a)
        return X, y

    Xtr, ytr = materialize(train_ids, f"{protocol.upper()} preprocess train")
    Xva, yva = materialize(val_ids, f"{protocol.upper()} preprocess val")
    return Xtr, ytr, Xva, yva


class GatedSweep(nn.Module):
    dim: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d = self.dim
        init = nn.initializers.xavier_uniform()
        wz_x = self.param("wz_x", init, (d, d))
        wz_h = self.param("wz_h", init, (d, d))
        bz = self.param("bz", nn.initializers.zeros, (d,))
        wr_x = self.param("wr_x", init, (d, d))
        wr_h = self.param("wr_h", init, (d, d))
        br = self.param("br", nn.initializers.zeros, (d,))
        wc_x = self.param("wc_x", init, (d, d))
        wc_h = self.param("wc_h", init, (d, d))
        bc = self.param("bc", nn.initializers.zeros, (d,))
        xt = jnp.swapaxes(x, 0, 1)
        if self.reverse:
            xt = xt[::-1]
        h0 = jnp.zeros((x.shape[0], d), x.dtype)

        def step(h, token):
            z = jax.nn.sigmoid(token @ wz_x + h @ wz_h + bz)
            r = jax.nn.sigmoid(token @ wr_x + h @ wr_h + br)
            cand = jnp.tanh(token @ wc_x + (r * h) @ wc_h + bc)
            h = (1.0 - z) * h + z * cand
            return h, h

        _, yt = jax.lax.scan(step, h0, xt)
        if self.reverse:
            yt = yt[::-1]
        return jnp.swapaxes(yt, 0, 1)


class BiMemory(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        f = GatedSweep(self.dim, reverse=False, name="fwd")(x)
        b = GatedSweep(self.dim, reverse=True, name="bwd")(x)
        y = nn.Dense(self.dim, name="merge")(jnp.concatenate([f, b], axis=-1))
        return nn.LayerNorm(name="norm")(x + y)


class SpatialEncoder(nn.Module):
    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        b, t, m, _, _ = x.shape
        h = nn.Dense(self.spatial_dim, name="in_proj")(x)
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
        h = nn.gelu(h + je + pe)

        order = jnp.asarray(JOINT_ORDER)
        inv = jnp.argsort(order)
        h = jnp.take(h, order, axis=3)
        h = h.reshape(b * t * m, JOINTS, self.spatial_dim)
        mem = GatedSweep(self.spatial_dim, reverse=False, name="joint_memory")(h)
        h = nn.LayerNorm(name="joint_memory_norm")(h + mem)
        h = h.reshape(b, t, m, JOINTS, self.spatial_dim)
        h = jnp.take(h, inv, axis=3)

        mask = jnp.asarray(PART_MASK_NP, h.dtype)
        counts = jnp.asarray(PART_COUNTS_NP, h.dtype)
        parts = jnp.einsum("btmvd,pv->btmpd", h, mask)
        parts = parts / counts[None, None, None, :, None]
        flat = parts.reshape(b, t, m * 10 * self.spatial_dim)
        y = nn.Dense(self.model_dim, name="part_fuse")(flat)
        y = nn.LayerNorm(name="out_norm")(nn.gelu(y))
        return nn.Dropout(self.dropout)(y, deterministic=not training)


class CrossStreamRouter(nn.Module):
    dim: int = 112
    residual_scale: float = 0.15

    @nn.compact
    def __call__(self, streams: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # streams: [B,T,S,D], deliberately AFTER each stream's frame memory.
        n = nn.LayerNorm(name="router_norm")(streams)
        scores = nn.Dense(1, name="score")(n)[..., 0]
        weights = jax.nn.softmax(scores, axis=2)
        context = jnp.sum(weights[..., None] * n, axis=2)
        delta = nn.Dense(self.dim, name="context_proj")(context)
        gate = jax.nn.sigmoid(nn.Dense(1, name="gate")(n))
        out = streams + self.residual_scale * gate * delta[:, :, None, :]
        return out, weights


class DescriptorHead(nn.Module):
    dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, frame_h: jnp.ndarray, training: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
        chunks = frame_h.reshape(frame_h.shape[0], 4, 4, self.dim).mean(axis=2)
        chunks = BiMemory(self.dim, name="chunk_memory")(chunks)
        pooled = jnp.concatenate(
            [frame_h.mean(axis=1), chunks.mean(axis=1)], axis=-1
        )
        pooled = nn.Dense(self.dim, name="hier_fuse")(pooled)
        pooled = nn.LayerNorm(name="hier_norm")(nn.gelu(pooled))
        pooled = nn.Dropout(self.dropout)(pooled, deterministic=not training)
        return chunks, pooled


class M4MotionPreserveT16(nn.Module):
    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Mapping[str, jnp.ndarray]:
        tok = x.reshape(x.shape[0], FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS)
        pose = tok[..., 0:3]
        disp = tok[..., 3:6]
        path = tok[..., 6:9]

        # Pose streams.
        joint = pose
        parents = jnp.asarray(PARENTS)
        bone = joint - jnp.take(joint, parents, axis=3)

        # Motion streams are full-segment summaries, not sparse-frame finite differences.
        joint_motion = jnp.concatenate([disp, path], axis=-1)
        parent_disp = jnp.take(disp, parents, axis=3)
        parent_path = jnp.take(path, parents, axis=3)
        bone_disp = disp - parent_disp
        bone_path = jnp.abs(path - parent_path)
        bone_motion = jnp.concatenate([bone_disp, bone_path], axis=-1)

        raw_streams = (joint, bone, joint_motion, bone_motion)
        spatial = []
        for i, s in enumerate(raw_streams):
            spatial.append(
                SpatialEncoder(
                    self.spatial_dim,
                    self.model_dim,
                    self.dropout,
                    name=f"spatial_{i}",
                )(s, training)
            )

        # Audit-driven change: temporalize each stream BEFORE cross-stream mixing.
        frame_streams = []
        for i, s in enumerate(spatial):
            frame_streams.append(
                BiMemory(self.model_dim, name=f"frame_memory_{i}")(s)
            )
        frame_stack = jnp.stack(frame_streams, axis=2)  # [B,T,4,D]
        mixed, router_weights = CrossStreamRouter(
            self.model_dim, name="cross_stream_after_frame"
        )(frame_stack)

        descriptors = []
        stream_logits = []
        chunk_states = []
        for i in range(NUM_STREAMS):
            chunks, desc = DescriptorHead(
                self.model_dim,
                self.dropout,
                name=f"descriptor_{i}",
            )(mixed[:, :, i], training)
            descriptors.append(desc)
            chunk_states.append(chunks)
            stream_logits.append(
                nn.Dense(NUM_CLASSES, name=f"classifier_{i}")(desc)
            )

        descs = jnp.stack(descriptors, axis=1)
        sl = jnp.stack(stream_logits, axis=1)
        prior = self.param("fusion_prior", nn.initializers.zeros, (NUM_STREAMS,))
        controller = nn.Dense(NUM_STREAMS, name="fusion_controller")(
            descs.reshape(descs.shape[0], -1)
        )
        fusion = jax.nn.softmax(
            prior[None, :] + 0.15 * jnp.tanh(controller), axis=-1
        )
        logits = jnp.einsum("bs,bsc->bc", fusion, sl)

        return {
            "logits": logits,
            "stream_logits": sl,
            "fusion_weights": fusion,
            "router_weights": router_weights,
            "spatial_stack": jnp.stack(spatial, axis=2),
            "frame_stack": frame_stack,
            "mixed_frame_stack": mixed,
            "descriptors": descs,
            "chunk_states": jnp.stack(chunk_states, axis=1),
        }


class State(train_state.TrainState):
    ema_params: Any


def smooth_ce(logits, labels, smoothing: float):
    onehot = jax.nn.one_hot(labels, NUM_CLASSES)
    targets = onehot * (1.0 - smoothing) + smoothing / NUM_CLASSES
    return -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)


def count_params(params) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))


def shard(x: np.ndarray, ndev: int) -> np.ndarray:
    return x.reshape(ndev, x.shape[0] // ndev, *x.shape[1:])


def iter_train(X, y, global_batch: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    usable = (len(idx) // global_batch) * global_batch
    idx = idx[:usable]
    for s in range(0, usable, global_batch):
        ii = idx[s:s + global_batch]
        yield X[ii], y[ii]


def iter_eval(X, y, global_batch: int):
    for s in range(0, len(X), global_batch):
        xb = X[s:s + global_batch]
        yb = y[s:s + global_batch]
        n = len(xb)
        if n < global_batch:
            px = np.zeros((global_batch, *X.shape[1:]), X.dtype)
            px[:n] = xb
            xb = px
            py = np.zeros((global_batch,), y.dtype)
            py[:n] = yb
            yb = py
        mask = np.zeros((global_batch,), np.float32)
        mask[:n] = 1.0
        yield xb, yb, mask


def audit_flops(model, params):
    try:
        dummy = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
        fn = jax.jit(
            lambda p, xx: model.apply({"params": p}, xx, training=False)["logits"]
        )
        compiled = fn.lower(params, dummy).compile()
        ca = compiled.cost_analysis()
        if isinstance(ca, list):
            ca = ca[0] if ca else {}
        flops = float(ca.get("flops", float("nan")))
        log(
            f"XLA inference audit: {flops:,.0f} FLOPs/clip = "
            f"{flops / 1e9:.9f} GFLOPs"
        )
        return flops
    except Exception as exc:
        log(f"GFLOPs audit unavailable on this TPU runtime: {exc}")
        return None


def find_dataset(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            for name in (
                "ntu120_3danno.pkl",
                "ntu120_3danno_clean.pkl",
                "ntu120.pkl",
            ):
                hits = list(root.rglob(name))
                if hits:
                    return hits[0]
    raise FileNotFoundError("Could not find NTU120 pkl. Pass --dataset PATH")


def train_protocol(args, annotations, split, protocol: str):
    devices = list(jax.local_devices())
    ndev = len(devices)
    if args.batch_size % ndev:
        raise ValueError(
            f"Global batch {args.batch_size} must be divisible by {ndev} TPU devices"
        )
    if args.eval_batch_size % ndev:
        raise ValueError(
            f"Eval batch {args.eval_batch_size} must be divisible by {ndev} TPU devices"
        )

    Xtr, ytr, Xva, yva = build_protocol_arrays(
        annotations,
        split,
        protocol,
        args.selector,
        args.max_train_samples,
        args.max_val_samples,
    )
    steps_per_epoch = len(Xtr) // args.batch_size
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_fraction))
    lr = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=warmup,
        decay_steps=total_steps,
        end_value=args.min_learning_rate,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(lr, weight_decay=args.weight_decay),
    )

    model = M4MotionPreserveT16(
        args.spatial_dim, args.model_dim, args.dropout
    )
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    dummy = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    params = model.init(
        {"params": init_key, "dropout": init_key},
        dummy,
        training=False,
    )["params"]
    log(f"{protocol.upper()} params={count_params(params):,}")
    if args.audit_first:
        audit_flops(model, params)

    state = State.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        ema_params=params,
    )
    state = jax.device_put_replicated(state, devices)
    rngs = jax.random.split(key, ndev)

    @partial(jax.pmap, axis_name="d", devices=devices)
    def p_train_step(state, rng, xb, yb):
        rng, drop = jax.random.split(rng)

        def loss_fn(p):
            out = model.apply(
                {"params": p},
                xb,
                training=True,
                rngs={"dropout": drop},
            )
            main = jnp.mean(
                smooth_ce(out["logits"], yb, args.label_smoothing)
            )
            sl = out["stream_logits"]
            aux = jnp.mean(
                smooth_ce(
                    sl.reshape(-1, NUM_CLASSES),
                    jnp.repeat(yb, NUM_STREAMS),
                    args.label_smoothing,
                )
            )
            loss = main + args.stream_aux_weight * aux
            acc = jnp.mean(jnp.argmax(out["logits"], axis=-1) == yb)
            return loss, (main, aux, acc)

        (loss, (main, aux, acc)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        grads = jax.lax.pmean(grads, "d")
        loss, main, aux, acc = jax.lax.pmean(
            (loss, main, aux, acc), "d"
        )
        state = state.apply_gradients(grads=grads)
        ema = jax.tree_util.tree_map(
            lambda e, p: args.ema_decay * e + (1.0 - args.ema_decay) * p,
            state.ema_params,
            state.params,
        )
        state = state.replace(ema_params=ema)
        return state, rng, (loss, main, aux, acc)

    @partial(jax.pmap, axis_name="d", devices=devices)
    def p_eval_step(ema_params, xb, yb, mask):
        out = model.apply({"params": ema_params}, xb, training=False)
        ce = smooth_ce(out["logits"], yb, 0.0)
        pred = jnp.argmax(out["logits"], axis=-1)
        correct = jnp.sum((pred == yb).astype(jnp.float32) * mask)
        loss_sum = jnp.sum(ce * mask)
        count = jnp.sum(mask)
        return jax.lax.psum(
            jnp.asarray([loss_sum, correct, count]), "d"
        )

    outdir = Path(args.outdir) / protocol
    outdir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    best_epoch = 0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss_sum = acc_sum = nstep = 0.0
        bar = tqdm(
            iter_train(Xtr, ytr, args.batch_size, args.seed + epoch),
            total=steps_per_epoch,
            desc=f"{protocol.upper()} TRAIN E{epoch:03d}/{args.epochs}",
            mininterval=0.5,
        )
        for xb, yb in bar:
            xb = shard(xb, ndev)
            yb = shard(yb, ndev)
            state, rngs, metrics = p_train_step(state, rngs, xb, yb)
            loss_v, _, _, acc_v = [
                float(np.asarray(v[0]))
                for v in jax.device_get(metrics)
            ]
            loss_sum += loss_v
            acc_sum += acc_v
            nstep += 1
            if nstep % args.progress_every == 0:
                bar.set_postfix(
                    loss=f"{loss_sum / nstep:.3f}",
                    acc=f"{100 * acc_sum / nstep:.2f}%",
                    best=f"{100 * best:.2f}%",
                )

        eval_loss = eval_correct = eval_count = 0.0
        for xb, yb, mask in tqdm(
            iter_eval(Xva, yva, args.eval_batch_size),
            desc=f"{protocol.upper()} VAL   E{epoch:03d}/{args.epochs}",
            leave=False,
            mininterval=0.5,
        ):
            xb = shard(xb, ndev)
            yb = shard(yb, ndev)
            mask = shard(mask, ndev)
            sums = p_eval_step(state.ema_params, xb, yb, mask)
            vals = np.asarray(jax.device_get(sums[0]))
            eval_loss += float(vals[0])
            eval_correct += float(vals[1])
            eval_count += float(vals[2])

        val_acc = eval_correct / max(eval_count, 1.0)
        val_loss = eval_loss / max(eval_count, 1.0)
        train_acc = acc_sum / max(nstep, 1.0)
        log(
            f"{protocol.upper()} E{epoch:03d} "
            f"train={100 * train_acc:.3f}% "
            f"val={100 * val_acc:.3f}% "
            f"loss={val_loss:.4f} time={time.time() - t0:.1f}s"
        )

        improved = val_acc > best + 1e-6
        if improved:
            best = val_acc
            best_epoch = epoch
            stale = 0
            single = jax.tree_util.tree_map(
                lambda z: jax.device_get(z[0]), state
            )
            payload = {
                "model": "M4MotionPreserveT16",
                "protocol": protocol,
                "selector": args.selector,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "params": single.params,
                "ema_params": single.ema_params,
                "opt_state": single.opt_state,
                "step": single.step,
                "config": vars(args),
            }
            (outdir / "best.msgpack").write_bytes(
                serialization.to_bytes(payload)
            )
            (outdir / "best.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "val_accuracy": val_acc,
                        "params": count_params(single.params),
                        "frames": FRAMES,
                        "token_channels": TOKEN_CHANNELS,
                    },
                    indent=2,
                )
            )
        else:
            stale += 1

        if stale >= args.patience:
            log(
                f"{protocol.upper()} early stop: "
                f"best={100 * best:.3f}% @ E{best_epoch}"
            )
            break

    del Xtr, ytr, Xva, yva
    return best, best_epoch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument(
        "--protocol", choices=["xsub", "xset", "both"], default="xsub"
    )
    p.add_argument(
        "--selector", choices=["segment", "uniform"], default="segment"
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument(
        "--batch-size", type=int, default=256,
        help="GLOBAL batch across all TPU cores",
    )
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=6e-4)
    p.add_argument("--min-learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.08)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--stream-aux-weight", type=float, default=0.15)
    p.add_argument("--spatial-dim", type=int, default=24)
    p.add_argument("--model-dim", type=int, default=112)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument(
        "--outdir",
        default="/kaggle/working/NestSAR_M4_MotionPreserve_T16_TPU",
    )
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--audit-first", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    log(
        f"JAX={jax.__version__} backend={jax.default_backend()} "
        f"devices={jax.devices()}"
    )
    if jax.default_backend() != "tpu":
        raise RuntimeError(
            "This runner is for TPU. In Kaggle choose a TPU accelerator and restart."
        )
    if jax.local_device_count() != 8:
        raise RuntimeError(
            f"Expected one Kaggle TPU exposing 8 local devices; "
            f"got {jax.local_device_count()}"
        )

    log(
        f"Using 8 TPU cores with pmap | T16 segment tokens | "
        f"features/token={FEATURES} | no attention | no GCN"
    )
    log(
        "Representation: 16 contiguous segments -> pose + net displacement + "
        "intra-segment path motion"
    )
    log(
        "Architecture: spatial streams -> per-stream frame memory -> "
        "cross-stream router -> chunk memory -> fusion"
    )

    dataset = find_dataset(args.dataset)
    log(f"Dataset={dataset}")
    anns, split = load_ntu(dataset)
    protocols = ["xsub", "xset"] if args.protocol == "both" else [args.protocol]
    summary = {}
    for pr in protocols:
        best, ep = train_protocol(args, anns, split, pr)
        summary[pr] = {
            "best_val_accuracy": best,
            "best_epoch": ep,
        }

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    log(f"DONE {summary}")


if __name__ == "__main__":
    main()

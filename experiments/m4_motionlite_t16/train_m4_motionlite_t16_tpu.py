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
FEATURES = 150
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
        x = np.concatenate([x, np.zeros((*x.shape[:-1], 1), np.float32)], axis=-1)
    return x[..., :3]


def canonicalize_raw(keypoints: np.ndarray) -> np.ndarray:
    x = to_tmvc(keypoints)
    if x.shape[2] < 25:
        pad = np.zeros((x.shape[0], x.shape[1], 25 - x.shape[2], 3), np.float32)
        x = np.concatenate([x, pad], axis=2)
    x = x[:, :, :25, :3]
    person_energy = np.sum(np.abs(x), axis=(0, 2, 3))
    x = x[:, np.argsort(-person_energy)]
    if x.shape[1] < 2:
        x = np.concatenate([x, np.zeros((x.shape[0], 2 - x.shape[1], 25, 3), np.float32)], axis=1)
    return x[:, :2]


def uniform_indices(total: int, n: int = 16) -> np.ndarray:
    if total <= 1:
        return np.zeros((n,), np.int64)
    return np.linspace(0, total - 1, n, dtype=np.float32).round().astype(np.int64)


def hybrid_motion_indices(x: np.ndarray, n: int = 16) -> np.ndarray:
    """8 coverage anchors + 8 high-motion frames, deterministic and CPU-side."""
    total = x.shape[0]
    if total <= n:
        return uniform_indices(total, n)

    n_anchor = n // 2
    anchors = list(dict.fromkeys(uniform_indices(total, n_anchor).tolist()))
    centered = x - x[:, :, 0:1, :]
    diff = np.zeros_like(centered)
    diff[1:] = centered[1:] - centered[:-1]
    energy = np.mean(np.abs(diff), axis=(1, 2, 3))
    if total >= 3:
        energy = np.convolve(energy, np.asarray([0.25, 0.5, 0.25], np.float32), mode="same")

    selected = set(anchors)
    min_gap = max(1, total // 32)
    for idx in np.argsort(-energy):
        idx = int(idx)
        if all(abs(idx - old) >= min_gap for old in selected):
            selected.add(idx)
            if len(selected) >= n:
                break

    for idx in uniform_indices(total, n):
        selected.add(int(idx))
        if len(selected) >= n:
            break

    out = np.asarray(sorted(selected), np.int64)
    if len(out) > n:
        out = out[uniform_indices(len(out), n)]
    if len(out) < n:
        out = uniform_indices(total, n)
    return out


def preprocess_keypoints(keypoints: np.ndarray, selector: str) -> np.ndarray:
    x = canonicalize_raw(keypoints)
    if selector == "motion":
        idx = hybrid_motion_indices(x, FRAMES)
    elif selector == "uniform":
        idx = uniform_indices(x.shape[0], FRAMES)
    else:
        raise ValueError(selector)
    x = x[idx]

    center = x[:, 0:1, 0:1, :]
    valid_center = np.any(np.abs(center) > 1e-8, axis=-1, keepdims=True)
    x = np.where(valid_center, x - center, x)
    valid_joint = np.any(np.abs(x) > 1e-8, axis=-1, keepdims=True)
    x = np.where(valid_joint, x, 0.0)
    nz = np.abs(x) > 1e-8
    if np.any(nz):
        rms = float(np.sqrt(np.mean(np.square(x[nz]))) + 1e-6)
        x = x / rms
    return np.nan_to_num(x).reshape(FRAMES, FEATURES).astype(np.float32)


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
    val_names = [f"{protocol}_val", f"{protocol}_test", f"{protocol}val", f"{protocol}test", f"val_{protocol}", f"test_{protocol}"]
    tk = next((lower[k] for k in train_names if k in lower), None)
    vk = next((lower[k] for k in val_names if k in lower), None)
    if tk is None or vk is None:
        raise KeyError(f"Cannot resolve {protocol}; split keys={list(split)}")
    return tk, vk


def build_protocol_arrays(annotations, split, protocol: str, selector: str, max_train: int = 0, max_val: int = 0):
    tk, vk = resolve_split(split, protocol)
    by_id = {sample_id(a, i): a for i, a in enumerate(annotations) if isinstance(a, Mapping)}
    train_ids = [str(v) for v in split[tk] if str(v) in by_id]
    val_ids = [str(v) for v in split[vk] if str(v) in by_id]
    if max_train:
        train_ids = train_ids[:max_train]
    if max_val:
        val_ids = val_ids[:max_val]

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


def lag_diff(x: jnp.ndarray, k: int) -> jnp.ndarray:
    zeros = jnp.zeros_like(x[:, :k])
    return jnp.concatenate([zeros, x[:, k:] - x[:, :-k]], axis=1)


class GatedSweep(nn.Module):
    dim: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d = self.dim
        init = nn.initializers.xavier_uniform()
        wz_x = self.param("wz_x", init, (d, d)); wz_h = self.param("wz_h", init, (d, d)); bz = self.param("bz", nn.initializers.zeros, (d,))
        wr_x = self.param("wr_x", init, (d, d)); wr_h = self.param("wr_h", init, (d, d)); br = self.param("br", nn.initializers.zeros, (d,))
        wc_x = self.param("wc_x", init, (d, d)); wc_h = self.param("wc_h", init, (d, d)); bc = self.param("bc", nn.initializers.zeros, (d,))
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
        je = self.param("joint_embed", nn.initializers.normal(0.02), (1, 1, 1, 25, self.spatial_dim))
        pe = self.param("person_embed", nn.initializers.normal(0.02), (1, 1, 2, 1, self.spatial_dim))
        h = nn.gelu(h + je + pe)

        order = jnp.asarray(JOINT_ORDER)
        inv = jnp.argsort(order)
        h = jnp.take(h, order, axis=3)
        h = h.reshape(b * t * m, 25, self.spatial_dim)
        mem = GatedSweep(self.spatial_dim, reverse=False, name="joint_memory")(h)
        h = nn.LayerNorm(name="joint_memory_norm")(h + mem)
        h = h.reshape(b, t, m, 25, self.spatial_dim)
        h = jnp.take(h, inv, axis=3)

        mask = jnp.asarray(PART_MASK_NP, h.dtype)
        counts = jnp.asarray(PART_COUNTS_NP, h.dtype)
        parts = jnp.einsum("btmvd,pv->btmpd", h, mask) / counts[None, None, None, :, None]
        flat = parts.reshape(b, t, m * 10 * self.spatial_dim)
        y = nn.Dense(self.model_dim, name="part_fuse")(flat)
        y = nn.LayerNorm(name="out_norm")(nn.gelu(y))
        return nn.Dropout(self.dropout)(y, deterministic=not training)


class TemporalHierarchy(nn.Module):
    dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = BiMemory(self.dim, name="frame_memory")(x)
        chunks = h.reshape(h.shape[0], 4, 4, self.dim).mean(axis=2)
        chunks = BiMemory(self.dim, name="chunk_memory")(chunks)
        pooled = jnp.concatenate([h.mean(axis=1), chunks.mean(axis=1)], axis=-1)
        pooled = nn.Dense(self.dim, name="hier_fuse")(pooled)
        pooled = nn.LayerNorm(name="hier_norm")(nn.gelu(pooled))
        pooled = nn.Dropout(self.dropout)(pooled, deterministic=not training)
        return h, pooled


class CrossStreamRouter(nn.Module):
    dim: int = 112
    residual_scale: float = 0.15

    @nn.compact
    def __call__(self, streams: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        n = nn.LayerNorm(name="router_norm")(streams)
        scores = nn.Dense(1, name="score")(n)[..., 0]
        weights = jax.nn.softmax(scores, axis=2)
        context = jnp.sum(weights[..., None] * n, axis=2)
        delta = nn.Dense(self.dim, name="context_proj")(context)
        gate = jax.nn.sigmoid(nn.Dense(1, name="gate")(n))
        out = streams + self.residual_scale * gate * delta[:, :, None, :]
        return out, weights


class M4MotionLiteT16(nn.Module):
    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Mapping[str, jnp.ndarray]:
        sk = x.reshape(x.shape[0], FRAMES, 2, 25, 3)
        root = sk[:, :, :, 0:1, :]
        joint = sk - root
        parents = jnp.asarray(PARENTS)
        bone = joint - jnp.take(joint, parents, axis=3)

        j_d1 = lag_diff(joint, 1)
        j_d2 = lag_diff(joint, 2) / 2.0
        j_d4 = lag_diff(joint, 4) / 4.0
        j_acc = lag_diff(j_d1, 1)
        joint_motion = jnp.concatenate([j_d1, j_d2, j_d4, j_acc], axis=-1)

        b_d1 = lag_diff(bone, 1)
        b_d2 = lag_diff(bone, 2) / 2.0
        b_d4 = lag_diff(bone, 4) / 4.0
        b_acc = lag_diff(b_d1, 1)
        bone_motion = jnp.concatenate([b_d1, b_d2, b_d4, b_acc], axis=-1)

        raw_streams = (joint, bone, joint_motion, bone_motion)
        encoded = []
        for i, s in enumerate(raw_streams):
            encoded.append(SpatialEncoder(self.spatial_dim, self.model_dim, self.dropout, name=f"spatial_{i}")(s, training))
        streams = jnp.stack(encoded, axis=2)
        streams, router_weights = CrossStreamRouter(self.model_dim, name="cross_stream")(streams)

        stream_logits = []
        descriptors = []
        for i in range(NUM_STREAMS):
            _, desc = TemporalHierarchy(self.model_dim, self.dropout, name=f"temporal_{i}")(streams[:, :, i], training)
            descriptors.append(desc)
            stream_logits.append(nn.Dense(NUM_CLASSES, name=f"classifier_{i}")(desc))
        descs = jnp.stack(descriptors, axis=1)
        sl = jnp.stack(stream_logits, axis=1)

        prior = self.param("fusion_prior", nn.initializers.zeros, (NUM_STREAMS,))
        controller = nn.Dense(NUM_STREAMS, name="fusion_controller")(descs.reshape(descs.shape[0], -1))
        fusion = jax.nn.softmax(prior[None, :] + 0.15 * jnp.tanh(controller), axis=-1)
        logits = jnp.einsum("bs,bsc->bc", fusion, sl)
        return {
            "logits": logits,
            "stream_logits": sl,
            "fusion_weights": fusion,
            "router_weights": router_weights,
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
            px = np.zeros((global_batch, *X.shape[1:]), X.dtype); px[:n] = xb; xb = px
            py = np.zeros((global_batch,), y.dtype); py[:n] = yb; yb = py
        mask = np.zeros((global_batch,), np.float32); mask[:n] = 1.0
        yield xb, yb, mask


def audit_flops(model, params):
    try:
        dummy = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
        fn = jax.jit(lambda p, xx: model.apply({"params": p}, xx, training=False)["logits"])
        compiled = fn.lower(params, dummy).compile()
        ca = compiled.cost_analysis()
        if isinstance(ca, list):
            ca = ca[0] if ca else {}
        flops = float(ca.get("flops", float("nan")))
        log(f"XLA inference audit: {flops:,.0f} FLOPs/clip = {flops / 1e9:.9f} GFLOPs")
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
            for name in ("ntu120_3danno.pkl", "ntu120_3danno_clean.pkl", "ntu120.pkl"):
                hits = list(root.rglob(name))
                if hits:
                    return hits[0]
    raise FileNotFoundError("Could not find NTU120 pkl. Pass --dataset PATH")


def train_protocol(args, annotations, split, protocol: str):
    devices = jax.local_devices()
    ndev = len(devices)
    if args.batch_size % ndev:
        raise ValueError(f"Global batch {args.batch_size} must be divisible by {ndev} TPU devices")
    if args.eval_batch_size % ndev:
        raise ValueError(f"Eval batch {args.eval_batch_size} must be divisible by {ndev} TPU devices")

    Xtr, ytr, Xva, yva = build_protocol_arrays(
        annotations, split, protocol, args.selector, args.max_train_samples, args.max_val_samples
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

    model = M4MotionLiteT16(args.spatial_dim, args.model_dim, args.dropout)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    dummy = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    params = model.init({"params": init_key, "dropout": init_key}, dummy, training=False)["params"]
    log(f"{protocol.upper()} params={count_params(params):,}")
    if args.audit_first:
        audit_flops(model, params)

    state = State.create(apply_fn=model.apply, params=params, tx=tx, ema_params=params)
    state = jax.device_put_replicated(state, devices)
    rngs = jax.random.split(key, ndev)

    @partial(jax.pmap, axis_name="d")
    def p_train_step(state, rng, xb, yb):
        rng, drop = jax.random.split(rng)

        def loss_fn(p):
            out = model.apply({"params": p}, xb, training=True, rngs={"dropout": drop})
            main = jnp.mean(smooth_ce(out["logits"], yb, args.label_smoothing))
            sl = out["stream_logits"]
            aux = jnp.mean(smooth_ce(sl.reshape(-1, NUM_CLASSES), jnp.repeat(yb, NUM_STREAMS), args.label_smoothing))
            loss = main + args.stream_aux_weight * aux
            acc = jnp.mean(jnp.argmax(out["logits"], axis=-1) == yb)
            return loss, (main, aux, acc)

        (loss, (main, aux, acc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, "d")
        loss, main, aux, acc = jax.lax.pmean((loss, main, aux, acc), "d")
        state = state.apply_gradients(grads=grads)
        ema = jax.tree_util.tree_map(
            lambda e, p: args.ema_decay * e + (1.0 - args.ema_decay) * p,
            state.ema_params,
            state.params,
        )
        state = state.replace(ema_params=ema)
        return state, rng, (loss, main, aux, acc)

    @partial(jax.pmap, axis_name="d")
    def p_eval_step(ema_params, xb, yb, mask):
        out = model.apply({"params": ema_params}, xb, training=False)
        ce = smooth_ce(out["logits"], yb, 0.0)
        pred = jnp.argmax(out["logits"], axis=-1)
        correct = jnp.sum((pred == yb).astype(jnp.float32) * mask)
        loss_sum = jnp.sum(ce * mask)
        count = jnp.sum(mask)
        return jax.lax.psum(jnp.asarray([loss_sum, correct, count]), "d")

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
            loss_v, _, _, acc_v = [float(np.asarray(v[0])) for v in jax.device_get(metrics)]
            loss_sum += loss_v
            acc_sum += acc_v
            nstep += 1
            if nstep % args.progress_every == 0:
                bar.set_postfix(loss=f"{loss_sum/nstep:.3f}", acc=f"{100*acc_sum/nstep:.2f}%", best=f"{100*best:.2f}%")

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
        log(f"{protocol.upper()} E{epoch:03d} train={100*train_acc:.3f}% val={100*val_acc:.3f}% loss={val_loss:.4f} time={time.time()-t0:.1f}s")

        improved = val_acc > best + 1e-6
        if improved:
            best = val_acc
            best_epoch = epoch
            stale = 0
            single = jax.tree_util.tree_map(lambda z: jax.device_get(z[0]), state)
            payload = {
                "model": "M4MotionLiteT16",
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
            (outdir / "best.msgpack").write_bytes(serialization.to_bytes(payload))
            (outdir / "best.json").write_text(
                json.dumps({"epoch": epoch, "val_accuracy": val_acc, "params": count_params(single.params)}, indent=2)
            )
        else:
            stale += 1

        if stale >= args.patience:
            log(f"{protocol.upper()} early stop: best={100*best:.3f}% @ E{best_epoch}")
            break

    del Xtr, ytr, Xva, yva
    return best, best_epoch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument("--protocol", choices=["xsub", "xset", "both"], default="xsub")
    p.add_argument("--selector", choices=["uniform", "motion"], default="motion")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256, help="GLOBAL batch across all TPU cores")
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
    p.add_argument("--outdir", default="/kaggle/working/NestSAR_M4_MotionLite_T16_TPU")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--audit-first", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    log(f"JAX={jax.__version__} backend={jax.default_backend()} devices={jax.devices()}")
    if jax.default_backend() != "tpu":
        raise RuntimeError("This runner is for TPU. In Kaggle choose Accelerator -> TPU VM / TPU v5e-8, then restart.")
    if jax.local_device_count() < 2:
        raise RuntimeError(f"Expected multi-core TPU, got {jax.local_device_count()} local devices")
    log(f"Using {jax.local_device_count()} TPU cores with pmap | T16 | no attention | no GCN")
    log("Motion representation: J/B + multi-scale JM/BM (Delta1, Delta2, Delta4, acceleration) + tiny 4-stream router")
    log(f"Selector={args.selector}; only 16 frames enter the neural network")

    dataset = find_dataset(args.dataset)
    log(f"Dataset={dataset}")
    anns, split = load_ntu(dataset)
    protocols = ["xsub", "xset"] if args.protocol == "both" else [args.protocol]
    summary = {}
    for pr in protocols:
        best, ep = train_protocol(args, anns, split, pr)
        summary[pr] = {"best_val_accuracy": best, "best_epoch": ep}
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE {summary}")


if __name__ == "__main__":
    main()

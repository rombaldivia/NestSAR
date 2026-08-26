#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NestSAR-HOPE Attention-Lite v2 PartTrace
========================================

Self-contained experimental JAX/Flax runner built on the public NestSAR repo.

Fixes three structural issues from Attention-Lite v1:
1) Preserve 10 anatomical part tokens through a lightweight temporal trace before collapse.
2) Add learned relative temporal bias to causal attention.
3) Make CMS f1/f2/f4/f8 structurally different with causal dilations 1/2/4/8.

This is an EXPERIMENTAL architecture. It does not overwrite the verified v1 baseline.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import serialization

import nestsar as ns


MODEL_NAME = "NestSAR-HOPE-Attention-Lite-v2-PartTrace"

FRAMES = 16
PERSONS = 2
JOINTS = 25
COORDS = 3
NUM_CLASSES = 120

PART_DIM = 32
PART_HEADS = 4
GLOBAL_DIM = 128
ATTENTION_DIM = 64
ATTENTION_HEADS = 4
LOCAL_KERNEL = 4
CMS_BOTTLENECK = 32

TEN_BODY_PARTS = (
    ("torso_core",             (0, 1, 20)),
    ("head_neck",              (2, 3)),
    ("left_upper_arm",         (4, 5)),
    ("left_forearm_hand",      (6, 7, 21, 22)),
    ("right_upper_arm",        (8, 9)),
    ("right_forearm_hand",     (10, 11, 23, 24)),
    ("left_upper_leg",         (12, 13)),
    ("left_lower_leg_foot",    (14, 15)),
    ("right_upper_leg",        (16, 17)),
    ("right_lower_leg_foot",   (18, 19)),
)


def _make_part_mask() -> jnp.ndarray:
    mask = np.zeros((10, JOINTS), dtype=np.float32)
    seen = []
    for p, (_, joints) in enumerate(TEN_BODY_PARTS):
        for j in joints:
            mask[p, j] = 1.0
            seen.append(j)
    if sorted(seen) != list(range(JOINTS)):
        raise ValueError("TEN_BODY_PARTS must cover joints 0..24 exactly once.")
    return jnp.asarray(mask)


PART_MASK = _make_part_mask()


def safe_logit(p: float) -> float:
    p = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return float(np.log(p / (1.0 - p)))


class PartMixer10(nn.Module):
    dim: int = PART_DIM
    heads: int = PART_HEADS
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        if x.ndim != 5 or x.shape[3] != 10:
            raise ValueError(f"Expected [B,T,M,10,D], got {x.shape}")
        if self.dim % self.heads:
            raise ValueError("Part dimension must be divisible by heads.")

        b, t, m, p, d = x.shape
        dh = self.dim // self.heads
        z = nn.LayerNorm(name="part_mix_norm")(x)
        q = nn.Dense(self.dim, use_bias=False, name="part_q")(z)
        k = nn.Dense(self.dim, use_bias=False, name="part_k")(z)
        v = nn.Dense(self.dim, use_bias=False, name="part_v")(z)
        q = q.reshape(b, t, m, p, self.heads, dh).transpose(0, 1, 2, 4, 3, 5)
        k = k.reshape(b, t, m, p, self.heads, dh).transpose(0, 1, 2, 4, 3, 5)
        v = v.reshape(b, t, m, p, self.heads, dh).transpose(0, 1, 2, 4, 3, 5)
        logits = jnp.einsum("btmhpd,btmhkd->btmhpk", q, k) / math.sqrt(dh)
        attn = jax.nn.softmax(logits, axis=-1)
        attn = nn.Dropout(rate=self.dropout, name="part_attn_dropout")(
            attn, deterministic=not training
        )
        ctx = jnp.einsum("btmhpk,btmhkd->btmhpd", attn, v)
        ctx = ctx.transpose(0, 1, 2, 4, 3, 5).reshape(b, t, m, p, d)
        ctx = nn.Dense(self.dim, name="part_out")(ctx)
        gate_logit = self.param(
            "part_mix_gate_logit",
            lambda key, shape: jnp.full(shape, safe_logit(0.10), dtype=jnp.float32),
            (1,),
        )
        gate = jax.nn.sigmoid(gate_logit)[0]
        return x + gate * ctx


class SharedPartTemporalTrace(nn.Module):
    dim: int = PART_DIM
    heads: int = PART_HEADS
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(f"Expected [N,T,D], got {x.shape}")
        n, t, d = x.shape
        dh = self.dim // self.heads
        h = nn.LayerNorm(name="trace_norm")(x)
        padded = jnp.pad(h, ((0, 0), (2, 0), (0, 0)))
        local = nn.Conv(
            features=self.dim,
            kernel_size=(3,),
            padding="VALID",
            feature_group_count=self.dim,
            use_bias=True,
            name="trace_dwconv",
        )(padded)
        local = jax.nn.silu(local)
        local_gate = jax.nn.sigmoid(
            self.param(
                "trace_local_gate_logit",
                lambda key, shape: jnp.full(shape, safe_logit(0.10), dtype=jnp.float32),
                (1,),
            )
        )[0]
        x = x + local_gate * local
        z = nn.LayerNorm(name="trace_attn_norm")(x)
        q = nn.Dense(self.dim, use_bias=False, name="trace_q")(z)
        k = nn.Dense(self.dim, use_bias=False, name="trace_k")(z)
        v = nn.Dense(self.dim, use_bias=False, name="trace_v")(z)
        q = q.reshape(n, t, self.heads, dh).transpose(0, 2, 1, 3)
        k = k.reshape(n, t, self.heads, dh).transpose(0, 2, 1, 3)
        v = v.reshape(n, t, self.heads, dh).transpose(0, 2, 1, 3)
        logits = jnp.einsum("nhtd,nhkd->nhtk", q, k) / math.sqrt(dh)
        rel_bias = self.param(
            "relative_time_bias",
            nn.initializers.zeros,
            (self.heads, FRAMES),
        )
        qpos = jnp.arange(t)[:, None]
        kpos = jnp.arange(t)[None, :]
        dist = jnp.clip(qpos - kpos, 0, FRAMES - 1)
        logits = logits + rel_bias[:, dist][None, ...]
        causal = kpos <= qpos
        logits = jnp.where(causal[None, None, :, :], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        ctx = jnp.einsum("nhtk,nhkd->nhtd", attn, v)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(n, t, d)
        ctx = nn.Dense(self.dim, name="trace_out")(ctx)
        gate = jax.nn.sigmoid(
            self.param(
                "trace_attn_gate_logit",
                lambda key, shape: jnp.full(shape, safe_logit(0.15), dtype=jnp.float32),
                (1,),
            )
        )[0]
        return nn.LayerNorm(name="trace_out_norm")(x + gate * ctx)


class PartTraceSpatialEncoder(nn.Module):
    part_dim: int = PART_DIM
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        if x.ndim != 5 or x.shape[2:] != (PERSONS, JOINTS, COORDS):
            raise ValueError(f"Expected [B,T,2,25,3], got {x.shape}")
        b, t, m, v, c = x.shape
        present = jnp.any(jnp.abs(x) > 1e-6, axis=-1, keepdims=True).astype(x.dtype)
        root = x[:, :, :, 0:1, :]
        centered = (x - root) * present
        prev = jnp.concatenate([centered[:, :1], centered[:, :-1]], axis=1)
        velocity = centered - prev
        joint_features = jnp.concatenate([centered, velocity], axis=-1)
        h = nn.Dense(self.part_dim, name="joint_projection")(joint_features)
        joint_emb = self.param(
            "joint_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, 1, 1, JOINTS, self.part_dim),
        )
        person_emb = self.param(
            "person_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, 1, PERSONS, 1, self.part_dim),
        )
        h = nn.LayerNorm(name="joint_norm")(h + joint_emb + person_emb)
        h = jax.nn.gelu(h)
        h = nn.Dropout(rate=self.dropout, name="joint_dropout")(h, deterministic=not training)
        h = h * present
        mask = PART_MASK.astype(h.dtype)
        numerator = jnp.einsum("btmjd,pj->btmpd", h, mask)
        counts = jnp.sum(mask, axis=1)[None, None, None, :, None]
        part_tokens = numerator / jnp.maximum(counts, 1.0)
        part_emb = self.param(
            "part_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, 1, 1, 10, self.part_dim),
        )
        part_tokens = nn.LayerNorm(name="part_norm")(part_tokens + part_emb)
        part_tokens = PartMixer10(
            dim=self.part_dim,
            heads=PART_HEADS,
            dropout=self.dropout,
            name="part_mixer",
        )(part_tokens, training)
        tracks = part_tokens.transpose(0, 2, 3, 1, 4).reshape(b * PERSONS * 10, t, self.part_dim)
        tracks = SharedPartTemporalTrace(
            dim=self.part_dim,
            heads=PART_HEADS,
            dropout=self.dropout,
            name="shared_part_trace",
        )(tracks, training)
        part_tokens = tracks.reshape(b, PERSONS, 10, t, self.part_dim).transpose(0, 3, 1, 2, 4)
        gate = jax.nn.sigmoid(nn.Dense(1, name="part_pool_gate")(part_tokens)[..., 0])
        person_desc = jnp.sum(gate[..., None] * part_tokens, axis=3) / jnp.maximum(
            jnp.sum(gate, axis=3, keepdims=True), 1e-6
        )
        first = person_desc[:, :, 0]
        second = person_desc[:, :, 1]
        pair = jnp.concatenate([first + second, jnp.abs(first - second), first * second], axis=-1)
        summary = nn.Dense(GLOBAL_DIM, name="person_fusion")(pair)
        summary = nn.LayerNorm(name="spatial_summary_norm")(summary)
        summary = jax.nn.gelu(summary)
        metrics = {
            "part_gate_mean": jnp.mean(gate),
            "part_token_rms": jnp.sqrt(jnp.mean(jnp.square(part_tokens)) + 1e-12),
        }
        return summary, metrics


class CausalAttentionWithRelativeBias(nn.Module):
    model_dim: int = GLOBAL_DIM
    attention_dim: int = ATTENTION_DIM
    heads: int = ATTENTION_HEADS
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        b, t, _ = x.shape
        dh = self.attention_dim // self.heads
        h = nn.LayerNorm(name="input_norm")(x)
        padded = jnp.pad(h, ((0, 0), (LOCAL_KERNEL - 1, 0), (0, 0)))
        local = nn.Conv(
            features=self.model_dim,
            kernel_size=(LOCAL_KERNEL,),
            padding="VALID",
            feature_group_count=self.model_dim,
            use_bias=True,
            name="hope_local_conv",
        )(padded)
        local = jax.nn.silu(local)
        local_gate = jax.nn.sigmoid(
            self.param(
                "hope_local_conv_gate_logit",
                lambda key, shape: jnp.full(shape, safe_logit(0.05), dtype=jnp.float32),
                (1,),
            )
        )[0]
        x = nn.LayerNorm(name="hope_local_conv_output_norm")(x + local_gate * local)
        z = nn.LayerNorm(name="attention_norm")(x)
        q = nn.Dense(self.attention_dim, use_bias=False, name="attention_q")(z)
        k = nn.Dense(self.attention_dim, use_bias=False, name="attention_k")(z)
        v = nn.Dense(self.attention_dim, use_bias=False, name="attention_v")(z)
        q = q.reshape(b, t, self.heads, dh).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, self.heads, dh).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.heads, dh).transpose(0, 2, 1, 3)
        logits = jnp.einsum("bhtd,bhkd->bhtk", q, k) / math.sqrt(dh)
        rel_bias = self.param("relative_time_bias", nn.initializers.zeros, (self.heads, FRAMES))
        qpos = jnp.arange(t)[:, None]
        kpos = jnp.arange(t)[None, :]
        dist = jnp.clip(qpos - kpos, 0, FRAMES - 1)
        logits = logits + rel_bias[:, dist][None, ...]
        causal = kpos <= qpos
        logits = jnp.where(causal[None, None, :, :], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        attn = nn.Dropout(rate=self.dropout, name="attention_dropout")(attn, deterministic=not training)
        ctx = jnp.einsum("bhtk,bhkd->bhtd", attn, v)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(b, t, self.attention_dim)
        ctx = nn.Dense(self.model_dim, name="attention_out")(ctx)
        attn_gate = jax.nn.sigmoid(
            self.param(
                "attention_gate_logit",
                lambda key, shape: jnp.full(shape, safe_logit(0.50), dtype=jnp.float32),
                (1,),
            )
        )[0]
        scaled = attn_gate * ctx
        return nn.LayerNorm(name="attention_output_norm")(x + scaled), scaled


class DilatedCMS(nn.Module):
    model_dim: int = GLOBAL_DIM
    bottleneck: int = CMS_BOTTLENECK
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        output = x
        for dilation in (1, 2, 4, 8):
            left_pad = 2 * dilation
            z = nn.LayerNorm(name=f"cms_f{dilation}_pre_norm")(output)
            z = jnp.pad(z, ((0, 0), (left_pad, 0), (0, 0)))
            z = nn.Conv(
                features=self.model_dim,
                kernel_size=(3,),
                kernel_dilation=(dilation,),
                padding="VALID",
                feature_group_count=self.model_dim,
                use_bias=False,
                name=f"cms_f{dilation}_temporal",
            )(z)
            h = nn.LayerNorm(name=f"cms_f{dilation}_norm")(output + z)
            h = nn.Dense(self.bottleneck, name=f"cms_f{dilation}_in")(h)
            h = jax.nn.silu(h)
            h = nn.Dropout(rate=self.dropout, name=f"cms_f{dilation}_dropout")(h, deterministic=not training)
            delta = nn.Dense(self.model_dim, name=f"cms_f{dilation}_out")(h)
            gate = jax.nn.sigmoid(
                self.param(
                    f"cms_f{dilation}_gate_logit",
                    lambda key, shape: jnp.full(shape, safe_logit(0.05), dtype=jnp.float32),
                    (1,),
                )
            )[0]
            output = output + gate * delta
        return nn.LayerNorm(name="cms_output_norm")(output)


class NestSARPartTraceV2(nn.Module):
    num_classes: int = NUM_CLASSES
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Dict[str, jnp.ndarray]:
        if x.ndim != 3 or x.shape[-1] != PERSONS * JOINTS * COORDS:
            raise ValueError(f"Expected [B,T,150], got {x.shape}")
        b, t, _ = x.shape
        xyz = x.reshape(b, t, PERSONS, JOINTS, COORDS)
        spatial, spatial_metrics = PartTraceSpatialEncoder(
            part_dim=PART_DIM,
            dropout=self.dropout,
            name="spatial_encoder",
        )(xyz, training)
        temporal, attn_residual = CausalAttentionWithRelativeBias(
            model_dim=GLOBAL_DIM,
            attention_dim=ATTENTION_DIM,
            heads=ATTENTION_HEADS,
            dropout=self.dropout,
            name="attention_lite_v2",
        )(spatial, training)
        temporal = DilatedCMS(
            model_dim=GLOBAL_DIM,
            bottleneck=CMS_BOTTLENECK,
            dropout=self.dropout,
            name="cms",
        )(temporal, training)
        pooled = jnp.mean(temporal, axis=1)
        logits = nn.Dense(self.num_classes, name="classifier")(pooled)
        out = {
            "logits": logits,
            "attention_residual_rms": jnp.sqrt(jnp.mean(jnp.square(attn_residual), axis=-1) + 1e-12),
        }
        out.update(spatial_metrics)
        return out


def tree_numel(tree) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(tree)))


def find_dataset(explicit: str) -> Path:
    if explicit and explicit != "auto":
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p.resolve()
    candidates = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            candidates.extend(root.rglob("ntu120_3danno.pkl"))
    if not candidates:
        raise FileNotFoundError("Could not locate ntu120_3danno.pkl under /kaggle/input.")
    return candidates[0].resolve()


def configure_nestsar(seed: int) -> None:
    ns.CFG = dataclasses.replace(
        ns.CFG,
        frames=FRAMES,
        persons=PERSONS,
        joints=JOINTS,
        coords=COORDS,
        num_classes=NUM_CLASSES,
        seed=seed,
    )


def make_optimizer(lr: float, wd: float, steps: int, warmup_fraction: float):
    warmup = max(1, int(steps * warmup_fraction))
    decay = max(1, steps - warmup)
    schedule = optax.join_schedules(
        [optax.linear_schedule(0.0, lr, warmup), optax.cosine_decay_schedule(lr, decay)],
        [warmup],
    )
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=wd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="auto")
    ap.add_argument("--protocol", choices=["xsub", "xset"], default="xsub")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.03)
    ap.add_argument("--warmup-fraction", type=float, default=0.10)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--outdir", default="/kaggle/working/NestSAR_PartTrace_v2")
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    configure_nestsar(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    model = NestSARPartTraceV2(dropout=args.dropout)
    dummy = jnp.zeros((1, FRAMES, PERSONS * JOINTS * COORDS), jnp.float32)
    variables = model.init({"params": rng, "dropout": rng}, dummy, training=True)
    params = variables["params"]

    print("=" * 110)
    print(MODEL_NAME)
    print("=" * 110)
    print("Backend:      ", jax.default_backend())
    print("Devices:      ", jax.devices())
    print("Frames:       ", FRAMES)
    print("Part tokens:  ", 10, "x D", PART_DIM)
    print("Global D:     ", GLOBAL_DIM)
    print("Attention:    ", f"D{ATTENTION_DIM}/H{ATTENTION_HEADS}")
    print("CMS dilation: ", "1/2/4/8")
    print("Parameters:   ", f"{tree_numel(params):,}")

    try:
        compiled = jax.jit(
            lambda p, xx: model.apply({"params": p}, xx, training=False)["logits"]
        ).lower(params, dummy).compile()
        cost = compiled.cost_analysis()
        if isinstance(cost, list) and cost:
            cost = cost[0]
        flops = float(cost.get("flops", 0.0)) if isinstance(cost, dict) else 0.0
        if flops:
            print("XLA GFLOPs:   ", f"{flops / 1e9:.9f}")
        else:
            print("XLA GFLOPs:   unavailable from backend cost analysis")
    except Exception as exc:
        print("XLA audit:     unavailable:", exc)

    if args.audit_only:
        out = model.apply({"params": params}, dummy, training=False)
        print("Smoke logits: ", out["logits"].shape)
        print("AUDIT-ONLY PASS")
        return

    dataset_path = find_dataset(args.dataset)
    raw = ns.load_pickle(dataset_path)
    train_samples, val_samples = ns.build_samples(
        raw,
        protocol=args.protocol,
        max_train=args.max_train_samples,
        max_val=args.max_val_samples,
        seed=args.seed,
    )
    train_ds = ns.SkeletonDataset(train_samples)
    val_ds = ns.SkeletonDataset(val_samples)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs)
    tx = make_optimizer(args.learning_rate, args.weight_decay, total_steps, args.warmup_fraction)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y, rng):
        def loss_fn(p):
            out = model.apply({"params": p}, x, training=True, rngs={"dropout": rng})
            logits = out["logits"]
            labels = jax.nn.one_hot(y, NUM_CLASSES)
            loss = jnp.mean(optax.softmax_cross_entropy(logits, labels))
            acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
            return loss, (acc, out["part_gate_mean"], out["part_token_rms"])
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    @jax.jit
    def eval_step(params, x, y):
        logits = model.apply({"params": params}, x, training=False)["logits"]
        return jnp.sum(jnp.argmax(logits, axis=-1) == y)

    outdir = Path(args.outdir) / args.protocol
    outdir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    run_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_rng = jax.random.fold_in(rng, epoch)
        loss_sum = 0.0
        acc_sum = 0.0
        batches = 0
        for x_np, y_np in ns.batch_iterator(
            train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed + epoch, drop_last=False
        ):
            step_rng = jax.random.fold_in(epoch_rng, batches)
            x = jnp.asarray(x_np, jnp.float32)
            y = jnp.asarray(y_np, jnp.int32)
            params, opt_state, loss, aux = train_step(params, opt_state, x, y, step_rng)
            loss_sum += float(loss)
            acc_sum += float(aux[0])
            batches += 1

        correct = 0
        count = 0
        for x_np, y_np in ns.batch_iterator(
            val_ds, batch_size=args.eval_batch_size, shuffle=False, seed=args.seed, drop_last=False
        ):
            x = jnp.asarray(x_np, jnp.float32)
            y = jnp.asarray(y_np, jnp.int32)
            correct += int(eval_step(params, x, y))
            count += len(y_np)

        val_acc = correct / max(count, 1)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            "train_accuracy": acc_sum / max(batches, 1),
            "val_accuracy": val_acc,
        }
        history.append(record)
        (outdir / "history.json").write_text(json.dumps(history, indent=2))
        print(
            f"E{epoch:03d}/{args.epochs} | loss={record['train_loss']:.4f} | "
            f"train={100*record['train_accuracy']:.2f}% | val={100*val_acc:.5f}%"
        )

        if val_acc > best:
            best = val_acc
            payload = {
                "model": MODEL_NAME,
                "epoch": epoch,
                "protocol": args.protocol,
                "seed": args.seed,
                "val_accuracy": val_acc,
                "params": params,
            }
            (outdir / "best.msgpack").write_bytes(serialization.to_bytes(payload))

    result = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "seed": args.seed,
        "epochs": args.epochs,
        "parameters": tree_numel(params),
        "best_val_accuracy": best,
        "wall_hours": (time.time() - run_start) / 3600.0,
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2))
    print("=" * 110)
    print("COMPLETE")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

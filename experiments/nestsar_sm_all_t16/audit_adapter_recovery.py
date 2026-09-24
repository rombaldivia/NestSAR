#!/usr/bin/env python3
from __future__ import annotations

"""Causal adapter-recovery audit for trained NestSAR R4.

The trained NestSAR checkpoint is frozen. Only one tiny shared rank-r residual
adapter is trained at a time, at one of four candidate bottleneck locations:

  pre_m4   : Spatial -> Adapter -> M4
  post_m4  : M4 -> Adapter -> Router
  pre_g4   : Router -> Adapter -> G4
  post_g4  : G4 descriptors -> Adapter -> classifiers/head

Every location uses the exact same adapter architecture and parameter count.
The adapter output projection is zero-initialized, so every run starts at the
exact original checkpoint function.

Adapter selection uses a deterministic stratified holdout carved only from the
training split. The official NTU validation split is evaluated only after the
best adapter epoch is selected.
"""

import argparse
import json
import math
import time
from contextlib import closing
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import serialization
from flax.training import train_state

from experiments.nestsar_sm_all_t16 import model as sm
from experiments.nestsar_sm_all_t16.streaming import worker as train_worker
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_bytes, atomic_json

LOCATIONS = ("pre_m4", "post_m4", "pre_g4", "post_g4")
NUM_CLASSES = 120
MODEL_DIM = 112
EXPECTED_BASE_PARAMS = 1_831_932


def report(path: Path, protocol: str, phase: str, current: int, total: int, **extra):
    payload = {
        "protocol": protocol,
        "phase": phase,
        "current": int(current),
        "total": int(max(total, 1)),
    }
    payload.update(extra)
    atomic_json(path, payload)


def tree_count(tree):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def load_checkpoint(path: Path):
    payload = serialization.msgpack_restore(path.read_bytes())
    if not isinstance(payload, dict) or "ema_params" not in payload or "config" not in payload:
        raise ValueError(f"Expected best checkpoint with ema_params/config: {path}")
    return payload, payload["ema_params"], dict(payload["config"])


def ce(logits, labels, smoothing):
    target = jax.nn.one_hot(labels, NUM_CLASSES)
    target = target * (1.0 - smoothing) + smoothing / NUM_CLASSES
    return -jnp.sum(target * jax.nn.log_softmax(logits), axis=-1)


class RecoveryAdapter(nn.Module):
    """Same normalized low-rank residual at every candidate location."""

    rank: int = 8

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm(name="norm")(x)
        h = nn.Dense(
            self.rank,
            kernel_init=nn.initializers.normal(0.02),
            bias_init=nn.initializers.zeros,
            name="down",
        )(h)
        h = nn.gelu(h)
        delta = nn.Dense(
            MODEL_DIM,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="up",
        )(h)
        return x + delta


class NestSARAdapterAudit(nn.Module):
    """Parameter-compatible R4 model with one extra shared audit adapter."""

    location: str
    adapter_rank: int = 8

    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10
    controller_dim: int = 16
    fast_rank: int = 4
    head_rank: int = 2
    sm_residual_scale: float = 0.08
    head_residual_scale: float = 0.15

    @nn.compact
    def __call__(self, x, training: bool = False):
        if self.location not in LOCATIONS:
            raise ValueError(self.location)
        if x.shape[1] != sm.FRAMES or x.shape[2] != sm.FEATURES:
            raise ValueError(f"Expected [B,{sm.FRAMES},{sm.FEATURES}], got {x.shape}")

        tok = x.reshape(
            x.shape[0],
            sm.FRAMES,
            sm.PERSONS,
            sm.JOINTS,
            sm.TOKEN_CHANNELS,
        )

        controller = sm.SharedSMController(
            controller_dim=self.controller_dim,
            head_rank=self.head_rank,
            name="sm_controller",
        )(tok)

        joint_valid = controller["joint_valid"]
        person_present = controller["person_present"].astype(bool)
        joint_valid = joint_valid.at[..., 0].set(person_present)
        valid_f = joint_valid[..., None].astype(tok.dtype)

        gamma = controller["gamma"][:, :, None, None, :]
        beta = controller["beta"][:, :, None, None, :]
        tok = (tok * gamma + valid_f * beta) * valid_f

        pose = tok[..., 0:3]
        full_disp = tok[..., 3:6]
        phase_a = tok[..., 6:9]
        phase_b = tok[..., 9:12]
        path = tok[..., 12:15]

        joint = pose
        parents = jnp.asarray(sm.base.PARENTS)
        parent_valid = jnp.take(joint_valid, parents, axis=3)
        bone_valid = joint_valid & parent_valid
        bone = (joint - jnp.take(joint, parents, axis=3)) * bone_valid[..., None]

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
        ) * bone_valid[..., None]

        raw_streams = (joint, bone, joint_motion, bone_motion)
        stream_valid = (joint_valid, bone_valid, joint_valid, bone_valid)

        spatial = []
        for i, (stream, valid) in enumerate(zip(raw_streams, stream_valid)):
            s = sm.MaskSafeSpatialEncoder(
                self.spatial_dim,
                self.model_dim,
                self.dropout,
                name=f"spatial_{i}",
            )(stream, valid, training)
            gate = controller["stream_gate"][:, :, i:i + 1]
            spatial.append(s * gate)

        spatial_stack = jnp.stack(spatial, axis=2)
        if self.location == "pre_m4":
            spatial_stack = RecoveryAdapter(
                rank=self.adapter_rank,
                name="audit_adapter",
            )(spatial_stack)
        spatial = [spatial_stack[:, :, i] for i in range(sm.NUM_STREAMS)]

        frame_streams = []
        for i, stream in enumerate(spatial):
            frame_streams.append(
                sm.SelfModBiMemory(
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
        if self.location == "post_m4":
            frame_stack = RecoveryAdapter(
                rank=self.adapter_rank,
                name="audit_adapter",
            )(frame_stack)

        mixed, router_weights = sm.base.CrossStreamRouter(
            self.model_dim,
            name="cross_stream_after_frame",
        )(frame_stack)

        if self.location == "pre_g4":
            mixed = RecoveryAdapter(
                rank=self.adapter_rank,
                name="audit_adapter",
            )(mixed)

        eta_slow = controller["eta"].reshape(
            x.shape[0], 4, sm.FRAMES // 4, 1
        ).mean(axis=2)
        alpha_slow = controller["alpha"].reshape(
            x.shape[0], 4, sm.FRAMES // 4, 1
        ).mean(axis=2)

        descriptors = []
        chunk_states = []

        for i in range(sm.NUM_STREAMS):
            chunks, desc = sm.SelfModDescriptorHead(
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

        descs = jnp.stack(descriptors, axis=1)

        if self.location == "post_g4":
            descs = RecoveryAdapter(
                rank=self.adapter_rank,
                name="audit_adapter",
            )(descs)

        stream_logits = []
        for i in range(sm.NUM_STREAMS):
            stream_logits.append(
                nn.Dense(
                    NUM_CLASSES,
                    name=f"classifier_{i}",
                )(descs[:, i])
            )

        sl = jnp.stack(stream_logits, axis=1)

        fusion = jax.nn.softmax(
            controller["fusion_logits"],
            axis=-1,
        )
        main_logits = jnp.einsum("bs,bsc->bc", fusion, sl)

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
            "spatial_stack": spatial_stack,
            "frame_stack": frame_stack,
            "mixed_frame_stack": mixed,
            "descriptors": descs,
            "chunk_states": jnp.stack(chunk_states, axis=1),
            "sm_eta_mean": jnp.mean(controller["eta"], axis=(1, 2)),
            "sm_alpha_mean": jnp.mean(controller["alpha"], axis=(1, 2)),
            "sm_head_coeff": controller["head_coeff"],
        }


class AdapterState(train_state.TrainState):
    ema_params: object


def combine_params(base_params, adapter_params):
    full = dict(base_params)
    full["audit_adapter"] = adapter_params
    return full


def make_adapter_model(config, location, adapter_rank):
    return NestSARAdapterAudit(
        location=location,
        adapter_rank=adapter_rank,
        spatial_dim=config["spatial_dim"],
        model_dim=config["model_dim"],
        dropout=config["dropout"],
        controller_dim=config["controller_dim"],
        fast_rank=config["fast_rank"],
        head_rank=config["head_rank"],
        sm_residual_scale=config["sm_residual_scale"],
        head_residual_scale=config["head_residual_scale"],
    )


def init_adapter_params(adapter_rank, seed):
    module = RecoveryAdapter(rank=adapter_rank)
    variables = module.init(
        jax.random.PRNGKey(seed),
        jnp.zeros((1, 1, 4, MODEL_DIM), jnp.float32),
    )
    return variables["params"]


def stratified_holdout(ids, labels, fraction, seed):
    ids = np.asarray(ids, np.int64)
    labels = np.asarray(labels)
    train_parts = []
    dev_parts = []
    for cls in range(NUM_CLASSES):
        cls_ids = ids[labels[ids] == cls].copy()
        if len(cls_ids) < 2:
            raise RuntimeError(f"Class {cls} has fewer than 2 training examples")
        rng = np.random.default_rng(seed + cls * 1009)
        rng.shuffle(cls_ids)
        ndev = int(round(len(cls_ids) * fraction))
        ndev = min(max(ndev, 1), len(cls_ids) - 1)
        dev_parts.append(cls_ids[:ndev])
        train_parts.append(cls_ids[ndev:])
    train_ids = np.concatenate(train_parts)
    dev_ids = np.concatenate(dev_parts)
    rng = np.random.default_rng(seed + 99991)
    rng.shuffle(train_ids)
    rng.shuffle(dev_ids)
    if set(train_ids.tolist()) & set(dev_ids.tolist()):
        raise RuntimeError("Adapter train/dev leakage")
    return train_ids.tolist(), dev_ids.tolist()


def build_steps(model, config, schedule):
    accum = int(config["accumulation_steps"])
    micro = int(config["micro_batch"])

    def per_sample(adapter_params, base_params, key, batch):
        full = combine_params(base_params, adapter_params)
        k1, k2 = jax.random.split(key)
        out = model.apply(
            {"params": full},
            batch["x"],
            training=True,
            rngs={"dropout": k1},
        )
        aug = model.apply(
            {"params": full},
            batch["xa"],
            training=True,
            rngs={"dropout": k2},
        )
        y = batch["y"]
        smooth = config["label_smoothing"]

        main = (
            ce(out["logits"], y, smooth)
            + ce(aug["logits"], y, smooth)
        ) / 2.0

        aux = (
            ce(out["stream_logits"], y[:, None], smooth).mean(axis=1)
            + ce(aug["stream_logits"], y[:, None], smooth).mean(axis=1)
        ) / 2.0

        temperature = config["consistency_temperature"]
        logp = jax.nn.log_softmax(out["logits"] / temperature)
        logq = jax.nn.log_softmax(aug["logits"] / temperature)
        kl = (
            0.5
            * temperature**2
            * jnp.sum(
                (jnp.exp(logp) - jnp.exp(logq))
                * (logp - logq),
                axis=-1,
            )
        )

        loss = (
            main
            + config["stream_aux_weight"] * aux
            + config["consistency_weight"] * kl
        )
        acc = (out["logits"].argmax(-1) == y).astype(jnp.float32)
        return jnp.stack([loss, main, aux, kl, acc], axis=-1)

    @jax.jit
    def train_step(state, base_params, key, batch):
        micros = jax.tree.map(
            lambda value: value.reshape(
                accum,
                micro,
                *value.shape[1:],
            ),
            batch,
        )
        denom = jnp.maximum(jnp.sum(batch["mask"]), 1.0)
        key, step_key = jax.random.split(key)
        drop_keys = jax.random.split(step_key, accum)
        zero = jax.tree.map(jnp.zeros_like, state.params)

        def accumulate(carry, inputs):
            grad_sum, metric_sum = carry
            micro_batch, drop_key = inputs

            def loss_fn(adapter_params):
                metrics = per_sample(
                    adapter_params,
                    base_params,
                    drop_key,
                    micro_batch,
                )
                totals = jnp.sum(
                    metrics * micro_batch["mask"][:, None],
                    axis=0,
                )
                return totals[0] / denom, totals

            (_, totals), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(state.params)

            return (
                jax.tree.map(
                    lambda a, b: a + b,
                    grad_sum,
                    grads,
                ),
                metric_sum + totals,
            ), None

        (grads, metrics), _ = jax.lax.scan(
            accumulate,
            (
                zero,
                jnp.zeros(5, jnp.float32),
            ),
            (
                micros,
                drop_keys,
            ),
        )

        grad_norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)
        ema = jax.tree.map(
            lambda e, p: (
                config["adapter_ema_decay"] * e
                + (1.0 - config["adapter_ema_decay"]) * p
            ),
            state.ema_params,
            state.params,
        )
        state = state.replace(ema_params=ema)

        return (
            state,
            key,
            jnp.r_[metrics, denom, grad_norm, schedule(state.step)],
        )

    @jax.jit
    def eval_step(base_params, adapter_params, batch):
        full = combine_params(base_params, adapter_params)
        out = model.apply(
            {"params": full},
            batch["x"],
            training=False,
        )
        y = batch["y"]
        mask = batch["mask"]
        loss = ce(out["logits"], y, 0.0)
        correct = (out["logits"].argmax(-1) == y).astype(jnp.float32)
        main_correct = (
            out["main_logits"].argmax(-1) == y
        ).astype(jnp.float32)

        values = jnp.stack(
            [loss, correct, main_correct, jnp.ones_like(mask)],
            axis=-1,
        )
        return jnp.sum(values * mask[:, None], axis=0)

    return train_step, eval_step


def evaluate(
    eval_step,
    base_params,
    adapter_params,
    dataset,
    ids,
    batch_size,
    config,
    protocol,
    status,
    phase,
):
    total_steps = math.ceil(len(ids) / batch_size)
    sums = np.zeros(4, np.float64)

    with closing(
        dataset.batches(
            ids,
            batch_size,
            config,
            training=False,
            protocol=protocol,
        )
    ) as batches:
        for bi in range(total_steps):
            batch, _ = next(batches)
            values = np.asarray(
                jax.block_until_ready(
                    eval_step(
                        base_params,
                        adapter_params,
                        jax.device_put(batch),
                    )
                )
            )
            sums += values
            if bi % 10 == 0 or bi + 1 == total_steps:
                report(
                    status,
                    protocol,
                    phase,
                    bi + 1,
                    total_steps,
                    val_acc=float(sums[1] / max(sums[3], 1.0)),
                )

    if int(round(sums[3])) != len(ids):
        raise RuntimeError(
            f"Evaluation accounting mismatch: {sums[3]} vs {len(ids)}"
        )

    return {
        "loss": float(sums[0] / sums[3]),
        "accuracy": float(sums[1] / sums[3]),
        "main_accuracy": float(sums[2] / sums[3]),
    }


def adapter_l2_norm(params):
    total = 0.0
    for leaf in jax.tree_util.tree_leaves(params):
        x = np.asarray(leaf, np.float64)
        total += float(np.sum(x * x))
    return math.sqrt(total)


def verify_zero_equivalence(
    original_model,
    audit_model,
    base_params,
    adapter_params,
    dataset,
    sample_ids,
):
    ids = np.asarray(sample_ids[:32], np.int64)
    x = jnp.asarray(np.asarray(dataset.canonical[ids], np.float32))
    base_logits = original_model.apply(
        {"params": base_params},
        x,
        training=False,
    )["logits"]
    audit_logits = audit_model.apply(
        {"params": combine_params(base_params, adapter_params)},
        x,
        training=False,
    )["logits"]
    max_abs = float(
        np.max(
            np.abs(
                np.asarray(jax.device_get(base_logits))
                - np.asarray(jax.device_get(audit_logits))
            )
        )
    )
    if max_abs > 2e-5:
        raise RuntimeError(
            f"Zero-adapter equivalence failed: max |delta logits|={max_abs}"
        )
    return max_abs


def run_location(
    location,
    protocol,
    dataset,
    train_ids,
    dev_ids,
    val_ids,
    base_params,
    checkpoint_payload,
    config,
    args,
    output,
    status,
):
    report(status, protocol, f"{location}: initialize", 0, 1)

    model = make_adapter_model(config, location, args.adapter_rank)
    adapter_params = init_adapter_params(
        args.adapter_rank,
        args.adapter_seed,
    )

    adapter_param_count = tree_count(adapter_params)
    if adapter_param_count <= 0:
        raise RuntimeError("Adapter has no parameters")

    original_model = train_worker.make_model(config)
    equivalence_error = verify_zero_equivalence(
        original_model,
        model,
        base_params,
        adapter_params,
        dataset,
        dev_ids,
    )

    steps_per_epoch = math.ceil(
        len(train_ids)
        / (args.micro_batch * args.accumulation_steps)
    )
    total_steps = max(args.epochs * steps_per_epoch, 2)
    warm_steps = max(1, steps_per_epoch * args.warmup_epochs)
    warm_steps = min(warm_steps, total_steps - 1)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=warm_steps,
        decay_steps=total_steps,
        end_value=args.min_learning_rate,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(
            schedule,
            weight_decay=args.weight_decay,
        ),
    )

    state = AdapterState.create(
        apply_fn=model.apply,
        params=adapter_params,
        tx=optimizer,
        ema_params=adapter_params,
    )
    key = jax.random.PRNGKey(args.adapter_seed)

    train_config = dict(config)
    train_config.update(
        micro_batch=args.micro_batch,
        accumulation_steps=args.accumulation_steps,
        adapter_ema_decay=args.ema_decay,
    )

    train_step, eval_step = build_steps(
        model,
        train_config,
        schedule,
    )

    # Exact baseline on the adapter-dev split.
    baseline_dev = evaluate(
        eval_step,
        base_params,
        adapter_params,
        dataset,
        dev_ids,
        args.eval_batch,
        train_config,
        protocol,
        status,
        f"{location}: baseline dev",
    )

    history = []
    best_dev = baseline_dev["accuracy"]
    best_epoch = 0
    best_adapter = jax.device_get(state.ema_params)
    bad_epochs = 0

    batch_size = args.micro_batch * args.accumulation_steps
    train_steps = math.ceil(len(train_ids) / batch_size)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        sums = np.zeros(8, np.float64)

        with closing(
            dataset.batches(
                train_ids,
                batch_size,
                train_config,
                epoch=epoch,
                training=True,
                protocol=protocol,
            )
        ) as batches:
            for bi in range(train_steps):
                batch, _ = next(batches)
                state, key, metrics = jax.block_until_ready(
                    train_step(
                        state,
                        base_params,
                        key,
                        jax.device_put(batch),
                    )
                )
                values = np.asarray(metrics)
                if not np.isfinite(values).all():
                    raise FloatingPointError(
                        f"{location}: non-finite adapter training metrics"
                    )
                sums += values

                if bi % 10 == 0 or bi + 1 == train_steps:
                    report(
                        status,
                        protocol,
                        f"{location}: train",
                        bi + 1,
                        train_steps,
                        epoch=epoch,
                        best=best_dev,
                        best_epoch=best_epoch,
                        train_acc=float(sums[4] / max(sums[5], 1.0)),
                        loss=float(sums[0] / max(sums[5], 1.0)),
                    )

        dev = evaluate(
            eval_step,
            base_params,
            state.ema_params,
            dataset,
            dev_ids,
            args.eval_batch,
            train_config,
            protocol,
            status,
            f"{location}: dev",
        )

        improved = dev["accuracy"] > best_dev + args.min_delta
        if improved:
            best_dev = dev["accuracy"]
            best_epoch = epoch
            best_adapter = jax.device_get(state.ema_params)
            bad_epochs = 0
        else:
            bad_epochs += 1

        row = {
            "epoch": epoch,
            "train_loss": float(sums[0] / max(sums[5], 1.0)),
            "train_accuracy": float(sums[4] / max(sums[5], 1.0)),
            "grad_norm": float(sums[6] / max(train_steps, 1)),
            "learning_rate": float(sums[7] / max(train_steps, 1)),
            "dev_loss": dev["loss"],
            "dev_accuracy": dev["accuracy"],
            "best_dev_accuracy": best_dev,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        atomic_json(output / f"{location}_history.json", history)

        report(
            status,
            protocol,
            f"{location}: epoch complete",
            1,
            1,
            epoch=epoch,
            best=best_dev,
            best_epoch=best_epoch,
            val_acc=dev["accuracy"],
        )

        if bad_epochs >= args.patience:
            break

    # Official validation is evaluated once, after model selection on train-derived dev.
    final_val = evaluate(
        eval_step,
        base_params,
        jax.device_put(best_adapter),
        dataset,
        val_ids,
        args.eval_batch,
        train_config,
        protocol,
        status,
        f"{location}: official validation",
    )

    base_val = float(checkpoint_payload["val_accuracy"])
    result = {
        "location": location,
        "adapter_rank": args.adapter_rank,
        "adapter_params": adapter_param_count,
        "base_params_frozen": EXPECTED_BASE_PARAMS,
        "zero_init_equivalence_max_abs_logits": equivalence_error,
        "train_samples": len(train_ids),
        "adapter_dev_samples": len(dev_ids),
        "official_val_samples": len(val_ids),
        "baseline_dev_accuracy": baseline_dev["accuracy"],
        "best_dev_accuracy": best_dev,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "official_checkpoint_val_accuracy": base_val,
        "adapter_official_val_accuracy": final_val["accuracy"],
        "adapter_official_main_accuracy": final_val["main_accuracy"],
        "official_val_gain_pp": 100.0 * (final_val["accuracy"] - base_val),
        "best_adapter_l2_norm": adapter_l2_norm(best_adapter),
        "selection_split": "deterministic stratified holdout from training split",
        "official_validation_used_for_adapter_selection": False,
    }

    adapter_payload = {
        "protocol": protocol,
        "location": location,
        "adapter_rank": args.adapter_rank,
        "adapter_params": best_adapter,
        "result": result,
    }
    atomic_bytes(
        output / f"{location}_best_adapter.msgpack",
        serialization.msgpack_serialize(adapter_payload),
    )
    atomic_json(output / f"{location}_result.json", result)

    # Free compiled executables before the next location.
    del state, train_step, eval_step, model
    jax.clear_caches()

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--adapter-rank", type=int, default=8)
    p.add_argument("--adapter-seed", type=int, default=20260924)
    p.add_argument("--dev-fraction", type=float, default=0.10)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--min-delta", type=float, default=1e-4)

    p.add_argument("--micro-batch", type=int, default=64)
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--eval-batch", type=int, default=256)

    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--min-learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)
    args = p.parse_args()

    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise RuntimeError(
            f"Expected one isolated GPU, got backend={jax.default_backend()}, "
            f"devices={jax.local_devices()}"
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.json"

    checkpoint_payload, base_params, config = load_checkpoint(
        Path(args.checkpoint)
    )

    if int(config.get("fast_rank", -1)) != 4:
        raise RuntimeError("Expected trained R4 checkpoint")
    if int(config.get("head_rank", -1)) != 2:
        raise RuntimeError("Expected Head Rank=2 checkpoint")
    if tree_count(base_params) != EXPECTED_BASE_PARAMS:
        raise RuntimeError(
            f"Base parameter mismatch: {tree_count(base_params)} "
            f"!= {EXPECTED_BASE_PARAMS}"
        )
    if config["model_dim"] != MODEL_DIM:
        raise RuntimeError(
            f"Adapter audit expects model_dim={MODEL_DIM}, got {config['model_dim']}"
        )

    dataset = Dataset(args.cache)
    full_train_ids = dataset.splits[f"{args.protocol}_train"]
    val_ids = dataset.splits[f"{args.protocol}_val"]

    split_seed = args.adapter_seed + (
        100000 if args.protocol == "xset" else 0
    )
    train_ids, dev_ids = stratified_holdout(
        full_train_ids,
        dataset.labels,
        args.dev_fraction,
        split_seed,
    )

    split_info = {
        "protocol": args.protocol,
        "full_training_samples": len(full_train_ids),
        "adapter_training_samples": len(train_ids),
        "adapter_dev_samples": len(dev_ids),
        "official_validation_samples": len(val_ids),
        "dev_fraction_requested": args.dev_fraction,
        "split_seed": split_seed,
        "adapter_seed": args.adapter_seed,
    }
    atomic_json(output / "split.json", split_info)

    results = {}
    for location in LOCATIONS:
        results[location] = run_location(
            location=location,
            protocol=args.protocol,
            dataset=dataset,
            train_ids=train_ids,
            dev_ids=dev_ids,
            val_ids=val_ids,
            base_params=base_params,
            checkpoint_payload=checkpoint_payload,
            config=config,
            args=args,
            output=output,
            status=status,
        )

    ordered = sorted(
        LOCATIONS,
        key=lambda loc: results[loc]["official_val_gain_pp"],
        reverse=True,
    )

    summary = {
        "audit": "NestSAR R4 frozen-backbone causal adapter recovery v1",
        "protocol": args.protocol,
        "checkpoint": str(args.checkpoint),
        "base_checkpoint_val_accuracy": float(
            checkpoint_payload["val_accuracy"]
        ),
        "base_params_frozen": EXPECTED_BASE_PARAMS,
        "adapter_architecture": (
            "LayerNorm -> Dense(112,rank) -> GELU -> "
            "zero-init Dense(rank,112) -> residual"
        ),
        "adapter_rank": args.adapter_rank,
        "adapter_params_each": results[LOCATIONS[0]]["adapter_params"],
        "all_locations_same_adapter_parameter_count": len(
            {results[loc]["adapter_params"] for loc in LOCATIONS}
        ) == 1,
        "selection_split": split_info,
        "results": results,
        "ranking_by_official_val_gain": ordered,
        "winner": ordered[0],
        "winner_gain_pp": results[ordered[0]]["official_val_gain_pp"],
        "runner_up": ordered[1],
        "winner_margin_over_runner_up_pp": (
            results[ordered[0]]["official_val_gain_pp"]
            - results[ordered[1]]["official_val_gain_pp"]
        ),
    }

    atomic_json(output / "adapter_recovery_summary.json", summary)
    report(
        status,
        args.protocol,
        "Done",
        1,
        1,
        done=True,
        best=results[ordered[0]]["best_dev_accuracy"],
        best_epoch=results[ordered[0]]["best_epoch"],
        val_acc=results[ordered[0]]["adapter_official_val_accuracy"],
    )

    print("=" * 120)
    print(f"{args.protocol.upper()} CAUSAL ADAPTER RECOVERY COMPLETE")
    print("=" * 120)
    print(json.dumps(summary, indent=2))
    print("REPORT:", output / "adapter_recovery_summary.json")


if __name__ == "__main__":
    main()

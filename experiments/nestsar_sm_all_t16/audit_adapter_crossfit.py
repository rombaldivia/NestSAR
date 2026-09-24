#!/usr/bin/env python3
from __future__ import annotations

"""Cross-fitted causal adapter recovery audit for trained NestSAR R4.

The trained R4 checkpoint is frozen. Only one tiny shared residual adapter is
trained at a time at four candidate bottleneck locations:

  pre_m4   : Spatial -> Adapter -> M4
  post_m4  : M4 -> Adapter -> Router
  pre_g4   : Router -> Adapter -> G4
  post_g4  : G4 descriptors -> Adapter -> classifiers/head

The adapter trains ONLY on the original NTU training split.

The original unseen validation split is stratified into two disjoint folds A/B.
A fixed adapter training trajectory is run for all requested epochs:
  * the epoch selected using fold A is evaluated on fold B;
  * the epoch selected using fold B is evaluated on fold A.

The two held-out test halves are then recombined. Thus every validation sample
is scored only by an adapter checkpoint whose epoch was selected without that
sample. No original NestSAR parameter is updated.
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
            "stream_logits": sl,
            "fusion_weights": fusion,
            "router_weights": router_weights,
            "spatial_stack": spatial_stack,
            "frame_stack": frame_stack,
            "mixed_frame_stack": mixed,
            "descriptors": descs,
            "chunk_states": jnp.stack(chunk_states, axis=1),
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


def stratified_two_folds(ids, labels, seed):
    ids = np.asarray(ids, np.int64)
    labels = np.asarray(labels)
    fold_a = []
    fold_b = []

    for cls in range(NUM_CLASSES):
        cls_ids = ids[labels[ids] == cls].copy()
        if len(cls_ids) < 2:
            raise RuntimeError(
                f"Validation class {cls} has fewer than two examples; "
                "cannot make two stratified folds."
            )
        rng = np.random.default_rng(seed + 1009 * cls)
        rng.shuffle(cls_ids)
        cut = (len(cls_ids) + 1) // 2
        fold_a.append(cls_ids[:cut])
        fold_b.append(cls_ids[cut:])

    fold_a = np.concatenate(fold_a)
    fold_b = np.concatenate(fold_b)

    rng = np.random.default_rng(seed + 99991)
    rng.shuffle(fold_a)
    rng.shuffle(fold_b)

    a = set(fold_a.tolist())
    b = set(fold_b.tolist())
    original = set(ids.tolist())

    if a & b:
        raise RuntimeError("Cross-fit validation fold leakage")
    if a | b != original:
        raise RuntimeError("Cross-fit folds do not partition validation exactly")

    return fold_a.tolist(), fold_b.tolist()


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

        return jnp.stack(
            [loss, main, aux, kl, acc],
            axis=-1,
        )

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
            (zero, jnp.zeros(5, jnp.float32)),
            (micros, drop_keys),
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

        values = jnp.stack(
            [
                loss,
                correct,
                jnp.ones_like(mask),
            ],
            axis=-1,
        )

        return jnp.sum(
            values * mask[:, None],
            axis=0,
        )

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
    epoch=0,
    best=None,
    best_epoch=0,
):
    steps = math.ceil(len(ids) / batch_size)
    sums = np.zeros(3, np.float64)

    with closing(
        dataset.batches(
            ids,
            batch_size,
            config,
            training=False,
            protocol=protocol,
        )
    ) as batches:
        for bi in range(steps):
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

            if bi % 10 == 0 or bi + 1 == steps:
                report(
                    status,
                    protocol,
                    phase,
                    bi + 1,
                    steps,
                    epoch=epoch,
                    best=best,
                    best_epoch=best_epoch,
                    val_acc=float(sums[1] / max(sums[2], 1.0)),
                )

    seen = int(round(sums[2]))
    if seen != len(ids):
        raise RuntimeError(
            f"{phase}: sample accounting mismatch {seen} != {len(ids)}"
        )

    return {
        "loss": float(sums[0] / sums[2]),
        "accuracy": float(sums[1] / sums[2]),
        "correct": int(round(sums[1])),
        "samples": seen,
    }


def verify_zero_equivalence(
    original_model,
    audit_model,
    base_params,
    adapter_params,
    dataset,
    ids,
):
    sample_ids = np.asarray(ids[:32], np.int64)
    x = jnp.asarray(
        np.asarray(dataset.canonical[sample_ids], np.float32)
    )

    original_logits = original_model.apply(
        {"params": base_params},
        x,
        training=False,
    )["logits"]

    audit_logits = audit_model.apply(
        {
            "params": combine_params(
                base_params,
                adapter_params,
            )
        },
        x,
        training=False,
    )["logits"]

    max_abs = float(
        np.max(
            np.abs(
                np.asarray(jax.device_get(original_logits))
                - np.asarray(jax.device_get(audit_logits))
            )
        )
    )

    if max_abs > 2e-5:
        raise RuntimeError(
            f"Zero-adapter equivalence failed: max |delta logits|={max_abs}"
        )

    return max_abs


def adapter_l2_norm(params):
    total = 0.0
    for leaf in jax.tree_util.tree_leaves(params):
        value = np.asarray(leaf, np.float64)
        total += float(np.sum(value * value))
    return math.sqrt(total)


def run_location(
    location,
    protocol,
    dataset,
    train_ids,
    fold_a_ids,
    fold_b_ids,
    base_params,
    checkpoint_payload,
    config,
    args,
    output,
    status,
):
    report(
        status,
        protocol,
        f"{location}: initialize",
        0,
        1,
    )

    model = make_adapter_model(
        config,
        location,
        args.adapter_rank,
    )

    init_params = init_adapter_params(
        args.adapter_rank,
        args.adapter_seed,
    )
    adapter_params_count = tree_count(init_params)

    original_model = train_worker.make_model(config)
    equivalence_error = verify_zero_equivalence(
        original_model,
        model,
        base_params,
        init_params,
        dataset,
        fold_a_ids,
    )

    batch_size = args.micro_batch * args.accumulation_steps
    steps_per_epoch = math.ceil(len(train_ids) / batch_size)
    total_steps = max(args.epochs * steps_per_epoch, 2)
    warm_steps = max(1, args.warmup_epochs * steps_per_epoch)
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
        params=init_params,
        tx=optimizer,
        ema_params=init_params,
    )

    key = jax.random.PRNGKey(
        args.adapter_seed
        + (100000 if protocol == "xset" else 0)
    )

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

    # Epoch 0 is the exact original model and is a valid candidate.
    baseline_a = evaluate(
        eval_step,
        base_params,
        init_params,
        dataset,
        fold_a_ids,
        args.eval_batch,
        train_config,
        protocol,
        status,
        f"{location}: baseline fold A",
    )

    baseline_b = evaluate(
        eval_step,
        base_params,
        init_params,
        dataset,
        fold_b_ids,
        args.eval_batch,
        train_config,
        protocol,
        status,
        f"{location}: baseline fold B",
    )

    baseline_full = (
        baseline_a["correct"] + baseline_b["correct"]
    ) / (
        baseline_a["samples"] + baseline_b["samples"]
    )

    checkpoint_acc = float(checkpoint_payload["val_accuracy"])
    tolerance = 1.0 / (
        baseline_a["samples"] + baseline_b["samples"]
    ) + 1e-9

    if abs(baseline_full - checkpoint_acc) > tolerance:
        raise RuntimeError(
            f"Cross-fit baseline {baseline_full} does not reproduce "
            f"checkpoint accuracy {checkpoint_acc}"
        )

    # best_for_a = checkpoint whose epoch was selected ONLY using fold A.
    # It will be tested on fold B.
    best_a_score = baseline_a["accuracy"]
    best_a_epoch = 0
    best_for_a = jax.device_get(init_params)

    # best_for_b = checkpoint whose epoch was selected ONLY using fold B.
    # It will be tested on fold A.
    best_b_score = baseline_b["accuracy"]
    best_b_epoch = 0
    best_for_b = jax.device_get(init_params)

    history = []
    train_steps = math.ceil(len(train_ids) / batch_size)

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
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
                        best=max(best_a_score, best_b_score),
                        best_epoch=max(best_a_epoch, best_b_epoch),
                        train_acc=float(
                            sums[4] / max(sums[5], 1.0)
                        ),
                        loss=float(
                            sums[0] / max(sums[5], 1.0)
                        ),
                    )

        # Both folds are observed because each is a selection fold for one
        # independent cross-fit direction. Fold B never selects the model
        # eventually tested on B, and vice versa.
        fold_a = evaluate(
            eval_step,
            base_params,
            state.ema_params,
            dataset,
            fold_a_ids,
            args.eval_batch,
            train_config,
            protocol,
            status,
            f"{location}: select on fold A",
            epoch=epoch,
            best=best_a_score,
            best_epoch=best_a_epoch,
        )

        fold_b = evaluate(
            eval_step,
            base_params,
            state.ema_params,
            dataset,
            fold_b_ids,
            args.eval_batch,
            train_config,
            protocol,
            status,
            f"{location}: select on fold B",
            epoch=epoch,
            best=best_b_score,
            best_epoch=best_b_epoch,
        )

        if fold_a["accuracy"] > best_a_score + args.min_delta:
            best_a_score = fold_a["accuracy"]
            best_a_epoch = epoch
            best_for_a = jax.device_get(state.ema_params)

        if fold_b["accuracy"] > best_b_score + args.min_delta:
            best_b_score = fold_b["accuracy"]
            best_b_epoch = epoch
            best_for_b = jax.device_get(state.ema_params)

        row = {
            "epoch": epoch,
            "train_loss": float(
                sums[0] / max(sums[5], 1.0)
            ),
            "train_accuracy": float(
                sums[4] / max(sums[5], 1.0)
            ),
            "grad_norm": float(
                sums[6] / max(train_steps, 1)
            ),
            "learning_rate": float(
                sums[7] / max(train_steps, 1)
            ),
            "fold_a_selection_accuracy": fold_a["accuracy"],
            "fold_b_selection_accuracy": fold_b["accuracy"],
            "best_fold_a_selection_accuracy": best_a_score,
            "best_fold_a_epoch": best_a_epoch,
            "best_fold_b_selection_accuracy": best_b_score,
            "best_fold_b_epoch": best_b_epoch,
            "epoch_seconds": time.perf_counter() - start_time,
        }
        history.append(row)

        atomic_json(
            output / f"{location}_history.json",
            history,
        )

        report(
            status,
            protocol,
            f"{location}: epoch complete",
            1,
            1,
            epoch=epoch,
            best=max(best_a_score, best_b_score),
            best_epoch=max(best_a_epoch, best_b_epoch),
        )

    # Cross-fitted held-out evaluation:
    #   selected on A -> test B
    #   selected on B -> test A
    test_b = evaluate(
        eval_step,
        base_params,
        jax.device_put(best_for_a),
        dataset,
        fold_b_ids,
        args.eval_batch,
        train_config,
        protocol,
        status,
        f"{location}: TEST B using A-selected epoch",
        epoch=best_a_epoch,
        best=best_a_score,
        best_epoch=best_a_epoch,
    )

    test_a = evaluate(
        eval_step,
        base_params,
        jax.device_put(best_for_b),
        dataset,
        fold_a_ids,
        args.eval_batch,
        train_config,
        protocol,
        status,
        f"{location}: TEST A using B-selected epoch",
        epoch=best_b_epoch,
        best=best_b_score,
        best_epoch=best_b_epoch,
    )

    crossfit_correct = test_a["correct"] + test_b["correct"]
    crossfit_samples = test_a["samples"] + test_b["samples"]
    crossfit_accuracy = crossfit_correct / crossfit_samples

    result = {
        "location": location,
        "adapter_rank": args.adapter_rank,
        "adapter_params": adapter_params_count,
        "base_params_frozen": EXPECTED_BASE_PARAMS,
        "zero_init_equivalence_max_abs_logits": equivalence_error,
        "training_samples": len(train_ids),
        "fold_a_samples": len(fold_a_ids),
        "fold_b_samples": len(fold_b_ids),
        "baseline_fold_a_accuracy": baseline_a["accuracy"],
        "baseline_fold_b_accuracy": baseline_b["accuracy"],
        "baseline_full_validation_accuracy": baseline_full,

        "selected_on_fold_a": {
            "best_epoch": best_a_epoch,
            "selection_accuracy": best_a_score,
            "heldout_test_fold": "B",
            "heldout_test_accuracy": test_b["accuracy"],
            "heldout_test_correct": test_b["correct"],
            "heldout_test_samples": test_b["samples"],
            "heldout_gain_pp_vs_baseline_fold_b": 100.0 * (
                test_b["accuracy"] - baseline_b["accuracy"]
            ),
        },

        "selected_on_fold_b": {
            "best_epoch": best_b_epoch,
            "selection_accuracy": best_b_score,
            "heldout_test_fold": "A",
            "heldout_test_accuracy": test_a["accuracy"],
            "heldout_test_correct": test_a["correct"],
            "heldout_test_samples": test_a["samples"],
            "heldout_gain_pp_vs_baseline_fold_a": 100.0 * (
                test_a["accuracy"] - baseline_a["accuracy"]
            ),
        },

        "crossfit_accuracy": crossfit_accuracy,
        "crossfit_correct": crossfit_correct,
        "crossfit_samples": crossfit_samples,
        "crossfit_gain_pp": 100.0 * (
            crossfit_accuracy - baseline_full
        ),
        "adapter_l2_norm_selected_on_a": adapter_l2_norm(
            best_for_a
        ),
        "adapter_l2_norm_selected_on_b": adapter_l2_norm(
            best_for_b
        ),
        "leakage_rule": (
            "Fold A selects only the model tested on B; "
            "fold B selects only the model tested on A."
        ),
        "fixed_training_epochs": args.epochs,
        "early_stopping_used": False,
    }

    atomic_bytes(
        output / f"{location}_selected_on_a.msgpack",
        serialization.msgpack_serialize(
            {
                "protocol": protocol,
                "location": location,
                "selected_on": "A",
                "test_fold": "B",
                "epoch": best_a_epoch,
                "adapter_params": best_for_a,
            }
        ),
    )

    atomic_bytes(
        output / f"{location}_selected_on_b.msgpack",
        serialization.msgpack_serialize(
            {
                "protocol": protocol,
                "location": location,
                "selected_on": "B",
                "test_fold": "A",
                "epoch": best_b_epoch,
                "adapter_params": best_for_b,
            }
        ),
    )

    atomic_json(
        output / f"{location}_crossfit_result.json",
        result,
    )

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
    p.add_argument("--fold-seed", type=int, default=20260924)

    p.add_argument("--epochs", type=int, default=20)
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
            f"Expected one isolated GPU; backend={jax.default_backend()}, "
            f"devices={jax.local_devices()}"
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.json"

    checkpoint_payload, base_params, config = load_checkpoint(
        Path(args.checkpoint)
    )

    if int(config.get("fast_rank", -1)) != 4:
        raise RuntimeError("Expected trained Fast Rank=4 checkpoint")
    if int(config.get("head_rank", -1)) != 2:
        raise RuntimeError("Expected Head Rank=2 checkpoint")
    if tree_count(base_params) != EXPECTED_BASE_PARAMS:
        raise RuntimeError(
            f"Base parameter mismatch: {tree_count(base_params)} "
            f"!= {EXPECTED_BASE_PARAMS}"
        )
    if int(config["model_dim"]) != MODEL_DIM:
        raise RuntimeError(
            f"Expected model_dim={MODEL_DIM}, got {config['model_dim']}"
        )

    dataset = Dataset(args.cache)
    train_ids = dataset.splits[f"{args.protocol}_train"]
    val_ids = dataset.splits[f"{args.protocol}_val"]

    fold_seed = args.fold_seed + (
        100000 if args.protocol == "xset" else 0
    )

    fold_a_ids, fold_b_ids = stratified_two_folds(
        val_ids,
        dataset.labels,
        fold_seed,
    )

    split_info = {
        "protocol": args.protocol,
        "training_samples": len(train_ids),
        "validation_samples": len(val_ids),
        "fold_a_samples": len(fold_a_ids),
        "fold_b_samples": len(fold_b_ids),
        "fold_seed": fold_seed,
        "adapter_seed": args.adapter_seed,
        "folds_partition_validation_exactly": True,
        "training_data_used_for_adapter_training_only": True,
        "validation_data_used_for_adapter_gradient_updates": False,
        "crossfit_rule": (
            "A selects checkpoint tested on B; "
            "B selects checkpoint tested on A."
        ),
    }

    atomic_json(
        output / "crossfit_split.json",
        split_info,
    )

    results = {}

    for location in LOCATIONS:
        results[location] = run_location(
            location=location,
            protocol=args.protocol,
            dataset=dataset,
            train_ids=train_ids,
            fold_a_ids=fold_a_ids,
            fold_b_ids=fold_b_ids,
            base_params=base_params,
            checkpoint_payload=checkpoint_payload,
            config=config,
            args=args,
            output=output,
            status=status,
        )

    gains = {
        loc: results[loc]["crossfit_gain_pp"]
        for loc in LOCATIONS
    }

    best_gain = max(gains.values())
    tolerance_pp = 1e-12
    tied_winners = [
        loc
        for loc in LOCATIONS
        if abs(gains[loc] - best_gain) <= tolerance_pp
    ]

    ordered = sorted(
        LOCATIONS,
        key=lambda loc: gains[loc],
        reverse=True,
    )

    runner_up_gain = gains[ordered[1]]

    summary = {
        "audit": "NestSAR R4 frozen-backbone causal adapter cross-fit v2",
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
        "same_parameter_count": (
            len(
                {
                    results[loc]["adapter_params"]
                    for loc in LOCATIONS
                }
            )
            == 1
        ),
        "crossfit_split": split_info,
        "results": results,
        "ranking_by_crossfit_gain": ordered,
        "tied_winners": tied_winners,
        "unique_winner": (
            tied_winners[0]
            if len(tied_winners) == 1
            else None
        ),
        "best_crossfit_gain_pp": best_gain,
        "margin_over_runner_up_pp": (
            best_gain - runner_up_gain
            if len(tied_winners) == 1
            else 0.0
        ),
    }

    atomic_json(
        output / "adapter_crossfit_summary.json",
        summary,
    )

    winner = summary["unique_winner"]
    winner_result = (
        results[winner]
        if winner is not None
        else results[ordered[0]]
    )

    report(
        status,
        args.protocol,
        "Done",
        1,
        1,
        done=True,
        val_acc=winner_result["crossfit_accuracy"],
    )

    print("=" * 120)
    print(f"{args.protocol.upper()} CROSS-FITTED CAUSAL ADAPTER AUDIT COMPLETE")
    print("=" * 120)
    print(json.dumps(summary, indent=2))
    print("REPORT:", output / "adapter_crossfit_summary.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""Warm-start late-stage geometry fine-tuning for NestSAR D128-MTS.

Purpose
-------
Run controlled continuation experiments from the proven D128-MTS EMA checkpoint
without changing inference architecture, parameter count, or FLOPs.

Only late-stage parameters are trainable:
  * descriptor_group      (G4 + hierarchical descriptor)
  * classifier_group
  * adaptive_head_u
  * adaptive_head_v

Everything up to and including Spatial/M4/Router/controller is frozen.

Training objective
------------------
Baseline terms are preserved:
  CE(two views) + stream auxiliary CE + logit consistency KL

Optional geometry terms:
  1. Descriptor consistency:
       1 - cosine(z, z_aug)

  2. Cross-view hard-negative descriptor margin:
       relu(m + max_{different-class j} cos(z_i, z_aug_j)
                - cos(z_i, z_aug_i))

The hard-negative term is symmetric across the two augmented views.
It uses only training-batch labels and never validation-derived class pairs.

Resume behavior
---------------
The new OUT_DIR has its own last.msgpack and resumes exactly from it.
If no last.msgpack exists, the optimizer is created fresh and model params/EMA
are initialized from the source checkpoint's EMA weights.
"""

import hashlib
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict

from experiments.nestsar_sm_all_t16.model_parallel import NestSARParallelT16
from experiments.nestsar_sm_all_t16.parallel_d128_config import (
    MODEL_NAME,
    EXPECTED_PARAMS,
    MODEL_DIM,
    M4_HALF_LIVES,
    G4_HALF_LIVES,
)
from experiments.nestsar_sm_all_t16.streaming import launch as launch_base
from experiments.nestsar_sm_all_t16.streaming import worker as base


EXPERIMENT_NAME = "NestSAR-D128-TRANSFERABLE-GEOMETRY-WARM-v2"
TRAINABLE_ROOTS = (
    "descriptor_group/hier_fuse",
    "descriptor_group/hier_norm",
    "classifier_group",
    "adaptive_head_u",
    "adaptive_head_v",
)


def make_model(config):
    return NestSARParallelT16(
        spatial_dim=config["spatial_dim"],
        model_dim=MODEL_DIM,
        dropout=config["dropout"],
        controller_dim=config["controller_dim"],
        fast_rank=config["fast_rank"],
        head_rank=config["head_rank"],
        sm_residual_scale=config["sm_residual_scale"],
        head_residual_scale=config["head_residual_scale"],
        m4_half_lives=M4_HALF_LIVES,
        g4_half_lives=G4_HALF_LIVES,
    )


def validate_geometry_config(config):
    extra_keys = {
        "source_checkpoint",
        "descriptor_consistency_weight",
        "descriptor_margin_weight",
        "descriptor_margin",
        "geometry_objective",
    }
    unknown = set(config) - (set(launch_base.DEFAULTS) | extra_keys)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    base_config = {
        k: v for k, v in config.items()
        if k in launch_base.DEFAULTS
    }

    # Reuse all proven D128 safety constraints.
    from experiments.nestsar_sm_all_t16.parallel_d128_config import (
        validate_d128_config,
    )

    c = validate_d128_config(base_config)

    source = str(config.get("source_checkpoint", "")).strip()
    if not source:
        raise ValueError("source_checkpoint is required")
    if not Path(source).is_file():
        raise FileNotFoundError(f"source_checkpoint not found: {source}")

    desc_w = float(config.get("descriptor_consistency_weight", 0.0))
    margin_w = float(config.get("descriptor_margin_weight", 0.0))
    margin = float(config.get("descriptor_margin", 0.10))
    objective = str(config.get("geometry_objective", "")).strip()

    if desc_w < 0 or margin_w < 0:
        raise ValueError("Geometry loss weights must be nonnegative")
    if not 0.0 <= margin <= 1.0:
        raise ValueError("descriptor_margin must be in [0,1]")
    if objective not in ("consistency", "consistency_margin"):
        raise ValueError(
            "geometry_objective must be 'consistency' or 'consistency_margin'"
        )
    if objective == "consistency" and margin_w != 0.0:
        raise ValueError(
            "consistency objective must use descriptor_margin_weight=0"
        )
    if objective == "consistency_margin" and margin_w <= 0.0:
        raise ValueError(
            "consistency_margin objective requires descriptor_margin_weight>0"
        )
    if desc_w <= 0.0:
        raise ValueError("descriptor_consistency_weight must be >0")

    c.update(
        source_checkpoint=source,
        descriptor_consistency_weight=desc_w,
        descriptor_margin_weight=margin_w,
        descriptor_margin=margin,
        geometry_objective=objective,
    )
    return c


def _count_params(tree):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def _source_params(path, template):
    payload = serialization.msgpack_restore(Path(path).read_bytes())

    if payload.get("model") != MODEL_NAME:
        raise ValueError(
            f"Warm-start source must be {MODEL_NAME}, "
            f"got {payload.get('model')!r}"
        )

    if "ema_params" not in payload:
        raise ValueError("Warm-start source checkpoint has no ema_params")

    params = serialization.from_state_dict(
        template,
        payload["ema_params"],
    )

    count = _count_params(params)
    if count != EXPECTED_PARAMS:
        raise RuntimeError(
            f"Warm-start parameter mismatch: {count} != {EXPECTED_PARAMS}"
        )

    return params, {
        "source_model": payload.get("model"),
        "source_epoch": int(payload.get("epoch", -1)),
        "source_val_accuracy": float(
            payload.get("val_accuracy", float("nan"))
        ),
    }


def _trainable_mask(params):
    flat = flatten_dict(params)
    mask_flat = {}

    for path in flat:
        text = "/".join(map(str, path))
        mask_flat[path] = (
            text.startswith("descriptor_group/hier_fuse/")
            or text.startswith("descriptor_group/hier_norm/")
            or text.startswith("classifier_group/")
            or text.startswith("adaptive_head_u/")
            or text.startswith("adaptive_head_v/")
        )

    mask = unflatten_dict(mask_flat)

    trainable = int(
        sum(
            np.asarray(value).size
            for path, value in flat.items()
            if mask_flat[path]
        )
    )

    return mask, trainable


def create_state_geometry(config, steps_per_epoch, model_factory=None):
    total = config["epochs"] * steps_per_epoch
    warm = max(1, int(total * config["warmup_fraction"]))
    warm = min(warm, max(total - 1, 1))

    schedule = optax.warmup_cosine_decay_schedule(
        0,
        config["learning_rate"],
        warm,
        max(total, warm + 1),
        end_value=config["min_learning_rate"],
    )

    model = (model_factory or make_model)(config)

    key, init = jax.random.split(
        jax.random.PRNGKey(config["seed"])
    )

    template = model.init(
        {"params": init, "dropout": init},
        jnp.zeros((1, base.FRAMES, base.FEATURES)),
        training=False,
    )["params"]

    count = _count_params(template)
    if count != EXPECTED_PARAMS:
        raise RuntimeError(
            f"Model parameter mismatch: {count} != {EXPECTED_PARAMS}"
        )

    params, source_meta = _source_params(
        config["source_checkpoint"],
        template,
    )

    mask, trainable_count = _trainable_mask(params)
    frozen_mask = jax.tree.map(lambda x: not x, mask)

    # Frozen leaves:
    #   masked AdamW passes their zeroed transform input through,
    #   then masked set_to_zero guarantees no residual update reaches them.
    optimizer = optax.chain(
        optax.clip_by_global_norm(config["grad_clip"]),
        optax.masked(
            optax.adamw(
                schedule,
                weight_decay=config["weight_decay"],
            ),
            mask,
        ),
        optax.masked(
            optax.set_to_zero(),
            frozen_mask,
        ),
    )

    state = base.State.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        ema_params=params,
    )

    # Visible in worker log; does not alter resume semantics.
    print(
        f"WARM START | E{source_meta['source_epoch']:02d} "
        f"| source val={100*source_meta['source_val_accuracy']:.4f}% "
        f"| trainable={trainable_count:,}/{EXPECTED_PARAMS:,} "
        f"({100*trainable_count/EXPECTED_PARAMS:.2f}%)",
        flush=True,
    )
    print(
        "TRAINABLE ROOTS:",
        ", ".join(TRAINABLE_ROOTS),
        flush=True,
    )

    return (
        model,
        state,
        key,
        schedule,
        int(math.ceil(warm / steps_per_epoch)),
    )


def _fused_descriptor(out):
    desc = out["descriptors"]
    fusion = out["fusion_weights"]
    fused = jnp.einsum("bs,bsd->bd", fusion, desc)
    norm2 = jnp.sum(jnp.square(fused), axis=-1, keepdims=True)
    return fused * jax.lax.rsqrt(jnp.maximum(norm2, 1e-12))


def _descriptor_consistency(z, za):
    # Symmetric stop-gradient form. Numerically this is 1-cos(z,za), but
    # both branches receive a balanced gradient.
    a = 1.0 - jnp.sum(z * jax.lax.stop_gradient(za), axis=-1)
    b = 1.0 - jnp.sum(za * jax.lax.stop_gradient(z), axis=-1)
    return 0.5 * (a + b)


def _hard_negative_margin(z, za, y, mask, margin):
    valid = mask > 0
    different = y[:, None] != y[None, :]
    valid_pair = different & valid[:, None] & valid[None, :]

    # z -> augmented-view hard negatives.
    sim_ab = z @ jax.lax.stop_gradient(za).T
    neg_ab = jnp.max(
        jnp.where(valid_pair, sim_ab, -1e9),
        axis=1,
    )
    pos_ab = jnp.sum(
        z * jax.lax.stop_gradient(za),
        axis=-1,
    )
    loss_ab = jax.nn.relu(margin + neg_ab - pos_ab)

    # augmented-view -> canonical-view hard negatives.
    sim_ba = za @ jax.lax.stop_gradient(z).T
    neg_ba = jnp.max(
        jnp.where(valid_pair, sim_ba, -1e9),
        axis=1,
    )
    pos_ba = jnp.sum(
        za * jax.lax.stop_gradient(z),
        axis=-1,
    )
    loss_ba = jax.nn.relu(margin + neg_ba - pos_ba)

    # If a pathological microbatch contains only one class, no valid negatives
    # exist. Set that sample's margin term to zero.
    has_negative = jnp.any(valid_pair, axis=1)
    loss = 0.5 * (loss_ab + loss_ba)
    return jnp.where(has_negative, loss, 0.0)


def build_steps_geometry(model, config):
    def per_sample(params, key, batch):
        k1, k2 = jax.random.split(key)

        out = model.apply(
            {"params": params},
            batch["x"],
            training=True,
            rngs={"dropout": k1},
        )

        aug = model.apply(
            {"params": params},
            batch["xa"],
            training=True,
            rngs={"dropout": k2},
        )

        y = batch["y"]
        smooth = config["label_smoothing"]

        main = (
            base.ce(out["logits"], y, smooth)
            + base.ce(aug["logits"], y, smooth)
        ) / 2.0

        aux = (
            base.ce(out["stream_logits"], y[:, None], smooth).mean(1)
            + base.ce(aug["stream_logits"], y[:, None], smooth).mean(1)
        ) / 2.0

        temperature = config["consistency_temperature"]
        logp = jax.nn.log_softmax(out["logits"] / temperature)
        logq = jax.nn.log_softmax(aug["logits"] / temperature)

        kl = (
            0.5
            * temperature ** 2
            * jnp.sum(
                (jnp.exp(logp) - jnp.exp(logq))
                * (logp - logq),
                axis=-1,
            )
        )

        z = _fused_descriptor(out)
        za = _fused_descriptor(aug)

        desc_cons = _descriptor_consistency(z, za)

        margin_loss = _hard_negative_margin(
            z,
            za,
            y,
            batch["mask"],
            config["descriptor_margin"],
        )

        loss = (
            main
            + config["stream_aux_weight"] * aux
            + config["consistency_weight"] * kl
            + config["descriptor_consistency_weight"] * desc_cons
            + config["descriptor_margin_weight"] * margin_loss
        )

        acc = (
            out["logits"].argmax(-1) == y
        ).astype(jnp.float32)

        aug_acc = (
            aug["logits"].argmax(-1) == y
        ).astype(jnp.float32)

        agreement = (
            out["logits"].argmax(-1)
            == aug["logits"].argmax(-1)
        ).astype(jnp.float32)

        # Keep the historical 9-column shape so base.run remains fully
        # compatible. Columns 1..3 are diagnostic-only there.
        return jnp.stack(
            [
                loss,
                main,
                desc_cons,
                margin_loss,
                acc,
                aug_acc,
                agreement,
                out["sm_eta_mean"],
                out["sm_alpha_mean"],
            ],
            axis=-1,
        )

    @jax.jit
    def train_step(state, key, batch):
        k = config["accumulation_steps"]

        micros = jax.tree.map(
            lambda x: x.reshape(
                k,
                config["micro_batch"],
                *x.shape[1:],
            ),
            batch,
        )

        denom = jnp.maximum(jnp.sum(batch["mask"]), 1)

        key, step_key = jax.random.split(key)
        drop_keys = jax.random.split(step_key, k)

        zero = jax.tree.map(
            jnp.zeros_like,
            state.params,
        )

        def accumulate(carry, inputs):
            gradient_sum, metric_sum = carry
            micro, drop = inputs

            def loss_fn(params):
                metrics = per_sample(
                    params,
                    drop,
                    micro,
                )
                totals = jnp.sum(
                    metrics * micro["mask"][:, None],
                    axis=0,
                )
                return totals[0] / denom, totals

            (_, totals), gradients = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(state.params)

            return (
                (
                    jax.tree.map(
                        lambda a, b: a + b,
                        gradient_sum,
                        gradients,
                    ),
                    metric_sum + totals,
                ),
                None,
            )

        (grads, metrics), _ = jax.lax.scan(
            accumulate,
            (
                zero,
                jnp.zeros(9),
            ),
            (
                micros,
                drop_keys,
            ),
        )

        norm = optax.global_norm(grads)

        state = state.apply_gradients(
            grads=grads
        )

        ema = jax.tree.map(
            lambda e, p:
                config["ema_decay"] * e
                + (1.0 - config["ema_decay"]) * p,
            state.ema_params,
            state.params,
        )

        return (
            state.replace(
                ema_params=ema
            ),
            key,
            jnp.r_[
                metrics,
                denom,
                norm,
            ],
        )

    @jax.jit
    def eval_step(params, batch):
        out = model.apply(
            {"params": params},
            batch["x"],
            training=False,
        )

        y = batch["y"]
        mask = batch["mask"]
        logits = out["logits"]

        top5 = jax.lax.top_k(
            logits,
            5,
        )[1]

        columns = [
            base.ce(logits, y, 0),
            (
                logits.argmax(-1) == y
            ).astype(jnp.float32),
            (
                out["main_logits"].argmax(-1) == y
            ).astype(jnp.float32),
            jnp.any(
                top5 == y[:, None],
                axis=-1,
            ).astype(jnp.float32),
            out["sm_eta_mean"],
            out["sm_alpha_mean"],
            jnp.ones_like(mask),
        ]

        return jnp.sum(
            jnp.stack(columns, -1)
            * mask[:, None],
            axis=0,
        )

    return train_step, eval_step


def _identity(config):
    source = Path(config["source_checkpoint"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    root = Path(__file__).resolve().parents[1]
    code = hashlib.sha256()
    for name in (
        "model_parallel.py",
        "streaming/worker_geometry_warm.py",
    ):
        p = root / name
        code.update(name.encode() + b"\0" + p.read_bytes() + b"\0")

    return {
        "version": EXPERIMENT_NAME,
        "architecture": MODEL_NAME,
        "parameters": EXPECTED_PARAMS,
        "source_checkpoint_sha256": digest,
        "geometry_objective": config["geometry_objective"],
        "descriptor_consistency_weight": config[
            "descriptor_consistency_weight"
        ],
        "descriptor_margin_weight": config[
            "descriptor_margin_weight"
        ],
        "descriptor_margin": config["descriptor_margin"],
        "trainable_roots": list(TRAINABLE_ROOTS),
        "warm_start_baseline_is_epoch0_best": True,
        "g4_frozen": True,
        "source_sha256": code.hexdigest(),
    }


def _source_baseline(path):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    if payload.get("model") != MODEL_NAME:
        raise ValueError(
            f"Warm-start source must be {MODEL_NAME}, "
            f"got {payload.get('model')!r}"
        )
    return float(payload["val_accuracy"]), int(payload.get("epoch", -1))


def _seed_public_baseline(config, protocol, outdir):
    """Make the source EMA checkpoint the immutable epoch-0 fallback best.

    This fixes warm-start bookkeeping: a continuation epoch is only considered
    best if it actually beats the source checkpoint.
    """
    out = Path(outdir) / protocol
    out.mkdir(parents=True, exist_ok=True)
    last = out / "last.msgpack"

    baseline, source_epoch = _source_baseline(config["source_checkpoint"])

    # Only seed on a fresh continuation. Resume must preserve any later winner.
    if not last.exists():
        source_bytes = Path(config["source_checkpoint"]).read_bytes()
        from experiments.nestsar_sm_all_t16.streaming.io_utils import (
            atomic_bytes,
            atomic_json,
        )
        atomic_bytes(out / "best.msgpack", source_bytes)
        atomic_json(
            out / "best.json",
            {
                "model": MODEL_NAME,
                "epoch": 0,
                "source_epoch": source_epoch,
                "val_accuracy": baseline,
                "params": EXPECTED_PARAMS,
                "warm_start_baseline": True,
            },
        )

    return baseline


def run(config, protocol, cache, outdir, allow_cpu=False):
    config = validate_geometry_config(config)

    # base.run/create_state intentionally use R4 globals. Override only inside
    # this isolated worker process.
    base.EXPECTED_PARAMS = EXPECTED_PARAMS

    baseline = _seed_public_baseline(config, protocol, outdir)

    original_create_state = base.create_state
    original_build_steps = base.build_steps
    original_stopping_update = base.stopping_update

    def stopping_update_from_source(
        best,
        bad_epochs,
        val_acc,
        epoch,
        warmup_epochs,
        min_delta,
    ):
        # base.run initializes best=-1. For warm continuation the real epoch-0
        # best is the source checkpoint, not the first continuation epoch.
        effective_best = baseline if best < 0 else best
        improved = val_acc > effective_best + min_delta
        new_best = val_acc if improved else effective_best
        bad = (
            0
            if improved or epoch <= warmup_epochs
            else bad_epochs + 1
        )
        return new_best, bad, improved

    base.create_state = create_state_geometry
    base.build_steps = build_steps_geometry
    base.stopping_update = stopping_update_from_source

    try:
        return base.run(
            config,
            protocol,
            cache,
            outdir,
            allow_cpu,
            model_factory=make_model,
            model_name=MODEL_NAME,
            model_identity=_identity(config),
            config_validator=validate_geometry_config,
        )
    finally:
        base.create_state = original_create_state
        base.build_steps = original_build_steps
        base.stopping_update = original_stopping_update


if __name__ == "__main__":
    print(EXPERIMENT_NAME, flush=True)
    print(
        "Warm-start D128 | frozen early hierarchy | "
        "descriptor geometry continuation",
        flush=True,
    )
    base.main(run)

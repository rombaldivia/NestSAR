"""From-zero D128 training with training-only EMA class-prototype margin.

This worker keeps the inference architecture exactly identical to the proven
NestSAR-FULL-PARALLEL-T16-D128-MTS-v1 model.  The only change is a training-time
class-geometry regularizer driven by a 120 x D EMA prototype bank.

The bank is stored in last.msgpack, so rerunning the same Kaggle cell resumes
optimizer, EMA weights, RNG, prototype bank, prototype seen-mask and epoch.
It is NOT stored in best.msgpack and is never used at inference.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from experiments.nestsar_sm_all_t16.model_parallel import NestSARParallelT16
from experiments.nestsar_sm_all_t16.parallel_d128_config import (
    EXPECTED_PARAMS,
    MODEL_DIM,
    M4_HALF_LIVES,
    G4_HALF_LIVES,
    implementation_identity as d128_implementation_identity,
    validate_d128_config,
)
from experiments.nestsar_sm_all_t16.preprocessing_corrected import FRAMES, FEATURES
from experiments.nestsar_sm_all_t16.streaming import launch as launch_base
from experiments.nestsar_sm_all_t16.streaming import worker as base


MODEL_NAME = "NestSAR-FULL-PARALLEL-T16-D128-MTS-PROTOMARGIN-v1"
NUM_CLASSES = 120

PROTOTYPE_DEFAULTS = {
    "prototype_margin": 0.20,
    "prototype_weight": 0.10,
    "prototype_momentum": 0.99,
    "prototype_delay_epochs": 4,
    "prototype_ramp_epochs": 10,
}

_STEPS_PER_EPOCH = None


def _normalize(x):
    return x * jax.lax.rsqrt(
        jnp.maximum(jnp.sum(jnp.square(x), axis=-1, keepdims=True), 1e-12)
    )


def _fused_descriptor(out):
    z = jnp.einsum(
        "bs,bsd->bd",
        out["fusion_weights"],
        out["descriptors"],
    )
    return _normalize(z)


def make_model(config):
    if int(config["model_dim"]) != MODEL_DIM:
        raise ValueError(
            f"This experiment requires model_dim={MODEL_DIM}, got {config['model_dim']}"
        )
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


def validate_prototype_config(config):
    allowed = set(launch_base.DEFAULTS) | set(PROTOTYPE_DEFAULTS)
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    d128_input = {
        k: v
        for k, v in config.items()
        if k in launch_base.DEFAULTS
    }
    c = validate_d128_config(d128_input)

    p = dict(PROTOTYPE_DEFAULTS)
    p.update(
        {
            k: config[k]
            for k in PROTOTYPE_DEFAULTS
            if k in config
        }
    )

    if not 0.0 <= float(p["prototype_margin"]) <= 1.0:
        raise ValueError("prototype_margin must be in [0,1]")
    if float(p["prototype_weight"]) < 0.0:
        raise ValueError("prototype_weight must be nonnegative")
    if not 0.0 <= float(p["prototype_momentum"]) < 1.0:
        raise ValueError("prototype_momentum must be in [0,1)")
    for key in ("prototype_delay_epochs", "prototype_ramp_epochs"):
        if not isinstance(p[key], int) or p[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if p["prototype_ramp_epochs"] == 0 and p["prototype_weight"] > 0:
        # Legal, but make the intended step schedule explicit.
        p["prototype_ramp_epochs"] = 0

    c.update(p)
    return c


def implementation_identity():
    worker_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "version": MODEL_NAME,
        "inference_architecture": d128_implementation_identity(),
        "training_only_change": "ema_class_prototype_hinge_margin",
        "worker_sha256": worker_sha,
        "prototype_bank_shape": [NUM_CLASSES, MODEL_DIM],
        "prototype_bank_in_inference": False,
    }


class PrototypeState(base.State):
    prototype_bank: object
    prototype_seen: object


def create_state(config, steps_per_epoch, model_factory=None):
    global _STEPS_PER_EPOCH
    _STEPS_PER_EPOCH = int(steps_per_epoch)

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

    optimizer = optax.chain(
        optax.clip_by_global_norm(config["grad_clip"]),
        optax.adamw(
            schedule,
            weight_decay=config["weight_decay"],
        ),
    )

    model = (model_factory or make_model)(config)

    key, init = jax.random.split(
        jax.random.PRNGKey(config["seed"])
    )

    params = model.init(
        {"params": init, "dropout": init},
        jnp.zeros((1, FRAMES, FEATURES)),
        training=False,
    )["params"]

    count = sum(x.size for x in jax.tree.leaves(params))
    if count != EXPECTED_PARAMS:
        raise RuntimeError(
            f"Model parameter mismatch: {count} != {EXPECTED_PARAMS}"
        )

    state = PrototypeState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        ema_params=params,
        prototype_bank=jnp.zeros(
            (NUM_CLASSES, MODEL_DIM),
            dtype=jnp.float32,
        ),
        prototype_seen=jnp.zeros(
            (NUM_CLASSES,),
            dtype=jnp.bool_,
        ),
    )

    return (
        model,
        state,
        key,
        schedule,
        int(math.ceil(warm / steps_per_epoch)),
    )


def _prototype_weight(step, config):
    if _STEPS_PER_EPOCH is None:
        raise RuntimeError("steps_per_epoch was not initialized")

    delay_steps = (
        int(config["prototype_delay_epochs"])
        * _STEPS_PER_EPOCH
    )
    ramp_steps = (
        int(config["prototype_ramp_epochs"])
        * _STEPS_PER_EPOCH
    )

    step_f = step.astype(jnp.float32)
    delay_f = jnp.asarray(delay_steps, jnp.float32)

    if ramp_steps <= 0:
        factor = (step_f >= delay_f).astype(jnp.float32)
    else:
        factor = jnp.clip(
            (step_f - delay_f)
            / jnp.asarray(ramp_steps, jnp.float32),
            0.0,
            1.0,
        )

    return (
        jnp.asarray(config["prototype_weight"], jnp.float32)
        * factor
    )


def _prototype_loss(z, labels, bank, seen, margin):
    """Per-sample true-prototype vs nearest-wrong-prototype hinge."""
    sims = z @ bank.T

    rows = jnp.arange(labels.shape[0])
    true_sim = sims[rows, labels]

    wrong_mask = (
        seen[None, :]
        & (
            jnp.arange(NUM_CLASSES)[None, :]
            != labels[:, None]
        )
    )

    wrong_sim = jnp.max(
        jnp.where(
            wrong_mask,
            sims,
            jnp.asarray(-1e9, sims.dtype),
        ),
        axis=-1,
    )

    valid = (
        seen[labels]
        & jnp.any(wrong_mask, axis=-1)
    )

    loss = jnp.where(
        valid,
        jax.nn.relu(
            jnp.asarray(margin, z.dtype)
            + wrong_sim
            - true_sim
        ),
        0.0,
    )

    active = (
        valid
        & (loss > 0)
    ).astype(jnp.float32)

    return loss, active


def build_steps(model, config):
    if _STEPS_PER_EPOCH is None:
        raise RuntimeError(
            "create_state must run before build_steps"
        )

    def per_sample(
        params,
        key,
        batch,
        prototype_bank,
        prototype_seen,
        prototype_weight,
    ):
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
        ) / 2

        aux = (
            base.ce(
                out["stream_logits"],
                y[:, None],
                smooth,
            ).mean(1)
            + base.ce(
                aug["stream_logits"],
                y[:, None],
                smooth,
            ).mean(1)
        ) / 2

        temperature = config["consistency_temperature"]

        logp = jax.nn.log_softmax(
            out["logits"] / temperature
        )
        logq = jax.nn.log_softmax(
            aug["logits"] / temperature
        )

        kl = (
            0.5
            * temperature ** 2
            * jnp.sum(
                (
                    jnp.exp(logp)
                    - jnp.exp(logq)
                )
                * (logp - logq),
                axis=-1,
            )
        )

        z = _fused_descriptor(out)
        za = _fused_descriptor(aug)

        bank = jax.lax.stop_gradient(
            prototype_bank
        )
        seen = jax.lax.stop_gradient(
            prototype_seen
        )

        proto_a, active_a = _prototype_loss(
            z,
            y,
            bank,
            seen,
            config["prototype_margin"],
        )
        proto_b, active_b = _prototype_loss(
            za,
            y,
            bank,
            seen,
            config["prototype_margin"],
        )

        proto = 0.5 * (
            proto_a + proto_b
        )
        proto_active = 0.5 * (
            active_a + active_b
        )

        loss = (
            main
            + config["stream_aux_weight"] * aux
            + config["consistency_weight"] * kl
            + prototype_weight * proto
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

        # One stable feature per sample for the EMA prototype bank.
        # This is stop-gradient and updated only AFTER the optimizer step,
        # so the current batch cannot trivially satisfy its own prototype loss.
        proto_feature = jax.lax.stop_gradient(
            _normalize(
                0.5 * (z + za)
            )
        )

        # Metrics 0..6 keep the exact historical layout expected by base.run.
        # Slots 7/8 are training-only diagnostics; base.run does not use them.
        metrics = jnp.stack(
            [
                loss,
                main,
                aux,
                kl,
                acc,
                aug_acc,
                agreement,
                proto,
                proto_active,
            ],
            axis=-1,
        )

        return metrics, proto_feature

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

        denom = jnp.maximum(
            jnp.sum(batch["mask"]),
            1,
        )

        key, step_key = jax.random.split(key)
        drop_keys = jax.random.split(step_key, k)

        zero_grads = jax.tree.map(
            jnp.zeros_like,
            state.params,
        )

        prototype_weight = _prototype_weight(
            state.step,
            config,
        )

        def accumulate(carry, inputs):
            (
                gradient_sum,
                metric_sum,
                class_sum,
                class_count,
            ) = carry

            micro, drop = inputs

            def loss_fn(params):
                metrics, proto_feature = per_sample(
                    params,
                    drop,
                    micro,
                    state.prototype_bank,
                    state.prototype_seen,
                    prototype_weight,
                )

                totals = jnp.sum(
                    metrics
                    * micro["mask"][:, None],
                    axis=0,
                )

                feature = (
                    proto_feature
                    * micro["mask"][:, None]
                )

                micro_class_sum = jnp.zeros(
                    (NUM_CLASSES, MODEL_DIM),
                    dtype=feature.dtype,
                ).at[
                    micro["y"]
                ].add(
                    feature
                )

                micro_class_count = jnp.zeros(
                    (NUM_CLASSES,),
                    dtype=jnp.float32,
                ).at[
                    micro["y"]
                ].add(
                    micro["mask"]
                )

                return (
                    totals[0] / denom,
                    (
                        totals,
                        micro_class_sum,
                        micro_class_count,
                    ),
                )

            (
                (_, stats),
                gradients,
            ) = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(
                state.params
            )

            (
                totals,
                micro_class_sum,
                micro_class_count,
            ) = stats

            return (
                (
                    jax.tree.map(
                        lambda a, b: a + b,
                        gradient_sum,
                        gradients,
                    ),
                    metric_sum + totals,
                    class_sum + micro_class_sum,
                    class_count + micro_class_count,
                ),
                None,
            )

        init_carry = (
            zero_grads,
            jnp.zeros(
                (9,),
                dtype=jnp.float32,
            ),
            jnp.zeros(
                (NUM_CLASSES, MODEL_DIM),
                dtype=jnp.float32,
            ),
            jnp.zeros(
                (NUM_CLASSES,),
                dtype=jnp.float32,
            ),
        )

        (
            (
                grads,
                metrics,
                class_sum,
                class_count,
            ),
            _,
        ) = jax.lax.scan(
            accumulate,
            init_carry,
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
            lambda e, p: (
                config["ema_decay"] * e
                + (1 - config["ema_decay"]) * p
            ),
            state.ema_params,
            state.params,
        )

        present = class_count > 0

        batch_mean = (
            class_sum
            / jnp.maximum(
                class_count[:, None],
                1.0,
            )
        )
        batch_mean = _normalize(batch_mean)

        momentum = jnp.asarray(
            config["prototype_momentum"],
            jnp.float32,
        )

        ema_candidate = _normalize(
            momentum * state.prototype_bank
            + (1.0 - momentum) * batch_mean
        )

        candidate = jnp.where(
            state.prototype_seen[:, None],
            ema_candidate,
            batch_mean,
        )

        new_bank = jnp.where(
            present[:, None],
            candidate,
            state.prototype_bank,
        )

        new_seen = (
            state.prototype_seen
            | present
        )

        state = state.replace(
            ema_params=ema,
            prototype_bank=new_bank,
            prototype_seen=new_seen,
        )

        # Historical base.run expects:
        # metrics[0:9], denom at index 9, norm after that.
        return (
            state,
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


def run(
    config,
    protocol,
    cache,
    outdir,
    allow_cpu=False,
):
    # Dedicated subprocess: safe to swap these hooks only here.
    base.EXPECTED_PARAMS = EXPECTED_PARAMS
    base.create_state = create_state
    base.build_steps = build_steps

    return base.run(
        config,
        protocol,
        cache,
        outdir,
        allow_cpu,
        model_factory=make_model,
        model_name=MODEL_NAME,
        model_identity=implementation_identity(),
        config_validator=validate_prototype_config,
    )


if __name__ == "__main__":
    print(MODEL_NAME, flush=True)
    print(
        f"inference params={EXPECTED_PARAMS:,} "
        f"model_dim={MODEL_DIM}",
        flush=True,
    )
    print(
        "prototype bank=120x128 training-only | "
        f"margin={PROTOTYPE_DEFAULTS['prototype_margin']} | "
        f"max_weight={PROTOTYPE_DEFAULTS['prototype_weight']} | "
        f"momentum={PROTOTYPE_DEFAULTS['prototype_momentum']} | "
        f"delay={PROTOTYPE_DEFAULTS['prototype_delay_epochs']} epochs | "
        f"ramp={PROTOTYPE_DEFAULTS['prototype_ramp_epochs']} epochs",
        flush=True,
    )
    base.main(run)

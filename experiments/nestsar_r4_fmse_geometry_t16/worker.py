from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import closing
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training import train_state

from experiments.nestsar_sm_all_t16.preprocessing_corrected import (
    FEATURES,
    FRAMES,
    VERSION as PREPROCESSING_VERSION,
)
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming.io_utils import (
    Reporter,
    atomic_bytes,
    atomic_json,
    read_json,
)
from experiments.nestsar_sm_all_t16.streaming import worker as r4_worker
from experiments.nestsar_r4_fmse_t16.model import NestSARR4FMSET16

from . import MODEL_IDENTITY, MODEL_NAME, VERSION
from .config import validate_config, geometry_scale
from .geometry import (
    DIM,
    NUM_CLASSES,
    fused_representations,
    local_geometry_terms,
    update_subcenters,
)

EXPECTED_PARAMS = 1_831_932


class State(train_state.TrainState):
    ema_params: object
    proto_desc: object
    proto_g4: object
    proto_desc_count: object
    proto_g4_count: object


def make_model(config):
    return NestSARR4FMSET16(
        **{
            k: config[k]
            for k in (
                "spatial_dim",
                "model_dim",
                "dropout",
                "controller_dim",
                "fast_rank",
                "head_rank",
                "sm_residual_scale",
                "head_residual_scale",
            )
        }
    )


def ce(logits, labels, smoothing):
    target = jax.nn.one_hot(labels, NUM_CLASSES)
    target = target * (1 - smoothing) + smoothing / NUM_CLASSES
    return -jnp.sum(target * jax.nn.log_softmax(logits), axis=-1)


def geometry_for_view(out, y, state, config):
    desc_rep, g4_rep = fused_representations(out)
    desc = local_geometry_terms(
        desc_rep,
        y,
        state.proto_desc,
        state.proto_desc_count,
        margin=config["geometry_margin"],
        rival_k=config["geometry_rival_k"],
        rival_temperature=config["geometry_rival_temperature"],
    )
    g4 = local_geometry_terms(
        g4_rep,
        y,
        state.proto_g4,
        state.proto_g4_count,
        margin=config["geometry_margin"],
        rival_k=config["geometry_rival_k"],
        rival_temperature=config["geometry_rival_temperature"],
    )
    return desc_rep, g4_rep, desc, g4


def build_steps(model, config):
    def per_sample(params, key, batch, state, geo_scale):
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
            ce(out["logits"], y, smooth)
            + ce(aug["logits"], y, smooth)
        ) / 2

        aux = (
            ce(out["stream_logits"], y[:, None], smooth).mean(1)
            + ce(aug["stream_logits"], y[:, None], smooth).mean(1)
        ) / 2

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

        desc_rep, g4_rep, desc_terms, g4_terms = geometry_for_view(
            out,
            y,
            state,
            config,
        )
        _, _, desc_aug_terms, g4_aug_terms = geometry_for_view(
            aug,
            y,
            state,
            config,
        )

        desc_loss = (desc_terms[0] + desc_aug_terms[0]) / 2
        g4_loss = (g4_terms[0] + g4_aug_terms[0]) / 2
        desc_active = (desc_terms[1] + desc_aug_terms[1]) / 2
        g4_active = (g4_terms[1] + g4_aug_terms[1]) / 2
        desc_pos = (desc_terms[2] + desc_aug_terms[2]) / 2
        desc_rival = (desc_terms[3] + desc_aug_terms[3]) / 2
        desc_gap = (desc_terms[4] + desc_aug_terms[4]) / 2
        g4_gap = (g4_terms[4] + g4_aug_terms[4]) / 2

        geo_added = geo_scale * (
            config["geometry_desc_weight"] * desc_loss
            + config["geometry_g4_weight"] * g4_loss
        )

        loss = (
            main
            + config["stream_aux_weight"] * aux
            + config["consistency_weight"] * kl
            + geo_added
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

        metrics = jnp.stack(
            [
                loss,
                main,
                aux,
                kl,
                acc,
                aug_acc,
                agreement,
                out["sm_eta_mean"],
                out["sm_alpha_mean"],
                geo_added,
                desc_loss,
                g4_loss,
                desc_active,
                g4_active,
                desc_pos,
                desc_rival,
                desc_gap,
                g4_gap,
            ],
            axis=-1,
        )
        return metrics, desc_rep, g4_rep

    @jax.jit
    def train_step(state, key, batch, geo_scale):
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
        zero = jax.tree.map(jnp.zeros_like, state.params)

        def accumulate(carry, inputs):
            gradient_sum, metric_sum = carry
            micro, drop = inputs

            def loss_fn(params):
                metrics, desc_rep, g4_rep = per_sample(
                    params,
                    drop,
                    micro,
                    state,
                    geo_scale,
                )
                totals = jnp.sum(
                    metrics * micro["mask"][:, None],
                    axis=0,
                )
                return totals[0] / denom, (
                    totals,
                    jax.lax.stop_gradient(desc_rep),
                    jax.lax.stop_gradient(g4_rep),
                )

            (_, (totals, desc_rep, g4_rep)), gradients = (
                jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            )

            return (
                (
                    jax.tree.map(
                        lambda a, b: a + b,
                        gradient_sum,
                        gradients,
                    ),
                    metric_sum + totals,
                ),
                (desc_rep, g4_rep),
            )

        (
            (grads, metrics),
            (desc_stack, g4_stack),
        ) = jax.lax.scan(
            accumulate,
            (
                zero,
                jnp.zeros(18, dtype=jnp.float32),
            ),
            (
                micros,
                drop_keys,
            ),
        )

        norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)

        ema = jax.tree.map(
            lambda e, p: (
                config["ema_decay"] * e
                + (1 - config["ema_decay"]) * p
            ),
            state.ema_params,
            state.params,
        )

        desc_batch = desc_stack.reshape(-1, DIM)
        g4_batch = g4_stack.reshape(-1, DIM)
        y_batch = micros["y"].reshape(-1)
        mask_batch = micros["mask"].reshape(-1)

        proto_desc, proto_desc_count = update_subcenters(
            state.proto_desc,
            state.proto_desc_count,
            desc_batch,
            y_batch,
            mask_batch,
            config["geometry_proto_momentum"],
        )
        proto_g4, proto_g4_count = update_subcenters(
            state.proto_g4,
            state.proto_g4_count,
            g4_batch,
            y_batch,
            mask_batch,
            config["geometry_proto_momentum"],
        )

        state = state.replace(
            ema_params=ema,
            proto_desc=proto_desc,
            proto_g4=proto_g4,
            proto_desc_count=proto_desc_count,
            proto_g4_count=proto_g4_count,
        )

        return state, key, jnp.r_[metrics, denom, norm]

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
        top5 = jax.lax.top_k(logits, 5)[1]
        columns = [
            ce(logits, y, 0),
            (logits.argmax(-1) == y).astype(jnp.float32),
            (out["main_logits"].argmax(-1) == y).astype(jnp.float32),
            jnp.any(top5 == y[:, None], axis=-1).astype(jnp.float32),
            out["sm_eta_mean"],
            out["sm_alpha_mean"],
            jnp.ones_like(mask),
        ]
        return jnp.sum(
            jnp.stack(columns, -1) * mask[:, None],
            axis=0,
        )

    return train_step, eval_step


def create_state(config, steps_per_epoch):
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

    model = make_model(config)
    key, init = jax.random.split(jax.random.PRNGKey(config["seed"]))
    params = model.init(
        {"params": init, "dropout": init},
        jnp.zeros((1, FRAMES, FEATURES), dtype=jnp.float32),
        training=False,
    )["params"]

    count = sum(x.size for x in jax.tree.leaves(params))
    if count != EXPECTED_PARAMS:
        raise RuntimeError(
            f"Model parameter mismatch: {count} != {EXPECTED_PARAMS}"
        )

    k = config["geometry_subcenters"]
    proto_shape = (NUM_CLASSES, k, DIM)
    proto_desc = jnp.zeros(proto_shape, dtype=jnp.float32)
    proto_g4 = jnp.zeros(proto_shape, dtype=jnp.float32)
    proto_count_desc = jnp.zeros((NUM_CLASSES, k), dtype=jnp.float32)
    proto_count_g4 = jnp.zeros((NUM_CLASSES, k), dtype=jnp.float32)

    state = State.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        ema_params=params,
        proto_desc=proto_desc,
        proto_g4=proto_g4,
        proto_desc_count=proto_count_desc,
        proto_g4_count=proto_count_g4,
    )
    return (
        model,
        state,
        key,
        schedule,
        int(math.ceil(warm / steps_per_epoch)),
    )


def save_checkpoint(path, state, key, metadata):
    payload = {
        "state": serialization.to_state_dict(jax.device_get(state)),
        "key": np.asarray(key),
        "metadata_json": json.dumps(metadata, allow_nan=False),
    }
    atomic_bytes(path, serialization.msgpack_serialize(payload))


def restore_checkpoint(path, template, expected_hash):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    meta = json.loads(payload["metadata_json"])
    if meta["config_hash"] != expected_hash:
        raise ValueError(
            "Resume config/data mismatch. Keep the original config "
            "or choose a new OUT_DIR."
        )
    state = serialization.from_state_dict(template, payload["state"])
    return jax.device_put(state), jax.device_put(payload["key"]), meta


def publish_best(out, metadata):
    name = metadata["best_checkpoint"]
    if Path(name).name != name or not name.startswith("best_epoch_"):
        raise ValueError("Invalid best-checkpoint filename")

    payload = (out / name).read_bytes()
    atomic_bytes(out / "best.msgpack", payload)
    atomic_json(
        out / "best.json",
        dict(
            model=metadata.get("model", MODEL_NAME),
            epoch=metadata["best_epoch"],
            val_accuracy=metadata["best"],
            params=EXPECTED_PARAMS,
            preprocessing_version=PREPROCESSING_VERSION,
            pipeline_version=VERSION,
            config_hash=metadata["config_hash"],
            model_identity=metadata.get("model_identity"),
        ),
    )


def write_result(out, protocol, metadata, digest, resumed=False):
    result = {
        "model": metadata.get("model", MODEL_NAME),
        "protocol": protocol,
        "best_val_accuracy": metadata["best"],
        "best_accuracy": metadata["best"],
        "best_epoch": metadata["best_epoch"],
        "last_epoch": metadata["epoch"],
        "epochs_run": metadata["epoch"],
        "params": EXPECTED_PARAMS,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.local_devices()],
        "config_hash": digest,
        "checkpoint": str(out / "best.msgpack"),
        "preprocessing_version": PREPROCESSING_VERSION,
        "pipeline_version": VERSION,
        "stopped_early": metadata["stopped_early"],
        "resumed_completed": resumed,
        "model_identity": metadata.get("model_identity"),
        "geometry": {
            "descriptor_weight": metadata["config"]["geometry_desc_weight"],
            "g4_weight": metadata["config"]["geometry_g4_weight"],
            "margin": metadata["config"]["geometry_margin"],
            "subcenters": metadata["config"]["geometry_subcenters"],
            "rival_k": metadata["config"]["geometry_rival_k"],
        },
    }
    atomic_json(out / "result.json", result)
    atomic_json(out.parent / f"result_{protocol}.json", result)
    return result


def run_signature(config, protocol, cache_signature):
    return {
        "config": config,
        "protocol": protocol,
        "cache": cache_signature,
        "parameters": EXPECTED_PARAMS,
        "pipeline_version": VERSION,
        "model": MODEL_NAME,
        "model_identity": MODEL_IDENTITY,
    }


def run(config, protocol, cache, outdir, allow_cpu=False):
    config = validate_config(config)
    if protocol not in ("xsub", "xset"):
        raise ValueError("Expected xsub or xset")

    out = Path(outdir) / protocol
    out.mkdir(parents=True, exist_ok=True)

    if not allow_cpu and (
        jax.default_backend() != "gpu"
        or len(jax.local_devices()) != 1
    ):
        raise RuntimeError(
            "Expected one isolated GPU; "
            f"backend={jax.default_backend()}, devices={jax.local_devices()}"
        )

    dataset = Dataset(cache)
    train_ids = dataset.splits[f"{protocol}_train"]
    val_ids = dataset.splits[f"{protocol}_val"]

    if config["max_train_samples"]:
        train_ids = train_ids[:config["max_train_samples"]]
    if config["max_val_samples"]:
        val_ids = val_ids[:config["max_val_samples"]]
    if not train_ids or not val_ids:
        raise ValueError("Empty train/validation split")

    batch_size = config["micro_batch"] * config["accumulation_steps"]
    steps = math.ceil(len(train_ids) / batch_size)

    signature = run_signature(
        config,
        protocol,
        dataset.meta["signature"],
    )
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()

    previous = read_json(out / "run_config.json")
    if previous is not None and previous != signature:
        raise ValueError(
            "OUT_DIR contains another model, implementation, or config/data. "
            "Choose a new OUT_DIR."
        )
    atomic_json(out / "run_config.json", signature)

    report = Reporter(out / "status.json")
    report(
        phase="JAX startup",
        current=0,
        total=1,
        protocol=protocol,
        epoch=0,
        best=None,
        best_epoch=0,
        completed_epoch=0,
    )

    model, state, key, schedule, warmup_epochs = create_state(
        config,
        steps,
    )

    metadata = {
        "config_hash": digest,
        "config": config,
        "epoch": 0,
        "best": -1.0,
        "best_epoch": 0,
        "model": MODEL_NAME,
        "model_identity": MODEL_IDENTITY,
        "bad_epochs": 0,
        "history": [],
        "warmup_epochs": warmup_epochs,
        "best_checkpoint": None,
        "stopped_early": False,
    }

    last = out / "last.msgpack"

    if last.exists():
        state, key, metadata = restore_checkpoint(
            last,
            state,
            digest,
        )
        if metadata["best_epoch"]:
            publish_best(out, metadata)
        atomic_json(out / "history.json", metadata["history"])

    report(
        best=(
            metadata["best"]
            if metadata["best_epoch"]
            else None
        ),
        best_epoch=metadata["best_epoch"],
        completed_epoch=metadata["epoch"],
    )

    if metadata["epoch"] >= config["epochs"] or metadata["stopped_early"]:
        result = write_result(
            out,
            protocol,
            metadata,
            digest,
            resumed=True,
        )
        report(
            phase="Done",
            current=1,
            total=1,
            done=True,
            epoch=metadata["epoch"],
        )
        return result

    train_step, eval_step = build_steps(model, config)
    compiled_train = None
    compiled_eval = None

    for epoch in range(
        metadata["epoch"] + 1,
        config["epochs"] + 1,
    ):
        epoch_t0 = time.perf_counter()
        geo_scale_value = geometry_scale(config, epoch)
        geo_scale_device = jnp.asarray(
            geo_scale_value,
            dtype=jnp.float32,
        )

        report(
            phase="Train",
            epoch=epoch,
            current=0,
            total=steps,
            val_acc=None,
            best=(
                metadata["best"]
                if metadata["best_epoch"]
                else None
            ),
            best_epoch=metadata["best_epoch"],
            geometry_scale=geo_scale_value,
        )

        # 18 metrics + denominator.
        train_sum = np.zeros(19, np.float64)

        timing = {
            "prepare_service_s": 0.0,
            "data_wait_s": 0.0,
            "h2d_s": 0.0,
            "gpu_train_s": 0.0,
            "gpu_eval_s": 0.0,
            "compile_train_s": 0.0,
            "compile_eval_s": 0.0,
            "warmup_train_s": 0.0,
            "warmup_eval_s": 0.0,
        }

        with closing(
            dataset.batches(
                train_ids,
                batch_size,
                config,
                epoch,
                True,
                protocol,
            )
        ) as batches:
            for index in range(steps):
                start = time.perf_counter()
                batch, prep_s = next(batches)
                timing["data_wait_s"] += time.perf_counter() - start
                timing["prepare_service_s"] += prep_s

                start = time.perf_counter()
                device_batch = jax.block_until_ready(jax.device_put(batch))
                timing["h2d_s"] += time.perf_counter() - start

                if compiled_train is None:
                    report(
                        phase="Compile train",
                        current=0,
                        total=steps,
                    )
                    start = time.perf_counter()
                    compiled_train = train_step.lower(
                        state,
                        key,
                        device_batch,
                        geo_scale_device,
                    ).compile()
                    timing["compile_train_s"] = time.perf_counter() - start

                    start = time.perf_counter()
                    warm = jax.block_until_ready(
                        compiled_train(
                            state,
                            key,
                            device_batch,
                            geo_scale_device,
                        )
                    )
                    del warm
                    timing["warmup_train_s"] = time.perf_counter() - start

                start = time.perf_counter()
                state, key, metrics = jax.block_until_ready(
                    compiled_train(
                        state,
                        key,
                        device_batch,
                        geo_scale_device,
                    )
                )
                timing["gpu_train_s"] += time.perf_counter() - start

                values = np.asarray(metrics)
                if not np.isfinite(values).all():
                    raise FloatingPointError(
                        f"Nonfinite training values at epoch {epoch}, "
                        f"batch {index + 1}"
                    )

                # 18 summed metrics + denominator. Final returned element is grad norm.
                train_sum += values[:19]

                if (
                    index % config["progress_every"] == 0
                    or index + 1 == steps
                ):
                    denom = max(train_sum[18], 1.0)
                    report(
                        phase="Train",
                        current=index + 1,
                        total=steps,
                        loss=float(train_sum[0] / denom),
                        train_acc=float(train_sum[4] / denom),
                        geometry=float(train_sum[9] / denom),
                        geo_desc=float(train_sum[10] / denom),
                        geo_g4=float(train_sum[11] / denom),
                        geo_desc_active=float(train_sum[12] / denom),
                        geo_g4_active=float(train_sum[13] / denom),
                        geometry_scale=geo_scale_value,
                        lr=float(schedule(state.step)),
                        wait_s=timing["data_wait_s"],
                        gpu_s=timing["gpu_train_s"],
                        **r4_worker.memory_snapshot(),
                    )

        if int(train_sum[18]) != len(train_ids):
            raise RuntimeError("Training sample accounting mismatch")

        eval_sum = np.zeros(7, np.float64)
        val_steps = math.ceil(len(val_ids) / config["eval_batch"])
        report(
            phase="Validate EMA",
            current=0,
            total=val_steps,
        )

        with closing(
            dataset.batches(
                val_ids,
                config["eval_batch"],
                config,
                protocol=protocol,
            )
        ) as batches:
            for index in range(val_steps):
                start = time.perf_counter()
                batch, prep_s = next(batches)
                timing["data_wait_s"] += time.perf_counter() - start
                timing["prepare_service_s"] += prep_s

                start = time.perf_counter()
                device_batch = jax.block_until_ready(jax.device_put(batch))
                timing["h2d_s"] += time.perf_counter() - start

                if compiled_eval is None:
                    report(
                        phase="Compile validation",
                        current=0,
                        total=val_steps,
                    )
                    start = time.perf_counter()
                    compiled_eval = eval_step.lower(
                        state.ema_params,
                        device_batch,
                    ).compile()
                    timing["compile_eval_s"] = time.perf_counter() - start

                    start = time.perf_counter()
                    jax.block_until_ready(
                        compiled_eval(
                            state.ema_params,
                            device_batch,
                        )
                    )
                    timing["warmup_eval_s"] = time.perf_counter() - start

                start = time.perf_counter()
                vals = np.asarray(
                    jax.block_until_ready(
                        compiled_eval(
                            state.ema_params,
                            device_batch,
                        )
                    )
                )
                timing["gpu_eval_s"] += time.perf_counter() - start

                if not np.isfinite(vals).all():
                    raise FloatingPointError("Nonfinite validation values")

                eval_sum += vals

                if (
                    index % config["progress_every"] == 0
                    or index + 1 == val_steps
                ):
                    report(
                        phase="Validate EMA",
                        current=index + 1,
                        total=val_steps,
                        val_acc=float(eval_sum[1] / eval_sum[6]),
                    )

        if int(eval_sum[6]) != len(val_ids):
            raise RuntimeError("Validation mask/count mismatch")

        val = float(eval_sum[1] / eval_sum[6])

        best, bad, improved = r4_worker.stopping_update(
            metadata["best"],
            metadata["bad_epochs"],
            val,
            epoch,
            warmup_epochs,
            config["min_delta"],
        )

        denom = train_sum[18]

        row = {
            "epoch": epoch,
            "train_loss": float(train_sum[0] / denom),
            "train_acc": float(train_sum[4] / denom),
            "augmented_train_acc": float(train_sum[5] / denom),
            "val_loss": float(eval_sum[0] / eval_sum[6]),
            "val_acc": val,
            "val_main_acc": float(eval_sum[2] / eval_sum[6]),
            "val_top5": float(eval_sum[3] / eval_sum[6]),
            "eta": float(eval_sum[4] / eval_sum[6]),
            "alpha": float(eval_sum[5] / eval_sum[6]),
            "geometry_scale": geo_scale_value,
            "geometry_added": float(train_sum[9] / denom),
            "geometry_desc_loss": float(train_sum[10] / denom),
            "geometry_g4_loss": float(train_sum[11] / denom),
            "geometry_desc_active": float(train_sum[12] / denom),
            "geometry_g4_active": float(train_sum[13] / denom),
            "geometry_desc_positive": float(train_sum[14] / denom),
            "geometry_desc_rival": float(train_sum[15] / denom),
            "geometry_desc_gap": float(train_sum[16] / denom),
            "geometry_g4_gap": float(train_sum[17] / denom),
            "proto_desc_filled": int(
                np.sum(np.asarray(state.proto_desc_count) > 0)
            ),
            "proto_g4_filled": int(
                np.sum(np.asarray(state.proto_g4_count) > 0)
            ),
            "train_samples": len(train_ids),
            "val_samples": len(val_ids),
            "bad_epochs": bad,
            "epoch_s": time.perf_counter() - epoch_t0,
            **timing,
            **r4_worker.memory_snapshot(),
        }

        previous_best = metadata["best_checkpoint"]
        metadata.update(
            epoch=epoch,
            best=best,
            bad_epochs=bad,
            stopped_early=bad >= config["patience"],
        )

        report(
            phase="Save checkpoint",
            current=1,
            total=1,
        )

        if improved:
            metadata.update(
                best_epoch=epoch,
                best_checkpoint=f"best_epoch_{epoch:04d}.msgpack",
            )
            best_payload = {
                "model": MODEL_NAME,
                "model_identity": MODEL_IDENTITY,
                "protocol": protocol,
                "epoch": epoch,
                "val_accuracy": val,
                "ema_params": jax.device_get(state.ema_params),
                "config": config,
                "preprocessing_version": PREPROCESSING_VERSION,
                "pipeline_version": VERSION,
                "cache_signature": dataset.meta["signature"],
                "train_samples": len(train_ids),
                "val_samples": len(val_ids),
                "training_only_geometry": {
                    "descriptor_weight": config["geometry_desc_weight"],
                    "g4_weight": config["geometry_g4_weight"],
                    "margin": config["geometry_margin"],
                    "subcenters": config["geometry_subcenters"],
                    "rival_k": config["geometry_rival_k"],
                },
            }
            atomic_bytes(
                out / metadata["best_checkpoint"],
                serialization.to_bytes(best_payload),
            )

        row.update(
            best_val_accuracy=best,
            best_epoch=metadata["best_epoch"],
        )
        metadata["history"].append(row)

        save_checkpoint(
            last,
            state,
            key,
            metadata,
        )

        if improved:
            publish_best(out, metadata)
            if (
                previous_best
                and previous_best != metadata["best_checkpoint"]
            ):
                (out / previous_best).unlink(missing_ok=True)

        atomic_json(
            out / "history.json",
            metadata["history"],
        )

        report(
            phase="Epoch complete",
            best=best,
            best_epoch=metadata["best_epoch"],
            completed_epoch=epoch,
            val_acc=val,
            current=1,
            total=1,
            bad_epochs=bad,
            geometry=float(row["geometry_added"]),
            geo_desc_active=float(row["geometry_desc_active"]),
            geo_g4_active=float(row["geometry_g4_active"]),
            geometry_scale=geo_scale_value,
            epoch_s=row["epoch_s"],
            wait_s=timing["data_wait_s"],
            gpu_s=timing["gpu_train_s"],
        )

        if metadata["stopped_early"]:
            break

    result = write_result(
        out,
        protocol,
        metadata,
        digest,
    )
    report(
        phase="Done",
        current=1,
        total=1,
        done=True,
        best=metadata["best"],
        best_epoch=metadata["best_epoch"],
        epoch=metadata["epoch"],
        completed_epoch=metadata["epoch"],
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--protocol", required=True, choices=("xsub", "xset"))
    p.add_argument("--cache", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    run(
        json.loads(Path(a.config).read_text()),
        a.protocol,
        a.cache,
        a.outdir,
        a.allow_cpu,
    )


if __name__ == "__main__":
    main()

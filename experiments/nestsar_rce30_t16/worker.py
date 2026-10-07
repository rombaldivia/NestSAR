from __future__ import annotations

"""Frozen-FMSE Stage-2 trainer for NestSAR-RCE30.

Only the RCE specialist is optimized.  The base FMSE EMA checkpoint is loaded
once, never placed in the optimizer state, and is therefore exactly protected.
"""

import argparse
import csv
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

from experiments.nestsar_r4_fmse_t16.model import NestSARR4FMSET16
from experiments.nestsar_sm_all_t16.preprocessing_corrected import (
    FRAMES,
    FEATURES,
    VERSION as PREPROCESSING_VERSION,
)
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming.io_utils import (
    Reporter,
    atomic_bytes,
    atomic_json,
    read_json,
)

from . import MODEL_IDENTITY, MODEL_NAME, VERSION
from .model import (
    NestSARRCE30T16,
    NUM_CLASSES,
    RIVALS_PER_TOP,
)


class RCEState(train_state.TrainState):
    ema_params: object


def ce(logits, labels, smoothing):
    target = jax.nn.one_hot(labels, NUM_CLASSES)
    target = target * (1.0 - smoothing) + smoothing / NUM_CLASSES
    return -jnp.sum(target * jax.nn.log_softmax(logits), axis=-1)


def _sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_base_checkpoint(path):
    path = Path(path)
    payload = serialization.msgpack_restore(path.read_bytes())
    if "ema_params" not in payload or "config" not in payload:
        raise ValueError(
            f"Expected FMSE best checkpoint with ema_params/config: {path}"
        )
    cfg = payload["config"]
    model = NestSARR4FMSET16(
        spatial_dim=cfg["spatial_dim"],
        model_dim=cfg["model_dim"],
        dropout=cfg["dropout"],
        controller_dim=cfg["controller_dim"],
        fast_rank=cfg["fast_rank"],
        head_rank=cfg["head_rank"],
        sm_residual_scale=cfg["sm_residual_scale"],
        head_residual_scale=cfg["head_residual_scale"],
    )
    params = jax.device_put(payload["ema_params"])
    return model, params, payload


def make_specialist(config):
    return NestSARRCE30T16(
        variant=config["rce_variant"],
        dim=40,
        blocks=2,
        rank=4,
        dropout=config["rce_dropout"],
    )


def specialist_parameter_count(model, rival_table):
    key = jax.random.PRNGKey(7)
    variables = model.init(
        {"params": key, "dropout": key},
        jnp.zeros((1, FRAMES, FEATURES), jnp.float32),
        jnp.zeros((1, NUM_CLASSES), jnp.float32),
        rival_table,
        training=False,
    )
    return int(sum(x.size for x in jax.tree.leaves(variables["params"])))


def validate_config(config):
    defaults = dict(
        epochs=40,
        patience=6,
        micro_batch=64,
        accumulation_steps=4,
        eval_batch=256,
        seed=128,
        rce_variant="rival",
        rce_learning_rate=8e-4,
        rce_min_learning_rate=2e-5,
        rce_warmup_fraction=0.08,
        rce_weight_decay=0.02,
        rce_grad_clip=1.0,
        rce_ema_decay=0.995,
        rce_dropout=0.05,
        label_smoothing=0.02,
        rival_weight=0.10,
        protect_weight=0.20,
        consistency_weight=0.02,
        gate_weight=0.005,
        protect_margin=1.0,
        fresh_augmentation=True,
        rotation_degrees=8.0,
        jitter_shift=1,
        min_delta=1e-6,
        progress_every=5,
        max_train_samples=0,
        max_val_samples=0,
        prefetch_batches=2,
        rival_graph_batch=256,
        rival_graph_topk=5,
    )
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown RCE config keys: {sorted(unknown)}")
    c = {**defaults, **config}
    if c["rce_variant"] not in ("rival", "fixed"):
        raise ValueError("rce_variant must be rival or fixed")
    if c["micro_batch"] < 1 or c["accumulation_steps"] < 1:
        raise ValueError("micro_batch and accumulation_steps must be positive")
    if c["epochs"] < 1 or c["patience"] < 1:
        raise ValueError("epochs/patience must be positive")
    return c


def build_training_rivals(
    *,
    dataset,
    train_ids,
    base_model,
    base_params,
    config,
    protocol,
    out,
):
    """Build directed rivals only from training samples/augmentations."""

    npy = out / "training_rivals.npy"
    meta_path = out / "training_rivals.json"

    if npy.is_file() and meta_path.is_file():
        table = np.load(npy)
        meta = json.loads(meta_path.read_text())
        if table.shape == (NUM_CLASSES, RIVALS_PER_TOP):
            return jax.device_put(table.astype(np.int32)), meta

    batch_size = int(config["rival_graph_batch"])
    topk = int(config["rival_graph_topk"])

    @jax.jit
    def infer(x):
        return base_model.apply(
            {"params": base_params},
            x,
            training=False,
        )["logits"]

    counts = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    steps = math.ceil(len(train_ids) / batch_size)

    with closing(
        dataset.batches(
            train_ids,
            batch_size,
            config,
            epoch=1,
            training=True,
            protocol=protocol,
        )
    ) as batches:
        for index in range(steps):
            batch, _ = next(batches)
            mask = batch["mask"].astype(bool)
            y = batch["y"][mask]
            if len(y) == 0:
                continue

            # Clean + strong training augmentation.  The graph is training-only.
            for key in ("x", "xa"):
                logits = np.asarray(
                    jax.block_until_ready(
                        infer(jax.device_put(batch[key]))
                    )
                )[mask]
                pred = np.argpartition(
                    logits,
                    -topk,
                    axis=1,
                )[:, -topk:]

                for rank in range(topk):
                    rival = pred[:, rank]
                    keep = rival != y
                    np.add.at(
                        counts,
                        (y[keep], rival[keep]),
                        1,
                    )

    table = np.zeros((NUM_CLASSES, RIVALS_PER_TOP), np.int32)
    for c in range(NUM_CLASSES):
        row = counts[c].copy()
        row[c] = -1
        order = np.argsort(row)[::-1]
        chosen = [int(v) for v in order if v != c][:RIVALS_PER_TOP]
        if len(chosen) != RIVALS_PER_TOP:
            raise RuntimeError(f"Could not derive rivals for class {c}")
        table[c] = chosen

    np.save(npy, table)
    meta = {
        "source": "training-only clean+augmented FMSE predictions",
        "protocol": protocol,
        "train_samples": len(train_ids),
        "topk_per_view": topk,
        "rivals_per_class": RIVALS_PER_TOP,
        "table": table.tolist(),
        "counts_top": [
            [
                int(k),
                int(counts[c, k]),
            ]
            for c in range(NUM_CLASSES)
            for k in table[c]
        ],
    }
    atomic_json(meta_path, meta)
    return jax.device_put(table), meta


def create_state(config, steps_per_epoch, specialist, rival_table):
    total = config["epochs"] * steps_per_epoch
    warm = max(1, int(total * config["rce_warmup_fraction"]))
    warm = min(warm, max(total - 1, 1))
    schedule = optax.warmup_cosine_decay_schedule(
        0.0,
        config["rce_learning_rate"],
        warm,
        max(total, warm + 1),
        end_value=config["rce_min_learning_rate"],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config["rce_grad_clip"]),
        optax.adamw(
            schedule,
            weight_decay=config["rce_weight_decay"],
        ),
    )

    key, init = jax.random.split(jax.random.PRNGKey(config["seed"]))
    variables = specialist.init(
        {"params": init, "dropout": init},
        jnp.zeros((1, FRAMES, FEATURES), jnp.float32),
        jnp.zeros((1, NUM_CLASSES), jnp.float32),
        rival_table,
        training=False,
    )
    params = variables["params"]
    state = RCEState.create(
        apply_fn=specialist.apply,
        params=params,
        tx=optimizer,
        ema_params=params,
    )
    return (
        state,
        key,
        schedule,
        int(math.ceil(warm / steps_per_epoch)),
    )


def build_steps(
    *,
    base_model,
    base_params,
    specialist,
    rival_table,
    config,
):
    @jax.jit
    def base_forward(x):
        return base_model.apply(
            {"params": base_params},
            x,
            training=False,
        )["logits"]

    def one_view(params, key, x, y, base_logits):
        out = specialist.apply(
            {"params": params},
            x,
            base_logits,
            rival_table,
            training=True,
            force_class=y,
            rngs={"dropout": key},
        )
        logits = out["logits"]
        main = ce(logits, y, config["label_smoothing"])

        rows = jnp.arange(y.shape[0])
        true_logit = logits[rows, y]

        candidate_mask = out["candidate_mask"]
        not_true = 1.0 - jax.nn.one_hot(y, NUM_CLASSES)
        rival_logits = jnp.where(
            (candidate_mask * not_true) > 0,
            logits,
            -1e9,
        )
        hardest = jnp.max(rival_logits, axis=-1)
        rival_loss = jax.nn.softplus(hardest - true_logit)

        base_logp = jax.nn.log_softmax(jax.lax.stop_gradient(base_logits))
        base_p = jnp.exp(base_logp)
        final_logp = jax.nn.log_softmax(logits)
        protect_kl = jnp.sum(
            base_p * (base_logp - final_logp),
            axis=-1,
        )

        top2 = jax.lax.top_k(base_logits, 2)[0]
        margin = top2[:, 0] - top2[:, 1]
        protect_mask = (
            (jnp.argmax(base_logits, axis=-1) == y)
            & (margin >= config["protect_margin"])
        ).astype(jnp.float32)
        protect = protect_kl * protect_mask

        natural_cov = out["natural_candidate_mask"][rows, y]

        return out, main, rival_loss, protect, natural_cov

    def per_sample(params, key, batch):
        k1, k2 = jax.random.split(key)

        base_clean = base_forward(batch["x"])
        base_aug = base_forward(batch["xa"])

        out, main, rival, protect, cov = one_view(
            params, k1, batch["x"], batch["y"], base_clean
        )
        aug, main_a, rival_a, protect_a, cov_a = one_view(
            params, k2, batch["xa"], batch["y"], base_aug
        )

        main = 0.5 * (main + main_a)
        rival = 0.5 * (rival + rival_a)
        protect = 0.5 * (protect + protect_a)
        coverage = 0.5 * (cov + cov_a)

        logp = jax.nn.log_softmax(out["logits"])
        logq = jax.nn.log_softmax(aug["logits"])
        consistency = 0.5 * jnp.sum(
            (jnp.exp(logp) - jnp.exp(logq)) * (logp - logq),
            axis=-1,
        )

        gate = 0.5 * (out["gate"] + aug["gate"])

        loss = (
            main
            + config["rival_weight"] * rival
            + config["protect_weight"] * protect
            + config["consistency_weight"] * consistency
            + config["gate_weight"] * gate
        )

        y = batch["y"]
        pred = jnp.argmax(out["logits"], axis=-1)
        base_pred = jnp.argmax(base_clean, axis=-1)
        acc = (pred == y).astype(jnp.float32)
        base_acc = (base_pred == y).astype(jnp.float32)
        fixed = ((base_pred != y) & (pred == y)).astype(jnp.float32)
        broken = ((base_pred == y) & (pred != y)).astype(jnp.float32)

        return jnp.stack(
            [
                loss,
                main,
                rival,
                protect,
                consistency,
                acc,
                base_acc,
                fixed,
                broken,
                gate,
                coverage,
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
        denom = jnp.maximum(jnp.sum(batch["mask"]), 1.0)
        key, step_key = jax.random.split(key)
        drop_keys = jax.random.split(step_key, k)
        zero = jax.tree.map(jnp.zeros_like, state.params)

        def accumulate(carry, inputs):
            gradient_sum, metric_sum = carry
            micro, drop = inputs

            def loss_fn(params):
                metrics = per_sample(params, drop, micro)
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
            (zero, jnp.zeros(11)),
            (micros, drop_keys),
        )

        norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)
        ema = jax.tree.map(
            lambda e, p: (
                config["rce_ema_decay"] * e
                + (1.0 - config["rce_ema_decay"]) * p
            ),
            state.ema_params,
            state.params,
        )
        return (
            state.replace(ema_params=ema),
            key,
            jnp.r_[metrics, denom, norm],
        )

    @jax.jit
    def eval_step(params, batch):
        base_logits = base_forward(batch["x"])
        out = specialist.apply(
            {"params": params},
            batch["x"],
            base_logits,
            rival_table,
            training=False,
        )
        logits = out["logits"]
        y = batch["y"]
        mask = batch["mask"]
        pred = jnp.argmax(logits, axis=-1)
        base_pred = jnp.argmax(base_logits, axis=-1)
        top5 = jax.lax.top_k(logits, 5)[1]
        rows = jnp.arange(y.shape[0])

        columns = [
            ce(logits, y, 0.0),
            (pred == y).astype(jnp.float32),
            (base_pred == y).astype(jnp.float32),
            jnp.any(top5 == y[:, None], axis=-1).astype(jnp.float32),
            ((base_pred != y) & (pred == y)).astype(jnp.float32),
            ((base_pred == y) & (pred != y)).astype(jnp.float32),
            out["gate"],
            out["natural_candidate_mask"][rows, y],
            jnp.ones_like(mask),
        ]
        return jnp.sum(
            jnp.stack(columns, axis=-1) * mask[:, None],
            axis=0,
        )

    return train_step, eval_step


def save_last(path, state, key, metadata):
    payload = {
        "state": serialization.to_state_dict(jax.device_get(state)),
        "key": np.asarray(key),
        "metadata_json": json.dumps(metadata, allow_nan=False),
    }
    atomic_bytes(path, serialization.msgpack_serialize(payload))


def restore_last(path, template, digest):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    meta = json.loads(payload["metadata_json"])
    if meta["config_hash"] != digest:
        raise ValueError(
            "Resume config/base/rival mismatch. Keep config or use new OUT_DIR."
        )
    state = serialization.from_state_dict(template, payload["state"])
    return (
        jax.device_put(state),
        jax.device_put(payload["key"]),
        meta,
    )


def run(
    *,
    config,
    protocol,
    cache,
    outdir,
    base_checkpoint,
    allow_cpu=False,
):
    config = validate_config(config)
    if protocol not in ("xsub", "xset"):
        raise ValueError("protocol must be xsub or xset")
    if not allow_cpu and (
        jax.default_backend() != "gpu"
        or len(jax.local_devices()) != 1
    ):
        raise RuntimeError(
            f"Expected one isolated GPU; got {jax.devices()}"
        )

    out = Path(outdir) / protocol
    out.mkdir(parents=True, exist_ok=True)
    report = Reporter(out / "status.json")

    dataset = Dataset(cache)
    train_ids = list(dataset.splits[f"{protocol}_train"])
    val_ids = list(dataset.splits[f"{protocol}_val"])

    if config["max_train_samples"]:
        train_ids = train_ids[: config["max_train_samples"]]
    if config["max_val_samples"]:
        val_ids = val_ids[: config["max_val_samples"]]

    report(
        phase="Load protected FMSE",
        current=0,
        total=1,
        protocol=protocol,
        epoch=0,
        best=None,
        best_epoch=0,
    )

    base_model, base_params, base_payload = load_base_checkpoint(
        base_checkpoint
    )
    base_sha = _sha256(base_checkpoint)

    # Sanity: protocol must match the frozen checkpoint.
    ckpt_protocol = base_payload.get("protocol")
    if ckpt_protocol is not None and ckpt_protocol != protocol:
        raise ValueError(
            f"Checkpoint protocol {ckpt_protocol} != worker {protocol}"
        )

    report(
        phase="Build training-only rivals",
        current=0,
        total=1,
    )
    rival_table, rival_meta = build_training_rivals(
        dataset=dataset,
        train_ids=train_ids,
        base_model=base_model,
        base_params=base_params,
        config=config,
        protocol=protocol,
        out=out,
    )

    specialist = make_specialist(config)
    param_count = specialist_parameter_count(
        specialist,
        rival_table,
    )

    batch_size = (
        config["micro_batch"]
        * config["accumulation_steps"]
    )
    steps = math.ceil(len(train_ids) / batch_size)

    state, key, schedule, warmup_epochs = create_state(
        config,
        steps,
        specialist,
        rival_table,
    )

    signature = {
        "model": MODEL_NAME,
        "identity": MODEL_IDENTITY,
        "variant": config["rce_variant"],
        "config": config,
        "protocol": protocol,
        "cache": dataset.meta["signature"],
        "base_checkpoint_sha256": base_sha,
        "specialist_parameters": param_count,
        "rival_table": np.asarray(rival_table).tolist(),
        "version": VERSION,
    }
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()

    previous = read_json(out / "run_config.json")
    if previous is not None and previous != signature:
        raise ValueError(
            "OUT_DIR contains another RCE config/base/rival graph."
        )
    atomic_json(out / "run_config.json", signature)

    # Exact-baseline preflight: zero-init correction must reproduce FMSE.
    dummy = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    base_dummy = base_model.apply(
        {"params": base_params},
        dummy,
        training=False,
    )["logits"]
    specialist_dummy = specialist.apply(
        {"params": state.params},
        dummy,
        base_dummy,
        rival_table,
        training=False,
    )
    baseline_error = float(
        jnp.max(
            jnp.abs(
                specialist_dummy["logits"] - base_dummy
            )
        )
    )
    if baseline_error > 1e-7:
        raise RuntimeError(
            f"RCE zero-init changed base prediction: {baseline_error}"
        )

    metadata = {
        "config_hash": digest,
        "epoch": 0,
        "best": -1.0,
        "best_epoch": 0,
        "bad_epochs": 0,
        "history": [],
        "stopped_early": False,
        "model": MODEL_NAME,
        "model_identity": MODEL_IDENTITY,
        "variant": config["rce_variant"],
        "specialist_parameters": param_count,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_sha,
        "base_checkpoint_val": float(base_payload.get("val_accuracy", -1)),
        "zero_init_baseline_max_abs_error": baseline_error,
        "rival_graph": rival_meta,
        "warmup_epochs": warmup_epochs,
        "best_checkpoint": None,
    }

    last = out / "last.msgpack"
    if last.is_file():
        state, key, metadata = restore_last(
            last,
            state,
            digest,
        )

    train_step, eval_step = build_steps(
        base_model=base_model,
        base_params=base_params,
        specialist=specialist,
        rival_table=rival_table,
        config=config,
    )

    compiled_train = None
    compiled_eval = None

    for epoch in range(
        metadata["epoch"] + 1,
        config["epochs"] + 1,
    ):
        report(
            phase="Train RCE",
            epoch=epoch,
            current=0,
            total=steps,
            best=metadata["best"] if metadata["best_epoch"] else None,
            best_epoch=metadata["best_epoch"],
        )
        train_sum = np.zeros(12, np.float64)

        with closing(
            dataset.batches(
                train_ids,
                batch_size,
                config,
                epoch=epoch,
                training=True,
                protocol=protocol,
            )
        ) as batches:
            for index in range(steps):
                batch, _ = next(batches)
                device_batch = jax.device_put(batch)

                if compiled_train is None:
                    compiled_train = train_step.lower(
                        state,
                        key,
                        device_batch,
                    ).compile()

                state, key, metrics = jax.block_until_ready(
                    compiled_train(
                        state,
                        key,
                        device_batch,
                    )
                )
                values = np.asarray(metrics)
                if not np.isfinite(values).all():
                    raise FloatingPointError(
                        f"Nonfinite training metrics E{epoch} B{index+1}"
                    )
                train_sum += values[:12]

                if (
                    index % config["progress_every"] == 0
                    or index + 1 == steps
                ):
                    denom = max(train_sum[11], 1.0)
                    report(
                        phase="Train RCE",
                        epoch=epoch,
                        current=index + 1,
                        total=steps,
                        loss=float(train_sum[0] / denom),
                        train_acc=float(train_sum[5] / denom),
                        base_acc=float(train_sum[6] / denom),
                        fixed=float(train_sum[7] / denom),
                        broken=float(train_sum[8] / denom),
                        gate=float(train_sum[9] / denom),
                        coverage=float(train_sum[10] / denom),
                        lr=float(schedule(state.step)),
                    )

        val_steps = math.ceil(
            len(val_ids) / config["eval_batch"]
        )
        eval_sum = np.zeros(9, np.float64)

        report(
            phase="Validate RCE EMA",
            epoch=epoch,
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
                batch, _ = next(batches)
                device_batch = jax.device_put(batch)

                if compiled_eval is None:
                    compiled_eval = eval_step.lower(
                        state.ema_params,
                        device_batch,
                    ).compile()

                vals = np.asarray(
                    jax.block_until_ready(
                        compiled_eval(
                            state.ema_params,
                            device_batch,
                        )
                    )
                )
                if not np.isfinite(vals).all():
                    raise FloatingPointError(
                        f"Nonfinite validation E{epoch}"
                    )
                eval_sum += vals

                if (
                    index % config["progress_every"] == 0
                    or index + 1 == val_steps
                ):
                    report(
                        phase="Validate RCE EMA",
                        epoch=epoch,
                        current=index + 1,
                        total=val_steps,
                        val_acc=float(eval_sum[1] / max(eval_sum[8], 1)),
                        base_acc=float(eval_sum[2] / max(eval_sum[8], 1)),
                    )

        n = eval_sum[8]
        val_acc = float(eval_sum[1] / n)
        base_acc = float(eval_sum[2] / n)
        fixed = float(eval_sum[4] / n)
        broken = float(eval_sum[5] / n)

        improved = val_acc > metadata["best"] + config["min_delta"]
        if improved:
            metadata["best"] = val_acc
            metadata["best_epoch"] = epoch
            metadata["bad_epochs"] = 0
            metadata["best_checkpoint"] = f"best_epoch_{epoch:04d}.msgpack"

            best_payload = {
                "model": MODEL_NAME,
                "model_identity": MODEL_IDENTITY,
                "variant": config["rce_variant"],
                "protocol": protocol,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "base_val_accuracy": base_acc,
                "specialist_ema_params": jax.device_get(state.ema_params),
                "specialist_parameters": param_count,
                "base_checkpoint": str(base_checkpoint),
                "base_checkpoint_sha256": base_sha,
                "rival_table": np.asarray(rival_table),
                "config": config,
                "preprocessing_version": PREPROCESSING_VERSION,
                "pipeline_version": VERSION,
                "cache_signature": dataset.meta["signature"],
            }
            atomic_bytes(
                out / metadata["best_checkpoint"],
                serialization.to_bytes(best_payload),
            )
            atomic_bytes(
                out / "best.msgpack",
                serialization.to_bytes(best_payload),
            )
        else:
            if epoch > warmup_epochs:
                metadata["bad_epochs"] += 1

        row = {
            "epoch": epoch,
            "train_loss": float(train_sum[0] / max(train_sum[11], 1)),
            "train_acc": float(train_sum[5] / max(train_sum[11], 1)),
            "train_base_acc": float(train_sum[6] / max(train_sum[11], 1)),
            "val_acc": val_acc,
            "base_val_acc": base_acc,
            "val_top5": float(eval_sum[3] / n),
            "fixed_rate": fixed,
            "broken_rate": broken,
            "net_fixed_minus_broken": fixed - broken,
            "gate_mean": float(eval_sum[6] / n),
            "candidate_coverage": float(eval_sum[7] / n),
            "best_val_accuracy": metadata["best"],
            "best_epoch": metadata["best_epoch"],
        }
        metadata["history"].append(row)
        metadata["epoch"] = epoch
        metadata["stopped_early"] = (
            metadata["bad_epochs"] >= config["patience"]
        )

        atomic_json(out / "history.json", metadata["history"])
        save_last(last, state, key, metadata)

        report(
            phase="Epoch complete",
            epoch=epoch,
            current=1,
            total=1,
            val_acc=val_acc,
            base_acc=base_acc,
            fixed=fixed,
            broken=broken,
            net_fixed_minus_broken=fixed - broken,
            gate=float(eval_sum[6] / n),
            coverage=float(eval_sum[7] / n),
            best=metadata["best"],
            best_epoch=metadata["best_epoch"],
            completed_epoch=epoch,
        )

        if metadata["stopped_early"]:
            break

    result = {
        "model": MODEL_NAME,
        "model_identity": MODEL_IDENTITY,
        "variant": config["rce_variant"],
        "protocol": protocol,
        "best_val_accuracy": metadata["best"],
        "best_epoch": metadata["best_epoch"],
        "last_epoch": metadata["epoch"],
        "specialist_parameters": param_count,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_sha,
        "base_checkpoint_val": metadata["base_checkpoint_val"],
        "checkpoint": str(out / "best.msgpack"),
        "stopped_early": metadata["stopped_early"],
        "zero_init_baseline_max_abs_error": metadata[
            "zero_init_baseline_max_abs_error"
        ],
    }
    atomic_json(out / "result.json", result)
    atomic_json(Path(outdir) / f"result_{protocol}.json", result)
    report(
        phase="Done",
        current=1,
        total=1,
        done=True,
        epoch=metadata["epoch"],
        best=metadata["best"],
        best_epoch=metadata["best_epoch"],
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()

    config = json.loads(Path(a.config).read_text())
    result = run(
        config=config,
        protocol=a.protocol,
        cache=a.cache,
        outdir=a.outdir,
        base_checkpoint=a.base_checkpoint,
        allow_cpu=a.allow_cpu,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

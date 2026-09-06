"""One protocol on one explicitly isolated GPU. Plain jit; no pmap shims."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import psutil
from flax import serialization
from flax.training import train_state

from .data import Dataset
from .io_utils import Reporter, atomic_bytes, atomic_json, read_json
from .model import M4LocalGlobalHandM4G4T32, EXPECTED_PARAMS
from .preprocessing import FRAMES, FEATURES, HAND_FRAMES, HAND_FEATURES


class State(train_state.TrainState):
    ema_params: object


def ce(logits, labels, smoothing):
    target = jax.nn.one_hot(labels, 120)
    target = target * (1 - smoothing) + smoothing / 120
    return -jnp.sum(target * jax.nn.log_softmax(logits), axis=-1)


def build_steps(model, config):
    def per_sample(params, key, batch):
        k1, k2 = jax.random.split(key)
        out = model.apply({"params": params}, batch["x"], batch["h"], training=True, rngs={"dropout": k1})
        aug = model.apply({"params": params}, batch["xa"], batch["ha"], training=True, rngs={"dropout": k2})
        y, smooth = batch["y"], config["label_smoothing"]
        main = (ce(out["logits"], y, smooth) + ce(aug["logits"], y, smooth)) / 2
        aux = (ce(out["stream_logits"], y[:, None], smooth).mean(1) +
               ce(aug["stream_logits"], y[:, None], smooth).mean(1)) / 2
        hand = (ce(out["hand_logits"], y, smooth) + ce(aug["hand_logits"], y, smooth)) / 2
        temperature = config["consistency_temperature"]
        logp = jax.nn.log_softmax(out["logits"] / temperature)
        logq = jax.nn.log_softmax(aug["logits"] / temperature)
        kl = 0.5 * temperature ** 2 * jnp.sum((jnp.exp(logp) - jnp.exp(logq)) * (logp - logq), -1)
        loss = main + config["stream_aux_weight"] * aux + config["hand_aux_weight"] * hand + config["consistency_weight"] * kl
        acc = (out["logits"].argmax(-1) == y).astype(jnp.float32)
        return jnp.stack([loss, main, aux, hand, kl, acc], axis=-1)

    @jax.jit
    def train_step(state, key, batch):
        # Sum microbatch gradients, then update AdamW and EMA ONCE per batch.
        k = config["accumulation_steps"]
        micros = jax.tree.map(lambda x: x.reshape(k, config["micro_batch"], *x.shape[1:]), batch)
        denom = jnp.maximum(jnp.sum(batch["mask"]), 1)
        key, step_key = jax.random.split(key)
        drop_keys = jax.random.split(step_key, k)
        zero = jax.tree.map(jnp.zeros_like, state.params)

        def accumulate(carry, inputs):
            gradient_sum, metric_sum = carry
            micro, drop = inputs

            def loss_fn(params):
                metrics = per_sample(params, drop, micro)
                totals = jnp.sum(metrics * micro["mask"][:, None], axis=0)
                return totals[0] / denom, totals

            (_, totals), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            return (jax.tree.map(lambda a, b: a + b, gradient_sum, gradients), metric_sum + totals), None

        (grads, metrics), _ = jax.lax.scan(accumulate, (zero, jnp.zeros(6)), (micros, drop_keys))
        norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)
        ema = jax.tree.map(lambda e, p: config["ema_decay"] * e + (1 - config["ema_decay"]) * p,
                           state.ema_params, state.params)
        return state.replace(ema_params=ema), key, jnp.r_[metrics, denom, norm]

    @jax.jit
    def eval_step(params, batch):
        out = model.apply({"params": params}, batch["x"], batch["h"], training=False)
        y, mask = batch["y"], batch["mask"]
        logits = out["logits"]
        top5 = jax.lax.top_k(logits, 5)[1]
        columns = [ce(logits, y, 0), (logits.argmax(-1) == y).astype(jnp.float32),
                   (out["main_logits"].argmax(-1) == y).astype(jnp.float32),
                   (out["hand_logits"].argmax(-1) == y).astype(jnp.float32),
                   jnp.any(top5 == y[:, None], axis=-1).astype(jnp.float32), jnp.ones_like(mask)]
        return jnp.sum(jnp.stack(columns, -1) * mask[:, None], axis=0)

    return train_step, eval_step


def create_state(config, steps_per_epoch):
    total = config["epochs"] * steps_per_epoch
    warm = max(1, int(math.ceil(total * config["warmup_fraction"])))
    # A one-epoch smoke run still has a well-defined schedule.
    warm = min(warm, max(total - 1, 1))
    schedule = optax.warmup_cosine_decay_schedule(0, config["learning_rate"], warm,
                                                max(total, warm + 1), end_value=config["min_learning_rate"])
    optimizer = optax.chain(optax.clip_by_global_norm(config["grad_clip"]),
                           optax.adamw(schedule, weight_decay=config["weight_decay"]))
    model = M4LocalGlobalHandM4G4T32(dropout=config["dropout"], hand_residual_scale=config["hand_residual_scale"])
    key, init = jax.random.split(jax.random.PRNGKey(config["seed"]))
    params = model.init({"params": init}, jnp.zeros((1, FRAMES, FEATURES)),
                        jnp.zeros((1, HAND_FRAMES, HAND_FEATURES)), training=False)["params"]
    count = sum(x.size for x in jax.tree.leaves(params))
    if count != EXPECTED_PARAMS:
        raise RuntimeError(f"Model parameter mismatch: {count} != {EXPECTED_PARAMS}")
    state = State.create(apply_fn=model.apply, params=params, tx=optimizer, ema_params=params)
    return model, state, key, schedule, int(math.ceil(warm / steps_per_epoch))


def stopping_update(best, bad_epochs, val_acc, epoch, warmup_epochs, min_delta):
    improved = val_acc > best + min_delta
    best = val_acc if improved else best
    bad_epochs = 0 if improved or epoch <= warmup_epochs else bad_epochs + 1
    return best, bad_epochs, improved


def save_checkpoint(path, state, key, metadata):
    payload = {"state": serialization.to_state_dict(jax.device_get(state)),
               "key": np.asarray(key), "metadata_json": json.dumps(metadata, allow_nan=False)}
    atomic_bytes(path, serialization.msgpack_serialize(payload))


def restore_checkpoint(path, template, expected_hash):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    meta = json.loads(payload["metadata_json"])
    if meta["config_hash"] != expected_hash:
        raise ValueError("Resume config/data mismatch. Keep the original config or choose a new OUT_DIR.")
    state = serialization.from_state_dict(template, payload["state"])
    return jax.device_put(state), jax.device_put(payload["key"]), meta


def memory_gib():
    # Diagnostics must not stop training on hosts with restricted /proc access.
    try:
        return psutil.Process().memory_info().rss / 2**30
    except (psutil.Error, OSError):
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def run(config, protocol, cache, outdir, allow_cpu=False):
    out = Path(outdir) / protocol
    out.mkdir(parents=True, exist_ok=True)
    report = Reporter(out / "status.json")
    report(phase="JAX startup", current=0, total=1, protocol=protocol, epoch=0,
           best=None, best_epoch=0, completed_epoch=0)
    if not allow_cpu and (jax.default_backend() != "gpu" or len(jax.local_devices()) != 1):
        raise RuntimeError(f"Expected one isolated GPU; backend={jax.default_backend()}, devices={jax.local_devices()}")
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
    signature = {"config": config, "protocol": protocol, "cache": dataset.meta["signature"],
                 "parameters": EXPECTED_PARAMS}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    previous_signature = read_json(out / "run_config.json")
    if previous_signature is not None and previous_signature != signature:
        raise ValueError("OUT_DIR contains another run's config/data. Choose a new OUT_DIR.")
    atomic_json(out / "run_config.json", signature)
    report(phase="Initialize M4/G4", train_samples=len(train_ids), val_samples=len(val_ids))
    model, state, key, schedule, warmup_epochs = create_state(config, steps)
    metadata = {"config_hash": digest, "epoch": 0, "best": -1.0, "best_epoch": 0,
                "bad_epochs": 0, "history": [], "warmup_epochs": warmup_epochs}
    last = out / "last.msgpack"
    if last.exists():
        state, key, metadata = restore_checkpoint(last, state, digest)
        if metadata["best_epoch"] > 0 and not (out / "best_ema.msgpack").is_file():
            raise FileNotFoundError("Resume requires the best_ema.msgpack beside last.msgpack")
        # Repair a crash between the atomic checkpoint and history-file writes.
        atomic_json(out / "history.json", metadata["history"])
    report(best=metadata["best"] if metadata["best_epoch"] else None,
           best_epoch=metadata["best_epoch"], completed_epoch=metadata["epoch"])
    if metadata["epoch"] >= config["epochs"] or metadata["bad_epochs"] >= config["patience"]:
        result = {"protocol": protocol, "best_accuracy": metadata["best"], "best_epoch": metadata["best_epoch"],
                  "epochs_run": metadata["epoch"], "resumed_completed": True, "config_hash": digest,
                  "parameters": EXPECTED_PARAMS, "backend": jax.default_backend(),
                  "devices": [str(d) for d in jax.local_devices()],
                  "checkpoint": str(out / "best_ema.msgpack"),
                  "stopped_early": metadata["bad_epochs"] >= config["patience"]}
        atomic_json(out / "result.json", result)
        report(phase="Done", current=1, total=1, done=True, best=metadata["best"],
               best_epoch=metadata["best_epoch"], epoch=metadata["epoch"], completed_epoch=metadata["epoch"])
        return result
    train_step, eval_step = build_steps(model, config)
    epoch_start = metadata["epoch"] + 1
    for epoch in range(epoch_start, config["epochs"] + 1):
        epoch_t0 = time.perf_counter()
        report(phase="Train", epoch=epoch, current=0, total=steps,
               best=metadata["best"] if metadata["best_epoch"] else None, best_epoch=metadata["best_epoch"])
        train_sum = np.zeros(7, np.float64)
        timing = {"prepare_service_s": 0.0, "data_wait_s": 0.0, "h2d_s": 0.0,
                  "gpu_train_s": 0.0, "gpu_eval_s": 0.0, "compile_train_s": 0.0, "compile_eval_s": 0.0}
        batches = iter(dataset.batches(train_ids, batch_size, config, epoch, training=True))
        for index in range(steps):
            start = time.perf_counter()
            batch, prep_s = next(batches)
            timing["data_wait_s"] += time.perf_counter() - start
            timing["prepare_service_s"] += prep_s
            start = time.perf_counter()
            device_batch = jax.device_put(batch)
            jax.block_until_ready(device_batch)
            timing["h2d_s"] += time.perf_counter() - start
            if epoch == epoch_start and index == 0:
                report(phase="Compile train", current=0, total=steps)
                start = time.perf_counter()
                compiled_train = train_step.lower(state, key, device_batch).compile()
                timing["compile_train_s"] = time.perf_counter() - start
            start = time.perf_counter()
            state, key, metrics = compiled_train(state, key, device_batch)
            values = np.asarray(jax.block_until_ready(metrics))
            timing["gpu_train_s"] += time.perf_counter() - start
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Nonfinite training values at epoch {epoch}, batch {index + 1}")
            train_sum += values[:7]
            if index % config["progress_every"] == 0 or index + 1 == steps:
                report(phase="Train", current=index+1, total=steps, loss=float(train_sum[0] / train_sum[6]),
                       train_acc=float(train_sum[5] / train_sum[6]), rss_gib=memory_gib(),
                       lr=float(schedule(state.step)), wait_s=timing["data_wait_s"], gpu_s=timing["gpu_train_s"])
        if int(train_sum[6]) != len(train_ids):
            raise RuntimeError("Training sample accounting mismatch")
        eval_sum = np.zeros(6, np.float64)
        val_steps = math.ceil(len(val_ids) / config["eval_batch"])
        report(phase="Validate EMA", current=0, total=val_steps)
        batches = iter(dataset.batches(val_ids, config["eval_batch"], config))
        for index in range(val_steps):
            start = time.perf_counter()
            batch, prep_s = next(batches)
            timing["data_wait_s"] += time.perf_counter() - start
            timing["prepare_service_s"] += prep_s
            start = time.perf_counter()
            device_batch = jax.device_put(batch)
            jax.block_until_ready(device_batch)
            timing["h2d_s"] += time.perf_counter() - start
            if epoch == epoch_start and index == 0:
                report(phase="Compile validation", current=0, total=val_steps)
                start = time.perf_counter()
                compiled_eval = eval_step.lower(state.ema_params, device_batch).compile()
                timing["compile_eval_s"] = time.perf_counter() - start
            start = time.perf_counter()
            vals = np.asarray(jax.block_until_ready(compiled_eval(state.ema_params, device_batch)))
            timing["gpu_eval_s"] += time.perf_counter() - start
            if not np.isfinite(vals).all():
                raise FloatingPointError("Nonfinite validation values")
            eval_sum += vals
            if index % config["progress_every"] == 0 or index + 1 == val_steps:
                report(phase="Validate EMA", current=index+1, total=val_steps, val_acc=float(eval_sum[1] / eval_sum[5]))
        if int(eval_sum[5]) != len(val_ids):
            raise RuntimeError("Validation mask/count mismatch")
        val = float(eval_sum[1] / eval_sum[5])
        best, bad, improved = stopping_update(metadata["best"], metadata["bad_epochs"], val, epoch,
                                             warmup_epochs, config["min_delta"])
        row = {"epoch": epoch, "train_loss": float(train_sum[0]/train_sum[6]),
               "train_acc": float(train_sum[5]/train_sum[6]), "val_loss": float(eval_sum[0]/eval_sum[5]),
               "val_acc": val, "val_main_acc": float(eval_sum[2]/eval_sum[5]),
               "val_hand_acc": float(eval_sum[3]/eval_sum[5]), "val_top5": float(eval_sum[4]/eval_sum[5]),
               "bad_epochs": bad, "epoch_s": time.perf_counter() - epoch_t0, **timing}
        metadata.update(epoch=epoch, best=best, bad_epochs=bad)
        metadata["history"].append(row)
        report(phase="Save checkpoint", current=1, total=1)
        if improved:
            metadata["best_epoch"] = epoch
            atomic_bytes(out / "best_ema.msgpack", serialization.to_bytes(jax.device_get(state.ema_params)))
        row.update(best_val_accuracy=best, best_epoch=metadata["best_epoch"])
        save_checkpoint(last, state, key, metadata)
        atomic_json(out / "history.json", metadata["history"])
        report(phase="Epoch complete", best=best, best_epoch=metadata["best_epoch"],
               completed_epoch=epoch, val_acc=val, current=1, total=1, bad_epochs=bad,
               epoch_s=row["epoch_s"], wait_s=timing["data_wait_s"], gpu_s=timing["gpu_train_s"])
        if bad >= config["patience"]:
            break
    result = {"protocol": protocol, "best_accuracy": metadata["best"], "best_epoch": metadata["best_epoch"],
              "epochs_run": metadata["epoch"], "parameters": EXPECTED_PARAMS,
              "backend": jax.default_backend(), "devices": [str(d) for d in jax.local_devices()],
              "config_hash": digest, "checkpoint": str(out / "best_ema.msgpack"),
              "stopped_early": metadata["bad_epochs"] >= config["patience"]}
    atomic_json(out / "result.json", result)
    report(phase="Done", current=1, total=1, done=True, best=metadata["best"],
           best_epoch=metadata["best_epoch"], epoch=metadata["epoch"], completed_epoch=metadata["epoch"])
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--protocol", required=True, choices=("xsub", "xset"))
    p.add_argument("--cache", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--allow-cpu", action="store_true", help="Synthetic validation only")
    a = p.parse_args()
    run(json.loads(Path(a.config).read_text()), a.protocol, a.cache, a.outdir, a.allow_cpu)

"""One protocol on one explicitly isolated GPU. Plain jit; no pmap shims."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import time
from contextlib import closing
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
from ..model import NestSARSMAllT16
from ..preprocessing_corrected import FRAMES, FEATURES, VERSION as PREPROCESSING_VERSION
from . import VERSION

EXPECTED_PARAMS = 1_826_556


def make_model(config):
    return NestSARSMAllT16(**{k: config[k] for k in (
        "spatial_dim", "model_dim", "dropout", "controller_dim", "fast_rank",
        "head_rank", "sm_residual_scale", "head_residual_scale")})


class State(train_state.TrainState):
    ema_params: object


def ce(logits, labels, smoothing):
    target = jax.nn.one_hot(labels, 120)
    target = target * (1 - smoothing) + smoothing / 120
    return -jnp.sum(target * jax.nn.log_softmax(logits), axis=-1)


def build_steps(model, config):
    def per_sample(params, key, batch):
        k1, k2 = jax.random.split(key)
        out = model.apply({"params": params}, batch["x"], training=True, rngs={"dropout": k1})
        aug = model.apply({"params": params}, batch["xa"], training=True, rngs={"dropout": k2})
        y, smooth = batch["y"], config["label_smoothing"]
        main = (ce(out["logits"], y, smooth) + ce(aug["logits"], y, smooth)) / 2
        aux = (ce(out["stream_logits"], y[:, None], smooth).mean(1) +
               ce(aug["stream_logits"], y[:, None], smooth).mean(1)) / 2
        temperature = config["consistency_temperature"]
        logp = jax.nn.log_softmax(out["logits"] / temperature)
        logq = jax.nn.log_softmax(aug["logits"] / temperature)
        kl = 0.5 * temperature ** 2 * jnp.sum((jnp.exp(logp) - jnp.exp(logq)) * (logp - logq), -1)
        loss = main + config["stream_aux_weight"] * aux + config["consistency_weight"] * kl
        acc = (out["logits"].argmax(-1) == y).astype(jnp.float32)
        aug_acc = (aug["logits"].argmax(-1) == y).astype(jnp.float32)
        agreement = (out["logits"].argmax(-1) == aug["logits"].argmax(-1)).astype(jnp.float32)
        return jnp.stack([loss, main, aux, kl, acc, aug_acc, agreement,
                          out["sm_eta_mean"], out["sm_alpha_mean"]], axis=-1)

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

        (grads, metrics), _ = jax.lax.scan(accumulate, (zero, jnp.zeros(9)), (micros, drop_keys))
        norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)
        ema = jax.tree.map(lambda e, p: config["ema_decay"] * e + (1 - config["ema_decay"]) * p,
                           state.ema_params, state.params)
        return state.replace(ema_params=ema), key, jnp.r_[metrics, denom, norm]

    @jax.jit
    def eval_step(params, batch):
        out = model.apply({"params": params}, batch["x"], training=False)
        y, mask = batch["y"], batch["mask"]
        logits = out["logits"]
        top5 = jax.lax.top_k(logits, 5)[1]
        columns = [ce(logits, y, 0), (logits.argmax(-1) == y).astype(jnp.float32),
                   (out["main_logits"].argmax(-1) == y).astype(jnp.float32),
                   jnp.any(top5 == y[:, None], axis=-1).astype(jnp.float32),
                   out["sm_eta_mean"], out["sm_alpha_mean"], jnp.ones_like(mask)]
        return jnp.sum(jnp.stack(columns, -1) * mask[:, None], axis=0)

    return train_step, eval_step


def create_state(config, steps_per_epoch):
    total = config["epochs"] * steps_per_epoch
    warm = max(1, int(total * config["warmup_fraction"]))
    # A one-epoch smoke run still has a well-defined schedule.
    warm = min(warm, max(total - 1, 1))
    schedule = optax.warmup_cosine_decay_schedule(0, config["learning_rate"], warm,
                                                max(total, warm + 1), end_value=config["min_learning_rate"])
    optimizer = optax.chain(optax.clip_by_global_norm(config["grad_clip"]),
                           optax.adamw(schedule, weight_decay=config["weight_decay"]))
    model = make_model(config)
    key, init = jax.random.split(jax.random.PRNGKey(config["seed"]))
    params = model.init({"params": init, "dropout": init}, jnp.zeros((1, FRAMES, FEATURES)), training=False)["params"]
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


def memory_snapshot():
    result = {"rss_gib": memory_gib()}
    try:
        full = psutil.Process().memory_full_info()
        for key in ("uss", "pss"):
            if hasattr(full, key):
                result[key + "_gib"] = getattr(full, key) / 2**30
    except (psutil.Error, OSError):
        pass
    try:
        result.update(host_available_gib=psutil.virtual_memory().available / 2**30,
                      host_swap_used_gib=psutil.swap_memory().used / 2**30)
    except (psutil.Error, OSError):
        pass
    return result


def publish_best(out, metadata):
    """Repair/publicize the best file referenced by the committed last state."""
    name = metadata["best_checkpoint"]
    if Path(name).name != name or not name.startswith("best_epoch_"):
        raise ValueError("Invalid best-checkpoint filename")
    payload = (out / name).read_bytes()
    atomic_bytes(out / "best.msgpack", payload)
    atomic_json(out / "best.json", dict(model="NestSAR-SM-ALL-T16-v1",
        epoch=metadata["best_epoch"], val_accuracy=metadata["best"], params=EXPECTED_PARAMS,
        preprocessing_version=PREPROCESSING_VERSION, pipeline_version=VERSION,
        config_hash=metadata["config_hash"]))


def write_result(out, protocol, metadata, digest, resumed=False):
    result = {"model": "NestSAR-SM-ALL-T16-v1", "protocol": protocol,
              "best_val_accuracy": metadata["best"], "best_accuracy": metadata["best"],
              "best_epoch": metadata["best_epoch"], "last_epoch": metadata["epoch"],
              "epochs_run": metadata["epoch"], "params": EXPECTED_PARAMS,
              "backend": jax.default_backend(), "devices": [str(d) for d in jax.local_devices()],
              "config_hash": digest, "checkpoint": str(out / "best.msgpack"),
              "preprocessing_version": PREPROCESSING_VERSION, "pipeline_version": VERSION,
              "stopped_early": metadata["stopped_early"], "resumed_completed": resumed}
    atomic_json(out / "result.json", result)
    atomic_json(out.parent / f"result_{protocol}.json", result)
    return result


def run(config, protocol, cache, outdir, allow_cpu=False):
    from .launch import validate_config
    config = validate_config(config)
    if protocol not in ("xsub", "xset"):
        raise ValueError("Expected xsub or xset")
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
                 "parameters": EXPECTED_PARAMS, "pipeline_version": VERSION}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    previous = read_json(out / "run_config.json")
    if previous is not None and previous != signature:
        raise ValueError("OUT_DIR contains another run's config/data. Choose a new OUT_DIR.")
    atomic_json(out / "run_config.json", signature)
    report(phase="Initialize SM-ALL", train_samples=len(train_ids), val_samples=len(val_ids))
    model, state, key, schedule, warmup_epochs = create_state(config, steps)
    metadata = {"config_hash": digest, "epoch": 0, "best": -1.0, "best_epoch": 0,
                "bad_epochs": 0, "history": [], "warmup_epochs": warmup_epochs,
                "best_checkpoint": None, "stopped_early": False}
    last = out / "last.msgpack"
    if last.exists():
        state, key, metadata = restore_checkpoint(last, state, digest)
        if metadata["best_epoch"]:
            publish_best(out, metadata)
        atomic_json(out / "history.json", metadata["history"])
    report(best=metadata["best"] if metadata["best_epoch"] else None,
           best_epoch=metadata["best_epoch"], completed_epoch=metadata["epoch"])
    if metadata["epoch"] >= config["epochs"] or metadata["stopped_early"]:
        result = write_result(out, protocol, metadata, digest, resumed=True)
        report(phase="Done", current=1, total=1, done=True, epoch=metadata["epoch"])
        return result

    train_step, eval_step = build_steps(model, config)
    compiled_train = compiled_eval = None
    for epoch in range(metadata["epoch"] + 1, config["epochs"] + 1):
        epoch_t0 = time.perf_counter()
        report(phase="Train", epoch=epoch, current=0, total=steps, val_acc=None,
               best=metadata["best"] if metadata["best_epoch"] else None, best_epoch=metadata["best_epoch"])
        train_sum = np.zeros(10, np.float64)
        timing = {"prepare_service_s": 0.0, "data_wait_s": 0.0, "h2d_s": 0.0,
                  "gpu_train_s": 0.0, "gpu_eval_s": 0.0, "compile_train_s": 0.0,
                  "compile_eval_s": 0.0, "warmup_train_s": 0.0, "warmup_eval_s": 0.0}
        with closing(dataset.batches(train_ids, batch_size, config, epoch, True, protocol)) as batches:
            for index in range(steps):
                start = time.perf_counter()
                batch, prep_s = next(batches)
                timing["data_wait_s"] += time.perf_counter() - start
                timing["prepare_service_s"] += prep_s
                start = time.perf_counter()
                device_batch = jax.block_until_ready(jax.device_put(batch))
                timing["h2d_s"] += time.perf_counter() - start
                if compiled_train is None:
                    report(phase="Compile train", current=0, total=steps)
                    start = time.perf_counter()
                    compiled_train = train_step.lower(state, key, device_batch).compile()
                    timing["compile_train_s"] = time.perf_counter() - start
                    start = time.perf_counter()
                    # Pure functional call; discard outputs so warm-up does not
                    # advance optimizer, EMA or RNG state.
                    warm = jax.block_until_ready(compiled_train(state, key, device_batch))
                    del warm
                    timing["warmup_train_s"] = time.perf_counter() - start
                start = time.perf_counter()
                state, key, metrics = jax.block_until_ready(compiled_train(state, key, device_batch))
                timing["gpu_train_s"] += time.perf_counter() - start
                values = np.asarray(metrics)
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"Nonfinite training values at epoch {epoch}, batch {index + 1}")
                train_sum += values[:10]
                if index % config["progress_every"] == 0 or index + 1 == steps:
                    report(phase="Train", current=index+1, total=steps, loss=float(train_sum[0] / train_sum[9]),
                           train_acc=float(train_sum[4] / train_sum[9]), **memory_snapshot(),
                           lr=float(schedule(state.step)), wait_s=timing["data_wait_s"], gpu_s=timing["gpu_train_s"])
        if int(train_sum[9]) != len(train_ids):
            raise RuntimeError("Training sample accounting mismatch")
        eval_sum = np.zeros(7, np.float64)
        val_steps = math.ceil(len(val_ids) / config["eval_batch"])
        report(phase="Validate EMA", current=0, total=val_steps)
        with closing(dataset.batches(val_ids, config["eval_batch"], config, protocol=protocol)) as batches:
            for index in range(val_steps):
                start = time.perf_counter()
                batch, prep_s = next(batches)
                timing["data_wait_s"] += time.perf_counter() - start
                timing["prepare_service_s"] += prep_s
                start = time.perf_counter()
                device_batch = jax.block_until_ready(jax.device_put(batch))
                timing["h2d_s"] += time.perf_counter() - start
                if compiled_eval is None:
                    report(phase="Compile validation", current=0, total=val_steps)
                    start = time.perf_counter()
                    compiled_eval = eval_step.lower(state.ema_params, device_batch).compile()
                    timing["compile_eval_s"] = time.perf_counter() - start
                    start = time.perf_counter()
                    jax.block_until_ready(compiled_eval(state.ema_params, device_batch))
                    timing["warmup_eval_s"] = time.perf_counter() - start
                start = time.perf_counter()
                vals = np.asarray(jax.block_until_ready(compiled_eval(state.ema_params, device_batch)))
                timing["gpu_eval_s"] += time.perf_counter() - start
                if not np.isfinite(vals).all():
                    raise FloatingPointError("Nonfinite validation values")
                eval_sum += vals
                if index % config["progress_every"] == 0 or index + 1 == val_steps:
                    report(phase="Validate EMA", current=index+1, total=val_steps, val_acc=float(eval_sum[1] / eval_sum[6]))
        if int(eval_sum[6]) != len(val_ids):
            raise RuntimeError("Validation mask/count mismatch")
        val = float(eval_sum[1] / eval_sum[6])
        best, bad, improved = stopping_update(metadata["best"], metadata["bad_epochs"], val, epoch,
                                             warmup_epochs, config["min_delta"])
        row = {"epoch": epoch, "train_loss": float(train_sum[0]/train_sum[9]),
               "train_acc": float(train_sum[4]/train_sum[9]), "augmented_train_acc": float(train_sum[5]/train_sum[9]),
               "val_loss": float(eval_sum[0]/eval_sum[6]), "val_acc": val,
               "val_main_acc": float(eval_sum[2]/eval_sum[6]), "val_top5": float(eval_sum[3]/eval_sum[6]),
               "eta": float(eval_sum[4]/eval_sum[6]), "alpha": float(eval_sum[5]/eval_sum[6]),
               "train_samples": len(train_ids), "val_samples": len(val_ids),
               "bad_epochs": bad, "epoch_s": time.perf_counter() - epoch_t0, **timing, **memory_snapshot()}
        previous_best = metadata["best_checkpoint"]
        metadata.update(epoch=epoch, best=best, bad_epochs=bad, stopped_early=bad >= config["patience"])
        report(phase="Save checkpoint", current=1, total=1)
        if improved:
            metadata.update(best_epoch=epoch, best_checkpoint=f"best_epoch_{epoch:04d}.msgpack")
            best_payload = {"model": "NestSAR-SM-ALL-T16-v1", "protocol": protocol, "epoch": epoch,
                            "val_accuracy": val, "ema_params": jax.device_get(state.ema_params),
                            "config": config, "preprocessing_version": PREPROCESSING_VERSION,
                            "pipeline_version": VERSION, "cache_signature": dataset.meta["signature"],
                            "train_samples": len(train_ids), "val_samples": len(val_ids)}
            atomic_bytes(out / metadata["best_checkpoint"], serialization.to_bytes(best_payload))
        row.update(best_val_accuracy=best, best_epoch=metadata["best_epoch"])
        metadata["history"].append(row)
        # Commit optimizer/EMA/key and best-file reference together, then repair
        # public aliases. Resume can recover a crash between either write.
        save_checkpoint(last, state, key, metadata)
        if improved:
            publish_best(out, metadata)
            if previous_best and previous_best != metadata["best_checkpoint"]:
                (out / previous_best).unlink(missing_ok=True)
        atomic_json(out / "history.json", metadata["history"])
        report(phase="Epoch complete", best=best, best_epoch=metadata["best_epoch"],
               completed_epoch=epoch, val_acc=val, current=1, total=1, bad_epochs=bad,
               epoch_s=row["epoch_s"], wait_s=timing["data_wait_s"], gpu_s=timing["gpu_train_s"])
        if metadata["stopped_early"]:
            break
    result = write_result(out, protocol, metadata, digest)
    report(phase="Done", current=1, total=1, done=True, best=metadata["best"],
           best_epoch=metadata["best_epoch"], epoch=metadata["epoch"], completed_epoch=metadata["epoch"])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--protocol", required=True, choices=("xsub", "xset"))
    p.add_argument("--cache", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--allow-cpu", action="store_true", help="Synthetic validation only")
    a = p.parse_args()
    run(json.loads(Path(a.config).read_text()), a.protocol, a.cache, a.outdir, a.allow_cpu)


if __name__ == "__main__":
    main()

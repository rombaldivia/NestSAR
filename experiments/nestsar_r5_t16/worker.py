"""Train NestSAR-R5 on one protocol and one isolated GPU.

Same loop, losses, optimizer, EMA, checkpoint format, resume and early stopping
as the R4-FMSE + LocalGeometry worker, minus the geometry loss (+0.005 pp,
archived as a negative result). Auxiliary heads: the M4 temporal mean and the
hand branch. Adds the fast-memory / interaction diagnostics to the history and,
at the end, a per-class report of the best EMA checkpoint next to the R4 recalls
of the classes R4 got most wrong.
"""
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

from experiments.nestsar_sm_all_t16.streaming import worker as r4_worker
from experiments.nestsar_sm_all_t16.streaming.io_utils import Reporter, atomic_bytes, atomic_json, read_json

from . import MODEL_IDENTITY as BASE_IDENTITY, MODEL_NAME, VERSION
from .mixing import mix_batch
from .config import model_kwargs, validate_config
from .data import Dataset
from .model import NUM_CLASSES, NestSARR5T16
from .preprocessing import FEATURES, FRAMES, VERSION as PREPROCESSING_VERSION

# Exact counts at the default sizes (checked at start-up and in the tests).
EXPECTED_PARAMS = {
    "full": 1_146_656,
    "no_hand_branch": 1_123_352,
    "coarse_parts": 1_128_224,
    "unidirectional_sweep": 1_144_304,
    "no_interaction": 1_131_072,
    "capped_fast_memory": 1_146_296,
    "no_fast_memory": 1_135_736,
}
# Strict MFLOPs per clip (jaxpr dot MACs x 2, scan bodies x length, batch 1).
STRICT_MFLOPS = {
    "full": 59.161216,
    "no_hand_branch": 56.094336,
    "coarse_parts": 57.667968,
    "unidirectional_sweep": 55.474816,
    "no_interaction": 58.31232,
    "capped_fast_memory": 59.161216,
    "no_fast_memory": 58.865536,
}

# R4-FMSE+Geometry per-class recall of its weakest classes (EXPERIMENT_REPORT, best checkpoints).
R4_WEAK_CLASS_RECALL = {
    "xsub": {"A073": 0.2785, "A072": 0.3635, "A074": 0.3965, "A071": 0.4330, "A084": 0.4703,
             "A091": 0.4791, "A105": 0.4922, "A082": 0.5096, "A075": 0.5114, "A012": 0.5331},
    "xset": {"A072": 0.3980, "A012": 0.4076, "A073": 0.4078, "A074": 0.4362, "A084": 0.4928,
             "A071": 0.5041, "A011": 0.5220, "A107": 0.5264, "A076": 0.5266, "A075": 0.5343},
}

TRAIN_METRICS = ("loss", "main", "aux", "kl", "acc", "aug_acc", "agreement", "eta", "alpha",
                 "g4_eta", "g4_alpha", "fast_scale_m4", "fast_scale_g4", "pair_scale",
                 "aux_m4_acc", "aux_hand_acc")
EVAL_COLUMNS = ("val_loss", "val_acc", "val_top5", "eta", "alpha", "aux_m4_acc", "aux_hand_acc", "count")


def implementation_identity(variant):
    """Hash of the files that define the model function and its inputs.

    A fix to the loop/launcher/data plumbing keeps a run resumable; any change to
    the model, preprocessing or configuration starts a new identity (new OUT_DIR).
    """
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("model.py", "preprocessing.py", "config.py"):
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    return dict(BASE_IDENTITY, version=VERSION, variant=variant, source_sha256=digest.hexdigest())


def make_model(config):
    return NestSARR5T16(**model_kwargs(config))


class State(train_state.TrainState):
    ema_params: object


def ce(logits, labels, smoothing, labels2=None, lam=None):
    """Cross-entropy with label smoothing; ``lam``/``labels2`` give the CutMix soft target (lam = 1: plain)."""
    target = jax.nn.one_hot(labels, NUM_CLASSES)
    if labels2 is not None:
        lam = lam.reshape(lam.shape + (1,) * (target.ndim - lam.ndim))
        target = lam * target + (1 - lam) * jax.nn.one_hot(labels2, NUM_CLASSES)
    target = target * (1 - smoothing) + smoothing / NUM_CLASSES
    return -jnp.sum(target * jax.nn.log_softmax(logits), axis=-1)


def build_steps(model, config):
    n_metrics = len(TRAIN_METRICS)

    def per_sample(params, key, batch):
        k1, k2 = jax.random.split(key)
        out = model.apply({"params": params}, batch["x"], training=True, rngs={"dropout": k1})
        aug = model.apply({"params": params}, batch["xa"], training=True, rngs={"dropout": k2})
        y, smooth = batch["y"], config["label_smoothing"]
        y2, lam = batch["y2"], batch["lam"]
        main = (ce(out["logits"], y, smooth, y2, lam) + ce(aug["logits"], y, smooth, y2, lam)) / 2
        aux = (ce(out["stream_logits"], y[:, None], smooth, y2[:, None], lam[:, None]).mean(1)
               + ce(aug["stream_logits"], y[:, None], smooth, y2[:, None], lam[:, None]).mean(1)) / 2
        temperature = config["consistency_temperature"]
        logp = jax.nn.log_softmax(out["logits"] / temperature)
        logq = jax.nn.log_softmax(aug["logits"] / temperature)
        kl = 0.5 * temperature ** 2 * jnp.sum((jnp.exp(logp) - jnp.exp(logq)) * (logp - logq), -1)
        loss = main + config["stream_aux_weight"] * aux + config["consistency_weight"] * kl
        hit = lambda logits: (logits.argmax(-1) == y).astype(jnp.float32)
        sl = out["stream_logits"]
        hand_acc = hit(sl[:, 1]) if sl.shape[1] > 1 else jnp.zeros_like(main)
        metrics = jnp.stack([
            loss, main, aux, kl, hit(out["logits"]), hit(aug["logits"]),
            (out["logits"].argmax(-1) == aug["logits"].argmax(-1)).astype(jnp.float32),
            out["sm_eta_mean"], out["sm_alpha_mean"], out["g4_eta_mean"], out["g4_alpha_mean"],
            out["fast_scale_m4"], out["fast_scale_g4"], out["pair_scale"], hit(sl[:, 0]), hand_acc,
        ], axis=-1)
        return metrics

    @jax.jit
    def train_step(state, key, batch):
        k = config["accumulation_steps"]
        if config.get("mix_prob", 0.0) > 0:       # static: with mixing off the key chain is untouched
            key, mix_key = jax.random.split(key)
            mixed_x, mixed_xa, y2, lam = mix_batch(mix_key, batch["x"], batch["xa"], batch["y"], batch["mask"],
                                                   config["mix_prob"])
        else:
            mixed_x, mixed_xa, y2, lam = batch["x"], batch["xa"], batch["y"], jnp.ones_like(batch["mask"])
        batch = dict(batch, x=mixed_x, xa=mixed_xa, y2=y2, lam=lam)
        micros = jax.tree.map(lambda x: x.reshape(k, config["micro_batch"], *x.shape[1:]), batch)
        denom = jnp.maximum(jnp.sum(batch["mask"]), 1)
        key, step_key = jax.random.split(key)
        drop_keys = jax.random.split(step_key, k)
        zero = jax.tree.map(jnp.zeros_like, state.params)

        def accumulate(carry, inputs):
            gradient_sum, metric_sum = carry
            micro, drop = inputs

            def loss_fn(params):
                totals = jnp.sum(per_sample(params, drop, micro) * micro["mask"][:, None], axis=0)
                return totals[0] / denom, totals

            (_, totals), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            return (jax.tree.map(lambda a, b: a + b, gradient_sum, gradients), metric_sum + totals), None

        (grads, metrics), _ = jax.lax.scan(
            accumulate, (zero, jnp.zeros(n_metrics, jnp.float32)), (micros, drop_keys))
        norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)
        ema = jax.tree.map(lambda e, p: config["ema_decay"] * e + (1 - config["ema_decay"]) * p,
                           state.ema_params, state.params)
        state = state.replace(ema_params=ema)
        return state, key, jnp.r_[metrics, denom, norm]

    @jax.jit
    def eval_step(params, batch):
        out = model.apply({"params": params}, batch["x"], training=False)
        y, mask, logits = batch["y"], batch["mask"], out["logits"]
        top5 = jax.lax.top_k(logits, 5)[1]
        sl = out["stream_logits"]
        hit = lambda lg: (lg.argmax(-1) == y).astype(jnp.float32)
        columns = [
            ce(logits, y, 0), hit(logits), jnp.any(top5 == y[:, None], axis=-1).astype(jnp.float32),
            out["sm_eta_mean"], out["sm_alpha_mean"], hit(sl[:, 0]),
            hit(sl[:, 1]) if sl.shape[1] > 1 else jnp.zeros_like(mask), jnp.ones_like(mask),
        ]
        return jnp.sum(jnp.stack(columns, -1) * mask[:, None], axis=0)

    @jax.jit
    def predict_step(params, batch):
        return model.apply({"params": params}, batch["x"], training=False)["logits"].argmax(-1)

    return train_step, eval_step, predict_step


def create_state(config, steps_per_epoch):
    total = config["epochs"] * steps_per_epoch
    warm = max(1, int(total * config["warmup_fraction"]))
    warm = min(warm, max(total - 1, 1))
    schedule = optax.warmup_cosine_decay_schedule(
        0, config["learning_rate"], warm, max(total, warm + 1), end_value=config["min_learning_rate"])
    optimizer = optax.chain(optax.clip_by_global_norm(config["grad_clip"]),
                            optax.adamw(schedule, weight_decay=config["weight_decay"]))
    model = make_model(config)
    key, init = jax.random.split(jax.random.PRNGKey(config["seed"]))
    params = model.init({"params": init, "dropout": init},
                        jnp.zeros((1, FRAMES, FEATURES), jnp.float32), training=False)["params"]
    count = sum(x.size for x in jax.tree.leaves(params))
    expected = EXPECTED_PARAMS[config["variant"]]
    if count != expected:
        raise RuntimeError(f"Model parameter mismatch for {config['variant']}: {count} != {expected}")
    state = State.create(apply_fn=model.apply, params=params, tx=optimizer, ema_params=params)
    return model, state, key, schedule, int(math.ceil(warm / steps_per_epoch))


def save_checkpoint(path, state, key, metadata):
    payload = {"state": serialization.to_state_dict(jax.device_get(state)), "key": np.asarray(key),
               "metadata_json": json.dumps(metadata, allow_nan=False)}
    atomic_bytes(path, serialization.msgpack_serialize(payload))


def restore_checkpoint(path, template, expected_hash):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    meta = json.loads(payload["metadata_json"])
    if meta["config_hash"] != expected_hash:
        raise ValueError("Resume config/data mismatch. Keep the original config or choose a new OUT_DIR.")
    state = serialization.from_state_dict(template, payload["state"])
    return jax.device_put(state), jax.device_put(payload["key"]), meta


def publish_best(out, metadata, params_count):
    name = metadata["best_checkpoint"]
    if Path(name).name != name or not name.startswith("best_epoch_"):
        raise ValueError("Invalid best-checkpoint filename")
    atomic_bytes(out / "best.msgpack", (out / name).read_bytes())
    atomic_json(out / "best.json", dict(
        model=metadata.get("model", MODEL_NAME), epoch=metadata["best_epoch"],
        val_accuracy=metadata["best"], params=params_count,
        preprocessing_version=PREPROCESSING_VERSION, pipeline_version=VERSION,
        config_hash=metadata["config_hash"], model_identity=metadata.get("model_identity")))


def write_result(out, protocol, metadata, digest, params_count, resumed=False):
    variant = metadata["config"]["variant"]
    result = {
        "model": metadata.get("model", MODEL_NAME),
        "variant": variant,
        "protocol": protocol,
        "best_val_accuracy": metadata["best"],
        "best_accuracy": metadata["best"],
        "best_epoch": metadata["best_epoch"],
        "last_epoch": metadata["epoch"],
        "epochs_run": metadata["epoch"],
        "params": params_count,
        "strict_mflops": STRICT_MFLOPS[variant],
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.local_devices()],
        "config_hash": digest,
        "checkpoint": str(out / "best.msgpack"),
        "preprocessing_version": PREPROCESSING_VERSION,
        "pipeline_version": VERSION,
        "stopped_early": metadata["stopped_early"],
        "resumed_completed": resumed,
        "model_identity": metadata.get("model_identity"),
    }
    atomic_json(out / "result.json", result)
    atomic_json(out.parent / f"result_{protocol}.json", result)
    return result


def per_class_report(out, protocol, dataset, val_ids, config, predict_step, report):
    """Confusion-based report of the best EMA checkpoint (written once)."""
    target = out / "per_class.json"
    if target.exists() or not (out / "best.msgpack").exists():
        return
    payload = serialization.msgpack_restore((out / "best.msgpack").read_bytes())
    params = jax.device_put(payload["ema_params"])
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    steps = math.ceil(len(val_ids) / config["eval_batch"])
    report(phase="Per-class report", current=0, total=steps)
    with closing(dataset.batches(val_ids, config["eval_batch"], config, protocol=protocol)) as batches:
        for index in range(steps):
            batch, _ = next(batches)
            pred = np.asarray(predict_step(params, jax.device_put(batch)))
            n = int(batch["mask"].sum())
            np.add.at(confusion, (batch["y"][:n], pred[:n]), 1)
            if index % config["progress_every"] == 0 or index + 1 == steps:
                report(phase="Per-class report", current=index + 1, total=steps)
    support = confusion.sum(1)
    recall = np.diag(confusion) / np.maximum(support, 1)
    names = [f"A{c + 1:03d}" for c in range(NUM_CLASSES)]
    off = confusion.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:20]
    pairs = [{"true": names[i // NUM_CLASSES], "pred": names[i % NUM_CLASSES],
              "count": int(off.flat[i])} for i in flat if off.flat[i] > 0]
    reference = R4_WEAK_CLASS_RECALL.get(protocol, {})
    atomic_json(target, {
        "protocol": protocol,
        "epoch": int(payload.get("epoch", 0)),
        "top1": float(np.trace(confusion) / max(confusion.sum(), 1)),
        "macro_recall": float(recall[support > 0].mean()) if (support > 0).any() else 0.0,
        "recall": {names[c]: float(recall[c]) for c in range(NUM_CLASSES)},
        "r4_weak_classes": {k: {"r4": v, "r5": float(recall[int(k[1:]) - 1]),
                                "delta_pp": 100 * (float(recall[int(k[1:]) - 1]) - v)}
                            for k, v in reference.items()},
        "top_confusions": pairs,
        "note": "Selected on the official validation split (best epoch); diagnostic, not an untouched test.",
    })


def run(config, protocol, cache, outdir, allow_cpu=False):
    config = validate_config(config)
    if protocol not in ("xsub", "xset"):
        raise ValueError("Expected xsub or xset")
    out = Path(outdir) / protocol
    out.mkdir(parents=True, exist_ok=True)
    if not allow_cpu and (jax.default_backend() != "gpu" or len(jax.local_devices()) != 1):
        raise RuntimeError(f"Expected one isolated GPU; backend={jax.default_backend()}, "
                           f"devices={jax.local_devices()}")

    variant = config["variant"]
    params_count = EXPECTED_PARAMS[variant]
    identity = implementation_identity(variant)
    dataset = Dataset(cache)
    if config.get("hand_filter", "none") != dataset.hand_filter:
        raise ValueError(f"config hand_filter={config.get('hand_filter', 'none')!r} but the hand cache was built with "
                         f"{dataset.hand_filter!r}")
    if config.get("body_align", "none") != dataset.body_align:
        raise ValueError(f"config body_align={config.get('body_align', 'none')!r} but the cache was built with "
                         f"{dataset.body_align!r}")
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
                 "parameters": params_count, "pipeline_version": VERSION, "model": MODEL_NAME,
                 "model_identity": identity}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    previous = read_json(out / "run_config.json")
    if previous is not None and previous != signature:
        raise ValueError("OUT_DIR contains another model, implementation, or config/data. Choose a new OUT_DIR.")
    atomic_json(out / "run_config.json", signature)

    report = Reporter(out / "status.json")
    report(phase="JAX startup", current=0, total=1, protocol=protocol, epoch=0, best=None,
           best_epoch=0, completed_epoch=0, variant=variant)
    model, state, key, schedule, warmup_epochs = create_state(config, steps)
    metadata = {"config_hash": digest, "config": config, "epoch": 0, "best": -1.0, "best_epoch": 0,
                "model": MODEL_NAME, "model_identity": identity, "bad_epochs": 0, "history": [],
                "warmup_epochs": warmup_epochs, "best_checkpoint": None, "stopped_early": False}
    last = out / "last.msgpack"
    if last.exists():
        state, key, metadata = restore_checkpoint(last, state, digest)
        if metadata["best_epoch"]:
            publish_best(out, metadata, params_count)
        atomic_json(out / "history.json", metadata["history"])
    report(best=metadata["best"] if metadata["best_epoch"] else None,
           best_epoch=metadata["best_epoch"], completed_epoch=metadata["epoch"])

    train_step, eval_step, predict_step = build_steps(model, config)
    if metadata["epoch"] >= config["epochs"] or metadata["stopped_early"]:
        per_class_report(out, protocol, dataset, val_ids, config, predict_step, report)
        result = write_result(out, protocol, metadata, digest, params_count, resumed=True)
        report(phase="Done", current=1, total=1, done=True, epoch=metadata["epoch"])
        return result

    compiled_train = compiled_eval = None
    guard = max(warmup_epochs, config["early_stop_guard_epoch"])
    n_train = len(TRAIN_METRICS)
    for epoch in range(metadata["epoch"] + 1, config["epochs"] + 1):
        epoch_t0 = time.perf_counter()
        report(phase="Train", epoch=epoch, current=0, total=steps, val_acc=None,
               best=metadata["best"] if metadata["best_epoch"] else None, best_epoch=metadata["best_epoch"])
        train_sum = np.zeros(n_train + 1, np.float64)
        timing = dict(prepare_service_s=0.0, data_wait_s=0.0, h2d_s=0.0, gpu_train_s=0.0,
                      gpu_eval_s=0.0, compile_train_s=0.0, compile_eval_s=0.0)
        grad_norm_sum = 0.0
        with closing(dataset.batches(train_ids, batch_size, config, epoch, True, protocol)) as batches:
            for index in range(steps):
                t = time.perf_counter()
                batch, prep_s = next(batches)
                timing["data_wait_s"] += time.perf_counter() - t
                timing["prepare_service_s"] += prep_s
                t = time.perf_counter()
                device_batch = jax.block_until_ready(jax.device_put(batch))
                timing["h2d_s"] += time.perf_counter() - t
                if compiled_train is None:
                    report(phase="Compile train", current=0, total=steps)
                    t = time.perf_counter()
                    compiled_train = train_step.lower(state, key, device_batch).compile()
                    timing["compile_train_s"] = time.perf_counter() - t
                t = time.perf_counter()
                state, key, metrics = jax.block_until_ready(compiled_train(state, key, device_batch))
                timing["gpu_train_s"] += time.perf_counter() - t
                values = np.asarray(metrics)
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"Nonfinite training values at epoch {epoch}, batch {index + 1}")
                train_sum += values[:n_train + 1]
                grad_norm_sum += float(values[-1])
                if index % config["progress_every"] == 0 or index + 1 == steps:
                    d = max(train_sum[n_train], 1.0)
                    report(phase="Train", current=index + 1, total=steps, loss=float(train_sum[0] / d),
                           train_acc=float(train_sum[4] / d), lr=float(schedule(state.step)),
                           wait_s=timing["data_wait_s"], gpu_s=timing["gpu_train_s"],
                           **r4_worker.memory_snapshot())
        if int(train_sum[n_train]) != len(train_ids):
            raise RuntimeError("Training sample accounting mismatch")

        eval_sum = np.zeros(len(EVAL_COLUMNS), np.float64)
        val_steps = math.ceil(len(val_ids) / config["eval_batch"])
        report(phase="Validate EMA", current=0, total=val_steps)
        with closing(dataset.batches(val_ids, config["eval_batch"], config, protocol=protocol)) as batches:
            for index in range(val_steps):
                t = time.perf_counter()
                batch, prep_s = next(batches)
                timing["data_wait_s"] += time.perf_counter() - t
                timing["prepare_service_s"] += prep_s
                device_batch = jax.block_until_ready(jax.device_put(batch))
                if compiled_eval is None:
                    report(phase="Compile validation", current=0, total=val_steps)
                    t = time.perf_counter()
                    compiled_eval = eval_step.lower(state.ema_params, device_batch).compile()
                    timing["compile_eval_s"] = time.perf_counter() - t
                t = time.perf_counter()
                vals = np.asarray(jax.block_until_ready(compiled_eval(state.ema_params, device_batch)))
                timing["gpu_eval_s"] += time.perf_counter() - t
                if not np.isfinite(vals).all():
                    raise FloatingPointError("Nonfinite validation values")
                eval_sum += vals
                if index % config["progress_every"] == 0 or index + 1 == val_steps:
                    report(phase="Validate EMA", current=index + 1, total=val_steps,
                           val_acc=float(eval_sum[1] / max(eval_sum[-1], 1)))
        if int(eval_sum[-1]) != len(val_ids):
            raise RuntimeError("Validation mask/count mismatch")
        n_val = eval_sum[-1]
        val = float(eval_sum[1] / n_val)
        best, bad, improved = r4_worker.stopping_update(
            metadata["best"], metadata["bad_epochs"], val, epoch, guard, config["min_delta"])

        d = train_sum[n_train]
        row = {"epoch": epoch}
        row.update({f"train_{name}": float(train_sum[i] / d) for i, name in enumerate(TRAIN_METRICS)})
        row.update({
            "train_loss": float(train_sum[0] / d), "train_acc": float(train_sum[4] / d),
            "augmented_train_acc": float(train_sum[5] / d),
            "val_loss": float(eval_sum[0] / n_val), "val_acc": val, "val_main_acc": val,
            "val_top5": float(eval_sum[2] / n_val), "eta": float(eval_sum[3] / n_val),
            "alpha": float(eval_sum[4] / n_val), "val_aux_m4_acc": float(eval_sum[5] / n_val),
            "val_aux_hand_acc": float(eval_sum[6] / n_val),
            "grad_norm_mean": grad_norm_sum / max(steps, 1),
            "train_samples": len(train_ids), "val_samples": len(val_ids), "bad_epochs": bad,
            "epoch_s": time.perf_counter() - epoch_t0, **timing, **r4_worker.memory_snapshot(),
        })

        previous_best = metadata["best_checkpoint"]
        metadata.update(epoch=epoch, best=best, bad_epochs=bad, stopped_early=bad >= config["patience"])
        report(phase="Save checkpoint", current=1, total=1)
        if improved:
            metadata.update(best_epoch=epoch, best_checkpoint=f"best_epoch_{epoch:04d}.msgpack")
            best_payload = {
                "model": MODEL_NAME, "model_identity": identity, "protocol": protocol, "epoch": epoch,
                "val_accuracy": val, "ema_params": jax.device_get(state.ema_params), "config": config,
                "preprocessing_version": PREPROCESSING_VERSION, "pipeline_version": VERSION,
                "cache_signature": dataset.meta["signature"], "train_samples": len(train_ids),
                "val_samples": len(val_ids), "params": params_count,
            }
            atomic_bytes(out / metadata["best_checkpoint"], serialization.to_bytes(best_payload))
        row.update(best_val_accuracy=best, best_epoch=metadata["best_epoch"])
        metadata["history"].append(row)
        save_checkpoint(last, state, key, metadata)
        if improved:
            publish_best(out, metadata, params_count)
            if previous_best and previous_best != metadata["best_checkpoint"]:
                (out / previous_best).unlink(missing_ok=True)
        atomic_json(out / "history.json", metadata["history"])
        report(phase="Epoch complete", best=best, best_epoch=metadata["best_epoch"], completed_epoch=epoch,
               val_acc=val, current=1, total=1, bad_epochs=bad, epoch_s=row["epoch_s"],
               wait_s=timing["data_wait_s"], gpu_s=timing["gpu_train_s"])
        if metadata["stopped_early"]:
            break

    per_class_report(out, protocol, dataset, val_ids, config, predict_step, report)
    result = write_result(out, protocol, metadata, digest, params_count)
    report(phase="Done", current=1, total=1, done=True, best=metadata["best"],
           best_epoch=metadata["best_epoch"], epoch=metadata["epoch"], completed_epoch=metadata["epoch"])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--protocol", required=True, choices=("xsub", "xset"))
    p.add_argument("--cache", required=True, help="R5 hand-cache directory")
    p.add_argument("--outdir", required=True)
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    print(MODEL_NAME, flush=True)
    run(json.loads(Path(a.config).read_text()), a.protocol, a.cache, a.outdir, a.allow_cpu)


if __name__ == "__main__":
    main()

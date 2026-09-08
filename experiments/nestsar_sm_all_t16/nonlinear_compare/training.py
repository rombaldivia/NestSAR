"""Shared training budgets; inner selection only; resumable active trials."""
from __future__ import annotations
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

from ..streaming.io_utils import atomic_bytes, atomic_json, read_json
from .data import make_batch
from .metrics import scores
from .models import make_model


class Trainer:
    def __init__(self, arm, config, trial):
        self.model = make_model(arm, config, trial)
        self.config, self.trial = config, trial
        self.optimizer = optax.chain(optax.clip_by_global_norm(config["grad_clip"]),
                                     optax.scale_by_adam(),
                                     optax.add_decayed_weights(trial["weight_decay"]),
                                     optax.scale(-1.0))
        self.compiled_step = self.compiled_eval = None

        def step(state, key, batch, learning_rate):
            key, dropout = jax.random.split(key)

            def loss(params):
                logits = self.model.apply({"params": params}, batch["x"], training=True,
                                          rngs={"dropout": dropout})
                labels = jax.nn.one_hot(batch["y"], 2)
                labels = (1-config["label_smoothing"])*labels + config["label_smoothing"]/2
                ce = optax.softmax_cross_entropy(logits, labels)
                value = jnp.sum(ce*batch["weights"])/jnp.maximum(batch["weights"].sum(), 1e-8)
                return value, (logits.argmax(-1) == batch["y"])

            (value, correct), grads = jax.value_and_grad(loss, has_aux=True)(state.params)
            updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
            updates = jax.tree.map(lambda u: learning_rate*u, updates)
            state = state.replace(step=state.step+1,
                                  params=optax.apply_updates(state.params, updates), opt_state=opt_state)
            metrics = jnp.asarray([value, jnp.sum(correct*batch["mask"]), batch["mask"].sum()])
            return state, key, metrics

        self.step = jax.jit(step)
        self.evaluate = jax.jit(lambda params, x: jax.nn.softmax(self.model.apply({"params": params}, x), -1))

    def initialize(self, shape, seed):
        key, init = jax.random.split(jax.random.PRNGKey(seed))
        params = self.model.init({"params": init, "dropout": init}, jnp.zeros((1, *shape)), training=False)["params"]
        state = TrainState.create(apply_fn=self.model.apply, params=params, tx=self.optimizer)
        state = state.replace(step=jnp.asarray(0, jnp.int32))
        return state, key

    def prepare(self, state, key, batch):
        times = dict(compile_s=0.0, warmup_s=0.0)
        if self.compiled_step is None:
            start = time.perf_counter()
            lr = jnp.asarray(self.trial["learning_rate"], jnp.float32)
            self.compiled_step = self.step.lower(state, key, batch, lr).compile()
            self.compiled_eval = self.evaluate.lower(state.params, batch["x"]).compile()
            times["compile_s"] = time.perf_counter()-start
            start = time.perf_counter()
            warm = jax.block_until_ready(self.compiled_step(state, key, batch, lr))
            jax.block_until_ready(self.compiled_eval(state.params, batch["x"]))
            del warm  # Warm-up never advances optimizer, RNG or model state.
            times["warmup_s"] = time.perf_counter()-start
        return times


def predict(trainer, params, x, y, positions, scale):
    probabilities = np.empty((len(positions), 2), np.float32)
    for offset in range(0, len(positions), trainer.config["batch_size"]):
        pos = positions[offset:offset+trainer.config["batch_size"]]
        batch = make_batch(x, y, pos, trainer.config["batch_size"], scale)
        xb = jax.block_until_ready(jax.device_put(batch["x"]))
        if trainer.compiled_eval is None:
            trainer.compiled_eval = trainer.evaluate.lower(params, xb).compile()
        value = jax.block_until_ready(trainer.compiled_eval(params, xb))
        probabilities[offset:offset+len(pos)] = np.asarray(value)[:len(pos)]
    return probabilities


def train_candidate(trainer, x, y, fit, select, scale, seed, directory, report):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    done = read_json(directory / "done.json")
    if done is not None:
        if not (directory / "best.msgpack").is_file():
            raise RuntimeError(f"Incomplete candidate checkpoint: {directory}")
        return done
    c = trainer.config
    state, key = trainer.initialize(x.shape[1:], seed)
    best_params = state.params
    metadata = dict(epoch=0, best=-1.0, best_epoch=0, bad_epochs=0, history=[],
                    stopped_early=False, seed=seed, trial=trainer.trial,
                    params=int(sum(p.size for p in jax.tree.leaves(state.params))))
    last = directory / "last.msgpack"
    if last.exists():
        payload = serialization.msgpack_restore(last.read_bytes())
        state = serialization.from_state_dict(state, payload["state"])
        state = jax.device_put(state)
        key = jnp.asarray(payload["key"])
        best_params = jax.device_put(payload["best_params"])
        metadata = payload["metadata"]
    counts = np.bincount(y[fit], minlength=2)
    weights = (len(fit)/(2*counts)).astype(np.float32)
    steps = math.ceil(len(fit)/c["batch_size"])
    report(phase="Compile/warm-up", epoch=metadata["epoch"], current=0, total=1,
           best=metadata["best"] if metadata["best_epoch"] else None, best_epoch=metadata["best_epoch"],
           train_acc=None, val_acc=None)
    example = jax.device_put(make_batch(x, y, fit[:c["batch_size"]], c["batch_size"], scale, weights))
    preparation = trainer.prepare(state, key, example)
    start_epoch = metadata["epoch"]+1
    for epoch in range(start_epoch, c["epochs"]+1):
        if metadata["stopped_early"]:
            break
        epoch_start = time.perf_counter()
        warm = min(c["warmup_epochs"], max(c["epochs"]-1, 1))
        fraction = min(1.0, epoch/warm) if epoch <= warm else (
            .02 + .98*.5*(1+math.cos(math.pi*(epoch-warm)/max(c["epochs"]-warm, 1))))
        lr = jnp.asarray(trainer.trial["learning_rate"]*fraction, jnp.float32)
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(fit)
        train_correct = train_count = loss_sum = 0.0
        timing = dict(gpu_train_s=0.0, inner_eval_s=0.0, h2d_s=0.0, batch_prepare_s=0.0)
        report(phase="Train", epoch=epoch, current=0, total=steps,
               best=metadata["best"] if metadata["best_epoch"] else None,
               best_epoch=metadata["best_epoch"], val_acc=None)
        for step, offset in enumerate(range(0, len(order), c["batch_size"])):
            t0 = time.perf_counter()
            pos = order[offset:offset+c["batch_size"]]
            batch = make_batch(x, y, pos, c["batch_size"], scale, weights)
            timing["batch_prepare_s"] += time.perf_counter()-t0
            t0 = time.perf_counter()
            batch = jax.block_until_ready(jax.device_put(batch))
            timing["h2d_s"] += time.perf_counter()-t0
            t0 = time.perf_counter()
            state, key, m = jax.block_until_ready(trainer.compiled_step(state, key, batch, lr))
            timing["gpu_train_s"] += time.perf_counter()-t0
            m = np.asarray(m)
            if not np.isfinite(m).all():
                raise FloatingPointError(f"Nonfinite training loss: {directory}, epoch {epoch}")
            loss_sum += float(m[0])*len(pos)
            train_correct += float(m[1])
            train_count += float(m[2])
            if step % 5 == 0 or step+1 == steps:
                report(current=step+1, loss=loss_sum/train_count, train_acc=train_correct/train_count)
        report(phase="Inner selection", current=0, total=math.ceil(len(select)/c["batch_size"]))
        t0 = time.perf_counter()
        selected = scores(y[select], predict(trainer, state.params, x, y, select, scale))
        timing["inner_eval_s"] = time.perf_counter()-t0
        improved = selected["balanced_accuracy"] > metadata["best"] + 1e-6
        if improved:
            best_params = state.params
            metadata.update(best=selected["balanced_accuracy"], best_epoch=epoch, selection=selected)
        metadata["bad_epochs"] = 0 if improved or epoch <= warm else metadata["bad_epochs"]+1
        metadata["stopped_early"] = metadata["bad_epochs"] >= c["patience"]
        metadata["epoch"] = epoch
        metadata["history"].append(dict(epoch=epoch, lr=float(lr), train_accuracy=train_correct/train_count,
                                        train_loss=loss_sum/train_count, selection=selected,
                                        wall_s=time.perf_counter()-epoch_start, **timing,
                                        **(preparation if epoch == start_epoch else {})))
        payload = dict(state=serialization.to_state_dict(jax.device_get(state)), key=np.asarray(key),
                       best_params=jax.device_get(best_params), metadata=metadata)
        atomic_bytes(last, serialization.msgpack_serialize(payload))
        report(best=metadata["best"], best_epoch=metadata["best_epoch"],
               val_acc=selected["balanced_accuracy"], current=1, total=1)
    atomic_bytes(directory / "best.msgpack", serialization.msgpack_serialize(jax.device_get(best_params)))
    atomic_json(directory / "done.json", metadata)
    last.unlink(missing_ok=True)
    return metadata

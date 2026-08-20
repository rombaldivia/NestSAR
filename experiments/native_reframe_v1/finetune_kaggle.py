#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint fine-tuning for NestSAR Native-Reframe v1 on Kaggle TPU.

Loads NESTSAR_INIT_PARAMS into the exact Native-Reframe architecture, resets
optimizer/scheduler, then fine-tunes with the environment configuration.
Includes colored tqdm bars for train, validation, and final reframed-window
validation. Uses the same modern-JAX pmap replication compatibility shim as
run_kaggle.py.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from tqdm.auto import tqdm

from experiments.native_reframe_v1 import train_kaggle as base


def _replicate_for_pmap(tree, devices):
    ndev = len(devices)
    if ndev < 1:
        raise RuntimeError("No JAX devices visible")

    def replicate_leaf(x):
        x = jnp.asarray(x)
        return jnp.broadcast_to(x, (ndev,) + x.shape)

    return jax.tree_util.tree_map(replicate_leaf, tree)


# Compatibility with current Kaggle JAX, where device_put_replicated is gone.
jax.device_put_replicated = _replicate_for_pmap


TRAIN_COLOUR = os.environ.get("NESTSAR_TQDM_TRAIN_COLOUR", "green")
VAL_COLOUR = os.environ.get("NESTSAR_TQDM_VAL_COLOUR", "blue")
REFRAME_COLOURS = ("cyan", "blue", "magenta", "yellow")


def evaluate_colored(params_repl, eval_step, samples, frames: int, ndev: int, desc: str, colour: str):
    ds = base.SkeletonDataset(samples, frames=frames, cache=False)
    total_correct = total_count = total_loss = 0.0
    md = mg = ng = 0.0
    batches = 0

    iterator = base.batch_iterator(
        ds,
        base.CFG.eval_batch,
        shuffle=False,
        seed=0,
        drop_last=False,
    )
    total_batches = (len(ds) + base.CFG.eval_batch - 1) // base.CFG.eval_batch
    bar = tqdm(
        iterator,
        total=total_batches,
        desc=desc,
        colour=colour,
        dynamic_ncols=True,
        leave=True,
    )

    for x, y in bar:
        xs, ys, mask = base.pad_and_shard_eval(x, y, base.CFG.eval_batch, ndev)
        out = eval_step(
            params_repl,
            jnp.asarray(xs),
            jnp.asarray(ys),
            jnp.asarray(mask),
        )
        correct = float(np.asarray(out["correct"][0]))
        count = float(np.asarray(out["count"][0]))
        loss_sum = float(np.asarray(out["loss_sum"][0]))
        total_correct += correct
        total_count += count
        total_loss += loss_sum
        md += float(np.asarray(out["memory_delta"][0]))
        mg += float(np.asarray(out["memory_gate"][0]))
        ng += float(np.asarray(out["nested_gate"][0]))
        batches += 1

        bar.set_postfix(
            acc=f"{100.0 * total_correct / max(1.0, total_count):.2f}%",
            loss=f"{total_loss / max(1.0, total_count):.4f}",
            refresh=False,
        )

    return {
        "frames": frames,
        "accuracy": 100.0 * total_correct / max(1.0, total_count),
        "loss": total_loss / max(1.0, total_count),
        "correct": int(total_correct),
        "count": int(total_count),
        "memory_delta": md / max(1, batches),
        "memory_gate": mg / max(1, batches),
        "nested_gate": ng / max(1, batches),
    }


def main():
    base.validate_config()
    base.seed_everything(base.CFG.seed)

    init_raw = os.environ.get("NESTSAR_INIT_PARAMS", "").strip()
    if not init_raw:
        raise ValueError("Set NESTSAR_INIT_PARAMS=/path/to/best_params.msgpack")
    init_path = Path(init_raw)
    if not init_path.is_file():
        raise FileNotFoundError(f"Initial checkpoint not found: {init_path}")

    devices = jax.local_devices()
    ndev = len(devices)
    if ndev < 1:
        raise RuntimeError("No JAX devices visible")
    if base.CFG.global_batch % ndev:
        raise ValueError(
            f"NESTSAR_GLOBAL_BATCH={base.CFG.global_batch} must be divisible by {ndev} devices"
        )
    if base.CFG.eval_batch % ndev:
        raise ValueError(
            f"NESTSAR_EVAL_BATCH={base.CFG.eval_batch} must be divisible by {ndev} devices"
        )

    outdir = Path(base.CFG.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    best_path = outdir / "best_params.msgpack"
    last_path = outdir / "last_state.msgpack"
    curve_path = outdir / "curve.jsonl"
    meta_path = outdir / "run_config.json"

    dataset_path = base.find_dataset()
    data = base.load_pickle(dataset_path)
    train_samples, val_samples, train_key, val_key = base.extract_splits(data, base.CFG.protocol)
    train_ds = base.SkeletonDataset(
        train_samples,
        frames=base.CFG.train_frames,
        cache=base.CFG.cache_train,
    )

    steps_per_epoch = len(train_ds) // base.CFG.global_batch
    if steps_per_epoch < 1:
        raise RuntimeError("Training split smaller than global batch")
    total_steps = base.CFG.epochs * steps_per_epoch

    model = base.build_model()
    rng = jax.random.PRNGKey(base.CFG.seed)
    state = base.create_state(model, rng, total_steps)
    params_count = base.tree_param_count(state.params)

    # Exact shape-checked checkpoint restore. Optimizer is intentionally reset.
    loaded_params = base.load_params(init_path, state.params)
    if base.tree_shape_signature(loaded_params) != base.tree_shape_signature(state.params):
        raise RuntimeError("Checkpoint parameter tree does not match Native-Reframe architecture")
    state = state.replace(params=loaded_params)
    base.audit_reframe_parameter_invariance(model, state.params)

    train_step, eval_step = base.make_pmapped_steps(model)
    state_repl = _replicate_for_pmap(state, devices)

    base.log("=" * 108)
    base.log("NESTSAR NATIVE-REFRAME v1 — CHECKPOINT FINE-TUNE")
    base.log("=" * 108)
    base.log(f"Backend:               {jax.default_backend()}")
    base.log(f"Visible JAX devices:   {ndev}")
    base.log(f"Protocol:              {base.CFG.protocol.upper()} ({train_key}/{val_key})")
    base.log(f"Dataset:               {dataset_path}")
    base.log(f"Train/val:             {len(train_samples):,}/{len(val_samples):,}")
    base.log(f"Initial checkpoint:    {init_path}")
    base.log("Optimizer state:       RESET (new AdamW + new cosine schedule)")
    base.log(f"Train frames:          {base.CFG.train_frames}")
    base.log(f"Reframe eval windows:  {base.CFG.eval_frames}")
    base.log(f"Parameters:            {params_count:,}")
    base.log(f"Global/local batch:    {base.CFG.global_batch}/{base.CFG.global_batch // ndev}")
    base.log(f"Eval global/local:     {base.CFG.eval_batch}/{base.CFG.eval_batch // ndev}")
    base.log(f"Epochs/patience:       {base.CFG.epochs}/{base.CFG.patience}")
    base.log(f"LR/min LR:             {base.CFG.learning_rate:.3e}/{base.CFG.min_learning_rate:.3e}")
    base.log(f"Dropout/WD/LS:         {base.CFG.dropout:.3f}/{base.CFG.weight_decay:.3f}/{base.CFG.label_smoothing:.3f}")
    base.log("Softmax attention:     NONE")
    base.log("Transformer/GCN/CNN:   NONE")
    base.log("=" * 108)

    meta = dict(base.dataclasses.asdict(base.CFG))
    meta["init_params"] = str(init_path)
    meta["optimizer_reset"] = True
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Verify we loaded the expected checkpoint before taking one optimizer step.
    init_val = evaluate_colored(
        state_repl.params,
        eval_step,
        val_samples,
        base.CFG.train_frames,
        ndev,
        desc=f"INIT VAL T{base.CFG.train_frames}",
        colour=VAL_COLOUR,
    )
    base.log(
        f"CHECKPOINT VERIFY | T{base.CFG.train_frames}="
        f"{init_val['accuracy']:.5f}% loss={init_val['loss']:.5f}"
    )

    # The initial checkpoint is a valid candidate. Stage-2 must beat it.
    best_acc = init_val["accuracy"]
    best_path.write_bytes(serialization.to_bytes(loaded_params))
    patience = 0

    for epoch in range(1, base.CFG.epochs + 1):
        t0 = time.time()
        losses, accs, memd, memg, nestg, gradn = [], [], [], [], [], []

        iterator = base.batch_iterator(
            train_ds,
            base.CFG.global_batch,
            shuffle=True,
            seed=base.CFG.seed + epoch,
            drop_last=True,
        )
        bar = tqdm(
            iterator,
            total=steps_per_epoch,
            desc=f"E{epoch:02d}/{base.CFG.epochs:02d} TRAIN",
            colour=TRAIN_COLOUR,
            dynamic_ncols=True,
            leave=True,
        )

        for step, (x, y) in enumerate(bar, start=1):
            xs, ys = base.shard_batch(x, y, ndev)
            keys = jax.random.split(
                jax.random.fold_in(rng, epoch * 1_000_000 + step),
                ndev,
            )
            state_repl, metrics = train_step(
                state_repl,
                jnp.asarray(xs),
                jnp.asarray(ys),
                keys,
            )

            loss = float(np.asarray(metrics["loss"][0]))
            acc = float(np.asarray(metrics["accuracy"][0]))
            losses.append(loss)
            accs.append(acc)
            memd.append(float(np.asarray(metrics["memory_delta"][0])))
            memg.append(float(np.asarray(metrics["memory_gate"][0])))
            nestg.append(float(np.asarray(metrics["nested_gate"][0])))
            gradn.append(float(np.asarray(metrics["grad_norm"][0])))

            if step == 1 or step % 20 == 0 or step == steps_per_epoch:
                bar.set_postfix(
                    loss=f"{np.mean(losses):.4f}",
                    acc=f"{100.0*np.mean(accs):.2f}%",
                    mem=f"{np.mean(memg):.3f}",
                    nested=f"{np.mean(nestg):.3f}",
                    refresh=False,
                )

        val = evaluate_colored(
            state_repl.params,
            eval_step,
            val_samples,
            base.CFG.train_frames,
            ndev,
            desc=f"E{epoch:02d}/{base.CFG.epochs:02d} VAL T{base.CFG.train_frames}",
            colour=VAL_COLOUR,
        )
        train_acc = 100.0 * float(np.mean(accs))

        base.log(
            f"FT E{epoch:03d} | train loss={np.mean(losses):.5f} acc={train_acc:.3f}% | "
            f"val T{base.CFG.train_frames}={val['accuracy']:.5f}% loss={val['loss']:.5f} | "
            f"memD={np.mean(memd):.5f} mem_gate={np.mean(memg):.4f} "
            f"nested_gate={np.mean(nestg):.4f} grad={np.mean(gradn):.4f} | "
            f"{(time.time()-t0)/60.0:.2f} min"
        )

        host_state = jax.tree_util.tree_map(lambda z: z[0], state_repl)
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            patience = 0
            base.save_params(best_path, host_state.params)
            base.log(f"FT BEST -> {best_acc:.5f}% | saved {best_path}")
        else:
            patience += 1
            base.log(f"FT patience -> {patience}/{base.CFG.patience}")

        last_payload = {
            "state": host_state,
            "epoch": np.int32(epoch),
            "best_acc": np.float32(best_acc),
            "patience": np.int32(patience),
        }
        last_path.write_bytes(serialization.to_bytes(last_payload))

        with curve_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_accuracy": train_acc,
                "val": val,
                "best_accuracy": best_acc,
                "patience": patience,
            }) + "\n")

        if patience >= base.CFG.patience:
            base.log(f"Fine-tune early stop: patience {patience}/{base.CFG.patience}")
            break

    host_state = jax.tree_util.tree_map(lambda z: z[0], state_repl)
    best_params = base.load_params(best_path, host_state.params)
    best_repl = _replicate_for_pmap(best_params, devices)

    base.log("=" * 108)
    base.log("FINAL FINE-TUNED REFRAME EVALUATION — SAME CHECKPOINT, NO WINDOW RETRAINING")
    base.log("=" * 108)
    final = {}
    for i, t in enumerate(base.CFG.eval_frames):
        gc.collect()
        metrics = evaluate_colored(
            best_repl,
            eval_step,
            val_samples,
            t,
            ndev,
            desc=f"REFRAME T{t:02d}",
            colour=REFRAME_COLOURS[i % len(REFRAME_COLOURS)],
        )
        final[str(t)] = metrics
        base.log(
            f"T{t:02d}: acc={metrics['accuracy']:.5f}% "
            f"({metrics['correct']}/{metrics['count']}) loss={metrics['loss']:.5f}"
        )

    summary = {
        "model": "NestSAR-Native-Reframe-v1-FineTune",
        "protocol": base.CFG.protocol,
        "parameters": params_count,
        "init_params": str(init_path),
        "optimizer_reset": True,
        "initial_t16_accuracy": init_val["accuracy"],
        "best_finetune_accuracy": best_acc,
        "train_frames": base.CFG.train_frames,
        "eval_frames": list(base.CFG.eval_frames),
        "same_checkpoint_all_windows": True,
        "reframe_results": final,
    }
    (outdir / "final_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    base.log(f"Done. Summary: {outdir / 'final_summary.json'}")


if __name__ == "__main__":
    main()

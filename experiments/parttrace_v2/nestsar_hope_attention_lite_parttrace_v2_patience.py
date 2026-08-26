#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patience-enabled trainer for NestSAR-HOPE Attention-Lite v2 PartTrace.

Reuses the architecture and helpers from the base PartTrace v2 implementation,
while adding configurable early stopping and a single dynamic tqdm line per worker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Ensure the repository root is importable when this script is launched by path
# from experiments/parttrace_v2 (e.g. by the parallel Kaggle launcher).
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from tqdm.auto import tqdm

import nestsar as ns
from nestsar_hope_attention_lite_parttrace_v2 import (
    MODEL_NAME,
    FRAMES,
    PERSONS,
    JOINTS,
    COORDS,
    NUM_CLASSES,
    NestSARPartTraceV2,
    tree_numel,
    find_dataset,
    configure_nestsar,
    make_optimizer,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="auto")
    ap.add_argument("--protocol", choices=["xsub", "xset"], default="xsub")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.03)
    ap.add_argument("--warmup-fraction", type=float, default=0.10)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--outdir", default="/kaggle/working/NestSAR_PartTrace_v2")
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    if args.patience <= 0:
        raise ValueError("--patience must be greater than zero")

    configure_nestsar(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    model = NestSARPartTraceV2(dropout=args.dropout)
    dummy = jnp.zeros((1, FRAMES, PERSONS * JOINTS * COORDS), jnp.float32)
    variables = model.init({"params": rng, "dropout": rng}, dummy, training=True)
    params = variables["params"]

    print("=" * 108, flush=True)
    print(MODEL_NAME + " + configurable patience", flush=True)
    print("=" * 108, flush=True)
    print("Protocol:   ", args.protocol.upper(), flush=True)
    print("Backend:    ", jax.default_backend(), flush=True)
    print("Devices:    ", jax.devices(), flush=True)
    print("Parameters: ", f"{tree_numel(params):,}", flush=True)
    print("Epochs:     ", args.epochs, flush=True)
    print("Patience:   ", args.patience, flush=True)

    try:
        compiled = jax.jit(
            lambda p, xx: model.apply({"params": p}, xx, training=False)["logits"]
        ).lower(params, dummy).compile()
        cost = compiled.cost_analysis()
        if isinstance(cost, list) and cost:
            cost = cost[0]
        flops = float(cost.get("flops", 0.0)) if isinstance(cost, dict) else 0.0
        if flops:
            print("XLA GFLOPs: ", f"{flops / 1e9:.9f}", flush=True)
    except Exception as exc:
        print("XLA audit unavailable:", exc, flush=True)

    if args.audit_only:
        out = model.apply({"params": params}, dummy, training=False)
        print("Smoke logits:", out["logits"].shape, flush=True)
        print("AUDIT-ONLY PASS", flush=True)
        return

    dataset_path = find_dataset(args.dataset)
    raw = ns.load_pickle(dataset_path)
    train_samples, val_samples = ns.build_samples(
        raw,
        protocol=args.protocol,
        max_train=args.max_train_samples,
        max_val=args.max_val_samples,
        seed=args.seed,
    )
    train_ds = ns.SkeletonDataset(train_samples)
    val_ds = ns.SkeletonDataset(val_samples)

    train_steps = math.ceil(len(train_ds) / args.batch_size)
    val_steps = math.ceil(len(val_ds) / args.eval_batch_size)
    total_steps = max(1, train_steps * args.epochs)
    tx = make_optimizer(args.learning_rate, args.weight_decay, total_steps, args.warmup_fraction)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y, step_rng):
        def loss_fn(p):
            out = model.apply({"params": p}, x, training=True, rngs={"dropout": step_rng})
            logits = out["logits"]
            labels = jax.nn.one_hot(y, NUM_CLASSES)
            loss = jnp.mean(optax.softmax_cross_entropy(logits, labels))
            correct = jnp.sum(jnp.argmax(logits, axis=-1) == y)
            return loss, (correct, logits)

        (loss, (correct, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, correct, logits

    @jax.jit
    def eval_step(params, x, y):
        logits = model.apply({"params": params}, x, training=False)["logits"]
        labels = jax.nn.one_hot(y, NUM_CLASSES)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, labels))
        correct = jnp.sum(jnp.argmax(logits, axis=-1) == y)
        return loss, correct

    outdir = Path(args.outdir) / args.protocol
    outdir.mkdir(parents=True, exist_ok=True)

    best = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    run_start = time.time()
    tqdm_position = int(os.environ.get("NESTSAR_TQDM_POSITION", "0"))

    # One reusable dynamic bar per protocol worker. It resets TRAIN -> VAL on the same line.
    bar = tqdm(
        total=train_steps,
        position=tqdm_position,
        leave=False,
        dynamic_ncols=True,
        mininterval=0.25,
    )

    completed_epochs = 0
    for epoch in range(1, args.epochs + 1):
        completed_epochs = epoch
        epoch_rng = jax.random.fold_in(rng, epoch)

        bar.reset(total=train_steps)
        bar.set_description(f"{args.protocol.upper()} TRAIN E{epoch:03}/{args.epochs}")

        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        batch_index = 0

        for x_np, y_np in ns.batch_iterator(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            drop_last=False,
        ):
            step_rng = jax.random.fold_in(epoch_rng, batch_index)
            x = jnp.asarray(x_np, jnp.float32)
            y = jnp.asarray(y_np, jnp.int32)
            params, opt_state, loss, correct, _ = train_step(
                params, opt_state, x, y, step_rng
            )
            batch_size_actual = len(y_np)
            train_loss_sum += float(loss) * batch_size_actual
            train_correct += int(correct)
            train_count += batch_size_actual
            batch_index += 1

            train_acc = train_correct / max(train_count, 1)
            train_loss = train_loss_sum / max(train_count, 1)
            bar.set_postfix_str(
                f"loss={train_loss:.4f} acc={100*train_acc:.2f}% best={100*max(best,0.0):.2f}%"
            )
            bar.update(1)

        bar.reset(total=val_steps)
        bar.set_description(f"{args.protocol.upper()} VAL   E{epoch:03}/{args.epochs}")

        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0

        for x_np, y_np in ns.batch_iterator(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=args.seed,
            drop_last=False,
        ):
            x = jnp.asarray(x_np, jnp.float32)
            y = jnp.asarray(y_np, jnp.int32)
            loss, correct = eval_step(params, x, y)
            batch_size_actual = len(y_np)
            val_loss_sum += float(loss) * batch_size_actual
            val_correct += int(correct)
            val_count += batch_size_actual

            live_val = val_correct / max(val_count, 1)
            live_best = max(best, live_val)
            bar.set_postfix_str(
                f"acc={100*live_val:.2f}% best={100*max(live_best,0.0):.2f}%"
            )
            bar.update(1)

        train_loss = train_loss_sum / max(train_count, 1)
        train_acc = train_correct / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        val_acc = val_correct / max(val_count, 1)

        improved = val_acc > best
        if improved:
            best = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            payload = {
                "model": MODEL_NAME,
                "epoch": epoch,
                "protocol": args.protocol,
                "seed": args.seed,
                "val_accuracy": val_acc,
                "params": params,
            }
            (outdir / "best.msgpack").write_bytes(serialization.to_bytes(payload))
        else:
            epochs_without_improvement += 1

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "best_val_accuracy": best,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(record)
        (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        # Keep tqdm on one line; only emit a final message when early stopping/finish happens.
        bar.set_postfix_str(
            f"acc={100*val_acc:.2f}% best={100*best:.2f}% patience={epochs_without_improvement}/{args.patience}"
        )
        bar.refresh()

        if epochs_without_improvement >= args.patience:
            break

    bar.close()

    early_stopped = completed_epochs < args.epochs
    result = {
        "model": MODEL_NAME,
        "protocol": args.protocol,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": completed_epochs,
        "patience": args.patience,
        "early_stopped": early_stopped,
        "parameters": tree_numel(params),
        "best_val_accuracy": best,
        "best_epoch": best_epoch,
        "wall_hours": (time.time() - run_start) / 3600.0,
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"{args.protocol.upper()} COMPLETE | best={100*best:.5f}% | "
        f"epoch={best_epoch} | completed={completed_epochs}/{args.epochs} | "
        f"patience={args.patience}",
        flush=True,
    )


if __name__ == "__main__":
    main()

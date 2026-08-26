#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from experiments.parttrace_v3_attention_lite.audit_model import (
    FRAMES, NUM_CLASSES, EXPECTED_BASE_PARAMS, EXPECTED_BASE_LEAVES,
    load_canonical_prefix, make_wrapper, tree_numel, tree_leaves,
)

EXPECTED_COUNTS = {"xsub": (63_026, 50_919), "xset": (54_468, 59_477)}
EXPECTED_TOTAL_PARAMS = 2_532_130
EXPECTED_TOTAL_LEAVES = 755


def parse_args():
    p = argparse.ArgumentParser(description="Train Attention-Lite + PartTrace v3 on one protocol/GPU")
    p.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    p.add_argument("--dataset", default="auto")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=6e-4)
    p.add_argument("--min-learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.08)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.998)
    p.add_argument("--predictive-loss-weight", type=float, default=0.10)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--outdir", default="/kaggle/working/NestSAR_PartTrace_v3")
    p.add_argument("--save-online", action="store_true")
    p.add_argument("--audit-first", action="store_true")
    return p.parse_args()


def find_dataset(ns, explicit: str):
    if explicit != "auto":
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    preferred = Path("/kaggle/input/models/paolamaydana/ntudanno/other/default/1/ntu120_3danno.pkl")
    if preferred.is_file():
        return preferred
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            hits = list(root.rglob("ntu120_3danno.pkl"))
            if hits:
                return hits[0]
    raise FileNotFoundError("ntu120_3danno.pkl not found")


def smoothed_ce(logits, y, smoothing: float):
    targets = jax.nn.one_hot(y, NUM_CLASSES)
    targets = targets * (1.0 - smoothing) + smoothing / NUM_CLASSES
    return -jnp.mean(jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1))


def main() -> int:
    a = parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required; got {jax.default_backend()}")
    devices = list(jax.local_devices())
    if len(devices) != 1:
        raise RuntimeError(f"Expected exactly one process-visible GPU; got {devices}")

    mod, source = load_canonical_prefix(a.protocol)
    ns = mod.ns
    base = mod.build_model()
    model = make_wrapper(base)

    rng = jax.random.PRNGKey(a.seed)
    rng, init_rng, drop_rng = jax.random.split(rng, 3)
    dummy = jnp.zeros((1, FRAMES, 150), jnp.float32)

    base_vars = base.init({"params": init_rng, "dropout": drop_rng}, dummy, training=True)
    bp, bl = tree_numel(base_vars["params"]), tree_leaves(base_vars["params"])
    if (bp, bl) != (EXPECTED_BASE_PARAMS, EXPECTED_BASE_LEAVES):
        raise RuntimeError(f"Base guard failed params={bp} leaves={bl}")

    rng, init_rng, drop_rng = jax.random.split(rng, 3)
    params = model.init({"params": init_rng, "dropout": drop_rng}, dummy, training=True)["params"]
    tp, tl = tree_numel(params), tree_leaves(params)
    if (tp, tl) != (EXPECTED_TOTAL_PARAMS, EXPECTED_TOTAL_LEAVES):
        raise RuntimeError(f"V3 guard failed params={tp}/{EXPECTED_TOTAL_PARAMS}, leaves={tl}/{EXPECTED_TOTAL_LEAVES}")

    print("="*112, flush=True)
    print("NESTSAR ATTENTION-LITE + PARTTRACE V3 — TRAIN", flush=True)
    print("="*112, flush=True)
    print(f"Protocol: {a.protocol.upper()} | GPU: {devices[0]}", flush=True)
    print(f"Canonical base: {source}", flush=True)
    print(f"Params: {tp:,} | leaves: {tl}", flush=True)
    print(f"Epochs={a.epochs} patience={a.patience} batch={a.batch_size} eval_batch={a.eval_batch_size}", flush=True)
    print(f"LR={a.learning_rate:g}->{a.min_learning_rate:g} warmup={a.warmup_fraction:.3f} wd={a.weight_decay:g}", flush=True)
    print(f"label_smoothing={a.label_smoothing:g} grad_clip={a.grad_clip:g} EMA={a.ema_decay:g} predictive_w={a.predictive_loss_weight:g}", flush=True)
    print("="*112, flush=True)

    if a.audit_first:
        compiled = jax.jit(lambda p, x: model.apply({"params": p}, x, training=False)["logits"]).lower(params, dummy).compile()
        cost = compiled.cost_analysis()
        if isinstance(cost, list) and cost:
            cost = cost[0]
        flops = float(cost.get("flops", 0.0)) if isinstance(cost, dict) else 0.0
        print(f"XLA GFLOPs: {flops/1e9:.9f}", flush=True)

    dataset = find_dataset(ns, a.dataset)
    raw = ns.load_pickle(dataset)
    train_samples, val_samples = ns.build_samples(
        raw, protocol=a.protocol,
        max_train=a.max_train_samples,
        max_val=a.max_val_samples,
        seed=a.seed,
    )
    if a.max_train_samples == 0 and a.max_val_samples == 0:
        exp = EXPECTED_COUNTS[a.protocol]
        if (len(train_samples), len(val_samples)) != exp:
            raise RuntimeError(f"Split mismatch got {(len(train_samples),len(val_samples))}, expected {exp}")

    train_ds = ns.SkeletonDataset(train_samples)
    val_ds = ns.SkeletonDataset(val_samples)
    train_steps = math.ceil(len(train_ds) / a.batch_size)
    total_steps = max(1, train_steps * a.epochs)
    warmup_steps = max(1, int(total_steps * a.warmup_fraction))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=a.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=a.min_learning_rate,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(a.grad_clip),
        optax.adamw(schedule, b1=0.9, b2=0.999, eps=1e-8, weight_decay=a.weight_decay),
    )
    opt_state = tx.init(params)
    ema_params = params

    @jax.jit
    def train_step(params, ema_params, opt_state, x, y, step_rng):
        def loss_fn(p):
            out = model.apply({"params": p}, x, training=True, rngs={"dropout": step_rng})
            ce = smoothed_ce(out["logits"], y, a.label_smoothing)
            aux = jnp.array(0.0, jnp.float32)
            if "prediction" in out and "motion_target" in out:
                aux = jnp.mean(jnp.square(out["prediction"] - jax.lax.stop_gradient(out["motion_target"])))
            loss = ce + a.predictive_loss_weight * aux
            acc = jnp.mean(jnp.argmax(out["logits"], axis=-1) == y)
            return loss, (ce, aux, acc, out["parttrace_gate"])
        (loss, (ce, aux, acc, gate)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = jax.tree_util.tree_map(lambda e, p: a.ema_decay * e + (1.0-a.ema_decay) * p, ema_params, params)
        return params, ema_params, opt_state, loss, ce, aux, acc, gate, grad_norm

    @jax.jit
    def eval_step(params, x, y):
        out = model.apply({"params": params}, x, training=False)
        correct = jnp.sum(jnp.argmax(out["logits"], axis=-1) == y)
        return correct, out["parttrace_gate"]

    outdir = Path(a.outdir) / a.protocol
    outdir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    start = time.time()

    for epoch in range(1, a.epochs + 1):
        tr_correct = tr_count = 0
        tr_loss = tr_ce = tr_aux = 0.0
        gate_last = 0.0
        gn_last = 0.0
        epoch_rng = jax.random.fold_in(rng, epoch)
        for bi, (x_np, y_np) in enumerate(ns.batch_iterator(train_ds, batch_size=a.batch_size, shuffle=True, seed=a.seed+epoch, drop_last=False)):
            x = jnp.asarray(x_np, jnp.float32)
            y = jnp.asarray(y_np, jnp.int32)
            step_rng = jax.random.fold_in(epoch_rng, bi)
            params, ema_params, opt_state, loss, ce, aux, acc, gate, gn = train_step(params, ema_params, opt_state, x, y, step_rng)
            n = len(y_np)
            tr_count += n
            tr_correct += int(round(float(acc) * n))
            tr_loss += float(loss) * n
            tr_ce += float(ce) * n
            tr_aux += float(aux) * n
            gate_last = float(gate)
            gn_last = float(gn)

        val_correct = val_count = 0
        val_gate = 0.0
        for x_np, y_np in ns.batch_iterator(val_ds, batch_size=a.eval_batch_size, shuffle=False, seed=0, drop_last=False):
            x = jnp.asarray(x_np, jnp.float32)
            y = jnp.asarray(y_np, jnp.int32)
            correct, gate = eval_step(ema_params, x, y)
            val_correct += int(correct)
            val_count += len(y_np)
            val_gate = float(gate)

        train_acc = tr_correct / max(1, tr_count)
        val_acc = val_correct / max(1, val_count)
        improved = val_acc > best
        if improved:
            best, best_epoch, stale = val_acc, epoch, 0
            payload = {"model":"AttentionLitePartTraceV3", "protocol":a.protocol, "epoch":epoch, "seed":a.seed, "val_accuracy":val_acc, "ema_params":ema_params}
            if a.save_online:
                payload["params"] = params
            (outdir / "best_ema.msgpack").write_bytes(serialization.to_bytes(payload))
        else:
            stale += 1

        rec = {
            "epoch": epoch,
            "train_loss": tr_loss/max(1,tr_count),
            "train_ce": tr_ce/max(1,tr_count),
            "train_aux": tr_aux/max(1,tr_count),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "best_val_accuracy": best,
            "best_epoch": best_epoch,
            "parttrace_gate": val_gate,
            "grad_norm_last": gn_last,
            "stale_epochs": stale,
        }
        history.append(rec)
        (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"{a.protocol.upper()} E{epoch:03}/{a.epochs} | train={100*train_acc:.2f}% | val={100*val_acc:.2f}% | "
            f"BEST={100*best:.2f}%@E{best_epoch:03} | gate={val_gate:.4f} | stale={stale}/{a.patience}",
            flush=True,
        )
        if a.patience > 0 and stale >= a.patience:
            break

    result = {
        "model":"AttentionLitePartTraceV3",
        "protocol":a.protocol,
        "seed":a.seed,
        "params":tp,
        "leaves":tl,
        "best_val_accuracy":best,
        "best_epoch":best_epoch,
        "epochs_completed":len(history),
        "epochs_requested":a.epochs,
        "patience":a.patience,
        "wall_hours":(time.time()-start)/3600.0,
        "args":vars(a),
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{a.protocol.upper()} COMPLETE | best={100*best:.5f}% @ epoch {best_epoch}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

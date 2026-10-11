"""Weak-class fine-tuning without freezing any weight (a continual-learning test of the HOPE premise).

Starting from the best EMA weights of a finished (or stopped) run, train a few more epochs with every weight
trainable, a low learning rate, and batches where ``weak_fraction`` of the clips come from the ``weak_k``
weakest classes (the rest are uniform replay of all classes). With ``outer_cms`` on, the slow levels
(periods 4 / 8) move every 4 / 8 steps, which is what Nested Learning claims protects old knowledge.

No test leakage: the weak classes are chosen on TRAINING clips only (per-class recall on pooled strong
views of a random training subset, where the model still errs), never on the official validation split.
Every epoch reports top-1, the weak-class mean recall, the mean recall of all other classes and their drop
("forgetting") against the starting checkpoint, all on the validation split. Report the FIXED last epoch.

    python -m experiments.nestsar_r5_t16.finetune --run-dir OUT --protocol xsub --cache HAND_CACHE \\
        [--epochs 4 --lr-scale 0.1 --weak-k 20 --weak-fraction 0.4 --outer-cms inherit|on|off]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import closing
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_bytes, atomic_json

from .config import validate_config
from .data import Dataset
from .model import NUM_CLASSES
from .worker import EXPECTED_PARAMS, build_steps, create_state

NAMES = [f"A{c + 1:03d}" for c in range(NUM_CLASSES)]


def load_best(run_dir, protocol):
    out = Path(run_dir) / protocol
    for name in ("best.msgpack",):
        if (out / name).is_file():
            payload = serialization.msgpack_restore((out / name).read_bytes())
            return payload["ema_params"], payload["config"], int(payload.get("epoch", 0))
    cands = sorted(out.glob("best_epoch_*.msgpack"))
    if not cands:
        raise FileNotFoundError(f"No best checkpoint in {out}")
    payload = serialization.msgpack_restore(cands[-1].read_bytes())
    return payload["ema_params"], payload["config"], int(payload.get("epoch", 0))


def predict(predict_step, params, dataset, ids, config, protocol, view="clean", epoch=0):
    """Predictions for ``ids``: on the canonical clips (view='clean') or on pooled strong views ('strong')."""
    preds, labels = [], []
    size = config["eval_batch"]
    training = view == "strong"
    with closing(dataset.batches(ids, size, config, epoch, training, protocol)) as batches:
        for _ in range(math.ceil(len(ids) / size)):
            b, _ = next(batches)
            n = int(b["mask"].sum())
            x = b["xa"] if training else b["x"]
            p = np.asarray(predict_step(params, jax.device_put({"x": x})))
            preds.append(p[:n]), labels.append(b["y"][:n])
    return np.concatenate(preds), np.concatenate(labels)


def recall_per_class(pred, y):
    hit = np.bincount(y, weights=(pred == y).astype(np.float64), minlength=NUM_CLASSES)
    support = np.bincount(y, minlength=NUM_CLASSES)
    return hit / np.maximum(support, 1), support


def choose_weak_classes(predict_step, params, dataset, train_ids, config, protocol, k, samples, seed):
    """k lowest-recall classes on pooled strong views of a random TRAINING subset (no validation data)."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(train_ids, size=min(samples, len(train_ids)), replace=False)
    pred, y = predict(predict_step, params, dataset, ids, config, protocol, view="strong", epoch=7777)
    rec, support = recall_per_class(pred, y)
    order = [c for c in np.argsort(rec) if support[c] >= 5]
    weak = sorted(int(c) for c in order[:k])
    return weak, {NAMES[c]: float(rec[c]) for c in weak}, float((pred == y).mean())


def epoch_ids(train_ids, labels, weak, fraction, rng):
    """One epoch of ids: ``fraction`` from the weak classes (with replacement), the rest uniform replay."""
    train_ids = np.asarray(train_ids)
    weak_ids = train_ids[np.isin(labels[train_ids], weak)]
    n = len(train_ids)
    n_weak = int(round(fraction * n)) if len(weak_ids) else 0
    replay = rng.permutation(train_ids)[: n - n_weak]
    return np.concatenate([replay, rng.choice(weak_ids, n_weak, replace=True)]) if n_weak else replay


def _mean(a, scale=1.0):
    return float(scale * np.mean(a)) if np.size(a) else None      # None: no such class in this split


def summarize(rec, base_rec, weak, support):
    valid = support > 0
    weak_mask = np.zeros(NUM_CLASSES, bool)
    weak_mask[weak] = True
    w, other = weak_mask & valid, valid & ~weak_mask
    return {
        "weak_mean_recall": _mean(rec[w]),
        "other_mean_recall": _mean(rec[other]),
        "weak_delta_pp": _mean(rec[w] - base_rec[w], 100),
        "other_delta_pp": _mean(rec[other] - base_rec[other], 100),
        "forgetting_pp": _mean(np.clip(base_rec - rec, 0, None)[other], 100),   # recall lost by non-weak classes
        "classes_worse_by_2pp": int(np.sum((base_rec - rec)[other] > 0.02)),
    }


def _f(x, scale=1.0, signed=False):
    return "n/a" if x is None else (f"{scale * x:+.2f}" if signed else f"{scale * x:.2f}")


def run(run_dir, protocol, cache, outdir=None, epochs=4, lr_scale=0.1, weak_k=20, weak_fraction=0.4,
        outer_cms="inherit", select_samples=12000, allow_cpu=False, max_train=0, max_val=0, log=print):
    if not allow_cpu and jax.default_backend() != "gpu":
        raise RuntimeError("Expected a GPU (use --allow-cpu for a local test)")
    params, base_cfg, base_epoch = load_best(run_dir, protocol)
    over = dict(epochs=epochs, learning_rate=base_cfg["learning_rate"] * lr_scale,
                min_learning_rate=base_cfg["learning_rate"] * lr_scale * 0.1, warmup_fraction=0.05)
    if outer_cms != "inherit":
        over["outer_cms"] = outer_cms == "on"
    config = validate_config(dict(base_cfg, **over))
    tag = f"ft_k{weak_k}_f{weak_fraction:g}_lr{lr_scale:g}_cms{'on' if config['outer_cms'] else 'off'}"
    out = Path(outdir) if outdir else Path(run_dir) / protocol / tag
    out.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(cache)
    train_ids = np.asarray(dataset.splits[f"{protocol}_train"])
    val_ids = np.asarray(dataset.splits[f"{protocol}_val"])
    if max_train:
        train_ids = train_ids[:max_train]
    if max_val:
        val_ids = val_ids[:max_val]
    labels = np.asarray(dataset.labels)
    batch = config["micro_batch"] * config["accumulation_steps"]
    steps = math.ceil(len(train_ids) / batch)

    model, state, key, schedule, _ = create_state(config, steps)
    params = jax.tree.map(jnp.asarray, params)
    if sum(x.size for x in jax.tree.leaves(params)) != EXPECTED_PARAMS[config["variant"]]:
        raise ValueError("Checkpoint does not match the variant of its own config")
    state = state.replace(params=params, ema_params=params,
                          opt_state=state.tx.init(params))          # fresh optimizer, every weight trainable
    train_step, _, predict_step = build_steps(model, config)

    weak, weak_train_recall, strong_train_acc = choose_weak_classes(
        predict_step, params, dataset, train_ids, config, protocol, weak_k, select_samples, config["seed"])
    log(f"[{protocol}] weak classes (chosen on training strong views, acc {100 * strong_train_acc:.2f}%): "
        + ", ".join(NAMES[c] for c in weak))

    pred, y = predict(predict_step, params, dataset, val_ids, config, protocol)
    base_rec, support = recall_per_class(pred, y)
    base_top1 = float((pred == y).mean())
    rows = [dict(epoch=0, top1=base_top1, **summarize(base_rec, base_rec, weak, support))]
    log(f"[{protocol}] start (run epoch {base_epoch}): top1 {100 * base_top1:.3f}%  "
        f"weak {_f(rows[0]['weak_mean_recall'], 100)}%  other {_f(rows[0]['other_mean_recall'], 100)}%")

    rng = np.random.default_rng(config["seed"] + 4242)
    compiled = None
    for e in range(1, epochs + 1):
        t0 = time.time()
        ids = epoch_ids(train_ids, labels, weak, weak_fraction, rng)
        n_steps = math.ceil(len(ids) / batch)
        loss_sum, hit_sum, count = 0.0, 0.0, 0.0
        with closing(dataset.batches(ids, batch, config, 1000 + e, True, protocol)) as batches:
            for _ in range(n_steps):
                b, _ = next(batches)
                b = jax.device_put(b)
                if compiled is None:
                    compiled = train_step.lower(state, key, b).compile()
                state, key, metrics = compiled(state, key, b)
                m = np.asarray(metrics)
                if not np.isfinite(m).all():
                    raise FloatingPointError(f"Nonfinite values in fine-tuning epoch {e}")
                loss_sum, hit_sum, count = loss_sum + m[0], hit_sum + m[4], count + m[-2]
        pred, y = predict(predict_step, state.ema_params, dataset, val_ids, config, protocol)
        rec, _ = recall_per_class(pred, y)
        row = dict(epoch=e, top1=float((pred == y).mean()), train_loss=float(loss_sum / max(count, 1)),
                   train_acc=float(hit_sum / max(count, 1)), seconds=time.time() - t0,
                   **summarize(rec, base_rec, weak, support))
        rows.append(row)
        log(f"[{protocol}] ft {e}/{epochs}: top1 {100 * row['top1']:.3f}% ({100 * (row['top1'] - base_top1):+.3f})  "
            f"weak {_f(row['weak_mean_recall'], 100)}% ({_f(row['weak_delta_pp'], signed=True)})  "
            f"other {_f(row['other_mean_recall'], 100)}% ({_f(row['other_delta_pp'], signed=True)})  "
            f"forgetting {_f(row['forgetting_pp'])} pp  [{row['seconds']:.0f}s]")
        atomic_bytes(out / f"ft_epoch_{e:02d}.msgpack", serialization.to_bytes(
            {"ema_params": jax.device_get(state.ema_params), "config": config, "weak_classes": weak, "epoch": e}))
        result = {"protocol": protocol, "variant": config["variant"], "run_dir": str(run_dir),
                  "start_epoch": base_epoch, "weak_classes": [NAMES[c] for c in weak],
                  "weak_train_strong_recall": weak_train_recall, "weak_fraction": weak_fraction,
                  "lr": config["learning_rate"], "outer_cms": config["outer_cms"], "dmgd": config["dmgd"],
                  "history": rows, "per_class_start": {NAMES[c]: float(base_rec[c]) for c in range(NUM_CLASSES)},
                  "per_class_last": {NAMES[c]: float(rec[c]) for c in range(NUM_CLASSES)},
                  "note": "Weak classes chosen on training data only. Report the fixed last epoch."}
        atomic_json(out / "finetune.json", json.loads(json.dumps(       # numpy scalars -> plain JSON numbers
            result, default=lambda o: o.item() if hasattr(o, "item") else str(o))))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--protocol", required=True, choices=("xsub", "xset"))
    ap.add_argument("--cache", required=True, help="R5 hand cache used by the run")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr-scale", type=float, default=0.1, help="fraction of the run's peak learning rate")
    ap.add_argument("--weak-k", type=int, default=20)
    ap.add_argument("--weak-fraction", type=float, default=0.4)
    ap.add_argument("--outer-cms", default="inherit", choices=("inherit", "on", "off"),
                    help="on vs off is the HOPE test: does multi-frequency updating reduce forgetting?")
    ap.add_argument("--select-samples", type=int, default=12000)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0)
    a = ap.parse_args(argv)
    run(a.run_dir, a.protocol, a.cache, a.outdir, a.epochs, a.lr_scale, a.weak_k, a.weak_fraction, a.outer_cms,
        a.select_samples, a.allow_cpu, a.max_train, a.max_val, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()

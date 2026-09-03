#!/usr/bin/env python3
from __future__ import annotations

"""Frozen Hand-M4/G4 champion + confusion-matrix expert (CME).

The 1,854,650-parameter Hand-M4/G4 champion is loaded from best.msgpack and is
never optimized. A tiny residual expert is trained on cached frozen features.
The expert can change only the classes listed by --pairs. Validation reports
both always-on and deterministic confusion-routed accuracy; checkpoint
selection uses routed accuracy.

Console contract: one persistent tqdm bar per worker. With the dual-T4 runner
this produces exactly two live bars total (XSUB and XSET), with TRAIN/VAL shown
in the postfix and no per-batch print lines.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import serialization
from tqdm.auto import tqdm

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_localglobal_t16 import preprocessing as lg
from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.model import (
    EXPECTED_PARAMS_D32,
    M4LocalGlobalHandM4G4T32,
)
from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.preprocessing import (
    HAND_FEATURES,
    HAND_FRAMES,
    hand_tokens_t32,
)

FRAMES = ju.FRAMES
FEATURES = ju.FEATURES
NUM_CLASSES = ju.NUM_CLASSES
DEFAULT_PAIRS = "71-72,73-76,74-84,16-17,106-107,11-12,12-30,10-34"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", choices=["xsub", "xset"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--pairs", default=DEFAULT_PAIRS)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--residual-scale", type=float, default=0.15)
    p.add_argument("--route-margin", type=float, default=0.20)
    p.add_argument("--target-weight", type=float, default=3.0)
    p.add_argument("--delta-l2", type=float, default=1e-4)
    p.add_argument("--preserve-kl-weight", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--cache-batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--min-learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.10)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--tqdm-position", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--base-accuracy-tolerance", type=float, default=5e-5)
    return p.parse_args()


def parse_pairs(spec: str) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split("-", 1)
        a1, b1 = int(left), int(right)
        if not (1 <= a1 <= NUM_CLASSES and 1 <= b1 <= NUM_CLASSES):
            raise ValueError(f"Pair outside 1..{NUM_CLASSES}: {item}")
        if a1 == b1:
            raise ValueError(f"Degenerate pair: {item}")
        pairs.append((a1 - 1, b1 - 1))
    if not pairs:
        raise ValueError("At least one confusion pair is required.")
    return tuple(pairs)


def unique_targets(pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    return np.asarray(sorted({x for pair in pairs for x in pair}), dtype=np.int32)


def count_params(tree) -> int:
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def restore_base(checkpoint: Path):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = serialization.msgpack_restore(checkpoint.read_bytes())
    if "ema_params" not in payload:
        raise KeyError(f"ema_params missing from {checkpoint}")
    cfg = payload.get("config", {})
    model = M4LocalGlobalHandM4G4T32(
        spatial_dim=int(cfg.get("spatial_dim", 24)),
        model_dim=int(cfg.get("model_dim", 112)),
        dropout=float(cfg.get("dropout", 0.10)),
        hand_dim=int(cfg.get("hand_dim", 32)),
        hand_residual_scale=float(cfg.get("hand_residual_scale", 0.10)),
    )
    params = payload["ema_params"]
    nparams = count_params(params)
    if nparams != EXPECTED_PARAMS_D32:
        raise RuntimeError(
            f"Frozen Hand checkpoint has {nparams:,} params; expected {EXPECTED_PARAMS_D32:,}."
        )
    return model, params, cfg


def checkpoint_expected_accuracy(checkpoint: Path) -> float | None:
    meta = checkpoint.with_name("best.json")
    if not meta.is_file():
        return None
    d = json.loads(meta.read_text(encoding="utf-8"))
    value = d.get("val_accuracy")
    if value is None:
        return None
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def make_split_ids(annotations, split, protocol: str, max_train: int, max_val: int):
    by_id, train_ids, val_ids = ju.resolve_protocol_ids(annotations, split, protocol)
    if max_train:
        train_ids = train_ids[:max_train]
    if max_val:
        val_ids = val_ids[:max_val]
    return by_id, train_ids, val_ids


def build_frozen_cache(by_id, ids, model, params, cache_batch_size: int):
    n = len(ids)
    logits = np.empty((n, NUM_CLASSES), dtype=np.float32)
    desc = np.empty((n, 32), dtype=np.float32)
    labels = np.empty((n,), dtype=np.int32)

    @jax.jit
    def frozen_forward(xb, hb):
        out = model.apply({"params": params}, xb, hb, training=False)
        return out["logits"], out["hand_descriptor"]

    for start in range(0, n, cache_batch_size):
        end = min(start + cache_batch_size, n)
        bs = end - start
        xb = np.empty((bs, FRAMES, FEATURES), dtype=np.float32)
        hb = np.empty((bs, HAND_FRAMES, HAND_FEATURES), dtype=np.float32)
        yb = np.empty((bs,), dtype=np.int32)
        for j, sid in enumerate(ids[start:end]):
            ann = by_id[sid]
            kp = base.annotation_keypoints(ann)
            xb[j] = lg.segment_phase_tokens_localglobal(kp)
            hb[j] = hand_tokens_t32(kp)
            yb[j] = base.annotation_label(ann)
        lo, de = frozen_forward(jnp.asarray(xb), jnp.asarray(hb))
        logits[start:end] = np.asarray(jax.device_get(lo), dtype=np.float32)
        desc[start:end] = np.asarray(jax.device_get(de), dtype=np.float32)
        labels[start:end] = yb
    return logits, desc, labels


def softmax_np(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def pair_route_mask(base_logits, pairs, targets, route_margin: float):
    probs = softmax_np(base_logits)
    order = np.argsort(base_logits, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    margin = probs[np.arange(len(probs)), top1] - probs[np.arange(len(probs)), top2]
    pair_route = np.zeros((len(base_logits),), dtype=bool)
    for a, b in pairs:
        pair_route |= ((top1 == a) & (top2 == b)) | ((top1 == b) & (top2 == a))
    uncertain_target = np.isin(top1, targets) & (margin < route_margin)
    return pair_route | uncertain_target


def make_features(base_logits, hand_desc, targets):
    probs = softmax_np(base_logits)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-8)), axis=1)
    entropy /= math.log(NUM_CLASSES)
    order = np.argsort(base_logits, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    margin = probs[np.arange(len(probs)), top1] - probs[np.arange(len(probs)), top2]
    selected = base_logits[:, targets]
    return np.concatenate(
        [hand_desc, selected, entropy[:, None], margin[:, None]], axis=1
    ).astype(np.float32)


class CMEHead(nn.Module):
    hidden_dim: int
    num_targets: int
    dropout: float

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.LayerNorm(name="input_norm")(x)
        x = nn.Dense(self.hidden_dim, name="hidden")(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout, name="dropout")(x, deterministic=not training)
        return nn.Dense(
            self.num_targets,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="delta",
        )(x)


def scatter_delta(delta_target: jnp.ndarray, target_ids: jnp.ndarray) -> jnp.ndarray:
    out = jnp.zeros((delta_target.shape[0], NUM_CLASSES), dtype=delta_target.dtype)
    return out.at[:, target_ids].set(delta_target)


def audit_head_flops(head: CMEHead, params, feature_dim: int) -> float | None:
    try:
        dummy = jnp.zeros((1, feature_dim), jnp.float32)
        fn = jax.jit(lambda p, x: head.apply({"params": p}, x, training=False))
        compiled = fn.lower(params, dummy).compile()
        ca = compiled.cost_analysis()
        if isinstance(ca, list):
            ca = ca[0] if ca else {}
        return float(ca.get("flops", float("nan")))
    except Exception:
        return None


def iter_indices(n: int, batch_size: int, rng: np.random.Generator | None = None):
    idx = np.arange(n)
    if rng is not None:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=1) == labels))


def target_accuracy(logits: np.ndarray, labels: np.ndarray, targets: np.ndarray) -> float:
    mask = np.isin(labels, targets)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.argmax(logits[mask], axis=1) == labels[mask]))


def main() -> None:
    args = parse_args()
    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU; backend={jax.default_backend()} devices={jax.local_devices()}"
        )

    pairs = parse_pairs(args.pairs)
    targets = unique_targets(pairs)
    target_ids_jax = jnp.asarray(targets)
    checkpoint = Path(args.checkpoint)
    model, base_params, _ = restore_base(checkpoint)
    expected_base_acc = checkpoint_expected_accuracy(checkpoint)

    dataset = base.find_dataset(args.dataset)
    annotations, split = base.load_ntu(dataset)
    by_id, train_ids, val_ids = make_split_ids(
        annotations, split, args.protocol, args.max_train_samples, args.max_val_samples
    )

    train_base, train_desc, ytr = build_frozen_cache(
        by_id, train_ids, model, base_params, args.cache_batch_size
    )
    val_base, val_desc, yva = build_frozen_cache(
        by_id, val_ids, model, base_params, args.cache_batch_size
    )

    base_val_acc = accuracy(val_base, yva)
    if expected_base_acc is not None and abs(base_val_acc - expected_base_acc) > args.base_accuracy_tolerance:
        raise RuntimeError(
            f"Frozen checkpoint reconstruction mismatch: cached={base_val_acc:.8f}, "
            f"best.json={expected_base_acc:.8f}, tolerance={args.base_accuracy_tolerance:.1e}"
        )

    xtr = make_features(train_base, train_desc, targets)
    xva = make_features(val_base, val_desc, targets)
    route_tr = pair_route_mask(train_base, pairs, targets, args.route_margin)
    route_va = pair_route_mask(val_base, pairs, targets, args.route_margin)
    is_target_tr = np.isin(ytr, targets)
    train_active = route_tr | is_target_tr

    head = CMEHead(args.hidden_dim, len(targets), args.dropout)
    key = jax.random.PRNGKey(args.seed + (0 if args.protocol == "xsub" else 100000))
    key, init_key = jax.random.split(key)
    params = head.init(
        {"params": init_key, "dropout": init_key},
        jnp.zeros((1, xtr.shape[1]), jnp.float32),
        training=False,
    )["params"]
    ema_params = params
    specialist_params = count_params(params)
    specialist_flops = audit_head_flops(head, params, xtr.shape[1])

    train_steps = math.ceil(len(ytr) / args.batch_size)
    eval_steps = math.ceil(len(yva) / args.eval_batch_size)
    total_steps = max(1, train_steps * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_fraction))
    lr = optax.warmup_cosine_decay_schedule(
        0.0, args.learning_rate, warmup_steps, total_steps, args.min_learning_rate
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(lr, weight_decay=args.weight_decay),
    )
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, ema_params, opt_state, rng, xb, bb, yb, active, target_label):
        rng, drop = jax.random.split(rng)

        def loss_fn(p):
            dsmall = head.apply({"params": p}, xb, training=True, rngs={"dropout": drop})
            dfull = scatter_delta(dsmall, target_ids_jax)
            active_f = active.astype(jnp.float32)[:, None]
            final = bb + args.residual_scale * dfull * active_f
            ce = optax.softmax_cross_entropy_with_integer_labels(final, yb)
            weights = 1.0 + (args.target_weight - 1.0) * target_label.astype(jnp.float32)
            ce_loss = jnp.sum(ce * weights) / jnp.maximum(jnp.sum(weights), 1.0)
            delta_l2 = jnp.mean(jnp.square(dsmall) * active_f)
            base_prob = jax.nn.softmax(bb, axis=-1)
            final_logprob = jax.nn.log_softmax(final, axis=-1)
            base_logprob = jax.nn.log_softmax(bb, axis=-1)
            kl = jnp.sum(base_prob * (base_logprob - final_logprob), axis=-1)
            non_target = 1.0 - target_label.astype(jnp.float32)
            preserve = jnp.sum(kl * non_target) / jnp.maximum(jnp.sum(non_target), 1.0)
            loss = ce_loss + args.delta_l2 * delta_l2 + args.preserve_kl_weight * preserve
            acc = jnp.mean(jnp.argmax(final, axis=-1) == yb)
            return loss, (ce_loss, delta_l2, preserve, acc)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = jax.tree_util.tree_map(
            lambda e, p: args.ema_decay * e + (1.0 - args.ema_decay) * p,
            ema_params,
            params,
        )
        return params, ema_params, opt_state, rng, loss, metrics

    @jax.jit
    def eval_delta(params, xb):
        return head.apply({"params": params}, xb, training=False)

    outdir = Path(args.outdir) / args.protocol
    outdir.mkdir(parents=True, exist_ok=True)

    bar = tqdm(
        total=args.epochs * (train_steps + eval_steps),
        desc=f"{args.protocol.upper()} CME",
        position=args.tqdm_position,
        leave=True,
        dynamic_ncols=True,
        mininterval=0.20,
        smoothing=0.05,
    )

    best = base_val_acc
    best_epoch = 0
    best_always = base_val_acc
    best_target = target_accuracy(val_base, yva, targets)
    stale = 0

    for epoch in range(1, args.epochs + 1):
        rng_np = np.random.default_rng(
            args.seed + epoch + (0 if args.protocol == "xsub" else 100000)
        )
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_seen = 0

        for step, ii in enumerate(iter_indices(len(ytr), args.batch_size, rng_np), start=1):
            params, ema_params, opt_state, key, loss, metrics = train_step(
                params, ema_params, opt_state, key,
                jnp.asarray(xtr[ii]), jnp.asarray(train_base[ii]), jnp.asarray(ytr[ii]),
                jnp.asarray(train_active[ii]), jnp.asarray(is_target_tr[ii]),
            )
            bs = len(ii)
            train_loss_sum += float(loss) * bs
            train_acc_sum += float(metrics[3]) * bs
            train_seen += bs
            if step == 1 or step % 4 == 0 or step == train_steps:
                bar.set_postfix_str(
                    f"E{epoch:02d}/{args.epochs} TRAIN loss={train_loss_sum/max(train_seen,1):.4f} "
                    f"acc={100*train_acc_sum/max(train_seen,1):.2f}%",
                    refresh=False,
                )
            bar.update(1)

        routed_correct = 0
        always_correct = 0
        target_correct = 0
        target_count = 0
        route_count = 0
        eval_count = 0

        for step, ii in enumerate(iter_indices(len(yva), args.eval_batch_size), start=1):
            dsmall = np.asarray(
                jax.device_get(eval_delta(ema_params, jnp.asarray(xva[ii]))), dtype=np.float32
            )
            dfull = np.zeros((len(ii), NUM_CLASSES), dtype=np.float32)
            dfull[:, targets] = dsmall
            always_logits = val_base[ii] + args.residual_scale * dfull
            route = route_va[ii]
            routed_logits = val_base[ii] + args.residual_scale * dfull * route[:, None].astype(np.float32)
            pred_routed = np.argmax(routed_logits, axis=1)
            pred_always = np.argmax(always_logits, axis=1)
            yb = yva[ii]
            routed_correct += int(np.sum(pred_routed == yb))
            always_correct += int(np.sum(pred_always == yb))
            route_count += int(np.sum(route))
            eval_count += len(ii)
            tm = np.isin(yb, targets)
            if np.any(tm):
                target_correct += int(np.sum(pred_routed[tm] == yb[tm]))
                target_count += int(np.sum(tm))
            if step == 1 or step % 4 == 0 or step == eval_steps:
                bar.set_postfix_str(
                    f"E{epoch:02d}/{args.epochs} VAL routed={100*routed_correct/max(eval_count,1):.3f}% "
                    f"best={100*best:.3f}% route={100*route_count/max(eval_count,1):.1f}%",
                    refresh=False,
                )
            bar.update(1)

        routed_acc = routed_correct / max(eval_count, 1)
        always_acc = always_correct / max(eval_count, 1)
        target_acc = target_correct / max(target_count, 1)
        route_rate = route_count / max(eval_count, 1)

        if routed_acc > best + 1e-7:
            best = routed_acc
            best_epoch = epoch
            best_always = always_acc
            best_target = target_acc
            stale = 0
            payload = {
                "model": "HandM4G4_CME_FrozenSpecialist",
                "protocol": args.protocol,
                "epoch": epoch,
                "val_accuracy": routed_acc,
                "val_always_accuracy": always_acc,
                "val_target_accuracy": target_acc,
                "base_val_accuracy": base_val_acc,
                "route_rate": route_rate,
                "params": params,
                "ema_params": ema_params,
                "config": vars(args),
                "base_checkpoint": str(checkpoint),
                "base_params_frozen": EXPECTED_PARAMS_D32,
                "specialist_params": specialist_params,
                "specialist_flops": specialist_flops,
                "target_classes_one_based": [int(x) + 1 for x in targets],
                "pairs_one_based": [[a + 1, b + 1] for a, b in pairs],
            }
            (outdir / "best.msgpack").write_bytes(serialization.to_bytes(payload))
            (outdir / "best.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "val_accuracy": routed_acc,
                        "val_always_accuracy": always_acc,
                        "val_target_accuracy": target_acc,
                        "base_val_accuracy": base_val_acc,
                        "gain_pp": 100.0 * (routed_acc - base_val_acc),
                        "route_rate": route_rate,
                        "base_checkpoint": str(checkpoint),
                        "base_params_frozen": EXPECTED_PARAMS_D32,
                        "specialist_params": specialist_params,
                        "specialist_flops": specialist_flops,
                        "target_classes_one_based": [int(x) + 1 for x in targets],
                        "pairs_one_based": [[a + 1, b + 1] for a, b in pairs],
                        "attention": False,
                        "base_trainable": False,
                        "config": vars(args),
                    }, indent=2,
                ), encoding="utf-8",
            )
        else:
            stale += 1

        bar.set_postfix_str(
            f"E{epoch:02d}/{args.epochs} VAL={100*routed_acc:.3f}% BASE={100*base_val_acc:.3f}% "
            f"GAIN={100*(routed_acc-base_val_acc):+.3f}pp BEST={100*best:.3f}% P={stale}/{args.patience}",
            refresh=True,
        )
        if stale >= args.patience:
            bar.total = bar.n
            bar.refresh()
            break

    bar.close()

    result = {
        "protocol": args.protocol,
        "base_accuracy": base_val_acc,
        "best_accuracy": best,
        "best_epoch": best_epoch,
        "gain_pp": 100.0 * (best - base_val_acc),
        "best_always_accuracy": best_always,
        "best_target_accuracy": best_target,
        "route_rate_initial": float(np.mean(route_va)),
        "base_checkpoint": str(checkpoint),
        "base_params_frozen": EXPECTED_PARAMS_D32,
        "specialist_params": specialist_params,
        "specialist_flops": specialist_flops,
        "pairs_one_based": [[a + 1, b + 1] for a, b in pairs],
        "target_classes_one_based": [int(x) + 1 for x in targets],
        "backend": jax.default_backend(),
        "visible_devices": [str(d) for d in jax.local_devices()],
        "base_trainable": False,
        "attention": False,
        "config": vars(args),
    }
    (Path(args.outdir) / f"result_{args.protocol}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Frozen R4 readout intervention. Exploratory: the backbone saw the probe dev set.

A refits the original head. B adds a nonlinear residual from exactly the original
head inputs. C adds the same kind of residual, with a matched parameter budget,
and three ordered, zero-mean contrasts of each pre/post-G4 chunk sequence.
Official evaluation labels never select epochs or update parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np

VERSION = "r4-frozen-readout-v1"
BASE_REF = "8225970527d009f7dfaa9fbe699aca53b3129312"
EXPECTED_PARAMS = 1_831_932
CLASSES, STREAMS, DIM = 120, 4, 112
VARIANTS = ("A_control", "B_descriptor", "C_temporal")
PREPROCESSING = "sm-all-personaware-p2-segmentpose-v2"
CACHE_VERSION = "sm-all-shared-cache-personaware-p2-r4-v1"
MODEL_KEYS = ("spatial_dim", "model_dim", "dropout", "controller_dim",
              "fast_rank", "head_rank", "sm_residual_scale", "head_residual_scale")
LIMITATION = (
    "Exploratory frozen-checkpoint diagnostic. Probe development groups come from "
    "official training and were already seen by the backbone. The backbone was "
    "also selected using the official evaluation partition. Probe seeds are not "
    "independent backbone training seeds. No causal ceiling or fresh benchmark "
    "claim follows from this experiment."
)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def atomic_bytes(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(value)
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_count(params):
    if isinstance(params, dict):
        return sum(tree_count(x) for x in params.values())
    return int(np.asarray(params).size)


def report(output, protocol, phase, current, total, **extra):
    atomic_json(Path(output) / "status.json", dict(
        protocol=protocol, phase=phase, current=int(current),
        total=max(1, int(total)), **extra))


def contrast_matrix():
    # Orthonormal nonconstant DCT-II columns: zero temporal mean, fixed order.
    t = np.arange(4, dtype=np.float64)[:, None]
    k = np.arange(1, 4, dtype=np.float64)[None, :]
    return (np.sqrt(0.5) * np.cos(np.pi * (t + 0.5) * k / 4)).astype(np.float32)


def temporal_contrasts(mixed, chunks):
    # mixed [B,16,4,112]; chunks [B,4,4,112] = stream,time,channel.
    b = mixed.shape[0]
    pre = mixed.reshape(b, 4, 4, STREAMS, DIM).mean(2).transpose(0, 2, 1, 3)
    states = np.stack([pre, chunks], axis=2)  # B,stream,stage,time,channel
    return np.einsum("bsktd,tq->bskqd", states, contrast_matrix(),
                     optimize=True).reshape(b, -1).astype(np.float32)


class CanonicalStore:
    """Read-only canonical cache; raw.npy is unnecessary for frozen extraction."""
    def __init__(self, cache, protocol):
        self.root = Path(cache)
        self.meta = json.loads((self.root / "manifest.json").read_text())
        sig = self.meta["signature"]
        if sig.get("preprocessing") != PREPROCESSING or sig.get("cache_version") != CACHE_VERSION:
            raise ValueError("Expected the existing person-aware P2 v3 canonical cache.")
        for name in ("canonical.npy", "labels.npy", "ids.json", "splits.json"):
            p = self.root / name
            if not p.is_file() or p.stat().st_size != self.meta["files"][name]:
                raise ValueError(f"Incomplete canonical cache: {p}")
        self.x = np.load(self.root / "canonical.npy", mmap_mode="r")
        self.y = np.load(self.root / "labels.npy", mmap_mode="r")
        self.names = json.loads((self.root / "ids.json").read_text())
        splits = json.loads((self.root / "splits.json").read_text())
        self.train = np.asarray(splits[protocol + "_train"], np.int64)
        self.test = np.asarray(splits[protocol + "_val"], np.int64)
        n = len(self.y)
        if self.x.shape != (n, 16, 750) or len(self.names) != n or n != 113945:
            raise ValueError(f"Unexpected canonical layout: {self.x.shape}, n={n}")
        expected = (63026, 50919) if protocol == "xsub" else (54468, 59477)
        if (len(self.train), len(self.test)) != expected:
            raise ValueError(f"{protocol}: official split counts differ from {expected}")
        both = np.concatenate([self.train, self.test])
        if not np.array_equal(np.sort(both), np.arange(n)):
            raise ValueError("Official splits overlap, duplicate samples, or omit samples.")
        if len(set(self.names)) != n or np.any((self.y < 0) | (self.y >= CLASSES)):
            raise ValueError("Invalid labels or duplicate sample names.")
        for i, name in enumerate(self.names):
            match = re.search(r"A(\d{3})", name)
            if match is None or int(match[1]) - 1 != self.y[i]:
                raise ValueError(f"Sample/label mismatch: {name}")
        self.groups = parse_groups(self.names, protocol)
        if set(self.groups[self.train]) & set(self.groups[self.test]):
            raise ValueError("Subject/setup overlap between official train and evaluation.")

    def identity(self):
        names = ("manifest.json", "labels.npy", "ids.json", "splits.json")
        p = self.root / "canonical.npy"
        return dict(signature=self.meta["signature"],
                    hashes={n: sha256(self.root / n) for n in names},
                    canonical_size=p.stat().st_size,
                    canonical_mtime_ns=p.stat().st_mtime_ns)


def parse_groups(names, protocol):
    field = "P" if protocol == "xsub" else "S"
    result = []
    for name in names:
        match = re.search(field + r"(\d{3})", name)
        if match is None:
            raise ValueError(f"Cannot recover {field} group from {name}")
        result.append(int(match[1]))
    return np.asarray(result, np.int32)


def group_dev_split(train, labels, groups, seed, fraction=0.20):
    """Deterministic group holdout ONLY inside official training."""
    unique = np.unique(groups[train])
    if len(unique) < 3:
        raise ValueError("At least three training groups are required.")
    count = min(len(unique) - 1, max(1, round(fraction * len(unique))))
    rng = np.random.default_rng(seed)
    for _ in range(200):
        dev_groups = rng.permutation(unique)[:count]
        is_dev = np.isin(groups[train], dev_groups)
        fit, dev = train[~is_dev], train[is_dev]
        if all(np.bincount(labels[x], minlength=CLASSES).min() >= 2 for x in (fit, dev)):
            return fit, dev, sorted(int(x) for x in dev_groups)
    raise ValueError("Could not form a training-only group split covering every class.")


def validate_identity(path, identity):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(f"Different run/data/config in {path.parent}; choose a new output.")
    else:
        atomic_json(path, identity)


def core_params(base):
    names = [f"classifier_{i}" for i in range(STREAMS)]
    names += ["adaptive_head_u", "adaptive_head_v"]
    return {k: {n: np.asarray(v).copy() for n, v in base[k].items()} for k in names}


def core_logits(params, desc, context, scale, xp):
    weights, coeff = context[:, :4], context[:, 4:]
    stream = xp.stack([desc[:, i] @ params[f"classifier_{i}"]["kernel"]
                       + params[f"classifier_{i}"]["bias"] for i in range(4)], axis=1)
    main = xp.einsum("bs,bsc->bc", weights, stream)
    fused = xp.einsum("bs,bsd->bd", weights, desc)
    delta = ((fused @ params["adaptive_head_u"]["kernel"]) * coeff
             @ params["adaptive_head_v"]["kernel"])
    return main + scale * delta


def hidden_sizes():
    b_input, c_input = STREAMS * DIM + 6, STREAMS * DIM + 6 + 4 * 2 * 3 * DIM
    hb = 224
    hb_params = (b_input + CLASSES + 1) * hb + CLASSES
    hc = max(1, round((hb_params - CLASSES) / (c_input + CLASSES + 1)))
    hc_params = (c_input + CLASSES + 1) * hc + CLASSES
    if abs(hc_params - hb_params) / hb_params > 0.02:
        raise AssertionError("B/C residual parameter budgets differ by more than 2%.")
    return {"A_control": 0, "B_descriptor": hb, "C_temporal": hc}


def init_probe(core, input_dim, hidden, seed):
    p = {"core": {k: {n: np.asarray(v).copy() for n, v in d.items()}
                  for k, d in core.items()}}
    if hidden:
        rng = np.random.default_rng(seed)
        p["residual"] = dict(
            w1=(rng.normal(size=(input_dim, hidden)) / np.sqrt(input_dim)).astype(np.float32),
            b1=np.zeros(hidden, np.float32),
            w2=np.zeros((hidden, CLASSES), np.float32),
            b2=np.zeros(CLASSES, np.float32))
    return p


class FeatureBank:
    def __init__(self, root):
        self.root = Path(root)
        self.arrays = {k: np.load(self.root / (k + ".npy"), mmap_mode="r")
                       for k in ("desc", "context", "temporal", "logits")}

    def inputs(self, ids, variant):
        d = np.asarray(self.arrays["desc"][ids], np.float32)
        c = np.asarray(self.arrays["context"][ids], np.float32)
        pieces = [d.reshape(len(ids), -1), c]
        if variant == "C_temporal":
            pieces.append(np.asarray(self.arrays["temporal"][ids], np.float32))
        return d, c, np.concatenate(pieces, axis=1)

    def normalization(self, ids, variant, batch=512):
        if variant == "A_control":
            return np.zeros(454, np.float32), np.ones(454, np.float32)
        total, squares = None, None
        for start in range(0, len(ids), batch):
            z = self.inputs(ids[start:start + batch], variant)[2].astype(np.float64)
            if total is None:
                total = np.zeros(z.shape[1], np.float64)
                squares = np.zeros_like(total)
            total += z.sum(0)
            squares += (z * z).sum(0)
        mean = total / len(ids)
        std = np.sqrt(np.maximum(squares / len(ids) - mean * mean, 0))
        return mean.astype(np.float32), np.maximum(std, 1e-3).astype(np.float32)

    def batch(self, ids, variant, labels, mean, std, size):
        n = len(ids)
        d, c, z = self.inputs(ids, variant)
        batch = dict(desc=np.zeros((size, 4, DIM), np.float32),
                     context=np.zeros((size, 6), np.float32),
                     z=np.zeros((size, z.shape[1]), np.float32),
                     y=np.zeros(size, np.int32), mask=np.zeros(size, np.float32))
        batch["desc"][:n], batch["context"][:n] = d, c
        batch["z"][:n] = (z - mean) / std
        batch["y"][:n], batch["mask"][:n] = labels[ids], 1
        return batch


def extract_features(args, store, base, config, core, checkpoint_hash):
    import jax
    import jax.numpy as jnp
    from .model import NestSARSMAllT16

    root = Path(args.output) / "features"
    root.mkdir(exist_ok=True)
    identity = dict(version=VERSION, checkpoint_sha256=checkpoint_hash,
                    cache=store.identity(), base_ref=BASE_REF, dtype="float32")
    validate_identity(root / "identity.json", identity)
    shapes = dict(desc=(len(store.y), 4, DIM), context=(len(store.y), 6),
                  temporal=(len(store.y), 4 * 2 * 3 * DIM), logits=(len(store.y), CLASSES))
    if (root / "complete.json").exists():
        bank = FeatureBank(root)
        if any(bank.arrays[k].shape != shape or bank.arrays[k].dtype != np.float32
               for k, shape in shapes.items()):
            raise ValueError("Feature cache shape/dtype mismatch.")
        return bank, json.loads((root / "complete.json").read_text())
    progress_file = root / "progress.json"
    start_at = json.loads(progress_file.read_text())["next_row"] if progress_file.exists() else 0
    if not start_at:
        required = sum(math.prod(shape) * 4 for shape in shapes.values())
        existing = sum((root / (k + ".npy")).stat().st_size for k in shapes
                       if (root / (k + ".npy")).exists())
        if shutil.disk_usage(root).free + existing < required + (512 << 20):
            raise RuntimeError(f"Need {required / 2**30:.2f} GiB for this protocol's features.")
    arrays = {k: np.lib.format.open_memmap(root / (k + ".npy"),
              mode="r+" if start_at else "w+", dtype=np.float32, shape=shape)
              for k, shape in shapes.items()}
    if any(a.shape != shapes[k] or a.dtype != np.float32 for k, a in arrays.items()):
        raise ValueError("Incomplete cache has the wrong layout.")
    model = NestSARSMAllT16(**{k: config[k] for k in MODEL_KEYS})

    @jax.jit
    def forward(x):
        out = model.apply({"params": base}, x, training=False)
        return {k: out[k] for k in ("descriptors", "fusion_weights", "sm_head_coeff",
                                   "mixed_frame_stack", "chunk_states", "logits")}

    # Verify the cached-core computation on a real batch, even when resuming.
    first = np.asarray(store.x[store.train[:min(16, len(store.train))]], np.float32)
    check = jax.device_get(forward(jnp.asarray(first)))
    context = np.concatenate([check["fusion_weights"], check["sm_head_coeff"]], -1)
    reconstructed = jax.device_get(core_logits(
        jax.tree.map(jnp.asarray, core), jnp.asarray(check["descriptors"]),
        jnp.asarray(context), config["head_residual_scale"], jnp))
    error = float(np.max(np.abs(reconstructed - check["logits"])))
    if not np.allclose(reconstructed, check["logits"], rtol=1e-5, atol=3e-5):
        raise RuntimeError(f"Original-head reconstruction failed: max error {error}")
    steps = math.ceil(len(store.y) / args.extract_batch)
    for start in range(start_at, len(store.y), args.extract_batch):
        end = min(start + args.extract_batch, len(store.y))
        x = np.zeros((args.extract_batch, 16, 750), np.float32)
        x[:end - start] = store.x[start:end]
        out = jax.device_get(forward(jnp.asarray(x)))
        values = dict(desc=out["descriptors"],
                      context=np.concatenate([out["fusion_weights"], out["sm_head_coeff"]], -1),
                      temporal=temporal_contrasts(out["mixed_frame_stack"], out["chunk_states"]),
                      logits=out["logits"])
        for key, value in values.items():
            if not np.isfinite(value[:end - start]).all():
                raise FloatingPointError(f"Nonfinite frozen feature: {key}")
            arrays[key][start:end] = value[:end - start]
        bi = start // args.extract_batch + 1
        if bi % 20 == 0 or end == len(store.y):
            for a in arrays.values():
                a.flush()
            atomic_json(progress_file, {"next_row": end})
        report(args.output, args.protocol, "Extract frozen features", bi, steps)
    for a in arrays.values():
        a.flush()
    info = dict(samples=len(store.y), original_head_max_abs_error=error,
                feature_bytes=sum(a.nbytes for a in arrays.values()),
                temporal_features="Three fixed zero-mean DCT contrasts per stream and pre/post-G4 stage")
    atomic_json(root / "complete.json", info)
    del forward, model, arrays
    jax.clear_caches()
    return FeatureBank(root), info


def make_steps(scale, args, steps_per_epoch):
    import jax
    import jax.numpy as jnp
    import optax

    total = args.epochs * steps_per_epoch
    warmup = min(steps_per_epoch, max(1, total - 1))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.learning_rate * 0.1, peak_value=args.learning_rate,
        warmup_steps=warmup, decay_steps=max(total, warmup + 1),
        end_value=args.learning_rate * 0.01)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adamw(schedule, weight_decay=args.weight_decay))

    def predict(params, batch, key=None):
        logits = core_logits(params["core"], batch["desc"], batch["context"], scale, jnp)
        if "residual" in params:
            p = params["residual"]
            h = jax.nn.gelu(batch["z"] @ p["w1"] + p["b1"])
            if key is not None and args.dropout:
                keep = 1 - args.dropout
                h = h * jax.random.bernoulli(key, keep, h.shape) / keep
            logits = logits + h @ p["w2"] + p["b2"]
        return logits

    @jax.jit
    def train_step(params, state, batch, key):
        def objective(p):
            logits = predict(p, batch, key)
            targets = jax.nn.one_hot(batch["y"], CLASSES)
            targets = targets * (1 - args.smoothing) + args.smoothing / CLASSES
            ce = optax.softmax_cross_entropy(logits, targets)
            loss = jnp.sum(ce * batch["mask"]) / jnp.maximum(batch["mask"].sum(), 1)
            acc = jnp.sum((logits.argmax(-1) == batch["y"]) * batch["mask"])
            return loss, acc
        (loss, correct), grads = jax.value_and_grad(objective, has_aux=True)(params)
        updates, state = tx.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss, correct

    return tx, train_step, jax.jit(predict)


def evaluate(predict, params, bank, ids, labels, variant, norm, batch_size):
    import jax
    import jax.numpy as jnp
    predictions = np.empty(len(ids), np.int16)
    nll = 0.0
    for start in range(0, len(ids), batch_size):
        ix = ids[start:start + batch_size]
        batch = bank.batch(ix, variant, labels, *norm, batch_size)
        logits = np.asarray(jax.device_get(predict(params, jax.tree.map(jnp.asarray, batch))))[:len(ix)]
        if not np.isfinite(logits).all():
            raise FloatingPointError("Nonfinite probe logits.")
        predictions[start:start + len(ix)] = logits.argmax(-1)
        shifted = logits.astype(np.float64) - logits.max(-1, keepdims=True)
        nll += float((np.log(np.exp(shifted).sum(-1)) - shifted[np.arange(len(ix)), labels[ix]]).sum())
    return dict(accuracy=float(np.mean(predictions == labels[ids])), nll=nll / len(ids)), predictions


def train_probe(args, bank, store, fit, dev, core, variant, seed, norm, scale):
    import jax
    import jax.numpy as jnp
    from flax import serialization

    root = Path(args.output) / f"seed_{seed}" / variant
    root.mkdir(parents=True, exist_ok=True)
    if (root / "fit_complete.json").exists():
        return root
    params = init_probe(core, len(norm[0]), hidden_sizes()[variant], seed)
    nparams = tree_count(params)
    tx, step, predict = make_steps(scale, args, math.ceil(len(fit) / args.batch_size))
    params = jax.tree.map(jnp.asarray, params)
    state = tx.init(params)
    last = root / "last.msgpack"
    history = []
    if last.exists():
        saved = serialization.msgpack_restore(last.read_bytes())
        params = jax.tree.map(jnp.asarray, saved["params"])
        state = serialization.from_state_dict(state, saved["optimizer"])
        best = saved["best_params"]
        best_loss, best_epoch, start_epoch = saved["best_loss"], saved["best_epoch"], saved["epoch"] + 1
        history = json.loads(saved["history_json"])
    else:
        dev_metrics, _ = evaluate(predict, params, bank, dev, store.y, variant, norm, args.batch_size)
        best, best_loss, best_epoch, start_epoch = jax.device_get(params), dev_metrics["nll"], 0, 1
        history.append(dict(epoch=0, dev=dev_metrics))
    np.savez(root / "normalization.npz", mean=norm[0], std=norm[1])
    for epoch in range(start_epoch, args.epochs + 1):
        order = np.random.default_rng(seed + 1000003 * epoch).permutation(fit)
        losses, correct, seen = 0.0, 0.0, 0
        total_batches = math.ceil(len(order) / args.batch_size)
        for bi, start in enumerate(range(0, len(order), args.batch_size)):
            ix = order[start:start + args.batch_size]
            batch = bank.batch(ix, variant, store.y, *norm, args.batch_size)
            key = jax.random.fold_in(jax.random.PRNGKey(seed), epoch * total_batches + bi)
            params, state, loss, acc = step(params, state, jax.tree.map(jnp.asarray, batch), key)
            loss, acc = float(loss), float(acc)
            if not np.isfinite(loss):
                raise FloatingPointError(f"{variant} seed={seed}: nonfinite training loss")
            losses += loss * len(ix)
            correct += acc
            seen += len(ix)
            if bi % 10 == 0 or bi + 1 == total_batches:
                report(args.output, args.protocol, f"{variant} s{seed}", bi + 1, total_batches,
                       epoch=epoch, train_acc=correct / seen, loss=losses / seen)
        dev_metrics, _ = evaluate(predict, params, bank, dev, store.y, variant, norm, args.batch_size)
        if dev_metrics["nll"] < best_loss - 1e-6:
            best, best_loss, best_epoch = jax.device_get(params), dev_metrics["nll"], epoch
        history.append(dict(epoch=epoch, train_accuracy=correct / seen,
                            train_loss=losses / seen, dev=dev_metrics))
        saved = dict(params=jax.device_get(params),
                     optimizer=serialization.to_state_dict(jax.device_get(state)),
                     best_params=best, best_loss=float(best_loss), best_epoch=int(best_epoch),
                     epoch=epoch, history_json=json.dumps(history))
        atomic_bytes(last, serialization.msgpack_serialize(saved))
        atomic_json(root / "history.json", history)
    # An epoch-zero selection is a valid negative result, never suppressed.
    atomic_bytes(root / "selected.msgpack", serialization.msgpack_serialize(best))
    atomic_json(root / "fit_complete.json", dict(
        variant=variant, seed=seed, trainable_parameters=nparams,
        residual_hidden_width=hidden_sizes()[variant], selected_epoch=int(best_epoch),
        selected_dev_nll=float(best_loss), completed_epochs=args.epochs,
        selection="minimum training-derived development NLL; epoch zero included",
        used_official_evaluation_for_selection=False))
    return root


def paired_metrics(reference, candidate, labels, groups, seed=937, replicates=1000):
    a, b = reference == labels, candidate == labels
    diff = b.astype(np.float64) - a.astype(np.float64)
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=diff)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(replicates, len(unique)))
    bootstrap = 100 * sums[sampled].sum(1) / counts[sampled].sum(1)
    return dict(gain_pp=100 * float(diff.mean()),
                fixed=int((~a & b).sum()), broken=int((a & ~b).sum()),
                group_bootstrap_95ci_pp=[float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
                resampling_groups=int(len(unique)), replicates=replicates,
                uncertainty_note="Exploratory cluster bootstrap; conditional on this backbone; not multiplicity-adjusted.")


def interpret(results):
    """Do not label a failure as a proven representation ceiling."""
    names = ("A_vs_base", "B_vs_A", "B_vs_base", "C_vs_B", "C_vs_base")
    stats = {}
    for name in names:
        items = [seed["comparisons"][name] for seed in results.values()]
        stats[name] = dict(mean_gain_pp=float(np.mean([x["gain_pp"] for x in items])),
                           min_gain_pp=float(min(x["gain_pp"] for x in items)),
                           all_seeds_positive=all(x["gain_pp"] > 0 for x in items),
                           all_cluster_intervals_above_zero=all(x["group_bootstrap_95ci_pp"][0] > 0 for x in items))
    messages = []
    if stats["A_vs_base"]["all_seeds_positive"]:
        messages.append("Ordinary head refitting improves this checkpoint; keep it as the adaptation control.")
    if stats["B_vs_A"]["all_seeds_positive"] and stats["B_vs_base"]["all_seeds_positive"]:
        messages.append("Descriptor-readout headroom is supported in this checkpoint; replicate with a clean selection protocol.")
    if stats["C_vs_B"]["all_seeds_positive"] and stats["C_vs_base"]["all_seeds_positive"]:
        messages.append("Extra pre-aggregation temporal contrasts help in this checkpoint; aggregation is a candidate limitation.")
    if not messages:
        messages.append("Inconclusive: these probes did not establish a consistent readout or aggregation improvement.")
    return dict(comparisons=stats, interpretation=messages,
                proven_single_bottleneck=False,
                negative_result_does_not_prove_information_absent=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, choices=("xsub", "xset"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[128, 28, 42])
    p.add_argument("--split-seed", type=int, default=20260925)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--extract-batch", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--smoothing", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.10)
    a = p.parse_args(argv)
    if a.epochs < 1 or min(a.batch_size, a.extract_batch) < 1:
        p.error("Epochs and batch sizes must be positive.")
    if not 0 <= a.dropout < 1 or not 0 <= a.smoothing < 1:
        p.error("Dropout and smoothing must be in [0,1).")
    if a.learning_rate <= 0 or a.weight_decay < 0 or len(set(a.seeds)) != len(a.seeds):
        p.error("Invalid optimizer settings or duplicate seeds.")
    return a


def main(argv=None):
    import jax
    import jax.numpy as jnp
    from flax import serialization

    args = parse_args(argv)
    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise RuntimeError(f"Expected one isolated GPU, got {jax.devices()}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    start_time = time.perf_counter()
    report(output, args.protocol, "Verify R4 checkpoint/cache", 0, 1)
    checkpoint_hash = sha256(checkpoint)
    payload = serialization.msgpack_restore(checkpoint.read_bytes())
    base, config = payload["ema_params"], payload["config"]
    if int(config["fast_rank"]) != 4 or int(config["head_rank"]) != 2 or tree_count(base) != EXPECTED_PARAMS:
        raise ValueError("This diagnostic requires the original Fast-R4/Head-R2 EMA checkpoint.")
    if int(config["model_dim"]) != DIM or int(config["spatial_dim"]) != 24:
        raise ValueError("Unexpected R4 dimensions.")
    if payload.get("protocol", args.protocol).lower() != args.protocol:
        raise ValueError("Checkpoint protocol mismatch.")
    store = CanonicalStore(args.cache, args.protocol)
    identity = dict(version=VERSION, script_sha256=sha256(__file__), args=vars(args),
                    checkpoint_sha256=checkpoint_hash, cache=store.identity())
    validate_identity(output / "run_identity.json", identity)
    if (output / "summary.json").exists():
        print("Completed result reused:", output / "summary.json", flush=True)
        report(output, args.protocol, "Done (reused)", 1, 1, done=True)
        return
    fit, dev, dev_groups = group_dev_split(store.train, store.y, store.groups, args.split_seed)
    np.savez(output / "probe_split.npz", fit=fit, dev=dev, official_evaluation=store.test)
    atomic_json(output / "protocol.json", dict(
        limitation=LIMITATION, fitting_samples=len(fit), development_samples=len(dev),
        development_groups=dev_groups, group_type="subject" if args.protocol == "xsub" else "setup",
        official_train_samples=len(store.train), official_evaluation_samples=len(store.test),
        probe_dev_seen_by_backbone=True, official_evaluation_used_for_probe_selection=False,
        pre_registered_variants=list(VARIANTS), probe_seeds=args.seeds))
    core = core_params(base)
    bank, extraction = extract_features(args, store, base, config, core, checkpoint_hash)
    del base
    # Baseline reproduction is an integrity gate, not a probe selection metric.
    # Run it before spending time fitting probes.
    test = store.test
    truth, groups = np.asarray(store.y[test]), store.groups[test]
    baseline = np.asarray(bank.arrays["logits"][test]).argmax(-1).astype(np.int16)
    base_accuracy = float(np.mean(baseline == truth))
    expected = payload.get("val_accuracy")
    if expected is not None and (not np.isfinite(expected) or abs(base_accuracy - float(expected)) > 2 / len(test)):
        raise RuntimeError(f"Clean baseline mismatch: reproduced {base_accuracy}, checkpoint {expected}")
    print(f"{args.protocol.upper()} baseline reproduced: {100 * base_accuracy:.4f}%", flush=True)
    # All fitting/selection completes before official probe scores are computed.
    norms = {}
    for variant in VARIANTS:
        report(output, args.protocol, f"Train-only normalization {variant}", 0, 1)
        norms[variant] = bank.normalization(fit, variant)
    for seed in args.seeds:
        for variant in VARIANTS:
            train_probe(args, bank, store, fit, dev, core, variant, seed,
                        norms[variant], config["head_residual_scale"])
    _, _, predict = make_steps(config["head_residual_scale"], args, math.ceil(len(fit) / args.batch_size))
    seed_results = {}
    for seed in args.seeds:
        records, predictions = {}, {"base": baseline}
        for vi, variant in enumerate(VARIANTS):
            report(output, args.protocol, f"Final evaluation {variant} s{seed}", vi, 3)
            root = output / f"seed_{seed}" / variant
            selected = serialization.msgpack_restore((root / "selected.msgpack").read_bytes())
            params = jax.tree.map(jnp.asarray, selected)
            evaluation, pred = evaluate(predict, params, bank, test, store.y, variant, norms[variant], args.batch_size)
            fit_metrics, _ = evaluate(predict, params, bank, fit, store.y, variant, norms[variant], args.batch_size)
            dev_metrics, _ = evaluate(predict, params, bank, dev, store.y, variant, norms[variant], args.batch_size)
            records[variant] = dict(json.loads((root / "fit_complete.json").read_text()),
                                   official_evaluation=evaluation, fit=fit_metrics, development=dev_metrics)
            predictions[variant] = pred
        comparisons = {}
        for name, left, right in (
                ("A_vs_base", "base", "A_control"), ("B_vs_base", "base", "B_descriptor"),
                ("C_vs_base", "base", "C_temporal"), ("B_vs_A", "A_control", "B_descriptor"),
                ("C_vs_B", "B_descriptor", "C_temporal")):
            comparisons[name] = paired_metrics(predictions[left], predictions[right], truth, groups)
        seed_results[str(seed)] = dict(variants=records, comparisons=comparisons)
        np.savez_compressed(output / f"seed_{seed}" / "predictions.npz",
                            sample_indices=test, labels=truth, groups=groups, **predictions)
    if sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("Source checkpoint changed during the diagnostic.")
    if store.identity() != identity["cache"]:
        raise RuntimeError("Source canonical cache changed during the diagnostic.")
    summary = dict(version=VERSION, protocol=args.protocol, limitation=LIMITATION,
                   checkpoint_sha256=checkpoint_hash, source_checkpoint_unchanged=True,
                   baseline_accuracy=base_accuracy, recorded_baseline_accuracy=expected,
                   extraction=extraction, seeds=seed_results, diagnosis=interpret(seed_results),
                   elapsed_seconds=time.perf_counter() - start_time,
                   note="No attention, LoRA, retrieval bank, or backbone updates are used in this diagnostic.")
    atomic_json(output / "summary.json", summary)
    report(output, args.protocol, "Done", 1, 1, done=True, val_acc=base_accuracy)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
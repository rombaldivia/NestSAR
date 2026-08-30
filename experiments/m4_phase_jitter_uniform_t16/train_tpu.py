#!/usr/bin/env python3
from __future__ import annotations

"""Phase-T16 + training-only segment-boundary jitter + fixed uniform fusion.

This experiment isolates a generalization change after the Phase-T16 audit:
  * keep the validated 15-channel phase representation;
  * keep spatial -> frame memory -> post-frame router -> chunk memory;
  * remove fusion_prior/fusion_controller and average the four stream logits;
  * precompute one canonical and one +/-1-frame boundary-jittered training view;
  * choose canonical vs jittered independently per sample each epoch;
  * validation/inference always use the canonical deterministic segmentation.

There is no inference-time augmentation and no consistency loss in this branch.
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization
from flax.training import train_state
import optax
from tqdm.auto import tqdm

from experiments.m4_motionpreserve_t16.jax10_compat import install as install_jax10_compat
install_jax10_compat()

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_motionpreserve_phase_t16 import train_phase_t16_tpu as phase

FRAMES = phase.FRAMES
PERSONS = phase.PERSONS
JOINTS = phase.JOINTS
TOKEN_CHANNELS = phase.TOKEN_CHANNELS
FEATURES = phase.FEATURES
NUM_CLASSES = phase.NUM_CLASSES
NUM_STREAMS = phase.NUM_STREAMS
EXPECTED_PARAMS = 1_816_130  # Phase-T16 minus fusion_controller (1796) and fusion_prior (4)
STREAM_NAMES = ("J", "B", "JM", "BM")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jittered_segment_bounds(total: int, n: int, max_shift: int, rng: np.random.Generator):
    """Jitter internal segment boundaries while keeping n non-empty segments."""
    canonical = base.segment_bounds(total, n)
    if total < n or max_shift <= 0:
        return canonical

    edges = np.asarray([canonical[0][0]] + [e for _, e in canonical], dtype=np.int64)
    base_edges = edges.copy()
    edges[0] = 0
    edges[-1] = total

    for i in range(1, n):
        shift = int(rng.integers(-max_shift, max_shift + 1))
        lo = int(edges[i - 1] + 1)
        hi = int(total - (n - i))
        edges[i] = int(np.clip(base_edges[i] + shift, lo, hi))

    return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]


def phase_tokens_from_bounds(keypoints: np.ndarray, bounds) -> np.ndarray:
    x = base.canonicalize_raw(keypoints)
    if x.shape[0] <= 0:
        return np.zeros((FRAMES, FEATURES), np.float32)

    tokens = np.zeros((FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS), np.float32)
    for i, (s, e) in enumerate(bounds):
        seg = x[s:e]
        pose = seg[(len(seg) - 1) // 2]
        if len(seg) >= 2:
            d = seg[1:] - seg[:-1]
            full_disp = np.sum(d, axis=0)
            cut = max(1, len(d) // 2)
            phase_a = np.sum(d[:cut], axis=0)
            phase_b = np.sum(d[cut:], axis=0) if cut < len(d) else np.zeros_like(full_disp)
            path = np.sum(np.abs(d), axis=0)
        else:
            full_disp = np.zeros_like(pose)
            phase_a = np.zeros_like(pose)
            phase_b = np.zeros_like(pose)
            path = np.zeros_like(pose)

        tokens[i, ..., 0:3] = pose
        tokens[i, ..., 3:6] = full_disp
        tokens[i, ..., 6:9] = phase_a
        tokens[i, ..., 9:12] = phase_b
        tokens[i, ..., 12:15] = path

    nz = np.abs(x) > 1e-8
    if np.any(nz):
        rms = float(np.sqrt(np.mean(np.square(x[nz]))) + 1e-6)
        tokens /= rms
    return np.nan_to_num(tokens).reshape(FRAMES, FEATURES).astype(np.float32)


def jitter_phase_tokens(keypoints: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    x = base.canonicalize_raw(keypoints)
    bounds = jittered_segment_bounds(x.shape[0], FRAMES, max_shift, rng)
    # phase_tokens_from_bounds canonicalizes again; duplicate work is preprocessing-only.
    return phase_tokens_from_bounds(keypoints, bounds)


class M4PhaseUniformT16(nn.Module):
    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Mapping[str, jnp.ndarray]:
        tok = x.reshape(x.shape[0], FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS)
        pose = tok[..., 0:3]
        full_disp = tok[..., 3:6]
        phase_a = tok[..., 6:9]
        phase_b = tok[..., 9:12]
        path = tok[..., 12:15]

        joint = pose
        parents = jnp.asarray(base.PARENTS)
        bone = joint - jnp.take(joint, parents, axis=3)

        joint_motion = jnp.concatenate([full_disp, phase_a, phase_b, path], axis=-1)
        parent_full = jnp.take(full_disp, parents, axis=3)
        parent_a = jnp.take(phase_a, parents, axis=3)
        parent_b = jnp.take(phase_b, parents, axis=3)
        parent_path = jnp.take(path, parents, axis=3)
        bone_motion = jnp.concatenate([
            full_disp - parent_full,
            phase_a - parent_a,
            phase_b - parent_b,
            jnp.abs(path - parent_path),
        ], axis=-1)

        raw_streams = (joint, bone, joint_motion, bone_motion)
        spatial = []
        for i, s in enumerate(raw_streams):
            spatial.append(base.SpatialEncoder(
                self.spatial_dim, self.model_dim, self.dropout, name=f"spatial_{i}"
            )(s, training))

        frame_streams = []
        for i, s in enumerate(spatial):
            frame_streams.append(base.BiMemory(
                self.model_dim, name=f"frame_memory_{i}"
            )(s))
        frame_stack = jnp.stack(frame_streams, axis=2)

        mixed, router_weights = base.CrossStreamRouter(
            self.model_dim, name="cross_stream_after_frame"
        )(frame_stack)

        descriptors = []
        stream_logits = []
        chunk_states = []
        for i in range(NUM_STREAMS):
            chunks, desc = base.DescriptorHead(
                self.model_dim, self.dropout, name=f"descriptor_{i}"
            )(mixed[:, :, i], training)
            descriptors.append(desc)
            chunk_states.append(chunks)
            stream_logits.append(nn.Dense(NUM_CLASSES, name=f"classifier_{i}")(desc))

        descs = jnp.stack(descriptors, axis=1)
        sl = jnp.stack(stream_logits, axis=1)

        # Audit result: learned fusion was slightly worse than a fixed mean.
        fusion = jnp.full((x.shape[0], NUM_STREAMS), 1.0 / NUM_STREAMS, dtype=sl.dtype)
        logits = jnp.mean(sl, axis=1)

        return {
            "logits": logits,
            "stream_logits": sl,
            "fusion_weights": fusion,
            "router_weights": router_weights,
            "spatial_stack": jnp.stack(spatial, axis=2),
            "frame_stack": frame_stack,
            "mixed_frame_stack": mixed,
            "descriptors": descs,
            "chunk_states": jnp.stack(chunk_states, axis=1),
        }


class State(train_state.TrainState):
    ema_params: Any


def count_params(params) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))


def smooth_ce(logits, labels, smoothing: float):
    return base.smooth_ce(logits, labels, smoothing)


def shard(x: np.ndarray, ndev: int) -> np.ndarray:
    return x.reshape(ndev, x.shape[0] // ndev, *x.shape[1:])


def resolve_protocol_ids(annotations, split, protocol: str):
    tk, vk = base.resolve_split(split, protocol)
    by_id = {
        base.sample_id(a, i): a
        for i, a in enumerate(annotations)
        if isinstance(a, Mapping)
    }
    train_ids = [str(v) for v in split[tk] if str(v) in by_id]
    val_ids = [str(v) for v in split[vk] if str(v) in by_id]
    if not train_ids or not val_ids:
        raise RuntimeError(f"Empty {protocol} split: train={len(train_ids)} val={len(val_ids)}")
    return by_id, train_ids, val_ids


def build_protocol_views(annotations, split, protocol: str, max_shift: int, seed: int,
                         max_train: int = 0, max_val: int = 0):
    by_id, train_ids, val_ids = resolve_protocol_ids(annotations, split, protocol)
    if max_train:
        train_ids = train_ids[:max_train]
    if max_val:
        val_ids = val_ids[:max_val]

    Xcan = np.empty((len(train_ids), FRAMES, FEATURES), np.float32)
    Xjit = np.empty_like(Xcan)
    ytr = np.empty((len(train_ids),), np.int32)

    for i, sid in enumerate(tqdm(train_ids, desc=f"{protocol.upper()} train canonical+jitter", mininterval=0.5)):
        a = by_id[sid]
        kp = base.annotation_keypoints(a)
        Xcan[i] = phase.segment_phase_tokens(kp)
        rng = np.random.default_rng(np.random.SeedSequence([seed, i, 9173]))
        Xjit[i] = jitter_phase_tokens(kp, max_shift, rng)
        ytr[i] = base.annotation_label(a)

    Xva = np.empty((len(val_ids), FRAMES, FEATURES), np.float32)
    yva = np.empty((len(val_ids),), np.int32)
    for i, sid in enumerate(tqdm(val_ids, desc=f"{protocol.upper()} val canonical", mininterval=0.5)):
        a = by_id[sid]
        Xva[i] = phase.segment_phase_tokens(base.annotation_keypoints(a))
        yva[i] = base.annotation_label(a)

    changed = float(np.mean(np.any(np.abs(Xcan - Xjit) > 1e-7, axis=(1, 2))))
    log(f"{protocol.upper()} jitter view differs from canonical for {100*changed:.2f}% of train samples")
    return Xcan, Xjit, ytr, Xva, yva


def iter_train_views(Xcan, Xjit, y, global_batch: int, seed: int, jitter_prob: float):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    usable = (len(idx) // global_batch) * global_batch
    idx = idx[:usable]
    for s in range(0, usable, global_batch):
        ii = idx[s:s + global_batch]
        xb = Xcan[ii].copy()
        use_jitter = rng.random(len(ii)) < jitter_prob
        if np.any(use_jitter):
            xb[use_jitter] = Xjit[ii[use_jitter]]
        yield xb, y[ii]


def iter_eval(X, y, global_batch: int):
    return base.iter_eval(X, y, global_batch)


def audit_flops(model, params):
    try:
        dummy = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
        fn = jax.jit(lambda p, xx: model.apply({"params": p}, xx, training=False)["logits"])
        compiled = fn.lower(params, dummy).compile()
        ca = compiled.cost_analysis()
        if isinstance(ca, list):
            ca = ca[0] if ca else {}
        flops = float(ca.get("flops", float("nan")))
        log(f"XLA inference audit: {flops:,.0f} FLOPs/clip = {flops / 1e9:.9f} GFLOPs")
        return flops
    except Exception as exc:
        log(f"GFLOPs audit unavailable: {exc}")
        return None


def train_protocol(args, annotations, split, protocol: str):
    devices = list(jax.local_devices())
    ndev = len(devices)
    if args.batch_size % ndev or args.eval_batch_size % ndev:
        raise ValueError("Global train/eval batches must be divisible by TPU device count")

    protocol_seed = args.seed + (0 if protocol == "xsub" else 100000)
    Xcan, Xjit, ytr, Xva, yva = build_protocol_views(
        annotations, split, protocol, args.jitter_max_shift, protocol_seed,
        args.max_train_samples, args.max_val_samples,
    )

    steps_per_epoch = len(ytr) // args.batch_size
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_fraction))
    lr = optax.warmup_cosine_decay_schedule(
        0.0, args.learning_rate, warmup, total_steps, end_value=args.min_learning_rate
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(lr, weight_decay=args.weight_decay),
    )

    model = M4PhaseUniformT16(args.spatial_dim, args.model_dim, args.dropout)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    dummy = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    params = model.init({"params": init_key, "dropout": init_key}, dummy, training=False)["params"]
    nparams = count_params(params)
    log(f"{protocol.upper()} params={nparams:,}")
    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(f"Parameter mismatch: got {nparams:,}, expected {EXPECTED_PARAMS:,}")
    if args.audit_first:
        audit_flops(model, params)

    state = State.create(apply_fn=model.apply, params=params, tx=tx, ema_params=params)
    state = jax.device_put_replicated(state, devices)
    rngs = jax.random.split(key, ndev)

    @partial(jax.pmap, axis_name="d", devices=devices)
    def p_train_step(state, rng, xb, yb):
        rng, drop = jax.random.split(rng)
        def loss_fn(p):
            out = model.apply({"params": p}, xb, training=True, rngs={"dropout": drop})
            main = jnp.mean(smooth_ce(out["logits"], yb, args.label_smoothing))
            sl = out["stream_logits"]
            aux = jnp.mean(smooth_ce(
                sl.reshape(-1, NUM_CLASSES), jnp.repeat(yb, NUM_STREAMS), args.label_smoothing
            ))
            loss = main + args.stream_aux_weight * aux
            acc = jnp.mean(jnp.argmax(out["logits"], axis=-1) == yb)
            return loss, (main, aux, acc)
        (loss, (main, aux, acc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, "d")
        loss, main, aux, acc = jax.lax.pmean((loss, main, aux, acc), "d")
        state = state.apply_gradients(grads=grads)
        ema = jax.tree_util.tree_map(
            lambda e, p: args.ema_decay * e + (1.0 - args.ema_decay) * p,
            state.ema_params, state.params,
        )
        state = state.replace(ema_params=ema)
        return state, rng, (loss, main, aux, acc)

    @partial(jax.pmap, axis_name="d", devices=devices)
    def p_eval_step(ema_params, xb, yb, mask):
        out = model.apply({"params": ema_params}, xb, training=False)
        ce = smooth_ce(out["logits"], yb, 0.0)
        pred = jnp.argmax(out["logits"], axis=-1)
        correct = jnp.sum((pred == yb).astype(jnp.float32) * mask)
        loss_sum = jnp.sum(ce * mask)
        count = jnp.sum(mask)
        return jax.lax.psum(jnp.asarray([loss_sum, correct, count]), "d")

    outdir = Path(args.outdir) / protocol
    outdir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    best_epoch = 0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss_sum = acc_sum = nstep = 0.0
        bar = tqdm(
            iter_train_views(Xcan, Xjit, ytr, args.batch_size, args.seed + epoch, args.jitter_prob),
            total=steps_per_epoch,
            desc=f"{protocol.upper()} TRAIN E{epoch:03d}/{args.epochs}",
            mininterval=0.5,
        )
        for xb, yb in bar:
            state, rngs, metrics = p_train_step(state, rngs, shard(xb, ndev), shard(yb, ndev))
            loss_v, _, _, acc_v = [float(np.asarray(v[0])) for v in jax.device_get(metrics)]
            loss_sum += loss_v; acc_sum += acc_v; nstep += 1
            if nstep % args.progress_every == 0:
                bar.set_postfix(loss=f"{loss_sum/nstep:.3f}", acc=f"{100*acc_sum/nstep:.2f}%", best=f"{100*best:.2f}%")

        eval_loss = eval_correct = eval_count = 0.0
        for xb, yb, mask in tqdm(iter_eval(Xva, yva, args.eval_batch_size),
                                  desc=f"{protocol.upper()} VAL   E{epoch:03d}/{args.epochs}",
                                  leave=False, mininterval=0.5):
            sums = p_eval_step(state.ema_params, shard(xb, ndev), shard(yb, ndev), shard(mask, ndev))
            vals = np.asarray(jax.device_get(sums[0]))
            eval_loss += float(vals[0]); eval_correct += float(vals[1]); eval_count += float(vals[2])

        val_acc = eval_correct / max(eval_count, 1.0)
        val_loss = eval_loss / max(eval_count, 1.0)
        train_acc = acc_sum / max(nstep, 1.0)
        log(f"{protocol.upper()} E{epoch:03d} train={100*train_acc:.3f}% val={100*val_acc:.3f}% loss={val_loss:.4f} time={time.time()-t0:.1f}s")

        if val_acc > best + 1e-6:
            best = val_acc; best_epoch = epoch; stale = 0
            single = jax.tree_util.tree_map(lambda z: jax.device_get(z[0]), state)
            payload = {
                "model": "M4PhaseJitterUniformT16",
                "protocol": protocol,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "params": single.params,
                "ema_params": single.ema_params,
                "opt_state": single.opt_state,
                "step": single.step,
                "config": vars(args),
                "representation": {
                    "frames": FRAMES,
                    "token_channels": TOKEN_CHANNELS,
                    "features_per_token": FEATURES,
                    "jitter_max_shift": args.jitter_max_shift,
                    "jitter_prob": args.jitter_prob,
                    "final_fusion": "uniform_mean",
                },
            }
            (outdir / "best.msgpack").write_bytes(serialization.to_bytes(payload))
            (outdir / "best.json").write_text(json.dumps({
                "epoch": epoch,
                "val_accuracy": val_acc,
                "params": count_params(single.params),
                "frames": FRAMES,
                "token_channels": TOKEN_CHANNELS,
                "jitter_max_shift": args.jitter_max_shift,
                "jitter_prob": args.jitter_prob,
                "final_fusion": "uniform_mean",
            }, indent=2))
        else:
            stale += 1

        if stale >= args.patience:
            log(f"{protocol.upper()} early stop: best={100*best:.3f}% @ E{best_epoch}")
            break

    del Xcan, Xjit, ytr, Xva, yva
    return best, best_epoch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument("--protocol", choices=["xsub", "xset", "both"], default="both")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=6e-4)
    p.add_argument("--min-learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.08)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--stream-aux-weight", type=float, default=0.15)
    p.add_argument("--spatial-dim", type=int, default=24)
    p.add_argument("--model-dim", type=int, default=112)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--jitter-max-shift", type=int, default=1)
    p.add_argument("--jitter-prob", type=float, default=0.50)
    p.add_argument("--outdir", default="/kaggle/working/NestSAR_M4_Phase_JitterUniform_T16_TPU")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--audit-first", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    log(f"JAX={jax.__version__} backend={jax.default_backend()} devices={jax.devices()}")
    if jax.default_backend() != "tpu" or jax.local_device_count() != 8:
        raise RuntimeError(f"Expected one TPU v5e-8; backend={jax.default_backend()} local_devices={jax.local_device_count()}")
    if not (0.0 <= args.jitter_prob <= 1.0):
        raise ValueError("--jitter-prob must be in [0,1]")

    log("Experiment: Phase-T16 + training-only segment jitter + fixed uniform final fusion")
    log(f"T16 | features/token={FEATURES} | jitter=+/-{args.jitter_max_shift} raw frame | jitter_prob={args.jitter_prob:.2f}")
    log("Inference: canonical segmentation only; no test-time augmentation")

    dataset = base.find_dataset(args.dataset)
    log(f"Dataset={dataset}")
    anns, split = base.load_ntu(dataset)
    protocols = ["xsub", "xset"] if args.protocol == "both" else [args.protocol]
    summary = {}
    for pr in protocols:
        best, ep = train_protocol(args, anns, split, pr)
        summary[pr] = {"best_val_accuracy": best, "best_epoch": ep}
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE {summary}")


if __name__ == "__main__":
    main()

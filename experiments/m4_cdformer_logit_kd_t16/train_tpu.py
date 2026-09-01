#!/usr/bin/env python3
from __future__ import annotations

"""Current T16 champion + sample-wise CD-Former JAX logit KD.

Teacher logits are precomputed once with the exact MMAction2KeypointDataset path.
The frozen teacher is not part of student training/inference compute after caching.
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
import optax
from tqdm.auto import tqdm

from experiments.m4_motionpreserve_t16.jax10_compat import install as install_jax10_compat
install_jax10_compat()

from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_t16 import train_tpu as cons

FRAMES = ju.FRAMES
FEATURES = ju.FEATURES
NUM_CLASSES = ju.NUM_CLASSES
NUM_STREAMS = ju.NUM_STREAMS
EXPECTED_PARAMS = ju.EXPECTED_PARAMS
DEFAULT_CACHE = Path("/kaggle/working/cdformer16_mmaction2_teacher_logits.npz")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kd_kl(student_logits, teacher_logits, temperature: float):
    t = jnp.asarray(temperature, dtype=student_logits.dtype)
    s_logp = jax.nn.log_softmax(student_logits / t, axis=-1)
    t_logp = jax.nn.log_softmax(teacher_logits / t, axis=-1)
    t_prob = jnp.exp(t_logp)
    kl = jnp.sum(t_prob * (t_logp - s_logp), axis=-1)
    return jnp.mean(kl) * (t * t)


def iter_train_kd(Xcan, Xjit, y, teacher_logits, global_batch: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    usable = (len(idx) // global_batch) * global_batch
    idx = idx[:usable]
    for s in range(0, usable, global_batch):
        ii = idx[s:s + global_batch]
        yield Xcan[ii], Xjit[ii], y[ii], teacher_logits[ii]


def load_teacher_targets(cache_path: Path, annotations, split, protocol: str, max_train: int):
    z = np.load(cache_path, allow_pickle=False)
    for key in ("ids", "logits", "labels"):
        if key not in z.files:
            raise KeyError(f"Teacher cache missing {key}")
    ids = np.asarray(z["ids"]).astype(str)
    logits = np.asarray(z["logits"], dtype=np.float32)
    labels = np.asarray(z["labels"], dtype=np.int32)
    if logits.shape != (len(ids), 120):
        raise RuntimeError(f"Bad teacher logits shape {logits.shape}")
    if len(set(ids.tolist())) != len(ids):
        raise RuntimeError("Teacher IDs are not unique")
    index = {sid: i for i, sid in enumerate(ids.tolist())}

    by_id, train_ids, _ = ju.resolve_protocol_ids(annotations, split, protocol)
    if max_train:
        train_ids = train_ids[:max_train]

    missing = [sid for sid in train_ids if sid not in index]
    if missing:
        raise RuntimeError(f"Teacher cache missing {len(missing)} {protocol} train IDs")

    pos = np.asarray([index[sid] for sid in train_ids], np.int64)
    out = logits[pos]
    out_labels = labels[pos]
    expected = np.asarray([ju.base.annotation_label(by_id[sid]) for sid in train_ids], np.int32)
    if not np.array_equal(out_labels, expected):
        bad = np.flatnonzero(out_labels != expected)[:10]
        raise RuntimeError(f"Teacher/student label mismatch at {bad.tolist()}")
    if not np.all(np.isfinite(out)):
        raise RuntimeError("Teacher logits contain non-finite values")
    return out, out_labels


def train_protocol(args, annotations, split, protocol: str):
    devices = list(jax.local_devices())
    ndev = len(devices)
    if args.batch_size % ndev or args.eval_batch_size % ndev:
        raise ValueError("Global train/eval batches must be divisible by TPU device count")

    protocol_seed = args.seed + (0 if protocol == "xsub" else 100000)
    Xcan, Xjit, ytr, Xva, yva = ju.build_protocol_views(
        annotations, split, protocol, args.jitter_max_shift, protocol_seed,
        args.max_train_samples, args.max_val_samples,
    )
    teacher_logits, teacher_labels = load_teacher_targets(
        Path(args.teacher_cache), annotations, split, protocol, args.max_train_samples
    )
    if len(teacher_logits) != len(ytr):
        raise RuntimeError(f"Teacher/student train length mismatch: {len(teacher_logits)} vs {len(ytr)}")
    if not np.array_equal(teacher_labels, ytr.astype(np.int32)):
        raise RuntimeError("Teacher/student training labels are not aligned")
    teacher_train_acc = float(np.mean(np.argmax(teacher_logits, axis=1) == ytr))
    log(f"{protocol.upper()} teacher train top1={100*teacher_train_acc:.3f}% targets={teacher_logits.shape}")

    steps_per_epoch = len(ytr) // args.batch_size
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_fraction))
    lr = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.learning_rate, warmup_steps=warmup,
        decay_steps=total_steps, end_value=args.min_learning_rate,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(lr, weight_decay=args.weight_decay),
    )

    model = ju.M4PhaseUniformT16(args.spatial_dim, args.model_dim, args.dropout)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    dummy = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    params = model.init({"params": init_key, "dropout": init_key}, dummy, training=False)["params"]
    nparams = ju.count_params(params)
    log(f"{protocol.upper()} student params={nparams:,}")
    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(f"Parameter mismatch: got {nparams:,}, expected {EXPECTED_PARAMS:,}")
    if args.audit_first:
        ju.audit_flops(model, params)

    # Loss-scale/gradient audit on the first real batch before full training.
    audit_n = min(args.batch_size, len(ytr))
    xa = jnp.asarray(Xcan[:audit_n])
    ya = jnp.asarray(ytr[:audit_n])
    ta = jnp.asarray(teacher_logits[:audit_n])
    audit_key = jax.random.PRNGKey(args.seed + 777)

    def audit_loss(p):
        out = model.apply({"params": p}, xa, training=True, rngs={"dropout": audit_key})
        ce = jnp.mean(ju.smooth_ce(out["logits"], ya, args.label_smoothing))
        kd = kd_kl(out["logits"], ta, args.kd_temperature)
        return ce + args.kd_weight * kd, (ce, kd)

    (audit_total, (audit_ce, audit_kd)), audit_grad = jax.value_and_grad(audit_loss, has_aux=True)(params)
    grad_norm = float(optax.global_norm(audit_grad))
    log(
        f"KD AUDIT ce={float(audit_ce):.6f} kd={float(audit_kd):.6f} "
        f"lambda*kd={args.kd_weight*float(audit_kd):.6f} total={float(audit_total):.6f} grad_norm={grad_norm:.6f}"
    )
    if not np.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError("KD gradient audit failed")

    state = ju.State.create(apply_fn=model.apply, params=params, tx=tx, ema_params=params)
    state = jax.device_put_replicated(state, devices)
    rngs = jax.random.split(key, ndev)

    @partial(jax.pmap, axis_name="d", devices=devices)
    def p_train_step(state, rng, xcan, xjit, yb, tlogits):
        rng, drop_can, drop_jit = jax.random.split(rng, 3)

        def loss_fn(p):
            out_can = model.apply({"params": p}, xcan, training=True, rngs={"dropout": drop_can})
            out_jit = model.apply({"params": p}, xjit, training=True, rngs={"dropout": drop_jit})

            ce_can = jnp.mean(ju.smooth_ce(out_can["logits"], yb, args.label_smoothing))
            ce_jit = jnp.mean(ju.smooth_ce(out_jit["logits"], yb, args.label_smoothing))
            main = 0.5 * (ce_can + ce_jit)

            sl_can = out_can["stream_logits"]
            sl_jit = out_jit["stream_logits"]
            labels4 = jnp.repeat(yb, NUM_STREAMS)
            aux_can = jnp.mean(ju.smooth_ce(sl_can.reshape(-1, NUM_CLASSES), labels4, args.label_smoothing))
            aux_jit = jnp.mean(ju.smooth_ce(sl_jit.reshape(-1, NUM_CLASSES), labels4, args.label_smoothing))
            aux = 0.5 * (aux_can + aux_jit)

            consistency = cons.symmetric_kl(out_can["logits"], out_jit["logits"], args.consistency_temperature)
            kd_can = kd_kl(out_can["logits"], tlogits, args.kd_temperature)
            kd_jit = kd_kl(out_jit["logits"], tlogits, args.kd_temperature)
            kd = 0.5 * (kd_can + kd_jit)

            loss = (
                main
                + args.stream_aux_weight * aux
                + args.consistency_weight * consistency
                + args.kd_weight * kd
            )

            acc_can = jnp.mean(jnp.argmax(out_can["logits"], axis=-1) == yb)
            acc_jit = jnp.mean(jnp.argmax(out_jit["logits"], axis=-1) == yb)
            agreement = jnp.mean(jnp.argmax(out_can["logits"], axis=-1) == jnp.argmax(out_jit["logits"], axis=-1))
            return loss, (main, aux, consistency, kd, acc_can, acc_jit, agreement)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, "d")
        loss = jax.lax.pmean(loss, "d")
        metrics = jax.lax.pmean(metrics, "d")
        state = state.apply_gradients(grads=grads)
        ema = jax.tree_util.tree_map(
            lambda e, p: args.ema_decay * e + (1.0 - args.ema_decay) * p,
            state.ema_params, state.params,
        )
        state = state.replace(ema_params=ema)
        return state, rng, (loss, *metrics)

    @partial(jax.pmap, axis_name="d", devices=devices)
    def p_eval_step(ema_params, xb, yb, mask):
        out = model.apply({"params": ema_params}, xb, training=False)
        ce = ju.smooth_ce(out["logits"], yb, 0.0)
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
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        sums = np.zeros(8, np.float64)  # loss main aux cons kd can jit agree
        nstep = 0
        bar = tqdm(
            iter_train_kd(Xcan, Xjit, ytr, teacher_logits, args.batch_size, args.seed + epoch),
            total=steps_per_epoch,
            desc=f"{protocol.upper()} LOGITKD TRAIN E{epoch:03d}/{args.epochs}",
            mininterval=0.5,
        )
        for xcan, xjit, yb, tlog in bar:
            xcan = ju.shard(xcan, ndev)
            xjit = ju.shard(xjit, ndev)
            yb = ju.shard(yb, ndev)
            tlog = ju.shard(tlog, ndev)
            state, rngs, metrics = p_train_step(state, rngs, xcan, xjit, yb, tlog)
            vals = [float(np.asarray(v[0])) for v in jax.device_get(metrics)]
            sums += np.asarray(vals, np.float64)
            nstep += 1
            if nstep % args.progress_every == 0:
                mean = sums / nstep
                bar.set_postfix(
                    loss=f"{mean[0]:.3f}", kd=f"{mean[4]:.3f}",
                    cons=f"{mean[3]:.4f}", can=f"{100*mean[5]:.2f}%",
                    jit=f"{100*mean[6]:.2f}%", agr=f"{100*mean[7]:.1f}%",
                    best=f"{100*best:.2f}%",
                )

        eval_loss = eval_correct = eval_count = 0.0
        for xb, yb, mask in tqdm(
            ju.iter_eval(Xva, yva, args.eval_batch_size),
            desc=f"{protocol.upper()} VAL E{epoch:03d}/{args.epochs}",
            leave=False, mininterval=0.5,
        ):
            xb = ju.shard(xb, ndev)
            yb = ju.shard(yb, ndev)
            mask = ju.shard(mask, ndev)
            vals = np.asarray(jax.device_get(p_eval_step(state.ema_params, xb, yb, mask)[0]))
            eval_loss += float(vals[0]); eval_correct += float(vals[1]); eval_count += float(vals[2])

        val_acc = eval_correct / max(eval_count, 1.0)
        val_loss = eval_loss / max(eval_count, 1.0)
        mean = sums / max(nstep, 1)
        row = {
            "epoch": epoch, "train_loss": float(mean[0]), "main_ce": float(mean[1]),
            "aux_ce": float(mean[2]), "consistency": float(mean[3]), "kd": float(mean[4]),
            "train_can_accuracy": float(mean[5]), "train_jit_accuracy": float(mean[6]),
            "agreement": float(mean[7]), "val_accuracy": float(val_acc), "val_loss": float(val_loss),
            "epoch_seconds": float(time.time() - t0),
        }
        history.append(row)
        (outdir / "history.json").write_text(json.dumps(history, indent=2))
        log(
            f"{protocol.upper()} E{epoch:03d} can={100*mean[5]:.3f}% jit={100*mean[6]:.3f}% "
            f"agree={100*mean[7]:.2f}% cons={mean[3]:.5f} kd={mean[4]:.5f} "
            f"val={100*val_acc:.3f}% loss={val_loss:.4f} time={row['epoch_seconds']:.1f}s"
        )

        if val_acc > best + 1e-6:
            best = val_acc; best_epoch = epoch; stale = 0
            single = jax.tree_util.tree_map(lambda z: jax.device_get(z[0]), state)
            payload = {
                "model": "M4PhaseJitterConsistencyCDFormerLogitKD_T16",
                "protocol": protocol, "epoch": epoch, "val_accuracy": val_acc,
                "params": single.params, "ema_params": single.ema_params,
                "opt_state": single.opt_state, "step": single.step,
                "config": vars(args),
                "teacher": {
                    "cache": str(Path(args.teacher_cache).resolve()),
                    "preprocessing": "MMAction2KeypointDataset 16f deterministic",
                    "temperature": args.kd_temperature,
                    "kd_weight": args.kd_weight,
                },
                "inference_extra_params": 0,
                "inference_extra_flops": 0,
            }
            (outdir / "best.msgpack").write_bytes(serialization.to_bytes(payload))
            (outdir / "best.json").write_text(json.dumps({
                "epoch": epoch, "val_accuracy": val_acc,
                "params": ju.count_params(single.params), "frames": FRAMES,
                "kd_weight": args.kd_weight, "kd_temperature": args.kd_temperature,
                "inference_extra_params": 0, "inference_extra_flops": 0,
            }, indent=2))
        else:
            stale += 1

        if stale >= args.patience:
            log(f"{protocol.upper()} early stop: best={100*best:.3f}% @ E{best_epoch}")
            break

    del Xcan, Xjit, ytr, Xva, yva, teacher_logits
    return best, best_epoch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument("--teacher-cache", default=str(DEFAULT_CACHE))
    p.add_argument("--protocol", choices=["xsub", "xset"], default="xsub")
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
    p.add_argument("--consistency-weight", type=float, default=0.08)
    p.add_argument("--consistency-temperature", type=float, default=1.0)
    p.add_argument("--kd-weight", type=float, default=0.20)
    p.add_argument("--kd-temperature", type=float, default=4.0)
    p.add_argument("--outdir", default="/kaggle/working/NestSAR_M4_CDFormer_LogitKD_T16_TPU")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--audit-first", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    log(f"JAX={jax.__version__} backend={jax.default_backend()} devices={jax.devices()}")
    if jax.default_backend() != "tpu" or jax.local_device_count() != 8:
        raise RuntimeError(f"Expected TPU8, got backend={jax.default_backend()} devices={jax.local_device_count()}")
    if not Path(args.teacher_cache).is_file():
        raise FileNotFoundError(args.teacher_cache)
    if args.kd_weight < 0 or args.kd_temperature <= 0:
        raise ValueError("Invalid KD configuration")

    log(
        f"Experiment: Phase+Jitter+Consistency T16 + CD-Former sample-wise logit KD | "
        f"KD={args.kd_weight:.3f} T={args.kd_temperature:.2f}"
    )
    log("Teacher logits: precomputed with deterministic MMAction2KeypointDataset")
    log("Inference architecture remains unchanged")

    dataset = ju.base.find_dataset(args.dataset)
    log(f"Dataset={dataset}")
    anns, split = ju.base.load_ntu(dataset)
    best, ep = train_protocol(args, anns, split, args.protocol)
    summary = {args.protocol: {"best_val_accuracy": best, "best_epoch": ep}}
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE {summary}")


if __name__ == "__main__":
    main()

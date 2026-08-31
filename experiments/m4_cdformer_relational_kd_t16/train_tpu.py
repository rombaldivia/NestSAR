#!/usr/bin/env python3
from __future__ import annotations

"""Phase+Jitter+Consistency T16 + CD-Former classifier-relation KD.

This is a controlled extension of the current T16 champion:
  * same Phase15 representation;
  * same fixed uniform fusion;
  * same canonical/jitter symmetric-KL consistency;
  * same inference graph and parameter count;
  * adds only a TRAINING loss that aligns each NestSAR stream classifier's
    class geometry with the validated CD-Former classifier geometry.

No CD-Former forward pass is used.  Therefore no CD-Former input preprocessing is
needed for this experiment and there are zero teacher parameters/FLOPs at inference.
"""

import argparse
import hashlib
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
EXPECTED_TEACHER_PARAMS = 11_284_344
DEFAULT_TEACHER = Path(
    "/kaggle/input/models/romelbaldivia/cdformer-jax/jax/default/1/"
    "cdformer16_teacher_jax.npz"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_teacher_relation(path: str | Path) -> tuple[jax.Array, dict]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CD-Former JAX NPZ not found: {path}")

    z = np.load(path, allow_pickle=False)
    files = list(z.files)
    total = int(sum(np.asarray(z[k]).size for k in files))
    if total != EXPECTED_TEACHER_PARAMS:
        raise RuntimeError(
            f"Teacher parameter guard failed: {total:,} != {EXPECTED_TEACHER_PARAMS:,}"
        )
    if "head.weight" not in files:
        raise KeyError("Validated teacher NPZ has no head.weight")

    W = np.asarray(z["head.weight"], dtype=np.float32)
    if W.shape != (120, 192):
        raise RuntimeError(f"Expected head.weight=(120,192), got {W.shape}")
    if not np.isfinite(W).all():
        raise RuntimeError("Teacher head.weight contains non-finite values")

    W64 = W.astype(np.float64)
    W64 /= np.maximum(np.linalg.norm(W64, axis=1, keepdims=True), 1e-12)
    G = (W64 @ W64.T).astype(np.float32)

    if not np.allclose(G, G.T, atol=1e-6):
        raise RuntimeError("Teacher class relation is not symmetric")
    if not np.allclose(np.diag(G), 1.0, atol=1e-5):
        raise RuntimeError("Teacher class relation diagonal is not ~1")

    meta = {
        "path": str(path),
        "sha256": sha256_file(path),
        "tensors": len(files),
        "parameters": total,
        "head_shape": list(W.shape),
    }
    return jnp.asarray(G, dtype=jnp.float32), meta


def relation_from_kernel(kernel: jax.Array) -> jax.Array:
    # Flax Dense kernel: [D,C] -> class vectors [C,D].
    W = jnp.asarray(kernel, dtype=jnp.float32).T
    W = W / (jnp.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
    return W @ W.T


def offdiag_mse(a: jax.Array, b: jax.Array) -> jax.Array:
    mask = 1.0 - jnp.eye(a.shape[0], dtype=jnp.float32)
    return jnp.sum(jnp.square(a - b) * mask) / jnp.sum(mask)


def relational_loss(params, teacher_relation: jax.Array) -> jax.Array:
    losses = []
    for i in range(NUM_STREAMS):
        kernel = params[f"classifier_{i}"]["kernel"]
        losses.append(offdiag_mse(relation_from_kernel(kernel), teacher_relation))
    return jnp.mean(jnp.stack(losses))


def iter_train_pairs(Xcan, Xjit, y, global_batch: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    usable = (len(idx) // global_batch) * global_batch
    idx = idx[:usable]
    for s in range(0, usable, global_batch):
        ii = idx[s:s + global_batch]
        yield Xcan[ii], Xjit[ii], y[ii]


def train_protocol(args, annotations, split, protocol: str, teacher_relation, teacher_meta):
    devices = list(jax.local_devices())
    ndev = len(devices)
    if args.batch_size % ndev:
        raise ValueError(f"Global batch {args.batch_size} must be divisible by {ndev}")
    if args.eval_batch_size % ndev:
        raise ValueError(f"Eval batch {args.eval_batch_size} must be divisible by {ndev}")

    protocol_seed = args.seed + (0 if protocol == "xsub" else 100000)
    Xcan, Xjit, ytr, Xva, yva = ju.build_protocol_views(
        annotations, split, protocol, args.jitter_max_shift, protocol_seed,
        args.max_train_samples, args.max_val_samples,
    )

    steps_per_epoch = len(ytr) // args.batch_size
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_fraction))
    lr = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=warmup,
        decay_steps=total_steps,
        end_value=args.min_learning_rate,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(lr, weight_decay=args.weight_decay),
    )

    model = ju.M4PhaseUniformT16(args.spatial_dim, args.model_dim, args.dropout)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    dummy = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    params = model.init(
        {"params": init_key, "dropout": init_key}, dummy, training=False
    )["params"]
    nparams = ju.count_params(params)
    log(f"{protocol.upper()} student params={nparams:,}")
    if nparams != EXPECTED_PARAMS:
        raise RuntimeError(f"Student parameter mismatch: {nparams:,} != {EXPECTED_PARAMS:,}")
    initial_rel = float(relational_loss(params, teacher_relation))
    log(f"{protocol.upper()} initial CD-rel={initial_rel:.8f}")
    if args.audit_first:
        ju.audit_flops(model, params)

    state = ju.State.create(apply_fn=model.apply, params=params, tx=tx, ema_params=params)
    state = jax.device_put_replicated(state, devices)
    rngs = jax.random.split(key, ndev)

    # Replicate the fixed 120x120 teacher target across TPU cores.
    teacher_rel_rep = jnp.broadcast_to(teacher_relation, (ndev,) + teacher_relation.shape)

    @partial(jax.pmap, axis_name="d", devices=devices)
    def p_train_step(state, rng, xcan, xjit, yb, gteacher):
        rng, drop_can, drop_jit = jax.random.split(rng, 3)

        def loss_fn(p):
            out_can = model.apply(
                {"params": p}, xcan, training=True, rngs={"dropout": drop_can}
            )
            out_jit = model.apply(
                {"params": p}, xjit, training=True, rngs={"dropout": drop_jit}
            )

            ce_can = jnp.mean(ju.smooth_ce(out_can["logits"], yb, args.label_smoothing))
            ce_jit = jnp.mean(ju.smooth_ce(out_jit["logits"], yb, args.label_smoothing))
            main = 0.5 * (ce_can + ce_jit)

            sl_can = out_can["stream_logits"]
            sl_jit = out_jit["stream_logits"]
            labels4 = jnp.repeat(yb, NUM_STREAMS)
            aux_can = jnp.mean(ju.smooth_ce(
                sl_can.reshape(-1, NUM_CLASSES), labels4, args.label_smoothing
            ))
            aux_jit = jnp.mean(ju.smooth_ce(
                sl_jit.reshape(-1, NUM_CLASSES), labels4, args.label_smoothing
            ))
            aux = 0.5 * (aux_can + aux_jit)

            consistency = cons.symmetric_kl(
                out_can["logits"], out_jit["logits"], args.consistency_temperature
            )
            rel = relational_loss(p, gteacher)

            loss = (
                main
                + args.stream_aux_weight * aux
                + args.consistency_weight * consistency
                + args.rel_weight * rel
            )

            acc_can = jnp.mean(jnp.argmax(out_can["logits"], axis=-1) == yb)
            acc_jit = jnp.mean(jnp.argmax(out_jit["logits"], axis=-1) == yb)
            agreement = jnp.mean(
                jnp.argmax(out_can["logits"], axis=-1)
                == jnp.argmax(out_jit["logits"], axis=-1)
            )
            return loss, (main, aux, consistency, rel, acc_can, acc_jit, agreement)

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
        sums = np.zeros(8, np.float64)  # loss main aux cons rel can jit agree
        nstep = 0

        bar = tqdm(
            iter_train_pairs(Xcan, Xjit, ytr, args.batch_size, args.seed + epoch),
            total=steps_per_epoch,
            desc=f"{protocol.upper()} CDREL TRAIN E{epoch:03d}/{args.epochs}",
            mininterval=0.5,
        )
        for xcan, xjit, yb in bar:
            xcan = ju.shard(xcan, ndev)
            xjit = ju.shard(xjit, ndev)
            yb = ju.shard(yb, ndev)
            state, rngs, metrics = p_train_step(
                state, rngs, xcan, xjit, yb, teacher_rel_rep
            )
            vals = [float(np.asarray(v[0])) for v in jax.device_get(metrics)]
            sums += np.asarray(vals, np.float64)
            nstep += 1
            if nstep % args.progress_every == 0:
                mean = sums / nstep
                bar.set_postfix(
                    loss=f"{mean[0]:.3f}", rel=f"{mean[4]:.4f}",
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
            eval_loss += float(vals[0])
            eval_correct += float(vals[1])
            eval_count += float(vals[2])

        val_acc = eval_correct / max(eval_count, 1.0)
        val_loss = eval_loss / max(eval_count, 1.0)
        mean = sums / max(nstep, 1)
        row = {
            "epoch": epoch,
            "train_loss": float(mean[0]),
            "main_ce": float(mean[1]),
            "aux_ce": float(mean[2]),
            "consistency": float(mean[3]),
            "relational": float(mean[4]),
            "train_can_accuracy": float(mean[5]),
            "train_jit_accuracy": float(mean[6]),
            "agreement": float(mean[7]),
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "epoch_seconds": float(time.time() - t0),
        }
        history.append(row)
        (outdir / "history.json").write_text(json.dumps(history, indent=2))

        log(
            f"{protocol.upper()} E{epoch:03d} "
            f"can={100*mean[5]:.3f}% jit={100*mean[6]:.3f}% "
            f"agree={100*mean[7]:.2f}% cons={mean[3]:.5f} rel={mean[4]:.6f} "
            f"val={100*val_acc:.3f}% loss={val_loss:.4f} time={row['epoch_seconds']:.1f}s"
        )

        improved = val_acc > best + 1e-6
        if improved:
            best = val_acc
            best_epoch = epoch
            stale = 0
            single = jax.tree_util.tree_map(lambda z: jax.device_get(z[0]), state)
            payload = {
                "model": "M4PhaseJitterConsistencyCDFormerRelKD_T16",
                "protocol": protocol,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "params": single.params,
                "ema_params": single.ema_params,
                "opt_state": single.opt_state,
                "step": single.step,
                "config": vars(args),
                "teacher": teacher_meta,
                "representation": {
                    "frames": FRAMES,
                    "token_channels": ju.TOKEN_CHANNELS,
                    "features_per_token": FEATURES,
                    "jitter_max_shift": args.jitter_max_shift,
                    "final_fusion": "uniform_mean",
                    "consistency_weight": args.consistency_weight,
                    "cdformer_transfer": "classifier_relation_offdiag_mse",
                    "rel_weight": args.rel_weight,
                    "teacher_forward": False,
                    "inference_extra_params": 0,
                    "inference_extra_flops": 0,
                },
            }
            (outdir / "best.msgpack").write_bytes(serialization.to_bytes(payload))
            (outdir / "best.json").write_text(json.dumps({
                "epoch": epoch,
                "val_accuracy": val_acc,
                "params": ju.count_params(single.params),
                "frames": FRAMES,
                "rel_weight": args.rel_weight,
                "teacher": teacher_meta,
                "teacher_forward": False,
                "inference_extra_params": 0,
                "inference_extra_flops": 0,
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
    p.add_argument("--teacher", default=str(DEFAULT_TEACHER))
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
    p.add_argument("--consistency-weight", type=float, default=0.08)
    p.add_argument("--consistency-temperature", type=float, default=1.0)
    p.add_argument("--rel-weight", type=float, default=0.03)
    p.add_argument(
        "--outdir",
        default="/kaggle/working/NestSAR_M4_CDFormer_RelKD_T16_TPU",
    )
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--audit-first", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    log(f"JAX={jax.__version__} backend={jax.default_backend()} devices={jax.devices()}")
    if jax.default_backend() != "tpu" or jax.local_device_count() != 8:
        raise RuntimeError(
            f"Expected one TPU v5e-8; backend={jax.default_backend()} "
            f"local_devices={jax.local_device_count()}"
        )
    if args.rel_weight < 0.0:
        raise ValueError("--rel-weight must be >= 0")

    teacher_relation, teacher_meta = load_teacher_relation(args.teacher)
    log(
        f"Teacher={teacher_meta['path']} | params={teacher_meta['parameters']:,} | "
        f"sha256={teacher_meta['sha256']}"
    )
    log(
        f"Experiment: T16 Phase+Jitter+Consistency + CD-Former relational KD | "
        f"lambda_rel={args.rel_weight:.4f}"
    )
    log("Teacher forward: OFF | inference architecture: unchanged")

    dataset = ju.base.find_dataset(args.dataset)
    log(f"Dataset={dataset}")
    anns, split = ju.base.load_ntu(dataset)
    protocols = ["xsub", "xset"] if args.protocol == "both" else [args.protocol]
    summary = {}
    for pr in protocols:
        best, ep = train_protocol(args, anns, split, pr, teacher_relation, teacher_meta)
        summary[pr] = {"best_val_accuracy": best, "best_epoch": ep}

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE {summary}")


if __name__ == "__main__":
    main()

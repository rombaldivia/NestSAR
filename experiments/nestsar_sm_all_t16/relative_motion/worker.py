"""One full 120-output NestSAR model on one device, with matched grouped data."""
from __future__ import annotations
import argparse
import hashlib
import io
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from ..streaming import worker as training
from ..streaming.io_utils import Reporter, atomic_bytes, atomic_json, read_json
from .data import Dataset


def evaluate_best(model, dataset, config, protocol, out, result, experiment):
    """Evaluate canonical views with best EMA; no train dropout/augmentation.

    best.msgpack has TOP-LEVEL ema_params. last.msgpack is a different schema.
    Fit/select/final confusion matrices and aligned predictions are saved once.
    """
    path = out / "best.msgpack"
    raw = path.read_bytes()
    checkpoint_hash = hashlib.sha256(raw).hexdigest()
    previous = read_json(out / "evaluation.json")
    if previous and previous.get("checkpoint_sha256") == checkpoint_hash:
        valid_files = all((out / name).is_file() and hashlib.sha256((out / name).read_bytes()).hexdigest() == sha
                          for name, sha in previous.get("files", {}).items())
        if valid_files and len(previous.get("files", {})) == 3 and previous.get("experiment") == experiment:
            return previous
    payload = serialization.msgpack_restore(raw)
    del raw
    if payload.get("experiment") != experiment or payload.get("protocol") != protocol:
        raise ValueError("Best checkpoint representation/split/protocol mismatch")
    params = jax.device_put(payload["ema_params"])
    del payload
    predict = jax.jit(lambda p, x: jax.lax.top_k(
        model.apply({"params": p}, x, training=False)["logits"], 5))
    report = Reporter(out / "status.json")
    common = dict(epoch=result["last_epoch"], completed_epoch=result["last_epoch"],
                  best=result["best_val_accuracy"], best_epoch=result["best_epoch"], done=False)
    evaluation = dict(checkpoint_sha256=checkpoint_hash, experiment=experiment, parts={}, files={})
    for part in ("fit", "select", "final"):
        ids = np.asarray(dataset.plan["indices"][part], np.int64)
        cap = config["max_train_samples"] if part == "fit" else config["max_val_samples"]
        if cap:
            ids = ids[:cap]
        top5 = np.empty((len(ids), 5), np.int32)
        size = config["eval_batch"]
        report(phase=f"Best EMA {part} (internal)", current=0, total=len(ids), **common)
        for start in range(0, len(ids), size):
            end = min(len(ids), start + size)
            batch, _ = dataset.batch(ids[start:end], np.arange(start, end), size, config, 0, False, protocol)
            scores, choices = jax.block_until_ready(predict(params, jax.device_put(batch["x"])))
            if not np.isfinite(np.asarray(scores)).all():
                raise FloatingPointError("Nonfinite best-checkpoint logits")
            top5[start:end] = np.asarray(choices)[:end-start]
            report(current=end)
        labels = np.asarray(dataset.labels[ids], np.int32)
        pred = top5[:, 0]
        confusion = np.bincount(120 * labels + pred, minlength=120*120).reshape(120, 120)
        support = confusion.sum(1)
        recall = np.divide(confusion.diagonal(), support, out=np.zeros(120, float), where=support > 0)
        metrics = dict(samples=len(ids), accuracy=float(np.mean(pred == labels)),
                       top5_accuracy=float(np.mean(np.any(top5 == labels[:, None], axis=1))),
                       per_class=[dict(action=i+1, support=int(support[i]),
                                       recall=float(recall[i]) if support[i] else None) for i in range(120)])
        if part == "select" and abs(metrics["accuracy"] - result["best_val_accuracy"]) > 1e-6:
            raise RuntimeError("Re-evaluated best EMA does not reproduce its selection accuracy")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, indices=ids, labels=labels, predictions=pred, top5=top5, confusion=confusion)
        filename = f"{part}_predictions.npz"
        contents = buffer.getvalue()
        atomic_bytes(out / filename, contents)
        evaluation["files"][filename] = hashlib.sha256(contents).hexdigest()
        evaluation["parts"][part] = metrics
    atomic_json(out / "evaluation.json", evaluation)
    return evaluation


def run_stage(cache, auxiliary, outdir, protocol, mode, config, plan, experiment, allow_cpu=False):
    if mode not in ("proxy", "relative") or experiment["mode"] != mode:
        raise ValueError("Invalid motion mode")
    dataset = Dataset(cache, auxiliary, plan)
    model = training.make_model(config).clone(motion_path=mode)
    result = training.run(config, protocol, cache, outdir, allow_cpu, model=model,
                          dataset=dataset, experiment=experiment)
    out = Path(outdir) / protocol
    evaluation = evaluate_best(model, dataset, config, protocol, out, result, experiment)
    result["internal_final_accuracy"] = evaluation["parts"]["final"]["accuracy"]
    result["evaluation_scope"] = plan["evaluation"]
    atomic_json(out / "result.json", result)
    Reporter(out / "status.json")(phase="Done (internal)", current=1, total=1, done=True,
        epoch=result["last_epoch"], completed_epoch=result["last_epoch"],
        best=result["best_val_accuracy"], best_epoch=result["best_epoch"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("cache", "auxiliary", "outdir", "protocol", "mode", "config", "plan", "experiment"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    run_stage(args.cache, args.auxiliary, args.outdir, args.protocol, args.mode,
              read_json(args.config), read_json(args.plan), read_json(args.experiment), args.allow_cpu)

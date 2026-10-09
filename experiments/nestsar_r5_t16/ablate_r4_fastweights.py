"""Inference-time counterfactuals on the trained R4-FMSE + LocalGeometry checkpoints.

Question: does R4 actually use its self-modifying fast memory and its capped
controller? Every mode keeps all trained weights and changes one thing at
inference on the full official validation split:

  trained            as trained (must reproduce the recorded best accuracy)
  fast_frozen        eta = 0, alpha = 1 in every fast memory: S stays at its
                     learned S0, so nothing is written in context (read = q^T S0)
  fast_frozen_m4     the same, frame-level (M4) memories only
  fast_frozen_g4     the same, chunk-level (G4) memories only
  fast_off           fast-memory reads set to zero (residual removed)
  uniform_fusion     stream fusion weights = 1/4 (controller fusion logits = 0)
  no_adaptive_head   rank-2 adaptive head delta removed (head_coeff = 0)
  controller_off     every capped controller knob neutral (FiLM gamma=1, beta=0,
                     stream gates = 1, uniform fusion, no adaptive head); the
                     fast memory keeps its eta/alpha

A drop is a counterfactual on a model trained WITH the component; it is not the
gain of the component in a model trained without it (that needs training, e.g.
the R5 variants ``no_fast_memory`` / ``capped_fast_memory``).

    python -m experiments.nestsar_r5_t16.ablate_r4_fastweights            # auto-find checkpoints + cache
    python -m experiments.nestsar_r5_t16.ablate_r4_fastweights --checkpoint-root DIR --r4-cache DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np

MODES = ("trained", "fast_frozen", "fast_frozen_m4", "fast_frozen_g4", "fast_off",
         "uniform_fusion", "no_adaptive_head", "controller_off")
REFERENCE_MODEL = "NestSAR-R4-FMSE-T16-v1"
REFERENCE_PARAMS = 1_831_932


PREFERRED_ROOT = "NestSAR_R4_FMSE_LOCAL_GEOMETRY_T16_v1"


def find_checkpoint_root(roots):
    """R4-FMSE run folders with xsub/best.msgpack; the LocalGeometry run is preferred
    (the plain FMSE run carries the same model name)."""
    found = []
    for root in roots:
        for pattern in ("*/xsub/best.json", "*/*/xsub/best.json", "*/*/*/xsub/best.json"):
            for path in sorted(glob.glob(str(Path(root) / pattern))):
                try:
                    meta = json.loads(Path(path).read_text())
                except (OSError, ValueError):
                    continue
                run = Path(path).parent.parent
                if meta.get("model") == REFERENCE_MODEL and Path(path).with_name("best.msgpack").exists() \
                        and run not in found:
                    found.append(run)
    found.sort(key=lambda d: (0 if d.name == PREFERRED_ROOT else 1, str(d)))
    if len(found) > 1:
        print("R4-FMSE checkpoint folders found: " + ", ".join(map(str, found)) + f"\nUsing {found[0]}", flush=True)
    return found[0] if found else None


def interceptor(mode):
    import jax.numpy as jnp
    from flax import linen as nn
    from experiments.nestsar_sm_all_t16 import model as r4

    def fn(next_fun, args, kwargs, context):
        module, method = context.module, context.method_name
        if method == "__call__" and isinstance(module, r4.SharedSMController):
            out = dict(next_fun(*args, **kwargs))
            if mode in ("uniform_fusion", "controller_off"):
                out["fusion_logits"] = jnp.zeros_like(out["fusion_logits"])
            if mode in ("no_adaptive_head", "controller_off"):
                out["head_coeff"] = jnp.zeros_like(out["head_coeff"])
            if mode == "controller_off":
                out["gamma"] = jnp.ones_like(out["gamma"])
                out["beta"] = jnp.zeros_like(out["beta"])
                out["stream_gate"] = jnp.ones_like(out["stream_gate"])
            return out
        if method == "__call__" and isinstance(module, r4.FastWeightDeltaResidual):
            x, eta, alpha = args
            is_m4 = x.shape[1] == 16
            frozen = (mode == "fast_frozen" or (mode == "fast_frozen_m4" and is_m4)
                      or (mode == "fast_frozen_g4" and not is_m4))
            if frozen:
                return next_fun(x, jnp.zeros_like(eta), jnp.ones_like(alpha), **kwargs)
            if mode == "fast_off":
                return jnp.zeros_like(next_fun(*args, **kwargs))
        return next_fun(*args, **kwargs)

    return nn.intercept_methods(fn) if mode != "trained" else None


def evaluate(model, params, dataset, ids, batch, mode, log_every=50):
    import contextlib
    import jax

    def forward(p, x):
        ctx = interceptor(mode)
        with (ctx if ctx is not None else contextlib.nullcontext()):
            out = model.apply({"params": p}, x, training=False)
        return out["logits"].argmax(-1), out["fusion_weights"], out["sm_eta_mean"], out["sm_alpha_mean"]

    fwd = jax.jit(forward)
    preds, fusion, eta, alpha = [], [], [], []
    steps = math.ceil(len(ids) / batch)
    t0 = time.time()
    for i in range(steps):
        chunk = np.asarray(ids[i * batch:(i + 1) * batch], np.int64)
        x = np.zeros((batch, *dataset.canonical.shape[1:]), np.float32)
        x[:len(chunk)] = dataset.canonical[chunk]
        p, f, e, a = jax.device_get(fwd(params, x))
        n = len(chunk)
        preds.append(p[:n]); fusion.append(f[:n]); eta.append(e[:n]); alpha.append(a[:n])
        if i % log_every == 0 or i + 1 == steps:
            print(f"    {mode:<18} {i + 1}/{steps}  {time.time() - t0:.0f}s", flush=True)
    return (np.concatenate(preds), np.concatenate(fusion), np.concatenate(eta), np.concatenate(alpha))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", default=None, help="dir with xsub/ and xset/ best.msgpack of R4-FMSE")
    ap.add_argument("--r4-cache", default=None)
    ap.add_argument("--protocols", default="xsub,xset")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-val-samples", type=int, default=0, help="0 = full validation split")
    ap.add_argument("--output", default=None, help="default: <checkpoint-root>/r4_fastweight_ablation.json")
    a = ap.parse_args(argv)

    import jax
    from flax import serialization
    from experiments.nestsar_r4_fmse_geometry_t16.worker import make_model
    from experiments.nestsar_r5_t16.data import candidate_r4_caches
    from experiments.nestsar_sm_all_t16.streaming.data import Dataset

    roots = [r for r in ("/kaggle/working", "/kaggle/input") if Path(r).is_dir()]
    ckpt_root = Path(a.checkpoint_root) if a.checkpoint_root else find_checkpoint_root(roots)
    if ckpt_root is None:
        raise SystemExit("No R4-FMSE checkpoints found (expected <run>/xsub/best.msgpack with best.json "
                         f"model={REFERENCE_MODEL}). Pass --checkpoint-root.")
    if a.r4_cache:
        cache = a.r4_cache
    else:
        found = candidate_r4_caches(roots, ("NestSAR_R4_EMA_REP_CONSISTENCY_CACHE_REBUILT_v1",))
        if not found:
            raise SystemExit("No R4 canonical cache found. Pass --r4-cache.")
        cache = str(found[0])
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    unknown = set(modes) - set(MODES)
    if unknown:
        raise SystemExit(f"Unknown modes {sorted(unknown)}; choose from {MODES}")
    if modes[0] != "trained":
        modes = ["trained"] + [m for m in modes if m != "trained"]

    print(f"Checkpoints: {ckpt_root}\nR4 cache: {cache}\nBackend: {jax.default_backend()} {jax.devices()}",
          flush=True)
    dataset = Dataset(cache)
    report = {"checkpoint_root": str(ckpt_root), "cache": cache, "backend": jax.default_backend(),
              "note": "Inference-time counterfactuals on trained weights; not training ablations.",
              "protocols": {}}
    for protocol in [p.strip() for p in a.protocols.split(",") if p.strip()]:
        path = ckpt_root / protocol / "best.msgpack"
        if not path.exists():
            print(f"{protocol}: no checkpoint at {path}, skipped", flush=True)
            continue
        payload = serialization.msgpack_restore(path.read_bytes())
        if payload.get("model") != REFERENCE_MODEL:
            raise SystemExit(f"{path} holds {payload.get('model')}, not {REFERENCE_MODEL}")
        params = payload["ema_params"]
        n_params = int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)))
        if n_params != REFERENCE_PARAMS:
            raise SystemExit(f"{path}: {n_params} params != {REFERENCE_PARAMS}")
        model = make_model(dict(payload["config"]))
        params = jax.device_put(params)
        ids = list(dataset.splits[f"{protocol}_val"])
        if a.max_val_samples:
            ids = ids[:a.max_val_samples]
        labels = np.asarray(dataset.labels[np.asarray(ids, np.int64)])
        if payload.get("cache_signature") not in (None, dataset.meta["signature"]):
            print(f"WARNING: {protocol} checkpoint was trained on a cache with another signature; "
                  "accuracies may not reproduce.", flush=True)
        print(f"\n{protocol.upper()}: epoch {payload.get('epoch')}, recorded val "
              f"{100 * float(payload.get('val_accuracy', float('nan'))):.4f}%, {len(ids):,} samples", flush=True)
        rows, base_pred = {}, None
        for mode in modes:
            pred, fusion, eta, alpha = evaluate(model, params, dataset, ids, a.batch, mode)
            acc = float(np.mean(pred == labels))
            row = {"accuracy": acc}
            if mode == "trained":
                base_pred = pred
                recorded = float(payload.get("val_accuracy", float("nan")))
                correct_diff = int(round(abs(acc - recorded) * len(ids))) if np.isfinite(recorded) else None
                cache_matches = payload.get("cache_signature") in (None, dataset.meta["signature"])
                same_split = payload.get("val_samples") in (None, len(ids))
                row.update(
                    recorded_accuracy=recorded,
                    correct_predictions_difference=correct_diff,
                    cache_matches_checkpoint=bool(cache_matches),
                    reproduced=bool(not a.max_val_samples and cache_matches and same_split
                                    and correct_diff == 0),
                    fusion_weight_min=float(fusion.min()), fusion_weight_max=float(fusion.max()),
                    fusion_weight_max_abs_from_uniform=float(np.abs(fusion - 0.25).max()),
                    eta_mean=float(eta.mean()), eta_p05=float(np.percentile(eta, 5)),
                    eta_p95=float(np.percentile(eta, 95)), alpha_mean=float(alpha.mean()),
                    alpha_p05=float(np.percentile(alpha, 5)), alpha_p95=float(np.percentile(alpha, 95)))
            else:
                right_before, right_now = base_pred == labels, pred == labels
                row.update(delta_pp=100 * (acc - rows["trained"]["accuracy"]),
                           fixed=int(np.sum(~right_before & right_now)),
                           broken=int(np.sum(right_before & ~right_now)),
                           changed_predictions=int(np.sum(pred != base_pred)))
            rows[mode] = row
            extra = (f"  delta {row['delta_pp']:+.3f} pp  (fixed {row['fixed']}, broken {row['broken']})"
                     if mode != "trained" else
                     f"  (recorded {100 * row['recorded_accuracy']:.4f}%; fusion weights in "
                     f"[{row['fusion_weight_min']:.3f}, {row['fusion_weight_max']:.3f}]; "
                     f"eta mean {row['eta_mean']:.3f}, alpha mean {row['alpha_mean']:.3f})")
            print(f"  {mode:<18} {100 * acc:7.3f}%{extra}", flush=True)
        report["protocols"][protocol] = {"epoch": payload.get("epoch"), "samples": len(ids), "modes": rows}

    output = Path(a.output) if a.output else ckpt_root / "r4_fastweight_ablation.json"
    try:
        output.write_text(json.dumps(report, indent=2))
    except OSError:                       # read-only input folder
        output = Path("/kaggle/working") / "r4_fastweight_ablation.json"
        output.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {output}")
    return report


if __name__ == "__main__":
    main()

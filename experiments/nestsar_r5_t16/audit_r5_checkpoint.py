"""Audit of trained NestSAR-R5 checkpoints (works on a stopped run: only best.msgpack is needed).

Per protocol, on the full official validation split (inference only, no training):

  reproduction     best EMA checkpoint must give back the recorded accuracy
  clean train acc  accuracy on a train subset in eval mode (no dropout, no augmentation),
                   so the train/val gap is not inflated by augmentation
  per class        recall, macro recall, weakest classes, top confusions, the R4 weak
                   classes (R4 vs R5), and the finger-driven classes as a group
  subsets          accuracy with one actor vs two actors
  calibration      confidence of right vs wrong predictions
  counterfactuals  one change at a time on the trained weights:
                     hand_zeroed       hand block set to 0 (the model sees only R4 tokens)
                     fast_scale_zero   fast/self-referential memory residual removed (every level)
                     fast_frozen       eta = 0, alpha = 1: fast memory read-only at its S0
                     pair_message_off  person-to-person message scale = 0
                   A drop is a counterfactual on a model trained WITH the component; it is
                   not the gain of the component in a model trained without it (that needs
                   the ablation variants).
  TTA              optional (--tta N): mean logits of the canonical view and N mildly augmented views
                   (yaw +-8 deg, +-1 frame boundary jitter); inference only
  R4 comparison    optional (--r4-checkpoint-root): both models on the same clips, who is
                   right when, the oracle union and the softmax-average ensemble
  history          overfit gap, gain per epoch, learned scales, timing (history.json)

    python -m experiments.nestsar_r5_t16.audit_r5_checkpoint --run-dir /kaggle/working/NestSAR_R5_T16_v1
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np

NUM_CLASSES = 120
MODES = ("trained", "hand_zeroed", "fast_scale_zero", "fast_frozen", "pair_message_off")
# NTU classes whose difference is in the fingers / hands (R4 weak list plus neighbours).
FINGER_CLASSES = tuple(f"A{c:03d}" for c in (11, 12, 71, 72, 73, 74, 75, 76, 84, 107))


def names():
    return [f"A{c + 1:03d}" for c in range(NUM_CLASSES)]


# --------------------------------------------------------------------------- counterfactuals
def counterfactual_params(params, mode):
    """Return a copy of ``params`` for the parameter-level counterfactuals (None = unchanged)."""
    if mode in ("trained", "hand_zeroed"):
        return params
    p = copy.deepcopy(params)
    if mode == "fast_scale_zero":
        for level in ("m4", "l2", "g4", "l8"):          # l2 / l8 exist only in the HOPE variants
            if level in p and "fast_scale" in p[level]:
                p[level]["fast_scale"] = np.zeros_like(np.asarray(p[level]["fast_scale"]))
        return p
    if mode == "fast_frozen":
        for level in ("m4", "g4"):
            fast = p.get(level, {}).get("fast")
            if fast is None or "gate_bias" not in fast:
                continue
            fast["gate"]["kernel"] = np.zeros_like(np.asarray(fast["gate"]["kernel"]))
            fast["surprise"] = np.zeros_like(np.asarray(fast["surprise"]))
            # eta = sigmoid(-30) ~ 0, alpha = 0.5 + 0.5 sigmoid(30) ~ 1: S never changes in the clip.
            fast["gate_bias"] = np.asarray([-30.0, 30.0], np.float32)
        return p
    if mode == "pair_message_off":
        if "pair_scale" in p:
            p["pair_scale"] = np.zeros_like(np.asarray(p["pair_scale"]))
        return p
    raise ValueError(mode)


def applicable(params, mode):
    if mode == "fast_scale_zero":
        return "fast_scale" in params.get("m4", {})
    if mode == "fast_frozen":
        return "gate_bias" in params.get("m4", {}).get("fast", {})
    if mode == "pair_message_off":
        return "pair_scale" in params
    return True


# --------------------------------------------------------------------------- evaluation
def make_forward(model):
    import jax

    @jax.jit
    def forward(params, x):
        out = model.apply({"params": params}, x, training=False)
        return out["logits"]

    return forward


def logits_for(forward, params, dataset, ids, batch, hand_start=None, label=""):
    """Logits [N, 120] for ``ids``; ``hand_start`` zeroes tokens from that feature on."""
    import jax
    ids = np.asarray(ids, np.int64)
    out = np.zeros((len(ids), NUM_CLASSES), np.float32)
    steps = math.ceil(len(ids) / batch)
    t0 = time.time()
    for i in range(steps):
        chunk = ids[i * batch:(i + 1) * batch]
        x = np.zeros((batch, *dataset.hand_shape), np.float32)
        x[:len(chunk)] = dataset.canonical(chunk)
        if hand_start is not None:
            x[..., hand_start:] = 0
        out[i * batch:i * batch + len(chunk)] = np.asarray(jax.device_get(forward(params, x)))[:len(chunk)]
        if i % 50 == 0 or i + 1 == steps:
            print(f"    {label:<18} {i + 1}/{steps}  {time.time() - t0:.0f}s", flush=True)
    return out


def tta_logits(forward, params, dataset, ids, batch, views, workers=4, seed=424242):
    """Mean logits over the canonical view and ``views`` augmented views, plus the per-view accuracies' inputs."""
    import concurrent.futures
    import jax
    from experiments.nestsar_r5_t16 import preprocessing as pp

    ids = np.asarray(ids, np.int64)
    total = np.zeros((len(ids), NUM_CLASSES), np.float32)
    per_view = []
    steps = math.ceil(len(ids) / batch)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for view in range(1, views + 1):
            def prepare(i, view=view):
                chunk = ids[i * batch:(i + 1) * batch]
                x = np.zeros((batch, pp.FRAMES, pp.FEATURES), np.float32)
                for j, idx in enumerate(chunk):
                    x[j] = pp.augmented_features(dataset.base.sample(idx), seed, view, int(idx), 8.0, 1,
                                                hand_filter=getattr(dataset, "hand_filter", "none"),
                                                body_align=getattr(dataset, "body_align", "none"))
                return chunk, x
            out = np.zeros_like(total)
            futures = [pool.submit(prepare, i) for i in range(min(steps, workers + 1))]
            nxt = len(futures)
            for i in range(steps):
                chunk, x = futures[i].result()
                if nxt < steps:
                    futures.append(pool.submit(prepare, nxt))
                    nxt += 1
                out[i * batch:i * batch + len(chunk)] = np.asarray(jax.device_get(forward(params, x)))[:len(chunk)]
            per_view.append(out)
            total += out
            print(f"    TTA view {view}/{views}  {time.time() - t0:.0f}s", flush=True)
    return per_view


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def confusion_matrix(labels, pred):
    c = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    np.add.at(c, (labels, pred), 1)
    return c


def class_report(labels, pred, reference):
    from experiments.nestsar_r5_t16.worker import R4_WEAK_CLASS_RECALL  # noqa: F401 (keeps one source)
    confusion = confusion_matrix(labels, pred)
    support = confusion.sum(1)
    recall = np.diag(confusion) / np.maximum(support, 1)
    nm = names()
    order = np.argsort(recall)
    off = confusion.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:20]
    pairs = [{"true": nm[i // NUM_CLASSES], "pred": nm[i % NUM_CLASSES], "count": int(off.flat[i])}
             for i in flat if off.flat[i] > 0]
    finger = [int(c[1:]) - 1 for c in FINGER_CLASSES]
    return {
        "macro_recall": float(recall[support > 0].mean()),
        "worst_15": [{"class": nm[c], "recall": float(recall[c]), "support": int(support[c])}
                     for c in order[:15]],
        "finger_classes": {"mean_recall": float(recall[finger].mean()),
                           "recall": {nm[c]: float(recall[c]) for c in finger}},
        "r4_weak_classes": {k: {"r4": v, "r5": float(recall[int(k[1:]) - 1]),
                                "delta_pp": 100 * (float(recall[int(k[1:]) - 1]) - v)}
                            for k, v in reference.items()},
        "top_confusions": pairs,
        "recall": {nm[c]: float(recall[c]) for c in range(NUM_CLASSES)},
    }


def calibration(prob, labels):
    pred = prob.argmax(1)
    conf = prob.max(1)
    right = pred == labels
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(right[m].mean() - conf[m].mean())
    return {"mean_confidence": float(conf.mean()), "accuracy": float(right.mean()),
            "confidence_when_right": float(conf[right].mean()) if right.any() else None,
            "confidence_when_wrong": float(conf[~right].mean()) if (~right).any() else None,
            "ece": float(ece)}


def actor_count(dataset, ids):
    """1 or 2 actors per clip, from the R4 tokens (second actor all-zero = absent)."""
    from experiments.nestsar_r5_t16 import preprocessing as pp
    ids = np.asarray(ids, np.int64)
    two = np.zeros(len(ids), bool)
    for start in range(0, len(ids), 2048):
        chunk = ids[start:start + 2048]
        body, _ = pp.split(dataset.canonical(chunk))
        two[start:start + len(chunk)] = np.abs(body[:, :, 1]).reshape(len(chunk), -1).max(1) > 0
    return two


# --------------------------------------------------------------------------- history
def history_report(run_protocol_dir):
    path = Path(run_protocol_dir) / "history.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    if not rows:
        return None
    last = rows[-1]
    gain5 = None
    if len(rows) >= 6:
        gain5 = 100 * (rows[-1]["val_acc"] - rows[-6]["val_acc"]) / 5
    keys = ("train_fast_scale_m4", "train_fast_scale_g4", "train_pair_scale", "eta", "alpha",
            "train_g4_eta", "train_g4_alpha", "val_aux_m4_acc", "val_aux_hand_acc", "val_top5",
            "epoch_s", "grad_norm_mean")
    return {
        "epochs_run": last["epoch"],
        "best_val_accuracy": max(r["val_acc"] for r in rows),
        "best_epoch": max(rows, key=lambda r: r["val_acc"])["epoch"],
        "last_train_acc_augmented_dropout": last.get("train_acc"),
        "last_val_acc": last["val_acc"],
        "last_gap_pp": 100 * (last.get("train_acc", float("nan")) - last["val_acc"]),
        "val_gain_pp_per_epoch_last5": gain5,
        "last_epoch_values": {k: last.get(k) for k in keys if k in last},
    }


# --------------------------------------------------------------------------- R4 comparison
def r4_comparison(r4_root, protocol, r5_dataset, ids, r5_logits, labels, batch):
    import jax
    from flax import serialization
    from experiments.nestsar_r4_fmse_geometry_t16.worker import make_model

    path = Path(r4_root) / protocol / "best.msgpack"
    if not path.exists():
        return {"skipped": f"no R4 checkpoint at {path}"}
    payload = serialization.msgpack_restore(path.read_bytes())
    model = make_model(dict(payload["config"]))
    params = jax.device_put(payload["ema_params"])
    base = r5_dataset.base

    @jax.jit
    def forward(p, x):
        return model.apply({"params": p}, x, training=False)["logits"]

    ids = np.asarray(ids, np.int64)
    out = np.zeros((len(ids), NUM_CLASSES), np.float32)
    for i in range(math.ceil(len(ids) / batch)):
        chunk = ids[i * batch:(i + 1) * batch]
        x = np.zeros((batch, *base.canonical.shape[1:]), np.float32)
        x[:len(chunk)] = base.canonical[chunk]
        out[i * batch:i * batch + len(chunk)] = np.asarray(jax.device_get(forward(params, x)))[:len(chunk)]
    r4_ok = out.argmax(1) == labels
    r5_ok = r5_logits.argmax(1) == labels
    ens = (softmax(out) + softmax(r5_logits)).argmax(1) == labels
    return {
        "r4_accuracy": float(r4_ok.mean()), "r5_accuracy": float(r5_ok.mean()),
        "both_right": float((r4_ok & r5_ok).mean()), "only_r4_right": float((r4_ok & ~r5_ok).mean()),
        "only_r5_right": float((~r4_ok & r5_ok).mean()), "both_wrong": float((~r4_ok & ~r5_ok).mean()),
        "oracle_union_accuracy": float((r4_ok | r5_ok).mean()),
        "softmax_average_ensemble_accuracy": float(ens.mean()),
        "same_prediction_rate": float((out.argmax(1) == r5_logits.argmax(1)).mean()),
        "note": "Ensemble = mean of the two softmax outputs; it is a diagnostic, not a model.",
    }


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/kaggle/working/NestSAR_R5_T16_v1",
                    help="R5 output folder with xsub/ and xset/ (best.msgpack, history.json)")
    ap.add_argument("--cache", default=None, help="R5 hand cache (default: found next to the R4 cache)")
    ap.add_argument("--r4-cache", default=None, help="R4 canonical cache (hint to relocate the base)")
    ap.add_argument("--r4-checkpoint-root", default=None, help="R4-FMSE run folder for the R4 comparison")
    ap.add_argument("--protocols", default="xsub,xset")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--train-samples", type=int, default=6000, help="clean train accuracy subset (0 = skip)")
    ap.add_argument("--tta", type=int, default=0, help="augmented views for test-time augmentation (0 = skip)")
    ap.add_argument("--max-val-samples", type=int, default=0, help="0 = full validation split")
    ap.add_argument("--output", default=None, help="default: <run-dir>/r5_audit.json")
    a = ap.parse_args(argv)

    import jax
    from flax import serialization
    from experiments.nestsar_r5_t16 import data as r5data
    from experiments.nestsar_r5_t16 import preprocessing as pp
    from experiments.nestsar_r5_t16.worker import EXPECTED_PARAMS, MODEL_NAME, R4_WEAK_CLASS_RECALL, make_model

    run = Path(a.run_dir)
    cache = a.cache
    if cache is None:
        roots = [r for r in ("/kaggle/working", "/kaggle/input") if Path(r).is_dir()]
        found = sorted(p for r in roots for p in Path(r).glob("NestSAR_R5_HAND_CACHE_v1*") if p.is_dir())
        if not found:
            raise SystemExit("No R5 hand cache found. Pass --cache.")
        cache = str(found[0])
    dataset = r5data.Dataset(cache, base=a.r4_cache)
    dataset.hand_shape = (pp.FRAMES, pp.FEATURES)
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    unknown = set(modes) - set(MODES)
    if unknown:
        raise SystemExit(f"Unknown modes {sorted(unknown)}; choose from {MODES}")
    if modes[0] != "trained":
        modes = ["trained"] + [m for m in modes if m != "trained"]
    print(f"Run: {run}\nHand cache: {cache}\nBackend: {jax.default_backend()} {jax.devices()}", flush=True)

    report = {"run_dir": str(run), "cache": str(cache), "backend": jax.default_backend(),
              "note": "Diagnostics on the official validation split (the same split used to pick the best "
                      "epoch); not an untouched test.", "protocols": {}}
    for protocol in [p.strip() for p in a.protocols.split(",") if p.strip()]:
        pdir = run / protocol
        path = pdir / "best.msgpack"
        if not path.exists():
            print(f"{protocol}: no checkpoint at {path}, skipped", flush=True)
            continue
        payload = serialization.msgpack_restore(path.read_bytes())
        if payload.get("model") != MODEL_NAME:
            raise SystemExit(f"{path} holds {payload.get('model')}, not {MODEL_NAME}")
        config = dict(payload["config"])
        params = payload["ema_params"]
        n_params = int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)))
        if n_params != EXPECTED_PARAMS[config["variant"]]:
            raise SystemExit(f"{path}: {n_params} params != {EXPECTED_PARAMS[config['variant']]}")
        model = make_model(config)
        forward = make_forward(model)
        ids = list(dataset.splits[f"{protocol}_val"])
        if a.max_val_samples:
            ids = ids[:a.max_val_samples]
        labels = np.asarray(dataset.labels[np.asarray(ids, np.int64)], np.int64)
        recorded = float(payload.get("val_accuracy", float("nan")))
        print(f"\n{protocol.upper()}: variant {config['variant']}, epoch {payload.get('epoch')}, recorded val "
              f"{100 * recorded:.4f}%, {len(ids):,} samples", flush=True)

        entry = {"variant": config["variant"], "epoch": payload.get("epoch"), "samples": len(ids),
                 "history": history_report(pdir)}
        modes_out, base_logits, base_pred = {}, None, None
        for mode in modes:
            p_mode = counterfactual_params(params, mode)
            if not applicable(params, mode):
                modes_out[mode] = {"skipped": "component absent in this variant"}
                print(f"  {mode:<18} skipped (component absent)", flush=True)
                continue
            lg = logits_for(forward, jax.device_put(p_mode), dataset, ids, a.batch,
                            hand_start=pp.R4_FEATURES if mode == "hand_zeroed" else None, label=mode)
            pred = lg.argmax(1)
            acc = float(np.mean(pred == labels))
            row = {"accuracy": acc}
            if mode == "trained":
                base_logits, base_pred = lg, pred
                diff = int(round(abs(acc - recorded) * len(ids))) if np.isfinite(recorded) else None
                same_split = payload.get("val_samples") in (None, len(ids))
                cache_ok = payload.get("cache_signature") in (None, dataset.meta["signature"])
                row.update(recorded_accuracy=recorded, correct_predictions_difference=diff,
                           cache_matches_checkpoint=bool(cache_ok),
                           reproduced=bool(not a.max_val_samples and cache_ok and same_split and diff == 0))
            else:
                before, now = base_pred == labels, pred == labels
                row.update(delta_pp=100 * (acc - modes_out["trained"]["accuracy"]),
                           fixed=int(np.sum(~before & now)), broken=int(np.sum(before & ~now)),
                           changed_predictions=int(np.sum(pred != base_pred)))
            modes_out[mode] = row
            extra = (f"  delta {row['delta_pp']:+.3f} pp  (fixed {row['fixed']}, broken {row['broken']})"
                     if mode != "trained" else
                     f"  (recorded {100 * recorded:.4f}%, reproduced={row['reproduced']})")
            print(f"  {mode:<18} {100 * acc:7.3f}%{extra}", flush=True)
        entry["modes"] = modes_out

        entry["per_class"] = class_report(labels, base_pred, R4_WEAK_CLASS_RECALL.get(protocol, {}))
        entry["calibration"] = calibration(softmax(base_logits), labels)
        two = actor_count(dataset, ids)
        entry["actors"] = {
            "one_actor": {"samples": int((~two).sum()),
                          "accuracy": float(np.mean(base_pred[~two] == labels[~two])) if (~two).any() else None},
            "two_actors": {"samples": int(two.sum()),
                           "accuracy": float(np.mean(base_pred[two] == labels[two])) if two.any() else None}}
        if a.train_samples:
            train_ids = np.asarray(dataset.splits[f"{protocol}_train"], np.int64)
            sub = np.random.default_rng(0).choice(train_ids, min(a.train_samples, len(train_ids)), replace=False)
            tl = logits_for(forward, jax.device_put(params), dataset, sub, a.batch, label="clean train")
            tacc = float(np.mean(tl.argmax(1) == np.asarray(dataset.labels[sub], np.int64)))
            entry["clean_train_accuracy"] = {"samples": len(sub), "accuracy": tacc,
                                             "gap_to_val_pp": 100 * (tacc - modes_out["trained"]["accuracy"])}
            print(f"  clean train acc    {100 * tacc:7.3f}%  gap to val {entry['clean_train_accuracy']['gap_to_val_pp']:.2f} pp",
                  flush=True)
        if a.tta:
            views = tta_logits(forward, jax.device_put(params), dataset, ids, a.batch, a.tta)
            base_acc = modes_out["trained"]["accuracy"]
            mean_logits = (base_logits + sum(views)) / (1 + len(views))
            mean_prob = (softmax(base_logits) + sum(softmax(v) for v in views)) / (1 + len(views))
            entry["tta"] = {
                "views": a.tta, "canonical_accuracy": base_acc,
                "single_view_accuracies": [float(np.mean(v.argmax(1) == labels)) for v in views],
                "mean_logits_accuracy": float(np.mean(mean_logits.argmax(1) == labels)),
                "mean_softmax_accuracy": float(np.mean(mean_prob.argmax(1) == labels)),
            }
            entry["tta"]["gain_pp"] = 100 * (entry["tta"]["mean_softmax_accuracy"] - base_acc)
            print(f"  TTA x{a.tta}: canonical {100 * base_acc:.3f}%  ->  {100 * entry['tta']['mean_softmax_accuracy']:.3f}% "
                  f"({entry['tta']['gain_pp']:+.3f} pp)", flush=True)
        if a.r4_checkpoint_root:
            entry["r4_comparison"] = r4_comparison(a.r4_checkpoint_root, protocol, dataset, ids,
                                                   base_logits, labels, a.batch)
            print("  R4 vs R5:", json.dumps({k: v for k, v in entry["r4_comparison"].items() if k != "note"}),
                  flush=True)
        c = entry["per_class"]
        print(f"  macro recall {100 * c['macro_recall']:.2f}%   finger classes {100 * c['finger_classes']['mean_recall']:.2f}%",
              flush=True)
        for k, v in c["r4_weak_classes"].items():
            print(f"    {k}: R4 {100 * v['r4']:.1f}%  R5 {100 * v['r5']:.1f}%  ({v['delta_pp']:+.1f} pp)", flush=True)
        report["protocols"][protocol] = entry

    output = Path(a.output) if a.output else run / "r5_audit.json"
    try:
        output.write_text(json.dumps(report, indent=2))
    except OSError:
        output = Path("/kaggle/working") / "r5_audit.json"
        output.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {output}")
    return report


if __name__ == "__main__":
    main()

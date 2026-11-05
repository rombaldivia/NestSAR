from __future__ import annotations
import argparse
import gc
import io
import json
import os
import time
from pathlib import Path

import jax
import numpy as np
import psutil
from flax import serialization

from ..streaming.io_utils import Reporter, atomic_bytes, atomic_json, read_json
from .config import ARMS, validate_config
from .data import TrainOnlyCache, fit_scale, grouped_plan, materialize_pair
from .metrics import paired_group_bootstrap, scores, summarize
from .training import Trainer, predict, train_candidate


def memory_rss_gib():
    try:
        return psutil.Process().memory_info().rss/2**30
    except (psutil.Error, OSError):
        # Optional telemetry must not fail training in a restricted PID namespace.
        return None


def run(cache_dir, outdir, protocol, config, allow_cpu=False):
    c = validate_config(config)
    if not allow_cpu and (jax.default_backend() != "gpu" or len(jax.local_devices()) != 1):
        raise RuntimeError("Each comparison worker must see exactly one GPU")
    out = Path(outdir) / protocol
    out.mkdir(parents=True, exist_ok=True)
    report = Reporter(out / "status.json")
    report(protocol=protocol, phase="Load train-only cache", epoch=0, current=0, total=1,
           best=None, best_epoch=0, done=False, failed=False)
    cache = TrainOnlyCache(cache_dir, protocol)
    plans = []
    for seed in c["seeds"]:
        plan = grouped_plan(cache.train, cache.groups, cache.data.labels, c["pairs"], seed, c)
        plan["grouping"] = "subject" if protocol == "xsub" else "setup"
        path = out / f"split_{seed}.json"
        if path.exists() and read_json(path) != plan:
            raise ValueError("Existing grouped split differs; choose a new output directory")
        atomic_json(path, plan)
        plans.append(plan)
    atomic_json(out / "hardware.json", dict(backend=jax.default_backend(), devices=[str(d) for d in jax.local_devices()],
                                           jax=jax.__version__, pid=os.getpid()))
    trainers = {}
    records = []
    for pair in c["pairs"]:
        pair_name = f"A{pair[0]:03d}_A{pair[1]:03d}"
        pair_out = out / pair_name
        complete = [read_json(pair_out / str(plan["seed"]) / "comparison.json") for plan in plans]
        if all(r is not None for r in complete):
            records.extend(complete)
            report(phase="Resume completed pair", pair=pair_name, current=1, total=1)
            continue
        prep_start = time.perf_counter()
        indices, y, arrays = materialize_pair(cache, pair, c, report)
        prep_s = time.perf_counter()-prep_start
        for repeat, plan in enumerate(plans):
            seed = plan["seed"]
            directory = pair_out / str(seed)
            directory.mkdir(parents=True, exist_ok=True)
            existing = read_json(directory / "comparison.json")
            if existing is not None:
                records.append(existing)
                continue
            partitions = {k: np.flatnonzero(np.isin(indices, v)) for k, v in plan["indices"].items()}
            for k, pos in partitions.items():
                if set(np.unique(y[pos])) != {0, 1}:
                    raise ValueError(f"Pair {pair}, seed {seed}, {k} lacks a class")
                cache.guard(indices[pos])
            # Files contain IDs/groups, so anyone can inspect exact matching.
            atomic_json(directory / "split_ids.json", {
                "plan_sha256": plan["sha256"], "grouping": plan["grouping"],
                "partitions": {k: [cache.ids[i] for i in indices[pos]] for k, pos in partitions.items()},
            })
            result = dict(protocol=protocol, pair=pair, seed=seed, plan_sha256=plan["sha256"],
                          grouping=plan["grouping"], pair_preparation_s=prep_s,
                          samples={k: len(v) for k, v in partitions.items()}, arms={})
            predictions = {}
            for arm in ARMS:
                arm_out = directory / arm
                arm_out.mkdir(parents=True, exist_ok=True)
                saved = read_json(arm_out / "result.json")
                prediction_file = arm_out / "final_predictions.npz"
                if saved is not None:
                    if not prediction_file.exists():
                        raise RuntimeError(f"Missing saved final predictions: {prediction_file}")
                    result["arms"][arm] = saved
                    with np.load(prediction_file, allow_pickle=False) as p:
                        if not np.array_equal(p["indices"], indices[partitions["final"]]):
                            raise ValueError("Saved final sample IDs do not match the grouped split")
                        predictions[arm] = p["probabilities"].argmax(1)
                    continue
                x = arrays[arm]
                scale = fit_scale(x, partitions["fit"])
                trials = []
                for trial_id, trial in enumerate(c["trials"]):
                    report(pair=pair_name, repeat=repeat+1, repeats=len(plans), arm=arm,
                           trial=trial_id+1, trials=len(c["trials"]),
                           best=None, best_epoch=0, epoch=0, train_acc=None, val_acc=None,
                           rss_gib=memory_rss_gib())
                    key = (arm, trial_id)
                    if key not in trainers:
                        trainers[key] = Trainer(arm, c, trial)
                    # Same initialization seed, sample order and trial grid across arms.
                    # GRU16/64 also start with identical parameter values/shapes.
                    model_seed = int(np.random.SeedSequence([seed, pair[0], pair[1]]).generate_state(1)[0])
                    meta = train_candidate(trainers[key], x, y, partitions["fit"], partitions["select"],
                                           scale, model_seed, arm_out/f"trial_{trial_id}", report)
                    trials.append(meta)
                winner = max(range(len(trials)), key=lambda i: (trials[i]["best"], -trials[i]["selection"]["nll"]))
                report(phase="Final diagnostic (frozen selection)", arm=arm,
                       best=trials[winner]["best"], best_epoch=trials[winner]["best_epoch"], current=0, total=1)
                params = jax.device_put(serialization.msgpack_restore((arm_out/f"trial_{winner}"/"best.msgpack").read_bytes()))
                final = partitions["final"]
                probabilities = predict(trainers[(arm, winner)], params, x, y, final, scale)
                saved = dict(selected_trial=winner, selected_config=c["trials"][winner],
                             selection=trials[winner]["selection"], best_epoch=trials[winner]["best_epoch"],
                             last_epoch=trials[winner]["epoch"], params=trials[winner]["params"],
                             final=scores(y[final], probabilities),
                             trials=[{k: v for k, v in t.items() if k != "history"} for t in trials])
                buf = io.BytesIO()
                np.savez_compressed(buf, indices=indices[final], ids=np.asarray([cache.ids[i] for i in indices[final]]),
                                    groups=cache.groups[indices[final]], y=y[final], probabilities=probabilities)
                atomic_bytes(prediction_file, buf.getvalue())
                if c["keep_checkpoints"]:
                    checkpoint = dict(params=jax.device_get(params), scale=scale, arm=arm,
                                      trial=c["trials"][winner], config=c, selection=saved["selection"])
                    atomic_bytes(arm_out / "selected_model.msgpack", serialization.msgpack_serialize(checkpoint))
                atomic_json(arm_out / "result.json", saved)
                for i in range(len(trials)):
                    (arm_out/f"trial_{i}"/"best.msgpack").unlink(missing_ok=True)
                result["arms"][arm] = saved
                predictions[arm] = probabilities.argmax(1)
                del params
            if result["arms"][ARMS[1]]["params"] != result["arms"][ARMS[2]]["params"]:
                raise AssertionError("The learned 16/64-frame controls must have identical parameter counts")
            final = partitions["final"]
            result["paired_differences"] = {
                "sequence64_minus_t16": paired_group_bootstrap(y[final], predictions[ARMS[0]], predictions[ARMS[2]],
                                                               cache.groups[indices[final]], seed, c["bootstrap_samples"]),
                "sequence64_minus_sequence16": paired_group_bootstrap(y[final], predictions[ARMS[1]], predictions[ARMS[2]],
                                                                      cache.groups[indices[final]], seed, c["bootstrap_samples"]),
            }
            atomic_json(directory / "comparison.json", result)
            records.append(result)
            atomic_json(out / "partial_summary.json", summarize(records, c["seeds"]))
        del arrays
        gc.collect()
    summary = summarize(records, c["seeds"])
    atomic_json(out / "comparisons.json", records)
    atomic_json(out / "summary.json", summary)
    report(phase="Complete: internal diagnostic", current=1, total=1, done=True,
           summaries={arm: summary[arm]["mean"] for arm in ARMS}, best=None, best_epoch=0,
           train_acc=None, val_acc=None)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--protocol", choices=("xsub", "xset"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    try:
        run(args.cache, args.outdir, args.protocol, json.loads(Path(args.config).read_text()), args.allow_cpu)
    except BaseException as exc:
        path = Path(args.outdir)/args.protocol/"status.json"
        status = read_json(path, {})
        status.update(failed=True, done=False, phase="Failed", error=str(exc))
        atomic_json(path, status)
        raise


if __name__ == "__main__":
    main()

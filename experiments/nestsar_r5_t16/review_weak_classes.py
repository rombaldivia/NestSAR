"""Review the existing heads of a saved R5/R7 checkpoint; no training or architecture changes.

    python -m experiments.nestsar_r5_t16.review_weak_classes --discover
    python -m experiments.nestsar_r5_t16.review_weak_classes --run-dir RUN [--cache HAND_CACHE]

Discovery reviews T16 graph_mr / graph_mr_pt / graph_mr_dct runs, in separate processes so each run
gets its original DCT input contract. Oracle counts use labels and are diagnostics, never accuracy
of a deployable selector or an upper bound on information present in the original skeletons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def input_contract(run):
    preprocessing = run.get("cache", {}).get("preprocessing", "")
    if not preprocessing:
        raise ValueError("Run has no recorded preprocessing contract")
    frames = re.search(r"-T(24|32)(?:\b|\+)", preprocessing)
    dct = re.search(r"bodydct(\d+)x", preprocessing)
    return (int(frames.group(1)) if frames else 16, int(dct.group(1)) if dct else 0)


def discover_runs(roots):
    found = {}
    for root in roots:
        for depth in range(1, 4):
            for protocol in ("xsub", "xset"):
                for path in Path(root).glob("*/" * depth + f"{protocol}/run_config.json"):
                    if not path.with_name("best.msgpack").is_file():
                        continue
                    try:
                        run = json.loads(path.read_text())
                        if run["config"]["variant"] not in ("graph_mr", "graph_mr_pt", "graph_mr_dct"):
                            continue
                        if input_contract(run)[0] != 16:
                            continue
                    except (OSError, ValueError, KeyError, TypeError):
                        continue
                    found[str(path.parent.parent.resolve())] = path.parent.parent
    return [found[k] for k in sorted(found)]


def find_cache(signature, explicit, roots):
    if explicit:
        return Path(explicit)
    matches = []
    for root in roots:
        for depth in range(1, 5):
            for path in Path(root).glob("*/" * depth + "manifest.json"):
                try:
                    manifest = json.loads(path.read_text())
                    if manifest.get("signature") == signature:
                        matches.append(path.parent)
                except (OSError, ValueError):
                    pass
    if not matches:
        raise FileNotFoundError("No hand cache matches this run. Attach the original cache or pass --cache.")
    return matches[0]


def class_report(result, weak_k=15):
    y, main, aux = (np.asarray(result[k]) for k in ("y", "main", "aux"))
    pred, apred = main.argmax(-1), aux.argmax(-1)
    correct = pred == y
    acorrect = apred == y[:, None]
    any_aux = acorrect.any(1)
    names = ["m4", "hand"][:aux.shape[1]]
    classes = []
    for label in np.unique(y):
        selected = y == label
        errors = selected & ~correct
        support = int(selected.sum())
        row = dict(class_id=f"A{label + 1:03d}", support=support, errors=int(errors.sum()),
                   recall=float(correct[selected].mean()),
                   errors_correct_in_any_aux=int((errors & any_aux).sum()),
                   errors_wrong_in_every_head=int((errors & ~any_aux).sum()))
        row["aux"] = {name: dict(recall=float(acorrect[selected, i].mean()),
                                 main_errors_corrected=int((errors & acorrect[:, i]).sum()),
                                 main_correct_broken=int((selected & correct & ~acorrect[:, i]).sum()))
                      for i, name in enumerate(names)}
        classes.append(row)
    weak = sorted(classes, key=lambda r: (r["recall"], r["class_id"]))[:weak_k]
    directions = []
    pairs, counts = np.unique(np.stack([y[~correct], pred[~correct]], axis=1), axis=0, return_counts=True)
    for (true, wrong), count in zip(pairs, counts):
        errors = (y == true) & (pred == wrong)
        directions.append(dict(true=f"A{true + 1:03d}", predicted=f"A{wrong + 1:03d}", count=int(count),
                               correct_in_any_aux=int((errors & any_aux).sum()),
                               wrong_in_every_head=int((errors & ~any_aux).sum())))
    checkpoint_accuracy = result.get("checkpoint_accuracy")
    measured = float(correct.mean())
    delta = None if checkpoint_accuracy is None else 100 * (measured - float(checkpoint_accuracy))
    oracle = correct | any_aux
    return dict(
        identity=result["identity"], variant=result["config"]["variant"], seed=result["config"]["seed"],
        samples=len(y), top1=measured, top5=float(np.any(np.argsort(main, axis=1)[:, -5:] == y[:, None], axis=1).mean()),
        recorded_accuracy=checkpoint_accuracy, reproduction_delta_pp=delta,
        reproduced=None if delta is None else bool(abs(delta) < 1e-6),
        existing_heads_oracle=float(oracle.mean()),
        main_errors_correct_in_any_aux=int((~correct & any_aux).sum()),
        errors_wrong_in_every_head=int((~correct & ~any_aux).sum()),
        aux_accuracy={name: float(acorrect[:, i].mean()) for i, name in enumerate(names)},
        weak_classes=weak, classes=classes,
        top_confusions=sorted(directions, key=lambda r: -r["count"])[:30],
        note="Descriptive diagnostics on the official held-out split. The oracle uses true labels and is not "
             "deployable accuracy. Failure of all existing heads does not prove missing input information. "
             "Do not select training changes repeatedly on this held-out report.",
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir")
    group.add_argument("--discover", action="store_true")
    ap.add_argument("--cache")
    ap.add_argument("--protocols", default="xsub,xset")
    ap.add_argument("--roots", nargs="+", default=["/kaggle/working", "/kaggle/input"])
    ap.add_argument("--output-root", default="/kaggle/working/NestSAR_WEAK_REVIEW")
    ap.add_argument("--weak-k", type=int, default=15)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    protocols = a.protocols.split(",")
    if not protocols or any(p not in ("xsub", "xset") for p in protocols) or a.weak_k < 1:
        ap.error("Use protocols xsub,xset and a positive --weak-k")
    if a.discover:
        if a.cache:
            ap.error("--cache is for one --run-dir; discovery matches each run's cache signature")
        runs = discover_runs(a.roots)
        if not runs:
            raise SystemExit("No saved T16 graph_mr/graph_mr_pt/graph_mr_dct runs found under --roots")
        print(f"Found {len(runs)} saved T16 runs; reviewing existing checkpoints only.", flush=True)
        failures = []
        for run in runs:
            command = [sys.executable, "-u", "-m", __spec__.name, "--run-dir", str(run),
                       "--protocols", a.protocols, "--output-root", a.output_root,
                       "--weak-k", str(a.weak_k), "--roots", *a.roots]
            if a.force:
                command.append("--force")
            if subprocess.run(command, check=False).returncode:
                failures.append(str(run))
        if failures:
            raise SystemExit("Some reviews failed; inspect their messages: " + ", ".join(failures))
        return
    run = Path(a.run_dir)
    available = [p for p in protocols if (run / p / "best.msgpack").is_file()]
    if not available:
        raise SystemExit(f"No requested checkpoints in {run}")
    records = {p: json.loads((run / p / "run_config.json").read_text()) for p in available}
    contracts = {input_contract(r) for r in records.values()}
    if len(contracts) != 1:
        raise SystemExit("Protocols have different input contracts; review each with --protocols separately")
    frames, dct = contracts.pop()
    # These must be selected before importing the model or preprocessing modules.
    os.environ.update(NESTSAR_FRAMES=str(frames), NESTSAR_BODY_DCT=str(dct))
    from .data import Dataset
    from .eval_heads import evaluate_bundle
    from experiments.nestsar_sm_all_t16.streaming.io_utils import atomic_json
    output = Path(a.output_root) / (run.name + "-" + hashlib.sha256(str(run.resolve()).encode()).hexdigest()[:8])
    mismatches = []
    for protocol, record in records.items():
        cache = find_cache(record["cache"], a.cache, a.roots)
        print(f"\n{protocol}: {run}\nCache: {cache}", flush=True)
        result = evaluate_bundle(run, protocol, Dataset(cache), force=a.force,
                                 prediction_cache_dir=output / protocol)
        report = class_report(result, a.weak_k)
        path = output / protocol / "weak_class_report.json"
        atomic_json(path, report)
        print(f"Top1 {100 * report['top1']:.4f}% | reproduction delta {report['reproduction_delta_pp']} pp", flush=True)
        for row in report["weak_classes"]:
            print(f"  {row['class_id']}: recall {100 * row['recall']:6.2f}% | errors {row['errors']:4d} | "
                  f"correct in existing aux {row['errors_correct_in_any_aux']:4d} | "
                  f"wrong in every head {row['errors_wrong_in_every_head']:4d}", flush=True)
        print(f"Saved {path}\n{report['note']}", flush=True)
        if report["reproduced"] is False:
            mismatches.append(protocol)
    if mismatches:
        raise SystemExit("Recorded accuracy did not reproduce for " + ", ".join(mismatches) +
                         ". Reports were saved; resolve the mismatch before choosing a new experiment.")


if __name__ == "__main__":
    main()

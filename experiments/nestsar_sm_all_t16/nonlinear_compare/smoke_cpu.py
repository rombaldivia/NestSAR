"""Small end-to-end two-process check, including resume and held-out poisoning."""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from .. import preprocessing_corrected as pp
from ..streaming.data import prepare
from ..streaming.io_utils import atomic_json
from .launch import run


def main(outdir):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    cache = root/"cache"
    if not (cache/"manifest.json").exists():
        samples, train, held = [], [], []
        for g in range(1, 11):
            for action in (71, 72):
                for rep in range(1, 4):
                    sid = f"S{g:03d}C001P{g:03d}R{rep:03d}A{action:03d}"
                    raw = np.zeros((2, 20, 25, 3), np.float32)
                    pose = np.random.default_rng(g*100+rep).normal(0, .05, (25, 3))
                    raw[0] = pose + [1, 2, 3]
                    raw[0, :, 10:20, 0] += .7*(action-71)
                    samples.append(dict(frame_dir=sid, label=action-1, keypoint=raw))
                    (train if g <= 8 else held).append(sid)
        splits = {f"{p}_{part}": v for p in ("xsub", "xset") for part, v in (("train", train), ("val", held))}
        with (root/"fixture.pkl").open("wb") as f:
            pickle.dump(dict(annotations=samples, split=splits), f)
        prepare(root/"fixture.pkl", cache, root/"prepare.json")
        # Accessing any held-out skeleton/features would fail finite checks.
        canonical = np.load(cache/"canonical.npy", mmap_mode="r+")
        raw = np.load(cache/"raw.npy", mmap_mode="r+")
        offsets = np.load(cache/"offsets.npy")
        for i in range(len(train), len(samples)):
            canonical[i] = np.nan
            raw[offsets[i]:offsets[i+1]] = np.nan
        canonical.flush()
        raw.flush()
    cfg = dict(pairs=[[71, 72]], seeds=[128], epochs=2, warmup_epochs=1, batch_size=8,
               mlp_width=8, gru_width=8, bootstrap_samples=100, min_class_samples=[4, 2, 2],
               trials=[dict(learning_rate=.01, weight_decay=.0001, dropout=.1),
                       dict(learning_rate=.005, weight_decay=.001, dropout=.2)])
    first = run(cache, root/"run", cfg, _allow_cpu=True)
    second = run(cache, root/"run", cfg, _allow_cpu=True)
    assert first == second
    results = {p: json.loads((root/"run"/p/"comparisons.json").read_text()) for p in ("xsub", "xset")}
    for p, rows in results.items():
        assert len(rows) == 1 and len(rows[0]["arms"]) == 3
        assert rows[0]["arms"]["sequence16_gru"]["params"] == rows[0]["arms"]["sequence64_gru"]["params"]
        assert not list((root/"run"/p).rglob("*.msgpack"))
    atomic_json(root/"validation.json", dict(cpu=True, simultaneous_protocol_workers=True,
                arms=3, trials_per_arm=2, epochs=2, completed_resume_identical=True,
                poisoned_official_heldout_not_accessed=True, no_retained_trial_weights=True,
                real_ntu_accuracy_measured=False, dual_t4_executed=False))
    print("CPU integration passed: both workers, all arms, inner selection, final predictions, resume, held-out protection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    main(parser.parse_args().outdir)

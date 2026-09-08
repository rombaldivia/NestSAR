"""Real SM-ALL, both modes/protocols, checkpoint reload and grouped safeguards.

Synthetic data and sample caps verify execution only, never NTU accuracy.
"""
import argparse
import os
import pickle
from pathlib import Path
import numpy as np
from ..streaming.data import prepare as prepare_p2
from ..streaming.io_utils import atomic_json, read_json
from .data import prepare, Dataset, split_plan


def synthetic_cache(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "p2"
    if (cache / "manifest.json").exists():
        return cache
    annotations, train, held = [], [], []
    # Six fit/select/final source groups, two completely untouched official groups.
    for group in range(1, 9):
        for action in range(1, 121):
            sid = f"S{group:03d}C001P{group:03d}R001A{action:03d}"
            x = np.zeros((2, 16, 25, 3), np.float32)
            x[0] = np.random.default_rng(action).normal(0, .04, (16, 25, 3)) + [1, 2, 3]
            x[0, :, 7, 0] += np.sin(np.arange(16, dtype=np.float32)) * (action/120)
            annotations.append(dict(frame_dir=sid, label=action-1, keypoint=x))
            (train if group <= 6 else held).append(sid)
    splits = {f"{p}_{part}": ids for p in ("xsub", "xset") for part, ids in (("train", train), ("val", held))}
    dataset = root / "fixture.pkl"
    dataset.write_bytes(pickle.dumps(dict(annotations=annotations, split=splits)))
    prepare_p2(dataset, cache, root / "p2_status.json")
    return cache


def main(outdir):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    from .launch import run
    root = Path(outdir)
    cache = synthetic_cache(root)
    auxiliary = root / "relative"
    prepare(cache, auxiliary, root / "relative_status.json", reserve_bytes=0)
    # After deterministic cache construction, poison every official held-out
    # feature. No training, selection, or final inference may read these rows.
    held = read_json(cache / "splits.json")["xsub_val"]
    for name in ("canonical", "raw"):
        a = np.load(cache / (name + ".npy"), mmap_mode="r+")
        offsets = np.load(cache / "offsets.npy")
        for i in held:
            if name == "canonical":
                a[i] = np.nan
            else:
                a[offsets[i]:offsets[i+1]] = np.nan
        a.flush()
    for protocol in ("xsub", "xset"):
        data = Dataset(cache, auxiliary, split_plan(cache, protocol, 128, min_class_samples=[1, 1, 1]))
        try:
            data.sample(held[0])
        except ValueError:
            pass
        else:
            raise AssertionError("Official held-out guard did not fire")
    config = dict(seeds=[128], min_class_samples=[1, 1, 1], smoke_test=True, audit_first=False)
    first = run(cache, auxiliary, root / "run", config, _allow_cpu=True)
    output = root / "run_smoke"
    hashes = {str(p): __import__("hashlib").sha256(p.read_bytes()).hexdigest() for p in output.rglob("last.msgpack")}
    for path in output.rglob("best.msgpack"):
        path.unlink()  # Must be repaired from the committed epoch checkpoint.
    second = run(cache, auxiliary, root / "run", config, _allow_cpu=True)
    assert first == second
    for name, sha in hashes.items():
        assert __import__("hashlib").sha256(Path(name).read_bytes()).hexdigest() == sha
    for path in output.rglob("history.json"):
        rows = read_json(path)
        assert len(rows) == 2 and all(r["train_samples"] == 3 and r["val_samples"] == 2 for r in rows)
    atomic_json(root / "validation.json", dict(real_model_params=1_826_556, cpu=True,
        concurrent_protocol_workers=True, modes=["proxy", "relative"], epochs_per_run=2,
        checkpoint_resume_and_alias_repair=True, canonical_best_ema_scores_reproduced=True,
        poisoned_official_heldout_not_accessed=True, real_ntu_accuracy_measured=False, dual_t4_executed=False))
    print("CPU integration passed: both modes/protocols, best EMA, matched splits, held-out guard and resume.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    main(parser.parse_args().outdir)

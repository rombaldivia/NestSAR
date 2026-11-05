"""End-to-end CPU smoke test of the exact Kaggle path on a tiny synthetic NTU pickle.

pickle -> R4 cache -> R5 hand cache -> detached launcher -> two concurrent
workers (real R5 model, 2 epochs) -> epoch-1 kill decision against a fake R4
reference -> result.json / per_class.json -> re-run (attach / finished) ->
interrupted-run resume. Checks execution and bookkeeping, not accuracy.

    JAX_PLATFORMS=cpu python -m experiments.nestsar_r5_t16.smoke_cpu --root /tmp/r5_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def synthetic_pickle(path, n=14, seed=0):
    rng = np.random.default_rng(seed)
    annotations = []
    for i in range(n):
        people = 2 if i % 3 == 0 else 1
        total = int(rng.integers(12, 90))
        x = np.zeros((people, total, 25, 3), np.float32)           # MTVC, like ntu120_3danno.pkl
        base = rng.normal(0, 0.3, (people, 1, 25, 3)) + np.array([0.0, 0.4, 3.0])
        t = np.arange(total)[None, :, None, None]
        x[:] = base + 0.05 * np.sin(0.3 * (i + 1) * t) + rng.normal(0, 0.01, (people, total, 25, 3))
        annotations.append(dict(frame_dir=f"S001C001P{i:03d}R001A{i % 120 + 1:03d}", keypoint=x,
                                label=int(i % 7), total_frames=total))
    ids = [a["frame_dir"] for a in annotations]
    split = dict(xsub_train=ids[:9], xsub_val=ids[9:], xset_train=ids[5:], xset_val=ids[:5])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(dict(annotations=annotations, split=split)))


def fake_reference(working):
    from experiments.nestsar_r5_t16 import PREFERRED_REFERENCE, REFERENCE_MODEL, REFERENCE_PARAMS
    for p in ("xsub", "xset"):
        d = working / PREFERRED_REFERENCE / p
        d.mkdir(parents=True, exist_ok=True)
        (d / "run_config.json").write_text(json.dumps({
            "model": REFERENCE_MODEL, "parameters": REFERENCE_PARAMS,
            "config": {"epochs": 60, "learning_rate": 6e-4, "seed": 128, "micro_batch": 64,
                       "accumulation_steps": 4}}))
        (d / "history.json").write_text(json.dumps([{"epoch": 1, "val_acc": 0.0}, {"epoch": 2, "val_acc": 0.0}]))


def run(cmd, env, timeout=3600):
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd)}\n--- stdout ---\n{r.stdout[-6000:]}\n--- stderr ---\n{r.stderr[-6000:]}")
    return r.stdout


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--variant", default="full")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if root.exists() and not a.keep:
        shutil.rmtree(root)
    working, inputs = root / "working", root / "input"
    working.mkdir(parents=True, exist_ok=True)
    synthetic_pickle(inputs / "ntu120" / "ntu120_3danno.pkl")
    fake_reference(working)
    outdir = working / "R5_smoke_run"
    env = dict(os.environ, PYTHONPATH=str(REPO), JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
               PYTHONUNBUFFERED="1")
    extra = json.dumps({"epochs": 2, "micro_batch": 2, "accumulation_steps": 2, "eval_batch": 4,
                        "progress_every": 1, "patience": 5})
    cmd = [sys.executable, "-m", "experiments.nestsar_r5_t16.kaggle_run", "--cpu-smoke",
           "--working", str(working), "--inputs", str(inputs), "--outdir", str(outdir),
           "--micro-batch", "2", "--kill-epoch", "1", "--poll-seconds", "2", "--extra-config", extra,
           "--variant", a.variant]
    log1 = run(cmd, env)
    print(log1[-3000:])
    assert "Model OK" in log1 and "hand cache OK" in log1, "preflight/cache messages missing"
    decision = json.loads((outdir / "kill_decision.json").read_text())
    assert decision["decision"] == "CONTINUE" and decision["epoch"] == 1, decision
    for p in ("xsub", "xset"):
        res = json.loads((outdir / p / "result.json").read_text())
        hist = json.loads((outdir / p / "history.json").read_text())
        pc = json.loads((outdir / p / "per_class.json").read_text())
        assert res["epochs_run"] == 2 and len(hist) == 2, (p, res, len(hist))
        from experiments.nestsar_r5_t16.worker import EXPECTED_PARAMS
        assert res["params"] == EXPECTED_PARAMS[a.variant] and res["variant"] == a.variant
        assert (outdir / p / "best.msgpack").exists()
        assert 0.0 <= pc["top1"] <= 1.0 and set(pc["r4_weak_classes"]) and len(pc["recall"]) == 120
        for row in hist:
            for k in ("val_acc", "val_top5", "eta", "alpha", "train_fast_scale_m4", "train_pair_scale",
                      "val_aux_hand_acc", "data_wait_s"):
                assert k in row and np.isfinite(row[k]), (p, k)

    # Re-running the cell must not start anything new.
    log2 = run(cmd, env)
    assert "already finished" in log2, log2[-2000:]

    # Interrupted run: drop the finished markers of XSUB and resume the worker directly.
    xsub = outdir / "xsub"
    for name in ("result.json", "per_class.json", "best.msgpack", "history.json"):
        (xsub / name).unlink()
    hand_cache = next(p for p in working.iterdir() if p.name.startswith("NestSAR_R5_HAND_CACHE_v1_"))
    run([sys.executable, "-m", "experiments.nestsar_r5_t16.worker", "--config", str(outdir / "config.json"),
         "--protocol", "xsub", "--cache", str(hand_cache), "--outdir", str(outdir), "--allow-cpu"], env)
    res = json.loads((xsub / "result.json").read_text())
    assert res["resumed_completed"] and (xsub / "best.msgpack").exists() and (xsub / "per_class.json").exists()
    assert len(json.loads((xsub / "history.json").read_text())) == 2

    # A different variant must refuse the same output folder.
    bad = subprocess.run([sys.executable, "-m", "experiments.nestsar_r5_t16.launch", "--cache", str(hand_cache),
                          "--outdir", str(outdir), "--variant", "no_hand_branch" if a.variant != "no_hand_branch" else "full",
                          "--cpu-smoke",
                          "--micro-batch", "2", "--extra-config", extra],
                         cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert bad.returncode != 0 and "different config" in (bad.stdout + bad.stderr), bad.stdout + bad.stderr
    report = {"passed": True, "variant": a.variant, "protocols": ["xsub", "xset"], "epochs": 2, "kill_decision": decision,
              "resume_completed": True, "variant_guard": True}
    (root / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

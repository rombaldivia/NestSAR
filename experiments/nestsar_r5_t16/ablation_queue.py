"""R9 ablation queue for a Kaggle T4 x2 session: one worker per GPU pulls the next pending run.

Every run uses the graph_mr champion recipe (NestSAR_T16_GRAPH_MR_REGNOMIX_SEED*), XSub only, and is named
NestSAR_T16_<VARIANT>_SEED<s>. A worker claims a job atomically (mkdir), starts it through kaggle_run, waits
for result.json, then saves val_probs.npz with eval_heads (needed for the paired McNemar tests). It does not
start a run that cannot finish before the session deadline. Re-running the cell skips finished runs.

    python -m experiments.nestsar_r5_t16.ablation_queue --gpu 0 --tier 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RECIPE = {"aug_strength": 1.0, "dropout": 0.3, "weight_decay": 0.05, "label_smoothing": 0.1,
          "recon_mask_ratio": 0.5}
EPOCHS, MICRO_BATCH = 38, 64

# (variant, seed) in priority order. Tier 1 decides what the paper can claim; tier 2 is the component table;
# tier 3 is the causal (streaming) question. Seeds 7 and 11 pair with the existing base runs.
TIERS = {
    1: [("graph_mr_no_fast", 11), ("graph_mr_mem_only", 7), ("graph_mr_shared", 7),
        ("graph_mr_mem_only", 11), ("graph_mr_one_level", 7), ("graph_mr_shared", 11),
        ("graph_mr_one_level", 11)],
    2: [("graph_mr_no_temporal", 7), ("graph_mr_no_hand", 7), ("graph_mr_no_inter", 7), ("graph_mr_coarse", 7),
        ("graph_mr_no_recon", 7), ("graph_mr_no_temporal", 11), ("graph_mr_no_hand", 11),
        ("graph_mr_no_inter", 11), ("graph_mr_coarse", 11), ("graph_mr_no_recon", 11)],
    3: [("graph_mr_causal", 7), ("graph_mr_causal_no_fast", 7), ("graph_mr_causal", 11),
        ("graph_mr_causal_no_fast", 11)],
}


def run_dir(working: Path, variant: str, seed: int) -> Path:
    return working / f"NestSAR_T16_{variant.upper()}_SEED{seed}"


def session_hours_left(limit_hours: float) -> float:
    up = float(Path("/proc/uptime").read_text().split()[0]) / 3600
    return limit_hours - up


def sh(cmd, env, log):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n")
        f.flush()
        return subprocess.run(list(map(str, cmd)), env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--tier", default="1", help="comma list of tiers, e.g. 1 or 1,2")
    ap.add_argument("--working", default="/kaggle/working")
    ap.add_argument("--hand-cache", default="/tmp/NestSAR_R5_HAND_CACHE_T16")
    ap.add_argument("--aug-pool", default="/tmp/NestSAR_AUGPOOL_T16")
    ap.add_argument("--hours-per-run", type=float, default=3.4, help="38 epochs at ~300 s, plus evaluation")
    ap.add_argument("--session-hours", type=float, default=12.0)
    a = ap.parse_args(argv)
    working = Path(a.working)
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo), NESTSAR_FRAMES="16", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    log = working / f"ABLATION_QUEUE_GPU{a.gpu}.log"
    jobs = [j for t in a.tier.split(",") for j in TIERS[int(t)]]
    if a.gpu == 0:
        # The existing champion / no-fast runs need val_probs.npz too (paired McNemar against them).
        from .ablation_report import ALIASES
        for names in ALIASES.values():
            for pattern in names:
                for s in (7, 11, 13):
                    d = working / pattern.format(s=s)
                    if (d / "xsub" / "best.msgpack").exists() and not (d / "xsub" / "val_probs.npz").exists():
                        print(f"[GPU0] val_probs for {d.name}", flush=True)
                        sh([sys.executable, "-u", "-m", "experiments.nestsar_r5_t16.eval_heads", "--cache",
                            a.hand_cache, "--protocol", "xsub", d], dict(env, CUDA_VISIBLE_DEVICES="0"), log)
    for variant, seed in jobs:
        out = run_dir(working, variant, seed)
        result, probs = out / "xsub" / "result.json", out / "xsub" / "val_probs.npz"
        claim = Path("/tmp/.ablation_claims") / out.name     # per session: a dead session never blocks a job
        try:
            claim.mkdir(parents=True)                       # atomic: the other GPU skips this job
        except FileExistsError:
            continue
        if not result.exists():
            left = session_hours_left(a.session_hours)
            if left < a.hours_per_run:
                claim.rmdir()
                print(f"[GPU{a.gpu}] {left:.1f} h left < {a.hours_per_run} h per run: stopping before {out.name}",
                      flush=True)
                return 0
            print(f"[GPU{a.gpu}] start {out.name} ({left:.1f} h left in the session)", flush=True)
            extra = json.dumps(dict(RECIPE, seed=seed))
            code = sh([sys.executable, "-u", "-m", "experiments.nestsar_r5_t16.kaggle_run",
                       "--hand-cache", a.hand_cache, "--aug-pool", a.aug_pool, "--pool-views", 6,
                       "--pool-dtype", "float16", "--r4-build-dir", "/tmp", "--micro-batch", MICRO_BATCH,
                       "--epochs", EPOCHS, "--ignore-kill", "--variant", variant, "--protocols", "xsub",
                       "--gpu-offset", a.gpu, "--no-follow", "--outdir", out, "--extra-config", extra], env, log)
            if code != 0:
                print(f"[GPU{a.gpu}] launch of {out.name} failed (exit {code}); see {log}", flush=True)
                continue
            t0 = time.time()
            while not result.exists():
                if time.time() - t0 > 1.6 * a.hours_per_run * 3600:
                    print(f"[GPU{a.gpu}] {out.name}: no result.json after {1.6 * a.hours_per_run:.1f} h; "
                          f"see {out}/launcher.log", flush=True)
                    break
                time.sleep(120)
            if not result.exists():
                continue
            time.sleep(30)
        if not probs.exists():
            genv = dict(env, CUDA_VISIBLE_DEVICES=str(a.gpu))
            sh([sys.executable, "-u", "-m", "experiments.nestsar_r5_t16.eval_heads", "--cache", a.hand_cache,
                "--protocol", "xsub", out], genv, log)
        acc = json.loads(result.read_text())["best_val_accuracy"]
        print(f"[GPU{a.gpu}] done  {out.name}: {100 * acc if acc <= 1 else acc:.2f}", flush=True)
    print(f"[GPU{a.gpu}] queue empty", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

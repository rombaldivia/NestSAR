"""Kaggle job queue: run a list of single-protocol training jobs on the two T4s, one job per GPU at a time.

    nohup python -m experiments.nestsar_r5_t16.kaggle_queue --jobs /kaggle/working/QUEUE_jobs.json \
        > /kaggle/working/QUEUE.log 2>&1 &

jobs.json = [{"name": ..., "dataset": "ntu120"|"ntu60", "protocol": "xsub"|"xset"|"xview", "variant": ...,
              "epochs": 38, "extra": {...config...}, "pool": "/tmp/...", "view_degrees": 15}, ...]

* Caches and augmentation pools are prepared first, serially (kaggle_run --prepare-only), so two jobs never
  build the same cache at once.
* A job is done when <outdir>/<protocol>/result.json exists; such jobs are skipped, so re-running the queue
  after a Kaggle restart (which wipes /tmp) rebuilds the caches and resumes unfinished runs from their last
  checkpoint (same commit = same run identity). A job that ends without a result is retried once.
* State is written to QUEUE_state.json after every change (the notebook monitor reads it).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKING = Path("/kaggle/working")
HAND = {"ntu120": "/tmp/NestSAR_R5_HAND_CACHE_T16", "ntu60": "/tmp/NestSAR_R5_HAND_CACHE_T16_NTU60"}


def find_pickle(dataset):
    tag = "120" if dataset == "ntu120" else "60"
    found = [p for p in glob.glob("/kaggle/input/**/*.pkl", recursive=True)
             if tag in Path(p).name and (tag == "120" or "120" not in Path(p).name)]
    if not found:
        raise SystemExit(f"no {dataset} PYSKL pickle under /kaggle/input")
    return sorted(found)[0]


def base_cmd(job):
    env = f"NESTSAR_DATASET={job['dataset']} NESTSAR_FRAMES=16"
    extra = dict(job.get("extra", {}))
    view = float(job.get("view_degrees", extra.get("aug_view_degrees", 15.0)))
    if view != 15.0:
        extra["aug_view_degrees"] = view
    args = (f"--pickle {find_pickle(job['dataset'])} --r4-build-dir /tmp --hand-cache {HAND[job['dataset']]} "
            f"--aug-pool {job['pool']} --pool-views 6 --pool-dtype float16 --variant {job['variant']}")
    return env, args, extra


def run(cmd, log):
    with open(log, "a") as fh:
        return subprocess.run(cmd, shell=True, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def outdir(job):
    return WORKING / job["name"]


def done(job):
    return (outdir(job) / job["protocol"] / "result.json").exists()


def alive(job):
    r = subprocess.run(["pgrep", "-f", f"{outdir(job)}( |/|$)"], capture_output=True, text=True)
    return bool(r.stdout.split())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--poll", type=float, default=60.0)
    a = ap.parse_args(argv)
    jobs = json.loads(Path(a.jobs).read_text())
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate job names")
    state_file = WORKING / "QUEUE_state.json"
    state = {j["name"]: {"status": "done" if done(j) else "pending", "tries": 0, "gpu": None} for j in jobs}

    def save():
        state_file.write_text(json.dumps({"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "jobs": state}, indent=1))

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    save()
    # 1) prepare every (dataset, pool) pair once, serially
    for key in dict.fromkeys((j["dataset"], j["pool"], float(j.get("view_degrees", 15.0)))
                             for j in jobs if state[j["name"]]["status"] != "done"):
        job = next(j for j in jobs if (j["dataset"], j["pool"], float(j.get("view_degrees", 15.0))) == key)
        env, args, extra = base_cmd(job)
        log(f"prepare {key[0]} pool={key[1]} view={key[2]}")
        rc = run(f"{env} {sys.executable} -u -m experiments.nestsar_r5_t16.kaggle_run {args} --prepare-only "
                 f"--extra-config '{json.dumps(dict(extra, aug_strength=1.0))}'", WORKING / "QUEUE_prepare.log")
        if rc != 0:
            raise SystemExit(f"prepare failed for {key}; see /kaggle/working/QUEUE_prepare.log")
    # 2) one job per GPU
    running = {}                                     # gpu -> job
    pending = [j for j in jobs if state[j["name"]]["status"] != "done"]
    while pending or running:
        for gpu, job in list(running.items()):
            if alive(job):
                continue
            st = state[job["name"]]
            if done(job):
                st["status"] = "done"
                log(f"done   {job['name']} (GPU{gpu})")
            elif st["tries"] < 2:
                st["status"] = "pending"
                pending.insert(0, job)
                log(f"retry  {job['name']} (ended without result.json)")
            else:
                st["status"] = "failed"
                log(f"FAILED {job['name']}; see {outdir(job)}/launcher.log")
            del running[gpu]
            save()
        for gpu in range(a.gpus):
            if gpu in running or not pending:
                continue
            job = pending.pop(0)
            if done(job):
                state[job["name"]]["status"] = "done"
                save()
                continue
            env, args, extra = base_cmd(job)
            st = state[job["name"]]
            st.update(status="running", gpu=gpu, tries=st["tries"] + 1, started=time.strftime("%H:%M"))
            log(f"start  {job['name']} on GPU{gpu} (try {st['tries']})")
            rc = run(f"{env} {sys.executable} -u -m experiments.nestsar_r5_t16.kaggle_run {args} "
                     f"--micro-batch 64 --epochs {job.get('epochs', 38)} --kill-epoch 999 "
                     f"--protocols {job['protocol']} --gpu-offset {gpu} --no-follow --outdir {outdir(job)} "
                     f"--extra-config '{json.dumps(extra)}'", WORKING / f"LAUNCH_{job['name']}.log")
            if rc != 0:
                log(f"launch failed ({rc}) for {job['name']}; see LAUNCH_{job['name']}.log")
            running[gpu] = job
            save()
            time.sleep(20)                           # let the launcher spawn its worker before polling
        time.sleep(a.poll)
    log("queue finished: " + json.dumps({k: v["status"] for k, v in state.items()}))


if __name__ == "__main__":
    main()

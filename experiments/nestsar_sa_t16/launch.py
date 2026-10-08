"""Launch NestSAR-SA on two GPUs (XSUB->GPU0, XSET->GPU1) with an early kill rule.

Kill rule at --kill-epoch (default 10), against the R4-FMSE + LocalGeometry run
trained by the same pipeline (its run_config is checked: same epochs/LR/seed):
    delta_p = SA_val_acc(epoch) - FMSE_val_acc(epoch)   for each protocol with a reference
    KILL if all deltas < 0, or if any delta < -kill_margin (default 0.5 pp).
Otherwise training continues to the normal end / early stopping.

Safe to re-run: workers resume from last.msgpack, a running launch is
re-attached instead of starting duplicate processes, and a run already stopped
by the kill rule is NOT restarted (unless --ignore-kill).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROTOCOLS = ("xsub", "xset")
_META = None


def meta():
    """Package constants, loaded by path so this file also works when loaded by path."""
    global _META
    if _META is None:
        spec = importlib.util.spec_from_file_location("nestsar_sa_meta", Path(__file__).with_name("__init__.py"))
        _META = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_META)
    return _META


WORKER = "experiments.nestsar_sa_t16.worker"
REF_LABEL = "R4-FMSE"


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def is_reference(d, p):
    """R4-FMSE + LocalGeometry run of protocol p: same pipeline, reference model and size."""
    rc = read_json(Path(d, p, "run_config.json")) or {}
    hist = read_json(Path(d, p, "history.json"))
    return (bool(hist) and rc.get("model") == meta().REFERENCE_MODEL
            and rc.get("parameters") == meta().REFERENCE_PARAMS)


def find_reference(explicit=None, root="/kaggle/working"):
    preferred = meta().PREFERRED_REFERENCE
    if explicit:
        cands = [explicit]
    else:
        cands = sorted({str(Path(p).parent.parent) for p in
                        glob.glob(str(Path(root) / "*" / "*" / "history.json"))})
        cands.sort(key=lambda d: (0 if Path(d).name == preferred else 1, d))
    for d in cands:
        if any(is_reference(d, p) for p in PROTOCOLS):
            return d
    return None


def reference_config_issues(ref, config):
    """Differences that would make epoch-by-epoch comparison unfair."""
    issues = []
    for p in PROTOCOLS:
        rc = (read_json(Path(ref, p, "run_config.json")) or {}).get("config", {})
        if not rc:
            continue
        for k in ("epochs", "learning_rate", "seed", "warmup_fraction", "min_learning_rate"):
            if k in rc and k in config and rc[k] != config[k]:
                issues.append(f"{p}: {k} reference={rc[k]} this run={config[k]}")
        if "micro_batch" in rc and rc.get("micro_batch", 0) * rc.get("accumulation_steps", 0) != \
                config["micro_batch"] * config["accumulation_steps"]:
            issues.append(f"{p}: effective batch differs")
    return issues


def val_at(history, epoch):
    for row in history or []:
        if int(row.get("epoch", -1)) == epoch:
            return float(row["val_acc"])
    return None


def worker_env(repo, gpu):
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               MALLOC_ARENA_MAX="2", CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONPATH=str(repo))
    if gpu is None:
        env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    else:
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS="cuda")
    return env


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def tail(path, n=25):
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def stop_workers(pids):
    for p in PROTOCOLS:
        if alive(pids.get(p)):
            try:
                os.killpg(os.getpgid(int(pids[p])), signal.SIGTERM)
            except OSError:
                pass


def _pct(v):
    return ("%.2f%%" % (100 * v)) if v is not None else "--"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--outdir", default="/kaggle/working/NestSAR_SA_T16_v1")
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--reference-dir", default=None)
    ap.add_argument("--working-root", default="/kaggle/working",
                    help="where to look for the R4-FMSE reference run")
    ap.add_argument("--kill-epoch", type=int, default=10)
    ap.add_argument("--kill-margin-pp", type=float, default=0.5)
    ap.add_argument("--ignore-kill", action="store_true",
                    help="resume a run that the kill rule already stopped")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--heartbeat-minutes", type=float, default=10.0,
                    help="print progress at least this often even if nothing changed")
    ap.add_argument("--cpu-smoke", action="store_true", help="local test only")
    ap.add_argument("--extra-config", default="{}", help="JSON merged into config (smoke tests)")
    a = ap.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if 256 % a.micro_batch:
        raise SystemExit("--micro-batch must divide 256 (effective batch stays 256 like R4)")
    config = dict(epochs=a.epochs, micro_batch=a.micro_batch,
                  accumulation_steps=256 // a.micro_batch)
    config.update(json.loads(a.extra_config))
    cfg_path = out / "config.json"
    old = read_json(cfg_path)
    if old is not None and old != config:
        raise SystemExit(f"{out} already holds a different config: {old}\n"
                         "Use the same settings to resume, or choose a new --outdir.")
    cfg_path.write_text(json.dumps(config, indent=2))

    previous = read_json(out / "kill_decision.json") or {}
    if previous.get("decision") == "KILL" and not a.ignore_kill:
        d = previous.get("delta_pp", {})
        print(f"This run was already stopped at epoch {previous.get('epoch')} by the kill rule "
              f"(XSUB {d.get('xsub', float('nan')):+.2f} pp, XSET {d.get('xset', float('nan')):+.2f} pp "
              f"vs {REF_LABEL}). Nothing was restarted.\nDetails: " + str(out / "kill_decision.json") +
              "\nTo resume it anyway, pass --ignore-kill.", flush=True)
        return

    ref = find_reference(a.reference_dir, a.working_root)
    ref_hist = ({p: (read_json(Path(ref, p, "history.json")) if is_reference(ref, p) else None)
                 for p in PROTOCOLS} if ref else {})
    print(f"Reference {REF_LABEL} run : {ref or 'NOT FOUND -> no kill rule, training runs to the end'}")
    if ref:
        from experiments.nestsar_sm_all_t16.streaming.launch import DEFAULTS
        issues = reference_config_issues(ref, dict(DEFAULTS, **config))
        for p in PROTOCOLS:
            v = val_at(ref_hist[p], a.kill_epoch)
            print(f"  {REF_LABEL} {p} @E{a.kill_epoch}: {('%.3f%%' % (100*v)) if v is not None else 'missing'}")
        for issue in issues:
            print("  WARNING (comparison not like-for-like): " + issue)

    pid_file = out / "pids.json"
    pids = read_json(pid_file) or {}
    procs = {}
    for i, p in enumerate(PROTOCOLS):
        if alive(pids.get(p)):
            print(f"{p}: already running (pid {pids[p]}), attaching")
            continue
        done = read_json(out / p / "result.json")
        if done is not None:
            print(f"{p}: already finished (best {100*done.get('best_val_accuracy', 0):.3f}%)")
            continue
        cmd = [sys.executable, "-m", WORKER, "--config", str(cfg_path),
               "--protocol", p, "--cache", a.cache, "--outdir", str(out)]
        if a.cpu_smoke:
            cmd.append("--allow-cpu")
        log = open(out / f"{p}.log", "a")
        proc = subprocess.Popen(cmd, cwd=repo, env=worker_env(repo, None if a.cpu_smoke else i),
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        procs[p] = proc
        pids[p] = proc.pid
        print(f"{p}: started pid {proc.pid} on {'CPU' if a.cpu_smoke else 'GPU' + str(i)}")
    pid_file.write_text(json.dumps(pids))
    sys.stdout.flush()

    decided = previous.get("decision")
    last_key, last_print = None, 0.0
    while True:
        rows, running = {}, False
        for p in PROTOCOLS:
            hist = read_json(out / p / "history.json") or []
            st = read_json(out / p / "status.json") or {}
            rows[p] = (hist, st)
            proc = procs.get(p)
            if proc is not None and proc.poll() is not None:
                if proc.returncode != 0 and read_json(out / p / "result.json") is None:
                    log_tail = tail(out / f"{p}.log", 200)
                    print(f"\n!!! {p} worker exited with code {proc.returncode}. "
                          f"Last log lines:\n{tail(out / f'{p}.log')}", flush=True)
                    if "RESOURCE_EXHAUSTED" in log_tail or "out of memory" in log_tail.lower():
                        print("Looks like GPU OOM: set micro-batch 32 (accumulation 8, same 256 batch) "
                              "and a NEW output folder.", flush=True)
                procs.pop(p)
            running |= alive(pids.get(p)) if p not in procs else procs[p].poll() is None

        parts, progress = [], []
        for p in PROTOCOLS:
            hist, st = rows[p]
            e = hist[-1]["epoch"] if hist else 0
            v = hist[-1]["val_acc"] if hist else None
            r = val_at(ref_hist.get(p), e) if hist else None
            delta = f" ({100*(v-r):+.2f} vs ref)" if (v is not None and r is not None) else ""
            parts.append(f"{p} E{e:02d} {str(st.get('phase', '?'))[:14]:<14} "
                         f"val {_pct(v)}{delta} best {_pct(st.get('best') or None)}")
            extra = f" train_acc {_pct(st.get('train_acc'))}" if st.get("train_acc") is not None else ""
            progress.append(f"{p} {st.get('current', '?')}/{st.get('total', '?')}{extra}")
        key = " | ".join(parts)
        now_t = time.time()
        if key != last_key:
            print(time.strftime("[%H:%M] ") + key, flush=True)
            last_key, last_print = key, now_t
        elif now_t - last_print >= 60 * a.heartbeat_minutes:
            print(time.strftime("[%H:%M]   ... ") + " | ".join(progress), flush=True)
            last_print = now_t

        if ref and decided is None:
            now = {p: val_at(rows[p][0], a.kill_epoch) for p in PROTOCOLS}
            refv = {p: val_at(ref_hist.get(p), a.kill_epoch) for p in PROTOCOLS}
            usable = [p for p in PROTOCOLS if refv[p] is not None]
            if usable and all(now[p] is not None for p in usable):
                d = {p: 100 * (now[p] - refv[p]) for p in usable}
                kill = all(x < 0 for x in d.values()) or min(d.values()) < -a.kill_margin_pp
                decided = "KILL" if kill else "CONTINUE"
                (out / "kill_decision.json").write_text(json.dumps(
                    {"decision": decided, "epoch": a.kill_epoch, "delta_pp": d,
                     "pt": now, "r4": refv, "reference_dir": ref}, indent=2))
                print(f"\n=== EPOCH {a.kill_epoch} DECISION: {decided}  ("
                      + ", ".join(f"{p.upper()} {x:+.2f} pp" for p, x in d.items())
                      + f" vs {REF_LABEL}) ===\n", flush=True)
                if kill:
                    stop_workers(pids)
                    print(f"Workers stopped. NestSAR-SA did not beat {REF_LABEL} early; see kill_decision.json",
                          flush=True)
                    return

        if not running:
            print("\nAll workers finished.")
            for p in PROTOCOLS:
                res = read_json(out / p / "result.json")
                if res:
                    print(f"  {p}: best {100*res['best_val_accuracy']:.3f}% @E{res['best_epoch']}")
                else:
                    print(f"  {p}: no result.json (stopped or failed; see {out / (p + '.log')})")
            sys.stdout.flush()
            return
        time.sleep(a.poll_seconds)


if __name__ == "__main__":
    main()

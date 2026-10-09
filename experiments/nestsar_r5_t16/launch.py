"""Launch NestSAR-R5 on two GPUs (XSUB -> GPU0, XSET -> GPU1) with an early kill rule.

At --kill-epoch (default 10) the EMA validation accuracy of each protocol is
compared with the R4-FMSE + LocalGeometry run trained by the same data
pipeline (found under --working-root, or /kaggle/input as a fallback):

    delta_p = R5_val(epoch) - R4_val(epoch)
    KILL     if every protocol that has a reference is more than
             --kill-margin-pp (default 1.0 pp) behind;
    CONTINUE otherwise (normal end / early stopping).

R5 is a different architecture, so its early curve is less predictive than a
one-block change; the margin is wider than the SA rule (0.5 pp).

Robustness: one launcher per output folder (file lock); workers resume from
last.msgpack; a worker that dies for a non-deterministic reason (killed, GPU
fault) is restarted up to --max-restarts times, while NaN, out-of-memory and
config/resume mismatches are reported instead of retried; a run already stopped
by the kill rule is not restarted (unless --ignore-kill); process liveness
ignores zombies and checks the command line, so a recycled PID is never
mistaken for (or killed as) a worker.
"""
from __future__ import annotations

import argparse
import fcntl
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
WORKER = "experiments.nestsar_r5_t16.worker"
LAUNCHER = "experiments.nestsar_r5_t16.launch"
REF_LABEL = "R4-FMSE"
NO_RETRY_MARKERS = ("Nonfinite", "FloatingPointError", "RESOURCE_EXHAUSTED", "out of memory",
                    "OUT_DIR contains another", "Resume config/data mismatch", "parameter mismatch",
                    "Empty train/validation", "is not an R5 hand cache", "base cache")
_META = None


def meta():
    """Package constants, loaded by path so this file also works when loaded by path."""
    global _META
    if _META is None:
        spec = importlib.util.spec_from_file_location("nestsar_r5_meta", Path(__file__).with_name("__init__.py"))
        _META = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_META)
    return _META


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_json(path, data):
    """Atomic JSON write (a crash never leaves a half-written decision or pid file)."""
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def is_reference(d, p):
    rc = read_json(Path(d, p, "run_config.json")) or {}
    hist = read_json(Path(d, p, "history.json"))
    return bool(hist) and rc.get("model") == meta().REFERENCE_MODEL and \
        rc.get("parameters") == meta().REFERENCE_PARAMS


def find_reference(explicit=None, roots=("/kaggle/working", "/kaggle/input")):
    preferred = meta().PREFERRED_REFERENCE
    if explicit:
        cands = [explicit]
    else:
        cands = set()
        for root in roots:
            for pattern in ("*/*/history.json", "*/*/*/history.json", "*/*/*/*/history.json"):
                cands |= {str(Path(p).parent.parent) for p in glob.glob(str(Path(root) / pattern))}
        cands = sorted(cands, key=lambda d: (0 if Path(d).name == preferred else 1, d))
    for d in cands:
        if any(is_reference(d, p) for p in PROTOCOLS):
            return d
    return None


def reference_config_issues(ref, config):
    """Differences that make the epoch-by-epoch comparison not like-for-like."""
    issues = []
    for p in PROTOCOLS:
        rc = (read_json(Path(ref, p, "run_config.json")) or {}).get("config", {})
        if not rc:
            continue
        for k in ("epochs", "learning_rate", "seed", "warmup_fraction", "min_learning_rate"):
            if k in rc and k in config and rc[k] != config[k]:
                issues.append(f"{p}: {k} reference={rc[k]} this run={config[k]}")
        if rc.get("micro_batch", 0) * rc.get("accumulation_steps", 0) != \
                config["micro_batch"] * config["accumulation_steps"]:
            issues.append(f"{p}: effective batch differs")
    return issues


def val_at(history, epoch):
    for row in history or []:
        if int(row.get("epoch", -1)) == epoch:
            return float(row["val_acc"])
    return None


def kill_decision(now, reference, margin_pp):
    """(decision, deltas_pp) or (None, {}) while any needed value is missing."""
    usable = [p for p in now if reference.get(p) is not None]
    if not usable or any(now[p] is None for p in usable):
        return None, {}
    deltas = {p: 100.0 * (now[p] - reference[p]) for p in usable}
    kill = all(d < -margin_pp for d in deltas.values())
    return ("KILL" if kill else "CONTINUE"), deltas


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


def alive(pid, expect=None):
    """True for a live, non-zombie process (whose command line contains `expect`, if given)."""
    try:
        pid = int(pid)
        os.kill(pid, 0)
    except (OSError, ValueError, TypeError):
        return False
    proc = Path(f"/proc/{pid}")
    if proc.exists():
        try:
            stat = (proc / "stat").read_text()
            if stat[stat.rindex(")") + 2:].split()[0] in ("Z", "X"):
                return False
            if expect is not None:
                cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                return expect in cmd
        except (OSError, ValueError, IndexError):
            return False
    return True


def tail(path, n=25):
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def stop_workers(pids):
    for p in list(pids):
        if alive(pids.get(p), WORKER):
            try:
                os.killpg(os.getpgid(int(pids[p])), signal.SIGTERM)
            except OSError:
                pass


def _pct(v):
    return ("%.2f%%" % (100 * v)) if v is not None else "--"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="R5 hand-cache directory")
    ap.add_argument("--outdir", default="/kaggle/working/NestSAR_R5_T16_v1")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--protocols", default="xsub,xset", help="comma list; GPU i runs the i-th protocol")
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--reference-dir", default=None)
    ap.add_argument("--working-root", default="/kaggle/working")
    ap.add_argument("--kill-epoch", type=int, default=10)
    ap.add_argument("--kill-margin-pp", type=float, default=1.0)
    ap.add_argument("--ignore-kill", action="store_true")
    ap.add_argument("--max-restarts", type=int, default=2, help="per worker, for non-deterministic crashes")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--heartbeat-minutes", type=float, default=10.0)
    ap.add_argument("--cpu-smoke", action="store_true", help="local test only")
    ap.add_argument("--extra-config", default="{}", help="JSON merged into config (smoke tests)")
    a = ap.parse_args(argv)

    protocols = tuple(p.strip() for p in a.protocols.split(",") if p.strip())
    if not protocols or len(set(protocols)) != len(protocols) or set(protocols) - set(PROTOCOLS):
        raise SystemExit(f"--protocols must be a subset of {PROTOCOLS}")
    repo = Path(__file__).resolve().parents[2]
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    lock = (out / "launch.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"Another launcher already manages {out}; nothing started.", flush=True)
        return

    if 256 % a.micro_batch:
        raise SystemExit("--micro-batch must divide 256 (the effective batch stays 256 like R4)")
    config = dict(epochs=a.epochs, micro_batch=a.micro_batch, accumulation_steps=256 // a.micro_batch,
                  variant=a.variant)
    config.update(json.loads(a.extra_config))
    sys.path.insert(0, str(repo))
    from experiments.nestsar_r5_t16.config import validate_config
    full_config = validate_config(config)
    cfg_path = out / "config.json"
    old = read_json(cfg_path)
    if old is not None and old != config:
        raise SystemExit(f"{out} already holds a different config: {old}\n"
                         "Use the same settings to resume, or choose a new --outdir.")
    write_json(cfg_path, config)

    previous = read_json(out / "kill_decision.json") or {}
    if previous.get("decision") == "KILL" and not a.ignore_kill:
        d = previous.get("delta_pp", {})
        print(f"This run was stopped at epoch {previous.get('epoch')} by the kill rule ("
              + ", ".join(f"{k.upper()} {v:+.2f} pp" for k, v in d.items())
              + f" vs {REF_LABEL}). Nothing was restarted.\nDetails: {out / 'kill_decision.json'}"
              "\nTo resume it anyway, pass --ignore-kill.", flush=True)
        return

    roots = [a.working_root] + (["/kaggle/input"] if a.working_root != "/kaggle/input" else [])
    ref = find_reference(a.reference_dir, roots)
    ref_hist = ({p: (read_json(Path(ref, p, "history.json")) if is_reference(ref, p) else None)
                 for p in PROTOCOLS} if ref else {})
    write_json(out / "reference.json", {"reference_dir": ref, "kill_epoch": a.kill_epoch,
                                        "kill_margin_pp": a.kill_margin_pp})
    print(f"Variant: {a.variant}   protocols: {', '.join(protocols)}", flush=True)
    print(f"Reference {REF_LABEL} run : {ref or 'NOT FOUND -> no kill rule, training runs to the end'}")
    if ref:
        for issue in reference_config_issues(ref, full_config):
            print("  WARNING (comparison not like-for-like): " + issue)
        for p in protocols:
            v = val_at(ref_hist.get(p), a.kill_epoch)
            print(f"  {REF_LABEL} {p} @E{a.kill_epoch}: {('%.3f%%' % (100 * v)) if v is not None else 'missing'}")
        print(f"  Kill rule: stop only if every protocol is more than {a.kill_margin_pp:.2f} pp behind "
              f"at epoch {a.kill_epoch}.")

    pids = read_json(out / "pids.json") or {}
    procs, cmds, gpus, restarts, attached = {}, {}, {}, {p: 0 for p in protocols}, set()

    def start(p):
        log = open(out / f"{p}.log", "a")
        proc = subprocess.Popen(cmds[p], cwd=repo, env=worker_env(repo, gpus[p]), stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        log.close()
        procs[p], pids[p] = proc, proc.pid
        write_json(out / "pids.json", pids)
        return proc

    for i, p in enumerate(protocols):
        gpus[p] = None if a.cpu_smoke else i
        cmds[p] = [sys.executable, "-m", WORKER, "--config", str(cfg_path), "--protocol", p,
                   "--cache", a.cache, "--outdir", str(out)] + (["--allow-cpu"] if a.cpu_smoke else [])
        if alive(pids.get(p), WORKER):
            print(f"{p}: already running (pid {pids[p]}), attaching")
            attached.add(p)
            continue
        done = read_json(out / p / "result.json")
        if done is not None:
            print(f"{p}: already finished (best {100 * done.get('best_val_accuracy', 0):.3f}%)")
            continue
        proc = start(p)
        print(f"{p}: started pid {proc.pid} on {'CPU' if a.cpu_smoke else 'GPU' + str(i)}")
    write_json(out / "pids.json", pids)
    sys.stdout.flush()

    decided = previous.get("decision")
    last_key, last_print = None, 0.0
    while True:
        rows, running = {}, False
        for p in protocols:
            rows[p] = (read_json(out / p / "history.json") or [], read_json(out / p / "status.json") or {})
            proc = procs.get(p)
            exited = proc is not None and proc.poll() is not None
            vanished = proc is None and p in attached and not alive(pids.get(p), WORKER)
            if exited or vanished:
                code = procs.pop(p).returncode if exited else "unknown (attached worker)"
                attached.discard(p)
                if code != 0 and read_json(out / p / "result.json") is None and decided != "KILL":
                    log_tail = tail(out / f"{p}.log", 300)
                    print(f"\n!!! {p} worker exited with code {code}. "
                          f"Last log lines:\n{tail(out / f'{p}.log')}", flush=True)
                    if "RESOURCE_EXHAUSTED" in log_tail or "out of memory" in log_tail.lower():
                        print("Looks like GPU OOM: use --micro-batch 32 (accumulation 8, same 256 batch) "
                              "and a NEW output folder.", flush=True)
                    deterministic = any(m in log_tail for m in NO_RETRY_MARKERS)
                    if not deterministic and restarts[p] < a.max_restarts:
                        restarts[p] += 1
                        time.sleep(min(30, a.poll_seconds))
                        new = start(p)
                        print(f"{p}: restarted (attempt {restarts[p]}/{a.max_restarts}) as pid {new.pid}; "
                              "it resumes from its last completed epoch.", flush=True)
                    elif deterministic:
                        print(f"{p}: not restarted (the failure would repeat).", flush=True)
            running |= (procs[p].poll() is None) if p in procs else alive(pids.get(p), WORKER)

        parts, progress = [], []
        for p in protocols:
            hist, st = rows[p]
            e = hist[-1]["epoch"] if hist else 0
            v = hist[-1]["val_acc"] if hist else None
            r = val_at(ref_hist.get(p), e) if hist else None
            delta = f" ({100 * (v - r):+.2f} vs ref)" if (v is not None and r is not None) else ""
            best = st.get("best")
            parts.append(f"{p} E{e:02d} {str(st.get('phase', '?'))[:14]:<14} "
                         f"val {_pct(v)}{delta} best {_pct(best)}")
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
            now = {p: val_at(rows[p][0], a.kill_epoch) for p in protocols}
            refv = {p: val_at(ref_hist.get(p), a.kill_epoch) for p in protocols}
            decision, deltas = kill_decision(now, refv, a.kill_margin_pp)
            if decision is not None:
                decided = decision
                write_json(out / "kill_decision.json",
                           {"decision": decision, "epoch": a.kill_epoch, "delta_pp": deltas, "r5": now,
                            "r4": refv, "margin_pp": a.kill_margin_pp, "reference_dir": ref})
                print(f"\n=== EPOCH {a.kill_epoch} DECISION: {decision}  ("
                      + ", ".join(f"{p.upper()} {x:+.2f} pp" for p, x in deltas.items())
                      + f" vs {REF_LABEL}) ===\n", flush=True)
                if decision == "KILL":
                    stop_workers(pids)
                    print(f"Workers stopped: R5 is more than {a.kill_margin_pp:.2f} pp behind {REF_LABEL} "
                          "on every protocol. See kill_decision.json", flush=True)
                    return

        if not running:
            print("\nAll workers finished.")
            for p in protocols:
                res = read_json(out / p / "result.json")
                if res:
                    print(f"  {p}: best {100 * res['best_val_accuracy']:.3f}% @E{res['best_epoch']}")
                else:
                    print(f"  {p}: no result.json (stopped or failed; see {out / (p + '.log')})")
            sys.stdout.flush()
            return
        time.sleep(a.poll_seconds)


if __name__ == "__main__":
    main()

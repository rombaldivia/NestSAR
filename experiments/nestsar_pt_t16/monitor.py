"""Read-only live monitor for a NestSAR-PT run in a Kaggle/Jupyter notebook.

Draws the same two progress rows as the R4 pipeline (streaming/notebook_progress.py:
HTML rows updated in place, no ipywidgets needed), prints one line per finished
epoch with its delta against the R4 reference, and the epoch-10 kill decision.

It only reads files written by the workers and the launcher, so stopping it
never affects training. Load it by path from the notebook kernel:

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "nestsar_pt_monitor", "/kaggle/working/NestSAR_PT_branch/experiments/nestsar_pt_t16/monitor.py")
    mon = importlib.util.module_from_spec(spec); spec.loader.exec_module(mon)
    mon.monitor("/kaggle/working/NestSAR_PT_T16_v1")
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

PROTOCOLS = ("xsub", "xset")
HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rj(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _val_by_epoch(history):
    return {int(r["epoch"]): float(r["val_acc"]) for r in history or []
            if "epoch" in r and r.get("val_acc") is not None}


def _ranges(epochs):
    epochs = sorted(epochs)
    if not epochs:
        return "none"
    parts, start, prev = [], epochs[0], epochs[0]
    for e in epochs[1:] + [None]:
        if e is not None and e == prev + 1:
            prev = e
            continue
        parts.append(f"E{start}" if start == prev else f"E{start}-E{prev}")
        if e is not None:
            start = prev = e
    return ", ".join(parts)


def _pct(v):
    return "--" if v is None else f"{100 * v:.2f}%"


def reference_report(working="/kaggle/working", kill_epoch=10):
    """Print every plain-R4 run the launcher could use; return the one it uses."""
    launch = _load("nestsar_pt_launch_ro", HERE / "launch.py")
    working = Path(working)
    chosen = launch.find_reference(None, str(working))
    print(f"R4 reference runs in {working} (plain R4: {launch.R4_PARAMS:,} params, no 'model' key):")
    roots = sorted({p.parent.parent for p in working.glob("*/*/history.json")})
    shown = 0
    for root in roots:
        cells = []
        for proto in PROTOCOLS:
            rc = _rj(root / proto / "run_config.json") or {}
            if "model" in rc or rc.get("parameters") != launch.R4_PARAMS:
                cells = None
                break
            vals = _val_by_epoch(_rj(root / proto / "history.json"))
            cells.append(f"{proto}: {_ranges(list(vals))}  E1 {_pct(vals.get(1))}  "
                         f"E{kill_epoch} {_pct(vals.get(kill_epoch))}")
        if cells:
            mark = "  <- used" if chosen and Path(chosen) == root else ""
            print(f"  {root.name}{mark}\n      " + "\n      ".join(cells))
            shown += 1
    if not shown:
        print("  (none)")
    if chosen:
        ref = {p: _val_by_epoch(_rj(Path(chosen, p, "history.json"))) for p in PROTOCOLS}
        missing = [p.upper() for p in PROTOCOLS if kill_epoch not in ref[p]]
        if missing:
            print(f"Epoch-{kill_epoch} kill rule INACTIVE: the reference has no "
                  f"{' / '.join(missing)} value at E{kill_epoch}. Training continues to the end.")
        else:
            print(f"Epoch-{kill_epoch} kill rule active.")
    else:
        print(f"No R4 reference found: epoch-{kill_epoch} kill rule inactive.")
    return chosen


def monitor(outdir="/kaggle/working/NestSAR_PT_T16_v1", working="/kaggle/working",
            kill_epoch=10, poll=2.0, max_minutes=None):
    out = Path(outdir)
    ref_dir = reference_report(working, kill_epoch)
    ref = ({p: _val_by_epoch(_rj(Path(ref_dir, p, "history.json"))) for p in PROTOCOLS}
           if ref_dir else {p: {} for p in PROTOCOLS})
    progress = _load("nestsar_notebook_progress", HERE.parent / "nestsar_sm_all_t16" / "streaming"
                     / "notebook_progress.py")
    print()
    bars = progress.make_bars()
    if hasattr(bars[0], "update_status"):   # notebook: HTML rows updated in place
        emit = lambda msg: print(msg, flush=True)
    else:                                    # terminal: keep tqdm bars intact
        from tqdm import tqdm
        emit = tqdm.write
    seen = {p: 0 for p in PROTOCOLS}
    decision_shown, dead_checks, t0 = False, 0, time.time()
    try:
        while True:
            for i, p in enumerate(PROTOCOLS):
                progress.update_bar(bars[i], p, i, _rj(out / p / "status.json") or {})
                history = _rj(out / p / "history.json") or []
                for row in history[seen[p]:]:
                    e, v = int(row["epoch"]), float(row["val_acc"])
                    rv = ref[p].get(e)
                    delta = f"  ({100 * (v - rv):+.2f} pp vs R4 {_pct(rv)})" if rv is not None else ""
                    emit(f"{p.upper()} E{e:02d}  val {_pct(v)}  top5 {_pct(row.get('val_top5'))}  "
                         f"train {_pct(row.get('train_acc'))}  {row.get('epoch_s', 0) / 60:.1f} min{delta}")
                seen[p] = len(history)
            decision = _rj(out / "kill_decision.json")
            if decision and not decision_shown:
                d = decision.get("delta_pp", {})
                emit(f"=== EPOCH {decision.get('epoch')} DECISION: {decision.get('decision')}  "
                     f"(XSUB {d.get('xsub', float('nan')):+.2f} pp, "
                     f"XSET {d.get('xset', float('nan')):+.2f} pp vs R4) ===")
                decision_shown = True
            results = {p: _rj(out / p / "result.json") for p in PROTOCOLS}
            if all(results.values()):
                emit("Training finished: " + "  ".join(
                    f"{p.upper()} best {_pct(r.get('best_val_accuracy'))} @E{r.get('best_epoch')}"
                    for p, r in results.items()))
                break
            pids = _rj(out / "pids.json") or {}
            running = any(_alive(pid) for pid in pids.values())
            dead_checks = 0 if (running or not pids) else dead_checks + 1
            if dead_checks >= 5:
                emit("Workers are not running (stopped by the kill rule, failed, or the session "
                     f"restarted). Logs: {out / 'xsub.log'} | {out / 'xset.log'}")
                break
            if max_minutes is not None and time.time() - t0 > 60 * max_minutes:
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\nMonitor stopped. Training keeps running; run this cell again to re-attach.")
    finally:
        for bar in bars:
            bar.close()

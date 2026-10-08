"""Read-only live monitor for a NestSAR-SA run in a Kaggle/Jupyter notebook.

Two progress bars (XSUB on GPU0, XSET on GPU1) with BEST / val / train acc /
loss, one line per finished epoch with its delta against the R4-FMSE reference,
and the epoch-10 kill decision. Bars are tqdm.notebook bars like the R4
launchers; if ipywidgets is unavailable it falls back to HTML rows that update
in place, and to plain tqdm outside a notebook.

It only reads files written by the workers and the launcher, so stopping it
never affects training. Load it by path from the notebook kernel:

    import importlib.util
    spec = importlib.util.spec_from_file_location("nestsar_sa_monitor", ".../experiments/nestsar_sa_t16/monitor.py")
    mon = importlib.util.module_from_spec(spec); spec.loader.exec_module(mon)
    mon.monitor("/kaggle/working/NestSAR_SA_T16_v1")
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


def _pct(v):
    return "--" if v is None else f"{100 * float(v):.2f}%"


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


# ---------------------------------------------------------------- bars
def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and getattr(ip, "kernel", None) is not None
    except Exception:
        return False


class _TqdmBars:
    """tqdm bars (notebook widgets in Jupyter, text in a terminal)."""

    def __init__(self, notebook):
        if notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        self._tqdm = tqdm
        self.bars = [tqdm(total=1, desc=f"{p.upper()} GPU{i} setup", position=i, leave=True,
                          mininterval=0.5, dynamic_ncols=True) for i, p in enumerate(PROTOCOLS)]
        self.tags = [None, None]
        self.write = (lambda msg: print(msg, flush=True)) if notebook else tqdm.write

    def update(self, i, status):
        bar = self.bars[i]
        phase = str(status.get("phase", "Starting"))
        total = max(int(status.get("total", 1) or 1), 1)
        epoch = int(status.get("epoch", 0) or 0)
        tag = (phase, epoch, total)
        if self.tags[i] != tag:
            bar.reset(total=total)
            self.tags[i] = tag
        bar.set_description_str(f"{PROTOCOLS[i].upper()} G{i} E{epoch:02d} {phase}", refresh=False)
        bar.n = min(max(int(status.get("current", 0) or 0), 0), total)
        bar.set_postfix(_stats(status), refresh=False)
        bar.refresh()

    def close(self):
        for bar in self.bars:
            bar.close()


class _HtmlBars:
    """Fallback without ipywidgets: R4's notebook_progress HTML rows, updated in place."""

    def __init__(self):
        np_mod = _load("nestsar_notebook_progress",
                       HERE.parent / "nestsar_sm_all_t16" / "streaming" / "notebook_progress.py")
        self._np = np_mod
        self.bars = np_mod.make_bars()
        self.write = lambda msg: print(msg, flush=True)

    def update(self, i, status):
        self._np.update_bar(self.bars[i], PROTOCOLS[i], i, status)

    def close(self):
        for bar in self.bars:
            bar.close()


def _stats(status):
    best, best_epoch = status.get("best"), int(status.get("best_epoch", 0) or 0)
    stats = {"BEST": f"{100 * best:.3f}%@E{best_epoch:02d}" if best is not None and best_epoch else "--"}
    if status.get("val_acc") is not None:
        stats["val"] = _pct(status["val_acc"])
    if status.get("train_acc") is not None:
        stats["tr"] = _pct(status["train_acc"])
    if status.get("loss") is not None:
        stats["loss"] = f"{float(status['loss']):.3f}"
    return stats


def make_bars(style="tqdm"):
    notebook = _in_notebook()
    if style == "tqdm":
        try:
            if notebook:
                import ipywidgets  # noqa: F401
            return _TqdmBars(notebook)
        except ImportError:
            pass
    if notebook:
        return _HtmlBars()
    return _TqdmBars(False)


# ---------------------------------------------------------------- reference
def reference_report(working="/kaggle/working", kill_epoch=10):
    """Print the R4-FMSE reference runs the launcher can use; return the one it uses."""
    launch = _load("nestsar_sa_launch_ro", HERE / "launch.py")
    working = Path(working)
    chosen = launch.find_reference(None, str(working))
    meta = launch.meta()
    print(f"{launch.REF_LABEL} reference runs in {working} "
          f"(model {meta.REFERENCE_MODEL}, {meta.REFERENCE_PARAMS:,} params):")
    shown = 0
    for root in sorted({p.parent.parent for p in working.glob("*/*/history.json")}):
        cells = []
        for proto in PROTOCOLS:
            if not launch.is_reference(root, proto):
                continue
            cfg = (_rj(root / proto / "run_config.json") or {}).get("config", {})
            vals = _val_by_epoch(_rj(root / proto / "history.json"))
            cells.append(f"{proto}: {_ranges(list(vals))}  E{kill_epoch} {_pct(vals.get(kill_epoch))}  "
                         f"best {_pct(max(vals.values()) if vals else None)}  "
                         f"(epochs={cfg.get('epochs')}, lr={cfg.get('learning_rate')}, seed={cfg.get('seed')})")
        if cells:
            mark = "  <- used" if chosen and Path(chosen) == root else ""
            print(f"  {root.name}{mark}\n      " + "\n      ".join(cells))
            shown += 1
    if not shown:
        print("  (none) -> no epoch-by-epoch reference; the kill rule is inactive.")
    elif chosen:
        have = [p.upper() for p in PROTOCOLS
                if kill_epoch in _val_by_epoch(_rj(Path(chosen, p, "history.json")))]
        print(f"Epoch-{kill_epoch} kill rule uses: {', '.join(have) if have else 'nothing (inactive)'}")
    return chosen


# ---------------------------------------------------------------- monitor
def monitor(outdir="/kaggle/working/NestSAR_SA_T16_v1", working="/kaggle/working",
            kill_epoch=10, poll=2.0, style="tqdm", max_minutes=None):
    out = Path(outdir)
    ref_dir = reference_report(working, kill_epoch)
    ref = ({p: _val_by_epoch(_rj(Path(ref_dir, p, "history.json"))) for p in PROTOCOLS}
           if ref_dir else {p: {} for p in PROTOCOLS})
    print()
    bars = make_bars(style)
    emit = bars.write
    seen = {p: 0 for p in PROTOCOLS}
    decision_shown, dead_checks, t0 = False, 0, time.time()
    try:
        while True:
            for i, p in enumerate(PROTOCOLS):
                bars.update(i, _rj(out / p / "status.json") or {})
                history = _rj(out / p / "history.json") or []
                for row in history[seen[p]:]:
                    e, v = int(row["epoch"]), float(row["val_acc"])
                    rv = ref[p].get(e)
                    delta = f"  ({100 * (v - rv):+.2f} pp vs ref {_pct(rv)})" if rv is not None else ""
                    emit(f"{p.upper()} E{e:02d}  train {_pct(row.get('train_acc'))}  val {_pct(v)}  "
                         f"top5 {_pct(row.get('val_top5'))}  {row.get('epoch_s', 0) / 60:.1f} min{delta}")
                seen[p] = len(history)
            decision = _rj(out / "kill_decision.json")
            if decision and not decision_shown:
                d = decision.get("delta_pp", {})
                emit(f"=== EPOCH {decision.get('epoch')} DECISION: {decision.get('decision')}  ("
                     + ", ".join(f"{k.upper()} {v:+.2f} pp" for k, v in d.items()) + " vs ref) ===")
                decision_shown = True
            results = {p: _rj(out / p / "result.json") for p in PROTOCOLS}
            if all(results.values()):
                decision = _rj(out / "kill_decision.json")
                if decision and not decision_shown:
                    emit(f"Epoch-{decision.get('epoch')} decision was {decision.get('decision')}: "
                         + ", ".join(f"{k.upper()} {v:+.2f} pp" for k, v in decision.get("delta_pp", {}).items()))
                emit("Training finished: " + "  ".join(
                    f"{p.upper()} best {_pct(r.get('best_val_accuracy'))} @E{r.get('best_epoch')}"
                    for p, r in results.items()))
                break
            pids = _rj(out / "pids.json") or {}
            running = any(_alive(pid) for pid in pids.values())
            dead_checks = 0 if (running or not pids) else dead_checks + 1
            if dead_checks >= 5:
                emit("Workers are not running (stopped by the kill rule, failed, or the session "
                     f"restarted). Logs: {out / 'xsub.log'} | {out / 'xset.log'} | {out / 'launcher.log'}")
                break
            if max_minutes is not None and time.time() - t0 > 60 * max_minutes:
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\nMonitor stopped. Training keeps running; run the cell again to re-attach.")
    finally:
        bars.close()

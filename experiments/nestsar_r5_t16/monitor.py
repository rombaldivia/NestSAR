"""Read-only live monitor for a NestSAR-R5 run in a Kaggle/Jupyter notebook.

Two progress bars (XSUB on GPU0, XSET on GPU1) with BEST / val / train acc /
loss, one line per finished epoch (val, top-5, hand-branch aux accuracy, fast
memory eta, delta vs the R4-FMSE reference at the same epoch) and the epoch-10
kill decision. Only reads files, so stopping it never affects training.

    import importlib.util
    spec = importlib.util.spec_from_file_location("nestsar_r5_monitor", ".../experiments/nestsar_r5_t16/monitor.py")
    mon = importlib.util.module_from_spec(spec); spec.loader.exec_module(mon)
    mon.monitor("/kaggle/working/NestSAR_R5_T16_v1")
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
    """Live, non-zombie worker process (a recycled PID of another program does not count)."""
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
            return b"nestsar_r5_t16.worker" in (proc / "cmdline").read_bytes()
        except (OSError, ValueError, IndexError):
            return False
    return True


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


def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and getattr(ip, "kernel", None) is not None
    except Exception:
        return False


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


class _TqdmBars:
    def __init__(self, notebook, protocols):
        from tqdm.notebook import tqdm as nb_tqdm
        from tqdm import tqdm as txt_tqdm
        tqdm = nb_tqdm if notebook else txt_tqdm
        self.protocols = protocols
        self.bars = [tqdm(total=1, desc=f"{p.upper()} GPU{i} setup", position=i, leave=True,
                          mininterval=0.5, dynamic_ncols=True) for i, p in enumerate(protocols)]
        self.tags = [None] * len(protocols)
        self.write = (lambda msg: print(msg, flush=True)) if notebook else txt_tqdm.write

    def update(self, i, status):
        bar = self.bars[i]
        phase = str(status.get("phase", "Starting"))
        total = max(int(status.get("total", 1) or 1), 1)
        epoch = int(status.get("epoch", 0) or 0)
        tag = (phase, epoch, total)
        if self.tags[i] != tag:
            bar.reset(total=total)
            self.tags[i] = tag
        bar.set_description_str(f"{self.protocols[i].upper()} G{i} E{epoch:02d} {phase}", refresh=False)
        bar.n = min(max(int(status.get("current", 0) or 0), 0), total)
        bar.set_postfix(_stats(status), refresh=False)
        bar.refresh()

    def close(self):
        for bar in self.bars:
            bar.close()


class _PlainBars:
    """Fallback without tqdm/ipywidgets: one printed line per status change."""

    def __init__(self, protocols):
        self.protocols = protocols
        self.last = [None] * len(protocols)
        self.write = lambda msg: print(msg, flush=True)

    def update(self, i, status):
        line = (f"{self.protocols[i].upper()} E{int(status.get('epoch', 0) or 0):02d} "
                f"{status.get('phase', 'Starting')} {status.get('current', 0)}/{status.get('total', 1)} "
                + " ".join(f"{k} {v}" for k, v in _stats(status).items()))
        key = (status.get("phase"), status.get("epoch"), int(status.get("current", 0) or 0) // 50)
        if key != self.last[i]:
            self.last[i] = key
            print(line, flush=True)

    def close(self):
        pass


def make_bars(protocols, style="tqdm"):
    notebook = _in_notebook()
    if style == "tqdm":
        try:
            if notebook:
                import ipywidgets  # noqa: F401
            return _TqdmBars(notebook, protocols)
        except ImportError:
            pass
    return _PlainBars(protocols)


def reference_report(working="/kaggle/working", kill_epoch=10, outdir=None):
    launch = _load("nestsar_r5_launch_ro", HERE / "launch.py")
    meta = launch.meta()
    recorded = _rj(Path(outdir) / "reference.json") if outdir else None
    if recorded is not None:             # exactly the run the launcher's kill rule uses
        chosen = recorded.get("reference_dir")
        kill_epoch = recorded.get("kill_epoch", kill_epoch)
    else:
        chosen = launch.find_reference(None, (working, "/kaggle/input"))
    print(f"{launch.REF_LABEL} reference ({meta.REFERENCE_MODEL}, {meta.REFERENCE_PARAMS:,} params, "
          f"{meta.REFERENCE_STRICT_MFLOPS:.2f} strict MFLOPs):")
    if not chosen:
        print("  (none found) -> no epoch-by-epoch reference; the kill rule is inactive.")
        return None
    for proto in PROTOCOLS:
        if not launch.is_reference(chosen, proto):
            continue
        cfg = (_rj(Path(chosen, proto, "run_config.json")) or {}).get("config", {})
        vals = _val_by_epoch(_rj(Path(chosen, proto, "history.json")))
        print(f"  {Path(chosen).name}/{proto}: {_ranges(list(vals))}  E{kill_epoch} {_pct(vals.get(kill_epoch))}  "
              f"best {_pct(max(vals.values()) if vals else None)}  "
              f"(epochs={cfg.get('epochs')}, lr={cfg.get('learning_rate')}, seed={cfg.get('seed')})")
    return chosen


def monitor(outdir="/kaggle/working/NestSAR_R5_T16_v1", working="/kaggle/working",
            kill_epoch=10, poll=2.0, style="tqdm", max_minutes=None):
    out = Path(outdir)
    cfg = _rj(out / "config.json") or {}
    deadline = time.time() + 30
    while not (out / "pids.json").exists() and time.time() < deadline:
        time.sleep(1)
    pids = _rj(out / "pids.json") or {}
    protocols = (tuple(p for p in PROTOCOLS if p in pids)
                 or tuple(p for p in PROTOCOLS if (out / p).exists()) or PROTOCOLS)
    print(f"Run: {out}  variant={cfg.get('variant', 'full')}  protocols={', '.join(protocols)}")
    ref_dir = reference_report(working, kill_epoch, out)
    ref = ({p: _val_by_epoch(_rj(Path(ref_dir, p, "history.json"))) for p in PROTOCOLS}
           if ref_dir else {p: {} for p in PROTOCOLS})
    print()
    bars = make_bars(protocols, style)
    emit = bars.write
    seen = {p: 0 for p in protocols}
    decision_shown, dead_checks, t0 = False, 0, time.time()
    try:
        while True:
            for i, p in enumerate(protocols):
                bars.update(i, _rj(out / p / "status.json") or {})
                history = _rj(out / p / "history.json") or []
                for row in history[seen[p]:]:
                    e, v = int(row["epoch"]), float(row["val_acc"])
                    rv = ref[p].get(e)
                    delta = f"  ({100 * (v - rv):+.2f} pp vs R4 {_pct(rv)})" if rv is not None else ""
                    emit(f"{p.upper()} E{e:02d}  train {_pct(row.get('train_acc'))}  val {_pct(v)}  "
                         f"top5 {_pct(row.get('val_top5'))}  hand-aux {_pct(row.get('val_aux_hand_acc'))}  "
                         f"eta {row.get('eta', 0):.3f}  {row.get('epoch_s', 0) / 60:.1f} min{delta}")
                seen[p] = len(history)
            decision = _rj(out / "kill_decision.json")
            if decision and not decision_shown:
                d = decision.get("delta_pp", {})
                emit(f"=== EPOCH {decision.get('epoch')} DECISION: {decision.get('decision')}  ("
                     + ", ".join(f"{k.upper()} {v:+.2f} pp" for k, v in d.items()) + " vs R4) ===")
                decision_shown = True
            results = {p: _rj(out / p / "result.json") for p in protocols}
            if all(results.values()):
                emit("Training finished: " + "  ".join(
                    f"{p.upper()} best {_pct(r.get('best_val_accuracy'))} @E{r.get('best_epoch')}"
                    for p, r in results.items()))
                for p in protocols:
                    pc = _rj(out / p / "per_class.json")
                    if pc and pc.get("r4_weak_classes"):
                        emit(f"{p.upper()} weak R4 classes (R4 -> R5): " + ", ".join(
                            f"{k} {100 * v['r4']:.0f}->{100 * v['r5']:.0f}"
                            for k, v in pc["r4_weak_classes"].items()))
                break
            pids = _rj(out / "pids.json") or {}
            running = any(_alive(pid) for pid in pids.values())
            dead_checks = 0 if (running or not pids) else dead_checks + 1
            if dead_checks >= 5:
                emit("Workers are not running (stopped by the kill rule, failed, or the session restarted). "
                     f"Logs: {out / 'launcher.log'} | " + " | ".join(str(out / f'{p}.log') for p in protocols))
                break
            if max_minutes is not None and time.time() - t0 > 60 * max_minutes:
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\nMonitor stopped. Training keeps running; run the cell again to re-attach.")
    finally:
        bars.close()

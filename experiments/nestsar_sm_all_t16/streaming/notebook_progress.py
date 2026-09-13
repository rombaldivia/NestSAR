"""Persistent dual-protocol progress display for Kaggle/Jupyter.

Kaggle can render every tqdm refresh as a new physical line, and some Kaggle frontends do
not load the ipywidgets model manager ("Error displaying widget: model not found").

Notebook mode therefore uses plain IPython HTML display IDs: two rows are displayed once and
updated in place with ``update_display``. This has no widget-model dependency. Plain terminal
execution still uses tqdm.
"""
from __future__ import annotations

from html import escape
import uuid


class _NotebookDisplayBar:
    def __init__(self, protocol: str, gpu: int, HTML, display, update_display):
        self.protocol = protocol
        self.gpu = gpu
        self._HTML = HTML
        self._display = display
        self._update_display = update_display
        self._display_id = f"nestsar-{protocol.lower()}-{gpu}-{uuid.uuid4().hex}"
        self._last_signature = None
        self._closed = False
        self._last_status = dict(phase="Starting", current=0, total=1, epoch=0)
        self._display(self._HTML(self._render(self._last_status)), display_id=self._display_id)

    @staticmethod
    def _progress_bar(current: int, total: int, done: bool, failed: bool) -> str:
        pct = 100.0 * current / max(total, 1)
        color = "#19a974" if done else ("#d64545" if failed else "#4c78ff")
        return (
            "<div style='height:11px;background:#e6e6e6;border-radius:7px;overflow:hidden;"
            "min-width:280px;flex:1 1 38%;'>"
            f"<div style='height:100%;width:{pct:.3f}%;background:{color};transition:width .12s linear'></div>"
            "</div>"
        )

    def _render(self, status: dict) -> str:
        phase = str(status.get("phase", "Starting"))
        epoch = int(status.get("epoch", 0) or 0)
        total = max(int(status.get("total", 1) or 1), 1)
        current = min(max(int(status.get("current", 0) or 0), 0), total)
        done = bool(status.get("done"))
        failed = "failed" in phase.lower()

        best = status.get("best")
        best_epoch = int(status.get("best_epoch", 0) or 0)
        best_text = f"{100 * float(best):.4f}%@E{best_epoch:02d}" if best is not None and best_epoch else "--"

        stats = [f"BEST={best_text}"]
        if status.get("val_acc") is not None:
            stats.append(f"val={100 * float(status['val_acc']):.2f}%")
        if status.get("train_acc") is not None:
            stats.append(f"tr={100 * float(status['train_acc']):.2f}%")
        if "loss" in status:
            stats.append(f"loss={float(status['loss']):.3f}")
        if "rss_gib" in status:
            stats.append(f"RAM={float(status['rss_gib']):.1f}G")
        if "wait_s" in status:
            stats.append(f"wait={float(status['wait_s']):.0f}s")
        if "gpu_s" in status:
            stats.append(f"GPU={float(status['gpu_s']):.0f}s")

        return (
            "<div style='display:flex;align-items:center;gap:10px;width:100%;"
            "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;"
            "padding:3px 0;'>"
            f"<div style='width:235px;white-space:nowrap'><b>{escape(self.protocol.upper())} G{self.gpu} E{epoch:02d}</b> "
            f"<span style='opacity:.68'>{escape(phase)}</span></div>"
            f"{self._progress_bar(current, total, done, failed)}"
            f"<div style='width:82px;text-align:right'>{current}/{total}</div>"
            f"<div style='flex:1 1 auto;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{escape('  '.join(stats))}</div>"
            "</div>"
        )

    def update_status(self, protocol: str, gpu: int, status: dict):
        phase = str(status.get("phase", "Starting"))
        epoch = int(status.get("epoch", 0) or 0)
        total = max(int(status.get("total", 1) or 1), 1)
        current = min(max(int(status.get("current", 0) or 0), 0), total)
        best = status.get("best")
        best_epoch = int(status.get("best_epoch", 0) or 0)
        signature = (
            protocol, gpu, phase, epoch, current, total,
            None if best is None else float(best), best_epoch,
            status.get("val_acc"), status.get("train_acc"), status.get("loss"),
            status.get("rss_gib"), status.get("wait_s"), status.get("gpu_s"),
            bool(status.get("done")),
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature
        self._last_status = dict(status)
        self._update_display(self._HTML(self._render(status)), display_id=self._display_id)

    def close(self):
        # Keep final row visible, matching tqdm(leave=True).
        self._closed = True


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and getattr(ip, "kernel", None) is not None
    except Exception:
        return False


def make_bars():
    if _is_notebook():
        try:
            from IPython.display import HTML, display, update_display
        except ImportError:
            pass
        else:
            # No ipywidgets: only basic notebook display IDs, which Kaggle supports even when
            # the Jupyter widget model manager is unavailable.
            return [
                _NotebookDisplayBar("XSUB", 0, HTML, display, update_display),
                _NotebookDisplayBar("XSET", 1, HTML, display, update_display),
            ]

    from tqdm import tqdm
    return [
        tqdm(
            total=1,
            desc=f"{protocol} GPU{gpu} setup",
            position=gpu,
            leave=True,
            mininterval=0.5,
            dynamic_ncols=True,
        )
        for gpu, protocol in enumerate(("XSUB", "XSET"))
    ]


def update_bar(bar, protocol: str, gpu: int, status: dict):
    if hasattr(bar, "update_status"):
        bar.update_status(protocol, gpu, status)
        return

    # Terminal fallback. Mutate n/total in place; avoid reset(), which can emit extra lines.
    phase = status.get("phase", "Starting")
    total = max(int(status.get("total", 1) or 1), 1)
    epoch = int(status.get("epoch", 0) or 0)
    bar.total = total
    bar.n = min(max(int(status.get("current", 0) or 0), 0), total)
    bar.set_description_str(f"{protocol.upper()} G{gpu} E{epoch:02d} {phase}", refresh=False)

    best = status.get("best")
    best_epoch = int(status.get("best_epoch", 0) or 0)
    stats = {
        "BEST": f"{100 * best:.4f}%@E{best_epoch:02d}" if best is not None and best_epoch else "--"
    }
    for key, label in (("val_acc", "val"), ("train_acc", "tr")):
        if status.get(key) is not None:
            stats[label] = f"{100 * status[key]:.2f}%"
    if "loss" in status:
        stats["loss"] = f"{status['loss']:.3f}"
    if "rss_gib" in status:
        stats["RAM"] = f"{status['rss_gib']:.1f}G"
    if "wait_s" in status:
        stats["wait"] = f"{status['wait_s']:.0f}s"
    if "gpu_s" in status:
        stats["GPU"] = f"{status['gpu_s']:.0f}s"
    bar.set_postfix(stats, refresh=False)
    bar.refresh()

"""Persistent dual-protocol progress display for Kaggle/Jupyter.

Kaggle's captured stdout/stderr can render every tqdm refresh as a new physical line.
This module avoids terminal control sequences in notebooks: two ipywidgets rows are displayed
once and only their widget values are updated afterwards. Terminal execution still uses tqdm.
"""
from __future__ import annotations

from html import escape


class _NotebookBar:
    def __init__(self, protocol: str, gpu: int, widgets):
        self.protocol = protocol
        self.gpu = gpu
        self._widgets = widgets
        self._last_signature = None
        self._closed = False

        self.title = widgets.HTML(
            value=f"<b>{escape(protocol.upper())} G{gpu}</b>",
            layout=widgets.Layout(width="240px"),
        )
        self.progress = widgets.IntProgress(
            value=0,
            min=0,
            max=1,
            description="",
            bar_style="",
            orientation="horizontal",
            layout=widgets.Layout(width="42%", min_width="320px"),
        )
        self.counter = widgets.HTML(
            value="0/1",
            layout=widgets.Layout(width="90px"),
        )
        self.stats = widgets.HTML(
            value="BEST=--",
            layout=widgets.Layout(width="auto", flex="1 1 auto"),
        )
        self.row = widgets.HBox(
            [self.title, self.progress, self.counter, self.stats],
            layout=widgets.Layout(
                width="100%",
                align_items="center",
                gap="8px",
                overflow="hidden",
            ),
        )

    def update_status(self, protocol: str, gpu: int, status: dict):
        phase = str(status.get("phase", "Starting"))
        epoch = int(status.get("epoch", 0) or 0)
        total = max(int(status.get("total", 1) or 1), 1)
        current = min(max(int(status.get("current", 0) or 0), 0), total)

        best = status.get("best")
        best_epoch = int(status.get("best_epoch", 0) or 0)
        best_text = f"{100 * float(best):.4f}%@E{best_epoch:02d}" if best is not None and best_epoch else "--"

        parts = [f"BEST={best_text}"]
        if status.get("val_acc") is not None:
            parts.append(f"val={100 * float(status['val_acc']):.2f}%")
        if status.get("train_acc") is not None:
            parts.append(f"tr={100 * float(status['train_acc']):.2f}%")
        if "loss" in status:
            parts.append(f"loss={float(status['loss']):.3f}")
        if "rss_gib" in status:
            parts.append(f"RAM={float(status['rss_gib']):.1f}G")
        if "wait_s" in status:
            parts.append(f"wait={float(status['wait_s']):.0f}s")
        if "gpu_s" in status:
            parts.append(f"GPU={float(status['gpu_s']):.0f}s")

        # Do not touch widgets when the visible state did not change. This also reduces
        # notebook comm traffic while the launcher polls status files several times/second.
        signature = (protocol, gpu, phase, epoch, current, total, tuple(parts), bool(status.get("done")))
        if signature == self._last_signature:
            return
        self._last_signature = signature

        self.progress.max = total
        self.progress.value = current
        if status.get("done"):
            self.progress.bar_style = "success"
        elif "failed" in phase.lower():
            self.progress.bar_style = "danger"
        else:
            self.progress.bar_style = ""

        self.title.value = (
            f"<b>{escape(protocol.upper())} G{gpu} E{epoch:02d}</b> "
            f"<span style='opacity:.72'>{escape(phase)}</span>"
        )
        self.counter.value = f"<code>{current}/{total}</code>"
        self.stats.value = "<code>" + escape("  ".join(parts)) + "</code>"

    def close(self):
        # Keep the final state visible. Closing an ipywidget removes/invalidates the display,
        # which is the opposite of tqdm(leave=True).
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
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            pass
        else:
            bars = [
                _NotebookBar("XSUB", 0, widgets),
                _NotebookBar("XSET", 1, widgets),
            ]
            panel = widgets.VBox(
                [bar.row for bar in bars],
                layout=widgets.Layout(width="100%", gap="4px"),
            )
            display(panel)
            return bars

    # Plain terminals keep normal carriage-return tqdm behavior.
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

    # Terminal fallback. Avoid reset(), which can print an extra completed line when total or
    # phase changes. Mutate total/n in place and refresh the same physical terminal line.
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

"""Which NTU benchmark the pipeline runs on (``NESTSAR_DATASET``, default ``ntu120``).

    ntu120: 120 classes, protocols xsub / xset   (default; every cache, signature and number unchanged)
    ntu60:  60 classes,  protocols xsub / xview  (PYSKL ``ntu60_3danno.pkl``: splits xsub_*/xview_*, labels 0..59)

The second protocol keeps the augmentation seed offset of xset, so seeds mean the same thing on both datasets.
"""
from __future__ import annotations

import os

DATASET = os.environ.get("NESTSAR_DATASET", "ntu120").lower()
_SPECS = {"ntu120": (120, ("xsub", "xset")), "ntu60": (60, ("xsub", "xview"))}
if DATASET not in _SPECS:
    raise RuntimeError(f"NESTSAR_DATASET must be one of {sorted(_SPECS)}, got {DATASET!r}")
NUM_CLASSES, PROTOCOLS = _SPECS[DATASET]


def protocol_seed(seed, protocol):
    """Same offsets as before for NTU-120 (xset +100000); xview reuses the second-protocol offset."""
    return seed + (0 if protocol == "xsub" else 100000)


def check_cache(labels, splits, where="cache"):
    """Refuse a cache built from another benchmark (e.g. an NTU-120 cache picked up in ntu60 mode)."""
    import numpy as np
    keys = {str(k).lower() for k in splits}
    missing = [f"{p}_{part}" for p in PROTOCOLS for part in ("train", "val") if f"{p}_{part}" not in keys]
    if missing:
        raise ValueError(f"{where} is not an {DATASET} cache: missing splits {missing} (has {sorted(keys)})")
    top = int(np.max(labels)) if len(labels) else -1
    if top >= NUM_CLASSES:
        raise ValueError(f"{where} has label {top} >= {NUM_CLASSES} classes of {DATASET}")

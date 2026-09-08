"""Freeze a matched experiment before looking at final internal groups."""
import math
from ..streaming.launch import validate_config as training_config


def validate_config(config=None):
    config = dict(config or {})
    defaults = dict(seeds=[128, 42, 28], select_fraction=.15, final_fraction=.15,
                    min_class_samples=[10, 3, 3], training={}, audit_first=True,
                    smoke_test=False)
    if set(config) - set(defaults):
        raise ValueError(f"Unknown experiment settings: {sorted(set(config)-set(defaults))}")
    c = dict(defaults, **config)
    if (not isinstance(c["seeds"], list) or not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"])
            or any(type(s) is not int or not 0 <= s < 2**32 - 100001 for s in c["seeds"])):
        raise ValueError("seeds must be distinct nonnegative 32-bit integers")
    for name in ("select_fraction", "final_fraction"):
        if not 0 < c[name] < .5:
            raise ValueError(f"Invalid {name}")
    if sum(c[k] for k in ("select_fraction", "final_fraction")) >= .7:
        raise ValueError("Keep at least 30% of groups for fitting")
    if (len(c["min_class_samples"]) != 3 or
            any(type(n) is not int or n < 1 for n in c["min_class_samples"])):
        raise ValueError("min_class_samples needs positive fit/select/final counts")
    if any(type(c[k]) is not bool for k in ("smoke_test", "audit_first")):
        raise ValueError("smoke_test and audit_first must be booleans")
    if "seed" in c["training"]:
        raise ValueError("Set seeds at experiment level, not training.seed")
    c["training"] = training_config(c["training"])
    for k, v in c["training"].items():
        if isinstance(v, (float, int)) and not math.isfinite(v):
            raise ValueError(f"Nonfinite training setting: {k}")
    if c["training"]["patience"] != 5:
        raise ValueError("This matched experiment uses patience=5 after warm-up")
    if c["training"]["max_train_samples"] or c["training"]["max_val_samples"]:
        raise ValueError("Sample caps are allowed only via smoke_test=True")
    if c["smoke_test"]:
        c["seeds"] = c["seeds"][:1]
        c["training"].update(epochs=2, micro_batch=2, accumulation_steps=2, eval_batch=4,
                             max_train_samples=3, max_val_samples=2, progress_every=1)
    return c

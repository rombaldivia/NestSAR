from copy import deepcopy
import math

PAIRS = [[71, 72], [69, 72], [73, 76], [74, 84], [11, 12],
         [16, 17], [89, 90], [106, 107], [56, 118], [110, 54]]
ARMS = ("t16_mlp", "sequence16_gru", "sequence64_gru")
DEFAULTS = dict(
    pairs=PAIRS, seeds=[128, 42, 28], epochs=60, patience=5,
    warmup_epochs=5, batch_size=64, mlp_width=256, gru_width=128,
    select_fraction=0.20, final_fraction=0.20,
    min_class_samples=[20, 8, 8], bootstrap_samples=2000,
    label_smoothing=0.05, grad_clip=1.0, max_pair_ram_mib=768,
    keep_checkpoints=False,
    trials=[dict(learning_rate=1e-3, weight_decay=1e-4, dropout=0.10),
            dict(learning_rate=5e-4, weight_decay=1e-3, dropout=0.30)],
)


def validate_config(config=None):
    config = config or {}
    if set(config) - set(DEFAULTS):
        raise ValueError(f"Unknown comparison settings: {sorted(set(config)-set(DEFAULTS))}")
    c = dict(deepcopy(DEFAULTS), **deepcopy(config))
    for key in ("epochs", "patience", "warmup_epochs", "batch_size", "mlp_width",
                "gru_width", "bootstrap_samples", "max_pair_ram_mib"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not c["seeds"] or len(c["seeds"]) != len(set(c["seeds"])):
        raise ValueError("Use at least one distinct integer split seed")
    if any(type(s) is not int or not 0 <= s < 2**31 for s in c["seeds"]):
        raise ValueError("Seeds must be integers in [0,2**31)")
    if not c["pairs"] or any(len(p) != 2 or len(set(p)) != 2 or
                            any(type(a) is not int or not 1 <= a <= 120 for a in p)
                            for p in c["pairs"]):
        raise ValueError("pairs must contain distinct one-based NTU action IDs")
    if len({tuple(sorted(p)) for p in c["pairs"]}) != len(c["pairs"]):
        raise ValueError("Duplicate action pairs")
    if len(c["min_class_samples"]) != 3 or any(type(n) is not int or n < 1 for n in c["min_class_samples"]):
        raise ValueError("min_class_samples needs fit/select/final positive counts")
    if not (0 < c["select_fraction"] < 0.4 and 0 < c["final_fraction"] < 0.4):
        raise ValueError("Selection and final group fractions must be in (0,0.4)")
    if not 0 <= c["label_smoothing"] < 1 or not 0 < c["grad_clip"] < float("inf"):
        raise ValueError("Invalid smoothing or gradient clipping")
    if not c["trials"] or len(c["trials"]) > 8:
        raise ValueError("Use 1..8 predefined trials; final scores never select a trial")
    for trial in c["trials"]:
        if set(trial) != {"learning_rate", "weight_decay", "dropout"}:
            raise ValueError("Each trial specifies learning_rate, weight_decay, dropout")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in trial.values()):
            raise ValueError("Trial settings must be finite numbers")
        if not (0 < trial["learning_rate"] <= .1 and 0 <= trial["weight_decay"] <= 1 and 0 <= trial["dropout"] < 1):
            raise ValueError("Invalid trial settings")
    if type(c["keep_checkpoints"]) is not bool:
        raise ValueError("keep_checkpoints must be bool")
    return c

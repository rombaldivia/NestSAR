"""Identity for the controlled D192 + multi-timescale parallel experiment."""
import hashlib
from pathlib import Path

MODEL_NAME = "NestSAR-FULL-PARALLEL-T16-D192-MTS-v1"
EXPECTED_PARAMS = 4_985_132
MODEL_DIM = 192
M4_HALF_LIVES = (1.0, 3.0, 7.0, 15.0)
G4_HALF_LIVES = (1.0, 2.0, 4.0, 8.0)


def validate_d192_config(config):
    """R4 training-safety checks, allowing only the controlled D192 width change."""
    from experiments.nestsar_sm_all_t16.streaming import launch as base

    unknown = set(config) - set(base.DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    c = dict(base.DEFAULTS, **config)
    c["model_dim"] = MODEL_DIM

    for k in (
        "epochs", "patience", "micro_batch", "accumulation_steps",
        "eval_batch", "progress_every",
    ):
        if not isinstance(c[k], int) or c[k] < 1:
            raise ValueError(f"{k} must be a positive integer")

    for k in ("seed", "jitter_shift", "max_train_samples", "max_val_samples"):
        if not isinstance(c[k], int) or c[k] < 0:
            raise ValueError(f"{k} must be a nonnegative integer")

    for k in ("dropout", "label_smoothing", "ema_decay"):
        if not 0 <= c[k] < 1:
            raise ValueError(f"Invalid {k}")

    if not 0 < c["warmup_fraction"] < 1 or c["consistency_temperature"] <= 0:
        raise ValueError("Invalid warmup/temperature")

    if not 0 < c["min_learning_rate"] <= c["learning_rate"] or c["grad_clip"] <= 0:
        raise ValueError("Invalid learning rate/gradient clipping")

    if not 0 <= c["rotation_degrees"] <= 20:
        raise ValueError("Keep yaw augmentation within 0..20 degrees")

    if any(
        c[k] < 0
        for k in (
            "weight_decay", "stream_aux_weight", "consistency_weight",
            "min_delta", "sm_residual_scale", "head_residual_scale",
        )
    ):
        raise ValueError("Loss weights/weight decay/min_delta must be nonnegative")

    if c["prefetch_batches"] not in (1, 2):
        raise ValueError("prefetch_batches must be 1 or 2 to bound host memory")

    # Controlled width ablation: every architecture knob except model_dim
    # stays identical to D128-MTS / Parallel-v2.
    for k in ("spatial_dim", "controller_dim", "fast_rank", "head_rank"):
        if c[k] != base.DEFAULTS[k]:
            raise ValueError(
                f"Keep {k}={base.DEFAULTS[k]} for the controlled D192 experiment"
            )

    if c["model_dim"] != MODEL_DIM:
        raise ValueError(f"Keep model_dim={MODEL_DIM} for this experiment")

    return c


def implementation_identity():
    root = Path(__file__).resolve().parent
    names = (
        "model.py",
        "model_parallel.py",
        "parallel_d192_config.py",
        "streaming/worker.py",
        "streaming/worker_parallel_d192.py",
        "streaming/launch.py",
    )
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    return {
        "version": MODEL_NAME,
        "source_sha256": digest.hexdigest(),
        "model_dim": MODEL_DIM,
        "m4_half_lives": list(M4_HALF_LIVES),
        "g4_half_lives": list(G4_HALF_LIVES),
    }

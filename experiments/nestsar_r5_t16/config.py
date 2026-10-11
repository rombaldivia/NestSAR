"""R5 configuration: the R4 training recipe, R5 model sizes, ablation variants.

The optimisation recipe (epochs, batch 256, AdamW + warmup-cosine, EMA, label
smoothing, auxiliary heads, canonical/augmented consistency, yaw + boundary
jitter augmentation, early stopping) is the R4 streaming recipe unchanged. The
training-only LocalGeometry loss is not used: it measured +0.005 pp on both
protocols (EXPERIMENT_REPORT of the geometry branch).
"""
from __future__ import annotations

from experiments.nestsar_sm_all_t16.streaming import launch as r4_launch

R4_MODEL_KEYS = ("spatial_dim", "model_dim", "controller_dim", "fast_rank", "head_rank",
                 "sm_residual_scale", "head_residual_scale")

MODEL_DEFAULTS = dict(
    spatial_dim=48, spatial_hidden=32, hand_dim=32, hand_hidden=24,
    person_dim=96, model_dim=176, fast_rank=8,
)

# Each ablation removes exactly one R5 fix (see model.py docstring).
VARIANTS = {
    "full": {},
    "no_hand_branch": {"hand_branch": False},
    "coarse_parts": {"fine_parts": False},
    # One direction with a wider state, so the joint sweep keeps roughly the same compute.
    "unidirectional_sweep": {"bidirectional_sweep": False, "spatial_hidden": 48},
    "no_interaction": {"interaction": False},
    "capped_fast_memory": {"fast_mode": "capped"},
    "no_fast_memory": {"fast_mode": "off"},
    # R6-HOPE: R5 front end + HOPE temporal core (self-referential memories, CMS levels 1/2/4/8,
    # CMS MLPs). Outer multi-frequency updates and DMGD-L2 are optimizer keys (outer_cms / dmgd,
    # "auto" = on for hope* variants). Each hope_no_* removes exactly one HOPE component.
    "hope": {"temporal": "hope"},
    "hope_no_selfref": {"temporal": "hope", "selfref": False},
    "hope_no_levels": {"temporal": "hope", "cms_levels": False},
    "hope_no_mlp": {"temporal": "hope", "cms_mlp": False},
    # Full HOPE backbone: Titans short conv instead of the level BiGRUs (the largest R5 block), wider
    # self-referential memory, Titans deep (MLP) memory with momentum surprise, and the HOPE CMS chain
    # (MLPs updated every 1/2/4/8 optimizer steps) in every level.
    "hope_core": {"temporal": "hope", "level_mixer": "conv", "selfref_dim": 64, "deep_memory": True,
                  "memory_hidden": 64, "cms_chain": True},
}

# Outer CMS: parameter tier of each temporal level = its in-clip period (optimizer steps per update).
# Inside a level, a CMS-chain MLP named mlp_p<f> uses its own period f instead.
OUTER_CMS_PERIODS = {"m4": 1, "l2": 2, "g4": 4, "l8": 8}

DEFAULTS = {k: v for k, v in r4_launch.DEFAULTS.items() if k not in R4_MODEL_KEYS}
DEFAULTS.update(MODEL_DEFAULTS)
DEFAULTS.update(variant="full", early_stop_guard_epoch=14)
# Strong training-only augmentation (preprocessing.strong_augmented_features). 0 = the R4 recipe
# (yaw +-8 deg and +-1 frame boundary jitter on one augmented view).
DEFAULTS.update(aug_strength=0.0, aug_clean_prob=0.2, prefetch_workers=1, hand_filter="none", aug_view_degrees=15.0, body_align="none", mix_prob=0.0, aug_pool="")
# HOPE optimizer side: "auto" = on for hope* variants, off otherwise (R5 keeps its exact optimizer).
DEFAULTS.update(outer_cms="auto", dmgd="auto", dmgd_momentum=0.90, dmgd_memory_lr=0.01, dmgd_mix=0.10,
                dmgd_cap=2.0)


def validate_config(config):
    config = dict(config or {})
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    c = dict(DEFAULTS, **config)

    # Reuse the R4 checks for every shared training key (R4 model keys at their R4 values).
    r4_view = {k: v for k, v in c.items() if k in r4_launch.DEFAULTS}
    r4_view.update({k: r4_launch.DEFAULTS[k] for k in R4_MODEL_KEYS})
    r4_launch.validate_config(r4_view)

    if c["variant"] not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
    for k, v in MODEL_DEFAULTS.items():
        if c[k] != v:
            raise ValueError(f"Keep {k}={v}: parameter/FLOP audits are recorded for the default sizes")
    if not isinstance(c["early_stop_guard_epoch"], int) or c["early_stop_guard_epoch"] < 0:
        raise ValueError("early_stop_guard_epoch must be a nonnegative integer")
    if not 0 <= c["aug_strength"] <= 2:
        raise ValueError("aug_strength must be in [0, 2] (0 = R4 augmentation)")
    if c["hand_filter"] not in ("none", "hampel", "smooth", "sun", "sun_smooth"):
        raise ValueError("hand_filter must be none, hampel, smooth, sun or sun_smooth")
    if not 0 <= c["mix_prob"] <= 1:
        raise ValueError("mix_prob must be in [0, 1]")
    if c["body_align"] not in ("none", "yaw"):
        raise ValueError("body_align must be none or yaw")
    if not 0 <= c["aug_view_degrees"] <= 90:
        raise ValueError("aug_view_degrees must be in [0, 90]")
    if not 0 <= c["aug_clean_prob"] <= 1:
        raise ValueError("aug_clean_prob must be in [0, 1]")
    if c["aug_pool"] and not c["aug_strength"] > 0:
        raise ValueError("aug_pool needs aug_strength > 0 (it replaces the live strong augmentation)")
    if c["prefetch_workers"] not in (1, 2, 3, 4):
        raise ValueError("prefetch_workers must be 1..4")
    hope = c["variant"].startswith("hope")
    for key in ("outer_cms", "dmgd"):
        if c[key] == "auto":
            c[key] = hope
        if not isinstance(c[key], bool):
            raise ValueError(f"{key} must be true, false or \"auto\"")
    if not 0 <= c["dmgd_momentum"] < 1:
        raise ValueError("dmgd_momentum must be in [0, 1)")
    if not 0 <= c["dmgd_mix"] <= 0.5:
        raise ValueError("dmgd_mix must be in [0, 0.5]")
    if not (c["dmgd_memory_lr"] > 0 and c["dmgd_cap"] > 0):
        raise ValueError("dmgd_memory_lr and dmgd_cap must be > 0")
    return c


def model_kwargs(config):
    kw = {k: config[k] for k in MODEL_DEFAULTS}
    kw["dropout"] = config["dropout"]
    kw.update(VARIANTS[config["variant"]])
    return kw

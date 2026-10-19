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
_CORE = VARIANTS["hope_core"]
VARIANTS.update({
    # Bidirectional HOPE memory (the clip is read both ways, shared weights): recovers the BiGRU's future context.
    "hope_core_bi": dict(_CORE, memory_bidir=True),
    # Spatial associative memory: attention over the 28 part tokens (both persons) in the tokenizer.
    "hope_core_attn": dict(_CORE, part_attention=True),
    "hope_core_bi_attn": dict(_CORE, memory_bidir=True, part_attention=True),
    # bi + attn with a 2x wider CMS chain (hidden 88): the most parameters per FLOP (+248 K params, +3.7 MFLOPs).
    "hope_core_wide": dict(_CORE, memory_bidir=True, part_attention=True, cms_chain_div=2),
})

# R7: training-only masked reconstruction on top of R5 (zero inference cost; see worker.build_steps).
VARIANTS["full_mr"] = {"masked_recon": True}
VARIANTS["hope_mr"] = {"temporal": "hope", "masked_recon": True}
# R8: replace the joint BiGRU sweep (50% of R5's FLOPs) by a parallel spatial mixer and spend the
# saving on a second temporal BiGRU. Both keep masked reconstruction (R7).
VARIANTS["graph2_mr"] = {"spatial_mixer": "graph", "temporal_stack": 2, "masked_recon": True}
VARIANTS["attn2_mr"] = {"spatial_mixer": "attn", "temporal_stack": 2, "masked_recon": True}
# Graph mixer + MR without the second temporal BiGRU: the T24 candidate at the R5 budget (~59 MFLOPs at T24).
VARIANTS["graph_mr"] = {"spatial_mixer": "graph", "masked_recon": True}
# Ablation of the nested fast memory on the graph_mr candidate: BiGRU levels only (fast_mode off).
VARIANTS["graph_mr_no_fast"] = {"spatial_mixer": "graph", "masked_recon": True, "fast_mode": "off"}
# graph_mr + part-temporal route (each of the 14 parts keeps its own 16-segment trajectory before pooling).
VARIANTS["graph_mr_pt"] = {"spatial_mixer": "graph", "masked_recon": True, "part_temporal": True}
# graph_mr + body DCT block (within-segment trajectory coefficients); needs NESTSAR_BODY_DCT=K > 0.
VARIANTS["graph_mr_dct"] = {"spatial_mixer": "graph", "masked_recon": True, "body_dct": True}
# graph_mr + FR-Head (CVPR 2023): training-only multi-level contrastive feature refinement (fr_head.py).
VARIANTS["graph_mr_fr"] = {"spatial_mixer": "graph", "masked_recon": True, "fr_head": True}
# FR-Head v2: every stage with an axis (joints S+T, parts S+T, frames + m4 T, descriptor) and the original
# squeeze (projection -> training-mode BatchNorm -> ReLU).
VARIANTS["graph_mr_fr2"] = {"spatial_mixer": "graph", "masked_recon": True, "fr_head": True, "fr_version": 2}
# graph_mr + 2 joint-level spatio-temporal blocks (graph aggregation with CTR-GCN refinement + depthwise temporal
# conv, dilations 1 and 2) between the spatial mixer and the parts. Zero-init residuals: equals graph_mr at init.
VARIANTS["graph_mr_st2"] = {"spatial_mixer": "graph", "masked_recon": True, "st_blocks": 2}

# Outer CMS: parameter tier of each temporal level = its in-clip period (optimizer steps per update).
# Inside a level, a CMS-chain MLP named mlp_p<f> uses its own period f instead.
OUTER_CMS_PERIODS = {"m4": 1, "l2": 2, "g4": 4, "l8": 8}

DEFAULTS = {k: v for k, v in r4_launch.DEFAULTS.items() if k not in R4_MODEL_KEYS}
DEFAULTS.update(MODEL_DEFAULTS)
DEFAULTS.update(variant="full", early_stop_guard_epoch=14)
# Strong training-only augmentation (preprocessing.strong_augmented_features). 0 = the R4 recipe
# (yaw +-8 deg and +-1 frame boundary jitter on one augmented view).
DEFAULTS.update(aug_strength=0.0, aug_clean_prob=0.2, prefetch_workers=1, hand_filter="none", aug_view_degrees=15.0, body_align="none", mix_prob=0.0, aug_pool="",
                aug_pool_views_use=0, aug_random_frame=False)   # 0 = draw from every pooled view; K>0 = only the first K (view-count ablation)
# HOPE optimizer side: "auto" = on for hope* variants, off otherwise (R5 keeps its exact optimizer).
DEFAULTS.update(outer_cms="auto", dmgd="auto", dmgd_momentum=0.90, dmgd_memory_lr=0.01, dmgd_mix=0.10,
                dmgd_cap=2.0)
# Masked reconstruction (only used by variants with masked_recon): fraction of the 16 segments of the
# augmented view zeroed at the input, and the weight of the relative-MSE reconstruction loss.
DEFAULTS.update(recon_weight=0.5, recon_mask_ratio=0.3)
# Logit distillation from a precomputed teacher cache (npz with ids/logits/labels, aligned by sample ID).
# Training-only: the student graph and its FLOPs are unchanged. kd_weight 0 = off.
DEFAULTS.update(kd_teacher="", kd_weight=0.0, kd_temperature=1.0)
# FR-Head (only used by variants with fr_head): official defaults --w-cl-loss 0.1,
# --w-multi-cl-loss 0.1 0.2 0.5 1 (low, mid, high, fin), alpha = tau = 0.125, memory momentum 0.9.
DEFAULTS.update(fr_weight=0.1, fr_level_weights=(0.1, 0.2, 0.5, 1.0), fr_alpha=0.125, fr_temperature=0.125,
                fr_momentum=0.9)


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
    from .preprocessing import BODY_DCT, IGNORE_DCT
    if IGNORE_DCT and (BODY_DCT == 0 or VARIANTS[c["variant"]].get("body_dct")):
        raise ValueError("NESTSAR_IGNORE_DCT=1 needs NESTSAR_BODY_DCT=K>0 (the DCT-width cache) and a non-DCT variant")
    if not IGNORE_DCT and bool(VARIANTS[c["variant"]].get("body_dct")) != (BODY_DCT > 0):
        raise ValueError(f"variant {c['variant']} with NESTSAR_BODY_DCT={BODY_DCT}: body_dct variants need "
                         "NESTSAR_BODY_DCT=K>0 and every other variant needs it unset (0)")
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
    if not isinstance(c["aug_pool_views_use"], int) or isinstance(c["aug_pool_views_use"], bool) \
            or c["aug_pool_views_use"] < 0:
        raise ValueError("aug_pool_views_use must be a nonnegative integer (0 = all pooled views)")
    if not isinstance(c["aug_random_frame"], bool):
        raise ValueError("aug_random_frame must be true or false")
    if c["aug_random_frame"] and (not c["aug_strength"] > 0 or c["aug_pool"]):
        raise ValueError("aug_random_frame needs live strong augmentation (aug_strength > 0, no aug_pool)")
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
    if not 0 <= c["recon_weight"] <= 5:
        raise ValueError("recon_weight must be in [0, 5]")
    if not 0 < c["recon_mask_ratio"] < 1:
        raise ValueError("recon_mask_ratio must be in (0, 1)")
    if not (c["dmgd_memory_lr"] > 0 and c["dmgd_cap"] > 0):
        raise ValueError("dmgd_memory_lr and dmgd_cap must be > 0")
    if not 0 <= c["kd_weight"] <= 10:
        raise ValueError("kd_weight must be in [0, 10]")
    if not 0 < c["kd_temperature"] <= 20:
        raise ValueError("kd_temperature must be in (0, 20]")
    if c["kd_weight"] > 0 and not c["kd_teacher"]:
        raise ValueError("kd_weight > 0 needs kd_teacher (path to the teacher logits npz)")
    c["fr_level_weights"] = tuple(float(w) for w in c["fr_level_weights"])
    if len(c["fr_level_weights"]) != 4 or min(c["fr_level_weights"]) < 0:
        raise ValueError("fr_level_weights must be 4 nonnegative weights (low, mid, high, fin)")
    if not 0 <= c["fr_weight"] <= 10:
        raise ValueError("fr_weight must be in [0, 10]")
    if not (0 < c["fr_alpha"] <= 1 and 0 < c["fr_temperature"] <= 1 and 0 <= c["fr_momentum"] < 1):
        raise ValueError("fr_alpha and fr_temperature must be in (0, 1], fr_momentum in [0, 1)")
    if VARIANTS[c["variant"]].get("fr_head") and c["fr_weight"] > 0 and c["mix_prob"] > 0:
        raise ValueError("FR-Head with mix_prob > 0 is not supported (its TP/FN/FP masks need one label per clip)")
    if c["kd_weight"] > 0 and c["mix_prob"] > 0:
        raise ValueError("kd with mix_prob > 0 is not supported (teacher targets are per clip, not per mix)")
    return c


def model_kwargs(config):
    kw = {k: config[k] for k in MODEL_DEFAULTS}
    kw["dropout"] = config["dropout"]
    kw.update(VARIANTS[config["variant"]])
    return kw

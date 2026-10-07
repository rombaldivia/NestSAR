"""NestSAR-RCE30-T16.

Protected FMSE/R4 generalist plus a compact rival-conditioned evidence path.

The evidence path is intentionally designed to add roughly 5 MFLOPs under
1 MAC = 2 FLOPs, matching the Astra prototype budget while implementing the
full architectural changes rather than another scalar or head-only patch.
"""

VERSION = "nestsar-rce30-t16-v1"
MODEL_NAME = "NestSAR-RCE30-T16-v1"

MODEL_IDENTITY = {
    "family": "NestSAR-R4-FMSE",
    "variant": "RCE30-v1",
    "frames": 16,
    "joints": 25,
    "evidence_dim": 40,
    "evidence_blocks": 2,
    "low_rank": 4,
    "candidate_topk": 3,
    "rivals_per_top": 2,
    "candidate_slots": 9,
    "query_modes": 4,
    "base_protected": True,
    "base_frozen_stage2": True,
    "zero_init_correction": True,
    "masked_local_correction": True,
    "training_only_rival_graph": True,
    "supports_fixed_query_ablation": True,
    "supports_rival_conditioned_query": True,
}

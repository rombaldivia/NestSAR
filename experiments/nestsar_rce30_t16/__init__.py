"""NestSAR-RCEX-T16.

Strong post-RCE30 architecture:
- protected frozen FMSE/R4 base;
- stream-oracle-aware candidate generation;
- learned global retrieval for candidate recall;
- rival-conditioned high-resolution evidence;
- no teacher-forced candidate injection during training;
- supervised intervention gate and margin protection.

This branch intentionally changes the full specialist learning problem rather
than tuning one scalar on RCE30.
"""

VERSION = "nestsar-rcex-t16-v1"
MODEL_NAME = "NestSAR-RCEX-T16-v1"

MODEL_IDENTITY = {
    "family": "NestSAR-R4-FMSE",
    "variant": "RCEX-v1",
    "frames": 16,
    "joints": 25,
    "evidence_dim": 40,
    "evidence_blocks": 2,
    "low_rank": 4,
    "base_topk": 3,
    "stream_top1_slots": 4,
    "retrieval_topk": 3,
    "rivals_for_base_top1": 2,
    "candidate_slots": 12,
    "query_modes": 4,
    "base_protected": True,
    "base_frozen_stage2": True,
    "zero_init_correction": True,
    "masked_local_correction": True,
    "teacher_forcing_candidates": False,
    "training_only_rival_graph": True,
    "uses_stream_oracle_candidates": True,
    "uses_global_evidence_retrieval": True,
    "uses_base_descriptor_context": True,
    "uses_stream_disagreement_gate": True,
    "supports_rival_conditioned_query": True,
}

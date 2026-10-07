"""Static architecture-level MAC/FLOP estimate for NestSAR-RCEX.

Convention: 1 MAC = 2 FLOPs.

This is intentionally separate from the old XLA/static NestSAR number.  It is
an auditable operator estimate for the new specialist only.  Run the full
compiler-independent model auditor before publication.
"""

import json

T = 16
J = 25
D = 40
R = 4
MODES = 4
CANDIDATES = 12
CLASSES = 120
STREAMS = 4

parts = {}

# Rich evidence encoder.
parts["family_views"] = 10 * 3 * (3 * 2) * T * J
parts["presence"] = 3 * 4 * T * J
parts["distal_views"] = 3 * (12 * 2) * T * J
parts["evidence_fuse"] = 70 * D * T * J

# Two blocks, three D40->R4->D40 branches over all 400 tokens.
parts["refine_blocks"] = 2 * 3 * (2 * D * R * T * J)

# Six temporal summaries and anatomical mixing.
parts["temporal_summary"] = 6 * D * D * J
parts["anatomical_mix"] = 3 * D * D * J

# Frozen FMSE descriptor context projection.
parts["base_context"] = 112 * D

# Global class-conditioned retrieval over all 120 classes.
parts["retrieval_query"] = CLASSES * D * D
parts["retrieval_key_value"] = 2 * D * D * J
parts["retrieval_attention"] = 2 * CLASSES * J * D
parts["retrieval_hidden"] = CLASSES * (3 * D) * D
parts["retrieval_score"] = CLASSES * D

# Rival-conditioned local scorer over 12 candidate slots.
parts["candidate_query"] = CANDIDATES * MODES * D * D
parts["candidate_key_value"] = 2 * D * D * J
parts["candidate_attention"] = 2 * CANDIDATES * MODES * J * D

candidate_feature_dim = (
    MODES * D
    + D                 # candidate class embedding
    + D                 # frozen FMSE context
    + STREAMS           # per-stream candidate support
    + 1                 # base logit
    + 1                 # retrieval logit
    + 1                 # base gap
)

parts["candidate_hidden_1"] = (
    CANDIDATES
    * candidate_feature_dim
    * (2 * D)
)
parts["candidate_hidden_2"] = (
    CANDIDATES
    * (2 * D)
    * D
)
parts["candidate_delta"] = CANDIDATES * D

# Gate MLP: 7 -> 16 -> 8 -> 1.
parts["gate"] = 7 * 16 + 16 * 8 + 8

macs = int(sum(parts.values()))
mflops = 2.0 * macs / 1e6

report = {
    "model": "NestSAR-RCEX-T16-v1",
    "components_macs": parts,
    "specialist_macs": macs,
    "specialist_mflops_1mac_eq_2flops": mflops,
    "fmse_reference_mflops_astra_accounting": 64.69,
    "estimated_total_mflops_same_reference": 64.69 + mflops,
    "delta_vs_rce30_rival_mflops": mflops - 5.27,
    "note": (
        "Static architecture estimate. The added cost comes mainly from the "
        "120-class global evidence retrieval that improves candidate recall. "
        "Use a compiler-independent full-model operator audit before publication."
    ),
}

print(json.dumps(report, indent=2))

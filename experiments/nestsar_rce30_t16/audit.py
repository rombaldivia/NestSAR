"""Static operator estimate for the RCE30 specialist.

This is a transparent architecture-level MAC/FLOP estimate, not a compiler
profile.  Convention: 1 MAC = 2 FLOPs.

The historical Astra design discussion used an FMSE full-model reference of
~64.69 MFLOPs under one consistent recurrent-iteration accounting scheme.
This file reports the specialist delta separately so it cannot be confused with
the older ~0.0296 XLA/static number or the D112 compiler-independent audit.
"""

import json

T = 16
J = 25
D = 40
R = 4
MODES = 4
C = 9
CLASSES = 120

parts = {}

# Ten families, each P1/P2/relative, each Dense 3->2.
parts["family_views"] = 10 * 3 * (3 * 2) * T * J
parts["presence"] = 3 * 4 * T * J
# Three distal views, each Dense 12->2.
parts["distal_views"] = 3 * (12 * 2) * T * J
parts["evidence_fuse"] = 70 * D * T * J

# 2 blocks * 3 branches * (D->R + R->D) over all 400 tokens.
parts["refine_blocks"] = 2 * 3 * (2 * D * R * T * J)

# Six temporal summaries -> D, once per joint.
parts["temporal_summary"] = 6 * D * D * J
# [z,parent,z-parent] -> D.
parts["anatomical_mix"] = 3 * D * D * J

common_macs = sum(parts.values())

# Fixed-query prototype.
fixed = dict(parts)
fixed["kv_projection"] = 2 * D * D * J
fixed["query_attention"] = 2 * MODES * J * D
fixed["fixed_hidden"] = (MODES * D) * 80
fixed["fixed_delta"] = 80 * CLASSES
fixed_macs = sum(fixed.values())

# Rival-conditioned prototype.
rival = dict(parts)
rival["kv_projection"] = 2 * D * D * J
rival["query_projection"] = C * MODES * D * D
rival["query_attention"] = 2 * C * MODES * J * D
# context(4D) + class embedding(D) + base logit + gap -> D -> 1
rival["candidate_scorer"] = C * (((MODES * D + D + 2) * D) + D)
rival_macs = sum(rival.values())

def summarize(name, table, macs):
    return {
        "variant": name,
        "components_macs": table,
        "specialist_macs": macs,
        "specialist_mflops_1mac_eq_2flops": 2.0 * macs / 1e6,
        "fmse_reference_mflops_astra_accounting": 64.69,
        "estimated_total_mflops_same_reference": 64.69 + 2.0 * macs / 1e6,
        "note": (
            "Static architecture estimate. Run the compiler-independent "
            "operator auditor before publication."
        ),
    }

report = {
    "fixed": summarize("fixed", fixed, fixed_macs),
    "rival": summarize("rival", rival, rival_macs),
}

print(json.dumps(report, indent=2))

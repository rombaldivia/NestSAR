"""NestSAR-R5-T16: the R4 architecture audit, fixed.

One early-fused trunk (instead of four averaged stream models), hands at full
resolution plus a 4x-rate hand branch, a bidirectional joint sweep, explicit
person-person relations, and a self-modifying fast memory without hard caps.
"""

VERSION = "nestsar-r5-t16-v1"
MODEL_NAME = "NestSAR-R5-T16-v1"
MODEL_IDENTITY = {
    "family": "NestSAR-R5",
    "frames": 16,
    "hand_subframes": 64,
    "trunk": "single early-fused trunk (J/B/JM/BM per joint), replaces four averaged stream models",
    "spatial": "bidirectional GRU sweep over joints, 14 parts (thumb / hand tip / wrist+hand kept separate)",
    "hand_branch": "4x-rate hand-relative tokens (tip, thumb, hand, wrist), bidirectional GRU, pooled to 16",
    "interaction": "explicit person-person geometry + layer-scaled cross-person message",
    "temporal": "M4/G4 nested BiMemory + surprise-gated self-modifying fast memory (no hard caps)",
    "removed": "shared capped controller, capped fusion, adaptive rank-2 head, training-only geometry loss",
    "attention": False,
}

# Early-kill reference: R4-FMSE + LocalGeometry trained by the same data pipeline.
REFERENCE_MODEL = "NestSAR-R4-FMSE-T16-v1"
REFERENCE_PARAMS = 1_831_932
REFERENCE_STRICT_MFLOPS = 60.873632
PREFERRED_REFERENCE = "NestSAR_R4_FMSE_LOCAL_GEOMETRY_T16_v1"
DEFAULT_OUTDIR = "/kaggle/working/NestSAR_R5_T16_v1"
DEFAULT_HAND_CACHE = "NestSAR_R5_HAND_CACHE_v1"

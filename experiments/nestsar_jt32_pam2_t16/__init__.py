"""NestSAR-JT32-PAM2-T16: joint-time token-preserving vNext architecture."""

VERSION = "nestsar-jt32-pam2-t16-v1"
MODEL_NAME = "NestSAR-JT32-PAM2-T16-v1"

MODEL_IDENTITY = {
    "family": "NestSAR-vNext",
    "variant": "JT32-PAM2-T16",
    "frames": 16,
    "joints": 25,
    "carrier_dim": 32,
    "blocks": 2,
    "heads": 4,
    "memory_rank": 4,
    "no_early_joint_pooling": True,
    "parallel_temporal_spatial_memory_spectral": True,
    "multiscale_readout": ["T16", "T8", "T4", "hands"],
    "old_m4_router_g4_removed": True,
}

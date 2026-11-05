"""NestSAR R4 FMSE T16 experiment.

Single architectural change from R4:
the joint-motion Spatial-2 input projection is factorized into four motion
components plus an equal-parameter low-rank mixer.

Everything after Spatial-2 is unchanged.
"""

VERSION = "nestsar-r4-fmse-t16-v1"
MODEL_NAME = "NestSAR-R4-FMSE-T16-v1"
MODEL_IDENTITY = {
    "family": "NestSAR-R4",
    "variant": "FMSE-v1",
    "frames": 16,
    "change": "factorized joint-motion spatial input projection",
    "motion_components": ["full_disp", "phase_a", "phase_b", "path"],
    "branch_width": 6,
    "mixer_rank": 4,
    "parameter_matched_to_r4": True,
    "inference_training_only_modules": False,
}

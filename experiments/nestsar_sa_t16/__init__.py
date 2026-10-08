"""NestSAR-SA-T16: R4-FMSE + per-frame spatial joint attention (trained with LocalGeometry)."""

VERSION = "nestsar-sa-t16-v1"
MODEL_NAME = "NestSAR-R4-FMSE-SA-T16-v1"
MODEL_IDENTITY = {
    "family": "NestSAR-R4",
    "variant": "FMSE-v1+SpatialJointAttention-v1+LocalGeometry-v1",
    "frames": 16,
    "change": "pre-norm 2-head softmax attention over the 50 joints (2 persons) of every frame, "
              "skeleton-hop bias, layer-scaled residual, inside each stream's spatial encoder",
    "temporal_attention": False,
    "training_only_geometry": True,
}

# Reference run for the early kill rule: R4-FMSE + LocalGeometry trained by the same pipeline.
REFERENCE_MODEL = "NestSAR-R4-FMSE-T16-v1"
REFERENCE_PARAMS = 1_831_932
PREFERRED_REFERENCE = "NestSAR_R4_FMSE_LOCAL_GEOMETRY_T16_v1"
DEFAULT_OUTDIR = "/kaggle/working/NestSAR_SA_T16_v1"

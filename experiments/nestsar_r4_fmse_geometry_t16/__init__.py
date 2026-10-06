"""NestSAR R4-FMSE + training-only local geometry regularization."""

VERSION = "nestsar-r4-fmse-local-geometry-t16-v1"
MODEL_NAME = "NestSAR-R4-FMSE-T16-v1"

MODEL_IDENTITY = {
    "family": "NestSAR-R4",
    "variant": "FMSE-v1+LocalGeometry-v1",
    "frames": 16,
    "inference_architecture": "NestSAR-R4-FMSE-T16-v1",
    "training_only_geometry": True,
    "subcenters_per_class": 2,
    "descriptor_geometry": True,
    "g4_geometry": True,
    "inference_parameter_delta": 0,
    "inference_flop_delta_from_geometry": 0,
}

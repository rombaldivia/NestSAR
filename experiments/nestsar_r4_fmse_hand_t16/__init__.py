"""NestSAR R4-FMSE + pre-pooling distal hand-relation motion residual."""

VERSION = "nestsar-r4-fmse-hand-t16-v1"
MODEL_NAME = "NestSAR-R4-FMSE-HAND-T16-v1"

MODEL_IDENTITY = {
    "family": "NestSAR-R4",
    "variant": "FMSE-HandRelations-v1",
    "frames": 16,
    "base": "NestSAR-R4-FMSE-T16-v1",
    "change": "pre-pooling distal hand/wrist motion relations inside joint-motion Spatial-2",
    "relation_pairs_0based": [[7,6],[21,6],[22,6],[11,10],[23,10],[24,10]],
    "relation_input_dim": 72,
    "relation_hidden_dim": 8,
    "relation_output_dim": 24,
    "relation_residual_scale": 0.10,
    "extra_parameters": 800,
    "inference_parameters": 1832732,
}

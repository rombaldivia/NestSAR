# Frozen G4 validation audit: 24 and 32 raw-frame windows

This audit evaluates an **existing** P2, P2 with training-only attention, or
G4 Temporal Moments checkpoint. It does not train, save weights, install packages, or edit the raw
skeleton. Both NTU120 protocols run on separate GPUs. An existing pair of
`best.msgpack` and `best.json` files is required. The runner reuses a compatible
cache if available, including an attached cache under `/kaggle/input`. When a
cache is absent, it builds one from the attached `ntu120_3danno.pkl` without
training and reports preparation on the same two tqdm rows. This preparation
needs sufficient host RAM and free disk. A cache for a different experiment is
left intact.

For each official validation clip, compare the cached original `[16,750]`
tokens with tokens rebuilt from centered **24** and **32** raw-frame windows.
The crop/padding step matches the supplied Graphormer notebook: crop the middle
when a clip is longer; repeat the final raw frame when shorter. The validation
path adds no random temporal jitter. This notebook is a Graphormer experiment,
so these modes should not be described as an exact CD-Former implementation.

Each variant then uses the exact corrected NestSAR preprocessing (including
its two-person policy and joint-validity mask) to build **16** tokens of 750
features, independently of window length. The G4 inference graph and FLOPs
are unchanged. The model has never been trained on these 24/32 window
distributions, so the scores are a frozen-model diagnostic. We do not copy the
Graphormer notebook's first-person selection or per-frame z-score, which would
also change the features seen by the frozen G4 model.

The inference graph matches each saved checkpoint. The G4 model is copied
exactly from `experiment/nestsar-g4-temporal-moments-t16` into
`model_g4_moments.py`, leaving the P2 model intact.

```python
from experiments.nestsar_sm_all_t16.validation_reframe import launch_all
results = launch_all({
    "checkpoint_root": "/kaggle/working/NestSAR_G4_TEMPORAL_MOMENTS_T16_FULL_OFFICIAL",
    "cache": "/kaggle/working/NestSAR_G4_TEMPORAL_MOMENTS_T16_CACHE",
    "dataset": None,  # Auto-find an attached NTU120 pickle if cache is absent.
    "outdir": "/kaggle/working/NestSAR_G4_VALIDATION_REFRAME_24_32_v1",
})
```

The parent launches one worker per GPU with independent CUDA visibility and
two live tqdm rows for each available checkpoint pair. `launch_all` discovers
every compatible P2, P2 with training-only attention, or G4 Temporal Moments
XSUB/XSET pair under `/kaggle/working` and `/kaggle/input`. It requires both
protocol weights *before* building a cache. Other model architectures need
their matching inference code and saved checkpoints; the attached Graphormer
notebook alone does not include those weights. The audit requires
matching model, parameter count, preprocessing version, and pipeline version.
Each worker
reads its EMA checkpoint; mismatched preprocessing/cache signatures fail before
inference. No training
split samples are scored. Results include the original/crop accuracies, top-5,
paired fixed/broken predictions, counts of cropped and padded clips, per-class
recalls, and sample predictions.
`best.json` is used to verify full-validation original-view score when present.

This is an **inference distribution-shift audit**: a lower frozen-model crop
score cannot rule out higher accuracy from a model trained on the crop. Scores
selected on these official held-out splits are diagnostic and should not be
presented as untouched final test results.

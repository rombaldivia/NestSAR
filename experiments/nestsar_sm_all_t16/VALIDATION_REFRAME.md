# Frozen validation reframe audit

This audit evaluates an **existing** P2 (or P2 with training-only attention)
checkpoint. It does not train, save weights, install packages, or edit the raw
skeleton. Both NTU120 protocols run on separate GPUs. An existing pair of
`best.msgpack` and `best.json` files is required. The runner reuses a compatible
cache if available, including an attached cache under `/kaggle/input`. When a
cache is absent, it builds one from the attached `ntu120_3danno.pkl` without
training and reports preparation on the same two tqdm rows. This preparation
needs sufficient host RAM and free disk. A cache for a different experiment is
left intact.

For each official validation clip, compare the cached original `[16,750]`
tokens with tokens rebuilt from center-cropped 64, 32 and 16 raw-frame windows.
Short clips retain their original length; no repeated or zero-padded raw frames
are introduced. Each variant uses the exact original P2 preprocessing and
inference graph. The CD-Former paper also uses a first-person input and
frame-wise normalization; neither is copied because both would separately
change the distribution seen by the frozen P2 checkpoint.

```python
from experiments.nestsar_sm_all_t16.validation_reframe import launch
results = launch({
    "checkpoint_root": "/kaggle/working/NestSAR_SM_ALL_T16_PERSON_AWARE_P2_v3",
    "cache": "/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3",
    "dataset": None,  # Auto-find an attached NTU120 pickle if cache is absent.
    "outdir": "/kaggle/working/NestSAR_P2_VALIDATION_REFRAME_v1",
})
```

The parent launches one worker per GPU with independent CUDA visibility and
two live tqdm rows. It locates an attached checkpoint pair when the preferred
root is absent and requires those weights *before* building a cache. Each worker
reads its EMA checkpoint; mismatched preprocessing/cache signatures fail before
inference. No training
split samples are scored. Results include the original/crop accuracies, top-5,
paired fixed/broken predictions, per-class recalls, and sample predictions.
`best.json` is used to verify full-validation original-view score when present.

This is an **inference distribution-shift audit**: a lower frozen-model crop
score cannot rule out higher accuracy from a model trained on the crop. Scores
selected on these official held-out splits are diagnostic and should not be
presented as untouched final test results.

# True parent-relative motion at T16

This experiment replaces the bone-motion proxy `abs(sum(abs(dj))-sum(abs(dp)))`
with `sum(abs(dj-dp))`, computed on native adjacent frames before segmentation.
Here `dj` and `dp` are signed child/parent displacements. Each xyz component is
accumulated separately. Both joints must be valid at both endpoints. Missing
tracks and self-parent root bones contribute zero. All segment-boundary
transitions are retained exactly once, including clips shorter than 16 frames.

This addresses one specific information loss. It does not preserve every
trajectory, movement direction, orientation, contact, or object cue. Summed
absolute travel remains sensitive to tracking noise. Accuracy gains, including
gains on hand-related classes, require retraining; no NTU result is claimed here.

## Fixed input and model

The input stays `[B,16,750]` and the parameter count stays **1,826,556**.
M4/G4 widths, streams, classifier, losses, and optimizer are retained.

| Channels per joint | Packed meaning |
|---|---|
| 0:3 | Representative pose |
| 3:6 | Full signed displacement |
| 6:9 | Early signed displacement |
| 9:12 | True parent-relative path |
| 12:15 | Total joint path |

Late displacement is redundant (`full = early + late`). The model reconstructs
`late = full - early` **before** the independently learned gamma/beta modulation.
Float32 reconstruction is numerically close to the original late block; it is
not promised to be bitwise identical. The true relative path uses the existing
shared path-channel gain. No additive shift is added to relative travel.

The two matched arms both unpack this interface and use identical visibility
handling. `proxy` keeps the old bone path; `relative` supplies the true path.
Only the BM path block differs at stream construction. Both arms initialize
with identical parameter trees and values. The default model interface remains
`motion_path="legacy"` for existing P2 callers.

Both arms also use a necessary numerical correction in fast-memory key/query
normalization: `sqrt(max(sum(x*x), eps**2))` instead of `max(norm(x), eps)`.
The old expression has undefined derivatives at zero and produced nonfinite
gradients in the full-model padded-batch smoke test. The corrected expression
retains the same forward normalization and parameter shapes while keeping
zero-padding gradients finite. The static-unrolled auditor uses the same fix.

Do not feed these packed tokens to an old model, or P2 tokens to the new modes.
New experiment checkpoints carry their mode, source/code signature and split
hash. Read **top-level `ema_params` from `best.msgpack`**; `last.msgpack` instead
contains resumable optimizer/EMA/RNG state under `state`.

Neural inference stays near the P2 model's reported **0.0294 GFLOPs**, with a
small arithmetic change from unpacking and path construction. `audit.json`
measures all three interfaces on the actual Kaggle GPU using the same static
unrolling convention and checks forward equivalence. Raw preprocessing is
excluded and scales with clip length. This is the P2 SM-ALL variant, not the
older 20.63-MFLOP Hand-M4/G4 variant.

CPU XLA cost analysis uses a different backend and is not interchangeable with
the earlier GPU report. Compare legacy/proxy/relative **within the same audit**
and retain its backend, JAX version and counting convention with any FLOP claim.

## Matched full-class experiment

The default run trains both arms for seeds **128, 42, 28**. For each arm/seed,
XSUB runs on GPU0 and XSET on GPU1 simultaneously. These are **120-output,
120-class** experiments, not the earlier binary nonlinear-classifier diagnostic.

All official training examples enter one internal group partition:

* Fit groups train the model.
* Selection groups choose the best EMA checkpoint and stop training.
* Final groups evaluate that checkpoint once after selection.

XSUB partitions whole subjects; XSET partitions whole setups. Selection/final
fractions default to 15% each, with at least two groups apiece. Class support
is checked for every partition. Seeds and all partitions are shared by both
arms. Splits are saved with sample indices, group IDs and hashes. Official
held-out examples are blocked from training and inference in this experiment.
Stateless per-clip cache construction may process all examples; it fits no
dataset or split statistics.

Use the final-group paired difference, per-class recall, corrected errors and
damaged correct predictions to judge this feature. Selection accuracy is
labelled `BEST_INNER` on both persistent notebook bars. These are **internal
scores, not official XSUB/XSET benchmark results**. Repeated partitions can
overlap; reported SD is descriptive, not an independent-sample confidence
interval. Do not use final-group results for repeated hyperparameter tuning.

Training defaults match the P2 schedule: up to 60 epochs; 64 examples per
microbatch, four accumulation steps; validation batch 256; fresh yaw/segment
jitter; EMA; patience **5 after warm-up**. Rotation and jitter regenerate all
motion summaries from the same augmented raw skeleton in one pass. No teacher,
specialist, class weighting or extra descriptor is added to this ablation.

## Kaggle

Enable **GPU T4 ×2** and Internet. Attach your P2 v3 cache or
`ntu120_3danno.pkl`. Run `kaggle_relative_motion.py` in the notebook kernel so
it owns both existing-style TQDM bars; workers write logs/status files only.
The launcher probes and reuses Kaggle's GPU runtime; it does not install CUDA
packages, create a pip environment, or delete old runtime directories.

Settings accepted through `NESTSAR_RELATIVE_SETTINGS`:

```python
{
    "cache_dir": None,  # Auto-detect an attached or existing P2 v3 cache.
    "auxiliary_dir": "/kaggle/working/NestSAR_TrueRelative_PathCache_v1",
    "outdir": "/kaggle/working/NestSAR_TrueRelative_T16_v1",
    "dataset": None,  # Only needed if the P2 cache must be built.
    "config": {
        "seeds": [128, 42, 28],
        "smoke_test": False,
        "audit_first": True,
        "training": {"epochs": 60, "patience": 5}
    }
}
```

Normal runs have no sample cap. `smoke_test=True` deliberately uses only three
fit/two selection examples for two epochs, marks results as smoke, and appends
`_smoke` to the output. It checks execution, not accuracy.

## Storage, resume and reports

The original P2 raw/canonical arrays are shared read-only by both workers.
One auxiliary float32 memmap adds `[N,16,2,25,3]`, approximately **1.019 GiB**
for 113,945 clips. Only batch-sized packed copies and up to two prefetched
batches per worker are allocated. There is no full augmented-feature cache.
Preparing a fresh source P2 cache still requires the existing pickle-loading
step; attaching that cache avoids repeating it.

The auxiliary builder commits progress after flushing/fsyncing each chunk and
verifies chunk hashes on interrupted-build resume and once per launcher reuse.
Completed shape,
dtype, file size and source signature are checked. Different output code,
configuration, data, mode or split is rejected for resume. Atomic checkpoint
publication preserves optimizer, EMA and RNG state; best aliases can be repaired.
Rerun the same cell/output to continue. Use a new output directory when changing
settings. Preserve the source cache, auxiliary cache and output as Kaggle
datasets/output before ending a session if you need them later.

`results.json` and `scores.csv` contain both arms' best selection and final
scores. Each `seed_*/{proxy,relative}/{xsub,xset}/` contains history, best/last
checkpoints, `evaluation.json`, and fit/select/final prediction NPZ files with
120×120 confusion matrices. Timings separate preparation, data wait, transfer,
compilation, warm-up and synchronized model execution.

CPU tests verify the lost-information example, native transition coverage,
padding/gaps, augmentation, packing, parameter identity, model behavior and
checkpoint training/reload. A concurrent two-worker synthetic integration is
provided in `smoke_cpu.py`; it does not measure T4 speed or real NTU accuracy.

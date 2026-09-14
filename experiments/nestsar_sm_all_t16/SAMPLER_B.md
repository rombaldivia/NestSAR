# Sampler B: local movement with the existing T16 SM-ALL network

Based on `experiment/nestsar-sm-all-person-aware-p2-t16` at
`0de94e304d15f2fe7ce591471f188f4a574aca60` (the latest full-official runner
and display-ID progress fix inspected on 2026-09-14).

This is **sampler B only**. It trains one XSUB model on GPU0 and one XSET model
on GPU1. No second seed, internal fold, specialist, distillation, or new neural
layer is enabled. Accuracy gains have not been established by this change.

## Unchanged training settings

| Setting | Value |
| --- | --- |
| XSUB | 63,026 training / 50,919 official held-out |
| XSET | 54,468 training / 59,477 official held-out |
| Seed | 128 |
| Epoch limit / patience | 60 / 5 after warm-up |
| Effective batch | 256, accumulated as 64 × 4 |
| Evaluation batch | 256 |
| Learning rate | 0 → 0.0006 warm-up, cosine to 0.00002 |
| Warm-up fraction | 0.08 |
| AdamW decay / gradient clip | 0.03 / 1.0 |
| EMA / dropout / label smoothing | 0.995 / 0.10 / 0.05 |
| Stream auxiliary / consistency weights | 0.15 / 0.08 |
| Consistency temperature | 1.0 |
| Fresh augmentation | yaw ±8°; segment-edge jitter ±1 frame |
| Prefetch / progress interval | 2 batches / 5 steps |
| Input / expected model parameters | `[16,750]` / 1,826,556 |

All other model values are inherited unchanged from `streaming.launch.DEFAULTS`.
The trainer also guards zero-weight **batch padding**: before the training forward,
unused batch slots receive a copy of a real sample but retain mask zero. The CPU
integration check exposed undefined gradients of the existing fast-weight norm
at an entirely zero padded sample. This guard avoids that case without changing
any real input, valid-sample gradient, sample count, model parameter or inference
operation. It applies equally to A and B in this source. Missing joints/people
within a real sample are not filled. Evaluation padding remains unchanged.

Both sample caps must be zero. Before fitting scales or training, the B preparation
process verifies the exact full split counts, disjointness, uniqueness and coverage
of all 113,945 samples. These are official protocol runs, not the earlier internal
grouped comparisons. Official held-out accuracy still selects the best EMA epoch,
as in the preceding run; it is not used to fit sampler scales.

## What the sampler changes

Only the 3 pose channels out of each joint's 15 channels can change. The other
12 channels (displacement, first/second-half displacement and coordinate-wise path)
retain exactly the existing values, transition coverage, normalization and jitter.
The network/model file is unchanged; its neural parameter and operation counts
are unchanged. Raw preprocessing adds O(T) work outside the neural FLOP count.

Eight offsets are computed at **every original raw timestamp**:

| Link | Child joint | Reference joint |
| --- | ---: | ---: |
| Left hand tip / wrist | 21 | 6 |
| Left thumb / wrist | 22 | 6 |
| Right hand tip / wrist | 23 | 10 |
| Right thumb / wrist | 24 | 10 |
| Left foot / ankle | 15 | 14 |
| Left ankle / hip | 14 | 12 |
| Right foot / ankle | 19 | 18 |
| Right ankle / hip | 18 | 16 |

Indices are zero-based. The dataset contains coarse hand-tip/thumb/foot landmarks,
not separately tracked fingers or toes. For link offset `r[t] = child[t]-reference[t]`,
the score uses `norm(r[t]-r[t-1])`. All four joints at the two endpoints must be valid.
The original raw validity mask survives centering and rotation.

For each protocol, all eligible transitions from **its training split only** feed
eight fixed-memory log histograms. Each link's positive-motion 90th percentile
defines its scale. The estimate uses a bin upper edge, within about 4.6% bin width.
Links with fewer than 32 positive observations or a scale below 0.00001 metre per
transition are disabled. Counts, overflow counts, scales and the training-membership
hash are saved. A sample that belongs to XSUB validation can legitimately belong
to XSET training; the calibrations remain separate.

After scaling, each link contributes at most 3. A large excursion immediately
followed by an almost cancelling return is excluded from the sampling score.
Other isolated edges are limited by their adjacent support; unsupported single
edges contribute zero. Hands and legs each receive half the score, with fixed
equal link weights within each group. Missing links cannot increase the remaining
links' weights. The score is assigned to the two adjacent frames.

Within each unchanged temporal segment, each person retains the baseline's frames
with the **maximum valid-joint count**. Among those candidates, B chooses the highest
reliable local-motion score, breaking ties toward the midpoint. Below the fixed
0.05 score threshold it uses the exact baseline midpoint rule. All 25 joints of
that person come from the chosen frame. Actor ordering and the existing independent
per-person timestamps remain unchanged; absent people stay zero.

These conservative tracking guards are heuristics: an exceptionally brief genuine
movement can also be suppressed. They affect frame selection only, never edit the
skeleton coordinates or the complete motion summaries. The thresholds are fixed
before the run, not tuned against held-out scores.

## Kaggle execution

Enable **GPU T4 ×2** and Internet; attach `ntu120_3danno.pkl`. Use the supplied
commit-pinned cell to call `kaggle_sampler_b.py` with `NESTSAR_B_SETTINGS`. From an
already verified checkout at `ROOT`, the call is:

```python
import runpy
nestsar_results = runpy.run_path(
    str(ROOT / "experiments/nestsar_sm_all_t16/kaggle_sampler_b.py"),
    init_globals={"NESTSAR_B_SETTINGS": {
        "dataset": None,
        "outdir": "/kaggle/working/NestSAR_SamplerB_T16_Seed128_v1",
        "cache_dir": "/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3",
    }},
)["nestsar_results"]
```

The parent keeps the latest runner's two persistent IPython display rows in Kaggle,
with `BEST=...%@E...` for each protocol; terminal execution uses tqdm. No ipywidgets
manager is required and workers print no tqdm lines. Setup/cache/calibration phases
use the same two rows. CUDA/JAX packages are never automatically installed. The
runner probes both existing GPUs in fresh subprocesses before importing the model.

The existing P2-v3 raw and canonical cache is reused, or built once in a CPU child
if missing. An incompatible old cache is rejected with an explicit message; nothing
is automatically deleted to make room. B adds two read-only **pose-only** float32
overlays (about 2.04 GiB combined) and 28 MiB of selected-frame indices. It does not
duplicate the roughly 10.2 GiB of full tokens that two new protocol caches would
need. Host memory remains bounded by batch buffers and file-backed mappings; raw
pickle data is not loaded by either training worker. Validation reads cached poses
and motion channels; it never recomputes raw-frame scores each epoch.

The augmented training view is rebuilt each epoch using the same sampler with its
frozen protocol calibration. Canonical and augmented views therefore both use B.
The canonical overlay is built once, even if training is resumed.

Re-running the same cell resumes committed last checkpoints, restores the best EMA
aliases, and reuses finished caches. A new output directory prevents loading an A
checkpoint into B. Dataset, split, policy and calibration identities are validated
before reuse and included in checkpoint signatures. Stops during cache creation
leave no completion marker, so incomplete overlays are rebuilt on the next run.

## Saved evidence

Under the output directory:

- `best_scores.json` and `results.json`: separate XSUB/XSET best scores and epochs.
- `xsub/` and `xset/`: `status.json`, `history.json`, `best.msgpack`, `last.msgpack`,
  and full run/checkpoint metadata. History records all sample counts plus data wait,
  preprocessing service, transfer, compilation and warmed GPU execution times.
- `sampler_b/calibration_<protocol>.json`: frozen scales, fitted training counts and
  split/source identity. Calibration is also embedded in each best checkpoint.
- `sampler_b/sampling_diagnostics.json`: per-split motion selections, midpoint fallbacks,
  changed selections and suspected tracking frames. This diagnoses whether B actually
  changes representative poses; no improvement is inferred from these counts alone.
- `sampler_b/indices_<protocol>.npy`: selected raw timestamp per sample, segment and
  person (`-1` for an absent person).
- `sampler_b/preparation_timing.json`: calibration and pose-cache CPU preparation time.
- `compute_audit.json`: unchanged model's complete neural inference audit on your GPU.

For standalone inference, use the matching checkpoint's calibration:

```python
from experiments.nestsar_sm_all_t16 import preprocessing_corrected as pp
from experiments.nestsar_sm_all_t16.sampler_b import SamplerB

sampler = SamplerB(checkpoint["sampler_b"]["calibration"])
tokens = pp.features(pp.ordered_raw(raw_keypoints, "MTVC"), pose_selector=sampler)
```

Do not evaluate a B checkpoint with the old midpoint preprocessor. Keep the same
calibration at inference; fitting it on a new validation/test set would change the
experiment. Review weak-class results after this matched full-training run before
adding further interventions.

## Local verification

**103 regression tests passed.** The real-model CPU check completed two epochs on
each protocol concurrently and resumed both checkpoints successfully. See
[`sampler_b_validation.json`](sampler_b_validation.json) for the saved evidence.
No full NTU120 retraining or dual-T4 execution was available in this workspace.

```bash
python -m pytest -q experiments/nestsar_sm_all_t16/test_sampler_b.py \
  experiments/nestsar_sm_all_t16/test_preprocessing_corrected.py \
  experiments/nestsar_sm_all_t16/streaming/tests
python -m experiments.nestsar_sm_all_t16.streaming.smoke_cpu \
  --sampler-b --outdir /tmp/nestsar_sampler_b_smoke
```

The CPU smoke fixture is explicitly synthetic and separate from the Kaggle entry
point. It exercises the real model, two concurrent workers, checkpoint/EMA persistence
and resume. It is not an NTU120 accuracy result or a dual-T4 speed measurement.

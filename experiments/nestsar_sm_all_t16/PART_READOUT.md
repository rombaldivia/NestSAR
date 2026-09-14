# Nonlinear part readout, T16

This is an optional architectural ablation on the verified sampler B branch.
It borrows Ta-CNN's idea of learning combinations of joint features before
pooling. It does **not** reproduce Ta-CNN, CAG or VAG, and it has not demonstrated
an NTU120 accuracy improvement.

## What changes

Each of the four spatial encoders already produces a 24-dimensional feature
for each joint using the existing joint memory. Its ten fixed part averages
are retained. Before discarding joint-level detail, the new residual:

1. Projects the joint features to eight channels using one shared linear map.
2. Forms two learned, signed joint combinations within each existing body part.
3. Centers those combinations using valid-joint means, then applies GELU.
4. Projects the resulting 16 features back to 24 and adds 0.1 times the result
   to the existing part descriptor.

Different opposing joint features can have identical averages but different
learned combinations. This is a representational capability, not evidence of
improved recognition. The features already passed through the joint memory;
the module does not reconstruct raw trajectories or missing finger joints.

All grouping weights are static learned parameters, not input-dependent
attention scores. Membership uses the existing ten anatomical groups. This
first ablation does not include learned cross-part grouping or new temporal
processing. M4, G4, stream fusion, classifier and T16 x 750 inputs are retained.
**The deployed spatial encoder changes.** This is not an exactly unchanged
architecture or training-only distillation.

## Safety and initialization

Missing joints contribute neither features nor normalization weight. Absent
people remain absent. A part needs two valid joints to produce a relational
residual. The output projection starts at zero, so common parameters and
initial predictions match the baseline with the same seed. Output weights
learn first; the other new weights begin receiving gradients after that.

The base model has 1,826,556 parameters. The new model has **1,829,060**, an
increase of **2,504 (0.137%)**. The runtime computes baseline and proposed-model
FLOPs with static-unrolled XLA on the same device before training. See
`compute_audit.json` for the actual device, convention and difference. CPU
audits are not T4 measurements; preprocessing is excluded from neural FLOPs.

The paired CPU audit on JAX 0.7.2 measured 68,797,520 baseline versus 73,111,440
proposed FLOPs per clip: +4,313,920, or +6.27%. Both statically unrolled models
matched their scan-based forward outputs exactly in this test. These CPU
numbers must not replace the historical T4 audit or be combined with its
29.07 MFLOPs baseline; compare values from the same device and convention.

## Kaggle

Select two T4 GPUs, enable Internet and attach `ntu120_3danno.pkl`. Run the
single cell in `kaggle_run_part_readout.py`. It checks out an immutable revision.
It uses the existing host JAX environment; it does not download another CUDA
runtime. The two persistent notebook progress rows, per-protocol BEST scores,
timing, checkpoint recovery and terminal tqdm behavior come from the previous
sampler B launcher.

The default experiment keeps the previous settings: seed 128, 60 epochs,
patience 5 after warm-up, microbatch 64 x accumulation 4, eval batch 256,
AdamW 6e-4 to 2e-5, EMA .995, weight decay .03, dropout .10, label smoothing
.05, fresh augmentation, two-batch prefetch. There is one run per protocol,
no automatic second seed or internal subset training.

| Protocol | GPU | Training samples | Evaluation samples |
| --- | --- | ---: | ---: |
| XSUB | 0 | 63,026 | 50,919 |
| XSET | 1 | 54,468 | 59,477 |

The output directory is new:
`/kaggle/working/NestSAR_PartReadout_T16_Seed128_v1`.
Previous baseline checkpoints are not resumed into the changed architecture.
The config, parameter budget, model identity and sampler calibration are saved
with the new checkpoints. Disabling the option preserves historical baseline
config defaults and checkpoint parameter names.

The raw/canonical cache is shared. If the previous sampler B output contains a
completed pose cache, it is reused without reselecting frames or duplicating
its approximately 2 GiB overlays. Otherwise the launcher builds the shared
pose cache once. `sampler_cache_dir` can specify a different existing cache;
the manifest, training calibration, split identity and file sizes are checked.
No input arrays are added for the readout itself.

## Interpret the result

Compare against sampler B with the same training settings and split, so the
readout is the only experimental change. Use paired per-class predictions to
inspect weak classes, and repeat a promising result across seeds. Similar
overall accuracy alone does not establish preserved fine-motion evidence or
identify the remaining bottleneck. The paper's NTU60 per-class gains are not
expected gains for this NTU120 experiment.

## Verification

`test_part_readout.py` checks discrimination when fixed part means coincide,
masking and gradients, absent actors and singleton parts, zero-initialized
baseline equivalence, unchanged common parameters, parameter accounting,
configuration checks and checkpoint metadata.

The integration command uses synthetic CPU data only:

```bash
python -m experiments.nestsar_sm_all_t16.streaming.smoke_cpu --sampler-b --part-readout --outdir /tmp/nestsar-part-smoke
```

It trains the real model with both workers concurrently, exercises a padded
last batch, checks restart/alias repair, loads best EMA checkpoints for
inference, and checks that the shared sampler cache is not duplicated.

Reference: Xu et al., [Topology-aware Convolutional Neural Network for Efficient
Skeleton-based Action Recognition](https://arxiv.org/abs/2112.04178), AAAI 2022;
[released implementation](https://github.com/hikvision-research/skelact).

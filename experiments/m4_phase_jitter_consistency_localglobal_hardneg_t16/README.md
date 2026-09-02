# LocalGlobal V2 + HardNeg T16

From-scratch ablation built directly on the verified LocalGlobal V2 champion.

## Frozen baseline

- XSUB: 75.3117696734%
- XSET: 75.9268288582%
- Parameters: 1,816,130
- Frames/tokens: 16
- Preprocessing: Local Pose + Global Motion V2
- Final fusion: uniform mean over J/B/JM/BM
- Canonical+jitter dual-view training
- Symmetric KL consistency weight: 0.08
- Stream auxiliary CE weight: 0.15
- EMA: 0.995
- Seed: 128

## Only experimental change

Add a training-only hard-negative ranking term to both canonical and jitter final logits:

`hardneg = mean(softplus(max_wrong_logit - true_logit + margin))`

Default settings:

- hardneg weight: 0.04
- margin: 0.20

Total loss:

`L = main_CE + 0.15*aux_stream_CE + 0.08*sym_KL + 0.04*hardneg`

## Controlled properties

- Training starts from random initialization; no champion checkpoint is loaded.
- Architecture is unchanged.
- Parameter count is unchanged.
- Inference graph is unchanged.
- Inference FLOPs are unchanged relative to LocalGlobal V2.
- Validation uses canonical LocalGlobal V2 preprocessing only.

## Kaggle Dual-T4 launcher

Use `experiments.m4_phase_jitter_consistency_localglobal_hardneg_t16.run_dual_t4`.
XSUB runs on physical GPU0 and XSET on physical GPU1.

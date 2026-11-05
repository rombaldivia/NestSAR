# LocalGlobal V2 + Hand-M4/G4-Lite T32

Clean from-scratch ablation based on the verified LocalGlobal V2 champion.

## Hypothesis

The remaining weak NTU120 classes contain fine hand/wrist temporal information
that is compressed too aggressively by the main T16 representation. Add a
small high-rate hand-only branch while keeping the proven whole-body model.

## Main path

Unchanged LocalGlobal V2:
- T16
- Phase15 local-pose/global-motion preprocessing
- J / B / JM / BM streams
- post-frame CrossStreamRouter
- fixed uniform four-stream final fusion
- canonical + +/-1 boundary-jitter consistency training

## New branch

No attention.

Eight NTU25 hand-region joints, zero-based:
`[6, 7, 21, 22, 10, 11, 23, 24]`

T32 token features:
- local XYZ: 48 values
- global per-raw-frame velocity XYZ: 48 values
- total: 96 values/token

Memory:
- Dense 96 -> 32
- same `base.BiMemory` / `GatedSweep` primitive used by NestSAR
- T32 frame memory
- average into 8 chunks x 4
- second `BiMemory`
- 32-D hand descriptor
- 120-way hand classifier

Final logits:
`main_logits + 0.10 * hand_logits`

Training-only hand auxiliary CE weight: `0.05`.

## Expected model size

Baseline: 1,816,130 params  
Hand branch: +38,520 params  
Total: **1,854,650 params**

The GPU preflight measures both baseline and new XLA FLOPs with the same audit
path before training. Absolute GPU cost-analysis values should not be mixed
with the older TPU reference without a matched audit.

## Baseline

LocalGlobal V2:
- XSUB 75.3117696734%
- XSET 75.9268288582%

## Run

Use `run_dual_t4.py` on Kaggle with exactly two visible T4 GPUs.

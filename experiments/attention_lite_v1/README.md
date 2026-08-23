# Attention-Lite v1

Versioned **NestSAR-HOPE-Attention-Lite D128** TPU v5e-8 experiment for NTU120 reproducibility and controlled training studies.

## Golden architecture — frozen

- 16 frames
- D128
- Attention D64 / H4 / Dh16
- 2,381,028 parameters
- 705 parameter leaves
- 0.060416900 GFLOPs per clip from the exact JAX/XLA forward cost audit
- TPU v5e-8, all 8 devices
- FAST/MEDIUM/SLOW/CONSOLIDATE stagewise optimizer structure
- RegMask 8% frame / 8% joint / 3% part
- EMA 0.995

Golden training defaults:

```text
epochs                  40
patience                 0   # disabled, matching original E40 source
dropout                0.22
learning_rate         1e-3
weight_decay           0.05
warmup_fraction        0.10
label_smoothing        0.05
grad_clip               1.0
predictive_loss_weight 0.10
initial_eta             0.02
initial_alpha           0.95
global_batch              32
grad_accum                 4
effective_batch           128
eval_batch                 32
```

Paper seeds: `28, 42, 128, 2026`. Use the same seed set and training recipe for XSUB and XSET when computing final mean ± std.

## Canonical sources are embedded in GitHub

Kaggle no longer needs a separate Attention-Lite source input.

The repository contains the exact validated XSUB source as an LZMA/base64 payload under:

```text
experiments/attention_lite_v1/canonical_payload/xsub/
```

`canonical_payload/build.py` reconstructs the source byte-for-byte and verifies:

```text
XSUB SHA256 e1080c4e02af96cf9dd0562415e73374d9d582ffa5e74c389ca3e47a05549aa6
XSET SHA256 8a446753a85bb8edba9c4c033cb49e7a9ebbbb317832c533d0f514b90720af0b
```

The validated XSET source is derived from the exact XSUB source using guarded protocol-only replacements recovered from the validated notebook. Both sources are independently SHA256-checked, checked for the 2,381,028-parameter / 705-leaf / D64-H4 architecture markers, and Python-compiled before training.

The fresh-process runner automatically materializes both sources:

```bash
python -u run_attention_lite_both.py --seed 28 --epochs 40 --patience 12
```

No `--xsub-source` or `--xset-source` is required for normal Kaggle use.

## Independent XSUB + XSET models

`run_attention_lite_both.py` and `train_both()` run **two independent models sequentially** on the full TPU8. XSUB finishes first; XSET then starts in a fresh Python subprocess from a fresh initialization with the same requested seed. XSET never inherits XSUB weights.

Editable training parameters include patience, dropout, learning rate, weight decay, warmup fraction, label smoothing, gradient clipping, predictive-loss weight, initial eta/alpha, batch size, gradient accumulation, and evaluation batch size. Architecture-defining values remain frozen.

Example output folders:

```text
/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_28_paper_p12/
/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSET_SEED_28_paper_p12/
```

## Four-seed summary

For a tagged reproducibility recipe:

```bash
python -m experiments.attention_lite_v1.multiseed_summary --run-tag paper_p12
```

Do not mix runs trained with different stopping or optimization recipes in the same mean ± std table.

## Merge gate

Do not merge this experiment runner to `main` until an actual TPU smoke/probe verifies the generated source and expected learning behavior. Static SHA256, marker, and Python compilation checks are strong source-integrity guards but are not substitutes for TPU runtime parity.

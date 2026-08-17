# NestSAR-HOPE v4.1 — audited D128/T16 XSUB baseline

This folder records the exact NestSAR-HOPE v4.1 configuration used for the strongest audited D128/T16 baseline.

## Locked reference result

- Dataset/protocol: NTU RGB+D 120, XSUB
- Frames: 16
- Model dimension: 128
- Memory dimension: 64
- Controller rank: 32
- Frame/chunk/clip/controller blocks: 2 / 2 / 2 / 2
- Chunk size: 4
- Clip size: 8
- Parameters: **2,033,988**
- Audited inference compute: **67,242,094 FLOPs = 67.242094 MFLOPs = 0.067242094 GFLOPs** using 1 MAC = 2 FLOPs
- Best audited XSUB validation accuracy: **73.24771%**
- Seed: 128

## Training recipe

- Epochs: 40
- Physical batch: 32
- Gradient accumulation: 4
- Effective batch: 128
- Eval batch: 64
- Peak LR: 1e-3
- Weight decay: 0.05
- Warmup fraction: 0.10
- Dropout: 0.22
- Label smoothing: 0.05
- Gradient clip: 1.0
- Predictive loss weight: 0.10
- Initial eta: 0.02
- Initial alpha: 0.95
- EMA: 0.995
- RegMask: frame 8%, joint 8%, part 3%
- CMS outer periods: 1 / 2 / 4 / 8
- DMGD-L2: momentum 0.90, memory LR 0.01, outer mix 0.10, projection cap 2.0
- Short-L3 post-write fix active because T16 gives L3 length 2

## Architecture

The inference model is the v3.3 bounded self-referential memory with K/V/Q/eta/alpha/main-memory states, SASM, repaired L3 state path, H4/L4, and the Short-L3 causal post-write fix. The outer trainer uses multi-frequency CMS updates and DMGD-L2 optimizer memory. There is no softmax attention, GCN/GNN, CNN backbone, or TCN backbone.

The exact self-contained Kaggle one-cell used for this baseline is kept separately as `NestSAR_HOPE_v4_1_SHORTL3FIX_Kaggle_16F_ONE_CELL.py`; the Python module in this folder is the exact outer v4.1 trainer/wrapper.

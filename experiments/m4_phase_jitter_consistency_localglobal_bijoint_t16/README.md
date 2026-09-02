# LocalGlobal V2 + Bi-Joint M4/G4 T16

Clean architecture ablation from verified LocalGlobal V2 champion commit `ba9913a3a606ae355eeacfeb1f21a95a60cd4027`.

## Only architecture change

The champion spatial encoder uses one forward `GatedSweep` over the 25 joints after the existing kinematic reorder. This branch replaces that one-way joint memory with the existing NestSAR `BiMemory` primitive:

- forward `GatedSweep`
- backward `GatedSweep`
- learned linear merge
- residual + LayerNorm

The original joint order, inverse reorder, 10-part pooling, part fusion, temporal frame `BiMemory`, post-frame router, descriptor heads, four classifiers, fixed uniform fusion, LocalGlobal V2 preprocessing, boundary-jitter consistency training, optimizer schedule, and seed are unchanged.

No attention. No QKV.

## Parameter budget

- LocalGlobal V2 champion: 1,816,130
- Bi-Joint: 1,834,946
- Added: 18,816 (+1.0361%)

The exact GPU-XLA inference cost is measured by `preflight_gpu.py` using the same runtime/method for baseline and Bi-Joint before training starts.

## Training

Both XSUB and XSET are trained from random initialization on separate Kaggle T4 GPUs with the exact LocalGlobal V2 recipe:

- 60 epochs
- patience 12
- batch 256 per protocol
- eval batch 512
- LR 6e-4, min LR 2e-5
- warmup 0.08
- weight decay 0.03
- label smoothing 0.05
- EMA 0.995
- stream auxiliary loss 0.15
- consistency 0.08
- seed 128
- boundary jitter +/-1 raw frame

This branch intentionally does **not** include Hand-M4/G4 T32 or adaptive/selective hand gating. If Bi-Joint is positive, it can be combined with the independently validated Hand-M4/G4 branch in a later factorial experiment.

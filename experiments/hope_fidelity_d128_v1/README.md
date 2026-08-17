# NestSAR-HOPE-Fidelity-D128-v1

This experiment starts from the exact audited NestSAR-HOPE v4.1 D128/T16 source tree and changes the temporal memory block to more closely follow the high-level HOPE structure while preserving the low-compute NestSAR skeleton pipeline.

## Reference baseline

- NestSAR-HOPE v4.1 D128/T16
- NTU120 XSUB best: **73.24771%**
- Params: **2,033,988**
- Audited compute: **0.067242094 GFLOPs**

## New forward architecture

```text
4-stream NestSAR spatial frontend
  -> SASM
  -> local causal depthwise temporal conv k=4
  -> bounded self-referential K/V/Q/eta/alpha/main-memory
  -> sequential CMS f1 -> f2 -> f4 -> f8
  -> existing L1/L2/L3 hierarchy
  -> Short-L3 causal post-write fix
  -> H4/L4 controller
  -> classifier + learned stream fusion
```

There is **no softmax attention**. The local convolution is temporal-only and depthwise; it is not a CNN/TCN backbone.

## Audited generated size

For T16 / D128 / M64 / R32:

- Params: **2,083,236**
- Increase over v4.1: **49,248 params (+2.42%)**
- Exact GFLOPs: **not yet locked; profile before paper claims**

The design intentionally replaces the original D128 FF tail with four narrow `128 -> 32 -> 128` CMS blocks, keeping dense per-token compute close to the original tail.

## Build

Run against the exact v4.1 source directory:

```bash
python experiments/hope_fidelity_d128_v1/build_from_v41.py \
  --root /path/to/exact/v4.1/source
```

It creates two new files in that source directory without modifying the baseline files:

- `nestsar_hope_fidelity_d128_v1_core.py`
- `nestsar_hope_fidelity_d128_v1_train.py`

## Scratch XSUB probe

```bash
python -u nestsar_hope_fidelity_d128_v1_train.py \
  --model nestsar_hope_fidelity_d128_v1 \
  --protocol xsub \
  --dataset /path/to/ntu120_3danno.pkl \
  --output-dir runs_hope_fidelity_d128_v1_probe_xsub \
  --seed 128 \
  --frames 16 \
  --num-classes 120 \
  --model-dim 128 \
  --memory-dim 64 \
  --frame-blocks 2 \
  --chunk-blocks 2 \
  --clip-blocks 2 \
  --controller-blocks 2 \
  --chunk-size 4 \
  --clip-size 8 \
  --controller-rank 32 \
  --dropout 0.22 \
  --batch-size 32 \
  --grad-accum-steps 4 \
  --eval-batch-size 64 \
  --epochs 3 \
  --patience 3 \
  --learning-rate 1e-3 \
  --weight-decay 0.05 \
  --warmup-fraction 0.10 \
  --label-smoothing 0.05 \
  --grad-clip 1.0 \
  --memory-residual-scale 0.25 \
  --predictive-loss-weight 0.10 \
  --initial-eta 0.02 \
  --initial-alpha 0.95 \
  --log-every-batches 200 \
  --resume none
```

For the paper result, train from random initialization (`--resume none`). The old 73.24771% v4.1 model is the locked comparison baseline, not a warm start.

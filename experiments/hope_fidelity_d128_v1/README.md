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

## Canonical generated size

For T16 / D128 / M64 / R32 / blocks 2-2-2-2 / CMS bottleneck 32:

- Params: **2,083,236**
- Increase over v4.1: **49,248 params (+2.42%)**
- Exact GFLOPs: **not yet locked; profile before paper claims**

For custom dimensions the trainer now reports the actual parameter count rather than incorrectly enforcing 2,083,236. Pass `--expected-params N` when you want a hard guard for a custom configuration.

## Universal CD-Former-style CLI

Use:

```bash
python -u experiments/hope_fidelity_d128_v1/run_universal.py --help
```

The same launcher supports one RTX 5080, Kaggle 2xT4, and Kaggle TPU v5e-8. `--batch-size` is the **global physical batch**; when SPMD is active it is sharded across devices.

### RTX 5080 — canonical recipe

```bash
python -u experiments/hope_fidelity_d128_v1/run_universal.py \
  --preset rtx5080 \
  --source-root /home/romelbaldivia/NestSAR_RTX5080_TQDM/NestSAR_HOPE_v4_1_RTX5080_16F_E40 \
  --dataset /home/romelbaldivia/Downloads/ntu120_3danno.pkl \
  --protocol xsub \
  --probe --fresh
```

This preset is one GPU, global B32, accumulation 4, effective batch 128.

### Kaggle 2xT4 — data parallel

```bash
python -u experiments/hope_fidelity_d128_v1/run_universal.py \
  --preset 2xt4 \
  --dataset auto \
  --protocol xsub \
  --probe --fresh
```

This preset uses both visible T4 GPUs with SPMD: global B32 -> local B16/GPU, accumulation 4 -> effective batch 128.

### Kaggle TPU v5e-8 — throughput preset

```bash
python -u experiments/hope_fidelity_d128_v1/run_universal.py \
  --preset tpu-v5e8 \
  --dataset auto \
  --protocol xsub \
  --probe --fresh
```

This preset uses all 8 TPU devices: global B128 -> local B16/device, accumulation 1 -> effective batch 128.

### TPU v5e-8 — strict batching comparison to v4.1

```bash
python -u experiments/hope_fidelity_d128_v1/run_universal.py \
  --preset tpu-v5e8-canonical \
  --dataset auto \
  --protocol xsub \
  --probe --fresh
```

This preserves B32 x accumulation4, with B32 sharded to B4/device.

## Configurable architecture and optimizer arguments

Example custom run:

```bash
python -u experiments/hope_fidelity_d128_v1/run_universal.py \
  --preset 2xt4 \
  --frames 32 \
  --model-dim 192 \
  --memory-dim 96 \
  --controller-rank 48 \
  --frame-blocks 2 \
  --chunk-blocks 2 \
  --clip-blocks 2 \
  --controller-blocks 2 \
  --chunk-size 4 \
  --clip-size 8 \
  --cms-bottleneck 48 \
  --batch-size 32 \
  --grad-accum-steps 4 \
  --eval-batch-size 64 \
  --learning-rate 6e-4 \
  --weight-decay 0.03 \
  --dropout 0.15 \
  --epochs 100 \
  --patience 20 \
  --dataset auto \
  --protocol xsub \
  --fresh
```

The CLI also exposes EMA/RegMask, self-reference stability constants, CMS periods and DMGD-L2 settings.

## Exact v4.1 source bundle

The universal launcher requires the exact v4.1 source tree. Locally pass `--source-root`. On Kaggle attach `NestSAR_HOPE_v4_1_SHORTL3FIX_Kaggle_16F_ONE_CELL.py`; the launcher auto-discovers it under `/kaggle/input`, verifies the embedded bundle SHA-256, extracts the exact source, and then builds the fidelity files.

## Build only

```bash
python experiments/hope_fidelity_d128_v1/build_from_v41.py \
  --root /path/to/exact/v4.1/source
```

It creates two new files without modifying the baseline files:

- `nestsar_hope_fidelity_d128_v1_core.py`
- `nestsar_hope_fidelity_d128_v1_train.py`

For the paper result, train from random initialization (`--resume none` or `--fresh`). The old 73.24771% v4.1 model is the locked comparison baseline, not a warm start.

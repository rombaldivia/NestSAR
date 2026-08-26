# NestSAR-HOPE Attention-Lite v2 PartTrace

Experimental branch: `experiment/parttrace-v2`

This experiment addresses three structural issues observed in the Attention-Lite v1 design:

1. **Premature spatial collapse** — 10 anatomical part tokens are now mixed and temporally processed before learned part pooling.
2. **No explicit temporal-distance encoding** — causal attention now has a learned relative temporal bias.
3. **CMS stages at identical temporal resolution** — `cms_f1`, `cms_f2`, `cms_f4`, and `cms_f8` now use causal depthwise convolutions with dilations 1, 2, 4, and 8.

The experiment is self-contained apart from the repository's public `nestsar.py` data-loading utilities. It does **not** overwrite the verified Attention-Lite v1 baseline.

## Kaggle audit first

```bash
python -u experiments/parttrace_v2/nestsar_hope_attention_lite_parttrace_v2.py \
  --audit-only
```

The audit prints parameter count, devices, output shape, and XLA FLOPs when the active JAX backend exposes cost analysis.

## XSUB training

```bash
python -u experiments/parttrace_v2/nestsar_hope_attention_lite_parttrace_v2.py \
  --protocol xsub \
  --dataset auto \
  --epochs 40 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --learning-rate 5e-4 \
  --weight-decay 0.03 \
  --seed 128
```

## XSET training

```bash
python -u experiments/parttrace_v2/nestsar_hope_attention_lite_parttrace_v2.py \
  --protocol xset \
  --dataset auto \
  --epochs 40 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --learning-rate 5e-4 \
  --weight-decay 0.03 \
  --seed 128
```

## Cheap smoke-training test

Before a full NTU120 run, use a reduced subset:

```bash
python -u experiments/parttrace_v2/nestsar_hope_attention_lite_parttrace_v2.py \
  --protocol xsub \
  --dataset auto \
  --epochs 2 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --max-train-samples 2048 \
  --max-val-samples 1024 \
  --seed 128
```

## Architecture

```text
[B,T,150]
  -> [B,T,2,25,3]
  -> joint projection D32
  -> 10 anatomical part tokens / person
  -> per-frame 10x10 PartMixer
  -> shared causal temporal trace per part (weights shared across 20 tracks)
  -> learned part pooling AFTER temporal reasoning
  -> two-person fusion -> D128
  -> causal depthwise Conv1D k=4
  -> causal attention D64 / H4 + relative temporal bias
  -> CMS dilated f1/f2/f4/f8 = dilations 1/2/4/8
  -> temporal mean pooling
  -> 120-class head
```

## Research status

This is an **experimental ablation**. Do not report its parameters, FLOPs, or accuracy as paper results until the audit and full official NTU120 runs have completed.

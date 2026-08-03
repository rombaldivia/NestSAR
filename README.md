# NestSAR

Nested multi-timescale memory networks for skeleton-based action recognition in JAX.

## Project status

This repository is under active research development. The current branch introduces the stable command-line interface and reproducibility metadata for a single-file trainer, `nestsar.py`, designed to run both on Kaggle and inside a local Python virtual environment.

The training engine has **not yet been ported** into the standardized file. At this stage, use `--dry-run` to validate paths, hyperparameters, presets, GPU mapping, configuration hashing, and experiment metadata without starting training.

## Current bootstrap usage

```bash
python nestsar.py \
  --preset legacy_4l_seed128 \
  --protocol both \
  --dataset auto \
  --gpu-map xsub:0,xset:1 \
  --dry-run
```

## Target training usage

After the validated legacy engine is ported into `nestsar.py`, the same interface will launch training:

```bash
python nestsar.py \
  --model nestsar_4l \
  --protocol both \
  --dataset auto \
  --seed 128 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --epochs 150 \
  --patience 40 \
  --gpu-map xsub:0,xset:1
```

## Planned model registry

- `h0`: direct-motion baseline
- `h2`: predictive-memory model
- `h3`: low-rank nested-memory ablation
- `nestsar_4l`: causal four-level NestSAR model

## Reproducibility goals

- Official NTU RGB+D 120 XSUB and XSET splits
- Shared preprocessing, optimizer, scheduler, evaluator, and checkpoint format
- Exact training resume with model, optimizer, scheduler, RNG, and early-stopping state
- Configuration snapshots and dataset fingerprints for every run
- The same command-line interface on Kaggle and local environments

## Current validated experimental reference

The current legacy NestSAR-4L experiment reached:

- XSUB: 63.259294%
- XSET: 61.216941%
- Seed: 128
- Physical batch size: 128

These values are treated as a legacy reproduction target, not yet as a fully reproduced result from the new standardized trainer.

## Data

The NTU RGB+D 120 dataset and generated checkpoints are not included in this repository.

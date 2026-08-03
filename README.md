# NestSAR

Nested multi-timescale memory networks for skeleton-based action recognition in JAX.

The repository provides one public entry point, `nestsar.py`, for Kaggle and local virtual environments. It includes the shared NTU RGB+D 120 loader, H0/H2/H3/NestSAR-4L model registry, training, evaluation, automatic GPU assignment, reproducibility metadata, checkpoints, early stopping, and exact resume support.

## Models

```text
h0           Direct-motion baseline
h2           Predictive-memory model
h3           Low-rank nested-memory ablation
nestsar_4l   Causal four-level NestSAR
```

## Kaggle: recommended workflow

Enable a GPU accelerator and Internet, attach the NTU120 pickle to the notebook, and execute this cell:

```python
!rm -rf /kaggle/working/NestSAR
!git clone --depth 1 https://github.com/rombaldivia/NestSAR.git /kaggle/working/NestSAR
%cd /kaggle/working/NestSAR

!python nestsar.py --list-gpus
```

The repository is cloned only once. After cloning, the trainer reads its verified source locally and does not need Internet.

### Full NestSAR-4L experiment

```python
!python -u nestsar.py \
    --model nestsar_4l \
    --protocol both \
    --dataset auto \
    --seed 128 \
    --frames 16 \
    --batch-size 128 \
    --eval-batch-size 256 \
    --epochs 150 \
    --patience 40 \
    --learning-rate 0.0002 \
    --weight-decay 0.03 \
    --warmup-fraction 0.10 \
    --label-smoothing 0.05 \
    --grad-clip 1.0 \
    --gpu-map auto \
    --resume auto
```

This replaces the old CD-Former-style `%%writefile` cell: the `.py` already exists in GitHub, so Kaggle only clones the repository and launches it with arguments.

## GPU behavior

```bash
python nestsar.py --list-gpus
```

With `--gpu-map auto`:

- No GPU: training stops unless `--allow-cpu` is explicitly used for a small test.
- One GPU: XSUB and XSET execute sequentially on that GPU.
- Two or more GPUs: XSUB uses the first visible GPU and XSET uses the second visible GPU in parallel.
- `--max-gpus N` limits how many visible devices are considered.
- `--gpu-map xsub:0,xset:1` defines an explicit mapping.

The current trainer parallelizes the two official protocols across at most two GPUs. It does not yet split one protocol across several GPUs.

## Other experiments

```bash
python -u nestsar.py --model h0 --protocol both --dataset auto --seed 28 --gpu-map auto
python -u nestsar.py --model h2 --protocol both --dataset auto --seed 28 --gpu-map auto
python -u nestsar.py --model h3 --protocol both --dataset auto --seed 28 --gpu-map auto
```

Use `--protocol xsub` or `--protocol xset` to run one official protocol.

## Smoke test

```bash
python -u nestsar.py \
    --model nestsar_4l \
    --protocol xsub \
    --dataset auto \
    --seed 128 \
    --batch-size 128 \
    --eval-batch-size 256 \
    --smoke-only
```

The smoke test checks shapes, finite values, temporal causality, state reset, forward pass, backward pass, and a real physical-batch optimization step.

## Resume

```bash
python -u nestsar.py [same training arguments] --resume auto
```

The last checkpoint stores model parameters, AdamW state, optimizer step, epoch, RNG state, best result, and early-stopping state. Resume continues from the next completed epoch when the configuration hash and run directory match.

## Outputs

Each experiment is stored under:

```text
runs/<model>/seed_<seed>/<config_hash>/
```

The run directory contains resolved configuration, environment metadata, history, logs, best/final results, and last/best checkpoints. Dataset files and generated checkpoints are excluded from Git.

## Local virtual environment

Install a CUDA-compatible JAX build for the local CUDA version, then install the remaining dependencies:

```bash
git clone https://github.com/rombaldivia/NestSAR.git
cd NestSAR
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python nestsar.py --list-gpus
```

Use an explicit dataset path outside Kaggle:

```bash
python -u nestsar.py \
    --model nestsar_4l \
    --protocol both \
    --dataset /absolute/path/ntu120_3danno.pkl \
    --seed 128 \
    --gpu-map auto \
    --resume auto
```

## Source integrity

The launcher verifies the compressed source and the reconstructed readable trainer before execution. The readable implementation can be exported without importing JAX:

```bash
python nestsar.py --export-source nestsar_readable.py
```

Verified readable-source SHA-256:

```text
8aa931b9423bbe4aaba2258563021797f738619ae6fd5f9a227ca9239dfb49d4
```

## Legacy reproduction target

The previous validated NestSAR-4L run reached:

- XSUB: 63.259294%
- XSET: 61.216941%
- Seed: 128
- Physical batch size: 128

These values are the legacy reproduction target. The standardized trainer must be executed on Kaggle before claiming that the final accuracies were reproduced by the refactored implementation.

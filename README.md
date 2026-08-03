# NestSAR

Nested multi-timescale memory networks for skeleton-based action recognition in JAX.

The repository uses **one readable Python file**, `nestsar.py`. The complete implementation is visible directly in that file: configuration, NTU RGB+D 120 loading and preprocessing, H0/H2/H3/NestSAR-4L networks, losses, optimizer, scheduler, training, evaluation, smoke tests, GPU assignment, checkpoints, exact resume, logging, and CLI arguments.

There are no compressed payloads, hidden generated modules, or secondary trainer scripts.

## Models inside `nestsar.py`

```text
h0           Direct-motion baseline
h2           Predictive-memory model
h3           Low-rank nested-memory ablation
nestsar_4l   Causal four-level NestSAR
```

## Kaggle

Enable a GPU accelerator and Internet, attach the NTU120 pickle to the notebook, and run:

```python
!rm -rf /kaggle/working/NestSAR
!git clone --depth 1 https://github.com/rombaldivia/NestSAR.git /kaggle/working/NestSAR
%cd /kaggle/working/NestSAR

!python nestsar.py --list-gpus
```

### Full NestSAR-4L experiment

```python
!python -u nestsar.py \
    --model nestsar_4l \
    --protocol both \
    --dataset auto \
    --seed 128 \
    --frames 16 \
    --num-classes 120 \
    --model-dim 128 \
    --memory-dim 64 \
    --dropout 0.15 \
    --batch-size 128 \
    --eval-batch-size 256 \
    --epochs 150 \
    --patience 40 \
    --learning-rate 0.0002 \
    --weight-decay 0.03 \
    --warmup-fraction 0.10 \
    --label-smoothing 0.05 \
    --grad-clip 1.0 \
    --frame-blocks 2 \
    --chunk-blocks 2 \
    --clip-blocks 2 \
    --controller-blocks 2 \
    --chunk-size 4 \
    --clip-size 8 \
    --controller-rank 32 \
    --max-train-samples 0 \
    --max-val-samples 0 \
    --gpu-map auto \
    --resume auto
```

This follows the same practical idea as the previous CD-Former workflow, but `%%writefile` is unnecessary because the complete readable `.py` is already versioned in GitHub.

## GPU behavior

```bash
python nestsar.py --list-gpus
```

With `--gpu-map auto`:

- No GPU: training stops unless `--allow-cpu` is used for a small test.
- One GPU: XSUB and XSET run sequentially on the same GPU.
- Two or more visible GPUs: XSUB uses the first GPU and XSET uses the second GPU in parallel.
- `--max-gpus N` limits the visible devices considered.
- `--gpu-map xsub:0,xset:1` defines an explicit assignment.

The current implementation parallelizes the two official protocols. It does not divide a single XSUB or XSET worker across several GPUs.

## Other experiments

```bash
python -u nestsar.py --model h0 --protocol both --dataset auto --seed 28 --gpu-map auto
python -u nestsar.py --model h2 --protocol both --dataset auto --seed 28 --gpu-map auto
python -u nestsar.py --model h3 --protocol both --dataset auto --seed 28 --gpu-map auto
```

Use `--protocol xsub` or `--protocol xset` to train only one official protocol.

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

The smoke test checks tensor shapes, finite values, temporal causality, state reset, forward pass, backward pass, and a physical-batch optimization step.

## Exact resume

```bash
python -u nestsar.py [same training arguments] --resume auto
```

The last checkpoint stores model parameters, AdamW state, optimizer step, epoch, RNG state, best result, and early-stopping state. Resume continues from the next completed epoch when the configuration hash matches.

## Outputs

Each experiment is stored under:

```text
runs/<model>/seed_<seed>/<config_hash>/
```

The directory contains resolved configuration, environment metadata, history, logs, final/best results, and last/best checkpoints. Datasets and generated checkpoints are excluded from Git.

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

Run with an explicit dataset path:

```bash
python -u nestsar.py \
    --model nestsar_4l \
    --protocol both \
    --dataset /absolute/path/ntu120_3danno.pkl \
    --seed 128 \
    --gpu-map auto \
    --resume auto
```

## Legacy reproduction target

The previous validated NestSAR-4L run reached:

- XSUB: 63.259294%
- XSET: 61.216941%
- Seed: 128
- Physical batch size: 128

These values remain the reproduction target. The standardized readable trainer still needs a complete Kaggle GPU run before those final accuracies can be claimed as reproduced.

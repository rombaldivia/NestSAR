<h1 align="center">NestSAR</h1>

<p align="center">
  <strong>Ultra-light nested-memory networks for skeleton-based action recognition</strong><br>
  JAX · NTU RGB+D 120 · edge-oriented · fixed 16-token neural processing
</p>

<p align="center">
  <img alt="XSUB" src="https://img.shields.io/badge/NTU120%20XSUB-76.32%25-success">
  <img alt="XSET" src="https://img.shields.io/badge/NTU120%20XSET-78.06%25-success">
  <img alt="Parameters" src="https://img.shields.io/badge/Params-1.83M-blue">
  <img alt="Compute" src="https://img.shields.io/badge/Compute-29.07%20MFLOPs%2Fclip-blueviolet">
  <img alt="Frames" src="https://img.shields.io/badge/Neural%20tokens-16-orange">
</p>

NestSAR is a research line for **low-compute Skeleton Action Recognition (SAR)**. The current architecture explores nested multi-timescale memory, HOPE-inspired low-rank self-modification, motion-preserving representations and adaptive cross-stream fusion while avoiding the usual heavy spatial/temporal backbones.

**No softmax attention · No Transformer · No GCN/GNN · No CNN/TCN · No T×T operation.**

> The raw NTU sequence can have a variable number of frames. The current edge-oriented pipeline summarizes the complete sequence into **16 motion-preserving temporal tokens**, so neural processing remains fixed at `T=16` rather than scaling directly with the raw frame count.

## Current results

### Best fully verified result

**NestSAR-SM-ALL-T16 v1 — corrected preprocessing v2**  
NTU RGB+D 120 · seed 128 · from-scratch training

| Protocol | Best validation accuracy | Best epoch |
| --- | ---: | ---: |
| **XSUB** | **76.321216%** | 24 |
| **XSET** | **78.062108%** | 26 |

| Metric | Value |
| --- | ---: |
| Parameters | **1,826,556** |
| Processing length | **16 tokens** |
| FLOPs / clip | **29,065,216** |
| MFLOPs / clip | **29.065216** |
| GFLOPs / clip | **0.029065216** |
| GMACs / clip (`1 MAC = 2 FLOPs`) | **0.014532608** |

The compute figure above is the **scan-corrected static-unrolled JAX/XLA audit**. Raw `lax.scan` cost analysis undercounts recurrent execution and is not used as the paper-facing FLOP value.

This result uses corrected mask-safe preprocessing, complete raw-frame transition accounting, fresh label-preserving augmentation, self-modifying M4/G4 memory, adaptive routing/fusion and a rank-2 dynamic head. **No CD-Former knowledge distillation and no distal specialist were used.**

Verified experiment branch: [`fix/nestsar-sm-all-preprocessing-v2`](https://github.com/rombaldivia/NestSAR/tree/fix/nestsar-sm-all-preprocessing-v2)  
Machine-readable record: [`verified_results.json`](https://github.com/rombaldivia/NestSAR/blob/fix/nestsar-sm-all-preprocessing-v2/experiments/nestsar_sm_all_t16/verified_results.json)

### Historical accuracy leader — re-audit required

A previous **M4G-H4 + SASM + L3Fix** line produced the highest historical accuracy currently recorded in the project ledger:

| Protocol | Historical score | Status |
| --- | ---: | --- |
| XSUB | **~76.44%** | Re-audit required |
| XSET | **~78.54%** | Re-audit required |

These values are intentionally **not presented as fully verified paper results yet**. The exact checkpoint, configuration and scan-corrected compute must be recovered/re-audited before publication-facing use.

## Recent model progression

| Variant | Tokens | XSUB | XSET | Params | Scan-corrected compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| LocalGlobal V2 | 16 | 75.3118% | 75.9268% | 1,816,130 | 28.545916 MFLOPs |
| HardNeg | 16 | 75.3098% | 76.0647% | 1,816,130 | — |
| Hand-M4/G4 T32 | 32 | 75.4335% | 76.1773% | 1,854,650 | 29.612176 MFLOPs |
| **SM-ALL-T16 corrected v2** | **16** | **76.3212%** | **78.0621%** | **1,826,556** | **29.065216 MFLOPs** |
| M4G-H4 + SASM + L3Fix | historical | **~76.44%** | **~78.54%** | re-audit | re-audit |

The older readable `nestsar.py` NestSAR-4L run on `main` reached **63.259294% XSUB / 61.216941% XSET**. It remains a legacy reproduction target, not the current research best.

## What changed in the corrected pipeline

Three preprocessing details materially affect the signal seen by the model:

1. **Missing people/joints stay missing.** Validity is captured from raw coordinates before centering, preventing absent zero skeletons from becoming artificial non-zero bodies.
2. **Movement between temporal segments is preserved.** Adjacent raw-frame differences are computed before segmentation, so boundary motion is not silently dropped.
3. **Motion is represented with fixed neural cost.** The full raw sequence is summarized into 16 LocalGlobal motion-preserving tokens while the neural graph remains fixed-length.

## Architecture direction

The current SM-ALL family retains a compact LocalGlobal M4/G4 topology and adds low-rank self-modifying fast-memory residuals. A shared controller modulates streams and the model learns adaptive fusion without materializing attention matrices.

For a temporal state, the HOPE-inspired delta-memory update is:

```text
pred_t = k_t^T S_(t-1)
err_t  = v_t - pred_t
S_t    = alpha_t S_(t-1) + eta_t k_t err_t^T
read_t = q_t^T S_t
```

`S_0` is learned by the outer NTU120 optimization and reset for every clip. This is a compressed, edge-oriented **HOPE-inspired** mechanism; it is not claimed to be a verbatim reproduction of the full language-model HOPE stack.

## Repository structure

The stable readable trainer remains available as [`nestsar.py`](./nestsar.py). Research variants are developed in versioned experiment directories/branches so architecture, preprocessing and compute changes can be audited independently.

```text
nestsar.py                                  readable baseline trainer
experiments/                               versioned research experiments
experiments/nestsar_sm_all_t16/            current SM-ALL T16 line
NESTSAR_EXPERIMENT_STATUS_2026-08-31.md    historical experiment ledger
```

## Kaggle

For the readable `main` trainer:

```bash
git clone --depth 1 https://github.com/rombaldivia/NestSAR.git
cd NestSAR
python nestsar.py --list-gpus
```

For the corrected SM-ALL-T16 research pipeline used by the verified result:

```bash
git clone --depth 1 \
  --branch fix/nestsar-sm-all-preprocessing-v2 \
  https://github.com/rombaldivia/NestSAR.git
cd NestSAR
```

The experiment contains the dual-T4 launcher, shared-cache pipeline, checkpoint/resume support, preprocessing regression tests and scan-corrected compute audit. XSUB and XSET can run simultaneously on two T4 GPUs.

## Reproducibility policy

Paper-facing NestSAR results should include the exact Git commit, protocol, seed, processing length, preprocessing version, checkpoint, parameter count, validation accuracy, confusion/per-class metrics and **scan-corrected** inference compute. Historical or partially recovered results stay explicitly labeled until they satisfy that audit trail.

---

**Research focus:** efficient Skeleton Action Recognition, nested/self-modifying memory, motion representation and edge AI.
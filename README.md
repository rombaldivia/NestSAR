<h1 align="center">NestSAR</h1>

<p align="center">
  <strong>Ultra-light nested-memory networks for skeleton-based action recognition</strong><br>
  JAX · NTU RGB+D 120 · edge-oriented · fixed 16-token neural processing
</p>

<p align="center">
  <img alt="XSUB" src="https://img.shields.io/badge/NTU120%20XSUB-76.97%25-success">
  <img alt="XSET" src="https://img.shields.io/badge/NTU120%20XSET-78.42%25-success">
  <img alt="Neural tokens" src="https://img.shields.io/badge/Neural%20tokens-16-orange">
  <img alt="JAX" src="https://img.shields.io/badge/Framework-JAX-blue">
</p>

NestSAR is a research line for **low-compute Skeleton Action Recognition (SAR)**. It explores nested multi-timescale memory, HOPE-inspired low-rank self-modification, motion-preserving skeleton representations and adaptive cross-stream fusion for accurate recognition under a very small compute budget.

**No softmax attention · No Transformer · No GCN/GNN · No CNN/TCN · No T×T operation.**

> Raw NTU clips may contain a variable number of frames. The current edge-oriented pipeline summarizes the complete sequence into **16 motion-preserving temporal tokens**, keeping neural processing fixed at `T=16` rather than directly proportional to raw clip length.

## Current best accuracy

### NTU RGB+D 120

| Protocol | Best validation accuracy |
| --- | ---: |
| **XSUB** | **76.971268%** |
| **XSET** | **78.423592%** |

These are the **current best NestSAR scores** recorded in the project.

## Audited SM-ALL-T16 reference

The previous fully documented **NestSAR-SM-ALL-T16 v1 — corrected preprocessing v2** run remains the current compute-audited reference:

| Protocol | Accuracy | Best epoch |
| --- | ---: | ---: |
| XSUB | 76.321216% | 24 |
| XSET | 78.062108% | 26 |

| Metric | Audited value |
| --- | ---: |
| Parameters | **1,826,556** |
| Processing length | **16 tokens** |
| FLOPs / clip | **29,065,216** |
| MFLOPs / clip | **29.065216** |
| GFLOPs / clip | **0.029065216** |
| GMACs / clip (`1 MAC = 2 FLOPs`) | **0.014532608** |

The compute value above comes from the **scan-corrected static-unrolled JAX/XLA audit**. Ordinary cost analysis of recurrent `lax.scan` graphs can undercount repeated recurrent execution, so the raw scan value is not used for paper-facing reporting.

The audited run used corrected mask-safe preprocessing, complete raw-frame transition accounting, fresh label-preserving augmentation, self-modifying M4/G4 memory, adaptive routing/fusion and a rank-2 dynamic head. It did **not** use CD-Former knowledge distillation or the distal specialist.

Verified reference branch: [`fix/nestsar-sm-all-preprocessing-v2`](https://github.com/rombaldivia/NestSAR/tree/fix/nestsar-sm-all-preprocessing-v2)  
Machine-readable audited record: [`verified_results.json`](https://github.com/rombaldivia/NestSAR/blob/fix/nestsar-sm-all-preprocessing-v2/experiments/nestsar_sm_all_t16/verified_results.json)

## Model progression

| Variant | Tokens | XSUB | XSET | Params | Compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| LocalGlobal V2 | 16 | 75.3118% | 75.9268% | 1,816,130 | 28.545916 MFLOPs |
| HardNeg | 16 | 75.3098% | 76.0647% | 1,816,130 | — |
| Hand-M4/G4 T32 | 32 | 75.4335% | 76.1773% | 1,854,650 | 29.612176 MFLOPs |
| SM-ALL-T16 corrected v2 | 16 | 76.3212% | 78.0621% | 1,826,556 | 29.065216 MFLOPs |
| **Current NestSAR best** | **16** | **76.971268%** | **78.423592%** | — | — |

The current-best row reports the latest accuracy result. Its exact checkpoint-linked parameter/FLOP audit should be attached before those fields are used in publication comparisons.

## Why NestSAR

NestSAR is designed around a different trade-off from conventional skeleton-recognition systems: preserve useful motion and relational information while keeping inference compact enough for edge deployment.

Key design goals include:

- **Fixed short neural sequence:** full raw clips are compressed to 16 informative temporal tokens.
- **Nested memory:** local and global temporal states operate at multiple timescales.
- **Self-modification:** low-rank fast-memory updates adapt internal state during a clip.
- **Motion preservation:** displacement, phase and path information are retained instead of relying only on sampled poses.
- **Adaptive fusion:** complementary skeleton streams are combined without quadratic attention.
- **Edge-oriented compute:** inference is kept in the tens-of-MFLOPs regime.

## Corrected preprocessing

Three data-path corrections materially improved the reliability of the representation:

1. **Missing people/joints remain zero.** Validity is captured from raw coordinates before centering so absent skeletons do not become artificial non-zero bodies.
2. **All adjacent-frame motion is preserved.** Differences are computed before temporal segmentation so movement across segment boundaries is not dropped.
3. **The complete clip contributes to fixed-cost tokens.** Raw sequences are summarized into a fixed 16-token neural representation while retaining motion statistics from the full sequence.

Recent experiments also investigate **true parent-relative motion paths**, separating real local articulation from whole-body displacement proxies.

## Self-modifying memory

The SM-ALL family keeps a compact LocalGlobal M4/G4 topology and adds low-rank self-modifying fast-memory residuals. A shared controller modulates stream features and adaptive fusion without materializing attention matrices.

```text
pred_t = k_t^T S_(t-1)
err_t  = v_t - pred_t
S_t    = alpha_t S_(t-1) + eta_t k_t err_t^T
read_t = q_t^T S_t
```

`S_0` is learned by the outer NTU120 optimization and reset for every clip. This is a compressed, edge-oriented **HOPE-inspired** mechanism and is not claimed to reproduce the full language-model HOPE stack verbatim.

## Repository structure

The readable baseline trainer remains available as [`nestsar.py`](./nestsar.py). New ideas are developed in versioned experiment directories and branches so architecture, preprocessing and compute changes can be audited independently.

```text
nestsar.py                                  readable baseline trainer
experiments/                               versioned research experiments
experiments/nestsar_sm_all_t16/            SM-ALL T16 research line
NESTSAR_EXPERIMENT_STATUS_2026-08-31.md    historical experiment ledger
```

## Kaggle

For the readable `main` trainer:

```bash
git clone --depth 1 https://github.com/rombaldivia/NestSAR.git
cd NestSAR
python nestsar.py --list-gpus
```

For the corrected SM-ALL-T16 reference pipeline:

```bash
git clone --depth 1 \
  --branch fix/nestsar-sm-all-preprocessing-v2 \
  https://github.com/rombaldivia/NestSAR.git
cd NestSAR
```

The research pipeline includes dual-T4 execution, shared-cache preprocessing, checkpoint/resume support, regression tests and scan-corrected compute auditing. XSUB and XSET can run simultaneously on two T4 GPUs.

## Reproducibility policy

Publication-facing NestSAR results should be tied to an exact Git commit, protocol, seed, processing length, preprocessing version, checkpoint, parameter count, validation accuracy, per-class/confusion metrics and **scan-corrected** inference compute.

---

**Research focus:** efficient Skeleton Action Recognition · nested/self-modifying memory · motion representation · edge AI
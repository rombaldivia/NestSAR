# NestSAR-SM-ALL-T16 v1

From-scratch LocalGlobal M4/G4 experiment with low-rank self-modifying fast weights.

## Verified result — corrected preprocessing v2

Final dual-T4 NTU RGB+D 120 run, seed 128, fixed neural processing length T=16:

| Protocol | Best validation accuracy | Best epoch | Last epoch |
| --- | ---: | ---: | ---: |
| XSUB | **76.321216%** | 24 | 29 |
| XSET | **78.062108%** | 26 | 31 |

Model size: **1,826,556 parameters**.

The run used corrected mask-safe preprocessing, complete raw-frame transition accounting, fresh per-epoch label-preserving augmentation, self-modifying M4/G4, cross-stream routing, adaptive fusion, and the rank-2 dynamic head. The distal specialist and CD-Former knowledge distillation were **not** used in this result.

## Verified compute — scan-corrected static audit

The paper/comparison FLOP number is measured with `compute_unrolled_audit.py`, which replaces only recurrent scan implementations with mathematically identical static unrolled loops before JAX/XLA cost analysis. This is necessary because ordinary cost analysis on `lax.scan`/while graphs can report loop-body cost without charging every recurrent iteration.

Audit environment: JAX 0.7.2, CUDA GPU, batch 1.

| Model | Parameters | FLOPs / clip | MFLOPs / clip | GFLOPs / clip |
| --- | ---: | ---: | ---: | ---: |
| LocalGlobal V2 | 1,816,130 | 28,545,916 | 28.545916 | 0.028545916 |
| Hand-M4/G4 T32 | 1,854,650 | 29,612,176 | 29.612176 | 0.029612176 |
| **NestSAR-SM-ALL-T16** | **1,826,556** | **29,065,216** | **29.065216** | **0.029065216** |

For SM-ALL-T16 this is also **0.014532608 GMACs/clip** under the convention `1 MAC = 2 FLOPs`.

Relative scan-corrected compute:

- SM-ALL vs LocalGlobal V2: **+0.519300 MFLOPs (+1.819%)**.
- SM-ALL vs Hand-M4/G4 T32: **-0.546960 MFLOPs (-1.847%)**.

### Important accounting note

The training result JSON reports `5,155,914 FLOPs = 0.005155914 GFLOPs/clip` from the ordinary compiled XLA graph. **Do not use that number as the paper FLOP count.** It undercounts recurrent scan execution. The verified reporting value for this model is:

**NestSAR-SM-ALL-T16 = 0.029065216 GFLOPs/clip = 29.065216 MFLOPs/clip.**

The launcher still keeps its historical raw-XLA runtime guard. That guard and the scan-corrected paper audit use different accounting paths and therefore should not be compared numerically as if they were the same FLOP definition.

## Hard constraints

- Raw NTU clip length is variable.
- Preprocessing converts the whole raw clip to exactly **16** LocalGlobal motion-preserving temporal tokens.
- Neural processing length is therefore fixed at T=16 and does not follow the original raw frame count.
- No attention, GCN, TCN, Transformer, or T x T operation.
- Training starts from random initialization; no champion checkpoint is loaded.

## Corrected preprocessing v2

The corrected path preserves model input shape `[16, 750]` while fixing three data-path defects:

1. Missing people/joints are identified before centering and remain exactly zero through centering and normalization.
2. Every valid adjacent raw-frame transition is computed before segmentation and assigned exactly once, preserving full displacement, phase-A displacement, phase-B displacement, and path magnitude.
3. The augmented training view is rebuilt from the raw skeleton every epoch using deterministic sample/epoch seeds, with up to ±8 degree yaw rotation and configured temporal-boundary jitter.

These preprocessing changes do not change the neural inference graph or its parameter count.

## Self-modification

The proven M4/G4 LocalGlobal topology is retained. A shared controller modulates input channels and stream features. M4 and G4 retain their original BiMemory and add a rank-2 fast-weight residual. For each temporal state:

```text
pred_t = k_t^T S_(t-1)
err_t  = v_t - pred_t
S_t    = alpha_t S_(t-1) + eta_t k_t err_t^T
read_t = q_t^T S_t
```

`S_0` is learned by the outer NTU120 optimization and is reset for every clip. M4 updates at the 16-token level; G4 uses the same rule on four chunk states. Fusion is dynamically modulated from a zero-initialized uniform state and the final adaptive head is rank-2 and evaluated once per clip.

This is a **HOPE-inspired low-rank self-modifying delta-memory adaptation**, deliberately compressed for NestSAR's edge-compute target. It is not claimed to be a verbatim reproduction of the full language-model HOPE stack.

## Kaggle execution

`run_dual_t4_corrected.py` reuses the original dual-T4 launcher while selecting the corrected training wrapper. GPU0 runs XSUB and GPU1 runs XSET. The corrected wrapper changes only the data path; `model.py` and the SM-ALL training/optimization logic remain unchanged.

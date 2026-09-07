# NestSAR-SM-ALL-T16 v1 — Person-aware P2 v3

From-scratch LocalGlobal M4/G4 experiment with low-rank self-modifying fast weights and explicit two-person representation.

## Verified person-aware P2 v3 result

Dual-T4 NTU RGB+D 120 run, seed 128, fixed neural processing length T=16:

| Protocol | Best held-out accuracy | Best epoch | Last epoch |
| --- | ---: | ---: | ---: |
| XSUB | **76.971268%** | 24 | 29 |
| XSET | **78.423592%** | 31 | 36 |

Model size: **1,826,556 parameters**.

Comparison with the preceding corrected-preprocessing-v2 baseline:

| Protocol | corrected-v2 | person-aware P2 v3 | Gain |
| --- | ---: | ---: | ---: |
| XSUB | 76.321216% | **76.971268%** | **+0.650052 pp** |
| XSET | 78.062108% | **78.423592%** | **+0.361484 pp** |

The person-aware run keeps the same 16 x 750 input and parameter budget. It adds explicit P1/P2 identity, mask-safe spatial encoding, P2-P1 relative pose/motion terms, and person/pair-presence signals to the SM controller. No attention, GCN, TCN, or Transformer is introduced.

### Evaluation-protocol note

`xsub_val` and `xset_val` are the official NTU120 held-out benchmark partitions. This run evaluated them every epoch for checkpoint selection and early stopping. The scores above are therefore verified experiment scores, but they should **not** be described as untouched one-shot final-test results in a paper. Future architecture/hyperparameter selection should use an internal validation split created only from the official training partition, then evaluate the official held-out partition after freezing the configuration.

## Verified compute — person-aware P2 v3

The reporting FLOP number is measured with `compute_unrolled_audit.py`, which replaces recurrent scans with mathematically equivalent static unrolled loops before JAX/XLA cost analysis. Ordinary scan/while cost analysis can undercount recurrent execution.

Audit environment: JAX 0.7.2, CUDA GPU, batch 1.

| Model | Parameters | FLOPs / clip | MFLOPs / clip | GFLOPs / clip |
| --- | ---: | ---: | ---: | ---: |
| LocalGlobal V2 | 1,816,130 | 28,545,916 | 28.545916 | 0.028545916 |
| Hand-M4/G4 T32 | 1,854,650 | 29,612,176 | 29.612176 | 0.029612176 |
| Corrected-v2 SM-ALL-T16 | 1,826,556 | 29,065,216 | 29.065216 | 0.029065216 |
| **Person-aware P2 v3** | **1,826,556** | **29,412,272** | **29.412272** | **0.029412272** |

Person-aware P2 v3 is also **0.014706136 GMACs/clip** under the convention `1 MAC = 2 FLOPs`.

Relative to corrected-v2, explicit person awareness adds **0.347056 MFLOPs/clip (+1.194%)**.

The raw compiled scan graph can report a much smaller number and must not be used as the paper compute figure. The verified reporting value for person-aware P2 v3 is **0.029412272 GFLOPs/clip**.

## Complete issue-closure audit

The person-aware v3 implementation passed the complete **70/70** issue-closure audit. Relevant verified dataset/representation checks include:

- NTU120 usable samples: **113,945**.
- XSUB: 63,026 train / 50,919 official held-out.
- XSET: 54,468 train / 59,477 official held-out.
- `Axxx` sample ID exactly matches zero-based label `Axxx - 1`.
- Official subject/setup membership matches the NTU120 protocols exactly.
- No train/held-out overlap in either protocol.
- Raw clips containing P2: **26,289**.
- Complete P2 loss after T16 preprocessing: **0**.
- P2 preservation: **100%**.
- Mean preserved P2 temporal tokens: **15.724 / 16**.
- Static-unrolled output equivalence maximum absolute difference: approximately **3.28e-7**.

## Person-aware preprocessing

Preprocessing version: `sm-all-personaware-p2-segmentpose-v2`.

The path preserves the `[16, 750]` model input while enforcing the following behavior:

1. Missing people/joints are identified before centering and remain exactly zero.
2. Two occupied NTU source tracks preserve their source order instead of being swapped by whole-clip motion energy.
3. Empty source tracks are compacted so a real actor is not stranded behind an empty P1 slot.
4. Representative pose is selected independently per person inside each temporal segment; if P2 is visible anywhere in the segment, a valid P2 pose is retained.
5. Every valid adjacent raw-frame transition is computed before segmentation and assigned exactly once.
6. Fresh per-epoch yaw/boundary augmentation remains zero-padding safe.

Pipeline/cache version: `sm-all-shared-cache-personaware-p2-v3`.

## Representation/model

- Learned `person_embed` remains explicit for P1 and P2.
- `MaskSafeSpatialEncoder` prevents Dense biases and embeddings from fabricating features for an absent P2.
- The SM controller no longer averages over the person axis.
- Its fixed 15-D summary contains P1 pose, P2 pose, P2-P1 relative pose, P2-P1 relative full displacement, and P1/P2/pair-presence bits.
- M4/G4, cross-stream routing, adaptive fusion, and the dynamic rank-2 head remain within the same 1,826,556-parameter budget.

## Shared-cache runner

Use `kaggle_cell.py` inside the notebook kernel with `runpy.run_path(...)`, or run:

```bash
python -m experiments.nestsar_sm_all_t16.run_dual_t4_corrected
```

The shared-cache runner:

- loads the NTU pickle once during preparation;
- validates both official protocols;
- writes shared float32 raw and canonical-token memory maps;
- runs XSUB on GPU0 and XSET on GPU1;
- computes canonical tokens once per sample;
- rebuilds only the augmented training view each epoch;
- keeps bounded batch prefetching;
- resumes optimizer, EMA, RNG, history, best checkpoint, and early-stopping state.

Person-aware v3 paths:

- output: `/kaggle/working/NestSAR_SM_ALL_T16_PERSON_AWARE_P2_v3`
- cache: `/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3`

Default training is 60 epochs, micro-batch 64, gradient accumulation 4 (effective batch 256), EMA 0.995, and patience 5 after the 8% learning-rate warm-up.

## Regression checks

```bash
python -m pytest -q \
  experiments/nestsar_sm_all_t16/test_preprocessing_corrected.py \
  experiments/nestsar_sm_all_t16/streaming/tests
```

## Hard constraints

- Raw NTU clip length is variable.
- Preprocessing converts the whole raw clip to exactly **16** LocalGlobal motion-preserving temporal tokens.
- Neural processing length is fixed at T=16 and does not follow raw frame count.
- No attention, GCN, TCN, Transformer, or T x T operation.
- Training starts from random initialization; no champion checkpoint is loaded.

## Self-modification

The proven M4/G4 LocalGlobal topology is retained. A shared controller modulates input channels and stream features. M4 and G4 keep their BiMemory and add a rank-2 fast-weight residual:

```text
pred_t = k_t^T S_(t-1)
err_t  = v_t - pred_t
S_t    = alpha_t S_(t-1) + eta_t k_t err_t^T
read_t = q_t^T S_t
```

`S_0` is learned by the outer NTU120 optimization and reset for every clip. M4 updates at the 16-token level; G4 uses the same rule on four chunk states. Fusion is dynamically modulated and the final adaptive rank-2 head is evaluated once per clip.

This is a **HOPE-inspired low-rank self-modifying delta-memory adaptation**, compressed for NestSAR's edge-compute target; it is not claimed to reproduce the full language-model HOPE stack verbatim.

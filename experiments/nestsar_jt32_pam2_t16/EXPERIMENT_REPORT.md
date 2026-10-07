# NestSAR-JT32-PAM2-T16 — experiment report

Updated: 2026-10-07

## Goal

Deeply redesign NestSAR to preserve a full joint-time carrier:

```
[B,16,2,25,15]
  -> factorized joint encoder
  -> [B,16,25,32]
  -> 2 parallel Joint-Time blocks
  -> T16/T8/T4/hands readout
```

Each block computes temporal attention, spatial attention, rank-4 parallel associative memory, spectral motion and FFN branches in parallel and adds them residually.

The old M4 -> Router -> G4 core is removed.

## Implemented architecture

- T = 16
- J = 25
- D = 32
- 2 blocks
- 4 attention heads
- memory rank = 4
- anatomical-hop spatial bias
- temporal relative bias
- parallel associative scan
- fixed DCT spectral branch
- multiscale T16/T8/T4 + hand readout
- no early joint pooling inside the core blocks

Exact model parameter count from the implemented source: **177,026**.

Approximate split:

- encoder + two JT blocks + final carrier norm: ~30.6k parameters
- readout/classifiers: ~146.5k parameters

Design compute target was **~0.030-0.035 strict GFLOPs**, but a final compiler-independent operator audit has not yet been committed. Do not publish the estimate as an exact count.

## Partial NTU120 result

XSUB at epoch 26:

- best validation: **66.892516%**
- best epoch: 26

This is far below FMSE (~77.33%) and already sufficient to diagnose strong underfitting. The run should not be treated as a competitive final model.

## NTU60 XSUB compatibility audit at epoch 26 EMA

16,487 processed validation samples:

| Metric | JT32 | FMSE-architecture reference | Delta |
|---|---:|---:|---:|
| Strict 120-way Top-1 | 74.495057% | 83.568872% | **-9.073815 pp** |
| Restricted A001-A060 Top-1 | 78.073634% | 85.685692% | **-7.612058 pp** |
| Strict Top-5 | 92.624492% | 95.529811% | -2.905319 pp |
| Restricted Top-5 | 94.996057% | 96.579123% | -1.583066 pp |
| Predictions in A061-A120 | 6.538485% | 3.821192% | +2.717293 pp |

The much smaller Top-5 drop than Top-1 drop shows that JT32 frequently finds the correct local class neighborhood but ranks the winner poorly.

## Global fixed/broken audit versus FMSE

On restricted NTU60:

- FMSE wrong -> JT32 correct: **592**
- FMSE correct -> JT32 wrong: **1,847**
- net correction: **-1,255**

This is the central failure pattern: broad replacement fixes some errors but destroys far more already-correct decisions.

## Biggest JT32 gains

Only a few classes improved:

- A054: **+1.81 pp**
- A030: **+1.09 pp**
- A060: **+0.72 pp**

## Biggest JT32 losses

Examples:

- A012: **-23.90 pp**
- A010: **-22.71 pp**
- A037: **-17.75 pp**
- A001: **-16.06 pp**
- A041: **-15.22 pp**
- A011: **-13.55 pp**
- A005: **-13.45 pp**
- A044: **-13.04 pp**
- A032: **-11.59 pp**
- A057: **-10.91 pp**

## Largest JT32 Top-5 -> Top-1 gaps

These show ranking/class-boundary weakness rather than total information loss:

- A012: Top-1 33.82%, Top-5 97.79%, gap **63.97 pp**
- A011: 45.05% -> 93.77%, gap **48.72 pp**
- A010: 39.93% -> 88.28%, gap **48.35 pp**
- A029: 56.36% -> 89.45%, gap **33.09 pp**
- A016: 66.30% -> 96.34%, gap **30.04 pp**
- A017: 59.12% -> 89.05%, gap **29.93 pp**

## Concrete implementation weaknesses

### 1. Input family over-compression

Each family concatenates P1, P2 and P2-P1 into 9 values, then projects:

```
9 -> 4
```

This is too aggressive for a model whose stated goal is information preservation.

### 2. Lost R4/FMSE motion families

The implementation restores parent-relative full displacement but does not reproduce the complete successful R4 bone-motion family:

- parent-relative phase A
- parent-relative phase B
- parent-relative path

Therefore JT32 accidentally removed motion information that FMSE/R4 already used.

### 3. Destructive final readout

The core successfully preserves [B,16,25,32], but the readout immediately does:

```python
fine_seq = mean(x, axis=joint)
```

and similar averaging for T8/T4 and hands.

Thus the classifier never directly consumes the full joint-time lattice. This contradicts the main architectural goal.

### 4. Backbone capacity is too small

Only ~30.6k parameters are in the encoder and two JT blocks. The sophisticated 400-token representation is therefore strongly capacity-limited even though most total parameters sit in readout/classifier layers.

### 5. R4 regularization was copied to a much smaller/different model

The run reused:

- dropout 0.10
- label smoothing 0.05
- auxiliary CE 0.15
- consistency KL 0.08

The auxiliary heads correspond to partial scale/region views, not the old full J/B/JM/BM streams, so forcing each partial readout to solve all 120 classes can create conflicting gradients.

## Advantages worth preserving

JT32 still provides useful ideas:

- explicit 16 x 25 carrier;
- parallel spatial/temporal processing;
- anatomical-hop bias;
- temporal relative bias;
- parallel associative memory;
- spectral temporal branch;
- high Top-5 despite poor Top-1, indicating nontrivial local representation.

These ideas should be reused **as a residual specialist**, not as a global replacement.

## Decision

**Do not continue JT32 as the main backbone and do not simply increase width.**

The new direction is:

- restore FMSE as the protected base;
- reuse joint-time processing only as a targeted fine-motion branch;
- replace mean pooling with learned token-query readout;
- initialize residual logit correction at zero;
- restrict corrections to local hard-class neighborhoods;
- protect confident FMSE predictions with a distillation/protection objective.

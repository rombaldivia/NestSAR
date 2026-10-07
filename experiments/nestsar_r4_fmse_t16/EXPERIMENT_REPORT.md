# NestSAR R4-FMSE T16 — experiment report

Updated: 2026-10-07

## Scope

This branch changes the joint-motion Spatial-2 representation while keeping the R4 training recipe, model size and downstream M4/Router/G4/descriptor path intact.

The successful change is the **Factorized Motion Spatial Encoder (FMSE)**:

- the four 3-D motion components are projected separately;
- a low-rank mixer combines them only after factorization;
- parameter count remains exactly **1,831,932**;
- this is the current strongest stable R4-family representation change.

## NTU120 results

| Model | XSUB | XSET | Params |
|---|---:|---:|---:|
| R4 reference | 76.910387% | 78.460581% | 1,831,932 |
| FMSE | **77.330662%** | 78.453856% | 1,831,932 |

Delta vs R4:

- XSUB: **+0.420275 pp**
- XSET: **-0.006725 pp** (effectively tied)

This is materially different from the many post-compression changes that produced only noise-level gains.

## Full NTU120 class-error audit using the strongest FMSE-architecture checkpoints

The later FMSE + training-only local-geometry checkpoints reproduce exactly and use the same FMSE inference graph.

### XSUB

- n: **50,919**
- Top-1: **77.336554%**
- Top-5: **93.682123%**
- macro recall: **78.952%**
- reproduction delta: **+0.000000 pp**

Weakest classes:

| Class | Action | Recall |
|---|---|---:|
| A073 | staple book | **27.85%** |
| A072 | make victory sign | **36.35%** |
| A074 | counting money | **39.65%** |
| A071 | make OK sign | **43.30%** |
| A084 | play magic cube | **47.03%** |
| A091 | open a box | **47.91%** |
| A105 | blow nose | **49.22%** |
| A082 | fold paper | **50.96%** |
| A075 | cutting nails | **51.14%** |
| A012 | writing | **53.31%** |

Largest confusion directions:

- A072 -> A071: **169**
- A073 -> A076: **154**
- A071 -> A072: **141**
- A074 -> A084: **87**
- A074 -> A075: **86**
- A072 -> A069: **75**
- A106 -> A050: **75**
- A118 -> A056: **72**

### XSET

- n: **59,477**
- Top-1: **78.458900%**
- Top-5: **94.043075%**
- macro recall: **78.419%**
- reproduction delta: **+0.000000 pp**

Weakest classes:

| Class | Action | Recall |
|---|---|---:|
| A072 | make victory sign | **39.80%** |
| A012 | writing | **40.76%** |
| A073 | staple book | **40.78%** |
| A074 | counting money | **43.62%** |
| A084 | play magic cube | **49.28%** |
| A071 | make OK sign | **50.41%** |
| A011 | reading | **52.20%** |
| A107 | wield knife towards other person | **52.64%** |
| A076 | cutting paper with scissors | **52.66%** |
| A075 | cutting nails | **53.43%** |

Largest confusion directions:

- A072 -> A071: **128**
- A071 -> A072: **119**
- A073 -> A076: **93**
- A076 -> A073: **91**
- A017 -> A016: **83**
- A016 -> A017: **79**
- A056 -> A118: **78**
- A012 -> A030: **77**

Seven weakest classes recur in both protocols: **A071, A072, A073, A074, A075, A084 and A012**.

This cross-protocol stability is the strongest evidence so far that the remaining bottleneck is concentrated, systematic fine-class discrimination rather than a generic capacity shortage.

The recurring error families are:

- micro hand/gesture geometry;
- fine object manipulation and local trajectory;
- temporal direction/order;
- inter-person direction/interaction semantics;
- reading/writing/typing-like local ranking.

The large Top-5/Top-1 gap shows that much of the missing accuracy is local ranking rather than complete loss of the correct class.

## NTU60 compatibility audit using the best FMSE-architecture XSUB weights

The strongest tested XSUB checkpoint used the FMSE inference architecture and was trained with the later geometry loss only during training. Its original NTU120 XSUB result was **77.336554%**.

Evaluation on the processed NTU60 XSUB subset (16,487 samples):

| Metric | Result |
|---|---:|
| Strict 120-way Top-1 | **83.568872%** |
| Strict 120-way Top-5 | **95.529811%** |
| Restricted A001-A060 Top-1 | **85.685692%** |
| Restricted A001-A060 Top-5 | **96.579123%** |
| Predictions in A061-A120 | **3.821192%** |

Important: this is a **diagnostic compatibility audit**, not a publishable NTU60 benchmark, because the checkpoint was trained on NTU120 and the processed NTU60 validation set contains 16,487 samples.

## Per-class strengths on NTU60

Best classes include:

| Class | FMSE accuracy |
|---|---:|
| A027 | 99.64% |
| A059 | 99.63% |
| A043 | 99.27% |
| A009 | 99.27% |
| A008 | 98.53% |
| A052 | 97.46% |
| A042 | 97.46% |
| A015 | 97.46% |
| A055 | 97.45% |
| A021 | 97.07% |

There are **21 classes at or above 90%** in this NTU60 diagnostic. This is strong evidence that the existing FMSE/R4 backbone should be protected rather than globally replaced.

## Per-class weaknesses on NTU60

Hard classes under 75%:

| Class | Top-1 | Top-5 | Main confusion |
|---|---:|---:|---|
| A012 | 57.72% | 94.85% | A030 |
| A011 | 58.61% | 91.94% | A012 |
| A002 | 61.82% | 81.45% | A019 |
| A010 | 62.64% | 91.94% | A034 |
| A029 | 66.55% | 90.55% | A030 |
| A017 | 69.71% | 93.80% | A016 |
| A030 | 69.82% | 97.45% | A012 |
| A016 | 71.43% | 97.80% | A017 |
| A041 | 71.74% | 92.75% | A037 |
| A047 | 74.28% | 94.57% | A044 |
| A044 | 74.64% | 93.12% | A047 |

The worst 10 classes account for **38.98% of all FMSE errors** in this diagnostic.

Strongest confusion directions:

- A016 -> A017: 23.44%
- A011 -> A012: 17.58%
- A010 -> A034: 17.58%
- A012 -> A030: 17.28%
- A030 -> A012: 15.27%
- A012 -> A011: 13.60%
- A017 -> A016: 13.50%
- A031 -> A032: 10.51%
- A034 -> A010: 9.78%
- A047 -> A044: 9.42%

## Interpretation

The dominant hard-class pattern is **local ranking failure**, not total representation failure:

- many hard classes have Top-5 above 90% while Top-1 is much lower;
- the correct class is often already in the local candidate set;
- broad model replacement is therefore unnecessarily destructive.

A002 is different: Top-5 is only **81.45%**, suggesting a stronger representation deficit than the other hard classes.

## Advantages

- First recent representation change with a clear XSUB gain.
- No parameter increase over R4.
- Preserves XSET.
- Very strong performance on many classical NTU actions.
- Compatible with the established R4 training/inference pipeline.
- Provides a strong base to protect while adding targeted fine-motion capacity.

## Weaknesses

- NTU120 remains at ~77-78%, far below the desired 80-85% range.
- Fine/confusable class neighborhoods dominate the remaining errors.
- Existing M4/Router/G4/descriptor path still compresses motion information after FMSE.
- A global 120-way classifier does not explicitly resolve local rival neighborhoods.
- Some classes (especially A002-like cases) may need genuinely richer representation, not just reranking.

## Recommended next improvement

Do **not** replace FMSE globally.

Build a **conditional fine-motion/rival specialist** that:

1. leaves the FMSE logits untouched by default;
2. taps richer pre-pooling motion information;
3. activates only for ambiguous local rival neighborhoods;
4. changes only logits inside the active cluster;
5. initializes the correction at zero;
6. uses a protection/distillation loss to avoid breaking confident FMSE decisions;
7. derives hard clusters from training-only statistics for the real NTU120 experiment, rather than hard-coding validation-derived clusters.

Diagnostic NTU60 neighborhoods worth investigating (not hard-coding directly):

- {A011, A012, A029, A030}
- {A016, A017}
- {A010, A034}
- {A037, A041, A044, A047}
- {A031, A032}

## Decision

**Keep FMSE as the protected backbone.**

The next strong architecture should be **NestSAR-RCE (Rival-Conditioned Evidence)** rather than another global replacement:

- FMSE supplies default logits;
- a D40 high-resolution pre-pooling evidence path preserves individual joint-time motion;
- rival-conditioned learned queries gather evidence for the current ambiguous class neighborhood;
- correction logits are zero-initialized and masked outside that neighborhood;
- confident FMSE decisions bypass the specialist;
- hard neighborhoods are learned from cross-fitted/augmented training-only predictions;
- fixed-vs-broken and per-class Top-5 become primary diagnostics.

This preserves the model's strong generalist behavior while targeting the exact cross-protocol fine-class failures observed in the audit.

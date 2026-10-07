# FMSE + Local Geometry — experiment report

Updated: 2026-10-07

## Goal

Test whether a training-only local prototype/subcenter geometry objective can improve the successful FMSE representation without changing inference cost or parameter count.

Inference architecture remains FMSE. Geometry banks exist only during training.

## Final NTU120 results

| Protocol | Best validation | Best epoch | Params |
|---|---:|---:|---:|
| XSUB | **77.336554%** | 33 | 1,831,932 |
| XSET | **78.458900%** | 25 | 1,831,932 |

Reference:

| Model | XSUB | XSET |
|---|---:|---:|
| R4 | 76.910387% | 78.460581% |
| FMSE | 77.330662% | 78.453856% |
| FMSE + Geometry | **77.336554%** | **78.458900%** |

Delta vs FMSE:

- XSUB: **+0.005892 pp**
- XSET: **+0.005044 pp**

These gains are noise-level and not architecturally meaningful.

## Geometry diagnostics

### XSUB

- Descriptor centers: **240/240**
- G4 centers: **240/240**
- Final lambda: **1.000**
- Final descriptor active: **9.96%**
- Final G4 active: **32.73%**
- Descriptor gap: **0.28524**
- G4 gap: **0.14694**
- Geometry loss contribution: **0.000518**
- At best epoch 33: D active **13.07%**, G4 active **35.01%**

### XSET

- Descriptor centers: **240/240**
- G4 centers: **240/240**
- Final lambda: **1.000**
- Final descriptor active: **15.96%**
- Final G4 active: **38.67%**
- Descriptor gap: **0.24929**
- G4 gap: **0.13170**
- Geometry loss contribution: **0.000826**
- At best epoch 25: D active **21.94%**, G4 active **42.56%**

## Class-error audit of the exact best checkpoints

The best XSUB/XSET checkpoints from this branch were audited end-to-end and reproduced with **+0.000000 pp** delta.

### XSUB

- Top-1: **77.336554%**
- Top-5: **93.682123%**
- macro recall: **78.952%**

Weakest classes are dominated by fine-grained actions:

- A073 staple book: 27.85%
- A072 make victory sign: 36.35%
- A074 counting money: 39.65%
- A071 make OK sign: 43.30%
- A084 play magic cube: 47.03%
- A091 open a box: 47.91%
- A105 blow nose: 49.22%
- A082 fold paper: 50.96%
- A075 cutting nails: 51.14%
- A012 writing: 53.31%

Largest confusions include A072->A071 (169), A073->A076 (154), A071->A072 (141), A074->A084 (87) and A074->A075 (86).

### XSET

- Top-1: **78.458900%**
- Top-5: **94.043075%**
- macro recall: **78.419%**

Weakest classes:

- A072 make victory sign: 39.80%
- A012 writing: 40.76%
- A073 staple book: 40.78%
- A074 counting money: 43.62%
- A084 play magic cube: 49.28%
- A071 make OK sign: 50.41%
- A011 reading: 52.20%
- A107 wield knife towards other person: 52.64%
- A076 cutting paper with scissors: 52.66%
- A075 cutting nails: 53.43%

Largest confusions include A072->A071 (128), A071->A072 (119), A073->A076 (93), A076->A073 (91), A017->A016 (83), A016->A017 (79), A056->A118 (78) and A012->A030 (77).

Seven weak classes recur in both protocols: **A071/A072/A073/A074/A075/A084/A012**.

This strongly reinforces the negative geometry conclusion: the residual error is structured around specific rival classes and fine evidence, not a universal late-embedding margin problem.

## What worked

- Prototype banks fully populated.
- Geometry ramp reached full strength.
- Loss remained active on a nontrivial fraction of samples.
- Training was numerically healthy.
- No inference parameters or FLOPs were added.

## What failed

Despite being active and correctly implemented, local subcenter geometry produced only about **+0.005 pp** on each protocol.

This repeats the earlier negative result from global prototype-margin experiments: changing final/late geometry does not materially improve validation separation.

## Interpretation

The mechanism is **not broken**; the hypothesis is weak.

The descriptor gaps are already relatively large while difficult fine-class errors remain. The remaining bottleneck is therefore not well described by a universal embedding-margin constraint.

The most plausible issue remains:

**fine-grained information is either compressed too early or local rival classes need specialized discrimination rather than a global metric objective.**

## Do not repeat

Do not spend another run simply changing:

- margin 0.10 -> 0.15/0.20;
- geometry weight;
- number of subcenters;
- rival-k;
- prototype EMA;
- global geometry strength.

Such sweeps are too close to two already-negative prototype/geometry experiments.

## Decision

**Archive as a clean negative result.**

Keep the checkpoints because they reproduce exactly and are the strongest FMSE-architecture weights currently available. Their inference graph is still plain FMSE.

Do not continue geometry-loss tuning. The full NTU120 class audit redirects the next architecture toward a protected FMSE base plus a rival-conditioned high-resolution evidence path.

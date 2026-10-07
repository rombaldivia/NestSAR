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

Keep the checkpoint because it is numerically the best XSUB FMSE-family checkpoint and its inference graph is still plain FMSE, but do not continue the geometry-loss research line.

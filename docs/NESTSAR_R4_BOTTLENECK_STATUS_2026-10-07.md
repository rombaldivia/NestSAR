# NestSAR R4 bottleneck — consolidated status

Updated: 2026-10-07

## Reference model

R4 family:

- XSUB: **76.910387%**
- XSET: **78.460581%**
- params: **1,831,932**
- historical XLA/static metric: ~0.0296 GFLOPs (not strict/math FLOPs)
- compiler-independent D112 reference audit: **61.500832 MFLOPs** using 1 MAC = 2 FLOPs

## Direct causal / stage evidence

Representation transfer audit:

- spatial excess: 0.00716
- M4: 0.01471
- Router: 0.03449
- G4: 0.06302
- Descriptor: 0.13777

Largest sharpening jump:

- G4 -> Descriptor: +0.07475
- Router -> G4: +0.02853
- M4 -> Router: +0.01978
- Spatial -> M4: +0.00755

Interpretation: later stages sharpen a representation whose train geometry transfers imperfectly to validation.

## Stream / router evidence

XSET stream oracle audit:

- J: 67.8952
- B: 68.5593
- JM: 66.8511
- BM: 68.0767
- any-stream oracle: 86.6755
- main: 78.4404
- final: 78.4606
- oracle-final gap: +8.2149 pp
- adaptive head gain: +0.0202 pp

Baseline router weights approximately:

- J: 1.22%
- B: 21.77%
- JM: 75.69%
- BM: 1.32%

Causal tests:

- removing Spatial-2 -> ~17.76%
- router bypass -> ~65.07%
- uniform router -> ~72.74%
- small scalar routing/spatial changes mostly hurt

Conclusion: Spatial-2 and the router are important, but scalar tuning around the current solution is not a promising path.

## Forgetting/controller evidence

Forensic gradient audit:

- balanced train subset loss 0.482656, accuracy 100%
- balanced validation subset loss 1.27496, accuracy 76.667%
- train gradient norm 0.385
- validation gradient norm 5.223
- train/val gradient cosine -0.0214
- sign conflicts ~49.64%

Controller:

- eta ~0.19943
- alpha ~0.99879

However causal forgetting/write sweeps produced only noise-level change (best ~+0.005 pp, p=0.648).

Conclusion: **global forgetting/write strength is not the main bottleneck**.

## Head/readout evidence

Frozen readout experiments:

XSET:
- fused linear: 68.3312
- concat linear: 76.1639
- static experts: 74.2130
- gated experts: 68.6630
- concat MLP: 77.0387, still -1.42184 pp vs base

Chosen nonlinear readout cross-protocol:
- XSUB: -1.11157 pp
- XSET: -1.42184 pp

Conclusion: final descriptors do not hide an easy +1-2 pp that a better generic classifier can recover.

## Width evidence

- D128 MTS: XSUB 77.00465, XSET 78.26219
- D192: XSUB 61.6351, XSET 63.5187

Conclusion: generic width is not the bottleneck; excessive width can collapse optimization/generalization.

## Reframe evidence

Native reframe without finetuning:

- XSUB T16 76.32 -> T24 74.96 -> T32 68.95
- XSET T16 78.06 -> T24 74.22 -> T32 65.33

Conclusion: simply increasing frame count after training does not solve the problem.

## Prototype / geometry evidence

Global prototype-margin: no useful gain.

FMSE + local two-subcenter geometry:

- XSUB 77.336554 vs FMSE 77.330662: **+0.005892 pp**
- XSET 78.458900 vs FMSE 78.453856: **+0.005044 pp**

Banks were fully populated and active, so this is a negative hypothesis result rather than an implementation failure.

## First clear positive architecture change

FMSE:

- R4 XSUB 76.910387 -> **77.330662** (+0.420275 pp)
- R4 XSET 78.460581 -> **78.453856** (-0.006725 pp)
- params unchanged at 1,831,932

Interpretation: changing **how motion is represented before later compression** is more productive than changing late geometry, generic heads or global memory scalars.

## NTU60 class audit evidence

Using the strongest FMSE-architecture XSUB checkpoint:

- restricted NTU60 Top-1: **85.685692%**
- restricted Top-5: **96.579123%**
- strict 120-way Top-1: **83.568872%**
- strict Top-5: **95.529811%**
- predictions in A061-A120: 3.821192%

This diagnostic is not a publishable NTU60 benchmark because weights were trained on NTU120.

The important structural result:

- many classes are already >=90%;
- worst 10 classes account for **38.98% of FMSE errors**;
- most hard classes have high Top-5 but low Top-1;
- errors are concentrated in local rival neighborhoods.

Representative hard neighborhoods:

- A011/A012/A029/A030
- A016/A017
- A010/A034
- A037/A041/A044/A047
- A031/A032

A002 is more representation-limited: Top-1 61.82%, Top-5 only 81.45%.

## Full NTU120 class-error audit

Using the strongest FMSE-architecture checkpoints (FMSE + training-only local geometry):

### XSUB

- n: **50,919**
- Top-1: **77.336554%**
- Top-5: **93.682123%**
- macro recall: **78.952%**
- checkpoint reproduction delta: **+0.000000 pp**

Weakest classes:

- A073 staple book: **27.85%**
- A072 make victory sign: **36.35%**
- A074 counting money: **39.65%**
- A071 make OK sign: **43.30%**
- A084 play magic cube: **47.03%**
- A091 open a box: **47.91%**
- A105 blow nose: **49.22%**
- A082 fold paper: **50.96%**
- A075 cutting nails: **51.14%**
- A012 writing: **53.31%**

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
- checkpoint reproduction delta: **+0.000000 pp**

Weakest classes:

- A072 make victory sign: **39.80%**
- A012 writing: **40.76%**
- A073 staple book: **40.78%**
- A074 counting money: **43.62%**
- A084 play magic cube: **49.28%**
- A071 make OK sign: **50.41%**
- A011 reading: **52.20%**
- A107 wield knife towards other person: **52.64%**
- A076 cutting paper with scissors: **52.66%**
- A075 cutting nails: **53.43%**

Largest confusion directions:

- A072 -> A071: **128**
- A071 -> A072: **119**
- A073 -> A076: **93**
- A076 -> A073: **91**
- A017 -> A016: **83**
- A016 -> A017: **79**
- A056 -> A118: **78**
- A012 -> A030: **77**

Seven classes are present in the weakest-10 list of both protocols:

- A071
- A072
- A073
- A074
- A075
- A084
- A012

This cross-protocol overlap is strong evidence that the remaining weakness is systematic rather than split-specific.

The recurrent error families are:

1. **micro hand/gesture geometry** — A069/A071/A072;
2. **fine object manipulation / local trajectory** — A073/A074/A075/A076/A082/A084/A091;
3. **direction / interaction semantics** — A016/A017, A050/A106, A056/A118, A107;
4. **local reading/writing/typing discrimination** — A011/A012/A030.

The large global Top-5 versus Top-1 gap means much of the missing accuracy is local ranking, not total absence of the correct class.

The next audit should compute per-class Top-5 recall for all 120 classes and separate:

- **Type R**: ranking failures, correct class usually in Top-5;
- **Type I**: representation failures, correct class frequently outside Top-5.

## JT32 replacement evidence

Deep JT32 global replacement failed:

- NTU120 XSUB best at epoch 26: **66.892516%**
- NTU60 restricted Top-1: **78.073634%**
- FMSE reference: **85.685692%**
- fixed 592 FMSE errors but broke 1,847 correct FMSE predictions
- net -1,255 clips

Conclusion: **do not replace the strong FMSE backbone globally**.

## Current bottleneck statement

The evidence now supports:

**The main remaining problem is concentrated fine-class rival discrimination, with a smaller subset of genuine representation failures.**

The full NTU120 audit strengthens this conclusion because the same weak classes and confusion pairs recur across both XSUB and XSET.

Not supported as primary solutions:

- generic width
- global forgetting strength
- global router scaling
- post-hoc generic MLP/readout
- global/local prototype margins
- naive longer-frame reframe
- tiny hand-only residuals
- full replacement by the under-capacity JT32 v1

## Current recommended direction — NestSAR-RCE

Protect FMSE and add **Rival-Conditioned Evidence (RCE)**:

1. FMSE/R4 remains the protected generalist and supplies the default logits.
2. A separate D40 high-resolution evidence path consumes richer pre-pooling joint-time motion.
3. Preserve complete motion families, including parent-relative phase/path terms that JT32-v1 accidentally dropped.
4. Use 1-2 parallel joint-time evidence blocks with temporal axial mixing, spatial axial mixing, spectral/DCT evidence and low-rank associative memory.
5. Replace generic mean readout with **rival-conditioned learned queries** over the joint-time tokens.
6. Build a small candidate set from FMSE Top-k plus training-derived rivals.
7. Zero-initialize the correction projection so the initial final logits exactly reproduce FMSE.
8. Mask correction logits outside the active candidate/rival set.
9. Gate intervention using FMSE margin, local entropy and specialist evidence; confident clips should bypass the specialist.
10. Use a protection/distillation loss to suppress destructive corrections on confident FMSE predictions.
11. Derive hard neighborhoods from cross-fitted or strongly augmented **training-only** predictions, not validation confusion.
12. Track **fixed vs broken** and per-class Top-5 as first-class diagnostics.
13. Treat Type-I representation failures separately from Type-R local ranking failures.

Current design estimates, not audited counts:

- added parameters: **~0.08-0.15M**
- worst-case invoked specialist compute: **~0.035-0.040 strict GFLOPs**
- if invoked on 20-30% of clips: estimated average added compute **~0.008-0.012 GFLOPs**

Do not publish those compute figures until a compiler-independent operator audit is committed.

This direction directly targets the observed class-error structure while preserving the strongest parts of the existing model.

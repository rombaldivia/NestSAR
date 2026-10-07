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

**The main remaining problem is concentrated local fine-class discrimination, with a smaller subset of true representation failures.**

Not supported as primary solutions:

- generic width
- global forgetting strength
- global router scaling
- post-hoc generic MLP/readout
- global/local prototype margins
- naive longer-frame reframe
- tiny hand-only residuals
- full replacement by the under-capacity JT32 v1

## Current recommended direction

Protect FMSE and add a **conditional rival/fine-motion specialist**:

1. base FMSE logits remain the default;
2. specialist sees richer pre-pooling joint-time motion;
3. specialist activates only for ambiguous local neighborhoods;
4. correction logits are zero-initialized;
5. logits outside the active rival cluster are unchanged;
6. protection/distillation loss suppresses destructive corrections on confident base predictions;
7. hard clusters for NTU120 training are derived from training-only statistics to avoid validation leakage;
8. separate representation-improvement treatment is allowed for A002-like failure modes.

This direction directly targets the observed fixed/broken cancellation problem while preserving the strongest parts of the existing model.

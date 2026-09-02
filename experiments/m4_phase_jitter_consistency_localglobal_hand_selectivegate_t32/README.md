# Selective Residual Hand Trust Gate (T32)

Mechanism diagnostic built on the validated LocalGlobal V2 + Hand-M4/G4-Lite T32 checkpoint.

The frozen model is unchanged. Only a 2,481-parameter trust gate is trained with 5-fold cross-fitting.

## Gate

Input: 153 features

- mean main descriptor: 112
- hand descriptor: 32
- main/hand top1-top2 margins: 2
- main/hand entropies: 2
- main/hand top1 confidences: 2
- same-top1 indicator: 1
- main probability of hand top1: 1
- hand probability of main top1: 1

MLP: `153 -> 16 -> 1`

Residual coefficient:

`alpha(x) = 0.20 + 0.15 * tanh(g(x))`

so `alpha(x)` is bounded to `[0.05, 0.35]` and initializes near the fixed common baseline `0.20`.

Final logits:

`main_logits + alpha(x) * hand_logits`

No attention. No change to the T16 main representation or T32 Hand-M4/G4 representation.

## Purpose

The previous sigmoid trust gate saturated close to 0.30 on most samples. This experiment tests whether explicit main/hand disagreement features plus a centered residual parameterization can learn meaningful sample-dependent trust rather than merely increasing the global hand coefficient.

This cross-fit result is a diagnostic only and must not be reported as the paper benchmark. If positive, the next clean ablation is a full from-scratch co-trained LocalGlobal + Hand-M4/G4 T32 + Selective Gate model.

# LocalGlobal V2 + Hand-M4/G4 T32 + Adaptive Trust Gate

Stage-1 diagnostic for sample-wise trust selection of the already validated T32 hand specialist.

## Motivation

The robust Hand-M4/G4 audit showed that the hand classifier adds about +1.27 to +1.59 pp to the stream oracle, yet the fixed 0.10 residual rescues only about 15-16% of samples where the main model is wrong and the hand classifier is correct. The next question is therefore not whether T32 works, but whether the model can learn when to trust it.

## Gate

The gate sees:

- mean 112-D main descriptor
- 32-D hand descriptor
- main top1-top2 logit margin
- hand top1-top2 logit margin
- main predictive entropy
- hand predictive entropy

It uses a 148 -> 16 -> 1 MLP and outputs:

`alpha(x) = 0.30 * sigmoid(g(x))`

Final logits are:

`main_logits + alpha(x) * hand_logits`

The gate adds exactly 2,401 parameters. No attention is used.

## Diagnostic protocol

This branch does **not** claim a paper benchmark. It freezes the existing Hand-M4/G4 checkpoint, caches validation logits/descriptors, performs stratified 5-fold cross-fitting, and evaluates each sample with a gate that was not trained on that sample. This gives a clean mechanism test for learnable trust selection.

If cross-fitted adaptive gating clearly beats fixed alpha on both XSUB and XSET, the next step is a proper from-scratch co-trained adaptive-gate ablation.

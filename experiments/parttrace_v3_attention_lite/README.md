# NestSAR PartTrace v3 — Attention-Lite anchored

This experiment is deliberately based on the exact canonical
`NestSAR-HOPE-Attention-Lite-D128-v1` source rather than the 110K-parameter
standalone PartTrace-v2 model.

## Non-negotiable baseline guards

The audit reconstructs the committed canonical Attention-Lite XSUB/XSET source
through `experiments.attention_lite_v1.canonical_integrated` and refuses to
continue unless the base initializes to exactly:

- 2,381,028 parameters
- 705 parameter leaves
- T16
- outer width D128
- attention D64 / H4 / Dh16

The canonical base keeps its original four streams and geometry path. PartTrace
is added as a residual information path; it does not replace the base.

## PartTrace residual branch

The residual branch uses the raw joint stream and preserves anatomical identity:

1. valid-presence root centering
2. presence-safe velocity and acceleration
3. 25 joints -> 10 anatomical parts
4. D64 part tokens
5. per-frame 10x10 part mixer
6. shared causal temporal trace with learned relative temporal bias
7. explicit left/right forearm-hand full-resolution traces
8. learned global part pooling
9. learned temporal pooling
10. gated residual logit fusion with the exact Attention-Lite logits

The residual fusion gate is initialized small so the network begins close to the
validated Attention-Lite behavior rather than replacing it at initialization.

## Why this design

PartTrace-v2 reached useful NTU120 accuracy at only ~110K parameters, but it had
silently discarded most of Attention-Lite's capacity, streams, and geometry.
PartTrace-v3 keeps those proven components and asks a cleaner question: does an
additional part-preserving fine-motion path improve the canonical model?

## First step

Run the audit before writing/trusting a full trainer:

```bash
python -u experiments/parttrace_v3_attention_lite/audit_model.py --protocol xsub
```

The audit prints base guards, total parameters, added parameters, and XLA GFLOPs.
Do not make a paper/training claim until those guards pass.

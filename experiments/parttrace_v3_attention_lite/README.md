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
3. explicit bone / torso-distance / hand-geometry features
4. learned joint-to-part pooling over 10 anatomical parts
5. configurable part-token width and part attention
6. shared causal temporal trace with learned relative temporal bias
7. explicit left/right forearm-hand full-resolution traces
8. independent global / left-hand / right-hand temporal pooling
9. configurable global and final Dense widths
10. bounded residual logit fusion plus dynamic four-stream controller

The residual fusion gate is initialized small so the network begins close to the
validated Attention-Lite behavior rather than replacing it at initialization.

## v3.2 candidate configuration

The current candidate configuration is:

- `part_dim=64`
- `part_heads=4`
- `global_dim=192`
- `dense_dim=192`
- `branch_dropout=0.12`
- 60 epochs, patience 10, seed 128
- base LR `4e-4`, branch LR `6e-4`, controller LR `1.5e-4`
- EMA `0.995`

The machine-readable preset and observed live snapshot are stored in:

`candidate_v32_d64_g192_d192.json`

### Current live snapshot — 2026-08-27

These are intermediate values from the running experiment, not final paper results:

| Protocol | Current phase | Partial train accuracy | Best completed validation | Best epoch | Partial loss |
| --- | --- | ---: | ---: | ---: | ---: |
| XSUB | TRAIN E005 | 66.38% | 59.90% | 4 | 1.839 |
| XSET | TRAIN E006 | 72.36% | 62.65% | 5 | 1.632 |

Do not compare the partial training accuracy directly against the validation best.
The candidate should only be promoted after the full run and exact XLA audit are complete.

## Why this design

PartTrace-v2 reached useful NTU120 accuracy at only ~110K parameters, but it had
silently discarded most of Attention-Lite's capacity, streams, and geometry.
PartTrace-v3 keeps those proven components and asks a cleaner question: does an
additional part-preserving fine-motion path improve the canonical model?

## Audit / training

Use the v3.2 Kaggle launcher with `%run` to keep exactly two persistent progress bars:

```python
%run experiments/parttrace_v3_attention_lite/run_both_t4_v32_kaggle.py \
    --gpu-xsub 0 \
    --gpu-xset 1 \
    --epochs 60 \
    --patience 10 \
    --seed 128 \
    --batch-size 32 \
    --eval-batch-size 64 \
    --part-dim 64 \
    --part-heads 4 \
    --global-dim 192 \
    --dense-dim 192 \
    --branch-dropout 0.12 \
    --audit-first
```

The progress bars display live accuracy, loss, and the best completed validation
accuracy/epoch for each protocol. Do not make a paper claim until the audit and
both full protocol runs are complete.

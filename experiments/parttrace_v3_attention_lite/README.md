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

---

# v3.3 TokenPreserve — T16 Pareto experiment

v3.3 returns to the strongest efficiency setting: **T16** and the exact
2,381,028-parameter Attention-Lite backbone. It removes the second PartTrace
temporal-attention backbone and instead preserves fine anatomical evidence until
a cheap learned readout.

## Architecture

1. raw T16 skeleton input
2. root-relative position + velocity + acceleration + bone + torso-distance features
3. learned joint-to-part pooling over 10 semantic parts
4. explicit left/right hand geometry injection
5. retain all `16 x 2 x 10 = 320` part-time-person tokens at D64
6. add learned time/person/part identity embeddings
7. `K=8` learned evidence queries cross-attend to all 320 tokens
8. tiny K-token self-mixer (K=8, so only an 8x8 attention map)
9. keep all K token identities until `K*D -> Dense192 -> 120`
10. add branch logits to the untouched Attention-Lite logits through a bounded residual gate

There is **no 320x320 self-attention** and no second full temporal backbone.
The v3.3 objective is to preserve more evidence while keeping the full model near
the original Attention-Lite Pareto point.

`global_dim` remains in the CLI for controlled width sweeps, but in v3.3 it is
the tiny readout-mixer FFN hidden width rather than a second global temporal trace.

## Recommended v3.3 configuration

- frames: `16`
- token/part dim: `64`
- heads: `4`
- preserved fine tokens: `320`
- learned readout tokens: `8`
- mixer hidden dim (`global_dim` CLI): `128`
- final dense dim: `192`
- dropout: `0.12`
- exact canonical Attention-Lite base unchanged
- no dynamic stream-controller correction in the v3.3 forward path

The exact parameter and GFLOP cost must be taken from `--audit-first`; the design
target is roughly 2.5M total parameters and <0.085 GFLOPs/clip.

## Kaggle dual-T4 run

Use `%run`, not `!python`, so XSUB and XSET each reuse one persistent notebook tqdm bar:

```python
%run experiments/parttrace_v3_attention_lite/run_both_t4_v33_kaggle.py \
    --gpu-xsub 0 \
    --gpu-xset 1 \
    --dataset auto \
    --frames 16 \
    --readout-tokens 8 \
    --epochs 60 \
    --patience 10 \
    --seed 128 \
    --batch-size 32 \
    --eval-batch-size 64 \
    --part-dim 64 \
    --part-heads 4 \
    --global-dim 128 \
    --dense-dim 192 \
    --branch-dropout 0.12 \
    --audit-first
```

---

# v3.4 TokenPreserve — pretrained / anti-collapse experiment

v3.4 keeps the T16/320-token inference design but fixes the main v3.3 training
mismatch: v3.3 preserved the Attention-Lite architecture but trained it from
scratch. v3.4 starts from a compatible trained Attention-Lite EMA checkpoint and
validates that base at epoch 0 before TokenPreserve training begins.

## Changes from v3.3

1. load a compatible canonical Attention-Lite EMA (`2,381,028` params / `705` leaves)
2. run a full epoch-0 base validation before optimization
3. freeze base gradients for 3 epochs, then ramp them over 3 epochs
4. reduce pretrained-base LR to `1e-4`
5. stronger branch auxiliary CE: `0.50 -> 0.20` as residual influence ramps in
6. residual gate starts at `0.10` full scale, bounded to at most `0.20`
7. lower preservation-branch masking: frame/joint/part = `0.03/0.04/0.01`
8. add learned query-specific part, person, and time score biases
9. add an anti-collapse diversity loss over the K readout attention maps
10. log query overlap, standalone TokenPreserve validation accuracy, base-gradient scale, and effective gate

The diversity loss is training-only and adds **zero inference GFLOPs**. Query
identity priors are tiny lookup tables, so the v3.4 inference cost should remain
very close to v3.3. Exact cost still comes from `--audit-first`.

## Recommended v3.4 run

```python
%run experiments/parttrace_v3_attention_lite/run_both_t4_v34_kaggle.py \
    --gpu-xsub 0 \
    --gpu-xset 1 \
    --dataset auto \
    --frames 16 \
    --base-checkpoint auto \
    --readout-tokens 8 \
    --epochs 60 \
    --patience 10 \
    --seed 128 \
    --batch-size 32 \
    --eval-batch-size 64 \
    --part-dim 64 \
    --part-heads 4 \
    --global-dim 128 \
    --dense-dim 192 \
    --branch-dropout 0.10 \
    --frame-mask-rate 0.03 \
    --joint-mask-rate 0.04 \
    --part-mask-rate 0.01 \
    --base-lr 1e-4 \
    --branch-lr 5e-4 \
    --gate-lr 1e-4 \
    --branch-aux-warmup-weight 0.50 \
    --branch-aux-final-weight 0.20 \
    --diversity-loss-weight 0.05 \
    --freeze-base-epochs 3 \
    --base-unfreeze-ramp-epochs 3 \
    --freeze-branch-epochs 2 \
    --branch-ramp-epochs 4 \
    --audit-first
```

`--base-checkpoint auto` searches Kaggle working/input storage for compatible
`best_ema.msgpack` files and prefers Attention-Lite/protocol-matching paths. The
worker refuses to silently fall back to scratch unless `--allow-scratch` is
explicitly supplied. Always inspect `PRETRAINED BASELINE VAL` before trusting a
v3.4 run.

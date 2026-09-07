# NestSAR-SM-ALL-T16 — Person-aware P2 revision

Branch: `experiment/nestsar-sm-all-person-aware-p2-t16`

This experiment keeps the neural input at **16 x 750** and keeps the historical
**1,826,556-parameter** SM-ALL budget while fixing the remaining person/P2 data
and representation gaps.

## Verified person-aware P2 v3 result

Dual-T4 NTU RGB+D 120 run, seed 128, fixed neural processing length T=16:

| Protocol | Best held-out accuracy | Best epoch | Last epoch |
| --- | ---: | ---: | ---: |
| XSUB | **76.971268%** | 24 | 29 |
| XSET | **78.423592%** | 31 | 36 |

Relative to the preceding corrected-preprocessing-v2 baseline:

| Protocol | corrected-v2 | person-aware P2 v3 | Gain |
| --- | ---: | ---: | ---: |
| XSUB | 76.321216% | **76.971268%** | **+0.650052 pp** |
| XSET | 78.062108% | **78.423592%** | **+0.361484 pp** |

Model size remains **1,826,556 parameters**.

Static-unrolled compute audit for the person-aware P2 v3 model:

- **29,412,272 FLOPs/clip**
- **29.412272 MFLOPs/clip**
- **0.029412272 GFLOPs/clip**
- **0.014706136 GMACs/clip** under the convention `1 MAC = 2 FLOPs`

Compared with corrected-v2 SM-ALL at 29.065216 MFLOPs/clip, person awareness adds
**0.347056 MFLOPs/clip (+1.194%)**.

The complete issue-closure audit passed **70/70 checks**. The full NTU120 cache
contains 113,945 usable samples; P2 is preserved in **100%** of raw clips in which
it is present, with zero complete P2 loss after T16 preprocessing.

### Evaluation-protocol note

`xsub_val` and `xset_val` are the official NTU120 held-out benchmark partitions.
This training run evaluated them every epoch for checkpoint selection and early
stopping. Therefore the numbers above are the verified scores of this experiment,
but they should not be described as untouched one-shot final-test scores in a
paper. Future architecture selection should use an internal validation split made
only from the official training partition, then evaluate the official held-out
partition after the model/configuration is frozen.

## Preprocessing

- Missing people/joints are still detected before centering and remain zero.
- Two occupied NTU source tracks now keep their source order; actors are no longer
  swapped according to whole-clip motion energy.
- Empty source tracks are compacted so a real actor is never stranded behind an
  empty P1 slot.
- Representative pose is chosen independently per person inside each temporal
  segment. If P2 exists anywhere in a segment, a valid P2 pose is selected even
  when the common segment midpoint is empty.
- Motion still uses every valid adjacent raw-frame transition exactly once.
- Fresh yaw/boundary augmentation is unchanged and preserves zero padding.

Preprocessing version: `sm-all-personaware-p2-segmentpose-v2`.

## Representation/model

- The existing learned `person_embed` remains explicit for P1 and P2.
- A mask-safe SpatialEncoder now prevents Dense biases/joint/person embeddings
  from fabricating features for an absent P2.
- The SM controller no longer averages over the person axis. Its fixed 15-D
  summary explicitly contains P1 pose, P2 pose, P2-P1 relative pose, P2-P1
  relative full displacement, and P1/P2/pair-presence bits.
- Input width, controller width, M4/G4, cross-stream routing, fusion and dynamic
  rank-2 head remain unchanged; no attention/GCN/TCN/Transformer is introduced.

Pipeline/cache version: `sm-all-shared-cache-personaware-p2-v3`.

Use new Kaggle paths so the v2 canonical cache cannot be reused accidentally:

- output: `/kaggle/working/NestSAR_SM_ALL_T16_PERSON_AWARE_P2_v3`
- cache: `/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3`

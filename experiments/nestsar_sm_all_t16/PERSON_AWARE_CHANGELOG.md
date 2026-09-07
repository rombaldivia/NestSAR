# NestSAR-SM-ALL-T16 — Person-aware P2 revision

Branch: `experiment/nestsar-sm-all-person-aware-p2-t16`

This experiment keeps the neural input at **16 x 750** and keeps the historical
**1,826,556-parameter** SM-ALL budget while fixing the remaining person/P2 data
and representation gaps.

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

The verified 76.321216% XSUB / 78.062108% XSET scores belong to the preceding
corrected-preprocessing-v2 baseline, not to this new person-aware experiment.

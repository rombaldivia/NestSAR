# NestSAR-RCEX-T16

Updated: 2026-10-07

RCEX is the strong replacement for the first RCE30 specialist experiment.

It keeps the proven FMSE/R4 generalist frozen, but substantially changes
candidate recall, gating, training supervision, and how frozen FMSE evidence is
used.

## Why RCE30 was changed

The first RCE30 run stayed almost exactly at the protected FMSE baseline while
its reported training accuracy reached about 100% within the first few epochs.

Source inspection exposed a real train/eval mismatch:

- training called the specialist with `force_class=y`;
- `CandidateBuilder` replaced one candidate slot with the true class;
- validation/inference did not know the true class and therefore could not do
  the same thing.

So the local ranker was trained on an easier candidate problem than the one it
faced at validation.  This can explain extremely high training accuracy with
almost no validation gain.

RCEX removes that mechanism completely.  The true label is never inserted into
the candidate set.

## Protected starting point

FMSE-architecture checkpoints:

- XSUB: **77.336554%**
- XSET: **78.458900%**

The FMSE EMA parameters remain outside the optimizer state.

## Full RCEX architecture

```
                           frozen FMSE/R4
                      +----------------------+
input ----------------> base final logits    |
        |             | 4 stream logits      |
        |             | 4 descriptors        |
        |             | fusion weights       |
        |             +----------+-----------+
        |                        |
        v                        |
rich joint-time evidence         |
[B,16,25,40]                     |
        |                        |
2 evidence blocks                |
        |                        |
per-joint temporal bank          |
[B,25,40]                        |
        |                        |
        +--> global 120-class retrieval
        |          |
        |          +--> retrieval Top-3
        |
        +------------------------------+
                                       |
candidate union:                        |
  final FMSE Top-3                      |
  + four stream Top-1                   |
  + retrieval Top-3                     |
  + two training-only rivals            |
                                       |
                           12 candidate slots
                                       |
                         rival-conditioned queries
                                       |
                           masked local correction
                                       |
final = base_logits + gate * correction
```

## Major changes vs RCE30

### 1. No teacher-forced candidates

RCE30:

```
training candidate set = natural candidates + TRUE CLASS
validation candidate set = natural candidates
```

RCEX:

```
training candidate set == validation candidate set
```

The true label is used only to compute losses and metrics.

### 2. Stream-oracle candidate routing

Previous R4 audits showed large complementary information across J/B/JM/BM
streams.  RCEX now uses it directly.

Candidate slots contain the Top-1 class from every frozen FMSE stream in
addition to final-model candidates.

This gives the specialist access to classes the final fusion may suppress even
when one stream strongly supports them.

### 3. Learned global evidence retrieval

A new 120-class class-conditioned retrieval head attends over all 25 joint
evidence tokens.

Its job is **candidate recall**, not final classification.

It is trained with global retrieval CE and contributes its Top-3 classes to the
local candidate set.

This gives Type-I / outside-Top3 errors a path into the specialist.

### 4. Candidate set is now multi-source

```
C(x) =
    Top3(final FMSE)
  + Top1(each of 4 FMSE streams)
  + Top3(global evidence retrieval)
  + 2 training-only rivals of final Top1
```

Fixed slots: **12** before duplicate merging.

### 5. Frozen FMSE descriptor context

The four 112-D FMSE descriptors are fused with the frozen FMSE fusion weights,
projected to D40, and supplied to every local rival query.

The specialist therefore does not have to reconstruct all high-level action
context from raw skeleton evidence.

### 6. Per-stream candidate support

For every candidate class, the local scorer sees all four frozen FMSE stream
logits for that class.

The scorer can learn patterns such as:

- JM strongly supports class A;
- B strongly supports class B;
- final fusion is uncertain;
- distal evidence favors A.

### 7. Rich evidence path remains

RCEX keeps the strong RCE evidence representation:

- pose;
- bone pose;
- full displacement;
- phase A;
- phase B;
- path;
- parent-relative full displacement;
- parent-relative phase A;
- parent-relative phase B;
- parent-relative path;
- explicit distal wrist/hand/tip/thumb motion;
- P1, P2 and inter-person relative evidence projected separately.

Carrier:

```
[B,16,25,40]
```

No global joint mean is used inside the evidence backbone.

### 8. Two parallel evidence blocks

Each block contains low-rank residual evidence branches for:

- channel evidence;
- temporal direction/difference;
- anatomical parent-relative difference.

All branches preserve the 16x25 carrier.

### 9. Order-sensitive temporal bank

Every joint keeps:

- temporal mean;
- last - first;
- second-half - first-half;
- DCT1;
- DCT2;
- DCT3.

These six temporal summaries are fused with an anatomical parent-relative
mixer.

### 10. Rival-conditioned readout is stronger

For candidate c against base Top-1 b:

```
q(c,b,m) = Wq(e_c - e_b + mode_m + 0.25 * base_context)
```

Four query modes attend over all 25 joint tokens.

Candidate scoring then consumes:

- four query contexts;
- candidate class embedding;
- frozen FMSE context;
- four stream-support logits;
- final FMSE candidate logit;
- retrieval candidate logit;
- final-FMSE logit gap.

### 11. Bounded zero-init correction

The raw local correction uses a zero-initialized output layer and:

```
delta = 3 * tanh(raw_delta)
```

So initial RCEX predictions are exactly FMSE, while corrections cannot diverge
without bound.

### 12. Gate no longer starts almost closed

RCE30 initialized the gate near 0.018 (bias -4), which scaled early correction
gradients heavily.

RCEX initializes near 0.18 (bias -1.5).

Because the correction itself is exactly zero-initialized, RCEX still equals
FMSE at initialization.

### 13. Explicit gate supervision

RCE30 only penalized gate size.

RCEX trains the gate with a binary target:

```
gate_target = 1
when base is wrong AND natural candidate set contains truth

gate_target = 0
otherwise
```

Gate inputs include:

- final FMSE margin;
- candidate entropy;
- candidate probability mass;
- retrieval margin;
- stream disagreement;
- fusion entropy;
- correction strength.

### 14. Stronger base protection

RCEX uses three separate protections:

1. KL(base || final) on base-correct clips;
2. true-class margin preservation on base-correct clips;
3. correction L2 on base-correct clips.

This directly targets the historical fixed-vs-broken cancellation problem.

### 15. Hard-example weighting

Examples receive more specialist weight when:

- FMSE is wrong;
- FMSE margin is low;
- frozen streams disagree.

Easy/high-confidence examples receive reduced CE pressure.

### 16. Training-only rival graph is stronger

Rivals are built from:

- clean training predictions;
- multiple fresh augmented training passes;
- rank-weighted final-model competitors;
- frozen stream winners.

No validation labels are used.

### 17. Diagnostics are stronger

Every epoch records:

- final validation accuracy;
- frozen base accuracy;
- fixed rate;
- broken rate;
- candidate coverage;
- retrieval Top-5;
- stream disagreement;
- gate mean.

The final per-class audit saves:

- base Top-1;
- RCEX Top-1;
- RCEX Top-5;
- delta;
- fixed;
- broken;
- candidate coverage;
- retrieval Top-5;
- gate mean;
- stream disagreement;
- Type-R / Type-I label.

## Training objective

```
L =
    hard_weight * CE(final)
  + 0.18 * CE(retrieval)
  + 0.10 * local_rival_loss
  + 0.25 * KL_protection
  + 0.20 * margin_protection
  + 0.08 * gate_supervision
  + 0.01 * correction_L2
  + 0.01 * clean_aug_consistency
```

Default specialist LR is reduced to **6e-4** because the new retrieval/gating
problem is better supervised and no longer needs the aggressive RCE30 update.

## Static compute estimate

The architecture-level estimate is:

- RCEX specialist: **~7.96 MFLOPs**
- increase over RCE30 rival specialist: **~+2.69 MFLOPs**

Using the same older Astra FMSE accounting reference (~64.69 MFLOPs):

- estimated combined total: **~72.65 MFLOPs**

This is still extremely small compared with conventional skeleton models.

These values are design/operator estimates, not final publication counts.  A
full compiler-independent audit is still required.

## Success criteria

Primary:

- improve both XSUB and XSET;
- candidate coverage materially higher than RCE30;
- net fixed > broken;
- hard classes improve without destroying strong classes.

Strong result:

- >= +0.50 pp both protocols.

Do **not** unfreeze FMSE unless frozen-base RCEX first produces positive
fixed-minus-broken behavior on both protocols.

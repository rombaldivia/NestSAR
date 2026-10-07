# NestSAR-RCE30-T16

Updated: 2026-10-07

This package implements the complete post-audit architecture proposed after the
FMSE/geometry/JT32/class-error work.  It is not another scalar tweak.

## Starting point

Protected base checkpoints:

- XSUB FMSE-architecture best: **77.336554%**
- XSET FMSE-architecture best: **78.458900%**

The base is loaded from the existing FMSE + training-only geometry checkpoints,
but geometry has no inference module: the frozen inference graph is FMSE.

The full NTU120 audit showed:

- XSUB Top-5 **93.682123%**
- XSET Top-5 **94.043075%**
- seven weakest classes recur across protocols:
  A071/A072/A073/A074/A075/A084/A012
- recurring rivals:
  A071<->A072, A073<->A076, A016<->A017,
  A056<->A118, A012->A030

Therefore RCE protects the strong generalist and adds local evidence only where
the base is ambiguous.

## What changed

### 1. Frozen FMSE base

The FMSE EMA parameters are never inserted into the RCE optimizer state.

Stage-2 training updates **only** RCE parameters.

This makes the protection structural, not merely a small learning rate.

### 2. Rich pre-pooling evidence

The specialist reconstructs complete joint-time evidence directly from the
canonical [B,16,750] input and keeps all 25 joints.

Ten evidence families are preserved:

1. pose
2. bone pose
3. full displacement
4. phase A
5. phase B
6. path
7. parent-relative full displacement
8. parent-relative phase A
9. parent-relative phase B
10. parent-relative path

For every family, P1, P2 and P2-P1 are projected **separately**.  This avoids
JT32-v1's destructive [P1,P2,relative] 9->4 bottleneck.

### 3. Distal evidence

Explicit wrist/hand/hand-tip/thumb motion relations are added for both persons,
including inter-person relative distal motion.

### 4. D40 joint-time carrier

The specialist carrier is:

    [B,16,25,40]

Individual joint identity survives through both evidence blocks.

### 5. Two efficient evidence-refinement blocks

Each block has three parallel low-rank residual branches:

- channel evidence
- temporal direction/difference evidence
- anatomical parent-relative evidence

Each branch is D40 -> rank4 -> D40.

This is the strong architecture change that preserves the ~5 MFLOP specialist
budget instead of using expensive full 400x400 attention.

### 6. Order-sensitive temporal evidence bank

The final evidence bank is per joint, not mean pooled.

For every joint it uses:

- temporal mean
- last minus first
- second-half minus first-half
- DCT component 1
- DCT component 2
- DCT component 3

These six summaries are projected back to D40 and combined with an anatomical
parent-relative mixer.

Output:

    [B,25,40]

### 7. Training-only rival graph

Before specialist training, the frozen FMSE model predicts the **training set
only**, on both clean and augmented views.

For every true class, the two strongest competing classes are saved.

No validation labels are used to construct routing.

### 8. Candidate routing

At inference:

    C(x) = Top3_FMSE(x) + 2 training-rivals for each Top3 class

There are nine fixed candidate slots before duplicate removal.

During training the true class is forced into the final candidate slot only for
the local ranking loss.  Natural candidate coverage is tracked separately.

### 9. Rival-conditioned learned queries

The default RCE variant uses class embeddings.

For candidate c against the FMSE Top-1 class b:

    q(c,b,m) = Wq(e_c - e_b + mode_m)

with four learned query modes.

Each candidate query attends to all 25 joint evidence tokens.  Therefore
A071/A072 can search different evidence from A016/A017 or A056/A118.

### 10. Ordinary fixed-query ablation

The same package also implements the Astra fixed-query prototype:

    rce_variant = "fixed"

It uses four learned global queries over the same evidence bank, then predicts
a 120-way residual that is still masked to the candidate set.

Thus the experiment can compare under an identical schedule:

- unchanged frozen FMSE
- fixed learned-query RCE
- rival-conditioned RCE

### 11. Hard masked correction

Corrections are exactly zero outside the active candidate set.

The candidate correction projection is zero-initialized.

Therefore at initialization:

    final_logits == base_logits

to machine precision.

The dual-T4 launcher verifies this for **both fixed and rival prototypes** before
starting expensive training.

### 12. Confidence gate

The gate sees:

- FMSE top1-top2 margin
- candidate entropy
- candidate probability mass
- specialist correction strength

Its final layer is initialized with zero kernel and bias -4, so intervention is
initially small.  Exact prediction equivalence is guaranteed by the zero
correction projection.

### 13. Protection loss

On confident, base-correct training examples:

    KL(stopgrad(p_base) || p_final)

penalizes unnecessary changes.

This directly targets the historical "fixed some / broke more" failure mode.

### 14. Local rival ranking loss

The specialist explicitly learns to raise the true candidate above the hardest
active rival:

    softplus(logit_rival - logit_true)

### 15. Fresh-view consistency

Clean and augmented RCE predictions retain a small symmetric consistency term,
but the old R4 stream-auxiliary loss is **not** copied into RCE.

### 16. Automatic Type-R / Type-I class audit

After training, the best EMA specialist is evaluated class by class.

The worker saves:

- Top-1
- Top-5
- base Top-1
- fixed
- broken
- net fixed
- candidate coverage
- gate mean

Weak classes are labelled diagnostically:

- Type-R: Top-1 <75%, Top-5 >=90% -> ranking problem
- Type-I: Top-1 <75%, Top-5 <90% -> representation problem

Files:

    xsub/per_class.csv
    xsub/class_audit.json
    xset/per_class.csv
    xset/class_audit.json

## Training stages

This implementation is Stage 2:

1. load protected FMSE EMA;
2. derive training-only rival graph;
3. initialize RCE with exact zero correction;
4. freeze FMSE;
5. train only RCE;
6. select best RCE EMA by final validation accuracy;
7. audit fixed-vs-broken and per-class Top-5.

Stage 3 (optional FMSE Spatial-2 unfreeze at ~0.05x LR) is intentionally not
mixed into the first controlled run.  It should only be attempted if Stage 2
has net fixed > broken.

## Compute

The static architecture audit uses 1 MAC = 2 FLOPs.

Current specialist estimates:

- fixed-query RCE: **~4.93 MFLOPs**
- rival-conditioned RCE: **~5.27 MFLOPs**

The earlier Astra design accounting used an FMSE full-model reference around
64.69 MFLOPs and two specialist prototypes around ~69.71 MFLOPs total.  The
implementation remains in that ~+5 MFLOP regime.

These are architecture estimates.  A compiler-independent operator audit is
still required before publishing a final FLOP number.

## Parameters

The specialist is intentionally compact.  Exact counts are printed by the
launcher preflight because fixed and rival variants differ slightly.

The base 1,831,932 FMSE parameters remain frozen and protected.

## Default training config

- epochs 40
- patience 6
- specialist LR 8e-4
- min LR 2e-5
- warmup 8%
- weight decay 0.02
- dropout 0.05
- label smoothing 0.02
- rival loss 0.10
- protection KL 0.20
- clean/aug consistency 0.02
- gate regularization 0.005
- protect margin 1.0
- microbatch 64
- accumulation 4
- eval batch 256
- seed 128

## Primary success criteria

A successful RCE run must satisfy more than raw Top-1:

1. beat FMSE on both XSUB and XSET;
2. net fixed > broken;
3. no meaningful destruction of strong classes;
4. candidate coverage remains high;
5. hard-class gains are concentrated in training-derived rival neighborhoods.

Strong target: >= +0.50 pp on both protocols.

## Files

- model.py — full evidence architecture, fixed and rival query variants
- worker.py — frozen-base Stage-2 training, training-only rival graph, losses,
  exact resume/checkpointing, per-class audit
- launch.py — dual-T4 launcher and exact-baseline preflight
- audit.py — transparent specialist MAC/FLOP estimate

# NestSAR FMSE + Conditional Rival Specialist — design record

Updated: 2026-10-07

## Motivation

The current evidence says the strong FMSE backbone should be protected, not globally replaced.

Key observations:

- FMSE NTU120: XSUB **77.330662%**, XSET **78.453856%**.
- Best FMSE-architecture XSUB checkpoint: **77.336554%**.
- NTU60 diagnostic: restricted Top-1 **85.685692%**, Top-5 **96.579123%**.
- 21/60 diagnostic classes are >=90%.
- worst 10 classes account for **38.98%** of FMSE errors.
- most hard classes have high Top-5 but low Top-1, indicating local ranking failures.
- JT32 global replacement fixed 592 FMSE errors but broke 1,847, net **-1,255**.

Therefore the next architecture should improve only ambiguous local neighborhoods while leaving strong FMSE decisions intact.

## Proposed architecture

```
                         +---------------------------+
Input -----------------> | Existing FMSE/R4 backbone | ---> base logits
                         +---------------------------+
              |
              | richer pre-pooling joint-time motion
              v
         Fine-motion token encoder
              |
       1-2 low-width axial blocks
              |
       learned query readout
              |
        local rival scorer
              |
              v
final_logits = base_logits + gate * local_correction
```

## Core rules

### 1. FMSE is frozen/protected conceptually

The specialist must not be allowed to erase the already-strong general representation.

Initial behavior:

```
final_logits ~= base_logits
```

Use zero-initialized correction projection and a gate initialized near zero.

### 2. No global 120-way replacement

The specialist should not relearn all 120 classes.

For an active local rival set C:

```
delta_logit[c] != 0 only for c in C
delta_logit[c] = 0 for c outside C
```

### 3. Preserve rich motion before pooling

Specialist inputs should include the complete successful motion families:

- pose
- bone
- full displacement
- phase A
- phase B
- path
- parent-relative full displacement
- parent-relative phase A
- parent-relative phase B
- parent-relative path
- optional distal relations
- P1/P2/inter-person relations where valid

Do not repeat JT32-v1's 9 -> 4 over-compression for every family.

### 4. Learned query readout, not joint mean

Use a few learned queries over the joint-time token bank, for example:

- global
- distal
- interaction
- temporal/frequency

With 400 tokens and only a few queries, this is cheap and avoids the destructive joint mean used by JT32-v1.

### 5. Confidence-sensitive activation

The specialist should intervene only when:

- FMSE top candidates lie inside a learned hard neighborhood;
- and the top-1/top-2 margin is small or local entropy is high.

Confident easy predictions should bypass the specialist.

### 6. Protection loss

Use a term that penalizes unnecessary changes on confident FMSE examples, e.g. KL(base || final) or an explicit gate-suppression objective.

This directly addresses the historical fixed-vs-broken cancellation pattern.

## Diagnostic hard neighborhoods

From the NTU60 diagnostic only:

- {A011, A012, A029, A030}
- {A016, A017}
- {A010, A034}
- {A037, A041, A044, A047}
- {A031, A032}

These are **diagnostic evidence**, not final hard-coded NTU120 clusters.

For the real NTU120 experiment, neighborhoods must be derived from **training-only** confusion/rival statistics to avoid validation leakage.

## Representation-failure path

A002 is different from most hard classes:

- Top-1: **61.82%**
- Top-5: **81.45%**

The correct class is absent from the top-5 much more often, so a local reranker alone may be insufficient.

The specialist should therefore distinguish:

- ranking-only ambiguous cases;
- representation-deficient cases requiring richer motion evidence.

## Initial capacity target

Start small enough to preserve efficiency but large enough to avoid JT32 underfitting:

- width: D32-D40
- 1-2 specialist blocks
- 4 or 5 heads with head dimension ~8
- learned query readout
- no large generic FFN
- no global 120-way auxiliary losses on partial region views

Target added strict compute before exact audit:

- **~0.005-0.015 GFLOPs when invoked**
- conditional average compute lower if the gate is sparse

Do not publish this as an exact FLOP count until the operator audit exists.

## Success criteria

Primary:

- improve both XSUB and XSET over FMSE;
- no >0.10 pp regression on either protocol;
- net fixed > broken on validation;
- largest gains should occur in training-derived rival clusters.

Strong result:

- >= +0.50 pp on both protocols without materially increasing average inference compute.

## Non-goals

Do not repeat:

- generic width scaling;
- global prototype margins;
- local geometry weight sweeps;
- global forgetting sweeps;
- router scalar sweeps;
- post-hoc generic MLP classifiers;
- native T24/T32 reframe;
- full replacement of FMSE by JT32-v1.

## Implementation status

**Design only. No specialist code is committed yet.**

This branch exists to preserve the evidence, constraints and next architecture before implementation.

# FMSE + Hand Relations — experiment report

Updated: 2026-10-07

## Goal

Test whether a tiny pre-pooling distal hand/wrist relation residual can improve fine-action discrimination while preserving the successful FMSE/R4 architecture.

## Architecture change

Six distal-to-wrist relations are formed from the 12-D joint-motion representation:

- left hand - left wrist
- left hand tip - left wrist
- left thumb - left wrist
- right hand - right wrist
- right hand tip - right wrist
- right thumb - right wrist

The six 12-D relations produce a 72-D vector, projected:

```
72 -> 8 -> 24
```

and injected only into the distal/wrist joints before the existing joint-memory and part-pooling path.

## Cost

- FMSE parameters: **1,831,932**
- Extra hand-relation parameters: **800**
- Total: **1,832,732**
- Approximate relation-projector compute: **~0.049 MFLOPs** under 1 MAC = 2 FLOPs

## Advantage

This was a clean test of the hypothesis that fine hand relations should be added **before** spatial pooling rather than after the final descriptor.

It preserved:

- FMSE;
- R4 memory topology;
- established optimizer/training recipe;
- essentially unchanged compute.

## Limitation / result status

The run did **not clear the improvement bar** strongly enough to justify continuing with a tiny hand-only residual.

No exact final XSUB/XSET terminal scores were committed into this branch at the time of this report, so this report intentionally does **not invent** them.

The important conclusion is architectural: an 800-parameter residual is too weak/narrow to address the broader fine-class problem revealed by the later class audit.

## Why the hypothesis was incomplete

The hard-class audit shows the remaining errors are not only distal-hand errors. They form multiple rival neighborhoods involving:

- local motion ordering;
- body-part interactions;
- similar action pairs;
- local class ranking;
- some genuine representation failures.

A hand-only residual cannot cover all of these.

## Recommended use of this branch

Treat it as evidence for two principles:

1. preserve fine relations before pooling;
2. do not expect a tiny single-region residual to solve the full NTU120 fine-class bottleneck.

Reuse the relation features only as one input family inside a larger conditional fine-motion specialist.

## Decision

**Do not scale this module by width alone.** Move to a broader fine-motion/rival-specialist design while keeping FMSE protected.

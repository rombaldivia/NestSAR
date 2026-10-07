# NestSAR FMSE class-error audit — NTU120 + NTU60 diagnostic

Updated: 2026-10-07

## Checkpoints and reproduction

Audit source commit used by the Kaggle report runner:

- `c22eca737b3d67a620561065aab7f06ef2aa33b1`

Audited EMA checkpoints:

- XSUB: `/kaggle/working/NestSAR_R4_FMSE_LOCAL_GEOMETRY_T16_v1/xsub/best.msgpack`
- XSET: `/kaggle/working/NestSAR_R4_FMSE_LOCAL_GEOMETRY_T16_v1/xset/best.msgpack`

The geometry mechanism is training-only, so the inference graph is still the FMSE architecture.

Checkpoint reproduction delta for both NTU120 protocols:

- **+0.000000 pp**

This confirms the audit exactly reproduces the saved checkpoints.

---

## NTU120 XSUB

- validation samples: **50,919**
- Top-1: **77.336554%**
- Top-5: **93.682123%**
- macro recall: **78.952%**

### Weakest classes

| Rank | Class | Action | Recall | Errors |
|---:|---|---|---:|---:|
| 1 | A073 | staple book | **27.85%** | 412 / 571 |
| 2 | A072 | make victory sign | **36.35%** | 366 / 575 |
| 3 | A074 | counting money | **39.65%** | 344 / 570 |
| 4 | A071 | make OK sign | **43.30%** | 326 / 575 |
| 5 | A084 | play magic cube | **47.03%** | 303 / 572 |
| 6 | A091 | open a box | **47.91%** | 299 / 574 |
| 7 | A105 | blow nose | **49.22%** | 292 / 575 |
| 8 | A082 | fold paper | **50.96%** | 282 / 575 |
| 9 | A075 | cutting nails | **51.14%** | 278 / 569 |
| 10 | A012 | writing | **53.31%** | 127 / 272 |

The ten weakest XSUB classes contribute **3,029 errors**.

### Largest confusion directions

| True | Predicted | Count |
|---|---|---:|
| A072 make victory sign | A071 make OK sign | **169** |
| A073 staple book | A076 cutting paper with scissors | **154** |
| A071 make OK sign | A072 make victory sign | **141** |
| A074 counting money | A084 play magic cube | **87** |
| A074 counting money | A075 cutting nails | **86** |
| A072 make victory sign | A069 thumb up | **75** |
| A106 hit other person with something | A050 punch/slap other person | **75** |
| A118 exchange things | A056 give something to other person | **72** |

---

## NTU120 XSET

- validation samples: **59,477**
- Top-1: **78.458900%**
- Top-5: **94.043075%**
- macro recall: **78.419%**

### Weakest classes

| Rank | Class | Action | Recall | Errors |
|---:|---|---|---:|---:|
| 1 | A072 | make victory sign | **39.80%** | 295 / 490 |
| 2 | A012 | writing | **40.76%** | 295 / 498 |
| 3 | A073 | staple book | **40.78%** | 289 / 488 |
| 4 | A074 | counting money | **43.62%** | 274 / 486 |
| 5 | A084 | play magic cube | **49.28%** | 248 / 489 |
| 6 | A071 | make OK sign | **50.41%** | 243 / 490 |
| 7 | A011 | reading | **52.20%** | 239 / 500 |
| 8 | A107 | wield knife towards other person | **52.64%** | 233 / 492 |
| 9 | A076 | cutting paper with scissors | **52.66%** | 231 / 488 |
| 10 | A075 | cutting nails | **53.43%** | 224 / 481 |

The ten weakest XSET classes contribute **2,571 errors**.

### Largest confusion directions

| True | Predicted | Count |
|---|---|---:|
| A072 make victory sign | A071 make OK sign | **128** |
| A071 make OK sign | A072 make victory sign | **119** |
| A073 staple book | A076 cutting paper with scissors | **93** |
| A076 cutting paper with scissors | A073 staple book | **91** |
| A017 take off a shoe | A016 wear a shoe | **83** |
| A016 wear a shoe | A017 take off a shoe | **79** |
| A056 give something to other person | A118 exchange things | **78** |
| A012 writing | A030 typing on keyboard | **77** |

---

## Cross-protocol stability

Seven classes appear in the weakest-10 list of **both** XSUB and XSET:

- **A071** make OK sign
- **A072** make victory sign
- **A073** staple book
- **A074** counting money
- **A075** cutting nails
- **A084** play magic cube
- **A012** writing

This overlap is important because it argues against a split-specific artifact. The same fine-discrimination failures survive both subject and setup changes.

The strongest recurring pair structures also survive protocols:

- **A071 <-> A072**
- **A073 <-> A076**
- **A016 <-> A017**
- **A056 <-> A118**
- **A012 -> A030**

These are better architecture targets than a generic global capacity increase.

---

## Error-family interpretation

### 1. Micro hand / gesture geometry

Examples:

- A071 OK sign
- A072 victory sign
- A069 thumb up

Evidence:

- A072 -> A071 is the largest XSUB confusion and the largest XSET confusion.
- A071 -> A072 is also one of the largest errors in both protocols.

Required evidence likely lives in:

- finger/hand/wrist relative geometry;
- left/right distal relations;
- local shape/orientation;
- fine temporal evolution of the hand.

### 2. Fine object manipulation / local trajectory

Examples:

- A073 staple book
- A076 cutting paper with scissors
- A074 counting money
- A075 cutting nails
- A084 play magic cube
- A082 fold paper
- A091 open a box

These classes share similar global posture and arm motion while differing in local trajectory, motion ordering and distal-object interaction.

### 3. Direction / interaction semantics

Examples:

- A016 wear shoe <-> A017 take off shoe
- A056 give something <-> A118 exchange things
- A106 hit person with something -> A050 punch/slap person
- A107 wield knife towards other person

These require directional temporal evidence and/or explicit inter-person relational features.

### 4. Local ranking vs missing representation

The global Top-5 / Top-1 gaps are large:

- XSUB: **93.682123% vs 77.336554%**
- XSET: **94.043075% vs 78.458900%**

This shows substantial correct-neighborhood information remains in the model.

The next audit should therefore save **per-class Top-5 recall** for all 120 classes and label weak classes as:

- **Type R — ranking failure:** correct class usually appears in Top-5;
- **Type I — information/representation failure:** correct class frequently absent from Top-5.

This distinction should decide whether a local rival reranker is sufficient or richer specialist evidence is required.

---

## NTU60 compatibility diagnostic

Processed NTU60 XSUB subset:

- n: **16,487**
- strict 120-way Top-1: **83.568872%**
- strict 120-way Top-5: **95.529811%**
- restricted A001-A060 Top-1: **85.685692%**
- restricted A001-A060 Top-5: **96.579123%**
- recovered by restricting to A001-A060: **349 clips**

Weak NTU60 classes:

- A012 writing: 57.72%
- A011 reading: 58.61%
- A002 eat meal/snack: 61.82%
- A010 clapping: 62.64%
- A029 playing with phone/tablet: 66.55%
- A017 take off a shoe: 69.71%
- A030 typing on keyboard: 69.82%
- A016 wear a shoe: 71.43%
- A041 sneeze/cough: 71.74%
- A047 touch neck: 74.28%

Largest NTU60 rival pairs:

- A016 -> A017: 64
- A010 -> A034: 48
- A011 -> A012: 48
- A012 -> A030: 47
- A030 -> A012: 42
- A012 -> A011: 37
- A017 -> A016: 37
- A031 -> A032: 29

This is a diagnostic compatibility result only, not a publication-fair NTU60 benchmark, because the weights were trained on NTU120.

---

## What the audit rules out

The audit, combined with previous causal experiments, argues against these as the next primary direction:

- another global width increase;
- another global forgetting/write sweep;
- another scalar router sweep;
- another global/local prototype-margin sweep;
- another generic final MLP/readout;
- another full replacement of FMSE;
- a tiny hand-only residual;
- naive T24/T32 reframe.

These mechanisms do not match the highly concentrated, rival-specific error structure.

---

## Recommended next architecture

The current recommendation is **NestSAR-RCE: Rival-Conditioned Evidence**.

Principles:

1. **Keep FMSE/R4 as the protected generalist.**
2. Add a separate high-resolution fine-evidence path.
3. Preserve individual joint-time motion before anatomical pooling.
4. Include complete motion families:
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
   - distal relations
   - inter-person relations
5. Use 1-2 low-width joint-time evidence blocks.
6. Use **rival-conditioned learned queries**, not a generic global descriptor.
7. Build a small candidate set from FMSE Top-k plus training-derived rivals.
8. Mask all specialist corrections outside the candidate set.
9. Zero-initialize the correction projection so the initial model exactly reproduces FMSE.
10. Gate intervention using FMSE margin / local entropy / specialist evidence.
11. Train candidate neighborhoods from cross-fitted or augmented **training-only** predictions to avoid validation leakage.
12. Track **fixed vs broken** as a first-class metric.
13. Keep a separate representation-improvement path for Type-I classes.

Design target, not audited compute:

- specialist width: **D40**
- 2 evidence blocks
- added parameters: approximately **0.08-0.15M**
- worst-case invoked specialist compute: approximately **0.035-0.040 strict GFLOPs**
- if invoked on 20-30% of clips, estimated average added compute: **~0.008-0.012 GFLOPs**

Do not publish the compute estimates until an exact operator audit exists.

---

## RCE implementation status

The complete post-audit architecture is now implemented on:

- branch: `experiment/nestsar-rce30-t16`
- package: `experiments/nestsar_rce30_t16/`

Implemented components:

- frozen/protected FMSE base;
- full pre-pooling evidence families;
- explicit distal + inter-person evidence;
- persistent D40 joint-time carrier;
- two rank-4 channel/temporal/anatomical evidence blocks;
- order-sensitive mean/direction/half/DCT temporal bank;
- training-only rival graph from clean+augmented training predictions;
- Top3 + 2 rivals/class candidate routing;
- rival-conditioned 4-mode learned queries;
- ordinary fixed-query ablation under the same schedule;
- zero-initialized local correction;
- candidate-masked logits;
- confidence gate;
- rival ranking loss;
- protection KL;
- clean/aug consistency;
- exact-baseline preflight for both fixed/rival prototypes;
- automatic per-class Top-1/Top-5/fixed/broken/Type-R/Type-I audit.

Static specialist compute estimate:

- fixed-query: ~4.93 MFLOPs;
- rival-conditioned: ~5.27 MFLOPs;

under 1 MAC = 2 FLOPs. Exact publishable compute still requires the operator auditor.

## RCE30 early-run diagnosis and RCEX correction

The first RCE30 run remained almost exactly at the protected FMSE baseline
while training accuracy reached approximately 100% in the first few epochs.

Source inspection identified a concrete train/eval mismatch:

- training passed `force_class=y` into the specialist;
- the candidate builder replaced one candidate slot with the ground-truth class;
- validation/inference did not and cannot do this.

Therefore the local ranker was optimized on an easier candidate problem than
the real inference problem.  The early near-100% training accuracy is not
evidence that the natural routing problem was solved.

A new full branch was created:

- `experiment/nestsar-rcex-t16`

RCEX removes true-class candidate injection entirely and adds:

- final FMSE Top-3 candidates;
- four FMSE stream Top-1 candidates;
- a learned global 120-class evidence-retrieval Top-3;
- two training-only rivals of the final Top-1;
- frozen FMSE descriptor context;
- per-stream candidate support;
- explicit gate supervision;
- KL + margin + correction-norm protection;
- hard-example weighting based on base error/margin/stream disagreement;
- stronger training-only rival mining from clean + multiple augmented passes.

The RCEX static specialist estimate is approximately **7.96 MFLOPs** under
1 MAC = 2 FLOPs, versus approximately 5.27 MFLOPs for the first rival RCE30
prototype.

## Decision

The strongest current diagnosis is:

> **NestSAR is already a strong generalist. The main remaining bottleneck is concentrated fine-class rival discrimination, with a smaller subset of true representation failures.**

The next serious architecture should therefore be a protected FMSE base plus a rival-conditioned high-resolution evidence path, not another global replacement.

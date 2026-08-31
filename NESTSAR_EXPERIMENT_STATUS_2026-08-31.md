# NestSAR Experiment Status — 2026-08-31

This is the current cross-branch research ledger for the recent M4-derived T16 line. It supersedes the 2026-08-28 ledger for **current experiment status**, while the older file remains as a historical snapshot.

## Status legend

- **Verified/consolidated**: completed run with stable checkpoint metadata and audited architecture/compute.
- **Verified ablation**: completed controlled experiment whose result is useful even if it did not improve the best score.
- **Pending**: branch/code exists, but no completed score is recorded yet.
- **Historical/re-audit required**: result exists in project history but exact checkpoint/config/compute should be re-audited before publication.

## Current T16 score and compute ledger

| Variant | Frames | XSUB | XSET | Params | Neural inference compute | Status | Main conclusion |
|---|---:|---:|---:|---:|---:|---|---|
| MotionLite | 16 | **70.634%** | — | **1,817,930** | **0.020089768 GFLOPs** | Verified/consolidated | Motion-aware T16 selection improved over older lightweight lines, but audit showed only ~60% temporal-path retention and a nearly useless pre-memory router. |
| MotionPreserve | 16 | **73.9763% @ E29** | **74.5212% @ E36** | **1,817,642** | **0.019704616 GFLOPs** | Verified/consolidated | Full-sequence segment summaries plus post-frame-memory routing produced the major gain: +3.34 pp XSUB over MotionLite at slightly lower neural compute. |
| Phase-T16 | 16 | **74.0372% @ E26** | **75.1702% @ E36** | **1,817,930** | **0.020194216 GFLOPs** | Verified/consolidated | Added first/second-half signed displacement. Small XSUB gain and stronger XSET gain; phase channels are used, but their controlled benefit is modest. |
| Phase + Jitter + Uniform Fusion | 16 | **74.607% @ E38 checkpoint** | **75.338% @ E33 checkpoint** | **1,816,130** | **0.020181636 GFLOPs** | Verified/consolidated | Training-only ±1-frame segment-boundary jitter plus fixed uniform final fusion improved generalization while removing the nearly-uniform learned fusion controller. |
| **Phase + Jitter + Consistency** | 16 | **74.8483% @ E42** | **75.5721% @ E31** | **1,816,130** | **0.020181636 GFLOPs** | **Verified/consolidated — current T16 champion** | Symmetric-KL canonical/jitter consistency (`lambda=0.08`) adds another +0.241 pp XSUB / +0.234 pp XSET over Jitter+Uniform with no inference-cost increase. |
| PhasePath + Jitter + Uniform | 16 | **74.5046% @ E24** | **75.3098% @ E37** | **1,816,274** | **0.020388036 GFLOPs** | Verified ablation — rejected | Replacing total path with separate path-A/path-B did not help: -0.102 pp XSUB / -0.028 pp XSET vs Jitter+Uniform, with slightly more compute. Preserve explicit total-path magnitude. |
| TotalPath + PathAsym + Jitter + Uniform | 16 | pending | pending | **1,816,274 expected** | XLA audit pending | Pending | Keeps the successful total-path statistic and adds normalized early-vs-late path asymmetry. This is the last cheap representation ablation before prioritizing T64→T16 KD. |

### Exact comparison of the recent progression

```text
MotionLite                 XSUB 70.634
MotionPreserve             XSUB 73.976   XSET 74.521
Phase-T16                  XSUB 74.037   XSET 75.170
Jitter + Uniform           XSUB 74.607   XSET 75.338
Jitter + Consistency       XSUB 74.848   XSET 75.572   <-- current champion
PhasePath + Jitter         XSUB 74.505   XSET 75.310   <-- rejected ablation
PathAsym + Jitter          pending
```

## Current champion audit

Checkpoint family: **M4PhaseJitterConsistencyT16**.

- XSUB canonical EMA accuracy: **74.852%**; jittered: **74.670%**; canonical↔jitter agreement: **89.650%**.
- XSET canonical EMA accuracy: **75.575%**; jittered: **75.243%**; canonical↔jitter agreement: **90.221%**.
- Consistency improved final validation accuracy but did **not** materially increase validation-time canonical↔jitter agreement. Its benefit is therefore best interpreted as useful regularization rather than learned full boundary invariance.
- Router-off collapse remains large: **-22.805 pp XSUB** and **-16.781 pp XSET** relative to canonical full accuracy. Keep the post-frame-memory router.
- Frame memory remains the strongest early stage by frozen linear probe: **+12.975 pp XSUB**, **+12.600 pp XSET** from spatial→frame-memory.
- Router immediate linear-probe gain remains small (**+1.375 pp XSUB**, **+1.312 pp XSET**), but router→chunk gain is large (**+10.913 pp XSUB**, **+8.188 pp XSET**). The router is useful through downstream temporal processing, not primarily through immediate linear separability.
- Removing total path is highly damaging: **-23.119 pp XSUB**, **-24.828 pp XSET**.
- Removing full displacement is also strongly damaging: **-18.040 pp XSUB**, **-14.708 pp XSET**.
- Removing both phase-displacement channels costs **-7.347 pp XSUB**, **-4.566 pp XSET** on the trained checkpoint. These are ablation sensitivities, not independent causal gains.
- Persistent hard/fine-grained classes include the 70–73 group and recurring classes such as 10/11, 28/29, 81/83 and related confusions.

## Active M4 experiment branches — verified 2026-08-31

| Branch | Head before ledger sync | Role / status |
|---|---|---|
| `experiment/m4-motionlite-t16-tpu` | `4c5e5d3556be6f266766f230e82bd52d28de25aa` | MotionLite baseline + representation audit |
| `experiment/m4-motionpreserve-t16-tpu` | `9f7871c0e7276d98ffebfa8cce08c105311cee70` | MotionPreserve + weight/activation audits |
| `experiment/m4-motionpreserve-phase-t16-tpu` | `c2455264baa7b091362717fc7f683b86f86b317b` | Phase-T16 + activation audit |
| `experiment/m4-phase-jitter-uniform-t16-tpu` | `4ac95090efac24f77afb4b4a73cc9d9ad67879b8` | Jitter + fixed uniform fusion + robustness audit |
| `experiment/m4-phase-jitter-consistency-t16-tpu` | `5bba3a6510d3b2e243b08e45a48467166bcb0a6f` | Current champion + consistency audit |
| `experiment/m4-phasepath-jitter-uniform-t16-tpu` | `03556dac0a4fd37c30be7f18a4819906b51c82a3` | Completed negative PhasePath ablation |
| `experiment/m4-pathasym-jitter-uniform-t16-tpu` | `c247993ee2450f39ee8872a5274e009bd3cd9aa7` | Pending TotalPath + PathAsym controlled ablation |

The SHAs above are the experiment heads **before this ledger file is synchronized onto the branches**. The ledger-sync commits will move each branch head without changing experiment code.

## Historical references retained

These remain useful context but are separate from the current audited ~20-MFLOP T16 line:

| Variant | XSUB | XSET | Status |
|---|---:|---:|---|
| HOPE v4.1 ShortL3Fix | **73.24771%** | — | Verified/consolidated historical efficient reference; ~2.034M params, ~0.06724 GFLOPs |
| M4G-H4 | **~76.325%** | — | Historical/re-audit required |
| M4G-H4 + SASM + L3Fix | **~76.44%** | **~78.54%** | Historical/re-audit required |
| `main` standardized `nestsar_4l` legacy reproduction target | **63.259294%** | **61.216941%** | Legacy reproduction target, not the current M4-derived T16 ceiling |

## Current research decision

1. Preserve **Phase + Jitter + Consistency** as the current T16 champion checkpoint family.
2. Do not increase consistency strength by default; training agreement approached 100%, while validation canonical↔jitter agreement remained ~90%.
3. Do not return to raw PhasePath; the controlled experiment was slightly worse while increasing compute.
4. Complete the cheap **TotalPath + PathAsym** controlled ablation.
5. If PathAsym does not produce a meaningful gain (roughly >=0.2–0.3 pp), stop adding handcrafted temporal channels and prioritize **T64→T16 knowledge distillation**.
6. Keep frame memory, post-frame cross-stream router, chunk memory and fixed uniform fusion unless a new controlled counterfactual demonstrates otherwise.
7. Do not claim preprocessing-free end-to-end 20 MFLOPs: the XLA numbers are **neural forward compute only**. MotionPreserve/Phase/Jitter preprocessing examines the full raw sequence before producing the 16 neural temporal tokens.

## Publication hygiene

For every paper-facing score archive: exact git commit, protocol, seed, frames/tokens, raw-sequence preprocessing disclosure, checkpoint path/SHA, best epoch, parameters, XLA neural GFLOPs, validation accuracy, confusion matrix/per-class metrics, and whether the number is a controlled training result or an inference-time counterfactual/ablation. Do not convert checkpoint ablation drops directly into independent causal gains.

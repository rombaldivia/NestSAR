# NestSAR Experiment Status — 2026-08-28

This file is the cross-branch experiment ledger for NestSAR. It separates **verified/consolidated**, **partial/interrupted**, and **historical/re-audit-required** results so intermediate experiments are not mistaken for paper-ready claims.

## Result status legend

- **Verified/consolidated**: architecture/compute/result has a stable project record and may be used for internal comparison.
- **Partial/interrupted**: result is from an incomplete run or an early epoch; do not cite as a final score.
- **Historical/re-audit required**: score exists in project history but should be re-audited from the exact checkpoint/config before publication.

## Accuracy and compute ledger

| Variant | Frames | XSUB | XSET | Params / compute | Status | Main conclusion |
|---|---:|---:|---:|---|---|---|
| HOPE v4.1 ShortL3Fix | 16 | **73.24771%** | — | **2,033,988 params**, **~0.06724 GFLOPs** | Verified/consolidated | Strong efficient baseline. More useful as an accuracy/compute reference than the experimental residual branches. |
| Attention-Lite canonical D128 | 16 | observed loaded baseline **70.44%** in v3.5 pretrain check | observed loaded baseline **70.66%** in v3.5 pretrain check | **2,381,028 params**, **705 leaves**, **0.060416900 GFLOPs** | Compute verified; accuracy checkpoint-dependent | Best canonical compute center. Preserve this path when evaluating additions. Historical stronger Attention-Lite checkpoints must be re-consolidated before a paper claim. |
| M4G-H4 | larger-frame historical line | **~76.325%** | — | re-audit exact variant | Historical/re-audit required | One of the strongest raw-accuracy NestSAR lines. Its four specialized streams remain the architectural teacher for new designs. |
| M4G-H4 + SASM + L3Fix | larger-frame historical line | **~76.44%** | **~78.54%** | re-audit exact variant | Historical/re-audit required | Historical accuracy leader. Exact checkpoint/config/compute must be re-audited before publication. |
| PartTrace v2 tiny | 16 | **~63.76% best** | **~65.14% partial** | **110,306 params**, **0.001846238 GFLOPs** | Partial / edge-Pareto | Excellent tiny branch, but removing too much Attention-Lite/HOPE capacity makes it unsuitable as the main accuracy path. |
| PartTrace v3.2 Dense | 16 | **59.90% @ E4** | **62.65% @ E5** | candidate total ~2.56M; exact audit required | Partial | Dense side branch learns, but did not show a path to the 73–76% accuracy line in early runs. |
| PartTrace v3.2 Dense | 64 | **35.76% @ E2** | **41.91% @ E2** | historical estimate ~0.417827 GFLOPs total | Partial | Naive T64 scaling was inefficient: many more tokens were collapsed into the same small final representation. More frames alone were not the solution. |
| TokenPreserve v3.3 | 16 | **54.76 E2 → 62.29 E3 → 65.62 E4** | **54.80 E2 → 61.03 E3 → 64.10 E4** | audit required | Partial | Faster early trajectory than v3.2. Preserving tokens helped, but the branch still lacked enough interaction/reasoning and no final consolidated gain was established. |
| TokenPreserve v3.4 | 16 | no consolidated final | no consolidated final | audit required | Experimental | Fixed checkpoint loading, branch training, masks and query diversity. Main lesson: preserving architecture is not equivalent to preserving learned weights. |
| Cross-Stream Memory v3.5 scratch | 16 | **63.83% best @ E6** | **64.04% best @ E7** | audit output should be archived from run | **Interrupted** | Base learned strongly, but memory correction gate saturated near its maximum while the memory head was weaker. Gate/control design was the main failure signal. |
| Cross-Stream Memory v3.6 scratch | 64 full input / 16 memory anchors | **61.97% best @ E4**; E5 train partial **72.36%**, MEM **62.78%** | **60.95% best @ E4**; E5 train partial **69.81%**, MEM **61.50%** | 640 side-memory tokens; exact XLA audit pending archival | **Ongoing / partial** | T64 base recovered rapidly after slow E1. E1–E5 run with branch correction off, so E6–E10 are the decisive test of uncertainty-gated memory. |

## Detailed observations from v3.5 and v3.6

### v3.5 T16 scratch

The interrupted run reached approximately:

- XSUB: best validation **63.83% @ E6**; partial E7 train **71.43%**; memory train **60.05%**; query overlap **0.249**; effective gate **~0.197**.
- XSET: best validation **64.04% @ E7**; partial E8 train **75.44%**; memory train **62.99%**; query overlap **0.237**; effective gate **~0.196**.

The maximum class correction gate was 0.20, so the correction mechanism was almost fully saturated even though the memory classifier was weaker than the main model. **Conclusion: v3.5 was learning, but the residual control mechanism was too aggressive.**

### v3.6 T64 scratch — current run

Current completed validation and partial E5 training:

- XSUB: E1 validation **15.71%** → best **61.97% @ E4**; E5 partial train **72.36%**, memory train **62.78%**, Q-overlap **0.277**.
- XSET: E1 validation **20.54%** → best **60.95% @ E4**; E5 partial train **69.81%**, memory train **61.50%**, Q-overlap **0.276**.
- During E1–E5: `branch_scale=0`, `effective gate=0`, `base_grad_scale=1`.

Therefore E1–E5 are primarily a clean test of the T64 Attention-Lite base while the side-memory classifier learns through auxiliary supervision. The architecture becomes a real cross-stream correction experiment only when the branch begins ramping from E6.

## Architecture conclusions

1. **More frames alone are not a reliable accuracy mechanism.** The old v3.2 T64 run underperformed badly because token count increased without increasing useful reasoning capacity.
2. **The four specialized streams are still the strongest design signal.** Joint, bone, joint-motion and bone-motion specialization appears more valuable than simply increasing token count or frame count.
3. **Late residual correction can hurt when its gate is not selective.** v3.5 showed a nearly saturated gate (~0.196/0.20) while its memory head remained weaker. v3.6 therefore delays the correction and caps it through uncertainty-aware gating.
4. **Token preservation is useful but not sufficient.** v3.3/v3.4 improved early learning/representation retention but did not prove a final jump over the strong base.
5. **The next high-value mechanism remains mid-level cross-stream interaction.** The target is to let J/B/JM/BM exchange evidence before late fusion while keeping each stream specialized.
6. **Protect the low-compute center.** Canonical Attention-Lite is ~0.0604 GFLOPs. New accuracy modules should be justified by measurable gains and exact XLA audits rather than width/frame increases by default.
7. **Historical accuracy target remains ~76%+.** HOPE v4.1 provides a verified efficient ~73.25% XSUB reference; the M4G-H4 family remains the historical raw-score target, subject to exact re-audit.

## Current decision rule for v3.6

Do not draw a final conclusion from E1–E5. The useful checkpoint is after the branch ramp:

- E6: branch scale 0.20
- E7: 0.40
- E8: 0.60
- E9: 0.80
- E10+: 1.00

At E10, compare validation against the E4/E5 base-only trajectory. If validation rises materially when the memory correction activates, keep the mechanism. If validation falls while memory/gate influence rises, remove or redesign the residual correction and move interaction earlier into the four-stream backbone.

## Publication hygiene

Only use final, re-audited runs in manuscript tables. For every paper-facing result archive: exact git commit, protocol, seed(s), frames, checkpoint SHA/path, parameters, leaves, XLA GFLOPs, best epoch, final/best validation accuracy, confusion matrix and per-class metrics. Partial values in this ledger are research diagnostics, not final claims.

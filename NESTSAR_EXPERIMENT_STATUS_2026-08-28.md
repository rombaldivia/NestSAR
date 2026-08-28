# NestSAR Experiment Status — 2026-08-28

Cross-branch research ledger. **Partial/interrupted results are diagnostics, not paper claims. Historical values marked re-audit must be checked from the exact checkpoint/config before publication.**

| Variant | Frames | XSUB | XSET | Compute / status | Conclusion |
|---|---:|---:|---:|---|---|
| HOPE v4.1 ShortL3Fix | 16 | **73.24771%** | — | **2,033,988 params**, **~0.06724 GFLOPs**, verified | Strong efficient baseline. |
| Attention-Lite canonical D128 | 16 | observed loaded baseline **70.44%** | observed loaded baseline **70.66%** | **2,381,028 params**, **705 leaves**, **0.060416900 GFLOPs**; accuracy checkpoint-dependent | Canonical low-compute center. |
| M4G-H4 | historical larger-frame line | **~76.325%** | — | historical, re-audit | Four-stream specialization remains the architectural teacher. |
| M4G-H4 + SASM + L3Fix | historical larger-frame line | **~76.44%** | **~78.54%** | historical, re-audit | Historical accuracy leader. |
| PartTrace v2 tiny | 16 | **~63.76% best** | **~65.14% partial** | **110,306 params**, **0.001846238 GFLOPs** | Edge-Pareto, not main accuracy line. |
| PartTrace v3.2 Dense | 16 | **59.90% @ E4** | **62.65% @ E5** | partial | Early trajectory below strong baseline. |
| PartTrace v3.2 Dense | 64 | **35.76% @ E2** | **41.91% @ E2** | partial; ~0.417827 GFLOPs historical estimate | Naive T64 scaling inefficient. |
| TokenPreserve v3.3 | 16 | **54.76 E2 → 62.29 E3 → 65.62 E4** | **54.80 E2 → 61.03 E3 → 64.10 E4** | partial | Preservation helped early learning, not enough for a proven final jump. |
| TokenPreserve v3.4 | 16 | no consolidated final | no consolidated final | experimental | Fixed checkpoint loading/training/masks/query diversity. |
| Cross-Stream Memory v3.5 scratch | 16 | **63.83% best @ E6** | **64.04% best @ E7** | interrupted | Gate saturated near max while memory head was weaker. |
| Cross-Stream Memory v3.6 scratch | 64 full / 16 memory anchors | **61.97% best @ E4**; E5 train partial **72.36%**, MEM **62.78%** | **60.95% best @ E4**; E5 train partial **69.81%**, MEM **61.50%** | ongoing, 640 memory tokens | E1–E5 branch correction off; E6–E10 decisive. |

## Current conclusions

1. More frames alone are not a reliable accuracy mechanism; representation quality and interaction matter more.
2. J/B/JM/BM specialized streams remain the strongest design signal.
3. v3.5 exposed a concrete failure: gate ~0.196–0.197 with max 0.20 while memory accuracy remained weaker.
4. Token preservation is useful but insufficient without stronger reasoning/interaction.
5. Highest-value next mechanism remains **mid-level cross-stream interaction before late fusion** while preserving stream specialization.
6. Protect the low-compute center: canonical Attention-Lite is **0.060416900 GFLOPs**.
7. Internal target remains **>76%**. HOPE v4.1 is the verified efficient ~73.25% XSUB reference; M4G-H4 is the historical raw-score target pending exact re-audit.

## v3.6 decision point

E1–E5 have `branch_scale=0`, `gate=0`, `base_grad_scale=1`. Ramp: E6=0.20, E7=0.40, E8=0.60, E9=0.80, E10+=1.00. Compare E10 validation against E4/E5. If validation falls as branch influence rises, move interaction earlier into the four-stream backbone.

## Publication hygiene

Archive exact commit, protocol, seeds, frames, checkpoint, params, leaves, XLA GFLOPs, best epoch, validation accuracy, confusion matrix and per-class metrics. Partial values above are research diagnostics only.

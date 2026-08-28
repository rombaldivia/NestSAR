# NestSAR Experiment Status — 2026-08-28

Cross-branch research ledger. Partial/interrupted scores are diagnostics, not paper claims; historical values marked re-audit require exact checkpoint/config verification.

| Variant | Frames | XSUB | XSET | Status / compute | Conclusion |
|---|---:|---:|---:|---|---|
| HOPE v4.1 ShortL3Fix | 16 | **73.24771%** | — | **2,033,988 params**, **~0.06724 GFLOPs**, verified | Strong efficient reference. |
| Attention-Lite D128 | 16 | observed baseline **70.44%** | observed baseline **70.66%** | **2,381,028 params**, **705 leaves**, **0.060416900 GFLOPs** | Canonical low-compute center. |
| M4G-H4 | historical | **~76.325%** | — | re-audit | Four specialist streams remain the architectural teacher. |
| M4G-H4 + SASM + L3Fix | historical | **~76.44%** | **~78.54%** | re-audit | Historical accuracy leader. |
| PartTrace v2 tiny | 16 | **~63.76%** | **~65.14% partial** | 110,306 params; 0.001846238 GFLOPs | Edge-Pareto, not accuracy line. |
| v3.2 Dense T16 | 16 | **59.90%@E4** | **62.65%@E5** | partial | Early trajectory below strong baseline. |
| v3.2 Dense T64 | 64 | **35.76%@E2** | **41.91%@E2** | partial; ~0.417827G historical estimate | More frames without better reasoning were inefficient. |
| v3.3 TokenPreserve | 16 | **65.62%@E4** | **64.10%@E4** | partial | Preservation helped early learning but no final jump was proven. |
| v3.5 Cross-Stream scratch | 16 | **63.83%@E6** | **64.04%@E7** | interrupted | Gate saturated (~0.196–0.197 / max 0.20) while memory head was weaker. |
| v3.6 Cross-Stream scratch | 64 full / 16 memory anchors | **61.97%@E4**; E5 train 72.36%, MEM 62.78% partial | **60.95%@E4**; E5 train 69.81%, MEM 61.50% partial | ongoing; 640 memory tokens | E1–E5 correction off; E6–E10 decide whether uncertainty-gated memory helps. |

## Conclusions

1. More frames alone are not a reliable accuracy mechanism.
2. J/B/JM/BM specialization remains the strongest design signal.
3. Late correction must be selective; v3.5 gate saturation is a concrete failure mode.
4. Token preservation is useful but insufficient without stronger interaction/reasoning.
5. Highest-value next mechanism: **mid-level cross-stream interaction before late fusion**, preserving four-stream specialization.
6. Protect the **0.060416900 GFLOPs** Attention-Lite center; additions need exact XLA-audited gains.
7. Internal accuracy target remains **>76%**. HOPE v4.1 is the verified efficient ~73.25% XSUB reference; M4G-H4 is the historical target pending re-audit.

## v3.6 decision rule

E1–E5: branch=0, gate=0, base gradients=1. Ramp E6=0.20, E7=0.40, E8=0.60, E9=0.80, E10+=1.00. Compare E10 validation against E4/E5. If validation drops as branch influence rises, move interaction earlier into the four-stream backbone.

## Publication hygiene

For paper-facing results archive exact git commit, protocol, seeds, frames, checkpoint, params/leaves, XLA GFLOPs, best epoch, validation accuracy, confusion matrix and per-class metrics.

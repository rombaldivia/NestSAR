# NestSAR Experiment Status — 2026-08-28

Cross-branch research ledger. **Partial/interrupted results are diagnostics, not paper claims. Historical values marked re-audit must be checked from the exact checkpoint/config before publication.**

| Variant | Frames | XSUB | XSET | Compute / status | Conclusion |
|---|---:|---:|---:|---|---|
| HOPE v4.1 ShortL3Fix | 16 | **73.24771%** | — | **2,033,988 params**, **~0.06724 GFLOPs**, verified | Strong efficient baseline. |
| Attention-Lite canonical D128 | 16 | observed loaded baseline **70.44%** | observed loaded baseline **70.66%** | **2,381,028 params**, **705 leaves**, **0.060416900 GFLOPs**; accuracy checkpoint-dependent | Canonical low-compute center; preserve this path when testing additions. |
| M4G-H4 | historical larger-frame line | **~76.325%** | — | historical, re-audit exact variant | One of the strongest raw-score lines; four-stream specialization remains the architectural teacher. |
| M4G-H4 + SASM + L3Fix | historical larger-frame line | **~76.44%** | **~78.54%** | historical, re-audit exact variant | Historical accuracy leader; do not publish until exact re-audit. |
| PartTrace v2 tiny | 16 | **~63.76% best** | **~65.14% partial** | **110,306 params**, **0.001846238 GFLOPs** | Useful edge-Pareto line, not the main accuracy path. |
| PartTrace v3.2 Dense | 16 | **59.90% @ E4** | **62.65% @ E5** | partial | Learns, but early trajectory did not reach the strong 73–76% line. |
| PartTrace v3.2 Dense | 64 | **35.76% @ E2** | **41.91% @ E2** | partial; historical estimate ~0.417827 GFLOPs | Naive T64 scaling increased tokens without enough useful reasoning. |
| TokenPreserve v3.3 | 16 | **54.76 E2 → 62.29 E3 → 65.62 E4** | **54.80 E2 → 61.03 E3 → 64.10 E4** | partial | Faster early trajectory than v3.2; preservation alone did not prove a final jump. |
| TokenPreserve v3.4 | 16 | no consolidated final | no consolidated final | experimental | Fixed checkpoint loading/training/masks/query diversity; architecture preservation != learned-weight preservation. |
| Cross-Stream Memory v3.5 scratch | 16 | **63.83% best @ E6** | **64.04% best @ E7** | interrupted | Main failure signal: effective gate saturated near max while memory classifier was weaker. |
| Cross-Stream Memory v3.6 scratch | 64 full / 16 memory anchors | **61.97% best @ E4**; E5 train partial **72.36%**, MEM **62.78%** | **60.95% best @ E4**; E5 train partial **69.81%**, MEM **61.50%** | **ongoing**, 640 side-memory tokens | T64 base recovered rapidly. E1–E5 have branch correction off; E6–E10 are the decisive uncertainty-gated-memory test. |

## v3.5 diagnostic

Interrupted T16 scratch run: XSUB partial E7 train 71.43%, MEM 60.05%, Q-overlap 0.249, gate ~0.197; XSET partial E8 train 75.44%, MEM 62.99%, Q-overlap 0.237, gate ~0.196. Gate maximum was 0.20. **Conclusion: the model was learning, but the weaker residual branch was allowed almost maximum correction strength.**

## v3.6 current diagnostic

XSUB E1 val 15.71% → 61.97% best @ E4; XSET E1 val 20.54% → 60.95% best @ E4. At E5 partial, XSUB train=72.36%, MEM=62.78%, Qov=0.277; XSET train=69.81%, MEM=61.50%, Qov=0.276. During E1–E5: `branch_scale=0`, `gate=0`, `base_grad_scale=1`. Therefore these epochs mainly test T64 Attention-Lite while the memory head learns through auxiliary supervision.

## Conclusions

1. More frames alone are not a reliable accuracy mechanism; representation quality and interaction matter more.
2. J/B/JM/BM specialist streams remain the strongest design signal.
3. Late correction must be selective; v3.5 gate saturation is a concrete failure mode.
4. Token preservation helps information retention but is not sufficient without stronger reasoning/interaction.
5. The highest-value next mechanism remains **mid-level cross-stream interaction before late fusion**, while keeping the four streams specialized.
6. Protect the low-compute center: canonical Attention-Lite is ~0.0604 GFLOPs; every added module must justify itself with exact XLA-audited accuracy gain.
7. Internal target remains **>76%**; HOPE v4.1 is the verified efficient ~73.25% XSUB reference and the M4G-H4 family is the historical raw-score target pending exact re-audit.

## v3.6 decision point

Branch ramp: E6=0.20, E7=0.40, E8=0.60, E9=0.80, E10+=1.00. Compare E10 validation against the E4/E5 base-only trajectory. If validation rises materially, retain the correction mechanism. If it falls as branch influence rises, move interaction earlier into the four-stream backbone.

## Publication hygiene

Archive exact git commit, protocol, seeds, frames, checkpoint, parameters, leaves, XLA GFLOPs, best epoch, validation accuracy, confusion matrix and per-class metrics for every paper-facing result. Do not cite partial values above as final scores.

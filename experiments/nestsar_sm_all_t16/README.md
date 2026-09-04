# NestSAR-SM-ALL-T16 v1

From-scratch LocalGlobal M4/G4 experiment with low-rank self-modifying fast weights.

## Hard constraints

- Raw NTU clip length is variable.
- Preprocessing always converts the whole raw clip to exactly **16** LocalGlobal motion-preserving temporal tokens.
- Neural processing length is therefore fixed at T=16 and does not follow the original raw frame count.
- No attention, GCN, TCN, Transformer, or T x T operation.
- XLA inference audit is mandatory in the dual-T4 launcher and training aborts if the compiled model exceeds **0.025 GFLOPs/clip**.
- Training starts from random initialization; no champion checkpoint is loaded.

## Self-modification

The proven M4/G4 LocalGlobal topology is retained. A shared controller modulates input channels and stream features. M4 and G4 retain their original BiMemory and add a rank-2 fast-weight residual. For each temporal state:

```text
pred_t = k_t^T S_(t-1)
err_t  = v_t - pred_t
S_t    = alpha_t S_(t-1) + eta_t k_t err_t^T
read_t = q_t^T S_t
```

`S_0` is learned by the outer NTU120 optimization and is reset for every clip. M4 updates at the 16-token level; G4 uses the same rule on four chunk states. Fusion is dynamically modulated from a zero-initialized uniform state and the final adaptive head is rank-2 and evaluated once per clip.

This is a **HOPE-inspired low-rank self-modifying delta-memory adaptation**, deliberately compressed for NestSAR's edge-compute target. It is not claimed to be a verbatim reproduction of the full language-model HOPE stack.

## Kaggle progress

`run_dual_t4.py` owns all notebook tqdm rendering. XSUB and XSET workers write plain structured status lines to log files, so the notebook sees exactly two persistent bars with in-place TRAIN/VAL updates and no interleaved child tqdm output.

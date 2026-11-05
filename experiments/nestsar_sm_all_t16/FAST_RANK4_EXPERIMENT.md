# NestSAR T16 Fast-Memory Rank-4 Experiment

Branch: `experiment/nestsar-fast-rank4-t16-dualt4`

Base: `experiment/nestsar-validation-reframe-t16`

## Controlled change

Only the fast-weight addressing rank is changed:

- baseline: rank = 2, state = [2, 112]
- experiment: rank = 4, state = [4, 112]
- adaptive head rank remains 2
- T = 16
- spatial_dim = 24
- model_dim = 112
- controller_dim = 16
- loss, optimizer, EMA, augmentation, preprocessing, M4/G4 topology, and dual-T4 progress behavior are unchanged

Expected initialized parameters: **1,831,932**.

The existing Person-Aware P2 preprocessing cache is reusable because the input representation is unchanged.

## Hypothesis

The rank-2 fast memory may suffer from addressing interference. Rank 4 doubles the fast-memory addressing/state capacity while changing only a very small fraction of total model compute.

## Kaggle

The existing dual-T4 launcher still owns the notebook progress display. Child workers do not render competing progress bars.

```bash
python -m experiments.nestsar_sm_all_t16.run_dual_t4 \
  --dataset /kaggle/input/models/paolamaydana/ntudanno/other/default/1/ntu120_3danno.pkl \
  --outdir /kaggle/working/NestSAR_SM_ALL_T16_R4_DualT4 \
  --fast-rank 4
```

For the streaming launcher, `fast_rank=4` is now the locked default and the parameter audit expects 1,831,932 parameters.

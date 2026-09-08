# Nonlinear T16 versus learned sequence diagnostic

This is the next **binary action-pair diagnostic**, extending the raw/DCT audit.
It does not train or replace the deployed SM-ALL model. It does not produce a
120-class benchmark accuracy or test a frozen specialist's routing accuracy.

The person-aware P2 v3 cache supplies exact `[16,750]` canonical tokens and raw
skeletons. An existing cache is reused; if absent, the repository's original
builder creates it once in a separate CPU process from the attached NTU pickle.
No JAX, CUDA or NVIDIA packages are installed by this runner.

| Arm | Input | Learned classifier |
| --- | --- | --- |
| `t16_mlp` | Existing corrected `[16,750]` tokens, flattened | Width-256 residual MLP, two residual blocks |
| `sequence16_gru` | 16 uniformly sampled poses, native-frame velocity and validity | Width-128, two-layer bidirectional GRU |
| `sequence64_gru` | 64 uniformly sampled poses, native-frame velocity and validity | Exactly the same GRU architecture and initialization as sequence16 |

Sequence channels are canonical xyz, raw consecutive-frame xyz velocity, pose
validity and velocity validity for each of 25 joints and two people. Canonical
scale is computed from the original clip's valid joints. Interpolation requires
both endpoints to be valid and never bridges a missing joint. The two sequence
resolutions share raw canonicalization/velocity work. Uniform sampling is not
lossless: especially for clips longer than 64 frames, these are diagnostic
representations, not upper bounds on the information in the full skeleton.

The 16/64 GRU comparison controls architecture and parameter count. Comparing the
MLP to a GRU also changes representation and architecture; it alone cannot assign
a score difference solely to temporal compression. Training compute is greater
for the 64-frame diagnostic; it is not an equal-FLOP deployment comparison.

## Evaluation protocol

- Same ten confusing pairs as the previous audit, one binary task per pair.
- Split seeds 128, 42, 28. XSUB groups by subject, XSET by setup.
- Each repeat assigns whole groups to fit / inner selection / final, with target
  group proportions 60% / 20% / 20%. Sample proportions depend on group sizes.
- One common partition serves all pairs, models and trials in that protocol and
  repeat. Splits are redrawn only to meet declared minimum class counts, never
  based on accuracy. Every pair contains both classes in all three partitions.
- All diagnostic examples come from the protocol's original official TRAIN
  partition. A guarded accessor blocks held-out raw/token reads. An internal-fold
  cache view is explicitly rejected to avoid accidentally splitting it again.
- RMS conditioning and class weights are fitted on FIT only. Zeros remain zero.
  Augmentation is disabled for all three diagnostic arms, so their input evidence
  is fixed. The production trainer's fresh augmentation is unaffected.
- Two predefined regularization/LR trials per arm; maximum 60 epochs, warm-up 5,
  patience 5 after warm-up. Same fit examples, epoch ordering, batch size,
  optimizer, smoothing and maximum epoch budget. Stop/choose checkpoints using
  inner balanced accuracy; ties between trials use inner NLL.
- The winning trial is frozen before FINAL is evaluated once. No refitting on
  selection/final, no threshold search on final and no automatic modality-limit
  claims. On resume, completed final predictions are read rather than reevaluated.
- Paired subject/setup cluster-bootstrap intervals are saved per pair/repeat for
  Seq64−T16 and Seq64−Seq16. These are conditional descriptive intervals, without
  multiple-testing correction. Repeated holdouts can overlap. The reported SD
  across repeats is descriptive, not an independent-repeat confidence interval.

Protocol summaries are an unweighted macro of the selected binary pairs. Pairs
can overlap in classes/samples. Never compare these numbers directly with the
76.97% / 78.42% 120-class scores, or with a different internal fold's accuracy.

## Kaggle

Select GPU T4 x2. Run `kaggle_nonlinear_compare.py` via `runpy.run_path` inside
the notebook kernel and pass `NESTSAR_COMPARE_SETTINGS` with `cache_dir`,
`outdir`, optional `dataset`, optional `config`, and optional `smoke_test`.

The default cache is `/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3`.
With `cache_dir=None`, one compatible input cache can be discovered automatically.
If no cache exists, `dataset=None` requires exactly one `ntu120_3danno.pkl` under
`/kaggle/input`; otherwise set its path explicitly. Cache construction displays
progress on both existing bars and requires one full pickle in host RAM plus
about 7.81 GiB for the raw/canonical cache and a 1 GiB output reserve on disk.
It runs before either training worker starts. This stateless per-clip conversion
builds the original dataset cache; all subsequent learning, model selection and
diagnostic evaluation read only the original official TRAIN examples.
The default output directory is
`/kaggle/working/NestSAR_P2_Nonlinear_Grouped_v1`.

XSUB runs on GPU0 and XSET on GPU1. Both progress bars are owned by the notebook
and update in place. `BEST_INNER` is the current trial's best internal selection
score and epoch, never an official benchmark score. The final table reports each
model's mean and SD across grouped repeats for both protocols.

Only one pair's dense feature arrays are resident per worker. They are built
once, then reused across splits, epochs and hyperparameter trials. The original
raw/T16 files remain shared memory maps. Fit normalization uses bounded chunks.
There is no second full-dataset feature cache. JIT functions are reused across
pairs; timings separate initial compilation/warm-up and synchronized GPU steps.

The active trial is checkpointed every completed epoch with optimizer, RNG and
best weights. Completed trials/pairs resume safely; changed code/config/cache
requires a new output directory. Temporary model weights are removed after the
selected final result is committed, keeping disk use bounded. Set
`keep_checkpoints=True` to keep the selected model and its normalization per arm,
pair and repeat; this increases disk use substantially.

Outputs: `results.json`, `pair_comparisons.csv`, per-protocol `summary.json`,
all split IDs/group IDs, trial histories, selected hyperparameters/epochs,
final sample-level probabilities, and paired confidence intervals.

## Verification

```bash
python -m pytest -q experiments/nestsar_sm_all_t16/nonlinear_compare/tests
python -m experiments.nestsar_sm_all_t16.nonlinear_compare.smoke_cpu --outdir /tmp/p2_compare_smoke
```

The CPU smoke test runs two concurrent workers, all three models and both inner
trials, checks completed-run resume, and poisons official held-out feature/raw
arrays with NaNs to catch unauthorized indexing. It does not measure NTU
accuracy or certify dual-T4 runtime performance.

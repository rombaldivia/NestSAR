# R4 frozen readout diagnostic

This is a small training intervention on cached features. It does not retrain
NestSAR. It cannot prove a single architectural bottleneck by itself.

Use the existing R4 EMA checkpoints, fixed T16 preprocessing, and P2 v3 cache:

    /kaggle/working/NestSAR_SM_ALL_T16_P2_R4/xsub/best.msgpack
    /kaggle/working/NestSAR_SM_ALL_T16_P2_R4/xset/best.msgpack
    /kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3

The worker requires Fast-R4/Head-R2 and 1,831,932 backbone parameters. It verifies
official split disjointness, counts, sample labels, preprocessing version, and
reconstruction of the original logits. Inputs open read-only. Full clean-baseline
reproduction is checked before probe fitting.

## Controlled comparisons

| Variant | Trainable change | Inputs |
| --- | --- | --- |
| A_control | Refit the existing head, initialized from the checkpoint | All four 112-D descriptors, four fusion weights, two dynamic-head coefficients |
| B_descriptor | Same head plus a nonlinear logit residual | Exactly the same information as A |
| C_temporal | Same head plus a nonlinear logit residual | B inputs plus three ordered zero-mean contrasts of pre-G4 and post-G4 four-chunk sequences |

The original head remains present in every variant. Residual output projections
start at zero, so every variant begins with the checkpoint's predictions.

B uses a hidden width of 224; C uses 39. Both are one-hidden-layer GELU residuals.
Their total trainable parameter budgets differ by less than 2%. Input dimension
and width necessarily differ; a failed C probe does not establish an information
ceiling. C receives fixed DCT-II contrasts, not extra temporal averages or raw
skeleton inputs. These contrasts retain order among the four chunks, and are
orthogonal to their constant temporal component. C does not test every possible
readout of all 16 tokens.

Defaults: 20 epochs per probe, batch 512, AdamW 3e-4, weight decay 1e-4, smoothing
0.05, residual dropout 0.10, one-epoch warm-up, cosine decay, norm clipping 1.0.
Probe seeds: 128, 28, 42. These are three readout initializations on the SAME
backbone checkpoint, not independent backbone training runs.

The features are extracted once per protocol in float32. This needs about 1.39 GiB
per protocol (about 2.77 GiB total), plus small probe checkpoints and predictions.
Runtime is not estimated: it depends on feature extraction and the Kaggle session.
Only existing numpy/JAX/Flax/Optax/tqdm packages are used; the launcher installs nothing.

## Selection and evaluation

Probe fitting uses approximately 80% of the official training examples, grouped
by subject for XSUB and setup for XSET. Development groups are sampled only from
the remaining official training groups. Every class must occur in both parts.
Normalization statistics use fitting examples only.

Every probe follows its full fixed training budget. Minimum development NLL
selects its checkpoint, with epoch zero eligible. There is no early stopping.
After fitting and selection have finished for all variants/seeds in a protocol,
the official evaluation labels are used for final scoring. A separate baseline
reproduction check before training verifies that the correct original model is loaded.

IMPORTANT: the backbone already trained on these probe-development examples,
and its original checkpoint was selected on official evaluation. This is
exploratory evidence, not a fresh held-out benchmark. A successful intervention
needs a baseline/candidate comparison with an internal development split fixed
before backbone training, followed by independent backbone seeds.

## Output and interpretation

Combined output:

    /kaggle/working/NestSAR_R4_FROZEN_READOUT_v1/summary.json

Per protocol/seed the output contains:

- Selected epoch, fitting/development/evaluation accuracy and NLL.
- A/B/C gains against the original checkpoint, B against A, and C against B.
- Corrected and newly broken predictions.
- Subject/setup cluster-bootstrap intervals (exploratory, no multiplicity correction).
- Prediction arrays, histories, fitted readouts, and normalization parameters.

B beating both A and the original checkpoint suggests nonlinear readout
headroom. C beating both B and the original checkpoint suggests useful temporal
information survives upstream of the descriptor and can improve this readout.
Neither result alone identifies a unique cause. Negative results are inconclusive.
The program never labels the model's architectural ceiling as proven.

## Running and resuming

The provided Kaggle notebook has one launch cell. Use Internet on and GPU T4 x2.
It checks out a pinned audit revision in a dedicated directory and launches
XSUB on GPU0 and XSET on GPU1. The notebook itself does not import JAX. Workers
run in fresh processes to avoid stale notebook imports and wrong GPU assignment.

The progress display reuses the existing persistent notebook bars, with tqdm
for terminal execution. Notebook display requires no ipywidgets installation.
Worker failures stop the other worker and print the failing log tail. Interrupting
the cell terminates its worker process groups.

Rerun the same configuration to resume feature extraction, resume an interrupted
probe at its last completed epoch, or reuse a completed probe. A run identity
guards checkpoint hashes, cache identity, code, and settings. If any setting
changes, choose a NEW output directory instead of mixing experiments.

## Validation

Seven CPU tests passed with JAX 0.7.2, Flax 0.11.2 and Optax 0.2.5:
real R4 model/head numerical equivalence and zero residual equivalence,
temporal contrast properties, training-only grouped splits, fitting-only
normalization and padding, parameter budgets and paired metrics, training all
three variants with exact interrupted-epoch resume, and cache-identity rejection.

The full NTU120/T4 experiment has not been run by the authoring environment:
the actual user checkpoints and canonical cache are available in Kaggle.
Run the provided notebook cell there and inspect the combined summary.

Local CPU test command, with the repository and dependencies on PYTHONPATH:

    JAX_PLATFORMS=cpu python -m unittest experiments.nestsar_sm_all_t16.test_frozen_readout -v

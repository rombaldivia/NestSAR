# NestSAR-SM-ALL-T16 v1

From-scratch LocalGlobal M4/G4 experiment with low-rank self-modifying fast weights.

## Shared-cache runner update

Use `kaggle_cell.py` inside the notebook kernel with `runpy.run_path(...)`, or
`python -m experiments.nestsar_sm_all_t16.run_dual_t4_corrected` from a terminal.
The current runner keeps **SM-ALL**, 1,826,556 parameters, and the **16 × 750**
input. It imports the existing `model.py` without modifying its equations.

- One preparation process loads the NTU pickle, validates both splits, writes a
  shared float32 raw cache and canonical-token cache, then exits before training.
- XSUB/GPU0 and XSET/GPU1 open the same files as read-only memory maps. Workers
  do not load a raw pickle or materialize complete protocol feature arrays.
- Canonical tokens are computed once per sample when building the cache.
  Validation only reads cached tokens. Training builds the augmented view once
  per sample/epoch, retaining the previous protocol/epoch/sample seeds.
- One producer thread per worker prepares at most two queued batches. The active
  batch and queued batches are the only dense input buffers held by that worker.
- Two parent-owned TQDM widget bars update in place. Each keeps
  `BEST=score%@Eepoch` first, including during validation and completed-run resume.
  Best scores become available only after complete EMA validation.
- Default training is 60 epochs, effective batch 256 (64 × 4 accumulation),
  EMA 0.995, and **patience 5 after the 8% learning-rate warm-up**. Padded tails
  are masked in all losses/metrics; every training and validation sample counts.
- Setup creates a pip-free venv when necessary and installs through the notebook
  Python's `pip --python`. It never invokes `ensurepip` or changes the kernel's
  JAX installation. JAX 0.7.2, Flax 0.11.2, Optax 0.2.5 and NumPy 2.2.6 are pinned.

The default new output directory is
`/kaggle/working/NestSAR_SM_ALL_T16_SharedCache_v2`; the cache is
`/kaggle/working/NestSAR_SM_ALL_SharedCache_v2`. Previous scores/checkpoints are
separate. Rerunning the same cell resumes the last completed epoch with optimizer,
EMA and RNG state. An atomic best-file reference repairs interrupted public
checkpoint/history writes. Configuration or dataset mismatches are rejected.

Disk tradeoff: canonical tokens occupy 48,000 bytes per sample (about 5.09 GiB
for 113,945 samples), plus the compressed two-person raw cache. Preparation checks
free disk space, allowing another 1 GiB for run outputs. It still needs one whole
raw pickle in RAM during conversion. Worker RSS includes mapped file pages;
PSS, USS, available host RAM and swap are recorded to distinguish shared pages
from private memory and memory pressure.

Outputs include `best_scores.json`, `results.json`, and each protocol's
`best.msgpack`, `best.json`, `last.msgpack`, `history.json`, and `status.json`.
Epoch history separates preprocessing service time, queue wait, host-to-device
transfer, compilation, warm-up, and GPU execution. Timed JAX execution blocks on
the full result, and warm-up outputs are discarded without advancing training.

`audit_checkpoint.py` now uses corrected SM-ALL inputs and separate legacy inputs
for the original Hand-M4/G4 checkpoint. For an older SM checkpoint trained before
the preprocessing correction, explicitly use `--sm-preprocessing legacy`.
FLOP auditing unrolls scans and checks forward equivalence; raw scan undercounts
are not used as a budget gate in the new runner.

Run regression checks with:

```bash
python -m pytest -q experiments/nestsar_sm_all_t16/test_preprocessing_corrected.py experiments/nestsar_sm_all_t16/streaming/tests
```

The optional integration check runs the real SM-ALL model on two concurrent CPU
workers with a tiny synthetic dataset, then checks resume and best-file repair:

```bash
python -m experiments.nestsar_sm_all_t16.streaming.smoke_cpu --outdir /tmp/sm_all_smoke
```

Set `NESTSAR_SETTINGS['smoke_test']=True` for a two-epoch Kaggle test with 256 train
and validation samples per protocol in a separate `_smoke` output directory.
The reusable cache is built for the attached dataset even in smoke mode.
This update has no new NTU accuracy or T4 speed result; the historical results
below belong to the preceding corrected runner. Gradient accumulation and keeping
the final partial training batch mean a new run need not reproduce its weights.

## Verified result — corrected preprocessing v2

Final dual-T4 NTU RGB+D 120 run, seed 128, fixed neural processing length T=16:

| Protocol | Best validation accuracy | Best epoch | Last epoch |
| --- | ---: | ---: | ---: |
| XSUB | **76.321216%** | 24 | 29 |
| XSET | **78.062108%** | 26 | 31 |

Model size: **1,826,556 parameters**.

The run used corrected mask-safe preprocessing, complete raw-frame transition accounting, fresh per-epoch label-preserving augmentation, self-modifying M4/G4, cross-stream routing, adaptive fusion, and the rank-2 dynamic head. The distal specialist and CD-Former knowledge distillation were **not** used in this result.

## Verified compute — scan-corrected static audit

The paper/comparison FLOP number is measured with `compute_unrolled_audit.py`, which replaces only recurrent scan implementations with mathematically identical static unrolled loops before JAX/XLA cost analysis. This is necessary because ordinary cost analysis on `lax.scan`/while graphs can report loop-body cost without charging every recurrent iteration.

Audit environment: JAX 0.7.2, CUDA GPU, batch 1.

| Model | Parameters | FLOPs / clip | MFLOPs / clip | GFLOPs / clip |
| --- | ---: | ---: | ---: | ---: |
| LocalGlobal V2 | 1,816,130 | 28,545,916 | 28.545916 | 0.028545916 |
| Hand-M4/G4 T32 | 1,854,650 | 29,612,176 | 29.612176 | 0.029612176 |
| **NestSAR-SM-ALL-T16** | **1,826,556** | **29,065,216** | **29.065216** | **0.029065216** |

For SM-ALL-T16 this is also **0.014532608 GMACs/clip** under the convention `1 MAC = 2 FLOPs`.

Relative scan-corrected compute:

- SM-ALL vs LocalGlobal V2: **+0.519300 MFLOPs (+1.819%)**.
- SM-ALL vs Hand-M4/G4 T32: **-0.546960 MFLOPs (-1.847%)**.

### Important accounting note

The training result JSON reports `5,155,914 FLOPs = 0.005155914 GFLOPs/clip` from the ordinary compiled XLA graph. **Do not use that number as the paper FLOP count.** It undercounts recurrent scan execution. The verified reporting value for this model is:

**NestSAR-SM-ALL-T16 = 0.029065216 GFLOPs/clip = 29.065216 MFLOPs/clip.**

The archived `run_dual_t4.py` keeps its historical raw-XLA guard. The updated
`run_dual_t4_corrected.py` uses the shared-cache runner and the unrolled audit.

## Hard constraints

- Raw NTU clip length is variable.
- Preprocessing converts the whole raw clip to exactly **16** LocalGlobal motion-preserving temporal tokens.
- Neural processing length is therefore fixed at T=16 and does not follow the original raw frame count.
- No attention, GCN, TCN, Transformer, or T x T operation.
- Training starts from random initialization; no champion checkpoint is loaded.

## Corrected preprocessing v2

The corrected path preserves model input shape `[16, 750]` while fixing three data-path defects:

1. Missing people/joints are identified before centering and remain exactly zero through centering and normalization.
2. Every valid adjacent raw-frame transition is computed before segmentation and assigned exactly once, preserving full displacement, phase-A displacement, phase-B displacement, and path magnitude.
3. The augmented training view is rebuilt from the raw skeleton every epoch using deterministic sample/epoch seeds, with up to ±8 degree yaw rotation and configured temporal-boundary jitter.

These preprocessing changes do not change the neural inference graph or its parameter count.

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

## Kaggle execution

`kaggle_cell.py` calls the shared-cache launcher in the notebook kernel, which
owns both progress bars. GPU0 runs XSUB and GPU1 runs XSET simultaneously after
cache preparation. Configure it through `NESTSAR_SETTINGS` (`dataset`, `outdir`,
`cache_dir`, `config`, `raw_layout`, `audit_first`, and `smoke_test`). The default
dataset lookup requires exactly one `ntu120_3danno.pkl` under `/kaggle/input`.

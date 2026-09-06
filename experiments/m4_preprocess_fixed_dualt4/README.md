# Fixed preprocessing, dual T4, runtime recovery and best scores

This branch contains the complete corrected Hand-M4/G4 package. From a checkout, execute `experiments/m4_preprocess_fixed_dualt4/kaggle_cell.py` inside the Kaggle notebook kernel using `runpy.run_path(...)`. Configure it through the `NESTSAR_SETTINGS` dictionary (see the file). The parent notebook owns both persistent bars; do not launch the entrypoint through a shell pipe if you want notebook widgets.

The code is based on NestSAR commit `1a570d554e9639183ead04d4282f478fee150eab`; this experiment does not modify the old experiment modules.

# NestSAR preprocessing fixes — Kaggle T4 × 2

Attach `ntu120_3danno.pkl` in Kaggle, select **GPU T4 × 2**, enable Internet for dependency setup, and execute the branch entrypoint in the notebook kernel. This directory contains the full `nestsar_fixed` package and tests; it does not import the old experiment harnesses. The pinned checkout cell supplied with this branch calls the entrypoint using `runpy.run_path`.

## What runs

| Setting | Value |
|---|---|
| GPU 0 | NTU120 XSUB training and EMA validation |
| GPU 1 | NTU120 XSET training and EMA validation |
| Model | Existing Hand-M4/G4, 1,854,650 parameters |
| Main input | 16 × 750, unchanged |
| Hand input | 32 × 96, unchanged |
| Epoch limit | 60 |
| Early stopping | Patience 5; failures count only after warm-up |
| Batch | Microbatch 64 × 4 accumulation steps = 256 per protocol |
| Validation batch | 256, padded last batch masked out |
| Seed | 128 for both protocols |
| Schedule | AdamW, peak 6e-4, minimum 2e-5, cosine, 8% warm-up |
| EMA | 0.995 |

The two workers train independent protocol models simultaneously. Gradient accumulation updates AdamW and EMA once per effective batch. Every real training sample, including the final partial batch, contributes to the loss. Canonical and augmented forwards both contribute classification, auxiliary and consistency losses.

This is the corrected Hand-M4/G4 experiment. CD-Former distillation and new architectural branches are subsequent experiments and are not included in this run.

## Corrections

1. Capture joint validity before centering. Keep missing joints and absent people zero. Compute RMS over all xyz coordinates of valid joints, including legitimate zeros. Carry the original mask into motion calculations. Missing primary roots use interpolated valid roots, with a valid-joint centroid fallback if no primary root exists.
2. Compute every adjacent raw-frame difference before segment aggregation. Mask both endpoints, then assign each transition to exactly one segment using its destination frame. This also handles clips shorter than 16 frames without counting repeated poses as repeated movement. Full displacement equals the sum of the two phase displacements; path is the sum of absolute valid differences.
3. Sample fresh mild yaw (±8°) and ±1-frame segmentation jitter every training epoch. Both people receive the same yaw, and both main and hand features are rebuilt from that transformed skeleton. Validation stays deterministic. Augmentation is proposed as label-preserving; its accuracy effect requires measurement.
4. Compute hand velocities from valid adjacent transitions between sampled frames, divided by elapsed raw frames, so missing joints cannot create velocity spikes. A valid joint at the centered origin stays valid.

The original clip-level person-energy ordering is retained. The raw pickle layout defaults to `M,T,V,C`; set `RAW_LAYOUT="TMVC"` only for a dataset explicitly stored in that layout. There is no shape guessing for ambiguous short clips, no silent split-ID dropping, and no automatic label offset conversion.

For an ablation with fixed jitter, set `fresh_augmentation=False` and `rotation_degrees=0.0`, and use a new `OUT_DIR`. To isolate the corrected deterministic preprocessing, also set `jitter_shift=0`. The first frozen-view option retains the two-forward consistency recipe.

## Progress, memory and resume

The notebook owns exactly two persistent TQDM widget bars. Workers write status files and logs; worker output is never streamed through `print()`. Progress therefore updates the existing bars instead of adding a console line for every batch. Terminal use falls back to two positioned TQDM bars.

Each bar keeps **BEST=score%@Eepoch** as its first statistic during training, validation and completion. XSUB and XSET have separate best scores. Before the first completed validation it shows `BEST=--`; partial validation accuracy is never promoted to the best score. Completed-run resume restores both the best score and its epoch. `best_scores.json` records both protocols together throughout the run.

The pickle is loaded once by a separate preparation process, which exits before training. It writes a shared float32 raw-skeleton cache. An entirely absent second person is omitted from disk and restored as zeros when reading a sample. Workers use read-only memory maps with at most two prefetched batches each; canonical/augmented feature arrays for the whole dataset are never duplicated in RAM. The OS may retain shared file pages in memory, so per-worker RSS is not the same as private memory usage.

Cache preparation checks disk space and refuses incomplete or mismatched caches. It records the full source-file SHA-256, split sizes and file sizes. The exact required disk space depends on the attached dataset; the runtime packages also occupy disk. There is one complete raw pickle in RAM during initial conversion. The runner cannot eliminate that peak while reading a monolithic pickle.

The runtime is isolated from the notebook kernel unless an already matching CUDA environment is available. Core versions: JAX 0.7.2, Flax 0.11.2, Optax 0.2.5. Each worker selects its GPU before importing JAX. No pmap replication compatibility shim is needed.

Version 1.0.1 repairs the reported Kaggle `ensurepip` failure: it creates the runtime with `with_pip=False` and uses the notebook Python's `pip --python <runtime-python>` to install dependencies. That documented pip option supports environments with no pip installed. Existing partial runtime directories are completed without clearing them. Checkpoints and the raw cache are preserved. See [pip's alternate-interpreter documentation](https://pip.pypa.io/en/stable/topics/python-option/).

Rerunning the same cell resumes from the last completed epoch with optimizer, RNG, EMA, patience and history state. A partially completed epoch is replayed deterministically. It does not resume old checkpoints trained with defective preprocessing. Keep the same output folder and settings to resume this experiment; choose a new output folder for a changed configuration. Keep the output/checkpoints and cache available when moving between Kaggle sessions.

An exclusive launcher lock prevents duplicate runs in the same output directory. Interrupting the cell terminates both owned worker processes. A failure in either worker terminates its peer and surfaces the log tail. Existing unrelated processes are not killed.

## Files produced

Under `OUT_DIR`:

- `xsub/` and `xset/`: `best_ema.msgpack`, `last.msgpack`, `history.json`, `run_config.json`, `status.json`, `result.json`.
- `results.json`: combined protocol results.
- `best_scores.json`: current best completed-validation score and epoch for each protocol.
- `compute_audit.json`: full inference audit performed on GPU 0 before training.
- `hardware.json`, `gpu0_probe.log`, `gpu1_probe.log`, `preprocessing_tests.log`, `prepare.log`, `xsub.log`, `xset.log`, and runtime-install logs when needed.

Timing fields distinguish compilation, CPU preparation service time, data wait, host-to-device transfer and synchronized model execution. Preparation overlaps GPU work, so service time must not be added to GPU time to infer epoch wall time. The `gpu_*` field names also apply to CPU smoke tests, where `backend` explicitly says CPU. Synchronization follows [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).

## Compute accounting and validation

The packaged model has the same parameter tree, identical initialized parameter values and identical forward outputs as the original Hand-M4/G4 model for the tested input. The fixes add no network layers or network input dimensions.

**The earlier approximately 20.63 MFLOP figure is not a verified full-loop cost.** In the local JAX 0.7.2 CPU audit, the old and packaged scan graphs both report 20,456,644 FLOPs. Expanding every recurrent iteration gives **66,523,220 FLOPs (0.06652322 GFLOPs)** for the same logits, with zero observed output difference. The notebook repeats the full unrolled audit on the T4 and saves that hardware/compiler-specific result. Raw-sequence preprocessing is excluded and scales with raw clip length.

Local validation:

- 42 passing regression tests for masks, short clips, transition conservation, augmentation replay, gradient accumulation, masked tails, splits, caches, patience, checkpoint serialization, pip-free runtime setup and best-score reporting.
- A real offline wheel installation into a pip-free environment passed while `ensurepip` was explicitly blocked, including recovery of a partially initialized runtime.
- The real 1,854,650-parameter model completed two synthetic epochs of both XSUB and XSET workflows on CPU, with validation, saving and completed-run resume.
- Resuming recovers full two-epoch history from the checkpoint.
- Original-versus-packaged parameter and forward-output equality passed.
- Static-unrolled versus scanned forward-output equality passed.

The complete Kaggle notebook has not been run on two T4s here, and the real NTU120 dataset was not available in this workspace. Set `NESTSAR_SETTINGS["smoke_test"]=True` for a short test on both Kaggle GPUs (256 training and 256 validation samples per protocol, two epochs, a separate output folder). Then set `smoke_test` to `False` for full training. Synthetic validation is a functionality check and provides no evidence of an accuracy gain.

## Source

Model equations were copied from [NestSAR commit 1a570d5](https://github.com/rombaldivia/NestSAR/tree/1a570d554e9639183ead04d4282f478fee150eab), specifically `experiments/m4_phase_jitter_consistency_localglobal_hand_m4g4_t32/model.py` and the memory/spatial primitives in `experiments/m4_motionpreserve_t16/train_m4_motionpreserve_t16_tpu.py`. Imports were detached to make this package self-contained. `AUDIT_UNROLL` only expands the same recurrence during the separate compute-audit process; training retains scans.

Runtime and display references: [JAX installation](https://docs.jax.dev/en/latest/installation.html), [TQDM notebook](https://tqdm.github.io/docs/notebook/).

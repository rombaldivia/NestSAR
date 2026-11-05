# NestSAR Full-Parallel T16

Branch: `experiment/nestsar-full-parallel-t16`

Current model identity: `NestSAR-FULL-PARALLEL-T16-v2`.

## Stability and integration fixes

- Keys and queries clamp the squared norm before the square root. This preserves
  forward normalization and prevents undefined gradients at zero. The fix applies
  to the parallel model, R4, and the unrolled FLOP-audit reference.
- The worker receives the model explicitly. Importing the parallel worker no
  longer replaces the R4 worker's functions globally.
- Best checkpoints, metadata, and results all identify the parallel model.
  Resume signatures include its identity and a source hash. R4/v1 output
  directories are rejected before their progress files are changed.
- The launcher defaults to a fresh `NestSAR_FULL_PARALLEL_T16_v2` output directory.
  The preprocessing cache can be reused; no packages are installed by the launcher.
- `NestSARParallelT16(parallel=False)` evaluates the new equations serially with
  exactly the same weights and shapes. This is a benchmark control, not R4.

The model still has **1,831,932 parameters**. No attention was added.
Existing R4 trained weights are not a checkpoint for the new affine architecture.

## Purpose

Remove the hidden token-serial execution paths in R4 while preserving the R4
hierarchy and ~1.83M parameter budget.

## Changes

### 1. GatedSweep -> ParallelAffineSweep

The historical GRU-like recurrence depends nonlinearly on `h[t-1]` and cannot
be converted exactly to an associative scan.

The replacement uses

```
h[t] = a[t] * h[t-1] + b[t]
```

with `a[t]` and `b[t]` computed only from `x[t]`.  Affine maps compose
associatively, allowing `jax.lax.associative_scan`.

The per-direction parameter count is deliberately matched to historical
GatedSweep:

```
6 * D^2 + 3 * D
```

### 2. FastWeightDeltaResidual -> exact associative form

The old update is

```
pred = k^T M
err  = v - pred
M'   = alpha M + eta k err^T
```

which can be rearranged exactly as

```
M' = A M + B

A = alpha I - eta k k^T
B = eta k v^T
```

and affine matrix maps compose associatively:

```
(A2,B2) o (A1,B1) = (A2 A1, A2 B1 + B2)
```

Therefore the fast-weight memory no longer requires a serial `lax.scan`.

### 3. Stream vectorization

- Joint/Bone spatial encoders are evaluated as one lifted-vmap pair.
- JointMotion/BoneMotion spatial encoders are evaluated as one lifted-vmap pair.
- All four M4 memories are one lifted-vmap operation.
- All four G4 descriptor memories are one lifted-vmap operation.
- The four classifiers use one grouped einsum.

Parameters remain independent across streams.

## Intentionally unchanged

- T16 preprocessing
- J/B/JM/BM semantics
- person-aware controller
- cross-stream router
- M4 -> Router -> G4 hierarchy
- rank-4 fast memory
- adaptive fusion/head
- training objective
- augmentation
- EMA validation
- no attention / no GCN / no TCN / no T x T operator

## Required audit

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.nestsar_sm_all_t16.audit_parallel --output parallel_audit.json
```

It verifies:

1. parallel fast-weight equivalence to the original serial delta recurrence;
2. exact parameter count and absence of recurrent `scan`/`while` in the forward graph;
3. matching outputs for serial and parallel execution of the new architecture;
4. two real optimizer/EMA updates with a masked, zero-padded sample;
5. R4 scan-corrected GFLOPs vs parallel GFLOPs on the same backend, verifying the
   unrolled R4 outputs before accepting its cost;
6. synchronized batch-1 and batch-256 latency for R4, the new serial control,
   and the parallel model. Compilation, preprocessing, and transfers are excluded.

The default dual-T4 launcher runs this audit before starting XSUB/GPU0 and
XSET/GPU1 and saves `parallel_audit.json`. That assignment runs two independent
training jobs; it is separate from parallelizing the model's token recurrences.

For a local CPU validation:

```bash
JAX_PLATFORMS=cpu python -m pytest -q experiments/nestsar_sm_all_t16/test_parallel.py experiments/nestsar_sm_all_t16/streaming/tests/test_training.py experiments/nestsar_sm_all_t16/streaming/tests/test_launcher.py experiments/nestsar_sm_all_t16/test_preprocessing_corrected.py
JAX_PLATFORMS=cpu python -m experiments.nestsar_sm_all_t16.streaming.smoke_cpu --model parallel --outdir /tmp/nestsar_parallel_smoke_v2
JAX_PLATFORMS=cpu python -m experiments.nestsar_sm_all_t16.audit_parallel --allow-cpu --batches 1 --repeats 5 --warmup 2 --output parallel_cpu_audit.json
```

The smoke check trains both protocol workers on tiny synthetic clips, exercises
zero padding and gradient accumulation, repairs checkpoint aliases on resume,
and rejects an R4 resume into parallel output. It does not measure NTU accuracy.

## Interpretation

The recurrent token updates use a tree prefix computation with logarithmic
dependency depth. Spatial, M4, router, and G4 layers still depend on preceding
layers. Gradient accumulation and successive optimizer steps remain sequential.
Vectorization does not guarantee that every operation runs concurrently on a GPU.

Use `same_equation_speedup` to isolate the scan execution change. `vs_r4_speedup`
also includes a different recurrence and stream vectorization. CPU timing cannot
establish T4 or phone speed. XLA FLOP estimates must be compared on the same
backend/compiler; do not substitute them directly for older GPU audit figures.

Do not interpret accuracy until this audit passes.  ParallelAffineSweep changes
the base recurrence and therefore requires fresh training.

## Local verification, 2026-10-03

`parallel_validation_cpu.json` records the executed checks and source identity:
37 focused tests passed; both synthetic protocol workers completed two epochs,
resumed successfully, repaired checkpoint aliases, and rejected an R4 resume.
The original masked-padding NaN failure is covered by a full training regression.
The parallel forward graph has zero recurrent scans; the serial control has eight.
Identical-equation predictions differed by at most `2.98e-7` in the CPU check.

CPU/JAX 0.7.2 batch-1 XLA estimates were 0.069916688 GFLOPs for R4 and
0.069913904 for the parallel model, effectively unchanged under this counter.
These are not replacements for the historical GPU estimate of about 0.02962.

The short CPU timing check (2 warmups, 5 samples) measured median 2.922 ms for R4,
2.126 ms for the new serial control, and 3.660 ms for the parallel model.
It therefore does **not** support a CPU batch-1 speedup claim. T4 latency,
training throughput, memory consumption, and retrained NTU120 accuracy remain
to be measured before deciding whether this architecture should replace R4.

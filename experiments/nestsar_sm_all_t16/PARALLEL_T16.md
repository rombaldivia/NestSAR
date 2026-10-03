# NestSAR Full-Parallel T16

Branch: `experiment/nestsar-full-parallel-t16`

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

## First required audit

```bash
python -m experiments.nestsar_sm_all_t16.audit_parallel
```

It verifies:

1. parallel fast-weight equivalence to the serial recurrence;
2. exact parameter count;
3. R4 scan-corrected GFLOPs vs parallel GFLOPs;
4. batch-1 and batch-256 latency.

Do not interpret accuracy until this audit passes.  ParallelAffineSweep changes
the base recurrence and therefore requires fresh training.

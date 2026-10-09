# NestSAR-R5-T16 — the R4 architecture audit, fixed

R5 keeps NestSAR's identity (recurrent sweeps + nested M4/G4 memory with a
self-modifying fast memory; no attention, no graph convolution, no TCN) and
fixes the six problems found when auditing R4-FMSE against its class errors.

## What changed and why

| # | R4 finding | R5 fix | Ablation variant |
|---|---|---|---|
| 1 | Thumb and hand tip were averaged with wrist + hand into one 24-D part, then a 480→112 bottleneck. The worst classes are exactly the ones that differ by thumb/tip (A071 OK sign ↔ A072 victory sign, A073, A074, A084…). | 14 parts: thumb and hand tip are their own parts. A hand branch reads hand-relative vectors (tip−hand, thumb−hand, hand−wrist), and the model adds scale-free hand-shape scalars (distances, angles). | `coarse_parts` |
| 2 | 16 segment tokens keep only net displacement and path length, so the rhythm of small repeated motions (counting money, writing, cutting nails) is erased. | 4×-rate hand block (64 sub-segments, built from every raw frame) processed by a bidirectional GRU before pooling to 16; per-axis reversal amount (path − \|net\|) for every joint. | `no_hand_branch` |
| 3 | The adaptive mechanisms were capped so hard they could barely act: fusion logits ±0.15 (every weight stays in [0.198, 0.311]), FiLM ±10 %, stream gates ±10 %, η ≤ 0.2, α ∈ [0.9, 0.999], fast residual ×0.08, rank-2 head ×0.15. "Learned fusion ≈ average" was a consequence of the cap. | The capped controller, capped fusion and adaptive head are removed. The fast memory stays (it is the HOPE/Titans-inspired part), with η ∈ (0,1) and α ∈ (0.5,1) driven by its own prediction error ("surprise") and a learnable layer scale instead of 0.08. With unit-norm keys the update is non-expansive, so no cap is needed for stability. | `capped_fast_memory`, `no_fast_memory` |
| 4 | The joint sweep was a one-way chain: the left arm never saw the right arm. | Bidirectional joint sweep (both directions advanced in one scan). | `unidirectional_sweep` (one direction, wider state, similar compute) |
| 5 | Four per-view models (J, B, JM, BM) at 62–68 % each, logits averaged. | One early-fused trunk: all four views embedded together per joint. | (R4 itself is the reference) |
| 6 | The two actors met only in one linear layer after pooling. | Explicit person–person geometry (hand→head/torso/hand distances, relative position and velocity, facing, closing speed) and a layer-scaled cross-person message; all zero when the pair is absent. | `no_interaction` |

The training recipe is the R4 one (60 epochs, batch 256, AdamW + warmup-cosine,
EMA 0.995, label smoothing, canonical/augmented consistency, yaw ±8° and ±1
frame boundary jitter, patience 5). The training-only LocalGeometry loss is not
used: it measured +0.005 pp. Auxiliary heads (weight 0.15): the M4 temporal mean
and the hand branch.

## Cost (measured, batch 1, T = 16)

Strict counter: MACs of every dot in the inference jaxpr, scan bodies × length,
1 MAC = 2 FLOPs (the same counter as the SA audit).

| Model | Params | Strict MFLOPs | Recurrent scans | Sequential steps |
|---|---:|---:|---:|---:|
| R4-FMSE (reference) | 1,831,932 | 60.87 | 28 | 340 |
| **R5 (full)** | **1,146,656** | **59.16** | **6** | **129** |

R5 has fewer FLOPs and 37 % fewer parameters than R4. Its sequential chain is
2.6× shorter, which matters on a GPU or a Jetson, where these small recurrent
matmuls are latency-bound. Every variant:

| Variant | Params | Strict MFLOPs |
|---|---:|---:|
| full | 1,146,656 | 59.16 |
| no_hand_branch | 1,123,352 | 56.09 |
| coarse_parts | 1,128,224 | 57.67 |
| unidirectional_sweep | 1,144,304 | 55.47 |
| no_interaction | 1,131,072 | 58.31 |
| capped_fast_memory | 1,146,296 | 59.16 |
| no_fast_memory | 1,135,736 | 58.87 |

Reproduce: `python -m experiments.nestsar_r5_t16.audit_flops --with-r4 --variants`

## Run on Kaggle — one cell (GPU T4 ×2, internet on)

```python
import importlib.util, os, subprocess, sys
from pathlib import Path
REPO, BRANCH = "/kaggle/working/NestSAR_R5_branch", "experiment/nestsar-r5-t16"
OUT = "/kaggle/working/NestSAR_R5_T16_v1"
if not Path(REPO, ".git").exists():
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH,
                    "https://github.com/rombaldivia/NestSAR.git", REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "fetch", "-q", "--depth", "1", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO, "reset", "-q", "--hard", "FETCH_HEAD"], check=True)
p = subprocess.Popen([sys.executable, "-u", "-m", "experiments.nestsar_r5_t16.kaggle_run",
                      "--outdir", OUT, "--no-follow"],
                     cwd=REPO, env=dict(os.environ, PYTHONPATH=REPO),
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
if p.wait() != 0:
    raise RuntimeError("Setup failed - see the messages above.")
spec = importlib.util.spec_from_file_location("nestsar_r5_monitor", f"{REPO}/experiments/nestsar_r5_t16/monitor.py")
mon = importlib.util.module_from_spec(spec); spec.loader.exec_module(mon)
mon.monitor(OUT)
```

What it does:

1. Checks the two GPUs, and that JAX can use each one.
2. Builds the model on CPU: exact parameter count, finite forward pass, finite
   gradients on a zero-padded batch.
3. Reuses the R4 canonical cache already in `/kaggle/working` (or
   `/kaggle/input`), or builds one from `ntu120*.pkl`.
4. Builds the R5 hand cache next to it (`NestSAR_R5_HAND_CACHE_v1_<hash>`,
   about 1.4 GiB, a few minutes). Every sample's R4 tokens are recomputed and
   must match the cache exactly.
5. Starts the dual-GPU launcher detached (XSUB → GPU0, XSET → GPU1). The
   monitor shows two bars and, per epoch, val / top-5 / hand-branch accuracy /
   fast-memory η and the delta against R4 at the same epoch.

Stopping the cell never stops training; running it again re-attaches.
Interrupted runs resume from `last.msgpack`. A worker that dies for a
non-deterministic reason is restarted up to twice. NaN, OOM and config
mismatches are reported instead of retried. Only one launcher can manage an
output folder. If a worker runs out of GPU memory, add `"--micro-batch", "32"`
(accumulation 8, the effective batch stays 256) and use a new `OUT`.

Expected time: about the same per epoch as R4 (both are limited by the
augmented-view preparation, about 3 ms per sample on one CPU thread).

## Early kill rule

At epoch 10 the launcher compares EMA validation accuracy with the R4-FMSE +
LocalGeometry run found in `/kaggle/working` or `/kaggle/input` (prefers
`NestSAR_R4_FMSE_LOCAL_GEOMETRY_T16_v1`; its `run_config.json` must say model
`NestSAR-R4-FMSE-T16-v1` with 1,831,932 parameters). R5 is a new architecture,
so its early curve predicts less than a one-block change does:

- **KILL** only if every protocol is more than 1.0 pp behind (`--kill-margin-pp`).
- **CONTINUE** otherwise.

A killed run is never restarted by re-running the cell (`--ignore-kill`
overrides). Without a reference run the rule is off and training runs to the end.

## Optional: does R4 use its fast memory? (inference-only, a few minutes on one GPU)

Run it before starting R5 or after it finishes (not while both GPUs train):

```python
import os, subprocess, sys
REPO = "/kaggle/working/NestSAR_R5_branch"
subprocess.run([sys.executable, "-u", "-m", "experiments.nestsar_r5_t16.ablate_r4_fastweights"],
               cwd=REPO, env=dict(os.environ, PYTHONPATH=REPO, CUDA_VISIBLE_DEVICES="0"), check=True)
```

The script loads the R4-FMSE + LocalGeometry best checkpoints and evaluates
the full validation split as trained and with:

- η = 0 and α = 1 (memory frozen at its learned initial state, for both levels
  or for M4 or G4 alone);
- the fast residual removed;
- uniform fusion;
- no adaptive head;
- every capped controller knob neutral.

It first checks that the trained accuracy reproduces exactly, then saves
`r4_fastweight_ablation.json`. These are counterfactuals on trained weights.
Training-time evidence comes from the R5 variants.

## Ablations

```python
# same cell, other variant and a new output folder, e.g.
[..., "--variant", "no_hand_branch", "--outdir", "/kaggle/working/NestSAR_R5_T16_no_hand_branch"]
# one protocol only (runs on GPU0):  "--protocols", "xsub"
```

## Outputs (`OUT/<protocol>/`)

- `history.json`: per-epoch training and validation metrics, η/α, fast-memory
  and interaction scales, auxiliary accuracies, timing and memory.
- `best.msgpack` / `best.json`: EMA weights of the best epoch.
- `last.msgpack`: resume state.
- `result.json`: also written to `OUT/result_<protocol>.json`.
- `per_class.json`: per-class recall of the best checkpoint, the top confusions,
  and R4 vs R5 recall on the classes R4 got most wrong.

Also in `OUT`: `kill_decision.json`, `reference.json`, `launcher.log`,
`<protocol>.log`.

## Local verification

```bash
JAX_PLATFORMS=cpu python -m pytest -q experiments/nestsar_r5_t16/tests
JAX_PLATFORMS=cpu python -m experiments.nestsar_r5_t16.smoke_cpu --root /tmp/r5_smoke
```

The tests cover:

- R4 tokens bit-exact against `preprocessing_corrected`, for canonical and
  augmented views.
- The hand block against an independent loop implementation, with every
  transition counted once.
- Parameter counts of every variant.
- Finite forward passes and gradients with zero-padded rows.
- That an absent person's hand data is ignored.
- The fast memory against a loop reference.
- The single-scan BiGRU against two separate GRUs.
- Cache build, validation, refusal and relocation.
- The kill rule and reference discovery.
- Worker restart vs no-retry, the launcher lock and zombie-safe liveness.
- The R4 counterfactual script.

The smoke test runs the exact Kaggle path on a synthetic pickle: caches,
detached launcher, two concurrent workers, kill decision, results, re-run,
resume, and refusal of a variant change.

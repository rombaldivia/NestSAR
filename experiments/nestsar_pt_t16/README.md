# NestSAR-PT-T16 — part tokens through the nested temporal memory

## Why

In R4 (`experiments/nestsar_sm_all_t16`, `MaskSafeSpatialEncoder`) every frame is
flattened — 2 persons × 10 parts × 24 dims → **one** 112-d vector — *before* any
temporal memory. The temporal hierarchy never sees the trajectory of an
individual body part: the hands (parts 3 and 5) are mixed with the whole body
first. Audited symptoms on the R4-FMSE checkpoints (NTU120): single streams reach
only 62–68 %, the learned fusion is ≈ a uniform mean (−0.14 / −0.21 pp), the
4-stream oracle is 85–86 %, and wider models (D128, D192) did not help.

## What changes

```
joints ─ mask-safe joint sweep ─ 10 part tokens (persons fused per part, valid-joint mean)
       ─ bidirectional cross-part GatedSweep            (spatial, every frame)
       ─ per-part self-modifying M4 memory               (weights shared across the 10 parts)
       ─ part read-out 10×Dp → 112                       (collapse only AFTER fast temporal memory)
       ─ R4 CrossStreamRouter → G4 descriptors → classifiers / fusion / adaptive head (unchanged)
```

Unchanged from R4: `SharedSMController`, the four mask-safe streams (J/B/JM/BM),
router, G4, heads, losses, augmentation, EMA, cache and the whole streaming
training pipeline (`experiments/nestsar_sm_all_t16/streaming`). No softmax
attention, no graph/GCN, no CNN/TCN, no T×T operation.

## Cost (measured, batch 1, T = 16, 1 MAC = 2 FLOPs)

| Model | Params | Strict MFLOPs |
|---|---:|---:|
| R4 (`fast_rank=4`) | 1,831,932 | 60.91 |
| **NestSAR-PT `part_dim=32` (default)** | **1,173,436** | **75.40** |
| NestSAR-PT `part_dim=40` | 1,276,828 | 97.89 |
| NestSAR-PT `part_dim=48` | 1,394,556 | 124.96 |

Reproduce: `python -m experiments.nestsar_pt_t16.audit_flops --part-dim 32 40 48 --with-r4`
(counts MACs from the JAXPR; scan bodies are multiplied by their length).

## Run on Kaggle (one cell, GPU T4 ×2, internet on)

```python
import os, subprocess, sys
from pathlib import Path
REPO, BRANCH = "/kaggle/working/NestSAR_PT_branch", "experiment/nestsar-pt-t16"
if not Path(REPO, ".git").exists():
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH,
                    "https://github.com/rombaldivia/NestSAR.git", REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "fetch", "-q", "--depth", "1", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO, "reset", "-q", "--hard", "FETCH_HEAD"], check=True)
p = subprocess.Popen([sys.executable, "-u", "-m", "experiments.nestsar_pt_t16.kaggle_run",
                      "--part-dim", "32", "--micro-batch", "64",
                      "--outdir", "/kaggle/working/NestSAR_PT_T16_v1"],
                     cwd=REPO, env=dict(os.environ, PYTHONPATH=REPO),
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
p.wait()
```

`kaggle_run` checks the two GPUs, verifies the exact parameter count on CPU,
reuses a compatible canonical cache (or builds one from `ntu120*.pkl`), and starts
the dual-GPU launcher **detached** (XSUB → GPU0, XSET → GPU1). Stopping the cell
does not stop training; re-running it re-attaches to the live log. Interrupted
runs resume from `last.msgpack`.

If a worker runs out of GPU memory, use `--micro-batch 32` (accumulation 8, the
effective batch stays 256) and a new `--outdir`.

## Live progress bars (optional second cell)

The launch cell prints text lines (one per phase change, plus a heartbeat every
10 minutes). For the two R4-style progress rows, stop the launch cell — training
keeps running — and run:

```python
import importlib.util, subprocess
REPO = "/kaggle/working/NestSAR_PT_branch"
subprocess.run(["git", "-C", REPO, "fetch", "-q", "--depth", "1", "origin", "experiment/nestsar-pt-t16"], check=True)
subprocess.run(["git", "-C", REPO, "reset", "-q", "--hard", "FETCH_HEAD"], check=True)
spec = importlib.util.spec_from_file_location("nestsar_pt_monitor", f"{REPO}/experiments/nestsar_pt_t16/monitor.py")
mon = importlib.util.module_from_spec(spec); spec.loader.exec_module(mon)
mon.monitor("/kaggle/working/NestSAR_PT_T16_v1")
```

`monitor.py` only reads the status/history files: it lists the R4 reference runs
(and whether the epoch-10 rule can decide), then shows the two progress rows,
one line per finished epoch with its delta vs R4, and the kill decision.
Stopping it never affects training.

## Early kill rule

At epoch 10 the launcher compares EMA validation accuracy with the plain R4 run
found in `/kaggle/working` (prefers `NestSAR_R4_EMA_REP_CONSISTENCY_T16_v1`; it must
have `parameters = 1,831,932` and no `model` key in `run_config.json`):

- **KILL** if NestSAR-PT is behind R4 on both protocols, or more than 0.5 pp behind
  on either one. The decision is written to `kill_decision.json` and a killed run
  is never restarted by re-running the cell (`--ignore-kill` overrides).
- **CONTINUE** otherwise, to the normal end / early stopping.

The comparison is against plain R4 (not FMSE) because FMSE adds training-only
geometry losses. If NestSAR-PT wins, the next step is adding those losses,
RegMask and 3-D rotation augmentation on top of this backbone.

## Outputs (`--outdir`)

`{xsub,xset}/history.json`, `best.json`, `best.msgpack`, `result.json` (same format
as R4, with `model_identity`), `kill_decision.json`, `launcher.log`,
`{xsub,xset}.log`.

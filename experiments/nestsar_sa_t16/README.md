# NestSAR-SA-T16 — R4-FMSE + per-frame spatial joint attention

## Idea

Titans/HOPE-style division of labour, adapted to skeletons:

- **Short-term, exact memory over space.** Softmax attention over the 50 joint
  tokens of each frame (2 persons × 25 joints), with a learned skeleton-hop bias
  (hops 0–7) and a separate bucket for the other person's joints. Absent joints and
  persons are masked out of the keys and their outputs are zeroed.
- **Long-term memory over time.** The unchanged R4-FMSE self-modifying M4/G4 nested
  memory, router, descriptors and heads.

Audits of R4-FMSE showed the weakness is spatial: single streams at 62–68 %, errors
concentrated on hand / fine-motion classes, and every temporal-memory variant
plateaued. The only change is one pre-norm, 2-head attention block (layer-scaled
residual, init 0.1) inside each stream's spatial encoder, after the joint sweep and
before part pooling. No temporal attention, no graph convolution, no CNN/TCN.

Training uses the proven R4-FMSE + LocalGeometry pipeline unchanged
(`experiments/nestsar_r4_fmse_geometry_t16`): same cache, augmentation, losses,
optimizer, schedule (60 epochs), EMA and checkpoints.

## Cost (measured, batch 1, T = 16, 1 MAC = 2 FLOPs)

| Model | Params | Strict MFLOPs |
|---|---:|---:|
| R4-FMSE (reference) | 1,831,932 | 60.87 |
| **NestSAR-SA** | **1,841,604** | **90.98** |

Reproduce: `python -m experiments.nestsar_sa_t16.audit_flops --with-r4`

## Run on Kaggle — one cell (GPU T4 ×2, internet on)

```python
import importlib.util, os, subprocess, sys
from pathlib import Path
REPO, BRANCH = "/kaggle/working/NestSAR_SA_branch", "experiment/nestsar-sa-t16"
OUT = "/kaggle/working/NestSAR_SA_T16_v1"
if not Path(REPO, ".git").exists():
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH,
                    "https://github.com/rombaldivia/NestSAR.git", REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "fetch", "-q", "--depth", "1", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO, "reset", "-q", "--hard", "FETCH_HEAD"], check=True)
p = subprocess.Popen([sys.executable, "-u", "-m", "experiments.nestsar_sa_t16.kaggle_run",
                      "--outdir", OUT, "--micro-batch", "64", "--no-follow"],
                     cwd=REPO, env=dict(os.environ, PYTHONPATH=REPO),
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="")
if p.wait() != 0:
    raise RuntimeError("Setup failed - see the messages above.")
spec = importlib.util.spec_from_file_location("nestsar_sa_monitor", f"{REPO}/experiments/nestsar_sa_t16/monitor.py")
mon = importlib.util.module_from_spec(spec); spec.loader.exec_module(mon)
mon.monitor(OUT)
```

What it does:
1. `kaggle_run` checks the two GPUs, builds the model on CPU and verifies the exact
   parameter count, reuses a compatible canonical cache (or builds one from
   `ntu120*.pkl`), and starts the dual-GPU launcher **detached** (XSUB → GPU0,
   XSET → GPU1).
2. `monitor` lists the R4-FMSE reference runs found, then shows two tqdm bars
   (BEST / val / train acc / loss) and one line per finished epoch with
   train acc, val acc, top-5 and the delta vs R4-FMSE at the same epoch.

Stopping the cell never stops training; running it again re-attaches. Interrupted
runs resume from `last.msgpack`. If a worker runs out of GPU memory, use
`--micro-batch 32` (accumulation 8, the effective batch stays 256) and a new `OUT`.

## Early kill rule

At epoch 10 the detached launcher compares EMA validation accuracy with the
R4-FMSE + LocalGeometry run in `/kaggle/working` (prefers
`NestSAR_R4_FMSE_LOCAL_GEOMETRY_T16_v1`; `run_config.json` must say
`model = NestSAR-R4-FMSE-T16-v1`, `parameters = 1,831,932`). It warns if the
reference used different epochs / learning rate / seed / effective batch.

- **KILL** if NestSAR-SA is behind on every protocol that has a reference value, or
  more than 0.5 pp behind on any of them. Written to `kill_decision.json`; a killed
  run is never restarted by re-running the cell (`--ignore-kill` overrides).
- **CONTINUE** otherwise, to the normal end / early stopping.

## Outputs (`OUT`)

`{xsub,xset}/history.json`, `best.json`, `best.msgpack`, `result.json` (same format
as R4-FMSE, with `model_identity`), `kill_decision.json`, `launcher.log`,
`{xsub,xset}.log`.

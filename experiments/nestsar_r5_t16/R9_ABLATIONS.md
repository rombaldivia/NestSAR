# R9: leave-one-out ablations of graph_mr

Question: which parts of the champion (`graph_mr`, 77.75 / 78.03 XSub with seeds 7 / 11) actually carry the
accuracy. Every run uses the champion recipe (38 epochs, batch 64 x 4, aug pool 6 views fp16, dropout 0.3,
wd 0.05, label smoothing 0.1, recon mask 0.5), XSub only, seeds 7 and 11 so each ablation pairs with a base
run clip by clip. `graph_mr` and `graph_mr_no_fast` are bit-for-bit unchanged (checked against the previous
commit on random inputs).

| Variant | Removes / changes | Params (incl. train-only heads) | Inference params | MFLOPs |
|---|---|---:|---:|---:|
| graph_mr | (base) | 1,300,768 | 1,104,994 | 40.88 |
| graph_mr_no_fast | nested fast memory (m4 and g4) | 1,289,848 | 1,094,074 | 40.59 |
| graph_mr_one_level | g4 (the 4-chunk time scale) | 829,084 | 633,310 | 37.29 |
| graph_mr_mem_only | the BiGRUs of m4/g4: the fast memory alone models time | 430,272 | 234,498 | 23.54 |
| graph_mr_no_temporal | every temporal model (frame features averaged) | 419,352 | 223,578 | 23.24 |
| graph_mr_no_hand | 4x-rate hand branch | 1,277,464 | 1,089,490 | 37.82 |
| graph_mr_coarse | 14 fine parts -> 10 parts | 1,282,336 | 1,086,562 | 39.39 |
| graph_mr_no_inter | person-person interaction | 1,285,184 | 1,089,410 | 40.03 |
| graph_mr_no_recon | masked reconstruction (training only) | 1,134,034 | 1,104,994 | 40.88 |
| graph_mr_shared | m4 and g4 share one NestedMemory (+ level embedding) | 860,236 | 664,462 | 40.88 |
| graph_mr_causal | forward-only GRUs (state 272), hand branch forward-only | 1,254,968 | 1,059,194 | 38.85 |
| graph_mr_causal_no_fast | causal and no fast memory | 1,244,048 | 1,048,274 | 38.55 |

## Plan (one run ~ 3.2 h on a T4; two GPUs per session)

- Tier 1 (7 runs, decides what the paper can claim): no_fast s11, mem_only s7/s11, shared s7/s11, one_level s7/s11.
- Tier 2 (10 runs, component table): no_temporal, no_hand, no_inter, coarse, no_recon at s7/s11.
- Tier 3 (4 runs, streaming story): causal and causal_no_fast at s7/s11.

## Kaggle cells

```python
# 1) code + caches + one queue per GPU (re-run in each new session: finished runs are skipped)
import os, subprocess, sys
from pathlib import Path
W, CODE, BR = Path("/kaggle/working"), Path("/tmp/NestSAR_code_r9"), "experiment/nestsar-r9-ablations"
TIERS = "1"            # next sessions: "1,2", then "1,2,3"
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
if not (CODE / ".git").exists():
    sh(f"git clone -q -b {BR} https://github.com/rombaldivia/NestSAR.git {CODE}")
sh(f"git -C {CODE} fetch -q origin {BR} && git -C {CODE} reset -q --hard origin/{BR}")
print(sh(f"git -C {CODE} log --oneline -1").stdout)
env = dict(os.environ, PYTHONPATH=str(CODE), NESTSAR_FRAMES="16")
subprocess.run([sys.executable, "-u", "-m", "experiments.nestsar_r5_t16.kaggle_run", "--prepare-only",
                "--hand-cache", "/tmp/NestSAR_R5_HAND_CACHE_T16", "--aug-pool", "/tmp/NestSAR_AUGPOOL_T16",
                "--pool-views", "6", "--pool-dtype", "float16", "--r4-build-dir", "/tmp",
                "--extra-config", '{"aug_strength": 1.0}'], cwd=CODE, env=env, check=True)
for g in (0, 1):
    subprocess.Popen([sys.executable, "-u", "-m", "experiments.nestsar_r5_t16.ablation_queue",
                      "--gpu", str(g), "--tier", TIERS], cwd=CODE, env=env, start_new_session=True,
                     stdout=open(W / f"QUEUE_R9_GPU{g}.out", "a"), stderr=subprocess.STDOUT)
print("colas lanzadas")
```

```python
# 2) progress
!tail -n 4 /kaggle/working/QUEUE_R9_GPU0.out /kaggle/working/QUEUE_R9_GPU1.out
```

```python
# 3) report (any time; missing runs are listed as pending)
!cd /tmp/NestSAR_code_r9 && PYTHONPATH=. python -m experiments.nestsar_r5_t16.ablation_report --working /kaggle/working --json /kaggle/working/R9_ablations.json
```

## Reading the report

`delta = ablation - reference` in pp; negative means the removed part was helping. McNemar compares two
models of the same seed on the same 50,919 clips (sample noise only). The base's seed-to-seed range is the
seed noise. "APORTA" needs the same sign on every seed, McNemar p < 0.01 on each, and a mean drop at least
as large as max(0.3 pp, seed range). With two seeds this is a coarse screen: anything near the threshold needs
a third seed (13) for the base and that ablation.

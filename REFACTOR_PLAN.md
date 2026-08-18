# NestSAR experiment-runner refactor

## Goal

Move validated research experiments out of giant Kaggle cells and into versioned GitHub modules, while keeping notebook usage to a few lines and preserving exact experimental behavior.

## Safety rule

Do not rewrite the mathematics while migrating. First reproduce the exact validated experiment, then clean internals in separate commits.

Golden Attention-Lite v1 guards:

- NTU120 XSUB
- 16 frames
- D128
- Attention D64 / H4 / Dh16
- 2,381,028 parameters
- 705 parameter leaves
- 0.0604169 GFLOPs per clip from JAX/XLA cost analysis
- Global batch 32 on TPU v5e-8
- Grad accumulation 4 / effective batch 128
- FAST/MEDIUM/SLOW/CONSOLIDATE periods 4/8/16/32 microbatches
- RegMask 8% / 8% / 3%
- EMA 0.995
- Probe reference: E3 XSUB = 58.75017%

## Target repository layout

```text
NestSAR/
├── nestsar.py                    # keep legacy trainer working during migration
├── nestsar_run.py                # stable experiment launcher
├── nestsar_lab/
│   ├── registry.py
│   ├── config.py
│   ├── data.py
│   ├── models/
│   │   ├── hope_attention_lite.py
│   │   └── ...
│   └── training/
│       ├── stagewise.py
│       ├── tpu.py
│       ├── checkpoints.py
│       └── metrics.py
├── experiments/
│   └── attention_lite_t16_xsub.py
└── tests/
    ├── test_registry.py
    ├── test_parameter_guards.py
    └── test_schedule_guards.py
```

Use `nestsar_lab` during migration instead of a `nestsar/` package because the repository already has a root `nestsar.py`; this avoids Python import ambiguity.

## Public notebook API target

```python
from nestsar_run import train
train("attention_lite_t16", epochs=40, dataset="auto", protocol="xsub")
```

After packaging is stable, Kaggle can become:

```python
!pip install -q --no-deps git+https://github.com/rombaldivia/NestSAR.git
from nestsar_run import train
train("attention_lite_t16", epochs=40, dataset="auto", protocol="xsub")
```

## Migration phases

1. Freeze the exact successful Attention-Lite source and parameter-tree guards.
2. Extract model, stagewise optimizer, TPU runtime, evaluation, and checkpoint code without changing formulas.
3. Add an experiment registry and `train()` launcher.
4. Make epoch-dependent schedule length derive from the requested epoch count; never hard-code the old 3-epoch `1478`-step horizon into long runs.
5. Add CI checks for syntax, registry, parameter count/leaves, and schedule math.
6. Run a TPU smoke test and compare the first three epochs against the validated probe before merging to `main`.
7. Only after parity is confirmed, deduplicate historical modules and improve internal APIs.

## Merge gate

Do not merge the refactor to `main` until the modular version passes the same architecture guards and a TPU probe reproduces the expected learning behavior without rollback/NaNs.

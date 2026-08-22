# Attention-Lite v1

Versioned **NestSAR-HOPE-Attention-Lite D128** TPU v5e-8 experiment for NTU120 reproducibility and controlled training studies.

## Golden architecture — frozen

The architecture remains fixed:

- 16 frames
- D128
- Attention D64 / H4 / Dh16
- 2,381,028 parameters
- 705 parameter leaves
- 0.060416900 GFLOPs per clip from the exact JAX/XLA forward cost audit
- TPU v5e-8, all 8 devices
- FAST/MEDIUM/SLOW/CONSOLIDATE stagewise optimizer structure
- RegMask 8% frame / 8% joint / 3% part
- EMA 0.995

Golden training defaults:

```text
epochs                  40
patience                 0   # disabled, matching original E40 source
dropout                0.22
learning_rate         1e-3
weight_decay           0.05
warmup_fraction        0.10
label_smoothing        0.05
grad_clip               1.0
predictive_loss_weight 0.10
initial_eta             0.02
initial_alpha           0.95
global_batch              32
grad_accum                 4
effective_batch           128
eval_batch                 32
```

Paper seeds: `28, 42, 128, 2026`. Use the same seed set and training recipe for XSUB and XSET when computing final mean ± std.

## Exact-source safety rule

The validated XSUB/XSET all-in-one sources remain the mathematical source of truth. The repository wrapper does **not** approximate or silently rebuild them from another architecture.

Expected Python source filenames:

```text
NestSAR_HOPE_Attention_Lite_TPUv5e8_XSUB_E40_ALL_IN_ONE.py
NestSAR_HOPE_Attention_Lite_TPUv5e8_XSET_E40_ALL_IN_ONE.py
```

The launcher now runs `source_resolver.py` **before** starting a child process. It searches `/kaggle/input`, `/kaggle/working`, the current directory, and the repository canonical folder. It also supports extracting a complete all-in-one code cell from known one-cell notebook inputs, including:

```text
NestSAR_HOPE_Attention_Lite_TPUv5e8_XSET_E40_ONE_CELL.ipynb
```

A source must contain the protocol-specific Attention-Lite golden markers (including the embedded source bundle and parameter guards) or it is rejected. If neither exact source is attached, the run stops immediately with an explicit `FileNotFoundError` instead of a generic `CalledProcessError`.

Preflight both sources from Kaggle:

```python
from experiments.attention_lite_v1.source_resolver import resolve_both_sources
sources = resolve_both_sources()
print(sources)
```

## Independent XSUB + XSET models

`train_both()` runs **two independent models sequentially** on the full TPU8. XSUB finishes first; XSET then starts in a fresh Python subprocess from a fresh initialization with the same requested seed. XSET never inherits XSUB weights.

```python
from nestsar_run import train_both

train_both(
    "attention_lite",
    seed=28,
    epochs=40,
    patience=12,
    dropout=0.22,
    learning_rate=1e-3,
    weight_decay=0.05,
    warmup_fraction=0.10,
    label_smoothing=0.05,
    grad_clip=1.0,
    predictive_loss_weight=0.10,
    initial_eta=0.02,
    initial_alpha=0.95,
    batch_size=32,
    grad_accum_steps=4,
    eval_batch_size=32,
    run_tag="p12",
    dataset="auto",
)
```

If auto-discovery cannot find the sources, pass them explicitly:

```python
train_both(
    "attention_lite",
    xsub_source="/kaggle/input/<dataset>/NestSAR_HOPE_Attention_Lite_TPUv5e8_XSUB_E40_ALL_IN_ONE.py",
    xset_source="/kaggle/input/<dataset>/NestSAR_HOPE_Attention_Lite_TPUv5e8_XSET_E40_ALL_IN_ONE.py",
    seed=28,
    epochs=40,
)
```

## One-protocol API

```python
from nestsar_run import train

train(
    "attention_lite",
    protocol="xsub",
    seed=28,
    epochs=40,
    patience=12,
    dropout=0.22,
    learning_rate=1e-3,
    weight_decay=0.05,
    warmup_fraction=0.10,
    label_smoothing=0.05,
    grad_clip=1.0,
    predictive_loss_weight=0.10,
    initial_eta=0.02,
    initial_alpha=0.95,
    batch_size=32,
    grad_accum_steps=4,
    eval_batch_size=32,
    run_tag="p12",
    dataset="auto",
)
```

`patience=None` or `patience=0` disables early stopping. Architecture-defining values (frames, model width, attention width/heads, parameter tree) are intentionally not exposed.

CLI supports `--protocol xsub`, `--protocol xset`, or `--protocol both`, plus the same training hyperparameter overrides.

## Output structure

Golden run example:

```text
/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_28/
```

Tagged custom run example:

```text
/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_28_p12/
```

Each run contains the canonical checkpoint/result files plus:

```text
generated_source/
logs/train.log
metadata/paths.json
metadata/run_manifest.json
```

A custom training configuration without `run_tag=` receives a stable `CUSTOM_<hash>` suffix to prevent accidental overwrite of a golden result.

## Four-seed summary

Golden, untagged runs can be aggregated with:

```bash
python -m experiments.attention_lite_v1.multiseed_summary
```

which writes:

```text
/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_4SEED_SUMMARY.json
```

## Merge gate

Do not merge this experiment runner to `main` until an actual TPU smoke/probe verifies that the generated source reproduces the golden Attention-Lite architecture and expected learning behavior. Static source validation and Python compilation are necessary but not substitutes for TPU parity.

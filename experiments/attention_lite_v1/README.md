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

Golden training defaults are:

```text
epochs                 40
patience                0   # disabled, matching the original E40 source
dropout               0.22
learning_rate        1e-3
weight_decay          0.05
warmup_fraction       0.10
label_smoothing       0.05
grad_clip              1.0
predictive_loss_weight 0.10
initial_eta            0.02
initial_alpha          0.95
global_batch             32
grad_accum                4
effective_batch          128
eval_batch                32
```

Paper seeds are:

```text
28, 42, 128, 2026
```

Use the same seed set for XSUB and XSET for the final mean ± std table.

## Safety strategy

The validated XSUB/XSET all-in-one source remains the mathematical source of truth. `trainer.py` does not recreate the model. It:

1. finds the exact validated protocol-specific all-in-one source;
2. verifies golden architecture markers;
3. patches the requested seed, output path, and training hyperparameters only;
4. keeps T16/D128/D64-H4-Dh16/parameter-count/leaves architecture guards fixed;
5. saves the generated source and SHA256 provenance;
6. executes it in a clean subprocess;
7. verifies the final result against the requested batch/accumulation configuration.

Expected canonical filenames:

```text
NestSAR_HOPE_Attention_Lite_TPUv5e8_XSUB_E40_ALL_IN_ONE.py
NestSAR_HOPE_Attention_Lite_TPUv5e8_XSET_E40_ALL_IN_ONE.py
```

Put the exact source in `/kaggle/input`, `/kaggle/working`, or pass `canonical_source=` explicitly.

## Deterministic output structure

A golden run uses:

```text
/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_28/
```

A custom training configuration is automatically separated with a stable hash unless `run_tag=` is supplied:

```text
/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_28_CUSTOM_a1b2c3d4/
```

or, with `run_tag="p12_lr8e4"`:

```text
/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_28_p12_lr8e4/
```

Each run contains the canonical checkpoint/result files plus:

```text
generated_source/
logs/train.log
metadata/paths.json
metadata/run_manifest.json
```

## Kaggle API — golden run

```python
from nestsar_run import train

train(
    "attention_lite",
    protocol="xsub",
    seed=28,
    epochs=40,
    dataset="auto",
)
```

`patience=None` preserves the original E40 behavior: no early stopping.

## Kaggle API — configurable training

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

A positive `patience` enables early stopping in the generated source. `patience=0` or `None` disables it.

Architecture-defining values such as frames, model width, attention width/heads, and parameter tree are intentionally not exposed here.

CLI equivalents are also available with `--patience`, `--dropout`, `--learning-rate`, `--weight-decay`, `--warmup-fraction`, `--label-smoothing`, `--grad-clip`, `--predictive-loss-weight`, `--initial-eta`, `--initial-alpha`, `--batch-size`, `--grad-accum-steps`, and `--eval-batch-size`.

## Four-seed summary

Golden, untagged runs can be aggregated with:

```bash
python -m experiments.attention_lite_v1.multiseed_summary
```

This writes:

```text
/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_4SEED_SUMMARY.json
```

## Merge gate

Do not merge this experiment runner to `main` until an actual TPU smoke/probe verifies that the generated source reproduces the golden Attention-Lite architecture and expected learning behavior. Static patch generation and Python compilation are necessary but not a substitute for TPU parity.

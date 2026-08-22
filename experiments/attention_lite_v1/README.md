# Attention-Lite v1

Versioned **NestSAR-HOPE-Attention-Lite D128** TPU v5e-8 experiment for the final NTU120 reproducibility study.

## Golden architecture — frozen

The paper runner must preserve the validated experiment:

- 16 frames
- D128
- Attention D64 / H4 / Dh16
- 2,381,028 parameters
- 705 parameter leaves
- 0.060416900 GFLOPs per clip from the exact JAX/XLA forward cost audit
- TPU v5e-8, all 8 devices
- global batch 32 / local batch 4
- gradient accumulation 4 / effective batch 128
- FAST/MEDIUM/SLOW/CONSOLIDATE periods 4/8/16/32 microbatches
- RegMask 8% frame / 8% joint / 3% part
- EMA 0.995
- 40 epochs for the paper multi-seed study

Only **protocol** and **seed** are experimental variables in paper mode.

Paper seeds are frozen to:

```text
28, 42, 128, 2026
```

Use the same seed set for XSUB and XSET.

## Safety strategy

The validated XSUB/XSET all-in-one source remains the mathematical source of truth until the modular refactor passes TPU parity. `trainer.py` does not recreate the model. It:

1. finds the exact validated protocol-specific all-in-one source;
2. verifies golden architecture markers;
3. patches only the seed and generated output path;
4. saves the generated source and SHA256 provenance;
5. executes it in a clean subprocess;
6. verifies the final `result.json` protocol, seed, parameters, leaves, frames, and batch/accumulation guards.

This avoids silently changing the successful Attention-Lite mathematics during the reproducibility runs.

Expected canonical source filenames:

```text
NestSAR_HOPE_Attention_Lite_TPUv5e8_XSUB_E40_ALL_IN_ONE.py
NestSAR_HOPE_Attention_Lite_TPUv5e8_XSET_E40_ALL_IN_ONE.py
```

Put the exact source in `/kaggle/input`, `/kaggle/working`, or pass `canonical_source=` explicitly.

## Deterministic output structure

Each run gets a seed-labeled main directory:

```text
/kaggle/working/
├── NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_28/
│   ├── best_ema.msgpack
│   ├── last_weights.msgpack
│   ├── last_stagewise_state.pkl
│   ├── history.json
│   ├── result.json
│   ├── generated_source/
│   │   └── attention_lite_xsub_seed_28_generated.py
│   ├── logs/
│   │   └── train.log
│   └── metadata/
│       ├── paths.json
│       └── run_manifest.json
├── NestSAR_HOPE_Attention_Lite_D128_XSET_SEED_28/
├── NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_42/
├── NestSAR_HOPE_Attention_Lite_D128_XSET_SEED_42/
├── NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_128/
├── NestSAR_HOPE_Attention_Lite_D128_XSET_SEED_128/
├── NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_2026/
└── NestSAR_HOPE_Attention_Lite_D128_XSET_SEED_2026/
```

The canonical trainer's checkpoint/result files stay at the run-root level so exact stagewise resume behavior is preserved.

## Kaggle API

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

Then run XSET with the same seed:

```python
train(
    "attention_lite",
    protocol="xset",
    seed=28,
    epochs=40,
    dataset="auto",
)
```

CLI equivalent:

```bash
python -m nestsar_run attention_lite --protocol xsub --seed 28 --epochs 40
python -m nestsar_run attention_lite --protocol xset --seed 28 --epochs 40
```

If auto-discovery cannot find the canonical source:

```bash
python -m nestsar_run attention_lite \
  --protocol xsub \
  --seed 28 \
  --canonical-source /kaggle/input/attention-lite-source/NestSAR_HOPE_Attention_Lite_TPUv5e8_XSUB_E40_ALL_IN_ONE.py
```

## Four-seed summary

After the eight protocol/seed directories are present:

```bash
python -m experiments.attention_lite_v1.multiseed_summary
```

This writes:

```text
/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_4SEED_SUMMARY.json
```

with the four individual seed results and mean ± sample standard deviation for XSUB and XSET.

## Merge gate

Do not merge this experiment runner to `main` until an actual TPU smoke/probe verifies that the generated source reproduces the golden Attention-Lite architecture and expected learning behavior. The branch intentionally keeps the validated source as the source of truth rather than approximating it.

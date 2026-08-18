# Attention-Lite v1

Versioned **NestSAR-HOPE-Attention-Lite D128** TPU v5e-8 XSUB experiment.

Validated architecture guards: 16 frames, D128, attention D64/H4/Dh16, 2,381,028 parameters, 705 leaves. The compiler audit measured 0.0604169 GFLOPs per clip.

The migration rule is strict: preserve the validated mathematics and parameter tree first; refactor internals only after TPU parity is confirmed.

Target notebook API:

```python
from nestsar_run import train
train("attention_lite", epochs=40, dataset="auto")
```

Target CLI:

```bash
python -m nestsar_run attention_lite --epochs 40 --dataset auto --seed 128
```

The experiment module is being migrated on this branch and must not be merged to `main` until the TPU smoke/probe and architecture guards pass.

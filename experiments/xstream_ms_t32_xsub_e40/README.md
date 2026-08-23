# NestSAR HOPE XStream-MS T32 — XSUB E40

Experimental NestSAR candidate for the NTU120 XSUB protocol.

This version keeps the D128 Attention-Lite temporal core, increases the temporal input to 32 frames using linear interpolation, and adds learned multiscale cross-stream interaction between joint, bone, joint-motion, and bone-motion representations before the existing late fusion.

## Locked experiment

- Frames: 32
- Model dimension: 128
- Temporal attention: D64 / 4 heads / head dim 16
- Cross-stream attention: D64 / 4 heads / head dim 16
- Cross-stream attention is performed across the 4 streams at the same temporal position
- Self-stream diagonal is masked in the cross-stream mixer
- Cross-stream hints are injected at L1, L2, and L3
- Cross gate max: 0.25
- Cross gate init: 0.05
- Cross-stream learned parameters: 47,736
- Expected total parameters: 2,428,764
- Expected leaves: 724
- Global batch: 32
- Gradient accumulation: 4
- Effective batch: 128
- RegMask: 8% frame / 8% joint / 3% part
- EMA: 0.995
- XSUB train: 63,026
- XSUB validation: 50,919
- Epochs: 40
- Microsteps per epoch: 1,970
- Total microsteps: 78,800
- Optimizer schedule steps: 19,700

The assembled training source is protected by SHA256:

`f1b3e79e7499adcc09bec2d6d87ed3f68b48d2c9c18813767b053c3179f2dbb1`

The source is stored in small exact chunks because the experiment is a self-contained historical all-in-one script. `run_kaggle.py` concatenates the chunks, verifies the checksum, writes the complete source into `/kaggle/working`, and launches it in a fresh Python process.

## Kaggle

Use a TPU v5e-8 notebook. Attach a Kaggle dataset containing `ntu120_3danno.pkl`; the experiment scans `/kaggle/input` for that filename.

```python
!git clone -q -b experiment/xstream-ms-t32 https://github.com/rombaldivia/NestSAR.git
%cd NestSAR
!python -u experiments/xstream_ms_t32_xsub_e40/run_kaggle.py
```

The training output is written to:

`/kaggle/working/NestSAR_HOPE_XSTREAM_MS_T32_D128_XSUB_E40`

The run saves `history.json`, `best_ema.msgpack`, `last_weights.msgpack`, `last_stagewise_state.pkl`, and `result.json`.

## Validation status

The generated all-in-one source has passed local Python syntax/static assembly checks. This new XStream-MS architecture has **not yet been runtime-validated on a TPU**. The script therefore performs hard guards on parameter count, leaf count, optimizer-tier counts, gradient count, TPU device count, finite updates, and gradient replication before committing to the full training run. If any guard fails, stop and inspect the error rather than bypassing it.

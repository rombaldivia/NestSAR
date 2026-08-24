# Attention-Lite FMI v1 — from zero

This trainer initializes the full Attention-Lite + native Fine Motion Injection model randomly and trains all parameters on NTU120. No seed-42 checkpoint weights are loaded; the seed-42 generated source is used only to reconstruct the validated Attention-Lite architecture definition. The FMI injector enhances the existing joint-motion tensor before the original NestSAR hierarchy, preserving a single-trunk architecture. Dual NVIDIA T4 training uses `jax.pmap` and tqdm batch progress.

Example:

```bash
python experiments/attention_lite_fmi_v1/train_fmi_from_zero_dual_t4.py --protocol xsub --epochs 60 --batch 64 --eval-batch 128 --hidden-dim 64 --seed 42
```

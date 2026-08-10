# -*- coding: utf-8 -*-
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import dataclasses
import inspect

# IMPORTANTE: cargar primero el wrapper. Él generaliza SMS antes de que el
# proyecto importe/cachée nestsar_sms_s1c_v2.
import nestsar_m4_geom_h4_jointfirst_wide128_v4 as exp

import jax
import jax.numpy as jnp
import numpy as np
import nestsar as ns
import nestsar_sms_s1c_v2 as sms

BASELINE_PARAMS = 2_337_988
BASELINE_GFLOPS = 0.211700416
TENPART_GFLOPS = 0.229786624


def tree_numel(tree):
    return int(sum(
        np.prod(np.asarray(x).shape, dtype=np.int64)
        for x in jax.tree_util.tree_leaves(tree)
    ))


def get_flops(compiled):
    raw = compiled.cost_analysis()
    if isinstance(raw, dict):
        return float(raw.get("flops", 0.0))
    if isinstance(raw, (list, tuple)):
        return float(sum(
            float(x.get("flops", 0.0))
            for x in raw if isinstance(x, dict)
        ))
    return 0.0


sms_source = inspect.getsource(sms.SpatialMemorySweep)
if "Se esperaban 5 partes" in sms_source or "parts != 5" in sms_source:
    raise RuntimeError(
        "El SpatialMemorySweep cargado todavía es la versión P=5. "
        "No continuar para evitar medir un modelo incorrecto."
    )

print("[SMS] módulo cargado:", sms.__file__)
print("[SMS] P dinámico verificado en el objeto Python cargado.")

ns.CFG = dataclasses.replace(
    ns.CFG,
    frames=64,
    persons=2,
    joints=25,
    coords=3,
    num_classes=120,
    model_dim=128,
    memory_dim=64,
    dropout=0.22,
    batch_size=16,
    grad_accum_steps=8,
    eval_batch_size=32,
    epochs=60,
    patience=15,
    learning_rate=1e-3,
    weight_decay=0.05,
    warmup_fraction=0.10,
    label_smoothing=0.10,
    grad_clip=1.0,
    predictive_loss_weight=0.10,
    initial_eta=0.10,
    initial_alpha=0.95,
    frame_blocks=2,
    chunk_blocks=2,
    clip_blocks=2,
    controller_blocks=2,
    chunk_size=4,
    clip_size=8,
    controller_rank=32,
)

print("[1/3] Construyendo JointFirst-Wide128 v4...")
model = ns.build_model(exp.MODEL_ID)
x = jnp.zeros((1, 64, 150), dtype=jnp.float32)
rng = jax.random.PRNGKey(128)

variables = model.init({"params": rng, "dropout": rng}, x, training=False)
params = variables["params"]
count = tree_numel(params)

print(f"Parámetros:             {count:,}")
print(f"Delta vs M4G-H4 5Part: {count - BASELINE_PARAMS:+,}")


def logits_fn(p, xx):
    return model.apply({"params": p}, xx, training=False)["logits"]


print("[2/3] Lowering...")
lowered = jax.jit(logits_fn).lower(params, x)

print("[3/3] Compilando scan normal...")
compiled = lowered.compile()
gflops = get_flops(compiled) / 1e9

print()
print("=" * 96)
print("M4G-H4 JOINTFIRST-WIDE128 v4 — XLA FAST")
print("=" * 96)
print(f"GFLOPs:                 {gflops:.9f}")
print(f"Baseline 5Part:          {BASELINE_GFLOPS:.9f}")
print(f"10Part anterior:         {TENPART_GFLOPS:.9f}")
print(f"Delta vs 5Part:          {gflops - BASELINE_GFLOPS:+.9f} G")
print(f"Incremento vs 5Part:     {(gflops / BASELINE_GFLOPS - 1.0) * 100:.2f}%")
print(f"Delta vs 10Part:         {gflops - TENPART_GFLOPS:+.9f} G")
print("=" * 96)

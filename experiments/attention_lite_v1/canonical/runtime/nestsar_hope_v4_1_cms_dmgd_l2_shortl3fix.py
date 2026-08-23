#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
NestSAR-HOPE v4.1 — 16f CMS + DMGD-L2 + Short-L3 Fix
====================================

Research adaptation combining two independently verified community ideas:

1) obekt/HOPE-nested-learning
   - real multi-frequency outer optimization
   - gradients buffered/averaged between update periods

2) erikl2/nested-learning
   - stable normalized self-modification
   - nested/DMGD internal gradient memory
   - L2-regression-style memory update

This remains NestSAR, not a verbatim copy of either language model:
- 4 skeleton streams
- SASM spatial associative memory
- repaired L3 state path
- bounded K/V/Q/eta/alpha/main-memory self-reference
- NO softmax attention, Transformer, GCN/GNN, CNN/TCN
- RegMask + EMA retained

Outer continuum defaults (optimizer-update units):
    FAST/L1          period 1
    MEDIUM/L2        period 2
    SLOW/L3          period 4
    CONSOLIDATE/L4   period 8

All non-L1/L2/L3/L4 parameters (stem, SASM, fusion, classifiers, etc.)
remain FAST so discriminative heads do not become under-trained.

DMGD-L2 is deliberately lightweight: instead of a large optimizer MLP per
parameter bucket, every parameter leaf owns a diagonal L2 predictor in optimizer
state. It predicts gradient momentum and is itself updated by a normalized
delta rule. This preserves the core nested-optimizer idea with tiny overhead and
ZERO inference parameters.
"""

import os
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import traverse_util
from flax.core import FrozenDict, freeze

import nestsar as ns
import nestsar_m4_regmask_ema_v3_safe as reg
import nestsar_hope_fullselfref_v3_3_shortl3fix as v3

MODEL_ID = "nestsar_hope_v4_1_cms_dmgd_l2_shortl3fix"
MODEL_MODE = "NestSAR_HOPE_v4_1_CMS_DMGD_L2_ShortL3Fix"
EXPECTED_PARAMS = v3.EXPECTED_PARAMS

CMS_P1 = int(os.environ.get("NESTSAR_CMS_PERIOD_L1", "1"))
CMS_P2 = int(os.environ.get("NESTSAR_CMS_PERIOD_L2", "2"))
CMS_P3 = int(os.environ.get("NESTSAR_CMS_PERIOD_L3", "4"))
CMS_P4 = int(os.environ.get("NESTSAR_CMS_PERIOD_L4", "8"))

DMGD_MOMENTUM = float(os.environ.get("NESTSAR_DMGD_MOMENTUM", "0.90"))
DMGD_MEMORY_LR = float(os.environ.get("NESTSAR_DMGD_MEMORY_LR", "0.01"))
DMGD_MIX = float(os.environ.get("NESTSAR_DMGD_MIX", "0.10"))
DMGD_PROJECTION_CAP = float(os.environ.get("NESTSAR_DMGD_PROJECTION_CAP", "2.0"))
DMGD_EPS = float(os.environ.get("NESTSAR_DMGD_EPS", "1e-6"))

PERIODS = {
    "fast": CMS_P1,
    "medium": CMS_P2,
    "slow": CMS_P3,
    "consolidate": CMS_P4,
}

if sorted(PERIODS.values()) != list(PERIODS.values()):
    raise ValueError(f"CMS periods must be non-decreasing: {PERIODS}")
if min(PERIODS.values()) < 1:
    raise ValueError(f"CMS periods must be >=1: {PERIODS}")
if not (0.0 <= DMGD_MOMENTUM < 1.0):
    raise ValueError("NESTSAR_DMGD_MOMENTUM must be in [0,1)")
if not (0.0 <= DMGD_MIX <= 0.5):
    raise ValueError("NESTSAR_DMGD_MIX must be in [0,0.5]")
if DMGD_MEMORY_LR <= 0.0 or DMGD_PROJECTION_CAP <= 0.0:
    raise ValueError("DMGD memory LR/cap must be >0")

class DMGDL2DiagState(NamedTuple):
    momentum: Any
    projection: Any
    count: jax.Array

def dmgd_l2_diag() -> optax.GradientTransformation:
    def init_fn(params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return DMGDL2DiagState(
            momentum=zeros,
            projection=zeros,
            count=jnp.asarray(0, dtype=jnp.int32),
        )

    def update_fn(updates, state, params=None):
        del params
        new_momentum = jax.tree_util.tree_map(
            lambda m, g: DMGD_MOMENTUM * m + (1.0 - DMGD_MOMENTUM) * g,
            state.momentum, updates,
        )
        prediction = jax.tree_util.tree_map(
            lambda p, g: jnp.tanh(p) * g,
            state.projection, updates,
        )
        new_projection = jax.tree_util.tree_map(
            lambda p, g, pred, target: jnp.clip(
                p - DMGD_MEMORY_LR
                * (pred - jax.lax.stop_gradient(target)) * g
                / (jnp.square(g) + DMGD_EPS),
                -DMGD_PROJECTION_CAP, DMGD_PROJECTION_CAP,
            ),
            state.projection, updates, prediction, new_momentum,
        )
        processed = jax.tree_util.tree_map(
            lambda g, pred: g + DMGD_MIX * pred,
            updates, prediction,
        )
        return processed, DMGDL2DiagState(
            momentum=new_momentum,
            projection=new_projection,
            count=state.count + jnp.asarray(1, dtype=jnp.int32),
        )
    return optax.GradientTransformation(init_fn, update_fn)

def _path_to_tier(path_tuple) -> str:
    text = "/".join(str(x) for x in path_tuple).lower()
    if "l4_slow_controller" in text:
        return "consolidate"
    if "l3_clip_memory" in text:
        return "slow"
    if "l2_chunk_memory" in text:
        return "medium"
    if "l1_frame_memory" in text:
        return "fast"
    return "fast"

def make_tier_labels(params):
    flat = traverse_util.flatten_dict(params)
    labeled = {path: _path_to_tier(path) for path in flat}
    tree = traverse_util.unflatten_dict(labeled)
    if isinstance(params, FrozenDict):
        tree = freeze(tree)
    return tree

def tier_parameter_counts(params):
    flat = traverse_util.flatten_dict(params)
    out = {name: 0 for name in PERIODS}
    leaves = {name: 0 for name in PERIODS}
    for path, value in flat.items():
        tier = _path_to_tier(path)
        out[tier] += int(value.size)
        leaves[tier] += 1
    return out, leaves

def _scaled_schedule(total_steps: int, period: int):
    base = ns.make_schedule(total_steps)
    max_index = max(0, int(total_steps) - 1)
    def schedule(local_count):
        global_like = jnp.minimum(local_count * period, max_index)
        return base(global_like)
    return schedule

def create_state(rng: jax.Array, model: nn.Module, total_steps: int):
    dummy = jnp.zeros(
        (2, ns.CFG.frames, ns.CFG.persons * ns.CFG.joints * ns.CFG.coords),
        dtype=jnp.float32,
    )
    variables = model.init({"params": rng, "dropout": rng}, dummy, training=True)
    params = variables["params"]
    counts, leaves = tier_parameter_counts(params)
    print("=" * 112)
    print("OUTER CONTINUUM PARAMETER MAP")
    for tier in ("fast", "medium", "slow", "consolidate"):
        print(
            f"{tier:12s} period={PERIODS[tier]:2d} | params={counts[tier]:,} | leaves={leaves[tier]} | "
            f"gradient window≈{ns.CFG.batch_size * ns.CFG.grad_accum_steps * PERIODS[tier]} samples"
        )
    print("=" * 112)

    labels = make_tier_labels(params)
    transforms = {}
    for tier, period in PERIODS.items():
        inner = optax.chain(
            dmgd_l2_diag(),
            optax.clip_by_global_norm(ns.CFG.grad_clip),
            optax.adamw(
                learning_rate=_scaled_schedule(total_steps, period),
                weight_decay=ns.CFG.weight_decay,
            ),
        )
        transforms[tier] = optax.MultiSteps(
            inner,
            every_k_schedule=max(1, ns.CFG.grad_accum_steps * period),
            use_grad_mean=True,
        )

    optimizer = optax.multi_transform(transforms, labels)
    base = ns.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return reg.EMAState(
        step=base.step, apply_fn=base.apply_fn, params=base.params,
        tx=base.tx, opt_state=base.opt_state, ema_params=base.params,
    )

_BASE_BUILD_MODEL = ns.build_model
_BASE_BUILD_STEPS = ns.build_steps
ns.MODEL_ALIASES[MODEL_ID] = MODEL_MODE

def build_model(model_id: str):
    if model_id == MODEL_ID:
        return v3.build_model(v3.MODEL_ID)
    return _BASE_BUILD_MODEL(model_id)

def build_steps(model, model_id: str):
    if model_id == MODEL_ID:
        return v3.build_steps(model, v3.MODEL_ID)
    return _BASE_BUILD_STEPS(model, model_id)

ns.create_state = create_state
ns.build_model = build_model
ns.build_steps = build_steps
ns.__file__ = __file__

print("=" * 120)
print("NESTSAR-HOPE v4.1 — FULL SELF-REFERENCE + OUTER CMS + DMGD-L2 + SHORT-L3 FIX")
print("=" * 120)
print("Inference model:       v3.3 bounded self-reference + causal short-L3 post-write read")
print("erikl2 adaptation:     stable normalized self-modification + diagonal L2 gradient memory")
print("obekt adaptation:      buffered multi-frequency outer updates")
print(f"Short-L3 fix:          post-write read for T<=2, blend={v3.SHORT_L3_POSTWRITE_BLEND}")
print(f"CMS periods:            L1={CMS_P1} | L2={CMS_P2} | L3={CMS_P3} | L4={CMS_P4}")
print(f"DMGD momentum:          {DMGD_MOMENTUM}")
print(f"DMGD memory LR:         {DMGD_MEMORY_LR}")
print(f"DMGD outer mix:         {DMGD_MIX}")
print(f"DMGD projection cap:    {DMGD_PROJECTION_CAP}")
print("Softmax attention:      NONE")
print("GCN/GNN/CNN/TCN:        NONE")
print("Inference extra params: ZERO")
print(f"Expected model params:  {EXPECTED_PARAMS:,}")
print("Recommended first run:  3-epoch probe; then 40 epochs if healthy")
print("GFLOPs:                 MEASURE before paper claims")
print("=" * 120)

if __name__ == "__main__":
    raise SystemExit(ns.main())

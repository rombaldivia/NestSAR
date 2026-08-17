#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
NestSAR-HOPE-Fidelity — universal JAX runtime
=============================================
Runs the generated trainer on one GPU, multiple GPUs, or a TPU slice.
For >1 device, state is replicated and the leading batch axis is sharded over
one data mesh using NamedSharding; the existing jax.jit step compiles SPMD.
"""

import os
os.environ.setdefault("JAX_THREEFRY_PARTITIONABLE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import nestsar as ns
import nestsar_m4_regmask_ema_v3_safe as reg
import nestsar_hope_fidelity_d128_v1_train as fidelity

EXPECTED_BACKEND = os.environ.get("NESTSAR_EXPECTED_BACKEND", "auto").lower()
REQUESTED_DEVICE_COUNT = int(os.environ.get("NESTSAR_DEVICE_COUNT", "0"))
SPMD_MODE = os.environ.get("NESTSAR_SPMD", "auto").lower()

if EXPECTED_BACKEND not in ("auto", "gpu", "tpu"):
    raise ValueError(f"Invalid NESTSAR_EXPECTED_BACKEND={EXPECTED_BACKEND!r}")
if SPMD_MODE not in ("auto", "on", "off"):
    raise ValueError(f"Invalid NESTSAR_SPMD={SPMD_MODE!r}")

BACKEND = jax.default_backend()
VISIBLE_DEVICES = list(jax.devices())
if EXPECTED_BACKEND != "auto" and BACKEND != EXPECTED_BACKEND:
    raise RuntimeError(
        f"Requested backend={EXPECTED_BACKEND}, got {BACKEND}; devices={VISIBLE_DEVICES}"
    )
if REQUESTED_DEVICE_COUNT > 0:
    if len(VISIBLE_DEVICES) < REQUESTED_DEVICE_COUNT:
        raise RuntimeError(
            f"Requested {REQUESTED_DEVICE_COUNT} devices, but JAX sees "
            f"{len(VISIBLE_DEVICES)}: {VISIBLE_DEVICES}"
        )
    DEVICES = VISIBLE_DEVICES[:REQUESTED_DEVICE_COUNT]
else:
    DEVICES = VISIBLE_DEVICES
if not DEVICES:
    raise RuntimeError("JAX sees no accelerator devices")

USE_SPMD = len(DEVICES) > 1 if SPMD_MODE == "auto" else SPMD_MODE == "on"

reg.EMA_DECAY = float(os.environ.get("NESTSAR_EMA_DECAY", str(reg.EMA_DECAY)))
reg.FRAME_MASK_PROB = float(os.environ.get("NESTSAR_FRAME_MASK_PROB", str(reg.FRAME_MASK_PROB)))
reg.JOINT_MASK_PROB = float(os.environ.get("NESTSAR_JOINT_MASK_PROB", str(reg.JOINT_MASK_PROB)))
reg.PART_MASK_PROB = float(os.environ.get("NESTSAR_PART_MASK_PROB", str(reg.PART_MASK_PROB)))

_BASE_CREATE_STATE = ns.create_state
_BASE_BUILD_STEPS = ns.build_steps
_BASE_BATCH_ITERATOR = ns.batch_iterator
MESH = None
REPLICATED = None


def _is_array_leaf(x) -> bool:
    return isinstance(x, (jax.Array, np.ndarray)) or (hasattr(x, "shape") and hasattr(x, "dtype"))


def _replicate_tree(tree):
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, REPLICATED) if _is_array_leaf(x) else x,
        tree,
    )


def _data_sharding_for(x):
    ndim = int(getattr(x, "ndim", np.ndim(x)))
    if ndim <= 0:
        return REPLICATED
    return NamedSharding(MESH, P("data", *([None] * (ndim - 1))))


def _place_batch(x):
    ndim = int(getattr(x, "ndim", np.ndim(x)))
    n = int(x.shape[0]) if ndim > 0 else 0
    if n >= len(DEVICES) and n % len(DEVICES) == 0:
        return jax.device_put(x, _data_sharding_for(x))
    return jax.device_put(x, REPLICATED)


def _spmd_batch_iterator(*args, **kwargs):
    for batch_x, batch_y in _BASE_BATCH_ITERATOR(*args, **kwargs):
        yield _place_batch(batch_x), _place_batch(batch_y)


def _spmd_create_state(rng, model, total_steps):
    state = _BASE_CREATE_STATE(rng, model, total_steps)
    state = _replicate_tree(state)
    leaves = jax.tree_util.tree_leaves(state.params)
    if not leaves:
        raise RuntimeError("No parameter leaves after initialization")
    first = leaves[0]
    if not getattr(first.sharding, "is_fully_replicated", False):
        raise RuntimeError(f"Parameter replication failed: {first.sharding}")
    print("=" * 118)
    print("NESTSAR UNIVERSAL SPMD STATE")
    print("=" * 118)
    print(f"Backend:             {BACKEND}")
    print(f"Devices used:        {len(DEVICES)}/{len(VISIBLE_DEVICES)}")
    print(f"Mesh:                {MESH}")
    print(f"Parameter sharding:  {first.sharding}")
    print("State:               replicated")
    print("Batch:               sharded on leading data axis")
    print("Remainder batch:     replicated, not padded/dropped")
    print("=" * 118)
    return state


def _spmd_build_steps(model, model_id: str):
    base_train_step, base_eval_step = _BASE_BUILD_STEPS(model, model_id)
    def train_step(state, batch_x, batch_y, dropout_rng):
        batch_x = _place_batch(batch_x)
        batch_y = _place_batch(batch_y)
        dropout_rng = jax.device_put(dropout_rng, REPLICATED)
        return base_train_step(state, batch_x, batch_y, dropout_rng)
    def eval_step(params, batch_x, batch_y):
        batch_x = _place_batch(batch_x)
        batch_y = _place_batch(batch_y)
        return base_eval_step(params, batch_x, batch_y)
    return train_step, eval_step

if USE_SPMD:
    MESH = Mesh(np.asarray(DEVICES, dtype=object), ("data",))
    REPLICATED = NamedSharding(MESH, P())
    ns.create_state = _spmd_create_state
    ns.build_steps = _spmd_build_steps
    ns.batch_iterator = _spmd_batch_iterator

print("=" * 118)
print("NESTSAR-HOPE-FIDELITY — UNIVERSAL ACCELERATOR RUNTIME")
print("=" * 118)
print(f"JAX:                 {jax.__version__}")
print(f"Backend:             {BACKEND}")
print(f"Visible devices:     {len(VISIBLE_DEVICES)}")
for i, device in enumerate(VISIBLE_DEVICES):
    print(f"  [{i}] {device}")
print(f"Selected devices:    {len(DEVICES)}")
print(f"SPMD:                {'ON' if USE_SPMD else 'OFF'}")
print(f"Model ID:            {fidelity.MODEL_ID}")
print(f"Canonical params:    {fidelity.EXPECTED_PARAMS:,}")
print(f"EMA decay:           {reg.EMA_DECAY}")
print(f"RegMask:             frame={reg.FRAME_MASK_PROB:.3f} | joint={reg.JOINT_MASK_PROB:.3f} | part={reg.PART_MASK_PROB:.3f}")
print("Softmax attention:   NONE")
print("=" * 118)

if __name__ == "__main__":
    raise SystemExit(ns.main())

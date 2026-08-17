#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
NestSAR-HOPE-Fidelity D128 v1 — TPU v5e-8 SPMD wrapper
=======================================================

This module keeps the model, loss, RegMask, EMA, CMS periods, DMGD-L2 and
optimizer semantics from nestsar_hope_fidelity_d128_v1_train.py, but places:

  * model/optimizer/EMA state: replicated on all 8 TPU chips
  * normal train/eval batches: sharded on the leading batch axis across 8 chips

The existing jax.jit train/eval functions then infer their input shardings from
these NamedSharding inputs and compile one SPMD program across the whole mesh.
Global batch reductions therefore remain global reductions, while each TPU chip
only materializes its local slice of the batch.

Default TPU recipe used by the launcher:
  global physical batch = 128
  TPU chips             = 8
  local batch/chip      = 16
  grad accumulation     = 1
  effective batch       = 128

This preserves the v4.1 outer sample windows:
  fast          128 samples
  medium        256 samples
  slow          512 samples
  consolidate  1024 samples

Tiny remainder batches that are not divisible by 8 are replicated across the
mesh rather than padded or discarded, preserving exact sample counts.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "tpu")
os.environ.setdefault("JAX_THREEFRY_PARTITIONABLE", "true")

import numpy as np
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import nestsar as ns

# Import the generated fidelity trainer first. It installs the exact model,
# RegMask/EMA steps and four-tier CMS/DMGD optimizer into `nestsar`.
import nestsar_hope_fidelity_d128_v1_train as fidelity

EXPECTED_DEVICES = 8

if jax.default_backend() != "tpu":
    raise RuntimeError(
        f"TPU SPMD wrapper requires backend=tpu, got {jax.default_backend()!r}"
    )
if jax.device_count() < EXPECTED_DEVICES:
    raise RuntimeError(
        f"TPU SPMD wrapper requires >=8 devices, got {jax.device_count()}"
    )

TPU_DEVICES = np.asarray(jax.devices()[:EXPECTED_DEVICES], dtype=object)
TPU_MESH = Mesh(TPU_DEVICES, ("data",))
REPLICATED = NamedSharding(TPU_MESH, P())
DATA_1D = NamedSharding(TPU_MESH, P("data"))
DATA_2D = NamedSharding(TPU_MESH, P("data", None))
DATA_3D = NamedSharding(TPU_MESH, P("data", None, None))
DATA_4D = NamedSharding(TPU_MESH, P("data", None, None, None))

_BASE_CREATE_STATE = ns.create_state
_BASE_BUILD_STEPS = ns.build_steps
_BASE_BATCH_ITERATOR = ns.batch_iterator


def _is_array_leaf(x) -> bool:
    return isinstance(x, (jax.Array, np.ndarray)) or (
        hasattr(x, "shape") and hasattr(x, "dtype")
    )


def _replicate_tree(tree):
    """Place every array leaf of a TrainState/optimizer pytree on all 8 chips."""
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, REPLICATED) if _is_array_leaf(x) else x,
        tree,
    )


def _data_sharding_for(x):
    ndim = int(getattr(x, "ndim", np.ndim(x)))
    if ndim == 1:
        return DATA_1D
    if ndim == 2:
        return DATA_2D
    if ndim == 3:
        return DATA_3D
    if ndim == 4:
        return DATA_4D
    if ndim <= 0:
        return REPLICATED
    return NamedSharding(TPU_MESH, P("data", *([None] * (ndim - 1))))


def _place_batch(x):
    """
    Shard a normal batch over all TPU chips. For a tiny last batch whose first
    dimension cannot be divided by 8, replicate it instead of padding/dropping.
    """
    n = int(x.shape[0]) if getattr(x, "ndim", 0) > 0 else 0
    if n >= EXPECTED_DEVICES and n % EXPECTED_DEVICES == 0:
        return jax.device_put(x, _data_sharding_for(x))
    return jax.device_put(x, REPLICATED)


def batch_iterator(*args, **kwargs):
    """Direct host -> TPU sharded placement before the main loop calls jnp.asarray."""
    for batch_x, batch_y in _BASE_BATCH_ITERATOR(*args, **kwargs):
        yield _place_batch(batch_x), _place_batch(batch_y)


def create_state(rng, model, total_steps):
    state = _BASE_CREATE_STATE(rng, model, total_steps)
    state = _replicate_tree(state)

    param_leaves = jax.tree_util.tree_leaves(state.params)
    if not param_leaves:
        raise RuntimeError("No parameter leaves after TPU replication")
    first = param_leaves[0]
    if not getattr(first.sharding, "is_fully_replicated", False):
        raise RuntimeError(f"Parameter replication failed: {first.sharding}")

    print("=" * 118)
    print("TPU v5e-8 SPMD STATE PLACEMENT")
    print("=" * 118)
    print(f"Mesh:               {TPU_MESH}")
    print(f"TPU chips:          {EXPECTED_DEVICES}")
    print(f"Parameter sharding: {first.sharding}")
    print("State placement:    REPLICATED on 8/8 chips")
    print("Batch placement:    SHARDED across data axis")
    print("Remainder batches:  REPLICATED, never padded/dropped")
    print("=" * 118)
    return state


def build_steps(model, model_id: str):
    base_train_step, base_eval_step = _BASE_BUILD_STEPS(model, model_id)

    def train_step(state, batch_x, batch_y, dropout_rng):
        # In the normal loop these are already sharded by batch_iterator. The
        # explicit placement also handles smoke tests / direct calls safely.
        batch_x = _place_batch(batch_x)
        batch_y = _place_batch(batch_y)
        dropout_rng = jax.device_put(dropout_rng, REPLICATED)
        return base_train_step(state, batch_x, batch_y, dropout_rng)

    def eval_step(params, batch_x, batch_y):
        batch_x = _place_batch(batch_x)
        batch_y = _place_batch(batch_y)
        return base_eval_step(params, batch_x, batch_y)

    return train_step, eval_step


def print_batch_layout(global_batch: int, eval_batch: int) -> None:
    if global_batch % EXPECTED_DEVICES != 0:
        raise ValueError(
            f"Training batch {global_batch} must be divisible by {EXPECTED_DEVICES}"
        )
    if eval_batch % EXPECTED_DEVICES != 0:
        raise ValueError(
            f"Eval batch {eval_batch} must be divisible by {EXPECTED_DEVICES}"
        )
    print("=" * 118)
    print("TPU v5e-8 DATA-PARALLEL LAYOUT")
    print("=" * 118)
    print(f"Global train batch: {global_batch}")
    print(f"Per-chip train:     {global_batch // EXPECTED_DEVICES}")
    print(f"Global eval batch:  {eval_batch}")
    print(f"Per-chip eval:      {eval_batch // EXPECTED_DEVICES}")
    print("Parameters:         replicated")
    print("Gradients:          global SPMD reduction through sharded batch loss")
    print("TPU utilization:    8/8 chips for normal full batches")
    print("=" * 118)


# Install only execution/sharding patches. Architecture and optimizer semantics
# remain those installed by the generated fidelity trainer.
ns.create_state = create_state
ns.build_steps = build_steps
ns.batch_iterator = batch_iterator
ns.__file__ = __file__

print("=" * 118)
print("NESTSAR-HOPE-FIDELITY D128 v1 — TRUE TPU v5e-8 SPMD")
print("=" * 118)
print(f"Model ID:           {fidelity.MODEL_ID}")
print(f"Expected params:    {fidelity.EXPECTED_PARAMS:,}")
print("Execution:          NamedSharding + jax.jit SPMD")
print("TPU chips used:     8/8 on normal batches")
print("Parameter layout:   replicated")
print("Batch layout:       sharded over mesh axis 'data'")
print("Softmax attention:  NONE")
print("=" * 118)


if __name__ == "__main__":
    print_batch_layout(ns.CFG.batch_size, ns.CFG.eval_batch_size)
    raise SystemExit(ns.main())

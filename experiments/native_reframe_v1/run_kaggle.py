#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kaggle launcher for NestSAR Native-Reframe v1.

Compatibility shim for modern JAX versions where ``jax.device_put_replicated``
was removed.  The training implementation still uses ``pmap`` with leading
replica axes, so we construct those replica axes explicitly and let ``pmap``
shard them over every visible local device.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp


def _replicate_for_pmap(tree, devices):
    ndev = len(devices)
    if ndev < 1:
        raise RuntimeError("No JAX devices visible")

    def replicate_leaf(x):
        x = jnp.asarray(x)
        return jnp.broadcast_to(x, (ndev,) + x.shape)

    return jax.tree_util.tree_map(replicate_leaf, tree)


# JAX >= current Kaggle removed device_put_replicated.  The base experiment
# calls it for TrainState and best parameters; provide an equivalent pmap input
# representation without relying on the removed API.
jax.device_put_replicated = _replicate_for_pmap

print(
    f"NESTSAR JAX REPLICATION COMPAT: PASS | "
    f"backend={jax.default_backend()} | visible_devices={len(jax.local_devices())}",
    flush=True,
)

TARGET = Path(__file__).with_name("train_kaggle.py")
runpy.run_path(str(TARGET), run_name="__main__")

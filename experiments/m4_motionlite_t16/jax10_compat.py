#!/usr/bin/env python3
from __future__ import annotations

"""JAX >=0.10 compatibility helpers for legacy pmap replication.

JAX removed ``jax.device_put_replicated`` from the public API.  The helper below
implements the documented drop-in behavior with ``jax.device_put`` and
``NamedSharding`` so the existing NestSAR pmap trainer can run unchanged.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def device_put_replicated_compat(tree, devices):
    devices = list(devices)
    if not devices:
        raise ValueError("devices must not be empty")
    mesh = Mesh(np.asarray(devices, dtype=object), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.stack([x] * len(devices)), sharding),
        tree,
    )


def install() -> None:
    """Install the removed API name only when the running JAX lacks it."""
    try:
        getattr(jax, "device_put_replicated")
        return
    except AttributeError:
        pass
    setattr(jax, "device_put_replicated", device_put_replicated_compat)
    print(
        f"JAX_COMPAT=PASS | JAX={jax.__version__} | installed device_put_replicated shim",
        flush=True,
    )

from __future__ import annotations

import jax
import jax.numpy as jnp

NUM_CLASSES = 120
DIM = 112

def l2_normalize(x, eps=1e-6):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)

def fused_representations(out):
    # Geometry should improve G4/descriptor features, not solve the loss by\n    # moving the already-audited fusion controller.\n    fusion = jax.lax.stop_gradient(out["fusion_weights"])\n    desc = jnp.einsum("bs,bsd->bd", fusion, out["descriptors"])
    chunks = out["chunk_states"]
    g4_stream = jnp.mean(chunks, axis=2)
    g4 = jnp.einsum("bs,bsd->bd", fusion, g4_stream)
    return l2_normalize(desc), l2_normalize(g4)

def local_geometry_terms(
    z,
    y,
    prototypes,
    counts,
    *,
    margin,
    rival_k,
    rival_temperature,
):
    z = l2_normalize(z)
    prototypes = l2_normalize(prototypes)

    sim = jnp.einsum("bd,ckd->bck", z, prototypes)
    valid = counts > 0

    valid_y = valid[y]
    sim_y = jnp.take_along_axis(sim, y[:, None, None], axis=1)[:, 0, :]
    pos = jnp.max(jnp.where(valid_y, sim_y, -1e9), axis=-1)
    pos_valid = jnp.any(valid_y, axis=-1)

    class_best = jnp.max(
        jnp.where(valid[None, :, :], sim, -1e9),
        axis=-1,
    )
    class_valid = jnp.any(valid, axis=-1)
    class_best = jnp.where(class_valid[None, :], class_best, -1e9)
    class_best = class_best.at[jnp.arange(z.shape[0]), y].set(-1e9)

    top = jax.lax.top_k(class_best, rival_k)[0]
    rival_valid = top[:, 0] > -1e8
    weights = jax.nn.softmax(top / rival_temperature, axis=-1)
    rival = jnp.sum(weights * top, axis=-1)

    valid_sample = pos_valid & rival_valid
    gap = pos - rival
    raw_hinge = margin - gap

    hardness = jax.lax.stop_gradient(
        jnp.clip(raw_hinge / margin, 0.0, 1.0)
    )
    active = valid_sample & (raw_hinge > 0)
    loss = jnp.where(
        active,
        hardness * jnp.maximum(raw_hinge, 0.0),
        0.0,
    )

    safe_pos = jnp.where(valid_sample, pos, 0.0)
    safe_rival = jnp.where(valid_sample, rival, 0.0)
    safe_gap = jnp.where(valid_sample, gap, 0.0)
    return (
        loss,
        active.astype(jnp.float32),
        safe_pos,
        safe_rival,
        safe_gap,
    )

def update_subcenters(
    prototypes,
    counts,
    z,
    y,
    mask,
    momentum,
):
    z = jax.lax.stop_gradient(l2_normalize(z))
    y = jax.lax.stop_gradient(y)
    mask = jax.lax.stop_gradient(mask)

    classes, k, dim = prototypes.shape
    if classes != NUM_CLASSES or dim != DIM:
        raise ValueError(f"Unexpected prototype bank shape: {prototypes.shape}")

    py = prototypes[y]
    cy = counts[y]
    sim = jnp.einsum("bd,bkd->bk", z, py)

    valid = cy > 0
    all_valid = jnp.all(valid, axis=-1)
    nearest = jnp.argmax(jnp.where(valid, sim, -1e9), axis=-1)
    first_empty = jnp.argmax((~valid).astype(jnp.int32), axis=-1)
    assignment = jnp.where(all_valid, nearest, first_empty)

    flat_index = y * k + assignment
    assignment_oh = jax.nn.one_hot(
        flat_index,
        classes * k,
        dtype=z.dtype,
    )
    assignment_oh = assignment_oh * mask[:, None]

    sums = assignment_oh.T @ z
    n = jnp.sum(assignment_oh, axis=0)

    old = prototypes.reshape(classes * k, dim)
    old_counts = counts.reshape(classes * k)
    means = sums / jnp.maximum(n[:, None], 1.0)

    initialized = old_counts > 0
    candidate = jnp.where(
        initialized[:, None],
        momentum * old + (1.0 - momentum) * means,
        means,
    )
    candidate = l2_normalize(candidate)

    new = jnp.where((n > 0)[:, None], candidate, old)
    new_counts = old_counts + n
    return (
        new.reshape(classes, k, dim),
        new_counts.reshape(classes, k),
    )

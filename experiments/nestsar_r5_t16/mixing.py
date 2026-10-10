"""Temporal CutMix on the 16 segment tokens, done on the accelerator inside the train step.

A token row is one segment of the clip ([FEATURES] = 750 body + 192 hand features), so a span of segments
of one clip can be replaced by the same span of another clip in the batch. The label is mixed in proportion
to the replaced span. Both views (``x`` and ``xa``) get the same partner and span, so the consistency loss
still compares two views of the same mixed clip. With ``prob = 0`` the batch is returned unchanged
(``lam = 1``), which keeps the loss numerically identical to the unmixed one.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

MIN_SPAN, MAX_SPAN = 0.2, 0.7      # fraction of the 16 segments that is replaced


def mix_batch(key, x, xa, y, mask, prob):
    """Return (x, xa, y2, lam): ``lam`` is the weight of the original label, ``y2`` the partner label."""
    batch, frames = x.shape[0], x.shape[1]
    if prob <= 0:
        return x, xa, y, jnp.ones(batch, jnp.float32)
    k_perm, k_do, k_len, k_start = jax.random.split(key, 4)
    partner = jax.random.permutation(k_perm, batch)
    do = ((jax.random.uniform(k_do, (batch,)) < prob) & (mask > 0) & (mask[partner] > 0)
          & (partner != jnp.arange(batch)))
    frac = jax.random.uniform(k_len, (batch,), minval=MIN_SPAN, maxval=MAX_SPAN)
    span = jnp.clip(jnp.round(frac * frames), 1, frames - 1).astype(jnp.int32)
    start = jnp.floor(jax.random.uniform(k_start, (batch,)) * (frames - span + 1)).astype(jnp.int32)
    t = jnp.arange(frames)[None, :]
    cut = (t >= start[:, None]) & (t < (start + span)[:, None]) & do[:, None]          # [B, frames]
    take = cut[..., None]
    lam = 1.0 - cut.sum(axis=1).astype(jnp.float32) / frames
    return (jnp.where(take, x[partner], x), jnp.where(take, xa[partner], xa),
            jnp.where(do, y[partner], y), lam)

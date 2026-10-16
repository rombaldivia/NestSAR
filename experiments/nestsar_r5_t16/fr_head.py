"""FR-Head (Zhou et al., CVPR 2023, "Learning Discriminative Representations for Skeleton Based Action
Recognition"), training-only, ported from the official ``RenovateNet`` (github.com/zhysora/FR-Head,
model/lib.py, version "V0", use_p_map=True).

Per refined feature z [B, C] (already through its cl_fc projection):
  * TP / FN / FP masks from the (detached) logits of the main classifier;
  * class memory f_mem: per class present among the TP samples, 0.9 * bank + 0.1 * TP centroid of the batch,
    else the bank (the bank is detached; the batch centroid keeps its gradient, as in the original);
  * FN / FP centroids per class from the batch (L1-normalised masks, gradient kept);
  * scores = cos(z, f_mem) with the true-class score moved by (1 - p_true) * alpha * (cos(z, FN) - 1) and
    by (1 - p_true) * alpha * (-cos(z, FP) - 1), each on classes that have FN / FP in the batch;
  * loss = CE(score_fn / tau, y) + CE(score_fp / tau, y), per sample.

Padding rows (mask 0) never enter a centroid, a TP mask or the bank. Nothing here is used at inference.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

LEVELS = ("low_s", "low_t", "mid_t", "high_t", "fin")
# FR-Head multi-level weights (low, mid, high, fin). Our low level has both the spatial and the temporal
# branch (the only stage that still has a joint axis), as the original ST_RenovateNet sums them.
LEVEL_OF = {"low_s": 0, "low_t": 0, "mid_t": 1, "high_t": 2, "fin": 3}


def _unit(a, axis):
    # sqrt(|a|^2 + tiny): same value as the original norm + 1e-12, but a finite (zero) gradient at a = 0
    # (classes without FN/FP in the batch have all-zero centroids).
    return a / (jnp.sqrt(jnp.sum(a * a, axis=axis, keepdims=True) + 1e-24) + 1e-12)


def renovate(z, y, logits, mask, bank, alpha=0.125, tau=0.125, momentum=0.9):
    """One FR branch. z [B,C], y [B] int, logits [B,K], mask [B] (1 = real sample), bank [C,K].

    Returns (per-sample loss [B], updated bank [C,K] (stop-gradient))."""
    k = bank.shape[1]
    logits = jax.lax.stop_gradient(logits)
    bank = jax.lax.stop_gradient(bank)
    m = mask.astype(z.dtype)[:, None]
    lbl = jax.nn.one_hot(y, k, dtype=z.dtype) * m
    pred = jax.nn.one_hot(jnp.argmax(logits, -1), k, dtype=z.dtype) * m
    prob = jax.nn.softmax(logits, -1).astype(z.dtype)
    tp = lbl * pred                      # (original also gates tp by prob > pred_threshold = 0: always true)
    fn = lbl - tp
    fp = pred - tp
    has_fn = (fn.sum(0) > 1e-8).astype(z.dtype)            # [K]
    has_fp = (fp.sum(0) > 1e-8).astype(z.dtype)

    l1 = lambda a: a / jnp.maximum(a.sum(0, keepdims=True), 1e-12)   # F.normalize(p=1, dim=0) on 0/1 masks
    f_fn = z.T @ l1(fn)                                     # [C,K]
    f_fp = z.T @ l1(fp)
    count = tp.sum(0, keepdims=True)                         # [1,K]
    f_tp = (z.T @ tp) / (count + 1e-12)
    keep = jnp.where(count > 1e-8, momentum, 1.0)
    f_mem = bank * keep + (1 - keep) * f_tp

    zn = _unit(z, 1)
    s_mem = zn @ _unit(f_mem, 0)                             # [B,K]
    p_map = (1 - prob) * lbl * alpha                         # only the true class column is nonzero
    s_fn = (zn @ _unit(f_fn, 0) - 1) * p_map * has_fn[None]
    s_fp = (-(zn @ _unit(f_fp, 0)) - 1) * p_map * has_fp[None]

    def ce(scores):
        logp = jax.nn.log_softmax(scores / tau, -1)
        return -jnp.take_along_axis(logp, y[:, None], axis=-1)[:, 0]

    loss = (ce(s_mem + s_fn) + ce(s_mem + s_fp)) * mask.astype(z.dtype)
    return loss, jax.lax.stop_gradient(f_mem)


def fr_loss(feats, y, logits, mask, banks, level_weights, alpha, tau, momentum):
    """Sum over the FR branches: sum_l w_level(l) * loss_l. feats/banks: dicts keyed by LEVELS.

    Returns (per-sample loss [B], new banks dict)."""
    total = jnp.zeros(y.shape, feats[LEVELS[0]].dtype)
    new = {}
    for name in LEVELS:
        loss, new[name] = renovate(feats[name], y, logits, mask, banks[name], alpha, tau, momentum)
        total = total + level_weights[LEVEL_OF[name]] * loss
    return total, new


def init_banks(key, hidden, num_classes):
    """torch.randn(h, K) per branch, as the original."""
    keys = jax.random.split(key, len(LEVELS))
    return {name: jax.random.normal(k, (hidden, num_classes), jnp.float32) for name, k in zip(LEVELS, keys)}

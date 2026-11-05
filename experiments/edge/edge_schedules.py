"""Deployment-only schedules of NestSAR's recurrences: same parameters, same arithmetic, different execution order.

BiGRUSweep (training code) advances the forward and backward GRUs in ONE scan with stacked weights, which halves the
sequential chain on a GPU. On a small CPU that is the wrong trade: each step then reads both directions' recurrent
matrices. For the temporal core (hidden 176) that is 2 x (176x352 + 176x176) fp32 = 743 KB per step, more than the
512 KB L2 of a Raspberry Pi Zero 2 W (Cortex-A53), so every step streams weights from DRAM. SplitBiGRUSweep runs
the forward scan and then the backward scan; each scan touches one direction's 372 KB, which stays cache-resident.

The parameter tree is identical (same module names, shapes and initialisers), so trained checkpoints load unchanged,
and the outputs equal the stacked version up to float32 round-off (checked in export_xla.py).

  with split_bigru():  ...build / trace NestSAR...
"""
import contextlib

import flax.linen as nn
import jax
import jax.numpy as jnp

from experiments.nestsar_r5_t16 import model as model_mod

_STACKED = model_mod.BiGRUSweep


class SplitBiGRUSweep(nn.Module):
    hidden: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.hidden
        x_fwd = nn.Dense(3 * h, name="input_fwd")(x)
        x_bwd = nn.Dense(3 * h, name="input_bwd")(x)[:, ::-1]
        init = nn.initializers.xavier_uniform()
        w_zr = self.param("hidden_zr", lambda k, s: jnp.stack([init(kk, s[1:]) for kk in jax.random.split(k)]),
                          (2, h, 2 * h))
        w_c = self.param("hidden_c", lambda k, s: jnp.stack([init(kk, s[1:]) for kk in jax.random.split(k)]),
                         (2, h, h))

        def run(xs, wzr, wc):                                                    # xs [N, L, 3h] -> [N, L, h]
            def step(state, xt):
                zr = jax.nn.sigmoid(xt[:, :2 * h] + state @ wzr)
                z, r = zr[:, :h], zr[:, h:]
                cand = jnp.tanh(xt[:, 2 * h:] + (r * state) @ wc)
                state = (1.0 - z) * state + z * cand
                return state, state
            _, ys = jax.lax.scan(step, jnp.zeros((xs.shape[0], h), xs.dtype), jnp.swapaxes(xs, 0, 1))
            return jnp.swapaxes(ys, 0, 1)

        fwd = run(x_fwd, w_zr[0], w_c[0])
        bwd = run(x_bwd, w_zr[1], w_c[1])[:, ::-1]
        return jnp.concatenate([fwd, bwd], axis=-1)


@contextlib.contextmanager
def split_bigru():
    """Build/trace NestSAR with the split (cache-friendly) BiGRU schedule; restores the training class after."""
    model_mod.BiGRUSweep = SplitBiGRUSweep
    try:
        yield
    finally:
        model_mod.BiGRUSweep = _STACKED

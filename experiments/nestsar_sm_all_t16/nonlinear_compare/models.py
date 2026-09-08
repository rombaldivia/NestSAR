"""Diagnostic models; none of these layers is added to deployed NestSAR."""
import jax
import jax.numpy as jnp
from flax import linen as nn


class TokenMLP(nn.Module):
    width: int = 256
    dropout: float = .1

    @nn.compact
    def __call__(self, x, training=False):
        h = nn.Dense(self.width)(x.reshape(x.shape[0], -1))
        h = nn.gelu(nn.LayerNorm()(h))
        for _ in range(2):
            r = nn.Dense(self.width*2)(nn.LayerNorm()(h))
            r = nn.Dropout(self.dropout)(nn.gelu(r), deterministic=not training)
            r = nn.Dense(self.width)(r)
            h = h + nn.Dropout(self.dropout)(r, deterministic=not training)
        h = nn.Dropout(self.dropout)(nn.LayerNorm()(h), deterministic=not training)
        return nn.Dense(2)(h)


class MaskedGRU(nn.Module):
    width: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x, valid):
        wx = self.param("wx", nn.initializers.glorot_uniform(), (x.shape[-1], 3*self.width))
        wh = self.param("wh", nn.initializers.orthogonal(), (self.width, 3*self.width))
        bias = self.param("bias", nn.initializers.zeros, (3*self.width,))
        projected = x @ wx + bias
        if self.reverse:
            projected, valid = projected[:, ::-1], valid[:, ::-1]

        def step(h, item):
            xt, keep = item
            xr, xz, xn = jnp.split(xt, 3, -1)
            hr, hz, hn = jnp.split(h @ wh, 3, -1)
            reset, retain = jax.nn.sigmoid(xr+hr), jax.nn.sigmoid(xz+hz)
            candidate = jnp.tanh(xn + reset*hn)
            h = jnp.where(keep[:, None], retain*h + (1-retain)*candidate, h)
            return h, h

        _, values = jax.lax.scan(step, jnp.zeros((x.shape[0], self.width), x.dtype),
                                 (projected.swapaxes(0, 1), valid.swapaxes(0, 1)))
        values = values.swapaxes(0, 1)
        return values[:, ::-1] if self.reverse else values


class SkeletonGRU(nn.Module):
    """Identical parameter count/equations at 16 and 64 uniformly sampled frames."""
    width: int = 128
    dropout: float = .1

    @nn.compact
    def __call__(self, x, training=False):
        valid = jnp.any(x[..., 6] > .5, axis=(2, 3))
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        phase = jnp.linspace(0, 1, x.shape[1])[None, :, None]
        phase = jnp.broadcast_to(phase, (*x.shape[:2], 1))
        h = nn.Dense(self.width)(jnp.concatenate([flat, phase], -1))
        h = nn.gelu(nn.LayerNorm()(h)) * valid[..., None]
        for layer in range(2):
            h = nn.Dropout(self.dropout)(h, deterministic=not training)
            forward = MaskedGRU(self.width, name=f"forward_{layer}")(h, valid)
            backward = MaskedGRU(self.width, reverse=True, name=f"backward_{layer}")(h, valid)
            h = nn.LayerNorm()(jnp.concatenate([forward, backward], -1))
            h = h * valid[..., None]
        mean = h.sum(1) / jnp.maximum(valid.sum(1, keepdims=True), 1)
        maximum = jnp.max(jnp.where(valid[..., None], h, -1e6), axis=1)
        maximum = jnp.where(valid.any(1, keepdims=True), maximum, 0)
        h = nn.Dense(self.width*2)(jnp.concatenate([mean, maximum], -1))
        h = nn.Dropout(self.dropout)(nn.gelu(h), deterministic=not training)
        return nn.Dense(2)(h)


def make_model(arm, config, trial):
    if arm == "t16_mlp":
        return TokenMLP(config["mlp_width"], trial["dropout"])
    if arm in ("sequence16_gru", "sequence64_gru"):
        return SkeletonGRU(config["gru_width"], trial["dropout"])
    raise ValueError(f"Unknown arm {arm}")

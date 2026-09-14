"""Lightweight experiment identity; importing launcher settings must not load JAX."""
VERSION = "local-subspace-v1"
RANK = 8
MIXTURES = 2
RESIDUAL_SCALE = 0.1


def extra_parameters(spatial_dim=24, streams=4):
    return streams * (spatial_dim * RANK + 25 * MIXTURES + MIXTURES * RANK * spatial_dim)

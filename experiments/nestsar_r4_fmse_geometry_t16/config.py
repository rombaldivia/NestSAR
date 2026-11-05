from __future__ import annotations

from experiments.nestsar_sm_all_t16.streaming import launch as r4_launch

GEOMETRY_DEFAULTS = dict(
    geometry_desc_weight=0.05,
    geometry_g4_weight=0.015,
    geometry_margin=0.10,
    geometry_rival_k=3,
    geometry_rival_temperature=0.05,
    geometry_proto_momentum=0.99,
    geometry_start_epoch=4,
    geometry_full_epoch=14,
    geometry_subcenters=2,
)

def validate_config(config):
    config = dict(config or {})
    base_keys = set(r4_launch.DEFAULTS)
    geo_keys = set(GEOMETRY_DEFAULTS)
    unknown = set(config) - base_keys - geo_keys
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    base_input = {k: v for k, v in config.items() if k in base_keys}
    c = r4_launch.validate_config(base_input)
    c.update(GEOMETRY_DEFAULTS)
    c.update({k: v for k, v in config.items() if k in geo_keys})

    for k in ("geometry_desc_weight", "geometry_g4_weight", "geometry_margin"):
        if not isinstance(c[k], (int, float)) or c[k] < 0:
            raise ValueError(f"{k} must be nonnegative")
    if c["geometry_margin"] <= 0:
        raise ValueError("geometry_margin must be positive")
    if not 0 < c["geometry_rival_temperature"]:
        raise ValueError("geometry_rival_temperature must be positive")
    if not 0 <= c["geometry_proto_momentum"] < 1:
        raise ValueError("geometry_proto_momentum must be in [0,1)")
    if c["geometry_subcenters"] != 2:
        raise ValueError("LocalGeometry-v1 is fixed at exactly two subcenters per class")
    if not isinstance(c["geometry_rival_k"], int) or not 1 <= c["geometry_rival_k"] <= 8:
        raise ValueError("geometry_rival_k must be an integer in [1,8]")
    for k in ("geometry_start_epoch", "geometry_full_epoch"):
        if not isinstance(c[k], int) or c[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    if c["geometry_full_epoch"] <= c["geometry_start_epoch"]:
        raise ValueError("geometry_full_epoch must be after geometry_start_epoch")
    return c

def geometry_scale(config, epoch):
    start = config["geometry_start_epoch"]
    full = config["geometry_full_epoch"]
    return float(max(0.0, min(1.0, (epoch - start) / (full - start))))

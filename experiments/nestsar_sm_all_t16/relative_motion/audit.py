"""Audit all three interfaces with the same parameter tree and physical clip."""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from ..streaming.audit import audit_model
from ..streaming.worker import make_model, EXPECTED_PARAMS
from ..streaming.io_utils import atomic_json, read_json
from .. import preprocessing_corrected as p2
from . import preprocessing as pp


def run(config, output):
    raw = np.zeros((64, 2, 25, 3), np.float32)
    raw[:, 0] = np.random.default_rng(31).normal(0, .1, (64, 25, 3)) + [1, 2, 3]
    legacy, packed = jnp.asarray(p2.features(raw)[None]), jnp.asarray(pp.features(raw)[None])
    model = make_model(config)
    params = model.init(jax.random.PRNGKey(128), legacy, training=False)["params"]
    if sum(p.size for p in jax.tree.leaves(params)) != EXPECTED_PARAMS:
        raise RuntimeError("Unexpected parameter count")
    records = {}
    for mode in ("legacy", "proxy", "relative"):
        selected = model.clone(motion_path=mode)
        records[mode] = audit_model(selected, params, legacy if mode == "legacy" else packed)
        jax.clear_caches()
    records["scope"] = "Neural forward only; raw preprocessing cost depends on clip length"
    records["relative_minus_legacy_flops"] = records["relative"]["flops"] - records["legacy"]["flops"]
    atomic_json(output, records)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(read_json(args.config), args.output)

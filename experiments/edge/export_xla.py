"""Serialize the edge-benchmark models as compiled-ready XLA programs (jax.export, platform "cpu"), weights baked in.

The device then needs only `jax` (no Flax, no model code): bench_xla.py deserializes and runs them. SGN is run
through an exact JAX port (sgn_jax.py, verified against PyTorch), so every model uses the same compiler (XLA).

  python -m experiments.edge.export_xla --out edge_xla            # random-init weights (latency is weight-independent)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

NESTSAR = {"nestsar": "graph_mr", "nestsar_tattn": "graph_mr_tattn", "nestsar_tconv": "graph_mr_tconv",
           "nestsar_notemp": "graph_mr_notemp"}
SGN = {"sgn40": 0.3125, "sgn_full": 1.0}


def main(argv=None):
    import jax
    import jax.numpy as jnp
    from jax import export
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="edge_xla")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, variant in NESTSAR.items():
        from experiments.nestsar_r5_t16 import preprocessing as pp, worker
        from experiments.nestsar_r5_t16.config import model_kwargs, validate_config
        from experiments.nestsar_r5_t16.model import NestSARR5T16
        m = NestSARR5T16(**model_kwargs(validate_config({"variant": variant})))
        shape = (1, pp.FRAMES, pp.FEATURES); k = jax.random.PRNGKey(0)
        p = m.init({"params": k, "dropout": k}, jnp.zeros(shape), training=False)["params"]
        f = jax.jit(lambda x, p=p, m=m: m.apply({"params": p}, x, training=False)["logits"])
        info = {"strict_mflops": worker.strict_mflops(variant), "params": int(sum(v.size for v in jax.tree.leaves(p)))}
        _save(export, f, shape, out, name, info, manifest)
    import torch
    from experiments.sgn_baseline.sgn import SGN as TorchSGN, strict_mflops
    from experiments.sgn_baseline.sgn_jax import from_torch, sgn_apply
    for name, w in SGN.items():
        torch.manual_seed(0); tm = TorchSGN(120, 20, w).eval(); P = from_torch(tm)
        f = jax.jit(lambda x, P=P: sgn_apply(P, x))
        x = np.random.default_rng(0).normal(0, .3, (1, 20, 75)).astype(np.float32)
        d = float(np.abs(np.asarray(f(x)) - tm(torch.from_numpy(x)).detach().numpy()).max())
        assert d < 1e-4, (name, d)
        info = {"strict_mflops": strict_mflops(TorchSGN(120, 20, w), 20), "params": int(sum(q.numel() for q in tm.parameters())),
                "max_abs_diff_vs_torch": d}
        _save(export, f, (1, 20, 75), out, name, info, manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def _save(export, f, shape, out, name, info, manifest):
    import jax
    exp = export.export(f, platforms=["cpu"])(jax.ShapeDtypeStruct(shape, np.float32))
    (out / f"{name}.jaxexport").write_bytes(exp.serialize())
    x = np.random.default_rng(1).normal(0, .3, shape).astype(np.float32)
    back = export.deserialize((out / f"{name}.jaxexport").read_bytes())
    d = float(np.abs(np.asarray(back.call(x)) - np.asarray(f(x))).max())
    manifest[name] = dict(info, input_shape=list(shape), roundtrip_max_abs_diff=d, jax_version=jax.__version__)
    print(f"{name:16s} exported  roundtrip |diff| {d:.1e}  {info['strict_mflops']:.1f} MFLOPs", flush=True)


if __name__ == "__main__":
    main()

"""Strict FLOP audit for R5 next to R4-FMSE, with the same counter.

Counts the MACs of every dot_general / conv in the inference jaxpr (batch 1,
T = 16), multiplying lax.scan bodies by their length; paper FLOPs = 2 x MACs.
This is the counter used for the SA audit (R4-FMSE = 60.87 MFLOPs). XLA's
cost_analysis is printed only for reference: it does not multiply scan bodies.

    python -m experiments.nestsar_r5_t16.audit_flops --with-r4 [--variants]
"""
from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from experiments.nestsar_sa_t16.audit_flops import count


def audit(model, name, features, frames=16):
    x = jnp.zeros((1, frames, features), jnp.float32)
    params = model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
                        x, training=False)["params"]
    fn = lambda p, xx: model.apply({"params": p}, xx, training=False)["logits"]
    acc = count(jax.make_jaxpr(fn)(params, x).jaxpr)
    try:
        raw = jax.jit(fn).lower(params, x).compile().cost_analysis()
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else {}
        xla = float(raw.get("flops", float("nan")))
    except Exception:  # backend dependent; reference only
        xla = float("nan")
    return {
        "model": name,
        "params": int(sum(np.prod(a.shape) for a in jax.tree_util.tree_leaves(params))),
        "macs": int(acc["macs"]),
        "strict_mflops": 2 * acc["macs"] / 1e6,
        "mflops_with_elementwise": (2 * acc["macs"] + acc["elementwise"] + acc["reductions"]) / 1e6,
        "while_loops_unbounded": int(acc["while_loops"]),
        "xla_cost_analysis_mflops": xla / 1e6,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-r4", action="store_true")
    ap.add_argument("--variants", action="store_true", help="also audit every ablation variant")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    from experiments.nestsar_r5_t16.config import VARIANTS, model_kwargs, validate_config
    from experiments.nestsar_r5_t16.model import NestSARR5T16
    from experiments.nestsar_r5_t16.preprocessing import FEATURES

    rows = []
    if a.with_r4:
        from experiments.nestsar_r4_fmse_t16.model import NestSARR4FMSET16
        rows.append(audit(NestSARR4FMSET16(), "R4-FMSE (reference)", 750))
    names = list(VARIANTS) if a.variants else ["full"]
    for v in names:
        cfg = validate_config({"variant": v})
        rows.append(audit(NestSARR5T16(**model_kwargs(cfg)), f"R5 {v}", FEATURES))
    for r in rows:
        print(f"{r['model']:<30} params {r['params']:>10,}  MACs {r['macs'] / 1e6:7.2f} M  "
              f"strict {r['strict_mflops']:7.2f} MFLOPs  (+elementwise {r['mflops_with_elementwise']:7.2f})  "
              f"XLA {r['xla_cost_analysis_mflops']:7.2f}  while={r['while_loops_unbounded']}", flush=True)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=2)
    return rows


if __name__ == "__main__":
    main()

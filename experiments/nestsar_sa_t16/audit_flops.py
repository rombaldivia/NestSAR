"""Compiler-independent FLOP audit (counted from the JAXPR, before XLA).

Counts the MACs of every dot_general / conv in the inference graph (batch 1,
T=16), multiplying the body of each lax.scan by its length and taking the most
expensive branch of every cond. Paper FLOPs = 2 x MACs. while_loop bodies cannot
be bounded statically and are reported (NestSAR-PT has none). XLA's
cost_analysis() is printed only for comparison: it does not multiply scan
bodies by their trip count, which is why it under-reports recurrent models.

    python -m experiments.nestsar_sa_t16.audit_flops --with-r4
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

ELEMENTWISE = {
    "add", "sub", "mul", "div", "neg", "max", "min", "exp", "log", "log1p", "tanh",
    "logistic", "sqrt", "rsqrt", "pow", "integer_pow", "abs", "sign", "erf", "sin",
    "cos", "select_n", "square", "exp2", "expm1", "atan2", "clamp", "erf_inv",
}
REDUCTIONS = {"reduce_sum", "reduce_max", "reduce_min", "reduce_prod", "argmax", "argmin",
              "cumsum", "cumprod", "cummax", "cumlogsumexp"}


def _size(aval):
    return int(np.prod(aval.shape)) if hasattr(aval, "shape") else 1


def _dot_macs(eqn):
    lhs = eqn.invars[0].aval
    (lc, _), _ = eqn.params["dimension_numbers"]
    k = int(np.prod([lhs.shape[d] for d in lc])) if lc else 1
    return _size(eqn.outvars[0].aval) * k


def _conv_macs(eqn):
    rhs = eqn.invars[1].aval
    spec = eqn.params["dimension_numbers"].rhs_spec
    spatial = int(np.prod([rhs.shape[d] for d in spec[2:]]))
    return _size(eqn.outvars[0].aval) * rhs.shape[spec[1]] * spatial


def count(jaxpr, mult=1, acc=None):
    if acc is None:
        acc = {"macs": 0, "elementwise": 0, "reductions": 0, "while_loops": 0, "prims": Counter()}
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        acc["prims"][name] += mult
        if name == "dot_general":
            acc["macs"] += mult * _dot_macs(eqn)
        elif name == "conv_general_dilated":
            acc["macs"] += mult * _conv_macs(eqn)
        elif name == "scan":
            count(eqn.params["jaxpr"].jaxpr, mult * int(eqn.params["length"]), acc)
        elif name == "cond":
            best = None
            for br in eqn.params["branches"]:
                sub = count(br.jaxpr, 1)
                if best is None or sub["macs"] > best["macs"]:
                    best = sub
            if best:
                for k in ("macs", "elementwise", "reductions", "while_loops"):
                    acc[k] += mult * best[k]
        elif name == "while":
            acc["while_loops"] += mult
        elif name in ELEMENTWISE:
            acc["elementwise"] += mult * _size(eqn.outvars[0].aval)
        elif name in REDUCTIONS:
            acc["reductions"] += mult * _size(eqn.invars[0].aval)
        else:
            for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
                if key in eqn.params:
                    sub = eqn.params[key]
                    count(getattr(sub, "jaxpr", sub), mult, acc)
    return acc


def audit(model, name):
    from experiments.nestsar_sm_all_t16.model import FRAMES, FEATURES
    x = jnp.zeros((1, FRAMES, FEATURES), jnp.float32)
    params = model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
                        x, training=False)["params"]
    fn = lambda p, xx: model.apply({"params": p}, xx, training=False)["logits"]
    acc = count(jax.make_jaxpr(fn)(params, x).jaxpr)
    try:
        raw = jax.jit(fn).lower(params, x).compile().cost_analysis()
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else {}
        xla = float(raw.get("flops", float("nan")))
    except Exception:  # cost_analysis is backend dependent; it is only a reference
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
    ap.add_argument("--with-r4", action="store_true", help="also audit the R4-FMSE baseline")
    ap.add_argument("--json", default=None, help="optional path to save the results")
    a = ap.parse_args(argv)

    from experiments.nestsar_sa_t16.model import NestSARSAT16
    rows = []
    if a.with_r4:
        from experiments.nestsar_r4_fmse_t16.model import NestSARR4FMSET16
        rows.append(audit(NestSARR4FMSET16(), "R4-FMSE (reference)"))
    rows.append(audit(NestSARSAT16(), "NestSAR-SA (R4-FMSE + spatial attention)"))
    for r in rows:
        print(f"{r['model']:<42} params {r['params']:>10,}  MACs {r['macs']/1e6:7.2f} M  "
              f"strict {r['strict_mflops']:7.2f} MFLOPs  (+elementwise {r['mflops_with_elementwise']:7.2f})  "
              f"XLA {r['xla_cost_analysis_mflops']:7.2f}  while={r['while_loops_unbounded']}", flush=True)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=2)
    return rows


if __name__ == "__main__":
    main()

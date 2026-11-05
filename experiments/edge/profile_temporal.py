"""Where does NestSAR's single-clip CPU latency go, and which equivalent graph forms of the temporal core are faster?

Compares, on 1 CPU thread, batch 1:
  * the full model and the NOTEMP variant (spatial front end only),
  * the full model with every jax.lax.scan compiled as-is (while loop) vs fully / partially unrolled.
Unrolling changes only how XLA schedules the recurrence, not the arithmetic: outputs are checked against the
reference within float32 round-off. Run:  python -m experiments.edge.profile_temporal
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import jax
import jax.numpy as jnp

_ORIG_SCAN = jax.lax.scan


def scan_with_unroll(unroll):
    def scan(f, init, xs=None, length=None, reverse=False, unroll_ignored=1, **kw):
        kw.pop("unroll", None)
        return _ORIG_SCAN(f, init, xs, length=length, reverse=reverse, unroll=unroll, **kw)
    return scan


def build(variant, unroll=1):
    from experiments.nestsar_r5_t16 import preprocessing as pp
    from experiments.nestsar_r5_t16 import model as model_mod
    from experiments.nestsar_r5_t16.config import model_kwargs, validate_config
    jax.lax.scan = scan_with_unroll(unroll)
    m =model_mod.NestSARR5T16(**model_kwargs(validate_config({"variant": variant})))
    shape = (1, pp.FRAMES, pp.FEATURES)
    k = jax.random.PRNGKey(0)
    p = m.init({"params": k, "dropout": k}, jnp.zeros(shape), training=False)["params"]
    f = jax.jit(lambda x: m.apply({"params": p}, x, training=False)["logits"])
    x = jnp.asarray(np.random.default_rng(1).normal(0, .3, shape).astype(np.float32))
    y = f(x).block_until_ready()
    jax.lax.scan = _ORIG_SCAN
    return f, x, np.asarray(y)


def bench(f, x, runs=300, warmup=30):
    for _ in range(warmup):
        f(x).block_until_ready()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); f(x).block_until_ready(); t.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(t))


def main():
    print(f"jax {jax.__version__}, 1 thread")
    f0, x, y0 = build("graph_mr_notemp")
    print(f"{'NOTEMP (spatial only)':34s} {bench(f0, x):7.2f} ms")
    f1, x, ref = build("graph_mr", 1)
    print(f"{'NestSAR, scan as while loop':34s} {bench(f1, x):7.2f} ms   (reference)")
    for u in (2, 4, True):
        fu, x, yu = build("graph_mr", u)
        d = float(np.abs(yu - ref).max())
        print(f"{'NestSAR, scan unroll=' + str(u):34s} {bench(fu, x):7.2f} ms   max|diff| {d:.1e}")


if __name__ == "__main__":
    main()

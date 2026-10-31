"""Edge latency benchmark with XLA (jax): batch 1, FP32. Needs only `pip install jax==<manifest jax_version>`.

  python3 bench_xla.py --models edge_xla --runs 500
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" python3 bench_xla.py ...   # 1 thread
  python3 bench_xla.py --models edge_xla --only nestsar --power-window 60      # steady load for the USB power meter
"""
import argparse
import json
import os
import platform
import resource
import socket
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax                     # noqa: E402
from jax import export         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="edge_xla")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--power-window", type=float, default=0)
    a = ap.parse_args()
    root = Path(a.models)
    manifest = json.loads((root / "manifest.json").read_text())
    tag = "t1" if "intra_op_parallelism_threads=1" in os.environ.get("XLA_FLAGS", "") else "tall"
    res = {"host": socket.gethostname(), "machine": platform.machine(), "jax": jax.__version__, "threads": tag, "models": {}}
    for name in a.only or list(manifest):
        f = jax.jit(export.deserialize((root / f"{name}.jaxexport").read_bytes()).call)
        x = np.random.default_rng(0).normal(0, 0.3, manifest[name]["input_shape"]).astype(np.float32)
        t0 = time.perf_counter(); f(x).block_until_ready(); compile_s = time.perf_counter() - t0
        for _ in range(a.warmup):
            f(x).block_until_ready()
        if a.power_window:
            print(f"{name}: continuous inference for {a.power_window:.0f}s - read P_active on the meter", flush=True)
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < a.power_window:
                f(x).block_until_ready(); n += 1
            print(f"  {n / a.power_window:.1f} clips/s", flush=True); continue
        ts = np.empty(a.runs)
        for i in range(a.runs):
            t = time.perf_counter(); f(x).block_until_ready(); ts[i] = (time.perf_counter() - t) * 1e3
        r = {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)), "mean_ms": float(ts.mean()),
             "clips_per_s": float(1e3 / np.median(ts)), "compile_s": compile_s,
             "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
             "strict_mflops": manifest[name]["strict_mflops"], "params": manifest[name]["params"]}
        res["models"][name] = r
        print(f"{name:16s} median {r['median_ms']:8.2f} ms  p90 {r['p90_ms']:8.2f}  {r['clips_per_s']:8.1f} clips/s  "
              f"{r['strict_mflops']:7.1f} MFLOPs", flush=True)
    if res["models"]:
        out = Path(f"bench_xla_{res['host']}_{tag}.json"); out.write_text(json.dumps(res, indent=1)); print("saved", out)


if __name__ == "__main__":
    main()

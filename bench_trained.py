"""Latency of the TRAINED NestSAR programs (edge_xla_trained/) on this device.

Same protocol as bench_xla.py: batch 1, FP32, XLA (jax 0.8.1), 50 warm-up calls, median of 300 synchronized calls,
with all threads and with 1 thread; the fastest setting is kept. Every (program, thread setting) runs in a fresh
process, so memory does not accumulate. The first and the last run are the random-weight program of the original
benchmark (edge_xla/nestsar_split), as a control of the session. If the on-device evaluation package is present
(pi_eval_ntu60_xsub/x.npy), the two full models are also timed on real NTU-60 clips held in memory.

    ./bench_trained.sh            about 50 min on a Raspberry Pi Zero 2 W, a few minutes on a PC
    RUNS=100 ./bench_trained.sh   faster

Interrupted? Run it again: finished measurements are kept in resultados_trained/ and skipped.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "resultados_trained"
RUNS = int(os.environ.get("RUNS", 300))
WARM = int(os.environ.get("WARMUP", 50))
ONE = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
ORDER = ["base120", "base120_hybrid", "base120_stacked", "base60", "flatstack", "flat", "nofast", "shared", "causal",
         "nohand", "nointer", "st2", "tattn", "tattn_stacked", "tconv", "tconv_stacked", "notemp", "notemp_stacked",
         "memonly", "memonly_stacked", "tavg", "tavg_stacked"]
LABEL = {"control": "CONTROL random weights (paper: 35.0)", "control_end": "CONTROL again, end of session",
         "base120": "NestSAR trained, NTU-120", "base120_hybrid": "NestSAR trained, hand scan stacked",
         "base120_stacked": "NestSAR trained, training schedule", "base60": "NestSAR trained, NTU-60 (Pi accuracy model)",
         "flatstack": "FLATSTACK", "flat": "FLAT (M4 only)", "nofast": "NOFAST", "shared": "SHARED", "causal": "CAUSAL",
         "nohand": "No hand branch", "nointer": "No interaction features", "st2": "ST2", "tattn": "TATTN",
         "tattn_stacked": "TATTN, hand scan stacked", "tconv": "TCONV", "tconv_stacked": "TCONV, hand scan stacked",
         "notemp": "NOTEMP", "notemp_stacked": "NOTEMP, hand scan stacked", "memonly": "MEMONLY",
         "memonly_stacked": "MEMONLY, hand scan stacked", "tavg": "Temporal average",
         "tavg_stacked": "Temporal average, hand scan stacked",
         "base120_real": "NestSAR trained, NTU-120, real clips", "base60_real": "NestSAR trained, NTU-60, real clips"}


def child(path, out, clips):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np
    import jax
    from jax import export
    f = jax.jit(export.deserialize(Path(path).read_bytes()).call)
    if clips:
        x = np.load(clips, mmap_mode="r")
        xs = [np.array(x[i:i + 1], np.float32) for i in np.linspace(0, len(x) - 1, 64).astype(int)]
    else:
        xs = [np.random.default_rng(0).normal(0, 0.3, (1, 16, 942)).astype(np.float32)]
    t0 = time.perf_counter(); f(xs[0]).block_until_ready(); compile_s = time.perf_counter() - t0
    for i in range(WARM):
        f(xs[i % len(xs)]).block_until_ready()
    ts = np.empty(RUNS)
    for i in range(RUNS):
        x = xs[i % len(xs)]
        t = time.perf_counter(); f(x).block_until_ready(); ts[i] = (time.perf_counter() - t) * 1e3
    Path(out).write_text(json.dumps({"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)),
                                     "p10_ms": float(np.percentile(ts, 10)), "mean_ms": float(ts.mean()), "runs": RUNS,
                                     "warmup": WARM, "compile_s": compile_s, "jax": jax.__version__,
                                     "input": "real clips" if clips else "random"}))


def vc(cmd):
    try:
        return subprocess.run(["vcgencmd", cmd], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def find_clips():
    for p in [HERE / "pi_eval_ntu60_xsub/x.npy", HERE.parent / "nestsar_edge_kit/pi_eval_ntu60_xsub/x.npy",
              Path.home() / "nestsar_edge_kit/pi_eval_ntu60_xsub/x.npy", Path.home() / "pi_eval_ntu60_xsub/x.npy"]:
        if p.exists():
            return str(p)
    return ""


def main():
    OUT.mkdir(exist_ok=True)
    man = json.loads((HERE / "edge_xla_trained/manifest.json").read_text())
    clips = find_clips()
    jobs = [("control", HERE / "edge_xla/nestsar_split.jaxexport", "")]
    for n in ORDER:
        if n in man:
            jobs.append((n, HERE / f"edge_xla_trained/{n}.jaxexport", ""))
            if clips and n in ("base120", "base60"):       # the same program on real NTU-60 clips held in memory
                jobs.append((n + "_real", HERE / f"edge_xla_trained/{n}.jaxexport", clips))
    jobs += [("control_end", HERE / "edge_xla/nestsar_split.jaxexport", "")]
    todo = [(n, p, c, tag) for n, p, c in jobs for tag in ("tall", "t1")]
    import platform
    print(f"== {platform.node()} {platform.machine()}, {os.cpu_count()} cores, {len(todo)} measurements, {RUNS} calls each; "
          f"real clips: {clips or 'not found (random input only)'}")
    print("== start", vc("measure_temp"), vc("get_throttled"), flush=True)
    t_start = time.time()
    for k, (n, p, c, tag) in enumerate(todo):
        out = OUT / f"{n}_{tag}.json"
        if not out.exists() or n.startswith("control"):
            env = dict(os.environ, RUNS=str(RUNS), WARMUP=str(WARM))
            if tag == "t1":
                env["XLA_FLAGS"] = ONE
            r = subprocess.run([sys.executable, __file__, "--child", str(p), str(out), c], env=env,
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"!! {n} [{tag}] failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}", flush=True)
                continue
        d = json.loads(out.read_text())
        el = time.time() - t_start
        print(f"[{k + 1:2d}/{len(todo)}] {n:18s} {'all threads' if tag == 'tall' else '1 thread   '} median {d['median_ms']:8.2f} ms  "
              f"p90 {d['p90_ms']:8.2f}  (compile {d['compile_s']:.0f}s, elapsed {el / 60:.0f} min)", flush=True)
    print("== end  ", vc("measure_temp"), vc("get_throttled"))
    lines = [f"{'program':46s} {'all thr':>9s} {'1 thr':>9s} {'BEST ms':>9s}", "-" * 78]
    for n, _, _ in jobs:
        v = {}
        for tag in ("tall", "t1"):
            f = OUT / f"{n}_{tag}.json"
            if f.exists():
                v[tag] = json.loads(f.read_text())["median_ms"]
        if v:
            fmt = lambda t: f"{v[t]:9.2f}" if t in v else "        -"
            lines.append(f"{LABEL.get(n, n):46s} {fmt('tall')} {fmt('t1')} {min(v.values()):9.2f}")
    txt = "\n".join(lines)
    print("\n" + txt)
    (OUT / "resumen.txt").write_text(txt + "\n")
    print(f"\nsaved in {OUT}/  (paste this table)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    else:
        main()

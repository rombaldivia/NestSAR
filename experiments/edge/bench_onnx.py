"""Edge latency benchmark (Raspberry Pi Zero 2 W, Jetson, PC): ONNX Runtime, batch 1, FP32.

  python3 bench_onnx.py --models edge_models --threads 4 --runs 1000
  python3 bench_onnx.py --models edge_models --only nestsar --power-window 60   # steady load for a USB power meter

Per model: warm-up, then `runs` timed single-clip inferences -> median / p90 / mean latency (ms), clips/s,
peak RSS. Results go to bench_<host>.json. Only needs numpy + onnxruntime (no JAX / PyTorch on the device).
Energy: read the USB meter while --power-window runs one model continuously (P_active), and once with
--idle-window at rest (P_idle); energy per clip = P_active x median latency (and (P_active - P_idle) x latency).
"""
import argparse
import json
import platform
import resource
import socket
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def session(path, threads):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="edge_models")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--power-window", type=float, default=0, help="seconds of continuous inference (power meter)")
    ap.add_argument("--idle-window", type=float, default=0, help="seconds of idle (power meter baseline)")
    a = ap.parse_args()
    root = Path(a.models)
    manifest = json.loads((root / "manifest.json").read_text())
    if a.idle_window:
        print(f"IDLE for {a.idle_window:.0f}s - read P_idle on the meter now", flush=True); time.sleep(a.idle_window); return
    names = a.only or list(manifest)
    host = socket.gethostname()
    res = {"host": host, "machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version(),
           "onnxruntime": ort.__version__, "threads": a.threads, "models": {}}
    for name in names:
        s = session(root / f"{name}.onnx", a.threads)
        inp = s.get_inputs()[0]
        x = np.random.default_rng(0).normal(0, 0.3, manifest[name]["input_shape"]).astype(np.float32)
        feed = {inp.name: x}
        for _ in range(a.warmup):
            s.run(None, feed)
        if a.power_window:
            print(f"{name}: continuous inference for {a.power_window:.0f}s - read P_active on the meter", flush=True)
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < a.power_window:
                s.run(None, feed); n += 1
            print(f"  {n} clips in {a.power_window:.0f}s = {n / a.power_window:.1f} clips/s", flush=True)
            continue
        ts = np.empty(a.runs)
        for i in range(a.runs):
            t = time.perf_counter(); s.run(None, feed); ts[i] = (time.perf_counter() - t) * 1e3
        r = {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)), "mean_ms": float(ts.mean()),
             "min_ms": float(ts.min()), "clips_per_s": float(1e3 / np.median(ts)),
             "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
             "strict_mflops": manifest[name]["strict_mflops"], "params": manifest[name]["params"]}
        res["models"][name] = r
        print(f"{name:16s} median {r['median_ms']:8.2f} ms  p90 {r['p90_ms']:8.2f}  {r['clips_per_s']:8.1f} clips/s  "
              f"{r['strict_mflops']:7.1f} MFLOPs", flush=True)
    if res["models"]:
        out = Path(f"bench_{host}_t{a.threads}.json"); out.write_text(json.dumps(res, indent=1)); print("saved", out)


if __name__ == "__main__":
    main()

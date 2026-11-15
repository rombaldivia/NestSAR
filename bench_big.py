"""Batch-one CPU latency of large published recognizers (FP32 ONNX Runtime), same protocol as bench_onnx.py:
50 warm-up calls, then the median of N synchronized calls, for 1 and 4 threads. Latency does not depend on the
weight values; SkateFormer uses its official released checkpoint (ntu60-xsub-joint, Hugging Face), HD-GCN and
InfoGCN are built from their official repositories with the NTU-60 joint configuration.

    python3 bench_big.py            # N = 30, all models (about 20 min on a Pi Zero 2 W)
    python3 bench_big.py 30 hdgcn   # one model
"""
import json
import sys
import time

import numpy as np
import onnxruntime as ort

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
MODELS = {   # name: (file, reported GFLOPs, reported NTU-60 XSub top-1, params M)
    "skateformer": ("skateformer_ntu60_xsub_joint.onnx", 3.62, 92.6, 2.03),
    "hdgcn": ("hdgcn.onnx", 3.44, 90.6, 1.66),
    "infogcn": ("infogcn.onnx", 3.34, 89.8, 1.56),
}
pick = sys.argv[2:] or list(MODELS)
rng = np.random.default_rng(0)
results = {}
for name in pick:
    f, gflops, acc, params = MODELS[name]
    for threads in (1, 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        s = ort.InferenceSession("edge_models/" + f, so, providers=["CPUExecutionProvider"])
        feed = {}
        for i in s.get_inputs():
            feed[i.name] = (np.arange(64, dtype=np.float32)[None] if i.name == "index_t"
                            else rng.normal(size=i.shape).astype(np.float32))
        for _ in range(50):
            s.run(None, feed)
        ts = []
        for _ in range(N):
            t = time.perf_counter()
            s.run(None, feed)
            ts.append(time.perf_counter() - t)
        med = 1e3 * float(np.median(ts))
        results[f"{name}_t{threads}"] = med
        print(f"{name:12s} {threads} thread(s): median {med:8.1f} ms   p90 {1e3 * np.percentile(ts, 90):8.1f} ms   "
              f"({N} calls; reported {gflops} GFLOPs, {params}M, {acc}% NTU-60 XSub)", flush=True)
json.dump(results, open("bench_big_result.json", "w"), indent=1)
print("saved bench_big_result.json")

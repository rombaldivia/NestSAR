"""Batch-one CPU latency of the released SkateFormer checkpoint (ntu60-xsub-joint, official weights from
huggingface.co/JeonghyeokDo/SkateFormer, exported to ONNX; matches PyTorch to 1.2e-6). Same protocol as
bench_onnx.py: 50 warm-up calls, then the median of N synchronized calls, for 1 and 4 threads.

    python3 bench_skateformer.py            # N = 30
"""
import sys
import time

import numpy as np
import onnxruntime as ort

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
path = "edge_models/skateformer_ntu60_xsub_joint.onnx"
rng = np.random.default_rng(0)
feed = {"x": rng.normal(size=(1, 3, 64, 24, 2)).astype(np.float32),
        "index_t": np.arange(64, dtype=np.float32)[None]}
for threads in (1, 4):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    for _ in range(50):
        s.run(None, feed)
    ts = []
    for _ in range(N):
        t = time.perf_counter()
        s.run(None, feed)
        ts.append(time.perf_counter() - t)
    print(f"SkateFormer (official ntu60-xsub-joint)  {threads} thread(s): median {1e3 * np.median(ts):8.1f} ms  "
          f"p90 {1e3 * np.percentile(ts, 90):8.1f} ms  ({N} calls)")

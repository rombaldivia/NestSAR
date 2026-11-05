"""On-device accuracy: run a packaged NestSAR program (make_pi_eval.py) over a full test split, one clip at a time.

    python3 eval_pi.py pi_eval_ntu60_xsub           # ~10 min for NTU-60 XSub on a Raspberry Pi Zero 2 W
Prints top-1, the per-clip latency of this run, and the reference top-1 of the same weights computed off-device.
"""
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
from jax import export


def main(folder):
    d = Path(folder)
    info = json.loads((d / "info.json").read_text())
    x = np.load(d / "x.npy", mmap_mode="r"); y = np.load(d / "y.npy")
    f = jax.jit(export.deserialize((d / "model.jaxexport").read_bytes()).call)
    f(np.asarray(x[:1])).block_until_ready()
    n = len(y); hits = 0; ts = np.empty(n); t_start = time.time()
    for i in range(n):
        t = time.perf_counter()
        pred = int(np.asarray(f(np.asarray(x[i:i + 1]))).argmax())
        ts[i] = (time.perf_counter() - t) * 1e3
        hits += pred == int(y[i])
        if (i + 1) % 500 == 0 or i + 1 == n:
            el = time.time() - t_start
            print(f"\r{i + 1:6d}/{n}  top-1 {100 * hits / (i + 1):6.2f}%  "
                  f"{np.median(ts[:i + 1]):6.2f} ms/clip  faltan {el / (i + 1) * (n - i - 1) / 60:5.1f} min",
                  end="", flush=True)
    res = dict(info, top1_on_device=100 * hits / n, median_ms=float(np.median(ts)), p90_ms=float(np.percentile(ts, 90)),
               device_jax=jax.__version__)
    (d / "result_on_device.json").write_text(json.dumps(res, indent=1))
    print(f"\n\n{info['run']} [{info['protocol']}]  {n} clips")
    print(f"  top-1 en el dispositivo     {res['top1_on_device']:.2f}%")
    print(f"  top-1 de referencia (GPU)   {info['top1_reference_training_schedule']:.2f}%")
    print(f"  latencia mediana            {res['median_ms']:.2f} ms/clip (incluye la lectura del clip)")
    print(f"  guardado en {d / 'result_on_device.json'}")


if __name__ == "__main__":
    main(sys.argv[1])

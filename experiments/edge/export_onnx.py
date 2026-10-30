"""Export the edge-benchmark models to ONNX (batch 1, FP32) and verify each against its source framework.

Latency does not depend on the weight VALUES (same graph, same shapes, same ops), so trained checkpoints are
optional: without --ckpt a model is exported with its initial weights. With --ckpt (NestSAR best.msgpack or
SGN best.pt) the trained weights are used, e.g. to also check accuracy on the device.

  python -m experiments.edge.export_onnx --out edge_models                       # all six, random init
  python -m experiments.edge.export_onnx --out edge_models --only nestsar --ckpt .../xsub/best.msgpack
Needs: jax==0.8.1, flax, jax2onnx==0.17.1, onnx, onnxruntime, torch (CPU).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

NESTSAR = {"nestsar": "graph_mr", "nestsar_tattn": "graph_mr_tattn", "nestsar_tconv": "graph_mr_tconv",
           "nestsar_notemp": "graph_mr_notemp"}
SGN = {"sgn40": 0.3125, "sgn_full": 1.0}


def unrolled_scan(f, init, xs, length=None, reverse=False, unroll=1, **kw):
    """Drop-in for jax.lax.scan that unrolls in Python: the exported graph has no ONNX Loop (ORT runs a Loop
    body through a sub-executor per step, which dominated NestSAR's CPU latency). Same math, same outputs."""
    import jax
    import jax.numpy as jnp
    n = length if xs is None else jax.tree.leaves(xs)[0].shape[0]
    carry, ys = init, []
    order = range(n - 1, -1, -1) if reverse else range(n)
    for i in order:
        carry, y = f(carry, None if xs is None else jax.tree.map(lambda a: a[i], xs))
        ys.append(y)
    if reverse:
        ys = ys[::-1]
    return carry, (None if ys[0] is None else jax.tree.map(lambda *a: jnp.stack(a), *ys))


def export_nestsar(name, variant, out, ckpt=None, unroll=True):
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from jax2onnx import to_onnx
    from experiments.nestsar_r5_t16 import preprocessing as pp, worker
    from experiments.nestsar_r5_t16.config import model_kwargs, validate_config
    from experiments.nestsar_r5_t16.model import NestSARR5T16
    m = NestSARR5T16(**model_kwargs(validate_config({"variant": variant})))
    k = jax.random.PRNGKey(0)
    shape = (1, pp.FRAMES, pp.FEATURES)
    params = m.init({"params": k, "dropout": k}, jnp.zeros(shape), training=False)["params"]
    if ckpt:
        params = serialization.from_state_dict(params, serialization.msgpack_restore(Path(ckpt).read_bytes())["ema_params"])
    f = lambda x: m.apply({"params": params}, x, training=False)["logits"]
    import onnx
    scan = jax.lax.scan
    if unroll:
        jax.lax.scan = unrolled_scan
    try:
        onnx.save(to_onnx(f, [shape], model_name=name), out / f"{name}.onnx")
    finally:
        jax.lax.scan = scan
    xs = [np.asarray(jax.random.normal(jax.random.PRNGKey(s), shape) * 0.3, np.float32) for s in range(3)]
    ref = [np.asarray(f(jnp.asarray(x))) for x in xs]
    return shape, xs, ref, {"params": int(sum(a.size for a in jax.tree.leaves(params))),
                            "strict_mflops": worker.strict_mflops(variant), "variant": variant}


def export_sgn(name, width, out, ckpt=None, classes=120, seg=20):
    import torch
    from experiments.sgn_baseline.sgn import SGN, strict_mflops
    torch.manual_seed(0)
    m = SGN(classes, seg, width).eval()
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        classes = sd["fc.weight"].shape[0]
        m = SGN(classes, seg, width).eval(); m.load_state_dict(sd)
    shape = (1, seg, 75)
    torch.onnx.export(m, torch.zeros(shape), out / f"{name}.onnx", input_names=["x"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    xs = [np.random.default_rng(s).normal(0, 0.3, shape).astype(np.float32) for s in range(3)]
    with torch.no_grad():
        ref = [m(torch.from_numpy(x)).numpy() for x in xs]
    return shape, xs, ref, {"params": int(sum(p.numel() for p in m.parameters())),
                            "strict_mflops": strict_mflops(SGN(classes, seg, width), seg), "width": width}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="edge_models")
    ap.add_argument("--only", nargs="*", default=None, help=f"subset of {list(NESTSAR) + list(SGN)}")
    ap.add_argument("--ckpt", default=None, help="trained weights for the (single) model in --only")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort
    names = a.only or list(NESTSAR) + list(SGN)
    if a.ckpt and len(names) != 1:
        raise SystemExit("--ckpt needs exactly one model in --only")
    manifest = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    for name in names:
        t = time.time()
        shape, xs, ref, info = (export_nestsar(name, NESTSAR[name], out, a.ckpt) if name in NESTSAR
                                else export_sgn(name, SGN[name], out, a.ckpt))
        s = ort.InferenceSession(str(out / f"{name}.onnx"), providers=["CPUExecutionProvider"])
        diff = max(float(np.abs(s.run(None, {s.get_inputs()[0].name: x})[0] - r).max()) for x, r in zip(xs, ref))
        if diff > 1e-3:
            raise SystemExit(f"{name}: ONNX differs from the source model by {diff}")
        manifest[name] = dict(info, input_shape=list(shape), max_abs_diff_vs_source=diff, trained=bool(a.ckpt),
                              onnx_bytes=(out / f"{name}.onnx").stat().st_size)
        print(f"{name:16s} ok  max|diff| {diff:.2e}  {info['strict_mflops']:.1f} MFLOPs  "
              f"{manifest[name]['onnx_bytes'] / 2**20:.1f} MB  ({time.time() - t:.0f}s)", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()

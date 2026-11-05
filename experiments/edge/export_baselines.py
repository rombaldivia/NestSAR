"""Export the external baselines for the edge benchmark: SkelAct (Two-Stream CNN, HCN-Baseline, Ta-CNN) and CTR-GCN.

Each model: official PyTorch code -> ONNX (edge_models/) -> JAX via onnx_jax -> jax.export (edge_xla/), verified
against ONNX Runtime. Appends to the manifests written by export_onnx.py / export_xla.py.

  python -m experiments.edge.export_baselines --onnx-out edge_models --xla-out edge_xla --ctrgcn-repo path/to/CTR-GCN
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def strict_mflops(model, shape):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:      # no torch.no_grad(): CTR-GCN's hooks need grad mode
        model(torch.randn(shape))
    return fc.get_total_flops() / 1e6


def main(argv=None):
    import jax
    import onnxruntime as ort
    import torch
    from jax import export
    from experiments.edge.onnx_jax import make_fn
    from experiments.edge.skelact_models import build
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-out", default="edge_models")
    ap.add_argument("--xla-out", default="edge_xla")
    ap.add_argument("--ctrgcn-repo", default=None, help="clone of github.com/Uason-Chen/CTR-GCN (skipped if absent)")
    a = ap.parse_args(argv)
    on, xo = Path(a.onnx_out), Path(a.xla_out)
    on.mkdir(parents=True, exist_ok=True); xo.mkdir(parents=True, exist_ok=True)
    man_o = json.loads((on / "manifest.json").read_text()) if (on / "manifest.json").exists() else {}
    man_x = json.loads((xo / "manifest.json").read_text()) if (xo / "manifest.json").exists() else {}
    models = {}
    for n in ("tscnn", "hcnb", "tacnn"):
        torch.manual_seed(0); models[n] = build(n)
    if a.ctrgcn_repo:
        sys.path.insert(0, str(Path(a.ctrgcn_repo).resolve()))
        from model.ctrgcn import Model
        torch.manual_seed(0)
        models["ctrgcn"] = (Model(num_class=120, num_point=25, num_person=2, graph="graph.ntu_rgb_d.Graph",
                                  graph_args={"labeling_mode": "spatial"}), (1, 3, 64, 25, 2))
    for n, (m, shape) in models.items():
        m.eval()
        mf = strict_mflops(m, shape)
        with torch.no_grad():
            torch.onnx.export(m, torch.randn(shape), on / f"{n}.onnx", input_names=["x"], output_names=["logits"],
                              opset_version=17, dynamo=False)
        params = int(sum(p.numel() for p in m.parameters()))
        fn, shp = make_fn(str(on / f"{n}.onnx")); f = jax.jit(fn)
        x = np.random.default_rng(1).normal(0, .3, shp).astype(np.float32)
        s = ort.InferenceSession(str(on / f"{n}.onnx"), providers=["CPUExecutionProvider"])
        ref = s.run(None, {"x": x})[0]
        d = float(np.abs(np.asarray(f(x)) - ref).max())
        if d > 1e-2 * max(1.0, float(np.abs(ref).max())):
            raise SystemExit(f"{n}: JAX port differs from ONNX Runtime by {d}")
        exp = export.export(f, platforms=["cpu"])(jax.ShapeDtypeStruct(tuple(shp), np.float32))
        (xo / f"{n}.jaxexport").write_bytes(exp.serialize())
        common = {"strict_mflops": mf, "params": params, "input_shape": list(shp)}
        man_o[n] = dict(common, trained=False, source="official PyTorch code")
        man_x[n] = dict(common, max_abs_diff_vs_onnxruntime=d, jax_version=jax.__version__,
                        source="official PyTorch code -> ONNX -> onnx_jax")
        print(f"{n:7s} {mf:8.1f} MFLOPs  {params / 1e6:.3f}M params  |JAX - ORT| {d:.1e}", flush=True)
    (on / "manifest.json").write_text(json.dumps(man_o, indent=1))
    (xo / "manifest.json").write_text(json.dumps(man_x, indent=1))


if __name__ == "__main__":
    main()

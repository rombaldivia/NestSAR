"""Minimal ONNX -> JAX interpreter for the CNN/GCN baselines (static shapes, inference only).

Lets every model be compiled by the same compiler (XLA) as NestSAR. Covers exactly the operators the exported
baselines use; anything else raises. Each exported program is verified against ONNX Runtime by the caller.
"""
import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx import numpy_helper


def _attrs(node):
    out = {}
    for a in node.attribute:
        v = onnx.helper.get_attribute_value(a)
        out[a.name] = v.decode() if isinstance(v, bytes) else v
    return out


def _const(x):
    return isinstance(x, (np.ndarray, np.generic))


def make_fn(model_path):
    m = onnx.load(model_path)
    g = m.graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    inp = [i.name for i in g.input if i.name not in inits][0]

    def run(x):
        env = dict(inits); env[inp] = x
        for n in g.node:
            a = _attrs(n); ins = [env[i] if i else None for i in n.input]; op = n.op_type
            if op == "Identity": r = ins[0]
            elif op == "Constant": r = numpy_helper.to_array(a["value"])
            elif op == "Shape": r = np.asarray(ins[0].shape, np.int64)
            elif op == "Gather":
                ax = a.get("axis", 0)
                r = np.asarray(np.take(ins[0], ins[1], axis=ax)) if _const(ins[0]) else jnp.take(ins[0], np.asarray(ins[1]), axis=ax)
            elif op == "Unsqueeze":
                r = ins[0]
                for ax in sorted(int(v) for v in np.asarray(ins[1]).reshape(-1)):
                    r = (np if _const(r) else jnp).expand_dims(r, ax)
            elif op == "Reshape":
                shp = [int(s) for s in np.asarray(ins[1])]
                shp = [ins[0].shape[i] if s == 0 else s for i, s in enumerate(shp)]
                r = (np if _const(ins[0]) else jnp).reshape(ins[0], shp)
            elif op == "Transpose": r = jnp.transpose(ins[0], a["perm"])
            elif op == "Concat": r = (np if all(_const(v) for v in ins) else jnp).concatenate(ins, axis=a["axis"])
            elif op == "Slice":
                starts, ends = np.asarray(ins[1]), np.asarray(ins[2])
                axes = np.asarray(ins[3]) if len(ins) > 3 and ins[3] is not None else np.arange(len(starts))
                steps = np.asarray(ins[4]) if len(ins) > 4 and ins[4] is not None else np.ones(len(starts), np.int64)
                sl = [slice(None)] * ins[0].ndim
                for s, e, ax, st in zip(starts, ends, axes, steps):
                    sl[int(ax)] = slice(int(s), int(min(e, 2 ** 31)), int(st))
                r = ins[0][tuple(sl)]
            elif op == "Add": r = ins[0] + ins[1]
            elif op == "Sub": r = ins[0] - ins[1]
            elif op == "Mul": r = ins[0] * ins[1]
            elif op == "Div": r = ins[0] / ins[1]
            elif op == "Relu": r = jax.nn.relu(ins[0])
            elif op == "Sigmoid": r = jax.nn.sigmoid(ins[0])
            elif op == "Tanh": r = jnp.tanh(ins[0])
            elif op == "PRelu": r = jnp.where(ins[0] >= 0, ins[0], ins[0] * ins[1])
            elif op == "MatMul": r = jnp.matmul(ins[0], ins[1])
            elif op == "Gemm":
                A = ins[0].T if a.get("transA", 0) else ins[0]; B = ins[1].T if a.get("transB", 0) else ins[1]
                r = a.get("alpha", 1.0) * (A @ B) + (a.get("beta", 1.0) * ins[2] if len(ins) > 2 and ins[2] is not None else 0)
            elif op == "Einsum": r = jnp.einsum(a["equation"], *ins)
            elif op in ("ReduceMean", "ReduceMax"):
                axes = a.get("axes", None)
                if axes is None and len(ins) > 1 and ins[1] is not None: axes = [int(v) for v in np.asarray(ins[1])]
                f = jnp.mean if op == "ReduceMean" else jnp.max
                r = f(ins[0], axis=tuple(axes) if axes is not None else None, keepdims=bool(a.get("keepdims", 1)))
            elif op == "GlobalAveragePool": r = jnp.mean(ins[0], axis=tuple(range(2, ins[0].ndim)), keepdims=True)
            elif op == "BatchNormalization":
                x_, sc, b, mu, var = ins[:5]; sh = (1, -1) + (1,) * (x_.ndim - 2)
                inv = sc / np.sqrt(var + a.get("epsilon", 1e-5))
                r = x_ * inv.reshape(sh) + (b - mu * inv).reshape(sh)
            elif op == "Conv":
                k = len(ins[1].shape) - 2
                pads = a.get("pads", [0] * (2 * k)); strides = a.get("strides", [1] * k); dil = a.get("dilations", [1] * k)
                r = jax.lax.conv_general_dilated(ins[0], ins[1], window_strides=strides,
                                                 padding=[(pads[i], pads[i + k]) for i in range(k)], rhs_dilation=dil,
                                                 feature_group_count=a.get("group", 1),
                                                 dimension_numbers=("NCHW", "OIHW", "NCHW") if k == 2 else ("NCH", "OIH", "NCH"))
                if len(ins) > 2 and ins[2] is not None: r = r + ins[2].reshape((1, -1) + (1,) * k)
            elif op == "MaxPool":
                ks, st = a["kernel_shape"], a.get("strides", a["kernel_shape"]); k = len(ks)
                pads = a.get("pads", [0] * (2 * k))
                r = jax.lax.reduce_window(ins[0], -jnp.inf, jax.lax.max, (1, 1) + tuple(ks), (1, 1) + tuple(st),
                                          [(0, 0), (0, 0)] + [(pads[i], pads[i + k]) for i in range(k)])
            else:
                raise NotImplementedError(f"ONNX op {op} not supported by onnx_jax")
            env[n.output[0]] = r
        return env[g.output[0].name]
    shape = [d.dim_value for d in [i for i in g.input if i.name == inp][0].type.tensor_type.shape.dim]
    return run, shape

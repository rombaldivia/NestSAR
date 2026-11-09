"""Fetch pinned ONNX exports and lower static graphs for TensorRT 8.2/10.

This build-time program runs in its own modern Python stage, never in the
JetPack 4 runtime. Every transformed model is checked against the original.
"""
import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as H, numpy_helper as NH
from onnxsim import simplify


class Lowerer:
    def __init__(self, model):
        self.model = model
        self.nodes = []
        self.serial = 0

    def name(self):
        self.serial += 1
        return "__jetson_{}".format(self.serial)

    def const(self, value, dtype=np.int64):
        name = self.name()
        self.model.graph.initializer.append(NH.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(self, kind, inputs, output=None, **attrs):
        output = output or self.name()
        self.nodes.append(H.make_node(kind, inputs, [output], **attrs))
        return output

    def reshape(self, x, shape):
        return self.op("Reshape", [x, self.const(shape)])

    def transpose(self, x, labels, ordered):
        perm = [labels.index(s) for s in ordered]
        return x if perm == list(range(len(labels))) else self.op("Transpose", [x], perm=perm)

    def einsum(self, node, shapes):
        """Binary contraction -> transpose/reshape/MatMul, including ellipses.

        TRT 8.2 cannot parse the ellipsis equations in the attention ablation.
        Avoid depending on its Einsum importer for any of the exports.
        """
        attrs = {a.name: H.get_attribute_value(a) for a in node.attribute}
        inputs, out = attrs["equation"].decode().replace(" ", "").split("->")
        terms = inputs.split(",")
        if len(terms) != 2:
            raise ValueError("Only binary Einsum is supported: {}".format(terms))
        ranks = [len(shapes[x]) for x in node.input]
        counts = [rank - len(t.replace("...", "")) if "..." in t else 0 for t, rank in zip(terms, ranks)]
        ellipsis = ["@{}".format(i) for i in range(max(counts))]

        def expand(term, n):
            if "..." not in term:
                return list(term)
            start, end = term.split("...")
            return list(start) + (ellipsis[-n:] if n else []) + list(end)

        left, right = [expand(t, n) for t, n in zip(terms, counts)]
        output = expand(out, len(ellipsis))
        if len(set(left)) != len(left) or len(set(right)) != len(right):
            raise ValueError("Diagonal Einsum is not supported")
        batch = [s for s in output if s in left and s in right]
        free_l = [s for s in left if s in output and s not in batch]
        free_r = [s for s in right if s in output and s not in batch]
        contracted = [s for s in left if s in right and s not in output]
        if set(left + right) != set(batch + free_l + free_r + contracted):
            raise ValueError("One-sided Einsum reduction is not supported")
        ld, rd = dict(zip(left, shapes[node.input[0]])), dict(zip(right, shapes[node.input[1]]))
        for s in contracted:
            if ld[s] != rd[s]:
                raise ValueError("Einsum contraction dimensions do not match")
        for s in batch:
            if ld[s] != rd[s] and min(ld[s], rd[s]) != 1:
                raise ValueError("Einsum batch dimensions cannot broadcast")
        a = self.transpose(node.input[0], left, batch + free_l + contracted)
        b = self.transpose(node.input[1], right, batch + contracted + free_r)
        product = lambda dims, keys: math.prod(dims[k] for k in keys)
        a = self.reshape(a, [ld[s] for s in batch] + [product(ld, free_l), product(ld, contracted)])
        b = self.reshape(b, [rd[s] for s in batch] + [product(rd, contracted), product(rd, free_r)])
        y = self.op("MatMul", [a, b])
        order = batch + free_l + free_r
        dims = [max(ld[s], rd[s]) for s in batch] + [ld[s] for s in free_l] + [rd[s] for s in free_r]
        y = self.reshape(y, dims)
        y = self.transpose(y, order, output)
        self.op("Identity", [y], output=node.output[0])

    def gelu(self, node):
        attr = {a.name: H.get_attribute_value(a) for a in node.attribute}
        if attr.get("approximate", b"none") != b"tanh":
            raise ValueError("Only the exported tanh GELU is supported")
        x = node.input[0]
        square = self.op("Mul", [x, x])
        cube = self.op("Mul", [square, x])
        inner = self.op("Mul", [cube, self.const(0.044715, np.float32)])
        inner = self.op("Add", [x, inner])
        inner = self.op("Mul", [inner, self.const(math.sqrt(2 / math.pi), np.float32)])
        inner = self.op("Tanh", [inner])
        inner = self.op("Add", [inner, self.const(1.0, np.float32)])
        half_x = self.op("Mul", [x, self.const(0.5, np.float32)])
        self.op("Mul", [half_x, inner], output=node.output[0])

    def run(self):
        model = onnx.shape_inference.infer_shapes(self.model, strict_mode=True, data_prop=True)
        shapes = {}
        for v in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
            dims = v.type.tensor_type.shape.dim
            if all(d.HasField("dim_value") for d in dims):
                shapes[v.name] = [d.dim_value for d in dims]
        for v in model.graph.initializer:
            shapes[v.name] = list(v.dims)
        constants = {v.name: NH.to_array(v) for v in self.model.graph.initializer}
        for node in self.model.graph.node:
            if node.op_type == "Einsum":
                self.einsum(node, shapes)
            elif node.op_type == "Gelu":
                self.gelu(node)
            elif node.op_type.startswith("Reduce") and node.op_type != "ReduceSum" and len(node.input) > 1:
                axes = constants.get(node.input[1])
                if axes is None:
                    raise ValueError("Reduction axes must be constant")
                attrs = {a.name: H.get_attribute_value(a) for a in node.attribute}
                noop = attrs.pop("noop_with_empty_axes", 0)
                if axes.size == 0 and noop:
                    self.op("Identity", [node.input[0]], output=node.output[0])
                else:
                    # Empty axes in opset 13 means reduce all dimensions.
                    if axes.size:
                        attrs["axes"] = axes.tolist()
                    self.op(node.op_type, [node.input[0]], output=node.output[0], **attrs)
            else:
                # These extra attributes were added after opset 13. Only their
                # default semantics are compatible; never discard a nondefault.
                remove = {"Reshape": {"allowzero": 0}, "Shape": {"start": 0},
                          "ScatterND": {"reduction": b"none"},
                          "Cast": {"saturate": 1},
                          "BatchNormalization": {"training_mode": 0}}.get(node.op_type, {})
                attrs = []
                for a in node.attribute:
                    if a.name in remove:
                        if H.get_attribute_value(a) != remove[a.name]:
                            raise ValueError("Cannot lower {}.{}".format(node.op_type, a.name))
                    else:
                        attrs.append(a)
                del node.attribute[:]
                node.attribute.extend(attrs)
                self.nodes.append(node)
        del self.model.graph.node[:]
        self.model.graph.node.extend(self.nodes)
        # The contract is deliberately narrow and checked below. This is a
        # semantic lowering, not just a change to the opset version number.
        del self.model.opset_import[:]
        self.model.opset_import.append(H.make_opsetid("", 13))
        self.model.ir_version = 7
        del self.model.graph.value_info[:]
        onnx.checker.check_model(self.model, full_check=True)
        return self.model


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])


def prepare(source, output, registry):
    output.mkdir(parents=True, exist_ok=True)
    original_manifest = json.loads((source / "manifest.json").read_text())
    manifest = {}
    for name in registry["models"]:
        path = source / (name + ".onnx")
        expected = registry["models"][name]["sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Source checksum mismatch: " + name)
        original = onnx.load(path)
        # Fold constant Range/NonZero/Scatter scaffolding before lowering. In
        # TensorRT 8.2 NonZero is unsupported, even for a constant expression.
        model, ok = simplify(original, check_n=0)
        if not ok:
            raise ValueError("Simplification failed: " + name)
        model = Lowerer(model).run()
        model, ok = simplify(model, check_n=0)
        if not ok:
            raise ValueError("Post-lowering simplification failed: " + name)
        forbidden = {"Einsum", "Gelu", "NonZero", "Loop", "Scan", "If"}
        remaining = forbidden.intersection(n.op_type for n in model.graph.node)
        if remaining:
            raise ValueError("Unsupported TensorRT 8.2 operations: {}".format(remaining))
        onnx.checker.check_model(model, full_check=True)
        source_session = session(original.SerializeToString())
        target_session = session(model.SerializeToString())
        shape = original_manifest[name]["input_shape"]
        samples = [np.zeros(shape, np.float32)] + [
            np.random.default_rng(s).normal(0, 0.3, shape).astype(np.float32) for s in range(3)]
        references, max_error = [], 0.0
        for x in samples:
            ref = source_session.run(None, {source_session.get_inputs()[0].name: x})[0]
            actual = target_session.run(None, {target_session.get_inputs()[0].name: x})[0]
            np.testing.assert_allclose(actual, ref, atol=1e-3, rtol=1e-4)
            max_error = max(max_error, float(np.max(np.abs(actual - ref))))
            references.append(ref)
        onnx.save(model, output / (name + ".onnx"))
        # Target-side tests compare TensorRT/CUDA output with the ORIGINAL
        # framework export, rather than trusting a successfully built engine.
        np.savez(output / (name + ".reference.npz"), x=np.stack(samples), y=np.stack(references))
        manifest[name] = dict(original_manifest[name], source_sha256=expected,
                              onnx_sha256=hashlib.sha256((output / (name + ".onnx")).read_bytes()).hexdigest(),
                              lowering_max_abs_diff=max_error, source_commit=registry["commit"],
                              onnx_opset=13, reference_file=name + ".reference.npz")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("{}: ONNX 13, reference check OK, max error {:.3g}".format(name, max_error), flush=True)
        del source_session, target_session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=Path(__file__).with_name("artifacts.json"))
    ap.add_argument("--source", type=Path, help="existing exported model directory; otherwise download pinned kit")
    ap.add_argument("--custom", action="store_true", help="prepare your own exported ONNX models, preserving trained metadata")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    registry = json.loads(args.registry.read_text())
    if args.custom:
        if args.source is None:
            ap.error("--custom requires --source")
        custom_manifest = json.loads((args.source / "manifest.json").read_text())
        registry = dict(commit=None, models={})
        for name, info in custom_manifest.items():
            if not name.replace("_", "").isalnum() or not isinstance(info.get("trained"), bool):
                ap.error("Custom model names must be alphanumeric/underscore and have boolean trained metadata")
            registry["models"][name] = dict(sha256=hashlib.sha256((args.source / (name + ".onnx")).read_bytes()).hexdigest())
    source = args.source or args.output.parent / "source_models"
    if not args.source:
        source.mkdir(parents=True, exist_ok=True)
        base = "https://raw.githubusercontent.com/rombaldivia/NestSAR/{}/edge_models/".format(registry["commit"])
        for filename in ["manifest.json"] + [n + ".onnx" for n in registry["models"]]:
            with urllib.request.urlopen(base + filename, timeout=120) as response:
                (source / filename).write_bytes(response.read())
    prepare(source, args.output, registry)


if __name__ == "__main__":
    main()

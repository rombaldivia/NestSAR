"""Static TensorRT 8.2 dtype compatibility for the shipped FP32 edge graphs.

The opset alone is insufficient: Gather/ScatterND do not accept BOOL data,
and reductions do not accept integer data. Masks and bounded population counts
can be represented exactly in FP32, then restored to their ONNX output types.
Constraints are from onnx/onnx-tensorrt 8.2-GA, commit
f42daeee49f2517a954c5601f0f76bef9ed94b62, docs/operators.md.
"""
import math

import onnx
from onnx import helper as H, numpy_helper as NH

F = onnx.TensorProto.FLOAT
H16 = onnx.TensorProto.FLOAT16
I32 = onnx.TensorProto.INT32
I64 = onnx.TensorProto.INT64
B = onnx.TensorProto.BOOL

FLOATS = {F, H16}
NUMERIC = FLOATS | {I32, I64}
ALL = NUMERIC | {B}

# This is deliberately restricted to operators in the shipped, static graphs.
# Input positions listed here are data tensors, not shape/index parameters.
DATA_TYPES = {
    **{k: (FLOATS, (0,)) for k in (
        "BatchNormalization", "Conv", "Gemm", "GlobalAveragePool", "MatMul",
        "MaxPool", "PRelu", "ReduceMax", "ReduceMean", "ReduceSum", "ReduceSumSquare",
        "Sigmoid", "Softmax", "Sqrt", "Tanh", "Dropout")},
    **{k: (NUMERIC, (0,)) for k in (
        "Abs", "Add", "Div", "Greater", "Max", "Mul", "Neg", "Relu", "Sub")},
    **{k: (ALL, (0,)) for k in (
        "Cast", "Concat", "Flatten", "Identity", "Reshape", "Slice", "Squeeze", "Transpose", "Unsqueeze")},
    "And": ({B}, (0, 1)),
    "Gather": (NUMERIC, (0,)),
    "ScatterND": (NUMERIC, (0, 2)),
}


def metadata(model):
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    types, shapes = {}, {}
    for v in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info):
        t = v.type.tensor_type
        types[v.name] = t.elem_type
        if all(d.HasField("dim_value") for d in t.shape.dim):
            shapes[v.name] = [d.dim_value for d in t.shape.dim]
    for v in inferred.graph.initializer:
        types[v.name] = v.data_type
        shapes[v.name] = list(v.dims)
    return types, shapes


def lower_masks(model):
    types, shapes = metadata(model)
    producers = {y: n for n in model.graph.node for y in n.output}
    existing = set(types)
    rewritten = []
    serial = 0

    def new_name():
        nonlocal serial
        while True:
            serial += 1
            name = "__trt82_mask_{}".format(serial)
            if name not in existing:
                existing.add(name)
                return name

    def cast(value, dtype):
        name = new_name()
        rewritten.append(H.make_node("Cast", [value], [name], to=dtype))
        return name

    for node in model.graph.node:
        dtype = types.get(node.input[0]) if node.input else None
        if node.op_type in ("Gather", "ScatterND") and dtype == B:
            original_output = node.output[0]
            node.input[0] = cast(node.input[0], F)
            if node.op_type == "ScatterND":
                node.input[2] = cast(node.input[2], F)
            node.output[0] = new_name()
            rewritten.append(node)
            rewritten.append(H.make_node("Cast", [node.output[0]], [original_output], to=B))
        elif node.op_type == "ReduceSum" and dtype in (I32, I64):
            parent = producers.get(node.input[0])
            if parent is None or parent.op_type != "Cast" or types.get(parent.input[0]) != B:
                raise ValueError("Cannot safely lower an arbitrary integer reduction: " + node.name)
            # Sum of at most 2**24 booleans is exactly representable in FP32.
            # Use the whole tensor size as a conservative bound.
            shape = shapes.get(parent.input[0])
            if shape is None or math.prod(shape) > 2 ** 24:
                raise ValueError("Boolean count exceeds the exact FP32 bound: " + node.name)
            original_output = node.output[0]
            node.input[0] = cast(parent.input[0], F)
            node.output[0] = new_name()
            rewritten.append(node)
            rewritten.append(H.make_node("Cast", [node.output[0]], [original_output], to=dtype))
        else:
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


def audit(model):
    types, _ = metadata(model)
    errors = []
    for n in model.graph.node:
        if n.op_type == "Constant":
            continue
        policy = DATA_TYPES.get(n.op_type)
        if policy is None:
            errors.append("Unreviewed operator: " + n.op_type)
            continue
        allowed, positions = policy
        for i in positions:
            if i < len(n.input) and types.get(n.input[i]) not in allowed:
                dtype = types.get(n.input[i], 0)
                errors.append("{} {} input {} has unsupported {}".format(
                    n.op_type, n.name, i, onnx.TensorProto.DataType.Name(dtype)))
    if errors:
        raise ValueError("TensorRT 8.2 dtype audit failed:\n" + "\n".join(errors))

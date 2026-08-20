#!/usr/bin/env python3
from __future__ import annotations

import ast
import os
import py_compile
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "run.py"
REAL_EXECV = os.execv

_HELPERS = r'''
def _fjh_single(module, x, tag):
    # Learned full-joint attention for a CrossStream hint.
    # Expected layout: [B, T, J=25, S=4, D=128].
    if not hasattr(x, "ndim") or x.ndim != 5:
        return x
    if x.shape[-3] != 25 or x.shape[-2] != 4:
        return x

    z = jnp.transpose(x, (0, 1, 3, 2, 4))  # [B,T,S,J,D]
    d_model = int(z.shape[-1])
    attn_dim = 64
    heads = 4
    head_dim = attn_dim // heads

    # Parameter-free RMS normalization keeps the residual branch stable.
    zn = z * jax.lax.rsqrt(jnp.mean(jnp.square(z), axis=-1, keepdims=True) + 1.0e-6)

    # Static T is part of the parameter name so one shared CrossStream module
    # can still own distinct L2/L3/L4 full-joint mixers.
    tlen = int(z.shape[1])
    pfx = f"{tag}_t{tlen}"
    init = nn.initializers.xavier_uniform()
    wq = module.param(f"{pfx}_q_kernel", init, (d_model, attn_dim))
    wk = module.param(f"{pfx}_k_kernel", init, (d_model, attn_dim))
    wv = module.param(f"{pfx}_v_kernel", init, (d_model, attn_dim))
    wo = module.param(f"{pfx}_o_kernel", init, (attn_dim, d_model))

    q = jnp.einsum("btsjd,da->btsja", zn, wq)
    k = jnp.einsum("btsjd,da->btsja", zn, wk)
    v = jnp.einsum("btsjd,da->btsja", zn, wv)

    q = q.reshape(q.shape[:-1] + (heads, head_dim))
    k = k.reshape(k.shape[:-1] + (heads, head_dim))
    v = v.reshape(v.shape[:-1] + (heads, head_dim))

    logits = jnp.einsum("btsjhd,btskhd->btshjk", q, k)
    logits = logits * (head_dim ** -0.5)
    weights = jax.nn.softmax(logits, axis=-1)
    ctx = jnp.einsum("btshjk,btskhd->btsjhd", weights, v)
    ctx = ctx.reshape(ctx.shape[:-2] + (attn_dim,))
    delta = jnp.einsum("btsja,ad->btsjd", ctx, wo)

    # gate_max=0.25, gate_init=0.05 exactly.
    gate_raw = module.param(
        f"{pfx}_gate_raw",
        nn.initializers.constant(-1.3862943611198906),
        (),
    )
    gate = 0.25 * jax.nn.sigmoid(gate_raw)
    y = z + gate * delta
    return jnp.transpose(y, (0, 1, 3, 2, 4))


def _fjh_mix_tree(module, out, prefix="full_joint"):
    if isinstance(out, tuple):
        return tuple(
            _fjh_mix_tree(module, value, f"{prefix}_{i}")
            for i, value in enumerate(out)
        )
    if isinstance(out, list):
        return [
            _fjh_mix_tree(module, value, f"{prefix}_{i}")
            for i, value in enumerate(out)
        ]
    if isinstance(out, dict):
        return {
            key: _fjh_mix_tree(module, value, f"{prefix}_{str(key).replace('-', '_')}")
            for key, value in out.items()
        }
    return _fjh_single(module, out, prefix)
'''


class _WrapReturns(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_FunctionDef(self, node):
        return node

    def visit_AsyncFunctionDef(self, node):
        return node

    def visit_ClassDef(self, node):
        return node

    def visit_Lambda(self, node):
        return node

    def visit_Return(self, node):
        node = self.generic_visit(node)
        if node.value is not None:
            node.value = ast.Call(
                func=ast.Name(id="_fjh_mix_tree", ctx=ast.Load()),
                args=[ast.Name(id="self", ctx=ast.Load()), node.value],
                keywords=[],
            )
            self.count += 1
        return node


def _patch_fulljoint(source: str) -> tuple[str, int, bool]:
    tree = ast.parse(source)
    helpers = ast.parse(_HELPERS).body

    class_hits = 0
    return_hits = 0
    compact_added = False
    insert_at = None

    for i, node in enumerate(tree.body):
        if isinstance(node, ast.ClassDef) and node.name == "CrossStreamMultiScaleHint":
            class_hits += 1
            insert_at = i

            call_fn = None
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__call__":
                    call_fn = item
                    break
            if call_fn is None:
                raise RuntimeError("CrossStreamMultiScaleHint.__call__ not found")

            dec_text = [ast.unparse(d) for d in call_fn.decorator_list]
            has_compact = any(text.endswith("compact") or text == "compact" for text in dec_text)
            if not has_compact:
                other_compact = False
                for item in node.body:
                    if item is call_fn or not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if any(
                        ast.unparse(d).endswith("compact") or ast.unparse(d) == "compact"
                        for d in item.decorator_list
                    ):
                        other_compact = True
                        break
                if other_compact:
                    raise RuntimeError(
                        "CrossStreamMultiScaleHint has another @compact method; cannot safely add learned full-joint parameters."
                    )
                call_fn.decorator_list.insert(
                    0,
                    ast.Attribute(
                        value=ast.Name(id="nn", ctx=ast.Load()),
                        attr="compact",
                        ctx=ast.Load(),
                    ),
                )
                compact_added = True

            wrapper = _WrapReturns()
            call_fn.body = [wrapper.visit(stmt) for stmt in call_fn.body]
            return_hits += wrapper.count

    if class_hits != 1:
        raise RuntimeError(f"Expected exactly one CrossStreamMultiScaleHint class, found {class_hits}")
    if return_hits < 1:
        raise RuntimeError("No CrossStreamMultiScaleHint return was wrapped")
    if insert_at is None:
        raise RuntimeError("Could not locate helper insertion point")

    tree.body[insert_at:insert_at] = helpers
    ast.fix_missing_locations(tree)
    patched = ast.unparse(tree) + "\n"
    return patched, return_hits, compact_added


def _patched_execv(path, argv):
    assembled = Path(argv[-1])
    source = assembled.read_text(encoding="utf-8")

    # Existing J25 manual-backward axis repair.
    needle = "    grad_cross = cross_vjp_exe("
    if source.count(needle) != 1:
        raise RuntimeError(
            "J25 cross-VJP axis patch guard failed; "
            f"grad_cross matches={source.count(needle)}"
        )
    replacement = """    # J25 cotangent axis repair for cross-stream VJP.\n    # [B,T,S,J,D] -> [B,T,J,S,D]\n    cot_all_hints = tuple(\n        jnp.transpose(c, (0, 1, 3, 2, 4))\n        for c in cot_all_hints\n    )\n\n    grad_cross = cross_vjp_exe("""
    source = source.replace(needle, replacement, 1)

    source = source.replace(
        "NestSAR_HOPE_XSTREAM_J25_T32_D128_XSUB_E",
        "NestSAR_HOPE_FULLJOINT_J25_T32_D128_XSUB_E",
    )
    source = source.replace(
        "NESTSAR-HOPE-XSTREAM-J25",
        "NESTSAR-HOPE-FULLJOINT-J25",
    )

    source, wrapped_returns, compact_added = _patch_fulljoint(source)

    assembled = assembled.with_name(assembled.name.replace("XStream_J25", "FullJoint_J25"))
    assembled.write_text(source, encoding="utf-8")
    py_compile.compile(str(assembled), doraise=True)

    required = [
        "def _fjh_single",
        "def _fjh_mix_tree",
        "full_joint",
        "btshjk",
        "0.25 * jax.nn.sigmoid",
    ]
    missing = [marker for marker in required if marker not in source]
    if missing:
        raise RuntimeError("FullJoint source audit failed: " + repr(missing))

    print("=" * 108)
    print("FULLJOINT-HIERARCHY PATCH: PASS")
    print("Existing L1 joint mixer: learned full 25-joint interaction")
    print("L2/L3/L4 mixers:        learned D64/H4 full 25-joint interaction inside CrossStream hints")
    print("Residual gate:           max=0.25 init=0.05")
    print("Wrapped CrossStream returns:", wrapped_returns)
    print("Added @nn.compact:          ", compact_added)
    print("Cross-VJP axis repair:      PASS [B,T,S,J,D] -> [B,T,J,S,D]")
    print("Runtime syntax audit:       PASS")
    print("Assembled FullJoint source: ", assembled)
    print("=" * 108)
    os.sys.stdout.flush()

    os.execv = REAL_EXECV
    REAL_EXECV(path, [argv[0], "-u", str(assembled)])


os.execv = _patched_execv
runpy.run_path(str(BASE), run_name="__main__")

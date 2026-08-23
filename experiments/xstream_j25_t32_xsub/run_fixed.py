#!/usr/bin/env python3
from __future__ import annotations

import os
import py_compile
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "run.py"
REAL_EXECV = os.execv


def _patched_execv(path, argv):
    assembled = Path(argv[-1])
    source = assembled.read_text(encoding="utf-8")

    # After the first J25 backward patch, cot_all_hints is assembled as
    # [B, T, S=4, J=25, D].  CrossStreamMultiScaleHint itself returns
    # [B, T, J=25, S=4, D], so jax.vjp requires the cotangent to have
    # that exact axis order.
    needle = "    grad_cross = cross_vjp_exe("
    if source.count(needle) != 1:
        raise RuntimeError(
            "J25 cross-VJP axis patch guard failed; "
            f"grad_cross matches={source.count(needle)}"
        )

    replacement = """    # J25 cotangent axis repair for cross-stream VJP.\n    # [B,T,S,J,D] -> [B,T,J,S,D]\n    cot_all_hints = tuple(\n        jnp.transpose(c, (0, 1, 3, 2, 4))\n        for c in cot_all_hints\n    )\n\n    grad_cross = cross_vjp_exe("""

    source = source.replace(needle, replacement, 1)

    marker = "jnp.transpose(c, (0, 1, 3, 2, 4))"
    if marker not in source:
        raise RuntimeError("J25 cross-VJP repair marker missing after patch.")

    assembled.write_text(source, encoding="utf-8")
    py_compile.compile(str(assembled), doraise=True)

    print("J25 cross-VJP axis repair: PASS  [B,T,S,J,D] -> [B,T,J,S,D]")
    print("Patched assembled source:       ", assembled)
    print("Launching repaired J25 process...")
    os.sys.stdout.flush()

    os.execv = REAL_EXECV
    REAL_EXECV(path, argv)


os.execv = _patched_execv
runpy.run_path(str(BASE), run_name="__main__")

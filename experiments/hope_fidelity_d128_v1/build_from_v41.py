#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build NestSAR-HOPE-Fidelity-D128-v1 from the exact audited v4.1 source tree.

This builder does NOT mutate the production v4.1 files. It creates:
  - nestsar_hope_fidelity_d128_v1_core.py
  - nestsar_hope_fidelity_d128_v1_train.py

Forward changes relative to audited v4.1:
  1) causal depthwise temporal convolution, k=4
  2) existing bounded self-referential K/V/Q/eta/alpha/main-memory
  3) sequential forward CMS: f1 -> f2 -> f4 -> f8
     with each CMS block D128 -> 32 -> D128

The existing D128/T16 skeleton frontend, four streams, SASM, L1/L2/L3
hierarchy, Short-L3 fix, H4/L4, classifier and stream fusion remain.

Audited generated parameter count for D128/M64/R32/T16: 2,083,236.
Softmax attention: NONE.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

EXPECTED_BASE_PARAMS = 2_033_988
EXPECTED_NEW_PARAMS = 2_083_236

BASE_CORE = "nestsar_hope_fullselfref_v3_3_shortl3fix.py"
BASE_TRAIN = "nestsar_hope_v4_1_cms_dmgd_l2_shortl3fix.py"
NEW_CORE = "nestsar_hope_fidelity_d128_v1_core.py"
NEW_TRAIN = "nestsar_hope_fidelity_d128_v1_train.py"


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {n}\n{old}")
    return text.replace(old, new, 1)


def build_core(root: Path) -> Path:
    src = root / BASE_CORE
    dst = root / NEW_CORE
    if not src.is_file():
        raise FileNotFoundError(src)

    s = src.read_text(encoding="utf-8")

    s = replace_exact(
        s,
        'MODEL_ID = "nestsar_m4_sasm_l3fix_hope_fullselfref_v3_shortl3fix"',
        'MODEL_ID = "nestsar_hope_fidelity_d128_v1_core"',
        "core model id",
    )
    s = replace_exact(
        s,
        'MODEL_MODE = "NestSAR_M4_SASM_L3Fix_HOPE_FullSelfRef_v3_ShortL3Fix"',
        'MODEL_MODE = "NestSAR_HOPE_Fidelity_D128_v1_Core"',
        "core model mode",
    )
    s = replace_exact(
        s,
        f"EXPECTED_PARAMS = {EXPECTED_BASE_PARAMS:_}",
        f"EXPECTED_PARAMS = {EXPECTED_NEW_PARAMS:_}",
        "core parameter guard",
    )

    # ------------------------------------------------------------------
    # HOPE local temporal convolution, causal k=4.
    # ------------------------------------------------------------------
    old = '''        h = nn.LayerNorm(name="input_norm")(x)

        # One outer-loop learned adapter puts the existing model representation
        # into the same compact per-head space used by all self-ref memories.
        u = nn.Dense(
'''

    new = '''        h = nn.LayerNorm(name="input_norm")(x)

        # HOPE local temporal mixing: causal depthwise convolution, window 4.
        # No softmax attention and no T x T attention matrix.
        local_in = jnp.pad(
            h,
            ((0, 0), (3, 0), (0, 0)),
            mode="constant",
        )
        local = nn.Conv(
            features=self.model_dim,
            kernel_size=(4,),
            strides=(1,),
            padding="VALID",
            feature_group_count=self.model_dim,
            use_bias=True,
            kernel_init=nn.initializers.lecun_normal(),
            name="hope_local_conv",
        )(local_in)
        local = jax.nn.silu(local)
        local_gate_logit = self.param(
            "hope_local_conv_gate_logit",
            lambda key, shape: jnp.full(
                shape, _safe_logit(0.05), dtype=jnp.float32
            ),
            (1,),
        )
        local_gate = jax.nn.sigmoid(local_gate_logit)[0]
        h = nn.LayerNorm(name="hope_local_conv_output_norm")(
            h + local_gate * local
        )

        # Existing self-referential memory input.
        u = nn.Dense(
'''
    s = replace_exact(s, old, new, "local convolution")

    # ------------------------------------------------------------------
    # Replace old single FF tail by sequential CMS f1/f2/f4/f8.
    # Four 128->32->128 blocks deliberately preserve approximately the
    # dense FLOPs/token of the old 128->128->128 FF tail.
    # ------------------------------------------------------------------
    start_marker = "        # Keep the proven M4 FF/residual tail unchanged."
    end_marker = "        # Compatibility with existing M4G-H4 aggregation."
    start = s.find(start_marker)
    end = s.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError("CMS replacement markers not found")

    cms = '''        # HOPE sequential Continuum Memory System.
        # Parameter update frequencies are assigned by the outer trainer.
        output = x

        for cms_frequency in (1, 2, 4, 8):
            cms_h = nn.LayerNorm(
                name=f"cms_f{cms_frequency}_norm"
            )(output)
            cms_h = nn.Dense(
                features=32,
                use_bias=True,
                kernel_init=nn.initializers.xavier_uniform(),
                name=f"cms_f{cms_frequency}_in",
            )(cms_h)
            cms_h = jax.nn.silu(cms_h)
            cms_h = nn.Dropout(
                rate=self.dropout,
                name=f"cms_f{cms_frequency}_dropout",
            )(cms_h, deterministic=not training)
            cms_delta = nn.Dense(
                features=self.model_dim,
                use_bias=True,
                kernel_init=nn.initializers.xavier_uniform(),
                name=f"cms_f{cms_frequency}_out",
            )(cms_h)
            cms_gate_logit = self.param(
                f"cms_f{cms_frequency}_gate_logit",
                lambda key, shape: jnp.full(
                    shape, _safe_logit(0.05), dtype=jnp.float32
                ),
                (1,),
            )
            cms_gate = jax.nn.sigmoid(cms_gate_logit)[0]
            output = output + cms_gate * cms_delta

        output = nn.LayerNorm(name="output_norm")(output)

'''
    s = s[:start] + cms + s[end:]

    # Banner corrections.
    s = s.replace(
        'print("Softmax attention:    NONE")',
        'print("HOPE local conv:      causal temporal window=4")\n'
        'print("HOPE memory:          self-referential K/V/Q/eta/alpha/main-memory")\n'
        'print("Sequential CMS:       f1 -> f2 -> f4 -> f8")\n'
        'print("CMS bottleneck:       32")\n'
        'print("Softmax attention:    NONE")',
        1,
    )
    s = s.replace(
        'print("GCN/GNN/CNN/TCN:      NONE")',
        'print("GCN/GNN:              NONE")\n'
        'print("CNN/TCN backbone:     NONE")\n'
        'print("Local temporal conv:  depthwise causal k=4")',
        1,
    )
    s = s.replace(
        'print("GFLOPs:               MEASURE before paper claims")',
        'print("GFLOPs:               PROFILE EXACTLY before paper claims")',
        1,
    )

    compile(s, str(dst), "exec")
    dst.write_text(s, encoding="utf-8")
    return dst


def build_train(root: Path) -> Path:
    src = root / BASE_TRAIN
    dst = root / NEW_TRAIN
    if not src.is_file():
        raise FileNotFoundError(src)

    s = src.read_text(encoding="utf-8")
    s = replace_exact(
        s,
        "import nestsar_hope_fullselfref_v3_3_shortl3fix as v3",
        "import nestsar_hope_fidelity_d128_v1_core as v3",
        "trainer core import",
    )
    s = replace_exact(
        s,
        'MODEL_ID = "nestsar_hope_v4_1_cms_dmgd_l2_shortl3fix"',
        'MODEL_ID = "nestsar_hope_fidelity_d128_v1"',
        "trainer model id",
    )
    s = replace_exact(
        s,
        'MODEL_MODE = "NestSAR_HOPE_v4_1_CMS_DMGD_L2_ShortL3Fix"',
        'MODEL_MODE = "NestSAR_HOPE_Fidelity_D128_v1"',
        "trainer model mode",
    )

    # Explicit CMS parameter names take priority over enclosing temporal level.
    pattern = re.compile(
        r"def _path_to_tier\(path_tuple\) -> str:\n.*?(?=\ndef make_tier_labels)",
        flags=re.DOTALL,
    )
    replacement = '''def _path_to_tier(path_tuple) -> str:
    text = "/".join(str(x) for x in path_tuple).lower()

    # Forward CMS memories: explicit HOPE continuum periods.
    if "cms_f8" in text:
        return "consolidate"
    if "cms_f4" in text:
        return "slow"
    if "cms_f2" in text:
        return "medium"
    if "cms_f1" in text:
        return "fast"

    # Existing NestSAR hierarchy for all remaining parameters.
    if "l4_slow_controller" in text:
        return "consolidate"
    if "l3_clip_memory" in text:
        return "slow"
    if "l2_chunk_memory" in text:
        return "medium"
    if "l1_frame_memory" in text:
        return "fast"
    return "fast"
'''
    s, n = pattern.subn(replacement, s, count=1)
    if n != 1:
        raise RuntimeError(f"trainer tier mapping replacement count={n}")

    # Add hard runtime parameter guard.
    old = '''    params = variables["params"]
    counts, leaves = tier_parameter_counts(params)'''
    new = f'''    params = variables["params"]
    parameter_count = sum(
        int(x.size) for x in jax.tree_util.tree_leaves(params)
    )
    if parameter_count != {EXPECTED_NEW_PARAMS:_}:
        raise RuntimeError(
            f"PARAMETER GUARD FAIL: {{parameter_count:,}} != {EXPECTED_NEW_PARAMS:,}"
        )
    print(f"PARAMETER GUARD: PASS ({{parameter_count:,}})")
    counts, leaves = tier_parameter_counts(params)'''
    s = replace_exact(s, old, new, "trainer parameter guard")

    s = s.replace(
        "NESTSAR-HOPE v4.1 — FULL SELF-REFERENCE + OUTER CMS + DMGD-L2 + SHORT-L3 FIX",
        "NESTSAR-HOPE-FIDELITY D128 v1 — SELF-MODIFYING MEMORY + SEQUENTIAL CMS",
    )

    compile(s, str(dst), "exec")
    dst.write_text(s, encoding="utf-8")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()

    core = build_core(root)
    train = build_train(root)

    print("=" * 110)
    print("NESTSAR-HOPE-FIDELITY D128 v1 GENERATED")
    print("=" * 110)
    print("Root:             ", root)
    print("Core:             ", core.name)
    print("Trainer:          ", train.name)
    print("Expected params:  ", f"{EXPECTED_NEW_PARAMS:,}")
    print("Frames / D / M:    16 / 128 / 64")
    print("Local conv:        causal depthwise k=4")
    print("Self-ref memory:   K/V/Q/eta/alpha/main-memory")
    print("Forward CMS:       f1 -> f2 -> f4 -> f8")
    print("Softmax attention: NONE")
    print("Initialization:    scratch when launched with --resume none")
    print("GFLOPs:            MUST BE PROFILED EXACTLY")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

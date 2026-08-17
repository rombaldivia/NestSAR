#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""TPU compatibility shim on top of run_universal_v2.

v2 normalizes the exact audited v4.1 bundle semantically. This v3 shim adds
one narrowly-scoped compatibility patch: the original v4.1 trainer contains a
hard GPU-only runtime guard inside run_experiment(). The universal worker
already verifies the requested accelerator backend and device count before
training, so for TPU we allow jax.default_backend() == "tpu" through that old
guard while preserving the original failure for CPU/other backends.

No model, loss, optimizer, schedule, data pipeline, parameter, or checkpoint
semantics are changed by this patch.
"""

import re
from pathlib import Path

import run_universal_v2 as v2

_BASE_NORMALIZE = v2.normalize_extracted_v41_robust


def normalize_extracted_v41_tpu(runtime: Path) -> None:
    # First perform all exact-source integrity checks and v2 normalizations.
    _BASE_NORMALIZE(runtime)

    core_path = runtime / "nestsar.py"
    core = core_path.read_text(encoding="utf-8")

    marker = "Backend TPU autorizado por runtime universal."
    if marker in core:
        print("[v4.1 PATCH] TPU backend guard already normalized")
        return

    # The exact v4.1 source has a hard GPU-only guard in run_experiment().
    # Replace only the actual raise line. Because it remains nested inside the
    # original `backend != gpu` branch, GPU behavior is untouched; TPU is
    # accepted; CPU/other backends still fail.
    pattern = re.compile(
        r'(?m)^(?P<i>[ \t]*)raise RuntimeError\("JAX no está usando GPU\. Activa GPU en Kaggle\."\)[ \t]*$'
    )
    m = pattern.search(core)
    if not m:
        raise RuntimeError(
            "Could not locate exact v4.1 GPU-only runtime guard for TPU normalization"
        )

    i = m.group("i")
    replacement = (
        f'{i}if jax.default_backend() != "tpu":\n'
        f'{i}    raise RuntimeError("JAX no está usando GPU/TPU compatible con esta corrida.")\n'
        f'{i}log("{marker}")'
    )
    core = core[:m.start()] + replacement + core[m.end():]

    core_path.write_text(core, encoding="utf-8")
    compile(core, str(core_path), "exec")
    print("[v4.1 PATCH] legacy GPU-only runtime guard -> GPU or verified TPU")


# run_universal_v2 imported run_universal as `base`; replace the normalizer that
# base.main() will call with this TPU-compatible version.
v2.base.normalize_extracted_v41 = normalize_extracted_v41_tpu


if __name__ == "__main__":
    raise SystemExit(v2.base.main())

#!/usr/bin/env python3
"""Process-isolated TPU4 wrapper for the validated Attention-Lite trainer.

This module intentionally does NOT import JAX. It is launched in a process whose
TPU visibility has already been restricted to exactly four chips. The normal
Attention-Lite trainer then generates the same validated model/training source,
with only runtime-topology guards changed from TPU8 to the four devices visible
inside this process.

Model math, seed patching, optimizer schedule, regularization, EMA, and all paper
architecture guards remain owned by ``experiments.attention_lite_v1.trainer``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import trainer as tr
from .paths import make_run_paths, validate_protocol, validate_seed

EXPECTED_ACTIVE_TPUS = 4

# The wrapper process itself must never initialize JAX. Changing this constant is
# enough for the launcher's batch-divisibility and manifest guards; the generated
# trainer performs the actual JAX/TPU initialization in its child process.
tr.EXPECTED_TPU_DEVICES = EXPECTED_ACTIVE_TPUS

_ORIGINAL_PATCH_OUTPUT = tr._patch_output
_ORIGINAL_VERIFY_RESULT = tr._verify_result


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"TPU4 runtime patch expected exactly one {label}; found {count}"
        )
    return source.replace(old, new, 1)


def _patch_tpu4_runtime(source: str) -> tuple[str, dict[str, int]]:
    """Patch only runtime topology/storage metadata after standard trainer patches."""
    counts: dict[str, int] = {}
    protocol = validate_protocol(os.environ.get("NESTSAR_PROTOCOL", "xsub"))

    # The canonical all-in-one extracts embedded modules under one shared directory.
    # Two concurrent OS processes must not write that directory at the same time.
    old_root = 'ROOT = Path("/kaggle/working/NestSAR_HOPE_FIDELITY_UNIVERSAL")'
    new_root = (
        f'ROOT = Path("/kaggle/working/'
        f'NestSAR_HOPE_FIDELITY_UNIVERSAL_{protocol.upper()}_PROC4")'
    )
    source = _replace_once(source, old_root, new_root, "protocol bundle extraction ROOT")
    counts["protocol_root"] = 1

    # In this process jax.devices() MUST already expose exactly four devices because
    # the parent launcher restricted TPU visibility before Python/JAX startup.
    guard_pattern = (
        r"if len\(\s*DEVICES\s*\) != 8:\s*\n\s*"
        r"raise RuntimeError\(\s*f\"Expected 8 TPU devices; found "
        r"\{len\(DEVICES\)\}\"\s*\)"
    )
    guard_replacement = (
        "if len(DEVICES) != 4:\n\n"
        "    raise RuntimeError(\n"
        "        f\"Expected 4 process-visible TPU devices; found {len(DEVICES)}\"\n"
        "    )"
    )
    source, n = re.subn(
        guard_pattern,
        guard_replacement,
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError("Could not patch canonical TPU8 device guard to isolated TPU4")
    counts["device_guard"] = n

    # Record the actual process-visible topology in result.json. No physical/global
    # TPU IDs are selected inside the generated source; it simply uses all devices
    # visible to this isolated process, avoiding the in-process global-id 4..7 bug.
    old_local_batch = '''    "local_batch":
        GLOBAL_BATCH
        //
        8,'''
    new_local_batch = '''    "local_batch":
        GLOBAL_BATCH
        //
        len(DEVICES),

    "active_tpu_devices":
        len(DEVICES),

    "device_ids":
        [int(device.id) for device in DEVICES],

    "runtime_topology":
        "isolated_process_tpu4",'''
    source = _replace_once(
        source,
        old_local_batch,
        new_local_batch,
        "result local_batch/runtime metadata",
    )
    counts["result_runtime_metadata"] = 1

    # Add a loud banner after the existing local-batch diagnostic.
    marker = '''print(
    "Local batch/TPU:",
    GLOBAL_BATCH
    //
    len(
        DEVICES
    )
)'''
    banner = marker + f'''

print(
    "ISOLATED TPU4 PROCESS | protocol={protocol.upper()} | visible device ids=",
    [int(device.id) for device in DEVICES],
)'''
    source = _replace_once(source, marker, banner, "TPU4 runtime banner")
    counts["runtime_banner"] = 1

    compile(source, "<Attention-Lite-isolated-TPU4>", "exec")
    return source, counts


def _patch_output_and_runtime(source: str, output: Path) -> tuple[str, int]:
    patched, output_count = _ORIGINAL_PATCH_OUTPUT(source, output)
    patched, runtime_counts = _patch_tpu4_runtime(patched)
    # _patch_output's public contract returns an integer count. Keep that contract;
    # the generated source itself and final result guards prove the runtime patch.
    if any(value != 1 for value in runtime_counts.values()):
        raise RuntimeError(f"Unexpected TPU4 runtime patch counts: {runtime_counts}")
    return patched, output_count


def _verify_result_tpu4(root: Path, protocol: str, seed: int, cfg: dict) -> None:
    _ORIGINAL_VERIFY_RESULT(root, protocol, seed, cfg)
    result_path = root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    active = int(result.get("active_tpu_devices", -1))
    if active != EXPECTED_ACTIVE_TPUS:
        raise RuntimeError(
            f"FINAL TPU4 GUARD FAILED: active_tpu_devices={active}, expected 4"
        )
    if result.get("runtime_topology") != "isolated_process_tpu4":
        raise RuntimeError(
            "FINAL TPU4 GUARD FAILED: result.json lacks isolated_process_tpu4 topology"
        )
    local_batch = int(result.get("local_batch", -1))
    expected_local = int(cfg["batch_size"] // EXPECTED_ACTIVE_TPUS)
    if local_batch != expected_local:
        raise RuntimeError(
            f"FINAL TPU4 GUARD FAILED: local_batch={local_batch}, expected {expected_local}"
        )


tr._patch_output = _patch_output_and_runtime
tr._verify_result = _verify_result_tpu4


def _paper_mode() -> bool:
    return os.environ.get("NESTSAR_PAPER_MODE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def main() -> int:
    protocol = validate_protocol(os.environ.get("NESTSAR_PROTOCOL", "xsub"))
    paper_mode = _paper_mode()
    seed = validate_seed(
        int(os.environ.get("NESTSAR_SEED", "128")),
        paper_mode=paper_mode,
    )
    tag = os.environ.get("NESTSAR_RUN_TAG", "").strip() or None
    root = make_run_paths(
        protocol,
        seed,
        base_dir=os.environ.get("NESTSAR_RUNS_ROOT", "/kaggle/working"),
        paper_mode=paper_mode,
        tag=tag,
        create=True,
    ).root

    print("=" * 108, flush=True)
    print("NESTSAR ATTENTION-LITE — ISOLATED TPU4 TRAINER WRAPPER", flush=True)
    print(f"Protocol: {protocol.upper()} | seed={seed}", flush=True)
    print("Expected process-visible TPUs: 4", flush=True)
    print(f"Output: {root}", flush=True)
    print("=" * 108, flush=True)

    # tr.main() generates the source and spawns the actual JAX process. This wrapper
    # has not imported JAX, so it does not acquire any TPU device itself.
    return int(tr.main())


if __name__ == "__main__":
    raise SystemExit(main())

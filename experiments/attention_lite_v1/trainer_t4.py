#!/usr/bin/env python3
"""Single-GPU T4 runtime adapter for validated Attention-Lite.

The canonical XSUB/XSET source remains the mathematical source of truth. This
adapter changes only the hardware runtime assumptions that are TPU-specific:

- require one visible GPU instead of eight TPU devices;
- use that one visible GPU as a one-device JAX mesh;
- isolate each protocol's embedded bundle extraction directory;
- record the GPU runtime topology in result.json.

All model math, optimizer tiers, seed patching, RegMask, EMA, validation, early
stopping, split counts, and paper architecture guards remain owned by trainer.py.

This module intentionally does not import JAX. The parent dual-T4 launcher sets
CUDA_VISIBLE_DEVICES before starting this process, and trainer.py then launches
the generated canonical source in a descendant process which inherits that GPU
visibility.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import trainer as tr
from .paths import validate_protocol

EXPECTED_ACTIVE_GPUS = 1

# trainer.py uses this constant for batch divisibility and run metadata. In the
# dual-T4 design each protocol process sees exactly one GPU.
tr.EXPECTED_TPU_DEVICES = EXPECTED_ACTIVE_GPUS

_ORIGINAL_PATCH_OUTPUT = tr._patch_output
_ORIGINAL_VERIFY_RESULT = tr._verify_result


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"T4 runtime patch expected exactly one {label}; found {count}"
        )
    return source.replace(old, new, 1)


def _patch_t4_runtime(source: str) -> tuple[str, dict[str, int]]:
    """Patch only canonical runtime topology/storage metadata for one T4."""
    counts: dict[str, int] = {}
    protocol = validate_protocol(os.environ.get("NESTSAR_PROTOCOL", "xsub"))
    physical_gpu = os.environ.get("NESTSAR_PHYSICAL_GPU", "unknown")

    # Both protocol jobs execute concurrently. The canonical all-in-one extracts
    # the embedded NestSAR bundle to one fixed directory, so give each worker its
    # own extraction root to prevent concurrent writes/import races.
    old_root = 'ROOT = Path("/kaggle/working/NestSAR_HOPE_FIDELITY_UNIVERSAL")'
    new_root = (
        f'ROOT = Path("/kaggle/working/'
        f'NestSAR_HOPE_FIDELITY_UNIVERSAL_{protocol.upper()}_T4")'
    )
    source = _replace_once(
        source,
        old_root,
        new_root,
        "protocol bundle extraction ROOT",
    )
    counts["protocol_root"] = 1

    # Replace the production TPU-only backend lock. The generated child sees only
    # one device because CUDA_VISIBLE_DEVICES is set before JAX is imported.
    backend_pattern = (
        r'if\s*\(\s*jax\.default_backend\(\)\s*!=\s*"tpu"\s*\)\s*:\s*\n'
        r'\s*raise RuntimeError\(\s*"This production probe is locked to TPU\."\s*\)'
    )
    backend_replacement = (
        'if (jax.default_backend() != "gpu"):\n\n'
        '    raise RuntimeError(\n'
        '        f"This T4 production run requires GPU; got {jax.default_backend()}."\n'
        '    )'
    )
    source, n = re.subn(
        backend_pattern,
        backend_replacement,
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError(
            "Could not patch canonical TPU backend guard to single-GPU T4"
        )
    counts["backend_guard"] = n

    device_pattern = (
        r'if len\(\s*DEVICES\s*\) != 8:\s*\n\s*'
        r'raise RuntimeError\(\s*f"Expected 8 TPU devices; found '
        r'\{len\(DEVICES\)\}"\s*\)'
    )
    device_replacement = (
        'if len(DEVICES) != 1:\n\n'
        '    raise RuntimeError(\n'
        '        f"Expected exactly 1 process-visible GPU; found {len(DEVICES)}"\n'
        '    )'
    )
    source, n = re.subn(
        device_pattern,
        device_replacement,
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError(
            "Could not patch canonical TPU8 device guard to single T4"
        )
    counts["device_guard"] = n

    # The canonical result hard-codes global_batch // 8 because TPU8 had eight
    # replicas. On one T4 the physical/local batch is the complete global batch.
    old_local_batch = '''    "local_batch":
        GLOBAL_BATCH
        //
        8,'''
    new_local_batch = f'''    "local_batch":
        GLOBAL_BATCH
        //
        len(DEVICES),

    "active_gpu_devices":
        len(DEVICES),

    "visible_device_ids":
        [int(device.id) for device in DEVICES],

    "physical_gpu_assignment":
        {physical_gpu!r},

    "runtime_topology":
        "isolated_single_t4",'''
    source = _replace_once(
        source,
        old_local_batch,
        new_local_batch,
        "result local_batch/GPU metadata",
    )
    counts["result_runtime_metadata"] = 1

    # Add a loud runtime banner after the canonical local-batch diagnostic.
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
    "DUAL-T4 RUNTIME | protocol={protocol.upper()} | "
    "physical GPU={physical_gpu} | process-visible device ids=",
    [int(device.id) for device in DEVICES],
)'''
    source = _replace_once(
        source,
        marker,
        banner,
        "single-T4 runtime banner",
    )
    counts["runtime_banner"] = 1

    compile(source, "<Attention-Lite-single-T4>", "exec")
    return source, counts


def _patch_output_and_runtime(source: str, output: Path) -> tuple[str, int]:
    patched, output_count = _ORIGINAL_PATCH_OUTPUT(source, output)
    patched, runtime_counts = _patch_t4_runtime(patched)
    if any(value != 1 for value in runtime_counts.values()):
        raise RuntimeError(f"Unexpected T4 runtime patch counts: {runtime_counts}")
    return patched, output_count


def _verify_result_t4(root: Path, protocol: str, seed: int, cfg: dict) -> None:
    _ORIGINAL_VERIFY_RESULT(root, protocol, seed, cfg)

    result_path = root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    active = int(result.get("active_gpu_devices", -1))
    if active != EXPECTED_ACTIVE_GPUS:
        raise RuntimeError(
            f"FINAL T4 GUARD FAILED: active_gpu_devices={active}, expected 1"
        )

    if result.get("runtime_topology") != "isolated_single_t4":
        raise RuntimeError(
            "FINAL T4 GUARD FAILED: result.json lacks isolated_single_t4 topology"
        )

    local_batch = int(result.get("local_batch", -1))
    expected_local = int(cfg["batch_size"])
    if local_batch != expected_local:
        raise RuntimeError(
            f"FINAL T4 GUARD FAILED: local_batch={local_batch}, expected {expected_local}"
        )


tr._patch_output = _patch_output_and_runtime
tr._verify_result = _verify_result_t4


def main() -> int:
    protocol = validate_protocol(os.environ.get("NESTSAR_PROTOCOL", "xsub"))
    gpu = os.environ.get("NESTSAR_PHYSICAL_GPU", "unknown")

    print("=" * 108, flush=True)
    print("NESTSAR ATTENTION-LITE — SINGLE T4 TRAINER ADAPTER", flush=True)
    print(f"Protocol: {protocol.upper()} | assigned physical GPU={gpu}", flush=True)
    print("Expected process-visible GPUs: 1", flush=True)
    print(
        "30-GB-RAM mode: dataset remains lazy at sample preprocessing; "
        "no parent-process dataset copy is created.",
        flush=True,
    )
    print("=" * 108, flush=True)

    return int(tr.main())


if __name__ == "__main__":
    raise SystemExit(main())

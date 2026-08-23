#!/usr/bin/env python3
"""True-parallel Attention-Lite runner v3.

Fixes the Flax/dataclasses bootstrap failure from v2 by executing each generated
trainer bootstrap inside a real, registered Python module rather than an orphan
``exec`` globals dictionary.  Python dataclasses (used by Flax Linen) resolves
``cls.__module__`` through ``sys.modules``; without that registration it crashes
with ``NoneType has no attribute __dict__`` while defining the first Linen class.

The wrapper also isolates the embedded NestSAR source modules between the serial
XSUB and XSET bootstraps.  XSUB keeps strong references to its imported module
objects, then those public module names are evicted before XSET imports its own
copies from the separate XSET extraction root.  Actual warmup/training remains
parallel on disjoint TPU4 meshes:

    XSUB -> TPU [0,1,2,3]
    XSET -> TPU [4,5,6,7]
"""
from __future__ import annotations

import builtins
import sys
import types
from typing import Any

import run_attention_lite_parallel as base
import run_attention_lite_parallel_safe as safe

RUNNER_API_VERSION = "attention-lite-parallel-v3-registered-modules-4plus4"

# Exact top-level module names extracted by the validated Attention-Lite bundle.
# They are imported during bootstrap and may contain references to each other.
EMBEDDED_MODULES = (
    "nestsar",
    "nestsar_fcjm_b2",
    "nestsar_hope_fullselfref_v3_3_shortl3fix",
    "nestsar_hope_v4_1_cms_dmgd_l2_shortl3fix",
    "nestsar_m4_geom_h4",
    "nestsar_m4_geom_h4_sasm_l3statefix_v1",
    "nestsar_m4_regmask_ema_v3_safe",
    "nestsar_selfweight_clean",
    "nestsar_sms_s1c_v2",
)


def _evict_embedded_modules() -> dict[str, Any]:
    """Remove public bundle module names before one protocol bootstrap.

    Returned objects are intentionally not restored.  The already-bootstrapped
    protocol keeps strong references in its execution globals; the next protocol
    must import fresh module objects from its own extraction root.
    """
    removed: dict[str, Any] = {}
    for name in EMBEDDED_MODULES:
        module = sys.modules.pop(name, None)
        if module is not None:
            removed[name] = module
    return removed


def _bootstrap_one_registered(prepared: base.PreparedRun) -> safe.BootstrappedRun:
    text = prepared.generated.read_text(encoding="utf-8")
    bootstrap, remainder = safe._split_source(text, prepared.generated)

    prepared.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = prepared.log_path.open("w", encoding="utf-8")
    tagged_print = base._make_thread_print(prepared.protocol, log_file)

    # CRITICAL: Flax Linen uses dataclasses.  dataclasses._is_type() looks up
    # sys.modules[cls.__module__].  A plain exec(dict, dict) gives classes a
    # synthetic __module__ name that is not registered, which caused the exact
    # NoneType.__dict__ failure seen on Kaggle.
    module_name = (
        f"_nestsar_attention_lite_{prepared.protocol}_"
        f"seed_{prepared.seed}_parallel_runtime"
    )
    sys.modules.pop(module_name, None)
    module = types.ModuleType(module_name)
    module.__file__ = str(prepared.generated)
    module.__package__ = None
    module.__dict__["print"] = tagged_print
    module.__dict__["__builtins__"] = builtins.__dict__
    sys.modules[module_name] = module
    globals_dict = module.__dict__

    # XSUB and XSET use different extraction roots, but Python import caching is
    # keyed by module name.  Clear bundle module names before this serial bootstrap
    # so this protocol imports its own objects.  Previously bootstrapped objects are
    # retained through that runtime's globals and the strong-reference snapshot.
    _evict_embedded_modules()

    try:
        exec(
            compile(bootstrap, str(prepared.generated) + ":bootstrap", "exec"),
            globals_dict,
            globals_dict,
        )
    except BaseException:
        safe._write_failure(prepared, "bootstrap")
        log_file.close()
        sys.modules.pop(module_name, None)
        raise

    required_runtime = (
        "MESH",
        "DEVICES",
        "REPLICATED",
        "BATCH_SHARDING",
        "VALID_SHARDING",
        "train_dataset",
        "val_dataset",
        "fast_params",
        "medium_params",
        "slow_params",
        "consolidate_params",
        "ema_params",
        "compute_gradient",
        "save_resume_state",
    )
    missing = [name for name in required_runtime if name not in globals_dict]
    if missing:
        log_file.close()
        sys.modules.pop(module_name, None)
        raise RuntimeError(
            f"{prepared.protocol.upper()} serial bootstrap incomplete; missing: {missing}"
        )

    active_devices = tuple(int(d.id) for d in globals_dict["DEVICES"])
    if active_devices != prepared.device_ids:
        log_file.close()
        sys.modules.pop(module_name, None)
        raise RuntimeError(
            f"{prepared.protocol.upper()} bootstrap device mismatch: "
            f"{active_devices} != {prepared.device_ids}"
        )

    # Keep imported bundle modules alive even after their public names are evicted
    # for the next protocol's bootstrap.
    globals_dict["__nestsar_dynamic_module_name__"] = module_name
    globals_dict["__nestsar_bundle_module_refs__"] = {
        name: sys.modules.get(name)
        for name in EMBEDDED_MODULES
        if sys.modules.get(name) is not None
    }

    tagged_print(
        "SERIAL BOOTSTRAP: PASS | "
        f"registered_module={module_name} | "
        f"active TPU IDs={list(active_devices)} | "
        "warmup/training held until both protocols are ready"
    )

    return safe.BootstrappedRun(
        prepared=prepared,
        globals_dict=globals_dict,
        remainder=remainder,
        log_file=log_file,
    )


def _close_runtime_registered(runtime: safe.BootstrappedRun | None) -> None:
    if runtime is None:
        return

    module_name = runtime.globals_dict.get("__nestsar_dynamic_module_name__")
    try:
        if not runtime.log_file.closed:
            runtime.log_file.flush()
            runtime.log_file.close()
    finally:
        if module_name:
            sys.modules.pop(str(module_name), None)


def main() -> int:
    # Patch only orchestration functions.  All source construction, SHA guards,
    # model math, optimizer config, TPU4 source patching, result guards and the
    # parallel training phase continue to come from the validated v1/v2 runners.
    safe._bootstrap_one = _bootstrap_one_registered
    safe._close_runtime = _close_runtime_registered
    safe.RUNNER_API_VERSION = RUNNER_API_VERSION

    builtins.print(
        "PARALLEL V3 FIX ACTIVE: registered exec modules + isolated bundle imports",
        flush=True,
    )
    return safe.main()


if __name__ == "__main__":
    raise SystemExit(main())

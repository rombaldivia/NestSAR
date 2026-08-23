#!/usr/bin/env python3
"""Safe true-parallel Attention-Lite XSUB + XSET runner for one TPU v5e-8 host.

This is the corrected parallel execution path.

The first parallel runner executed two complete all-in-one trainers concurrently
from byte zero.  That is unsafe because the validated all-in-one source performs
process-global bootstrap work (sys.path changes, imports and monkey-patches of the
same embedded NestSAR modules) before it reaches the actual training loop.

This runner keeps one JAX runtime and the requested 4+4 TPU split, but separates
execution into two phases:

1. SERIAL BOOTSTRAP
   XSUB is fully initialized first, then XSET.  Imports, monkey-patches, model/state
   construction, dataset setup and TPU4 sharding are therefore deterministic and
   cannot race through sys.modules/sys.path.

2. PARALLEL WARMUP + TRAINING
   Once both independent protocol states exist, their production graph warmups and
   E1..E40 loops execute concurrently on disjoint TPU4 meshes:
       XSUB -> TPU [0,1,2,3]
       XSET -> TPU [4,5,6,7]

The mathematical model/training source remains the SHA-verified canonical source;
this file only changes orchestration/topology.  The two protocols keep separate
parameter trees, optimizer states, RNGs, datasets, output folders and result files.
"""
from __future__ import annotations

import builtins
import json
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import run_attention_lite_parallel as base

RUNNER_API_VERSION = "attention-lite-parallel-v2-serial-bootstrap-parallel-train"
TRAIN_SPLIT_MARKER = "# 22. COMPILE WARMUP ON ONE REAL B32"


@dataclass
class BootstrappedRun:
    prepared: base.PreparedRun
    globals_dict: dict[str, Any]
    remainder: str
    log_file: TextIO


def _split_source(text: str, generated: Path) -> tuple[str, str]:
    marker_index = text.find(TRAIN_SPLIT_MARKER)
    if marker_index < 0:
        raise RuntimeError(
            f"Could not find parallel phase split marker in {generated}: "
            f"{TRAIN_SPLIT_MARKER!r}"
        )

    # Start phase 2 at the beginning of the marker line.  Everything before this
    # point is initialization/bootstrap; everything after it is warmup + training.
    line_start = text.rfind("\n", 0, marker_index) + 1
    bootstrap = text[:line_start]
    remainder = text[line_start:]

    if "for epoch in range(" not in remainder:
        raise RuntimeError(f"Training loop is not present in phase 2 for {generated}")
    if "result.json" not in remainder:
        raise RuntimeError(f"Final result writer is not present in phase 2 for {generated}")
    if "import nestsar as ns" in remainder:
        raise RuntimeError(
            f"Phase split is too early for {generated}; process-global NestSAR imports leaked into phase 2"
        )

    compile(bootstrap, str(generated) + ":bootstrap", "exec")
    compile(remainder, str(generated) + ":train", "exec")
    return bootstrap, remainder


def _write_failure(prepared: base.PreparedRun, stage: str) -> str:
    text = traceback.format_exc()
    failure_path = prepared.root / f"parallel_{stage}_failure_traceback.txt"
    failure_path.write_text(text, encoding="utf-8")
    with base.PRINT_LOCK:
        builtins.print(
            f"[{prepared.protocol.upper()}] {stage.upper()} FAILED\n"
            f"Traceback saved: {failure_path}\n{text}",
            flush=True,
        )
    return text


def _bootstrap_one(prepared: base.PreparedRun) -> BootstrappedRun:
    text = prepared.generated.read_text(encoding="utf-8")
    bootstrap, remainder = _split_source(text, prepared.generated)

    prepared.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = prepared.log_path.open("w", encoding="utf-8")
    tagged_print = base._make_thread_print(prepared.protocol, log_file)
    globals_dict: dict[str, Any] = {
        "__name__": f"__nestsar_{prepared.protocol}_parallel__",
        "__file__": str(prepared.generated),
        "print": tagged_print,
    }

    try:
        exec(
            compile(bootstrap, str(prepared.generated) + ":bootstrap", "exec"),
            globals_dict,
            globals_dict,
        )
    except BaseException:
        _write_failure(prepared, "bootstrap")
        log_file.close()
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
        raise RuntimeError(
            f"{prepared.protocol.upper()} serial bootstrap incomplete; missing: {missing}"
        )

    active_devices = tuple(int(d.id) for d in globals_dict["DEVICES"])
    if active_devices != prepared.device_ids:
        log_file.close()
        raise RuntimeError(
            f"{prepared.protocol.upper()} bootstrap device mismatch: "
            f"{active_devices} != {prepared.device_ids}"
        )

    tagged_print(
        "SERIAL BOOTSTRAP: PASS | "
        f"active TPU IDs={list(active_devices)} | "
        "production warmup/training held until both protocols are ready"
    )

    return BootstrappedRun(
        prepared=prepared,
        globals_dict=globals_dict,
        remainder=remainder,
        log_file=log_file,
    )


def _finish_result_guard(prepared: base.PreparedRun) -> None:
    base.tr._verify_result(
        prepared.root,
        prepared.protocol,
        prepared.seed,
        prepared.cfg,
    )

    result_path = prepared.root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if int(result.get("active_tpu_devices", -1)) != 4:
        raise RuntimeError(
            f"{prepared.protocol.upper()} result did not record active_tpu_devices=4"
        )
    got_ids = tuple(int(x) for x in result.get("device_ids", []))
    if got_ids != prepared.device_ids:
        raise RuntimeError(
            f"{prepared.protocol.upper()} result device_ids mismatch: "
            f"{got_ids} != {prepared.device_ids}"
        )


def _train_one(runtime: BootstrappedRun, barrier: threading.Barrier) -> None:
    prepared = runtime.prepared
    barrier.wait()
    try:
        runtime.globals_dict["print"](
            "PARALLEL PHASE START | production graph warmup + training"
        )
        exec(
            compile(runtime.remainder, str(prepared.generated) + ":train", "exec"),
            runtime.globals_dict,
            runtime.globals_dict,
        )
        _finish_result_guard(prepared)
        runtime.globals_dict["print"]("PARALLEL TRAIN + FINAL GUARDS: PASS")
    except BaseException:
        _write_failure(prepared, "train")
        raise
    finally:
        runtime.log_file.flush()
        runtime.log_file.close()


def _close_runtime(runtime: BootstrappedRun | None) -> None:
    if runtime is None:
        return
    try:
        if not runtime.log_file.closed:
            runtime.log_file.flush()
            runtime.log_file.close()
    except Exception:
        pass


def main() -> int:
    args = base.parse_args()
    paper_mode = not args.non_paper
    seed = base.validate_seed(args.seed, paper_mode=paper_mode)

    if args.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if args.patience < 0:
        raise ValueError("patience must be >= 0")

    cfg = base._configure_training_env(args)
    if cfg["batch_size"] % 4 or cfg["eval_batch_size"] % 4:
        raise ValueError(
            "Parallel 4+4 mode requires batch_size and eval_batch_size divisible by 4"
        )

    devices = list(base.jax.devices())
    if base.jax.default_backend() != "tpu":
        raise RuntimeError(
            f"Parallel runner requires TPU backend; got {base.jax.default_backend()}"
        )
    if len(devices) != 8:
        raise RuntimeError(
            f"Parallel runner requires exactly 8 host-visible TPU devices; found {len(devices)}"
        )

    builtins.print("=" * 108, flush=True)
    builtins.print(f"RUNNER API: {RUNNER_API_VERSION}", flush=True)
    builtins.print("TRUE PARALLEL MODE WITH SERIAL IMPORT/BOOTSTRAP ISOLATION", flush=True)
    builtins.print("XSUB -> TPU [0,1,2,3]", flush=True)
    builtins.print("XSET -> TPU [4,5,6,7]", flush=True)
    builtins.print(
        f"Global batch/protocol={cfg['batch_size']} | "
        f"local batch/TPU={cfg['batch_size'] // 4} | "
        f"accum={cfg['grad_accum_steps']} | effective={cfg['effective_batch']}",
        flush=True,
    )
    builtins.print("=" * 108, flush=True)

    sources = base.ensure_canonical_sources(verbose=True)
    runs_root = Path(args.runs_root).expanduser()

    xsub = base._prepare_one(
        protocol="xsub",
        canonical_source=Path(sources["xsub"]),
        device_ids=(0, 1, 2, 3),
        seed=seed,
        cfg=cfg,
        runs_root=runs_root,
        run_tag=args.run_tag,
        paper_mode=paper_mode,
    )
    xset = base._prepare_one(
        protocol="xset",
        canonical_source=Path(sources["xset"]),
        device_ids=(4, 5, 6, 7),
        seed=seed,
        cfg=cfg,
        runs_root=runs_root,
        run_tag=args.run_tag,
        paper_mode=paper_mode,
    )

    builtins.print("GENERATED SOURCE PATCH/COMPILE PREFLIGHT: PASS", flush=True)
    builtins.print(f"XSUB generated: {xsub.generated}", flush=True)
    builtins.print(f"XSET generated: {xset.generated}", flush=True)

    xsub_runtime: BootstrappedRun | None = None
    xset_runtime: BootstrappedRun | None = None
    try:
        # Critical fix: do NOT race process-global imports/monkey-patches.
        builtins.print("\n[XSUB] Starting SERIAL bootstrap...", flush=True)
        xsub_runtime = _bootstrap_one(xsub)
        builtins.print("[XSET] Starting SERIAL bootstrap...", flush=True)
        xset_runtime = _bootstrap_one(xset)
    except BaseException:
        _close_runtime(xsub_runtime)
        _close_runtime(xset_runtime)
        raise

    builtins.print("=" * 108, flush=True)
    builtins.print("BOTH SERIAL BOOTSTRAPS: PASS", flush=True)
    builtins.print("Independent protocol states now resident on disjoint TPU4 meshes.", flush=True)
    builtins.print("=" * 108, flush=True)

    if args.preflight_only:
        _close_runtime(xsub_runtime)
        _close_runtime(xset_runtime)
        builtins.print("PREFLIGHT-ONLY — training was not started.", flush=True)
        return 0

    barrier = threading.Barrier(2)
    failures: list[str] = []
    runtimes = (xsub_runtime, xset_runtime)

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="nestsar-train") as pool:
        future_to_runtime = {
            pool.submit(_train_one, runtime, barrier): runtime
            for runtime in runtimes
        }
        for future in as_completed(future_to_runtime):
            runtime = future_to_runtime[future]
            prepared = runtime.prepared
            try:
                future.result()
                builtins.print(
                    f"{prepared.protocol.upper()} COMPLETE: {prepared.root}",
                    flush=True,
                )
            except BaseException as exc:
                failures.append(
                    f"{prepared.protocol.upper()}: {type(exc).__name__}: {exc} | "
                    f"trace={prepared.root / 'parallel_train_failure_traceback.txt'}"
                )

    if failures:
        raise RuntimeError(
            "Parallel Attention-Lite training failed. Exact per-protocol traceback files were saved:\n"
            + "\n".join(failures)
        )

    builtins.print("=" * 108, flush=True)
    builtins.print("XSUB + XSET TRUE PARALLEL TRAINING COMPLETE", flush=True)
    builtins.print(f"XSUB: {xsub.root}", flush=True)
    builtins.print(f"XSET: {xset.root}", flush=True)
    builtins.print("=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

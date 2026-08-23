#!/usr/bin/env python3
"""Run Attention-Lite XSUB and XSET concurrently on one TPU v5e-8 host.

Why this runner exists
----------------------
Kaggle TPU devices are owned by one JAX runtime. Starting two independent Python
processes against the same TPU commonly fails with /dev/vfio "device busy" errors.
This runner therefore imports JAX exactly once, splits the eight visible TPU devices
into two disjoint four-device meshes, and executes the two validated trainers in
parallel threads inside the same process:

    XSUB -> TPU [0,1,2,3]
    XSET -> TPU [4,5,6,7]

The global training recipe remains common to both protocols. With the validated
GLOBAL_BATCH=32 and GRAD_ACCUM=4, each protocol still sees global batch 32 and
effective batch 128; only the sharding topology changes from 8x4 samples/device to
4x8 samples/device. For a controlled multi-seed table, use this same parallel mode
for every seed (including seed 128) rather than mixing with historical TPU8 runs.
"""
from __future__ import annotations

import argparse
import builtins
import json
import os
import re
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Environment must be set before JAX import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import jax

from experiments.attention_lite_v1.canonical_integrated import ensure_canonical_sources
from experiments.attention_lite_v1.paths import (
    make_run_paths,
    validate_seed,
    write_path_manifest,
)
from experiments.attention_lite_v1 import trainer as tr

RUNNER_API_VERSION = "attention-lite-parallel-v1-single-jax-runtime-4plus4"
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class PreparedRun:
    protocol: str
    device_ids: tuple[int, ...]
    generated: Path
    root: Path
    log_path: Path
    cfg: dict[str, Any]
    seed: int


def _set_optional_env(name: str, value: Any) -> None:
    if value is not None:
        os.environ[name] = str(value)


def _configure_training_env(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["NESTSAR_EPOCHS"] = str(args.epochs)
    os.environ["NESTSAR_PATIENCE"] = str(args.patience)
    _set_optional_env("NESTSAR_DROPOUT", args.dropout)
    _set_optional_env("NESTSAR_LEARNING_RATE", args.learning_rate)
    _set_optional_env("NESTSAR_WEIGHT_DECAY", args.weight_decay)
    _set_optional_env("NESTSAR_WARMUP_FRACTION", args.warmup_fraction)
    _set_optional_env("NESTSAR_LABEL_SMOOTHING", args.label_smoothing)
    _set_optional_env("NESTSAR_GRAD_CLIP", args.grad_clip)
    _set_optional_env("NESTSAR_PREDICTIVE_LOSS_WEIGHT", args.predictive_loss_weight)
    _set_optional_env("NESTSAR_INITIAL_ETA", args.initial_eta)
    _set_optional_env("NESTSAR_INITIAL_ALPHA", args.initial_alpha)
    _set_optional_env("NESTSAR_BATCH_SIZE", args.batch_size)
    _set_optional_env("NESTSAR_GRAD_ACCUM_STEPS", args.grad_accum_steps)
    _set_optional_env("NESTSAR_EVAL_BATCH_SIZE", args.eval_batch_size)
    return tr._resolve_config()


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Parallel patch expected exactly one {label}; found {count}")
    return text.replace(old, new, 1)


def _patch_parallel_runtime(
    source: str,
    *,
    protocol: str,
    device_ids: tuple[int, ...],
) -> tuple[str, dict[str, int]]:
    """Patch only runtime sharding/storage, never model math or optimizer semantics."""
    if len(device_ids) != 4:
        raise ValueError("Parallel Attention-Lite currently requires exactly 4 TPU devices per protocol")
    if len(set(device_ids)) != 4:
        raise ValueError(f"Duplicate TPU device IDs: {device_ids}")

    counts: dict[str, int] = {}

    # Avoid two threads writing/extracting the embedded source bundle into one directory.
    old_root = 'ROOT = Path("/kaggle/working/NestSAR_HOPE_FIDELITY_UNIVERSAL")'
    new_root = f'ROOT = Path("/kaggle/working/NestSAR_HOPE_FIDELITY_UNIVERSAL_{protocol.upper()}")'
    source = _replace_once(source, old_root, new_root, "bundle ROOT")
    counts["protocol_root"] = 1

    # The original validated source takes all eight devices. Select a disjoint four-device
    # submesh while keeping jax.devices() itself visible to the single host runtime.
    old_devices = """DEVICES = list(
    jax.devices()
)"""
    ids_literal = repr(tuple(int(x) for x in device_ids))
    new_devices = f"""ALL_TPU_DEVICES = list(
    jax.devices()
)

DEVICE_IDS = {ids_literal}

if len(ALL_TPU_DEVICES) != 8:
    raise RuntimeError(
        f\"Parallel runner expected 8 host-visible TPU devices; found {{len(ALL_TPU_DEVICES)}}\"
    )

DEVICES = [
    ALL_TPU_DEVICES[index]
    for index in DEVICE_IDS
]"""
    source = _replace_once(source, old_devices, new_devices, "DEVICES selection")
    counts["device_selection"] = 1

    source, n = re.subn(
        r"if len\(\s*DEVICES\s*\) != 8:\s*\n\s*raise RuntimeError\(\s*f\"Expected 8 TPU devices; found \{len\(DEVICES\)\}\"\s*\)",
        "if len(DEVICES) != 4:\n\n    raise RuntimeError(\n        f\"Expected 4 active TPU devices; found {len(DEVICES)}\"\n    )",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError("Could not patch TPU8 active-device guard to TPU4 submesh")
    counts["active_device_guard"] = n

    # Correct provenance: parallel topology means local batch = global/4, not global/8.
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
        list(DEVICE_IDS),'''
    source = _replace_once(source, old_local_batch, new_local_batch, "result local_batch metadata")
    counts["result_runtime_metadata"] = 1

    # Add an unmistakable runtime banner immediately after mesh creation diagnostics.
    marker = '''print(
    "Local batch/TPU:",
    GLOBAL_BATCH
    //
    len(
        DEVICES
    )
)'''
    banner = marker + f'''\n\nprint(\n    "PARALLEL PROTOCOL: {protocol.upper()} | DEVICE_IDS={ids_literal} | ACTIVE_TPUS=4"\n)'''
    source = _replace_once(source, marker, banner, "parallel runtime banner")
    counts["runtime_banner"] = 1

    compile(source, f"<Attention-Lite-{protocol.upper()}-parallel>", "exec")
    return source, counts


def _prepare_one(
    *,
    protocol: str,
    canonical_source: Path,
    device_ids: tuple[int, ...],
    seed: int,
    cfg: dict[str, Any],
    runs_root: Path,
    run_tag: str | None,
    paper_mode: bool,
) -> PreparedRun:
    paths = make_run_paths(
        protocol,
        seed,
        base_dir=runs_root,
        paper_mode=paper_mode,
        tag=run_tag,
        create=True,
    )

    if (paths.root / "result.json").exists() and os.environ.get("NESTSAR_ALLOW_OVERWRITE", "0") != "1":
        raise FileExistsError(
            f"Completed run already exists: {paths.root}. Change seed/tag or set NESTSAR_ALLOW_OVERWRITE=1."
        )

    source = canonical_source.read_text(encoding="utf-8")
    tr._assert_golden_source(source, protocol)
    source_sha = tr._sha256_text(source)

    patch_counts: dict[str, int] = {}
    patched, c = tr._patch_seed(source, seed)
    patch_counts.update({f"seed_{k}": v for k, v in c.items()})
    patched, c = tr._patch_cfg_block(patched, cfg)
    patch_counts.update({f"cfg_{k}": v for k, v in c.items()})
    patched, c = tr._patch_locked_training(patched, cfg)
    patch_counts.update({f"locked_{k}": v for k, v in c.items()})
    patched, c = tr._patch_config_guards(patched, cfg)
    patch_counts.update(c)
    patched, c = tr._patch_early_stopping(patched)
    patch_counts.update(c)
    patched, patch_counts["output"] = tr._patch_output(patched, paths.root)
    patched, c = _patch_parallel_runtime(
        patched,
        protocol=protocol,
        device_ids=device_ids,
    )
    patch_counts.update({f"parallel_{k}": v for k, v in c.items()})

    patched_sha = tr._sha256_text(patched)
    generated = paths.source_dir / f"attention_lite_{protocol}_seed_{seed}_parallel4_generated.py"
    generated.write_text(patched, encoding="utf-8")
    compile(patched, str(generated), "exec")

    write_path_manifest(paths)
    manifest_path = tr._write_manifest(
        paths,
        protocol,
        seed,
        cfg,
        tr._is_golden_recipe(cfg),
        canonical_source,
        source_sha,
        patched_sha,
        patch_counts,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime"] = {
        "execution": "single_process_parallel_threads",
        "host_visible_tpu_devices": 8,
        "active_tpu_devices": 4,
        "device_ids": list(device_ids),
        "global_batch": int(cfg["batch_size"]),
        "local_batch_per_tpu": int(cfg["batch_size"] // 4),
        "effective_batch": int(cfg["effective_batch"]),
        "note": "Parallel 4+4 topology; do not mix with historical 8-device seed runs in one reproducibility mean/std.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return PreparedRun(
        protocol=protocol,
        device_ids=device_ids,
        generated=generated,
        root=paths.root,
        log_path=paths.logs_dir / "train_parallel.log",
        cfg=cfg,
        seed=seed,
    )


def _make_thread_print(protocol: str, log_file):
    prefix = f"[{protocol.upper()}]"

    def tagged_print(*args, **kwargs):
        sep = kwargs.pop("sep", " ")
        end = kwargs.pop("end", "\n")
        flush = kwargs.pop("flush", False)
        file_arg = kwargs.pop("file", None)
        if kwargs:
            raise TypeError(f"Unsupported print kwargs in parallel runner: {sorted(kwargs)}")
        text = sep.join(str(x) for x in args)
        with PRINT_LOCK:
            # Respect explicit non-stdout file targets, otherwise tee to notebook + log.
            if file_arg not in (None, sys.stdout):
                builtins.print(*args, sep=sep, end=end, file=file_arg, flush=flush)
                return
            builtins.print(prefix, text, end=end, flush=True)
            log_file.write(prefix + " " + text + end)
            log_file.flush()

    return tagged_print


def _execute_one(run: PreparedRun, start_barrier: threading.Barrier) -> None:
    code_text = run.generated.read_text(encoding="utf-8")
    code = compile(code_text, str(run.generated), "exec")
    run.log_path.parent.mkdir(parents=True, exist_ok=True)

    with run.log_path.open("w", encoding="utf-8") as log_file:
        globals_dict = {
            "__name__": "__main__",
            "__file__": str(run.generated),
            "print": _make_thread_print(run.protocol, log_file),
        }
        start_barrier.wait()
        try:
            exec(code, globals_dict, globals_dict)
        except BaseException:
            with PRINT_LOCK:
                builtins.print(f"[{run.protocol.upper()}] PARALLEL TRAIN FAILED", flush=True)
                traceback.print_exc()
            raise

    tr._verify_result(run.root, run.protocol, run.seed, run.cfg)
    result = json.loads((run.root / "result.json").read_text(encoding="utf-8"))
    if int(result.get("active_tpu_devices", -1)) != 4:
        raise RuntimeError(f"{run.protocol.upper()} result did not record active_tpu_devices=4")
    if tuple(int(x) for x in result.get("device_ids", [])) != run.device_ids:
        raise RuntimeError(
            f"{run.protocol.upper()} device_ids mismatch: {result.get('device_ids')} != {run.device_ids}"
        )



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XSUB and XSET concurrently on disjoint TPU4 submeshes")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--runs-root", default="/kaggle/working")
    p.add_argument("--run-tag", default="parallel4x4")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--non-paper", action="store_true")

    p.add_argument("--dropout", type=float, default=0.22)
    p.add_argument("--learning-rate", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-fraction", type=float, default=0.10)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--predictive-loss-weight", type=float, default=0.10)
    p.add_argument("--initial-eta", type=float, default=0.02)
    p.add_argument("--initial-alpha", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=32)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paper_mode = not args.non_paper
    seed = validate_seed(args.seed, paper_mode=paper_mode)
    if args.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if args.patience < 0:
        raise ValueError("patience must be >= 0")

    cfg = _configure_training_env(args)
    if cfg["batch_size"] % 4 or cfg["eval_batch_size"] % 4:
        raise ValueError("Parallel mode requires batch_size and eval_batch_size divisible by 4")

    devices = list(jax.devices())
    if jax.default_backend() != "tpu":
        raise RuntimeError(f"Parallel runner requires TPU backend; got {jax.default_backend()}")
    if len(devices) != 8:
        raise RuntimeError(f"Parallel runner requires exactly 8 host-visible TPU devices; found {len(devices)}")

    print("=" * 108, flush=True)
    print(f"RUNNER API: {RUNNER_API_VERSION}", flush=True)
    print("TRUE PARALLEL MODE: XSUB and XSET execute at the same time", flush=True)
    print("XSUB -> TPU [0,1,2,3]", flush=True)
    print("XSET -> TPU [4,5,6,7]", flush=True)
    print(f"Global batch/protocol: {cfg['batch_size']} | local batch/TPU: {cfg['batch_size'] // 4}", flush=True)
    print(f"Grad accumulation: {cfg['grad_accum_steps']} | effective batch/protocol: {cfg['effective_batch']}", flush=True)
    print("=" * 108, flush=True)

    sources = ensure_canonical_sources(verbose=True)
    runs_root = Path(args.runs_root).expanduser()
    xsub = _prepare_one(
        protocol="xsub",
        canonical_source=Path(sources["xsub"]),
        device_ids=(0, 1, 2, 3),
        seed=seed,
        cfg=cfg,
        runs_root=runs_root,
        run_tag=args.run_tag,
        paper_mode=paper_mode,
    )
    xset = _prepare_one(
        protocol="xset",
        canonical_source=Path(sources["xset"]),
        device_ids=(4, 5, 6, 7),
        seed=seed,
        cfg=cfg,
        runs_root=runs_root,
        run_tag=args.run_tag,
        paper_mode=paper_mode,
    )

    print("PARALLEL GENERATED SOURCE PREFLIGHT: PASS", flush=True)
    print(f"XSUB generated: {xsub.generated}", flush=True)
    print(f"XSET generated: {xset.generated}", flush=True)
    if args.preflight_only:
        print("PREFLIGHT-ONLY — no TPU training started.", flush=True)
        return 0

    barrier = threading.Barrier(2)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="nestsar-protocol") as pool:
        future_to_run = {
            pool.submit(_execute_one, xsub, barrier): xsub,
            pool.submit(_execute_one, xset, barrier): xset,
        }
        for future in as_completed(future_to_run):
            run = future_to_run[future]
            try:
                future.result()
                print(f"{run.protocol.upper()} PARALLEL TRAIN COMPLETE: {run.root}", flush=True)
            except BaseException as exc:
                failures.append(f"{run.protocol.upper()}: {exc!r}")

    if failures:
        raise RuntimeError("Parallel Attention-Lite run failed:\n" + "\n".join(failures))

    print("=" * 108, flush=True)
    print("XSUB + XSET TRUE PARALLEL TRAINING COMPLETE", flush=True)
    print(f"XSUB: {xsub.root}", flush=True)
    print(f"XSET: {xset.root}", flush=True)
    print("=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

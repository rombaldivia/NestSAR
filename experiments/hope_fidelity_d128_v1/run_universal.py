#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
NestSAR-HOPE-Fidelity universal launcher
========================================

CD-Former-style argparse entry point for the same HOPE-Fidelity source on:
  --preset rtx5080
  --preset 2xt4
  --preset tpu-v5e8
  --preset tpu-v5e8-canonical
  --preset canonical

`--batch-size` is always the GLOBAL physical batch. With SPMD enabled it is
sharded across the selected devices. `--grad-accum-steps` then gives the global
effective batch = global physical batch * accumulation.

The exact v4.1 self-contained one-cell may be supplied with --bundle-cell. On
Kaggle it is auto-discovered under /kaggle/input. Locally you can instead pass
--source-root pointing at the exact audited v4.1 source directory.
"""

import argparse
import base64
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

MODEL_ID = "nestsar_hope_fidelity_d128_v1"
BUNDLE_FILENAME = "NestSAR_HOPE_v4_1_SHORTL3FIX_Kaggle_16F_ONE_CELL.py"
EXPECTED_BUNDLE_SHA256 = "c720c2afd9c32648ece4ac4b23e916f325039ba41684ebe4127ae285c6e216dd"
CANONICAL_PARAMS = 2_083_236

HERE = Path(__file__).resolve().parent
BUILDER = HERE / "build_from_v41.py"
WORKER = HERE / "nestsar_universal_runtime.py"

REQUIRED_V41 = (
    "nestsar.py",
    "nestsar_fcjm_b2.py",
    "nestsar_selfweight_clean.py",
    "nestsar_sms_s1c_v2.py",
    "nestsar_m4_geom_h4.py",
    "nestsar_m4_regmask_ema_v3_safe.py",
    "nestsar_m4_geom_h4_sasm_l3statefix_v1.py",
    "nestsar_hope_fullselfref_v3_3_shortl3fix.py",
    "nestsar_hope_v4_1_cms_dmgd_l2_shortl3fix.py",
)

PRESETS = {
    "canonical": dict(
        backend="auto", device_count=0, spmd="auto",
        batch_size=32, grad_accum_steps=4, eval_batch_size=64,
    ),
    "rtx5080": dict(
        backend="gpu", device_count=1, spmd="off",
        batch_size=32, grad_accum_steps=4, eval_batch_size=64,
    ),
    "2xt4": dict(
        backend="gpu", device_count=2, spmd="on",
        batch_size=32, grad_accum_steps=4, eval_batch_size=64,
    ),
    "tpu-v5e8": dict(
        backend="tpu", device_count=8, spmd="on",
        batch_size=128, grad_accum_steps=1, eval_batch_size=256,
    ),
    "tpu-v5e8-canonical": dict(
        backend="tpu", device_count=8, spmd="on",
        batch_size=32, grad_accum_steps=4, eval_batch_size=64,
    ),
}


def default_runtime_root() -> Path:
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working/NestSAR_HOPE_FIDELITY_UNIVERSAL")
    return Path.cwd() / ".nestsar_hope_fidelity_runtime"


def find_dataset(value: str) -> Path:
    if value != "auto":
        p = Path(value).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    candidates = [
        Path.home() / "Downloads" / "ntu120_3danno.pkl",
        Path.home() / "Downloads" / "ntu120_3danno_clean.pkl",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        for name in ("ntu120_3danno.pkl", "ntu120_3danno_clean.pkl"):
            hits = sorted(kaggle.rglob(name))
            if hits:
                return hits[0].resolve()
    raise FileNotFoundError("NTU120 pickle not found. Pass --dataset /path/ntu120_3danno.pkl")


def valid_source_root(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_V41)


def find_bundle(explicit: str | None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates += [Path.cwd() / BUNDLE_FILENAME]
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        candidates += sorted(kaggle.rglob(BUNDLE_FILENAME))
    for p in candidates:
        try:
            p = p.resolve()
        except Exception:
            continue
        if p.is_file():
            return p
    return None


def normalize_extracted_v41(runtime: Path) -> None:
    """Reproduce the audited one-cell's post-extraction trainer fixes.

    The exact self-contained v4.1 artifact stores an immutable source ZIP and
    then applies three trainer-only patches before launch: optimizer-step LR
    accounting, effective-batch logging / redundant-smoke omission, and the LR
    display step. The universal launcher extracts the same immutable ZIP, so it
    must apply those same idempotent patches before building HOPE-Fidelity.
    """
    core_path = runtime / "nestsar.py"
    core = core_path.read_text(encoding="utf-8")

    old_total = '''    steps_per_epoch = max(1, math.ceil(len(train_dataset) / CFG.batch_size))
    total_steps = CFG.epochs * steps_per_epoch

    model = build_model(model_id)
'''
    new_total = '''    steps_per_epoch = max(1, math.ceil(len(train_dataset) / CFG.batch_size))
    total_micro_steps = CFG.epochs * steps_per_epoch
    total_steps = max(
        1,
        math.ceil(total_micro_steps / max(1, CFG.grad_accum_steps)),
    )
    optimizer_steps_per_epoch = (
        steps_per_epoch / max(1, CFG.grad_accum_steps)
    )

    model = build_model(model_id)
'''
    if old_total in core:
        core = core.replace(old_total, new_total, 1)
        print("[v4.1 PATCH] LR schedule counts effective optimizer updates")
    elif "total_micro_steps = CFG.epochs * steps_per_epoch" not in core:
        raise RuntimeError("Could not normalize v4.1 total_steps accounting")

    old_block = '''    parameter_count = count_parameters(state.params)
    log(f"Parámetros: {parameter_count:,}")

    train_step, eval_step = build_steps(model, model_id)
    smoke_test(model, state, train_step)
    if smoke_only:
'''
    new_block = '''    parameter_count = count_parameters(state.params)
    log(f"Parámetros: {parameter_count:,}")
    log(
        f"Batch físico={CFG.batch_size} | "
        f"acumulación={CFG.grad_accum_steps} | "
        f"batch efectivo≈{CFG.batch_size * CFG.grad_accum_steps} | "
        f"updates/época≈{optimizer_steps_per_epoch:.2f}"
    )

    train_step, eval_step = build_steps(model, model_id)
    log("Smoke test OMITIDO para esta corrida.")
    if smoke_only:
'''
    if old_block in core:
        core = core.replace(old_block, new_block, 1)
        print("[v4.1 PATCH] effective-batch logging / redundant smoke omission")
    elif 'log("Smoke test OMITIDO para esta corrida.")' not in core:
        raise RuntimeError("Could not normalize v4.1 training-start block")

    old_lr = "lr = float(np.asarray(make_schedule(total_steps)(state.step)))"
    new_lr = '''lr_step = state.step // max(1, CFG.grad_accum_steps)
            lr = float(np.asarray(make_schedule(total_steps)(lr_step)))'''
    if old_lr in core:
        core = core.replace(old_lr, new_lr, 1)
        print("[v4.1 PATCH] displayed LR uses effective optimizer step")
    elif (
        "lr_step = state.step // max(1, CFG.grad_accum_steps)" not in core
        and "lr_step = state.step // CFG.grad_accum_steps" not in core
        and "lr_step = int(np.asarray(state.step)) // max(1, CFG.grad_accum_steps)" not in core
    ):
        raise RuntimeError("Could not normalize v4.1 LR logger")

    if "grad_accum_steps" not in core or "--grad-accum-steps" not in core:
        raise RuntimeError("Extracted v4.1 source has no gradient-accumulation CLI support")
    core_path.write_text(core, encoding="utf-8")

    reg_path = runtime / "nestsar_m4_regmask_ema_v3_safe.py"
    reg = reg_path.read_text(encoding="utf-8")
    ema_pattern = re.compile(r"^EMA_DECAY\s*=\s*[0-9.]+\s*$", re.MULTILINE)
    match = ema_pattern.search(reg)
    if not match:
        raise RuntimeError("Could not find EMA_DECAY in exact v4.1 RegMask source")
    if match.group(0) != "EMA_DECAY = 0.995":
        reg = reg[:match.start()] + "EMA_DECAY = 0.995" + reg[match.end():]
        print("[v4.1 PATCH] EMA_DECAY -> 0.995")
    for marker in (
        "FRAME_MASK_PROB = 0.08",
        "JOINT_MASK_PROB = 0.08",
        "PART_MASK_PROB = 0.03",
    ):
        if marker not in reg:
            raise RuntimeError(f"Unexpected v4.1 RegMask source; missing {marker!r}")
    reg_path.write_text(reg, encoding="utf-8")

    compile(core, str(core_path), "exec")
    compile(reg, str(reg_path), "exec")
    print("Exact v4.1 post-extraction normalization: PASS")


def extract_exact_bundle(cell: Path, runtime: Path) -> Path:
    text = cell.read_text(encoding="utf-8")
    patterns = (
        r"BUNDLE_B64\s*=\s*r'''(.*?)'''",
        r'BUNDLE_B64\s*=\s*r"""(.*?)"""',
        r"BUNDLE_B64\s*=\s*'''(.*?)'''",
        r'BUNDLE_B64\s*=\s*"""(.*?)"""',
    )
    payload = None
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.DOTALL)
        if m:
            payload = "".join(m.group(1).split())
            break
    if payload is None:
        raise RuntimeError(f"Could not extract BUNDLE_B64 from {cell}")
    raw = base64.b64decode(payload.encode("ascii"), validate=True)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXPECTED_BUNDLE_SHA256:
        raise RuntimeError(
            f"Exact v4.1 bundle SHA mismatch: {actual} != {EXPECTED_BUNDLE_SHA256}"
        )
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(__import__("io").BytesIO(raw), "r") as zf:
        zf.extractall(runtime)
    if not valid_source_root(runtime):
        missing = [name for name in REQUIRED_V41 if not (runtime / name).is_file()]
        raise RuntimeError("Extracted v4.1 bundle missing: " + ", ".join(missing))
    print(f"Exact v4.1 bundle: PASS | SHA256={actual}")
    normalize_extracted_v41(runtime)
    return runtime


def resolve_source_root(args, runtime: Path) -> Path:
    if args.source_root:
        p = Path(args.source_root).expanduser().resolve()
        if not valid_source_root(p):
            raise FileNotFoundError(
                "--source-root is not a complete v4.1 source tree: " + str(p)
            )
        return p
    if valid_source_root(Path.cwd()):
        return Path.cwd().resolve()
    bundle = find_bundle(args.bundle_cell)
    if bundle is None:
        raise FileNotFoundError(
            f"Could not find {BUNDLE_FILENAME}. Pass --bundle-cell /path/{BUNDLE_FILENAME} "
            "or --source-root /path/to/exact/v4.1/source"
        )
    return extract_exact_bundle(bundle, runtime)


def apply_preset(args):
    p = PRESETS[args.preset]
    if args.backend is None:
        args.backend = p["backend"]
    if args.device_count is None:
        args.device_count = p["device_count"]
    if args.spmd is None:
        args.spmd = p["spmd"]
    if args.batch_size is None:
        args.batch_size = p["batch_size"]
    if args.grad_accum_steps is None:
        args.grad_accum_steps = p["grad_accum_steps"]
    if args.eval_batch_size is None:
        args.eval_batch_size = p["eval_batch_size"]
    if args.probe:
        if args.epochs is None:
            args.epochs = 3
        if args.patience is None:
            args.patience = 3
    else:
        if args.epochs is None:
            args.epochs = 40
        if args.patience is None:
            args.patience = 12
    return args


def build_parser():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Universal NestSAR-HOPE-Fidelity trainer for RTX5080, 2xT4 and TPU v5e-8",
    )

    ap.add_argument("--preset", choices=tuple(PRESETS), default="canonical")
    ap.add_argument("--source-root")
    ap.add_argument("--bundle-cell")
    ap.add_argument("--runtime-root", default=str(default_runtime_root()))
    ap.add_argument("--dataset", default="auto")
    ap.add_argument("--output-dir")
    ap.add_argument("--protocol", choices=("xsub", "xset"), default="xsub")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--resume", choices=("none", "auto"), default="auto")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke-only", action="store_true")

    ap.add_argument("--backend", choices=("auto", "gpu", "tpu"), default=None)
    ap.add_argument("--device-count", type=int, default=None, help="0=all visible")
    ap.add_argument("--device-ids", help="GPU physical ids before JAX import, e.g. 0 or 0,1")
    ap.add_argument("--spmd", choices=("auto", "on", "off"), default=None)
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.90)

    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--num-classes", type=int, default=120)
    ap.add_argument("--model-dim", type=int, default=128)
    ap.add_argument("--memory-dim", type=int, default=64)
    ap.add_argument("--controller-rank", type=int, default=32)
    ap.add_argument("--frame-blocks", type=int, default=2)
    ap.add_argument("--chunk-blocks", type=int, default=2)
    ap.add_argument("--clip-blocks", type=int, default=2)
    ap.add_argument("--controller-blocks", type=int, default=2)
    ap.add_argument("--chunk-size", type=int, default=4)
    ap.add_argument("--clip-size", type=int, default=8)
    ap.add_argument("--cms-bottleneck", type=int, default=32)
    ap.add_argument("--expected-params", type=int, default=0, help="0=canonical guard or dynamic custom count")

    ap.add_argument("--batch-size", type=int, default=None, help="GLOBAL physical batch")
    ap.add_argument("--grad-accum-steps", type=int, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=None, help="GLOBAL eval batch")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-fraction", type=float, default=0.10)
    ap.add_argument("--dropout", type=float, default=0.22)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--predictive-loss-weight", type=float, default=0.10)
    ap.add_argument("--memory-residual-scale", type=float, default=0.25)
    ap.add_argument("--initial-eta", type=float, default=0.02)
    ap.add_argument("--initial-alpha", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--log-every-batches", type=int, default=200)

    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--frame-mask-prob", type=float, default=0.08)
    ap.add_argument("--joint-mask-prob", type=float, default=0.08)
    ap.add_argument("--part-mask-prob", type=float, default=0.03)

    ap.add_argument("--selfref-dgd-scale", type=float, default=1.0)
    ap.add_argument("--selfref-eta-max", type=float, default=0.05)
    ap.add_argument("--selfref-residual-beta", type=float, default=0.10)
    ap.add_argument("--selfref-matrix-norm-cap", type=float, default=4.0)
    ap.add_argument("--selfref-vector-norm-cap", type=float, default=1.0)
    ap.add_argument("--selfref-alpha-min", type=float, default=0.90)
    ap.add_argument("--selfref-alpha-max", type=float, default=0.999)
    ap.add_argument("--selfref-state-init-scale", type=float, default=0.12)
    ap.add_argument("--short-l3-blend", type=float, default=1.0)

    ap.add_argument("--cms-period-l1", type=int, default=1)
    ap.add_argument("--cms-period-l2", type=int, default=2)
    ap.add_argument("--cms-period-l3", type=int, default=4)
    ap.add_argument("--cms-period-l4", type=int, default=8)
    ap.add_argument("--dmgd-momentum", type=float, default=0.90)
    ap.add_argument("--dmgd-memory-lr", type=float, default=0.01)
    ap.add_argument("--dmgd-mix", type=float, default=0.10)
    ap.add_argument("--dmgd-projection-cap", type=float, default=2.0)
    return ap


def main() -> int:
    args = apply_preset(build_parser().parse_args())

    if args.device_count is not None and args.device_count < 0:
        raise ValueError("--device-count must be >= 0")
    if args.batch_size < 1 or args.grad_accum_steps < 1 or args.eval_batch_size < 1:
        raise ValueError("batch/accum/eval batch must be >=1")
    if args.cms_bottleneck < 1:
        raise ValueError("--cms-bottleneck must be >=1")

    effective_batch = args.batch_size * args.grad_accum_steps
    if args.device_count and args.device_count > 1 and args.spmd != "off":
        if args.batch_size % args.device_count != 0:
            raise ValueError("Global --batch-size must be divisible by --device-count for SPMD full batches")
        if args.eval_batch_size % args.device_count != 0:
            raise ValueError("--eval-batch-size must be divisible by --device-count for SPMD")

    runtime = Path(args.runtime_root).expanduser().resolve()
    source_root = resolve_source_root(args, runtime)
    dataset = find_dataset(args.dataset)

    if not args.skip_build:
        subprocess.run([sys.executable, "-u", str(BUILDER), "--root", str(source_root)], check=True)

    generated = (
        source_root / "nestsar_hope_fidelity_d128_v1_core.py",
        source_root / "nestsar_hope_fidelity_d128_v1_train.py",
    )
    missing = [str(p) for p in generated if not p.is_file()]
    if missing:
        raise FileNotFoundError("Generated fidelity sources missing: " + ", ".join(missing))

    if args.output_dir:
        output = Path(args.output_dir).expanduser().resolve()
    else:
        output = source_root / "runs_universal" / f"{args.preset}_{args.protocol}_f{args.frames}_d{args.model_dim}_m{args.memory_dim}_b{args.batch_size}x{args.grad_accum_steps}"
    if args.fresh and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    resume = "none" if args.fresh else args.resume

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env["JAX_THREEFRY_PARTITIONABLE"] = "true"
    env["NESTSAR_EXPECTED_BACKEND"] = args.backend
    env["NESTSAR_DEVICE_COUNT"] = str(args.device_count or 0)
    env["NESTSAR_SPMD"] = args.spmd

    if args.backend in ("gpu", "tpu"):
        env["JAX_PLATFORMS"] = args.backend
    else:
        env.pop("JAX_PLATFORMS", None)

    if args.backend == "gpu":
        if args.device_ids:
            env["CUDA_VISIBLE_DEVICES"] = args.device_ids
        elif args.device_count and args.device_count > 0:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.device_count))
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.gpu_memory_fraction)
        env.setdefault("MALLOC_ARENA_MAX", "2")
    elif args.backend == "tpu":
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)

    cache = source_root / ".jax_cache_universal"
    cache.mkdir(parents=True, exist_ok=True)
    env["JAX_COMPILATION_CACHE_DIR"] = str(cache)
    env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "1"
    env["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "-1"

    env.update({
        "NESTSAR_CMS_BOTTLENECK": str(args.cms_bottleneck),
        "NESTSAR_EXPECTED_PARAMS": str(args.expected_params),
        "NESTSAR_EMA_DECAY": str(args.ema_decay),
        "NESTSAR_FRAME_MASK_PROB": str(args.frame_mask_prob),
        "NESTSAR_JOINT_MASK_PROB": str(args.joint_mask_prob),
        "NESTSAR_PART_MASK_PROB": str(args.part_mask_prob),
        "NESTSAR_SELFREF_DGD_SCALE": str(args.selfref_dgd_scale),
        "NESTSAR_SELFREF_ETA_MAX": str(args.selfref_eta_max),
        "NESTSAR_SELFREF_RESIDUAL_BETA": str(args.selfref_residual_beta),
        "NESTSAR_SELFREF_MATRIX_NORM_CAP": str(args.selfref_matrix_norm_cap),
        "NESTSAR_SELFREF_VECTOR_NORM_CAP": str(args.selfref_vector_norm_cap),
        "NESTSAR_SELFREF_ALPHA_MIN": str(args.selfref_alpha_min),
        "NESTSAR_SELFREF_ALPHA_MAX": str(args.selfref_alpha_max),
        "NESTSAR_SELFREF_STATE_INIT_SCALE": str(args.selfref_state_init_scale),
        "NESTSAR_SHORT_L3_POSTWRITE_BLEND": str(args.short_l3_blend),
        "NESTSAR_CMS_PERIOD_L1": str(args.cms_period_l1),
        "NESTSAR_CMS_PERIOD_L2": str(args.cms_period_l2),
        "NESTSAR_CMS_PERIOD_L3": str(args.cms_period_l3),
        "NESTSAR_CMS_PERIOD_L4": str(args.cms_period_l4),
        "NESTSAR_DMGD_MOMENTUM": str(args.dmgd_momentum),
        "NESTSAR_DMGD_MEMORY_LR": str(args.dmgd_memory_lr),
        "NESTSAR_DMGD_MIX": str(args.dmgd_mix),
        "NESTSAR_DMGD_PROJECTION_CAP": str(args.dmgd_projection_cap),
    })
    env["PYTHONPATH"] = str(source_root) + os.pathsep + str(HERE) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable, "-u", str(WORKER),
        "--model", MODEL_ID,
        "--protocol", args.protocol,
        "--dataset", str(dataset),
        "--output-dir", str(output),
        "--seed", str(args.seed),
        "--frames", str(args.frames),
        "--num-classes", str(args.num_classes),
        "--model-dim", str(args.model_dim),
        "--memory-dim", str(args.memory_dim),
        "--frame-blocks", str(args.frame_blocks),
        "--chunk-blocks", str(args.chunk_blocks),
        "--clip-blocks", str(args.clip_blocks),
        "--controller-blocks", str(args.controller_blocks),
        "--chunk-size", str(args.chunk_size),
        "--clip-size", str(args.clip_size),
        "--controller-rank", str(args.controller_rank),
        "--dropout", str(args.dropout),
        "--batch-size", str(args.batch_size),
        "--grad-accum-steps", str(args.grad_accum_steps),
        "--eval-batch-size", str(args.eval_batch_size),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--warmup-fraction", str(args.warmup_fraction),
        "--label-smoothing", str(args.label_smoothing),
        "--grad-clip", str(args.grad_clip),
        "--memory-residual-scale", str(args.memory_residual_scale),
        "--predictive-loss-weight", str(args.predictive_loss_weight),
        "--initial-eta", str(args.initial_eta),
        "--initial-alpha", str(args.initial_alpha),
        "--log-every-batches", str(args.log_every_batches),
        "--resume", resume,
    ]
    # The inherited v4.1 CLI names this guard --allow-cpu, but semantically it
    # means "allow a non-GPU JAX backend". We add it only for TPU; the universal
    # worker has already hard-checked backend=tpu before ns.main() runs.
    if args.backend == "tpu":
        cmd.append("--allow-cpu")
    if args.smoke_only:
        cmd.append("--smoke-only")

    devices_text = "auto/all visible" if not args.device_count else str(args.device_count)
    local_batch = (
        args.batch_size // args.device_count
        if args.device_count and args.device_count > 0 and args.spmd != "off"
        else args.batch_size
    )
    print("=" * 120)
    print("NESTSAR-HOPE-FIDELITY — UNIVERSAL RUN CONFIG")
    print("=" * 120)
    print(f"Preset:              {args.preset}")
    print(f"Backend:             {args.backend}")
    print(f"Devices requested:   {devices_text}")
    print(f"SPMD:                {args.spmd}")
    print(f"Global batch:        {args.batch_size}")
    print(f"Local batch/device:  {local_batch} (for requested fixed device count)")
    print(f"Grad accumulation:   {args.grad_accum_steps}")
    print(f"Effective batch:     {effective_batch}")
    print(f"Eval batch:          {args.eval_batch_size}")
    print(f"Frames:              {args.frames}")
    print(f"D / M / R:           {args.model_dim} / {args.memory_dim} / {args.controller_rank}")
    print(f"Blocks F/C/K/L4:     {args.frame_blocks}/{args.chunk_blocks}/{args.clip_blocks}/{args.controller_blocks}")
    print(f"Chunk / clip:        {args.chunk_size} / {args.clip_size}")
    print(f"CMS bottleneck:      {args.cms_bottleneck}")
    print(f"CMS periods:         {args.cms_period_l1}/{args.cms_period_l2}/{args.cms_period_l3}/{args.cms_period_l4}")
    print(f"Epochs / patience:   {args.epochs} / {args.patience}")
    print(f"LR / WD:             {args.learning_rate:g} / {args.weight_decay:g}")
    print(f"Source root:         {source_root}")
    print(f"Dataset:             {dataset}")
    print(f"Output:              {output}")
    print(f"Canonical params:    {CANONICAL_PARAMS:,} (guard only for canonical config)")
    if effective_batch != 128:
        print(f"WARNING: effective batch is {effective_batch}, not canonical 128")
    print("=" * 120)
    print("COMMAND:")
    print(" ".join(shlex.quote(x) for x in cmd))
    print("=" * 120)

    if args.dry_run:
        return 0
    return subprocess.call(cmd, cwd=source_root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())

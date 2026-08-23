#!/usr/bin/env python3
"""Safe launcher for validated NestSAR Attention-Lite protocol trainers.

The validated XSUB/XSET all-in-one source remains the mathematical source of
truth. This wrapper patches only seed, output path, and explicitly requested
training hyperparameters. Architecture-defining values remain guarded.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .paths import PAPER_SEEDS, make_run_paths, validate_protocol, validate_seed, write_path_manifest

EXPECTED_PARAMS = 2_381_028
EXPECTED_LEAVES = 705
EXPECTED_GFLOPS = 0.060416900
EXPECTED_FRAMES = 16
EXPECTED_DMODEL = 128
EXPECTED_ATTN_DIM = 64
EXPECTED_HEADS = 4
EXPECTED_HEAD_DIM = 16
EXPECTED_TPU_DEVICES = 8

GOLDEN = {
    "epochs": 40,
    "patience": 0,
    "dropout": 0.22,
    "learning_rate": 1.0e-3,
    "weight_decay": 0.05,
    "warmup_fraction": 0.10,
    "label_smoothing": 0.05,
    "grad_clip": 1.0,
    "predictive_loss_weight": 0.10,
    "initial_eta": 0.02,
    "initial_alpha": 0.95,
    "batch_size": 32,
    "grad_accum_steps": 4,
    "eval_batch_size": 32,
}

CANONICAL_FILENAMES = {
    "xsub": "NestSAR_HOPE_Attention_Lite_TPUv5e8_XSUB_E40_ALL_IN_ONE.py",
    "xset": "NestSAR_HOPE_Attention_Lite_TPUv5e8_XSET_E40_ALL_IN_ONE.py",
}

GOLDEN_MARKERS = (
    "NESTSAR-HOPE-ATTENTION-LITE D128",
    "EXPECTED_PARAMS = 2_381_028",
    "EXPECTED_LEAVES = 705",
    "GLOBAL_BATCH = 32",
    "GRAD_ACCUM = 4",
    "EFFECTIVE_BATCH = 128",
    "ATTENTION_DIM = 64",
    "ATTENTION_HEADS = 4",
    "HEAD_DIM = ATTENTION_DIM // ATTENTION_HEADS",
    "RegMask",
    "EMA",
)

ENV_OVERRIDES = {
    "dropout": ("NESTSAR_DROPOUT", float),
    "learning_rate": ("NESTSAR_LEARNING_RATE", float),
    "weight_decay": ("NESTSAR_WEIGHT_DECAY", float),
    "warmup_fraction": ("NESTSAR_WARMUP_FRACTION", float),
    "label_smoothing": ("NESTSAR_LABEL_SMOOTHING", float),
    "grad_clip": ("NESTSAR_GRAD_CLIP", float),
    "predictive_loss_weight": ("NESTSAR_PREDICTIVE_LOSS_WEIGHT", float),
    "initial_eta": ("NESTSAR_INITIAL_ETA", float),
    "initial_alpha": ("NESTSAR_INITIAL_ALPHA", float),
    "batch_size": ("NESTSAR_BATCH_SIZE", int),
    "grad_accum_steps": ("NESTSAR_GRAD_ACCUM_STEPS", int),
    "eval_batch_size": ("NESTSAR_EVAL_BATCH_SIZE", int),
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _paper_mode() -> bool:
    return os.environ.get("NESTSAR_PAPER_MODE", "1").strip().lower() not in {"0", "false", "no"}


def _discover_source(protocol: str) -> Path:
    explicit = os.environ.get("NESTSAR_CANONICAL_SOURCE", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"NESTSAR_CANONICAL_SOURCE does not exist: {path}")
        return path.resolve()

    filename = CANONICAL_FILENAMES[protocol]
    repo_local = Path(__file__).resolve().parent / "canonical" / filename
    candidates: list[Path] = [repo_local]
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if root.exists():
            try:
                candidates.extend(root.rglob(filename))
            except OSError:
                pass

    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        "Validated Attention-Lite source not found. Expected:\n"
        f"  {filename}\n"
        "Place the exact validated all-in-one .py in /kaggle/input or set\n"
        "NESTSAR_CANONICAL_SOURCE=/path/to/the/file."
    )


def _assert_golden_source(source: str, protocol: str) -> None:
    missing = [marker for marker in GOLDEN_MARKERS if marker not in source]
    if missing:
        raise RuntimeError(
            "Canonical source failed Attention-Lite golden fingerprint. Missing: "
            + ", ".join(missing)
        )
    protocol_marker = f'"protocol":\n        "{protocol}"'
    title_marker = f"REAL NTU120 {protocol.upper()} 40-EPOCH FULL RUN"
    if protocol_marker not in source and title_marker not in source:
        raise RuntimeError(f"Source does not look like the validated {protocol.upper()} trainer.")
    if "EPOCHS = 40" not in source:
        raise RuntimeError("Validated source must contain EPOCHS = 40 before patching")


def _env_value(name: str, cast, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except Exception as exc:
        raise ValueError(f"Invalid {name}={raw!r}") from exc


def _resolve_config() -> dict[str, Any]:
    cfg = dict(GOLDEN)
    cfg["epochs"] = _env_value("NESTSAR_EPOCHS", int, GOLDEN["epochs"])
    cfg["patience"] = _env_value("NESTSAR_PATIENCE", int, GOLDEN["patience"])
    for key, (env_name, cast) in ENV_OVERRIDES.items():
        cfg[key] = _env_value(env_name, cast, GOLDEN[key])

    if cfg["epochs"] <= 0:
        raise ValueError("epochs must be > 0")
    if cfg["patience"] < 0:
        raise ValueError("patience must be >= 0; use 0/None to disable")
    if cfg["batch_size"] <= 0 or cfg["grad_accum_steps"] <= 0 or cfg["eval_batch_size"] <= 0:
        raise ValueError("batch sizes and grad_accum_steps must be > 0")
    if cfg["batch_size"] % EXPECTED_TPU_DEVICES:
        raise ValueError(f"batch_size must be divisible by {EXPECTED_TPU_DEVICES} TPU devices")
    if cfg["eval_batch_size"] % EXPECTED_TPU_DEVICES:
        raise ValueError(f"eval_batch_size must be divisible by {EXPECTED_TPU_DEVICES} TPU devices")
    if not (0.0 <= cfg["dropout"] < 1.0):
        raise ValueError("dropout must be in [0,1)")
    if not (0.0 <= cfg["label_smoothing"] < 1.0):
        raise ValueError("label_smoothing must be in [0,1)")
    if not (0.0 <= cfg["warmup_fraction"] < 1.0):
        raise ValueError("warmup_fraction must be in [0,1)")
    if cfg["learning_rate"] <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if cfg["weight_decay"] < 0.0 or cfg["grad_clip"] <= 0.0:
        raise ValueError("weight_decay must be >=0 and grad_clip >0")
    if cfg["predictive_loss_weight"] < 0.0:
        raise ValueError("predictive_loss_weight must be >=0")
    if not (0.0 < cfg["initial_alpha"] <= 1.0):
        raise ValueError("initial_alpha must be in (0,1]")
    if cfg["initial_eta"] <= 0.0:
        raise ValueError("initial_eta must be >0")

    cfg["effective_batch"] = int(cfg["batch_size"] * cfg["grad_accum_steps"])
    return cfg


def _is_golden_recipe(cfg: dict[str, Any]) -> bool:
    return all(cfg[k] == v for k, v in GOLDEN.items())


def _custom_tag(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return "CUSTOM_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _replace_required(pattern: str, replacement: str, text: str, label: str, *, count: int = 0, flags: int = re.MULTILINE) -> tuple[str, int]:
    patched, n = re.subn(pattern, replacement, text, count=count, flags=flags)
    if n < 1:
        raise RuntimeError(f"Could not patch {label}; canonical source format changed")
    return patched, n


def _patch_seed(source: str, seed: int) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    source, counts["SEED"] = _replace_required(
        r"(?m)^SEED\s*=\s*128\s*$", f"SEED = {seed}", source, "SEED"
    )
    source, counts["config_seed"] = _replace_required(
        r"(?m)^(\s*)seed\s*=\s*128,\s*$", rf"\1seed={seed},", source, "Config seed"
    )
    source, counts["prng_seed"] = re.subn(
        r"jax\.random\.PRNGKey\(\s*128\s*\)", f"jax.random.PRNGKey({seed})", source
    )
    if counts["prng_seed"] < 1:
        raise RuntimeError("Could not patch bootstrap PRNG seed; refusing a fake multi-seed run")
    return source, counts


def _patch_cfg_block(source: str, cfg: dict[str, Any]) -> tuple[str, dict[str, int]]:
    start = source.find("ns.CFG = dataclasses.replace(")
    end_marker = "# ==========================================================================================\n# 3. HOPE-ATTENTION LITE TEMPORAL BLOCK"
    end = source.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError("Could not isolate Attention-Lite ns.CFG block")

    before, block, after = source[:start], source[start:end], source[end:]
    counts: dict[str, int] = {}
    field_values = {
        "dropout": cfg["dropout"],
        "batch_size": cfg["batch_size"],
        "grad_accum_steps": cfg["grad_accum_steps"],
        "eval_batch_size": cfg["eval_batch_size"],
        "learning_rate": cfg["learning_rate"],
        "weight_decay": cfg["weight_decay"],
        "warmup_fraction": cfg["warmup_fraction"],
        "label_smoothing": cfg["label_smoothing"],
        "grad_clip": cfg["grad_clip"],
        "predictive_loss_weight": cfg["predictive_loss_weight"],
        "initial_eta": cfg["initial_eta"],
        "initial_alpha": cfg["initial_alpha"],
    }
    for field, value in field_values.items():
        pattern = rf"(?m)^(\s*){re.escape(field)}\s*=\s*[^,\n]+,\s*$"
        block, n = re.subn(pattern, rf"\1{field}={value!r},", block, count=1)
        if n != 1:
            raise RuntimeError(f"Could not patch ns.CFG.{field}")
        counts[field] = n

    marker = f"    eval_batch_size={cfg['eval_batch_size']!r},"
    injection = marker + f"\n    epochs={cfg['epochs']},\n    patience={cfg['patience']},"
    if marker not in block:
        raise RuntimeError("Could not inject ns.CFG epochs/patience")
    block = block.replace(marker, injection, 1)
    counts["epochs_metadata"] = 1
    counts["patience_metadata"] = 1
    return before + block + after, counts


def _patch_locked_training(source: str, cfg: dict[str, Any]) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    replacements = [
        (r"(?m)^EPOCHS\s*=\s*40\s*$", f"EPOCHS = {cfg['epochs']}", "EPOCHS"),
        (r"(?m)^GLOBAL_BATCH\s*=\s*32\s*$", f"GLOBAL_BATCH = {cfg['batch_size']}", "GLOBAL_BATCH"),
        (r"(?m)^GRAD_ACCUM\s*=\s*4\s*$", f"GRAD_ACCUM = {cfg['grad_accum_steps']}", "GRAD_ACCUM"),
        (r"(?m)^EFFECTIVE_BATCH\s*=\s*128\s*$", f"EFFECTIVE_BATCH = {cfg['effective_batch']}", "EFFECTIVE_BATCH"),
    ]
    for pattern, replacement, label in replacements:
        source, counts[label] = _replace_required(pattern, replacement, source, label)

    epoch_line = f"EPOCHS = {cfg['epochs']}"
    source = source.replace(epoch_line, epoch_line + f"\nPATIENCE = {cfg['patience']}", 1)
    counts["PATIENCE"] = 1

    fast_pattern = r"(FAST_ACCUM_TX\s*=\s*\[.*?every_k_schedule\s*=\s*)4(,.*?for _ in range\(NUM_FAST_FRAGMENTS\).*?\])"
    source, counts["FAST_ACCUM_TX"] = _replace_required(
        fast_pattern,
        rf"\g<1>{cfg['grad_accum_steps']}\g<2>",
        source,
        "FAST_ACCUM_TX every_k_schedule",
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )

    assert_pattern = (
        r"assert MICROSTEPS_PER_EPOCH == \d+\s*\n"
        r"assert TOTAL_MICROSTEPS == [\d_]+\s*\n"
        r"assert TOTAL_STEPS == [\d_]+"
    )
    assert_replacement = (
        "assert MICROSTEPS_PER_EPOCH == math.ceil(TRAIN_EXPECTED / GLOBAL_BATCH)\n"
        "assert TOTAL_MICROSTEPS == MICROSTEPS_PER_EPOCH * EPOCHS\n"
        "assert TOTAL_STEPS == math.ceil(TOTAL_MICROSTEPS / GRAD_ACCUM)"
    )
    source, counts["schedule_asserts"] = _replace_required(
        assert_pattern, assert_replacement, source, "schedule assertions", count=1
    )
    return source, counts


def _patch_config_guards(source: str, cfg: dict[str, Any]) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    source, counts["batch_guard"] = _replace_required(
        r"assert ns\.CFG\.batch_size == 32",
        f"assert ns.CFG.batch_size == {cfg['batch_size']}",
        source,
        "batch guard",
        count=1,
    )
    source, counts["accum_guard"] = _replace_required(
        r"assert ns\.CFG\.grad_accum_steps == 4",
        f"assert ns.CFG.grad_accum_steps == {cfg['grad_accum_steps']}",
        source,
        "grad accumulation guard",
        count=1,
    )

    guarded = {
        "dropout": cfg["dropout"],
        "learning_rate": cfg["learning_rate"],
        "weight_decay": cfg["weight_decay"],
        "label_smoothing": cfg["label_smoothing"],
        "predictive_loss_weight": cfg["predictive_loss_weight"],
    }
    for field, value in guarded.items():
        pattern = rf"assert abs\(\s*ns\.CFG\.{field}\s*-\s*[^\n]+\s*\) < 1e-12"
        replacement = f"assert abs(ns.CFG.{field} - {value!r}) < 1e-12"
        source, counts[f"guard_{field}"] = _replace_required(
            pattern,
            replacement,
            source,
            f"{field} guard",
            count=1,
            flags=re.MULTILINE,
        )
    return source, counts


def _patch_early_stopping(source: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    marker = "best_val = -1.0\nbest_epoch = 0"
    if marker not in source:
        raise RuntimeError("Could not locate best metric initialization")
    source = source.replace(marker, marker + "\nepochs_without_improvement = 0", 1)
    counts["patience_state"] = 1

    best_pattern = r"\n    if val_acc > best_val:\n.*?\n    # Always save current weights\."
    best_replacement = r'''
    if val_acc > best_val:

        best_val = val_acc
        best_epoch = epoch
        epochs_without_improvement = 0

        save_weights(
            OUT / "best_ema.msgpack",
            online_params,
            ema_params,
            epoch,
            val_metrics,
        )

        print()
        print(f"🔥 NEW BEST E{epoch}: {100.0 * best_val:.5f}%")

    else:
        epochs_without_improvement += 1
        if PATIENCE > 0:
            print(
                f"Patience: {epochs_without_improvement}/{PATIENCE} "
                f"| best={100.0 * best_val:.5f}%@E{best_epoch}"
            )

    # Always save current weights.'''
    source, counts["best_block"] = _replace_required(
        best_pattern,
        best_replacement,
        source,
        "best/patience block",
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )

    resume_marker = '''    print(
        "Resume state saved."
    )'''
    if resume_marker not in source:
        raise RuntimeError("Could not locate resume-state completion marker")
    early_stop = resume_marker + '''

    if PATIENCE > 0 and epochs_without_improvement >= PATIENCE:
        print(
            f"EARLY STOP at E{epoch}: no validation improvement for "
            f"{epochs_without_improvement} epochs."
        )
        break'''
    source = source.replace(resume_marker, early_stop, 1)
    counts["early_stop_break"] = 1

    result_marker = '''    "epochs":
        EPOCHS,'''
    if result_marker not in source:
        raise RuntimeError("Could not locate result epochs field")
    result_injection = result_marker + '''

    "patience":
        PATIENCE,

    "epochs_ran":
        len(history),'''
    source = source.replace(result_marker, result_injection, 1)
    counts["result_patience"] = 1
    return source, counts


def _patch_output(source: str, output: Path) -> tuple[str, int]:
    pattern = r"OUT\s*=\s*Path\(\s*(?:\"[^\"]*\"\s*)+\)"
    replacement = f"OUT = Path({str(output)!r})"
    patched, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError("Could not uniquely patch OUT path; canonical source format changed")
    return patched, count


def _write_manifest(paths, protocol: str, seed: int, cfg: dict[str, Any], golden_recipe: bool, source_path: Path, source_sha: str, patched_sha: str, patch_counts: dict[str, int]) -> Path:
    manifest = {
        "model": "NestSAR-HOPE-Attention-Lite-D128-v1",
        "protocol": protocol,
        "seed": seed,
        "paper_seeds": list(PAPER_SEEDS),
        "golden_training_recipe": golden_recipe,
        "training_config": cfg,
        "architecture": {
            "frames": EXPECTED_FRAMES,
            "model_dim": EXPECTED_DMODEL,
            "attention_dim": EXPECTED_ATTN_DIM,
            "attention_heads": EXPECTED_HEADS,
            "head_dim": EXPECTED_HEAD_DIM,
            "parameters": EXPECTED_PARAMS,
            "leaves": EXPECTED_LEAVES,
            "gflops_xla_forward": EXPECTED_GFLOPS,
            "expected_tpu_devices": EXPECTED_TPU_DEVICES,
        },
        "canonical_source": str(source_path),
        "canonical_source_sha256": source_sha,
        "generated_source_sha256": patched_sha,
        "patch_counts": patch_counts,
        "output": str(paths.root),
        "created_unix": time.time(),
    }
    target = paths.metadata_dir / "run_manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def _verify_result(root: Path, protocol: str, seed: int, cfg: dict[str, Any]) -> None:
    result_path = root / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"Training process exited but result.json is missing: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checks = {
        "protocol": (str(result.get("protocol", "")).lower(), protocol),
        "seed": (int(result.get("seed", -1)), seed),
        "parameters": (int(result.get("parameters", -1)), EXPECTED_PARAMS),
        "leaves": (int(result.get("leaves", -1)), EXPECTED_LEAVES),
        "frames": (int(result.get("frames", -1)), EXPECTED_FRAMES),
        "global_batch": (int(result.get("global_batch", -1)), int(cfg["batch_size"])),
        "grad_accum": (int(result.get("grad_accum", -1)), int(cfg["grad_accum_steps"])),
        "effective_batch": (int(result.get("effective_batch", -1)), int(cfg["effective_batch"])),
        "patience": (int(result.get("patience", -1)), int(cfg["patience"])),
    }
    bad = [f"{k}: got {got!r}, expected {want!r}" for k, (got, want) in checks.items() if got != want]
    if bad:
        raise RuntimeError("FINAL RUN GUARD FAILED:\n" + "\n".join(bad))


def main() -> int:
    protocol = validate_protocol(os.environ.get("NESTSAR_PROTOCOL", "xsub"))
    paper_mode = _paper_mode()
    seed = validate_seed(int(os.environ.get("NESTSAR_SEED", "128")), paper_mode=paper_mode)
    cfg = _resolve_config()
    golden_recipe = _is_golden_recipe(cfg)

    requested_tag = os.environ.get("NESTSAR_RUN_TAG", "").strip() or None
    tag = requested_tag if requested_tag else (None if golden_recipe else _custom_tag(cfg))

    base_dir = os.environ.get("NESTSAR_RUNS_ROOT", "/kaggle/working")
    paths = make_run_paths(
        protocol,
        seed,
        base_dir=base_dir,
        paper_mode=paper_mode,
        tag=tag,
        create=True,
    )

    if (paths.root / "result.json").exists() and os.environ.get("NESTSAR_ALLOW_OVERWRITE", "0") != "1":
        raise FileExistsError(
            f"Completed run already exists: {paths.root}\n"
            "Refusing to overwrite. Change seed/run_tag/config or set NESTSAR_ALLOW_OVERWRITE=1."
        )

    source_path = _discover_source(protocol)
    source = source_path.read_text(encoding="utf-8")
    _assert_golden_source(source, protocol)
    source_sha = _sha256_text(source)

    patch_counts: dict[str, int] = {}
    patched, c = _patch_seed(source, seed)
    patch_counts.update({f"seed_{k}": v for k, v in c.items()})
    patched, c = _patch_cfg_block(patched, cfg)
    patch_counts.update({f"cfg_{k}": v for k, v in c.items()})
    patched, c = _patch_locked_training(patched, cfg)
    patch_counts.update({f"locked_{k}": v for k, v in c.items()})
    patched, c = _patch_config_guards(patched, cfg)
    patch_counts.update(c)
    patched, c = _patch_early_stopping(patched)
    patch_counts.update(c)
    patched, patch_counts["output"] = _patch_output(patched, paths.root)
    patched_sha = _sha256_text(patched)

    generated = paths.source_dir / f"attention_lite_{protocol}_seed_{seed}_generated.py"
    generated.write_text(patched, encoding="utf-8")
    write_path_manifest(paths)
    _write_manifest(paths, protocol, seed, cfg, golden_recipe, source_path, source_sha, patched_sha, patch_counts)

    print("=" * 108, flush=True)
    print("NESTSAR ATTENTION-LITE — CONFIGURABLE RUN", flush=True)
    print("=" * 108, flush=True)
    print(f"Protocol:           {protocol.upper()}", flush=True)
    print(f"Seed:               {seed}", flush=True)
    print(f"Golden recipe:      {'YES' if golden_recipe else 'NO — custom training config'}", flush=True)
    print(f"Epochs/patience:    {cfg['epochs']}/{cfg['patience']} (0 patience = disabled)", flush=True)
    print(f"Dropout:            {cfg['dropout']}", flush=True)
    print(f"LR / WD / warmup:   {cfg['learning_rate']} / {cfg['weight_decay']} / {cfg['warmup_fraction']}", flush=True)
    print(f"Label smoothing:    {cfg['label_smoothing']}", flush=True)
    print(f"Grad clip:          {cfg['grad_clip']}", flush=True)
    print(f"Pred loss weight:   {cfg['predictive_loss_weight']}", flush=True)
    print(f"eta / alpha:        {cfg['initial_eta']} / {cfg['initial_alpha']}", flush=True)
    print(f"Batch/accum/eff:    {cfg['batch_size']}/{cfg['grad_accum_steps']}/{cfg['effective_batch']}", flush=True)
    print(f"Eval batch:         {cfg['eval_batch_size']}", flush=True)
    print(f"Parameters guard:   {EXPECTED_PARAMS:,}", flush=True)
    print(f"Leaves guard:       {EXPECTED_LEAVES}", flush=True)
    print(f"GFLOPs reference:   {EXPECTED_GFLOPS:.9f}", flush=True)
    print(f"Canonical source:   {source_path}", flush=True)
    print(f"Generated source:   {generated}", flush=True)
    print(f"MAIN OUTPUT FOLDER: {paths.root}", flush=True)
    print("=" * 108, flush=True)

    log_path = paths.logs_dir / "train.log"
    env = os.environ.copy()
    env["NESTSAR_SEED"] = str(seed)
    env["NESTSAR_PROTOCOL"] = protocol
    env["NESTSAR_OUTPUT"] = str(paths.root)

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(generated)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = proc.wait()

    if return_code != 0:
        raise SystemExit(return_code)

    _verify_result(paths.root, protocol, seed, cfg)
    print("FINAL ARCHITECTURE/RUN GUARD: PASS", flush=True)
    print(f"Output: {paths.root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

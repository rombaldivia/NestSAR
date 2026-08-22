#!/usr/bin/env python3
"""Paper-safe launcher for the validated Attention-Lite all-in-one trainer.

This wrapper deliberately does NOT rewrite the model/training mathematics. It finds
one of the already validated protocol-specific all-in-one sources, checks its golden
architecture fingerprints, patches only the experimental seed and output path, saves
the generated source for provenance, and executes it in a clean subprocess.

The exact XSUB/XSET all-in-one file therefore remains the source of truth until the
modular trainer passes TPU parity. This follows REFACTOR_PLAN.md's migration rule.
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

from .paths import PAPER_SEEDS, make_run_paths, validate_protocol, validate_seed, write_path_manifest

EXPECTED_PARAMS = 2_381_028
EXPECTED_LEAVES = 705
EXPECTED_GFLOPS = 0.060416900
EXPECTED_FRAMES = 16
EXPECTED_DMODEL = 128
EXPECTED_ATTN_DIM = 64
EXPECTED_HEADS = 4
EXPECTED_HEAD_DIM = 16
EXPECTED_GLOBAL_BATCH = 32
EXPECTED_GRAD_ACCUM = 4
EXPECTED_EFFECTIVE_BATCH = 128
EXPECTED_TPU_DEVICES = 8
EXPECTED_EPOCHS = 40

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
        "Validated Attention-Lite source not found. Expected one of:\n"
        f"  {filename}\n"
        "Place the exact validated all-in-one .py in /kaggle/input or set\n"
        "NESTSAR_CANONICAL_SOURCE=/path/to/the/file. The wrapper refuses to\n"
        "reconstruct or approximate the paper model."
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
        raise RuntimeError(
            f"Source does not look like the validated {protocol.upper()} trainer."
        )

    if "EPOCHS = 40" not in source:
        raise RuntimeError("Validated paper trainer must contain EPOCHS = 40")


def _replace_once_or_more(pattern: str, replacement: str, text: str, label: str) -> tuple[str, int]:
    patched, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count < 1:
        raise RuntimeError(f"Could not patch {label}; canonical source format changed")
    return patched, count


def _patch_seed(source: str, seed: int) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}

    # Trainer-level experiment seed.
    source, counts["SEED"] = _replace_once_or_more(
        r"(?m)^SEED\s*=\s*128\s*$",
        f"SEED = {seed}",
        source,
        "SEED",
    )

    # Bootstrap Config seed. This controls model/data RNG in the validated source.
    source, counts["config_seed"] = _replace_once_or_more(
        r"(?m)^(\s*)seed\s*=\s*128,\s*$",
        rf"\1seed={seed},",
        source,
        "Config seed",
    )

    # Exact bootstrap model initialization used by the all-in-one source.
    source, counts["prng_seed"] = re.subn(
        r"jax\.random\.PRNGKey\(\s*128\s*\)",
        f"jax.random.PRNGKey({seed})",
        source,
        flags=re.MULTILINE,
    )
    if counts["prng_seed"] < 1:
        # Some saved variants format the call over multiple lines with a bare rng assignment.
        source, counts["prng_seed"] = re.subn(
            r"rng\s*=\s*jax\.random\.PRNGKey\(\s*128\s*\)",
            f"rng = jax.random.PRNGKey({seed})",
            source,
            flags=re.MULTILINE,
        )
    if counts["prng_seed"] < 1:
        raise RuntimeError("Could not patch bootstrap PRNG seed; refusing a fake multi-seed run")

    return source, counts


def _patch_output(source: str, output: Path) -> tuple[str, int]:
    # Match the validated multiline block:
    # OUT = Path(
    #     "/kaggle/working/"
    #     "NestSAR_..."
    # )
    pattern = r"OUT\s*=\s*Path\(\s*(?:\"[^\"]*\"\s*)+\)"
    replacement = f"OUT = Path({str(output)!r})"
    patched, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError("Could not uniquely patch OUT path; canonical source format changed")
    return patched, count


def _write_manifest(paths, protocol: str, seed: int, source_path: Path, source_sha: str, patched_sha: str, patch_counts: dict[str, int]) -> Path:
    manifest = {
        "model": "NestSAR-HOPE-Attention-Lite-D128-v1",
        "protocol": protocol,
        "seed": seed,
        "paper_seeds": list(PAPER_SEEDS),
        "epochs": EXPECTED_EPOCHS,
        "frames": EXPECTED_FRAMES,
        "model_dim": EXPECTED_DMODEL,
        "attention_dim": EXPECTED_ATTN_DIM,
        "attention_heads": EXPECTED_HEADS,
        "head_dim": EXPECTED_HEAD_DIM,
        "parameters": EXPECTED_PARAMS,
        "leaves": EXPECTED_LEAVES,
        "gflops_xla_forward": EXPECTED_GFLOPS,
        "global_batch": EXPECTED_GLOBAL_BATCH,
        "grad_accum": EXPECTED_GRAD_ACCUM,
        "effective_batch": EXPECTED_EFFECTIVE_BATCH,
        "expected_tpu_devices": EXPECTED_TPU_DEVICES,
        "canonical_source": str(source_path),
        "canonical_source_sha256": source_sha,
        "generated_source_sha256": patched_sha,
        "seed_patch_counts": patch_counts,
        "output": str(paths.root),
        "created_unix": time.time(),
    }
    target = paths.metadata_dir / "run_manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def _verify_result(root: Path, protocol: str, seed: int) -> None:
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
        "global_batch": (int(result.get("global_batch", -1)), EXPECTED_GLOBAL_BATCH),
        "grad_accum": (int(result.get("grad_accum", -1)), EXPECTED_GRAD_ACCUM),
        "effective_batch": (int(result.get("effective_batch", -1)), EXPECTED_EFFECTIVE_BATCH),
    }
    bad = [f"{k}: got {got!r}, expected {want!r}" for k, (got, want) in checks.items() if got != want]
    if bad:
        raise RuntimeError("FINAL PAPER-RUN GUARD FAILED:\n" + "\n".join(bad))


def main() -> int:
    protocol = validate_protocol(os.environ.get("NESTSAR_PROTOCOL", "xsub"))
    paper_mode = _paper_mode()
    seed = validate_seed(int(os.environ.get("NESTSAR_SEED", "128")), paper_mode=paper_mode)
    epochs = int(os.environ.get("NESTSAR_EPOCHS", str(EXPECTED_EPOCHS)))
    if paper_mode and epochs != EXPECTED_EPOCHS:
        raise ValueError(
            f"Paper mode is locked to {EXPECTED_EPOCHS} epochs; got {epochs}. "
            "Set NESTSAR_PAPER_MODE=0 only for non-paper experiments."
        )

    base_dir = os.environ.get("NESTSAR_RUNS_ROOT", "/kaggle/working")
    paths = make_run_paths(protocol, seed, base_dir=base_dir, paper_mode=paper_mode, create=True)

    if (paths.root / "result.json").exists() and os.environ.get("NESTSAR_ALLOW_OVERWRITE", "0") != "1":
        raise FileExistsError(
            f"Completed run already exists: {paths.root}\n"
            "Refusing to overwrite. Use a new seed/protocol or explicitly set NESTSAR_ALLOW_OVERWRITE=1."
        )

    source_path = _discover_source(protocol)
    source = source_path.read_text(encoding="utf-8")
    _assert_golden_source(source, protocol)
    source_sha = _sha256_text(source)

    patched, patch_counts = _patch_seed(source, seed)
    patched, _ = _patch_output(patched, paths.root)
    patched_sha = _sha256_text(patched)

    generated = paths.source_dir / f"attention_lite_{protocol}_seed_{seed}_generated.py"
    generated.write_text(patched, encoding="utf-8")
    write_path_manifest(paths)
    _write_manifest(paths, protocol, seed, source_path, source_sha, patched_sha, patch_counts)

    print("=" * 108, flush=True)
    print("NESTSAR ATTENTION-LITE — PAPER MULTI-SEED RUN", flush=True)
    print("=" * 108, flush=True)
    print(f"Protocol:           {protocol.upper()}", flush=True)
    print(f"Seed:               {seed}", flush=True)
    print(f"Epochs:             {epochs}", flush=True)
    print(f"Parameters guard:   {EXPECTED_PARAMS:,}", flush=True)
    print(f"Leaves guard:       {EXPECTED_LEAVES}", flush=True)
    print(f"GFLOPs reference:   {EXPECTED_GFLOPS:.9f}", flush=True)
    print(f"Batch/accum/eff:    {EXPECTED_GLOBAL_BATCH}/{EXPECTED_GRAD_ACCUM}/{EXPECTED_EFFECTIVE_BATCH}", flush=True)
    print(f"Canonical source:   {source_path}", flush=True)
    print(f"Canonical SHA256:   {source_sha}", flush=True)
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

    _verify_result(paths.root, protocol, seed)
    print("PAPER-RUN FINAL GUARD: PASS", flush=True)
    print(f"Output: {paths.root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

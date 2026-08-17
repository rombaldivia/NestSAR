#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Robust compatibility shim for the HOPE-Fidelity universal launcher.

The exact audited v4.1 self-contained artifact contains an immutable source ZIP.
Different v4.1 one-cell revisions applied the same scheduler/smoke/LR fixes with
slightly different surrounding source text, so exact multi-line string matching
is too brittle. This shim performs the same normalization semantically, then
hands control back to run_universal.py.
"""

import re
from pathlib import Path

import run_universal as base


def normalize_extracted_v41_robust(runtime: Path) -> None:
    core_path = runtime / "nestsar.py"
    core = core_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # 1) LR schedule must count optimizer updates, not microbatches.
    # ------------------------------------------------------------------
    if "total_micro_steps = CFG.epochs * steps_per_epoch" not in core:
        old_total = """    steps_per_epoch = max(1, math.ceil(len(train_dataset) / CFG.batch_size))
    total_steps = CFG.epochs * steps_per_epoch

    model = build_model(model_id)
"""
        new_total = """    steps_per_epoch = max(1, math.ceil(len(train_dataset) / CFG.batch_size))
    total_micro_steps = CFG.epochs * steps_per_epoch
    total_steps = max(
        1,
        math.ceil(total_micro_steps / max(1, CFG.grad_accum_steps)),
    )
    optimizer_steps_per_epoch = (
        steps_per_epoch / max(1, CFG.grad_accum_steps)
    )

    model = build_model(model_id)
"""
        if old_total in core:
            core = core.replace(old_total, new_total, 1)
            print("[v4.1 PATCH] LR schedule counts effective optimizer updates")
        else:
            # Flexible fallback for cosmetic/comment differences.
            pattern = re.compile(
                r"(?m)^(?P<i>\s*)steps_per_epoch\s*=\s*max\(1,\s*math\.ceil\(len\(train_dataset\)\s*/\s*CFG\.batch_size\)\)\s*\n"
                r"(?P=i)total_steps\s*=\s*CFG\.epochs\s*\*\s*steps_per_epoch\s*$"
            )
            m = pattern.search(core)
            if not m:
                raise RuntimeError("Could not normalize v4.1 total_steps accounting")
            i = m.group("i")
            replacement = (
                f"{i}steps_per_epoch = max(1, math.ceil(len(train_dataset) / CFG.batch_size))\n"
                f"{i}total_micro_steps = CFG.epochs * steps_per_epoch\n"
                f"{i}total_steps = max(\n"
                f"{i}    1,\n"
                f"{i}    math.ceil(total_micro_steps / max(1, CFG.grad_accum_steps)),\n"
                f"{i})\n"
                f"{i}optimizer_steps_per_epoch = (\n"
                f"{i}    steps_per_epoch / max(1, CFG.grad_accum_steps)\n"
                f"{i})"
            )
            core = core[:m.start()] + replacement + core[m.end():]
            print("[v4.1 PATCH] LR schedule normalized by semantic fallback")
    else:
        print("[v4.1 PATCH] LR schedule already normalized")

    # ------------------------------------------------------------------
    # 2) Effective-batch logging and redundant core smoke omission.
    #    Match semantics, not one exact surrounding block.
    # ------------------------------------------------------------------
    if "batch efectivo≈" not in core:
        parameter_log = '    log(f"Parámetros: {parameter_count:,}")'
        if parameter_log in core:
            logging = '''    log(f"Parámetros: {parameter_count:,}")
    log(
        f"Batch físico={CFG.batch_size} | "
        f"acumulación={CFG.grad_accum_steps} | "
        f"batch efectivo≈{CFG.batch_size * CFG.grad_accum_steps} | "
        f"updates/época≈{optimizer_steps_per_epoch:.2f}"
    )'''
            core = core.replace(parameter_log, logging, 1)
            print("[v4.1 PATCH] effective-batch logging inserted")
        else:
            print("[v4.1 PATCH] WARNING: parameter log anchor not found; continuing")

    if 'log("Smoke test OMITIDO para esta corrida.")' in core:
        print("[v4.1 PATCH] redundant smoke already omitted")
    else:
        # Only replace an actual call line, never the function definition.
        smoke_pattern = re.compile(
            r"(?m)^(?P<i>[ \t]+)smoke_test\([^\n]*\)\s*$"
        )
        matches = list(smoke_pattern.finditer(core))
        if matches:
            # Prefer the call occurring after build_steps in run_experiment.
            chosen = None
            build_pos = core.find("train_step, eval_step = build_steps(model, model_id)")
            if build_pos >= 0:
                for m in matches:
                    if m.start() > build_pos:
                        chosen = m
                        break
            if chosen is None:
                chosen = matches[0]
            indent = chosen.group("i")
            replacement = indent + 'log("Smoke test OMITIDO para esta corrida.")'
            core = core[:chosen.start()] + replacement + core[chosen.end():]
            print("[v4.1 PATCH] redundant core smoke omitted by semantic match")
        else:
            # Not correctness-critical. Some exact v4.1 revisions already omit
            # the smoke using different wording.
            print("[v4.1 PATCH] no core smoke call found; leaving training-start logic unchanged")

    # ------------------------------------------------------------------
    # 3) Displayed LR must use optimizer-update step under accumulation.
    # ------------------------------------------------------------------
    effective_lr_markers = (
        "lr_step = state.step // max(1, CFG.grad_accum_steps)",
        "lr_step = state.step // CFG.grad_accum_steps",
        "lr_step = int(np.asarray(state.step)) // max(1, CFG.grad_accum_steps)",
    )
    if any(marker in core for marker in effective_lr_markers):
        print("[v4.1 PATCH] displayed LR already uses effective optimizer step")
    else:
        old_lr = "lr = float(np.asarray(make_schedule(total_steps)(state.step)))"
        new_lr = """lr_step = state.step // max(1, CFG.grad_accum_steps)
            lr = float(np.asarray(make_schedule(total_steps)(lr_step)))"""
        if old_lr in core:
            core = core.replace(old_lr, new_lr, 1)
            print("[v4.1 PATCH] displayed LR uses effective optimizer step")
        else:
            pattern = re.compile(
                r"(?m)^(?P<i>\s*)lr\s*=\s*float\(np\.asarray\(make_schedule\(total_steps\)\(state\.step\)\)\)\s*$"
            )
            m = pattern.search(core)
            if not m:
                raise RuntimeError("Could not normalize v4.1 LR logger")
            i = m.group("i")
            replacement = (
                f"{i}lr_step = state.step // max(1, CFG.grad_accum_steps)\n"
                f"{i}lr = float(np.asarray(make_schedule(total_steps)(lr_step)))"
            )
            core = core[:m.start()] + replacement + core[m.end():]
            print("[v4.1 PATCH] displayed LR normalized by semantic fallback")

    if "grad_accum_steps" not in core or "--grad-accum-steps" not in core:
        raise RuntimeError("Extracted v4.1 source has no gradient-accumulation CLI support")

    core_path.write_text(core, encoding="utf-8")

    # ------------------------------------------------------------------
    # 4) Canonical EMA / RegMask values.
    # ------------------------------------------------------------------
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
    print("Exact v4.1 robust post-extraction normalization: PASS")


base.normalize_extracted_v41 = normalize_extracted_v41_robust

if __name__ == "__main__":
    raise SystemExit(base.main())

#!/usr/bin/env python3
"""Corrected single-T4 Attention-Lite adapter.

This module reuses trainer_t4's hardware/runtime adapter and replaces only the
notebook-output patch.  The previous quiet patch inserted ``disable=True`` before
the positional ``iterator`` argument of ``tqdm`` and therefore generated invalid
Python ("positional argument follows keyword argument").

V2 inserts ``disable=True`` immediately *after* the positional iterator in both
the train and validation progress bars.  Training/evaluation semantics are
unchanged; only tqdm rendering is disabled.
"""
from __future__ import annotations

import re

from . import trainer_t4 as base

RUNNER_API_VERSION = "attention-lite-single-t4-v2-valid-quiet-tqdm"


def _patch_tqdm_quiet_v2(source: str) -> tuple[str, int]:
    """Disable train+val tqdm without changing their positional iterator."""
    # Canonical form has a positional iterator first:
    #
    #     pbar = tqdm(
    #         iterator,
    #         total=...,
    #         desc=...,
    #     )
    #
    # Put the keyword only AFTER that positional argument.
    pattern = (
        r"(?m)^([ \t]*)pbar[ \t]*=[ \t]*tqdm\([ \t]*\n"
        r"(?:[ \t]*\n)*"
        r"([ \t]*)iterator,[ \t]*\n"
    )

    def repl(match: re.Match[str]) -> str:
        call_indent = match.group(1)
        arg_indent = match.group(2)
        return (
            f"{call_indent}pbar = tqdm(\n"
            f"{arg_indent}iterator,\n"
            f"{arg_indent}disable=True,\n"
        )

    patched, count = re.subn(pattern, repl, source)
    if count != 2:
        raise RuntimeError(
            f"Expected exactly 2 tqdm progress bars (train+val); found {count}"
        )

    # Strong syntax-shape guards before base._patch_t4_runtime compiles the full
    # generated source.
    if patched.count("disable=True") != 2:
        raise RuntimeError("Quiet tqdm patch did not create exactly two disable=True markers")
    if re.search(r"tqdm\(\s*disable=True,\s*iterator,", patched, flags=re.DOTALL):
        raise RuntimeError("Invalid tqdm argument order detected after quiet patch")

    return patched, count


# base._patch_t4_runtime resolves this name from trainer_t4 module globals at call
# time, so replacing it here repairs both source selftests and real generated jobs.
base._patch_tqdm_quiet = _patch_tqdm_quiet_v2


def main() -> int:
    print("=" * 108, flush=True)
    print(f"RUNNER API: {RUNNER_API_VERSION}", flush=True)
    print("QUIET TQDM V2: positional iterator preserved; disable=True follows it", flush=True)
    print("=" * 108, flush=True)
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())

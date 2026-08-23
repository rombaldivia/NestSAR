#!/usr/bin/env python3
"""Static selftest for corrected dual-T4 quiet source patch."""
from __future__ import annotations

import os
from pathlib import Path

from .canonical_integrated import ensure_canonical_sources
from . import trainer_t4_v2 as fixed


def main() -> int:
    sources_raw = ensure_canonical_sources(verbose=True)

    for protocol in ("xsub", "xset"):
        path = Path(sources_raw[protocol]).resolve()
        text = path.read_text(encoding="utf-8")

        old_protocol = os.environ.get("NESTSAR_PROTOCOL")
        old_gpu = os.environ.get("NESTSAR_PHYSICAL_GPU")
        os.environ["NESTSAR_PROTOCOL"] = protocol
        os.environ["NESTSAR_PHYSICAL_GPU"] = "0" if protocol == "xsub" else "1"

        try:
            # trainer_t4_v2 patched trainer_t4._patch_tqdm_quiet at import time.
            patched, counts = fixed.base._patch_t4_runtime(text)
        finally:
            if old_protocol is None:
                os.environ.pop("NESTSAR_PROTOCOL", None)
            else:
                os.environ["NESTSAR_PROTOCOL"] = old_protocol
            if old_gpu is None:
                os.environ.pop("NESTSAR_PHYSICAL_GPU", None)
            else:
                os.environ["NESTSAR_PHYSICAL_GPU"] = old_gpu

        # This is the guard that failed in the previous commit.
        compile(patched, f"<t4-v2-{protocol}-selftest>", "exec")

        expected_counts = {
            "protocol_root": 1,
            "backend_guard": 1,
            "device_guard": 1,
            "result_runtime_metadata": 1,
            "runtime_banner": 1,
            "tqdm_quiet": 2,
        }
        if counts != expected_counts:
            raise RuntimeError(
                f"{protocol.upper()} T4 patch-count guard failed: {counts}"
            )

        if patched.count("disable=True") != 2:
            raise RuntimeError(
                f"{protocol.upper()} expected two disable=True markers"
            )

        # Explicitly reject the exact syntax ordering that caused the crash.
        bad = "tqdm(\n        disable=True,\n\n        iterator,"
        if bad in patched:
            raise RuntimeError(
                f"{protocol.upper()} still has keyword-before-positional tqdm ordering"
            )

        print(
            f"{protocol.upper()} T4 QUIET SOURCE: PASS | counts={counts}",
            flush=True,
        )

    print("=" * 108, flush=True)
    print("NESTSAR ATTENTION-LITE DUAL-T4 QUIET V2 SELFTEST: PASS", flush=True)
    print("Generated XSUB/XSET sources compile successfully.", flush=True)
    print("Per-batch tqdm rendering: DISABLED", flush=True)
    print("Epoch/validation summaries: RETAINED", flush=True)
    print("=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

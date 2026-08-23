#!/usr/bin/env python3
"""Static preflight for the dual-T4 Attention-Lite runtime adapter.

No JAX runtime is created here. The test reconstructs both canonical protocol
sources, applies the single-T4 hardware-only patch, verifies that each patched
source compiles, and checks the runtime markers needed by the dual-GPU launcher.
"""
from __future__ import annotations

from pathlib import Path

from .canonical_integrated import ensure_canonical_sources
from .trainer_t4 import _patch_t4_runtime


def main() -> int:
    sources_raw = ensure_canonical_sources(verbose=True)

    for protocol in ("xsub", "xset"):
        path = Path(sources_raw[protocol]).resolve()
        text = path.read_text(encoding="utf-8")

        # trainer_t4 reads protocol/GPU assignment from the environment at runtime.
        import os
        old_protocol = os.environ.get("NESTSAR_PROTOCOL")
        old_gpu = os.environ.get("NESTSAR_PHYSICAL_GPU")
        os.environ["NESTSAR_PROTOCOL"] = protocol
        os.environ["NESTSAR_PHYSICAL_GPU"] = "0" if protocol == "xsub" else "1"
        try:
            patched, counts = _patch_t4_runtime(text)
        finally:
            if old_protocol is None:
                os.environ.pop("NESTSAR_PROTOCOL", None)
            else:
                os.environ["NESTSAR_PROTOCOL"] = old_protocol
            if old_gpu is None:
                os.environ.pop("NESTSAR_PHYSICAL_GPU", None)
            else:
                os.environ["NESTSAR_PHYSICAL_GPU"] = old_gpu

        compile(patched, f"<t4-{protocol}-selftest>", "exec")

        required = {
            "protocol_root",
            "backend_guard",
            "device_guard",
            "result_runtime_metadata",
            "runtime_banner",
        }
        if set(counts) != required or any(counts[key] != 1 for key in required):
            raise RuntimeError(
                f"{protocol.upper()} T4 patch-count guard failed: {counts}"
            )

        markers = (
            'jax.default_backend() != "gpu"',
            "Expected exactly 1 process-visible GPU",
            '"runtime_topology":\n        "isolated_single_t4"',
            f"NestSAR_HOPE_FIDELITY_UNIVERSAL_{protocol.upper()}_T4",
        )
        missing = [marker for marker in markers if marker not in patched]
        if missing:
            raise RuntimeError(
                f"{protocol.upper()} T4 patched source missing markers: {missing}"
            )

        print(
            f"{protocol.upper()} T4 SOURCE PATCH: PASS | "
            f"counts={counts}",
            flush=True,
        )

    print("=" * 108, flush=True)
    print("NESTSAR ATTENTION-LITE DUAL-T4 STATIC SELFTEST: PASS", flush=True)
    print("No JAX/CUDA runtime was initialized by this test.", flush=True)
    print("=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

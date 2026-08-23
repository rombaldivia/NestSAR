#!/usr/bin/env python3
"""Static self-test for repository-bundled Attention-Lite sources."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .source_resolver import (
    _extract_embedded_bundle,
    resolve_both_sources,
    validate_canonical_source_text,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    sources = resolve_both_sources(verbose=True)

    texts = {}
    for protocol in ("xsub", "xset"):
        path = Path(sources[protocol])
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        validate_canonical_source_text(text, protocol, origin=str(path))
        compile(text, str(path), "exec")
        texts[protocol] = text

    xsub_bundle = _extract_embedded_bundle(texts["xsub"])
    xset_bundle = _extract_embedded_bundle(texts["xset"])
    if xsub_bundle != xset_bundle:
        raise RuntimeError("XSUB/XSET embedded v4.1 bundles differ")

    print("=" * 108)
    print("NESTSAR ATTENTION-LITE — REPOSITORY CANONICAL SELFTEST: PASS")
    print("=" * 108)
    for protocol in ("xsub", "xset"):
        path = Path(sources[protocol])
        print(f"{protocol.upper()} | {path}")
        print(f"       sha256={sha256(path)} | bytes={path.stat().st_size:,}")
    print("Embedded BUNDLE_B64: IDENTICAL across protocols")
    print("Architecture guards: 2,381,028 params / 705 leaves / T16 / D64-H4")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

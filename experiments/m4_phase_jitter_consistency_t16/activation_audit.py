#!/usr/bin/env python3
from __future__ import annotations

"""Read-only activation/robustness audit for the Jitter+Consistency champion.

The consistency experiment changes TRAINING ONLY. Its inference architecture,
15-channel Phase-T16 representation, fixed uniform fusion, router, temporal
hierarchy, and checkpoint parameter tree are identical to Jitter+Uniform.
Therefore the already validated Jitter+Uniform activation audit is the exact
appropriate inference-time diagnostic here.

This dedicated entry point exists so experiment provenance is explicit while
reusing one audited implementation for:
  * canonical validation accuracy
  * +/-1 boundary-jitter validation accuracy
  * canonical<->jitter prediction agreement
  * router-OFF contribution
  * per-stream / leave-one-stream-out counterfactuals
  * pose / full-displacement / phase-A / phase-B / path ablations
  * frozen Ridge probes through spatial, frame memory, router, chunk memory,
    and descriptor stages
  * activation effective rank
  * router robustness
  * hardest and most jitter-sensitive classes

The audit is read-only and never modifies a checkpoint.
"""

import sys
from pathlib import Path

from flax import serialization

from experiments.m4_phase_jitter_uniform_t16 import activation_audit as base_audit


def _checkpoint_from_argv() -> Path | None:
    try:
        i = sys.argv.index("--checkpoint")
    except ValueError:
        return None
    if i + 1 >= len(sys.argv):
        return None
    return Path(sys.argv[i + 1])


def _print_consistency_metadata() -> None:
    ckpt = _checkpoint_from_argv()
    if ckpt is None or not ckpt.is_file():
        return
    payload = serialization.msgpack_restore(ckpt.read_bytes())
    model = payload.get("model", "unknown")
    cfg = payload.get("config", {})
    rep = payload.get("representation", {})
    print("=" * 122, flush=True)
    print("JITTER+CONSISTENCY CHECKPOINT PROVENANCE", flush=True)
    print("=" * 122, flush=True)
    print(f"model                  : {model}", flush=True)
    print(f"consistency            : {rep.get('consistency', 'unknown')}", flush=True)
    print(f"consistency weight     : {rep.get('consistency_weight', cfg.get('consistency_weight', 'unknown'))}", flush=True)
    print(f"consistency temperature: {rep.get('consistency_temperature', cfg.get('consistency_temperature', 'unknown'))}", flush=True)
    print(f"jitter max shift       : {rep.get('jitter_max_shift', cfg.get('jitter_max_shift', 'unknown'))}", flush=True)
    print(f"final fusion           : {rep.get('final_fusion', 'uniform_mean')}", flush=True)
    print("Reusing validated Jitter+Uniform inference audit because architecture/representation are identical.", flush=True)
    print("=" * 122, flush=True)


def main() -> None:
    _print_consistency_metadata()
    base_audit.main()


if __name__ == "__main__":
    main()

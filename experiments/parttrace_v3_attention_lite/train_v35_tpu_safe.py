#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safety wrapper for the v3.5 TPU trainer.

Adds experiment-integrity guards without duplicating the main trainer:
- checkpoint payload protocol must match the requested NTU120 protocol when declared;
- derivative PartTrace/TokenPreserve/CrossStream checkpoints are not auto-used as the base;
- saved checkpoints record the exact branch/base ramp scales of their best epoch.
"""
from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import jax.numpy as jnp

from experiments.parttrace_v3_attention_lite import train_v35_tpu as tr

_ORIGINAL_TO_BYTES = tr.serialization.to_bytes
_DERIVATIVE_MARKERS = ("parttrace", "tokenpreserve", "crossstream", "cross-stream")


def _safe_load_compatible_base(spec: str, protocol: str, template):
    errors = []
    explicit = spec.lower() not in ("auto", "none", "scratch")
    for path in tr._checkpoint_candidates(spec, protocol):
        try:
            payload = tr.serialization.msgpack_restore(path.read_bytes())
            if isinstance(payload, Mapping):
                declared = payload.get("protocol")
                if declared is not None:
                    if isinstance(declared, bytes):
                        declared = declared.decode("utf-8", errors="replace")
                    declared = str(declared).lower().strip()
                    if declared and declared != protocol:
                        errors.append(
                            f"{path}: declared protocol={declared!r}, requested={protocol!r}"
                        )
                        continue

                # In AUTO mode, insist on a clean Attention-Lite lineage. An explicit
                # path is allowed to override this for intentional continuation runs.
                model_name = payload.get("model")
                if model_name is not None and not explicit:
                    if isinstance(model_name, bytes):
                        model_name = model_name.decode("utf-8", errors="replace")
                    model_text = str(model_name).lower()
                    if any(marker in model_text for marker in _DERIVATIVE_MARKERS):
                        errors.append(
                            f"{path}: derivative checkpoint model={model_name!r}; skipped in auto mode"
                        )
                        continue

            roots = []
            if isinstance(payload, Mapping):
                for key in ("ema_params", "params"):
                    if key in payload:
                        roots.append(payload[key])
            roots.append(payload)
            for root in roots:
                match_path, subtree = tr._find_matching_subtree(root, template)
                if subtree is not None:
                    loaded = tr.jax.tree_util.tree_map(lambda z: jnp.asarray(z), subtree)
                    return loaded, path, match_path
            errors.append(f"{path}: no canonical-shaped Attention-Lite subtree")
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return None, None, errors


def _to_bytes_with_runtime_scales(payload):
    if isinstance(payload, dict) and payload.get("model") == tr.MODEL_NAME:
        payload = dict(payload)
        epoch = int(payload.get("epoch", 0))
        args = payload.get("args", {}) or {}
        freeze_branch = int(args.get("freeze_branch_epochs", 2))
        ramp_branch = int(args.get("branch_ramp_epochs", 4))
        freeze_base = int(args.get("freeze_base_epochs", 3))
        ramp_base = int(args.get("base_unfreeze_ramp_epochs", 3))
        payload["branch_scale"] = tr.branch_scale_for_epoch(
            epoch, freeze_branch, ramp_branch
        ) if epoch > 0 else 0.0
        payload["base_grad_scale"] = tr.base_grad_scale_for_epoch(
            epoch, freeze_base, ramp_base
        ) if epoch > 0 else 0.0
        payload["checkpoint_semantics"] = (
            "Evaluate this EMA with the stored branch_scale; epoch-0 is canonical base only."
        )
    return _ORIGINAL_TO_BYTES(payload)


def main() -> int:
    original_loader = tr.load_compatible_base
    original_to_bytes = tr.serialization.to_bytes
    tr.load_compatible_base = _safe_load_compatible_base
    tr.serialization.to_bytes = _to_bytes_with_runtime_scales
    try:
        return int(tr.main())
    finally:
        tr.load_compatible_base = original_loader
        tr.serialization.to_bytes = original_to_bytes


if __name__ == "__main__":
    raise SystemExit(main())

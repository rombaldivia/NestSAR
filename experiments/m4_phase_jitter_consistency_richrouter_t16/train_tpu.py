#!/usr/bin/env python3
from __future__ import annotations

"""Train the T16 champion with ONLY the post-frame router replaced.

All optimization, Phase15 preprocessing, canonical+jitter dual-view training,
consistency loss, EMA, uniform final fusion, batch size and scheduler are reused
from the verified Phase+Jitter+Consistency trainer.
"""

import json
import sys
from pathlib import Path

from flax import serialization

from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_t16 import train_tpu as cons
from experiments.m4_phase_jitter_consistency_richrouter_t16.model import (
    M4PhaseRichRouterT16,
    EXPECTED_PARAMS,
)

DEFAULT_OUTDIR = "/kaggle/working/NestSAR_M4_Phase_JitterConsistency_RichRouter_T16_TPU"


def _arg_value(flag: str, default: str) -> str:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def _patch_checkpoint_metadata(outdir: Path) -> None:
    for protocol in ("xsub", "xset"):
        d = outdir / protocol
        msg = d / "best.msgpack"
        js = d / "best.json"

        if msg.is_file():
            payload = serialization.msgpack_restore(msg.read_bytes())
            if isinstance(payload, dict):
                payload["model"] = "M4PhaseJitterConsistencyRichRouterT16"
                rep = dict(payload.get("representation", {}))
                rep.update({
                    "router": "RichCrossStreamRouter",
                    "router_position": "post_frame_memory",
                    "router_context_mlp": "112->224->112",
                    "router_gate": "vector_224_to_112",
                    "router_residual_scale": 0.15,
                    "router_only_architecture_change": True,
                })
                payload["representation"] = rep
                msg.write_bytes(serialization.msgpack_serialize(payload))

        if js.is_file():
            meta = json.loads(js.read_text())
            meta.update({
                "model": "M4PhaseJitterConsistencyRichRouterT16",
                "router": "RichCrossStreamRouter",
                "router_context_mlp": "112->224->112",
                "router_gate": "vector_224_to_112",
                "router_residual_scale": 0.15,
                "router_only_architecture_change": True,
            })
            js.write_text(json.dumps(meta, indent=2))

    manifest = {
        "experiment": "M4PhaseJitterConsistencyRichRouterT16",
        "baseline": "M4PhaseJitterConsistencyT16",
        "architecture_change": "post-frame CrossStreamRouter only",
        "expected_params": EXPECTED_PARAMS,
        "router": {
            "stream_scores": "Dense(112->1)+softmax across four streams",
            "context": "weighted normalized stream sum",
            "context_mlp": "112->224->112 with GELU",
            "gate": "vector gate Dense(224->112) from [local_stream, context]",
            "residual_scale": 0.15,
        },
        "unchanged": [
            "Phase15 representation",
            "16 temporal tokens",
            "four J/B/JM/BM streams",
            "spatial encoders",
            "per-stream frame BiMemory",
            "descriptor/chunk memory",
            "uniform final fusion",
            "canonical+jitter dual-view training",
            "symmetric-KL consistency weight 0.08",
            "stream auxiliary CE weight 0.15",
            "EMA 0.995",
        ],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    # Reuse the exact champion training loop while swapping only the model class.
    ju.M4PhaseUniformT16 = M4PhaseRichRouterT16
    ju.EXPECTED_PARAMS = EXPECTED_PARAMS
    cons.EXPECTED_PARAMS = EXPECTED_PARAMS

    outdir = Path(_arg_value("--outdir", DEFAULT_OUTDIR))

    print("=" * 120, flush=True)
    print("M4 PHASE+JITTER+CONSISTENCY — RICH ROUTER ABLATION", flush=True)
    print(f"EXPECTED_PARAMS={EXPECTED_PARAMS:,}", flush=True)
    print("ONLY ARCHITECTURE CHANGE: post-frame CrossStreamRouter", flush=True)
    print("=" * 120, flush=True)

    cons.main()
    _patch_checkpoint_metadata(outdir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""Single-T4 from-scratch worker for LocalGlobal V2 + BiJoint M4/G4."""

import json
import time
from pathlib import Path

import jax
from flax import serialization

from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_t16 import train_tpu as cons
from experiments.m4_phase_jitter_consistency_localglobal_t16 import train_tpu as lg_train
from experiments.m4_phase_jitter_consistency_localglobal_bijoint_t16.model import (
    EXPECTED_PARAMS,
    M4PhaseUniformBiJointT16,
)

BASELINE_ACCURACY = {
    "xsub": 0.7531176967340285,
    "xset": 0.7592682885821410,
}


def compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


class SilentProgress:
    """tqdm-compatible iterable that avoids piped carriage-return spam."""

    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else ()

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return None

    def refresh(self, *args, **kwargs):
        return None

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def install_clean_progress(protocol: str) -> None:
    ju.tqdm = SilentProgress
    cons.tqdm = SilentProgress

    original_build = ju.build_protocol_views

    def clean_build(*args, **kwargs):
        start = time.time()
        print(
            f"{protocol.upper()} PREPROCESSING START | "
            "LocalGlobal canonical+jitter train + canonical val",
            flush=True,
        )
        result = original_build(*args, **kwargs)
        print(
            f"{protocol.upper()} PREPROCESSING READY | "
            f"time={time.time()-start:.1f}s",
            flush=True,
        )
        return result

    ju.build_protocol_views = clean_build


def patch_saved_metadata(outdir: Path, protocol: str) -> None:
    d = outdir / protocol
    msg = d / "best.msgpack"
    js = d / "best.json"

    if msg.is_file():
        payload = serialization.msgpack_restore(msg.read_bytes())
        if isinstance(payload, dict):
            payload["model"] = "M4LocalGlobalBiJointT16"
            rep = dict(payload.get("representation", {}))
            rep.update(
                {
                    "preprocessing": "local_pose_global_motion_v2",
                    "spatial_joint_memory": "bidirectional_bimemory",
                    "joint_order": "existing_kinematic_order",
                    "attention": False,
                    "qkv": False,
                    "architecture_change": True,
                    "training_from_scratch": True,
                    "runtime_backend": "single_visible_gpu",
                }
            )
            payload["representation"] = rep
            msg.write_bytes(serialization.msgpack_serialize(payload))

    if js.is_file():
        meta = json.loads(js.read_text())
        meta.update(
            {
                "model": "M4LocalGlobalBiJointT16",
                "preprocessing": "local_pose_global_motion_v2",
                "spatial_joint_memory": "bidirectional_bimemory",
                "attention": False,
                "qkv": False,
                "architecture_change": True,
                "training_from_scratch": True,
            }
        )
        js.write_text(json.dumps(meta, indent=2))


def main() -> None:
    # EXACT LocalGlobal V2 preprocessing for both canonical and jitter views.
    lg_train.install_preprocessing_override()

    # Patch only the neural model constructed by the established consistency trainer.
    ju.M4PhaseUniformT16 = M4PhaseUniformBiJointT16
    ju.EXPECTED_PARAMS = EXPECTED_PARAMS
    cons.EXPECTED_PARAMS = EXPECTED_PARAMS

    args = cons.parse_args()

    if args.protocol not in ("xsub", "xset"):
        raise ValueError("GPU worker must run exactly one protocol: xsub or xset")

    install_clean_progress(args.protocol)

    print("=" * 120, flush=True)
    print(
        f"NESTSAR LOCALGLOBAL V2 + BI-JOINT M4/G4 | "
        f"{args.protocol.upper()} | SINGLE T4 WORKER",
        flush=True,
    )
    print("=" * 120, flush=True)
    print("JAX:", jax.__version__, flush=True)
    print("BACKEND:", jax.default_backend(), flush=True)
    print("LOCAL DEVICES:", jax.local_device_count(), flush=True)
    print("DEVICES:", jax.local_devices(), flush=True)
    print("EXPECTED PARAMS:", f"{EXPECTED_PARAMS:,}", flush=True)
    print("JOINT MEMORY: BIDIRECTIONAL", flush=True)
    print("ATTENTION: NONE", flush=True)
    print("TRAINING: RANDOM INITIALIZATION / FROM SCRATCH", flush=True)

    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, got backend={jax.default_backend()} "
            f"count={jax.local_device_count()}"
        )

    dataset = ju.base.find_dataset(args.dataset)
    print("DATASET:", dataset, flush=True)

    annotations, split = ju.base.load_ntu(dataset)
    best, epoch = cons.train_protocol(
        args,
        annotations,
        split,
        args.protocol,
    )

    outdir = Path(args.outdir)
    patch_saved_metadata(outdir, args.protocol)

    baseline = BASELINE_ACCURACY[args.protocol]
    result = {
        "protocol": args.protocol,
        "best_val_accuracy": best,
        "best_epoch": epoch,
        "baseline_localglobal_v2": baseline,
        "delta_vs_baseline_pp": 100.0 * (best - baseline),
        "expected_params": EXPECTED_PARAMS,
        "backend": "gpu",
        "visible_devices": [str(d) for d in jax.local_devices()],
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "seed": args.seed,
        "preprocessing": "local_pose_global_motion_v2",
        "spatial_joint_memory": "bidirectional_bimemory",
        "attention": False,
        "training_from_scratch": True,
    }

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"result_{args.protocol}.json").write_text(
        compact_json(result)
    )

    print("=" * 120, flush=True)
    print("GPU WORKER DONE", compact_json(result), flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()

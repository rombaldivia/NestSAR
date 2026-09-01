#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
from flax import serialization

from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_t16 import train_tpu as cons
from experiments.m4_phase_jitter_consistency_localglobal_t16 import train_tpu as lg_train

EXPECTED_PARAMS = 1_816_130


def compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _patch_protocol_metadata(outdir: Path, protocol: str) -> None:
    d = outdir / protocol
    msg = d / "best.msgpack"
    js = d / "best.json"
    if msg.is_file():
        payload = serialization.msgpack_restore(msg.read_bytes())
        if isinstance(payload, dict):
            payload["model"] = "M4PhaseJitterConsistencyLocalPoseGlobalMotionT16"
            rep = dict(payload.get("representation", {}))
            rep.update({"preprocessing": "local_pose_global_motion_v2", "pose_coordinate_frame": "exact_base_canonicalize_raw_framewise_root_centered", "motion_coordinate_frame": "constant_first_valid_person0_root", "architecture_change": False, "preprocessing_only_change": True, "runtime_backend": "single_visible_gpu"})
            payload["representation"] = rep
            msg.write_bytes(serialization.msgpack_serialize(payload))
    if js.is_file():
        meta = json.loads(js.read_text())
        meta.update({"model": "M4PhaseJitterConsistencyLocalPoseGlobalMotionT16", "preprocessing": "local_pose_global_motion_v2", "architecture_change": False, "preprocessing_only_change": True, "runtime_backend": "single_visible_gpu"})
        js.write_text(compact_json(meta))


def main() -> None:
    lg_train.install_preprocessing_override()
    ju.EXPECTED_PARAMS = EXPECTED_PARAMS
    cons.EXPECTED_PARAMS = EXPECTED_PARAMS
    args = cons.parse_args()
    if args.protocol not in ("xsub", "xset"):
        raise ValueError("GPU worker must run exactly one protocol: xsub or xset")
    print("=" * 120, flush=True)
    print(f"NESTSAR LOCALGLOBAL V2 | {args.protocol.upper()} | SINGLE T4 WORKER", flush=True)
    print("=" * 120, flush=True)
    print("JAX:", jax.__version__, flush=True)
    print("BACKEND:", jax.default_backend(), flush=True)
    print("LOCAL DEVICES:", jax.local_device_count(), flush=True)
    print("DEVICES:", jax.local_devices(), flush=True)
    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, got backend={jax.default_backend()} count={jax.local_device_count()}")
    dataset = ju.base.find_dataset(args.dataset)
    print("DATASET:", dataset, flush=True)
    annotations, split = ju.base.load_ntu(dataset)
    best, epoch = cons.train_protocol(args, annotations, split, args.protocol)
    outdir = Path(args.outdir)
    _patch_protocol_metadata(outdir, args.protocol)
    result = {"protocol": args.protocol, "best_val_accuracy": best, "best_epoch": epoch, "expected_params": EXPECTED_PARAMS, "backend": "gpu", "visible_devices": [str(d) for d in jax.local_devices()], "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size, "seed": args.seed, "preprocessing": "local_pose_global_motion_v2"}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"result_{args.protocol}.json").write_text(compact_json(result))
    print("=" * 120, flush=True)
    print("GPU WORKER DONE", compact_json(result), flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()

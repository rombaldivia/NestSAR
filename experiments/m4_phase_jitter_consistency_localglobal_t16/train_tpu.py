#!/usr/bin/env python3
from __future__ import annotations

"""Train champion T16 with ONLY LocalPose+GlobalMotion preprocessing changed."""

import json
import sys
from pathlib import Path

from flax import serialization

from experiments.m4_motionpreserve_phase_t16 import train_phase_t16_tpu as phase
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju
from experiments.m4_phase_jitter_consistency_t16 import train_tpu as cons
from experiments.m4_phase_jitter_consistency_localglobal_t16 import preprocessing as lg

EXPECTED_PARAMS = 1_816_130
DEFAULT_OUTDIR = "/kaggle/working/NestSAR_M4_Phase_JitterConsistency_LocalGlobal_T16_TPU"


def _arg_value(flag: str, default: str) -> str:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def install_preprocessing_override() -> None:
    # Canonical training/validation view in ju.build_protocol_views().
    phase.segment_phase_tokens = lg.segment_phase_tokens_localglobal

    # Jittered training view. ju.jitter_phase_tokens() resolves this module global
    # at call time, so replacing it preserves the exact jitter-boundary machinery
    # while changing only how pose/motion channels are derived inside each bound.
    ju.phase_tokens_from_bounds = lg.phase_tokens_from_bounds_localglobal


def _patch_checkpoint_metadata(outdir: Path) -> None:
    for protocol in ("xsub", "xset"):
        d = outdir / protocol
        msg = d / "best.msgpack"
        js = d / "best.json"

        if msg.is_file():
            payload = serialization.msgpack_restore(msg.read_bytes())
            if isinstance(payload, dict):
                payload["model"] = "M4PhaseJitterConsistencyLocalPoseGlobalMotionT16"
                rep = dict(payload.get("representation", {}))
                rep.update({
                    "preprocessing": "local_pose_global_motion",
                    "pose_coordinate_frame": "person0_root_centered_every_frame",
                    "motion_coordinate_frame": "constant_first_valid_person0_root",
                    "motion_channels": [
                        "full_displacement_xyz",
                        "first_half_displacement_xyz",
                        "second_half_displacement_xyz",
                        "absolute_path_xyz",
                    ],
                    "normalization": "single_rms_from_local_centered_pose_coordinates",
                    "architecture_change": False,
                    "preprocessing_only_change": True,
                })
                payload["representation"] = rep
                msg.write_bytes(serialization.msgpack_serialize(payload))

        if js.is_file():
            meta = json.loads(js.read_text())
            meta.update({
                "model": "M4PhaseJitterConsistencyLocalPoseGlobalMotionT16",
                "preprocessing": "local_pose_global_motion",
                "pose_coordinate_frame": "person0_root_centered_every_frame",
                "motion_coordinate_frame": "constant_first_valid_person0_root",
                "normalization": "single_rms_from_local_centered_pose_coordinates",
                "architecture_change": False,
                "preprocessing_only_change": True,
            })
            js.write_text(json.dumps(meta, indent=2))

    manifest = {
        "experiment": "M4PhaseJitterConsistencyLocalPoseGlobalMotionT16",
        "baseline": "M4PhaseJitterConsistencyT16",
        "baseline_accuracy": {
            "xsub": 0.748482884581394,
            "xset": 0.7557206987575029,
        },
        "architecture_change": False,
        "preprocessing_only_change": True,
        "expected_params": EXPECTED_PARAMS,
        "representation": {
            "frames": 16,
            "persons": 2,
            "joints": 25,
            "channels_per_person_joint": 15,
            "features_per_token": 750,
            "pose": "local frame-wise person0 root centered xyz",
            "motion": "differences in coordinates using one first-valid-root reference",
            "full_displacement": "global-motion coordinates",
            "phase_a": "global-motion coordinates",
            "phase_b": "global-motion coordinates",
            "absolute_path": "global-motion coordinates",
            "normalization": "baseline local-pose RMS",
        },
        "unchanged": [
            "Phase15 dimensionality",
            "16 temporal tokens",
            "J/B/JM/BM streams",
            "spatial encoders",
            "per-stream frame BiMemory",
            "original simple post-frame CrossStreamRouter",
            "descriptor/chunk memory",
            "uniform final fusion",
            "canonical+jitter dual-view training",
            "symmetric-KL consistency weight 0.08",
            "stream auxiliary CE weight 0.15",
            "EMA 0.995",
            "seed 128",
        ],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    install_preprocessing_override()

    # Model and optimizer remain the exact champion setup.
    ju.EXPECTED_PARAMS = EXPECTED_PARAMS
    cons.EXPECTED_PARAMS = EXPECTED_PARAMS

    outdir = Path(_arg_value("--outdir", DEFAULT_OUTDIR))

    print("=" * 120, flush=True)
    print("M4 PHASE+JITTER+CONSISTENCY — LOCAL POSE + GLOBAL MOTION", flush=True)
    print(f"EXPECTED_PARAMS={EXPECTED_PARAMS:,}", flush=True)
    print("ARCHITECTURE CHANGE: NONE", flush=True)
    print("PREPROCESSING CHANGE ONLY:", flush=True)
    print("  pose   = frame-wise root-centered coordinates", flush=True)
    print("  motion = first-valid-root global-reference coordinates", flush=True)
    print("=" * 120, flush=True)

    cons.main()
    _patch_checkpoint_metadata(outdir)


if __name__ == "__main__":
    main()

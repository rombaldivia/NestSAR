#!/usr/bin/env python3
from __future__ import annotations

"""Phase-aware M4-MotionPreserve-T16.

This experiment keeps the validated MotionPreserve architecture unchanged and
changes only the 16-token temporal representation.  Every contiguous segment
stores:

  pose xyz
  full signed displacement xyz
  first-half signed displacement xyz
  second-half signed displacement xyz
  accumulated absolute path motion xyz

The extra first/second-half channels preserve intra-segment phase/direction that
was identified as a residual bottleneck by the activation audit.
"""

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base

jax = base.jax
jnp = base.jnp
nn = base.nn
serialization = base.serialization

FRAMES = base.FRAMES
PERSONS = base.PERSONS
JOINTS = base.JOINTS
XYZ = base.XYZ
NUM_CLASSES = base.NUM_CLASSES
NUM_STREAMS = base.NUM_STREAMS

TOKEN_CHANNELS = 15
FEATURES = PERSONS * JOINTS * TOKEN_CHANNELS  # 750
EXPECTED_PARAMS = 1_817_930


def segment_phase_tokens(keypoints: np.ndarray) -> np.ndarray:
    """Compress the full sequence into 16 phase-aware temporal tokens."""
    x = base.canonicalize_raw(keypoints)
    total = x.shape[0]
    if total <= 0:
        return np.zeros((FRAMES, FEATURES), np.float32)

    tokens = np.zeros(
        (FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS), dtype=np.float32
    )

    for i, (s, e) in enumerate(base.segment_bounds(total, FRAMES)):
        seg = x[s:e]
        pose = seg[(len(seg) - 1) // 2]

        if len(seg) >= 2:
            d = seg[1:] - seg[:-1]
            full_disp = np.sum(d, axis=0)

            # Split TRANSITIONS, not frames, so phase_a + phase_b == full_disp.
            cut = max(1, len(d) // 2)
            phase_a = np.sum(d[:cut], axis=0)
            if cut < len(d):
                phase_b = np.sum(d[cut:], axis=0)
            else:
                phase_b = np.zeros_like(full_disp)

            path = np.sum(np.abs(d), axis=0)
        else:
            full_disp = np.zeros_like(pose)
            phase_a = np.zeros_like(pose)
            phase_b = np.zeros_like(pose)
            path = np.zeros_like(pose)

        tokens[i, ..., 0:3] = pose
        tokens[i, ..., 3:6] = full_disp
        tokens[i, ..., 6:9] = phase_a
        tokens[i, ..., 9:12] = phase_b
        tokens[i, ..., 12:15] = path

    # Preserve relative scale between pose/displacement/path exactly as in the
    # validated MotionPreserve preprocessing.
    nz = np.abs(x) > 1e-8
    if np.any(nz):
        rms = float(np.sqrt(np.mean(np.square(x[nz]))) + 1e-6)
        tokens = tokens / rms

    return np.nan_to_num(tokens).reshape(FRAMES, FEATURES).astype(np.float32)


def preprocess_keypoints(keypoints: np.ndarray, selector: str = "segment") -> np.ndarray:
    if selector == "segment":
        return segment_phase_tokens(keypoints)

    if selector == "uniform":
        x = base.canonicalize_raw(keypoints)
        idx = base.uniform_indices(x.shape[0], FRAMES)
        pose = x[idx]
        tokens = np.zeros(
            (FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS), dtype=np.float32
        )
        tokens[..., 0:3] = pose
        nz = np.abs(x) > 1e-8
        if np.any(nz):
            rms = float(np.sqrt(np.mean(np.square(x[nz]))) + 1e-6)
            tokens = tokens / rms
        return np.nan_to_num(tokens).reshape(FRAMES, FEATURES).astype(np.float32)

    raise ValueError(selector)


class M4MotionPreservePhaseT16(nn.Module):
    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Mapping[str, jnp.ndarray]:
        tok = x.reshape(
            x.shape[0], FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS
        )
        pose = tok[..., 0:3]
        full_disp = tok[..., 3:6]
        phase_a = tok[..., 6:9]
        phase_b = tok[..., 9:12]
        path = tok[..., 12:15]

        joint = pose
        parents = jnp.asarray(base.PARENTS)
        bone = joint - jnp.take(joint, parents, axis=3)

        # Phase-aware motion streams: 12 channels each.
        joint_motion = jnp.concatenate(
            [full_disp, phase_a, phase_b, path], axis=-1
        )

        parent_full = jnp.take(full_disp, parents, axis=3)
        parent_a = jnp.take(phase_a, parents, axis=3)
        parent_b = jnp.take(phase_b, parents, axis=3)
        parent_path = jnp.take(path, parents, axis=3)

        bone_full = full_disp - parent_full
        bone_a = phase_a - parent_a
        bone_b = phase_b - parent_b
        bone_path = jnp.abs(path - parent_path)
        bone_motion = jnp.concatenate(
            [bone_full, bone_a, bone_b, bone_path], axis=-1
        )

        raw_streams = (joint, bone, joint_motion, bone_motion)
        spatial = []
        for i, s in enumerate(raw_streams):
            spatial.append(
                base.SpatialEncoder(
                    self.spatial_dim,
                    self.model_dim,
                    self.dropout,
                    name=f"spatial_{i}",
                )(s, training)
            )

        # Keep the validated ordering: frame memory BEFORE cross-stream routing.
        frame_streams = []
        for i, s in enumerate(spatial):
            frame_streams.append(
                base.BiMemory(
                    self.model_dim,
                    name=f"frame_memory_{i}",
                )(s)
            )

        frame_stack = jnp.stack(frame_streams, axis=2)
        mixed, router_weights = base.CrossStreamRouter(
            self.model_dim,
            name="cross_stream_after_frame",
        )(frame_stack)

        descriptors = []
        stream_logits = []
        chunk_states = []
        for i in range(NUM_STREAMS):
            chunks, desc = base.DescriptorHead(
                self.model_dim,
                self.dropout,
                name=f"descriptor_{i}",
            )(mixed[:, :, i], training)
            descriptors.append(desc)
            chunk_states.append(chunks)
            stream_logits.append(
                nn.Dense(NUM_CLASSES, name=f"classifier_{i}")(desc)
            )

        descs = jnp.stack(descriptors, axis=1)
        sl = jnp.stack(stream_logits, axis=1)
        prior = self.param(
            "fusion_prior", nn.initializers.zeros, (NUM_STREAMS,)
        )
        controller = nn.Dense(
            NUM_STREAMS, name="fusion_controller"
        )(descs.reshape(descs.shape[0], -1))
        fusion = jax.nn.softmax(
            prior[None, :] + 0.15 * jnp.tanh(controller), axis=-1
        )
        logits = jnp.einsum("bs,bsc->bc", fusion, sl)

        return {
            "logits": logits,
            "stream_logits": sl,
            "fusion_weights": fusion,
            "router_weights": router_weights,
            "spatial_stack": jnp.stack(spatial, axis=2),
            "frame_stack": frame_stack,
            "mixed_frame_stack": mixed,
            "descriptors": descs,
            "chunk_states": jnp.stack(chunk_states, axis=1),
        }


def install_phase_overrides() -> None:
    """Patch only the representation/model globals used by the proven trainer."""
    base.TOKEN_CHANNELS = TOKEN_CHANNELS
    base.FEATURES = FEATURES
    base.preprocess_keypoints = preprocess_keypoints
    base.segment_motion_tokens = segment_phase_tokens
    base.M4MotionPreserveT16 = M4MotionPreservePhaseT16


def rewrite_checkpoint_metadata(outdir: str, protocol: str) -> None:
    ckpt = Path(outdir) / protocol / "best.msgpack"
    if not ckpt.is_file():
        return
    payload = serialization.msgpack_restore(ckpt.read_bytes())
    payload["model"] = "M4MotionPreservePhaseT16"
    payload["representation"] = {
        "frames": FRAMES,
        "token_channels": TOKEN_CHANNELS,
        "features_per_token": FEATURES,
        "channels": [
            "pose_xyz",
            "full_displacement_xyz",
            "first_half_displacement_xyz",
            "second_half_displacement_xyz",
            "path_motion_xyz",
        ],
    }
    ckpt.write_bytes(serialization.to_bytes(payload))


def main() -> None:
    install_phase_overrides()
    args = base.parse_args()

    base.log(
        f"JAX={jax.__version__} backend={jax.default_backend()} "
        f"devices={jax.devices()}"
    )
    if jax.default_backend() != "tpu":
        raise RuntimeError("This runner requires a TPU backend")
    if jax.local_device_count() != 8:
        raise RuntimeError(
            f"Expected 8 local TPU devices; got {jax.local_device_count()}"
        )

    base.log(
        f"Using 8 TPU cores with pmap | T16 PHASE tokens | "
        f"features/token={FEATURES} | no attention | no GCN"
    )
    base.log(
        "Representation: pose + full displacement + first-half displacement + "
        "second-half displacement + path motion"
    )
    base.log(
        "Architecture unchanged: spatial -> frame memory -> post-frame router -> "
        "chunk memory -> fusion"
    )

    dataset = base.find_dataset(args.dataset)
    base.log(f"Dataset={dataset}")
    anns, split = base.load_ntu(dataset)
    protocols = ["xsub", "xset"] if args.protocol == "both" else [args.protocol]

    summary = {}
    for pr in protocols:
        best, ep = base.train_protocol(args, anns, split, pr)
        rewrite_checkpoint_metadata(args.outdir, pr)
        summary[pr] = {
            "best_val_accuracy": best,
            "best_epoch": ep,
            "expected_params": EXPECTED_PARAMS,
            "token_channels": TOKEN_CHANNELS,
        }

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    base.log(f"DONE {summary}")


# Install at import time so preflight can use the same patched base utilities.
install_phase_overrides()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""PhasePath-T16 + training-only segment jitter + fixed uniform fusion.

This is a clean representation ablation on top of the successful Jitter+Uniform
experiment.  The network and training recipe are unchanged except that total
intra-segment path is split into first-half and second-half path channels.

Per segment / person / joint token:
  pose xyz
  full signed displacement xyz
  first-half signed displacement xyz
  second-half signed displacement xyz
  first-half absolute path xyz
  second-half absolute path xyz

Training uses one canonical and one +/-1-frame boundary-jittered precomputed
view, choosing between them per sample each epoch exactly as Jitter+Uniform.
Validation/inference use canonical segmentation only.  No consistency loss is
used in this branch.
"""

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization
from tqdm.auto import tqdm

from experiments.m4_motionpreserve_t16 import train_m4_motionpreserve_t16_tpu as base
from experiments.m4_phase_jitter_uniform_t16 import train_tpu as ju

FRAMES = ju.FRAMES
PERSONS = ju.PERSONS
JOINTS = ju.JOINTS
NUM_CLASSES = ju.NUM_CLASSES
NUM_STREAMS = ju.NUM_STREAMS

TOKEN_CHANNELS = 18
FEATURES = PERSONS * JOINTS * TOKEN_CHANNELS  # 900
EXPECTED_PARAMS = 1_816_274


def phasepath_tokens_from_bounds(keypoints: np.ndarray, bounds) -> np.ndarray:
    """Build 16 PhasePath tokens from explicit contiguous segment bounds."""
    x = base.canonicalize_raw(keypoints)
    if x.shape[0] <= 0:
        return np.zeros((FRAMES, FEATURES), np.float32)

    tokens = np.zeros(
        (FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS), dtype=np.float32
    )

    for i, (s, e) in enumerate(bounds):
        seg = x[s:e]
        pose = seg[(len(seg) - 1) // 2]

        if len(seg) >= 2:
            d = seg[1:] - seg[:-1]
            full_disp = np.sum(d, axis=0)

            # Split transitions so phase_a + phase_b == full_disp exactly
            # up to floating point arithmetic.
            cut = max(1, len(d) // 2)
            da = d[:cut]
            db = d[cut:]

            phase_a = np.sum(da, axis=0)
            phase_b = np.sum(db, axis=0) if len(db) else np.zeros_like(full_disp)

            path_a = np.sum(np.abs(da), axis=0)
            path_b = np.sum(np.abs(db), axis=0) if len(db) else np.zeros_like(full_disp)
        else:
            full_disp = np.zeros_like(pose)
            phase_a = np.zeros_like(pose)
            phase_b = np.zeros_like(pose)
            path_a = np.zeros_like(pose)
            path_b = np.zeros_like(pose)

        tokens[i, ..., 0:3] = pose
        tokens[i, ..., 3:6] = full_disp
        tokens[i, ..., 6:9] = phase_a
        tokens[i, ..., 9:12] = phase_b
        tokens[i, ..., 12:15] = path_a
        tokens[i, ..., 15:18] = path_b

    # Preserve the exact geometric normalization convention used by the
    # validated MotionPreserve / Phase models.
    nz = np.abs(x) > 1e-8
    if np.any(nz):
        rms = float(np.sqrt(np.mean(np.square(x[nz]))) + 1e-6)
        tokens /= rms

    return np.nan_to_num(tokens).reshape(FRAMES, FEATURES).astype(np.float32)


def segment_phasepath_tokens(keypoints: np.ndarray) -> np.ndarray:
    x = base.canonicalize_raw(keypoints)
    bounds = base.segment_bounds(x.shape[0], FRAMES)
    return phasepath_tokens_from_bounds(keypoints, bounds)


def jitter_phasepath_tokens(
    keypoints: np.ndarray,
    max_shift: int,
    rng: np.random.Generator,
) -> np.ndarray:
    x = base.canonicalize_raw(keypoints)
    bounds = ju.jittered_segment_bounds(x.shape[0], FRAMES, max_shift, rng)
    return phasepath_tokens_from_bounds(keypoints, bounds)


class M4PhasePathUniformT16(nn.Module):
    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> Mapping[str, jnp.ndarray]:
        tok = x.reshape(
            x.shape[0], FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS
        )

        pose = tok[..., 0:3]
        full_disp = tok[..., 3:6]
        phase_a = tok[..., 6:9]
        phase_b = tok[..., 9:12]
        path_a = tok[..., 12:15]
        path_b = tok[..., 15:18]

        joint = pose
        parents = jnp.asarray(base.PARENTS)
        bone = joint - jnp.take(joint, parents, axis=3)

        # Motion streams now preserve direction AND motion magnitude per half.
        joint_motion = jnp.concatenate(
            [full_disp, phase_a, phase_b, path_a, path_b], axis=-1
        )

        parent_full = jnp.take(full_disp, parents, axis=3)
        parent_a = jnp.take(phase_a, parents, axis=3)
        parent_b = jnp.take(phase_b, parents, axis=3)
        parent_path_a = jnp.take(path_a, parents, axis=3)
        parent_path_b = jnp.take(path_b, parents, axis=3)

        bone_motion = jnp.concatenate(
            [
                full_disp - parent_full,
                phase_a - parent_a,
                phase_b - parent_b,
                jnp.abs(path_a - parent_path_a),
                jnp.abs(path_b - parent_path_b),
            ],
            axis=-1,
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

        # Keep the audit-supported fixed uniform fusion.
        fusion = jnp.full(
            (x.shape[0], NUM_STREAMS),
            1.0 / NUM_STREAMS,
            dtype=sl.dtype,
        )
        logits = jnp.mean(sl, axis=1)

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


def build_protocol_views(
    annotations,
    split,
    protocol: str,
    max_shift: int,
    seed: int,
    max_train: int = 0,
    max_val: int = 0,
):
    by_id, train_ids, val_ids = ju.resolve_protocol_ids(
        annotations, split, protocol
    )

    if max_train:
        train_ids = train_ids[:max_train]
    if max_val:
        val_ids = val_ids[:max_val]

    Xcan = np.empty((len(train_ids), FRAMES, FEATURES), np.float32)
    Xjit = np.empty_like(Xcan)
    ytr = np.empty((len(train_ids),), np.int32)

    for i, sid in enumerate(
        tqdm(
            train_ids,
            desc=f"{protocol.upper()} PhasePath train canonical+jitter",
            mininterval=0.5,
        )
    ):
        a = by_id[sid]
        kp = base.annotation_keypoints(a)
        Xcan[i] = segment_phasepath_tokens(kp)
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, i, 9173])
        )
        Xjit[i] = jitter_phasepath_tokens(kp, max_shift, rng)
        ytr[i] = base.annotation_label(a)

    Xva = np.empty((len(val_ids), FRAMES, FEATURES), np.float32)
    yva = np.empty((len(val_ids),), np.int32)
    for i, sid in enumerate(
        tqdm(
            val_ids,
            desc=f"{protocol.upper()} PhasePath val canonical",
            mininterval=0.5,
        )
    ):
        a = by_id[sid]
        Xva[i] = segment_phasepath_tokens(base.annotation_keypoints(a))
        yva[i] = base.annotation_label(a)

    changed = float(
        np.mean(np.any(np.abs(Xcan - Xjit) > 1e-7, axis=(1, 2)))
    )
    ju.log(
        f"{protocol.upper()} jitter view differs from canonical for "
        f"{100 * changed:.2f}% of train samples"
    )
    return Xcan, Xjit, ytr, Xva, yva


def install_overrides() -> None:
    """Reuse the proven Jitter+Uniform trainer with PhasePath globals."""
    ju.TOKEN_CHANNELS = TOKEN_CHANNELS
    ju.FEATURES = FEATURES
    ju.EXPECTED_PARAMS = EXPECTED_PARAMS
    ju.M4PhaseUniformT16 = M4PhasePathUniformT16
    ju.build_protocol_views = build_protocol_views


def rewrite_checkpoint_metadata(outdir: str, protocol: str) -> None:
    ckpt = Path(outdir) / protocol / "best.msgpack"
    if not ckpt.is_file():
        return

    payload = serialization.msgpack_restore(ckpt.read_bytes())
    payload["model"] = "M4PhasePathJitterUniformT16"
    payload["representation"] = {
        "frames": FRAMES,
        "token_channels": TOKEN_CHANNELS,
        "features_per_token": FEATURES,
        "channels": [
            "pose_xyz",
            "full_displacement_xyz",
            "first_half_displacement_xyz",
            "second_half_displacement_xyz",
            "first_half_path_xyz",
            "second_half_path_xyz",
        ],
        "jitter_max_shift": payload.get("config", {}).get(
            "jitter_max_shift", 1
        ),
        "jitter_prob": payload.get("config", {}).get(
            "jitter_prob", 0.5
        ),
        "final_fusion": "uniform_mean",
        "consistency": None,
    }
    ckpt.write_bytes(serialization.to_bytes(payload))


def main() -> None:
    install_overrides()
    args = ju.parse_args()

    ju.log(
        f"JAX={jax.__version__} backend={jax.default_backend()} "
        f"devices={jax.devices()}"
    )
    if jax.default_backend() != "tpu" or jax.local_device_count() != 8:
        raise RuntimeError(
            f"Expected one TPU v5e-8; backend={jax.default_backend()} "
            f"local_devices={jax.local_device_count()}"
        )
    if not (0.0 <= args.jitter_prob <= 1.0):
        raise ValueError("--jitter-prob must be in [0,1]")

    ju.log(
        "Experiment: PhasePath-T16 + training-only segment jitter + "
        "fixed uniform fusion"
    )
    ju.log(
        f"T16 | features/token={FEATURES} | token_channels={TOKEN_CHANNELS} | "
        f"jitter=+/-{args.jitter_max_shift} raw frame | "
        f"jitter_prob={args.jitter_prob:.2f}"
    )
    ju.log(
        "Representation: pose + full disp + phase-A/B disp + "
        "phase-A/B path"
    )
    ju.log(
        "Inference: canonical segmentation only; no TTA; no consistency loss"
    )

    dataset = base.find_dataset(args.dataset)
    ju.log(f"Dataset={dataset}")
    anns, split = base.load_ntu(dataset)
    protocols = (
        ["xsub", "xset"] if args.protocol == "both" else [args.protocol]
    )

    summary = {}
    for pr in protocols:
        best, ep = ju.train_protocol(args, anns, split, pr)
        rewrite_checkpoint_metadata(args.outdir, pr)
        summary[pr] = {
            "best_val_accuracy": best,
            "best_epoch": ep,
            "expected_params": EXPECTED_PARAMS,
            "token_channels": TOKEN_CHANNELS,
            "features_per_token": FEATURES,
        }

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    ju.log(f"DONE {summary}")


install_overrides()


if __name__ == "__main__":
    main()

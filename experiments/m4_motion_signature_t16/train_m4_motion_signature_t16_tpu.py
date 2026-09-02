#!/usr/bin/env python3
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
from flax import linen as nn

from experiments.m4_motionlite_t16 import train_m4_motionlite_t16_tpu as base

# Semantic distance pairs chosen to capture hand/head, hand/hand,
# hand/root and foot/root dynamics without building a dense VxV graph.
IMPORTANT_PAIRS = (
    (7, 3),
    (11, 3),
    (7, 11),
    (7, 0),
    (11, 0),
    (15, 0),
    (19, 0),
    (15, 19),
)


def part_motion_energy(velocity: jnp.ndarray) -> jnp.ndarray:
    """Average relative motion magnitude for each of the 10 anatomical parts.

    Args:
        velocity: [B, T, M, V, 3]

    Returns:
        [B, T, M, 10]
    """
    mask = jnp.asarray(base.PART_MASK_NP, velocity.dtype)
    counts = jnp.asarray(base.PART_COUNTS_NP, velocity.dtype)
    speed = jnp.sqrt(jnp.sum(jnp.square(velocity), axis=-1) + 1e-8)
    energy = jnp.einsum("btmv,pv->btmp", speed, mask)
    return energy / counts[None, None, None, :]


def distance_dynamics(joint: jnp.ndarray) -> jnp.ndarray:
    """Temporal derivative of selected semantic inter-joint distances.

    Args:
        joint: root-relative skeleton [B, T, M, 25, 3]

    Returns:
        [B, T, 8]
    """
    values = []
    for a, b in IMPORTANT_PAIRS:
        diff = joint[:, :, :, a, :] - joint[:, :, :, b, :]
        dist = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1) + 1e-8)
        dist = jnp.mean(dist, axis=2)  # average over persons -> [B,T]
        dd = base.lag_diff(dist[..., None], 1)[..., 0]
        values.append(dd)
    return jnp.stack(values, axis=-1)


def build_motion_signature(joint: jnp.ndarray) -> jnp.ndarray:
    """Build a compact 21-D analytical motion descriptor per frame.

    Components:
      - 10 body-part relative-motion energies
      - 8 semantic distance dynamics
      - mean relative joint speed
      - max relative joint speed
      - mean acceleration magnitude
    """
    velocity = base.lag_diff(joint, 1)
    acceleration = base.lag_diff(velocity, 1)

    # Root/torso-relative velocity suppresses global translation and encourages
    # subject/setup invariance while retaining local action dynamics.
    torso_velocity = velocity[:, :, :, 0:1, :]
    relative_velocity = velocity - torso_velocity

    energy = part_motion_energy(relative_velocity)
    energy = jnp.mean(energy, axis=2)  # [B,T,10]

    speed = jnp.sqrt(jnp.sum(jnp.square(relative_velocity), axis=-1) + 1e-8)
    accel_mag = jnp.sqrt(jnp.sum(jnp.square(acceleration), axis=-1) + 1e-8)

    mean_speed = jnp.mean(speed, axis=(2, 3))[..., None]
    max_speed = jnp.max(speed, axis=(2, 3))[..., None]
    mean_accel = jnp.mean(accel_mag, axis=(2, 3))[..., None]

    dist_motion = distance_dynamics(joint)

    return jnp.concatenate(
        [energy, dist_motion, mean_speed, max_speed, mean_accel],
        axis=-1,
    )


class M4MotionSignatureT16(nn.Module):
    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False):
        sk = x.reshape(x.shape[0], base.FRAMES, 2, 25, 3)
        root = sk[:, :, :, 0:1, :]
        joint = sk - root

        parents = jnp.asarray(base.PARENTS)
        bone = joint - jnp.take(joint, parents, axis=3)

        # Existing MotionLite multi-scale motion streams.
        j_d1 = base.lag_diff(joint, 1)
        j_d2 = base.lag_diff(joint, 2) / 2.0
        j_d4 = base.lag_diff(joint, 4) / 4.0
        j_acc = base.lag_diff(j_d1, 1)
        joint_motion = jnp.concatenate([j_d1, j_d2, j_d4, j_acc], axis=-1)

        b_d1 = base.lag_diff(bone, 1)
        b_d2 = base.lag_diff(bone, 2) / 2.0
        b_d4 = base.lag_diff(bone, 4) / 4.0
        b_acc = base.lag_diff(b_d1, 1)
        bone_motion = jnp.concatenate([b_d1, b_d2, b_d4, b_acc], axis=-1)

        raw_streams = (joint, bone, joint_motion, bone_motion)
        encoded = []
        for i, stream in enumerate(raw_streams):
            encoded.append(
                base.SpatialEncoder(
                    self.spatial_dim,
                    self.model_dim,
                    self.dropout,
                    name=f"spatial_{i}",
                )(stream, training)
            )
        streams = jnp.stack(encoded, axis=2)  # [B,T,4,D]

        # 21-D analytical descriptor -> tiny 16-D token -> model-dim residual.
        signature = build_motion_signature(joint)
        motion_token = nn.Dense(16, name="motion_signature_proj")(signature)
        motion_token = nn.gelu(motion_token)
        motion_token = nn.LayerNorm(name="motion_signature_norm")(motion_token)
        motion_residual = nn.Dense(self.model_dim, name="motion_signature_fuse")(
            motion_token
        )

        # Conservative initialization: sigmoid(-2) ~= 0.119. The model can
        # learn to increase or suppress the analytical motion contribution.
        motion_gate_logit = self.param(
            "motion_signature_gate",
            nn.initializers.constant(-2.0),
            (1,),
        )
        motion_gate = jax.nn.sigmoid(motion_gate_logit)
        streams = streams + motion_gate * motion_residual[:, :, None, :]

        streams, router_weights = base.CrossStreamRouter(
            self.model_dim,
            name="cross_stream",
        )(streams)

        stream_logits = []
        descriptors = []
        for i in range(base.NUM_STREAMS):
            _, desc = base.TemporalHierarchy(
                self.model_dim,
                self.dropout,
                name=f"temporal_{i}",
            )(streams[:, :, i], training)
            descriptors.append(desc)
            stream_logits.append(
                nn.Dense(base.NUM_CLASSES, name=f"classifier_{i}")(desc)
            )

        descs = jnp.stack(descriptors, axis=1)
        sl = jnp.stack(stream_logits, axis=1)

        prior = self.param(
            "fusion_prior",
            nn.initializers.zeros,
            (base.NUM_STREAMS,),
        )
        controller = nn.Dense(base.NUM_STREAMS, name="fusion_controller")(
            descs.reshape(descs.shape[0], -1)
        )
        fusion = jax.nn.softmax(
            prior[None, :] + 0.15 * jnp.tanh(controller),
            axis=-1,
        )
        logits = jnp.einsum("bs,bsc->bc", fusion, sl)

        return {
            "logits": logits,
            "stream_logits": sl,
            "fusion_weights": fusion,
            "router_weights": router_weights,
            "motion_gate": motion_gate,
        }


def main() -> None:
    # Reuse the audited MotionLite TPU training/eval pipeline while replacing
    # only the model class. This keeps optimizer, EMA, splits and FLOP audit
    # directly comparable to the existing experiment.
    base.M4MotionLiteT16 = M4MotionSignatureT16

    # Patience is intentionally reduced from the base default (12) to 5.
    # An explicit user-provided --patience still wins.
    if "--patience" not in sys.argv:
        sys.argv.extend(["--patience", "5"])

    # Give this experiment its own output directory unless explicitly set.
    if "--outdir" not in sys.argv:
        sys.argv.extend(
            [
                "--outdir",
                "/kaggle/working/NestSAR_M4_MotionSignature_T16_TPU",
            ]
        )

    base.log(
        "MotionSignature: 21-D/frame = 10 part energies + 8 distance dynamics "
        "+ mean/max relative speed + mean acceleration | gated residual fusion"
    )
    base.log("Early-stopping default patience=5")
    base.main()


if __name__ == "__main__":
    main()

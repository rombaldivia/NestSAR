#!/usr/bin/env python3
from __future__ import annotations

"""T64 teacher matched to the current T16 Phase+Jitter+Consistency champion.

Same learnable architecture/parameter shapes as T16, but 64 temporal phase tokens
and 16 chunks of 4 tokens each.  Training keeps the proven dual canonical/jitter
consistency objective.  Validation/inference uses canonical segmentation only.

Host-memory control: canonical/jitter/validation token caches are stored as float16
and cast back to float32 batch-wise before TPU transfer.  This keeps one protocol's
cache around ~16-18 GB rather than ~32-36 GB.
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
from experiments.m4_phase_jitter_consistency_t16 import train_tpu as cons

FRAMES = 64
PERSONS = 2
JOINTS = 25
TOKEN_CHANNELS = 15
FEATURES = PERSONS * JOINTS * TOKEN_CHANNELS  # 750
NUM_CLASSES = 120
NUM_STREAMS = 4
CHUNK_SIZE = 4
CHUNKS = FRAMES // CHUNK_SIZE  # 16
EXPECTED_PARAMS = 1_816_130
CACHE_DTYPE = np.float16


def phase_tokens_from_bounds(keypoints: np.ndarray, bounds) -> np.ndarray:
    x = base.canonicalize_raw(keypoints)
    if x.shape[0] <= 0:
        return np.zeros((FRAMES, FEATURES), np.float32)

    tokens = np.zeros((FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS), np.float32)
    for i, (s, e) in enumerate(bounds):
        seg = x[s:e]
        pose = seg[(len(seg) - 1) // 2]
        if len(seg) >= 2:
            d = seg[1:] - seg[:-1]
            full_disp = np.sum(d, axis=0)
            cut = max(1, len(d) // 2)
            phase_a = np.sum(d[:cut], axis=0)
            phase_b = np.sum(d[cut:], axis=0) if cut < len(d) else np.zeros_like(full_disp)
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

    nz = np.abs(x) > 1e-8
    if np.any(nz):
        rms = float(np.sqrt(np.mean(np.square(x[nz]))) + 1e-6)
        tokens /= rms
    return np.nan_to_num(tokens).reshape(FRAMES, FEATURES).astype(np.float32)


def segment_phase_tokens64(keypoints: np.ndarray) -> np.ndarray:
    x = base.canonicalize_raw(keypoints)
    bounds = base.segment_bounds(x.shape[0], FRAMES)
    return phase_tokens_from_bounds(keypoints, bounds)


def jitter_phase_tokens64(keypoints: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    x = base.canonicalize_raw(keypoints)
    bounds = ju.jittered_segment_bounds(x.shape[0], FRAMES, max_shift, rng)
    return phase_tokens_from_bounds(keypoints, bounds)


def build_protocol_views64(
    annotations,
    split,
    protocol: str,
    max_shift: int,
    seed: int,
    max_train: int = 0,
    max_val: int = 0,
):
    by_id, train_ids, val_ids = ju.resolve_protocol_ids(annotations, split, protocol)
    if max_train:
        train_ids = train_ids[:max_train]
    if max_val:
        val_ids = val_ids[:max_val]

    Xcan = np.empty((len(train_ids), FRAMES, FEATURES), CACHE_DTYPE)
    Xjit = np.empty_like(Xcan)
    ytr = np.empty((len(train_ids),), np.int32)

    for i, sid in enumerate(tqdm(
        train_ids,
        desc=f"{protocol.upper()} T64 train canonical+jitter",
        mininterval=0.5,
    )):
        a = by_id[sid]
        kp = base.annotation_keypoints(a)
        Xcan[i] = segment_phase_tokens64(kp).astype(CACHE_DTYPE)
        rng = np.random.default_rng(np.random.SeedSequence([seed, i, 64017]))
        Xjit[i] = jitter_phase_tokens64(kp, max_shift, rng).astype(CACHE_DTYPE)
        ytr[i] = base.annotation_label(a)

    Xva = np.empty((len(val_ids), FRAMES, FEATURES), CACHE_DTYPE)
    yva = np.empty((len(val_ids),), np.int32)
    for i, sid in enumerate(tqdm(
        val_ids,
        desc=f"{protocol.upper()} T64 val canonical",
        mininterval=0.5,
    )):
        a = by_id[sid]
        Xva[i] = segment_phase_tokens64(base.annotation_keypoints(a)).astype(CACHE_DTYPE)
        yva[i] = base.annotation_label(a)

    changed = float(np.mean(np.any(np.abs(Xcan.astype(np.float32) - Xjit.astype(np.float32)) > 1e-5, axis=(1, 2))))
    gib = (Xcan.nbytes + Xjit.nbytes + Xva.nbytes) / (1024 ** 3)
    cons.log(
        f"{protocol.upper()} T64 jitter differs for {100*changed:.2f}% | "
        f"cached_host={gib:.2f} GiB dtype=float16"
    )
    return Xcan, Xjit, ytr, Xva, yva


def iter_train_pairs64(Xcan, Xjit, y, global_batch: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    usable = (len(idx) // global_batch) * global_batch
    idx = idx[:usable]
    for s in range(0, usable, global_batch):
        ii = idx[s:s + global_batch]
        yield (
            np.asarray(Xcan[ii], dtype=np.float32),
            np.asarray(Xjit[ii], dtype=np.float32),
            y[ii],
        )


def iter_eval64(X, y, global_batch: int):
    n = len(y)
    for s in range(0, n, global_batch):
        e = min(s + global_batch, n)
        k = e - s
        xb = np.zeros((global_batch, FRAMES, FEATURES), np.float32)
        yb = np.zeros((global_batch,), np.int32)
        mask = np.zeros((global_batch,), np.float32)
        xb[:k] = np.asarray(X[s:e], dtype=np.float32)
        yb[:k] = y[s:e]
        mask[:k] = 1.0
        yield xb, yb, mask


class DescriptorHeadT64(nn.Module):
    dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, frame_h: jnp.ndarray, training: bool):
        chunks = frame_h.reshape(frame_h.shape[0], CHUNKS, CHUNK_SIZE, self.dim).mean(axis=2)
        chunks = base.BiMemory(self.dim, name="chunk_memory")(chunks)
        pooled = jnp.concatenate([frame_h.mean(axis=1), chunks.mean(axis=1)], axis=-1)
        pooled = nn.Dense(self.dim, name="hier_fuse")(pooled)
        pooled = nn.LayerNorm(name="hier_norm")(nn.gelu(pooled))
        pooled = nn.Dropout(self.dropout)(pooled, deterministic=not training)
        return chunks, pooled


class M4PhaseUniformT64Teacher(nn.Module):
    spatial_dim: int = 24
    model_dim: int = 112
    dropout: float = 0.10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Mapping[str, jnp.ndarray]:
        tok = x.reshape(x.shape[0], FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS)
        pose = tok[..., 0:3]
        full_disp = tok[..., 3:6]
        phase_a = tok[..., 6:9]
        phase_b = tok[..., 9:12]
        path = tok[..., 12:15]

        joint = pose
        parents = jnp.asarray(base.PARENTS)
        bone = joint - jnp.take(joint, parents, axis=3)

        joint_motion = jnp.concatenate([full_disp, phase_a, phase_b, path], axis=-1)
        parent_full = jnp.take(full_disp, parents, axis=3)
        parent_a = jnp.take(phase_a, parents, axis=3)
        parent_b = jnp.take(phase_b, parents, axis=3)
        parent_path = jnp.take(path, parents, axis=3)
        bone_motion = jnp.concatenate([
            full_disp - parent_full,
            phase_a - parent_a,
            phase_b - parent_b,
            jnp.abs(path - parent_path),
        ], axis=-1)

        raw_streams = (joint, bone, joint_motion, bone_motion)
        spatial = []
        for i, s in enumerate(raw_streams):
            spatial.append(base.SpatialEncoder(
                self.spatial_dim,
                self.model_dim,
                self.dropout,
                name=f"spatial_{i}",
            )(s, training))

        frame_streams = []
        for i, s in enumerate(spatial):
            frame_streams.append(base.BiMemory(
                self.model_dim,
                name=f"frame_memory_{i}",
            )(s))
        frame_stack = jnp.stack(frame_streams, axis=2)

        mixed, router_weights = base.CrossStreamRouter(
            self.model_dim,
            name="cross_stream_after_frame",
        )(frame_stack)

        descriptors = []
        stream_logits = []
        chunk_states = []
        for i in range(NUM_STREAMS):
            chunks, desc = DescriptorHeadT64(
                self.model_dim,
                self.dropout,
                name=f"descriptor_{i}",
            )(mixed[:, :, i], training)
            descriptors.append(desc)
            chunk_states.append(chunks)
            stream_logits.append(nn.Dense(NUM_CLASSES, name=f"classifier_{i}")(desc))

        descs = jnp.stack(descriptors, axis=1)
        sl = jnp.stack(stream_logits, axis=1)
        fusion = jnp.full((x.shape[0], NUM_STREAMS), 1.0 / NUM_STREAMS, dtype=sl.dtype)
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


def install_overrides() -> None:
    # Uniform infrastructure used dynamically by consistency trainer.
    ju.FRAMES = FRAMES
    ju.TOKEN_CHANNELS = TOKEN_CHANNELS
    ju.FEATURES = FEATURES
    ju.EXPECTED_PARAMS = EXPECTED_PARAMS
    ju.M4PhaseUniformT16 = M4PhaseUniformT64Teacher
    ju.build_protocol_views = build_protocol_views64
    ju.iter_eval = iter_eval64

    # Consistency trainer local globals.
    cons.FRAMES = FRAMES
    cons.FEATURES = FEATURES
    cons.EXPECTED_PARAMS = EXPECTED_PARAMS
    cons.iter_train_pairs = iter_train_pairs64


def rewrite_checkpoint_metadata(outdir: str, protocol: str) -> None:
    ckpt = Path(outdir) / protocol / "best.msgpack"
    if not ckpt.is_file():
        return
    payload = serialization.msgpack_restore(ckpt.read_bytes())
    payload["model"] = "M4PhaseJitterConsistencyT64Teacher"
    payload["teacher_role"] = "T64 feature/logit teacher candidate for T16 distillation"
    payload["representation"] = {
        "frames": FRAMES,
        "token_channels": TOKEN_CHANNELS,
        "features_per_token": FEATURES,
        "channels": [
            "pose_xyz",
            "full_displacement_xyz",
            "first_half_displacement_xyz",
            "second_half_displacement_xyz",
            "total_path_xyz",
        ],
        "chunk_size": CHUNK_SIZE,
        "chunks": CHUNKS,
        "jitter_max_shift": payload.get("config", {}).get("jitter_max_shift", 1),
        "final_fusion": "uniform_mean",
        "consistency": "symmetric_kl",
        "consistency_weight": payload.get("config", {}).get("consistency_weight", 0.08),
        "cache_storage_dtype": "float16_precompute_float32_model_input",
    }
    ckpt.write_bytes(serialization.to_bytes(payload))


def main() -> None:
    install_overrides()
    args = cons.parse_args()

    cons.log(f"JAX={jax.__version__} backend={jax.default_backend()} devices={jax.devices()}")
    if jax.default_backend() != "tpu" or jax.local_device_count() != 8:
        raise RuntimeError(
            f"Expected one TPU v5e-8; backend={jax.default_backend()} "
            f"local_devices={jax.local_device_count()}"
        )
    if args.consistency_weight < 0.0:
        raise ValueError("--consistency-weight must be >= 0")
    if args.consistency_temperature <= 0.0:
        raise ValueError("--consistency-temperature must be > 0")

    cons.log("Experiment: matched T64 teacher | Phase + Jitter + Consistency + Uniform Fusion")
    cons.log(
        f"T64 | chunks={CHUNKS}x{CHUNK_SIZE} | features/token={FEATURES} | "
        f"jitter=+/-{args.jitter_max_shift} raw frame | "
        f"symKL_weight={args.consistency_weight:.3f}"
    )
    cons.log("Same parameter shapes as T16 champion; extra cost is temporal compute only")

    dataset = ju.base.find_dataset(args.dataset)
    cons.log(f"Dataset={dataset}")
    anns, split = ju.base.load_ntu(dataset)
    protocols = ["xsub", "xset"] if args.protocol == "both" else [args.protocol]

    summary = {}
    for pr in protocols:
        best, ep = cons.train_protocol(args, anns, split, pr)
        rewrite_checkpoint_metadata(args.outdir, pr)
        summary[pr] = {
            "best_val_accuracy": best,
            "best_epoch": ep,
            "expected_params": EXPECTED_PARAMS,
            "frames": FRAMES,
            "chunks": CHUNKS,
            "features_per_token": FEATURES,
        }

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "summary.json").write_text(json.dumps(summary, indent=2))
    cons.log(f"DONE {summary}")


install_overrides()

if __name__ == "__main__":
    main()

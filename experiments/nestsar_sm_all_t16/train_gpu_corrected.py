#!/usr/bin/env python3
from __future__ import annotations

"""Corrected data-path wrapper for NestSAR-SM-ALL-T16.

The SM-ALL model itself is imported unchanged from train_gpu/model.py. This
wrapper replaces only the preprocessing/materialization functions used by the
existing trainer:
  * missing people/joints remain exactly zero after centering,
  * every valid adjacent raw-frame transition is counted once,
  * the augmented view is rebuilt fresh every epoch from the raw skeleton.

The neural input remains exactly [16, 750], so the M4/G4/self-modifying model
parameterization and inference graph are not changed by this file.
"""

import numpy as np

from experiments.nestsar_sm_all_t16 import preprocessing_corrected as pp
from experiments.nestsar_sm_all_t16 import train_gpu as legacy

_CTX = {}
ROTATION_DEGREES = 8.0


def _raw_from_annotation(annotation):
    # NTU120 MMAction-style pickle used by this project is M,T,V,C.
    return pp.ordered_raw(legacy.base.annotation_keypoints(annotation), layout="MTVC")


def build_protocol_arrays(args, annotations, split, protocol: str):
    """Materialize corrected canonical train/val views only.

    The second training view is intentionally NOT materialized here. It is
    rebuilt from the raw skeleton inside iter_train_pairs for every epoch.
    """
    by_id, train_ids, val_ids = legacy.ju.resolve_protocol_ids(
        annotations, split, protocol
    )
    if args.max_train_samples:
        train_ids = train_ids[: args.max_train_samples]
    if args.max_val_samples:
        val_ids = val_ids[: args.max_val_samples]
    if not train_ids or not val_ids:
        raise RuntimeError(
            f"Resolved empty {protocol} arrays: train={len(train_ids)} val={len(val_ids)}"
        )

    Xcan = np.empty((len(train_ids), legacy.FRAMES, legacy.FEATURES), np.float32)
    ytr = np.empty((len(train_ids),), np.int32)
    Xva = np.empty((len(val_ids), legacy.FRAMES, legacy.FEATURES), np.float32)
    yva = np.empty((len(val_ids),), np.int32)

    legacy.emit_status(protocol, 0, args.epochs, "PREP_TRAIN", 0, len(train_ids))
    for i, sid in enumerate(train_ids):
        annotation = by_id[sid]
        raw = _raw_from_annotation(annotation)
        Xcan[i] = pp.features(raw)
        ytr[i] = legacy.base.annotation_label(annotation)
        done = i + 1
        if done % args.preprocess_progress_every == 0 or done == len(train_ids):
            legacy.emit_status(protocol, 0, args.epochs, "PREP_TRAIN", done, len(train_ids))

    legacy.emit_status(protocol, 0, args.epochs, "PREP_VAL", 0, len(val_ids))
    for i, sid in enumerate(val_ids):
        annotation = by_id[sid]
        raw = _raw_from_annotation(annotation)
        Xva[i] = pp.features(raw)
        yva[i] = legacy.base.annotation_label(annotation)
        done = i + 1
        if done % args.preprocess_progress_every == 0 or done == len(val_ids):
            legacy.emit_status(protocol, 0, args.epochs, "PREP_VAL", done, len(val_ids))

    protocol_seed = args.seed + (0 if protocol == "xsub" else 100000)
    _CTX.clear()
    _CTX.update(
        args=args,
        protocol=protocol,
        protocol_seed=protocol_seed,
        by_id=by_id,
        train_ids=list(train_ids),
    )

    # Placeholder retained only to satisfy the unchanged legacy trainer API.
    # iter_train_pairs ignores it and constructs a fresh augmented batch.
    Xfresh_placeholder = np.empty((0,), np.float32)

    print(
        "PREPROCESS_CORRECTED|"
        f"protocol={protocol}|version={pp.VERSION}|train={len(ytr)}|val={len(yva)}|"
        f"processing_frames={legacy.FRAMES}|features={legacy.FEATURES}|"
        f"fresh_aug=1|rotation_deg={ROTATION_DEGREES}|jitter={args.jitter_max_shift}",
        flush=True,
    )
    return Xcan, Xfresh_placeholder, ytr, Xva, yva


def iter_train_pairs(Xcan, _unused, y, batch_size: int, seed: int):
    """Yield corrected canonical + freshly rebuilt augmented batches.

    legacy.train_protocol calls this with seed=args.seed+epoch, which lets us
    recover the epoch deterministically without changing the trainer/model.
    """
    if not _CTX:
        raise RuntimeError("Corrected preprocessing context was not initialized")
    args = _CTX["args"]
    epoch = int(seed - args.seed)
    if epoch < 1:
        raise RuntimeError(f"Invalid recovered epoch {epoch} from seed {seed}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    usable = (len(idx) // batch_size) * batch_size
    idx = idx[:usable]

    by_id = _CTX["by_id"]
    train_ids = _CTX["train_ids"]
    protocol_seed = _CTX["protocol_seed"]

    for start in range(0, usable, batch_size):
        ii = idx[start : start + batch_size]
        xaug = np.empty((len(ii), legacy.FRAMES, legacy.FEATURES), np.float32)
        for j, sample_pos in enumerate(ii):
            sid = train_ids[int(sample_pos)]
            raw = _raw_from_annotation(by_id[sid])
            _, xaug[j] = pp.training_views(
                raw,
                seed=protocol_seed,
                epoch=epoch,
                sample_index=int(sample_pos),
                rotation_degrees=ROTATION_DEGREES,
                shift=args.jitter_max_shift,
            )
        yield Xcan[ii], xaug, y[ii]


def install_patch():
    # The unchanged legacy train_protocol resolves these names from its module
    # globals at runtime, so replacing only these two functions is sufficient.
    legacy.build_protocol_arrays = build_protocol_arrays
    legacy.iter_train_pairs = iter_train_pairs


def main():
    install_patch()
    args = legacy.parse_args()

    print("=" * 118, flush=True)
    print(
        f"NESTSAR-SM-ALL-T16 v1 + CORRECTED PREPROCESSING | {args.protocol.upper()} | SINGLE T4",
        flush=True,
    )
    print(
        "MODEL UNCHANGED | MASK-SAFE CENTERING | COMPLETE RAW TRANSITIONS | FRESH EPOCH AUGMENTATION",
        flush=True,
    )
    print(
        f"INPUT=[16,{legacy.FEATURES}] ROT=+/-{ROTATION_DEGREES:.1f}deg JITTER=+/-{args.jitter_max_shift} "
        f"FAST_RANK={args.fast_rank} HEAD_RANK={args.head_rank}",
        flush=True,
    )
    print("JAX", legacy.jax.__version__, "BACKEND", legacy.jax.default_backend(), flush=True)
    print("DEVICES", legacy.jax.local_devices(), flush=True)
    print("=" * 118, flush=True)

    dataset = legacy.base.find_dataset(args.dataset)
    annotations, split = legacy.base.load_ntu(dataset)
    legacy.train_protocol(args, annotations, split, args.protocol)


if __name__ == "__main__":
    main()

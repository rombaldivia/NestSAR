#!/usr/bin/env python3
from __future__ import annotations

"""CME T32 pipeline using the verified LocalGlobal + Hand-M4/G4-Lite T32 champion.

The base champion is frozen.  Cache generation reconstructs the exact deployed
Hand-M4/G4 model:

  main path: LocalGlobal T16
  hand path: Hand-M4/G4-Lite T32

Only CME is trainable.  No attention / no QKV / no GCN.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from experiments.m4_confusion_memory_expert_t32 import pipeline as cme
from experiments.m4_phase_jitter_consistency_localglobal_t16 import preprocessing as lg
from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.model import (
    M4LocalGlobalHandM4G4T32,
)
from experiments.m4_phase_jitter_consistency_localglobal_hand_m4g4_t32.preprocessing import (
    hand_tokens_t32,
)


BASE_KIND = "hand"


def _parse_args():
    # The original CME parser predates Hand support and accepts only
    # localglobal/bijoint.  Accept --base-kind hand from the launcher by
    # temporarily presenting the old parser with a legal placeholder, then
    # restore the real kind immediately after parsing.
    argv = list(sys.argv[1:])
    if "--base-kind" in argv:
        i = argv.index("--base-kind")
        if i + 1 < len(argv):
            argv[i + 1] = "localglobal"
    args = cme.make_parser().parse_args(argv)
    args.base_kind = BASE_KIND
    return args


def load_hand_base(ckpt: Path):
    payload = serialization.msgpack_restore(ckpt.read_bytes())
    if not isinstance(payload, dict):
        raise RuntimeError("Hand base checkpoint is not a dict payload")

    params = payload.get("ema_params", payload.get("params"))
    if params is None:
        raise RuntimeError("Hand base checkpoint has no ema_params/params")

    cfg = payload.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}

    spatial_dim = int(cfg.get("spatial_dim", 24))
    model_dim = int(cfg.get("model_dim", 112))
    dropout = float(cfg.get("dropout", 0.10))
    hand_dim = int(cfg.get("hand_dim", 32))
    hand_scale = float(cfg.get("hand_residual_scale", 0.10))

    if hand_dim != 32:
        raise RuntimeError(
            f"Expected audited Hand-M4/G4 D32 checkpoint, got hand_dim={hand_dim}"
        )

    model = M4LocalGlobalHandM4G4T32(
        spatial_dim=spatial_dim,
        model_dim=model_dim,
        dropout=dropout,
        hand_dim=hand_dim,
        hand_residual_scale=hand_scale,
    )
    return model, params, payload


def build_cache_hand(args) -> None:
    root = Path(args.cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.base_ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    if cme.cache_complete(root, sha, args.protocol, BASE_KIND):
        cme.emit("cache_ready", protocol=args.protocol, reused=True)
        return

    if jax.default_backend() != "gpu" or jax.local_device_count() != 1:
        raise RuntimeError(
            f"cache builder requires exactly one visible GPU, got {jax.local_devices()}"
        )

    anns, split = cme.raw.load_ntu(Path(args.dataset))
    by_id, train_ids, val_ids = cme.split_ids(anns, split, args.protocol)
    model, params, payload = load_hand_base(ckpt)

    @jax.jit
    def infer(p, main_x, hand_x):
        return model.apply(
            {"params": p},
            main_x,
            hand_x,
            training=False,
        )["logits"]

    def materialize(ids: Sequence[str], stem: str):
        n = len(ids)

        xt = np.lib.format.open_memmap(
            root / f"{stem}_tokens.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n, cme.T32, cme.TOKEN_FEATURES),
        )
        zl = np.lib.format.open_memmap(
            root / f"{stem}_logits.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n, cme.NUM_CLASSES),
        )
        yy = np.lib.format.open_memmap(
            root / f"{stem}_y.npy",
            mode="w+",
            dtype=np.int32,
            shape=(n,),
        )

        bs = int(args.base_batch_size)
        main_buf = []
        hand_buf = []
        row_buf = []

        for i, sid in enumerate(ids):
            ann = by_id[sid]
            kp = cme.raw.annotation_keypoints(ann)

            # New information for CME itself: T32 upper-body/fine-motion tokens.
            xt[i] = cme.specialist_tokens(kp)
            yy[i] = cme.raw.annotation_label(ann)

            # Exact frozen Hand champion inputs.
            main_buf.append(lg.segment_phase_tokens_localglobal(kp))
            hand_buf.append(hand_tokens_t32(kp))
            row_buf.append(i)

            if len(main_buf) >= bs or i == n - 1:
                xb = np.asarray(main_buf, np.float32)
                hb = np.asarray(hand_buf, np.float32)
                pred = np.asarray(
                    jax.device_get(
                        infer(
                            params,
                            jnp.asarray(xb),
                            jnp.asarray(hb),
                        )
                    ),
                    np.float32,
                )
                zl[np.asarray(row_buf)] = pred
                main_buf.clear()
                hand_buf.clear()
                row_buf.clear()

        xt.flush()
        zl.flush()
        yy.flush()

    cme.emit(
        "cache_start",
        protocol=args.protocol,
        train=len(train_ids),
        val=len(val_ids),
        base_kind=BASE_KIND,
    )

    materialize(train_ids, "train")
    materialize(val_ids, "val")

    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "protocol": args.protocol,
                "base_kind": BASE_KIND,
                "base_ckpt": str(ckpt),
                "base_ckpt_sha256": sha,
                "base_architecture": "M4LocalGlobalHandM4G4T32",
                "base_main_frames": 16,
                "base_hand_frames": 32,
                "base_hand_dim": int(cfg.get("hand_dim", 32)),
                "base_hand_residual_scale": float(cfg.get("hand_residual_scale", 0.10)),
                "train_samples": len(train_ids),
                "val_samples": len(val_ids),
                "cme_frames": cme.T32,
                "cme_token_features": cme.TOKEN_FEATURES,
                "dtype": "float32",
            },
            indent=2,
        )
    )

    cme.emit("cache_ready", protocol=args.protocol, reused=False)


def main() -> int:
    args = _parse_args()

    if args.mode == "cache":
        build_cache_hand(args)
    elif args.mode == "train":
        # Training code is the same tested CME learner; only the frozen base
        # logits used to create the cache are different.
        cme.train_specialist(args)
    else:
        raise ValueError(args.mode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

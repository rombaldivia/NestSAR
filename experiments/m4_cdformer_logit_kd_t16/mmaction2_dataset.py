#!/usr/bin/env python3
from __future__ import annotations

"""CD-Former MMAction2-style NTU keypoint preprocessing used for teacher inference."""

from typing import Dict, List
import numpy as np


class MMAction2KeypointDataset:
    def __init__(
        self,
        data: Dict,
        idx_list: List[int],
        num_frames: int = 32,
        jitter: int = 6,
        drop_tokens: float = 0.2,
        is_train: bool = True,
    ):
        self.samples = [data["annotations"][i] for i in idx_list]
        self.num_frames = int(num_frames)
        self.jitter = int(jitter)
        self.drop_tokens = float(drop_tokens)
        self.is_train = bool(is_train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[int(idx)]
        kp = np.asarray(item["keypoint"], dtype=np.float32)
        kp = kp[0] if kp.ndim == 4 else kp
        label = int(item["label"])

        if self.is_train and self.jitter:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            kp = np.roll(kp, shift, axis=0)

        T = kp.shape[0]
        if T > self.num_frames:
            s = (T - self.num_frames) // 2
            kp = kp[s:s + self.num_frames]
        elif T < self.num_frames:
            if T <= 0:
                raise RuntimeError("Empty keypoint sequence")
            pad = np.repeat(kp[-1][None], self.num_frames - T, axis=0)
            kp = np.concatenate([kp, pad], axis=0)

        # Exact CD-Former data-path normalization: independently per frame,
        # across the 25 joints for each coordinate channel.
        kp = (kp - kp.mean(axis=1, keepdims=True)) / (
            kp.std(axis=1, keepdims=True) + 1e-5
        )

        if self.is_train and np.random.rand() < 0.5:
            kp[..., 0] *= -1

        if self.is_train and self.drop_tokens > 0:
            n_drop = int(round(self.num_frames * self.drop_tokens))
            if n_drop:
                idx_drop = np.random.choice(self.num_frames, n_drop, replace=False)
                kp[idx_drop] = 0

        kp = np.nan_to_num(kp).astype(np.float32, copy=False)
        if kp.shape != (self.num_frames, 25, 3):
            raise RuntimeError(
                f"MMAction2KeypointDataset expected {(self.num_frames,25,3)}, got {kp.shape}"
            )
        return kp, np.int32(label)

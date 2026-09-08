"""Only official-train examples may enter a diagnostic worker."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from .. import preprocessing_corrected as pp
from ..streaming.data import Dataset

EXPECTED_PREPROCESSING = "sm-all-personaware-p2-segmentpose-v2"
ID_PATTERN = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})(?:\D|$)")


class TrainOnlyCache:
    def __init__(self, cache, protocol):
        if protocol not in ("xsub", "xset"):
            raise ValueError("Expected xsub or xset")
        self.data = Dataset(cache)
        if pp.VERSION != EXPECTED_PREPROCESSING:
            raise ValueError("This comparison requires the person-aware P2 v3 preprocessing")
        if "internal_split" in self.data.meta["signature"]:
            raise ValueError("Use the original official P2 cache, not an internal-fold cache view")
        self.ids = json.loads((Path(cache) / "ids.json").read_text())
        self.train = np.asarray(self.data.splits[f"{protocol}_train"], np.int64)
        held = np.asarray(self.data.splits[f"{protocol}_val"], np.int64)
        n = len(self.ids)
        if n != self.data.meta["samples"] or self.data.canonical.shape != (n, 16, 750):
            raise ValueError("Unexpected ID count or T16 cache dimensions")
        for a in (self.train, held):
            if not len(a) or len(np.unique(a)) != len(a) or np.any((a < 0) | (a >= n)):
                raise ValueError("Invalid or duplicate split indices")
        if np.intersect1d(self.train, held).size:
            raise ValueError("Official train and held-out IDs overlap")
        self.allowed = np.zeros(n, bool)
        self.allowed[self.train] = True
        self.groups = np.full(n, -1, np.int32)
        # Metadata checks do not read held-out raw skeletons or canonical tokens.
        for i in self.train:
            m = ID_PATTERN.search(self.ids[i])
            if m is None:
                raise ValueError(f"Cannot parse NTU sample ID: {self.ids[i]}")
            setup, _, subject, _, action = map(int, m.groups())
            if int(self.data.labels[i]) != action - 1:
                raise ValueError(f"ID/label mismatch: {self.ids[i]}")
            self.groups[i] = subject if protocol == "xsub" else setup
        self.protocol = protocol

    def guard(self, indices):
        a = np.asarray(indices, np.int64)
        if np.any((a < 0) | (a >= len(self.allowed))) or not self.allowed[a].all():
            raise ValueError("Official held-out access blocked before reading any features")
        return a

    def pair_indices(self, pair):
        labels = self.data.labels[self.train]
        return self.train[np.isin(labels, np.asarray(pair) - 1)]

    def canonical(self, indices):
        return self.data.canonical[self.guard(indices)]

    def raw(self, index):
        self.guard([index])
        return self.data.sample(index)


def grouped_plan(train_indices, groups, labels, pairs, seed, config):
    """One common subject/setup partition for every pair and every model.

    Rejection is based only on class support, never on model performance.
    Repeats may share final groups; they are not independent test datasets.
    """
    relevant = train_indices[np.isin(labels[train_indices], np.unique(pairs) - 1)]
    unique = np.unique(groups[relevant])
    if len(unique) < 6:
        raise ValueError("Need at least six subject/setup groups for fit/select/final")
    ns = max(2, int(round(len(unique) * config["select_fraction"])))
    nf = max(2, int(round(len(unique) * config["final_fraction"])))
    if ns + nf >= len(unique) - 1:
        raise ValueError("Too few fit groups")
    rng = np.random.default_rng(seed)
    for attempt in range(10000):
        order = rng.permutation(unique)
        partition = {"fit": order[ns+nf:], "select": order[nf:ns+nf], "final": order[:nf]}
        subsets = {k: relevant[np.isin(groups[relevant], g)] for k, g in partition.items()}
        supported = all(
            np.count_nonzero(labels[subsets[k]] == action-1) >= minimum
            for k, minimum in zip(("fit", "select", "final"), config["min_class_samples"])
            for action in np.unique(pairs)
        )
        if supported:
            body = {"seed": seed, "attempt": attempt,
                    "groups": {k: sorted(map(int, g)) for k, g in partition.items()},
                    "indices": {k: sorted(map(int, a)) for k, a in subsets.items()}}
            body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            return body
    raise ValueError("Cannot obtain supported disjoint groups; reduce min_class_samples explicitly")


def prepare_sequence(raw):
    raw = np.asarray(raw, np.float32)
    local, valid, scale = pp.canonicalize_raw(raw)
    velocity = np.zeros_like(raw)
    velocity_valid = np.zeros_like(valid)
    velocity_valid[1:] = valid[1:] & valid[:-1]
    velocity[1:] = np.where(velocity_valid[1:, ..., None], np.diff(raw, axis=0), 0)
    return local, valid, scale, velocity, velocity_valid


def sequence_features(raw, frames, prepared=None):
    """Uniformly resample canonical pose and native adjacent-frame velocity.

    Both interpolation endpoints must be valid. No interpolation bridges a
    missing joint. This is a sampled 16/64-frame sequence, not a lossless raw
    sequence or a claim of an information ceiling. The same code serves both
    resolution controls; no DCT or pretrained features are used.
    """
    raw = np.asarray(raw, np.float32)
    if not len(raw):
        return np.zeros((frames, 2, 25, 8), np.float32)
    local, valid, scale, velocity, velocity_valid = prepare_sequence(raw) if prepared is None else prepared
    grid = np.linspace(0, len(raw)-1, frames)
    lo, hi = np.floor(grid).astype(int), np.ceil(grid).astype(int)
    alpha = (grid-lo).astype(np.float32)[:, None, None, None]

    def sample(values, mask):
        keep = mask[lo] & mask[hi]
        values = ((1-alpha)*values[lo] + alpha*values[hi]) / scale
        return np.where(keep[..., None], values, 0), keep

    pose, pm = sample(local, valid)
    motion, vm = sample(velocity, velocity_valid)
    return np.concatenate([pose, motion, pm[..., None], vm[..., None]], -1).astype(np.float32)


def materialize_pair(cache, pair, config, report):
    indices = cache.pair_indices(pair)
    # Only one pair is resident. Both protocols share the original raw/T16 maps.
    required = len(indices) * (16*750 + (16+64)*2*25*8) * 4
    if required > config["max_pair_ram_mib"] * 2**20:
        raise MemoryError(f"Pair feature buffers need {required/2**20:.1f} MiB")
    report(phase="Prepare pair", pair=f"A{pair[0]:03d}/A{pair[1]:03d}",
           current=0, total=len(indices), best=None, best_epoch=0, epoch=0)
    x = {"t16_mlp": np.asarray(cache.canonical(indices), np.float32),
         "sequence16_gru": np.empty((len(indices), 16, 2, 25, 8), np.float32),
         "sequence64_gru": np.empty((len(indices), 64, 2, 25, 8), np.float32)}
    for j, i in enumerate(indices):
        raw = cache.raw(int(i))
        prepared = prepare_sequence(raw)
        x["sequence16_gru"][j] = sequence_features(raw, 16, prepared)
        x["sequence64_gru"][j] = sequence_features(raw, 64, prepared)
        if j % 32 == 0 or j+1 == len(indices):
            report(current=j+1)
    for a in x.values():
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite comparison features")
    y = (cache.data.labels[indices] == pair[1]-1).astype(np.int32)
    return indices, y, x


def fit_scale(x, fit_indices):
    """Fit-only RMS conditioning preserves zeros; no final/selection statistics."""
    sequence = x.ndim == 5
    shape = (6,) if sequence else x.shape[1:]
    squares = np.zeros(shape, np.float64)
    count = np.zeros(shape, np.float64)
    for start in range(0, len(fit_indices), 64):
        b = x[fit_indices[start:start+64]]
        if sequence:
            weight = np.concatenate([np.repeat(b[..., 6:7], 3, -1),
                                     np.repeat(b[..., 7:8], 3, -1)], -1)
            squares += np.sum(np.square(b[..., :6], dtype=np.float64), axis=(0, 1, 2, 3))
            count += weight.sum(axis=(0, 1, 2, 3))
        else:
            squares += np.sum(np.square(b, dtype=np.float64), axis=0)
            count += len(b)
    return np.maximum(np.sqrt(squares / np.maximum(count, 1)), 1e-3).astype(np.float32)


def make_batch(x, y, positions, batch_size, scale, class_weights=None):
    b = np.zeros((batch_size, *x.shape[1:]), np.float32)
    b[:len(positions)] = x[positions]
    if b.ndim == 5:
        b[..., :6] /= scale
    else:
        b /= scale
    labels = np.zeros(batch_size, np.int32)
    labels[:len(positions)] = y[positions]
    mask = np.zeros(batch_size, np.float32)
    mask[:len(positions)] = 1
    weights = mask if class_weights is None else mask * class_weights[labels]
    return dict(x=b, y=labels, mask=mask, weights=weights)

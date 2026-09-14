"""Training-calibrated local-motion pose sampling; no learned/inference layers.

Coordinates are NTU metres in [T,2,25,3], before scale normalization. Only the
representative pose changes. Masks, actor slots, bins and motion summaries do not.
"""
from __future__ import annotations

import numpy as np

VERSION = "local-motion-sampler-b-v1"
# Zero-based Kinect/NTU links: tip/wrist, thumb/wrist, foot/ankle, ankle/hip.
LINK_NAMES = ("left_tip_wrist", "left_thumb_wrist", "right_tip_wrist",
              "right_thumb_wrist", "left_foot_ankle", "left_ankle_hip",
              "right_foot_ankle", "right_ankle_hip")
CHILD = np.array([21, 22, 23, 24, 15, 14, 19, 18])
PARENT = np.array([6, 6, 10, 10, 14, 12, 18, 16])
POLICY = dict(scale_quantile=0.90, minimum_positive_observations=32,
              scale_floor=1e-5, transition_cap=3.0, reversal_ratio=0.2,
              neighbor_cap=2.0, minimum_frame_score=0.05,
              histogram_bins=512, histogram_min=1e-8, histogram_max=1e2)


def relative_motion(x, valid):
    """Differences of raw-frame offsets; all four joint endpoints must exist."""
    x, valid = np.asarray(x, np.float32), np.asarray(valid, bool)
    if x.ndim != 4 or x.shape[1:] != (2, 25, 3) or valid.shape != x.shape[:-1]:
        raise ValueError("Sampler B expects [T,2,25,3] and a matching raw mask")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite sampler coordinates")
    # Float64 subtraction avoids introducing extra rounding in small offsets.
    offsets = x[:, :, CHILD].astype(np.float64) - x[:, :, PARENT]
    good = valid[:, :, CHILD] & valid[:, :, PARENT]
    eligible = good[1:] & good[:-1]
    delta = np.where(eligible[..., None], np.diff(offsets, axis=0), 0.0)
    return delta, eligible


class ScaleAccumulator:
    """Fixed-memory log histograms, observing every eligible training transition.

    Zero motions are counted separately. Positive-motion q90 is approximated by
    its bin's upper edge (about 4.6% bin width); no validation data is observed.
    """
    def __init__(self):
        self.edges = np.geomspace(POLICY["histogram_min"], POLICY["histogram_max"],
                                  POLICY["histogram_bins"] + 1)
        self.histogram = np.zeros((8, POLICY["histogram_bins"]), np.int64)
        self.valid_count = np.zeros(8, np.int64)
        self.overflow_count = np.zeros(8, np.int64)
        self.samples = 0

    def observe(self, delta, eligible):
        magnitude = np.linalg.norm(delta, axis=-1)
        self.samples += 1
        self.valid_count += eligible.sum(axis=(0, 1))
        for link in range(8):
            v = magnitude[..., link][eligible[..., link]]
            v = v[v > self.edges[0]]
            self.overflow_count[link] += np.count_nonzero(v > self.edges[-1])
            bins = np.clip(np.searchsorted(self.edges, v, side="right") - 1,
                           0, len(self.edges) - 2)
            self.histogram[link] += np.bincount(bins, minlength=len(self.edges) - 1)

    def finish(self):
        positive = self.histogram.sum(axis=1)
        scales, enabled = [], []
        for link, count in enumerate(positive):
            rank = max(1, int(np.ceil(POLICY["scale_quantile"] * count)))
            index = min(int(np.searchsorted(self.histogram[link].cumsum(), rank)),
                        len(self.edges) - 2)
            q = float(self.edges[index + 1]) if count else 0.0
            scales.append(max(q, POLICY["scale_floor"]))
            enabled.append(bool(count >= POLICY["minimum_positive_observations"]
                                and q >= POLICY["scale_floor"]
                                and q < self.edges[-1]))
        return dict(version=VERSION, policy=POLICY, links=list(LINK_NAMES),
                    scale=scales, enabled=enabled, samples_observed=self.samples,
                    valid_transitions=self.valid_count.tolist(),
                    positive_transitions=positive.tolist(),
                    histogram_overflow=self.overflow_count.tolist())


class SamplerB:
    def __init__(self, calibration):
        calibration = dict(calibration)
        # flax.serialization.to_bytes encodes Python lists as numeric-keyed state
        # dictionaries. Accept both the JSON calibration and an embedded best
        # checkpoint restored by msgpack_restore without requiring Flax at inference.
        for field in ("links", "scale", "enabled", "valid_transitions",
                      "positive_transitions", "histogram_overflow"):
            value = calibration.get(field)
            if isinstance(value, dict) and set(value) == {str(i) for i in range(8)}:
                calibration[field] = [value[str(i)] for i in range(8)]
        if calibration.get("version") != VERSION or calibration.get("policy") != POLICY:
            raise ValueError("Sampler calibration version/policy mismatch")
        if calibration.get("links") != list(LINK_NAMES):
            raise ValueError("Sampler calibration link order mismatch")
        self.calibration = calibration
        self.scale = np.asarray(calibration["scale"], np.float64)
        self.enabled = np.asarray(calibration["enabled"], bool)
        if self.scale.shape != (8,) or self.enabled.shape != (8,) or not (
                np.isfinite(self.scale).all() and (self.scale > 0).all()):
            raise ValueError("Invalid sampler scales")

    def frame_scores(self, delta, eligible):
        total = len(delta) + 1
        scores = np.zeros((total, 2), np.float64)
        suspect = np.zeros((total, 2), bool)
        if not len(delta):
            return scores, suspect
        magnitude = np.linalg.norm(delta, axis=-1)
        normalized = magnitude / self.scale
        good = eligible & self.enabled[None, None, :]
        # A very large excursion followed immediately by an almost cancelling
        # return is a candidate tracking spike. Suppress its two scoring edges,
        # and avoid selecting that frame by motion. Coordinates are NEVER edited.
        if len(delta) > 1:
            excursion = (good[:-1] & good[1:]
                         & (normalized[:-1] > POLICY["transition_cap"])
                         & (normalized[1:] > POLICY["transition_cap"])
                         & (np.linalg.norm(delta[:-1] + delta[1:], axis=-1)
                            < POLICY["reversal_ratio"] * (magnitude[:-1] + magnitude[1:])))
            suspect[1:-1] = excursion.any(axis=-1)
            good[:-1] &= ~excursion
            good[1:] &= ~excursion
        bounded = np.where(good, np.minimum(normalized, POLICY["transition_cap"]), 0.0)
        neighbors = np.zeros_like(bounded)
        if len(bounded) > 1:
            neighbors[1:] = bounded[:-1]
            neighbors[:-1] = np.maximum(neighbors[:-1], bounded[1:])
        # An unsupported single edge gets zero weight. Sustained movement stays;
        # isolated large edges cannot exceed twice their adjacent support.
        reliable = np.minimum(bounded, POLICY["neighbor_cap"] * neighbors)
        # Equal hand/leg weights, equal link weights within each group. Fixed
        # denominators prevent missing links from inflating the remaining score.
        energy = 0.5 * reliable[..., :4].mean(-1) + 0.5 * reliable[..., 4:].mean(-1)
        scores[:-1] = energy
        scores[1:] = np.maximum(scores[1:], energy)
        scores[suspect] = 0.0
        return scores, suspect

    def select(self, local, valid, starts, ends, motion=None):
        """Select one complete 25-joint pose per person/bin, plus diagnostics."""
        pose = np.zeros((len(starts), 2, 25, 3), np.float32)
        selected = np.full((len(starts), 2), -1, np.int32)
        stats = dict(present_segments=0, motion_selected=0, midpoint_fallback=0,
                     changed_from_midpoint=0, suspect_frames=0)
        if not len(local):
            return pose, selected, stats
        scores, suspect = self.frame_scores(*(relative_motion(local, valid) if motion is None else motion))
        stats["suspect_frames"] = int(suspect.sum())
        for seg, (start, end) in enumerate(zip(starts, ends)):
            if end <= start:
                continue
            midpoint = 0.5 * (int(start) + int(end) - 1)
            frames = np.arange(start, end)
            for person in range(2):
                counts = valid[start:end, person].sum(-1)
                if not counts.any():
                    continue
                # Exactly the baseline's per-person maximum-valid-joint policy.
                candidates = frames[counts == counts.max()]
                base = candidates[np.argmin(np.abs(candidates - midpoint))]
                chosen = base
                values = scores[candidates, person]
                peak = values.max()
                if peak >= POLICY["minimum_frame_score"]:
                    near_peak = candidates[np.isclose(values, peak, rtol=1e-5, atol=1e-8)]
                    chosen = near_peak[np.argmin(np.abs(near_peak - midpoint))]
                    stats["motion_selected"] += 1
                else:
                    # Exact baseline fallback, including absent/intermittent P2.
                    stats["midpoint_fallback"] += 1
                stats["present_segments"] += 1
                stats["changed_from_midpoint"] += int(chosen != base)
                selected[seg, person] = chosen
                pose[seg, person] = local[chosen, person]
        return pose, selected, stats

    def __call__(self, local, valid, starts, ends):
        return self.select(local, valid, starts, ends)[0]

"""Northwestern-UCLA (NW-UCLA) -> the PYSKL-style pickle the NestSAR pipeline reads (NESTSAR_DATASET=nwucla).

Input: the widely used preprocessed release ``all_sqe`` (one JSON per clip, e.g. ``a01_s01_e00_v01.json``, key
"skeletons" = [T][20][3] Kinect-v1 joints; the release linked from the CTR-GCN README). Output: ``nwucla_3danno.pkl``
with ``annotations`` (frame_dir, label 0..9, keypoint [1, T, 25, 3]) and ``split`` = {xview_train, xview_val}.

Standard cross-view protocol: train on views 1 and 2, test on view 3. The 10 action ids (a01-a06, a08, a09, a11,
a12) are mapped, in sorted order, to labels 0..9.

Joints: Kinect v1 (20) is NTU / Kinect v2 (25) minus spine-shoulder and the four hand tip / thumb joints, in the same
order for the first 20 except that v1 joint 2 is the shoulder centre. v1 joint i -> NTU joint i (i < 20), and the
shoulder centre is also copied to NTU joint 20 (spine shoulder). The hand tip / thumb joints (21-24) do not exist and
are left exactly zero, which the NestSAR preprocessing treats as missing joints (masked, never a fake position).

    python -m experiments.nestsar_r5_t16.prepare_nwucla --src /kaggle/input/.../all_sqe --out /kaggle/working/nwucla_3danno.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

NAME = re.compile(r"a(\d+)_s(\d+)_e(\d+)_v(\d+)", re.I)


def load_clip(path):
    d = json.loads(Path(path).read_text())
    sk = d.get("skeletons", d.get("skeleton", d.get("data")))
    x = np.asarray(sk, np.float32)
    if x.ndim != 3 or x.shape[1:] != (20, 3):
        raise ValueError(f"{path}: expected [T, 20, 3] Kinect-v1 joints, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: non-finite coordinates")
    return x


def to_ntu25(x20):
    t = x20.shape[0]
    out = np.zeros((1, t, 25, 3), np.float32)          # [M=1, T, V=25, C=3]
    out[0, :, :20] = x20
    out[0, :, 20] = x20[:, 2]                          # spine shoulder <- shoulder centre
    return out                                          # joints 21-24 (hand tips / thumbs) stay exactly 0 = missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with the all_sqe/*.json clips (searched recursively)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(p for p in Path(a.src).rglob("*.json") if NAME.search(p.stem))
    if not files:
        raise SystemExit(f"no aXX_sXX_eXX_vXX.json files under {a.src}")
    meta = [(p, *map(int, NAME.search(p.stem).groups())) for p in files]
    actions = sorted({m[1] for m in meta})
    if len(actions) != 10:
        raise SystemExit(f"expected 10 NW-UCLA actions, found {len(actions)}: {actions}")
    label_of = {act: i for i, act in enumerate(actions)}
    annotations, split = [], {"xview_train": [], "xview_val": []}
    for p, act, sub, rep, view in meta:
        x = to_ntu25(load_clip(p))
        name = p.stem
        annotations.append({"frame_dir": name, "label": label_of[act], "keypoint": x, "total_frames": x.shape[1]})
        if view in (1, 2):
            split["xview_train"].append(name)
        elif view == 3:
            split["xview_val"].append(name)
        else:
            raise SystemExit(f"{name}: unexpected view {view}")
    Path(a.out).write_bytes(pickle.dumps({"annotations": annotations, "split": split}))
    lengths = np.array([x["total_frames"] for x in annotations])
    print(f"clips {len(annotations)}  train {len(split['xview_train'])}  val {len(split['xview_val'])}  "
          f"actions {actions} -> labels 0..9  frames median {int(np.median(lengths))} "
          f"[{lengths.min()}, {lengths.max()}]  -> {a.out}")


if __name__ == "__main__":
    main()

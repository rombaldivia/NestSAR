"""Run a trained NestSAR model on one skeleton clip. Weights are downloaded once from the GitHub release.

    python infer.py --input S001C001P001R001A001.skeleton                 # NTU RGB+D (60 or 120) raw file
    python infer.py --model ntu120-xset --input clip.skeleton
    python infer.py --model nwucla --input a01_s01_e00_v03.json           # NW-UCLA all_sqe clip
    python infer.py --input clip.npy                                      # array [M, T, 25, 3] or [T, 25, 3]
    python infer.py --input pi_eval_ntu60_xsub/x.npy --index 7            # tokenised clip [N, 16, 942] (+ y.npy)
    python infer.py --list                                                # every released checkpoint

--model takes a short name (ntu60-xsub, ntu60-xview, ntu120-xsub, ntu120-xset, nwucla; best seed of the paper's
three) or any full checkpoint name from --list (other seeds, the 38-epoch ablation runs, variants).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = "rombaldivia/NestSAR"
TAG = "weights-v1"
URL = f"https://github.com/{REPO}/releases/download/{TAG}/"
CACHE = Path(os.environ.get("NESTSAR_CACHE", Path.home() / ".cache" / "nestsar"))

# Main models of the paper (60 epochs on NTU, 80 on NW-UCLA): the seed with the best validation accuracy.
SHORT = {
    "ntu60-xsub": "nestsar_ntu60_xsub_ep60_s23",     # 85.70
    "ntu60-xview": "nestsar_ntu60_xview_ep60_s23",   # 91.98
    "ntu120-xsub": "nestsar_ntu120_xsub_ep60_s7",    # 78.62
    "ntu120-xset": "nestsar_ntu120_xset_ep60_s7",    # 80.45
    "nwucla": "nestsar_nwucla_xview_ep80_s23",       # 90.73
}
MANIFESTS = ("manifest_manifest_papa.json.json", "manifest_romel_weights_NestSAR.json.json")

NTU120 = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup", "throw", "sitting down",
    "standing up (from sitting position)", "clapping", "reading", "writing", "tear up paper", "wear jacket",
    "take off jacket", "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap",
    "take off a hat/cap", "cheer up", "hand waving", "kicking something", "reach into pocket", "hopping (one foot jumping)",
    "jump up", "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard",
    "pointing to something with finger", "taking a selfie", "check time (from watch)", "rub two hands together",
    "nod head/bow", "shake head", "wipe face", "salute", "put the palms together", "cross hands in front (say stop)",
    "sneeze/cough", "staggering", "falling", "touch head (headache)", "touch chest (stomachache/heart pain)",
    "touch back (backache)", "touch neck (neckache)", "nausea or vomiting condition", "use a fan (with hand or paper)/feeling warm",
    "punching/slapping other person", "kicking other person", "pushing other person", "pat on back of other person",
    "point finger at the other person", "hugging other person", "giving something to other person",
    "touch other person's pocket", "handshaking", "walking towards each other", "walking apart from each other",
    "put on headphone", "take off headphone", "shoot at the basket", "bounce ball", "tennis bat swing",
    "juggling table tennis balls", "hush (quite)", "flick hair", "thumb up", "thumb down", "make ok sign",
    "make victory sign", "staple book", "counting money", "cutting nails", "cutting paper (using scissors)",
    "snapping fingers", "open bottle", "sniff (smell)", "squat down", "toss a coin", "fold paper", "ball up paper",
    "play magic cube", "apply cream on face", "apply cream on hand back", "put on bag", "take off bag",
    "put something into a bag", "take something out of a bag", "open a box", "move heavy objects", "shake fist",
    "throw up cap/hat", "hands up (both hands)", "cross arms", "arm circles", "arm swings", "running on the spot",
    "butt kicks (kick backward)", "cross toe touch", "side kick", "yawn", "stretch oneself", "blow nose",
    "hit other person with something", "wield knife towards other person", "knock over other person (hit with body)",
    "grab other person's stuff", "shoot at other person with a gun", "step on foot", "high-five", "cheers and drink",
    "carry something with other person", "take a photo of other person", "follow other person",
    "whisper in other person's ear", "exchange things with other person", "support somebody with hand",
    "finger-guessing game (playing rock-paper-scissors)",
]
NWUCLA = ["pick up with one hand", "pick up with two hands", "drop trash", "walk around", "sit down", "stand up",
          "donning", "doffing", "throw", "carry"]


def fetch(name):
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} ...", file=sys.stderr)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(URL + name, tmp)
        tmp.replace(path)
    return path


def released():
    out = []
    for m in MANIFESTS:
        out += json.loads(fetch(m).read_text())
    return {m["name"]: m for m in out}


def read_ntu_skeleton(path):
    """NTU RGB+D .skeleton text file -> [M, T, 25, 3], one track per body ID in order of first appearance.
    Frames without any body are dropped. The NestSAR preprocessing keeps the two most active tracks if more exist."""
    import numpy as np
    it = iter(Path(path).read_text().split())
    frames, ids = [], []
    for _ in range(int(next(it))):
        bodies = {}
        for _ in range(int(next(it))):
            body_id = next(it)
            for _ in range(9):
                next(it)
            joints = np.zeros((25, 3), np.float32)
            for j in range(int(next(it))):
                vals = [next(it) for _ in range(12)]
                if j < 25:
                    joints[j] = [float(v) for v in vals[:3]]
            bodies[body_id] = joints
            if body_id not in ids:
                ids.append(body_id)
        if bodies:
            frames.append(bodies)
    if not frames:
        raise ValueError(f"{path}: no skeleton in any frame")
    x = np.zeros((len(ids), len(frames), 25, 3), np.float32)
    for t, bodies in enumerate(frames):
        for b, joints in bodies.items():
            x[ids.index(b), t] = joints
    return x


def read_input(path):
    import numpy as np
    p = Path(path)
    if p.suffix == ".skeleton":
        return read_ntu_skeleton(p)
    if p.suffix == ".json":                          # NW-UCLA all_sqe clip, Kinect v1 joints [T, 20, 3]
        from experiments.nestsar_r5_t16.prepare_nwucla import load_clip, to_ntu25
        return to_ntu25(load_clip(p))
    x = np.load(p).astype(np.float32)
    if x.ndim == 3:
        x = x[None]
    if x.ndim != 4 or x.shape[-2:] != (25, 3):
        raise ValueError(f"expected [M, T, 25, 3] or [T, 25, 3], got {x.shape}")
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ntu60-xsub", help=f"one of {sorted(SHORT)} or a name from --list")
    ap.add_argument("--input", help=".skeleton (NTU), .json (NW-UCLA all_sqe) or .npy [M, T, 25, 3]")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--index", type=int, default=0,
                    help="clip to use when --input is an already tokenised array [N, 16, 942] (e.g. x.npy of the "
                         "Pi evaluation package); a y.npy next to it gives the true label")
    ap.add_argument("--list", action="store_true", help="list every released checkpoint")
    a = ap.parse_args()

    if a.list:
        for name, m in sorted(released().items()):
            mark = "*" if name in SHORT.values() else ("p" if m.get("paper_main") else " ")
            print(f"{mark} {m['top1']:6.2f}  {name}")
        print("\n* default for its --model short name   p model of the paper's main table")
        return
    if not a.input:
        ap.error("--input is required (or use --list)")

    name = SHORT.get(a.model, a.model)
    meta = json.loads(fetch(name + ".json").read_text())
    dataset = name.split("_")[1]
    os.environ["NESTSAR_DATASET"] = dataset          # sets the class count before the model module is imported
    os.environ.setdefault("NESTSAR_FRAMES", "16")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax
    import numpy as np
    from flax import serialization
    from experiments.nestsar_r5_t16 import preprocessing as r5pp
    from experiments.nestsar_r5_t16.config import model_kwargs, validate_config
    from experiments.nestsar_r5_t16.model import NestSARR5T16
    from experiments.nestsar_sm_all_t16.preprocessing_corrected import ordered_raw

    config = validate_config(meta["run_config"]["config"])
    model = NestSARR5T16(**model_kwargs(config))
    params = serialization.msgpack_restore(fetch(name + ".msgpack").read_bytes())["ema_params"]

    truth, src = None, Path(a.input)
    arr = np.load(src, mmap_mode="r") if src.suffix == ".npy" else None
    if arr is not None and arr.shape[-2:] == (r5pp.FRAMES, r5pp.FEATURES):      # already tokenised
        tokens = np.asarray(arr[a.index] if arr.ndim == 3 else arr, np.float32)
        what = f"tokenised clip {a.index}" if arr.ndim == 3 else "tokenised clip"
        if arr.ndim == 3 and (src.parent / "y.npy").exists():
            truth = int(np.load(src.parent / "y.npy")[a.index])
    else:
        raw = ordered_raw(read_input(a.input), "MTVC")
        tokens = r5pp.features(raw, hand_filter=config["hand_filter"], body_align=config["body_align"])
        what = f"{raw.shape[0]} frames"
    logits = model.apply({"params": params}, jax.numpy.asarray(tokens[None]), training=False)["logits"][0]
    prob = np.asarray(jax.nn.softmax(logits))
    names = NWUCLA if dataset == "nwucla" else NTU120[:60] if dataset == "ntu60" else NTU120
    acc = meta["result"].get("best_accuracy")
    acc = 100 * acc if acc is not None and acc <= 1 else acc
    print(f"{name}  (validation top-1 {acc:.2f}%)  input {src.name}: {what}")
    for k in np.argsort(-prob)[:a.top]:
        tag = f"{k:2d}" if dataset == "nwucla" else f"A{k + 1:03d}"
        print(f"  {tag}  {prob[k] * 100:5.1f}%  {names[k]}" + ("   <- true" if k == truth else ""))
    if truth is not None:
        print(f"  true class: {names[truth]}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

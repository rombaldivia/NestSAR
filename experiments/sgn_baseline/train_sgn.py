"""Train/evaluate SGN (any width) on a PYSKL NTU pickle with the original SGN recipe.

Recipe (microsoft/SGN main.py / data.py): seq translation to joint-2 of the first frame, empty frames removed,
the two actors' frames interleaved into ONE 25-joint sequence (turn_two_to_one), seg=20 segments with one
random frame each, random 3-axis rotation (theta 0.3 rad; 0.5 for cross-view), batch 64, Adam lr 1e-3,
weight decay 1e-4, label smoothing 0.1, 120 epochs, LR x0.1 at 60/90/110. Validation every epoch with one
random sampling (as SGN), best checkpoint re-tested with SGN's 5-sampling average and with 1 sampling.

  python -m experiments.sgn_baseline.train_sgn --pickle ntu60_3danno.pkl --split xsub --width 0.3125 --out DIR
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .sgn import SGN, strict_mflops


def sgn_sequence(keypoint):
    """PYSKL keypoint [M, T, 25, 3] -> SGN sequence [N, 75] (translated, empty frames dropped, two-to-one)."""
    kp = np.asarray(keypoint, np.float32)[:2]
    m, t = kp.shape[:2]
    x = np.zeros((t, 150), np.float32)
    x[:, :75] = kp[0].reshape(t, 75)
    if m > 1:
        x[:, 75:] = kp[1].reshape(t, 75)
    p1 = np.any(x[:, :75] != 0, axis=1)
    p2 = np.any(x[:, 75:] != 0, axis=1)
    first = np.flatnonzero(p1)
    if len(first):
        origin = x[first[0], 3:6].copy()                 # joint-2 (middle of the spine), first frame of actor 1
        x[p1, :75] -= np.tile(origin, 25)
        x[p2, 75:] -= np.tile(origin, 25)
    x = x[p1 | p2]                                       # remove empty frames
    out = []
    for f in x:                                          # turn_two_to_one
        a, b = f[:75], f[75:]
        if not a.any():
            out.append(b)
        elif not b.any():
            out.append(a)
        else:
            out += [a, b]
    return np.asarray(out, np.float32) if out else np.zeros((1, 75), np.float32)


def sample(seq, seg, rng):
    if len(seq) < seg:
        seq = np.concatenate([seq, np.zeros((seg - len(seq), 75), np.float32)])
    ave = len(seq) // seg
    return seq[np.arange(seg) * ave + rng.integers(ave, size=seg)]


def rotation(theta, n, gen):
    r = (torch.rand(n, 3, generator=gen) * 2 - 1) * theta
    c, s = r.cos(), r.sin()
    o, z = torch.ones(n), torch.zeros(n)
    rx = torch.stack([o, z, z, z, c[:, 0], s[:, 0], z, -s[:, 0], c[:, 0]], 1).view(n, 3, 3)
    ry = torch.stack([c[:, 1], z, -s[:, 1], z, o, z, s[:, 1], z, c[:, 1]], 1).view(n, 3, 3)
    rz = torch.stack([c[:, 2], s[:, 2], z, -s[:, 2], c[:, 2], z, z, z, o], 1).view(n, 3, 3)
    return rz @ ry @ rx


class LabelSmoothing(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.classes, self.s = classes, smoothing

    def forward(self, logits, y):
        logp = logits.log_softmax(-1)
        target = torch.full_like(logp, self.s / (self.classes - 1)).scatter_(1, y[:, None], 1 - self.s)
        return (-target * logp).sum(-1).mean()


def batches(seqs, labels, idx, seg, bs, rng, drop_last):
    for i in range(0, len(idx) - (bs - 1 if drop_last else 0), bs):
        j = idx[i:i + bs]
        if len(j) == 0:
            break
        yield torch.from_numpy(np.stack([sample(seqs[k], seg, rng) for k in j])), torch.from_numpy(labels[j])


@torch.no_grad()
def evaluate(model, seqs, labels, idx, seg, dev, crops, seed=0):
    model.eval()
    rng = np.random.default_rng(seed)
    correct = 0
    for i in range(0, len(idx), 256):
        j = idx[i:i + 256]
        logits = 0
        for _ in range(crops):
            x = torch.from_numpy(np.stack([sample(seqs[k], seg, rng) for k in j])).to(dev)
            logits = logits + model(x).float().softmax(-1)
        correct += int((logits.argmax(-1).cpu() == torch.from_numpy(labels[j])).sum())
    return correct / len(idx)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--split", required=True, help="xsub | xview | xset")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--seg", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-samples", type=int, default=0, help="debug: cap train/val sizes")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = pickle.load(open(a.pickle, "rb"))
    ann = data["annotations"]
    index = {x["frame_dir"]: i for i, x in enumerate(ann)}
    tr = np.asarray([index[n] for n in data["split"][f"{a.split}_train"] if n in index])
    va = np.asarray([index[n] for n in data["split"][f"{a.split}_val"] if n in index])
    if a.max_samples:
        tr, va = tr[:a.max_samples], va[:a.max_samples]
    labels = np.asarray([x["label"] for x in ann], np.int64)
    classes = int(labels.max()) + 1
    t0 = time.time()
    seqs = {i: sgn_sequence(ann[i]["keypoint"]) for i in np.concatenate([tr, va])}
    print(f"{len(tr)} train / {len(va)} val, {classes} classes, sequences ready in {time.time() - t0:.0f}s", flush=True)
    del data, ann

    theta = 0.5 if a.split == "xview" else 0.3
    model = SGN(classes, a.seg, a.width).to(dev)
    mflops = strict_mflops(SGN(classes, a.seg, a.width), a.seg)
    params = sum(p.numel() for p in model.parameters())
    info = {"width": a.width, "channels": model.c, "params": params, "strict_mflops_per_forward": mflops,
            "strict_mflops_5crop": 5 * mflops, "split": a.split, "seed": a.seed, "epochs": a.epochs}
    print(json.dumps(info), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 90, 110], gamma=0.1)
    crit = LabelSmoothing(classes, 0.1)
    best, best_ep, hist = -1.0, 0, []
    status = out / "status.json"
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = rng.permutation(tr)
        tl, tc, tn, t1 = 0.0, 0, 0, time.time()
        nb = len(perm) // a.batch
        for bi, (x, y) in enumerate(batches(seqs, labels, perm, a.seg, a.batch, rng, True)):
            n = x.shape[0]
            r = rotation(theta, n, gen)
            x = (x.view(n, a.seg, 25, 3) @ r[:, None].transpose(-1, -2)).reshape(n, a.seg, 75)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.item() * n
            tc += int((logits.argmax(-1) == y).sum())
            tn += n
            if bi % 50 == 0:
                status.write_text(json.dumps({"phase": "Train", "epoch": ep, "epochs": a.epochs, "current": bi, "total": nb,
                                              "best": best, "best_epoch": best_ep}))
        sched.step()
        acc = evaluate(model, seqs, labels, va, a.seg, dev, crops=1, seed=ep)
        if acc > best:
            best, best_ep = acc, ep
            torch.save(model.state_dict(), out / "best.pt")
        hist.append({"epoch": ep, "train_loss": tl / tn, "train_acc": tc / tn, "val_acc_1crop": acc,
                     "seconds": time.time() - t1})
        (out / "history.json").write_text(json.dumps(hist))
        status.write_text(json.dumps({"phase": "Eval", "epoch": ep, "epochs": a.epochs, "current": nb, "total": nb,
                                      "best": best, "best_epoch": best_ep, "last_val": acc}))
        print(f"E{ep:3d} loss {tl / tn:.3f} train {100 * tc / tn:.2f} val(1 crop) {100 * acc:.2f} best {100 * best:.2f} (E{best_ep}) "
              f"{time.time() - t1:.0f}s", flush=True)
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    res = dict(info, best_epoch=best_ep, val_1crop_selection=best,
               test_1crop=evaluate(model, seqs, labels, va, a.seg, dev, crops=1, seed=12345),
               test_5crop=evaluate(model, seqs, labels, va, a.seg, dev, crops=5, seed=12345))
    (out / "result.json").write_text(json.dumps(res, indent=1))
    status.write_text(json.dumps({"phase": "Done", "epoch": a.epochs, "epochs": a.epochs, "best": best, "best_epoch": best_ep}))
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

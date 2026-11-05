"""NESTSAR_DATASET=ntu60: 60 classes, xsub/xview splits, an NTU-120 cache is refused, a training step runs.
NTU-120 (default) keeps its constants. The NTU-60 checks run in a subprocess (the dataset is read at import)."""
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]


def _pickle(path, splits, labels, n=14, seed=0):
    rng = np.random.default_rng(seed)
    ann = []
    for i in range(n):
        people = 2 if i % 3 == 0 else 1
        total = int(rng.integers(12, 90))
        x = (rng.normal(0, 0.3, (people, 1, 25, 3)) + np.array([0.0, 0.4, 3.0])
             + rng.normal(0, 0.01, (people, total, 25, 3))).astype(np.float32)
        ann.append(dict(frame_dir=f"S001C00{1 + i % 3}P{i:03d}R001A{i % 60 + 1:03d}", keypoint=x,
                        label=int(labels[i]), total_frames=total))
    ids = [a["frame_dir"] for a in ann]
    split = {f"{splits[0]}_train": ids[:9], f"{splits[0]}_val": ids[9:],
             f"{splits[1]}_train": ids[5:], f"{splits[1]}_val": ids[:5]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(dict(annotations=ann, split=split)))


SCRIPT = r"""
import json, sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
root = Path(sys.argv[1])
from experiments.nestsar_sm_all_t16.streaming import dataset_spec as spec
from experiments.nestsar_sm_all_t16.streaming.data import prepare, Dataset as R4
from experiments.nestsar_r5_t16 import data as r5data, worker, launch
from experiments.nestsar_r5_t16.config import validate_config
assert (spec.NUM_CLASSES, spec.PROTOCOLS) == (60, ("xsub", "xview"))
assert launch.PROTOCOLS == ("xsub", "xview") and launch.find_reference(roots=[str(root)]) is None
meta = prepare(root / "ntu60.pkl", root / "r4_60", root / "s.json")
assert set(meta["split_counts"]) == {"xsub_train", "xsub_val", "xview_train", "xview_val"}
# an NTU-120 cache (xset splits, labels up to 119) must be refused in ntu60 mode, at build and at load
for name in ("ntu120_xset.pkl", "ntu120_labels.pkl"):
    try:
        prepare(root / name, root / ("bad_" + name), root / "b.json")
    except ValueError as e:
        print("refused", name, "->", str(e)[:60])
    else:
        raise AssertionError(f"{name} was accepted")
r5data.build(root / "r4_60", root / "hand_60")
ds = r5data.Dataset(root / "hand_60")
cfg = validate_config({"variant": "graph_mr", "micro_batch": 2, "accumulation_steps": 1, "epochs": 1,
                       "aug_strength": 1.0})
model, state, key, _, _ = worker.create_state(cfg, 4)
n = sum(x.size for x in jax.tree.leaves(state.params))
assert n == worker.expected_params("graph_mr") < worker.EXPECTED_PARAMS["graph_mr"], n
idx = np.asarray(ds.splits["xview_train"][:2])
b, _ = ds.batch(idx, np.arange(2), 2, dict(cfg, fresh_augmentation=True, jitter_shift=1), 1, True, "xview")
out = model.apply({"params": state.params}, jnp.asarray(b["x"]), training=False)
assert out["logits"].shape == (2, 60)
new_state, _, metrics = worker.build_steps(model, cfg)[0](state, key, {k: jnp.asarray(v) for k, v in b.items()})
assert np.isfinite(np.asarray(metrics)).all()
print(json.dumps({"params": int(n), "ntu120_params": worker.EXPECTED_PARAMS["graph_mr"],
                  "mflops": worker.strict_mflops("graph_mr"), "dataset": worker.implementation_identity("graph_mr")["dataset"]}))
"""


def test_ntu60_mode(tmp_path):
    labels60 = np.arange(14) % 60
    _pickle(tmp_path / "ntu60.pkl", ("xsub", "xview"), labels60)
    _pickle(tmp_path / "ntu120_xset.pkl", ("xsub", "xset"), labels60)
    _pickle(tmp_path / "ntu120_labels.pkl", ("xsub", "xview"), 100 + np.arange(14) % 20)
    env = dict(os.environ, NESTSAR_DATASET="ntu60", JAX_PLATFORMS="cpu", PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["dataset"] == "ntu60" and out["params"] < out["ntu120_params"]
    assert 40.0 < out["mflops"] < 40.882816


def test_ntu120_default_is_unchanged():
    from experiments.nestsar_r5_t16 import launch, worker
    from experiments.nestsar_r5_t16.model import NUM_CLASSES
    from experiments.nestsar_sm_all_t16.streaming import dataset_spec as spec
    assert (NUM_CLASSES, spec.PROTOCOLS, launch.PROTOCOLS) == (120, ("xsub", "xset"), ("xsub", "xset"))
    assert worker.class_head_delta("graph_mr") == 0 and "dataset" not in worker.implementation_identity("graph_mr")
    assert worker.strict_mflops("graph_mr") == 40.882816
    assert spec.protocol_seed(7, "xsub") == 7 and spec.protocol_seed(7, "xset") == 100007
    with pytest.raises(ValueError):
        spec.check_cache(np.zeros(3, np.int32), {"xsub_train": [], "xsub_val": [], "xview_train": [], "xview_val": []})

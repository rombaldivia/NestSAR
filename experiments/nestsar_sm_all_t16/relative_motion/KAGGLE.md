# One Kaggle cell: current relative-path branch

Enable Internet and GPU T4 ×2. Attach the P2 v3 cache or `ntu120_3danno.pkl`.
This resolves the branch once, checks out that exact commit, and runs both
protocol workers with two persistent TQDM bars. Record the printed commit with
your scores. For a permanently pinned run, replace the `COMMIT = ...` lookup
with the complete SHA of the reviewed commit.

```python
import importlib, runpy, subprocess, sys
from pathlib import Path

REPO = "https://github.com/rombaldivia/NestSAR.git"
BRANCH = "experiment/nestsar-true-relative-path-t16"

def git(*args):
    p = subprocess.run(["git", *map(str, args)], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-6000:])
    return p.stdout.strip()

COMMIT = git("ls-remote", "--exit-code", REPO, "refs/heads/" + BRANCH).split()[0]
ROOT = Path("/kaggle/working") / ("NestSAR_Relative_" + COMMIT[:12])
if not (ROOT / ".git").exists():
    if ROOT.exists() and any(ROOT.iterdir()):
        raise RuntimeError(f"Preserving nonempty directory: {ROOT}")
    ROOT.mkdir(parents=True, exist_ok=True)
    git("init", "-q", ROOT)
    git("-C", ROOT, "remote", "add", "origin", REPO)
if git("-C", ROOT, "remote", "get-url", "origin") != REPO:
    raise RuntimeError("Unexpected checkout origin")
if git("-C", ROOT, "status", "--porcelain"):
    raise RuntimeError(f"Preserving local changes in {ROOT}; use a clean checkout")
git("-C", ROOT, "fetch", "-q", "--depth=1", "origin", COMMIT)
git("-C", ROOT, "checkout", "-q", "--detach", COMMIT)
assert git("-C", ROOT, "rev-parse", "HEAD") == COMMIT
for name in list(sys.modules):
    if name == "experiments" or name.startswith("experiments."):
        del sys.modules[name]
sys.path.insert(0, str(ROOT))
importlib.invalidate_caches()
print("NestSAR source:", COMMIT)
nestsar_results = runpy.run_path(
    str(ROOT / "experiments/nestsar_sm_all_t16/kaggle_relative_motion.py"),
    init_globals={"NESTSAR_RELATIVE_SETTINGS": {
        "cache_dir": None,
        "outdir": "/kaggle/working/NestSAR_TrueRelative_T16_v1",
        "config": {"seeds": [128, 42, 28], "smoke_test": False, "audit_first": True,
                   "training": {"epochs": 60, "patience": 5}}
    }}
)["nestsar_relative_results"]
```

This trains all 120 classes on internal partitions of the official training
splits. It runs proxy/relative arms in sequence and XSUB/XSET concurrently
within each arm. The bars show each arm's best **internal selection** score;
`scores.csv` also reports the untouched internal final-group scores and paired
differences. This is an ablation to decide whether to promote the feature to a
subsequent official full-training run.

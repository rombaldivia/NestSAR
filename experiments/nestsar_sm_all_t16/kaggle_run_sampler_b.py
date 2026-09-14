# Paste this entire file into ONE Kaggle cell. Select GPU T4 x2 and Internet ON.
from pathlib import Path
import runpy
import subprocess
import tempfile

REPO = "https://github.com/rombaldivia/NestSAR.git"
BRANCH = "experiment/nestsar-sampler-b-local-motion-t16"
COMMIT = "1cce0167fe2d5bc91e43f3e927fb641fbe59452f"
ROOT = Path("/kaggle/working/NestSAR_SamplerB_" + COMMIT[:12])
DATASET = None  # Auto-detect attached ntu120_3danno.pkl; or set its full path.

def command(*args):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stdout + "\n" + result.stderr)[-5000:])
    return result.stdout.strip()

# Preserve any edited/non-git source directory by using a fresh checkout.
if ROOT.exists() and (
    not (ROOT / ".git").is_dir()
    or command("git", "-C", ROOT, "status", "--porcelain")
):
    ROOT = Path(tempfile.mkdtemp(prefix="NestSAR_SamplerB_", dir="/kaggle/working"))
if not (ROOT / ".git").is_dir():
    command("git", "clone", "--quiet", "--depth", "1", "--single-branch",
            "--branch", BRANCH, REPO, ROOT)
if command("git", "-C", ROOT, "rev-parse", "HEAD") != COMMIT:
    command("git", "-C", ROOT, "fetch", "--quiet", "--depth", "1", "origin", COMMIT)
    command("git", "-C", ROOT, "checkout", "--quiet", "--detach", COMMIT)
assert command("git", "-C", ROOT, "rev-parse", "HEAD") == COMMIT
print("Source:", COMMIT)

nestsar_results = runpy.run_path(
    str(ROOT / "experiments/nestsar_sm_all_t16/kaggle_sampler_b.py"),
    init_globals={"NESTSAR_B_SETTINGS": {
        "dataset": DATASET,
        "outdir": "/kaggle/working/NestSAR_SamplerB_T16_Seed128_v1",
        "cache_dir": "/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3",
        "config": {
            "epochs": 60, "patience": 5, "seed": 128,
            "micro_batch": 64, "accumulation_steps": 4, "eval_batch": 256,
            "fresh_augmentation": True, "prefetch_batches": 2,
            "max_train_samples": 0, "max_val_samples": 0,
        },
    }},
)["nestsar_results"]

# Paste this file into ONE Kaggle cell. Select GPU T4 x2 and Internet ON.
from pathlib import Path
import runpy
import subprocess
import tempfile

REPO = "https://github.com/rombaldivia/NestSAR.git"
BRANCH = "experiment/nestsar-part-readout-t16"
COMMIT = "4b63284b1fb6786492a490ba01522f62cddb1723"
ROOT = Path("/kaggle/working/NestSAR_PartReadout_" + COMMIT[:12])
DATASET = None  # Auto-detect attached ntu120_3danno.pkl, or specify its full path.

def command(*args):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stdout + "\n" + result.stderr)[-5000:])
    return result.stdout.strip()

if ROOT.exists() and (
    not (ROOT / ".git").is_dir()
    or command("git", "-C", ROOT, "status", "--porcelain")
):
    ROOT = Path(tempfile.mkdtemp(prefix="NestSAR_PartReadout_", dir="/kaggle/working"))
if not (ROOT / ".git").is_dir():
    command("git", "clone", "--quiet", "--depth", "1", "--single-branch",
            "--branch", BRANCH, REPO, ROOT)
if command("git", "-C", ROOT, "rev-parse", "HEAD") != COMMIT:
    command("git", "-C", ROOT, "fetch", "--quiet", "--depth", "1", "origin", COMMIT)
    command("git", "-C", ROOT, "checkout", "--quiet", "--detach", COMMIT)
assert command("git", "-C", ROOT, "rev-parse", "HEAD") == COMMIT
print("Source:", COMMIT)

nestsar_results = runpy.run_path(
    str(ROOT / "experiments/nestsar_sm_all_t16/kaggle_part_readout.py"),
    init_globals={"NESTSAR_PART_SETTINGS": {
        "dataset": DATASET,
        "outdir": "/kaggle/working/NestSAR_PartReadout_T16_Seed128_v1",
        "cache_dir": "/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_v3",
        "sampler_cache_dir": None,  # Reuse previous B overlays when available.
        "audit_first": True,  # Paired baseline/readout FLOP audit on the T4.
        "config": {
            "epochs": 60, "patience": 5, "seed": 128,
            "micro_batch": 64, "accumulation_steps": 4, "eval_batch": 256,
            "fresh_augmentation": True, "prefetch_batches": 2,
            "max_train_samples": 0, "max_val_samples": 0,
        },
    }},
)["nestsar_results"]

"""Stand-in worker for launcher tests (behaviour chosen by <outdir>/fake_plan.json)."""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--protocol")
    ap.add_argument("--cache")
    ap.add_argument("--outdir")
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.outdir)
    plan = json.loads((out / "fake_plan.json").read_text())["mode"]
    d = out / a.protocol
    d.mkdir(parents=True, exist_ok=True)
    attempts_file = d / "attempts"
    attempts = int(attempts_file.read_text()) + 1 if attempts_file.exists() else 1
    attempts_file.write_text(str(attempts))
    if plan == "crash_once" and attempts == 1:
        print("transient failure (simulated)", flush=True)
        sys.exit(3)
    if plan == "nan":
        print("FloatingPointError: Nonfinite training values at epoch 2, batch 7", flush=True)
        sys.exit(1)
    if plan == "kill":
        (d / "history.json").write_text(json.dumps([{"epoch": 1, "val_acc": 0.10}]))
        (d / "status.json").write_text(json.dumps({"phase": "Train", "epoch": 2}))
        time.sleep(120)
        sys.exit(0)
    (d / "history.json").write_text(json.dumps([{"epoch": 1, "val_acc": 0.80}]))
    (d / "result.json").write_text(json.dumps({"best_val_accuracy": 0.80, "best_epoch": 1}))


if __name__ == "__main__":
    main()

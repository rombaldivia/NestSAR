#!/usr/bin/env python3
"""Aggregate Attention-Lite paper runs into mean/std summaries."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from .paths import MODEL_SLUG, PAPER_SEEDS, run_folder_name


def _load_one(base: Path, protocol: str, seed: int) -> dict:
    root = base / run_folder_name(protocol, seed)
    path = root / "result.json"
    if not path.is_file():
        return {
            "protocol": protocol,
            "seed": seed,
            "status": "missing",
            "root": str(root),
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_acc = data.get("best_val_accuracy")
    if raw_acc is None:
        raise KeyError(f"best_val_accuracy missing from {path}")
    acc = float(raw_acc)
    # Canonical result stores accuracy as a fraction (e.g. 0.7311).
    if acc <= 1.0:
        acc *= 100.0
    return {
        "protocol": protocol,
        "seed": seed,
        "status": "complete",
        "root": str(root),
        "best_epoch": int(data.get("best_epoch", -1)),
        "best_val_accuracy_percent": acc,
        "parameters": int(data.get("parameters", -1)),
        "leaves": int(data.get("leaves", -1)),
        "accepted": int(data.get("accepted", -1)),
        "rollback": int(data.get("rollback", -1)),
    }


def summarize(base_dir: str | Path = "/kaggle/working") -> dict:
    base = Path(base_dir)
    output: dict[str, object] = {
        "model": "NestSAR-HOPE-Attention-Lite-D128-v1",
        "expected_seeds": list(PAPER_SEEDS),
        "protocols": {},
    }
    for protocol in ("xsub", "xset"):
        rows = [_load_one(base, protocol, seed) for seed in PAPER_SEEDS]
        values = [r["best_val_accuracy_percent"] for r in rows if r["status"] == "complete"]
        stats = {
            "n_complete": len(values),
            "n_expected": len(PAPER_SEEDS),
            "mean_accuracy_percent": statistics.mean(values) if values else None,
            "std_accuracy_percent_sample": statistics.stdev(values) if len(values) >= 2 else None,
            "min_accuracy_percent": min(values) if values else None,
            "max_accuracy_percent": max(values) if values else None,
        }
        output["protocols"][protocol] = {"runs": rows, "stats": stats}
    return output


def _print_table(summary: dict) -> None:
    print("=" * 92)
    print("NESTSAR ATTENTION-LITE — 4-SEED PAPER SUMMARY")
    print("=" * 92)
    for protocol in ("xsub", "xset"):
        block = summary["protocols"][protocol]
        print(f"\n{protocol.upper()}")
        print("-" * 92)
        for row in block["runs"]:
            if row["status"] == "complete":
                print(
                    f"seed {row['seed']:4d} | best E{row['best_epoch']:02d} | "
                    f"{row['best_val_accuracy_percent']:.5f}% | rollback={row['rollback']}"
                )
            else:
                print(f"seed {row['seed']:4d} | MISSING | {row['root']}")
        stats = block["stats"]
        if stats["n_complete"]:
            mean = stats["mean_accuracy_percent"]
            std = stats["std_accuracy_percent_sample"]
            std_text = f"{std:.5f}" if std is not None else "n/a"
            print(
                f"MEAN ± STD: {mean:.5f}% ± {std_text} pp "
                f"({stats['n_complete']}/{stats['n_expected']} seeds complete)"
            )
        else:
            print("No completed runs found.")
    print("=" * 92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="/kaggle/working")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summary = summarize(args.runs_root)
    _print_table(summary)
    out = Path(args.output) if args.output else Path(args.runs_root) / f"{MODEL_SLUG}_4SEED_SUMMARY.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

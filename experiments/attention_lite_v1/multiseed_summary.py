#!/usr/bin/env python3
"""Aggregate Attention-Lite reproducibility runs into mean/std summaries."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .paths import MODEL_SLUG, PAPER_SEEDS, run_folder_name, sanitize_tag

LEGACY_SEED128 = {
    "xsub": "NestSAR_HOPE_ATTENTION_LITE_D128_XSUB_E40",
    "xset": "NestSAR_HOPE_ATTENTION_LITE_D128_XSET_E40",
}


def _candidate_result_paths(
    base: Path,
    protocol: str,
    seed: int,
    run_tag: str | None,
) -> list[tuple[Path, str]]:
    root = base / run_folder_name(protocol, seed, run_tag)
    candidates = [(root / "result.json", "seed_labeled")]
    # The historical seed-128 checkpoint used the golden untagged recipe.  Never mix
    # it into a tagged/custom configuration's statistics.
    if seed == 128 and run_tag is None:
        candidates.append((base / LEGACY_SEED128[protocol] / "result.json", "legacy_seed128"))
    return candidates


def _load_one(base: Path, protocol: str, seed: int, run_tag: str | None) -> dict:
    candidates = _candidate_result_paths(base, protocol, seed, run_tag)
    selected: tuple[Path, str] | None = None
    for path, source_kind in candidates:
        if path.is_file():
            selected = (path, source_kind)
            break

    expected_root = base / run_folder_name(protocol, seed, run_tag)
    if selected is None:
        return {
            "protocol": protocol,
            "seed": seed,
            "run_tag": run_tag,
            "status": "missing",
            "root": str(expected_root),
            "searched": [str(p) for p, _ in candidates],
        }

    path, source_kind = selected
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_acc = data.get("best_val_accuracy")
    if raw_acc is None:
        raise KeyError(f"best_val_accuracy missing from {path}")
    acc = float(raw_acc)
    if acc <= 1.0:
        acc *= 100.0

    result_seed = int(data.get("seed", -1))
    result_protocol = str(data.get("protocol", "")).lower()
    if result_seed != seed:
        raise RuntimeError(f"Seed mismatch in {path}: result={result_seed}, expected={seed}")
    if result_protocol != protocol:
        raise RuntimeError(f"Protocol mismatch in {path}: result={result_protocol}, expected={protocol}")

    return {
        "protocol": protocol,
        "seed": seed,
        "run_tag": run_tag,
        "status": "complete",
        "source_kind": source_kind,
        "root": str(path.parent),
        "result_path": str(path),
        "best_epoch": int(data.get("best_epoch", -1)),
        "best_val_accuracy_percent": acc,
        "parameters": int(data.get("parameters", -1)),
        "leaves": int(data.get("leaves", -1)),
        "accepted": int(data.get("accepted", -1)),
        "rollback": int(data.get("rollback", -1)),
    }


def summarize(
    base_dir: str | Path = "/kaggle/working",
    *,
    run_tag: str | None = None,
) -> dict:
    base = Path(base_dir)
    clean_tag = sanitize_tag(run_tag)
    output: dict[str, object] = {
        "model": "NestSAR-HOPE-Attention-Lite-D128-v1",
        "run_tag": clean_tag,
        "expected_seeds": list(PAPER_SEEDS),
        "protocols": {},
    }
    for protocol in ("xsub", "xset"):
        rows = [_load_one(base, protocol, seed, clean_tag) for seed in PAPER_SEEDS]
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
    print("NESTSAR ATTENTION-LITE — 4-SEED REPRODUCIBILITY SUMMARY")
    print(f"RUN TAG: {summary.get('run_tag') or '<golden/untagged>'}")
    print("=" * 92)
    for protocol in ("xsub", "xset"):
        block = summary["protocols"][protocol]
        print(f"\n{protocol.upper()}")
        print("-" * 92)
        for row in block["runs"]:
            if row["status"] == "complete":
                legacy = " legacy" if row.get("source_kind") == "legacy_seed128" else ""
                print(
                    f"seed {row['seed']:4d} | best E{row['best_epoch']:02d} | "
                    f"{row['best_val_accuracy_percent']:.5f}% | rollback={row['rollback']}{legacy}"
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
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summary = summarize(args.runs_root, run_tag=args.run_tag)
    _print_table(summary)

    clean_tag = sanitize_tag(args.run_tag)
    if args.output:
        out = Path(args.output)
    else:
        suffix = f"_{clean_tag}" if clean_tag else ""
        out = Path(args.runs_root) / f"{MODEL_SLUG}_4SEED_SUMMARY{suffix}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run Person-Aware P2 SM-ALL on the complete official NTU120 XSUB/XSET splits.

This launcher deliberately rejects internal/subset cache views. Training uses every sample
from the official protocol training split and validation uses every sample from the official
held-out split:

    XSUB train=63,026  val=50,919
    XSET train=54,468  val=59,477

The two protocols are then launched in parallel through streaming.launch (GPU0/GPU1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.nestsar_sm_all_t16.streaming.data import prepare
from experiments.nestsar_sm_all_t16.streaming.launch import find_dataset, run

EXPECTED = {
    "xsub_train": 63026,
    "xsub_val": 50919,
    "xset_train": 54468,
    "xset_val": 59477,
}


def verify_full_official_cache(cache: Path) -> dict:
    manifest_path = cache / "manifest.json"
    splits_path = cache / "splits.json"
    if not manifest_path.is_file() or not splits_path.is_file():
        raise FileNotFoundError(f"Incomplete cache: {cache}")

    manifest = json.loads(manifest_path.read_text())
    splits = json.loads(splits_path.read_text())
    counts = {name: len(splits.get(name, [])) for name in EXPECTED}
    if counts != EXPECTED:
        raise RuntimeError(
            "Refusing subset/internal split cache. "
            f"Expected {EXPECTED}, found {counts}."
        )

    signature = manifest.get("signature", {})
    if "internal_split" in signature:
        raise RuntimeError(
            "Refusing cache view tagged as internal_split. Use the original full official cache."
        )

    for protocol in ("xsub", "xset"):
        train = set(map(int, splits[f"{protocol}_train"]))
        val = set(map(int, splits[f"{protocol}_val"]))
        if train & val:
            raise RuntimeError(f"{protocol.upper()} train/val overlap detected")

    if int(manifest.get("samples", -1)) != 113945:
        raise RuntimeError(
            f"Expected 113945 valid NTU120 annotations, found {manifest.get('samples')}"
        )

    return {
        "samples": manifest["samples"],
        "split_counts": counts,
        "signature": signature,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--outdir",
        default="/kaggle/working/NestSAR_SM_ALL_T16_PERSON_AWARE_P2_FULL_OFFICIAL",
    )
    parser.add_argument(
        "--cache",
        default="/kaggle/working/NestSAR_SM_ALL_PERSON_AWARE_P2_CACHE_FULL_OFFICIAL",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    dataset = find_dataset(args.dataset)
    out = Path(args.outdir)
    cache = Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    # Build/reuse the exact branch cache before launching any GPU training, then hard-check
    # that it is the official full XSUB/XSET split rather than an internal-fold view.
    prepare(
        dataset=dataset,
        cache=cache,
        status=out / "full_cache_prepare_status.json",
        layout="MTVC",
    )
    audit = verify_full_official_cache(cache)

    print("=" * 110)
    print("NESTSAR PERSON-AWARE P2 — FULL OFFICIAL NTU120")
    print("=" * 110)
    print(f"Dataset: {dataset}")
    print(f"Samples: {audit['samples']}")
    for name, count in audit["split_counts"].items():
        print(f"{name:10s}: {count:,}")
    print("Subset caps: OFF (max_train_samples=0, max_val_samples=0)")
    print("Assignment: XSUB -> GPU0 | XSET -> GPU1")
    print("=" * 110)

    config = {
        "epochs": args.epochs,
        "patience": args.patience,
        "max_train_samples": 0,
        "max_val_samples": 0,
    }

    results = run(
        dataset=str(dataset),
        outdir=str(out),
        cache_dir=str(cache),
        config=config,
        raw_layout="MTVC",
        audit_first=True,
    )

    print("\nFULL OFFICIAL RUN COMPLETE")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()

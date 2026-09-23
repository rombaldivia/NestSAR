"""Run NestSAR P2 + G4 temporal moments on full NTU120 XSUB/XSET.

The experiment changes only G4 chunk construction:
plain four-frame mean -> mean + temporal trend + curvature.

XSUB runs on GPU0 and XSET on GPU1 through the existing streaming launcher.
The launcher owns two persistent tqdm notebook bars.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.nestsar_sm_all_t16.streaming.data import prepare
from experiments.nestsar_sm_all_t16.streaming import launch as streaming_launch


EXPECTED_SPLITS = {
    "xsub_train": 63026,
    "xsub_val": 50919,
    "xset_train": 54468,
    "xset_val": 59477,
}

EXPECTED_PARAMS = 1_827_452


def verify_full_official_cache(cache: Path) -> dict:
    manifest_path = cache / "manifest.json"
    splits_path = cache / "splits.json"

    if not manifest_path.is_file() or not splits_path.is_file():
        raise FileNotFoundError(f"Incomplete cache: {cache}")

    manifest = json.loads(manifest_path.read_text())
    splits = json.loads(splits_path.read_text())

    counts = {
        name: len(splits.get(name, []))
        for name in EXPECTED_SPLITS
    }

    if counts != EXPECTED_SPLITS:
        raise RuntimeError(
            "Refusing subset/internal split cache. "
            f"Expected {EXPECTED_SPLITS}, found {counts}."
        )

    signature = manifest.get("signature", {})
    if "internal_split" in signature:
        raise RuntimeError(
            "Refusing cache tagged as internal_split."
        )

    for protocol in ("xsub", "xset"):
        train = set(map(int, splits[f"{protocol}_train"]))
        val = set(map(int, splits[f"{protocol}_val"]))
        if train & val:
            raise RuntimeError(
                f"{protocol.upper()} train/heldout overlap detected"
            )

    if int(manifest.get("samples", -1)) != 113945:
        raise RuntimeError(
            "Expected 113945 usable NTU120 samples, "
            f"found {manifest.get('samples')}"
        )

    return {
        "samples": int(manifest["samples"]),
        "split_counts": counts,
        "signature": signature,
    }


def run_g4_temporal_moments(
    dataset=None,
    outdir="/kaggle/working/NestSAR_G4_TEMPORAL_MOMENTS_T16_FULL_OFFICIAL",
    cache="/kaggle/working/NestSAR_G4_TEMPORAL_MOMENTS_T16_CACHE",
    epochs=60,
    patience=5,
    audit_first=True,
):
    dataset = streaming_launch.find_dataset(dataset)

    out = Path(outdir)
    cache = Path(cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    # Build or reuse the exact official cache first, then hard-verify it.
    prepare(
        dataset=dataset,
        cache=cache,
        status=out / "full_cache_prepare_status.json",
        layout="MTVC",
    )
    audit = verify_full_official_cache(cache)

    print("=" * 112)
    print("NESTSAR P2 + G4 TEMPORAL MOMENTS — FULL NTU120")
    print("=" * 112)
    print(f"Dataset: {dataset}")
    print(f"Samples: {audit['samples']:,}")
    print(f"Parameters expected: {EXPECTED_PARAMS:,}")
    print("G4 chunker: mean + temporal trend + curvature")
    print("Frames: T16 -> four G4 chunks")
    print("Assignment: XSUB -> GPU0 | XSET -> GPU1")
    print("Progress: two tqdm notebook bars")
    print("Subset caps: OFF")
    for name, count in audit["split_counts"].items():
        print(f"{name:10s}: {count:,}")
    print("=" * 112)

    config = {
        "epochs": int(epochs),
        "patience": int(patience),
        "max_train_samples": 0,
        "max_val_samples": 0,
    }

    # Deliberately DO NOT monkey-patch make_bars/update_bar:
    # streaming.launch uses tqdm.notebook in Jupyter/Kaggle.
    results = streaming_launch.run(
        dataset=str(dataset),
        outdir=str(out),
        cache_dir=str(cache),
        config=config,
        raw_layout="MTVC",
        audit_first=bool(audit_first),
    )

    print("\n" + "=" * 112)
    print("G4 TEMPORAL MOMENTS RUN COMPLETE")
    print("=" * 112)
    print(json.dumps(results, indent=2, default=str))

    return results


if __name__ == "__main__":
    run_g4_temporal_moments()

"""Train clean P2 NestSAR with an attention supervisor that exists only in training.

The deployed/evaluated graph is the original P2-v3 NestSAR:
J/B/JM/BM -> Spatial -> M4 -> Router -> G4 -> classifiers.

During training only, a single 4-head attention block sees the full M4
[B,16,4,112] representation and contributes an auxiliary CE loss. Its output
never enters the deployed NestSAR forward path.

XSUB runs on GPU0 and XSET on GPU1 using the existing two-tqdm launcher.
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

DEPLOY_PARAMS = 1_826_556
TRAINING_PARAMS = 1_893_428


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
        raise RuntimeError("Refusing cache tagged as internal_split.")

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


def run_training_attention(
    dataset=None,
    outdir="/kaggle/working/NestSAR_TRAIN_ONLY_ATTENTION_T16_FULL_OFFICIAL",
    cache="/kaggle/working/NestSAR_TRAIN_ONLY_ATTENTION_T16_CACHE",
    epochs=60,
    patience=5,
    attention_aux_weight=0.20,
    audit_first=True,
):
    dataset = streaming_launch.find_dataset(dataset)

    out = Path(outdir)
    cache = Path(cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    prepare(
        dataset=dataset,
        cache=cache,
        status=out / "full_cache_prepare_status.json",
        layout="MTVC",
    )

    audit = verify_full_official_cache(cache)

    print("=" * 116)
    print("NESTSAR P2 + TRAINING-ONLY ATTENTION SUPERVISOR — FULL NTU120")
    print("=" * 116)
    print(f"Dataset: {dataset}")
    print(f"Samples: {audit['samples']:,}")
    print(f"Deploy parameters:   {DEPLOY_PARAMS:,}")
    print(f"Training parameters: {TRAINING_PARAMS:,}")
    print("Attention: 1 block | 4 heads | 64 M4 tokens | D=112")
    print(f"Attention CE weight: {float(attention_aux_weight):.3f}")
    print("Attention evaluation/deployment: OFF")
    print("Assignment: XSUB -> GPU0 | XSET -> GPU1")
    print("Progress: two tqdm notebook bars")
    print("Subset caps: OFF")

    for name, count in audit["split_counts"].items():
        print(f"{name:10s}: {count:,}")

    print("=" * 116)

    config = {
        "epochs": int(epochs),
        "patience": int(patience),
        "attention_heads": 4,
        "attention_dropout": 0.10,
        "attention_aux_weight": float(attention_aux_weight),
        "max_train_samples": 0,
        "max_val_samples": 0,
    }

    # No UI monkey patch: streaming.launch uses tqdm.notebook in Kaggle.
    results = streaming_launch.run(
        dataset=str(dataset),
        outdir=str(out),
        cache_dir=str(cache),
        config=config,
        raw_layout="MTVC",
        audit_first=bool(audit_first),
    )

    print("\n" + "=" * 116)
    print("TRAINING-ONLY ATTENTION RUN COMPLETE")
    print("=" * 116)
    print(json.dumps(results, indent=2, default=str))

    return results


if __name__ == "__main__":
    run_training_attention()

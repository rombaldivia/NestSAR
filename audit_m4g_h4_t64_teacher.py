#!/usr/bin/env python3
from __future__ import annotations

"""Recovery/provenance audit for the historical M4G-H4 T64 teacher.

This script is intentionally model-import-free.  The historical JointFirst-Wide128
wrapper in the repository is not self-contained because it expects helper modules
that are not present on the branch.  Before implementing feature KD, this audit
searches attached Kaggle assets for:

  * candidate .msgpack checkpoints
  * sibling/nearby JSON metadata
  * historical helper source files
  * EMA/online parameter trees and parameter counts
  * likely frame/protocol/accuracy/seed metadata
  * SHA256 hashes for reproducibility

It does not claim that a checkpoint is the T64 teacher merely from its filename.
The output is a ranked recovery report that must be inspected before model loading.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from flax import serialization
except Exception:
    serialization = None

TARGET_TERMS = (
    "m4g", "m4", "h4", "jointfirst", "joint_first", "wide128", "wide_128",
    "geom", "geometry", "regmask", "ema", "seed", "xsub", "xset", "t64",
    "64f", "64frame", "64_frame",
)

REQUIRED_SOURCES = (
    "nestsar_m4_geom_h4_jointfirst_wide128_v4.py",
    "nestsar_m4_geom_h4.py",
    "nestsar_sms_s1c_v2.py",
    "nestsar_m4_regmask_ema_v3_safe.py",
    "patch_sms_exact_guard_v5.py",
)

INTERESTING_KEYS = (
    "model", "model_id", "mode", "architecture", "protocol", "seed", "frames",
    "num_frames", "frame_count", "best_val_accuracy", "val_accuracy", "accuracy",
    "best_epoch", "epoch", "step", "ema_decay", "checkpoint_contains_ema",
    "regularization_variant", "params", "parameters", "gflops", "flops",
    "dataset", "config_hash",
)


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def scalarize(v: Any):
    if isinstance(v, (str, bool, int, float)) or v is None:
        return v
    try:
        a = np.asarray(v)
        if a.ndim == 0:
            x = a.item()
            if isinstance(x, (str, bool, int, float)):
                return x
    except Exception:
        pass
    return None


def collect_metadata(obj: Any, prefix: str = "", depth: int = 0, out=None):
    if out is None:
        out = {}
    if depth > 5:
        return out
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            ks = str(k)
            p = f"{prefix}.{ks}" if prefix else ks
            low = ks.lower()
            sv = scalarize(v)
            if sv is not None and any(key in low for key in INTERESTING_KEYS):
                out[p] = sv
            if isinstance(v, Mapping):
                collect_metadata(v, p, depth + 1, out)
    return out


def numeric_leaf_stats(tree: Any):
    leaves = []

    def walk(x, name=""):
        if isinstance(x, Mapping):
            for k, v in x.items():
                walk(v, f"{name}/{k}" if name else str(k))
            return
        if isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, f"{name}/{i}")
            return
        try:
            a = np.asarray(x)
        except Exception:
            return
        if a.dtype.kind not in "biufc" or a.ndim == 0:
            return
        leaves.append((name, tuple(int(s) for s in a.shape), int(a.size), str(a.dtype)))

    walk(tree)
    return {
        "leaf_count": len(leaves),
        "parameter_count": int(sum(x[2] for x in leaves)),
        "largest_leaves": sorted(
            ({"name": n, "shape": s, "size": z, "dtype": d} for n, s, z, d in leaves),
            key=lambda r: r["size"], reverse=True,
        )[:20],
    }


def restore_msgpack(path: Path):
    if serialization is None:
        return {"ok": False, "error": "flax.serialization unavailable"}
    try:
        payload = serialization.msgpack_restore(path.read_bytes())
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    out = {
        "ok": True,
        "top_level_type": type(payload).__name__,
        "top_level_keys": list(payload.keys())[:100] if isinstance(payload, Mapping) else [],
        "metadata": collect_metadata(payload),
    }
    if isinstance(payload, Mapping):
        for key in ("ema_params", "params"):
            if key in payload:
                out[key] = numeric_leaf_stats(payload[key])
    return out


def nearby_jsons(path: Path):
    candidates = []
    sibling = path.with_suffix(".json")
    if sibling.is_file():
        candidates.append(sibling)
    try:
        for p in path.parent.glob("*.json"):
            if p not in candidates:
                candidates.append(p)
    except Exception:
        pass
    return candidates[:30]


def text_score(text: str) -> int:
    low = text.lower()
    score = 0
    weights = {
        "m4g": 8, "m4": 3, "h4": 7, "jointfirst": 8, "wide128": 8,
        "t64": 8, "64f": 7, "64 frame": 7, "frames": 1, "xsub": 2,
        "xset": 2, "ema": 2, "regmask": 2, "geometry": 2, "geom": 1,
    }
    for term, weight in weights.items():
        if term in low:
            score += weight
    return score


def scan_roots(roots):
    checkpoints = []
    source_hits = {name: [] for name in REQUIRED_SOURCES}

    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        print(f"SCAN_ROOT={root}", flush=True)
        for dirpath, dirnames, filenames in os.walk(root):
            # Avoid repository internals / caches that cannot contain teacher assets.
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".cache"}]
            base = Path(dirpath)
            for fn in filenames:
                p = base / fn
                if fn in source_hits:
                    source_hits[fn].append(str(p))
                if p.suffix.lower() == ".msgpack":
                    checkpoints.append(p)
    return checkpoints, source_hits


def audit_checkpoint(path: Path):
    entry = {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256(path),
        "filename_score": text_score(str(path)),
    }

    metas = []
    meta_text = ""
    for jp in nearby_jsons(path):
        obj = safe_json(jp)
        if obj is None:
            continue
        flat = collect_metadata(obj)
        metas.append({
            "path": str(jp),
            "sha256": sha256(jp),
            "interesting": flat,
        })
        try:
            meta_text += " " + json.dumps(obj, default=str)[:200000]
        except Exception:
            pass
    entry["nearby_json"] = metas
    entry["metadata_score"] = text_score(meta_text)
    entry["msgpack"] = restore_msgpack(path)

    combined = str(path) + " " + meta_text + " " + json.dumps(entry["msgpack"], default=str)
    entry["total_score"] = text_score(combined)

    flags = []
    low = combined.lower()
    if "ema_params" in low or "checkpoint_contains_ema" in low:
        flags.append("EMA_PRESENT_OR_REFERENCED")
    if any(x in low for x in ("t64", "64f", '"frames": 64', "frames=64", "num_frames": 64)):
        flags.append("T64_REFERENCED")
    if "xsub" in low:
        flags.append("XSUB_REFERENCED")
    if "xset" in low:
        flags.append("XSET_REFERENCED")
    if "jointfirst" in low or "joint_first" in low:
        flags.append("JOINTFIRST_REFERENCED")
    if "wide128" in low or "wide_128" in low:
        flags.append("WIDE128_REFERENCED")
    entry["flags"] = flags
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=["/kaggle/input", "/kaggle/working"])
    ap.add_argument("--out", default="/kaggle/working/m4g_h4_t64_teacher_recovery.json")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    print("=" * 120)
    print("NESTSAR M4G-H4 T64 TEACHER — RECOVERY / PROVENANCE AUDIT")
    print("=" * 120)
    print("This audit does NOT assume a filename proves teacher identity.")
    print("It inventories candidate checkpoints, metadata and missing historical source modules.")

    ckpts, source_hits = scan_roots(args.roots)
    print(f"MSG_PACK_FILES_FOUND={len(ckpts)}")

    audited = []
    for i, p in enumerate(ckpts, 1):
        print(f"AUDIT_CHECKPOINT {i}/{len(ckpts)}: {p}", flush=True)
        try:
            audited.append(audit_checkpoint(p))
        except Exception as exc:
            audited.append({
                "path": str(p),
                "error": f"{type(exc).__name__}: {exc}",
                "total_score": -1,
            })

    audited.sort(key=lambda x: (x.get("total_score", -1), x.get("size_bytes", 0)), reverse=True)

    report = {
        "roots": args.roots,
        "required_historical_sources": list(REQUIRED_SOURCES),
        "source_hits": source_hits,
        "checkpoint_count": len(ckpts),
        "candidates": audited,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 120)
    print("HISTORICAL SOURCE RECOVERY")
    print("=" * 120)
    for name in REQUIRED_SOURCES:
        hits = source_hits[name]
        print(f"{name:48s} : {'FOUND' if hits else 'MISSING'}")
        for p in hits[:10]:
            print(f"    {p}")

    print("\n" + "=" * 120)
    print(f"TOP {min(args.top, len(audited))} CHECKPOINT CANDIDATES")
    print("=" * 120)
    for rank, row in enumerate(audited[: args.top], 1):
        mp = row.get("msgpack", {})
        ema_n = mp.get("ema_params", {}).get("parameter_count") if isinstance(mp, Mapping) else None
        raw_n = mp.get("params", {}).get("parameter_count") if isinstance(mp, Mapping) else None
        print(f"[{rank:02d}] score={row.get('total_score', -1):3d} size={row.get('size_bytes', 0):,}")
        print(f"     {row.get('path')}")
        print(f"     flags={row.get('flags', [])}")
        print(f"     params={raw_n} ema_params={ema_n}")
        if row.get("nearby_json"):
            for meta in row["nearby_json"][:3]:
                print(f"     meta={meta['path']}")
                interesting = meta.get("interesting", {})
                if interesting:
                    print(f"       {interesting}")

    print("\n" + "=" * 120)
    print(f"REPORT={out}")
    print("NEXT STEP: send the TOP CANDIDATES + HISTORICAL SOURCE RECOVERY block before KD implementation.")
    print("=" * 120)


if __name__ == "__main__":
    main()

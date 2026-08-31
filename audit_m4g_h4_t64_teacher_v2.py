#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from flax import serialization


KEYWORDS = {
    "m4g_h4": 8,
    "m4g-h4": 8,
    "m4_geom_h4": 7,
    "geom_h4": 6,
    "jointfirst": 6,
    "joint_first": 6,
    "wide128": 5,
    "wide_128": 5,
    "t64": 7,
    "64f": 7,
    "64 frames": 7,
    '"frames": 64': 7,
    "frames=64": 7,
    '"num_frames": 64': 7,
    "xsub": 2,
    "xset": 2,
    "ema_params": 3,
    "checkpoint_contains_ema": 3,
    "best": 1,
}

INTERESTING_META_KEYS = (
    "model", "model_id", "model_name", "architecture", "variant",
    "frames", "num_frames", "protocol", "seed", "epoch", "best_epoch",
    "best_val_accuracy", "val_accuracy", "accuracy", "params", "parameters",
    "checkpoint_contains_ema", "ema_decay", "git_commit", "commit", "config",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def count_tree(tree: Any) -> int | None:
    try:
        leaves = []
        def walk(x):
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    walk(v)
            elif hasattr(x, "shape"):
                leaves.append(x)
        walk(tree)
        if not leaves:
            return None
        return int(sum(np.prod(np.asarray(x).shape) for x in leaves))
    except Exception:
        return None


def summarize_msgpack(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "restore_ok": False,
        "top_keys": [],
        "params_count": None,
        "ema_params_count": None,
    }
    try:
        obj = serialization.msgpack_restore(path.read_bytes())
        out["restore_ok"] = True
        if isinstance(obj, dict):
            out["top_keys"] = sorted(str(k) for k in obj.keys())[:100]
            if "params" in obj:
                out["params_count"] = count_tree(obj["params"])
            if "ema_params" in obj:
                out["ema_params_count"] = count_tree(obj["ema_params"])
            if out["params_count"] is None and out["ema_params_count"] is None:
                # Some checkpoints are directly a parameter pytree.
                direct = count_tree(obj)
                if direct is not None:
                    out["direct_tree_count"] = direct
        else:
            out["object_type"] = type(obj).__name__
            out["direct_tree_count"] = count_tree(obj)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list, tuple)):
                out.update(flatten_json(v, key))
            else:
                out[key] = v
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten_json(v, f"{prefix}[{i}]"))
    return out


def nearby_metadata(path: Path) -> list[dict[str, Any]]:
    candidates = []
    seen = set()
    for p in (
        path.with_suffix(".json"),
        path.parent / "summary.json",
        path.parent / "config.json",
        path.parent / "metadata.json",
        path.parent / "best.json",
    ):
        if p.is_file() and p not in seen:
            seen.add(p)
            candidates.append(p)

    # Also inspect a few JSON files in the immediate checkpoint directory.
    try:
        for p in sorted(path.parent.glob("*.json"))[:20]:
            if p not in seen:
                seen.add(p)
                candidates.append(p)
    except Exception:
        pass

    out = []
    for p in candidates:
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            flat = flatten_json(obj)
            interesting = {}
            for k, v in flat.items():
                lk = k.lower()
                if any(term in lk for term in INTERESTING_META_KEYS):
                    interesting[k] = v
            out.append({
                "path": str(p),
                "interesting": interesting,
                "raw_excerpt": json.dumps(obj, default=str)[:30000],
            })
        except Exception as exc:
            out.append({"path": str(p), "error": f"{type(exc).__name__}: {exc}"})
    return out


def score_text(text: str) -> tuple[int, list[str]]:
    low = text.lower()
    score = 0
    hits = []
    for keyword, weight in KEYWORDS.items():
        if keyword in low:
            score += weight
            hits.append(keyword)
    # Strong preference for evidence of both model family and T64.
    has_m4 = any(k in low for k in ("m4g_h4", "m4g-h4", "m4_geom_h4", "geom_h4"))
    has_t64 = any(k in low for k in ("t64", "64f", "64 frames", '"frames": 64', "frames=64", '"num_frames": 64'))
    if has_m4 and has_t64:
        score += 20
        hits.append("M4+T64_COMBO")
    return score, hits


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    meta = nearby_metadata(path)
    msg = summarize_msgpack(path)
    combined = str(path) + "\n" + json.dumps(meta, default=str) + "\n" + json.dumps(msg, default=str)
    score, hits = score_text(combined)

    low = combined.lower()
    flags = []
    if "ema_params" in low or "checkpoint_contains_ema" in low:
        flags.append("EMA_PRESENT_OR_REFERENCED")
    if any(x in low for x in ("t64", "64f", "64 frames", '"frames": 64', "frames=64", '"num_frames": 64')):
        flags.append("T64_REFERENCED")
    if "xsub" in low:
        flags.append("XSUB_REFERENCED")
    if "xset" in low:
        flags.append("XSET_REFERENCED")
    if "jointfirst" in low or "joint_first" in low:
        flags.append("JOINTFIRST_REFERENCED")
    if "wide128" in low or "wide_128" in low:
        flags.append("WIDE128_REFERENCED")

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "score": score,
        "hits": hits,
        "flags": flags,
        "msgpack": msg,
        "metadata": meta,
    }


def find_files(roots: list[str], pattern: str) -> list[Path]:
    found = []
    seen = set()
    for root_s in roots:
        root = Path(root_s)
        if not root.exists():
            continue
        try:
            for p in root.rglob(pattern):
                try:
                    rp = p.resolve()
                except Exception:
                    rp = p
                if p.is_file() and str(rp) not in seen:
                    seen.add(str(rp))
                    found.append(p)
        except Exception as exc:
            print(f"WARN scan {root}: {type(exc).__name__}: {exc}", flush=True)
    return found


def source_recovery(repo: Path) -> dict[str, Any]:
    wanted = [
        "nestsar_m4_geom_h4_jointfirst_wide128_v4.py",
        "nestsar_m4_regmask_ema_v3_safe.py",
        "patch_sms_exact_guard_v5.py",
        "nestsar_m4_geom_h4.py",
        "nestsar_sms_s1c_v2.py",
    ]
    result = {}
    for name in wanted:
        p = repo / name
        result[name] = {
            "exists": p.is_file(),
            "path": str(p),
            "sha256": sha256(p) if p.is_file() else None,
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["/kaggle/input", "/kaggle/working"])
    ap.add_argument("--repo", default="/kaggle/working/NestSAR")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default="/kaggle/working/m4g_h4_t64_teacher_recovery_v2.json")
    args = ap.parse_args()

    print("=" * 120)
    print("NESTSAR M4G-H4 T64 TEACHER — RECOVERY / PROVENANCE AUDIT V2")
    print("=" * 120)
    print("READ_ONLY=TRUE")
    print("ROOTS=", args.roots)

    sources = source_recovery(Path(args.repo))
    print("\nHISTORICAL SOURCE RECOVERY")
    print("-" * 120)
    for name, info in sources.items():
        print(f"{name:52s} {'FOUND' if info['exists'] else 'MISSING'}")

    checkpoints = find_files(args.roots, "*.msgpack")
    print(f"\nCHECKPOINTS FOUND: {len(checkpoints)}")

    results = []
    for i, p in enumerate(checkpoints, 1):
        try:
            r = inspect_checkpoint(p)
            results.append(r)
        except Exception as exc:
            results.append({
                "path": str(p),
                "score": -1,
                "error": f"{type(exc).__name__}: {exc}",
            })
        if i % 25 == 0:
            print(f"inspected {i}/{len(checkpoints)}", flush=True)

    results.sort(key=lambda x: (x.get("score", -1), x.get("size_bytes", 0)), reverse=True)

    print("\nTOP CHECKPOINT CANDIDATES")
    print("=" * 120)
    for rank, r in enumerate(results[:args.top], 1):
        print(f"#{rank:02d} score={r.get('score')} path={r.get('path')}")
        print(f"    flags={r.get('flags', [])}")
        print(f"    hits={r.get('hits', [])}")
        msg = r.get("msgpack", {})
        print(
            "    params="
            f"{msg.get('params_count')} ema_params={msg.get('ema_params_count')} "
            f"direct={msg.get('direct_tree_count')} restore_ok={msg.get('restore_ok')}"
        )
        metas = r.get("metadata", [])
        for m in metas[:3]:
            interesting = m.get("interesting", {})
            if interesting:
                preview = json.dumps(interesting, default=str)[:1200]
                print(f"    meta={m.get('path')} -> {preview}")
        print()

    payload = {
        "audit": "M4G-H4 T64 teacher recovery v2",
        "roots": args.roots,
        "historical_sources": sources,
        "checkpoint_count": len(checkpoints),
        "candidates": results,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("=" * 120)
    print("AUDIT_V2_COMPLETE")
    print("REPORT:", out)
    print("=" * 120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

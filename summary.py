"""Summary table of all benchmark JSONs in a folder: best latency per runtime, best overall, and accuracy.

  python3 summary.py resultados
Accuracy = NTU-60 XSub top-1 (single model): ours (3 seeds) for NestSAR/SGN, SkelAct 5-seed reproductions for
Ta-CNN / HCN-Baseline / Two-Stream CNN. NestSAR variants are from the 38-epoch ablation (not the final model).
"""
import glob
import json
import sys
from pathlib import Path

ACC = {"nestsar": "85.55", "sgn40": "83.22", "sgn_full": "87.43", "tacnn": "88.53", "hcnb": "87.67", "tscnn": "83.89",
       "nestsar_tattn": "84.30*", "nestsar_tconv": "83.39*", "nestsar_notemp": "81.87*", "ctrgcn": "-",
       "nestsar_u4": "85.55", "nestsar_unrolled": "85.55", "nestsar_split": "85.55", "nestsar_split_u4": "85.55"}
NAMES = {"nestsar": "NestSAR", "sgn40": "SGN-40 (mismo costo)", "sgn_full": "SGN completo", "tacnn": "Ta-CNN",
         "hcnb": "HCN-Baseline", "tscnn": "Two-Stream CNN", "nestsar_tattn": "NestSAR-TATTN",
         "nestsar_tconv": "NestSAR-TCONV", "nestsar_notemp": "NestSAR-NOTEMP", "ctrgcn": "CTR-GCN",
         "nestsar_u4": "NestSAR [unroll 4]", "nestsar_unrolled": "NestSAR [unroll todo]",
         "nestsar_split": "NestSAR [GRU separada]", "nestsar_split_u4": "NestSAR [separada+u4]"}


def main(folder):
    rows, host = {}, None
    for f in sorted(glob.glob(str(Path(folder) / "bench_*.json"))):
        r = json.load(open(f)); host = host or r.get("host")
        rt = "XLA" if Path(f).name.startswith("bench_xla") else "ORT"
        for n, m in r["models"].items():
            d = rows.setdefault(n, {"mflops": m["strict_mflops"], "ORT": None, "XLA": None, "rss": 0})
            d[rt] = m["median_ms"] if d[rt] is None else min(d[rt], m["median_ms"])
            d["rss"] = max(d["rss"], m.get("peak_rss_mb", 0))
    best = lambda d: min(v for v in (d["ORT"], d["XLA"]) if v is not None)
    fmt = lambda v: f"{v:8.2f}" if v is not None else "       -"
    lines = [f"Equipo: {host}",
             f"{'Modelo':24s} {'MFLOPs':>8s} {'ORT ms':>8s} {'XLA ms':>8s} {'MEJOR ms':>9s} {'clips/s':>8s} {'NTU-60 XSub':>12s}",
             "-" * 86]
    for n, d in sorted(rows.items(), key=lambda kv: best(kv[1])):
        b = best(d)
        lines.append(f"{NAMES.get(n, n):24s} {d['mflops']:8.1f} {fmt(d['ORT'])} {fmt(d['XLA'])} {b:9.2f} {1e3 / b:8.1f} {ACC.get(n, '-'):>12s}")
    lines += ["", "* variantes de la ablación (38 épocas), no el modelo final.",
              "MEJOR = el runtime más rápido para cada modelo (mínimo entre ONNX Runtime y XLA, 1 hilo o todos).",
              "NestSAR [..] = el mismo modelo y los mismos pesos con otro orden de ejecución (salida idéntica)."]
    txt = "\n".join(lines); print("\n" + txt)
    (Path(folder) / "resumen.md").write_text("```\n" + txt + "\n```\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "resultados")

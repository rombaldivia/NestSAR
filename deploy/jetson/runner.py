"""CUDA-only model runner for Jetson Nano and Orin Nano; Python 3.6+."""
import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Debe ser un entero positivo")
    return number


def load_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("manifest.json debe contener modelos")
    for name, info in manifest.items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Nombre de modelo no valido: " + name)
        shape = info.get("input_shape")
        if not isinstance(shape, list) or not shape or shape[0] != 1 or not all(isinstance(x, int) and x > 0 for x in shape):
            raise ValueError("Se requiere input_shape estatico, batch 1: " + name)
    return manifest


def verify(engine, root, name, info, precision):
    reference = root / info.get("reference_file", name + ".reference.npz")
    if not reference.is_file():
        raise ValueError("Falta la referencia de {}: prepara el modelo con prepare_models.py".format(name))
    max_error = 0.0
    count = 0
    atol, rtol = (1e-3, 1e-3) if precision == "fp32" else (3e-2, 2e-2)
    with np.load(str(reference), allow_pickle=False) as data:
        if len(data["x"]) != len(data["y"]) or len(data["x"]) < 1:
            raise ValueError("Referencias vacias o incompletas")
        for x, expected in zip(data["x"], data["y"]):
            actual = engine.run(x)
            if not np.isfinite(actual).all():
                raise ValueError("La salida CUDA contiene NaN o Inf")
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            count += 1
    return dict(cases=count, max_abs_diff=max_error, atol=atol, rtol=rtol)


def worker(args):
    from trt_engine import Engine
    manifest = load_manifest(args.models)
    name = args.only[0]
    info = manifest[name]
    path = args.models / (name + ".onnx")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != info.get("onnx_sha256"):
        raise ValueError("El ONNX no coincide con el manifest: " + name)
    if args.command == "predict" and info.get("trained") is not True:
        raise ValueError("{} tiene pesos sin entrenar; se permite benchmark, pero no prediccion".format(name))
    if args.command == "predict" and args.input is None:
        raise ValueError("predict requiere --input /data/clip.npy")
    engine = Engine(path, args.cache, args.precision, args.workspace_mb)
    try:
        print("{} | {} | TensorRT {} | {}".format(name, engine.info["gpu"], engine.info["tensorrt"],
                                                  args.precision), flush=True)
        validation = verify(engine, args.models, name, info, args.precision)
        print("  CUDA verificada: {} casos, max|diff|={:.3g}".format(validation["cases"], validation["max_abs_diff"]), flush=True)
        result = dict(model=name, hardware=engine.info, trained=info.get("trained", False),
                      input_shape=info["input_shape"], source_commit=info.get("source_commit"),
                      onnx_sha256=digest, validation=validation, cache_hit=engine.cache_hit,
                      build_or_load_s=engine.build_or_load_s, strict_mflops=info.get("strict_mflops"))
        if args.command == "predict":
            x = np.load(str(args.input), allow_pickle=False)
            y = engine.run(x)
            np.save(str(args.output / (name + "_logits.npy")), y, allow_pickle=False)
            result.update(class_index=int(np.argmax(y[0])), logits_file=name + "_logits.npy")
            print("  class_index={} (indice base 0)".format(result["class_index"]), flush=True)
        else:
            x = np.random.RandomState(0).normal(0, .3, info["input_shape"]).astype(np.float32)
            for _ in range(args.warmup):
                engine.run(x)
            samples = []
            for _ in range(args.runs):
                start = time.perf_counter()
                engine.run(x)
                samples.append((time.perf_counter() - start) * 1000)
            median = float(np.median(samples))
            result.update(median_ms=median, p90_ms=float(np.percentile(samples, 90)),
                          mean_ms=float(np.mean(samples)), clips_per_s=1000 / median,
                          runs=args.runs, warmup=args.warmup,
                          timing_scope="host input -> H2D -> TensorRT CUDA -> sync -> D2H -> host output",
                          peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
            print("  {:.2f} ms/clip | {:.1f} clips/s | pesos {}".format(
                median, 1000 / median, "entrenados" if result["trained"] else "sin entrenar"), flush=True)
        destination = args.output / (name + ".json")
        destination.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        engine.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="NestSAR y baselines en GPU CUDA mediante TensorRT; sin CPU fallback")
    ap.add_argument("command", nargs="?", default="benchmark", choices=["benchmark", "predict", "check", "list"])
    ap.add_argument("--models", type=Path, default=Path("/opt/models"))
    ap.add_argument("--cache", type=Path, default=Path("/cache"))
    ap.add_argument("--output", type=Path, default=Path("/results"))
    ap.add_argument("--only", nargs="+", help="modelos; por defecto todos")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--workspace-mb", type=positive, default=128)
    ap.add_argument("--runs", type=positive, default=100)
    ap.add_argument("--warmup", type=positive, default=10)
    ap.add_argument("--input", type=Path, help="tensor .npy FP32 preprocesado para el modelo")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.command == "check":
        import tensorrt as trt
        from cuda_runtime import Cuda
        cuda = Cuda()
        print(json.dumps(dict(cuda.info, tensorrt=trt.__version__, backend="TensorRT/CUDA"), indent=2))
        return
    manifest = load_manifest(args.models)
    names = args.only or list(manifest)
    unknown = set(names) - set(manifest)
    if unknown:
        ap.error("Modelos desconocidos: {}. Disponibles: {}".format(sorted(unknown), sorted(manifest)))
    if args.command == "list":
        print(json.dumps({n: dict(manifest[n], backend="TensorRT/CUDA") for n in names}, indent=2))
        return
    if args.command == "predict" and len(names) != 1:
        ap.error("predict requiere --only con exactamente un modelo")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    # A fresh process per model releases all CUDA/TensorRT allocations between
    # models. This matters on the original Nano with 2 GB shared RAM.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    args.output = args.output / stamp
    args.output.mkdir()
    failures, results = {}, []
    for name in names:
        command = [sys.executable, str(Path(__file__).resolve()), args.command, "--worker", "--only", name,
                   "--models", str(args.models), "--cache", str(args.cache), "--output", str(args.output),
                   "--precision", args.precision, "--workspace-mb", str(args.workspace_mb),
                   "--runs", str(args.runs), "--warmup", str(args.warmup)]
        if args.input:
            command += ["--input", str(args.input)]
        completed = subprocess.run(command)
        if completed.returncode:
            failures[name] = completed.returncode
        else:
            results.append(json.loads((args.output / (name + ".json")).read_text()))
    summary = dict(results=results, failures=failures, backend="TensorRT/CUDA",
                   note="Synthetic preprocessed inputs. Latency is not accuracy or camera-to-action latency.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Resultados: " + str(args.output), flush=True)
    if failures:
        raise SystemExit("Fallo CUDA/TensorRT en: {}. No se ejecuto en CPU.".format(", ".join(failures)))


if __name__ == "__main__":
    main()

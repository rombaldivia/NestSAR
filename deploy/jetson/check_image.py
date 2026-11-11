"""Image checks that run on native ARM64 without a Jetson GPU/driver.

These check Python/NumPy, packaged bindings/libraries, model/reference hashes
and I/O. Actual TensorRT import, engine building and CUDA remain device checks.
"""
import argparse
import ctypes.util
import glob
import hashlib
import importlib.util
import json
import platform
import struct
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--models", type=Path, default=Path("/opt/models"))
    args = ap.parse_args()
    actual = "{}.{}".format(*sys.version_info[:2])
    if actual != args.python or platform.machine() != "aarch64":
        raise RuntimeError("Wrong Python/platform: {} {}".format(actual, platform.machine()))
    if np.__version__ != "1.24.4":
        raise RuntimeError("Unexpected NumPy version: " + np.__version__)
    np.testing.assert_array_equal(np.ones((2, 2), np.float32) @ np.ones((2, 2), np.float32), np.full((2, 2), 2))
    spec = importlib.util.find_spec("tensorrt")
    if spec is None or not spec.origin:
        raise RuntimeError("The NVIDIA image has no TensorRT Python bindings")
    extensions = list(Path(spec.origin).parent.rglob("*.so"))
    if not extensions:
        raise RuntimeError("The TensorRT Python extension is missing")
    for path in extensions:
        header = path.read_bytes()[:20]
        if header[:4] != b"\x7fELF" or struct.unpack_from("<H", header, 18)[0] != 183:
            raise RuntimeError("Non-AArch64 TensorRT extension: " + str(path))
    libraries = {n: ctypes.util.find_library(n) for n in ("nvinfer", "nvonnxparser", "nvinfer_plugin")}
    if not all(libraries.values()):
        raise RuntimeError("Missing TensorRT libraries: " + str(libraries))
    if not glob.glob("/usr/local/cuda*/targets/aarch64-linux/lib/libcudart.so*"):
        raise RuntimeError("CUDA runtime library is missing")
    manifest = json.loads((args.models / "manifest.json").read_text())
    if not manifest:
        raise RuntimeError("No prepared models")
    for name, info in manifest.items():
        path = args.models / (name + ".onnx")
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["onnx_sha256"]:
            raise RuntimeError("Model checksum mismatch: " + name)
        with np.load(str(args.models / info["reference_file"]), allow_pickle=False) as data:
            if data["x"].shape[1:] != tuple(info["input_shape"]) or data["x"].dtype != np.float32:
                raise RuntimeError("Invalid references: " + name)
    print(json.dumps(dict(check="image-files-and-numpy", python=actual, platform=platform.machine(),
                          numpy=np.__version__, tensorrt_bindings=spec.origin, libraries=libraries,
                          models=len(manifest), cuda_execution_tested=False), indent=2))


if __name__ == "__main__":
    main()

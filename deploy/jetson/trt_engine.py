"""TensorRT 8.2 and 10 engine construction/inference; Python 3.6 compatible."""
import ctypes as C
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from cuda_runtime import Cuda


class Engine:
    def __init__(self, path, cache, precision="fp32", workspace_mb=128, device=0):
        import tensorrt as trt
        self.trt = trt
        self.cuda = Cuda(device)
        self.buffers = []
        self.stream = None
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.info = dict(self.cuda.info, tensorrt=trt.__version__, precision=precision,
                         workspace_mb=workspace_mb, backend="TensorRT/CUDA", platform=os.uname().machine)
        fingerprint = dict(self.info, onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           engine_recipe=1)
        key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        cache.mkdir(parents=True, exist_ok=True)
        engine_path = cache / (path.stem + "-" + key + ".engine")
        start = time.perf_counter()
        self.cache_hit = engine_path.exists()
        if self.cache_hit:
            self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
            if self.engine is None:
                raise RuntimeError("Cache TensorRT invalida: elimina {} y repite".format(engine_path))
        else:
            builder = trt.Builder(self.logger)
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(flags)
            parser = trt.OnnxParser(network, self.logger)
            if not parser.parse(path.read_bytes()):
                errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise RuntimeError("TensorRT no puede importar {}:\n{}".format(path.name, errors))
            config = builder.create_builder_config()
            workspace = workspace_mb * 1024 * 1024
            if hasattr(config, "set_memory_pool_limit"):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
            else:
                config.max_workspace_size = workspace
            if hasattr(trt.BuilderFlag, "TF32"):
                config.clear_flag(trt.BuilderFlag.TF32)
            if precision == "fp16":
                if not builder.platform_has_fast_fp16:
                    raise RuntimeError("La GPU no anuncia FP16 rapido; usa --precision fp32")
                config.set_flag(trt.BuilderFlag.FP16)
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                raise RuntimeError("No se pudo construir el motor CUDA: revisa el log de TensorRT")
            self.engine = self.runtime.deserialize_cuda_engine(plan)
            if self.engine is None:
                raise RuntimeError("TensorRT genero un motor que no puede cargar")
            temp = engine_path.with_suffix(".{}.tmp".format(os.getpid()))
            temp.write_bytes(bytes(plan))
            temp.replace(engine_path)
        self.build_or_load_s = time.perf_counter() - start
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("No hay memoria para el contexto CUDA")
        self.v3 = hasattr(self.engine, "num_io_tensors")
        count = self.engine.num_io_tensors if self.v3 else self.engine.num_bindings
        inputs, outputs = [], []
        self.bindings = [0] * count
        for i in range(count):
            if self.v3:
                name = self.engine.get_tensor_name(i)
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = self.engine.get_tensor_dtype(name)
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            else:
                name = self.engine.get_binding_name(i)
                shape = tuple(self.engine.get_binding_shape(i))
                dtype = self.engine.get_binding_dtype(i)
                is_input = self.engine.binding_is_input(i)
            if any(d <= 0 for d in shape) or dtype != trt.float32:
                raise RuntimeError("Se requiere I/O FP32 con forma estatica: {} {} {}".format(name, shape, dtype))
            host = np.empty(shape, np.float32)
            ptr = self.cuda.malloc(host.nbytes)
            self.buffers.append(ptr)
            self.bindings[i] = ptr.value
            if self.v3 and not self.context.set_tensor_address(name, ptr.value):
                raise RuntimeError("No se pudo asignar el tensor " + name)
            (inputs if is_input else outputs).append((name, host, ptr))
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("Se requiere una entrada y una salida logits")
        self.input_name, self.input, self.input_ptr = inputs[0]
        self.output_name, self.output, self.output_ptr = outputs[0]
        self.stream = self.cuda.stream()

    def run(self, x):
        if x.dtype != np.float32 or x.shape != self.input.shape:
            raise ValueError("Entrada requerida: FP32 {}; recibida {} {}".format(self.input.shape, x.dtype, x.shape))
        x = np.ascontiguousarray(x)
        self.cuda.check(self.cuda.lib.cudaMemcpy(self.input_ptr, C.c_void_p(x.ctypes.data), x.nbytes, 1))
        if self.v3:
            ok = self.context.execute_async_v3(self.stream.value)
        else:
            ok = self.context.execute_async_v2(self.bindings, self.stream.value)
        if not ok:
            raise RuntimeError("Fallo la inferencia CUDA")
        self.cuda.check(self.cuda.lib.cudaStreamSynchronize(self.stream))
        self.cuda.check(self.cuda.lib.cudaMemcpy(C.c_void_p(self.output.ctypes.data), self.output_ptr,
                                               self.output.nbytes, 2))
        return self.output.copy()

    def close(self):
        if self.stream is not None:
            self.cuda.check(self.cuda.lib.cudaStreamSynchronize(self.stream))
        # Drop context before freeing the memory it references.
        self.context = None
        for ptr in self.buffers:
            self.cuda.check(self.cuda.lib.cudaFree(ptr))
        self.buffers = []
        if self.stream is not None:
            self.cuda.check(self.cuda.lib.cudaStreamDestroy(self.stream))
            self.stream = None
        self.engine = None
        self.runtime = None

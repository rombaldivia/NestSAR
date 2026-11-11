"""Exercise both API paths with simulated CUDA memory, without claiming GPU execution."""
import ctypes as C
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trt_engine


class SimulatedCuda:
    gpu = "test-gpu"

    def __init__(self, device):
        self.info = dict(device=device, gpu=self.gpu, cuda_runtime=12060, cuda_driver=12060,
                         compute_capability=[8, 7])
        self.owned = {}
        self.lib = self

    def malloc(self, size):
        value = C.create_string_buffer(size)
        pointer = C.cast(value, C.c_void_p)
        self.owned[pointer.value] = value
        return pointer

    def stream(self):
        return C.c_void_p(17)

    def check(self, code):
        if code:
            raise RuntimeError("simulated CUDA error")

    def cudaMemcpy(self, destination, source, size, kind):
        C.memmove(destination, source, size)
        return 0

    def cudaStreamSynchronize(self, stream):
        assert stream.value == 17
        return 0

    def cudaFree(self, pointer):
        del self.owned[pointer.value]
        return 0

    def cudaStreamDestroy(self, stream):
        assert stream.value == 17
        return 0


def fake_trt(major):
    calls = []

    class Context:
        def __init__(self):
            self.addresses = {}
            self.success = True

        def set_tensor_address(self, name, pointer):
            self.addresses[name] = pointer
            return True

        def calculate(self, input_pointer, output_pointer):
            x = np.ctypeslib.as_array((C.c_float * 2).from_address(input_pointer))
            y = np.ctypeslib.as_array((C.c_float * 2).from_address(output_pointer))
            y[:] = x * 2 + 1

        def execute_async_v2(self, bindings, stream):
            assert major == 8 and stream == 17
            calls.append("v2")
            self.calculate(*bindings)
            return self.success

        def execute_async_v3(self, stream):
            assert major == 10 and stream == 17
            calls.append("v3")
            self.calculate(self.addresses["x"], self.addresses["y"])
            return self.success

    class Engine8:
        num_bindings = 2
        def create_execution_context(self): return Context()
        def get_binding_name(self, i): return ["x", "y"][i]
        def get_binding_shape(self, i): return (1, 2)
        def get_binding_dtype(self, i): return "float32"
        def binding_is_input(self, i): return i == 0

    class Engine10:
        num_io_tensors = 2
        def create_execution_context(self): return Context()
        def get_tensor_name(self, i): return ["x", "y"][i]
        def get_tensor_shape(self, name): return (1, 2)
        def get_tensor_dtype(self, name): return "float32"
        def get_tensor_mode(self, name): return "input" if name == "x" else "output"

    class Runtime:
        def __init__(self, logger): pass
        def deserialize_cuda_engine(self, plan):
            assert bytes(plan) == b"valid-plan"
            return Engine8() if major == 8 else Engine10()

    class Builder:
        platform_has_fast_fp16 = True
        def __init__(self, logger): calls.append("build")
        def create_network(self, flags): return object()
        def create_builder_config(self):
            config = SimpleNamespace(clear_flag=lambda flag: None, set_flag=lambda flag: None)
            if major == 10:
                config.set_memory_pool_limit = lambda kind, size: calls.append(("workspace", size))
            return config
        def build_serialized_network(self, network, config):
            if major == 8:
                assert config.max_workspace_size == 128 * 1024 * 1024
            return b"valid-plan"

    class Parser:
        def __init__(self, network, logger): pass
        def parse(self, data): return data == b"valid-onnx"

    return SimpleNamespace(__version__="{}.2".format(major),
        Logger=SimpleNamespace(WARNING=1, __call__=lambda _: None), Runtime=Runtime,
        Builder=Builder, OnnxParser=Parser, float32="float32",
        init_libnvinfer_plugins=lambda logger, name: None,
        NetworkDefinitionCreationFlag=SimpleNamespace(EXPLICIT_BATCH=0),
        BuilderFlag=SimpleNamespace(TF32=1, FP16=2),
        MemoryPoolType=SimpleNamespace(WORKSPACE=0), TensorIOMode=SimpleNamespace(INPUT="input")), calls


@pytest.mark.parametrize("major", [8, 10])
def test_binding_paths_transfers_cache_and_failure(monkeypatch, tmp_path, major):
    trt, calls = fake_trt(major)
    class Logger:
        WARNING = 1
        def __init__(self, level): pass
    trt.Logger = Logger
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(trt_engine, "Cuda", SimulatedCuda)
    path = tmp_path / "model.onnx"
    path.write_bytes(b"valid-onnx")
    engine = trt_engine.Engine(path, tmp_path / "cache")
    assert not engine.cache_hit
    np.testing.assert_array_equal(engine.run(np.array([[2, -1]], np.float32)), [[5, -1]])
    assert ("v2" if major == 8 else "v3") in calls
    engine.context.success = False
    with pytest.raises(RuntimeError, match="inferencia CUDA"):
        engine.run(np.zeros((1, 2), np.float32))
    cuda = engine.cuda
    engine.close()
    assert not cuda.owned
    engine = trt_engine.Engine(path, tmp_path / "cache")
    assert engine.cache_hit and calls.count("build") == 1
    engine.close()
    monkeypatch.setattr(SimulatedCuda, "gpu", "a-different-gpu")
    engine = trt_engine.Engine(path, tmp_path / "cache")
    assert not engine.cache_hit and calls.count("build") == 2
    engine.close()

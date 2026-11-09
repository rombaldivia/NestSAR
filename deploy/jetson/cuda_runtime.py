"""Small CUDA runtime binding, compatible with Python 3.6 and CUDA 10/12.

No PyCUDA compiler, PyTorch, JAX, CPU inference, or extra CUDA installation.
"""
import ctypes as C
import ctypes.util
import glob


class Cuda:
    def __init__(self, device=0):
        candidates = [ctypes.util.find_library("cudart"), "libcudart.so"]
        candidates += glob.glob("/usr/local/cuda*/targets/aarch64-linux/lib/libcudart.so*")
        candidates += glob.glob("/usr/local/cuda*/lib64/libcudart.so*")
        self.lib = None
        for candidate in candidates:
            if candidate:
                try:
                    self.lib = C.CDLL(candidate)
                    break
                except OSError:
                    pass
        if self.lib is None:
            raise RuntimeError("No se encuentra libcudart: ejecuta con --runtime nvidia y JetPack compatible")
        self.lib.cudaGetErrorString.restype = C.c_char_p
        self.lib.cudaGetErrorString.argtypes = [C.c_int]
        signatures = {
            "cudaGetDeviceCount": [C.POINTER(C.c_int)], "cudaSetDevice": [C.c_int],
            "cudaRuntimeGetVersion": [C.POINTER(C.c_int)],
            "cudaDriverGetVersion": [C.POINTER(C.c_int)],
            "cudaMalloc": [C.POINTER(C.c_void_p), C.c_size_t],
            "cudaFree": [C.c_void_p],
            "cudaMemcpy": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int],
            "cudaStreamCreate": [C.POINTER(C.c_void_p)],
            "cudaStreamSynchronize": [C.c_void_p], "cudaStreamDestroy": [C.c_void_p],
        }
        for name, args in signatures.items():
            getattr(self.lib, name).argtypes = args
            getattr(self.lib, name).restype = C.c_int
        count = C.c_int()
        self.check(self.lib.cudaGetDeviceCount(C.byref(count)))
        if count.value == 0 or not 0 <= device < count.value:
            raise RuntimeError("CUDA no expone la GPU solicitada; no se permite ejecutar en CPU")
        self.check(self.lib.cudaSetDevice(device))
        runtime, driver = C.c_int(), C.c_int()
        self.check(self.lib.cudaRuntimeGetVersion(C.byref(runtime)))
        self.check(self.lib.cudaDriverGetVersion(C.byref(driver)))
        gpu_name = C.create_string_buffer(256)
        cuda_driver = C.CDLL("libcuda.so.1")
        cuda_driver.cuInit.argtypes = [C.c_uint]
        cuda_driver.cuDeviceGetName.argtypes = [C.c_char_p, C.c_int, C.c_int]
        cuda_driver.cuDeviceGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int]
        if cuda_driver.cuInit(0) != 0 or cuda_driver.cuDeviceGetName(gpu_name, len(gpu_name), device) != 0:
            raise RuntimeError("El controlador CUDA no permite consultar la GPU")
        major, minor = C.c_int(), C.c_int()
        if cuda_driver.cuDeviceGetAttribute(C.byref(major), 75, device) != 0:
            raise RuntimeError("No se puede consultar compute capability")
        if cuda_driver.cuDeviceGetAttribute(C.byref(minor), 76, device) != 0:
            raise RuntimeError("No se puede consultar compute capability")
        self.info = dict(device=device, gpu=gpu_name.value.decode(), cuda_runtime=runtime.value,
                         cuda_driver=driver.value, compute_capability=[major.value, minor.value])

    def check(self, code):
        if code:
            raise RuntimeError("CUDA {}: {}".format(code, self.lib.cudaGetErrorString(code).decode()))

    def malloc(self, size):
        ptr = C.c_void_p()
        self.check(self.lib.cudaMalloc(C.byref(ptr), size))
        return ptr

    def stream(self):
        ptr = C.c_void_p()
        self.check(self.lib.cudaStreamCreate(C.byref(ptr)))
        return ptr

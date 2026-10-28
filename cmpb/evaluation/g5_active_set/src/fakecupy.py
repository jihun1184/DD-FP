"""fakecupy.py -- numpy-backed stand-in for the subset of CuPy used by the worklist files (A1 test tooling).

Purpose: run the *unmodified* frozen ``boutry_bucket_gpu`` and the new active-frontier driver on a CPU,
with the CUDA kernel source text compiled through ``shim/cuda_shim.h`` and executed by real threads.
Evidence tier produced by this module is "SHIM" (kernel source text + host driver executed on CPU
threads); it is NOT GPU execution and does not exercise nvcc, warps, or the GPU memory model.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

SHIM_DIR = Path(__file__).resolve().parent.parent / "shim"
CACHE = Path(os.environ.get("SHIM_CACHE", tempfile.gettempdir())) / "af_shim_cache"

_KERNEL_RE = re.compile(r"__global__\s+void\s+(\w+)\s*\((.*?)\)\s*\{", re.S)


def _strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return re.sub(r"//[^\n]*", "", code)


def parse_kernels(code: str):
    out = {}
    for m in _KERNEL_RE.finditer(_strip_comments(code)):
        params = []
        for p in m.group(2).split(","):
            toks = p.replace("*", " * ").split()
            name = toks[-1]
            typ = " ".join(toks[:-1]).replace(" *", "*")
            params.append((typ, name, typ.endswith("*")))
        out[m.group(1)] = params
    return out


_DTYPE = {"float": np.float32, "int": np.int32, "unsigned int": np.uint32}


class FArray(np.ndarray):
    """numpy array that quacks like a cupy array for the calls the worklist code makes."""

    def get(self):
        return np.array(self, copy=True).view(np.ndarray)


def _wrap(a) -> FArray:
    return np.ascontiguousarray(a).view(FArray)


class _Event:
    def __init__(self):
        self.t = 0.0

    def record(self, stream=None):
        self.t = time.perf_counter()

    def synchronize(self):
        pass


class _Stream:
    class null:
        @staticmethod
        def synchronize():
            pass


class _Cuda:
    Stream = _Stream
    Event = _Event

    @staticmethod
    def get_elapsed_time(a, b):
        return (b.t - a.t) * 1e3


class RawModule:
    def __init__(self, code, options=(), *, sanitize=False, backend=None):
        self.code = code
        self.kernels = parse_kernels(code)
        wrappers = []
        for name, params in self.kernels.items():
            casts = ", ".join(f"*({typ}*)a[{i}]" for i, (typ, _, _) in enumerate(params))
            wrappers.append(
                f'extern "C" void shim_{name}(long grid, long block, void** a, unsigned long seed) {{\n'
                f"  shim_run(grid, block, seed, [&]() {{ {name}({casts}); }});\n}}\n"
            )
        src = '#include "cuda_shim.h"\n' + code + "\n" + "\n".join(wrappers)
        flags = ["-O2", "-std=c++17", "-shared", "-fPIC", "-pthread", "-w", f"-I{SHIM_DIR}"]
        if sanitize:
            flags += ["-fsanitize=thread", "-g"]
        key = hashlib.sha256((src + " ".join(flags) + (SHIM_DIR / "cuda_shim.h").read_text()).encode()).hexdigest()[:24]
        CACHE.mkdir(parents=True, exist_ok=True)
        so = CACHE / f"{key}.so"
        if not so.exists():
            cpp = CACHE / f"{key}.cpp"
            cpp.write_text(src)
            tmp = CACHE / f"{key}.{os.getpid()}.so"
            r = subprocess.run(["g++", *flags, str(cpp), "-o", str(tmp)], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("shim compile failed:\n" + r.stderr[-4000:])
            os.replace(tmp, so)
        self.lib = ctypes.CDLL(str(so))
        self.lib.shim_set_jitter.argtypes = [ctypes.c_double]

    def get_function(self, name):
        return _Kernel(self, name, self.kernels[name])


class _Kernel:
    def __init__(self, mod, name, params):
        self.mod, self.name, self.params = mod, name, params
        self.fn = getattr(mod.lib, f"shim_{name}")
        self.fn.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong]
        self.fn.restype = None

    def __call__(self, grid, block, args):
        fcp = FakeCupy.current
        assert len(args) == len(self.params), f"{self.name}: {len(args)} args for {len(self.params)} params"
        keep, ptrs, named = [], [], {}
        for (typ, pname, is_ptr), a in zip(self.params, args):
            named[pname] = a
            if is_ptr:
                base = typ.replace("const", "").replace("*", "").strip()
                want = _DTYPE[base]
                if not (isinstance(a, np.ndarray) and a.dtype == want and a.flags["C_CONTIGUOUS"] and a.flags["WRITEABLE"]):
                    raise TypeError(f"{self.name}.{pname}: need C-contiguous writeable {np.dtype(want)} array, got {getattr(a, 'dtype', type(a))}")
                v = ctypes.c_void_p(a.ctypes.data)
            elif typ == "int":
                iv = int(a)
                if not (-2**31 <= iv < 2**31):
                    raise OverflowError(f"{self.name}.{pname}={iv} does not fit int32")
                v = ctypes.c_int32(iv)
            elif typ == "float":
                v = ctypes.c_float(float(a))
            else:
                raise TypeError(f"unsupported kernel param type {typ!r}")
            keep.append(v)
            ptrs.append(ctypes.addressof(v))
        arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        g = int(grid[0]); b = int(block[0])
        if g > 0:
            fcp.launch_count += 1
            self.fn(g, b, ctypes.cast(arr, ctypes.c_void_p), ctypes.c_ulong(fcp.seed + fcp.launch_count))
        for h in fcp.hooks:
            h(self.name, named, g, b)


class FakeCupy:
    nan = np.nan
    float32, int32, uint32 = np.float32, np.int32, np.uint32
    cuda = _Cuda
    current = None
    __version__ = "fakecupy-shim"

    def __init__(self, seed=1, threads=4, jitter=0.0, sanitize=False):
        self.seed, self.hooks, self.launch_count, self.sanitize = seed, [], 0, sanitize
        os.environ["SHIM_THREADS"] = str(threads)
        self._jitter = jitter
        FakeCupy.current = self

    def RawModule(self, code, options=()):
        m = RawModule(code, options, sanitize=self.sanitize)
        m.lib.shim_set_jitter(self._jitter)
        return m

    @staticmethod
    def asarray(a, dtype=None):
        return _wrap(np.array(a, dtype=dtype, copy=True))

    @staticmethod
    def ascontiguousarray(a, dtype=None):
        return _wrap(np.ascontiguousarray(a, dtype=dtype))

    @staticmethod
    def asnumpy(a):
        return np.array(a, copy=True).view(np.ndarray)

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return _wrap(np.zeros(shape, dtype=dtype))

    @staticmethod
    def full(shape, fill, dtype=np.float32):
        return _wrap(np.full(shape, fill, dtype=dtype))

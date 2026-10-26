# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal NVRTC + CUDA driver API loader via ctypes (no host compiler, no extra deps).

Uses the NVRTC shipped with torch's CUDA pip wheels (e.g. nvidia-cuda-nvrtc-cu12) and the driver's
libcuda. Kernels run in torch's primary context on torch's current stream and take raw device
pointers of torch tensors -> zero-copy interop. Compiled cubins are cached in
$FAST_MARIO_ENV_CACHE (default: $XDG_CACHE_HOME/fast_mario_env or ~/.cache/fast_mario_env).

Set FAST_MARIO_ENV_NVRTC to the full path of libnvrtc.so to override the NVRTC lookup.
"""
import ctypes
import ctypes.util
import glob
import hashlib
import os

import torch

_nvrtc = None
_cuda = None


def cache_dir():
    d = os.environ.get("FAST_MARIO_ENV_CACHE")
    if not d:
        d = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "fast_mario_env")
    return os.path.join(d, "cuda")


def _nvrtc_candidates():
    env = os.environ.get("FAST_MARIO_ENV_NVRTC")
    if env:
        yield env
    dirs = []
    try:  # torch's CUDA 12 wheels
        import nvidia.cuda_nvrtc as m
        dirs += [os.path.join(p, "lib") for p in m.__path__]
    except ImportError:
        pass
    try:  # newer layouts (nvidia/cu13/lib, ...)
        import nvidia
        for p in nvidia.__path__:
            dirs += glob.glob(os.path.join(p, "*", "lib"))
    except ImportError:
        pass
    dirs.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    for d in dirs:
        yield from sorted(glob.glob(os.path.join(d, "libnvrtc.so*")))
    lib = ctypes.util.find_library("nvrtc")
    if lib:
        yield lib


def _load():
    global _nvrtc, _cuda
    if _cuda is not None:
        return
    err = None
    for c in _nvrtc_candidates():
        try:
            _nvrtc = ctypes.CDLL(c)
            break
        except OSError as e:
            err = e
    if _nvrtc is None:
        raise RuntimeError("fast_mario_env.cuda: libnvrtc not found. Install torch with CUDA wheels "
                           "(they ship nvidia-cuda-nvrtc) or set FAST_MARIO_ENV_NVRTC. " + (str(err) if err else ""))
    _cuda = ctypes.CDLL("libcuda.so.1")


def nvrtc_version():
    _load()
    a, b = ctypes.c_int(), ctypes.c_int()
    _nvrtc.nvrtcVersion(ctypes.byref(a), ctypes.byref(b))
    return a.value, b.value


def _chk_cu(r, what):
    if r != 0:
        s = ctypes.c_char_p()
        _cuda.cuGetErrorString(r, ctypes.byref(s))
        raise RuntimeError(f"{what}: CUDA error {r} {s.value.decode() if s.value else ''}")


def compile_cubin(src, name, opts, arch):
    _load()
    key = hashlib.sha1((src + repr(opts) + arch + repr(nvrtc_version())).encode()).hexdigest()[:16]
    cache = cache_dir()
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{name}_{key}.cubin")
    if os.path.exists(path):
        return open(path, "rb").read(), ""
    prog = ctypes.c_void_p()
    r = _nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), (name + ".cu").encode(), 0, None, None)
    if r != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed ({r})")
    allopts = [f"--gpu-architecture=sm_{arch}"] + list(opts)
    arr = (ctypes.c_char_p * len(allopts))(*[o.encode() for o in allopts])
    r = _nvrtc.nvrtcCompileProgram(prog, len(allopts), arr)
    n = ctypes.c_size_t()
    _nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value)
    _nvrtc.nvrtcGetProgramLog(prog, log)
    log = log.value.decode(errors="replace")
    if r != 0:
        raise RuntimeError("NVRTC compile failed:\n" + log)
    _nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value)
    _nvrtc.nvrtcGetCUBIN(prog, buf)
    _nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    data = buf.raw
    with open(path + ".tmp", "wb") as f:
        f.write(data)
    os.replace(path + ".tmp", path)
    return data, log


class Module:
    def __init__(self, src, name="mod", opts=(), device=0):
        _load()
        torch.cuda.init()
        torch.zeros(1, device=f"cuda:{device}")  # make torch's primary context current
        maj, mi = torch.cuda.get_device_capability(device)
        self.cubin, self.log = compile_cubin(src, name, opts, f"{maj}{mi}")
        ctx = ctypes.c_void_p()
        _chk_cu(_cuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        if not ctx.value:
            dev = ctypes.c_int()
            _chk_cu(_cuda.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")
            _chk_cu(_cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "retain")
            _chk_cu(_cuda.cuCtxSetCurrent(ctx), "setcurrent")
        self.mod = ctypes.c_void_p()
        _chk_cu(_cuda.cuModuleLoadData(ctypes.byref(self.mod), self.cubin), "cuModuleLoadData")
        self._f = {}

    def func(self, name):
        if name not in self._f:
            f = ctypes.c_void_p()
            _chk_cu(_cuda.cuModuleGetFunction(ctypes.byref(f), self.mod, name.encode()), f"get {name}")
            self._f[name] = f
        return self._f[name]

    def attr(self, name, a):
        v = ctypes.c_int()
        _cuda.cuFuncGetAttribute(ctypes.byref(v), a, self.func(name))
        return v.value

    def regs(self, name):
        return self.attr(name, 4)  # CU_FUNC_ATTRIBUTE_NUM_REGS

    def local_bytes(self, name):
        return self.attr(name, 3)  # CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES

    def set_max_smem(self, name, nbytes):
        _chk_cu(_cuda.cuFuncSetAttribute(self.func(name), 8, int(nbytes)), "setattr smem")

    def launch(self, name, grid, block, args, smem=0, stream=None):
        """args: torch tensors (device pointers), python ints (int64), floats (double) or ctypes values."""
        f = self.func(name)
        cargs = []
        for a in args:
            if isinstance(a, torch.Tensor):
                cargs.append(ctypes.c_uint64(a.data_ptr()))
            elif isinstance(a, bool):
                cargs.append(ctypes.c_int32(int(a)))
            elif isinstance(a, int):
                cargs.append(ctypes.c_int64(a))
            elif isinstance(a, float):
                cargs.append(ctypes.c_double(a))
            else:
                cargs.append(a)
        pp = (ctypes.c_void_p * len(cargs))(*[ctypes.cast(ctypes.byref(c), ctypes.c_void_p) for c in cargs])
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        g = grid if isinstance(grid, tuple) else (grid, 1, 1)
        b = block if isinstance(block, tuple) else (block, 1, 1)
        _chk_cu(_cuda.cuLaunchKernel(f, g[0], g[1], g[2], b[0], b[1], b[2], int(smem), ctypes.c_void_p(stream),
                                     pp, None), f"launch {name}")

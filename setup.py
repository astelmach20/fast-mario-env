"""Build the native emulator (csrc/fastmario.cpp) as a plain shared library loaded through ctypes.

No Python C-API is used, so the library is compiled with a direct compiler call:

* compiler: $CXX if set, else the first of c++ / g++ / clang++ on PATH, else `python -m ziglang c++`
  (`pip install ziglang`) as a no-system-compiler fallback;
* flags: -O3 -std=c++17 -ffp-contract=off (keeps float rounding identical to OpenCV's INTER_AREA);
  on x86-64, -march=x86-64-v3 (AVX2) is tried first and dropped if the compiler rejects it;
* $FAST_MARIO_ENV_MARCH overrides the -march value ("native", "x86-64-v2", ...; "none" disables it);
* $FAST_MARIO_ENV_CXXFLAGS appends extra flags.
"""
import os
import platform
import shlex
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join("csrc", "fastmario.cpp")
DEPS = [SRC, os.path.join("csrc", "nes_core.hpp")]


def find_cxx():
    env = os.environ.get("CXX")
    if env:
        return shlex.split(env)
    for c in ("c++", "g++", "clang++"):
        if shutil.which(c):
            return [c]
    try:
        import ziglang  # noqa: F401
        return [sys.executable, "-m", "ziglang", "c++"]
    except ImportError:
        pass
    raise RuntimeError("fast-mario-env: no C++ compiler found. Install g++ or clang++, set CXX, or "
                       "`pip install ziglang` (used automatically as a fallback).")


def default_march():
    m = os.environ.get("FAST_MARIO_ENV_MARCH")
    if m:
        return None if m == "none" else m
    if platform.machine().lower() in ("x86_64", "amd64"):
        try:  # only if the build machine itself has AVX2 (the library is built where it is installed)
            with open("/proc/cpuinfo") as f:
                if " avx2" not in f.read():
                    return None
        except OSError:
            pass
        return "x86-64-v3"
    return None


class NativeLib(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class BuildNative(build_ext):
    def get_ext_filename(self, ext_name):
        # plain library name (not a CPython extension): fast_mario_env/libfastmario.so
        parts = ext_name.split(".")
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        return os.path.join(*parts[:-1], parts[-1] + suffix)

    def build_extension(self, ext):
        out = os.path.abspath(self.get_ext_fullpath(ext.name))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cxx = find_cxx()
        base = ["-O3", "-std=c++17", "-ffp-contract=off", "-fno-strict-aliasing", "-fPIC", "-shared", "-pthread",
                "-Wall", "-Wno-unused-function"]
        extra = shlex.split(os.environ.get("FAST_MARIO_ENV_CXXFLAGS", ""))
        src = os.path.join(HERE, SRC)
        march = default_march()
        attempts = []
        if march:
            attempts.append([f"-march={march}"])
            if "-" in march:  # zig cc spells CPU models with underscores (x86_64_v3)
                attempts.append([f"-march={march.replace('-', '_')}"])
        attempts.append([])
        last = None
        for arch in attempts:
            cmd = cxx + base + arch + extra + [src, "-o", out]
            print("building native library:", " ".join(cmd), flush=True)
            r = subprocess.run(cmd)
            if r.returncode == 0:
                return
            last = r.returncode
            if arch:
                print(f"compile with {arch[0]} failed; retrying", flush=True)
        raise RuntimeError(f"fast-mario-env: native build failed (exit {last}); see the compiler output above")


setup(ext_modules=[NativeLib("fast_mario_env.libfastmario")], cmdclass={"build_ext": BuildNative})

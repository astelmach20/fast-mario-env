# SPDX-License-Identifier: GPL-3.0-or-later
"""ctypes bindings for the native emulator library (built from csrc/ at install time)."""
from __future__ import annotations

import ctypes
import glob
import os

LIB_ENV_VAR = "FAST_MARIO_ENV_LIB"
HERE = os.path.dirname(os.path.abspath(__file__))

_P = ctypes.c_void_p
_lib = None


def lib_path() -> str:
    p = os.environ.get(LIB_ENV_VAR)
    if p:
        return p
    for pat in ("libfastmario*.so", "libfastmario*.dylib", "libfastmario*.dll", "libfastmario*.pyd"):
        hits = sorted(glob.glob(os.path.join(HERE, pat)))
        if hits:
            return hits[0]
    raise ImportError(
        "fast_mario_env: native library libfastmario not found next to the package. It is compiled when the "
        "package is installed (`pip install .`, needs a C++17 compiler; set CXX to choose one). "
        f"Alternatively point {LIB_ENV_VAR} at a prebuilt libfastmario.so.")


def lib():
    global _lib
    if _lib is not None:
        return _lib
    L = ctypes.CDLL(lib_path())
    i64, i32, dbl = ctypes.c_int64, ctypes.c_int, ctypes.c_double
    L.fm_create.restype = _P
    L.fm_create.argtypes = [_P, i64, i32, _P, _P, i32, i32, dbl, i32, i32, i32, _P, i32]
    L.fm_set_buffers.argtypes = [_P] * 9
    for f in ("fm_reset", "fm_step_wait", "fm_destroy"):
        getattr(L, f).argtypes = [_P]
    L.fm_step_async.argtypes = [_P, _P]
    L.fm_frames.restype = L.fm_bad_ops.restype = i64
    L.fm_frames.argtypes = L.fm_bad_ops.argtypes = [_P]
    L.fm_env_ram.argtypes = [_P, i32, _P]
    L.fm_state_bytes.restype = i64
    L.fm_env_bytes.restype = i64
    L.fm_env_bytes.argtypes = [_P]
    L.fm_palette.argtypes = [_P]
    L.fm_resize_ref.argtypes = [_P, i32, i32, _P, i32, i32]
    L.fm_resize_fast.argtypes = [_P, _P]
    L.fm_raw_create.restype = _P
    L.fm_raw_create.argtypes = [_P, i64]
    L.fm_raw_frame.argtypes = [_P, i32, i32]
    L.fm_raw_ram.restype = L.fm_raw_screen.restype = _P
    L.fm_raw_ram.argtypes = L.fm_raw_screen.argtypes = [_P]
    L.fm_raw_backup.argtypes = L.fm_raw_restore.argtypes = L.fm_raw_destroy.argtypes = [_P]
    L.fm_raw_bad_ops.restype = i64
    L.fm_raw_bad_ops.argtypes = [_P]
    _lib = L
    return L


def ptr(a):
    """ctypes pointer to a contiguous numpy array's data."""
    return a.ctypes.data_as(_P)

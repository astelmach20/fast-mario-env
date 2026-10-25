"""FastVecEnv: N bit-exact Super Mario Bros envs stepped by a C++ thread pool in this process.

    from fast_mario_env import FastVecEnv
    venv = FastVecEnv("1-1", n_envs=64, seed=0, sticky=0.25, noop_max=30)
    obs = venv.reset()                       # (N, 4, 84, 84) uint8
    venv.step_async(actions); ...; obs, rew, term, trunc, infos = venv.step_wait()

Semantics are those of nes-py's SuperMarioBrosEnv + JoypadSpace(SIMPLE_MOVEMENT) behind the
wrapper stack in tests/reference_env.py: frame-level sticky actions (numpy PCG64 stream per env),
frame skip with optional 2-frame max-pool, life loss and flag as episode end, random no-op starts,
grayscale + cv2 INTER_AREA 84x84, time limit, frame stack and auto-reset. ctypes releases the GIL
for each native call, so step_async() returns immediately and the learner can run while the
emulators work.
"""
from __future__ import annotations

import ctypes
import json
import os
import time

import numpy as np

from ._native import lib, ptr as _ptr
from .rom import check_rom, decode_level, load_rom

OBS_HW = 84
N_ACTIONS = 7  # gym_super_mario_bros.actions.SIMPLE_MOVEMENT
SIMPLE_MOVEMENT = [["NOOP"], ["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"], ["A"], ["left"]]
INFO_FIELDS = ("x_pos", "flag_get", "life", "score", "time", "world", "stage", "coins", "y_pos", "status")
INFO_KEYS = ("x_pos", "flag_get", "life", "score", "time", "world", "stage")  # keys put in each info dict


def palette_rgb() -> np.ndarray:
    """nes-py master palette as (64, 3) uint8 RGB."""
    p = np.zeros(64, np.uint32)
    lib().fm_palette(_ptr(p))
    return np.stack([(p >> 16) & 255, (p >> 8) & 255, p & 255], -1).astype(np.uint8)


def gray_lut() -> np.ndarray:
    """palette index (0-255, taken & 63) -> grayscale value, identical to cv2.cvtColor(RGB2GRAY)
    (OpenCV's 14-bit fixed-point BT.601 weights with round-half-up)."""
    rgb = palette_rgb().astype(np.int64)
    g = (rgb[:, 0] * 4899 + rgb[:, 1] * 9617 + rgb[:, 2] * 1868 + (1 << 13)) >> 14
    return np.ascontiguousarray(g[np.arange(256) & 63], np.uint8)


def rng_state(seed: int) -> list[int]:
    """numpy PCG64 state of np.random.default_rng(seed), as the 6 words the native side expects."""
    st = np.random.default_rng(seed).bit_generator.state
    s, inc = st["state"]["state"], st["state"]["inc"]
    m = (1 << 64) - 1
    return [s >> 64, s & m, inc >> 64, inc & m, st["has_uint32"], st["uinteger"]]


class RawNES:
    """A single raw emulator (frame stepping only, no env logic); used by the fidelity tests."""

    def __init__(self, rom_bytes: bytes):
        check_rom(rom_bytes)
        L = lib()
        self._rom = np.frombuffer(rom_bytes, np.uint8).copy()
        self.h = L.fm_raw_create(_ptr(self._rom), len(self._rom))
        if not self.h:
            raise ValueError("unsupported ROM")
        u8p = ctypes.POINTER(ctypes.c_uint8)
        self.ram = np.ctypeslib.as_array(ctypes.cast(L.fm_raw_ram(self.h), u8p), (0x800,))
        self.screen = np.ctypeslib.as_array(ctypes.cast(L.fm_raw_screen(self.h), u8p), (240, 256))

    def frame(self, buttons: int, render: bool = True):
        lib().fm_raw_frame(self.h, int(buttons), int(bool(render)))

    def backup(self):
        lib().fm_raw_backup(self.h)

    def restore(self):
        lib().fm_raw_restore(self.h)

    def bad_ops(self) -> int:
        return int(lib().fm_raw_bad_ops(self.h))

    def __del__(self):
        h, self.h = getattr(self, "h", None), None
        if h:
            try:
                lib().fm_raw_destroy(h)
            except Exception:
                pass


class MonitorCSV:
    """Writes an SB3 VecMonitor-compatible monitor.monitor.csv (columns r,l,t,x_pos,flag_get)."""

    def __init__(self, path: str):
        if not path.endswith(".monitor.csv"):
            path = os.path.join(path, "monitor.monitor.csv") if os.path.isdir(path) else path + ".monitor.csv"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = open(path, "w")
        self.f.write("#%s\n" % json.dumps({"t_start": time.time(), "env_id": "None"}))
        self.f.write("r,l,t,x_pos,flag_get\n")
        self.f.flush()

    def write(self, e: dict):
        self.f.write(f"{e['r']},{e['l']},{e['t']},{e['x_pos']},{e['flag_get']}\n")
        self.f.flush()

    def close(self):
        self.f.close()


class FastVecEnv:
    """N Mario envs stepped by a C++ thread pool.

    reset() -> obs (N, k, 84, 84) uint8
    step(actions) / step_async(actions) + step_wait() ->
        obs, reward float32 (N,), terminated bool (N,), truncated bool (N,), infos (list of N dicts)
    infos[i] has x_pos/flag_get/life/score/time/world/stage; when env i finished an episode in this
    step it also has 'episode' = dict(r, l, t, x_pos, flag_get), 'final_obs' (the last stacked obs
    before the automatic reset) and 'TimeLimit.truncated'.

    Args:
        level: "w-s" (e.g. "1-1"), or a list of N levels (one per env).
        n_envs: number of environments.
        seed: env i uses the numpy PCG64 stream ``seed * 1000 + env_offset + i`` for sticky actions
            and no-op starts (the reference stack uses the same scheme).
        frame_stack: number of stacked frames k.
        sticky: probability of repeating the previous action, sampled per NES frame.
        noop_max: up to this many random no-op agent steps after each reset (0 disables).
        max_steps: time limit in agent steps (truncation).
        maxpool: max over the gray frames of the last two skipped frames (as ALE does).
        skip: NES frames per agent step.
        n_threads: worker threads (default min(N, os.cpu_count())); 0 or 1 steps inline.
        copy_obs: if False, the returned obs is a view of the internal buffer (valid until the next step).
        rom_path / rom_mode: see fast_mario_env.rom.
        monitor_path: if given, episodes are written to an SB3-compatible monitor CSV.
    """

    def __init__(self, level="1-1", n_envs=8, seed=0, frame_stack=4, sticky=0.0, noop_max=0,
                 max_steps=2000, maxpool=False, skip=4, n_threads=None, copy_obs=True, rom_path=None,
                 rom_mode="vanilla", monitor_path=None, env_offset=0):
        L = lib()
        if not 1 <= skip <= 16:
            raise ValueError("skip must be in 1..16")
        if frame_stack < 1:
            raise ValueError("frame_stack must be >= 1")
        self.n, self.k, self.copy_obs = n_envs, frame_stack, copy_obs
        self.num_envs = n_envs
        levels = [level] * n_envs if isinstance(level, str) else list(level)
        if len(levels) != n_envs:
            raise ValueError(f"got {len(levels)} levels for {n_envs} envs")
        self.levels = levels
        self._lv = np.array([decode_level(lv) for lv in levels], np.int32).reshape(-1)
        self._rng = np.array([rng_state(seed * 1000 + env_offset + i) for i in range(n_envs)],
                             np.uint64).reshape(-1)
        self._rom = np.frombuffer(load_rom(rom_path, rom_mode), np.uint8).copy()
        self._lut = gray_lut()
        if n_threads is None:
            n_threads = min(n_envs, os.cpu_count() or 1)
        self.n_threads = n_threads
        self.h = L.fm_create(_ptr(self._rom), len(self._rom), n_envs, _ptr(self._lv), _ptr(self._lut), skip,
                             max_steps, float(sticky), int(noop_max), int(bool(maxpool)), frame_stack,
                             _ptr(self._rng), int(n_threads))
        if not self.h:
            raise RuntimeError("fm_create failed (unsupported ROM or arguments)")
        n, k = n_envs, frame_stack
        self._obs = np.zeros((n, k, OBS_HW, OBS_HW), np.uint8)
        self._fin = np.zeros((n, k, OBS_HW, OBS_HW), np.uint8)
        self._rew = np.zeros(n, np.float32)
        self._term = np.zeros(n, np.uint8)
        self._trunc = np.zeros(n, np.uint8)
        self._info = np.zeros((n, len(INFO_FIELDS)), np.int64)
        self._epr = np.zeros(n, np.float64)
        self._epl = np.zeros(n, np.int64)
        self.set_buffers(self._obs, self._fin, self._rew, self._term, self._trunc, self._info, self._epr, self._epl)
        self.monitor = MonitorCSV(monitor_path) if monitor_path else None
        self.t0 = time.time()
        self._act = np.zeros(n, np.int64)
        self._closed = False

    # -- low-level: let the native side write into caller-owned (e.g. pinned torch) buffers
    def set_buffers(self, obs, fin, rew, term, trunc, info, ep_ret, ep_len):
        """Point the native outputs at other buffers. Arguments are numpy arrays or raw addresses (int)
        with the dtypes/shapes of the default buffers; the caller keeps them alive."""
        P = ctypes.c_void_p
        args = [P(x) if isinstance(x, int) else _ptr(x) for x in (obs, fin, rew, term, trunc, info, ep_ret, ep_len)]
        lib().fm_set_buffers(self.h, *args)

    @property
    def frames(self) -> int:
        """Total emulated NES frames (all envs)."""
        return int(lib().fm_frames(self.h))

    def bad_ops(self) -> int:
        """Number of unimplemented opcodes executed (0 for correct emulation)."""
        return int(lib().fm_bad_ops(self.h))

    def bytes_per_env(self) -> int:
        return int(lib().fm_env_bytes(self.h))

    def _out(self):
        return self._obs.copy() if self.copy_obs else self._obs

    def reset(self):
        lib().fm_reset(self.h)
        return self._out()

    def step_async(self, actions):
        a = np.ascontiguousarray(np.asarray(actions).reshape(-1), np.int64)
        if a.shape[0] != self.n:
            raise ValueError(f"expected {self.n} actions, got {a.shape[0]}")
        self._act = a  # the pool copies the actions before returning, but keep it alive anyway
        lib().fm_step_async(self.h, _ptr(a))

    def step_wait(self):
        lib().fm_step_wait(self.h)
        term = self._term.astype(bool)
        trunc = self._trunc.astype(bool)
        inf = self._info.tolist()
        infos = []
        for i in range(self.n):
            r = inf[i]
            info = {"x_pos": r[0], "flag_get": bool(r[1]), "life": r[2], "score": r[3],
                    "time": r[4], "world": r[5], "stage": r[6]}
            if term[i] or trunc[i]:
                info["final_obs"] = self._fin[i].copy()
                info["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
                e = dict(r=round(float(self._epr[i]), 6), l=int(self._epl[i]), t=round(time.time() - self.t0, 6),
                         x_pos=info["x_pos"], flag_get=info["flag_get"])
                info["episode"] = e
                if self.monitor:
                    self.monitor.write(e)
            infos.append(info)
        return self._out(), self._rew.copy(), term, trunc, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def ram(self, i: int) -> np.ndarray:
        """Copy of env i's 2 KB work RAM."""
        out = np.zeros(0x800, np.uint8)
        lib().fm_env_ram(self.h, int(i), _ptr(out))
        return out

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        lib().fm_destroy(self.h)
        self.h = None
        if self.monitor:
            self.monitor.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def resize_area(gray: np.ndarray, fast: bool = True) -> np.ndarray:
    """240x256 uint8 -> 84x84 with the native port of cv2.resize(..., INTER_AREA)."""
    g = np.ascontiguousarray(gray, np.uint8)
    if g.shape != (240, 256):
        raise ValueError("expected a (240, 256) uint8 image")
    out = np.zeros((OBS_HW, OBS_HW), np.uint8)
    if fast:
        lib().fm_resize_fast(_ptr(g), _ptr(out))
    else:
        lib().fm_resize_ref(_ptr(g), 256, 240, _ptr(out), OBS_HW, OBS_HW)
    return out


__all__ = ["FastVecEnv", "RawNES", "MonitorCSV", "palette_rgb", "gray_lut", "rng_state", "resize_area",
           "OBS_HW", "N_ACTIONS", "SIMPLE_MOVEMENT", "INFO_FIELDS", "INFO_KEYS"]

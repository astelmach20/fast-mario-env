"""Frame-level fidelity: the native NES core vs nes-py's own emulator (lib_nes_env), same ROM and inputs.

Compares the full 2 KB RAM after every frame and the full 240x256 RGB screen after every frame the
native core rendered (rendering is switched on for a random subset of frames; between renders the
core skips pixel output but must keep all CPU-visible PPU state identical). Random controller input,
plus random state backup/restore on both sides.
"""
import ctypes
import glob
import os

import numpy as np
import pytest
from conftest import requires_ref, scaled

import fast_mario_env as fme

MOVES = [0x00, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40, 0x41, 0x20, 0x10, 0x08, 0x04]


class NesPyRaw:
    """Direct ctypes access to nes-py's compiled emulator (what NESEnv uses internally)."""

    def __init__(self, rom_path):
        import nes_py
        path = glob.glob(os.path.join(os.path.dirname(nes_py.__file__), "lib_nes_env*"))[0]
        L = self.L = ctypes.cdll.LoadLibrary(path)
        L.Initialize.argtypes = [ctypes.c_wchar_p]
        L.Initialize.restype = ctypes.c_void_p
        for f in ("Controller", "Screen", "Memory"):
            getattr(L, f).restype = ctypes.c_void_p
        L.Controller.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        for f in ("Screen", "Memory", "Reset", "Step", "Backup", "Restore", "Close"):
            getattr(L, f).argtypes = [ctypes.c_void_p]
        self.h = L.Initialize(rom_path)
        L.Reset(self.h)
        u8p = ctypes.POINTER(ctypes.c_uint8)
        self.ram = np.ctypeslib.as_array(ctypes.cast(L.Memory(self.h), u8p), (0x800,))
        self.screen = np.ctypeslib.as_array(ctypes.cast(L.Screen(self.h), u8p), (240, 256, 4))  # BGRx
        self.joy = np.ctypeslib.as_array(ctypes.cast(L.Controller(self.h, 0), u8p), (1,))

    def frame(self, buttons):
        self.joy[0] = buttons
        self.L.Step(self.h)

    def close(self):
        self.L.Close(self.h)


def run(seed, frames, render_p, backup_p=0.002):
    rom_path = fme.find_rom()
    ref = NesPyRaw(rom_path)
    raw = fme.RawNES(open(rom_path, "rb").read())
    pal = fme.palette_rgb()
    rng = np.random.default_rng(seed)
    btn, bad_ram, bad_scr, n_scr, n_restore, first_bad = 0, 0, 0, 0, 0, None
    have_backup = False
    for f in range(frames):
        if f % int(rng.integers(1, 12)) == 0:
            btn = MOVES[int(rng.integers(0, len(MOVES)))] if rng.random() < 0.97 else int(rng.integers(0, 256))
        u = rng.random()
        if u < backup_p:
            ref.L.Backup(ref.h); raw.backup(); have_backup = True
        elif u < 2 * backup_p and have_backup:
            ref.L.Restore(ref.h); raw.restore(); n_restore += 1
        render = bool(rng.random() < render_p)
        ref.frame(btn)
        raw.frame(btn, render)
        if not np.array_equal(ref.ram, raw.ram):
            bad_ram += 1
            if first_bad is None:
                first_bad = dict(frame=f, kind="ram", addrs=[hex(x) for x in np.flatnonzero(ref.ram != raw.ram)[:10]])
        if render:
            n_scr += 1
            if not np.array_equal(ref.screen[:, :, 2::-1], pal[raw.screen & 63]):
                bad_scr += 1
                first_bad = first_bad or dict(frame=f, kind="screen")
        if bad_ram > 5:
            break
    res = dict(frames=f + 1, ram_mismatch_frames=bad_ram, screens_checked=n_scr, screen_mismatch_frames=bad_scr,
               restores=n_restore, bad_ops=raw.bad_ops(), first_bad=first_bad)
    ref.close()
    return res


@requires_ref
@pytest.mark.parametrize("seed,render_p", [(0, 0.3), (1, 0.3), (2, 1.0)])
def test_raw_frames_vs_nes_py(seed, render_p):
    r = run(seed, scaled(4000), render_p)
    print(r)
    assert r["ram_mismatch_frames"] == 0 and r["screen_mismatch_frames"] == 0, r
    assert r["screens_checked"] > 0 and r["restores"] > 0 and r["bad_ops"] == 0

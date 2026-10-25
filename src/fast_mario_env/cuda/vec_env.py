"""CudaVecEnv: thousands of bit-exact Super Mario Bros envs stepped on the GPU.

    from fast_mario_env.cuda import CudaVecEnv
    env = CudaVecEnv("1-1", n_envs=8192, seed=0, sticky=0.25, noop_max=30)
    obs = env.reset()                         # (N, 4, 84, 84) uint8 CUDA tensor (internal buffer)
    obs, rew, term, trunc, info = env.step(actions)   # actions: (N,) int64 CUDA tensor
    # rew f32 (N,), term/trunc bool (N,), info: dict of CUDA tensors (x_pos, flag_get, life, score,
    # time, world, stage, coins, y_pos, status: int32 (N,)), ep_ret f64 / ep_len int32 (valid where
    # term|trunc), final_obs (N,4,84,84) (valid where term|trunc).

Semantics are those of fast_mario_env.FastVecEnv with maxpool=False and frame_stack=4: SIMPLE_MOVEMENT,
frame-level sticky actions with the same numpy PCG64 streams (env i seeded seed*1000+i), frame skip,
LifeLoss, NoopStart, gray + INTER_AREA 84x84, TimeLimit, frame stack, auto-reset.
All tensors returned by step()/reset() are views of internal buffers, overwritten by the next call
(clone() them if you need to keep them). No host<->device copies happen in step().
"""
import ctypes
import math
import os

import numpy as np
import torch

from ..env import gray_lut, rng_state
from ..rom import decode_level, load_rom
from . import cujit

KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")

INFO_FIELDS = ("x_pos", "flag_get", "life", "score", "time", "world", "stage", "coins", "y_pos", "status")
OBS = 84


# ----------------------------------------------------------------- host-side tables
def _area_tab(ssize, dsize):
    """cv::computeResizeAreaTab (== compute_area_tab in csrc/fastmario.cpp), double precision."""
    scale = 1.0 / (dsize / ssize)
    tab = []
    for dx in range(dsize):
        fsx1 = dx * scale
        fsx2 = fsx1 + scale
        cell = min(scale, ssize - fsx1)
        sx1 = int(math.ceil(fsx1)); sx2 = int(math.floor(fsx2))
        sx2 = min(sx2, ssize - 1)
        sx1 = min(sx1, sx2)
        if sx1 - fsx1 > 1e-3:
            tab.append((dx, sx1 - 1, np.float32((sx1 - fsx1) / cell)))
        for sx in range(sx1, sx2):
            tab.append((dx, sx, np.float32(1.0 / cell)))
        if fsx2 - sx2 > 1e-3:
            tab.append((dx, sx2, np.float32(min(min(fsx2 - sx2, 1.0), cell) / cell)))
    return tab


def resize_tables():
    M, D = 6, OBS
    xt = _area_tab(256, D)
    xs = np.zeros((M, D), np.int32); xw = np.zeros((M, D), np.float32); cnt = [0] * D
    for d, s, a in xt:
        assert cnt[d] < M
        xs[cnt[d], d] = s; xw[cnt[d], d] = a; cnt[d] += 1
    yt = _area_tab(240, D)
    ye = -np.ones((240, 2), np.int32); yw = np.zeros((240, 2), np.float32); yc = [0] * 240
    for j, (d, s, a) in enumerate(yt):
        first = j == 0 or yt[j - 1][0] != d
        last = j == len(yt) - 1 or yt[j + 1][0] != d
        k = yc[s]; assert k < 2; yc[s] += 1
        if k == 1:
            assert ye[s, 0] & 0xff == d - 1  # a row spanning two outputs is consecutive in ytab
        ye[s, k] = d | (int(first) << 8) | (int(last) << 9); yw[s, k] = a
    assert all(c >= 1 for c in yc)
    gr = np.zeros((D, 2), np.int32)
    for d in range(D):
        rows = [s for (dd, s, a) in yt if dd == d]
        assert rows == list(range(rows[0], rows[-1] + 1))
        gr[d] = rows[0], rows[-1]
    return xs, xw, ye, yw, gr


def _op_cycles():
    import re
    src = open(os.path.join(KERNEL_DIR, "nes_cuda.cu")).read()
    body = src[src.index("OPERATION_CYCLES[0x100] = {") + 27: src.index("};", src.index("OPERATION_CYCLES[0x100]"))]
    v = [int(x) for x in re.findall(r"\d+", body)]
    assert len(v) == 256
    return v


def decode_table():
    """Decode table for the v2 interpreter: official opcodes get a class/mode/op; the rest use the slow
    path (exact nes-py cascade). bits: 0-3 kind, 4-7 mode, 8 page-cross penalty, 12-15 op, 16-23 opcode,
    24-31 base cycles."""
    K_SLOW, K_IMPL, K_BRANCH, K_READ, K_STORE, K_RMW, K_ACC = range(7)
    M_NONE, M_IMM, M_ZP, M_ZPX, M_ZPY, M_ABS, M_ABSX, M_ABSY, M_INDX, M_INDY = range(10)
    cyc = _op_cycles()
    ent = {}
    for op in [0x00, 0x08, 0x18, 0x20, 0x28, 0x38, 0x40, 0x48, 0x4C, 0x58, 0x60, 0x68, 0x6C, 0x78, 0x88, 0x8A,
               0x98, 0x9A, 0xA8, 0xAA, 0xB8, 0xBA, 0xC8, 0xCA, 0xD8, 0xE8, 0xEA, 0xF8]:
        ent[op] = (K_IMPL, M_NONE, 0, 0)
    for op in [0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0]:
        ent[op] = (K_BRANCH, M_NONE, 0, 0)
    m1 = [M_INDX, M_ZP, M_IMM, M_ABS, M_INDY, M_ZPX, M_ABSY, M_ABSX]
    alu1 = [0, 1, 2, 3, None, 4, 5, 6]  # ORA AND EOR ADC (STA) LDA CMP SBC
    for opn in range(8):
        for mi in range(8):
            op = (opn << 5) | (mi << 2) | 1
            if op == 0x89:
                continue
            pcx = int(mi in (4, 6, 7) and opn != 4)
            ent[op] = (K_STORE, m1[mi], pcx, 0) if opn == 4 else (K_READ, m1[mi], pcx, alu1[opn])
    rmw = {0: 0, 1: 1, 2: 2, 3: 3, 6: 4, 7: 5}  # ASL ROL LSR ROR DEC INC
    for opn, r in rmw.items():
        base = opn << 5
        for mi, md in ((1, M_ZP), (3, M_ABS), (5, M_ZPX), (7, M_ABSX)):
            ent[base | (mi << 2) | 2] = (K_RMW, md, int(mi == 7), r)
        if opn < 4:
            ent[base | 0x0A] = (K_ACC, M_NONE, 0, r)
    ent[0x86] = (K_STORE, M_ZP, 0, 1); ent[0x8E] = (K_STORE, M_ABS, 0, 1); ent[0x96] = (K_STORE, M_ZPY, 0, 1)
    A_BIT, A_LDY, A_CPY, A_CPX, A_LDX = 7, 8, 9, 10, 11
    for op, md, pcx in ((0xA2, M_IMM, 0), (0xA6, M_ZP, 0), (0xAE, M_ABS, 0), (0xB6, M_ZPY, 0), (0xBE, M_ABSY, 1)):
        ent[op] = (K_READ, md, pcx, A_LDX)
    ent[0x24] = (K_READ, M_ZP, 0, A_BIT); ent[0x2C] = (K_READ, M_ABS, 0, A_BIT)
    ent[0x84] = (K_STORE, M_ZP, 0, 2); ent[0x8C] = (K_STORE, M_ABS, 0, 2); ent[0x94] = (K_STORE, M_ZPX, 0, 2)
    for op, md, pcx in ((0xA0, M_IMM, 0), (0xA4, M_ZP, 0), (0xAC, M_ABS, 0), (0xB4, M_ZPX, 0), (0xBC, M_ABSX, 1)):
        ent[op] = (K_READ, md, pcx, A_LDY)
    for op, md in ((0xC0, M_IMM), (0xC4, M_ZP), (0xCC, M_ABS)):
        ent[op] = (K_READ, md, 0, A_CPY)
    for op, md in ((0xE0, M_IMM), (0xE4, M_ZP), (0xEC, M_ABS)):
        ent[op] = (K_READ, md, 0, A_CPX)
    t = np.zeros(256, np.uint32)
    for op in range(256):
        k, md, pcx, a = ent.get(op, (K_SLOW, M_NONE, 0, 0))
        t[op] = k | (md << 4) | (pcx << 8) | (a << 12) | (op << 16) | (cyc[op] << 24)
    assert len(ent) == 151, len(ent)
    return t


def decode_table3():
    """(256, 2) uint32: word0 = decode_table(); word1 = micro-op fields for the uniform (branch-free) fast
    path. bit0 fast-eligible, 1-4 alu, 5-7 src, 8-10 dst, 11-12 areg, 13 setNZ, 14 setC, 15 setV,
    16-18 ctrl, 22 flag value, 24-31 flag mask."""
    F_N, F_V, F_D, F_I, F_Z, F_C = 0x01, 0x02, 0x10, 0x20, 0x40, 0x80
    PASS, OR, AND, EOR, ADC, SBC, CMP, BIT, ASL, ROL, LSR, ROR, INC, DEC, FLAG = range(15)
    MEM, A, X, Y, SP, P, PULL = range(7)
    NONE, dA, dX, dY, dSP, dP, dMEM, PUSH = range(8)
    C_NONE, C_JMP, C_JSR, C_RTS, C_BR = range(5)
    D = decode_table()
    U = np.zeros(256, np.uint32)

    def f(alu=PASS, src=MEM, dst=NONE, ar=0, nz=0, c=0, v=0, ctrl=C_NONE, fval=0, fmask=0):
        return 1 | alu << 1 | src << 5 | dst << 8 | ar << 11 | nz << 13 | c << 14 | v << 15 | ctrl << 16 | fval << 22 | fmask << 24
    impl = {0x08: f(src=P, dst=PUSH), 0x18: f(alu=FLAG, fmask=F_C), 0x20: f(ctrl=C_JSR), 0x28: f(src=PULL, dst=dP),
            0x38: f(alu=FLAG, fmask=F_C, fval=1), 0x48: f(src=A, dst=PUSH), 0x4C: f(ctrl=C_JMP),
            0x58: f(alu=FLAG, fmask=F_I), 0x60: f(ctrl=C_RTS), 0x68: f(src=PULL, dst=dA, nz=1),
            0x78: f(alu=FLAG, fmask=F_I, fval=1), 0x88: f(alu=DEC, src=Y, dst=dY, nz=1), 0x8A: f(src=X, dst=dA, nz=1),
            0x98: f(src=Y, dst=dA, nz=1), 0x9A: f(src=X, dst=dSP), 0xA8: f(src=A, dst=dY, nz=1), 0xAA: f(src=A, dst=dX, nz=1),
            0xB8: f(alu=FLAG, fmask=F_V), 0xBA: f(src=SP, dst=dX, nz=1), 0xC8: f(alu=INC, src=Y, dst=dY, nz=1),
            0xCA: f(alu=DEC, src=X, dst=dX, nz=1), 0xD8: f(alu=FLAG, fmask=F_D), 0xE8: f(alu=INC, src=X, dst=dX, nz=1),
            0xEA: f(), 0xF8: f(alu=FLAG, fmask=F_D, fval=1)}
    read_alu = {0: f(alu=OR, dst=dA, nz=1), 1: f(alu=AND, dst=dA, nz=1), 2: f(alu=EOR, dst=dA, nz=1),
                3: f(alu=ADC, dst=dA, nz=1, c=1, v=1), 4: f(dst=dA, nz=1), 5: f(alu=CMP, nz=1, c=1),
                6: f(alu=SBC, dst=dA, nz=1, c=1, v=1), 7: f(alu=BIT), 8: f(dst=dY, nz=1), 9: f(alu=CMP, ar=2, nz=1, c=1),
                10: f(alu=CMP, ar=1, nz=1, c=1), 11: f(dst=dX, nz=1)}
    rmw_alu = [(ASL, 1), (ROL, 1), (LSR, 1), (ROR, 1), (DEC, 0), (INC, 0)]
    for op in range(256):
        d = int(D[op]); k = d & 15; o = (d >> 12) & 15
        if k == 1:
            U[op] = impl.get(op, 0)
        elif k == 2:
            U[op] = f(ctrl=C_BR)
        elif k == 3:
            U[op] = read_alu[o]
        elif k == 4:
            U[op] = f(src=[A, X, Y][o], dst=dMEM)
        elif k in (5, 6):
            al, c = rmw_alu[o]
            U[op] = f(alu=al, src=MEM if k == 5 else A, dst=dMEM if k == 5 else dA, nz=1, c=c)
    return np.stack([D, U], 1)


def load_cart(rom):
    d = np.frombuffer(rom, np.uint8)
    assert bytes(d[:4]) == b"NES\x1a"
    mirroring = d[6] & 0xB
    mapper = ((d[6] >> 4) & 0xF) | (d[7] & 0xF0)
    banks, vbanks = int(d[4]), int(d[5])
    assert mapper == 0 and not (d[6] & 0x6) and banks in (1, 2) and vbanks == 1
    prg = d[16:16 + 0x4000 * banks]
    if banks == 1:
        prg = np.concatenate([prg, prg])
    chr_ = d[16 + 0x4000 * banks: 16 + 0x4000 * banks + 0x2000]
    nt = {0: [0, 0, 0x400, 0x400], 1: [0, 0x400, 0, 0x400]}.get(int(mirroring), [0, 0, 0, 0])
    return np.concatenate([prg, chr_]).astype(np.uint8), np.array(nt, np.int32)


# ----------------------------------------------------------------- ctypes mirrors of kernel structs
_P = ctypes.c_uint64


class Cfg(ctypes.Structure):
    _fields_ = [("skip", ctypes.c_int), ("max_steps", ctypes.c_int), ("noop_max", ctypes.c_int),
                ("sticky", ctypes.c_double)]


class EnvArrays(ctypes.Structure):
    _fields_ = [(f, _P) for f in ("x_last", "time_last", "prev", "lives", "t", "ep_len", "ep_ret", "rng", "level",
                                  "frames")]


class StepArgs(ctypes.Structure):
    _fields_ = ([("n", ctypes.c_int), ("np", ctypes.c_int), ("nc", ctypes.c_int), ("cfg", Cfg),
                 ("state", _P), ("snap", _P), ("A", EnvArrays)] +
                [(f, _P) for f in ("actions", "newobs", "rew", "term", "trunc", "info", "ep_ret_out", "ep_len_out",
                                   "reset_idx", "cstate", "cx", "ct", "clives", "cdraws", "cart", "nt", "xs", "xw",
                                   "ye", "yw", "lut", "bad_ops", "gr", "dbg_screen")])


AUXW = 1072 + 4 * 288 + (256 + 12 * 256) // 4 + 84  # per-env render log words (nes_cuda2.cu AX_*)
SC_WORDS = 0x1128 // 4 + 8 + 35  # byte region + 4 int64 + 35 scalars (see SC_LIST; checked via k_nw)

_modules = {}


def get_module(skip=4, extra_opts=(), device=0):
    key = (skip, tuple(extra_opts), device)
    if key not in _modules:
        src = "".join(open(os.path.join(KERNEL_DIR, f)).read() for f in ("nes_cuda.cu", "nes_cuda2.cu"))
        opts = ["--fmad=false", "--device-int128", "-std=c++17", f"-DSKIP_K={skip}"] + list(extra_opts)
        _modules[key] = cujit.Module(src, "nes_cuda", opts, device)
    return _modules[key]


def _p(t):
    return ctypes.c_uint64(t.data_ptr()).value


class CudaVecEnv:
    """N bit-exact Super Mario Bros envs on the GPU, one CUDA thread per env (see module docstring).

    Args mirror FastVecEnv: level ("w-s" or a list of N levels), n_envs, seed, sticky, noop_max,
    max_steps, skip, rom_path, env_offset (env i uses RNG stream seed*1000 + env_offset + i).
    Only frame_stack=4, maxpool=False and rom_mode="vanilla" are supported.
    kernel: "v2" (optimized, default) or "v1" (straightforward reference port).
    """

    def __init__(self, level="1-1", n_envs=1024, seed=0, sticky=0.0, noop_max=0, max_steps=2000, skip=4,
                 frame_stack=4, device="cuda", block=64, rom_path=None, rom_mode="vanilla", extra_opts=(),
                 kernel="v2", env_offset=0, render_kernel="k_render2"):
        if frame_stack != 4:
            raise ValueError("CudaVecEnv supports frame_stack=4 only")
        if rom_mode != "vanilla":
            raise ValueError("CudaVecEnv supports rom_mode='vanilla' only")
        if not 1 <= skip <= 16:
            raise ValueError("skip must be in 1..16")
        if kernel not in ("v1", "v2"):
            raise ValueError("kernel must be 'v1' or 'v2'")
        self.dev = torch.device(device)
        di = self.dev.index if self.dev.index is not None else torch.cuda.current_device()
        self.mod = get_module(skip, extra_opts, di)
        chk = torch.zeros(2, dtype=torch.int32, device=self.dev)
        self.mod.launch("k_nw", 1, 1, [chk])
        assert chk[0].item() == SC_WORDS, chk
        self.n = n = n_envs
        self.np_ = np_ = (n + 31) // 32 * 32
        self.block = block
        levels = [level] * n if isinstance(level, str) else list(level)
        if len(levels) != n:
            raise ValueError(f"got {len(levels)} levels for {n} envs")
        uniq = sorted(set(levels), key=levels.index)
        self.levels = uniq
        lv = np.array([decode_level(lv_) for lv_ in uniq], np.int32).reshape(-1)
        cart, nt = load_cart(load_rom(rom_path, rom_mode))
        xs, xw, ye, yw, gr = resize_tables()
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.dev)  # noqa: E731
        self.cart, self.nt = T(cart), T(nt)
        self.xs, self.xw, self.ye, self.yw, self.gr = T(xs), T(xw), T(ye), T(yw), T(gr)
        self.lut = T(gray_lut())
        self.cfg = Cfg(skip, max_steps, noop_max, float(sticky))
        self.noop_max = noop_max
        nl = len(uniq)
        z = lambda *s, dt=torch.int64: torch.zeros(*s, dtype=dt, device=self.dev)  # noqa: E731
        u32 = torch.int32
        # ---- boot each level (one thread per level), then the reset cache (one thread per (level, k))
        self.bstate = z(SC_WORDS * nl, dt=u32)
        self.bx, self.bt = z(nl), z(nl)
        self.obs0 = z(nl, OBS, OBS, dt=torch.uint8)
        tabs = [self.cart, self.nt, self.xs, self.xw, self.ye, self.yw, self.lut, self.gr]
        self.mod.launch("k_boot", (nl + 31) // 32, 32, [ctypes.c_int(nl), T(lv), self.bstate, self.bx, self.bt,
                                                         self.obs0] + tabs)
        nk = noop_max + 1
        nc = nl * nk
        self.nc = nc
        self.cstate = z(SC_WORDS * nc, dt=u32)
        cscratch = z(SC_WORDS * nc, dt=u32)
        self.cx, self.ct = z(nc), z(nc)
        self.clives, self.cdraws = z(nc, dt=u32), z(nc, dt=u32)
        self.cobs = z(nc, OBS, OBS, dt=torch.uint8)
        self.mod.launch("k_cache", (nc + 31) // 32, 32,
                        [ctypes.c_int(nl), ctypes.c_int(noop_max), ctypes.c_int(skip), ctypes.c_double(sticky),
                         ctypes.c_int(max_steps), self.bstate, self.bx, self.bt, self.obs0, self.cstate, cscratch,
                         self.cx, self.ct, self.clives, self.cdraws, self.cobs] + tabs)
        torch.cuda.synchronize(self.dev)
        del cscratch
        # ---- per-env buffers
        self.state = z(SC_WORDS * np_, dt=u32)
        self.snap = z(SC_WORDS * np_, dt=u32)
        lvl_idx = np.array([uniq.index(l) for l in levels], np.int32)
        rng = np.array([rng_state(seed * 1000 + env_offset + i) for i in range(n)], np.uint64).reshape(-1)
        self.e = dict(x_last=z(n), time_last=z(n), prev=z(n, dt=u32), lives=z(n, dt=u32), t=z(n, dt=u32),
                      ep_len=z(n, dt=u32), ep_ret=z(n, dt=torch.float64),
                      rng=torch.from_numpy(rng.view(np.int64)).to(self.dev), level=T(lvl_idx), frames=z(n))
        self.actions = z(n)
        self.newobs = z(n, OBS, OBS, dt=torch.uint8)
        self.obs = z(n, 4, OBS, OBS, dt=torch.uint8)
        self.final_obs = z(n, 4, OBS, OBS, dt=torch.uint8)
        self.rew = z(n, dt=torch.float32)
        self.term = z(n, dt=torch.bool)
        self.trunc = z(n, dt=torch.bool)
        self.info_t = z(n, len(INFO_FIELDS), dt=u32)
        self.ep_ret = z(n, dt=torch.float64)
        self.ep_len = z(n, dt=u32)
        self.reset_idx = z(n, dt=u32)
        self.bad = z(1, dt=u32)
        self.info = {k: self.info_t[:, i] for i, k in enumerate(INFO_FIELDS)}
        self.info["ep_ret"] = self.ep_ret
        self.info["ep_len"] = self.ep_len
        self.info["final_obs"] = self.final_obs
        self._args = StepArgs()
        a = self._args
        a.n, a.np, a.nc, a.cfg = n, np_, nc, self.cfg
        a.state, a.snap = _p(self.state), _p(self.snap)
        a.A = EnvArrays(*[_p(self.e[f]) for f, _ in EnvArrays._fields_])
        for f in ("actions", "newobs", "rew", "term", "trunc"):
            setattr(a, f, _p(getattr(self, f)))
        a.info, a.ep_ret_out, a.ep_len_out, a.reset_idx = _p(self.info_t), _p(self.ep_ret), _p(self.ep_len), \
            _p(self.reset_idx)
        a.cstate, a.cx, a.ct, a.clives, a.cdraws = (_p(x) for x in (self.cstate, self.cx, self.ct, self.clives,
                                                                   self.cdraws))
        a.cart, a.nt, a.xs, a.xw, a.ye, a.yw, a.lut, a.gr = (_p(x) for x in tabs)
        a.bad_ops = _p(self.bad)
        self.aux = z(n * AUXW, dt=u32)
        self.dtab = T(decode_table3().reshape(-1).view(np.int32))
        self.kernel = kernel
        self.render_grid = torch.cuda.get_device_properties(self.dev).multi_processor_count * 6
        self.render_kernel = render_kernel
        self.block2 = int(next((o.split("=")[1] for o in extra_opts if o.startswith("-DBLOCK2=")), 128))

    # ------------------------------------------------------------- API
    def _launch(self, mode):
        if mode == 0 and self.kernel == "v2":
            b2 = self.block2
            self.mod.launch("k_step2", (self.n + b2 - 1) // b2, b2, [self._args, self.dtab, self.aux])
            self.mod.launch(self.render_kernel, min(self.n, self.render_grid), 256, [self._args, self.aux])
            self.mod.launch("k_reset", (self.n + 127) // 128, 128, [self._args])
        else:
            g = (self.n + self.block - 1) // self.block
            self.mod.launch("k_step", g, self.block, [self._args, ctypes.c_int(mode)])
        self.mod.launch("k_stack", self.n, 128, [ctypes.c_int(self.n), self.obs, self.newobs, self.final_obs,
                                                 self.term, self.trunc, self.reset_idx, self.cobs,
                                                 ctypes.c_int(mode)])

    def reset(self):
        self._launch(1)
        return self.obs

    def step(self, actions):
        if actions.dtype != torch.int64 or actions.device != self.obs.device or not actions.is_contiguous():
            actions = actions.to(self.obs.device, torch.int64).contiguous()
        if actions.data_ptr() != self.actions.data_ptr():
            self.actions.copy_(actions.view(-1))
        self._launch(0)
        return self.obs, self.rew, self.term, self.trunc, self.info

    def ram(self):
        """(N, 2048) uint8 RAM of every env (debug / fidelity; makes a copy)."""
        w = self.state[: (0x800 // 4) * self.np_].view(0x800 // 4, self.np_)[:, : self.n]
        return w.t().contiguous().view(torch.uint8).view(self.n, 0x800)

    @property
    def frames(self):
        return int(self.e["frames"].sum().item())

    def bad_ops(self):
        return int(self.bad.item())

    def bytes_per_env(self):
        tot = 0
        for t in [self.state, self.snap, self.newobs, self.obs, self.final_obs, self.aux]:
            tot += t.numel() * t.element_size()
        for t in self.e.values():
            tot += t.numel() * t.element_size()
        return tot / self.n

// fast-mario-env CUDA port: one thread per Super Mario Bros env, bit-exact with csrc/nes_core.hpp +
// csrc/fastmario.cpp (which are bit-exact with nes-py + gym-super-mario-bros).
//
// Layout: every env's console state lives in global memory, *interleaved by 32-bit word*
// (word w of env e at S[w * NP + e]), so that lanes of a warp touching the same emulated
// address (common: same game code) produce one coalesced transaction. CPU/PPU scalars are
// loaded into registers for the duration of a kernel and stored back at the end.
//
// Compiled at runtime with NVRTC (see cujit.py), options: --fmad=false (exact float order for
// the INTER_AREA port), --device-int128 (numpy PCG64).
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef short int16_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;

#define FI __device__ __forceinline__

#ifndef SKIP_K
#define SKIP_K 4
#endif

// ------------------------------------------------------------------ state layout
#define OFF_RAM 0x0000
#define OFF_VRAM 0x0800
#define OFF_OAM 0x1000
#define OFF_PAL 0x1100
#define OFF_SL 0x1120
#define NB_BYTES 0x1128
#define SC_W0 (NB_BYTES / 4)

#define SC_LIST                                                                                        \
    X64(iter) X64(exec_at) X64(ppu_calls) X64(nmi_thresh)                                              \
    X(uint8_t, zero_skip) X(uint16_t, pc) X(uint8_t, sp) X(uint8_t, a) X(uint8_t, x) X(uint8_t, y)     \
    X(uint8_t, p) X(uint8_t, joy_strobe0) X(uint8_t, joy_strobe1) X(uint8_t, joy_buttons0)             \
    X(uint8_t, joy_buttons1) X(uint8_t, joy_bits0) X(uint8_t, joy_bits1) X(uint8_t, pstate)            \
    X(uint8_t, even) X(uint8_t, vblank) X(uint8_t, s0hit) X(uint8_t, first_write)                      \
    X(uint8_t, data_buffer) X(uint8_t, oam_addr) X(uint8_t, fine_x) X(uint8_t, show_bg)                \
    X(uint8_t, show_spr) X(uint8_t, hide_edge_bg) X(uint8_t, hide_edge_spr) X(uint8_t, long_spr)       \
    X(uint8_t, nmi_en) X(uint8_t, bg_page) X(uint8_t, spr_page) X(uint8_t, nsl) X(int32_t, pcyc)       \
    X(int32_t, scanline) X(uint16_t, v) X(uint16_t, t) X(uint16_t, inc)

enum {
#define X64(n) W_##n, W_##n##_hi,
#define X(t, n) W_##n,
    SC_LIST
#undef X
#undef X64
    W_END
};
#define NW (SC_W0 + W_END)

struct Sc {
#define X64(n) int64_t n;
#define X(t, n) t n;
    SC_LIST
#undef X
#undef X64
};

// A view of one env's interleaved state.
struct Mem {
    uint32_t* w;  // = base + e
    int n;        // interleave stride (number of envs in the array, padded)
    FI uint32_t& W(int i) const { return w[(size_t)i * n]; }
    FI uint8_t& B(int off) const { return ((uint8_t*)(w + (size_t)(off >> 2) * n))[off & 3]; }
};
FI Mem mem_of(uint32_t* base, int e, int n) { Mem m; m.w = base + e; m.n = n; return m; }

FI void load_sc(Sc& s, const Mem& m) {
#define X64(n) s.n = (int64_t)((uint64_t)m.W(SC_W0 + W_##n) | ((uint64_t)m.W(SC_W0 + W_##n##_hi) << 32));
#define X(t, n) s.n = (t)m.W(SC_W0 + W_##n);
    SC_LIST
#undef X
#undef X64
}
FI void store_sc(const Sc& s, const Mem& m) {
#define X64(n) m.W(SC_W0 + W_##n) = (uint32_t)(uint64_t)s.n; m.W(SC_W0 + W_##n##_hi) = (uint32_t)((uint64_t)s.n >> 32);
#define X(t, n) m.W(SC_W0 + W_##n) = (uint32_t)s.n;
    SC_LIST
#undef X
#undef X64
}
FI void copy_bytes(const Mem& d, const Mem& s) {
    for (int i = 0; i < SC_W0; i++) d.W(i) = s.W(i);
}

// ------------------------------------------------------------------ constants
enum : uint8_t { F_N = 0x01, F_V = 0x02, F_ONE = 0x04, F_B = 0x08, F_D = 0x10, F_I = 0x20, F_Z = 0x40, F_C = 0x80 };
enum { PRE_RENDER = 0, RENDER = 1, POST_RENDER = 2, VERTICAL_BLANK = 3 };
__constant__ uint8_t OPERATION_CYCLES[0x100] = {
    7, 6, 0, 0, 0, 3, 5, 0, 3, 2, 2, 0, 0, 4, 6, 0, 2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    6, 6, 0, 0, 3, 3, 5, 0, 4, 2, 2, 0, 4, 4, 6, 0, 2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    6, 6, 0, 0, 0, 3, 5, 0, 3, 2, 2, 0, 3, 4, 6, 0, 2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    6, 6, 0, 0, 0, 3, 5, 0, 4, 2, 2, 0, 5, 4, 6, 0, 2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    0, 6, 0, 0, 3, 3, 3, 0, 2, 0, 2, 0, 4, 4, 4, 0, 2, 6, 0, 0, 4, 4, 4, 0, 2, 5, 2, 0, 0, 5, 0, 0,
    2, 6, 2, 0, 3, 3, 3, 0, 2, 2, 2, 0, 4, 4, 4, 0, 2, 5, 0, 0, 4, 4, 4, 0, 2, 4, 2, 0, 4, 4, 4, 0,
    2, 6, 0, 0, 3, 3, 5, 0, 2, 2, 2, 0, 4, 4, 6, 0, 2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    2, 6, 0, 0, 3, 3, 5, 0, 2, 2, 2, 2, 4, 4, 6, 0, 2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
};
#define CYCLES_PER_FRAME 29781
#define SCANLINE_END_CYCLE 341
#define VISIBLE_SCANLINES 240
#define FRAME_END_SCANLINE 261

// Cartridge (shared, read-only): prg[0x8000] | chr[0x2000] | nt[4] (uint16)
struct Cart {
    const uint8_t* __restrict__ prg;
    const uint8_t* __restrict__ chr;
    int nt0, nt1, nt2, nt3;
};

// resize tables (see python side): per output column dx up to RM taps; per source row up to 2 entries
#define RD 84
#define RM 6
struct Rs {
    const int* __restrict__ xs;    // [RM][RD]
    const float* __restrict__ xw;  // [RM][RD]
    const int* __restrict__ ye;    // [240][2]: dy | first<<8 | last<<9 , -1 = none
    const float* __restrict__ yw;  // [240][2]
    const uint8_t* __restrict__ lut;  // [256] palette value -> gray
    const int* __restrict__ gr;    // [84][2]: first / last source row of output row dy
};
#define DBUF_ROWS 12

// ------------------------------------------------------------------ console
struct Nes {
    Sc s;
    Mem m;
    Cart c;
    bool render;
    int64_t cur_iter;
    int skip;
    int bad_ops;
    // render state
    int32_t tc_key;
    uint8_t tc_lo, tc_hi, tc_attr;
    const Rs* rs;
    uint8_t* obs;      // 84*84 output for rendered frames
    uint8_t* screen;   // optional 240*256 palette values (debug)
    uint8_t row[256];  // gray of the scanline being rendered
    float sum[RD];     // vertical INTER_AREA accumulator
    // The emulated frame (29781*3 PPU calls) is 1-2 calls longer than a PPU frame, so its start
    // drifts through the PPU frame: when it lies in the visible area, rows complete in rotated
    // order and the start line(s) are rendered partly at the frame start and partly (re-rendered)
    // at its end. Output rows whose source rows include those lines are deferred to frame end;
    // their source rows are kept in dbuf.
    int spec0, spec1, gmin, gmax, dmin, dmax;
    uint8_t dbuf[DBUF_ROWS][256];

    FI uint8_t& RAM(int a) { return m.B(OFF_RAM + a); }
    FI uint8_t& VRAM(int a) { return m.B(OFF_VRAM + a); }
    FI uint8_t& OAM(int a) { return m.B(OFF_OAM + a); }
    FI uint8_t& PAL(int a) { return m.B(OFF_PAL + a); }
    FI uint8_t& SL(int a) { return m.B(OFF_SL + a); }
    FI int ntoff(int t) const { return t == 0 ? c.nt0 : t == 1 ? c.nt1 : t == 2 ? c.nt2 : c.nt3; }

    FI uint8_t rd_prg(uint16_t a) const { return __ldg(c.prg + (a & 0x7fff)); }
    FI uint8_t rd_prg_fast(uint16_t a) const { return a >= 0x8000 ? rd_prg(a) : 0xff; }

    // ---------------- init (Emulator(rom) + reset)
    FI void power_on() {
        for (int i = 0; i < SC_W0; i++) m.W(i) = 0;
        s.iter = s.exec_at = s.ppu_calls = s.nmi_thresh = 0;
        s.zero_skip = 0; s.pc = 0; s.sp = s.a = s.x = s.y = s.p = 0;
        s.joy_strobe0 = s.joy_strobe1 = 1; s.joy_buttons0 = s.joy_buttons1 = 0; s.joy_bits0 = s.joy_bits1 = 0;
        s.pstate = 0; s.even = s.vblank = s.s0hit = s.first_write = s.data_buffer = s.oam_addr = s.fine_x = 0;
        s.show_bg = s.show_spr = s.hide_edge_bg = s.hide_edge_spr = s.long_spr = s.nmi_en = s.bg_page = 0;
        s.spr_page = s.nsl = 0; s.pcyc = s.scanline = 0; s.v = s.t = s.inc = 0;
        reset();
    }
    FI void reset() {
        s.zero_skip = 0;
        s.a = s.x = s.y = 0;
        s.p = 0b00110100;
        s.pc = (uint16_t)(rd_prg(0xfffc) | rd_prg(0xfffd) << 8);
        s.sp = 0xfd;
        s.iter = 0; s.exec_at = 0; s.ppu_calls = 0;
        s.long_spr = 0; s.nmi_en = 0; s.vblank = 0; s.show_bg = 1; s.show_spr = 1; s.even = 1;
        s.first_write = 1; s.bg_page = 0; s.spr_page = 0; s.v = 0; s.pcyc = 0; s.scanline = 0;
        s.oam_addr = 0; s.fine_x = 0; s.t = 0; s.inc = 1; s.pstate = PRE_RENDER; s.nsl = 0;
        update_nmi_thresh();
    }

    // ---------------- CPU bus
    FI uint8_t rd(uint16_t a) {
        if (a >= 0x8000) return rd_prg(a);
        if (a < 0x2000) return RAM(a & 0x7ff);
        return rd_io(a);
    }
    __device__ __noinline__ uint8_t rd_io(uint16_t a) {
        if (a < 0x4000) {
            switch (a & 0x2007) {
                case 0x2002: sync(); return ppu_status();
                case 0x2007: sync(); return ppu_get_data();
                case 0x2004: return OAM(s.oam_addr);
                default: return 0;
            }
        }
        if (a == 0x4016) return joy_read(0);
        if (a == 0x4017) return joy_read(1);
        return 0;
    }
    FI void wr(uint16_t a, uint8_t v) {
        if (a < 0x2000) { RAM(a & 0x7ff) = v; return; }
        wr_io(a, v);
    }
    __device__ __noinline__ void wr_io(uint16_t a, uint8_t v) {
        if (a < 0x4000) {
            switch (a & 0x2007) {
                case 0x2000: sync(); ppu_control(v); tc_key = -1; break;
                case 0x2001: sync(); ppu_mask(v); break;
                case 0x2003: sync(); s.oam_addr = v; break;
                case 0x2004: sync(); OAM(s.oam_addr) = v; s.oam_addr++; tc_key = -1; break;
                case 0x2005: sync(); ppu_scroll(v); break;
                case 0x2006: sync(); ppu_addr(v); break;
                case 0x2007: sync(); ppu_set_data(v); tc_key = -1; break;
                default: break;
            }
            return;
        }
        if (a == 0x4014) {  // OAMDMA
            sync();
            skip += 513 + (int)((cur_iter + 1) & 1);
            uint16_t addr = (uint16_t)(v << 8);
            if (addr < 0x2000) {
                int base = addr & 0x7ff;
                if ((s.oam_addr & 3) == 0) {
                    int ow = s.oam_addr >> 2;
                    for (int j = 0; j < 64; j++) m.W((OFF_OAM >> 2) + ((ow + j) & 63)) = m.W((base >> 2) + j);
                } else {
                    for (int j = 0; j < 256; j++) OAM((s.oam_addr + j) & 255) = RAM(base + j);
                }
            }
            tc_key = -1;
            return;
        }
        if (a == 0x4016) {
            s.joy_strobe0 = v & 1;
            if (!s.joy_strobe0) s.joy_bits0 = s.joy_buttons0;
            s.joy_strobe1 = v & 1;
            if (!s.joy_strobe1) s.joy_bits1 = s.joy_buttons1;
        }
    }
    FI uint8_t joy_read(int i) {
        uint8_t r;
        if (i == 0) {
            if (s.joy_strobe0) r = s.joy_buttons0 & 1;
            else { r = s.joy_bits0 & 1; s.joy_bits0 >>= 1; }
        } else {
            if (s.joy_strobe1) r = s.joy_buttons1 & 1;
            else { r = s.joy_bits1 & 1; s.joy_bits1 >>= 1; }
        }
        return r | 0x40;
    }

    // ---------------- PPU registers
    FI void ppu_control(uint8_t c_) {
        s.nmi_en = (c_ & 0x80) != 0;
        s.long_spr = (c_ & 0x20) != 0;
        s.bg_page = (c_ & 0x10) ? 1 : 0;
        s.spr_page = (c_ & 0x8) ? 1 : 0;
        s.inc = (c_ & 0x4) ? 0x20 : 1;
        s.t = (uint16_t)((s.t & ~0xc00) | ((c_ & 0x3) << 10));
    }
    FI void ppu_mask(uint8_t mm) {
        s.hide_edge_bg = !(mm & 0x2);
        s.hide_edge_spr = !(mm & 0x4);
        s.show_bg = (mm & 0x8) != 0;
        s.show_spr = (mm & 0x10) != 0;
    }
    FI uint8_t ppu_status() {
        uint8_t st = (uint8_t)(s.s0hit << 6 | s.vblank << 7);
        s.vblank = 0;
        s.first_write = 1;
        return st;
    }
    FI void ppu_addr(uint8_t a) {
        if (s.first_write) {
            s.t = (uint16_t)((s.t & ~0xff00) | ((a & 0x3f) << 8));
            s.first_write = 0;
        } else {
            s.t = (uint16_t)((s.t & ~0xff) | a);
            s.v = s.t;
            s.first_write = 1;
        }
    }
    FI uint8_t ppu_get_data() {
        uint8_t d = pb_read(s.v);
        s.v = (uint16_t)(s.v + s.inc);
        if (s.v < 0x3f00) { uint8_t tmp = d; d = s.data_buffer; s.data_buffer = tmp; }
        return d;
    }
    FI void ppu_set_data(uint8_t d) {
        pb_write(s.v, d);
        s.v = (uint16_t)(s.v + s.inc);
    }
    FI void ppu_scroll(uint8_t sc) {
        if (s.first_write) {
            s.t = (uint16_t)((s.t & ~0x1f) | ((sc >> 3) & 0x1f));
            s.fine_x = sc & 0x7;
            s.first_write = 0;
        } else {
            s.t = (uint16_t)((s.t & ~0x73e0) | ((sc & 0x7) << 12) | ((sc & 0xf8) << 2));
            s.first_write = 1;
        }
    }

    // ---------------- picture bus
    FI uint8_t pb_read(uint16_t a) {
        if (a < 0x2000) return __ldg(c.chr + a);
        if (a < 0x3eff) {
            int t = a < 0x2400 ? 0 : a < 0x2800 ? 1 : a < 0x2c00 ? 2 : 3;
            return VRAM(ntoff(t) + (a & 0x3ff));
        }
        if (a < 0x3fff) return PAL(a & 0x1f);
        return 0;
    }
    FI void pb_write(uint16_t a, uint8_t v) {
        if (a < 0x2000) return;
        if (a < 0x3eff) {
            int t = a < 0x2400 ? 0 : a < 0x2800 ? 1 : a < 0x2c00 ? 2 : 3;
            VRAM(ntoff(t) + (a & 0x3ff)) = v;
        } else if (a < 0x3fff) {
            if (a == 0x3f10) PAL(0) = v;
            else PAL(a & 0x1f) = v;
        }
    }
    FI uint8_t nt_read(uint16_t a) { return VRAM(ntoff((a >> 10) & 3) + (a & 0x3ff)); }

    // ---------------- PPU timing
    FI void update_nmi_thresh() {
        int line;
        switch (s.pstate) {
            case PRE_RENDER: line = -1; break;
            case RENDER: line = s.scanline; break;
            case POST_RENDER: line = 240; break;
            default: line = s.scanline; break;
        }
        int64_t P = (int64_t)(line + 1) * 341 + (s.pcyc - 1);
        const int64_t PN = 242LL * 341;
        int64_t d = P <= PN ? PN - P + 1 : (262LL * 341 - P) + PN + 1;
        s.nmi_thresh = s.ppu_calls + d - 8;
    }
    FI void sync_to(int64_t target) {
        if (target > s.ppu_calls) {
            ppu_run(target - s.ppu_calls);
            update_nmi_thresh();
        }
    }
    FI void sync() { sync_to(3 * (cur_iter + 1)); }

    FI void coarse_x_inc_n(int n) {
        if (n <= 0) return;
        int tot = (s.v & 0x1f) + n;
        s.v = (uint16_t)((s.v & ~0x1f) | (tot & 31));
        if ((tot >> 5) & 1) s.v ^= 0x0400;
    }
    FI int inc_count(int a, int b) const {
        int r = (7 - s.fine_x) & 7;
        return (b - r + 7) / 8 - (a - r + 7) / 8;
    }
    FI void fine_y_inc() {
        if ((s.v & 0x7000) != 0x7000) {
            s.v += 0x1000;
        } else {
            s.v &= ~0x7000;
            int y = (s.v & 0x03E0) >> 5;
            if (y == 29) { y = 0; s.v ^= 0x0800; }
            else if (y == 31) y = 0;
            else y += 1;
            s.v = (uint16_t)((s.v & ~0x03E0) | (y << 5));
        }
    }
    FI void hcopy() { s.v = (uint16_t)((s.v & ~0x41f) | (s.t & 0x41f)); }
    FI void vcopy() { s.v = (uint16_t)((s.v & ~0x7be0) | (s.t & 0x7be0)); }

    FI void sprite_eval() {
        int nsl = 0;
        int range = s.long_spr ? 16 : 8;
        for (int i = s.oam_addr / 4; i < 64; ++i) {
            int diff = s.scanline - OAM(i * 4);
            if (0 <= diff && diff < range) {
                SL(nsl++) = (uint8_t)i;
                if (nsl >= 8) break;
            }
        }
        s.nsl = (uint8_t)nsl;
    }

    FI uint8_t bg_pixel(int x, uint8_t* opaque) {
        int x_fine = (s.fine_x + x) % 8;
        uint16_t addr = (uint16_t)(0x2000 | (s.v & 0x0FFF));
        uint8_t tile = nt_read(addr);
        addr = (uint16_t)((tile * 16) + ((s.v >> 12) & 0x7));
        addr |= s.bg_page << 12;
        uint8_t cc = (__ldg(c.chr + addr) >> (7 ^ x_fine)) & 1;
        cc |= ((__ldg(c.chr + addr + 8) >> (7 ^ x_fine)) & 1) << 1;
        *opaque = cc;
        uint16_t aaddr = (uint16_t)(0x23C0 | (s.v & 0x0C00) | ((s.v >> 4) & 0x38) | ((s.v >> 2) & 0x07));
        uint8_t attribute = nt_read(aaddr);
        int shift = ((s.v >> 4) & 4) | (s.v & 2);
        cc |= ((attribute >> shift) & 0x3) << 2;
        return cc;
    }
    FI uint8_t spr_bits(int i, int x, int y) {
        uint8_t spr_x = OAM(i * 4 + 3);
        uint8_t spr_y = (uint8_t)(OAM(i * 4 + 0) + 1), tile = OAM(i * 4 + 1), attribute = OAM(i * 4 + 2);
        int length = s.long_spr ? 16 : 8;
        int x_shift = (x - spr_x) % 8, y_offset = (y - spr_y) % length;
        if ((attribute & 0x40) == 0) x_shift ^= 7;
        if ((attribute & 0x80) != 0) y_offset ^= (length - 1);
        uint16_t address;
        if (!s.long_spr) {
            address = (uint16_t)(tile * 16 + y_offset);
            if (s.spr_page) address += 0x1000;
        } else {
            y_offset = (y_offset & 7) | ((y_offset & 8) << 1);
            address = (uint16_t)((tile >> 1) * 32 + y_offset);
            address |= (tile & 1) << 12;
        }
        uint8_t cc = (pb_read(address) >> x_shift) & 1;
        cc |= ((pb_read((uint16_t)(address + 8)) >> x_shift) & 1) << 1;
        return cc;
    }

    // visible dots [x0, x1) without pixel output: only CPU-visible state
    FI void dots_fast(int x0, int x1) {
        if (!s.show_bg) return;
        if (!s.s0hit && s.show_spr && s.nsl > 0 && SL(0) == 0) {
            int sx = OAM(3);
            int lo = x0 > sx ? x0 : sx;
            int hi = x1 < sx + 8 ? x1 : sx + 8;
            if (s.hide_edge_spr && lo < 8) lo = 8;
            if (lo < hi) {
                coarse_x_inc_n(inc_count(x0, lo));
                int y = s.scanline;
                for (int x = lo; x < hi; x++) {
                    if (!s.s0hit && (!s.hide_edge_bg || x >= 8)) {
                        uint8_t op;
                        bg_pixel(x, &op);
                        if (op && spr_bits(0, x, y)) s.s0hit = 1;
                    }
                    if ((s.fine_x + x) % 8 == 7) coarse_x_inc_n(1);
                }
                coarse_x_inc_n(inc_count(hi, x1));
                return;
            }
        }
        coarse_x_inc_n(inc_count(x0, x1));
    }

    FI void fetch_tile() {
        int32_t key = (s.bg_page << 16) | s.v;
        if (key == tc_key) return;
        uint8_t tile = nt_read((uint16_t)(0x2000 | (s.v & 0x0FFF)));
        uint16_t addr = (uint16_t)(((tile * 16) + ((s.v >> 12) & 0x7)) | (s.bg_page << 12));
        tc_lo = __ldg(c.chr + addr);
        tc_hi = __ldg(c.chr + addr + 8);
        uint16_t aaddr = (uint16_t)(0x23C0 | (s.v & 0x0C00) | ((s.v >> 4) & 0x38) | ((s.v >> 2) & 0x07));
        int shift = ((s.v >> 4) & 4) | (s.v & 2);
        tc_attr = (uint8_t)(((nt_read(aaddr) >> shift) & 0x3) << 2);
        tc_key = key;
    }

    // visible dots with pixel output (same results as the CPU core's dots_render); writes gray to row[]
    __device__ __noinline__ void dots_render(int x0, int x1) {
        const int y = s.scanline;
        uint8_t pa[256];
        if (!s.show_bg) {
            for (int x = x0; x < x1; x++) pa[x] = 0;
        } else {
            int x = x0;
            while (x < x1) {
                int xf = (s.fine_x + x) & 7;
                int span = 8 - xf;
                if (span > x1 - x) span = x1 - x;
                fetch_tile();
                for (int j = 0; j < span; j++) {
                    int sh = 7 - (xf + j);
                    uint8_t b = (uint8_t)(((tc_lo >> sh) & 1) | (((tc_hi >> sh) & 1) << 1));
                    pa[x + j] = b ? (uint8_t)(b | tc_attr) : 0;
                }
                if (s.hide_edge_bg)
                    for (int j = 0; j < span && x + j < 8; j++) pa[x + j] = 0;
                if (xf + span == 8) coarse_x_inc_n(1);
                x += span;
            }
        }
        if (s.show_spr && s.nsl) {
            int xs = (s.hide_edge_spr && x0 < 8) ? 8 : x0;
            int nsl = s.nsl;
            for (int x = xs; x < x1; x++) {
                // first opaque sprite of the line list covering x (== the CPU core's sprite line buffer)
                uint8_t col = 0, fl = 0;
                for (int k = 0; k < nsl; k++) {
                    int i = SL(k);
                    int sx = OAM(i * 4 + 3);
                    if (x < sx || x >= sx + 8) continue;
                    uint8_t cc = spr_bits(i, x, y);
                    if (cc) {
                        uint8_t attribute = OAM(i * 4 + 2);
                        col = (uint8_t)(cc | 0x10 | ((attribute & 0x3) << 2));
                        fl = (uint8_t)((!(attribute & 0x20) ? 1 : 0) | (i == 0 ? 2 : 0));
                        break;
                    }
                }
                if (!col) continue;
                uint8_t bgOpaque = pa[x] & 3;
                if ((fl & 2) && bgOpaque && !s.s0hit && s.show_bg) s.s0hit = 1;
                if (!bgOpaque || (fl & 1)) pa[x] = col;
            }
        }
        uint8_t* dst = (y >= gmin && y <= gmax) ? dbuf[y - gmin] : row;
        for (int x = x0; x < x1; x++) dst[x] = __ldg(rs->lut + PAL(pa[x]));
        if (screen) for (int x = x0; x < x1; x++) screen[y * 256 + x] = PAL(pa[x]);
        if (x1 == 256 && obs && y != spec0 && y != spec1) finish_row(y, dst);
    }

    // INTER_AREA 256x240 -> 84x84, streamed by source row (same float op order as cv2 / the CPU core)
    FI float hpass(const uint8_t* r, int dx) {
        float b = 0.f;
        for (int k = 0; k < RM; k++) {
            float w = __ldg(rs->xw + k * RD + dx);
            if (w == 0.f) break;
            b += (float)r[__ldg(rs->xs + k * RD + dx)] * w;
        }
        return b;
    }
    FI void put(int dy, int dx, float sm) {
        int iv = __float2int_rn(sm);
        obs[dy * RD + dx] = (uint8_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv);
    }
    __device__ __noinline__ void finish_row(int y, const uint8_t* r) {
        int e0 = __ldg(rs->ye + 2 * y), e1 = __ldg(rs->ye + 2 * y + 1);
        float w0 = __ldg(rs->yw + 2 * y), w1 = __ldg(rs->yw + 2 * y + 1);
        bool u0 = e0 >= 0 && ((e0 & 0xff) < dmin || (e0 & 0xff) > dmax);
        bool u1 = e1 >= 0 && ((e1 & 0xff) < dmin || (e1 & 0xff) > dmax);
        if (!u0 && !u1) return;
        for (int dx = 0; dx < RD; dx++) {
            float b = hpass(r, dx);
            if (u0) {
                float v = w0 * b;
                float sm = (e0 & 0x100) ? v : sum[dx] + v;
                if (e0 & 0x200) put(e0 & 0xff, dx, sm);
                sum[dx] = sm;
            }
            if (u1) {
                float v = w1 * b;
                float sm = (e1 & 0x100) ? v : sum[dx] + v;
                if (e1 & 0x200) put(e1 & 0xff, dx, sm);
                sum[dx] = sm;
            }
        }
    }
    // frame start (render frames): find the line(s) that may be re-rendered at the frame end
    FI void render_begin() {
        spec0 = spec1 = -1;
        if (s.pstate == RENDER) { spec0 = s.scanline; if (s.scanline + 1 < 240) spec1 = s.scanline + 1; }
        else if (s.pstate == PRE_RENDER) spec0 = 0;
        dmin = 1000; dmax = -1; gmin = 1000; gmax = -1;
        for (int q = 0; q < 2; q++) {
            int y = q ? spec1 : spec0;
            if (y < 0) continue;
            for (int j = 0; j < 2; j++) {
                int e = __ldg(rs->ye + 2 * y + j);
                if (e < 0) continue;
                int d = e & 0xff;
                if (d < dmin) dmin = d;
                if (d > dmax) dmax = d;
            }
        }
        if (dmax >= 0) { gmin = __ldg(rs->gr + 2 * dmin); gmax = __ldg(rs->gr + 2 * dmax + 1); }
    }
    __device__ __noinline__ void render_end() {
        if (!obs || dmax < 0) return;
        for (int dy = dmin; dy <= dmax; dy++) {
            int r0 = __ldg(rs->gr + 2 * dy), r1 = __ldg(rs->gr + 2 * dy + 1);
            for (int dx = 0; dx < RD; dx++) {
                float sm = 0.f;
                for (int y = r0; y <= r1; y++) {
                    int j = ((__ldg(rs->ye + 2 * y) & 0xff) == dy && __ldg(rs->ye + 2 * y) >= 0) ? 0 : 1;
                    float v = __ldg(rs->yw + 2 * y + j) * hpass(dbuf[y - gmin], dx);
                    sm = (y == r0) ? v : sm + v;
                }
                put(dy, dx, sm);
            }
        }
    }

    FI void dots(int x0, int x1) {
        if (render) dots_render(x0, x1);
        else dots_fast(x0, x1);
    }

    FI void push(uint8_t v) { RAM((0x100 | s.sp) & 0x7ff) = v; s.sp--; }
    FI uint8_t pop() { ++s.sp; return RAM((0x100 | s.sp) & 0x7ff); }

    __device__ __noinline__ void nmi(int64_t call_index) {
        int64_t mm = (call_index - 1) / 3;
        push((uint8_t)(s.pc >> 8));
        push((uint8_t)s.pc);
        push((uint8_t)(s.p | 0b00100000));
        s.p |= F_I;
        s.pc = (uint16_t)(rd_prg(0xfffa) | rd_prg(0xfffb) << 8);
        int64_t v = (s.zero_skip && mm == s.exec_at) ? 0 : (s.exec_at - mm + 1);
        s.exec_at = mm + v + 6;
        s.zero_skip = 0;
    }

    __device__ __noinline__ void ppu_run(int64_t n) {
        while (n > 0) {
            int cc = s.pcyc;
            switch (s.pstate) {
                case PRE_RENDER: {
                    bool rend = s.show_bg && s.show_spr;
                    int end = SCANLINE_END_CYCLE - (!s.even && rend);
                    int64_t k = cc >= end ? 1 : (n < end - cc ? n : end - cc);
                    int a = cc, b = (int)(cc + k);
                    if (a <= 1 && 1 < b) s.vblank = s.s0hit = 0;
                    if (rend) {
                        if (a <= 258 && 258 < b) hcopy();
                        if (a <= 304 && 281 < b) vcopy();
                    }
                    s.ppu_calls += k; n -= k;
                    if (b - 1 >= end) { s.pstate = RENDER; s.scanline = 0; s.pcyc = 1; }
                    else s.pcyc = b;
                    break;
                }
                case RENDER: {
                    if (cc >= 1 && cc <= 256) {
                        int64_t k = n < 257 - cc ? n : 257 - cc;
                        dots(cc - 1, (int)(cc - 1 + k));
                        s.pcyc += (int)k; s.ppu_calls += k; n -= k;
                    } else if (cc == 257) {
                        if (s.show_bg) fine_y_inc();
                        s.pcyc++; s.ppu_calls++; n--;
                    } else if (cc == 258) {
                        if (s.show_bg && s.show_spr) hcopy();
                        s.pcyc++; s.ppu_calls++; n--;
                    } else if (cc > 258 && cc < SCANLINE_END_CYCLE) {
                        int64_t k = n < SCANLINE_END_CYCLE - cc ? n : SCANLINE_END_CYCLE - cc;
                        s.pcyc += (int)k; s.ppu_calls += k; n -= k;
                    } else if (cc >= SCANLINE_END_CYCLE) {
                        sprite_eval();
                        ++s.scanline;
                        s.pcyc = 1; s.ppu_calls++; n--;
                        if (s.scanline >= VISIBLE_SCANLINES) s.pstate = POST_RENDER;
                    } else {
                        s.pcyc++; s.ppu_calls++; n--;
                    }
                    break;
                }
                case POST_RENDER: {
                    if (cc >= SCANLINE_END_CYCLE) {
                        ++s.scanline; s.pcyc = 1; s.pstate = VERTICAL_BLANK;
                        s.ppu_calls++; n--;
                    } else {
                        int64_t k = n < SCANLINE_END_CYCLE - cc ? n : SCANLINE_END_CYCLE - cc;
                        s.pcyc += (int)k; s.ppu_calls += k; n -= k;
                    }
                    break;
                }
                default: {
                    if (cc >= SCANLINE_END_CYCLE) {
                        ++s.scanline;
                        s.pcyc = 1;
                        if (s.scanline >= FRAME_END_SCANLINE) {
                            s.pstate = PRE_RENDER; s.scanline = 0; s.even = !s.even;
                        }
                        s.ppu_calls++; n--;
                    } else if (cc == 1 && s.scanline == 241) {
                        s.vblank = 1;
                        s.ppu_calls++; n--;
                        s.pcyc = 2;
                        if (s.nmi_en) nmi(s.ppu_calls);
                    } else {
                        int lim = SCANLINE_END_CYCLE - cc;
                        if (s.scanline == 241 && cc < 1) lim = 1 - cc;
                        int64_t k = n < lim ? n : lim;
                        s.pcyc += (int)k; s.ppu_calls += k; n -= k;
                    }
                    break;
                }
            }
        }
    }

    // ---------------- CPU
    FI void set_ZN(uint8_t v) { s.p = (uint8_t)((s.p & ~(F_Z | F_N)) | (v ? 0 : F_Z) | ((v & 0x80) ? F_N : 0)); }
    FI void setf(uint8_t f, bool on) { s.p = on ? (s.p | f) : (s.p & ~f); }
    FI bool getf(uint8_t f) const { return (s.p & f) != 0; }
    FI uint16_t read_address(uint16_t a) {
        uint8_t lo = rd(a);
        uint8_t hi = rd((uint16_t)(a + 1));
        return (uint16_t)(lo | hi << 8);
    }
    FI void page_crossed(uint16_t a, uint16_t b, int inc = 1) {
        if ((a & 0xff00) != (b & 0xff00)) skip += inc;
    }

    FI bool implied(uint8_t op) {
        switch (op) {
            case 0x00:  // BRK
                ++s.pc;
                push((uint8_t)(s.pc >> 8));
                push((uint8_t)s.pc);
                push((uint8_t)(s.p | 0b00100000 | (1 << 4)));
                s.p |= F_I;
                s.pc = read_address(0xfffe);
                skip += 7;
                break;
            case 0x08: push(s.p); break;
            case 0x18: setf(F_C, false); break;
            case 0x20:
                push((uint8_t)((s.pc + 1) >> 8));
                push((uint8_t)(s.pc + 1));
                s.pc = read_address(s.pc);
                break;
            case 0x28: s.p = pop(); break;
            case 0x38: setf(F_C, true); break;
            case 0x40: s.p = pop(); s.pc = pop(); s.pc |= pop() << 8; break;
            case 0x48: push(s.a); break;
            case 0x4C: s.pc = read_address(s.pc); break;
            case 0x58: setf(F_I, false); break;
            case 0x60: s.pc = pop(); s.pc |= pop() << 8; ++s.pc; break;
            case 0x68: s.a = pop(); set_ZN(s.a); break;
            case 0x6C: {
                uint16_t loc = read_address(s.pc);
                uint16_t page = loc & 0xff00;
                uint8_t lo = rd(loc);
                uint8_t hi = rd((uint16_t)(page | ((loc + 1) & 0xff)));
                s.pc = (uint16_t)(lo | hi << 8);
                break;
            }
            case 0x78: setf(F_I, true); break;
            case 0x88: --s.y; set_ZN(s.y); break;
            case 0x8a: s.a = s.x; set_ZN(s.a); break;
            case 0x98: s.a = s.y; set_ZN(s.a); break;
            case 0x9a: s.sp = s.x; break;
            case 0xa8: s.y = s.a; set_ZN(s.y); break;
            case 0xaa: s.x = s.a; set_ZN(s.x); break;
            case 0xb8: setf(F_V, false); break;
            case 0xba: s.x = s.sp; set_ZN(s.x); break;
            case 0xc8: ++s.y; set_ZN(s.y); break;
            case 0xca: --s.x; set_ZN(s.x); break;
            case 0xd8: setf(F_D, false); break;
            case 0xe8: ++s.x; set_ZN(s.x); break;
            case 0xea: break;
            case 0xf8: setf(F_D, true); break;
            default: return false;
        }
        return true;
    }

    FI bool branch(uint8_t op) {
        if ((op & 0x1f) != 0x10) return false;
        bool br = op & 0x20;
        switch (op >> 6) {
            case 0: br = !(br ^ getf(F_N)); break;
            case 1: br = !(br ^ getf(F_V)); break;
            case 2: br = !(br ^ getf(F_C)); break;
            default: br = !(br ^ getf(F_Z)); break;
        }
        if (br) {
            int8_t offset = (int8_t)rd(s.pc++);
            ++skip;
            uint16_t newPC = (uint16_t)(s.pc + offset);
            page_crossed(s.pc, newPC, 2);
            s.pc = newPC;
        } else {
            ++s.pc;
        }
        return true;
    }

    FI bool type0(uint8_t op) {
        if ((op & 0x3) != 0x0) return false;
        uint16_t location = 0;
        switch ((op & 0x1c) >> 2) {
            case 0: location = s.pc++; break;
            case 1: location = rd(s.pc++); break;
            case 3: location = read_address(s.pc); s.pc += 2; break;
            case 5: location = (rd(s.pc++) + s.x) & 0xff; break;
            case 7:
                location = read_address(s.pc); s.pc += 2;
                page_crossed(location, (uint16_t)(location + s.x));
                location = (uint16_t)(location + s.x);
                break;
            default: return false;
        }
        switch ((op & 0xe0) >> 5) {
            case 1: {
                uint16_t operand = rd(location);
                setf(F_Z, !(s.a & operand));
                setf(F_V, operand & 0x40);
                setf(F_N, operand & 0x80);
                break;
            }
            case 4: wr(location, s.y); break;
            case 5: s.y = rd(location); set_ZN(s.y); break;
            case 6: {
                uint16_t diff = (uint16_t)(s.y - rd(location));
                setf(F_C, !(diff & 0x100));
                set_ZN((uint8_t)diff);
                break;
            }
            case 7: {
                uint16_t diff = (uint16_t)(s.x - rd(location));
                setf(F_C, !(diff & 0x100));
                set_ZN((uint8_t)diff);
                break;
            }
            default: return false;
        }
        return true;
    }

    FI bool type1(uint8_t op) {
        if ((op & 0x3) != 0x1) return false;
        uint16_t location = 0;
        int opn = (op & 0xe0) >> 5;
        const int STA = 4;
        switch ((op & 0x1c) >> 2) {
            case 0: {
                uint8_t za = (uint8_t)(s.x + rd(s.pc++));
                uint8_t lo = rd(za & 0xff);
                uint8_t hi = rd((za + 1) & 0xff);
                location = (uint16_t)(lo | hi << 8);
                break;
            }
            case 1: location = rd(s.pc++); break;
            case 2: location = s.pc++; break;
            case 3: location = read_address(s.pc); s.pc += 2; break;
            case 4: {
                uint8_t za = rd(s.pc++);
                uint8_t lo = rd(za & 0xff);
                uint8_t hi = rd((za + 1) & 0xff);
                location = (uint16_t)(lo | hi << 8);
                if (opn != STA) page_crossed(location, (uint16_t)(location + s.y));
                location = (uint16_t)(location + s.y);
                break;
            }
            case 5: location = (rd(s.pc++) + s.x) & 0xff; break;
            case 6:
                location = read_address(s.pc); s.pc += 2;
                if (opn != STA) page_crossed(location, (uint16_t)(location + s.y));
                location = (uint16_t)(location + s.y);
                break;
            default:
                location = read_address(s.pc); s.pc += 2;
                if (opn != STA) page_crossed(location, (uint16_t)(location + s.x));
                location = (uint16_t)(location + s.x);
                break;
        }
        switch (opn) {
            case 0: s.a |= rd(location); set_ZN(s.a); break;
            case 1: s.a &= rd(location); set_ZN(s.a); break;
            case 2: s.a ^= rd(location); set_ZN(s.a); break;
            case 3: {
                uint8_t operand = rd(location);
                uint16_t sum_ = (uint16_t)(s.a + operand + getf(F_C));
                setf(F_C, sum_ & 0x100);
                setf(F_V, (s.a ^ sum_) & (operand ^ sum_) & 0x80);
                s.a = (uint8_t)sum_;
                set_ZN(s.a);
                break;
            }
            case 4: wr(location, s.a); break;
            case 5: s.a = rd(location); set_ZN(s.a); break;
            case 6: {
                uint16_t diff = (uint16_t)(s.a - rd(location));
                setf(F_C, !(diff & 0x100));
                set_ZN((uint8_t)diff);
                break;
            }
            default: {
                uint16_t subtrahend = rd(location);
                uint16_t diff = (uint16_t)(s.a - subtrahend - !getf(F_C));
                setf(F_C, !(diff & 0x100));
                setf(F_V, (s.a ^ diff) & (~subtrahend ^ diff) & 0x80);
                s.a = (uint8_t)diff;
                set_ZN((uint8_t)diff);
                break;
            }
        }
        return true;
    }

    FI bool type2(uint8_t op) {
        if ((op & 0x3) != 2) return false;
        uint16_t location = 0;
        int opn = (op & 0xe0) >> 5;
        int mode = (op & 0x1c) >> 2;
        const int ASL = 0, ROL = 1, LSR = 2, ROR = 3, STX = 4, LDX = 5, DEC = 6;
        switch (mode) {
            case 0: location = s.pc++; break;
            case 1: location = rd(s.pc++); break;
            case 2: break;
            case 3: location = read_address(s.pc); s.pc += 2; break;
            case 5: {
                location = rd(s.pc++);
                uint8_t index = (opn == LDX || opn == STX) ? s.y : s.x;
                location = (location + index) & 0xff;
                break;
            }
            case 7: {
                location = read_address(s.pc); s.pc += 2;
                uint8_t index = (opn == LDX || opn == STX) ? s.y : s.x;
                page_crossed(location, (uint16_t)(location + index));
                location = (uint16_t)(location + index);
                break;
            }
            default: return false;
        }
        uint16_t operand = 0;
        switch (opn) {
            case ASL:
            case ROL:
                if (mode == 2) {
                    bool prev_C = getf(F_C);
                    setf(F_C, s.a & 0x80);
                    s.a <<= 1;
                    s.a = (uint8_t)(s.a | (prev_C && (opn == ROL)));
                    set_ZN(s.a);
                } else {
                    bool prev_C = getf(F_C);
                    operand = rd(location);
                    setf(F_C, operand & 0x80);
                    operand = (uint16_t)(operand << 1 | (prev_C && (opn == ROL)));
                    set_ZN((uint8_t)operand);
                    wr(location, (uint8_t)operand);
                }
                break;
            case LSR:
            case ROR:
                if (mode == 2) {
                    bool prev_C = getf(F_C);
                    setf(F_C, s.a & 1);
                    s.a >>= 1;
                    s.a = (uint8_t)(s.a | (prev_C && (opn == ROR)) << 7);
                    set_ZN(s.a);
                } else {
                    bool prev_C = getf(F_C);
                    operand = rd(location);
                    setf(F_C, operand & 1);
                    operand = (uint16_t)(operand >> 1 | (prev_C && (opn == ROR)) << 7);
                    set_ZN((uint8_t)operand);
                    wr(location, (uint8_t)operand);
                }
                break;
            case STX: wr(location, s.x); break;
            case LDX: s.x = rd(location); set_ZN(s.x); break;
            case DEC: { int tmp = rd(location) - 1; set_ZN((uint8_t)tmp); wr(location, (uint8_t)tmp); break; }
            default:  { int tmp = rd(location) + 1; set_ZN((uint8_t)tmp); wr(location, (uint8_t)tmp); break; }
        }
        return true;
    }

    FI int execute() {
        skip = 0;
        uint8_t op = rd(s.pc++);
        if (implied(op) || branch(op) || type1(op) || type2(op) || type0(op)) skip += OPERATION_CYCLES[op];
        else bad_ops++;
        return skip;
    }

    // One nes-py Emulator::step(): 29781 CPU iterations, 3 PPU calls each.
    __device__ __noinline__ void run_frame() {
        const int64_t end = s.iter + CYCLES_PER_FRAME;
        tc_key = -1;
        if (render) render_begin();
        else { spec0 = spec1 = -1; gmin = dmin = 1000; gmax = dmax = -1; }
        while (s.exec_at < end) {
            const int64_t J = s.exec_at;
            if (3 * (J + 1) >= s.nmi_thresh) {
                cur_iter = J;
                sync_to(3 * (J + 1));
                if (s.exec_at != J) continue;
            }
            if (rd_prg_fast(s.pc) == 0x4C && (uint16_t)(rd_prg_fast(s.pc + 1) | rd_prg_fast(s.pc + 2) << 8) == s.pc) {
                int64_t lim_end = end - J;
                int64_t lim_nmi = s.nmi_thresh - 3 * (J + 1);
                int64_t k1 = lim_end > 0 ? (lim_end + 2) / 3 : 0;
                int64_t k2 = lim_nmi > 0 ? (lim_nmi + 8) / 9 : 0;
                int64_t k = k1 < k2 ? k1 : k2;
                if (k > 1) {
                    s.exec_at = J + 3 * (k - 1);
                    s.zero_skip = 0;
                    continue;
                }
            }
            cur_iter = J;
            int S = execute();
            s.exec_at = J + (S >= 1 ? S : 1);
            s.zero_skip = S <= 0;
        }
        sync_to(3 * end);
        s.iter = end;
        if (render) render_end();
    }

    // save / restore full state (scalars from registers) to / from another interleaved buffer
    FI void save_to(const Mem& d) { copy_bytes(d, m); store_sc(s, d); }
    FI void restore_from(const Mem& src) { copy_bytes(m, src); load_sc(s, src); }
};

// ------------------------------------------------------------------ numpy PCG64
struct Pcg64 {
    unsigned __int128 state, inc;
    int has32;
    uint32_t uinteger;
    FI uint64_t next64() {
        const unsigned __int128 M = ((unsigned __int128)0x2360ED051FC65DA4ULL << 64) | 0x4385DF649FCCF645ULL;
        state = state * M + inc;
        uint64_t hi = (uint64_t)(state >> 64), lo = (uint64_t)state;
        unsigned rot = (unsigned)(state >> 122);
        uint64_t x = hi ^ lo;
        return (x >> rot) | (x << ((64 - rot) & 63));
    }
    FI uint32_t next32() {
        if (has32) { has32 = 0; return uinteger; }
        uint64_t n = next64();
        has32 = 1;
        uinteger = (uint32_t)(n >> 32);
        return (uint32_t)(n & 0xffffffff);
    }
    FI double random() { return (double)(next64() >> 11) * (1.0 / 9007199254740992.0); }
    FI int64_t integers(int64_t high) {
        uint64_t rng = (uint64_t)(high - 1);
        if (rng == 0) return 0;
        if (rng == 0xFFFFFFFFULL) return next32();
        uint32_t rng_excl = (uint32_t)rng + 1;
        uint64_t mm = (uint64_t)next32() * rng_excl;
        uint32_t leftover = (uint32_t)(mm & 0xffffffff);
        if (leftover < rng_excl) {
            uint32_t threshold = (uint32_t)((0xFFFFFFFFu - (uint32_t)rng) % rng_excl);
            while (leftover < threshold) {
                mm = (uint64_t)next32() * rng_excl;
                leftover = (uint32_t)(mm & 0xffffffff);
            }
        }
        return (int64_t)(mm >> 32);
    }
};

// ------------------------------------------------------------------ SMB env layer
__constant__ uint8_t SIMPLE_MOVEMENT[7] = {0x00, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40};

FI int64_t digits(Nes& n, int a, int cnt) {
    int64_t v = 0;
    for (int i = 0; i < cnt; i++) {
        int b = n.RAM(a + i);
        v = v * (b >= 100 ? 1000 : b >= 10 ? 100 : 10) + b;
    }
    return v;
}
FI int64_t smb_x(Nes& n) { return (int64_t)n.RAM(0x6d) * 0x100 + n.RAM(0x86); }
FI int64_t smb_time(Nes& n) { return digits(n, 0x7f8, 3); }
FI bool smb_busy(Nes& n) { int s = n.RAM(0x0e); return s <= 5 || s == 7; }
FI bool smb_world_over(Nes& n) { return n.RAM(0x0770) == 2; }
FI bool smb_flag(Nes& n) {
    if (smb_world_over(n)) return true;
    for (int a = 0x16; a <= 0x1A; a++) {
        int r = n.RAM(a);
        if (r == 0x2D || r == 0x31) return n.RAM(0x001D) == 3;
    }
    return false;
}
#define N_INFO 10
struct Info { int x_pos, flag_get, life, score, time, world, stage, coins, y_pos, status; };
FI void smb_info(Nes& n, Info& i) {
    i.coins = (int)digits(n, 0x7ed, 2);
    i.flag_get = smb_flag(n);
    i.life = n.RAM(0x075a);
    i.score = (int)digits(n, 0x7de, 6);
    i.stage = n.RAM(0x075c) + 1;
    int st = n.RAM(0x0756);
    i.status = st == 0 ? 0 : st == 1 ? 1 : 2;
    i.time = (int)smb_time(n);
    i.world = n.RAM(0x075f) + 1;
    i.x_pos = (int)smb_x(n);
    int yp = n.RAM(0x03b8);
    i.y_pos = n.RAM(0x00b5) < 1 ? 255 + (255 - yp) : 255 - yp;
}

struct Cfg {
    int skip, max_steps, noop_max;
    double sticky;
};

struct EnvS {  // env-layer scalars
    int64_t x_last, time_last;
    int prev, lives, t, ep_len;
    double ep_ret;
    bool obs_is_backup;
    Info info;
};

FI void emu_frame(Nes& n, EnvS& E, uint8_t a, bool render) {
    n.s.joy_buttons0 = a;
    n.render = render;
    n.run_frame();
    if (render) E.obs_is_backup = false;
}
FI void hacks(Nes& n, EnvS& E, bool render) {
    uint8_t cat = n.RAM(0x06DE);
    if (cat > 1 && cat < 255) n.RAM(0x06DE) = 1;
    for (;;) {
        if (!(smb_busy(n) || smb_world_over(n))) break;
        n.RAM(0x07A0) = 0;
        emu_frame(n, E, 0, render);
    }
}
FI double frame_step(Nes& n, EnvS& E, uint8_t a, bool render, bool* done) {
    emu_frame(n, E, a, render);
    int64_t x = smb_x(n);
    int64_t xr = x - E.x_last; E.x_last = x;
    if (xr < -5 || xr > 5) xr = 0;
    int64_t tm = smb_time(n);
    int64_t tp = tm - E.time_last; E.time_last = tm;
    if (tp > 0) tp = 0;
    int ps = n.RAM(0x0e);
    bool dying = ps == 0x0b || n.RAM(0x00b5) > 1, dead = ps == 0x06;
    int64_t dp = (dying || dead) ? -25 : 0;
    int64_t r = xr + tp + dp;
    if (r < -15) r = -15;
    if (r > 15) r = 15;
    smb_info(n, E.info);
    *done = dying || dead || E.info.flag_get;
    if (!*done) hacks(n, E, render);
    return (double)r;
}
// reference _agent_step (no max-pool). snap: scratch state used to re-render on early termination.
// draws: incremented by the number of sticky random() draws (frames iterated).
FI double agent_step(Nes& n, EnvS& E, Pcg64& rng, const Cfg& cfg, int a, bool want_obs, bool* term,
                     const Mem& snap, int* draws) {
    const int K = cfg.skip;
    double total = 0.0;
    bool done = false;
    uint8_t acts[16];
    if (want_obs) n.save_to(snap);
    int i = 0;
    for (i = 0; i < K; i++) {
        int ae = a;
        (*draws)++;
        if (cfg.sticky != 0.0 && rng.random() < cfg.sticky) ae = E.prev;
        E.prev = ae;
        bool render = want_obs && (i == K - 1);
        acts[i] = (uint8_t)ae;
        total += frame_step(n, E, SIMPLE_MOVEMENT[ae], render, &done);
        if (done) break;
    }
    int last = done ? i : K - 1;
    if (want_obs && done && last < K - 1) {
        // early termination: replay from the step-start snapshot, rendering frame `last`
        n.restore_from(snap);
        for (int j = 0; j < last; j++) {
            emu_frame(n, E, SIMPLE_MOVEMENT[acts[j]], false);
            hacks(n, E, false);
        }
        emu_frame(n, E, SIMPLE_MOVEMENT[acts[last]], true);
    }
    bool t_ = done;
    int life = E.info.life;
    if (E.lives >= 0 && life < E.lives) t_ = true;
    E.lives = life;
    if (E.info.flag_get) t_ = true;
    *term = t_;
    return total;
}

FI void setup_nes(Nes& n, uint32_t* base, int e, int np, const uint8_t* cart, const int* nt, const Rs* rs) {
    n.m = mem_of(base, e, np);
    n.c.prg = cart; n.c.chr = cart + 0x8000;
    n.c.nt0 = nt[0]; n.c.nt1 = nt[1]; n.c.nt2 = nt[2]; n.c.nt3 = nt[3];
    n.render = false; n.cur_iter = 0; n.skip = 0; n.bad_ops = 0; n.tc_key = -1;
    n.tc_lo = n.tc_hi = n.tc_attr = 0;
    n.rs = rs;
    n.obs = 0;
    n.screen = 0;
}

// ------------------------------------------------------------------ kernels
// level table entries: world, stage, area
// boot: one thread per level; writes backup state (interleaved, stride nl), x_last0/time_last0, obs0
extern "C" __global__ void k_boot(int nl, const int* lv, uint32_t* bstate, int64_t* bx, int64_t* bt, uint8_t* obs0,
                                  const uint8_t* cart, const int* nt, const int* xs, const float* xw,
                                  const int* ye, const float* yw, const uint8_t* lut, const int* gr) {
    int L = blockIdx.x * blockDim.x + threadIdx.x;
    if (L >= nl) return;
    Rs rs{xs, xw, ye, yw, lut, gr};
    Nes n;
    setup_nes(n, bstate, L, nl, cart, nt, &rs);
    n.obs = obs0 + (size_t)L * 84 * 84;
    n.power_on();
    int w = lv[3 * L], st = lv[3 * L + 1], ar = lv[3 * L + 2];
    EnvS E;
    auto fa = [&](uint8_t a) { n.s.joy_buttons0 = a; n.render = true; n.run_frame(); };
    fa(8); fa(0);
    while (smb_time(n) == 0) {
        fa(8);
        n.RAM(0x075f) = (uint8_t)(w - 1); n.RAM(0x075c) = (uint8_t)(st - 1); n.RAM(0x0760) = (uint8_t)(ar - 1);
        fa(0);
        n.RAM(0x07A0) = 0;
    }
    int64_t tl = smb_time(n);
    while (smb_time(n) >= tl) { tl = smb_time(n); fa(8); fa(0); }
    store_sc(n.s, n.m);
    bx[L] = smb_x(n);
    bt[L] = smb_time(n);
    (void)E;
}

// reset cache: entry c = L * (noop_max + 1) + k : state after the reference reset() drew k noops.
extern "C" __global__ void k_cache(int nl, int noop_max, int skip, double sticky, int max_steps,
                                   const uint32_t* bstate, const int64_t* bx, const int64_t* bt, const uint8_t* obs0,
                                   uint32_t* cstate, uint32_t* cscratch, int64_t* cx, int64_t* ct, int* clives,
                                   int* cdraws, uint8_t* cobs,
                                   const uint8_t* cart, const int* nt, const int* xs, const float* xw,
                                   const int* ye, const float* yw, const uint8_t* lut, const int* gr) {
    int nk = noop_max + 1;
    int nc = nl * nk;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    int L = c / nk, k = c % nk;
    Rs rs{xs, xw, ye, yw, lut, gr};
    Nes n;
    setup_nes(n, cstate, c, nc, cart, nt, &rs);
    n.obs = cobs + (size_t)c * 84 * 84;
    Mem bm = mem_of((uint32_t*)bstate, L, nl);
    Mem snap = mem_of(cscratch, c, nc);
    Cfg cfg{skip, max_steps, noop_max, 0.0};  // noop action 0 with prev 0: sticky never changes it
    Pcg64 rng; rng.state = 0; rng.inc = 0; rng.has32 = 0; rng.uinteger = 0;
    EnvS E;
    n.restore_from(bm);
    E.x_last = bx[L]; E.time_last = bt[L]; E.obs_is_backup = true; E.prev = 0; E.lives = -1;
    int draws = 0;
    for (int j = 0; j < k; j++) {
        bool term;
        agent_step(n, E, rng, cfg, 0, j == k - 1, &term, snap, &draws);
        if (term) {
            n.restore_from(bm);
            E.x_last = bx[L]; E.time_last = bt[L]; E.obs_is_backup = true; E.prev = 0; E.lives = -1;
        }
    }
    store_sc(n.s, n.m);
    cx[c] = E.x_last; ct[c] = E.time_last; clives[c] = E.lives; cdraws[c] = draws;
    if (E.obs_is_backup)
        for (int i = 0; i < 84 * 84; i++) cobs[(size_t)c * 84 * 84 + i] = obs0[(size_t)L * 84 * 84 + i];
    (void)sticky;
}

// Env scalars (SoA)
struct EnvArrays {
    int64_t* x_last; int64_t* time_last;
    int* prev; int* lives; int* t; int* ep_len; double* ep_ret;
    uint64_t* rng;  // 6 per env: state_hi, state_lo, inc_hi, inc_lo, has32, uinteger
    int* level;
    int64_t* frames;
};

FI void do_reset(Nes& n, EnvS& E, Pcg64& rng, const Cfg& cfg, int L, const uint32_t* cstate, int nc,
                 const int64_t* cx, const int64_t* ct, const int* clives, const int* cdraws, int* reset_idx, int e) {
    int nk = cfg.noop_max + 1;
    int k = 0;
    if (cfg.noop_max) k = (int)rng.integers(cfg.noop_max + 1);
    int c = L * nk + k;
    if (cfg.sticky != 0.0) {
        int d = cdraws[c];
        for (int j = 0; j < d; j++) rng.next64();
    }
    n.restore_from(mem_of((uint32_t*)cstate, c, nc));
    E.x_last = cx[c]; E.time_last = ct[c]; E.lives = clives[c];
    E.prev = 0; E.t = 0; E.ep_ret = 0.0; E.ep_len = 0;
    reset_idx[e] = c;
}

FI void load_env(EnvS& E, Pcg64& rng, const EnvArrays& A, int e) {
    E.x_last = A.x_last[e]; E.time_last = A.time_last[e];
    E.prev = A.prev[e]; E.lives = A.lives[e]; E.t = A.t[e]; E.ep_len = A.ep_len[e]; E.ep_ret = A.ep_ret[e];
    E.obs_is_backup = false;
    const uint64_t* r = A.rng + 6 * (size_t)e;
    rng.state = ((unsigned __int128)r[0] << 64) | r[1];
    rng.inc = ((unsigned __int128)r[2] << 64) | r[3];
    rng.has32 = (int)r[4]; rng.uinteger = (uint32_t)r[5];
}
FI void store_env(const EnvS& E, const Pcg64& rng, const EnvArrays& A, int e) {
    A.x_last[e] = E.x_last; A.time_last[e] = E.time_last;
    A.prev[e] = E.prev; A.lives[e] = E.lives; A.t[e] = E.t; A.ep_len[e] = E.ep_len; A.ep_ret[e] = E.ep_ret;
    uint64_t* r = A.rng + 6 * (size_t)e;
    r[0] = (uint64_t)(rng.state >> 64); r[1] = (uint64_t)rng.state;
    r[2] = (uint64_t)(rng.inc >> 64); r[3] = (uint64_t)rng.inc;
    r[4] = (uint64_t)rng.has32; r[5] = rng.uinteger;
}

struct StepArgs {
    int n, np, nc;
    Cfg cfg;
    uint32_t* state; uint32_t* snap;
    EnvArrays A;
    const int64_t* actions;
    uint8_t* newobs;   // (n, 84, 84): frame produced by this step (terminal frame on done)
    float* rew; uint8_t* term; uint8_t* trunc; int* info;  // (n, N_INFO)
    double* ep_ret_out; int* ep_len_out;
    int* reset_idx;    // cache entry used for the reset obs, -1 if no reset
    const uint32_t* cstate; const int64_t* cx; const int64_t* ct; const int* clives; const int* cdraws;
    const uint8_t* cart; const int* nt; const int* xs; const float* xw; const int* ye; const float* yw;
    const uint8_t* lut;
    int* bad_ops;
    const int* gr;
    uint8_t* dbg_screen;  // optional: 240*256 palette values of env 0's rendered frames
};

// mode 0: step; mode 1: reset all envs
extern "C" __global__ void k_step(StepArgs P, int mode) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= P.n) return;
    Rs rs{P.xs, P.xw, P.ye, P.yw, P.lut, P.gr};
    Nes n;
    setup_nes(n, P.state, e, P.np, P.cart, P.nt, &rs);
    EnvS E;
    Pcg64 rng;
    load_env(E, rng, P.A, e);
    int L = P.A.level[e];
    if (mode == 1) {
        do_reset(n, E, rng, P.cfg, L, P.cstate, P.nc, P.cx, P.ct, P.clives, P.cdraws, P.reset_idx, e);
        store_sc(n.s, n.m);
        store_env(E, rng, P.A, e);
        return;
    }
    load_sc(n.s, n.m);
    n.obs = P.newobs + (size_t)e * 84 * 84;
    if (e == 0) n.screen = P.dbg_screen;
    Mem snap = mem_of(P.snap, e, P.np);
    int a = (int)P.actions[e];
    bool te;
    int draws = 0;
    double r = agent_step(n, E, rng, P.cfg, a, true, &te, snap, &draws);
    E.t++;
    bool tr = E.t >= P.cfg.max_steps;
    E.ep_ret += r; E.ep_len++;
    P.rew[e] = (float)r; P.term[e] = te; P.trunc[e] = tr;
    int* inf = P.info + (size_t)e * N_INFO;
    inf[0] = E.info.x_pos; inf[1] = E.info.flag_get; inf[2] = E.info.life; inf[3] = E.info.score;
    inf[4] = E.info.time; inf[5] = E.info.world; inf[6] = E.info.stage; inf[7] = E.info.coins;
    inf[8] = E.info.y_pos; inf[9] = E.info.status;
    P.A.frames[e] += draws;
    if (te || tr) {
        P.ep_ret_out[e] = E.ep_ret; P.ep_len_out[e] = E.ep_len;
        do_reset(n, E, rng, P.cfg, L, P.cstate, P.nc, P.cx, P.ct, P.clives, P.cdraws, P.reset_idx, e);
    } else {
        P.reset_idx[e] = -1;
    }
    store_sc(n.s, n.m);
    store_env(E, rng, P.A, e);
    if (n.bad_ops) atomicAdd(P.bad_ops, n.bad_ops);
}

// Frame stack update (fastmario VecEnv::do_env semantics), one block per env, k = 4 frames.
// stack: (n, 4, 84*84); newobs: (n, 84*84); final: (n, 4, 84*84) (written only for done envs)
extern "C" __global__ void k_stack(int n, uint8_t* stack, const uint8_t* newobs, uint8_t* final_, const uint8_t* term,
                                   const uint8_t* trunc, const int* reset_idx, const uint8_t* cobs, int mode) {
    const int O = 84 * 84 / 4;  // words per frame
    int e = blockIdx.x;
    if (e >= n) return;
    uint32_t* st = (uint32_t*)(stack + (size_t)e * 4 * 84 * 84);
    const uint32_t* no = (const uint32_t*)(newobs + (size_t)e * 84 * 84);
    int ri = reset_idx[e];
    const uint32_t* ro = ri >= 0 ? (const uint32_t*)(cobs + (size_t)ri * 84 * 84) : 0;
    if (mode == 1) {  // reset: zeros + reset obs
        for (int i = threadIdx.x; i < O; i += blockDim.x) {
            st[i] = 0; st[O + i] = 0; st[2 * O + i] = 0; st[3 * O + i] = ro[i];
        }
        return;
    }
    bool done = term[e] || trunc[e];
    for (int i = threadIdx.x; i < O; i += blockDim.x) {
        uint32_t f1 = st[O + i], f2 = st[2 * O + i], f3 = st[3 * O + i], nw = no[i];
        if (done) {
            uint32_t* fo = (uint32_t*)(final_ + (size_t)e * 4 * 84 * 84);
            fo[i] = f1; fo[O + i] = f2; fo[2 * O + i] = f3; fo[3 * O + i] = nw;
            st[i] = 0; st[O + i] = 0; st[2 * O + i] = 0; st[3 * O + i] = ro[i];
        } else {
            st[i] = f1; st[O + i] = f2; st[2 * O + i] = f3; st[3 * O + i] = nw;
        }
    }
}

extern "C" __global__ void k_nw(int* o) { o[0] = NW; o[1] = SC_W0; }

// raw frame-level debug: one console from power-on, buttons[f] per frame, render every frame
extern "C" __global__ void k_raw(int nf, const int* buttons, uint8_t* ram_out, uint8_t* scr_out, uint32_t* st, uint32_t* st_out,
                                 const uint8_t* cart, const int* nt, const int* xs, const float* xw,
                                 const int* ye, const float* yw, const uint8_t* lut, const int* gr) {
    Rs rs{xs, xw, ye, yw, lut, gr};
    Nes n;
    setup_nes(n, st, 0, 1, cart, nt, &rs);
    n.power_on();
    for (int f = 0; f < nf; f++) {
        n.screen = scr_out + (size_t)f * 240 * 256;
        n.s.joy_buttons0 = buttons[f] & 0xff; n.render = buttons[f] < 0 ? false : true; n.run_frame();
        for (int i = 0; i < 0x800; i++) ram_out[(size_t)f * 0x800 + i] = n.RAM(i);
        if (st_out) { store_sc(n.s, n.m); for (int i = 0; i < NW; i++) st_out[(size_t)f * NW + i] = n.m.W(i); }
    }
}

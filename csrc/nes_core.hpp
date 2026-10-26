// SPDX-License-Identifier: GPL-3.0-or-later
// fast-mario-env NES core: a behaviour-identical re-implementation of the nes-py 8.2.1 (SimpleNES-derived)
// emulator, restricted to mapper 0 (NROM, e.g. Super Mario Bros).
//
// Exactness contract: for the same ROM and controller inputs, RAM / VRAM / OAM / palette /
// CPU registers and the rendered screen are bit-identical to nes-py after every frame.
// nes-py quirks are reproduced on purpose (instruction-atomic CPU with skip cycles, PPU
// sprite-0 / scrolling model, bit-reversed flag byte layout, NT3 mirroring of $3000, ...).
//
// Speed comes from:
//   * no std::function / unordered_map / virtual calls on the bus (direct switch)
//   * catch-up PPU: the PPU is only advanced when the CPU touches a PPU register, when an
//     NMI could be due, or at the end of a frame; between those points whole scanline
//     segments are advanced with closed-form updates of the scroll registers
//   * pixel output is optional per frame (`render`); with it off only the state the CPU can
//     observe is computed (scroll counters, sprite evaluation, exact sprite-0 hit)
//
// State is a single POD struct so snapshot/restore is a memcpy (and the layout is
// GPU-portable).
#pragma once
#include <cstdint>
#include <cstring>
#ifdef FM_PROF
#include <x86intrin.h>
namespace fm { inline uint64_t prof_pc[65536];
inline uint64_t prof_ppu = 0, prof_eval = 0, prof_cpu_instr = 0, prof_idle = 0; }
#define FM_T0 uint64_t _t0 = __rdtsc();
#define FM_T1(v) v += __rdtsc() - _t0;
#else
#define FM_T0
#define FM_T1(v)
#endif

namespace fm {

// Flag bits in the layout produced by nes-py's `bool N:1, V:1, ONE:1, B:1, D:1, I:1, Z:1, C:1`
// bitfield on little-endian SysV (N is the LSB).  Pushed/pulled with this layout.
enum : uint8_t { F_N = 0x01, F_V = 0x02, F_ONE = 0x04, F_B = 0x08, F_D = 0x10, F_I = 0x20, F_Z = 0x40, F_C = 0x80 };

enum PpuState : uint8_t { PRE_RENDER = 0, RENDER = 1, POST_RENDER = 2, VERTICAL_BLANK = 3 };

static const uint8_t OPERATION_CYCLES[0x100] = {
    7, 6, 0, 0, 0, 3, 5, 0, 3, 2, 2, 0, 0, 4, 6, 0, 2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    6, 6, 0, 0, 3, 3, 5, 0, 4, 2, 2, 0, 4, 4, 6, 0, 2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    6, 6, 0, 0, 0, 3, 5, 0, 3, 2, 2, 0, 3, 4, 6, 0, 2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    6, 6, 0, 0, 0, 3, 5, 0, 4, 2, 2, 0, 5, 4, 6, 0, 2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    0, 6, 0, 0, 3, 3, 3, 0, 2, 0, 2, 0, 4, 4, 4, 0, 2, 6, 0, 0, 4, 4, 4, 0, 2, 5, 2, 0, 0, 5, 0, 0,
    2, 6, 2, 0, 3, 3, 3, 0, 2, 2, 2, 0, 4, 4, 4, 0, 2, 5, 0, 0, 4, 4, 4, 0, 2, 4, 2, 0, 4, 4, 4, 0,
    2, 6, 0, 0, 3, 3, 5, 0, 2, 2, 2, 0, 4, 4, 6, 0, 2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    2, 6, 0, 0, 3, 3, 5, 0, 2, 2, 2, 2, 4, 4, 6, 0, 2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
};

static const int CYCLES_PER_FRAME = 29781;
static const int SCANLINE_END_CYCLE = 341;
static const int VISIBLE_SCANLINES = 240;
static const int FRAME_END_SCANLINE = 261;

// Read-only cartridge data (shared by all emulators running the same ROM).
struct Cart {
    uint8_t prg[0x8000];
    uint8_t chr[0x2000];
    uint32_t prg_size;      // 0x4000 or 0x8000
    uint16_t nt[4];         // name table offsets into VRAM (mirroring)
};

// Load an iNES file. Returns 0 on success, negative on unsupported ROMs.
inline int cart_load(Cart* c, const uint8_t* data, size_t n) {
    if (n < 16 || memcmp(data, "NES\x1a", 4) != 0) return -1;
    uint8_t mirroring = data[6] & 0xB;
    int mapper = ((data[6] >> 4) & 0xf) | (data[7] & 0xf0);
    bool ext_ram = data[6] & 0x2;
    int banks = data[4], vbanks = data[5];
    if (mapper != 0 || ext_ram || (data[6] & 0x4)) return -2;     // NROM, no PRG-RAM, no trainer
    if (banks < 1 || banks > 2 || vbanks != 1) return -3;          // CHR-ROM only
    if (n < 16 + 0x4000u * banks + 0x2000u) return -4;
    memset(c, 0, sizeof(*c));
    memcpy(c->prg, data + 16, 0x4000 * banks);
    if (banks == 1) memcpy(c->prg + 0x4000, data + 16, 0x4000);  // NROM-128 mirror
    memcpy(c->chr, data + 16 + 0x4000 * banks, 0x2000);
    c->prg_size = 0x4000 * banks;
    switch (mirroring) {
        case 0: c->nt[0] = c->nt[1] = 0; c->nt[2] = c->nt[3] = 0x400; break;      // HORIZONTAL
        case 1: c->nt[0] = c->nt[2] = 0; c->nt[1] = c->nt[3] = 0x400; break;      // VERTICAL
        default: c->nt[0] = c->nt[1] = c->nt[2] = c->nt[3] = 0; break;            // unsupported -> 0
    }
    return 0;
}

// Complete machine state (POD).
struct NesState {
    // ---- scheduling (nes-py: CPU::cycles / skip_cycles, Emulator::step loop) ----
    int64_t iter;        // CPU iterations completed (multiple of 29781 at frame boundaries)
    int64_t exec_at;     // iteration at which the next instruction executes
    int64_t ppu_calls;   // PPU::cycle calls completed (== 3*iter at frame boundaries)
    int64_t nmi_thresh;  // must sync PPU before executing at J if 3*(J+1) >= nmi_thresh
    uint8_t zero_skip;   // last instruction left skip_cycles == 0 (illegal opcode)
    // ---- CPU ----
    uint16_t pc;
    uint8_t sp, a, x, y, p;
    // ---- controllers ----
    uint8_t joy_strobe[2], joy_buttons[2], joy_bits[2];
    // ---- PPU ----
    uint8_t pstate;
    uint8_t even, vblank, s0hit, first_write, data_buffer, oam_addr, fine_x;
    uint8_t show_bg, show_spr, hide_edge_bg, hide_edge_spr, long_spr, nmi_en, bg_page, spr_page;
    uint8_t nsl;
    uint8_t sl[8];
    int32_t pcyc, scanline;
    uint16_t v, t, inc;
    uint8_t oam[256];
    uint8_t pal[32];
    uint8_t vram[0x800];
    uint8_t ram[0x800];
};

struct Nes {
    NesState s;
    const Cart* cart = nullptr;
    uint8_t* screen = nullptr;  // optional 240*256 row-major palette-RAM values (debug / verification)
    uint8_t* grayR = nullptr;   // 240*256 row-major grayscale output
    const uint8_t* gray_lut = nullptr;  // palette value -> gray (cv2 RGB2GRAY of nes-py's palette)
    bool render = false;
    int64_t cur_iter = 0;       // iteration of the instruction being executed
    int skip = 0;               // skip cycles accumulated by the executing instruction
    int64_t bad_ops = 0;

    // ------------------------------------------------------------------ init
    void power_on(const Cart* c) {  // nes-py: Emulator(rom) (zeroed memory) + Reset()
        cart = c;
        memset(&s, 0, sizeof(s));
        s.joy_strobe[0] = s.joy_strobe[1] = 1;
        reset();
    }
    void reset() {
        // CPU::reset(bus)
        s.zero_skip = 0;
        s.a = s.x = s.y = 0;
        s.p = 0b00110100;
        s.pc = (uint16_t)(rd_prg(0xfffc) | rd_prg(0xfffd) << 8);
        s.sp = 0xfd;
        s.iter = 0; s.exec_at = 0; s.ppu_calls = 0;
        // PPU::reset()
        s.long_spr = 0; s.nmi_en = 0; s.vblank = 0; s.show_bg = 1; s.show_spr = 1; s.even = 1;
        s.first_write = 1; s.bg_page = 0; s.spr_page = 0; s.v = 0; s.pcyc = 0; s.scanline = 0;
        s.oam_addr = 0; s.fine_x = 0; s.t = 0; s.inc = 1; s.pstate = PRE_RENDER; s.nsl = 0;
        update_nmi_thresh();
    }

    // ------------------------------------------------------------------ CPU bus
    inline uint8_t rd_prg(uint16_t a) const {
        return cart->prg[a & 0x7fff];  // 16 KB images are mirrored at load
    }
    inline uint8_t rd_prg_fast(uint16_t a) const { return a >= 0x8000 ? rd_prg(a) : 0xff; }
    inline uint8_t rd(uint16_t a) {
        if (a >= 0x8000) return rd_prg(a);
        if (a < 0x2000) return s.ram[a & 0x7ff];
        if (a < 0x4000) {
            switch (a & 0x2007) {
                case 0x2002: sync(); return ppu_status();
                case 0x2007: sync(); return ppu_get_data();
                case 0x2004: return s.oam[s.oam_addr];
                default: return 0;
            }
        }
        if (a == 0x4016) return joy_read(0);
        if (a == 0x4017) return joy_read(1);
        return 0;
    }
    inline void wr(uint16_t a, uint8_t v) {
        if (a < 0x2000) { s.ram[a & 0x7ff] = v; return; }
        if (a < 0x4000) {
            switch (a & 0x2007) {
                case 0x2000: sync(); ppu_control(v); invalidate_caches(); break;
                case 0x2001: sync(); ppu_mask(v); break;
                case 0x2003: sync(); s.oam_addr = v; break;
                case 0x2004: sync(); s.oam[s.oam_addr++] = v; invalidate_caches(); break;
                case 0x2005: sync(); ppu_scroll(v); break;
                case 0x2006: sync(); ppu_addr(v); break;
                case 0x2007: sync(); ppu_set_data(v); invalidate_caches(); break;
                default: break;
            }
            return;
        }
        if (a == 0x4014) {  // OAMDMA
            sync();
            skip += 513 + (int)((cur_iter + 1) & 1);
            uint16_t addr = (uint16_t)(v << 8);
            if (addr < 0x2000) {
                const uint8_t* page = &s.ram[addr & 0x7ff];
                memcpy(s.oam + s.oam_addr, page, 256 - s.oam_addr);
                if (s.oam_addr) memcpy(s.oam, page + (256 - s.oam_addr), s.oam_addr);
            }
            invalidate_caches();
            return;
        }
        if (a == 0x4016) {
            for (int i = 0; i < 2; i++) {
                s.joy_strobe[i] = v & 1;
                if (!s.joy_strobe[i]) s.joy_bits[i] = s.joy_buttons[i];
            }
        }
    }
    inline uint8_t joy_read(int i) {
        uint8_t r;
        if (s.joy_strobe[i]) r = s.joy_buttons[i] & 1;
        else { r = s.joy_bits[i] & 1; s.joy_bits[i] >>= 1; }
        return r | 0x40;
    }

    // ------------------------------------------------------------------ PPU registers
    inline void ppu_control(uint8_t c) {
        s.nmi_en = (c & 0x80) != 0;
        s.long_spr = (c & 0x20) != 0;
        s.bg_page = (c & 0x10) ? 1 : 0;
        s.spr_page = (c & 0x8) ? 1 : 0;
        s.inc = (c & 0x4) ? 0x20 : 1;
        s.t = (uint16_t)((s.t & ~0xc00) | ((c & 0x3) << 10));
    }
    inline void ppu_mask(uint8_t m) {
        s.hide_edge_bg = !(m & 0x2);
        s.hide_edge_spr = !(m & 0x4);
        s.show_bg = (m & 0x8) != 0;
        s.show_spr = (m & 0x10) != 0;
    }
    inline uint8_t ppu_status() {
        uint8_t st = (uint8_t)(s.s0hit << 6 | s.vblank << 7);
        s.vblank = 0;
        s.first_write = 1;
        return st;
    }
    inline void ppu_addr(uint8_t a) {
        if (s.first_write) {
            s.t = (uint16_t)((s.t & ~0xff00) | ((a & 0x3f) << 8));
            s.first_write = 0;
        } else {
            s.t = (uint16_t)((s.t & ~0xff) | a);
            s.v = s.t;
            s.first_write = 1;
        }
    }
    inline uint8_t ppu_get_data() {
        uint8_t d = pb_read(s.v);
        s.v = (uint16_t)(s.v + s.inc);
        if (s.v < 0x3f00) { uint8_t tmp = d; d = s.data_buffer; s.data_buffer = tmp; }
        return d;
    }
    inline void ppu_set_data(uint8_t d) {
        pb_write(s.v, d);
        s.v = (uint16_t)(s.v + s.inc);
    }
    inline void ppu_scroll(uint8_t sc) {
        if (s.first_write) {
            s.t = (uint16_t)((s.t & ~0x1f) | ((sc >> 3) & 0x1f));
            s.fine_x = sc & 0x7;
            s.first_write = 0;
        } else {
            s.t = (uint16_t)((s.t & ~0x73e0) | ((sc & 0x7) << 12) | ((sc & 0xf8) << 2));
            s.first_write = 1;
        }
    }

    // ------------------------------------------------------------------ picture bus
    inline uint8_t pb_read(uint16_t a) const {
        if (a < 0x2000) return cart->chr[a];
        if (a < 0x3eff) {
            int t = a < 0x2400 ? 0 : a < 0x2800 ? 1 : a < 0x2c00 ? 2 : 3;
            return s.vram[cart->nt[t] + (a & 0x3ff)];
        }
        if (a < 0x3fff) return s.pal[a & 0x1f];
        return 0;
    }
    inline void pb_write(uint16_t a, uint8_t v) {
        if (a < 0x2000) return;  // CHR-ROM: read only
        if (a < 0x3eff) {
            int t = a < 0x2400 ? 0 : a < 0x2800 ? 1 : a < 0x2c00 ? 2 : 3;
            s.vram[cart->nt[t] + (a & 0x3ff)] = v;
        } else if (a < 0x3fff) {
            if (a == 0x3f10) s.pal[0] = v;
            else s.pal[a & 0x1f] = v;
        }
    }
    // name table read for addresses known to be in 0x2000..0x2fff
    inline uint8_t nt_read(uint16_t a) const {
        return s.vram[cart->nt[(a >> 10) & 3] + (a & 0x3ff)];
    }

    // ------------------------------------------------------------------ PPU timing
    // distance (in PPU calls, inclusive) until the vblank-start call, minus a safety margin
    inline void update_nmi_thresh() {
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
    inline void sync_to(int64_t target) {
        if (target > s.ppu_calls) {
            ppu_run(target - s.ppu_calls);
            update_nmi_thresh();
        }
    }
    inline void sync() { sync_to(3 * (cur_iter + 1)); }

    inline void coarse_x_inc_n(int n) {
        if (n <= 0) return;
        int tot = (s.v & 0x1f) + n;
        s.v = (uint16_t)((s.v & ~0x1f) | (tot & 31));
        if ((tot >> 5) & 1) s.v ^= 0x0400;
    }
    // number of x in [a, b) with (fine_x + x) % 8 == 7
    inline int inc_count(int a, int b) const {
        int r = (7 - s.fine_x) & 7;
        return (b - r + 7) / 8 - (a - r + 7) / 8;
    }
    inline void fine_y_inc() {
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
    inline void hcopy() { s.v = (uint16_t)((s.v & ~0x41f) | (s.t & 0x41f)); }
    inline void vcopy() { s.v = (uint16_t)((s.v & ~0x7be0) | (s.t & 0x7be0)); }

    inline void sprite_eval() {
#ifdef FM_PROF
        prof_eval++;
#endif
        s.nsl = 0;
        int range = s.long_spr ? 16 : 8;
        for (int i = s.oam_addr / 4; i < 64; ++i) {
            int diff = s.scanline - s.oam[i * 4];
            if (0 <= diff && diff < range) {
                s.sl[s.nsl++] = (uint8_t)i;
                if (s.nsl >= 8) break;
            }
        }
    }

    // background pixel (2-bit opaque part + attribute bits) at dot x for the current v
    inline uint8_t bg_pixel(int x, uint8_t* opaque) const {
        int x_fine = (s.fine_x + x) % 8;
        uint16_t addr = (uint16_t)(0x2000 | (s.v & 0x0FFF));
        uint8_t tile = nt_read(addr);
        addr = (uint16_t)((tile * 16) + ((s.v >> 12) & 0x7));
        addr |= s.bg_page << 12;
        uint8_t c = (cart->chr[addr] >> (7 ^ x_fine)) & 1;
        c |= ((cart->chr[addr + 8] >> (7 ^ x_fine)) & 1) << 1;
        *opaque = c;
        uint16_t aaddr = (uint16_t)(0x23C0 | (s.v & 0x0C00) | ((s.v >> 4) & 0x38) | ((s.v >> 2) & 0x07));
        uint8_t attribute = nt_read(aaddr);
        int shift = ((s.v >> 4) & 4) | (s.v & 2);
        c |= ((attribute >> shift) & 0x3) << 2;
        return c;
    }
    // 2-bit pattern value of sprite i at (x, y) (caller checked x range)
    inline uint8_t spr_bits(int i, int x, int y) const {
        uint8_t spr_x = s.oam[i * 4 + 3];
        uint8_t spr_y = (uint8_t)(s.oam[i * 4 + 0] + 1), tile = s.oam[i * 4 + 1], attribute = s.oam[i * 4 + 2];
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
        uint8_t c = (pb_read(address) >> x_shift) & 1;
        c |= ((pb_read((uint16_t)(address + 8)) >> x_shift) & 1) << 1;
        return c;
    }

    // visible dots x in [x0, x1) of the current scanline (render off): only CPU-visible state
    inline void dots_fast(int x0, int x1) {
        if (!s.show_bg) return;  // no scroll increments, sprite-0 hit impossible
        if (!s.s0hit && s.show_spr && s.nsl > 0 && s.sl[0] == 0) {
            int sx = s.oam[3];
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

    // ---- render caches (derived data, not part of NesState; invalidated conservatively)
    int32_t tc_key = -1;           // (bg_page << 16) | v of the cached background tile fetch
    uint8_t tc_lo = 0, tc_hi = 0, tc_attr = 0;
    int32_t sp_line = -1;          // scanline the sprite line buffer was built for (-1 = invalid)
    uint8_t sp_col[256], sp_flags[256];  // first opaque list sprite per x: colour (0 = none), bit0 fg, bit1 sprite 0
    inline void invalidate_caches() { tc_key = -1; sp_line = -1; }

    void build_sprite_line(int y) {
        memset(sp_col, 0, sizeof(sp_col));
        for (int k = 0; k < s.nsl; k++) {
            int i = s.sl[k];
            int sx = s.oam[i * 4 + 3];
            uint8_t attribute = s.oam[i * 4 + 2];
            uint8_t hi_bits = (uint8_t)(0x10 | ((attribute & 0x3) << 2));
            uint8_t fl = (uint8_t)((!(attribute & 0x20) ? 1 : 0) | (i == 0 ? 2 : 0));
            int xe = sx + 8 < 256 ? sx + 8 : 256;
            for (int x = sx; x < xe; x++) {
                if (sp_col[x]) continue;
                uint8_t c = spr_bits(i, x, y);
                if (c) { sp_col[x] = (uint8_t)(c | hi_bits); sp_flags[x] = fl; }
            }
        }
        sp_line = y;
    }

    // spread tables: byte k of SPREAD[b] = bit (7-k) of b  (pattern bit for the pixel at fine x k)
    struct Spread {
        uint64_t t[256];
        Spread() {
            for (int b = 0; b < 256; b++) {
                uint64_t v = 0;
                for (int k = 0; k < 8; k++) v |= (uint64_t)((b >> (7 - k)) & 1) << (8 * k);
                t[b] = v;
            }
        }
    };
    static const Spread& spread() { static Spread sp; return sp; }

    inline void fetch_tile() {
        int32_t key = (s.bg_page << 16) | s.v;
        if (key == tc_key) return;
        uint8_t tile = nt_read((uint16_t)(0x2000 | (s.v & 0x0FFF)));
        uint16_t addr = (uint16_t)(((tile * 16) + ((s.v >> 12) & 0x7)) | (s.bg_page << 12));
        tc_lo = cart->chr[addr];
        tc_hi = cart->chr[addr + 8];
        uint16_t aaddr = (uint16_t)(0x23C0 | (s.v & 0x0C00) | ((s.v >> 4) & 0x38) | ((s.v >> 2) & 0x07));
        int shift = ((s.v >> 4) & 4) | (s.v & 2);
        tc_attr = (uint8_t)(((nt_read(aaddr) >> shift) & 0x3) << 2);
        tc_key = key;
    }

    // visible dots with pixel output (same results as PPU::cycle's RENDER pixel branch)
    void dots_render(int x0, int x1) {
        const int y = s.scanline;
        uint8_t pa[256];  // palette address per dot
        if (!s.show_bg) {
            for (int x = x0; x < x1; x++) pa[x] = 0;
        } else {
            const Spread& SP = spread();
            int x = x0;
            while (x < x1) {
                int xf = (s.fine_x + x) & 7;
                int span = 8 - xf;
                if (span > x1 - x) span = x1 - x;
                fetch_tile();
                uint64_t bits = SP.t[tc_lo] | (SP.t[tc_hi] << 1);
                for (int j = 0; j < span; j++) {
                    uint8_t b = (uint8_t)((bits >> (8 * (xf + j))) & 3);
                    pa[x + j] = b ? (uint8_t)(b | tc_attr) : 0;
                }
                if (s.hide_edge_bg)
                    for (int j = 0; j < span && x + j < 8; j++) pa[x + j] = 0;
                if (xf + span == 8) { coarse_x_inc_n(1); }
                x += span;
            }
        }
        if (s.show_spr && s.nsl) {
            if (sp_line != y) build_sprite_line(y);
            int xs = (s.hide_edge_spr && x0 < 8) ? 8 : x0;
            for (int x = xs; x < x1; x++) {
                uint8_t c = sp_col[x];
                if (!c) continue;
                uint8_t bgOpaque = pa[x] & 3;
                uint8_t fl = sp_flags[x];
                if ((fl & 2) && bgOpaque && !s.s0hit && s.show_bg) s.s0hit = 1;
                if (!bgOpaque || (fl & 1)) pa[x] = c;
            }
        }
        if (screen) {
            uint8_t* row = screen + y * 256;
            for (int x = x0; x < x1; x++) row[x] = s.pal[pa[x]];
        }
        if (grayR) {
            uint8_t gp[32];
            for (int i = 0; i < 32; i++) gp[i] = gray_lut[s.pal[i]];
            uint8_t* g = grayR + y * 256;
            for (int x = x0; x < x1; x++) g[x] = gp[pa[x]];
        }
    }

    inline void dots(int x0, int x1) {
        if (render) dots_render(x0, x1);
        else dots_fast(x0, x1);
    }

    void nmi(int64_t call_index) {  // call_index: 1-based global PPU call that raised vblank
        int64_t m = (call_index - 1) / 3;
        // CPU::interrupt(NMI)
        push((uint8_t)(s.pc >> 8));
        push((uint8_t)s.pc);
        push((uint8_t)(s.p | 0b00100000));
        s.p |= F_I;
        s.pc = (uint16_t)(rd_prg(0xfffa) | rd_prg(0xfffb) << 8);
        int64_t v = (s.zero_skip && m == s.exec_at) ? 0 : (s.exec_at - m + 1);
        s.exec_at = m + v + 6;
        s.zero_skip = 0;
    }

    // Advance the PPU by n PPU::cycle calls.
    void ppu_run(int64_t n) {
        FM_T0
        ppu_run_(n);
        FM_T1(prof_ppu)
    }
    void ppu_run_(int64_t n) {
        while (n > 0) {
            int c = s.pcyc;
            switch (s.pstate) {
                case PRE_RENDER: {
                    bool rend = s.show_bg && s.show_spr;
                    int end = SCANLINE_END_CYCLE - (!s.even && rend);
                    int64_t k = c >= end ? 1 : (n < end - c ? n : end - c);
                    int a = c, b = (int)(c + k);  // calls with cycle values [a, b)
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
                    if (c >= 1 && c <= 256) {
                        int64_t k = n < 257 - c ? n : 257 - c;
                        dots(c - 1, (int)(c - 1 + k));
                        s.pcyc += (int)k; s.ppu_calls += k; n -= k;
                    } else if (c == 257) {
                        if (s.show_bg) fine_y_inc();
                        s.pcyc++; s.ppu_calls++; n--;
                    } else if (c == 258) {
                        if (s.show_bg && s.show_spr) hcopy();
                        s.pcyc++; s.ppu_calls++; n--;
                    } else if (c > 258 && c < SCANLINE_END_CYCLE) {
                        int64_t k = n < SCANLINE_END_CYCLE - c ? n : SCANLINE_END_CYCLE - c;
                        s.pcyc += (int)k; s.ppu_calls += k; n -= k;
                    } else if (c >= SCANLINE_END_CYCLE) {
                        sprite_eval();
                        sp_line = -1;
                        ++s.scanline;
                        s.pcyc = 1; s.ppu_calls++; n--;
                        if (s.scanline >= VISIBLE_SCANLINES) s.pstate = POST_RENDER;
                    } else {  // c == 0 (unreachable in practice)
                        s.pcyc++; s.ppu_calls++; n--;
                    }
                    break;
                }
                case POST_RENDER: {
                    if (c >= SCANLINE_END_CYCLE) {
                        ++s.scanline; s.pcyc = 1; s.pstate = VERTICAL_BLANK;
                        s.ppu_calls++; n--;
                    } else {
                        int64_t k = n < SCANLINE_END_CYCLE - c ? n : SCANLINE_END_CYCLE - c;
                        s.pcyc += (int)k; s.ppu_calls += k; n -= k;
                    }
                    break;
                }
                default: {  // VERTICAL_BLANK
                    if (c >= SCANLINE_END_CYCLE) {
                        ++s.scanline;
                        s.pcyc = 1;
                        if (s.scanline >= FRAME_END_SCANLINE) {
                            s.pstate = PRE_RENDER; s.scanline = 0; s.even = !s.even;
                        }
                        s.ppu_calls++; n--;
                    } else if (c == 1 && s.scanline == 241) {
                        s.vblank = 1;
                        s.ppu_calls++; n--;
                        s.pcyc = 2;
                        if (s.nmi_en) nmi(s.ppu_calls);
                    } else {
                        int lim = SCANLINE_END_CYCLE - c;
                        if (s.scanline == 241 && c < 1) lim = 1 - c;
                        int64_t k = n < lim ? n : lim;
                        s.pcyc += (int)k; s.ppu_calls += k; n -= k;
                    }
                    break;
                }
            }
        }
    }

    // ------------------------------------------------------------------ CPU
    inline void set_ZN(uint8_t v) {
        s.p = (uint8_t)((s.p & ~(F_Z | F_N)) | (v ? 0 : F_Z) | ((v & 0x80) ? F_N : 0));
    }
    inline void setf(uint8_t f, bool on) { s.p = on ? (s.p | f) : (s.p & ~f); }
    inline bool getf(uint8_t f) const { return (s.p & f) != 0; }
    inline uint16_t read_address(uint16_t a) {
        uint8_t lo = rd(a);
        uint8_t hi = rd((uint16_t)(a + 1));
        return (uint16_t)(lo | hi << 8);
    }
    inline void push(uint8_t v) { s.ram[(0x100 | s.sp--) & 0x7ff] = v; }
    inline uint8_t pop() { return s.ram[(0x100 | ++s.sp) & 0x7ff]; }
    inline void page_crossed(uint16_t a, uint16_t b, int inc = 1) {
        if ((a & 0xff00) != (b & 0xff00)) skip += inc;
    }

    void interrupt_brk() {
        ++s.pc;
        push((uint8_t)(s.pc >> 8));
        push((uint8_t)s.pc);
        push((uint8_t)(s.p | 0b00100000 | (1 << 4)));
        s.p |= F_I;
        s.pc = read_address(0xfffe);
        skip += 7;
    }

    __attribute__((always_inline)) inline bool implied(uint8_t op) {
        switch (op) {
            case 0x00: interrupt_brk(); break;                              // BRK
            case 0x08: push(s.p); break;                                    // PHP
            case 0x18: setf(F_C, false); break;                             // CLC
            case 0x20:                                                      // JSR
                push((uint8_t)((s.pc + 1) >> 8));
                push((uint8_t)(s.pc + 1));
                s.pc = read_address(s.pc);
                break;
            case 0x28: s.p = pop(); break;                                  // PLP
            case 0x38: setf(F_C, true); break;                              // SEC
            case 0x40: s.p = pop(); s.pc = pop(); s.pc |= pop() << 8; break; // RTI
            case 0x48: push(s.a); break;                                    // PHA
            case 0x4C: s.pc = read_address(s.pc); break;                    // JMP
            case 0x58: setf(F_I, false); break;                             // CLI
            case 0x60: s.pc = pop(); s.pc |= pop() << 8; ++s.pc; break;     // RTS
            case 0x68: s.a = pop(); set_ZN(s.a); break;                     // PLA
            case 0x6C: {                                                    // JMP (ind)
                uint16_t loc = read_address(s.pc);
                uint16_t page = loc & 0xff00;
                uint8_t lo = rd(loc);
                uint8_t hi = rd((uint16_t)(page | ((loc + 1) & 0xff)));
                s.pc = (uint16_t)(lo | hi << 8);
                break;
            }
            case 0x78: setf(F_I, true); break;                              // SEI
            case 0x88: --s.y; set_ZN(s.y); break;                           // DEY
            case 0x8a: s.a = s.x; set_ZN(s.a); break;                       // TXA
            case 0x98: s.a = s.y; set_ZN(s.a); break;                       // TYA
            case 0x9a: s.sp = s.x; break;                                   // TXS
            case 0xa8: s.y = s.a; set_ZN(s.y); break;                       // TAY
            case 0xaa: s.x = s.a; set_ZN(s.x); break;                       // TAX
            case 0xb8: setf(F_V, false); break;                             // CLV
            case 0xba: s.x = s.sp; set_ZN(s.x); break;                      // TSX
            case 0xc8: ++s.y; set_ZN(s.y); break;                           // INY
            case 0xca: --s.x; set_ZN(s.x); break;                           // DEX
            case 0xd8: setf(F_D, false); break;                             // CLD
            case 0xe8: ++s.x; set_ZN(s.x); break;                           // INX
            case 0xea: break;                                               // NOP
            case 0xf8: setf(F_D, true); break;                              // SED
            default: return false;
        }
        return true;
    }

    __attribute__((always_inline)) inline bool branch(uint8_t op) {
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

    __attribute__((always_inline)) inline bool type0(uint8_t op) {
        if ((op & 0x3) != 0x0) return false;
        uint16_t location = 0;
        switch ((op & 0x1c) >> 2) {
            case 0: location = s.pc++; break;                                     // immediate
            case 1: location = rd(s.pc++); break;                                 // zero page
            case 3: location = read_address(s.pc); s.pc += 2; break;              // absolute
            case 5: location = (rd(s.pc++) + s.x) & 0xff; break;                  // zp,X
            case 7:                                                               // abs,X
                location = read_address(s.pc); s.pc += 2;
                page_crossed(location, (uint16_t)(location + s.x));
                location = (uint16_t)(location + s.x);
                break;
            default: return false;
        }
        switch ((op & 0xe0) >> 5) {
            case 1: {  // BIT
                uint16_t operand = rd(location);
                setf(F_Z, !(s.a & operand));
                setf(F_V, operand & 0x40);
                setf(F_N, operand & 0x80);
                break;
            }
            case 4: wr(location, s.y); break;                       // STY
            case 5: s.y = rd(location); set_ZN(s.y); break;         // LDY
            case 6: {                                               // CPY
                uint16_t diff = (uint16_t)(s.y - rd(location));
                setf(F_C, !(diff & 0x100));
                set_ZN((uint8_t)diff);
                break;
            }
            case 7: {                                               // CPX
                uint16_t diff = (uint16_t)(s.x - rd(location));
                setf(F_C, !(diff & 0x100));
                set_ZN((uint8_t)diff);
                break;
            }
            default: return false;
        }
        return true;
    }

    __attribute__((always_inline)) inline bool type1(uint8_t op) {
        if ((op & 0x3) != 0x1) return false;
        uint16_t location = 0;
        int opn = (op & 0xe0) >> 5;
        const int STA = 4;
        switch ((op & 0x1c) >> 2) {
            case 0: {  // (zp,X)
                uint8_t za = (uint8_t)(s.x + rd(s.pc++));
                uint8_t lo = rd(za & 0xff);
                uint8_t hi = rd((za + 1) & 0xff);
                location = (uint16_t)(lo | hi << 8);
                break;
            }
            case 1: location = rd(s.pc++); break;
            case 2: location = s.pc++; break;
            case 3: location = read_address(s.pc); s.pc += 2; break;
            case 4: {  // (zp),Y
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
            case 0: s.a |= rd(location); set_ZN(s.a); break;   // ORA
            case 1: s.a &= rd(location); set_ZN(s.a); break;   // AND
            case 2: s.a ^= rd(location); set_ZN(s.a); break;   // EOR
            case 3: {                                          // ADC
                uint8_t operand = rd(location);
                uint16_t sum = (uint16_t)(s.a + operand + getf(F_C));
                setf(F_C, sum & 0x100);
                setf(F_V, (s.a ^ sum) & (operand ^ sum) & 0x80);
                s.a = (uint8_t)sum;
                set_ZN(s.a);
                break;
            }
            case 4: wr(location, s.a); break;                  // STA
            case 5: s.a = rd(location); set_ZN(s.a); break;    // LDA
            case 6: {                                          // CMP
                uint16_t diff = (uint16_t)(s.a - rd(location));
                setf(F_C, !(diff & 0x100));
                set_ZN((uint8_t)diff);
                break;
            }
            default: {                                         // SBC
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

    __attribute__((always_inline)) inline bool type2(uint8_t op) {
        if ((op & 0x3) != 2) return false;
        uint16_t location = 0;
        int opn = (op & 0xe0) >> 5;
        int mode = (op & 0x1c) >> 2;
        const int ASL = 0, ROL = 1, LSR = 2, ROR = 3, STX = 4, LDX = 5, DEC = 6;
        switch (mode) {
            case 0: location = s.pc++; break;
            case 1: location = rd(s.pc++); break;
            case 2: break;  // accumulator
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

    // decode cascade of nes-py's CPU::cycle, specialised per opcode (constant-folded)
    template <int OP>
    __attribute__((always_inline)) inline void exec_t() {
        if (implied(OP) || branch(OP) || type1(OP) || type2(OP) || type0(OP)) skip += OPERATION_CYCLES[OP];
        else bad_ops++;
    }
    inline int execute() {
        skip = 0;
        uint8_t op = rd(s.pc++);
        switch (op) {
            case 0x00: exec_t<0x00>(); break;
            case 0x01: exec_t<0x01>(); break;
            case 0x02: exec_t<0x02>(); break;
            case 0x03: exec_t<0x03>(); break;
            case 0x04: exec_t<0x04>(); break;
            case 0x05: exec_t<0x05>(); break;
            case 0x06: exec_t<0x06>(); break;
            case 0x07: exec_t<0x07>(); break;
            case 0x08: exec_t<0x08>(); break;
            case 0x09: exec_t<0x09>(); break;
            case 0x0a: exec_t<0x0a>(); break;
            case 0x0b: exec_t<0x0b>(); break;
            case 0x0c: exec_t<0x0c>(); break;
            case 0x0d: exec_t<0x0d>(); break;
            case 0x0e: exec_t<0x0e>(); break;
            case 0x0f: exec_t<0x0f>(); break;
            case 0x10: exec_t<0x10>(); break;
            case 0x11: exec_t<0x11>(); break;
            case 0x12: exec_t<0x12>(); break;
            case 0x13: exec_t<0x13>(); break;
            case 0x14: exec_t<0x14>(); break;
            case 0x15: exec_t<0x15>(); break;
            case 0x16: exec_t<0x16>(); break;
            case 0x17: exec_t<0x17>(); break;
            case 0x18: exec_t<0x18>(); break;
            case 0x19: exec_t<0x19>(); break;
            case 0x1a: exec_t<0x1a>(); break;
            case 0x1b: exec_t<0x1b>(); break;
            case 0x1c: exec_t<0x1c>(); break;
            case 0x1d: exec_t<0x1d>(); break;
            case 0x1e: exec_t<0x1e>(); break;
            case 0x1f: exec_t<0x1f>(); break;
            case 0x20: exec_t<0x20>(); break;
            case 0x21: exec_t<0x21>(); break;
            case 0x22: exec_t<0x22>(); break;
            case 0x23: exec_t<0x23>(); break;
            case 0x24: exec_t<0x24>(); break;
            case 0x25: exec_t<0x25>(); break;
            case 0x26: exec_t<0x26>(); break;
            case 0x27: exec_t<0x27>(); break;
            case 0x28: exec_t<0x28>(); break;
            case 0x29: exec_t<0x29>(); break;
            case 0x2a: exec_t<0x2a>(); break;
            case 0x2b: exec_t<0x2b>(); break;
            case 0x2c: exec_t<0x2c>(); break;
            case 0x2d: exec_t<0x2d>(); break;
            case 0x2e: exec_t<0x2e>(); break;
            case 0x2f: exec_t<0x2f>(); break;
            case 0x30: exec_t<0x30>(); break;
            case 0x31: exec_t<0x31>(); break;
            case 0x32: exec_t<0x32>(); break;
            case 0x33: exec_t<0x33>(); break;
            case 0x34: exec_t<0x34>(); break;
            case 0x35: exec_t<0x35>(); break;
            case 0x36: exec_t<0x36>(); break;
            case 0x37: exec_t<0x37>(); break;
            case 0x38: exec_t<0x38>(); break;
            case 0x39: exec_t<0x39>(); break;
            case 0x3a: exec_t<0x3a>(); break;
            case 0x3b: exec_t<0x3b>(); break;
            case 0x3c: exec_t<0x3c>(); break;
            case 0x3d: exec_t<0x3d>(); break;
            case 0x3e: exec_t<0x3e>(); break;
            case 0x3f: exec_t<0x3f>(); break;
            case 0x40: exec_t<0x40>(); break;
            case 0x41: exec_t<0x41>(); break;
            case 0x42: exec_t<0x42>(); break;
            case 0x43: exec_t<0x43>(); break;
            case 0x44: exec_t<0x44>(); break;
            case 0x45: exec_t<0x45>(); break;
            case 0x46: exec_t<0x46>(); break;
            case 0x47: exec_t<0x47>(); break;
            case 0x48: exec_t<0x48>(); break;
            case 0x49: exec_t<0x49>(); break;
            case 0x4a: exec_t<0x4a>(); break;
            case 0x4b: exec_t<0x4b>(); break;
            case 0x4c: exec_t<0x4c>(); break;
            case 0x4d: exec_t<0x4d>(); break;
            case 0x4e: exec_t<0x4e>(); break;
            case 0x4f: exec_t<0x4f>(); break;
            case 0x50: exec_t<0x50>(); break;
            case 0x51: exec_t<0x51>(); break;
            case 0x52: exec_t<0x52>(); break;
            case 0x53: exec_t<0x53>(); break;
            case 0x54: exec_t<0x54>(); break;
            case 0x55: exec_t<0x55>(); break;
            case 0x56: exec_t<0x56>(); break;
            case 0x57: exec_t<0x57>(); break;
            case 0x58: exec_t<0x58>(); break;
            case 0x59: exec_t<0x59>(); break;
            case 0x5a: exec_t<0x5a>(); break;
            case 0x5b: exec_t<0x5b>(); break;
            case 0x5c: exec_t<0x5c>(); break;
            case 0x5d: exec_t<0x5d>(); break;
            case 0x5e: exec_t<0x5e>(); break;
            case 0x5f: exec_t<0x5f>(); break;
            case 0x60: exec_t<0x60>(); break;
            case 0x61: exec_t<0x61>(); break;
            case 0x62: exec_t<0x62>(); break;
            case 0x63: exec_t<0x63>(); break;
            case 0x64: exec_t<0x64>(); break;
            case 0x65: exec_t<0x65>(); break;
            case 0x66: exec_t<0x66>(); break;
            case 0x67: exec_t<0x67>(); break;
            case 0x68: exec_t<0x68>(); break;
            case 0x69: exec_t<0x69>(); break;
            case 0x6a: exec_t<0x6a>(); break;
            case 0x6b: exec_t<0x6b>(); break;
            case 0x6c: exec_t<0x6c>(); break;
            case 0x6d: exec_t<0x6d>(); break;
            case 0x6e: exec_t<0x6e>(); break;
            case 0x6f: exec_t<0x6f>(); break;
            case 0x70: exec_t<0x70>(); break;
            case 0x71: exec_t<0x71>(); break;
            case 0x72: exec_t<0x72>(); break;
            case 0x73: exec_t<0x73>(); break;
            case 0x74: exec_t<0x74>(); break;
            case 0x75: exec_t<0x75>(); break;
            case 0x76: exec_t<0x76>(); break;
            case 0x77: exec_t<0x77>(); break;
            case 0x78: exec_t<0x78>(); break;
            case 0x79: exec_t<0x79>(); break;
            case 0x7a: exec_t<0x7a>(); break;
            case 0x7b: exec_t<0x7b>(); break;
            case 0x7c: exec_t<0x7c>(); break;
            case 0x7d: exec_t<0x7d>(); break;
            case 0x7e: exec_t<0x7e>(); break;
            case 0x7f: exec_t<0x7f>(); break;
            case 0x80: exec_t<0x80>(); break;
            case 0x81: exec_t<0x81>(); break;
            case 0x82: exec_t<0x82>(); break;
            case 0x83: exec_t<0x83>(); break;
            case 0x84: exec_t<0x84>(); break;
            case 0x85: exec_t<0x85>(); break;
            case 0x86: exec_t<0x86>(); break;
            case 0x87: exec_t<0x87>(); break;
            case 0x88: exec_t<0x88>(); break;
            case 0x89: exec_t<0x89>(); break;
            case 0x8a: exec_t<0x8a>(); break;
            case 0x8b: exec_t<0x8b>(); break;
            case 0x8c: exec_t<0x8c>(); break;
            case 0x8d: exec_t<0x8d>(); break;
            case 0x8e: exec_t<0x8e>(); break;
            case 0x8f: exec_t<0x8f>(); break;
            case 0x90: exec_t<0x90>(); break;
            case 0x91: exec_t<0x91>(); break;
            case 0x92: exec_t<0x92>(); break;
            case 0x93: exec_t<0x93>(); break;
            case 0x94: exec_t<0x94>(); break;
            case 0x95: exec_t<0x95>(); break;
            case 0x96: exec_t<0x96>(); break;
            case 0x97: exec_t<0x97>(); break;
            case 0x98: exec_t<0x98>(); break;
            case 0x99: exec_t<0x99>(); break;
            case 0x9a: exec_t<0x9a>(); break;
            case 0x9b: exec_t<0x9b>(); break;
            case 0x9c: exec_t<0x9c>(); break;
            case 0x9d: exec_t<0x9d>(); break;
            case 0x9e: exec_t<0x9e>(); break;
            case 0x9f: exec_t<0x9f>(); break;
            case 0xa0: exec_t<0xa0>(); break;
            case 0xa1: exec_t<0xa1>(); break;
            case 0xa2: exec_t<0xa2>(); break;
            case 0xa3: exec_t<0xa3>(); break;
            case 0xa4: exec_t<0xa4>(); break;
            case 0xa5: exec_t<0xa5>(); break;
            case 0xa6: exec_t<0xa6>(); break;
            case 0xa7: exec_t<0xa7>(); break;
            case 0xa8: exec_t<0xa8>(); break;
            case 0xa9: exec_t<0xa9>(); break;
            case 0xaa: exec_t<0xaa>(); break;
            case 0xab: exec_t<0xab>(); break;
            case 0xac: exec_t<0xac>(); break;
            case 0xad: exec_t<0xad>(); break;
            case 0xae: exec_t<0xae>(); break;
            case 0xaf: exec_t<0xaf>(); break;
            case 0xb0: exec_t<0xb0>(); break;
            case 0xb1: exec_t<0xb1>(); break;
            case 0xb2: exec_t<0xb2>(); break;
            case 0xb3: exec_t<0xb3>(); break;
            case 0xb4: exec_t<0xb4>(); break;
            case 0xb5: exec_t<0xb5>(); break;
            case 0xb6: exec_t<0xb6>(); break;
            case 0xb7: exec_t<0xb7>(); break;
            case 0xb8: exec_t<0xb8>(); break;
            case 0xb9: exec_t<0xb9>(); break;
            case 0xba: exec_t<0xba>(); break;
            case 0xbb: exec_t<0xbb>(); break;
            case 0xbc: exec_t<0xbc>(); break;
            case 0xbd: exec_t<0xbd>(); break;
            case 0xbe: exec_t<0xbe>(); break;
            case 0xbf: exec_t<0xbf>(); break;
            case 0xc0: exec_t<0xc0>(); break;
            case 0xc1: exec_t<0xc1>(); break;
            case 0xc2: exec_t<0xc2>(); break;
            case 0xc3: exec_t<0xc3>(); break;
            case 0xc4: exec_t<0xc4>(); break;
            case 0xc5: exec_t<0xc5>(); break;
            case 0xc6: exec_t<0xc6>(); break;
            case 0xc7: exec_t<0xc7>(); break;
            case 0xc8: exec_t<0xc8>(); break;
            case 0xc9: exec_t<0xc9>(); break;
            case 0xca: exec_t<0xca>(); break;
            case 0xcb: exec_t<0xcb>(); break;
            case 0xcc: exec_t<0xcc>(); break;
            case 0xcd: exec_t<0xcd>(); break;
            case 0xce: exec_t<0xce>(); break;
            case 0xcf: exec_t<0xcf>(); break;
            case 0xd0: exec_t<0xd0>(); break;
            case 0xd1: exec_t<0xd1>(); break;
            case 0xd2: exec_t<0xd2>(); break;
            case 0xd3: exec_t<0xd3>(); break;
            case 0xd4: exec_t<0xd4>(); break;
            case 0xd5: exec_t<0xd5>(); break;
            case 0xd6: exec_t<0xd6>(); break;
            case 0xd7: exec_t<0xd7>(); break;
            case 0xd8: exec_t<0xd8>(); break;
            case 0xd9: exec_t<0xd9>(); break;
            case 0xda: exec_t<0xda>(); break;
            case 0xdb: exec_t<0xdb>(); break;
            case 0xdc: exec_t<0xdc>(); break;
            case 0xdd: exec_t<0xdd>(); break;
            case 0xde: exec_t<0xde>(); break;
            case 0xdf: exec_t<0xdf>(); break;
            case 0xe0: exec_t<0xe0>(); break;
            case 0xe1: exec_t<0xe1>(); break;
            case 0xe2: exec_t<0xe2>(); break;
            case 0xe3: exec_t<0xe3>(); break;
            case 0xe4: exec_t<0xe4>(); break;
            case 0xe5: exec_t<0xe5>(); break;
            case 0xe6: exec_t<0xe6>(); break;
            case 0xe7: exec_t<0xe7>(); break;
            case 0xe8: exec_t<0xe8>(); break;
            case 0xe9: exec_t<0xe9>(); break;
            case 0xea: exec_t<0xea>(); break;
            case 0xeb: exec_t<0xeb>(); break;
            case 0xec: exec_t<0xec>(); break;
            case 0xed: exec_t<0xed>(); break;
            case 0xee: exec_t<0xee>(); break;
            case 0xef: exec_t<0xef>(); break;
            case 0xf0: exec_t<0xf0>(); break;
            case 0xf1: exec_t<0xf1>(); break;
            case 0xf2: exec_t<0xf2>(); break;
            case 0xf3: exec_t<0xf3>(); break;
            case 0xf4: exec_t<0xf4>(); break;
            case 0xf5: exec_t<0xf5>(); break;
            case 0xf6: exec_t<0xf6>(); break;
            case 0xf7: exec_t<0xf7>(); break;
            case 0xf8: exec_t<0xf8>(); break;
            case 0xf9: exec_t<0xf9>(); break;
            case 0xfa: exec_t<0xfa>(); break;
            case 0xfb: exec_t<0xfb>(); break;
            case 0xfc: exec_t<0xfc>(); break;
            case 0xfd: exec_t<0xfd>(); break;
            case 0xfe: exec_t<0xfe>(); break;
            case 0xff: exec_t<0xff>(); break;
        }
        return skip;
    }

    // One nes-py Emulator::step(): 29781 CPU iterations, 3 PPU calls each.
    void run_frame() {
        const int64_t end = s.iter + CYCLES_PER_FRAME;
        invalidate_caches();
        while (s.exec_at < end) {
            const int64_t J = s.exec_at;
            if (3 * (J + 1) >= s.nmi_thresh) {
                cur_iter = J;
                sync_to(3 * (J + 1));
                if (s.exec_at != J) continue;  // NMI delayed this instruction
            }
            // idle-loop fast-forward: `JMP abs` to itself has no side effects; skip whole
            // iterations (3 cycles each) that would run before the frame end / NMI sync point
            if (rd_prg_fast(s.pc) == 0x4C && (uint16_t)(rd_prg_fast(s.pc + 1) | rd_prg_fast(s.pc + 2) << 8) == s.pc) {
                // iterations J + 3i with J + 3i < end and 3*(J + 3i + 1) < nmi_thresh
                int64_t lim_end = end - J;                          // 3i < lim_end
                int64_t lim_nmi = s.nmi_thresh - 3 * (J + 1);       // 9i < lim_nmi
                int64_t k1 = lim_end > 0 ? (lim_end + 2) / 3 : 0;
                int64_t k2 = lim_nmi > 0 ? (lim_nmi + 8) / 9 : 0;
                int64_t k = k1 < k2 ? k1 : k2;
                if (k > 1) {
#ifdef FM_PROF
                    prof_idle += k - 1;
#endif
                    s.exec_at = J + 3 * (k - 1);  // leave the last one to the normal path
                    s.zero_skip = 0;
                    continue;
                }
            }
            cur_iter = J;
#ifdef FM_PROF
            prof_cpu_instr++; prof_pc[s.pc]++;
#endif
            int S = execute();
            s.exec_at = J + (S >= 1 ? S : 1);
            s.zero_skip = S <= 0;
        }
        sync_to(3 * end);
        s.iter = end;
    }
};

}  // namespace fm

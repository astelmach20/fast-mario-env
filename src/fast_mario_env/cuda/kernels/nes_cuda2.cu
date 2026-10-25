// ---------------------------------------------------------------------------------------------
// v2 step kernel (compiled in the same unit as nes_cuda.cu, reusing its layout/env types).
//
// Differences from v1 (all semantics-preserving):
//  * one PPU catch-up site: an instruction that may touch a PPU register / OAM DMA is decoded first,
//    then the PPU is synced to 3*(J+1) (the target every sync() of that instruction would use; such
//    mid-instruction syncs can never reach the NMI because of nmi_thresh), then it executes.
//  * one run_frame site: the env layer (sticky, skip, hacks, early-termination replay) is a phase
//    machine around a single inlined frame loop, so all CPU/PPU scalars stay in registers.
//  * official opcodes run through a decode table + short per-class switches; everything else
//    (unofficial opcodes, code outside PRG, JMP (ind) through I/O) takes the original cascade.
//  * PRG/CHR/decode table live in shared memory.
// ---------------------------------------------------------------------------------------------

enum { K_SLOW = 0, K_IMPL = 1, K_BRANCH = 2, K_READ = 3, K_STORE = 4, K_RMW = 5, K_ACC = 6 };
enum { M_NONE = 0, M_IMM, M_ZP, M_ZPX, M_ZPY, M_ABS, M_ABSX, M_ABSY, M_INDX, M_INDY };
enum { A_ORA = 0, A_AND, A_EOR, A_ADC, A_LDA, A_CMP, A_SBC, A_BIT, A_LDY, A_CPY, A_CPX, A_LDX };
enum { S_STA = 0, S_STX, S_STY };
enum { R_ASL = 0, R_ROL, R_LSR, R_ROR, R_DEC, R_INC };

// per-env render log (AoS, AUXW words per env), written by k_step2 on render frames, read by k_render
#define NSEG 288
enum { AX_HDR = 0, AX_NSEG = 1, AX_VER = 2, AX_ISL = 3, AX_LP = 8, AX_PP = 248, AX_V0 = 488, AX_V0PAL = 1000,
       AX_V0OAM = 1008, AX_REC = 1072, AX_SCR = 1072 + 4 * NSEG,
       AUXW = AX_SCR + (256 + DBUF_ROWS * 256) / 4 + RD };  // AX_SCR: inline-render scratch (row, dbuf, sum)

#ifdef PROF
__device__ unsigned long long g_prof[16];
#define PADD(i, v) atomicAdd(&g_prof[i], (unsigned long long)(v))
#else
#define PADD(i, v)
#endif
struct Core {
    Sc s;
    Mem m;
    const uint8_t* __restrict__ prg;  // smem or global (read-only)
    const uint8_t* __restrict__ chr;
    int nt0, nt1, nt2, nt3;
    bool render;
    int skip;
    int bad_ops;
    int32_t tc_key;
    uint8_t tc_lo, tc_hi, tc_attr;
    const Rs* rs;
    uint8_t* obs;
#ifdef SCRATCH_LOCAL
    float sum[RD];
#else
    float* sum;      // inline-render scratch in global memory (keeps Core out of local memory)
#endif
    int spec0, spec1, gmin, gmax, dmin, dmax;
    bool sl_lazy, s0f;  // lazy sprite evaluation: SL/nsl stale, recompute for line sl_line
    int sl_line;
    // deferred rendering (render && !inline_r): log line segments, k_render draws them
    bool inline_r, conflict, dirty, lopen, lv_since, lo_since;
    uint32_t* aux;
    uint32_t rtag, nseg, vv, vo, ed;
    uint32_t ow0, ow1, ow2, ow3;   // open segment
    int run_y, run_start, run_cnt;
    FI bool logging() const { return render && !inline_r; }
    FI void cow_vram_hook() {  // before a VRAM/palette write during a logged frame
        if (vv == 0) {
            for (int i = 0; i < 512; i++) aux[AX_V0 + i] = m.W((OFF_VRAM >> 2) + i);
            for (int i = 0; i < 8; i++) aux[AX_V0PAL + i] = m.W((OFF_PAL >> 2) + i);
        } else if (lv_since) conflict = true;
        if (vv >= 0xfffe) conflict = true;
        vv++; lv_since = false; dirty = true;
    }
    FI void cow_oam_hook() {
        if (vo == 0) {
            for (int i = 0; i < 64; i++) aux[AX_V0OAM + i] = m.W((OFF_OAM >> 2) + i);
        } else if (lo_since) conflict = true;
        if (vo >= 0x3ffe) conflict = true;
        vo++; lo_since = false; dirty = true;
    }
    FI uint32_t lpack(int start, int cnt) const { return (uint32_t)start | ((uint32_t)cnt << 9) | (rtag << 16); }
    FI void run_close() {
        if (run_y < 0) return;
        uint32_t old = aux[AX_LP + run_y];
        aux[AX_PP + run_y] = (old >> 16) == rtag ? old : lpack(0, 0);
        aux[AX_LP + run_y] = lpack(run_start, run_cnt);
        run_y = -1;
    }
    FI void seg_flush() {
        if (!lopen) return;
        lopen = false;
        uint32_t k = nseg++;
        uint32_t* r = aux + AX_REC + 4 * k;
        r[0] = ow0; r[1] = ow1; r[2] = ow2; r[3] = ow3;
        int y = ow0 & 0xff;
        if (y == run_y && run_cnt < 127) run_cnt++;
        else { run_close(); run_y = y; run_start = (int)k; run_cnt = 1; }
    }
    FI void log_dots(int x0, int x1, uint16_t vb) {
        const uint32_t y = (uint32_t)s.scanline;
        if (lopen && (ow0 & 0xff) == y && (ow0 >> 17) == (uint32_t)x0 && !dirty) {
            ow0 = (ow0 & 0x1ffff) | ((uint32_t)x1 << 17);
        } else {
            seg_flush();
            if (nseg >= NSEG) { conflict = true; return; }
            lopen = true; dirty = false;
            ow0 = y | ((uint32_t)x0 << 8) | ((uint32_t)x1 << 17);
            ow1 = (uint32_t)vb | ((uint32_t)s.fine_x << 16) | ((uint32_t)s.bg_page << 19) | ((uint32_t)s.spr_page << 20) |
                  ((uint32_t)s.long_spr << 21) | ((uint32_t)s.show_bg << 22) | ((uint32_t)s.show_spr << 23) |
                  ((uint32_t)s.hide_edge_bg << 24) | ((uint32_t)s.hide_edge_spr << 25);
            ow2 = ed; ow3 = vv | (vo << 16);
        }
        lv_since = lo_since = true;
    }
    FI void log_begin() {
        rtag = (rtag + 1) & 0xffff;
        aux[AX_HDR] = rtag | (1u << 16);
        nseg = 0; vv = vo = 0; ed = 1;
        lv_since = lo_since = dirty = lopen = false; conflict = false; run_y = -1;
        sl_materialize();
        aux[AX_ISL] = m.W((OFF_SL >> 2)); aux[AX_ISL + 1] = m.W((OFF_SL >> 2) + 1); aux[AX_ISL + 2] = s.nsl;
    }
    FI void log_end() {
        seg_flush(); run_close();
        aux[AX_NSEG] = nseg; aux[AX_VER] = vv | (vo << 16);
    }
#ifdef SCRATCH_LOCAL
    uint8_t row[256];
    uint8_t dbuf[DBUF_ROWS][256];
#else
    uint8_t* row;
    uint8_t (*dbuf)[256];
#endif

    FI uint8_t& RAM(int a) { return m.B(OFF_RAM + a); }
    FI uint8_t& VRAM(int a) { return m.B(OFF_VRAM + a); }
    FI uint8_t& OAM(int a) { return m.B(OFF_OAM + a); }
    FI uint8_t& PAL(int a) { return m.B(OFF_PAL + a); }
    FI uint8_t& SL(int a) { return m.B(OFF_SL + a); }
    FI int ntoff(int t) const { return t == 0 ? nt0 : t == 1 ? nt1 : t == 2 ? nt2 : nt3; }
    FI uint8_t rd_prg(uint16_t a) const { return prg[a & 0x7fff]; }
    FI uint8_t rd_prg_fast(uint16_t a) const { return a >= 0x8000 ? rd_prg(a) : 0xff; }
    FI uint8_t CHR(int a) const { return chr[a]; }

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
        if (s.first_write) { s.t = (uint16_t)((s.t & ~0xff00) | ((a & 0x3f) << 8)); s.first_write = 0; }
        else { s.t = (uint16_t)((s.t & ~0xff) | a); s.v = s.t; s.first_write = 1; }
    }
    FI uint8_t ppu_get_data() {
        uint8_t d = pb_read(s.v);
        s.v = (uint16_t)(s.v + s.inc);
        if (s.v < 0x3f00) { uint8_t tmp = d; d = s.data_buffer; s.data_buffer = tmp; }
        return d;
    }
    FI void ppu_set_data(uint8_t d) { pb_write(s.v, d); s.v = (uint16_t)(s.v + s.inc); }
    FI void ppu_scroll(uint8_t sc) {
        if (s.first_write) { s.t = (uint16_t)((s.t & ~0x1f) | ((sc >> 3) & 0x1f)); s.fine_x = sc & 0x7; s.first_write = 0; }
        else { s.t = (uint16_t)((s.t & ~0x73e0) | ((sc & 0x7) << 12) | ((sc & 0xf8) << 2)); s.first_write = 1; }
    }
    FI uint8_t pb_read(uint16_t a) {
        if (a < 0x2000) return CHR(a);
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
            if (a == 0x3f10) PAL(0) = v; else PAL(a & 0x1f) = v;
        }
    }
    FI uint8_t nt_read(uint16_t a) { return VRAM(ntoff((a >> 10) & 3) + (a & 0x3ff)); }

    // ---------------- CPU bus (I/O without sync: the caller pre-synced)
    FI uint8_t joy_read(int i) {
        uint8_t r;
        if (i == 0) { if (s.joy_strobe0) r = s.joy_buttons0 & 1; else { r = s.joy_bits0 & 1; s.joy_bits0 >>= 1; } }
        else { if (s.joy_strobe1) r = s.joy_buttons1 & 1; else { r = s.joy_bits1 & 1; s.joy_bits1 >>= 1; } }
        return r | 0x40;
    }
    FI uint8_t rd_io(uint16_t a) {
        if (a < 0x4000) {
            switch (a & 0x2007) {
                case 0x2002: return ppu_status();
                case 0x2007: if (logging()) dirty = true; return ppu_get_data();
                case 0x2004: return OAM(s.oam_addr);
                default: return 0;
            }
        }
        if (a == 0x4016) return joy_read(0);
        if (a == 0x4017) return joy_read(1);
        return 0;
    }
    FI uint8_t rd(uint16_t a) {
        if (a >= 0x8000) return rd_prg(a);
        if (a < 0x2000) return RAM(a & 0x7ff);
        return rd_io(a);
    }
    FI void wr_io(uint16_t a, uint8_t v, int64_t J) {
        if (a < 0x4000) {
            if (logging()) {
                dirty = true;
                const int r_ = a & 0x2007;
                if (r_ == 0x2004) cow_oam_hook();
                if (r_ == 0x2007 && s.v >= 0x2000 && s.v < 0x3fff) cow_vram_hook();
            }
            switch (a & 0x2007) {
                case 0x2000: sl_materialize(); ppu_control(v); tc_key = -1; break;
                case 0x2001: ppu_mask(v); break;
                case 0x2003: sl_materialize(); s.oam_addr = v; break;
                case 0x2004: sl_materialize(); OAM(s.oam_addr) = v; s.oam_addr++; tc_key = -1; break;
                case 0x2005: ppu_scroll(v); break;
                case 0x2006: ppu_addr(v); break;
                case 0x2007: ppu_set_data(v); tc_key = -1; break;
                default: break;
            }
            return;
        }
        if (a == 0x4014) {
            sl_materialize();
            if (logging()) { dirty = true; if ((uint16_t)(v << 8) < 0x2000) cow_oam_hook(); }
            skip += 513 + (int)((J + 1) & 1);
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
    FI void wr(uint16_t a, uint8_t v, int64_t J) {
        if (a < 0x2000) { RAM(a & 0x7ff) = v; return; }
        wr_io(a, v, J);
    }

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
    FI void sprite_eval_line(int line) {
        int nsl = 0;
        int range = s.long_spr ? 16 : 8;
        for (int i = s.oam_addr / 4; i < 64; ++i) {
            int diff = line - OAM(i * 4);
            if (0 <= diff && diff < range) {
                SL(nsl++) = (uint8_t)i;
                if (nsl >= 8) break;
            }
        }
        s.nsl = (uint8_t)nsl;
    }
    // end-of-line evaluation: only the "sprite 0 is SL[0]" bit is needed by dots_fast; the full list
    // is materialized when something consumes it (rendering, OAM/oam_addr/ctrl writes, state store).
    FI void sprite_eval() {
        sl_lazy = true; sl_line = s.scanline;
        int diff = s.scanline - OAM(0);
        s0f = s.oam_addr < 4 && 0 <= diff && diff < (s.long_spr ? 16 : 8);
    }
    // Sprite-0 wait loop `LDA $2002; AND #$40; BEQ/BNE *-5` at s.pc, PPU synced for the LDA at J: skip
    // the iterations whose status reads provably see the current s0hit (they only clear vblank, which
    // is 0 after the first read, and set first_write). Bounded by the next dot that can change s0hit
    // (sprite-0 candidate line / pre-render clear), the NMI threshold and the frame end. Returns the
    // number of skipped iterations (0 = execute normally).
    FI int64_t poll_skip(int64_t J, int64_t fend) {
        const uint16_t pc0 = s.pc;
        if (pc0 < 0x8000 || pc0 > 0xfff0) return 0;
        if (rd_prg(pc0 + 3) != 0x29 || rd_prg(pc0 + 4) != 0x40 || rd_prg(pc0 + 6) != 0xF9) return 0;
        const uint8_t bop = rd_prg(pc0 + 5);
        if (bop != 0xF0 && bop != 0xD0) return 0;
        const bool wait_set = bop == 0xF0;
        if (wait_set == (s.s0hit != 0)) return 0;  // this read exits the loop
        const int64_t FRAME = 262LL * 341;
        int line;
        switch (s.pstate) {
            case PRE_RENDER: line = -1; break;
            case RENDER: line = s.scanline; break;
            case POST_RENDER: line = 240; break;
            default: line = s.scanline; break;
        }
        const int64_t P = (int64_t)(line + 1) * 341 + (s.pcyc - 1);
        int64_t d = ((int64_t)1 << 62);  // ppu calls until s0hit may change
        if (wait_set) {
            if (s.show_bg && s.show_spr) {
                const int range = s.long_spr ? 16 : 8;
                const int o0 = OAM(0);
                const int lo = o0 + 1 > 1 ? o0 + 1 : 1, hi = o0 + range < 239 ? o0 + range : 239;  // lines 1..239 via eval formula
                const bool f_ok = s.oam_addr < 4;
                if (s.pstate == RENDER) {
                    if (s.pcyc <= 256 && s0f) d = 0;
                    int L = line + 1 > lo ? line + 1 : lo;
                    if (f_ok && L <= hi) { int64_t q = (int64_t)(L + 1) * 341 - P; if (q < d) d = q; }
                } else {
                    // next frame: line 0 uses the (already made or pending) line-239 evaluation
                    bool l0 = (s.pstate == PRE_RENDER || (s.pstate == VERTICAL_BLANK) || s.pstate == POST_RENDER) ? s0f : false;
                    if (s.pstate == RENDER) l0 = false;
                    int64_t base = FRAME - P;
                    if (s.pstate == PRE_RENDER) base = -P;  // line 0 of this frame
                    if (l0) { int64_t q = base + 341; if (q < d) d = q; }
                    if (f_ok && lo <= hi) { int64_t q = base + (int64_t)(lo + 1) * 341; if (q < d) d = q; }
                }
            }
        } else {
            // s0hit is cleared at pre-render dot 1 (P = 0 of the next frame, or now)
            d = (s.pstate == PRE_RENDER) ? (s.pcyc <= 1 ? 0 : FRAME - P) : FRAME - P;
        }
        int64_t H = d == ((int64_t)1 << 62) ? ((int64_t)1 << 62) : s.ppu_calls + d - 3;
        if (s.nmi_thresh - 1 < H) H = s.nmi_thresh - 1;
        const uint16_t p2 = (uint16_t)(pc0 + 7);
        const int64_t C = (int64_t)OPERATION_CYCLES[0xAD] + OPERATION_CYCLES[0x29] + OPERATION_CYCLES[bop] + 1 +
                          (((p2 ^ pc0) & 0xff00) ? 2 : 0);
        // all skipped instructions < J + C*m: need 3*(J + C*m + 1) <= H and J + C*m <= fend
        int64_t m = (H / 3 - 1 - J) / C;
        const int64_t mf = (fend - J) / C;
        if (mf < m) m = mf;
        if (m <= 0 || H / 3 - 1 - J < 0) return 0;
        s.vblank = 0; s.first_write = 1;
        s.a = wait_set ? 0 : 0x40;
        s.p = (uint8_t)((s.p & ~(F_Z | F_N)) | (s.a ? 0 : F_Z));
        s.exec_at = J + C * m; s.zero_skip = 0;
        return m;
    }
    FI void sl_materialize() { if (sl_lazy) { sprite_eval_line(sl_line); sl_lazy = false; } }
    FI void sl_init() { sl_lazy = false; s0f = s.nsl > 0 && SL(0) == 0; }
    FI uint8_t bg_opaque(int x) {
        int x_fine = (s.fine_x + x) % 8;
        uint16_t addr = (uint16_t)(0x2000 | (s.v & 0x0FFF));
        uint8_t tile = nt_read(addr);
        addr = (uint16_t)((tile * 16) + ((s.v >> 12) & 0x7));
        addr |= s.bg_page << 12;
        uint8_t cc = (CHR(addr) >> (7 ^ x_fine)) & 1;
        cc |= ((CHR(addr + 8) >> (7 ^ x_fine)) & 1) << 1;
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
    FI void dots_fast(int x0, int x1) {
        if (!s.show_bg) return;
        if (!s.s0hit && s.show_spr && s0f) {
            int sx = OAM(3);
            int lo = x0 > sx ? x0 : sx;
            int hi = x1 < sx + 8 ? x1 : sx + 8;
            if (s.hide_edge_spr && lo < 8) lo = 8;
            if (lo < hi) {
                coarse_x_inc_n(inc_count(x0, lo));
                int y = s.scanline;
                for (int x = lo; x < hi; x++) {
                    if (!s.s0hit && (!s.hide_edge_bg || x >= 8)) {
                        if (bg_opaque(x) && spr_bits(0, x, y)) s.s0hit = 1;
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
        tc_lo = CHR(addr);
        tc_hi = CHR(addr + 8);
        uint16_t aaddr = (uint16_t)(0x23C0 | (s.v & 0x0C00) | ((s.v >> 4) & 0x38) | ((s.v >> 2) & 0x07));
        int shift = ((s.v >> 4) & 4) | (s.v & 2);
        tc_attr = (uint8_t)(((nt_read(aaddr) >> shift) & 0x3) << 2);
        tc_key = key;
    }
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
    FI void finish_row(int y, const uint8_t* r) {
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
    FI void render_end() {
        if (dmax < 0) return;
        for (int dy = dmin; dy <= dmax; dy++) {
            int r0 = __ldg(rs->gr + 2 * dy), r1 = __ldg(rs->gr + 2 * dy + 1);
            for (int dx = 0; dx < RD; dx++) {
                float sm = 0.f;
                for (int y = r0; y <= r1; y++) {
                    int e0 = __ldg(rs->ye + 2 * y);
                    int j = (e0 >= 0 && (e0 & 0xff) == dy) ? 0 : 1;
                    float v = __ldg(rs->yw + 2 * y + j) * hpass(dbuf[y - gmin], dx);
                    sm = (y == r0) ? v : sm + v;
                }
                put(dy, dx, sm);
            }
        }
    }
    FI void dots_render(int x0, int x1) {
        sl_materialize();
        const int y = s.scanline;
        uint8_t* dst = (y >= gmin && y <= gmax) ? dbuf[y - gmin] : row;
        // background palette address per dot -> dst (temporarily), then sprites, then gray
        if (!s.show_bg) {
            for (int x = x0; x < x1; x++) dst[x] = 0;
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
                    dst[x + j] = (s.hide_edge_bg && x + j < 8) ? 0 : (b ? (uint8_t)(b | tc_attr) : 0);
                }
                if (xf + span == 8) coarse_x_inc_n(1);
                x += span;
            }
        }
        if (s.show_spr && s.nsl) {
            int xs = (s.hide_edge_spr && x0 < 8) ? 8 : x0;
            int nsl = s.nsl;
            for (int x = xs; x < x1; x++) {
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
                uint8_t bgOpaque = dst[x] & 3;
                if ((fl & 2) && bgOpaque && !s.s0hit && s.show_bg) s.s0hit = 1;
                if (!bgOpaque || (fl & 1)) dst[x] = col;
            }
        }
        // gray (palette may change between segments: convert now)
        uint32_t g[8];
        for (int i = 0; i < 8; i++) {
            uint32_t pw = m.W((OFF_PAL >> 2) + i);
            g[i] = (uint32_t)__ldg(rs->lut + (pw & 0xff)) | ((uint32_t)__ldg(rs->lut + ((pw >> 8) & 0xff)) << 8) |
                   ((uint32_t)__ldg(rs->lut + ((pw >> 16) & 0xff)) << 16) | ((uint32_t)__ldg(rs->lut + (pw >> 24)) << 24);
        }
        for (int x = x0; x < x1; x++) { int p = dst[x]; dst[x] = (uint8_t)(g[p >> 2] >> (8 * (p & 3))); }
        if (x1 == 256 && y != spec0 && y != spec1) finish_row(y, dst);
    }
    FI void push(uint8_t v) { RAM((0x100 | s.sp) & 0x7ff) = v; s.sp--; }
    FI uint8_t pop() { ++s.sp; return RAM((0x100 | s.sp) & 0x7ff); }
    FI void nmi(int64_t call_index) {
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
    FI void ppu_run(int64_t n) {
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
                        if (render && inline_r) dots_render(cc - 1, (int)(cc - 1 + k));
                        else {
                            const uint16_t vb = s.v;
                            dots_fast(cc - 1, (int)(cc - 1 + k));
                            if (render) log_dots(cc - 1, (int)(cc - 1 + k), vb);
                        }
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
                        if (logging()) { ed = ((uint32_t)s.scanline << 1) | ((uint32_t)s.oam_addr << 9) | ((uint32_t)s.long_spr << 17) | (vo << 18); lo_since = true; }
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
                        if (s.scanline >= FRAME_END_SCANLINE) { s.pstate = PRE_RENDER; s.scanline = 0; s.even = !s.even; }
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
    FI uint16_t read_address(uint16_t a) { uint8_t lo = rd(a); uint8_t hi = rd((uint16_t)(a + 1)); return (uint16_t)(lo | hi << 8); }
    FI void page_crossed(uint16_t a, uint16_t b, int inc = 1) { if ((a & 0xff00) != (b & 0xff00)) skip += inc; }

    // the original nes-py decode cascade (rare path), I/O already synced
    FI void slow_exec(int64_t J) {
        uint8_t op = rd(s.pc++);
        if (slow_implied(op) || slow_branch(op) || slow_type1(op, J) || slow_type2(op, J) || slow_type0(op, J))
            skip += OPERATION_CYCLES[op];
        else bad_ops++;
    }
    FI bool slow_implied(uint8_t op) {
        switch (op) {
            case 0x00:
                ++s.pc; push((uint8_t)(s.pc >> 8)); push((uint8_t)s.pc); push((uint8_t)(s.p | 0b00100000 | (1 << 4)));
                s.p |= F_I; s.pc = read_address(0xfffe); skip += 7; break;
            case 0x08: push(s.p); break;
            case 0x18: setf(F_C, false); break;
            case 0x20: push((uint8_t)((s.pc + 1) >> 8)); push((uint8_t)(s.pc + 1)); s.pc = read_address(s.pc); break;
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
    FI bool slow_branch(uint8_t op) {
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
        } else ++s.pc;
        return true;
    }
    FI bool slow_type0(uint8_t op, int64_t J) {
        if ((op & 0x3) != 0x0) return false;
        uint16_t location = 0;
        switch ((op & 0x1c) >> 2) {
            case 0: location = s.pc++; break;
            case 1: location = rd(s.pc++); break;
            case 3: location = read_address(s.pc); s.pc += 2; break;
            case 5: location = (rd(s.pc++) + s.x) & 0xff; break;
            case 7: location = read_address(s.pc); s.pc += 2; page_crossed(location, (uint16_t)(location + s.x));
                    location = (uint16_t)(location + s.x); break;
            default: return false;
        }
        switch ((op & 0xe0) >> 5) {
            case 1: { uint16_t operand = rd(location); setf(F_Z, !(s.a & operand)); setf(F_V, operand & 0x40); setf(F_N, operand & 0x80); break; }
            case 4: wr(location, s.y, J); break;
            case 5: s.y = rd(location); set_ZN(s.y); break;
            case 6: { uint16_t diff = (uint16_t)(s.y - rd(location)); setf(F_C, !(diff & 0x100)); set_ZN((uint8_t)diff); break; }
            case 7: { uint16_t diff = (uint16_t)(s.x - rd(location)); setf(F_C, !(diff & 0x100)); set_ZN((uint8_t)diff); break; }
            default: return false;
        }
        return true;
    }
    FI bool slow_type1(uint8_t op, int64_t J) {
        if ((op & 0x3) != 0x1) return false;
        uint16_t location = 0;
        int opn = (op & 0xe0) >> 5;
        const int STA = 4;
        switch ((op & 0x1c) >> 2) {
            case 0: { uint8_t za = (uint8_t)(s.x + rd(s.pc++)); uint8_t lo = rd(za & 0xff); uint8_t hi = rd((za + 1) & 0xff);
                      location = (uint16_t)(lo | hi << 8); break; }
            case 1: location = rd(s.pc++); break;
            case 2: location = s.pc++; break;
            case 3: location = read_address(s.pc); s.pc += 2; break;
            case 4: { uint8_t za = rd(s.pc++); uint8_t lo = rd(za & 0xff); uint8_t hi = rd((za + 1) & 0xff);
                      location = (uint16_t)(lo | hi << 8);
                      if (opn != STA) page_crossed(location, (uint16_t)(location + s.y));
                      location = (uint16_t)(location + s.y); break; }
            case 5: location = (rd(s.pc++) + s.x) & 0xff; break;
            case 6: location = read_address(s.pc); s.pc += 2;
                    if (opn != STA) page_crossed(location, (uint16_t)(location + s.y));
                    location = (uint16_t)(location + s.y); break;
            default: location = read_address(s.pc); s.pc += 2;
                     if (opn != STA) page_crossed(location, (uint16_t)(location + s.x));
                     location = (uint16_t)(location + s.x); break;
        }
        switch (opn) {
            case 0: s.a |= rd(location); set_ZN(s.a); break;
            case 1: s.a &= rd(location); set_ZN(s.a); break;
            case 2: s.a ^= rd(location); set_ZN(s.a); break;
            case 3: { uint8_t operand = rd(location); uint16_t sum_ = (uint16_t)(s.a + operand + getf(F_C));
                      setf(F_C, sum_ & 0x100); setf(F_V, (s.a ^ sum_) & (operand ^ sum_) & 0x80); s.a = (uint8_t)sum_; set_ZN(s.a); break; }
            case 4: wr(location, s.a, J); break;
            case 5: s.a = rd(location); set_ZN(s.a); break;
            case 6: { uint16_t diff = (uint16_t)(s.a - rd(location)); setf(F_C, !(diff & 0x100)); set_ZN((uint8_t)diff); break; }
            default: { uint16_t subtrahend = rd(location); uint16_t diff = (uint16_t)(s.a - subtrahend - !getf(F_C));
                       setf(F_C, !(diff & 0x100)); setf(F_V, (s.a ^ diff) & (~subtrahend ^ diff) & 0x80); s.a = (uint8_t)diff;
                       set_ZN((uint8_t)diff); break; }
        }
        return true;
    }
    FI bool slow_type2(uint8_t op, int64_t J) {
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
            case 5: { location = rd(s.pc++); uint8_t index = (opn == LDX || opn == STX) ? s.y : s.x; location = (location + index) & 0xff; break; }
            case 7: { location = read_address(s.pc); s.pc += 2; uint8_t index = (opn == LDX || opn == STX) ? s.y : s.x;
                      page_crossed(location, (uint16_t)(location + index)); location = (uint16_t)(location + index); break; }
            default: return false;
        }
        uint16_t operand = 0;
        switch (opn) {
            case ASL: case ROL:
                if (mode == 2) { bool prev_C = getf(F_C); setf(F_C, s.a & 0x80); s.a <<= 1; s.a = (uint8_t)(s.a | (prev_C && (opn == ROL))); set_ZN(s.a); }
                else { bool prev_C = getf(F_C); operand = rd(location); setf(F_C, operand & 0x80);
                       operand = (uint16_t)(operand << 1 | (prev_C && (opn == ROL))); set_ZN((uint8_t)operand); wr(location, (uint8_t)operand, J); }
                break;
            case LSR: case ROR:
                if (mode == 2) { bool prev_C = getf(F_C); setf(F_C, s.a & 1); s.a >>= 1; s.a = (uint8_t)(s.a | (prev_C && (opn == ROR)) << 7); set_ZN(s.a); }
                else { bool prev_C = getf(F_C); operand = rd(location); setf(F_C, operand & 1);
                       operand = (uint16_t)(operand >> 1 | (prev_C && (opn == ROR)) << 7); set_ZN((uint8_t)operand); wr(location, (uint8_t)operand, J); }
                break;
            case STX: wr(location, s.x, J); break;
            case LDX: s.x = rd(location); set_ZN(s.x); break;
            case DEC: { int tmp = rd(location) - 1; set_ZN((uint8_t)tmp); wr(location, (uint8_t)tmp, J); break; }
            default:  { int tmp = rd(location) + 1; set_ZN((uint8_t)tmp); wr(location, (uint8_t)tmp, J); break; }
        }
        return true;
    }
};

// ---------------------------------------------------------------------------- SMB helpers on Core
FI int64_t digits2(Core& n, int a, int cnt) {
    int64_t v = 0;
    for (int i = 0; i < cnt; i++) { int b = n.RAM(a + i); v = v * (b >= 100 ? 1000 : b >= 10 ? 100 : 10) + b; }
    return v;
}
FI bool smb_flag2(Core& n) {
    if (n.RAM(0x0770) == 2) return true;
    for (int a = 0x16; a <= 0x1A; a++) { int r = n.RAM(a); if (r == 0x2D || r == 0x31) return n.RAM(0x001D) == 3; }
    return false;
}

#ifndef BLOCK2
#define BLOCK2 128
#endif

// mode 0: step all envs (mode 1 = reset is handled by the v1 kernel)
#ifndef MINB
#define MINB 4
#endif
extern "C" __global__ void __launch_bounds__(BLOCK2, MINB) k_step2(StepArgs P, const uint2* __restrict__ dtab_g, uint32_t* __restrict__ aux) {
#ifdef ROM_SMEM
    __shared__ uint32_t sh_rom[(0x8000 + 0x2000) / 4];
#endif
    __shared__ uint2 sh_dtab[256];
#ifdef ROM_SMEM
    for (int i = threadIdx.x; i < (0x8000 + 0x2000) / 4; i += blockDim.x) sh_rom[i] = ((const uint32_t*)P.cart)[i];
#endif
    for (int i = threadIdx.x; i < 256; i += blockDim.x) sh_dtab[i] = dtab_g[i];
    __syncthreads();
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned wmask = __ballot_sync(0xffffffffu, e < P.n);
    if (e >= P.n) return;

    Rs rs{P.xs, P.xw, P.ye, P.yw, P.lut, P.gr};
    Core n;
    n.m = mem_of(P.state, e, P.np);
#ifndef ROM_SMEM
    n.prg = P.cart; n.chr = P.cart + 0x8000;   // read-only, L1-cached
#else
    n.prg = (const uint8_t*)sh_rom; n.chr = (const uint8_t*)sh_rom + 0x8000;
#endif
    n.nt0 = P.nt[0]; n.nt1 = P.nt[1]; n.nt2 = P.nt[2]; n.nt3 = P.nt[3];
    n.bad_ops = 0; n.tc_key = -1; n.tc_lo = n.tc_hi = n.tc_attr = 0; n.rs = &rs;
    n.obs = P.newobs + (size_t)e * 84 * 84;
    n.aux = aux + (size_t)e * AUXW;
#ifndef SCRATCH_LOCAL
    n.row = (uint8_t*)(n.aux + AX_SCR);
    n.dbuf = (uint8_t(*)[256])(n.aux + AX_SCR + 64);
    n.sum = (float*)(n.aux + AX_SCR + 64 + DBUF_ROWS * 64);
#endif
    n.rtag = n.aux[AX_HDR] & 0xffff;
    n.inline_r = false; n.conflict = false; n.lopen = false; n.dirty = false; n.run_y = -1;
#ifdef INLINE_RENDER
    n.inline_r = true; n.aux[AX_HDR] = n.rtag;
#endif
    load_sc(n.s, n.m);
    n.sl_init();
    EnvS E;
    Pcg64 rng;
    const int L = P.A.level[e];
    const int K = P.cfg.skip;
    const double sticky = P.cfg.sticky;
    Mem snap = mem_of(P.snap, e, P.np);
    copy_bytes(snap, n.m); store_sc(n.s, snap);

    const int a_in = (int)P.actions[e];
    uint8_t acts[16];
    double total;
    bool done;
    int i, j, last, ph, draws;
#ifdef PROF
    long long pr_t0 = clock64(), pr_ppu = 0, pr_it = 0, pr_sync = 0, pr_ins = 0, pr_fr = 0;
#endif
    enum { MD_ATTEMPT = 0, MD_FSTART, MD_RUN, MD_FEND, MD_DONE };
    int mode = MD_ATTEMPT, attempt = 0;
    int ae = 0; uint8_t buttons = 0;
    int64_t fend = 0, tgt = 0, J = 0;
    bool pend = false;
    uint32_t D = 0, U = 0; uint16_t loc = 0, npc = 0; uint8_t b1 = 0, b2 = 0; int extra = 0;
    // flat state machine; every iteration re-converges the warp (lanes that finished their step idle)
#ifndef PARK
#define PARK 1
#endif
    for (;;) {
        // lanes with expensive work pending (PPU catch-up / slow instruction) park until PARK lanes of the
        // warp have some (or nobody has cheap work), so those divergent blocks run for many lanes at once
        const bool want = mode == MD_RUN && (pend || tgt > n.s.ppu_calls);
        const unsigned bw = __ballot_sync(wmask, want);
        const unsigned bf = __ballot_sync(wmask, mode != MD_DONE && !want);
        if ((bw | bf) == 0) break;
        if (want && !(__popc(bw) >= PARK || bf == 0)) continue;
        if (mode == MD_ATTEMPT) {
        if (attempt) {  // a logged frame could not be deferred -> redo the step with inline rendering
        copy_bytes(n.m, snap); load_sc(n.s, snap); n.sl_init();
        n.inline_r = true; n.conflict = false; n.lopen = false; n.run_y = -1;
        n.aux[AX_HDR] = n.rtag;  // tells k_render the obs was produced inline
#ifdef PROF
        PADD(7, 1);
#endif
    }
    load_env(E, rng, P.A, e);
    total = 0.0; done = false;
    i = 0; j = 0; last = 0; ph = 0; draws = 0;
    // first frame
    ae = a_in;
    draws++;
    if (sticky != 0.0 && rng.random() < sticky) ae = E.prev;
    E.prev = ae; acts[0] = (uint8_t)ae;
    buttons = SIMPLE_MOVEMENT[ae];
    n.render = (K - 1 == 0);

            mode = MD_FSTART;
        }
        if (mode == MD_FSTART) {
#ifdef PROF
        pr_fr++;
#endif
        // ================================================================ one emulated frame
        n.s.joy_buttons0 = buttons;
#ifdef NO_RENDER
        n.render = false;
#endif
        {
            fend = n.s.iter + CYCLES_PER_FRAME;
            n.tc_key = -1;
            if (n.render && !n.inline_r) { n.log_begin(); n.spec0 = n.spec1 = -1; n.gmin = n.dmin = 1000; n.gmax = n.dmax = -1; }
            else if (n.render) n.render_begin();
            else { n.spec0 = n.spec1 = -1; n.gmin = n.dmin = 1000; n.gmax = n.dmax = -1; }
            tgt = 0; pend = false;
        }
            mode = MD_RUN;
        }
        if (mode == MD_RUN) {
            do {
#ifdef PROF
                pr_it++;
                if (tgt > n.s.ppu_calls) { long long c0 = clock64(); n.ppu_run(tgt - n.s.ppu_calls); n.update_nmi_thresh(); pr_ppu += clock64() - c0; pr_sync++; }
#else
                if (tgt > n.s.ppu_calls) { n.ppu_run(tgt - n.s.ppu_calls); n.update_nmi_thresh(); }
#endif
                tgt = 0;
                if (n.conflict) { mode = MD_FEND; break; }
                if (!pend) {
                    if (n.s.exec_at >= fend) {
                        if (3 * fend > n.s.ppu_calls) { tgt = 3 * fend; PADD(11, 1); continue; }
                        mode = MD_FEND; break;
                    }
                    J = n.s.exec_at;
                    if (3 * (J + 1) >= n.s.nmi_thresh && 3 * (J + 1) > n.s.ppu_calls) { tgt = 3 * (J + 1); PADD(10, 1); continue; }
                    const uint16_t pc = n.s.pc;
                    if (n.rd_prg_fast(pc) == 0x4C && (uint16_t)(n.rd_prg_fast(pc + 1) | n.rd_prg_fast(pc + 2) << 8) == pc) {
                        int64_t lim_end = fend - J;
                        int64_t lim_nmi = n.s.nmi_thresh - 3 * (J + 1);
                        int64_t k1 = lim_end > 0 ? (lim_end + 2) / 3 : 0;
                        int64_t k2 = lim_nmi > 0 ? (lim_nmi + 8) / 9 : 0;
                        int64_t k = k1 < k2 ? k1 : k2;
                        if (k > 1) { n.s.exec_at = J + 3 * (k - 1); n.s.zero_skip = 0; continue; }
                    }
                    // ---- decode (uniform: every lane runs the same straight-line code)
                    const bool inprg = pc >= 0x8000 && pc <= 0xfffc;
                    const uint8_t op = n.prg[pc & 0x7fff];
                    const uint2 DD = sh_dtab[op];
                    D = DD.x; U = DD.y;
                    b1 = n.prg[(pc + 1) & 0x7fff]; b2 = n.prg[(pc + 2) & 0x7fff];
                    const uint16_t abs16 = (uint16_t)(b1 | (b2 << 8));
                    const int mode = (D >> 4) & 15;
                    const int kind = D & 15;
                    const bool mx = mode == M_ZPX || mode == M_ABSX || mode == M_INDX;
                    const uint8_t idx = mx ? n.s.x : (mode == M_ZPY || mode == M_ABSY || mode == M_INDY) ? n.s.y : 0;
                    const bool isind = mode == M_INDX || mode == M_INDY;
                    const uint8_t za = (uint8_t)(b1 + (mode == M_INDX ? n.s.x : 0));
                    uint16_t ptr = 0;
                    if (isind && inprg) ptr = (uint16_t)(n.RAM(za) | (n.RAM((za + 1) & 0xff) << 8));
                    const bool zpm = mode == M_ZP || mode == M_ZPX || mode == M_ZPY;
                    const uint16_t base = isind ? ptr : zpm ? (uint16_t)b1 : abs16;
                    const bool post = mode == M_ABSX || mode == M_ABSY || mode == M_INDY;
                    loc = zpm ? (uint16_t)((b1 + idx) & 0xff) : mode == M_INDX ? ptr : (uint16_t)(base + (post ? idx : 0));
                    if (mode == M_IMM) loc = (uint16_t)(pc + 1);
                    extra = ((D & 0x100) && post && ((base ^ loc) & 0xff00)) ? 1 : 0;
                    const int len = mode == M_NONE ? 1 : (mode == M_ABS || mode == M_ABSX || mode == M_ABSY) ? 3 : 2;
                    npc = (uint16_t)(pc + len);
                    const bool memop = kind >= K_READ && kind <= K_RMW;
                    const bool ioloc = loc >= 0x2000 && loc < 0x8000;
                    bool needsync = !inprg || kind == K_SLOW ||
                                    (memop && ioloc && (loc < 0x4000 || (loc == 0x4014 && kind != K_READ)));
                    if (inprg && kind == K_IMPL && op == 0x6C && ioloc) { D = (D & ~15u) | K_SLOW; needsync = true; }
                    if (!inprg) D = K_SLOW;
                    const bool slow = needsync || !(U & 1) || (memop && ioloc);
                    if (slow) {
                        pend = true;
                        if (needsync && 3 * (J + 1) > n.s.ppu_calls) tgt = 3 * (J + 1);
                        continue;  // executed (batched) in a later iteration
                    }
                }
                if (!pend) {
#ifdef PROF
                    pr_ins++;
#endif
                    // ---- uniform fast execute: no I/O, official opcode, operands in RAM/PRG
                    Sc& s = n.s;
                    const int alu = (U >> 1) & 15, src = (U >> 5) & 7, dst = (U >> 8) & 7, ar = (U >> 11) & 3;
                    const int ctrl = (U >> 16) & 7;
                    const uint8_t sp = s.sp;
                    const bool pull = src == 6, isrts = ctrl == 3;
                    if (pull || isrts) loc = (uint16_t)(0x100 | (uint8_t)(sp + 1));
                    const bool rdp = (src == 0 && ((D >> 4) & 15) != M_NONE) || pull || isrts;
                    uint8_t v = 0, hi = 0;
                    if (rdp && loc < 0x2000) v = n.RAM(loc & 0x7ff);
                    if (loc >= 0x8000) v = n.prg[loc & 0x7fff];
                    if (isrts) hi = n.RAM(0x100 | (uint8_t)(sp + 2));
                    const uint8_t sv = src == 0 || src == 6 ? v : src == 1 ? s.a : src == 2 ? s.x : src == 3 ? s.y : src == 4 ? sp : s.p;
                    const uint8_t av = ar == 0 ? s.a : ar == 1 ? s.x : s.y;
                    const int C = (s.p & F_C) ? 1 : 0;
                    const uint8_t o = alu == 5 ? (uint8_t)~sv : sv;
                    const int sum = av + o + C;
                    const int diff = av - sv;
                    uint8_t r;
                    switch (alu) {
                        case 1: r = av | sv; break;
                        case 2: r = av & sv; break;
                        case 3: r = av ^ sv; break;
                        case 4: case 5: r = (uint8_t)sum; break;
                        case 6: r = (uint8_t)diff; break;
                        case 8: r = (uint8_t)(sv << 1); break;
                        case 9: r = (uint8_t)((sv << 1) | C); break;
                        case 10: r = (uint8_t)(sv >> 1); break;
                        case 11: r = (uint8_t)((sv >> 1) | (C << 7)); break;
                        case 12: r = (uint8_t)(sv + 1); break;
                        case 13: r = (uint8_t)(sv - 1); break;
                        default: r = sv; break;
                    }
                    uint8_t p = s.p;
                    if (U & (1u << 13)) p = (uint8_t)((p & ~(F_Z | F_N)) | (r ? 0 : F_Z) | ((r & 0x80) ? F_N : 0));
                    const int cout = (alu == 4 || alu == 5) ? ((sum >> 8) & 1) : alu == 6 ? (diff >= 0) :
                                     (alu == 8 || alu == 9) ? (sv >> 7) : (sv & 1);
                    if (U & (1u << 14)) p = cout ? (p | F_C) : (p & ~F_C);
                    if (U & (1u << 15)) p = ((av ^ sum) & (o ^ sum) & 0x80) ? (p | F_V) : (p & ~F_V);
                    if (alu == 7) p = (uint8_t)((p & ~(F_Z | F_V | F_N)) | ((av & sv) ? 0 : F_Z) | ((sv & 0x40) ? F_V : 0) | ((sv & 0x80) ? F_N : 0));
                    const uint8_t fm = (uint8_t)(U >> 24);
                    if (alu == 14) p = (U & (1u << 22)) ? (p | fm) : (p & ~fm);
                    if (dst == 5) p = r;
                    s.p = p;
                    if (dst == 1) s.a = r;
                    if (dst == 2) s.x = r;
                    if (dst == 3) s.y = r;
                    // stack pointer
                    uint8_t nsp = sp;
                    if (dst == 4) nsp = r;
                    if (pull) nsp = (uint8_t)(sp + 1);
                    if (isrts) nsp = (uint8_t)(sp + 2);
                    if (dst == 7) nsp = (uint8_t)(sp - 1);
                    if (ctrl == 2) nsp = (uint8_t)(sp - 2);
                    s.sp = nsp;
                    // memory writes
                    const uint16_t ret = (uint16_t)(npc + 1);
                    const bool w0 = (dst == 6 && loc < 0x2000) || dst == 7 || ctrl == 2;
                    const int wa0 = dst == 6 ? (loc & 0x7ff) : (0x100 | sp);
                    const uint8_t wv0 = ctrl == 2 ? (uint8_t)(ret >> 8) : r;
                    if (w0) n.RAM(wa0) = wv0;
                    if (ctrl == 2) n.RAM(0x100 | (uint8_t)(sp - 1)) = (uint8_t)ret;
                    // program counter + cycles
                    int S = (int)(D >> 24) + extra;
                    uint16_t npc2 = npc;
                    if (ctrl == 1 || ctrl == 2) npc2 = (uint16_t)(b1 | (b2 << 8));
                    if (isrts) npc2 = (uint16_t)((v | (hi << 8)) + 1);
                    if (ctrl == 4) {
                        const uint8_t bop = (uint8_t)(D >> 16);
                        const uint8_t f = (bop >> 6) == 0 ? F_N : (bop >> 6) == 1 ? F_V : (bop >> 6) == 2 ? F_C : F_Z;
                        const bool br = !(((bop & 0x20) != 0) ^ ((p & f) != 0));
                        const uint16_t p2 = (uint16_t)(npc + 1);
                        const uint16_t newPC = (uint16_t)(p2 + (int8_t)b1);
                        npc2 = br ? newPC : p2;
                        S += br ? 1 + (((p2 ^ newPC) & 0xff00) ? 2 : 0) : 0;
                    }
                    s.pc = npc2;
                    n.s.exec_at = J + (S >= 1 ? S : 1);
                    n.s.zero_skip = S <= 0;
                    continue;
                }
                pend = false;
#ifdef PROF
                pr_ins++; PADD(8, 1); if (((D & 15) == K_READ) && loc == 0x2002) PADD(9, 1); if ((D & 15) == K_STORE || (D & 15) == K_RMW) PADD(12, 1);
#endif
#ifndef NO_POLL_SKIP
                if ((D & 15) == K_READ && loc == 0x2002 && ((D >> 16) & 0xff) == 0xAD) {
                    const int64_t m = n.poll_skip(J, fend);
                    if (m > 0) {
#ifdef PROF
                        PADD(13, m);
#endif
                        continue;
                    }
                }
#endif
                // ---- slow execute (I/O, rare opcodes): the v2 per-class switches
                n.skip = 0;
                const int kind = D & 15;
                Sc& s = n.s;
                switch (kind) {
                    case K_SLOW: n.slow_exec(J); break;
                    case K_BRANCH: {
                        const uint8_t op = (uint8_t)(D >> 16);
                        bool br = op & 0x20;
                        uint8_t f = (op >> 6) == 0 ? F_N : (op >> 6) == 1 ? F_V : (op >> 6) == 2 ? F_C : F_Z;
                        br = !(br ^ ((s.p & f) != 0));
                        if (br) {
                            uint16_t p2 = (uint16_t)(npc + 1);
                            uint16_t newPC = (uint16_t)(p2 + (int8_t)b1);
                            n.skip += 1 + (((p2 ^ newPC) & 0xff00) ? 2 : 0);
                            s.pc = newPC;
                        } else s.pc = (uint16_t)(npc + 1);
                        break;
                    }
                    case K_READ: {
                        s.pc = npc;
                        uint8_t v = n.rd(loc);
                        switch ((D >> 12) & 15) {
                            case A_ORA: s.a |= v; n.set_ZN(s.a); break;
                            case A_AND: s.a &= v; n.set_ZN(s.a); break;
                            case A_EOR: s.a ^= v; n.set_ZN(s.a); break;
                            case A_ADC: case A_SBC: {
                                uint8_t o = ((D >> 12) & 15) == A_ADC ? v : (uint8_t)~v;
                                uint16_t sum_ = (uint16_t)(s.a + o + ((s.p & F_C) ? 1 : 0));
                                n.setf(F_C, sum_ & 0x100);
                                n.setf(F_V, (s.a ^ sum_) & (o ^ sum_) & 0x80);
                                s.a = (uint8_t)sum_;
                                n.set_ZN(s.a);
                                break;
                            }
                            case A_LDA: s.a = v; n.set_ZN(v); break;
                            case A_LDX: s.x = v; n.set_ZN(v); break;
                            case A_LDY: s.y = v; n.set_ZN(v); break;
                            case A_CMP: case A_CPX: case A_CPY: {
                                int ai = (D >> 12) & 15;
                                uint8_t r = ai == A_CMP ? s.a : ai == A_CPX ? s.x : s.y;
                                uint16_t diff = (uint16_t)(r - v);
                                n.setf(F_C, !(diff & 0x100));
                                n.set_ZN((uint8_t)diff);
                                break;
                            }
                            default: {  // BIT
                                n.setf(F_Z, !(s.a & v));
                                n.setf(F_V, v & 0x40);
                                n.setf(F_N, v & 0x80);
                                break;
                            }
                        }
                        break;
                    }
                    case K_STORE: {
                        s.pc = npc;
                        int si = (D >> 12) & 15;
                        n.wr(loc, si == S_STA ? s.a : si == S_STX ? s.x : s.y, J);
                        break;
                    }
                    case K_RMW: case K_ACC: {
                        s.pc = npc;
                        uint8_t v = kind == K_ACC ? s.a : n.rd(loc);
                        int ri = (D >> 12) & 15;
                        uint8_t r;
                        switch (ri) {
                            case R_ASL: case R_ROL: { bool pc_ = (s.p & F_C) != 0; n.setf(F_C, v & 0x80); r = (uint8_t)((v << 1) | (pc_ && ri == R_ROL)); break; }
                            case R_LSR: case R_ROR: { bool pc_ = (s.p & F_C) != 0; n.setf(F_C, v & 1); r = (uint8_t)((v >> 1) | ((pc_ && ri == R_ROR) << 7)); break; }
                            case R_DEC: r = (uint8_t)(v - 1); break;
                            default: r = (uint8_t)(v + 1); break;
                        }
                        n.set_ZN(r);
                        if (kind == K_ACC) s.a = r; else n.wr(loc, r, J);
                        break;
                    }
                    default: {  // implied
                        const uint8_t op = (uint8_t)(D >> 16);
                        s.pc = npc;
                        switch (op) {
                            case 0x00:
                                ++s.pc; n.push((uint8_t)(s.pc >> 8)); n.push((uint8_t)s.pc); n.push((uint8_t)(s.p | 0b00100000 | (1 << 4)));
                                s.p |= F_I; s.pc = (uint16_t)(n.rd_prg(0xfffe) | n.rd_prg(0xffff) << 8); n.skip += 7; break;
                            case 0x08: n.push(s.p); break;
                            case 0x18: n.setf(F_C, false); break;
                            case 0x20: n.push((uint8_t)((npc + 1) >> 8)); n.push((uint8_t)(npc + 1)); s.pc = loc; break;
                            case 0x28: s.p = n.pop(); break;
                            case 0x38: n.setf(F_C, true); break;
                            case 0x40: s.p = n.pop(); s.pc = n.pop(); s.pc |= n.pop() << 8; break;
                            case 0x48: n.push(s.a); break;
                            case 0x4C: s.pc = loc; break;
                            case 0x58: n.setf(F_I, false); break;
                            case 0x60: s.pc = n.pop(); s.pc |= n.pop() << 8; ++s.pc; break;
                            case 0x68: s.a = n.pop(); n.set_ZN(s.a); break;
                            case 0x6C: {
                                uint8_t lo = n.rd(loc);
                                uint8_t hi = n.rd((uint16_t)((loc & 0xff00) | ((loc + 1) & 0xff)));
                                s.pc = (uint16_t)(lo | hi << 8);
                                break;
                            }
                            case 0x78: n.setf(F_I, true); break;
                            case 0x88: --s.y; n.set_ZN(s.y); break;
                            case 0x8a: s.a = s.x; n.set_ZN(s.a); break;
                            case 0x98: s.a = s.y; n.set_ZN(s.a); break;
                            case 0x9a: s.sp = s.x; break;
                            case 0xa8: s.y = s.a; n.set_ZN(s.y); break;
                            case 0xaa: s.x = s.a; n.set_ZN(s.x); break;
                            case 0xb8: n.setf(F_V, false); break;
                            case 0xba: s.x = s.sp; n.set_ZN(s.x); break;
                            case 0xc8: ++s.y; n.set_ZN(s.y); break;
                            case 0xca: --s.x; n.set_ZN(s.x); break;
                            case 0xd8: n.setf(F_D, false); break;
                            case 0xe8: ++s.x; n.set_ZN(s.x); break;
                            case 0xf8: n.setf(F_D, true); break;
                            default: break;  // NOP
                        }
                        break;
                    }
                }
                if (kind != K_SLOW) n.skip += (int)(D >> 24) + extra;
                const int S = n.skip;
                n.s.exec_at = J + (S >= 1 ? S : 1);
                n.s.zero_skip = S <= 0;
            } while (0);
        }
        if (mode == MD_FEND) {
            n.s.iter = fend;
            if (n.conflict) { attempt++; mode = MD_ATTEMPT; continue; }
            if (n.render) { if (n.inline_r) n.render_end(); else n.log_end(); E.obs_is_backup = false; }
            mode = MD_FSTART;
            do {
        // ================================================================ env layer (phase machine)
        if (ph == 0) {  // main frame i finished: SuperMarioBrosEnv reward / done / info
            int64_t x = (int64_t)n.RAM(0x6d) * 0x100 + n.RAM(0x86);
            int64_t xr = x - E.x_last; E.x_last = x;
            if (xr < -5 || xr > 5) xr = 0;
            int64_t tm = digits2(n, 0x7f8, 3);
            int64_t tp = tm - E.time_last; E.time_last = tm;
            if (tp > 0) tp = 0;
            int ps = n.RAM(0x0e);
            bool dying = ps == 0x0b || n.RAM(0x00b5) > 1, dead = ps == 0x06;
            int64_t r = xr + tp + ((dying || dead) ? -25 : 0);
            if (r < -15) r = -15;
            if (r > 15) r = 15;
            total += (double)r;
            Info& I = E.info;
            I.coins = (int)digits2(n, 0x7ed, 2);
            I.flag_get = smb_flag2(n);
            I.life = n.RAM(0x075a);
            I.score = (int)digits2(n, 0x7de, 6);
            I.stage = n.RAM(0x075c) + 1;
            int st = n.RAM(0x0756);
            I.status = st == 0 ? 0 : st == 1 ? 1 : 2;
            I.time = (int)tm;
            I.world = n.RAM(0x075f) + 1;
            I.x_pos = (int)x;
            int yp = n.RAM(0x03b8);
            I.y_pos = n.RAM(0x00b5) < 1 ? 255 + (255 - yp) : 255 - yp;
            done = dying || dead || I.flag_get;
            if (done) {
                last = i;
                if (last < K - 1) {  // early termination: replay from the step snapshot, render frame `last`
                    copy_bytes(n.m, snap); load_sc(n.s, snap); n.sl_init();
                    j = 0;
                    if (last == 0) { ph = 4; n.render = true; } else { ph = 2; n.render = false; }
                    buttons = SIMPLE_MOVEMENT[acts[0]];
                    continue;
                }
                mode = MD_DONE; break;
            }
            uint8_t cat = n.RAM(0x06DE);
            if (cat > 1 && cat < 255) n.RAM(0x06DE) = 1;
            ph = 1;
        } else if (ph == 2) {
            uint8_t cat = n.RAM(0x06DE);
            if (cat > 1 && cat < 255) n.RAM(0x06DE) = 1;
            ph = 3;
        } else if (ph == 4) {
            mode = MD_DONE; break;
        }
        // ph 1 / 3: _skip_occupied_states
        {
            int ps = n.RAM(0x0e);
            if (ps <= 5 || ps == 7 || n.RAM(0x0770) == 2) { n.RAM(0x07A0) = 0; buttons = 0; continue; }
        }
        if (ph == 1) {
            i++;
            if (i == K) { mode = MD_DONE; break; }
            ae = a_in;
            draws++;
            if (sticky != 0.0 && rng.random() < sticky) ae = E.prev;
            E.prev = ae; acts[i] = (uint8_t)ae;
            buttons = SIMPLE_MOVEMENT[ae];
            n.render = (i == K - 1);
            ph = 0;
        } else {  // ph 3
            j++;
            buttons = SIMPLE_MOVEMENT[acts[j]];
            if (j == last) { ph = 4; n.render = true; } else { ph = 2; n.render = false; }
        }
                } while (0);
        }
    }
#ifdef PROF
    PADD(0, clock64() - pr_t0); PADD(1, pr_ppu); PADD(2, pr_it); PADD(3, pr_sync); PADD(4, pr_ins); PADD(5, pr_fr); PADD(6, 1);
#endif
    n.sl_materialize();
    // ---------------------------------------------------------------- reference step() tail
    bool te = done;
    int life = E.info.life;
    if (E.lives >= 0 && life < E.lives) te = true;
    E.lives = life;
    if (E.info.flag_get) te = true;
    E.t++;
    bool tr = E.t >= P.cfg.max_steps;
    E.ep_ret += total; E.ep_len++;
    P.rew[e] = (float)total; P.term[e] = te; P.trunc[e] = tr;
    int* inf = P.info + (size_t)e * N_INFO;
    inf[0] = E.info.x_pos; inf[1] = E.info.flag_get; inf[2] = E.info.life; inf[3] = E.info.score;
    inf[4] = E.info.time; inf[5] = E.info.world; inf[6] = E.info.stage; inf[7] = E.info.coins;
    inf[8] = E.info.y_pos; inf[9] = E.info.status;
    P.A.frames[e] += draws;
    if (te || tr) {
        P.ep_ret_out[e] = E.ep_ret; P.ep_len_out[e] = E.ep_len;
        int nk = P.cfg.noop_max + 1;
        int k = 0;
        if (P.cfg.noop_max) k = (int)rng.integers(P.cfg.noop_max + 1);
        int c = L * nk + k;
        if (sticky != 0.0) { int d = P.cdraws[c]; for (int q = 0; q < d; q++) rng.next64(); }
        E.x_last = P.cx[c]; E.time_last = P.ct[c]; E.lives = P.clives[c];
        E.prev = 0; E.t = 0; E.ep_ret = 0.0; E.ep_len = 0;
        P.reset_idx[e] = c;
    } else {
        P.reset_idx[e] = -1;
    }
    store_sc(n.s, n.m);
    store_env(E, rng, P.A, e);
    if (n.bad_ops) atomicAdd(P.bad_ops, n.bad_ops);
}

// ---------------------------------------------------------------------------------------------
// k_render: draw the logged render frame of every env (one 256-thread block per env, one thread per
// dot, rows top to bottom) and INTER_AREA-resize it to 84x84 exactly like Core::finish_row.
// Every dot's segment is the last one (in emulation time) that covered it: the latest pass over its
// line (lp) or, for dots that pass did not reach, the earlier pass (pp).
// ---------------------------------------------------------------------------------------------
struct RView {
    const uint8_t* chr;
    const uint8_t* vram; const uint8_t* pal; const uint8_t* oam;
    int nt0, nt1, nt2, nt3;
    FI int ntoff(int t) const { return t == 0 ? nt0 : t == 1 ? nt1 : t == 2 ? nt2 : nt3; }
    FI uint8_t pb_read(uint16_t a) const {
        if (a < 0x2000) return chr[a];
        if (a < 0x3eff) {
            int t = a < 0x2400 ? 0 : a < 0x2800 ? 1 : a < 0x2c00 ? 2 : 3;
            return vram[ntoff(t) + (a & 0x3ff)];
        }
        if (a < 0x3fff) return pal[a & 0x1f];
        return 0;
    }
    FI uint8_t nt_read(uint16_t a) const { return vram[ntoff((a >> 10) & 3) + (a & 0x3ff)]; }
    FI uint8_t spr_bits(int i, int x, int y, bool long_spr, int spr_page) const {
        uint8_t spr_x = oam[i * 4 + 3];
        uint8_t spr_y = (uint8_t)(oam[i * 4 + 0] + 1), tile = oam[i * 4 + 1], attribute = oam[i * 4 + 2];
        int length = long_spr ? 16 : 8;
        int x_shift = (x - spr_x) % 8, y_offset = (y - spr_y) % length;
        if ((attribute & 0x40) == 0) x_shift ^= 7;
        if ((attribute & 0x80) != 0) y_offset ^= (length - 1);
        uint16_t address;
        if (!long_spr) {
            address = (uint16_t)(tile * 16 + y_offset);
            if (spr_page) address += 0x1000;
        } else {
            y_offset = (y_offset & 7) | ((y_offset & 8) << 1);
            address = (uint16_t)((tile >> 1) * 32 + y_offset);
            address |= (tile & 1) << 12;
        }
        uint8_t cc = (pb_read(address) >> x_shift) & 1;
        cc |= ((pb_read((uint16_t)(address + 8)) >> x_shift) & 1) << 1;
        return cc;
    }
};

extern "C" __global__ void __launch_bounds__(256) k_render(StepArgs P, const uint32_t* __restrict__ aux) {
    __shared__ uint32_t s_chr[0x2000 / 4];
    __shared__ uint32_t s_vram[2][512];
    __shared__ uint32_t s_pal[2][8];
    __shared__ uint32_t s_oam[2][64];
    __shared__ uint32_t s_rec[4 * NSEG];
    __shared__ uint32_t s_lp[240], s_pp[240];
    __shared__ uint8_t s_sl[NSEG][9];
    __shared__ uint8_t s_row[256];
    const int tid = threadIdx.x;
    for (int i = tid; i < 0x2000 / 4; i += 256) s_chr[i] = ((const uint32_t*)P.cart)[0x8000 / 4 + i];
    for (int e = blockIdx.x; e < P.n; e += gridDim.x) {  // persistent blocks: CHR loaded once
    const uint32_t* A = aux + (size_t)e * AUXW;
    const uint32_t hdr = A[AX_HDR];
    if (!((hdr >> 16) & 1)) continue;  // produced inline by k_step2 (uniform per block)
    __syncthreads();  // previous env's smem readers are done
    const uint32_t tag = hdr & 0xffff;
    const int nseg = (int)A[AX_NSEG];
    const uint32_t ver = A[AX_VER];
    const bool vcopy = (ver & 0xffff) != 0, ocopy = (ver >> 16) != 0;
    for (int i = tid; i < 512; i += 256) {
        uint32_t w = P.state[(size_t)((OFF_VRAM >> 2) + i) * P.np + e];
        s_vram[1][i] = w; s_vram[0][i] = vcopy ? A[AX_V0 + i] : w;
    }
    if (tid < 8) {
        uint32_t w = P.state[(size_t)((OFF_PAL >> 2) + tid) * P.np + e];
        s_pal[1][tid] = w; s_pal[0][tid] = vcopy ? A[AX_V0PAL + tid] : w;
    }
    if (tid < 64) {
        uint32_t w = P.state[(size_t)((OFF_OAM >> 2) + tid) * P.np + e];
        s_oam[1][tid] = w; s_oam[0][tid] = ocopy ? A[AX_V0OAM + tid] : w;
    }
    for (int i = tid; i < 4 * nseg; i += 256) s_rec[i] = A[AX_REC + i];
    for (int i = tid; i < 240; i += 256) { s_lp[i] = A[AX_LP + i]; s_pp[i] = A[AX_PP + i]; }
    __syncthreads();
    const int nt0 = P.nt[0], nt1 = P.nt[1], nt2 = P.nt[2], nt3 = P.nt[3];
    // sprite list of every segment (the evaluation it was rendered with)
    for (int r = tid; r < nseg; r += 256) {
        const uint32_t ed = s_rec[4 * r + 2];
        if (ed & 1) {
            for (int q = 0; q < 8; q++) s_sl[r][q] = (uint8_t)(A[AX_ISL + (q >> 2)] >> (8 * (q & 3)));
            s_sl[r][8] = (uint8_t)A[AX_ISL + 2];
        } else {
            const int line = (ed >> 1) & 0xff, oam_addr = (ed >> 9) & 0xff, range = ((ed >> 17) & 1) ? 16 : 8;
            const uint8_t* oam = (const uint8_t*)s_oam[((ed >> 18) == 0 && ocopy) ? 0 : 1];
            int nsl = 0;
            for (int i = oam_addr / 4; i < 64; ++i) {
                int diff = line - oam[i * 4];
                if (0 <= diff && diff < range) {
                    s_sl[r][nsl++] = (uint8_t)i;
                    if (nsl >= 8) break;
                }
            }
            s_sl[r][8] = (uint8_t)nsl;
        }
    }
    __syncthreads();
    const Rs rs{P.xs, P.xw, P.ye, P.yw, P.lut, P.gr};
    uint8_t* obs = P.newobs + (size_t)e * 84 * 84;
    float sum = 0.f;
    const int x = tid;
    for (int y = 0; y < 240; y++) {
        // ---- the segment that last drew dot (y, x)
        int rsel = -1;
        {
            uint32_t L = s_lp[y];
            if ((L >> 16) == tag) {
                int st = L & 511, cn = (L >> 9) & 127;
                for (int k = st; k < st + cn; k++) {
                    uint32_t w0 = s_rec[4 * k];
                    if ((int)((w0 >> 8) & 511) <= x && x < (int)(w0 >> 17)) rsel = k;
                }
                if (rsel < 0) {
                    uint32_t Q = s_pp[y];
                    if ((Q >> 16) == tag) {
                        st = Q & 511; cn = (Q >> 9) & 127;
                        for (int k = st; k < st + cn; k++) {
                            uint32_t w0 = s_rec[4 * k];
                            if ((int)((w0 >> 8) & 511) <= x && x < (int)(w0 >> 17)) rsel = k;
                        }
                    }
                }
            }
        }
        uint8_t gray = 0;
        if (rsel < 0) atomicAdd(P.bad_ops, 1 << 20);  // must not happen: every dot is drawn in a frame
        else {
            const uint32_t w0 = s_rec[4 * rsel], w1 = s_rec[4 * rsel + 1], w3 = s_rec[4 * rsel + 3];
            const int x0 = (w0 >> 8) & 511;
            const int fine_x = (w1 >> 16) & 7;
            const int bg_page = (w1 >> 19) & 1, spr_page = (w1 >> 20) & 1;
            const bool long_spr = (w1 >> 21) & 1, show_bg = (w1 >> 22) & 1, show_spr = (w1 >> 23) & 1;
            const bool heb = (w1 >> 24) & 1, hes = (w1 >> 25) & 1;
            const int vsel = ((w3 & 0xffff) == 0 && vcopy) ? 0 : 1;
            const int osel = ((w3 >> 16) == 0 && ocopy) ? 0 : 1;
            RView V{(const uint8_t*)s_chr, (const uint8_t*)s_vram[vsel], (const uint8_t*)s_pal[vsel],
                    (const uint8_t*)s_oam[osel], nt0, nt1, nt2, nt3};
            uint8_t px = 0;
            if (show_bg) {
                uint16_t v = (uint16_t)(w1 & 0xffff);
                const int rr = (7 - fine_x) & 7;
                const int ninc = (x - rr + 7) / 8 - (x0 - rr + 7) / 8;
                if (ninc > 0) {
                    int tot = (v & 0x1f) + ninc;
                    v = (uint16_t)((v & ~0x1f) | (tot & 31));
                    if ((tot >> 5) & 1) v ^= 0x0400;
                }
                const int xf = (fine_x + x) & 7;
                uint8_t tile = V.nt_read((uint16_t)(0x2000 | (v & 0x0FFF)));
                uint16_t addr = (uint16_t)(((tile * 16) + ((v >> 12) & 0x7)) | (bg_page << 12));
                uint8_t lo = V.chr[addr], hi = V.chr[addr + 8];
                uint16_t aaddr = (uint16_t)(0x23C0 | (v & 0x0C00) | ((v >> 4) & 0x38) | ((v >> 2) & 0x07));
                int shift = ((v >> 4) & 4) | (v & 2);
                uint8_t at = (uint8_t)(((V.nt_read(aaddr) >> shift) & 0x3) << 2);
                int sh = 7 - xf;
                uint8_t b = (uint8_t)(((lo >> sh) & 1) | (((hi >> sh) & 1) << 1));
                px = (heb && x < 8) ? 0 : (b ? (uint8_t)(b | at) : 0);
            }
            const int nsl = s_sl[rsel][8];
            if (show_spr && nsl && !(hes && x < 8)) {
                uint8_t col = 0, fl = 0;
                for (int k = 0; k < nsl; k++) {
                    int i = s_sl[rsel][k];
                    int sx = V.oam[i * 4 + 3];
                    if (x < sx || x >= sx + 8) continue;
                    uint8_t cc = V.spr_bits(i, x, y, long_spr, spr_page);
                    if (cc) {
                        uint8_t attribute = V.oam[i * 4 + 2];
                        col = (uint8_t)(cc | 0x10 | ((attribute & 0x3) << 2));
                        fl = (uint8_t)(!(attribute & 0x20) ? 1 : 0);
                        break;
                    }
                }
                if (col) {
                    uint8_t bgOpaque = px & 3;
                    if (!bgOpaque || (fl & 1)) px = col;
                }
            }
            gray = __ldg(rs.lut + V.pal[px]);
        }
        s_row[x] = gray;
        __syncthreads();
        if (tid < RD) {
            const int dx = tid;
            const int e0 = __ldg(rs.ye + 2 * y), e1 = __ldg(rs.ye + 2 * y + 1);
            const float wy0 = __ldg(rs.yw + 2 * y), wy1 = __ldg(rs.yw + 2 * y + 1);
            if (e0 >= 0 || e1 >= 0) {
                float b = 0.f;
                for (int k = 0; k < RM; k++) {
                    float w = __ldg(rs.xw + k * RD + dx);
                    if (w == 0.f) break;
                    b += (float)s_row[__ldg(rs.xs + k * RD + dx)] * w;
                }
                if (e0 >= 0) {
                    float v = wy0 * b;
                    float sm = (e0 & 0x100) ? v : sum + v;
                    if (e0 & 0x200) { int iv = __float2int_rn(sm); obs[(e0 & 0xff) * RD + dx] = (uint8_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv); }
                    sum = sm;
                }
                if (e1 >= 0) {
                    float v = wy1 * b;
                    float sm = (e1 & 0x100) ? v : sum + v;
                    if (e1 & 0x200) { int iv = __float2int_rn(sm); obs[(e1 & 0xff) * RD + dx] = (uint8_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv); }
                    sum = sm;
                }
            }
        }
        __syncthreads();
    }
    }
}

// ---------------------------------------------------------------------------------------------
// k_render2: same output as k_render, but one thread per 8-dot span (warp = row, 8 rows per phase):
// tile fetches and sprite pattern fetches are done once per span instead of once per dot, and the
// resize runs once per 8 rows (2 block barriers per 8 rows instead of per row).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) k_render2(StepArgs P, const uint32_t* __restrict__ aux) {
    __shared__ uint32_t s_chr[0x2000 / 4];
    __shared__ uint32_t s_vram[2][512];
    __shared__ uint32_t s_pal[2][8];
    __shared__ uint32_t s_oam[2][64];
    __shared__ uint32_t s_rec[4 * NSEG];
    __shared__ uint32_t s_lp[240], s_pp[240];
    __shared__ uint8_t s_sl[NSEG][9];
    __shared__ uint32_t s_rows[8][64];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < 0x2000 / 4; i += 256) s_chr[i] = ((const uint32_t*)P.cart)[0x8000 / 4 + i];
    const int nt0 = P.nt[0], nt1 = P.nt[1], nt2 = P.nt[2], nt3 = P.nt[3];
    const Rs rs{P.xs, P.xw, P.ye, P.yw, P.lut, P.gr};
    for (int e = blockIdx.x; e < P.n; e += gridDim.x) {
        const uint32_t* A = aux + (size_t)e * AUXW;
        const uint32_t hdr = A[AX_HDR];
        if (!((hdr >> 16) & 1)) continue;  // produced inline by k_step2 (uniform per block)
        __syncthreads();
        const uint32_t tag = hdr & 0xffff;
        const int nseg = (int)A[AX_NSEG];
        const uint32_t ver = A[AX_VER];
        const bool vcopy = (ver & 0xffff) != 0, ocopy = (ver >> 16) != 0;
        for (int i = tid; i < 512; i += 256) {
            uint32_t w = P.state[(size_t)((OFF_VRAM >> 2) + i) * P.np + e];
            s_vram[1][i] = w; s_vram[0][i] = vcopy ? A[AX_V0 + i] : w;
        }
        if (tid < 8) {
            uint32_t w = P.state[(size_t)((OFF_PAL >> 2) + tid) * P.np + e];
            s_pal[1][tid] = w; s_pal[0][tid] = vcopy ? A[AX_V0PAL + tid] : w;
        }
        if (tid < 64) {
            uint32_t w = P.state[(size_t)((OFF_OAM >> 2) + tid) * P.np + e];
            s_oam[1][tid] = w; s_oam[0][tid] = ocopy ? A[AX_V0OAM + tid] : w;
        }
        for (int i = tid; i < 4 * nseg; i += 256) s_rec[i] = A[AX_REC + i];
        for (int i = tid; i < 240; i += 256) { s_lp[i] = A[AX_LP + i]; s_pp[i] = A[AX_PP + i]; }
        __syncthreads();
        for (int r = tid; r < nseg; r += 256) {
            const uint32_t ed = s_rec[4 * r + 2];
            if (ed & 1) {
                for (int q = 0; q < 8; q++) s_sl[r][q] = (uint8_t)(A[AX_ISL + (q >> 2)] >> (8 * (q & 3)));
                s_sl[r][8] = (uint8_t)A[AX_ISL + 2];
            } else {
                const int line = (ed >> 1) & 0xff, oam_addr = (ed >> 9) & 0xff, range = ((ed >> 17) & 1) ? 16 : 8;
                const uint8_t* oam = (const uint8_t*)s_oam[((ed >> 18) == 0 && ocopy) ? 0 : 1];
                int nsl = 0;
                for (int i = oam_addr / 4; i < 64; ++i) {
                    int diff = line - oam[i * 4];
                    if (0 <= diff && diff < range) {
                        s_sl[r][nsl++] = (uint8_t)i;
                        if (nsl >= 8) break;
                    }
                }
                s_sl[r][8] = (uint8_t)nsl;
            }
        }
        __syncthreads();
        uint8_t* obs = P.newobs + (size_t)e * 84 * 84;
        float sum = 0.f;
        for (int y0 = 0; y0 < 240; y0 += 8) {
            {   // ---- dots phase: row y, dots xa .. xa+7
                const int y = y0 + warp, xa = lane * 8;
                const uint32_t L = s_lp[y], Q = s_pp[y];
                const bool lok = (L >> 16) == tag, qok = (Q >> 16) == tag;
                int cur = -2;
                // per-segment state
                RView V{(const uint8_t*)s_chr, nullptr, nullptr, nullptr, nt0, nt1, nt2, nt3};
                uint32_t w1 = 0; int x0 = 0, nsl = 0;
                uint32_t sp[8];
                int tkey = -1; uint8_t tlo = 0, thi = 0, tat = 0;
                unsigned long long outw = 0;
#pragma unroll 1
                for (int j = 0; j < 8; j++) {
                    const int x = xa + j;
                    int rsel = -1;
                    if (lok) {
                        const int st = L & 511, cn = (L >> 9) & 127;
                        for (int k = st; k < st + cn; k++) {
                            uint32_t w0 = s_rec[4 * k];
                            if ((int)((w0 >> 8) & 511) <= x && x < (int)(w0 >> 17)) rsel = k;
                        }
                        if (rsel < 0 && qok) {
                            const int st2 = Q & 511, cn2 = (Q >> 9) & 127;
                            for (int k = st2; k < st2 + cn2; k++) {
                                uint32_t w0 = s_rec[4 * k];
                                if ((int)((w0 >> 8) & 511) <= x && x < (int)(w0 >> 17)) rsel = k;
                            }
                        }
                    }
                    uint8_t gray = 0;
                    if (rsel < 0) { atomicAdd(P.bad_ops, 1 << 20); }
                    else {
                        if (rsel != cur) {  // segment state for this span
                            cur = rsel; tkey = -1;
                            const uint32_t w0 = s_rec[4 * rsel], w3 = s_rec[4 * rsel + 3];
                            w1 = s_rec[4 * rsel + 1];
                            x0 = (w0 >> 8) & 511;
                            const int vsel = ((w3 & 0xffff) == 0 && vcopy) ? 0 : 1;
                            const int osel = ((w3 >> 16) == 0 && ocopy) ? 0 : 1;
                            V.vram = (const uint8_t*)s_vram[vsel]; V.pal = (const uint8_t*)s_pal[vsel];
                            V.oam = (const uint8_t*)s_oam[osel];
                            nsl = s_sl[rsel][8];
                            const bool long_spr = (w1 >> 21) & 1;
                            const int spr_page = (w1 >> 20) & 1;
#pragma unroll
                            for (int k = 0; k < 8; k++) {
                                sp[k] = 0;
                                if (k < nsl) {
                                    const int i = s_sl[rsel][k];
                                    const int sx = V.oam[i * 4 + 3];
                                    // pattern row of sprite i on line y (independent of the dot)
                                    uint8_t spr_y = (uint8_t)(V.oam[i * 4 + 0] + 1), tile = V.oam[i * 4 + 1], attribute = V.oam[i * 4 + 2];
                                    int length = long_spr ? 16 : 8;
                                    int y_offset = (y - spr_y) % length;
                                    if ((attribute & 0x80) != 0) y_offset ^= (length - 1);
                                    uint16_t address;
                                    if (!long_spr) {
                                        address = (uint16_t)(tile * 16 + y_offset);
                                        if (spr_page) address += 0x1000;
                                    } else {
                                        y_offset = (y_offset & 7) | ((y_offset & 8) << 1);
                                        address = (uint16_t)((tile >> 1) * 32 + y_offset);
                                        address |= (tile & 1) << 12;
                                    }
                                    const uint8_t lo = V.pb_read(address), hi = V.pb_read((uint16_t)(address + 8));
                                    sp[k] = (uint32_t)lo | ((uint32_t)hi << 8) | ((uint32_t)attribute << 16) | ((uint32_t)sx << 24);
                                }
                            }
                        }
                        const int fine_x = (w1 >> 16) & 7;
                        const bool show_bg = (w1 >> 22) & 1, show_spr = (w1 >> 23) & 1;
                        const bool heb = (w1 >> 24) & 1, hes = (w1 >> 25) & 1;
                        uint8_t px = 0;
                        if (show_bg) {
                            uint16_t v = (uint16_t)(w1 & 0xffff);
                            const int rr = (7 - fine_x) & 7;
                            const int ninc = (x - rr + 7) / 8 - (x0 - rr + 7) / 8;
                            if (ninc > 0) {
                                int tot = (v & 0x1f) + ninc;
                                v = (uint16_t)((v & ~0x1f) | (tot & 31));
                                if ((tot >> 5) & 1) v ^= 0x0400;
                            }
                            if ((int)v != tkey) {
                                tkey = v;
                                const int bg_page = (w1 >> 19) & 1;
                                uint8_t tile = V.nt_read((uint16_t)(0x2000 | (v & 0x0FFF)));
                                uint16_t addr = (uint16_t)(((tile * 16) + ((v >> 12) & 0x7)) | (bg_page << 12));
                                tlo = V.chr[addr]; thi = V.chr[addr + 8];
                                uint16_t aaddr = (uint16_t)(0x23C0 | (v & 0x0C00) | ((v >> 4) & 0x38) | ((v >> 2) & 0x07));
                                int shift = ((v >> 4) & 4) | (v & 2);
                                tat = (uint8_t)(((V.nt_read(aaddr) >> shift) & 0x3) << 2);
                            }
                            const int sh = 7 - ((fine_x + x) & 7);
                            uint8_t b = (uint8_t)(((tlo >> sh) & 1) | (((thi >> sh) & 1) << 1));
                            px = (heb && x < 8) ? 0 : (b ? (uint8_t)(b | tat) : 0);
                        }
                        if (show_spr && nsl && !(hes && x < 8)) {
                            uint8_t col = 0, fl = 0;
                            bool found = false;
#pragma unroll
                            for (int k = 0; k < 8; k++) {
                                const uint32_t q = sp[k];
                                const int sx = (int)(q >> 24);
                                if (!found && k < nsl && x >= sx && x < sx + 8) {
                                    const uint8_t attribute = (uint8_t)(q >> 16);
                                    int x_shift = (x - sx) % 8;
                                    if ((attribute & 0x40) == 0) x_shift ^= 7;
                                    uint8_t cc = (uint8_t)((((q & 0xff) >> x_shift) & 1) | ((((q >> 8) & 0xff) >> x_shift) & 1) << 1);
                                    if (cc) {
                                        col = (uint8_t)(cc | 0x10 | ((attribute & 0x3) << 2));
                                        fl = (uint8_t)(!(attribute & 0x20) ? 1 : 0);
                                        found = true;
                                    }
                                }
                            }
                            if (col) {
                                uint8_t bgOpaque = px & 3;
                                if (!bgOpaque || (fl & 1)) px = col;
                            }
                        }
                        gray = __ldg(rs.lut + V.pal[px]);
                    }
                    outw |= (unsigned long long)gray << (8 * j);
                }
                s_rows[warp][lane * 2] = (uint32_t)outw; s_rows[warp][lane * 2 + 1] = (uint32_t)(outw >> 32);
            }
            __syncthreads();
            if (tid < RD) {
                const int dx = tid;
                for (int r = 0; r < 8; r++) {
                    const int y = y0 + r;
                    const uint8_t* row = (const uint8_t*)s_rows[r];
                    const int e0 = __ldg(rs.ye + 2 * y), e1 = __ldg(rs.ye + 2 * y + 1);
                    const float wy0 = __ldg(rs.yw + 2 * y), wy1 = __ldg(rs.yw + 2 * y + 1);
                    if (e0 >= 0 || e1 >= 0) {
                        float b = 0.f;
                        for (int k = 0; k < RM; k++) {
                            float w = __ldg(rs.xw + k * RD + dx);
                            if (w == 0.f) break;
                            b += (float)row[__ldg(rs.xs + k * RD + dx)] * w;
                        }
                        if (e0 >= 0) {
                            float v = wy0 * b;
                            float sm = (e0 & 0x100) ? v : sum + v;
                            if (e0 & 0x200) { int iv = __float2int_rn(sm); obs[(e0 & 0xff) * RD + dx] = (uint8_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv); }
                            sum = sm;
                        }
                        if (e1 >= 0) {
                            float v = wy1 * b;
                            float sm = (e1 & 0x100) ? v : sum + v;
                            if (e1 & 0x200) { int iv = __float2int_rn(sm); obs[(e1 & 0xff) * RD + dx] = (uint8_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv); }
                            sum = sm;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

// deferred reset: after k_render consumed the terminal frame's PPU memory
extern "C" __global__ void k_reset(StepArgs P) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= P.n) return;
    int c = P.reset_idx[e];
    if (c < 0) return;
    for (int w = 0; w < NW; w++) P.state[(size_t)w * P.np + e] = P.cstate[(size_t)w * P.nc + c];
}

// SPDX-License-Identifier: GPL-3.0-or-later
// fast-mario-env: batched Super Mario Bros environment in C++ (threads, no processes, GIL-free via ctypes).
//
// Semantics reproduce the reference nes-py wrapper stack (tests/reference_env.py) exactly:
//   nes-py SuperMarioBrosEnv(target=(w,s)) + JoypadSpace(SIMPLE_MOVEMENT)
//   -> [Sticky(p), frame level, numpy PCG64 stream] -> SkipMax(k) -> LifeLoss -> [NoopStart]
//   -> gray (cv2 RGB2GRAY LUT) -> INTER_AREA 84x84 (exact port of OpenCV's float area resize)
//   -> TimeLimit(max_steps) ; frame stack + auto-reset as in tests/reference_env.RefVecEnv.
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#include "nes_core.hpp"

namespace fm {

// nes-py master palette (xRGB)
static const uint32_t PALETTE[64] = {
    0x7C7C7C, 0x0000FC, 0x0000BC, 0x4428BC, 0x940084, 0xA80020, 0xA81000, 0x881400,
    0x503000, 0x007800, 0x006800, 0x005800, 0x004058, 0x000000, 0x000000, 0x000000,
    0xBCBCBC, 0x0078F8, 0x0058F8, 0x6844FC, 0xD800CC, 0xE40058, 0xF83800, 0xE45C10,
    0xAC7C00, 0x00B800, 0x00A800, 0x00A844, 0x008888, 0x000000, 0x000000, 0x000000,
    0xF8F8F8, 0x3CBCFC, 0x6888FC, 0x9878F8, 0xF878F8, 0xF85898, 0xF87858, 0xFCA044,
    0xF8B800, 0xB8F818, 0x58D854, 0x58F898, 0x00E8D8, 0x787878, 0x000000, 0x000000,
    0xFCFCFC, 0xA4E4FC, 0xB8B8F8, 0xD8B8F8, 0xF8B8F8, 0xF8A4C0, 0xF0D0B0, 0xFCE0A8,
    0xF8D878, 0xD8F878, 0xB8F8B8, 0xB8F8D8, 0x00FCFC, 0xF8D8F8, 0x000000, 0x000000};

static const uint8_t SIMPLE_MOVEMENT[7] = {0x00, 0x80, 0x81, 0x82, 0x83, 0x01, 0x40};

// ------------------------------------------------------------------ numpy PCG64 (Generator)
struct Pcg64 {
    unsigned __int128 state = 0, inc = 0;
    int has32 = 0;
    uint32_t uinteger = 0;
    static unsigned __int128 mult() {
        return ((unsigned __int128)0x2360ED051FC65DA4ULL << 64) | 0x4385DF649FCCF645ULL;
    }
    uint64_t next64() {
        state = state * mult() + inc;
        uint64_t hi = (uint64_t)(state >> 64), lo = (uint64_t)state;
        unsigned rot = (unsigned)(state >> 122);
        uint64_t x = hi ^ lo;
        return (x >> rot) | (x << ((64 - rot) & 63));
    }
    uint32_t next32() {
        if (has32) { has32 = 0; return uinteger; }
        uint64_t n = next64();
        has32 = 1;
        uinteger = (uint32_t)(n >> 32);
        return (uint32_t)(n & 0xffffffff);
    }
    double random() { return (double)(next64() >> 11) * (1.0 / 9007199254740992.0); }
    // Generator.integers(0, high) (int64, endpoint=False), high <= 2^32
    int64_t integers(int64_t high) {
        uint64_t rng = (uint64_t)(high - 1);
        if (rng == 0) return 0;
        if (rng == 0xFFFFFFFFULL) return next32();
        uint32_t rng_excl = (uint32_t)rng + 1;
        uint64_t m = (uint64_t)next32() * rng_excl;
        uint32_t leftover = (uint32_t)(m & 0xffffffff);
        if (leftover < rng_excl) {
            uint32_t threshold = (uint32_t)((0xFFFFFFFFu - (uint32_t)rng) % rng_excl);
            while (leftover < threshold) {
                m = (uint64_t)next32() * rng_excl;
                leftover = (uint32_t)(m & 0xffffffff);
            }
        }
        return (int64_t)(m >> 32);
    }
};

// ------------------------------------------------------------------ OpenCV INTER_AREA (float path)
struct AreaTab { int di, si; float alpha; };

static int compute_area_tab(int ssize, int dsize, double scale, AreaTab* tab) {
    int k = 0;
    for (int dx = 0; dx < dsize; dx++) {
        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, ssize - fsx1);
        int sx1 = (int)std::ceil(fsx1), sx2 = (int)std::floor(fsx2);
        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);
        if (sx1 - fsx1 > 1e-3) { tab[k].di = dx; tab[k].si = sx1 - 1; tab[k++].alpha = (float)((sx1 - fsx1) / cellWidth); }
        for (int sx = sx1; sx < sx2; sx++) { tab[k].di = dx; tab[k].si = sx; tab[k++].alpha = float(1.0 / cellWidth); }
        if (fsx2 - sx2 > 1e-3) {
            tab[k].di = dx; tab[k].si = sx2;
            tab[k++].alpha = (float)(std::min(std::min(fsx2 - sx2, 1.), cellWidth) / cellWidth);
        }
    }
    return k;
}

static inline uint8_t sat_round(float v) {
    int iv = (int)lrintf(v);
    return (uint8_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv);
}

// Reference implementation (row-major src), identical operation order to cv::resizeArea_<uchar,float>.
static void area_resize_ref(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    std::vector<AreaTab> xtab(sw * 2), ytab(sh * 2);
    int xn = compute_area_tab(sw, dw, 1. / ((double)dw / sw), xtab.data());
    int yn = compute_area_tab(sh, dh, 1. / ((double)dh / sh), ytab.data());
    std::vector<float> buf(dw), sum(dw, 0.f);
    int prev_dy = ytab[0].di;
    for (int j = 0; j < yn; j++) {
        float beta = ytab[j].alpha;
        int dy = ytab[j].di, sy = ytab[j].si;
        const uint8_t* S = src + sy * sw;
        for (int dx = 0; dx < dw; dx++) buf[dx] = 0.f;
        for (int k = 0; k < xn; k++) buf[xtab[k].di] += S[xtab[k].si] * xtab[k].alpha;
        if (dy != prev_dy) {
            uint8_t* D = dst + prev_dy * dw;
            for (int dx = 0; dx < dw; dx++) { D[dx] = sat_round(sum[dx]); sum[dx] = beta * buf[dx]; }
            prev_dy = dy;
        } else {
            for (int dx = 0; dx < dw; dx++) sum[dx] += beta * buf[dx];
        }
    }
    uint8_t* D = dst + prev_dy * dw;
    for (int dx = 0; dx < dw; dx++) D[dx] = sat_round(sum[dx]);
}

// 16x16 byte block transpose (4 rounds of row i / row i+8 byte interleave)
static inline void tr16(const uint8_t* src, int ss, uint8_t* dst, int ds) {
#if !defined(__SSE2__)
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) dst[j * ds + i] = src[i * ss + j];
#else
    __m128i a[16], b[16];
    for (int i = 0; i < 16; i++) a[i] = _mm_loadu_si128((const __m128i*)(src + i * ss));
    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < 8; i++) {
            b[2 * i] = _mm_unpacklo_epi8(a[i], a[i + 8]);
            b[2 * i + 1] = _mm_unpackhi_epi8(a[i], a[i + 8]);
        }
        for (int i = 0; i < 16; i++) a[i] = b[i];
    }
    for (int i = 0; i < 16; i++) _mm_storeu_si128((__m128i*)(dst + i * ds), a[i]);
#endif
}
// row-major 240x256 -> column-major (x*240 + y)
static void transpose_gray(const uint8_t* gR, uint8_t* gT) {
    for (int by = 0; by < 240; by += 16)
        for (int bx = 0; bx < 256; bx += 16) tr16(gR + by * 256 + bx, 256, gT + bx * 240 + by, 240);
}

// Fast 256x240 -> 84x84 resize from a column-major (x*240+y) gray image, same float results.
struct AreaResizer {
    static const int SW = 256, SH = 240, D = 84, M = 6;
    int xs[M][D]; float xw[M][D];   // per output column: up to M (source col, weight), zero-padded at the end
    int ys[M][D]; float yw[M][D];
    void init() {
        AreaTab xt[SW * 2], yt[SH * 2];
        int xn = compute_area_tab(SW, D, 1. / ((double)D / SW), xt);
        int yn = compute_area_tab(SH, D, 1. / ((double)D / SH), yt);
        fill(xt, xn, xs, xw);
        fill(yt, yn, ys, yw);
    }
    static void fill(const AreaTab* t, int n, int (*idx)[D], float (*w)[D]) {
        int cnt[D] = {0};
        for (int m = 0; m < M; m++) for (int d = 0; d < D; d++) { idx[m][d] = 0; w[m][d] = 0.f; }
        for (int k = 0; k < n; k++) {
            int d = t[k].di;
            if (cnt[d] >= M) { fprintf(stderr, "AreaResizer: M too small\n"); abort(); }
            idx[cnt[d]][d] = t[k].si; w[cnt[d]][d] = t[k].alpha; cnt[d]++;
        }
    }
    // gT: 256*240 column-major gray; bufT scratch D*SH floats
    void run(const uint8_t* __restrict gT, uint8_t* __restrict out, float* __restrict bufT) const {
        for (int dx = 0; dx < D; dx++) {
            float* __restrict b = bufT + dx * SH;
            for (int r = 0; r < SH; r++) b[r] = 0.f;
            for (int m = 0; m < M; m++) {
                const float w = xw[m][dx];
                if (w == 0.f) break;  // padding only at the end (adding S*0 would be a no-op anyway)
                const uint8_t* __restrict col = gT + xs[m][dx] * SH;
                for (int r = 0; r < SH; r++) b[r] += (float)col[r] * w;
            }
        }
        for (int dx = 0; dx < D; dx++) {
            const float* b = bufT + dx * SH;
            for (int dy = 0; dy < D; dy++) {
                float sacc = 0.f;
                for (int m = 0; m < M; m++) {
                    const float w = yw[m][dy];
                    if (w == 0.f) break;
                    sacc += w * b[ys[m][dy]];
                }
                out[dy * D + dx] = sat_round(sacc);
            }
        }
    }
};

// ------------------------------------------------------------------ SMB RAM helpers
static inline int64_t digits(const uint8_t* r, int n) {  // int(''.join(map(str, ram[a:a+n])))
    int64_t v = 0;
    for (int i = 0; i < n; i++) {
        int b = r[i];
        v = v * (b >= 100 ? 1000 : b >= 10 ? 100 : 10) + b;
    }
    return v;
}
struct Info {
    int64_t x_pos, flag_get, life, score, time, world, stage, coins, y_pos, status;
};
enum { N_INFO = 10 };

struct SmbRam {
    const uint8_t* r;
    int64_t x_pos() const { return (int64_t)r[0x6d] * 0x100 + r[0x86]; }
    int64_t time() const { return digits(r + 0x7f8, 3); }
    int player_state() const { return r[0x000e]; }
    int y_viewport() const { return r[0x00b5]; }
    bool is_dying() const { return player_state() == 0x0b || y_viewport() > 1; }
    bool is_dead() const { return player_state() == 0x06; }
    bool is_busy() const { int s = player_state(); return s <= 5 || s == 7; }
    bool is_world_over() const { return r[0x0770] == 2; }
    bool is_stage_over() const {
        static const int addrs[5] = {0x16, 0x17, 0x18, 0x19, 0x1A};
        for (int a : addrs) if (r[a] == 0x2D || r[a] == 0x31) return r[0x001D] == 3;
        return false;
    }
    bool flag_get() const { return is_world_over() || is_stage_over(); }
    Info info() const {
        Info i;
        i.coins = digits(r + 0x7ed, 2);
        i.flag_get = flag_get();
        i.life = r[0x075a];
        i.score = digits(r + 0x7de, 6);
        i.stage = r[0x075c] + 1;
        int st = r[0x0756];
        i.status = st == 0 ? 0 : st == 1 ? 1 : 2;  // small / tall / fireball
        i.time = time();
        i.world = r[0x075f] + 1;
        i.x_pos = x_pos();
        int yp = r[0x03b8];
        i.y_pos = y_viewport() < 1 ? 255 + (255 - yp) : 255 - yp;
        return i;
    }
};

struct Config {
    int skip = 4, max_steps = 2000, noop_max = 0, maxpool = 0, frame_stack = 4;
    double sticky = 0.0;
};

static const int GSZ = 256 * 240, OSZ = 84 * 84;

// Per-level boot snapshot (shared by every env on that level).
struct Level {
    int world, stage, area;
    NesState backup;
    int64_t x_last0, time_last0;
    uint8_t obs0[OSZ];
    std::vector<uint8_t> screen0;  // palette values at backup (debug)
};

struct Env {
    Nes nes;
    const Level* lvl = nullptr;
    const Config* cfg = nullptr;
    const AreaResizer* rs = nullptr;
    int64_t x_last = 0, time_last = 0;
    Pcg64 rng;
    int prev = 0, lives = -1, t = 0, ep_len = 0;
    double ep_ret = 0.0;
    bool obs_is_backup = false;  // live screen == boot/backup screen (right after a reset)
    std::vector<uint8_t> gR, gR_prev, gT;  // row-major gray (current / previous frame-step), transposed scratch
    std::vector<float> bufT;
    NesState snaps[16];
    uint8_t acts[16];
    Info info{};
    uint64_t frames = 0;

    void init(const Level* L, const Config* c, const AreaResizer* r, const uint8_t* lut, const Cart* cart) {
        lvl = L; cfg = c; rs = r;
        gR.assign(GSZ, 0); gR_prev.assign(GSZ, 0); gT.assign(GSZ, 0); bufT.assign(84 * 240, 0.f);
        nes.cart = cart; nes.gray_lut = lut; nes.grayR = gR.data();
        restore_backup();
    }
    void restore_backup() {  // nes-py reset(): _will_reset, _restore, _did_reset
        nes.s = lvl->backup;
        x_last = lvl->x_last0; time_last = lvl->time_last0;
        obs_is_backup = true;
    }

    inline void emu_frame(uint8_t a, bool render) {
        nes.s.joy_buttons[0] = a;
        nes.render = render;
        nes.run_frame();
        frames++;
        if (render) obs_is_backup = false;
    }
    inline void hacks(bool render) {  // SuperMarioBrosEnv._did_step for a not-done single-stage env
        uint8_t* ram = nes.s.ram;
        uint8_t cat = ram[0x06DE];
        if (cat > 1 && cat < 255) ram[0x06DE] = 1;  // _skip_change_area
        for (;;) {                                  // _skip_occupied_states
            SmbRam R{ram};
            if (!(R.is_busy() || R.is_world_over())) break;
            ram[0x07A0] = 0;
            emu_frame(0, render);
        }
    }
    // nes-py env.step(byte) for SuperMarioBrosEnv(target=...): reward, done, info, hacks
    double frame_step(uint8_t a, bool render, bool* done) {
        emu_frame(a, render);
        SmbRam R{nes.s.ram};
        int64_t x = R.x_pos();
        int64_t xr = x - x_last; x_last = x;
        if (xr < -5 || xr > 5) xr = 0;
        int64_t tm = R.time();
        int64_t tp = tm - time_last; time_last = tm;
        if (tp > 0) tp = 0;
        bool dying = R.is_dying(), dead = R.is_dead();
        int64_t dp = (dying || dead) ? -25 : 0;
        int64_t r = xr + tp + dp;
        if (r < -15) r = -15;
        if (r > 15) r = 15;
        info = R.info();
        *done = dying || dead || info.flag_get;
        if (!*done) hacks(render);
        return (double)r;
    }
    // reference: RefMarioEnv._agent_step ; want_obs: the gray frame(s) for the observation must be produced
    double agent_step(int a, bool want_obs, bool* term) {
        const int K = cfg->skip;
        const bool mp = cfg->maxpool != 0;
        double total = 0.0;
        bool done = false;
        int i = 0;
        for (i = 0; i < K; i++) {
            int ae = a;
            if (cfg->sticky != 0.0 && rng.random() < cfg->sticky) ae = prev;
            prev = ae;
            bool render = want_obs && (i == K - 1 || (mp && i == K - 2));
            if (want_obs) snaps[i] = nes.s;
            acts[i] = (uint8_t)ae;
            total += frame_step(SIMPLE_MOVEMENT[ae], render, &done);
            if (want_obs && mp && i == K - 2) gR_prev = gR;
            if (done) break;
        }
        int last = done ? i : K - 1;
        if (want_obs && done && last < K - 1) {
            // early termination: re-emulate the last frame-step(s) from snapshots with pixels on
            if (mp && last >= 1) {
                nes.s = snaps[last - 1];
                emu_frame(SIMPLE_MOVEMENT[acts[last - 1]], true);
                hacks(true);  // frame-step last-1 was not done
                gR_prev = gR;
            } else {
                nes.s = snaps[last];
            }
            emu_frame(SIMPLE_MOVEMENT[acts[last]], true);
        }
        if (want_obs) pooled = mp && last >= 1;
        bool t_ = done;
        int life = (int)info.life;
        if (lives >= 0 && life < lives) t_ = true;
        lives = life;
        if (info.flag_get) t_ = true;
        *term = t_;
        return total;
    }
    bool pooled = false;

    void make_obs(uint8_t* out, bool allow_pool) {
        if (obs_is_backup) { memcpy(out, lvl->obs0, OSZ); return; }
        if (allow_pool && pooled) {
            for (int k = 0; k < GSZ; k++) gR_prev[k] = gR_prev[k] > gR[k] ? gR_prev[k] : gR[k];
            transpose_gray(gR_prev.data(), gT.data());
        } else {
            transpose_gray(gR.data(), gT.data());
        }
        rs->run(gT.data(), out, bufT.data());
    }

    // reference: RefMarioEnv.reset
    void reset(uint8_t* obs_out) {
        restore_backup(); prev = 0; lives = -1;
        if (cfg->noop_max) {
            int64_t k = rng.integers(cfg->noop_max + 1);
            for (int64_t j = 0; j < k; j++) {
                bool term;
                agent_step(0, j == k - 1, &term);
                if (term) { restore_backup(); prev = 0; lives = -1; }
            }
        }
        t = 0; ep_ret = 0.0; ep_len = 0;
        make_obs(obs_out, false);
    }
    // reference: RefMarioEnv.step
    void step(int a, uint8_t* obs_out, float* rew, bool* term, bool* trunc) {
        bool te;
        double r = agent_step(a, true, &te);
        t++;
        *trunc = t >= cfg->max_steps;
        ep_ret += r; ep_len++;
        *rew = (float)r; *term = te;
        make_obs(obs_out, true);
    }
};

// ------------------------------------------------------------------ boot (SuperMarioBrosEnv.__init__)
struct Boot {
    Nes nes;
    std::vector<uint8_t> gR, screen;
    void fa(uint8_t a) { nes.s.joy_buttons[0] = a; nes.render = true; nes.run_frame(); }
    int64_t time() { return SmbRam{nes.s.ram}.time(); }
};

static void boot_level(const Cart* cart, const uint8_t* lut, const AreaResizer* rs, Level* L) {
    Boot b;
    b.gR.assign(GSZ, 0); b.screen.assign(GSZ, 0);
    b.nes.grayR = b.gR.data(); b.nes.screen = b.screen.data(); b.nes.gray_lut = lut;
    b.nes.power_on(cart);
    uint8_t* ram = b.nes.s.ram;
    b.fa(8); b.fa(0);
    while (b.time() == 0) {
        b.fa(8);
        ram[0x075f] = (uint8_t)(L->world - 1); ram[0x075c] = (uint8_t)(L->stage - 1); ram[0x0760] = (uint8_t)(L->area - 1);
        b.fa(0);
        ram[0x07A0] = 0;
    }
    int64_t tl = b.time();
    while (b.time() >= tl) { tl = b.time(); b.fa(8); b.fa(0); }
    L->backup = b.nes.s;
    L->x_last0 = SmbRam{ram}.x_pos();
    L->time_last0 = b.time();
    std::vector<float> bufT(84 * 240);
    std::vector<uint8_t> gT(GSZ);
    transpose_gray(b.gR.data(), gT.data());
    rs->run(gT.data(), L->obs0, bufT.data());
    L->screen0 = b.screen;
}

// ------------------------------------------------------------------ vec env + thread pool
struct VecEnv {
    int n = 0, K = 4;
    Config cfg;
    Cart cart;
    AreaResizer rs;
    uint8_t lut[256];
    std::vector<Level*> levels;
    std::vector<Env> envs;
    // outputs (numpy-owned)
    uint8_t* obs = nullptr;    // (n, K, 84, 84) stacked
    uint8_t* fin = nullptr;    // (n, K, 84, 84) final stacked obs for done envs
    float* rew = nullptr;
    uint8_t* term = nullptr;
    uint8_t* trunc = nullptr;
    int64_t* info = nullptr;   // (n, N_INFO)
    double* ep_ret = nullptr;
    int64_t* ep_len = nullptr;
    std::vector<int> actions;
    std::vector<uint8_t> scratch;  // n*OSZ
    // pool
    std::vector<std::thread> th;
    std::mutex mu;
    std::condition_variable cv_go, cv_done;
    uint64_t gen = 0;
    int busy = 0;
    bool stop = false, pending = false;
    std::atomic<int> next{0};
    int job = 0;  // 1 = step, 2 = reset

    void do_env(int i) {
        Env& e = envs[i];
        uint8_t* st = obs + (size_t)i * K * OSZ;
        uint8_t* o = scratch.data() + (size_t)i * OSZ;
        if (job == 2) {
            e.reset(o);
            memset(st, 0, (size_t)K * OSZ);
            memcpy(st + (size_t)(K - 1) * OSZ, o, OSZ);
            return;
        }
        float r; bool te, tr;
        e.step(actions[i], o, &r, &te, &tr);
        rew[i] = r; term[i] = te; trunc[i] = tr;
        const Info& f = e.info;
        int64_t* inf = info + (size_t)i * N_INFO;
        inf[0] = f.x_pos; inf[1] = f.flag_get; inf[2] = f.life; inf[3] = f.score; inf[4] = f.time;
        inf[5] = f.world; inf[6] = f.stage; inf[7] = f.coins; inf[8] = f.y_pos; inf[9] = f.status;
        memmove(st, st + OSZ, (size_t)(K - 1) * OSZ);
        if (te || tr) {
            uint8_t* fo = fin + (size_t)i * K * OSZ;
            memcpy(fo, st, (size_t)(K - 1) * OSZ);
            memcpy(fo + (size_t)(K - 1) * OSZ, o, OSZ);
            ep_ret[i] = e.ep_ret; ep_len[i] = e.ep_len;
            e.reset(o);
            memset(st, 0, (size_t)(K - 1) * OSZ);
        }
        memcpy(st + (size_t)(K - 1) * OSZ, o, OSZ);
    }
    void drain() {
        for (;;) {
            int i = next.fetch_add(1);
            if (i >= n) break;
            do_env(i);
        }
    }
    void worker() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_go.wait(lk, [&] { return stop || gen != seen; });
                if (stop) return;
                seen = gen;
            }
            drain();
            {
                std::lock_guard<std::mutex> lk(mu);
                if (--busy == 0) cv_done.notify_all();
            }
        }
    }
    void launch(int j) {
        job = j;
        next.store(0);
        if (th.empty()) { pending = true; return; }
        {
            std::lock_guard<std::mutex> lk(mu);
            busy = (int)th.size();
            gen++;
            pending = true;
        }
        cv_go.notify_all();
    }
    void wait() {
        if (!pending) return;
        if (th.empty()) { drain(); pending = false; return; }
        std::unique_lock<std::mutex> lk(mu);
        cv_done.wait(lk, [&] { return busy == 0; });
        pending = false;
    }
    ~VecEnv() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv_go.notify_all();
        for (auto& t : th) t.join();
        for (auto* l : levels) delete l;
    }
};

}  // namespace fm

using namespace fm;

extern "C" {

// rng: n * 6 uint64 (state_hi, state_lo, inc_hi, inc_lo, has_uint32, uinteger)
// levels: n * 3 ints (world, stage, area)
void* fm_create(const uint8_t* rom, int64_t rom_len, int n, const int* levels, const uint8_t* lut,
                int skip, int max_steps, double sticky, int noop_max, int maxpool, int frame_stack,
                const uint64_t* rng, int n_threads) {
    if (skip < 1 || skip > 16) return nullptr;
    VecEnv* v = new VecEnv();
    if (cart_load(&v->cart, rom, (size_t)rom_len) != 0) { delete v; return nullptr; }
    v->n = n; v->K = frame_stack;
    v->cfg.skip = skip; v->cfg.max_steps = max_steps; v->cfg.sticky = sticky;
    v->cfg.noop_max = noop_max; v->cfg.maxpool = maxpool; v->cfg.frame_stack = frame_stack;
    memcpy(v->lut, lut, 256);
    v->rs.init();
    v->envs.resize(n);
    v->actions.assign(n, 0);
    v->scratch.assign((size_t)n * OSZ, 0);
    for (int i = 0; i < n; i++) {
        int w = levels[3 * i], s = levels[3 * i + 1], a = levels[3 * i + 2];
        Level* L = nullptr;
        for (auto* l : v->levels) if (l->world == w && l->stage == s && l->area == a) L = l;
        if (!L) {
            L = new Level();
            L->world = w; L->stage = s; L->area = a;
            boot_level(&v->cart, v->lut, &v->rs, L);
            v->levels.push_back(L);
        }
        Env& e = v->envs[i];
        e.init(L, &v->cfg, &v->rs, v->lut, &v->cart);
        const uint64_t* r = rng + 6 * i;
        e.rng.state = ((unsigned __int128)r[0] << 64) | r[1];
        e.rng.inc = ((unsigned __int128)r[2] << 64) | r[3];
        e.rng.has32 = (int)r[4];
        e.rng.uinteger = (uint32_t)r[5];
    }
    if (n_threads > 1)
        for (int t = 0; t < n_threads; t++) v->th.emplace_back([v] { v->worker(); });
    return v;
}

void fm_set_buffers(void* h, uint8_t* obs, uint8_t* fin, float* rew, uint8_t* term, uint8_t* trunc,
                    int64_t* info, double* ep_ret, int64_t* ep_len) {
    VecEnv* v = (VecEnv*)h;
    v->obs = obs; v->fin = fin; v->rew = rew; v->term = term; v->trunc = trunc;
    v->info = info; v->ep_ret = ep_ret; v->ep_len = ep_len;
}

void fm_reset(void* h) { VecEnv* v = (VecEnv*)h; v->wait(); v->launch(2); v->wait(); }
void fm_step_async(void* h, const int64_t* actions) {
    VecEnv* v = (VecEnv*)h;
    v->wait();
    for (int i = 0; i < v->n; i++) v->actions[i] = (int)actions[i];
    v->launch(1);
}
void fm_step_wait(void* h) { ((VecEnv*)h)->wait(); }
void fm_destroy(void* h) { VecEnv* v = (VecEnv*)h; v->wait(); delete v; }
int64_t fm_frames(void* h) {
    VecEnv* v = (VecEnv*)h; int64_t f = 0;
    for (auto& e : v->envs) f += (int64_t)e.frames;
    return f;
}
int64_t fm_bad_ops(void* h) {
    VecEnv* v = (VecEnv*)h; int64_t f = 0;
    for (auto& e : v->envs) f += e.nes.bad_ops;
    return f;
}
// copy env i's RAM (0x800) out (debug / fidelity)
void fm_env_ram(void* h, int i, uint8_t* out) { memcpy(out, ((VecEnv*)h)->envs[i].nes.s.ram, 0x800); }
int64_t fm_state_bytes() { return (int64_t)sizeof(NesState); }
int64_t fm_env_bytes(void* h) {
    VecEnv* v = (VecEnv*)h; (void)v;
    return (int64_t)(sizeof(Env) + 3 * GSZ + 84 * 240 * sizeof(float) + OSZ);
}

void fm_palette(uint32_t* out) { memcpy(out, PALETTE, sizeof(PALETTE)); }

void fm_resize_ref(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    area_resize_ref(src, sw, sh, dst, dw, dh);
}
// src row-major 240x256 -> fast path (via transpose)
void fm_resize_fast(const uint8_t* src, uint8_t* dst) {
    static AreaResizer rs; static bool inited = false;
    if (!inited) { rs.init(); inited = true; }
    std::vector<uint8_t> gT(GSZ); std::vector<float> buf(84 * 240);
    transpose_gray(src, gT.data());
    rs.run(gT.data(), dst, buf.data());
}

// ---------------- raw emulator (frame-level fidelity tests vs nes-py)
struct Raw { Cart cart; Nes nes; std::vector<uint8_t> screen, gR; uint8_t lut[256]; NesState backup; };
void* fm_raw_create(const uint8_t* rom, int64_t len) {
    Raw* r = new Raw();
    if (cart_load(&r->cart, rom, (size_t)len) != 0) { delete r; return nullptr; }
    r->screen.assign(GSZ, 0); r->gR.assign(GSZ, 0);
    for (int i = 0; i < 256; i++) r->lut[i] = (uint8_t)i;
    r->nes.screen = r->screen.data(); r->nes.grayR = r->gR.data(); r->nes.gray_lut = r->lut;
    r->nes.power_on(&r->cart);
    return r;
}
void fm_raw_frame(void* h, int buttons, int render) {
    Raw* r = (Raw*)h;
    r->nes.s.joy_buttons[0] = (uint8_t)buttons;
    r->nes.render = render != 0;
    r->nes.run_frame();
}
uint8_t* fm_raw_ram(void* h) { return ((Raw*)h)->nes.s.ram; }
uint8_t* fm_raw_screen(void* h) { return ((Raw*)h)->screen.data(); }
void fm_raw_backup(void* h) { Raw* r = (Raw*)h; r->backup = r->nes.s; }
void fm_raw_restore(void* h) { Raw* r = (Raw*)h; r->nes.s = r->backup; }
void fm_raw_destroy(void* h) { delete (Raw*)h; }
int64_t fm_raw_bad_ops(void* h) { return ((Raw*)h)->nes.bad_ops; }
}

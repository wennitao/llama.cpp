// fa-exc-gpu: the GPU half of the heterogeneous prefill attention split, standalone and self-checking.
//
// The NPU runs the UNION of a 256-row query tile's four 64-row sub-block selections. The split keeps
// the blocks that at least c_star sub-blocks want on the NPU and gives the rest ("exceptions") to the
// Adreno, one list per 64-row sub-block. The two halves are merged by the online-softmax fold of
// htp-ops.h (HTP_FA_FOLD_MAGIC): the GPU leaves an UNNORMALISED (m, l, acc) triple per row and the
// HMX kernel folds it into its final normalization.
//
// This tool builds that GPU half only, so it can be validated before any integration exists:
//   - kernel: mllm's bs_fa2_fused, ported to the llama.cpp KV layout (V and Q read in place, only the
//     selected K blocks staged transposed), made GQA-aware, and with the OWRITE epilogue replaced by
//     the fold triple written straight into the hexagon rpcmem buffer (ION-aliased into OpenCL).
//   - check: a double-precision host reference for attention over the exception blocks only, compared
//     against the emitted m, l and acc SEPARATELY - the fold consumes the unnormalised triple, so a
//     tool that only checks acc/l would miss an anchor that is off by a constant.
//
// Selection granularity follows the device: flash-attn-ops.c reads src[5] as one block list per
// (query block, KV head, sequence). So the exception lists are per (KV head, 64-row sub-block) and the
// G query heads of a group share both the list and the staged K.
//
// Everything here runs on the GPU and the CPU. No HTP graph is built; the HTP0 device is used only to
// get an rpcmem buffer, which is what the fold buffer must live in once this is wired up.

#include "ggml.h"
#include "ggml-backend.h"
#include "htp-ops.h"
#include "flash-attn-ops.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <time.h>
#include <vector>

extern "C" int rpcmem_to_fd(void * po);

static double now_ms() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

#define DIE(...)      do { fprintf(stderr, "fa-exc-gpu: " __VA_ARGS__); fputc('\n', stderr); exit(1); } while (0)
#define CHECK(c, ...) do { if (!(c)) DIE(__VA_ARGS__); } while (0)
#define CL_CHECK(e, what) do { cl_int _e = (e); if (_e != CL_SUCCESS) DIE("%s failed (%d)", (what), _e); } while (0)

// The kernel's lane mapping is 8x16 for QK and 16x8 for PV, which only adds up to 128 lanes at D=128.
#define FA_D_FIXED 128
#define FA2_BQ     32
#define FA2_BK     64

// ---------------------------------------------------------------------------------------

struct options {
    int  d = 128, nh = 16, nkvh = 8, lq = 1024, kv = 4096;
    int  bq = 256, bs = 64;          // query tile and selection block size (C2)
    int  topk = 8;                   // blocks each sub-block selects (shared pool + exceptions)
    int  exc = 2;                    // of those, how many are exceptions (go to the GPU)
    int  cstar = 2;                  // blocks wanted by >= cstar sub-blocks stay on the NPU
    int  empty_mod = 0;              // every empty_mod-th sub-block gets no exceptions (0 = off)
    int  iters = 20;
    bool pack_q = false;             // A/B: pack Q to half in mllm's layout instead of reading it in place
    bool no_mask = false;            // drop the causal mask (timing arm only, not a valid partial)
    bool check = true;
    bool verbose = false;
};

static void usage() {
    printf("usage: llama-fa-exc-gpu [options]\n"
           "  --d N         head dim (must be %d, the kernel's lane mapping needs it)\n"
           "  --nh N        query heads          (default 16)\n"
           "  --nkvh N      KV heads             (default 8)\n"
           "  --lq N        query tokens         (default 1024, multiple of the query tile)\n"
           "  --kv N        KV length            (default 4096, multiple of 64, >= lq)\n"
           "  --bq N        query tile rows      (default 256)\n"
           "  --bs N        selection block      (must be %d)\n"
           "  --topk N      blocks selected per 64-row sub-block (default 8)\n"
           "  --exc N       of those, exceptions per sub-block -> the GPU (default 2)\n"
           "  --cstar N     NPU keeps blocks wanted by >= cstar of the 4 sub-blocks (default 2)\n"
           "  --empty-mod N every N-th sub-block gets 0 exceptions, to exercise empty rows (0 = off)\n"
           "  --iters N     timed iterations     (default 20)\n"
           "  --pack-q      pack Q to half (mllm layout) instead of reading f32 Q in place\n"
           "  --no-mask     skip the causal mask (timing only; the partial is then NOT foldable)\n"
           "  --no-check    skip the double-precision host reference\n"
           "  -v            verbose\n", FA_D_FIXED, FA2_BK);
}

// ---------------------------------------------------------------------------------------
// The kernels. stage_k transposes only the selected blocks; fa_exc is mllm's bs_fa2_fused with
// V/Q/mask read in place, GQA folded in, and the fold triple as the epilogue.
// ---------------------------------------------------------------------------------------

static const char * cl_src = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define FA2_BQ  32
#define FA2_BK  64
#ifndef FA_D
#define FA_D 128
#endif

// HTP_FA_M_INITIAL_VAL. NOT -INFINITY: fa_fold_diag_vec clamps (m - M) and feeds it to an HVX exp,
// so an infinite anchor would poison the merge instead of weighing zero.
#define M_EMPTY (-10000.0f)

// Kt[kv_head][d][key] over the union of this KV head's exception blocks. The QK loop reads 8
// consecutive KEYS at one d, which llama.cpp's K layout (256 B contiguous per (token, kv_head))
// cannot serve. One work item moves 8 dims of one key, so the 256 B source row is read whole.
__kernel void stage_k(__global const uchar * base,
                      __global half * Kt,
                      const int k_off, const int k_nb1, const int k_nb2,
                      const int abs_off, const int cnt_off,
                      const int abs_stride, const int n_stage) {
    const int ki  = get_global_id(0);
    const int dg  = get_global_id(1);
    const int kvh = get_global_id(2);
    __global const int * cnt = (__global const int *)(base + cnt_off);
    if (ki >= cnt[kvh]) return;
    __global const int * abs_tab = (__global const int *)(base + abs_off);
    const int tok = abs_tab[kvh * abs_stride + (ki >> 6)] * FA2_BK + (ki & (FA2_BK - 1));
    __global const half * src = (__global const half *)(base + k_off + (long) tok * k_nb1 + (long) kvh * k_nb2);
    const half8 kr = vload8(0, src + dg * 8);
    __global half * dst = Kt + (long) kvh * FA_D * n_stage + (long) (dg * 8) * n_stage + ki;
    const long st = n_stage;
    dst[0]      = kr.s0; dst[st]     = kr.s1; dst[st * 2] = kr.s2; dst[st * 3] = kr.s3;
    dst[st * 4] = kr.s4; dst[st * 5] = kr.s5; dst[st * 6] = kr.s6; dst[st * 7] = kr.s7;
}

// Pack Q to mllm's Qp[bh][d/4][q][4] half layout. Only built for the --pack-q arm.
__kernel void pack_q(__global const uchar * base, __global half * Qp,
                     const int q_off, const int q_nb1, const int q_nb2, const int Sq) {
    const int q  = get_global_id(0);
    const int g  = get_global_id(1);
    const int bh = get_global_id(2);
    if (q >= Sq || g >= FA_D / 4) return;
    __global const float * src = (__global const float *)(base + q_off + (long) bh * q_nb2 + (long) q * q_nb1) + g * 4;
    __global half * dst = Qp + (long) bh * (FA_D / 4) * Sq * 4 + ((long) g * Sq + q) * 4;
    for (int kk = 0; kk < 4; ++kk) dst[kk] = (half) src[kk];
}

__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void fa_exc(__read_only image1d_buffer_t Kt_img,
            __read_only image1d_buffer_t V_img,
            __global uchar * base,
            __global const half * Qp,
            const int q_off, const int q_nb1, const int q_nb2,
            const int mask_off, const int mask_nb1, const int use_mask,
            const int idx_off, const int abs_off, const int cnt_off, const int lcnt_off, const int err_off,
            const int f_m, const int f_l, const int f_acc,
            const int Sq, const int G, const int num_sb, const int qb_per_sb, const int top_k,
            const int abs_stride, const int n_stage,
            const int v_base_tex, const int v_tok_tex, const int v_head_tex,
            const float scale) {
    const int t  = get_local_id(0);
    const int qb = get_global_id(1);
    const int bh = get_global_id(2);
    const int q0 = qb * FA2_BQ;
    if (q0 >= Sq) return;
    const int kvh = bh / G;
    const int sb  = qb / qb_per_sb;

    __global const int * idx_row = (__global const int *)(base + idx_off) + ((long) kvh * num_sb + sb) * top_k;
    const int cnt_row = ((__global const int *)(base + lcnt_off))[(long) kvh * num_sb + sb];
    __global const int * abs_tab = (__global const int *)(base + abs_off);
    __global const int * cnt     = (__global const int *)(base + cnt_off);
    const int n_blk_kvh = cnt[kvh] >> 6;

    __local float S_lds[FA2_BQ][FA2_BK];
    __local float m_run[FA2_BQ]; __local float l_run[FA2_BQ]; __local float a_sh[FA2_BQ];

    const int pv_mt = t % (FA_D / 8); const int pv_nt = t / (FA_D / 8); const int pv_d0 = pv_mt * 8;
    float8 o0 = (float8)0, o1 = (float8)0, o2 = (float8)0, o3 = (float8)0;
    if (t < FA2_BQ) { m_run[t] = -INFINITY; l_run[t] = 0.0f; }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int M_4kv   = n_stage >> 2;
    const int kt_head = kvh * FA_D * M_4kv;
    const int v_head  = v_base_tex + kvh * v_head_tex;
#ifdef PACK_Q
    __global const half * Qph = Qp + (long) bh * (FA_D / 4) * Sq * 4;
#else
    __global const float * Qh = (__global const float *)(base + q_off + (long) bh * q_nb2);
    const int q_row_f = q_nb1 >> 2;
#endif
    __global const half * Mk = (__global const half *)(base + mask_off);
    const int mask_row_h = mask_nb1 >> 1;

    // The list length comes from cnt_row, NOT from a -1 sentinel. The producer of these lists is an
    // argsort-DESC of a 0/1 membership row, whose tail is the NON-members -- and the non-members of
    // an exception row are exactly that tile's SHARED blocks, which the NPU also computes. Reading
    // past the count would hand both engines the same block and the LSE merge would double-count
    // it, shifting that block's logit by +ln2 with nothing raised anywhere.
    const int n_list = (cnt_row < top_k) ? cnt_row : top_k;
    for (int s = 0; s < n_list; ++s) {
        const int cb = idx_row[s];
        if (cb < 0) break;               // defensive: a padded list is still honoured
        if (cb >= n_blk_kvh) {           // host validated this table, so reaching here is a bug
            if (t == 0) atomic_inc((volatile __global int *)(base + err_off));
            break;
        }
        const int kt0 = cb * FA2_BK;                                   // compact key offset into Kt
        const int vk0 = abs_tab[kvh * abs_stride + cb] * FA2_BK;       // absolute key offset for V and the mask
        {
            const int mt = t % (FA2_BK / 8); const int nt = t / (FA2_BK / 8);
            const int kbase = kt_head + (kt0 >> 2) + mt * 2;
            float8 c0 = (float8)0, c1 = (float8)0;
            for (int i = 0; i < FA_D; i += 4) {
                const int tb = kbase + i * M_4kv;
                half8 B0, B1, B2, B3;
                B0.s0123 = read_imageh(Kt_img, tb);              B0.s4567 = read_imageh(Kt_img, tb + 1);
                B1.s0123 = read_imageh(Kt_img, tb + M_4kv);      B1.s4567 = read_imageh(Kt_img, tb + M_4kv + 1);
                B2.s0123 = read_imageh(Kt_img, tb + 2 * M_4kv);  B2.s4567 = read_imageh(Kt_img, tb + 2 * M_4kv + 1);
                B3.s0123 = read_imageh(Kt_img, tb + 3 * M_4kv);  B3.s4567 = read_imageh(Kt_img, tb + 3 * M_4kv + 1);
                const float8 b0 = convert_float8(B0), b1 = convert_float8(B1), b2 = convert_float8(B2), b3 = convert_float8(B3);
#ifdef PACK_Q
                const half8 w = vload8(0, Qph + ((long)(i >> 2) * Sq + q0 + nt * 2) * 4);
                c0 += b0*w.s0; c0 += b1*w.s1; c0 += b2*w.s2; c0 += b3*w.s3;
                c1 += b0*w.s4; c1 += b1*w.s5; c1 += b2*w.s6; c1 += b3*w.s7;
#else
                const float4 wa = vload4(0, Qh + (long)(q0 + nt * 2)     * q_row_f + i);
                const float4 wb = vload4(0, Qh + (long)(q0 + nt * 2 + 1) * q_row_f + i);
                c0 += b0*wa.s0; c0 += b1*wa.s1; c0 += b2*wa.s2; c0 += b3*wa.s3;
                c1 += b0*wb.s0; c1 += b1*wb.s1; c1 += b2*wb.s2; c1 += b3*wb.s3;
#endif
            }
            const int kb = vk0 + mt * 8;
            #define EMITS(NN, CV) { const int qg = q0 + nt * 2 + (NN); \
                float8 v = (CV) * scale; \
                if (use_mask) v += convert_float8(vload8(0, Mk + (long) qg * mask_row_h + kb)); \
                vstore8(v, 0, &S_lds[nt * 2 + (NN)][mt * 8]); }
            EMITS(0, c0); EMITS(1, c1);
            #undef EMITS
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (t < FA2_BQ) {
            const int q = t; __local float * row = S_lds[q];
            float8 mx = (float8)(-INFINITY);
            for (int k = 0; k < FA2_BK; k += 8) mx = fmax(mx, vload8(0, row + k));
            float4 m4 = fmax(mx.lo, mx.hi); float2 m2 = fmax(m4.lo, m4.hi); float m_tile = fmax(m2.s0, m2.s1);
            const float m_old = m_run[q]; const float m_new = fmax(m_old, m_tile);
            const float a = (m_old == -INFINITY) ? 0.0f : native_exp(m_old - m_new);
            float8 l8 = (float8)0;
            for (int k = 0; k < FA2_BK; k += 8) {
                const float8 s8 = vload8(0, row + k);
                const float8 p = select(native_exp(s8 - m_new), (float8)0, isinf(s8));
                vstore8(p, 0, row + k); l8 += p;
            }
            float4 l4 = l8.lo + l8.hi; float2 l2 = l4.lo + l4.hi;
            l_run[q] = a * l_run[q] + (l2.s0 + l2.s1); m_run[q] = m_new; a_sh[q] = a;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        o0 *= a_sh[pv_nt * 4 + 0]; o1 *= a_sh[pv_nt * 4 + 1]; o2 *= a_sh[pv_nt * 4 + 2]; o3 *= a_sh[pv_nt * 4 + 3];
        for (int kk = 0; kk < FA2_BK; kk += 4) {
            const int vt = v_head + (vk0 + kk) * v_tok_tex + pv_mt * 2;
            half8 V0, V1, V2, V3;
            V0.s0123 = read_imageh(V_img, vt);                   V0.s4567 = read_imageh(V_img, vt + 1);
            V1.s0123 = read_imageh(V_img, vt + v_tok_tex);       V1.s4567 = read_imageh(V_img, vt + v_tok_tex + 1);
            V2.s0123 = read_imageh(V_img, vt + 2 * v_tok_tex);   V2.s4567 = read_imageh(V_img, vt + 2 * v_tok_tex + 1);
            V3.s0123 = read_imageh(V_img, vt + 3 * v_tok_tex);   V3.s4567 = read_imageh(V_img, vt + 3 * v_tok_tex + 1);
            const float8 v0 = convert_float8(V0), v1 = convert_float8(V1), v2 = convert_float8(V2), v3 = convert_float8(V3);
            float4 p0 = vload4(0, &S_lds[pv_nt * 4 + 0][kk]); float4 p1 = vload4(0, &S_lds[pv_nt * 4 + 1][kk]);
            float4 p2 = vload4(0, &S_lds[pv_nt * 4 + 2][kk]); float4 p3 = vload4(0, &S_lds[pv_nt * 4 + 3][kk]);
            o0 += v0*p0.s0; o0 += v1*p0.s1; o0 += v2*p0.s2; o0 += v3*p0.s3;
            o1 += v0*p1.s0; o1 += v1*p1.s1; o1 += v2*p1.s2; o1 += v3*p1.s3;
            o2 += v0*p2.s0; o2 += v1*p2.s1; o2 += v2*p2.s2; o2 += v3*p2.s3;
            o3 += v0*p3.s0; o3 += v1*p3.s1; o3 += v2*p3.s2; o3 += v3*p3.s3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Fold epilogue: the UNNORMALISED triple, not O. A row with no exception block (or one whose
    // every key is masked) leaves the exact empty partial the HMX fold weighs at zero.
    barrier(CLK_LOCAL_MEM_FENCE);
    const long r0 = (long) bh * Sq + q0;
    __global float * Fm = (__global float *)(base + f_m);
    __global float * Fl = (__global float *)(base + f_l);
    __global float * Fa = (__global float *)(base + f_acc);
    if (t < FA2_BQ) {
        const float l = l_run[t];
        Fm[r0 + t] = (l > 0.0f) ? m_run[t] : M_EMPTY;
        Fl[r0 + t] = l;
    }
    #define OWRITE(NN, OV) { const int q = pv_nt * 4 + (NN); const float l = l_run[q]; \
        vstore8((l > 0.0f) ? (OV) : (float8)0, 0, Fa + (r0 + q) * FA_D + pv_d0); }
    OWRITE(0, o0); OWRITE(1, o1); OWRITE(2, o2); OWRITE(3, o3);
    #undef OWRITE
}
)CL";

// ---------------------------------------------------------------------------------------

struct layout {
    size_t lcnt = 0;
    size_t q = 0, k = 0, v = 0, mask = 0, fold = 0, idx = 0, abs = 0, cnt = 0, err = 0, total = 0;
    size_t off_m = 0, off_l = 0, off_acc = 0;     // from the fold header, 128-aligned
    size_t q_nb1 = 0, q_nb2 = 0, kv_nb1 = 0, kv_nb2 = 0, mask_nb1 = 0;
};

static inline uint16_t f2h(float f) { return ggml_fp32_to_fp16(f); }
static inline float    h2f(uint16_t h) { return ggml_fp16_to_fp32(h); }

struct cmp {
    double max_abs = 0, sum2 = 0, ref2 = 0;
    void add(double got, double want) {
        const double d = got - want;
        max_abs = std::max(max_abs, fabs(d));
        sum2 += d * d; ref2 += want * want;
    }
    double nmse() const { return sum2 / (ref2 > 0 ? ref2 : 1.0); }
};

int main(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](int & dst) { CHECK(i + 1 < argc, "%s needs a value", a.c_str()); dst = atoi(argv[++i]); };
        if      (a == "--d")          next(o.d);
        else if (a == "--nh")         next(o.nh);
        else if (a == "--nkvh")       next(o.nkvh);
        else if (a == "--lq")         next(o.lq);
        else if (a == "--kv")         next(o.kv);
        else if (a == "--bq")         next(o.bq);
        else if (a == "--bs")         next(o.bs);
        else if (a == "--topk")       next(o.topk);
        else if (a == "--exc")        next(o.exc);
        else if (a == "--cstar")      next(o.cstar);
        else if (a == "--empty-mod")  next(o.empty_mod);
        else if (a == "--iters")      next(o.iters);
        else if (a == "--pack-q")     o.pack_q = true;
        else if (a == "--no-mask")    o.no_mask = true;
        else if (a == "--no-check")   o.check = false;
        else if (a == "-v")           o.verbose = true;
        else { usage(); return 1; }
    }
    setvbuf(stdout, nullptr, _IONBF, 0);

    // ---- geometry the kernel cannot bend ----
    CHECK(o.d == FA_D_FIXED, "--d must be %d: the kernel maps 128 lanes as 8x16 (QK) and 16x8 (PV), which only closes at D=%d", FA_D_FIXED, FA_D_FIXED);
    CHECK(o.bs == FA2_BK, "--bs must be %d (the kernel's KV tile)", FA2_BK);
    CHECK(o.bq % o.bs == 0, "--bq must be a multiple of --bs");
    CHECK(o.bs % FA2_BQ == 0, "--bs must be a multiple of the kernel's %d-row query block", FA2_BQ);
    CHECK(o.nh > 0 && o.nkvh > 0 && o.nh % o.nkvh == 0, "--nh must be a multiple of --nkvh");
    CHECK(o.lq > 0 && o.lq % o.bq == 0, "--lq must be a multiple of --bq");
    CHECK(o.kv >= o.lq && o.kv % o.bs == 0, "--kv must be >= --lq and a multiple of %d", o.bs);
    CHECK(o.exc >= 0 && o.exc < o.topk, "--exc must be in [0, --topk)");
    const int R = o.bq / o.bs;
    CHECK(o.cstar >= 2 && o.cstar <= R, "--cstar must be in [2, %d]: below 2 nothing is an exception, above %d nothing is shared", R, R);
    CHECK(o.iters > 0, "--iters must be > 0");

    const int    G       = o.nh / o.nkvh;
    const int    num_sb  = o.lq / o.bs;          // 64-row sub-blocks per KV head
    const int    n_tiles = o.lq / o.bq;
    const int    num_qb  = o.lq / FA2_BQ;        // kernel query blocks
    const int    qb_per_sb = o.bs / FA2_BQ;
    const int    nblk    = o.kv / o.bs;
    const float  scale   = 1.0f / sqrtf((float) o.d);
    const size_t D       = (size_t) o.d;
    const size_t rows    = (size_t) o.lq * o.nh;

    // ---- buffer layout: everything the GPU touches lives in one rpcmem buffer ----
    layout L;
    L.q_nb1    = D * sizeof(float);                       // Q: [d, lq, nh], contiguous f32
    L.q_nb2    = (size_t) o.lq * L.q_nb1;
    L.kv_nb1   = (size_t) o.nkvh * D * sizeof(uint16_t);  // K/V after llama.cpp's permute (C2)
    L.kv_nb2   = D * sizeof(uint16_t);
    L.mask_nb1 = (size_t) o.kv * sizeof(uint16_t);

    auto bump = [](size_t & off, size_t bytes) { size_t at = off; off = (off + bytes + 4095) & ~(size_t) 4095; return at; };
    size_t off = 0;
    L.q    = bump(off, (size_t) o.nh * L.q_nb2);
    L.k    = bump(off, (size_t) o.kv * L.kv_nb1);
    L.v    = bump(off, (size_t) o.kv * L.kv_nb1);
    L.mask = bump(off, (size_t) GGML_PAD(o.lq, 64) * L.mask_nb1);
    const size_t ml_bytes = (rows * sizeof(float) + 127) & ~(size_t) 127;
    L.off_m   = 128;
    L.off_l   = L.off_m + ml_bytes;
    L.off_acc = L.off_l + ml_bytes;
    const size_t fold_bytes = L.off_acc + rows * D * sizeof(float);
    L.fold = bump(off, fold_bytes);
    L.idx  = bump(off, (size_t) o.nkvh * num_sb * std::max(o.exc, 1) * sizeof(int32_t));
    L.abs  = bump(off, (size_t) o.nkvh * std::max(nblk, 1) * sizeof(int32_t));
    L.cnt  = bump(off, (size_t) o.nkvh * sizeof(int32_t));
    L.lcnt = bump(off, (size_t) o.nkvh * num_sb * sizeof(int32_t));
    L.err  = bump(off, 256);
    L.total = (off + (1u << 20) - 1) & ~((size_t) (1u << 20) - 1);

    // Every offset the kernel casts must satisfy the alignment its loads assume.
    CHECK((L.fold + L.off_acc) % 32 == 0, "acc base must be 32 B aligned for the vstore8");
    CHECK(L.kv_nb1 % 8 == 0 && L.kv_nb2 % 8 == 0, "K/V strides must be whole RGBA-half texels");
    CHECK(L.v % 8 == 0, "V base must be a whole texel offset");
    CHECK(L.mask_nb1 % 16 == 0, "mask row stride must be 16 B for the vload8 of half");
    CHECK(L.q_nb1 % 16 == 0 && L.q_nb2 % 16 == 0, "Q strides must be 16 B for the vload4 of float");
    CHECK(L.total < (size_t) 1 << 31, "buffer must stay under 2 GB: kernel offsets are int");

    // ---- rpcmem buffer (the fold buffer has to live where the DSP can read it) ----
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        if (!strcmp(ggml_backend_dev_name(ggml_backend_dev_get(i)), "HTP0")) dev = ggml_backend_dev_get(i);
    }
    CHECK(dev, "HTP0 device not found (the fold buffer must be rpcmem)");
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    CHECK(be, "failed to init HTP0");
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), L.total);
    CHECK(buf, "failed to allocate %zu MB on HTP0", L.total >> 20);
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
    const int fd = rpcmem_to_fd(base);
    CHECK(fd >= 0, "rpcmem_to_fd failed (%d): the ION alias needs the dmabuf fd", fd);
    memset(base, 0, L.total);

    // ---- inputs ----
    float *    q    = (float *)    (base + L.q);
    uint16_t * k    = (uint16_t *) (base + L.k);
    uint16_t * v    = (uint16_t *) (base + L.v);
    uint16_t * mask = (uint16_t *) (base + L.mask);
    {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> u(-0.5f, 0.5f);
        for (size_t i = 0; i < (size_t) o.nh * o.lq * D; i++) q[i] = u(rng);
        for (size_t i = 0; i < (size_t) o.kv * o.nkvh * D; i++) { k[i] = f2h(u(rng)); v[i] = f2h(u(rng)); }
        // llama.cpp causal mask: query token t may see keys [0, (kv - lq) + t]
        const int mask_rows = GGML_PAD(o.lq, 64);
        for (int r = 0; r < mask_rows; r++) {
            const int last = (o.kv - o.lq) + std::min(r, o.lq - 1);
            for (int j = 0; j < o.kv; j++) mask[(size_t) r * o.kv + j] = f2h((!o.no_mask && j > last) ? -INFINITY : 0.0f);
        }
    }

    // ---- the split: per (KV head, 256-row tile) a shared pool, per 64-row sub-block its exceptions ----
    std::vector<std::vector<int>> sel((size_t) o.nkvh * num_sb);     // what the sub-block selects
    std::vector<std::vector<int>> exc((size_t) o.nkvh * num_sb);     // what the GPU owns
    std::vector<std::vector<int>> pool((size_t) o.nkvh * n_tiles);   // what the NPU keeps
    {
        const int n_pool = o.topk - o.exc;
        std::mt19937 rng(9876);
        for (int kvh = 0; kvh < o.nkvh; kvh++) {
            for (int tl = 0; tl < n_tiles; tl++) {
                // every sub-block of the tile must be able to select the pool, so bound it by the first one
                const int lim0 = ((o.kv - o.lq) + tl * o.bq + o.bs - 1) / o.bs;
                CHECK(lim0 + 1 >= n_pool + R * o.exc,
                      "tile %d has only %d causally allowed blocks but the split needs %d (pool %d + %d x exc %d); raise --kv or lower --topk",
                      tl, lim0 + 1, n_pool + R * o.exc, n_pool, R, o.exc);
                std::vector<int> free_blk(lim0 + 1);
                for (int b = 0; b <= lim0; b++) free_blk[b] = b;
                std::shuffle(free_blk.begin(), free_blk.end(), rng);
                size_t take = 0;
                std::vector<char> used((size_t) nblk, 0);
                std::vector<int> & pl = pool[(size_t) kvh * n_tiles + tl];
                for (int i = 0; i < n_pool; i++) { pl.push_back(free_blk[take]); used[free_blk[take++]] = 1; }
                std::sort(pl.begin(), pl.end());
                for (int j = 0; j < R; j++) {
                    const int sb = tl * R + j;
                    std::vector<int> & ex = exc[(size_t) kvh * num_sb + sb];
                    const bool empty_sb = o.empty_mod > 0 && ((kvh * num_sb + sb) % o.empty_mod) == 0;
                    if (!empty_sb && o.exc > 0) {
                        // the sub-block's own diagonal block first: it is partly masked, so the mask
                        // path is exercised instead of only fully visible blocks
                        const int diag = ((o.kv - o.lq) + tl * o.bq + j * o.bs) / o.bs;
                        if (!used[diag]) { ex.push_back(diag); used[diag] = 1; }
                        while ((int) ex.size() < o.exc) {
                            CHECK(take < free_blk.size(), "ran out of distinct blocks for tile %d", tl);
                            const int b = free_blk[take++];
                            if (!used[b]) { ex.push_back(b); used[b] = 1; }
                        }
                        std::sort(ex.begin(), ex.end());
                    }
                    std::vector<int> & sl = sel[(size_t) kvh * num_sb + sb];
                    sl = pl;
                    sl.insert(sl.end(), ex.begin(), ex.end());
                    std::sort(sl.begin(), sl.end());
                    CHECK(std::adjacent_find(sl.begin(), sl.end()) == sl.end(),
                          "selection of kvh %d sub-block %d has a duplicate block: the LSE merge SUMS, so a block on both engines is counted twice", kvh, sb);
                }
            }
        }
    }

    // C3 invariants, checked rather than assumed: the c(b) >= cstar rule must reproduce exactly the
    // sets built above, and shared must be disjoint from every sub-block's exceptions.
    size_t n_exc_pairs = 0, n_empty_sb = 0;
    for (int kvh = 0; kvh < o.nkvh; kvh++) {
        for (int tl = 0; tl < n_tiles; tl++) {
            std::vector<int> cnt_b((size_t) nblk, 0);
            for (int j = 0; j < R; j++) for (int b : sel[(size_t) kvh * num_sb + tl * R + j]) cnt_b[b]++;
            std::vector<int> derived_shared;
            for (int b = 0; b < nblk; b++) if (cnt_b[b] >= o.cstar) derived_shared.push_back(b);
            CHECK(derived_shared == pool[(size_t) kvh * n_tiles + tl],
                  "kvh %d tile %d: the c(b) >= %d rule does not reproduce the shared list", kvh, tl, o.cstar);
            for (int j = 0; j < R; j++) {
                const std::vector<int> & ex = exc[(size_t) kvh * num_sb + tl * R + j];
                std::vector<int> derived_exc;
                for (int b : sel[(size_t) kvh * num_sb + tl * R + j]) if (cnt_b[b] < o.cstar) derived_exc.push_back(b);
                CHECK(derived_exc == ex, "kvh %d tile %d sub-block %d: derived exceptions differ from the built list", kvh, tl, j);
                for (int b : ex) CHECK(std::find(derived_shared.begin(), derived_shared.end(), b) == derived_shared.end(),
                                       "kvh %d tile %d sub-block %d: block %d is on BOTH engines; the LSE merge would count it twice", kvh, tl, j, b);
                if (ex.empty()) n_empty_sb++; else n_exc_pairs += ex.size();
            }
        }
    }
    CHECK(n_exc_pairs > 0, "no exception blocks at all: nothing for the GPU to do");

    // ---- stage table: per KV head the union of its exception blocks, compacted ----
    std::vector<std::vector<int>> stage_abs((size_t) o.nkvh);
    std::vector<std::vector<int>> to_compact((size_t) o.nkvh, std::vector<int>((size_t) nblk, -1));
    for (int kvh = 0; kvh < o.nkvh; kvh++) {
        std::vector<char> seen((size_t) nblk, 0);
        for (int sb = 0; sb < num_sb; sb++) for (int b : exc[(size_t) kvh * num_sb + sb]) seen[b] = 1;
        for (int b = 0; b < nblk; b++) if (seen[b]) { to_compact[kvh][b] = (int) stage_abs[kvh].size(); stage_abs[kvh].push_back(b); }
    }
    int n_stage_blk = 1;
    size_t n_stage_used = 0;
    for (int kvh = 0; kvh < o.nkvh; kvh++) { n_stage_blk = std::max(n_stage_blk, (int) stage_abs[kvh].size()); n_stage_used += stage_abs[kvh].size(); }
    const int n_stage = n_stage_blk * o.bs;
    CHECK(n_stage % 4 == 0, "staged key capacity must be a whole texel");

    const int top_k = std::max(o.exc, 1);
    int32_t * idx_h = (int32_t *) (base + L.idx);
    int32_t * abs_h = (int32_t *) (base + L.abs);
    int32_t * cnt_h = (int32_t *) (base + L.cnt);
    int32_t * lcnt_h = (int32_t *) (base + L.lcnt);
    for (int kvh = 0; kvh < o.nkvh; kvh++) {
        cnt_h[kvh] = (int32_t) stage_abs[kvh].size() * o.bs;
        for (int c = 0; c < n_stage_blk; c++) abs_h[(size_t) kvh * n_stage_blk + c] = c < (int) stage_abs[kvh].size() ? stage_abs[kvh][c] : -1;
        for (int sb = 0; sb < num_sb; sb++) {
            const std::vector<int> & ex = exc[(size_t) kvh * num_sb + sb];
            int32_t * row = idx_h + ((size_t) kvh * num_sb + sb) * top_k;
            for (int s = 0; s < top_k; s++) row[s] = s < (int) ex.size() ? to_compact[kvh][ex[s]] : -1;
            // The kernel reads exactly this many entries. The in-graph split emits an argsort of a
            // 0/1 membership row with NO -1 padding, and that row's tail is the tile's SHARED
            // blocks, so a kernel that trusted a sentinel would fold blocks the NPU also computes.
            lcnt_h[(size_t) kvh * num_sb + sb] = (int32_t) ex.size();
            // the kernel breaks on the first -1, so padding must be a suffix, and every live entry
            // must point back at the block it came from
            bool pad = false;
            for (int s = 0; s < top_k; s++) {
                if (row[s] < 0) { pad = true; continue; }
                CHECK(!pad, "kvh %d sub-block %d: -1 padding is not a suffix", kvh, sb);
                CHECK(row[s] < (int) stage_abs[kvh].size(), "kvh %d sub-block %d slot %d: compact id %d out of range", kvh, sb, s, row[s]);
                CHECK(stage_abs[kvh][row[s]] == ex[s], "kvh %d sub-block %d slot %d: compact id maps to block %d, expected %d", kvh, sb, s, stage_abs[kvh][row[s]], ex[s]);
            }
        }
    }

    // ---- fold header (C1) ----
    // flags 0: the partial is already resident when the HMX op starts. This tool is not a LIVE
    // producer, so no handshake region is built. Fields are named, not positional: the header grew.
    struct htp_fa_fold_hdr hdr = {};
    hdr.magic   = HTP_FA_FOLD_MAGIC;
    hdr.rows    = (uint32_t) rows;
    hdr.neq1    = (uint32_t) o.lq;
    hdr.dv      = (uint32_t) D;
    hdr.off_m   = (uint32_t) L.off_m;
    hdr.off_l   = (uint32_t) L.off_l;
    hdr.off_acc = (uint32_t) L.off_acc;
    memcpy(base + L.fold, &hdr, sizeof(hdr));
    CHECK(L.off_m % 128 == 0 && L.off_l % 128 == 0 && L.off_acc % 128 == 0, "fold offsets must be 128-aligned");

    printf("shape d=%d nh=%d nkvh=%d (G=%d) lq=%d kv=%d | tile=%d sub-block=%d topk=%d cstar=%d exc=%d\n",
           o.d, o.nh, o.nkvh, G, o.lq, o.kv, o.bq, o.bs, o.topk, o.cstar, o.exc);
    printf("split  %zu exception (sub-block, block) pairs, %zu sub-blocks with none | staged %zu of %d blocks per KV head (cap %d)\n",
           n_exc_pairs, n_empty_sb, n_stage_used / (size_t) o.nkvh, nblk, n_stage_blk);
    printf("buffer %zu MB rpcmem, fold %.1f MB at +%zu\n", L.total >> 20, fold_bytes / 1048576.0, L.fold);
    if (o.no_mask) printf("WARNING: --no-mask, so exception blocks past the causal limit are attended; TIMING ONLY, the partial is not foldable\n");

    // ---- OpenCL ----
    cl_platform_id plat = nullptr; cl_device_id cdev = nullptr; cl_uint n = 0; cl_int err = 0;
    CL_CHECK(clGetPlatformIDs(1, &plat, &n), "clGetPlatformIDs");
    CHECK(n > 0, "no OpenCL platform");
    CL_CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &cdev, &n), "clGetDeviceIDs");
    CHECK(n > 0, "no OpenCL GPU");
    char dname[256] = {0}; clGetDeviceInfo(cdev, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
    size_t ext_len = 0; clGetDeviceInfo(cdev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_len);
    std::string ext(ext_len, '\0'); clGetDeviceInfo(cdev, CL_DEVICE_EXTENSIONS, ext_len, &ext[0], nullptr);
    // The ION allocation type rides on cl_qcom_ext_host_ptr; the Adreno 830 driver (0800.70) does
    // NOT advertise a separate cl_qcom_ion_host_ptr string even though CL_MEM_ION_HOST_PTR_QCOM
    // works -- verified standalone, and ggml_hexagon_hetero_alias does not check either.
    CHECK(ext.find("cl_qcom_ext_host_ptr") != std::string::npos,
          "device has no cl_qcom_ext_host_ptr: the GPU cannot write the fold buffer in place");
    size_t max_img_buf = 0; clGetDeviceInfo(cdev, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, sizeof(max_img_buf), &max_img_buf, nullptr);

    cl_context ctx = clCreateContext(nullptr, 1, &cdev, nullptr, nullptr, &err); CL_CHECK(err, "clCreateContext");
    cl_command_queue cq = clCreateCommandQueueWithProperties(ctx, cdev, nullptr, &err); CL_CHECK(err, "clCreateCommandQueue");

    char opts[192];
    snprintf(opts, sizeof(opts), "-cl-std=CL2.0 -cl-fast-relaxed-math -DFA_D=%d%s", o.d, o.pack_q ? " -DPACK_Q" : "");
    size_t src_len = strlen(cl_src);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &cl_src, &src_len, &err); CL_CHECK(err, "clCreateProgramWithSource");
    if (clBuildProgram(prog, 1, &cdev, opts, nullptr, nullptr) != CL_SUCCESS) {
        size_t ln = 0; clGetProgramBuildInfo(prog, cdev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ln);
        std::string log(ln, '\0'); clGetProgramBuildInfo(prog, cdev, CL_PROGRAM_BUILD_LOG, ln, &log[0], nullptr);
        DIE("OpenCL build failed:\n%s", log.c_str());
    }
    cl_kernel k_stage = clCreateKernel(prog, "stage_k", &err); CL_CHECK(err, "kernel stage_k");
    cl_kernel k_fa    = clCreateKernel(prog, "fa_exc",  &err); CL_CHECK(err, "kernel fa_exc");
    cl_kernel k_packq = nullptr;
    if (o.pack_q) { k_packq = clCreateKernel(prog, "pack_q", &err); CL_CHECK(err, "kernel pack_q"); }

    cl_mem_ion_host_ptr ion = {};
    ion.ext_host_ptr.allocation_type  = CL_MEM_ION_HOST_PTR_QCOM;
    ion.ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
    ion.ion_filedesc = fd; ion.ion_hostptr = base;
    cl_mem alias = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM, L.total, &ion, &err);
    CL_CHECK(err, "ION alias of the rpcmem buffer");

    const size_t kt_bytes = (size_t) o.nkvh * D * n_stage * sizeof(uint16_t);
    cl_mem kt = clCreateBuffer(ctx, CL_MEM_READ_WRITE, kt_bytes, nullptr, &err); CL_CHECK(err, "staged K buffer");
    cl_mem qp = nullptr;
    if (o.pack_q) { qp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t) o.nh * o.lq * D * sizeof(uint16_t), nullptr, &err); CL_CHECK(err, "packed Q buffer"); }

    // Both images are RGBA half, so a texel is 8 B and every offset below is in texels.
    cl_image_format fmt = { CL_RGBA, CL_HALF_FLOAT };
    cl_image_desc dk; memset(&dk, 0, sizeof(dk));
    dk.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    dk.image_width = kt_bytes / 8; dk.buffer = kt;
    CHECK(dk.image_width <= max_img_buf, "staged K needs %zu texels, device max is %zu", dk.image_width, max_img_buf);
    cl_mem kt_img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &dk, nullptr, &err); CL_CHECK(err, "staged K image");
    dk.image_width = L.total / 8; dk.buffer = alias;
    CHECK(dk.image_width <= max_img_buf, "V image needs %zu texels over the shared buffer, device max is %zu", dk.image_width, max_img_buf);
    cl_mem v_img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &dk, nullptr, &err); CL_CHECK(err, "V image");

    printf("gpu    %s | image max buffer %zu texels | opts '%s'\n", dname, max_img_buf, opts);

    // ---- kernel args ----
    int ai = 0;
    auto sa = [&](cl_kernel kk, int idx, size_t sz, const void * p) { CL_CHECK(clSetKernelArg(kk, idx, sz, p), "clSetKernelArg"); };
    const int32_t k_off = (int32_t) L.k, k_nb1 = (int32_t) L.kv_nb1, k_nb2 = (int32_t) L.kv_nb2;
    const int32_t abs_off = (int32_t) L.abs, cnt_off = (int32_t) L.cnt, err_off = (int32_t) L.err;
    const int32_t abs_stride = n_stage_blk;
    ai = 0;
    sa(k_stage, ai++, sizeof(alias), &alias); sa(k_stage, ai++, sizeof(kt), &kt);
    sa(k_stage, ai++, 4, &k_off); sa(k_stage, ai++, 4, &k_nb1); sa(k_stage, ai++, 4, &k_nb2);
    sa(k_stage, ai++, 4, &abs_off); sa(k_stage, ai++, 4, &cnt_off);
    sa(k_stage, ai++, 4, &abs_stride); sa(k_stage, ai++, 4, &n_stage);

    const int32_t q_off = (int32_t) L.q, q_nb1 = (int32_t) L.q_nb1, q_nb2 = (int32_t) L.q_nb2;
    const int32_t mask_off = (int32_t) L.mask, mask_nb1 = (int32_t) L.mask_nb1;
    const int32_t use_mask = o.no_mask ? 0 : 1;
    const int32_t idx_off = (int32_t) L.idx;
    const int32_t lcnt_off = (int32_t) L.lcnt;
    const int32_t f_m = (int32_t) (L.fold + L.off_m), f_l = (int32_t) (L.fold + L.off_l), f_acc = (int32_t) (L.fold + L.off_acc);
    const int32_t v_base_tex = (int32_t) (L.v / 8), v_tok_tex = (int32_t) (L.kv_nb1 / 8), v_head_tex = (int32_t) (L.kv_nb2 / 8);
    cl_mem qp_arg = o.pack_q ? qp : kt;   // unused when PACK_Q is off, but a live cl_mem is still required
    ai = 0;
    sa(k_fa, ai++, sizeof(kt_img), &kt_img); sa(k_fa, ai++, sizeof(v_img), &v_img);
    sa(k_fa, ai++, sizeof(alias), &alias);   sa(k_fa, ai++, sizeof(qp_arg), &qp_arg);
    sa(k_fa, ai++, 4, &q_off); sa(k_fa, ai++, 4, &q_nb1); sa(k_fa, ai++, 4, &q_nb2);
    sa(k_fa, ai++, 4, &mask_off); sa(k_fa, ai++, 4, &mask_nb1); sa(k_fa, ai++, 4, &use_mask);
    sa(k_fa, ai++, 4, &idx_off); sa(k_fa, ai++, 4, &abs_off); sa(k_fa, ai++, 4, &cnt_off); sa(k_fa, ai++, 4, &lcnt_off); sa(k_fa, ai++, 4, &err_off);
    sa(k_fa, ai++, 4, &f_m); sa(k_fa, ai++, 4, &f_l); sa(k_fa, ai++, 4, &f_acc);
    sa(k_fa, ai++, 4, &o.lq); sa(k_fa, ai++, 4, &G); sa(k_fa, ai++, 4, &num_sb); sa(k_fa, ai++, 4, &qb_per_sb); sa(k_fa, ai++, 4, &top_k);
    sa(k_fa, ai++, 4, &abs_stride); sa(k_fa, ai++, 4, &n_stage);
    sa(k_fa, ai++, 4, &v_base_tex); sa(k_fa, ai++, 4, &v_tok_tex); sa(k_fa, ai++, 4, &v_head_tex);
    sa(k_fa, ai++, 4, &scale);
    if (o.pack_q) {
        ai = 0;
        sa(k_packq, ai++, sizeof(alias), &alias); sa(k_packq, ai++, sizeof(qp), &qp);
        sa(k_packq, ai++, 4, &q_off); sa(k_packq, ai++, 4, &q_nb1); sa(k_packq, ai++, 4, &q_nb2); sa(k_packq, ai++, 4, &o.lq);
    }

    const size_t g_stage[3] = { (size_t) n_stage, D / 8, (size_t) o.nkvh };
    const size_t g_packq[3] = { (size_t) o.lq, D / 4, (size_t) o.nh };
    const size_t g_fa[3]    = { 128, (size_t) num_qb, (size_t) o.nh };
    const size_t l_fa[3]    = { 128, 1, 1 };

    // ---- stage, then prove the transpose moved the right bytes ----
    CL_CHECK(clEnqueueNDRangeKernel(cq, k_stage, 3, nullptr, g_stage, nullptr, 0, nullptr, nullptr), "stage_k");
    if (o.pack_q) CL_CHECK(clEnqueueNDRangeKernel(cq, k_packq, 3, nullptr, g_packq, nullptr, 0, nullptr, nullptr), "pack_q");
    CL_CHECK(clFinish(cq), "clFinish stage");
    {
        std::vector<uint16_t> kt_host(kt_bytes / 2);
        CL_CHECK(clEnqueueReadBuffer(cq, kt, CL_TRUE, 0, kt_bytes, kt_host.data(), 0, nullptr, nullptr), "read staged K");
        size_t bad = 0;
        for (int kvh = 0; kvh < o.nkvh; kvh++) {
            for (size_t c = 0; c < stage_abs[kvh].size(); c++) {
                for (int i = 0; i < o.bs; i++) {
                    const int tok = stage_abs[kvh][c] * o.bs + i;
                    const uint16_t * src = (const uint16_t *) (base + L.k + (size_t) tok * L.kv_nb1 + (size_t) kvh * L.kv_nb2);
                    const size_t ki = c * o.bs + i;
                    for (size_t e = 0; e < D; e++) {
                        if (kt_host[(size_t) kvh * D * n_stage + e * n_stage + ki] != src[e]) bad++;
                    }
                }
            }
        }
        CHECK(bad == 0, "staged K differs from the source in %zu of %zu halves: the transpose is wrong", bad, n_stage_used * D);
    }

    // ---- poison the fold area, so a row the kernel never writes cannot pass as a plausible zero ----
    const uint32_t poison = 0x7fc0dead;
    {
        uint32_t * w = (uint32_t *) (base + L.fold + L.off_m);
        const size_t nw = (fold_bytes - L.off_m) / 4;
        for (size_t i = 0; i < nw; i++) w[i] = poison;
        memset(base + L.err, 0, 256);
    }

    CL_CHECK(clEnqueueNDRangeKernel(cq, k_fa, 3, nullptr, g_fa, l_fa, 0, nullptr, nullptr), "fa_exc");
    CL_CHECK(clFinish(cq), "clFinish fa");

    const int32_t dev_err = *(const int32_t *) (base + L.err);
    CHECK(dev_err == 0, "kernel reported %d out-of-range block ids", dev_err);
    {
        const float * fm = (const float *) (base + L.fold + L.off_m);
        const float * fl = (const float *) (base + L.fold + L.off_l);
        const float * fa = (const float *) (base + L.fold + L.off_acc);
        size_t left = 0;
        for (size_t r = 0; r < rows; r++) {
            if (((const uint32_t *) fm)[r] == poison) left++;
            if (((const uint32_t *) fl)[r] == poison) left++;
            for (size_t e = 0; e < D; e++) if (((const uint32_t *) fa)[r * D + e] == poison) left++;
        }
        CHECK(left == 0, "%zu fold words were never written by the kernel", left);
    }

    // ---- timing ----
    double ms_stage = 0, ms_fa = 0, ms_packq = 0;
    {
        double t0 = now_ms();
        for (int i = 0; i < o.iters; i++) CL_CHECK(clEnqueueNDRangeKernel(cq, k_stage, 3, nullptr, g_stage, nullptr, 0, nullptr, nullptr), "stage_k");
        CL_CHECK(clFinish(cq), "clFinish stage");
        ms_stage = (now_ms() - t0) / o.iters;
        if (o.pack_q) {
            t0 = now_ms();
            for (int i = 0; i < o.iters; i++) CL_CHECK(clEnqueueNDRangeKernel(cq, k_packq, 3, nullptr, g_packq, nullptr, 0, nullptr, nullptr), "pack_q");
            CL_CHECK(clFinish(cq), "clFinish pack_q");
            ms_packq = (now_ms() - t0) / o.iters;
        }
        t0 = now_ms();
        for (int i = 0; i < o.iters; i++) CL_CHECK(clEnqueueNDRangeKernel(cq, k_fa, 3, nullptr, g_fa, l_fa, 0, nullptr, nullptr), "fa_exc");
        CL_CHECK(clFinish(cq), "clFinish fa");
        ms_fa = (now_ms() - t0) / o.iters;
    }

    // ---- host reference in double over the exception blocks only ----
    const float * fm = (const float *) (base + L.fold + L.off_m);
    const float * fl = (const float *) (base + L.fold + L.off_l);
    const float * fa = (const float *) (base + L.fold + L.off_acc);
    size_t n_owned = 0, n_empty = 0;
    if (o.check) {
        cmp c_m, c_l, c_acc, c_out;
        std::vector<double> accd(D);
        for (int h = 0; h < o.nh; h++) {
            const int kvh = h / G;
            for (int tk = 0; tk < o.lq; tk++) {
                const size_t r = (size_t) h * o.lq + tk;
                const std::vector<int> & ex = exc[(size_t) kvh * num_sb + tk / o.bs];
                const float * qr = q + ((size_t) h * o.lq + tk) * D;
                const uint16_t * mrow = mask + (size_t) tk * o.kv;
                double m_p = -INFINITY, l_p = 0;
                std::fill(accd.begin(), accd.end(), 0.0);
                std::vector<double> s;
                s.reserve(ex.size() * o.bs);
                for (int b : ex) {
                    for (int i = 0; i < o.bs; i++) {
                        const int key = b * o.bs + i;
                        const uint16_t * kr = (const uint16_t *) (base + L.k + (size_t) key * L.kv_nb1 + (size_t) kvh * L.kv_nb2);
                        double dot = 0;
                        for (size_t e = 0; e < D; e++) dot += (double) qr[e] * h2f(kr[e]);
                        const double sv = dot * scale + (double) h2f(mrow[key]);
                        s.push_back(sv);
                        if (sv > m_p) m_p = sv;
                    }
                }
                if (std::isfinite(m_p)) {
                    size_t si = 0;
                    for (int b : ex) {
                        for (int i = 0; i < o.bs; i++, si++) {
                            if (!std::isfinite(s[si])) continue;
                            const double p = exp(s[si] - m_p);
                            l_p += p;
                            const int key = b * o.bs + i;
                            const uint16_t * vr = (const uint16_t *) (base + L.v + (size_t) key * L.kv_nb1 + (size_t) kvh * L.kv_nb2);
                            for (size_t e = 0; e < D; e++) accd[e] += p * h2f(vr[e]);
                        }
                    }
                }
                if (l_p > 0) {
                    n_owned++;
                    CHECK(fl[r] > 0.0f, "row %zu (head %d token %d) owns %zu exception blocks but the GPU left it empty", r, h, tk, ex.size());
                    CHECK(std::isfinite(fm[r]) && std::isfinite(fl[r]), "row %zu has a non-finite m or l", r);
                    c_m.add(fm[r], m_p);
                    c_l.add(fl[r], l_p);
                    for (size_t e = 0; e < D; e++) {
                        CHECK(std::isfinite(fa[r * D + e]), "row %zu acc[%zu] is not finite", r, e);
                        c_acc.add(fa[r * D + e], accd[e]);
                        c_out.add(fa[r * D + e] / fl[r], accd[e] / l_p);
                    }
                } else {
                    n_empty++;
                    CHECK(fm[r] == HTP_FA_M_INITIAL_VAL, "row %zu is unowned but m is %g, not the empty sentinel %g", r, (double) fm[r], (double) HTP_FA_M_INITIAL_VAL);
                    CHECK(fl[r] == 0.0f, "row %zu is unowned but l is %g", r, (double) fl[r]);
                    for (size_t e = 0; e < D; e++) CHECK(fa[r * D + e] == 0.0f, "row %zu is unowned but acc[%zu] is %g", r, e, (double) fa[r * D + e]);
                }
            }
        }
        printf("check  m    max_abs %.3e  nmse %.3e\n", c_m.max_abs, c_m.nmse());
        printf("check  l    max_abs %.3e  nmse %.3e\n", c_l.max_abs, c_l.nmse());
        printf("check  acc  max_abs %.3e  nmse %.3e\n", c_acc.max_abs, c_acc.nmse());
        printf("check  o    max_abs %.3e  nmse %.3e   (acc/l, for scale only; the fold reads the triple)\n", c_out.max_abs, c_out.nmse());
        printf("rows   %zu owned, %zu empty (of %zu)%s\n", n_owned, n_empty, rows,
               n_empty == 0 ? "  WARNING: the empty-partial path was not exercised, pass --empty-mod 5" : "");
    } else {
        printf("check  skipped (--no-check)\n");
    }

    // one unit = 1 KV head x 1 64-row sub-block x 1 64-key block, and covers all G query heads
    const double units = (double) n_exc_pairs;
    const double flops = 4.0 * (double) G * (double) o.bs * (double) o.bs * (double) D * (double) n_exc_pairs;
    printf("time   fa %8.3f ms  %7.1f GF/s  %6.3f us/unit | stage %7.3f ms (%.1f%% of fa)%s\n",
           ms_fa, flops / (ms_fa * 1e-3) / 1e9, ms_fa * 1e3 / units, ms_stage, 100.0 * ms_stage / ms_fa,
           o.pack_q ? "" : "  [Q read in place]");
    if (o.pack_q) printf("time   pack_q %.3f ms\n", ms_packq);

    // header must survive the run: the HMX side ignores the fold unless it matches exactly
    const struct htp_fa_fold_hdr * hb = (const struct htp_fa_fold_hdr *) (base + L.fold);
    CHECK(hb->magic == HTP_FA_FOLD_MAGIC && hb->rows == rows && hb->neq1 == (uint32_t) o.lq && hb->dv == (uint32_t) D,
          "the fold header was corrupted by the run");

    if (o.verbose) {
        // the anchor the HMX side rebuilds is m * log2(e) + scale; print all three so a future
        // mismatch is read off directly instead of being hunted through nmse
        for (size_t r = 0; r < std::min<size_t>(rows, 4); r++)
            printf("  [row %zu] m %.6f (x log2e %.6f, +scale %.6f) l %.6f acc0 %.6f\n", r, (double) fm[r],
                   (double) fm[r] * 1.44269504, (double) fm[r] * 1.44269504 + scale, (double) fl[r], (double) fa[r * D]);
    }

    clReleaseMemObject(v_img); clReleaseMemObject(kt_img); clReleaseMemObject(kt);
    if (qp) clReleaseMemObject(qp);
    clReleaseMemObject(alias);
    if (k_packq) clReleaseKernel(k_packq);
    clReleaseKernel(k_fa); clReleaseKernel(k_stage); clReleaseProgram(prog);
    clReleaseCommandQueue(cq); clReleaseContext(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    return 0;
}

// fa-hetero-live: the REAL heterogeneous prefill handshake, both engines in one process.
//
// THE QUESTION THIS TOOL EXISTS TO ANSWER: does the GPU's work hide under the NPU's, or does the
// HTP block on it? hmx_flash_attn_ext publishes `ready` at op entry but takes its ONE wait at the
// FIRST (sequence, query block, KV head) tile's fold, so a single end-of-op handshake can leave
// the HTP idle for most of the GPU's runtime. Three arms over the SAME data answer it:
//
//   A npu-only  mask = shared set only, no fold                    -- the baseline shape
//   B serial    GPU kernel to completion, THEN the NPU op with the partial already resident
//               (flags = 0), so GPU time and NPU time are separate numbers
//   C live      HTP_FA_FOLD_F_LIVE: the NPU op is launched, a host thread waits for the DSP to
//               raise `ready`, runs the GPU kernel, writes `done`. The real pipeline.
//   D staged    HTP_FA_FOLD_F_STAGED: the GPU runs one launch per KV head in ascending order and
//               the relay writes done[k] as head k completes; the NPU runs its exception-free
//               tiles first (plain path, no fold, no wait), then the exception tiles in the same
//               head order, waiting once per head. --exc-heads N leaves heads >= N without
//               exceptions so that reordering has something to reorder.
//
// C ~= max(A, B_gpu) means the overlap is real. C ~= A + B_gpu means it is not. The DSP also
// records how long it idled (wait_us / wait_us_max in the status region), which separates "the
// GPU was slow" from "the HTP blocked".
//
// By default the NPU arms run the REAL sparse kernel (src[5] block lists + src[6] per-row counts):
// arm A attends to the union of pool and exceptions per (tile, KV head) -- the fused-query-block
// baseline the split is judged against -- and the fold arms to the pool alone. --dense-npu
// restores the older mask-only arms, where the HMX kernel walks every KV block and the shadow the
// GPU can hide in is several times longer than a sparse NPU's: the optimistic case.
//
// The two halves come from examples/fa-fold-check (the HTP op and the fold buffer) and
// examples/fa-exc-gpu (the exception kernel and its ION alias), merged into one allocation.

#include "ggml.h"
#include "ggml-backend.h"
#include "htp-ops.h"
#include "flash-attn-ops.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sched.h>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

extern "C" int rpcmem_to_fd(void * po);

static double now_ms() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

#define DIE(...)      do { fprintf(stderr, "fa-hetero-live: " __VA_ARGS__); fputc('\n', stderr); exit(1); } while (0)
#define CHECK(c, ...) do { if (!(c)) DIE(__VA_ARGS__); } while (0)
#define CL_CHECK(e, what) do { cl_int _e = (e); if (_e != CL_SUCCESS) DIE("%s failed (%d)", (what), _e); } while (0)

// The kernel's lane mapping is 8x16 for QK and 16x8 for PV, which only adds up to 128 lanes at D=128.
#define FA_D_FIXED 128
#define FA2_BQ     32                        // rows per GPU work-group (two per 64-row sub-block)
#define FA2_BK     64
#define FA_TILE_BQ 256                       // query tile the split is defined on
#define FA_TILE_R  (FA_TILE_BQ / FA2_BK)     // 64-row sub-blocks per tile

// aarch64 EL0 cache maintenance, as in examples/hetero-decode-attn. The control words are polled
// and written by this CPU while the DSP writes and polls them from the other side.
static inline uint64_t cnt_now() {
    uint64_t v;
    asm volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(v) : : "memory");
    return v;
}
static inline void dc_civac(const void * p, size_t n) {
    uintptr_t a = (uintptr_t) p & ~(uintptr_t) 63; const uintptr_t e = (uintptr_t) p + n;
    for (; a < e; a += 64) asm volatile("dc civac, %0" : : "r"(a) : "memory");
    asm volatile("dsb sy" : : : "memory");
}
static inline void dc_cvac(const void * p, size_t n) {
    uintptr_t a = (uintptr_t) p & ~(uintptr_t) 63; const uintptr_t e = (uintptr_t) p + n;
    for (; a < e; a += 64) asm volatile("dc cvac, %0" : : "r"(a) : "memory");
    asm volatile("dsb sy" : : : "memory");
}

// ---------------------------------------------------------------------------------------

struct options {
    int  d = 128, nh = 16, nkvh = 8, lq = 1024, kv = 4096;
    int  topk = 8;                   // blocks each 64-row sub-block selects (shared pool + exceptions)
    int  exc = 2;                    // of those, how many are exceptions (go to the GPU)
    int  cstar = 2;                  // blocks wanted by >= cstar sub-blocks stay on the NPU
    int  iters = 20;
    int  cpu = -1;                   // pin the relay thread to this core (-1 = leave it to the scheduler)
    int  exc_heads = -1;             // KV heads [0, exc_heads) carry exceptions; the rest select the pool alone (-1 = all)
    bool sparse = true;              // NPU arms use src[5]/src[6] (false: mask-only, dense walk)
    bool instore = false;            // fold in the store threads (HTP_FA_FOLD_F_INSTORE) in every fold arm
    int  head_start = 0;             // the first K tiles of the staged order keep their exceptions on the NPU (union, plain path)
    int  hpl = 1;                    // KV heads per GPU launch in arm D (coarser staging, fuller GPU)
    bool stage_qb = false;           // arm D stages by 256-row query block (all heads per launch) instead of by KV head
    bool check = true;
    bool verbose = false;
};

static void usage() {
    printf("usage: llama-fa-hetero-live [options]\n"
           "  --d N       head dim (must be %d, the GPU kernel's lane mapping needs it)\n"
           "  --nh N      query heads             (default 16)\n"
           "  --nkvh N    KV heads                (default 8)\n"
           "  --lq N      query tokens            (default 1024, multiple of %d)\n"
           "  --kv N      KV length               (default 4096, multiple of %d, >= lq)\n"
           "  --exc N     exception blocks per 64-row sub-block -> the GPU (default 2)\n"
           "  --cstar N   NPU keeps blocks wanted by >= cstar of the %d sub-blocks (default 2)\n"
           "  --topk N    blocks selected per sub-block, shared pool is topk - exc (default 8)\n"
           "  --iters N   timed iterations per arm (default 20)\n"
           "  --cpu N     pin the relay thread to core N (default: no pinning)\n"
           "  --exc-heads N  only KV heads < N have exceptions; the rest have none (default: all)\n"
           "  --dense-npu the NPU arms express the split through the mask only (no src[5]; the old shape)\n"
           "  --instore   fold in the store threads (no diagonal build, no norm pass) in arms B, C, D\n"
           "  --stage-qb  arm D: GPU launches per 256-row query block (all heads), NPU keeps its default order\n"
           "  --head-start K  the first K tiles (KV head 0 first) run the UNION on the NPU with no GPU work,\n"
           "              a shadow for the GPU launch; arm D then never blocks if K tiles outlast it\n"
           "  --no-check  skip the double-precision host reference (timing only)\n"
           "  -v          verbose\n", FA_D_FIXED, FA_TILE_BQ, FA2_BK, FA_TILE_R);
}

// ---------------------------------------------------------------------------------------
// The GPU half: examples/fa-exc-gpu's kernels with Q read in place, staged K moved into the
// shared rpcmem buffer (one image serves both the staged K and V), and the pack-Q arm dropped.
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
// consecutive KEYS at one d, which the K tensor's layout (D contiguous per (kv_head, token))
// cannot serve. One work item moves 8 dims of one key, so the source row is read whole.
__kernel void stage_k(__global uchar * base,
                      const int k_off, const int k_nb1, const int k_nb2, const int kt_off,
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
    __global half * dst = (__global half *)(base + kt_off) + (long) kvh * FA_D * n_stage + (long) (dg * 8) * n_stage + ki;
    const long st = n_stage;
    dst[0]      = kr.s0; dst[st]     = kr.s1; dst[st * 2] = kr.s2; dst[st * 3] = kr.s3;
    dst[st * 4] = kr.s4; dst[st * 5] = kr.s5; dst[st * 6] = kr.s6; dst[st * 7] = kr.s7;
}

// One work-group = 32 query rows of one head over that sub-block's exception blocks. Emits the
// UNNORMALISED (m, l, acc) per row; a row without exceptions leaves the empty partial.
__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void fa_exc(__read_only image1d_buffer_t buf_img,
            __global uchar * base,
            const int q_off, const int q_nb1, const int q_nb2,
            const int mask_off, const int mask_nb1,
            const int idx_off, const int abs_off, const int cnt_off, const int lcnt_off, const int err_off,
            const int f_m, const int f_l, const int f_acc,
            const int Sq, const int G, const int num_sb, const int qb_per_sb, const int top_k,
            const int abs_stride, const int n_stage,
            const int kt_base_tex, const int v_base_tex, const int v_tok_tex, const int v_head_tex,
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
    const int kt_head = kt_base_tex + kvh * FA_D * M_4kv;
    const int v_head  = v_base_tex + kvh * v_head_tex;
    __global const float * Qh = (__global const float *)(base + q_off + (long) bh * q_nb2);
    const int q_row_f = q_nb1 >> 2;
    __global const half * Mk = (__global const half *)(base + mask_off);
    const int mask_row_h = mask_nb1 >> 1;

    const int n_list = (cnt_row < top_k) ? cnt_row : top_k;
    for (int s = 0; s < n_list; ++s) {
        const int cb = idx_row[s];
        if (cb < 0) break;
        if (cb >= n_blk_kvh) {
            if (t == 0) atomic_inc((volatile __global int *)(base + err_off));
            break;
        }
        const int kt0 = cb * FA2_BK;
        const int vk0 = abs_tab[kvh * abs_stride + cb] * FA2_BK;
        {
            const int mt = t % (FA2_BK / 8); const int nt = t / (FA2_BK / 8);
            // rows past Sq (a ragged last ubatch) read a clamped row and are never written
            const int qr0 = min(q0 + nt * 2,     Sq - 1);
            const int qr1 = min(q0 + nt * 2 + 1, Sq - 1);
            const int kbase = kt_head + (kt0 >> 2) + mt * 2;
            float8 c0 = (float8)0, c1 = (float8)0;
            for (int i = 0; i < FA_D; i += 4) {
                const int tb = kbase + i * M_4kv;
                half8 B0, B1, B2, B3;
                B0.s0123 = read_imageh(buf_img, tb);              B0.s4567 = read_imageh(buf_img, tb + 1);
                B1.s0123 = read_imageh(buf_img, tb + M_4kv);      B1.s4567 = read_imageh(buf_img, tb + M_4kv + 1);
                B2.s0123 = read_imageh(buf_img, tb + 2 * M_4kv);  B2.s4567 = read_imageh(buf_img, tb + 2 * M_4kv + 1);
                B3.s0123 = read_imageh(buf_img, tb + 3 * M_4kv);  B3.s4567 = read_imageh(buf_img, tb + 3 * M_4kv + 1);
                const float8 b0 = convert_float8(B0), b1 = convert_float8(B1), b2 = convert_float8(B2), b3 = convert_float8(B3);
                const float4 wa = vload4(0, Qh + (long) qr0 * q_row_f + i);
                const float4 wb = vload4(0, Qh + (long) qr1 * q_row_f + i);
                c0 += b0*wa.s0; c0 += b1*wa.s1; c0 += b2*wa.s2; c0 += b3*wa.s3;
                c1 += b0*wb.s0; c1 += b1*wb.s1; c1 += b2*wb.s2; c1 += b3*wb.s3;
            }
            const int kb = vk0 + mt * 8;
            #define EMITS(NN, CV, QR) { float8 v = (CV) * scale; \
                v += convert_float8(vload8(0, Mk + (long) (QR) * mask_row_h + kb)); \
                vstore8(v, 0, &S_lds[nt * 2 + (NN)][mt * 8]); }
            EMITS(0, c0, qr0); EMITS(1, c1, qr1);
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
            V0.s0123 = read_imageh(buf_img, vt);                   V0.s4567 = read_imageh(buf_img, vt + 1);
            V1.s0123 = read_imageh(buf_img, vt + v_tok_tex);       V1.s4567 = read_imageh(buf_img, vt + v_tok_tex + 1);
            V2.s0123 = read_imageh(buf_img, vt + 2 * v_tok_tex);   V2.s4567 = read_imageh(buf_img, vt + 2 * v_tok_tex + 1);
            V3.s0123 = read_imageh(buf_img, vt + 3 * v_tok_tex);   V3.s4567 = read_imageh(buf_img, vt + 3 * v_tok_tex + 1);
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

    barrier(CLK_LOCAL_MEM_FENCE);
    const long r0 = (long) bh * Sq + q0;
    __global float * Fm = (__global float *)(base + f_m);
    __global float * Fl = (__global float *)(base + f_l);
    __global float * Fa = (__global float *)(base + f_acc);
    if (t < FA2_BQ && q0 + t < Sq) {
        const float l = l_run[t];
        Fm[r0 + t] = (l > 0.0f) ? m_run[t] : M_EMPTY;
        Fl[r0 + t] = l;
    }
    #define OWRITE(NN, OV) { const int q = pv_nt * 4 + (NN); const float l = l_run[q]; \
        if (q0 + q < Sq) vstore8((l > 0.0f) ? (OV) : (float8)0, 0, Fa + (r0 + q) * FA_D + pv_d0); }
    OWRITE(0, o0); OWRITE(1, o1); OWRITE(2, o2); OWRITE(3, o3);
    #undef OWRITE
}
)CL";

// ---------------------------------------------------------------------------------------

struct layout {
    size_t q = 0, k = 0, v = 0, kt = 0, img_end = 0;
    size_t mask = 0, gmask = 0, dst = 0, fold = 0, idx = 0, abs = 0, cnt = 0, lcnt = 0, err = 0, total = 0;
    size_t off_m = 0, off_l = 0, off_acc = 0, off_ctl = 0, off_exc = 0, fold_bytes = 0;
    size_t sel_a = 0, cnt_a = 0, sel_s = 0, cnt_s = 0;   // src[5]/src[6]: union (arm A) and pool (fold arms)
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
    void merge(const cmp & o) { max_abs = std::max(max_abs, o.max_abs); sum2 += o.sum2; ref2 += o.ref2; }
    double nmse() const { return sum2 / (ref2 > 0 ? ref2 : 1.0); }
};

static ggml_tensor * place(ggml_backend_buffer_t buf, uint8_t * base, size_t off, ggml_tensor * t, const char * name) {
    t->buffer = buf;
    t->data   = base + off;
    ggml_set_name(t, name);
    ggml_backend_buffer_init_tensor(buf, t);
    return t;
}

int main(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](int & dst) { CHECK(i + 1 < argc, "%s needs a value", a.c_str()); dst = atoi(argv[++i]); };
        if      (a == "--d")        next(o.d);
        else if (a == "--nh")       next(o.nh);
        else if (a == "--nkvh")     next(o.nkvh);
        else if (a == "--lq")       next(o.lq);
        else if (a == "--kv")       next(o.kv);
        else if (a == "--exc")      next(o.exc);
        else if (a == "--cstar")    next(o.cstar);
        else if (a == "--topk")     next(o.topk);
        else if (a == "--iters")    next(o.iters);
        else if (a == "--cpu")      next(o.cpu);
        else if (a == "--exc-heads") next(o.exc_heads);
        else if (a == "--dense-npu") o.sparse = false;
        else if (a == "--instore")   o.instore = true;
        else if (a == "--head-start") next(o.head_start);
        else if (a == "--hpl")       next(o.hpl);
        else if (a == "--stage-qb")  o.stage_qb = true;
        else if (a == "--no-check") o.check = false;
        else if (a == "-v")         o.verbose = true;
        else { usage(); return 1; }
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    // The handshake pairs ONE host relay with ONE op in flight. An async backend would let the
    // next op publish `ready` before this one's `done` is written.
    setenv("GGML_HEXAGON_ASYNC", "0", 1);

    // ---- geometry neither half can bend ----
    CHECK(o.d == FA_D_FIXED, "--d must be %d: the GPU kernel maps 128 lanes as 8x16 (QK) and 16x8 (PV)", FA_D_FIXED);
    CHECK(o.nh > 0 && o.nkvh > 0 && o.nh % o.nkvh == 0, "--nh must be a multiple of --nkvh");
    CHECK(o.lq > 0 && o.lq % FA_TILE_BQ == 0, "--lq must be a multiple of %d", FA_TILE_BQ);
    CHECK(o.kv >= o.lq && o.kv % FA2_BK == 0, "--kv must be >= --lq and a multiple of %d", FA2_BK);
    CHECK(o.exc > 0 && o.exc < o.topk, "--exc must be in (0, --topk): with no exceptions there is nothing for the GPU to do");
    CHECK(o.cstar >= 2 && o.cstar <= FA_TILE_R, "--cstar must be in [2, %d]: below 2 nothing is an exception, above %d nothing is shared", FA_TILE_R, FA_TILE_R);
    CHECK(o.iters > 0, "--iters must be > 0");
    if (o.exc_heads < 0) o.exc_heads = o.nkvh;
    CHECK(o.exc_heads >= 1 && o.exc_heads <= o.nkvh, "--exc-heads must be in [1, --nkvh]");
    CHECK(o.head_start >= 0, "--head-start must be >= 0");
    CHECK(o.hpl >= 1 && o.nkvh % o.hpl == 0, "--hpl must divide --nkvh");

    const int    G         = o.nh / o.nkvh;
    const int    num_sb    = o.lq / FA2_BK;      // 64-row sub-blocks
    const int    n_tiles   = o.lq / FA_TILE_BQ;
    const int    num_qb    = o.lq / FA2_BQ;      // GPU kernel query blocks
    const int    qb_per_sb = FA2_BK / FA2_BQ;
    const int    nblk      = o.kv / FA2_BK;
    const int    mask_rows = GGML_PAD(o.lq, 64);
    const float  scale     = 1.0f / sqrtf((float) o.d);
    const size_t D         = (size_t) o.d;
    const size_t rows      = (size_t) o.lq * o.nh;

    // ---- the split, built before the layout: the staged-K region sizes itself from it ----
    //
    // The shared pool is per TILE and common to every KV head: the NPU's share rides on the mask,
    // and a ggml mask has no head dimension. The exceptions stay per (KV head, sub-block), which
    // is the granularity flash-attn-ops.c reads src[5] at.
    std::vector<std::vector<int>> pool((size_t) n_tiles);
    std::vector<std::vector<int>> exc((size_t) o.nkvh * num_sb);
    std::vector<std::vector<int>> sel((size_t) o.nkvh * num_sb);
    {
        const int n_pool = o.topk - o.exc;
        std::mt19937 rng(9876);
        for (int tl = 0; tl < n_tiles; tl++) {
            // every row of the tile must see every pool block, so bound by the tile's FIRST row
            const int lim0 = ((o.kv - o.lq) + tl * FA_TILE_BQ) / FA2_BK;
            CHECK(lim0 + 1 >= n_pool + FA_TILE_R * o.exc,
                  "tile %d has only %d causally allowed blocks but the split needs %d (pool %d + %d x exc %d); raise --kv or lower --topk/--exc",
                  tl, lim0 + 1, n_pool + FA_TILE_R * o.exc, n_pool, FA_TILE_R, o.exc);
            std::vector<int> perm(lim0 + 1);
            for (int b = 0; b <= lim0; b++) perm[b] = b;
            std::shuffle(perm.begin(), perm.end(), rng);
            std::vector<char> in_pool((size_t) nblk, 0);
            std::vector<int> & pl = pool[tl];
            for (int i = 0; i < n_pool; i++) { pl.push_back(perm[i]); in_pool[perm[i]] = 1; }
            std::sort(pl.begin(), pl.end());
            for (int kvh = 0; kvh < o.nkvh; kvh++) {
                std::shuffle(perm.begin(), perm.end(), rng);
                std::vector<char> used = in_pool;
                size_t take = 0;
                for (int j = 0; j < FA_TILE_R; j++) {
                    const int sb = tl * FA_TILE_R + j;
                    std::vector<int> & ex = exc[(size_t) kvh * num_sb + sb];
                    if (kvh >= o.exc_heads) {
                        // no exceptions on this head: its sub-blocks select exactly the pool
                        sel[(size_t) kvh * num_sb + sb] = pl;
                        continue;
                    }
                    // the sub-block's own diagonal block first: it is partly masked, so the mask
                    // path is exercised instead of only fully visible blocks
                    const int diag = ((o.kv - o.lq) + tl * FA_TILE_BQ + j * FA2_BK) / FA2_BK;
                    if (!used[diag]) { ex.push_back(diag); used[diag] = 1; }
                    while ((int) ex.size() < o.exc) {
                        CHECK(take < perm.size(), "ran out of distinct blocks for tile %d kvh %d", tl, kvh);
                        const int b = perm[take++];
                        if (!used[b]) { ex.push_back(b); used[b] = 1; }
                    }
                    std::sort(ex.begin(), ex.end());
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
    // sets built above, and the shared pool must be disjoint from every sub-block's exceptions.
    size_t n_exc_pairs = 0;
    for (int kvh = 0; kvh < o.nkvh; kvh++) {
        for (int tl = 0; tl < n_tiles; tl++) {
            std::vector<int> cnt_b((size_t) nblk, 0);
            for (int j = 0; j < FA_TILE_R; j++) for (int b : sel[(size_t) kvh * num_sb + tl * FA_TILE_R + j]) cnt_b[b]++;
            std::vector<int> derived_shared;
            for (int b = 0; b < nblk; b++) if (cnt_b[b] >= o.cstar) derived_shared.push_back(b);
            CHECK(derived_shared == pool[tl], "kvh %d tile %d: the c(b) >= %d rule does not reproduce the shared list", kvh, tl, o.cstar);
            for (int j = 0; j < FA_TILE_R; j++) {
                const std::vector<int> & ex = exc[(size_t) kvh * num_sb + tl * FA_TILE_R + j];
                std::vector<int> derived_exc;
                for (int b : sel[(size_t) kvh * num_sb + tl * FA_TILE_R + j]) if (cnt_b[b] < o.cstar) derived_exc.push_back(b);
                CHECK(derived_exc == ex, "kvh %d tile %d sub-block %d: derived exceptions differ from the built list", kvh, tl, j);
                for (int b : ex) CHECK(!std::binary_search(pool[tl].begin(), pool[tl].end(), b),
                                       "kvh %d tile %d sub-block %d: block %d is on BOTH engines; the LSE merge would count it twice", kvh, tl, j, b);
                n_exc_pairs += ex.size();
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
    int    n_stage_blk = 1;
    size_t n_stage_used = 0;
    for (int kvh = 0; kvh < o.nkvh; kvh++) { n_stage_blk = std::max(n_stage_blk, (int) stage_abs[kvh].size()); n_stage_used += stage_abs[kvh].size(); }
    const int n_stage = n_stage_blk * FA2_BK;
    CHECK(n_stage % 4 == 0, "staged key capacity must be a whole texel");

    // ---- buffer layout: ONE rpcmem allocation for both engines ----
    //
    // K, V and the staged K sit first so the image the GPU reads them through covers only the
    // head of the buffer, well inside CL_DEVICE_IMAGE_MAX_BUFFER_SIZE.
    layout L;
    L.q_nb1    = D * sizeof(float);                       // Q:    [d, lq, nh]  contiguous f32
    L.q_nb2    = (size_t) o.lq * L.q_nb1;
    L.kv_nb1   = D * sizeof(uint16_t);                    // K/V:  [d, kv, nkvh] contiguous f16
    L.kv_nb2   = (size_t) o.kv * L.kv_nb1;
    L.mask_nb1 = (size_t) o.kv * sizeof(uint16_t);

    auto bump = [](size_t & off, size_t bytes) { size_t at = off; off = (off + bytes + 4095) & ~(size_t) 4095; return at; };
    size_t off = 0;
    L.q     = bump(off, (size_t) o.nh * L.q_nb2);
    L.k     = bump(off, (size_t) o.nkvh * L.kv_nb2);
    L.v     = bump(off, (size_t) o.nkvh * L.kv_nb2);
    L.kt    = bump(off, (size_t) o.nkvh * D * n_stage * sizeof(uint16_t));
    L.img_end = off;
    L.mask  = bump(off, (size_t) mask_rows * L.mask_nb1);
    L.gmask = bump(off, (size_t) mask_rows * L.mask_nb1);
    L.dst   = bump(off, (size_t) o.lq * o.nh * D * sizeof(float));
    const size_t ml_bytes = (rows * sizeof(float) + 127) & ~(size_t) 127;
    L.off_m   = 128;
    L.off_l   = L.off_m + ml_bytes;
    L.off_acc = L.off_l + ml_bytes;
    L.off_ctl = (L.off_acc + rows * D * sizeof(float) + 127) & ~(size_t) 127;
    L.off_exc = (L.off_ctl + HTP_FA_HETERO_STATUS_OFF + HTP_FA_FOLD_ST_N * sizeof(uint32_t) + 127) & ~(size_t) 127;
    L.fold_bytes = L.off_exc + (size_t) o.nkvh * num_sb * nblk * sizeof(float);   // sparse_exc_mem rows
    L.fold_bytes = (L.fold_bytes + 127) & ~(size_t) 127;
    L.fold  = bump(off, L.fold_bytes);

    // ---- block lists for the sparse NPU kernel: union per (tile, KV head) for arm A, pool for the fold arms ----
    std::vector<std::vector<int>> uni((size_t) n_tiles * o.nkvh);
    int u_max_a = 1, u_max_s = 1;
    for (int tl = 0; tl < n_tiles; tl++) {
        u_max_s = std::max(u_max_s, (int) pool[tl].size());
        for (int kvh = 0; kvh < o.nkvh; kvh++) {
            std::vector<char> in((size_t) nblk, 0);
            for (int b : pool[tl]) in[b] = 1;
            for (int j = 0; j < FA_TILE_R; j++) for (int b : exc[(size_t) kvh * num_sb + tl * FA_TILE_R + j]) in[b] = 1;
            std::vector<int> & u = uni[(size_t) kvh * n_tiles + tl];
            for (int b = 0; b < nblk; b++) if (in[b]) u.push_back(b);
            u_max_a = std::max(u_max_a, (int) u.size());
        }
    }
    // Head start: the first K tiles of the staged order (KV head ascending, tile ascending) attend
    // to the tile UNION on the NPU and hand the GPU nothing, so the GPU launch runs in their shadow.
    // Same attention as arm A for those rows; the fold arms' reference follows.
    std::vector<char> npu_union((size_t) o.nkvh * n_tiles, 0);
    {
        int left = o.head_start;
        for (int kvh = 0; kvh < o.nkvh && left > 0; kvh++) {
            for (int tl = 0; tl < n_tiles && left > 0; tl++, left--) {
                npu_union[(size_t) kvh * n_tiles + tl] = 1;
                for (int j = 0; j < FA_TILE_R; j++) exc[(size_t) kvh * num_sb + tl * FA_TILE_R + j].clear();
                u_max_s = std::max(u_max_s, (int) uni[(size_t) kvh * n_tiles + tl].size());
            }
        }
        CHECK(left == 0, "--head-start %d exceeds the %d tiles", o.head_start, o.nkvh * n_tiles);
        n_exc_pairs = 0;
        for (auto & e : exc) n_exc_pairs += e.size();
        CHECK(n_exc_pairs > 0, "head start swallowed every exception: nothing for the GPU to do");
    }
    L.sel_a = bump(off, (size_t) u_max_a * n_tiles * o.nkvh * sizeof(int32_t));
    L.cnt_a = bump(off, (size_t) n_tiles * o.nkvh * sizeof(float));
    L.sel_s = bump(off, (size_t) u_max_s * n_tiles * o.nkvh * sizeof(int32_t));
    L.cnt_s = bump(off, (size_t) n_tiles * o.nkvh * sizeof(float));
    L.idx   = bump(off, (size_t) o.nkvh * num_sb * o.exc * sizeof(int32_t));
    L.abs   = bump(off, (size_t) o.nkvh * n_stage_blk * sizeof(int32_t));
    L.cnt   = bump(off, (size_t) o.nkvh * sizeof(int32_t));
    L.lcnt  = bump(off, (size_t) o.nkvh * num_sb * sizeof(int32_t));
    L.err   = bump(off, 256);
    L.total = (off + (1u << 20) - 1) & ~((size_t) (1u << 20) - 1);

    // Alignments both halves assume. A wrong one here is a plausible-looking tensor, not a crash.
    CHECK((L.fold + L.off_acc) % 32 == 0, "acc base must be 32 B aligned for the kernel's vstore8");
    CHECK(L.off_m % 128 == 0 && L.off_l % 128 == 0 && L.off_acc % 128 == 0, "fold offsets must be 128-aligned (the device rejects otherwise)");
    CHECK(L.off_ctl % 128 == 0 && L.off_ctl >= 256, "off_ctl must be 128-aligned and >= 256 (it must clear the two header lines the DSP invalidates)");
    CHECK(L.off_ctl >= L.off_acc + rows * D * sizeof(float), "the control region must not overlap m/l/acc");
    CHECK(L.off_exc % 128 == 0 && L.off_exc >= L.off_ctl + HTP_FA_HETERO_STATUS_OFF + HTP_FA_FOLD_ST_N * sizeof(uint32_t),
          "the exception table must be 128-aligned and clear of the control region");
    CHECK(L.kv_nb1 % 8 == 0 && L.kv_nb2 % 8 == 0 && L.v % 8 == 0 && L.kt % 8 == 0, "K/V/Kt strides and bases must be whole RGBA-half texels");
    CHECK(L.mask_nb1 % 16 == 0, "mask row stride must be 16 B for the vload8 of half");
    CHECK(L.q_nb1 % 16 == 0 && L.q_nb2 % 16 == 0, "Q strides must be 16 B for the vload4 of float");
    CHECK(L.total < (size_t) 1 << 31, "buffer must stay under 2 GB: the kernel's offsets are int");

    // ---- rpcmem buffer + HTP0 backend ----
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("HTP0");
    CHECK(dev, "HTP0 device not found");
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    CHECK(be, "failed to init HTP0");
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), L.total);
    CHECK(buf, "failed to allocate %zu MB on HTP0", L.total >> 20);
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
    const int fd = rpcmem_to_fd(base);
    CHECK(fd >= 0, "rpcmem_to_fd failed (%d): the ION alias needs the dmabuf fd", fd);
    memset(base, 0, L.total);

    // ---- inputs ----
    float *    q     = (float *)    (base + L.q);
    uint16_t * k     = (uint16_t *) (base + L.k);
    uint16_t * v     = (uint16_t *) (base + L.v);
    uint16_t * mask  = (uint16_t *) (base + L.mask);    // NPU: shared blocks only, and causal
    uint16_t * gmask = (uint16_t *) (base + L.gmask);   // GPU: causal only, its block list does the rest
    float *    dst   = (float *)    (base + L.dst);
    {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> u(-0.5f, 0.5f);
        for (size_t i = 0; i < (size_t) o.nh * o.lq * D; i++) q[i] = u(rng);
        for (size_t i = 0; i < (size_t) o.kv * o.nkvh * D; i++) { k[i] = f2h(u(rng)); v[i] = f2h(u(rng)); }
        const uint16_t neg = f2h(-INFINITY), zero = f2h(0.0f);
        for (int r = 0; r < mask_rows; r++) {
            // llama.cpp causal mask: query token t may see keys [0, (kv - lq) + t]
            const int last = (o.kv - o.lq) + std::min(r, o.lq - 1);
            const std::vector<int> & pl = pool[std::min(r, o.lq - 1) / FA_TILE_BQ];
            uint16_t * mr = mask + (size_t) r * o.kv, * gr = gmask + (size_t) r * o.kv;
            for (int j = 0; j < o.kv; j++) { mr[j] = neg; gr[j] = j > last ? neg : zero; }
            for (int b : pl) {
                for (int j = b * FA2_BK; j < (b + 1) * FA2_BK; j++) mr[j] = j > last ? neg : zero;
            }
        }
    }

    // ---- device tables ----
    int32_t * idx_h  = (int32_t *) (base + L.idx);
    int32_t * abs_h  = (int32_t *) (base + L.abs);
    int32_t * cnt_h  = (int32_t *) (base + L.cnt);
    int32_t * lcnt_h = (int32_t *) (base + L.lcnt);
    for (int kvh = 0; kvh < o.nkvh; kvh++) {
        cnt_h[kvh] = (int32_t) stage_abs[kvh].size() * FA2_BK;
        for (int c = 0; c < n_stage_blk; c++) abs_h[(size_t) kvh * n_stage_blk + c] = c < (int) stage_abs[kvh].size() ? stage_abs[kvh][c] : -1;
        for (int sb = 0; sb < num_sb; sb++) {
            const std::vector<int> & ex = exc[(size_t) kvh * num_sb + sb];
            int32_t * row = idx_h + ((size_t) kvh * num_sb + sb) * o.exc;
            for (int s = 0; s < o.exc; s++) row[s] = s < (int) ex.size() ? to_compact[kvh][ex[s]] : -1;
            lcnt_h[(size_t) kvh * num_sb + sb] = (int32_t) ex.size();
            for (int s = 0; s < (int) ex.size(); s++) {
                CHECK(row[s] >= 0 && row[s] < (int) stage_abs[kvh].size(), "kvh %d sub-block %d slot %d: compact id %d out of range", kvh, sb, s, row[s]);
                CHECK(stage_abs[kvh][row[s]] == ex[s], "kvh %d sub-block %d slot %d: compact id maps to block %d, expected %d", kvh, sb, s, stage_abs[kvh][row[s]], ex[s]);
            }
        }
    }

    // ---- src[5]/src[6] for the sparse NPU arms. Element (i, tile, kvh) of a [u, NBq, nkvh] I32 tensor ----
    {
        int32_t * sa_h = (int32_t *) (base + L.sel_a); float * ca_h = (float *) (base + L.cnt_a);
        int32_t * ss_h = (int32_t *) (base + L.sel_s); float * cs_h = (float *) (base + L.cnt_s);
        for (int kvh = 0; kvh < o.nkvh; kvh++) {
            for (int tl = 0; tl < n_tiles; tl++) {
                const std::vector<int> & u = uni[(size_t) kvh * n_tiles + tl];
                int32_t * ra = sa_h + ((size_t) kvh * n_tiles + tl) * u_max_a;
                int32_t * rs = ss_h + ((size_t) kvh * n_tiles + tl) * u_max_s;
                for (int i = 0; i < u_max_a; i++) ra[i] = i < (int) u.size() ? u[i] : 0;
                const std::vector<int> & sr = npu_union[(size_t) kvh * n_tiles + tl] ? u : pool[tl];
                for (int i = 0; i < u_max_s; i++) rs[i] = i < (int) sr.size() ? sr[i] : 0;
                ca_h[(size_t) kvh * n_tiles + tl] = (float) u.size();
                cs_h[(size_t) kvh * n_tiles + tl] = (float) sr.size();
            }
        }
    }

    // ---- fold header. flags is rewritten per arm; everything else is fixed ----
    uint8_t * const fold = base + L.fold;
    {
        // STAGED: the graph's sparse_exc_mem rows -- per (KV head, 64-token sub-block) a 0/1 f32
        // per KV block; the kernel's per-tile test is "any nonzero", no packing anywhere
        float * ex_w = (float *) (fold + L.off_exc);
        memset(ex_w, 0, (size_t) o.nkvh * num_sb * nblk * sizeof(float));
        for (int kvh = 0; kvh < o.nkvh; kvh++)
            for (int sb = 0; sb < num_sb; sb++)
                for (int b : exc[(size_t) kvh * num_sb + sb]) ex_w[((size_t) kvh * num_sb + sb) * nblk + b] = 1.0f;
        dc_cvac(ex_w, (size_t) o.nkvh * num_sb * nblk * sizeof(float));
    }
    uint8_t * const ctl  = fold + L.off_ctl;
    const uint32_t slot  = 0;
    const uint32_t timeout_us = 200000;
    auto write_hdr = [&](uint32_t flags) {
        struct htp_fa_fold_hdr hdr = {};
        hdr.magic      = HTP_FA_FOLD_MAGIC;
        hdr.rows       = (uint32_t) rows;
        hdr.neq1       = (uint32_t) o.lq;
        hdr.dv         = (uint32_t) D;
        hdr.off_m      = (uint32_t) L.off_m;
        hdr.off_l      = (uint32_t) L.off_l;
        hdr.off_acc    = (uint32_t) L.off_acc;
        hdr.flags      = flags | (o.instore ? HTP_FA_FOLD_F_INSTORE : 0u) | (o.stage_qb && (flags & HTP_FA_FOLD_F_STAGED) ? HTP_FA_FOLD_F_STAGE_QB : 0u);
        hdr.slot       = slot;
        hdr.off_ctl    = (uint32_t) L.off_ctl;
        hdr.timeout_us = timeout_us;
        hdr.off_exc    = (uint32_t) L.off_exc;
        hdr.exc_nbk    = (uint32_t) nblk;
        memcpy(fold, &hdr, sizeof(hdr));
        dc_cvac(fold, sizeof(hdr));
    };
    write_hdr(0);
    volatile uint32_t * ready_w  = (volatile uint32_t *) (ctl + (size_t) slot * HTP_FA_HETERO_SLOT_STRIDE);
    volatile uint32_t * done_w   = (volatile uint32_t *) (ctl + (size_t) slot * HTP_FA_HETERO_SLOT_STRIDE + 128);
    volatile uint32_t * dwait_w  = (volatile uint32_t *) (ctl + (size_t) slot * HTP_FA_HETERO_SLOT_STRIDE + 8);
    volatile uint32_t * status_w = (volatile uint32_t *) (ctl + HTP_FA_HETERO_STATUS_OFF);
    // Only the status region. The per-slot wait word shares its 64 B line with `ready`, and
    // cleaning that line from the CPU would write the CPU's copy of `ready` back over the DSP's.
    auto reset_status = [&]() {
        memset((void *) status_w, 0, HTP_FA_FOLD_ST_N * sizeof(uint32_t));
        dc_cvac((const void *) status_w, HTP_FA_FOLD_ST_N * sizeof(uint32_t));
    };
    reset_status();
    // Push the zeroed slot line out of the CPU's cache once. A dirty copy of `ready` could
    // otherwise be written back over the sequence the DSP publishes later.
    dc_cvac((const void *) ready_w, HTP_FA_HETERO_SLOT_STRIDE);

    printf("shape  d=%d nh=%d nkvh=%d (G=%d) lq=%d kv=%d | tile=%d sub-block=%d topk=%d cstar=%d exc=%d | fold %s\n",
           o.d, o.nh, o.nkvh, G, o.lq, o.kv, FA_TILE_BQ, FA2_BK, o.topk, o.cstar, o.exc, o.instore ? "in-store" : "diagonal");
    printf("split  shared %zu blocks/tile (NPU), %zu exception pairs (GPU) on %d of %d KV heads, head start %d tiles | staged %zu of %d blocks per KV head | npu %s (u_max union %d, pool %d)\n",
           pool[0].size(), n_exc_pairs, o.exc_heads, o.nkvh, o.head_start, n_stage_used / (size_t) o.nkvh, nblk,
           o.sparse ? "sparse src[5]" : "dense-with-mask", u_max_a, u_max_s);
    printf("buffer %zu MB rpcmem | fold at +%zu, m +%zu l +%zu acc +%zu ctl +%zu (%zu KB)\n",
           L.total >> 20, L.fold, L.off_m, L.off_l, L.off_acc, L.off_ctl, L.fold_bytes >> 10);

    // ---- OpenCL ----
    cl_platform_id plat = nullptr; cl_device_id cdev = nullptr; cl_uint n = 0; cl_int err = 0;
    CL_CHECK(clGetPlatformIDs(1, &plat, &n), "clGetPlatformIDs");
    CHECK(n > 0, "no OpenCL platform");
    CL_CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &cdev, &n), "clGetDeviceIDs");
    CHECK(n > 0, "no OpenCL GPU");
    char dname[256] = {0}; clGetDeviceInfo(cdev, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
    size_t ext_len = 0; clGetDeviceInfo(cdev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_len);
    std::string ext(ext_len, '\0'); clGetDeviceInfo(cdev, CL_DEVICE_EXTENSIONS, ext_len, &ext[0], nullptr);
    // The ION allocation type rides on cl_qcom_ext_host_ptr; the Adreno 830 driver does NOT
    // advertise a separate cl_qcom_ion_host_ptr string even though CL_MEM_ION_HOST_PTR_QCOM works.
    CHECK(ext.find("cl_qcom_ext_host_ptr") != std::string::npos,
          "device has no cl_qcom_ext_host_ptr: the GPU cannot write the fold buffer in place");
    size_t max_img_buf = 0; clGetDeviceInfo(cdev, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, sizeof(max_img_buf), &max_img_buf, nullptr);

    cl_context ctx_cl = clCreateContext(nullptr, 1, &cdev, nullptr, nullptr, &err); CL_CHECK(err, "clCreateContext");
    cl_command_queue cq = clCreateCommandQueueWithProperties(ctx_cl, cdev, nullptr, &err); CL_CHECK(err, "clCreateCommandQueue");

    char opts[128];
    snprintf(opts, sizeof(opts), "-cl-std=CL2.0 -cl-fast-relaxed-math -DFA_D=%d", o.d);
    size_t src_len = strlen(cl_src);
    cl_program prog = clCreateProgramWithSource(ctx_cl, 1, &cl_src, &src_len, &err); CL_CHECK(err, "clCreateProgramWithSource");
    if (clBuildProgram(prog, 1, &cdev, opts, nullptr, nullptr) != CL_SUCCESS) {
        size_t ln = 0; clGetProgramBuildInfo(prog, cdev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ln);
        std::string log(ln, '\0'); clGetProgramBuildInfo(prog, cdev, CL_PROGRAM_BUILD_LOG, ln, &log[0], nullptr);
        DIE("OpenCL build failed:\n%s", log.c_str());
    }
    cl_kernel k_stage = clCreateKernel(prog, "stage_k", &err); CL_CHECK(err, "kernel stage_k");
    cl_kernel k_fa    = clCreateKernel(prog, "fa_exc",  &err); CL_CHECK(err, "kernel fa_exc");

    cl_mem_ion_host_ptr ion = {};
    ion.ext_host_ptr.allocation_type   = CL_MEM_ION_HOST_PTR_QCOM;
    ion.ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
    ion.ion_filedesc = fd; ion.ion_hostptr = base;
    cl_mem alias = clCreateBuffer(ctx_cl, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM, L.total, &ion, &err);
    CL_CHECK(err, "ION alias of the rpcmem buffer");

    // One RGBA-half image over [0, img_end): the staged K and V are both read through it.
    cl_image_format fmt = { CL_RGBA, CL_HALF_FLOAT };
    cl_image_desc dk; memset(&dk, 0, sizeof(dk));
    dk.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    dk.image_width = L.img_end / 8;
    dk.buffer      = alias;
    CHECK(dk.image_width <= max_img_buf, "K/V/Kt image needs %zu texels, device max is %zu", dk.image_width, max_img_buf);
    cl_mem buf_img = clCreateImage(ctx_cl, CL_MEM_READ_ONLY, &fmt, &dk, nullptr, &err); CL_CHECK(err, "K/V/Kt image");

    printf("gpu    %s | image %zu of max %zu texels | opts '%s'\n", dname, dk.image_width, max_img_buf, opts);

    // ---- kernel args ----
    auto sa = [&](cl_kernel kk, int idx, size_t sz, const void * p) { CL_CHECK(clSetKernelArg(kk, idx, sz, p), "clSetKernelArg"); };
    const int32_t k_off = (int32_t) L.k, k_nb1 = (int32_t) L.kv_nb1, k_nb2 = (int32_t) L.kv_nb2, kt_off = (int32_t) L.kt;
    const int32_t abs_off = (int32_t) L.abs, cnt_off = (int32_t) L.cnt, err_off = (int32_t) L.err;
    const int32_t abs_stride = n_stage_blk;
    int ai = 0;
    sa(k_stage, ai++, sizeof(alias), &alias);
    sa(k_stage, ai++, 4, &k_off); sa(k_stage, ai++, 4, &k_nb1); sa(k_stage, ai++, 4, &k_nb2); sa(k_stage, ai++, 4, &kt_off);
    sa(k_stage, ai++, 4, &abs_off); sa(k_stage, ai++, 4, &cnt_off);
    sa(k_stage, ai++, 4, &abs_stride); sa(k_stage, ai++, 4, &n_stage);

    const int32_t q_off = (int32_t) L.q, q_nb1 = (int32_t) L.q_nb1, q_nb2 = (int32_t) L.q_nb2;
    const int32_t gmask_off = (int32_t) L.gmask, mask_nb1 = (int32_t) L.mask_nb1;
    const int32_t idx_off = (int32_t) L.idx, lcnt_off = (int32_t) L.lcnt;
    const int32_t f_m = (int32_t) (L.fold + L.off_m), f_l = (int32_t) (L.fold + L.off_l), f_acc = (int32_t) (L.fold + L.off_acc);
    const int32_t kt_base_tex = (int32_t) (L.kt / 8), v_base_tex = (int32_t) (L.v / 8);
    const int32_t v_tok_tex = (int32_t) (L.kv_nb1 / 8), v_head_tex = (int32_t) (L.kv_nb2 / 8);
    ai = 0;
    sa(k_fa, ai++, sizeof(buf_img), &buf_img); sa(k_fa, ai++, sizeof(alias), &alias);
    sa(k_fa, ai++, 4, &q_off); sa(k_fa, ai++, 4, &q_nb1); sa(k_fa, ai++, 4, &q_nb2);
    sa(k_fa, ai++, 4, &gmask_off); sa(k_fa, ai++, 4, &mask_nb1);
    sa(k_fa, ai++, 4, &idx_off); sa(k_fa, ai++, 4, &abs_off); sa(k_fa, ai++, 4, &cnt_off); sa(k_fa, ai++, 4, &lcnt_off); sa(k_fa, ai++, 4, &err_off);
    sa(k_fa, ai++, 4, &f_m); sa(k_fa, ai++, 4, &f_l); sa(k_fa, ai++, 4, &f_acc);
    sa(k_fa, ai++, 4, &o.lq); sa(k_fa, ai++, 4, &G); sa(k_fa, ai++, 4, &num_sb); sa(k_fa, ai++, 4, &qb_per_sb); sa(k_fa, ai++, 4, &o.exc);
    sa(k_fa, ai++, 4, &abs_stride); sa(k_fa, ai++, 4, &n_stage);
    sa(k_fa, ai++, 4, &kt_base_tex); sa(k_fa, ai++, 4, &v_base_tex); sa(k_fa, ai++, 4, &v_tok_tex); sa(k_fa, ai++, 4, &v_head_tex);
    sa(k_fa, ai++, 4, &scale);

    const size_t g_stage[3] = { (size_t) n_stage, D / 8, (size_t) o.nkvh };
    const size_t g_fa[3]    = { 128, (size_t) num_qb, (size_t) o.nh };
    const size_t l_fa[3]    = { 128, 1, 1 };

    // ---- stage once, then prove the transpose moved the right bytes ----
    CL_CHECK(clEnqueueNDRangeKernel(cq, k_stage, 3, nullptr, g_stage, nullptr, 0, nullptr, nullptr), "stage_k");
    CL_CHECK(clFinish(cq), "clFinish stage");
    {
        const uint16_t * kt_dev = (const uint16_t *) (base + L.kt);
        size_t bad = 0;
        for (int kvh = 0; kvh < o.nkvh; kvh++) {
            for (size_t c = 0; c < stage_abs[kvh].size(); c++) {
                for (int i = 0; i < FA2_BK; i++) {
                    const int tok = stage_abs[kvh][c] * FA2_BK + i;
                    const uint16_t * src = (const uint16_t *) (base + L.k + (size_t) tok * L.kv_nb1 + (size_t) kvh * L.kv_nb2);
                    const size_t ki = c * FA2_BK + i;
                    for (size_t e = 0; e < D; e++) if (kt_dev[(size_t) kvh * D * n_stage + e * n_stage + ki] != src[e]) bad++;
                }
            }
        }
        CHECK(bad == 0, "staged K differs from the source in %zu of %zu halves: the transpose is wrong", bad, n_stage_used * D);
    }
    double ms_stage = 0;
    {
        const double t0 = now_ms();
        for (int i = 0; i < o.iters; i++) CL_CHECK(clEnqueueNDRangeKernel(cq, k_stage, 3, nullptr, g_stage, nullptr, 0, nullptr, nullptr), "stage_k");
        CL_CHECK(clFinish(cq), "clFinish stage");
        ms_stage = (now_ms() - t0) / o.iters;
    }

    // ---- graphs: identical NPU work in all three arms, the fold is the only difference ----
    struct arm_graph {
        ggml_context * ctx = nullptr;
        ggml_cgraph *  gf  = nullptr;
    };
    auto make_graph = [&](bool with_fold) {
        // sparse: arm A (no fold) attends to the union, the fold arms to the pool; the mask is causal only
        const bool   uni_sel  = !with_fold;
        const size_t mask_off = o.sparse ? L.gmask : L.mask;
        arm_graph g;
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        g.ctx = ggml_init(ip);
        ggml_tensor * tq = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, o.d, o.lq, o.nh, 1);
        ggml_tensor * tk = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, o.d, o.kv, o.nkvh, 1);
        ggml_tensor * tv = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, o.d, o.kv, o.nkvh, 1);
        ggml_tensor * tm = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, o.kv, mask_rows, 1, 1);
        place(buf, base, L.q, tq, "q"); place(buf, base, L.k, tk, "k");
        place(buf, base, L.v, tv, "v"); place(buf, base, mask_off, tm, "mask");
        ggml_tensor * out = ggml_flash_attn_ext(g.ctx, tq, tk, tv, tm, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        place(buf, base, L.dst, out, "dst");
        if (o.sparse) {
            ggml_tensor * ts = ggml_new_tensor_4d(g.ctx, GGML_TYPE_I32, uni_sel ? u_max_a : u_max_s, n_tiles, o.nkvh, 1);
            ggml_tensor * tc = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, n_tiles, o.nkvh, 1, 1);
            place(buf, base, uni_sel ? L.sel_a : L.sel_s, ts, "sel");
            place(buf, base, uni_sel ? L.cnt_a : L.cnt_s, tc, "cnt");
            ggml_flash_attn_ext_set_sparse(out, ts, FA2_BK, FA_TILE_BQ);
            ggml_flash_attn_ext_set_sparse_cnt(out, tc);
        }
        if (with_fold) {
            ggml_tensor * tf = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, (int64_t) (L.fold_bytes / 4));
            place(buf, base, L.fold, tf, "fold");
            out->src[7] = tf;
        }
        g.gf = ggml_new_graph(g.ctx);
        ggml_build_forward_expand(g.gf, out);
        return g;
    };
    auto compute = [&](arm_graph & g, const char * arm) {
        const ggml_status st = ggml_backend_graph_compute(be, g.gf);
        CHECK(st == GGML_STATUS_SUCCESS, "arm %s: graph compute failed (%d). A live arm fails here when the DSP gave up on the producer", arm, (int) st);
    };

    const size_t dst_elems = (size_t) o.lq * o.nh * D;
    std::vector<float> out_a(dst_elems), out_b(dst_elems), out_c(dst_elems);

    // ================= arm A: npu-only =================
    double ms_npu_only = 0;
    {
        arm_graph g = make_graph(false);
        memset(dst, 0, dst_elems * sizeof(float));
        compute(g, "A");
        memcpy(out_a.data(), dst, dst_elems * sizeof(float));
        const double t0 = now_ms();
        for (int i = 0; i < o.iters; i++) compute(g, "A");
        ms_npu_only = (now_ms() - t0) / o.iters;
        ggml_free(g.ctx);
    }

    // ================= arm B: serial =================
    // The GPU runs to completion first, so the partial is resident before the op starts (flags = 0)
    // and the two engines never overlap. This is the pair of numbers arm C is judged against.
    const uint32_t poison = 0x7fc0dead;
    auto poison_partial = [&]() {
        uint32_t * w = (uint32_t *) (fold + L.off_m);
        const size_t nw = (L.off_ctl - L.off_m) / 4;
        for (size_t i = 0; i < nw; i++) w[i] = poison;
        memset(base + L.err, 0, 256);
        dc_cvac(fold + L.off_m, L.off_ctl - L.off_m);
    };
    auto gpu_fa_once = [&]() {
        CL_CHECK(clEnqueueNDRangeKernel(cq, k_fa, 3, nullptr, g_fa, l_fa, 0, nullptr, nullptr), "fa_exc");
        CL_CHECK(clFinish(cq), "clFinish fa");
    };

    write_hdr(0);
    poison_partial();
    gpu_fa_once();
    const int32_t dev_err = *(const int32_t *) (base + L.err);
    CHECK(dev_err == 0, "the GPU kernel reported %d out-of-range block ids", dev_err);
    {
        const uint32_t * fm = (const uint32_t *) (fold + L.off_m);
        const uint32_t * fl = (const uint32_t *) (fold + L.off_l);
        const uint32_t * fa = (const uint32_t *) (fold + L.off_acc);
        size_t left = 0;
        for (size_t r = 0; r < rows; r++) {
            if (fm[r] == poison) left++;
            if (fl[r] == poison) left++;
            for (size_t e = 0; e < D; e++) if (fa[r * D + e] == poison) left++;
        }
        CHECK(left == 0, "%zu fold words were never written by the GPU kernel", left);
    }

    double ms_gpu = 0, ms_gpu_pipe = 0;
    {
        double t0 = now_ms();
        for (int i = 0; i < o.iters; i++) gpu_fa_once();                 // launch + finish, what the relay pays
        ms_gpu = (now_ms() - t0) / o.iters;
        t0 = now_ms();
        for (int i = 0; i < o.iters; i++) CL_CHECK(clEnqueueNDRangeKernel(cq, k_fa, 3, nullptr, g_fa, l_fa, 0, nullptr, nullptr), "fa_exc");
        CL_CHECK(clFinish(cq), "clFinish fa");
        ms_gpu_pipe = (now_ms() - t0) / o.iters;
    }

    double ms_npu_fold = 0;
    uint32_t st_b[HTP_FA_FOLD_ST_N] = {};
    {
        reset_status();
        arm_graph g = make_graph(true);
        memset(dst, 0, dst_elems * sizeof(float));
        compute(g, "B");
        memcpy(out_b.data(), dst, dst_elems * sizeof(float));
        const double t0 = now_ms();
        for (int i = 0; i < o.iters; i++) compute(g, "B");
        ms_npu_fold = (now_ms() - t0) / o.iters;
        ggml_free(g.ctx);
        dc_civac((const void *) status_w, sizeof(st_b));
        memcpy(st_b, (const void *) status_w, sizeof(st_b));
    }

    // ================= arm C: live =================
    //
    // The relay thread must already be spinning when the op publishes `ready`, and it must know the
    // sequence the DSP will publish NEXT. Re-reading the ready word at the top of each handshake
    // would race the op that publishes before the relay gets there.
    std::atomic<int>  hs_ok{0};
    std::atomic<bool> hs_abort{false};
    std::atomic<uint32_t> hs_bad_seq{0};
    uint32_t seq_last = 0;
    double ms_live = 0;
    uint32_t st_c[HTP_FA_FOLD_ST_N] = {};
    {
        write_hdr(HTP_FA_FOLD_F_LIVE);
        // A stale partial from arm B in this buffer would let arm C pass its check with a
        // handshake that did nothing. Poison it so only a live producer can be right.
        poison_partial();
        dc_civac((const void *) ready_w, 64);
        seq_last = *ready_w;
        reset_status();

        const int n_hs = 1 + o.iters;
        auto relay = [&](int n) {
            if (o.cpu >= 0) {
                cpu_set_t set; CPU_ZERO(&set); CPU_SET(o.cpu, &set);
                sched_setaffinity(0, sizeof(set), &set);
            }
            uint32_t last = seq_last;
            for (int i = 0; i < n; i++) {
                const uint64_t t0 = cnt_now();
                uint32_t cur;
                for (;;) {
                    dc_civac((const void *) ready_w, 4);
                    cur = __atomic_load_n(ready_w, __ATOMIC_ACQUIRE);
                    if (cur != last) break;
                    if (hs_abort.load(std::memory_order_relaxed)) return;
                    if (cnt_now() - t0 > 19200ull * 5000ull) { hs_abort = true; return; }   // 5 s
                }
                // A skipped sequence means the relay missed an op: it would then answer op N+1
                // with op N's partial, which is a plausible tensor, not an error.
                if (cur != last + 1) { hs_bad_seq = cur; hs_abort = true; return; }
                last = cur;
                gpu_fa_once();
                __atomic_store_n(done_w, cur, __ATOMIC_RELEASE);
                dc_cvac((const void *) done_w, sizeof(uint32_t));
                hs_ok++;
            }
        };
        std::thread th(relay, n_hs);

        arm_graph g = make_graph(true);
        memset(dst, 0, dst_elems * sizeof(float));
        compute(g, "C");                                   // this one folds a partial that was poison
        memcpy(out_c.data(), dst, dst_elems * sizeof(float));
        reset_status();                                    // drop the warmup's wait from the stats
        const double t0 = now_ms();
        for (int i = 0; i < o.iters; i++) compute(g, "C");
        ms_live = (now_ms() - t0) / o.iters;
        ggml_free(g.ctx);

        hs_abort = true;
        th.join();
        dc_civac((const void *) status_w, sizeof(st_c));
        memcpy(st_c, (const void *) status_w, sizeof(st_c));
        CHECK(hs_bad_seq.load() == 0, "the relay missed a handshake: ready jumped to %u", hs_bad_seq.load());
        CHECK(hs_ok.load() == n_hs, "the relay completed %d of %d handshakes", hs_ok.load(), n_hs);
    }

    // stages: KV heads (hpl per launch) or 256-row query blocks (all heads per launch)
    const int n_stages = o.stage_qb ? n_tiles : o.nkvh;
    const int qb32_per_tile = FA_TILE_BQ / FA2_BQ;

    // ================= arm D: staged live =================
    //
    // One GPU launch per KV head, enqueued together and signalled in order as each event
    // completes, so head k's partial is announced as soon as it exists. The DSP runs the
    // exception-free tiles first and waits once per head at its first exception tile.
    double ms_live_st = 0, ms_gpu_staged = 0;
    uint32_t st_d[HTP_FA_FOLD_ST_N] = {};
    std::vector<float> out_d(dst_elems);
    {
        std::vector<cl_event> ev((size_t) n_stages);
        auto gpu_fa_staged = [&](bool signal, uint32_t seq) {
            if (o.stage_qb) {
                for (int st = 0; st < n_stages; st++) {
                    const size_t off3[3] = { 0, (size_t) st * qb32_per_tile, 0 };
                    const size_t gsz[3]  = { 128, (size_t) qb32_per_tile, (size_t) o.nh };
                    CL_CHECK(clEnqueueNDRangeKernel(cq, k_fa, 3, off3, gsz, l_fa, 0, nullptr, &ev[st]), "fa_exc tile");
                    CL_CHECK(clFlush(cq), "clFlush");
                }
                for (int st = 0; st < n_stages; st++) {
                    CL_CHECK(clWaitForEvents(1, &ev[st]), "clWaitForEvents");
                    if (signal) {
                        __atomic_store_n(done_w + st, seq, __ATOMIC_RELEASE);
                        dc_cvac((const void *) (done_w + st), sizeof(uint32_t));
                    }
                    clReleaseEvent(ev[st]);
                }
                return;
            }
            for (int kvh = 0; kvh < o.nkvh; kvh += o.hpl) {
                const size_t off3[3] = { 0, 0, (size_t) kvh * G };
                const size_t gsz[3]  = { 128, (size_t) num_qb, (size_t) G * o.hpl };
                CL_CHECK(clEnqueueNDRangeKernel(cq, k_fa, 3, off3, gsz, l_fa, 0, nullptr, &ev[kvh]), "fa_exc head");
                // one submission per launch: batched into one command buffer, the driver reports
                // every event complete together at the END, and the staging degenerates to arm C
                CL_CHECK(clFlush(cq), "clFlush");
            }
            for (int kvh = 0; kvh < o.nkvh; kvh += o.hpl) {
                CL_CHECK(clWaitForEvents(1, &ev[kvh]), "clWaitForEvents");
                for (int h = kvh; h < kvh + o.hpl; h++) {
                    if (signal) {
                        __atomic_store_n(done_w + h, seq, __ATOMIC_RELEASE);
                        dc_cvac((const void *) (done_w + h), sizeof(uint32_t));
                    }
                }
                clReleaseEvent(ev[kvh]);
            }
        };
        {
            const double t0 = now_ms();
            for (int i = 0; i < o.iters; i++) gpu_fa_staged(false, 0);
            ms_gpu_staged = (now_ms() - t0) / o.iters;
        }

        write_hdr(HTP_FA_FOLD_F_LIVE | HTP_FA_FOLD_F_STAGED);
        poison_partial();
        dc_civac((const void *) ready_w, 64);
        const uint32_t seq_last_d = *ready_w;
        reset_status();
        hs_ok = 0; hs_abort = false; hs_bad_seq = 0;

        const int n_hs = 1 + o.iters;
        auto relay = [&](int n) {
            if (o.cpu >= 0) {
                cpu_set_t set; CPU_ZERO(&set); CPU_SET(o.cpu, &set);
                sched_setaffinity(0, sizeof(set), &set);
            }
            uint32_t last = seq_last_d;
            for (int i = 0; i < n; i++) {
                const uint64_t t0 = cnt_now();
                uint32_t cur;
                for (;;) {
                    dc_civac((const void *) ready_w, 4);
                    cur = __atomic_load_n(ready_w, __ATOMIC_ACQUIRE);
                    if (cur != last) break;
                    if (hs_abort.load(std::memory_order_relaxed)) return;
                    if (cnt_now() - t0 > 19200ull * 5000ull) { hs_abort = true; return; }
                }
                if (cur != last + 1) { hs_bad_seq = cur; hs_abort = true; return; }
                last = cur;
                gpu_fa_staged(true, cur);
                hs_ok++;
            }
        };
        std::thread th(relay, n_hs);

        arm_graph g = make_graph(true);
        memset(dst, 0, dst_elems * sizeof(float));
        compute(g, "D");
        memcpy(out_d.data(), dst, dst_elems * sizeof(float));
        reset_status();
        const double t0 = now_ms();
        for (int i = 0; i < o.iters; i++) compute(g, "D");
        ms_live_st = (now_ms() - t0) / o.iters;
        ggml_free(g.ctx);

        hs_abort = true;
        th.join();
        dc_civac((const void *) status_w, sizeof(st_d));
        memcpy(st_d, (const void *) status_w, sizeof(st_d));
        CHECK(hs_bad_seq.load() == 0, "staged: the relay missed a handshake: ready jumped to %u", hs_bad_seq.load());
        CHECK(hs_ok.load() == n_hs, "staged: the relay completed %d of %d handshakes", hs_ok.load(), n_hs);
    }

    // ---- results ----
    const double wait_ms     = st_c[HTP_FA_FOLD_ST_WAIT_US] / 1e3;
    const double wait_max_ms = st_c[HTP_FA_FOLD_ST_WAIT_US_MAX] / 1e3;
    const double serial_ms   = ms_gpu + ms_npu_fold;
    printf("\n");
    printf("time   stage    %8.3f ms  (hoistable: it depends only on K and the lists)\n", ms_stage);
    printf("time   A npu-only          %8.3f ms\n", ms_npu_only);
    printf("time   B serial  gpu %8.3f + npu %8.3f = %8.3f ms   (gpu back-to-back %8.3f ms)\n",
           ms_gpu, ms_npu_fold, serial_ms, ms_gpu_pipe);
    printf("time   C live              %8.3f ms   htp blocked %8.3f ms (max %8.3f) waits %u timeouts %u\n",
           ms_live, wait_ms, wait_max_ms, st_c[HTP_FA_FOLD_ST_WAITS], st_c[HTP_FA_FOLD_ST_TIMEOUTS]);
    printf("read   live vs serial %+.1f%%  |  live vs max(npu,gpu) %+.1f%%  |  the GPU hid %.0f%% of its %.3f ms\n",
           100.0 * (ms_live / serial_ms - 1.0),
           100.0 * (ms_live / std::max(ms_npu_fold, ms_gpu) - 1.0),
           100.0 * (1.0 - std::min(wait_ms, ms_gpu) / (ms_gpu > 0 ? ms_gpu : 1.0)), ms_gpu);
    const double wait_d_ms     = st_d[HTP_FA_FOLD_ST_WAIT_US] / 1e3;
    const double wait_d_max_ms = st_d[HTP_FA_FOLD_ST_WAIT_US_MAX] / 1e3;
    printf("time   D staged            %8.3f ms   htp blocked %8.3f ms (max %8.3f) stage waits that spun %u over %d ops, timeouts %u | gpu %d launches (%s) %8.3f ms\n",
           ms_live_st, wait_d_ms, wait_d_max_ms, st_d[HTP_FA_FOLD_ST_TILES_BLOCKED], o.iters, st_d[HTP_FA_FOLD_ST_TIMEOUTS],
           n_stages, o.stage_qb ? "per query block, all heads" : "per KV head", ms_gpu_staged);
    printf("read   staged vs npu-only %+.1f%%  |  staged vs serial %+.1f%%  |  staged vs live %+.1f%%  |  the GPU hid %.0f%% of its %.3f ms\n",
           100.0 * (ms_live_st / ms_npu_only - 1.0), 100.0 * (ms_live_st / serial_ms - 1.0), 100.0 * (ms_live_st / ms_live - 1.0),
           100.0 * (1.0 - std::min(wait_d_ms, ms_gpu_staged) / (ms_gpu_staged > 0 ? ms_gpu_staged : 1.0)), ms_gpu_staged);
    if (!o.sparse) {
        printf("note   --dense-npu: the NPU arms walk every KV block, so the shadow the GPU hides in is the optimistic case\n");
    }

    // ---- checks ----
    int fails = 0;
    auto fail = [&](const char * fmt, ...) {
        va_list ap; va_start(ap, fmt);
        fprintf(stderr, "FAIL: "); vfprintf(stderr, fmt, ap); fputc('\n', stderr); va_end(ap);
        fails++;
    };

    // Guards that do not need the reference.
    {
        const struct htp_fa_fold_hdr * hb = (const struct htp_fa_fold_hdr *) fold;
        if (hb->magic != HTP_FA_FOLD_MAGIC || hb->rows != rows || hb->neq1 != (uint32_t) o.lq || hb->dv != (uint32_t) D ||
            hb->off_ctl != (uint32_t) L.off_ctl) fail("the fold header was corrupted by the run");
        if (st_b[HTP_FA_FOLD_ST_WAITS] != 0) fail("arm B waited %u times: flags = 0 must not touch the handshake", st_b[HTP_FA_FOLD_ST_WAITS]);
        if (st_c[HTP_FA_FOLD_ST_WAITS] != (uint32_t) o.iters) fail("arm C recorded %u waits, expected %d (one per op)", st_c[HTP_FA_FOLD_ST_WAITS], o.iters);
        if (st_c[HTP_FA_FOLD_ST_TIMEOUTS] != 0) fail("arm C timed out %u times: those ops folded nothing", st_c[HTP_FA_FOLD_ST_TIMEOUTS]);
        if (st_c[HTP_FA_FOLD_ST_SLOT] != slot) fail("the DSP reported slot %u, expected %u", st_c[HTP_FA_FOLD_ST_SLOT], slot);
        if (st_d[HTP_FA_FOLD_ST_TIMEOUTS] != 0) fail("arm D timed out %u times: those ops folded nothing", st_d[HTP_FA_FOLD_ST_TIMEOUTS]);

        double d_cb = 0, d_ab = 0, d_db = 0;
        for (size_t i = 0; i < dst_elems; i++) {
            d_cb = std::max(d_cb, (double) fabsf(out_c[i] - out_b[i]));
            d_ab = std::max(d_ab, (double) fabsf(out_a[i] - out_b[i]));
            d_db = std::max(d_db, (double) fabsf(out_d[i] - out_b[i]));
        }
        printf("check  C vs B  max_abs %.3e (must be 0: same partial, same kernel)\n", d_cb);
        if (d_cb != 0.0) fail("arm C differs from arm B by %.3e; the live path is not folding the same partial", d_cb);
        printf("check  D vs B  max_abs %.3e (must be 0: same partial, tiles in another order)\n", d_db);
        if (o.instore ? !(d_db < 1e-3) : d_db != 0.0) {   // in-store: exception-free heads take the HMX f16 norm in D, f32 in B
            fail("arm D differs from arm B by %.3e; the staged path is not folding the same partial", d_db);
            // where: per (256-row tile, KV head), the unit the staged order permutes
            for (int kvh = 0; kvh < o.nkvh; kvh++) {
                printf("       D-B kvh %d:", kvh);
                for (int tl = 0; tl < n_tiles; tl++) {
                    double m = 0;
                    for (int g = 0; g < G; g++)
                        for (int t = tl * FA_TILE_BQ; t < (tl + 1) * FA_TILE_BQ; t++)
                            for (size_t e = 0; e < D; e++) {
                                const size_t i = ((size_t) t * o.nh + kvh * G + g) * D + e;
                                m = std::max(m, (double) fabsf(out_d[i] - out_b[i]));
                            }
                    printf(" tile %d %.1e", tl, m);
                }
                printf("\n");
            }
        }
        if (o.sparse) {
            printf("check  A vs B  max_abs %.3e (same attention, exceptions on different engines)\n", d_ab);
        } else {
            printf("check  A vs B  max_abs %.3e (must be > 0: the fold has to change the answer)\n", d_ab);
            if (!(d_ab > 0.0)) fail("arm A equals arm B: the fold changed nothing, so neither arm proves anything");
        }
    }

    if (o.check) {
        // Host reference in double. Per row: softmax over the shared blocks alone (what arm A must
        // reproduce), over the shared blocks UNION this row's exceptions (arms B and C), and the
        // unnormalised exception-only triple (what the GPU must have left in the fold buffer).
        std::vector<float> ref_shared(dst_elems), ref_full(dst_elems), ref_uni(dst_elems);
        const float * fm = (const float *) (fold + L.off_m);
        const float * fl = (const float *) (fold + L.off_l);
        const float * fa = (const float *) (fold + L.off_acc);
        const int nthr = std::max(1, std::min((int) std::thread::hardware_concurrency(), o.nh));
        std::vector<cmp> c_m(nthr), c_l(nthr), c_acc(nthr);
        std::vector<std::thread> thr;
        for (int w = 0; w < nthr; w++) {
            thr.emplace_back([&, w]() {
                std::vector<double> acc_s(D), acc_p(D), acc_u(D), sc((size_t) (o.topk + FA_TILE_R * o.exc) * FA2_BK);
                for (int h = w; h < o.nh; h += nthr) {
                    const int kvh = h / G;
                    for (int t = 0; t < o.lq; t++) {
                        const size_t r  = (size_t) h * o.lq + t;
                        const float * qr = q + r * D;
                        const uint16_t * mrow = gmask + (size_t) t * o.kv;
                        double m_s = -INFINITY, l_s = 0, m_p = -INFINITY, l_p = 0, m_u = -INFINITY, l_u = 0;
                        std::fill(acc_s.begin(), acc_s.end(), 0.0);
                        std::fill(acc_p.begin(), acc_p.end(), 0.0);
                        std::fill(acc_u.begin(), acc_u.end(), 0.0);
                        // scores first, then the sums off the same values
                        auto pass = [&](const std::vector<int> & blocks, double & m_o, double & l_o, std::vector<double> & acc_o) {
                            size_t n = 0;
                            for (int b : blocks) {
                                for (int i = 0; i < FA2_BK; i++, n++) {
                                    const int key = b * FA2_BK + i;
                                    if (!std::isfinite(h2f(mrow[key]))) { sc[n] = -INFINITY; continue; }
                                    const uint16_t * kr = (const uint16_t *) (base + L.k + (size_t) kvh * L.kv_nb2 + (size_t) key * L.kv_nb1);
                                    double dot = 0;
                                    for (size_t e = 0; e < D; e++) dot += (double) qr[e] * h2f(kr[e]);
                                    sc[n] = dot * scale;
                                    m_o = std::max(m_o, sc[n]);
                                }
                            }
                            if (!std::isfinite(m_o)) return;
                            n = 0;
                            for (int b : blocks) {
                                for (int i = 0; i < FA2_BK; i++, n++) {
                                    if (!std::isfinite(sc[n])) continue;
                                    const int key = b * FA2_BK + i;
                                    const uint16_t * vr = (const uint16_t *) (base + L.v + (size_t) kvh * L.kv_nb2 + (size_t) key * L.kv_nb1);
                                    const double p = exp(sc[n] - m_o);
                                    l_o += p;
                                    for (size_t e = 0; e < D; e++) acc_o[e] += p * h2f(vr[e]);
                                }
                            }
                        };
                        const std::vector<int> & exl = exc[(size_t) kvh * num_sb + t / FA2_BK];
                        pass(pool[t / FA_TILE_BQ], m_s, l_s, acc_s);
                        pass(exl, m_p, l_p, acc_p);
                        // arm A's policy is the TILE union: every sub-block's exceptions, for every row
                        pass(uni[(size_t) kvh * n_tiles + t / FA_TILE_BQ], m_u, l_u, acc_u);
                        CHECK(l_s > 0, "head %d token %d sees no shared key: the npu-only arm would divide by zero", h, t);
                        if (!exl.empty()) CHECK(l_p > 0, "head %d token %d owns exceptions but every key is masked", h, t);
                        float * rs = ref_shared.data() + ((size_t) t * o.nh + h) * D;
                        float * rf = ref_full.data()   + ((size_t) t * o.nh + h) * D;
                        float * ru = ref_uni.data()    + ((size_t) t * o.nh + h) * D;
                        for (size_t e = 0; e < D; e++) ru[e] = (float) (acc_u[e] / l_u);
                        const bool nu = npu_union[(size_t) kvh * n_tiles + t / FA_TILE_BQ] != 0;
                        const double M  = std::max(m_s, m_p);
                        const double ws = exp(m_s - M), wp = exl.empty() ? 0.0 : exp(m_p - M);
                        const double S  = ws * l_s + wp * l_p;
                        for (size_t e = 0; e < D; e++) {
                            rs[e] = (float) (acc_s[e] / l_s);
                            rf[e] = nu ? ru[e] : (float) ((ws * acc_s[e] + wp * acc_p[e]) / S);
                        }
                        if (!exl.empty()) {   // an empty row holds the M_EMPTY sentinel, not -inf
                            c_m[w].add(fm[r], m_p);
                            c_l[w].add(fl[r], l_p);
                            for (size_t e = 0; e < D; e++) c_acc[w].add(fa[r * D + e], acc_p[e]);
                        }
                    }
                }
            });
        }
        for (auto & t : thr) t.join();
        cmp cm, cl_, ca;
        for (int w = 0; w < nthr; w++) { cm.merge(c_m[w]); cl_.merge(c_l[w]); ca.merge(c_acc[w]); }
        printf("check  partial m   max_abs %.3e  nmse %.3e\n", cm.max_abs, cm.nmse());
        printf("check  partial l   max_abs %.3e  nmse %.3e\n", cl_.max_abs, cl_.nmse());
        printf("check  partial acc max_abs %.3e  nmse %.3e\n", ca.max_abs, ca.nmse());
        if (cm.nmse() > 1e-8 || cl_.nmse() > 1e-8 || ca.nmse() > 1e-8) fail("the GPU partial does not match the host reference");

        auto vs = [&](const char * name, const std::vector<float> & got, const std::vector<float> & want) {
            cmp c;
            for (size_t i = 0; i < dst_elems; i++) c.add(got[i], want[i]);
            printf("check  %-12s max_abs %.3e  nmse %.3e\n", name, c.max_abs, c.nmse());
            if (!(c.nmse() < 1e-4)) fail("%s does not match its reference (nmse %.3e)", name, c.nmse());
        };
        if (o.sparse) vs("A vs tile-union", out_a, ref_uni); else vs("A vs shared", out_a, ref_shared);
        vs("B vs union",  out_b, ref_full);
        vs("C vs union",  out_c, ref_full);
        vs("D vs union",  out_d, ref_full);
    } else {
        printf("check  skipped (--no-check)\n");
    }

    if (o.verbose) {
        dc_civac((const void *) dwait_w, sizeof(uint32_t));
        printf("  slot wait word %u us | done_seen %u seq_wanted %u\n",
               *dwait_w, st_c[HTP_FA_FOLD_ST_DONE_SEEN], st_c[HTP_FA_FOLD_ST_SEQ_WANTED]);
    }

    clReleaseMemObject(buf_img); clReleaseMemObject(alias);
    clReleaseKernel(k_fa); clReleaseKernel(k_stage); clReleaseProgram(prog);
    clReleaseCommandQueue(cq); clReleaseContext(ctx_cl);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    if (fails) { fprintf(stderr, "fa-hetero-live: %d check(s) failed\n", fails); return 1; }
    printf("ok\n");
    return 0;
}

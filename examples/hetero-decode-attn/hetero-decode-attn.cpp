// hetero-decode-attn: decode attention split between the HTP (tail of the KV range) and the
// Adreno GPU (head of the KV range), with both engines streaming the same KV cache at once.
//
// What it measures, for the heterogeneous decode design of docs/backend/snapdragon/heterogeneous-npu-gpu.md
// section 4g: (1) a GQA-aware OpenCL decode kernel on the real ggml KV layout that emits the
// split-KV partial format of the HVX decode kernel (M, S, acc[DV]); (2) its time and bandwidth
// alone; (3) the HVX decode op alone over its range; (4) both at once, started within a few
// microseconds of each other through the sync-probe handshake, so DRAM contention shows up
// as the slowdown of each side.
//
// Layout: one hexagon rpcmem buffer holds everything and is aliased into OpenCL zero-copy.
// The HTP graph is [sync-probe handshake] -> [FLASH_ATTN_EXT over K/V views of the HTP range]
// -> [sync-probe ping]; the probe stamps give the FA op's start and end on the HTP qtimer,
// which is the same counter as the CPU's cntvct (section 4g). The GPU kernel is pre-launched
// and spins on a fine-grain SVM word the host releases when the handshake fires.

#include "ggml.h"
#include "ggml-backend.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "htp-ops.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sched.h>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

extern "C" int rpcmem_to_fd(void * po);

// ---------------------------------------------------------------------------------------
// clocks / cache maintenance (aarch64 EL0), as in hetero-sync-probe
// ---------------------------------------------------------------------------------------

static inline uint64_t cnt_now() {
    uint64_t v;
    asm volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(v) : : "memory");
    return v;
}
static inline double ticks_us(double t) { return t / 19.2; }
static inline uint64_t realtime_ns() {
    timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
static double g_rt_off_ns = 0;
static void calibrate_realtime() {
    double best = 1e30, best_off = 0;
    for (int i = 0; i < 64; i++) {
        const uint64_t c0 = cnt_now(), r = realtime_ns(), c1 = cnt_now();
        const double w = (double) (c1 - c0);
        if (w < best) { best = w; best_off = (double) r - ((double) c0 + (double) c1) * 0.5 * 1e9 / 19.2e6; }
    }
    g_rt_off_ns = best_off;
}
static inline double cnt_of_realtime(uint64_t ns) { return ((double) ns - g_rt_off_ns) * 19.2e6 / 1e9; }

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
// options
// ---------------------------------------------------------------------------------------

struct options {
    int    kv        = 4096;
    double gpu_frac  = 0.5;     // fraction of the KV range given to the GPU (block aligned)
    int    iters     = 10;
    int    span      = 256;     // keys per GPU work-group (one split)
    int    nh        = 16;
    int    nkvh      = 8;
    int    d         = 128;
    int    cpu       = 7;
    int    verbose   = 0;
    int    mask_frac_permille = 10;   // ~1% of keys masked, exercises the -inf path
    int    layers    = 1;      // >1 = sustained token mode: chain N pre-enqueued GPU kernels, one relayed per layer
    int    pad_mb    = 0;      // place K/V this many MB into a larger buffer (alias/TLB footprint experiment)
    int    integrated = 0;     // drive the backend's own hetero path (GGML_HEXAGON_HETERO_FRAC = gpu_frac) on a full-range FA op and check vs CPU
    bool   run_htp = true, run_gpu = true, run_both = true;
    // --cluster: cluster-page decode kernel measurement (synthetic shadow + host page lists)
    int    cluster   = 0;
    int    density   = 25;     // percent of the candidate pages listed per KV head
    int    sel_scatter = 1;    // 1 = random pages, 0 = the first n pages (contiguous in the shadow)
    double skew      = 1.0;    // per-head list length skew: head lengths run from skew x mean to (2 - skew) x mean
    int    nslots    = 2;      // staging ring depth per thread (2, 4, 8)
    int    desc1d    = 0;      // 1 = one 1D descriptor per shadow page, 0 = 64 x 256 B rows
    int    window    = 256;    // dense recent window (positions [kv - window, kv))
    int    perm_identity = 0;  // 1 = pages in positional order (pure bandwidth), 0 = random permutation
    int    pmu       = 0;      // capture AXI read requests per op (GGML_HEXAGON_PROFILE PMU mode)
    int    seed      = 1234;
};

static void usage() {
    printf("usage: llama-hetero-decode-attn [--kv N] [--gpu-frac F] [--iters N] [--span N] [--nh N] [--nkvh N]\n"
           "         [--cpu N] [--modes htp,gpu,both] [--mask-permille N] [--layers N] [--pad-mb N] [--integrated] [-v]\n"
           "       llama-hetero-decode-attn --cluster [--kv N] [--density PCT] [--sel contig|scatter] [--skew F]\n"
           "         [--nslots N] [--desc 1d|2d] [--window W] [--perm identity|random] [--pmu] [--iters N] [--seed N]\n");
}

static bool parse(int argc, char ** argv, options & o) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next_i = [&](int & v) { if (i + 1 >= argc) return false; v = (int) strtol(argv[++i], nullptr, 0); return true; };
        auto next_d = [&](double & v) { if (i + 1 >= argc) return false; v = strtod(argv[++i], nullptr); return true; };
        if (a == "--kv")            { if (!next_i(o.kv)) return false; }
        else if (a == "--gpu-frac") { if (!next_d(o.gpu_frac)) return false; }
        else if (a == "--iters")    { if (!next_i(o.iters)) return false; }
        else if (a == "--span")     { if (!next_i(o.span)) return false; }
        else if (a == "--nh")       { if (!next_i(o.nh)) return false; }
        else if (a == "--nkvh")     { if (!next_i(o.nkvh)) return false; }
        else if (a == "--cpu")      { if (!next_i(o.cpu)) return false; }
        else if (a == "--mask-permille") { if (!next_i(o.mask_frac_permille)) return false; }
        else if (a == "--layers")   { if (!next_i(o.layers)) return false; }
        else if (a == "--pad-mb")   { if (!next_i(o.pad_mb)) return false; }
        else if (a == "--integrated") { o.integrated = 1; }
        else if (a == "--cluster")  { o.cluster = 1; }
        else if (a == "--density")  { if (!next_i(o.density)) return false; }
        else if (a == "--skew")     { if (!next_d(o.skew)) return false; }
        else if (a == "--nslots")   { if (!next_i(o.nslots)) return false; }
        else if (a == "--window")   { if (!next_i(o.window)) return false; }
        else if (a == "--seed")     { if (!next_i(o.seed)) return false; }
        else if (a == "--pmu")      { o.pmu = 1; }
        else if (a == "--sel")      { if (i + 1 >= argc) return false; o.sel_scatter = std::string(argv[++i]) != "contig"; }
        else if (a == "--desc")     { if (i + 1 >= argc) return false; o.desc1d = std::string(argv[++i]) == "1d"; }
        else if (a == "--perm")     { if (i + 1 >= argc) return false; o.perm_identity = std::string(argv[++i]) == "identity"; }
        else if (a == "-v")         { o.verbose = 1; }
        else if (a == "--modes") {
            if (i + 1 >= argc) return false;
            std::string m = argv[++i];
            o.run_htp = m.find("htp") != std::string::npos;
            o.run_gpu = m.find("gpu") != std::string::npos;
            o.run_both = m.find("both") != std::string::npos;
        }
        else { usage(); return false; }
    }
    if (o.kv % 64 || o.kv < 64) { fprintf(stderr, "--kv must be a positive multiple of 64\n"); return false; }
    if (o.d != 128) { fprintf(stderr, "only d=128 is built into the GPU kernel\n"); return false; }
    if (o.nh % o.nkvh) { fprintf(stderr, "nh must be a multiple of nkvh\n"); return false; }
    if (o.gpu_frac < 0 || o.gpu_frac > 1) { fprintf(stderr, "--gpu-frac in [0,1]\n"); return false; }
    return true;
}

// ---------------------------------------------------------------------------------------
// GPU kernel: one work-group per (KV head, split); FA_D lanes; the G query heads of the KV
// head share every K/V read. Emits the HVX split-KV partial: 128-byte header (M, S), then
// acc[FA_D] f32. Scores are scale*q.k + mask, exp is natural: the same units as the HVX
// kernel, so the HTP merge can combine partials from both engines.
// ---------------------------------------------------------------------------------------

static const char * cl_src_fa = R"CL(
#ifndef FA_D
#define FA_D 128
#endif
#ifndef FA_G
#define FA_G 2
#endif
#define HTP_M_INIT (-10000.0f)

__kernel void fa_dec_gqa(__global uchar * base,
                         uint q_off, uint k_off, uint v_off, uint m_off, uint p_off,
                         uint nbq2, uint nbk1, uint nbk2, uint nbv1, uint nbv2,
                         uint n_kv, uint span, uint nsplit, float scale, uint part_stride,
                         __global uint * svm, uint want, uint max_spin) {
    const int t   = get_local_id(0);
    const int kvh = get_group_id(1);
    const int sp  = get_group_id(2);

    __local float Q_l[FA_G][FA_D];
    __local float S_l[FA_G][FA_D];
    __local float sh_a[FA_G];
    __local float sh_m[FA_G];
    __local float sh_l[FA_G];

    if (want) {
        if (t == 0) {
            if (kvh == 0 && sp == 0) {
                atomic_store_explicit((volatile __global atomic_uint *)(svm + 1), 1u, memory_order_seq_cst, memory_scope_all_svm_devices);
            }
            uint it = 0;
            while (atomic_load_explicit((volatile __global atomic_uint *) svm, memory_order_acquire, memory_scope_all_svm_devices) != want && it < max_spin) { it++; }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    for (int g = 0; g < FA_G; ++g) {
        Q_l[g][t] = ((__global const float *)(base + q_off + (kvh * FA_G + g) * nbq2))[t] * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int kv_begin = sp * span;
    const int kv_end   = min((int) n_kv, kv_begin + (int) span);

    float m_run = -INFINITY, l_run = 0.0f;   // valid on lanes t < FA_G (lane t owns head t)
    float o_acc[FA_G];
    for (int g = 0; g < FA_G; ++g) o_acc[g] = 0.0f;

    if (kv_begin < kv_end) {
        for (int bs = kv_begin; bs < kv_end; bs += FA_D) {
            const int blk_n = min(FA_D, kv_end - bs);

            float s[FA_G];
            for (int g = 0; g < FA_G; ++g) s[g] = -INFINITY;
            if (t < blk_n) {
                __global const half * krow = (__global const half *)(base + k_off + (ulong)(bs + t) * nbk1 + kvh * nbk2);
                float8 acc[FA_G];
                for (int g = 0; g < FA_G; ++g) acc[g] = (float8)(0.0f);
                #pragma unroll
                for (int d8 = 0; d8 < FA_D / 8; ++d8) {
                    const float8 k8 = convert_float8(vload8(d8, krow));
                    for (int g = 0; g < FA_G; ++g) acc[g] += k8 * vload8(d8, Q_l[g]);
                }
                float mval = 0.0f;
                if (m_off) mval = (float)((__global const half *)(base + m_off))[bs + t];
                for (int g = 0; g < FA_G; ++g) {
                    const float8 a = acc[g];
                    s[g] = (a.s0 + a.s1 + a.s2 + a.s3 + a.s4 + a.s5 + a.s6 + a.s7) + mval;
                }
            }
            for (int g = 0; g < FA_G; ++g) S_l[g][t] = s[g];
            barrier(CLK_LOCAL_MEM_FENCE);

            if (t < FA_G) {
                float m_t = -INFINITY;
                for (int c = 0; c < FA_D; ++c) m_t = fmax(m_t, S_l[t][c]);
                const float m_new = fmax(m_run, m_t);
                const float a = (m_run == -INFINITY) ? 0.0f : native_exp(m_run - m_new);
                float l = 0.0f;
                for (int c = 0; c < FA_D; ++c) {
                    const float sc = S_l[t][c];
                    const float p = (isinf(sc) || m_new == -INFINITY) ? 0.0f : native_exp(sc - m_new);
                    S_l[t][c] = p;
                    l += p;
                }
                l_run = a * l_run + l;
                m_run = m_new;
                sh_a[t] = a;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            {
                float pv[FA_G];
                for (int g = 0; g < FA_G; ++g) pv[g] = 0.0f;
                __global const half * vb = (__global const half *)(base + v_off + (ulong) bs * nbv1 + kvh * nbv2) + t;
                const int vstep = nbv1 / 2;
                #pragma unroll 8
                for (int c = 0; c < blk_n; ++c) {
                    const float vf = (float) vb[(ulong) c * vstep];
                    for (int g = 0; g < FA_G; ++g) pv[g] += S_l[g][c] * vf;
                }
                for (int g = 0; g < FA_G; ++g) o_acc[g] = o_acc[g] * sh_a[g] + pv[g];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (t < FA_G) { sh_m[t] = (kv_begin < kv_end) ? m_run : HTP_M_INIT; sh_l[t] = l_run; }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int g = 0; g < FA_G; ++g) {
        const int row = kvh * FA_G + g;   // decode: one token, row = query head
        __global float * part = (__global float *)(base + p_off + ((ulong) row * nsplit + sp) * part_stride);
        if (t == 0) { part[0] = sh_m[g]; part[1] = sh_l[g]; }
        part[32 + t] = o_acc[g];
    }
}
)CL";

// ---------------------------------------------------------------------------------------

struct gpu_side {
    cl_platform_id plat = nullptr; cl_device_id dev = nullptr; cl_context ctx = nullptr; cl_command_queue q = nullptr;
    cl_program prog = nullptr; cl_kernel k_fa = nullptr; cl_mem alias = nullptr; uint32_t * svm = nullptr;
};

static bool cl_check(cl_int err, const char * what) {
    if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL: %s failed (%d)\n", what, err); return false; }
    return true;
}

static bool gpu_init(gpu_side & g, const options & o, uint8_t * base, size_t buf_size, int fd) {
    cl_int err; cl_uint n = 0;
    if (!cl_check(clGetPlatformIDs(1, &g.plat, &n), "clGetPlatformIDs") || !n) return false;
    if (!cl_check(clGetDeviceIDs(g.plat, CL_DEVICE_TYPE_GPU, 1, &g.dev, &n), "clGetDeviceIDs") || !n) return false;
    char name[256] = {0}; clGetDeviceInfo(g.dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    cl_device_svm_capabilities svm = 0; clGetDeviceInfo(g.dev, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm), &svm, nullptr);
    if (!(svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) || !(svm & CL_DEVICE_SVM_ATOMICS)) { fprintf(stderr, "gpu: no fine-grain SVM atomics\n"); return false; }
    g.ctx = clCreateContext(nullptr, 1, &g.dev, nullptr, nullptr, &err); if (!cl_check(err, "clCreateContext")) return false;
    cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    g.q = clCreateCommandQueueWithProperties(g.ctx, g.dev, qp, &err); if (!cl_check(err, "queue")) return false;

    char opts[128]; snprintf(opts, sizeof(opts), "-cl-std=CL2.0 -DFA_D=%d -DFA_G=%d", o.d, o.nh / o.nkvh);
    g.prog = clCreateProgramWithSource(g.ctx, 1, &cl_src_fa, nullptr, &err); if (!cl_check(err, "program")) return false;
    err = clBuildProgram(g.prog, 1, &g.dev, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t ln = 0; clGetProgramBuildInfo(g.prog, g.dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ln);
        std::string log(ln, '\0'); clGetProgramBuildInfo(g.prog, g.dev, CL_PROGRAM_BUILD_LOG, ln, log.data(), nullptr);
        fprintf(stderr, "OpenCL build failed (%d):\n%s\n", err, log.c_str()); return false;
    }
    g.k_fa = clCreateKernel(g.prog, "fa_dec_gqa", &err); if (!cl_check(err, "kernel")) return false;

    cl_uint page = 0; clGetDeviceInfo(g.dev, CL_DEVICE_PAGE_SIZE_QCOM, sizeof(page), &page, nullptr); if (!page) page = 4096;
    cl_mem_ion_host_ptr ion = {};
    ion.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
    ion.ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
    ion.ion_filedesc = fd; ion.ion_hostptr = base;
    g.alias = clCreateBuffer(g.ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM, buf_size, &ion, &err);
    if (!cl_check(err, "ION alias")) return false;
    g.svm = (uint32_t *) clSVMAlloc(g.ctx, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, 4096, 64);
    if (!g.svm) { fprintf(stderr, "clSVMAlloc failed\n"); return false; }
    printf("gpu: %s, alias %zu MB, kernel opts '%s'\n", name, buf_size >> 20, opts);
    return true;
}

// ---------------------------------------------------------------------------------------
// layout inside the shared buffer
// ---------------------------------------------------------------------------------------

struct layout {
    size_t ready = 0, done = 4096, rec1 = 8192, rec2 = 12288, flags_end = 16384;
    size_t q = 65536, mask = 131072, dst = 262144, parts = 524288, parts_size = 1 << 20;
    size_t k = 2 << 20, v = 0, total = 0;
    size_t nbk1 = 0, nbk2 = 0;     // K/V: position-major, heads contiguous within a position
    size_t part_stride = 0;
};

static ggml_tensor * place(ggml_backend_buffer_t buf, uint8_t * base, size_t off, ggml_tensor * t, const char * name) {
    t->buffer = buf; t->data = base + off; ggml_set_name(t, name);
    ggml_backend_buffer_init_tensor(buf, t);
    return t;
}

static void noop_custom(ggml_tensor *, int, int, void *) {}

static void set_probe_params(ggml_tensor * t, int mode /*0 handshake, 1 ping*/, const layout & L, int timeout_ms) {
    int32_t * p = t->op_params; memset(p, 0, sizeof(t->op_params));
    p[HTP_SYNC_PROBE_P_MAGIC] = (int32_t) HTP_SYNC_PROBE_MAGIC;
    p[HTP_SYNC_PROBE_P_MODE] = mode;
    p[HTP_SYNC_PROBE_P_TIMEOUT_US] = timeout_ms * 1000;
    p[HTP_SYNC_PROBE_P_INVAL] = 0;         // dcinva per poll
    p[HTP_SYNC_PROBE_P_READY_OFF] = (int32_t) L.ready;
    p[HTP_SYNC_PROBE_P_DONE_OFF]  = (int32_t) L.done;
    p[HTP_SYNC_PROBE_P_PAYLOAD_OFF] = (int32_t) L.done + 128;
    p[HTP_SYNC_PROBE_P_READY_VAL] = 1; p[HTP_SYNC_PROBE_P_DONE_VAL] = 1;
}

// CPU reference: full softmax attention of head h over keys [k0, k1). Natural units.
static void ref_attn(const options & o, const uint8_t * base, const layout & L, int h, int k0, int k1, std::vector<double> & out,
                     double * M_out, double * S_out) {
    const int kvh = h / (o.nh / o.nkvh);
    const float * q = (const float *) (base + L.q + (size_t) h * o.d * 4);
    const ggml_fp16_t * mask = (const ggml_fp16_t *) (base + L.mask);
    const double scale = 1.0 / sqrt((double) o.d);
    std::vector<double> s(k1 - k0);
    double M = -INFINITY;
    for (int p = k0; p < k1; ++p) {
        const ggml_fp16_t * kr = (const ggml_fp16_t *) (base + L.k + (size_t) p * L.nbk1 + (size_t) kvh * L.nbk2);
        double dot = 0;
        for (int d = 0; d < o.d; ++d) dot += (double) q[d] * ggml_fp16_to_fp32(kr[d]);
        const double mv = ggml_fp16_to_fp32(mask[p]);
        s[p - k0] = dot * scale + mv;
        M = std::max(M, s[p - k0]);
    }
    out.assign(o.d, 0.0);
    double S = 0;
    for (int p = k0; p < k1; ++p) {
        if (std::isinf(s[p - k0])) continue;
        const double w = exp(s[p - k0] - M);
        S += w;
        const ggml_fp16_t * vr = (const ggml_fp16_t *) (base + L.v + (size_t) p * L.nbk1 + (size_t) kvh * L.nbk2);
        for (int d = 0; d < o.d; ++d) out[d] += w * ggml_fp16_to_fp32(vr[d]);
    }
    if (M_out) *M_out = M;
    if (S_out) *S_out = S;
}

struct stat_acc {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    double med() { if (v.empty()) return NAN; auto s = v; std::sort(s.begin(), s.end()); return s[s.size() / 2]; }
    double mn() { return v.empty() ? NAN : *std::min_element(v.begin(), v.end()); }
    double mx() { return v.empty() ? NAN : *std::max_element(v.begin(), v.end()); }
    bool empty() const { return v.empty(); }
};

// ---------------------------------------------------------------------------------------
// --cluster: the page-list decode kernel (docs cluster-sparse-decode.md, Stage 1). One layer's
// shadow (htp-ops.h, struct htp_fa_cluster_header) is built here from a permutation of the
// positional K/V; per-head page lists are host-written; the FA node is tagged the way the backend
// tags it (op_params[11..15], src[7]) and checked against a CPU reference over exactly the keys
// the kernel was told to attend: the listed pages (mask ignored) plus the dense tail (masked).
// ---------------------------------------------------------------------------------------

static std::vector<std::string> g_prof_lines;
static void prof_log_cb(ggml_log_level level, const char * text, void *) {
    if (strstr(text, "profile-op FLASH_ATTN_EXT")) { g_prof_lines.emplace_back(text); return; }
    if (level != GGML_LOG_LEVEL_DEBUG) fputs(text, stderr);
}
// usec and AXI read requests (pmu[7]) of the last captured FLASH_ATTN_EXT profile line
static bool prof_last(double * usec, double * axi) {
    if (g_prof_lines.empty()) return false;
    const std::string & s = g_prof_lines.back();
    const size_t p = s.find("usec ");
    if (p == std::string::npos) return false;
    *usec = atof(s.c_str() + p + 5);
    *axi  = 0;
    const size_t b = s.find("pmu [");
    if (b != std::string::npos) {
        const size_t e = s.find(']', b);
        const size_t c = s.rfind(',', e);
        if (c != std::string::npos && c > b) *axi = atof(s.c_str() + c + 1);
    }
    return true;
}

// CPU reference over an explicit key set of (position, apply_mask) pairs, normalised.
static void ref_attn_keys(const options & o, const uint8_t * base, const layout & L, int h,
                          const std::vector<std::pair<int, bool>> & keys, std::vector<double> & out) {
    const int kvh = h / (o.nh / o.nkvh);
    const float * q = (const float *) (base + L.q + (size_t) h * o.d * 4);
    const ggml_fp16_t * mask = (const ggml_fp16_t *) (base + L.mask);
    const double scale = 1.0 / sqrt((double) o.d);
    std::vector<double> s(keys.size());
    double M = -INFINITY;
    for (size_t i = 0; i < keys.size(); ++i) {
        const int p = keys[i].first;
        const ggml_fp16_t * kr = (const ggml_fp16_t *) (base + L.k + (size_t) p * L.nbk1 + (size_t) kvh * L.nbk2);
        double dot = 0;
        for (int d = 0; d < o.d; ++d) dot += (double) q[d] * ggml_fp16_to_fp32(kr[d]);
        s[i] = dot * scale + (keys[i].second ? (double) ggml_fp16_to_fp32(mask[p]) : 0.0);
        M = std::max(M, s[i]);
    }
    out.assign(o.d, 0.0);
    double S = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (std::isinf(s[i])) continue;
        const double w = exp(s[i] - M);
        S += w;
        const int p = keys[i].first;
        const ggml_fp16_t * vr = (const ggml_fp16_t *) (base + L.v + (size_t) p * L.nbk1 + (size_t) kvh * L.nbk2);
        for (int d = 0; d < o.d; ++d) out[d] += w * ggml_fp16_to_fp32(vr[d]);
    }
    for (int d = 0; d < o.d; ++d) out[d] = S > 0 ? out[d] / S : 0.0;
}

static int run_cluster(const options & o, ggml_backend_t be, ggml_backend_buffer_t buf, uint8_t * base, const layout & L,
                       const htp_fa_cluster_header & hdr, size_t shadow_bytes) {
    const int W = o.window;
    const int covered_end = o.kv - W;
    if (W % 64 || covered_end <= 0) { fprintf(stderr, "--window must be a multiple of 64 and smaller than kv\n"); return 1; }
    if (o.nslots < 2 || o.nslots > HTP_FA_CLUSTER_MAX_SLOTS || (o.nslots & (o.nslots - 1))) { fprintf(stderr, "--nslots must be 2, 4 or 8\n"); return 1; }
    const int n_cand = covered_end / 64;
    const int G = o.nh / o.nkvh;
    uint8_t * lb = base + hdr.layer0_off;   // layer 0

    // header, empty directory, chunk table (1024-key chunks up to covered_end)
    memcpy(base, &hdr, sizeof(hdr));
    auto * dir = (htp_fa_cluster_dir *) (base + HTP_FA_CLUSTER_DIR_OFF);
    memset(dir, 0, sizeof(*dir));
    auto * chunks = (htp_fa_cluster_chunk *) (lb + hdr.off_chunks);
    uint32_t n_chunks = 0;
    for (int pos = 0; pos < covered_end; pos += (int) hdr.chunk_keys) {
        const int pend = std::min(pos + (int) hdr.chunk_keys, covered_end);
        chunks[n_chunks++] = { (uint32_t) pos, (uint32_t) pend, (uint32_t) (pos / 64), (uint32_t) ((pend - pos) / 64) };
    }

    // pages: a permutation of the covered positions, 64 per page, all heads share the page index space
    std::mt19937 rng(o.seed);
    std::vector<int> perm(covered_end);
    for (int i = 0; i < covered_end; ++i) perm[i] = i;
    if (!o.perm_identity) std::shuffle(perm.begin(), perm.end(), rng);
    auto * pos_map = (uint32_t *) (lb + hdr.off_pos_map);
    const size_t head_stride = (size_t) hdr.n_pages_max * hdr.page_bytes;
    for (int p = 0; p < n_cand; ++p) {
        for (int i = 0; i < 64; ++i) {
            const int pos = perm[p * 64 + i];
            pos_map[p * 64 + i] = (uint32_t) pos;
            for (int h = 0; h < o.nkvh; ++h) {
                const size_t poff = (size_t) h * head_stride + (size_t) p * hdr.page_bytes + (size_t) i * o.d * 2;
                memcpy(lb + hdr.off_k_pages + poff, base + L.k + (size_t) pos * L.nbk1 + (size_t) h * L.nbk2, (size_t) o.d * 2);
                memcpy(lb + hdr.off_v_pages + poff, base + L.v + (size_t) pos * L.nbk1 + (size_t) h * L.nbk2, (size_t) o.d * 2);
            }
        }
        for (int h = 0; h < o.nkvh; ++h) {   // page descriptor = f16 mean of its keys (for the later selection stages)
            std::vector<float> acc(o.d, 0.0f);
            for (int i = 0; i < 64; ++i) {
                const ggml_fp16_t * kr = (const ggml_fp16_t *) (lb + hdr.off_k_pages + (size_t) h * head_stride + (size_t) p * hdr.page_bytes + (size_t) i * o.d * 2);
                for (int d = 0; d < o.d; ++d) acc[d] += ggml_fp16_to_fp32(kr[d]);
            }
            ggml_fp16_t * c = (ggml_fp16_t *) (lb + hdr.off_centroids + ((size_t) h * hdr.n_pages_max + p) * hdr.centroid_bytes);
            for (int d = 0; d < o.d; ++d) c[d] = ggml_fp32_to_fp16(acc[d] / 64.0f);
        }
    }

    // per-head lists, address ordered
    std::vector<std::vector<uint16_t>> lists(o.nkvh);
    size_t pages_total = 0;
    for (int h = 0; h < o.nkvh; ++h) {
        double mult = 1.0;
        if (o.nkvh > 1) mult = 1.0 + (o.skew - 1.0) * (1.0 - 2.0 * h / (o.nkvh - 1));   // skew .. 2 - skew, mean 1
        int n = (int) lround(n_cand * (o.density / 100.0) * mult);
        n = std::max(0, std::min(n_cand, n));
        std::vector<uint16_t> & l = lists[h];
        if (o.sel_scatter) {
            std::vector<int> all(n_cand);
            for (int i = 0; i < n_cand; ++i) all[i] = i;
            std::shuffle(all.begin(), all.end(), rng);
            all.resize(n);
            std::sort(all.begin(), all.end());
            for (int x : all) l.push_back((uint16_t) x);
        } else {
            for (int i = 0; i < n; ++i) l.push_back((uint16_t) i);
        }
        pages_total += l.size();
        uint8_t * e = lb + hdr.off_host_sel + (size_t) h * hdr.host_sel_stride;
        *(uint32_t *) e = (uint32_t) n;
        if (n) memcpy(e + 128, l.data(), l.size() * 2);
    }
    // publish: data first, then the directory (the kernel reads the directory to find the data)
    dc_cvac(base, shadow_bytes);
    dir->n_chunks = n_chunks; dir->n_pages_pub = (uint32_t) n_cand; dir->covered_end = (uint32_t) covered_end;
    dc_cvac(dir, sizeof(*dir));

    // FA node over the full positional range, tagged the way the backend tags it
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(8, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const size_t kv_bytes = (size_t) o.kv * L.nbk1;
    ggml_tensor * shadow = place(buf, base, 0, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (shadow_bytes / 4)), "shadow");
    ggml_tensor * q  = place(buf, base, L.q, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, o.d, 1, o.nh, 1), "q");
    ggml_tensor * kb = place(buf, base, L.k, ggml_new_tensor_1d(ctx, GGML_TYPE_F16, (int64_t) (kv_bytes / 2)), "kbase");
    ggml_tensor * vb = place(buf, base, L.v, ggml_new_tensor_1d(ctx, GGML_TYPE_F16, (int64_t) (kv_bytes / 2)), "vbase");
    ggml_tensor * mb = place(buf, base, L.mask, ggml_new_tensor_1d(ctx, GGML_TYPE_F16, o.kv), "mbase");
    ggml_tensor * kf = ggml_view_4d(ctx, kb, o.d, o.kv, o.nkvh, 1, L.nbk1, L.nbk2, kv_bytes, 0);
    ggml_tensor * vf = ggml_view_4d(ctx, vb, o.d, o.kv, o.nkvh, 1, L.nbk1, L.nbk2, kv_bytes, 0);
    ggml_tensor * mf = ggml_view_2d(ctx, mb, o.kv, 1, (size_t) o.kv * 2, 0);
    kf->buffer = vf->buffer = mf->buffer = buf;
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, kf, vf, mf, 1.0f / sqrtf((float) o.d), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    place(buf, base, L.dst, fa, "fa_cluster");
    ggml_cgraph * g = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(g, fa);
    if (!ggml_backend_supports_op(be, fa)) { fprintf(stderr, "HTP0 rejects the FA op\n"); return 1; }

    uint32_t flags = (o.desc1d ? HTP_FA_CLUSTER_FLAG_DESC1D : 0) | HTP_FA_CLUSTER_FLAG_ECHO;
    { int lg = 0; while ((1 << lg) < o.nslots) lg++; flags |= (uint32_t) lg << 1; }
    auto set_mode = [&](bool on) {
        fa->op_params[HTP_FA_CLUSTER_OPP_MAGIC]   = on ? (int32_t) HTP_FA_CLUSTER_MAGIC : 0;
        fa->op_params[HTP_FA_CLUSTER_OPP_LAYER]   = 0;
        fa->op_params[HTP_FA_CLUSTER_OPP_DENSITY] = 0;    // host-written lists
        fa->op_params[HTP_FA_CLUSTER_OPP_WINDOW]  = W;
        fa->op_params[HTP_FA_CLUSTER_OPP_FLAGS]   = (int32_t) flags;
        fa->src[7] = on ? shadow : nullptr;
    };

    const size_t blk_bytes   = 2 * (size_t) 64 * o.d * 2;                       // K + V of one block of one head
    const size_t bytes_dense = (size_t) (o.kv / 64) * o.nkvh * blk_bytes;
    const size_t bytes_list  = pages_total * blk_bytes + (size_t) (W / 64) * o.nkvh * blk_bytes;
    printf("cluster mode: kv %d, window %d, %d candidate pages per head, density %d%% %s, skew %.2f -> %zu listed pages (mean %.1f/head), nslots %d, desc %s, perm %s, mask %d permille\n",
           o.kv, W, n_cand, o.density, o.sel_scatter ? "scattered" : "contiguous", o.skew, pages_total, (double) pages_total / o.nkvh,
           o.nslots, o.desc1d ? "1d" : "2d", o.perm_identity ? "identity" : "random", o.mask_frac_permille);

    struct arm_res { stat_acc wall, usec, axi; double worst = 0; };
    auto run_arm = [&](const char * name, bool on, arm_res & r) {
        set_mode(on);
        for (int it = -1; it < o.iters; ++it) {
            memset(base + L.dst, 0, (size_t) o.nh * o.d * 4);
            g_prof_lines.clear();
            const uint64_t t0 = cnt_now();
            ggml_backend_graph_compute(be, g);
            const uint64_t t1 = cnt_now();
            dc_civac(base + L.dst, (size_t) o.nh * o.d * 4);
            if (it < 0) continue;
            r.wall.add(ticks_us((double) (t1 - t0)));
            double us = 0, ax = 0;
            if (prof_last(&us, &ax)) { r.usec.add(us); r.axi.add(ax); }
        }
        double worst = 0;
        for (int h = 0; h < o.nh; ++h) {
            std::vector<std::pair<int, bool>> keys;
            if (on) {
                for (uint16_t p : lists[h / G]) for (int i = 0; i < 64; ++i) keys.push_back({ (int) pos_map[p * 64 + i], false });
                for (int p = covered_end; p < o.kv; ++p) keys.push_back({ p, true });
            } else {
                for (int p = 0; p < o.kv; ++p) keys.push_back({ p, true });
            }
            std::vector<double> ref;
            ref_attn_keys(o, base, L, h, keys, ref);
            const float * out = (const float *) (base + L.dst + (size_t) h * o.d * 4);
            for (int dd = 0; dd < o.d; ++dd) worst = std::max(worst, fabs((double) out[dd] - ref[dd]));
        }
        r.worst = worst;
        const size_t bytes = on ? bytes_list : bytes_dense;
        const double gbs = r.usec.empty() ? 0.0 : bytes / (r.usec.med() * 1e-6) / 1e9;
        printf("  %-10s graph_compute med %7.1f us | FA op med %7.1f us (min %7.1f max %7.1f) | %6.2f MB -> %5.1f GB/s | AXI rd req med %.0f (%.0f B/req) | max|err| %.2e %s\n",
               name, r.wall.med(), r.usec.med(), r.usec.mn(), r.usec.mx(), bytes / 1048576.0, gbs,
               r.axi.med(), r.axi.med() > 0 ? bytes / r.axi.med() : 0.0, worst, worst < 2e-2 ? "OK" : "MISMATCH");
    };
    arm_res dense, list;
    run_arm("dense", false, dense);
    run_arm("page list", true, list);
    {
        bool ok = true;
        dc_civac(lb + hdr.off_echo_sel, (size_t) o.nkvh * hdr.host_sel_stride);
        for (int h = 0; h < o.nkvh; ++h) {
            const uint8_t * e = lb + hdr.off_echo_sel + (size_t) h * hdr.host_sel_stride;
            if (*(const uint32_t *) e != lists[h].size() || (lists[h].size() && memcmp(e + 128, lists[h].data(), lists[h].size() * 2))) ok = false;
        }
        printf("  echoed lists %s; page list vs dense FA op: %.2fx time, %.2fx bytes\n", ok ? "match the host lists" : "DO NOT match (page path not taken?)",
               dense.usec.empty() || list.usec.empty() ? 0.0 : list.usec.med() / dense.usec.med(), (double) bytes_list / bytes_dense);
        if (!ok) { ggml_free(ctx); return 3; }
    }
    ggml_free(ctx);
    return (dense.worst < 2e-2 && list.worst < 2e-2) ? 0 : 2;
}

int main(int argc, char ** argv) {
    options o;
    if (!parse(argc, argv, o)) return 1;
    setvbuf(stdout, nullptr, _IONBF, 0);
    setenv("GGML_HEXAGON_SYNC_PROBE", "1", 1);
    setenv("GGML_HEXAGON_ASYNC", "1", 1);
    setenv("GGML_HEXAGON_FA_SELECT", "1", 0);   // HVX path (the decode kernel lives there)
    if (o.integrated) {
        char fr[32]; snprintf(fr, sizeof(fr), "%.3f", o.gpu_frac);
        if (o.gpu_frac > 0) setenv("GGML_HEXAGON_HETERO_FRAC", fr, 1); else unsetenv("GGML_HEXAGON_HETERO_FRAC");
        setenv("GGML_HEXAGON_HETERO_SPAN", std::to_string(o.span).c_str(), 1);
        setenv("GGML_HEXAGON_ASYNC", "0", 1);
    }
    if (o.cluster) {
        setenv("GGML_HEXAGON_ASYNC", "0", 1);
        setenv("GGML_HEXAGON_PROFILE", o.pmu ? "0x3,0x41,0xce,0x43,0xcf,0x7d,0x8c,0x40" : "1", 1);
        unsetenv("GGML_HEXAGON_HETERO_FRAC");
        unsetenv("GGML_HEXAGON_CLUSTER_ATTN");   // the tool tags its own node
        ggml_log_set(prof_log_cb, nullptr);
    }
    if (o.cpu >= 0) { cpu_set_t set; CPU_ZERO(&set); CPU_SET(o.cpu, &set); if (sched_setaffinity(0, sizeof(set), &set)) perror("sched_setaffinity"); }
    calibrate_realtime();

    const int gpu_kv = ((int) (o.gpu_frac * o.kv) / 64) * 64;   // GPU owns [0, gpu_kv), HTP owns [gpu_kv, kv)
    const int htp_kv = o.kv - gpu_kv;
    const int nsplit = std::max(1, (gpu_kv + o.span - 1) / o.span);

    layout L;
    L.nbk2 = (size_t) o.d * 2;
    L.nbk1 = L.nbk2 * o.nkvh;
    L.part_stride = 128 + (size_t) o.d * 4;
    const size_t kv_bytes = (size_t) o.kv * L.nbk1;
    htp_fa_cluster_header shadow_hdr = {};
    size_t shadow_bytes = 0;
    if (o.cluster) {
        // one layer's shadow at the front of the buffer; the positional data moves up behind it
        shadow_bytes = (size_t) htp_fa_cluster_layout(&shadow_hdr, 1, (uint32_t) o.kv, (uint32_t) o.nkvh, (uint32_t) o.d);
        shadow_bytes = ((shadow_bytes + (1 << 20) - 1) >> 20) << 20;
        L.q += shadow_bytes; L.mask += shadow_bytes; L.dst += shadow_bytes; L.parts += shadow_bytes; L.k += shadow_bytes;
    }
    L.k += (size_t) o.pad_mb << 20;
    L.v = L.k + kv_bytes;
    L.total = ((L.v + kv_bytes + (1 << 20) - 1) >> 20) << 20;
    if ((size_t) o.nh * nsplit * L.part_stride > L.parts_size) { fprintf(stderr, "too many GPU splits for the partial area\n"); return 1; }

    // backend + buffer
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) if (!strcmp(ggml_backend_dev_name(ggml_backend_dev_get(i)), "HTP0")) dev = ggml_backend_dev_get(i);
    if (!dev) { fprintf(stderr, "HTP0 not found\n"); return 1; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), L.total);
    if (!buf) { fprintf(stderr, "buffer alloc (%zu MB) failed\n", L.total >> 20); return 1; }
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
    const int fd = rpcmem_to_fd(base);

    // data
    {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        float * q = (float *) (base + L.q);
        for (int i = 0; i < o.nh * o.d; ++i) q[i] = u(rng);
        ggml_fp16_t * m = (ggml_fp16_t *) (base + L.mask);
        std::uniform_int_distribution<int> pm(0, 999);
        for (int p = 0; p < o.kv; ++p) m[p] = ggml_fp32_to_fp16(pm(rng) < o.mask_frac_permille ? -INFINITY : 0.0f);
        ggml_fp16_t * k = (ggml_fp16_t *) (base + L.k);
        ggml_fp16_t * v = (ggml_fp16_t *) (base + L.v);
        const size_t n = kv_bytes / 2;
        for (size_t i = 0; i < n; ++i) { k[i] = ggml_fp32_to_fp16(u(rng)); v[i] = ggml_fp32_to_fp16(u(rng)); }
        memset(base + L.parts, 0, L.parts_size);
        memset(base, 0, L.flags_end);
        dc_cvac(base, L.total);
    }

    if (o.cluster) {
        const int rc = run_cluster(o, be, buf, base, L, shadow_hdr, shadow_bytes);
        ggml_backend_buffer_free(buf); ggml_backend_free(be);
        return rc;
    }

    // HTP graph: probe(handshake) -> FA over [gpu_kv, kv) -> probe(ping)
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(16, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * flags = place(buf, base, 0, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (L.flags_end / 4)), "flags");
    ggml_tensor * fargs[1] = { flags };
    ggml_tensor * probe1 = place(buf, base, L.rec1, ggml_custom_4d(ctx, GGML_TYPE_F32, 64, 1, 1, 1, fargs, 1, noop_custom, 1, nullptr), "probe1");
    ggml_tensor * probe2 = place(buf, base, L.rec2, ggml_custom_4d(ctx, GGML_TYPE_F32, 64, 1, 1, 1, fargs, 1, noop_custom, 1, nullptr), "probe2");
    set_probe_params(probe1, 0, L, 2000);
    set_probe_params(probe2, 1, L, 2000);

    ggml_tensor * q  = place(buf, base, L.q, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, o.d, 1, o.nh, 1), "q");
    ggml_tensor * kb = place(buf, base, L.k, ggml_new_tensor_1d(ctx, GGML_TYPE_F16, (int64_t) (kv_bytes / 2)), "kbase");
    ggml_tensor * vb = place(buf, base, L.v, ggml_new_tensor_1d(ctx, GGML_TYPE_F16, (int64_t) (kv_bytes / 2)), "vbase");
    ggml_tensor * mb = place(buf, base, L.mask, ggml_new_tensor_1d(ctx, GGML_TYPE_F16, o.kv), "mbase");
    ggml_tensor * fa = nullptr;
    if (htp_kv > 0) {
        ggml_tensor * kh = ggml_view_4d(ctx, kb, o.d, htp_kv, o.nkvh, 1, L.nbk1, L.nbk2, kv_bytes, (size_t) gpu_kv * L.nbk1);
        ggml_tensor * vh = ggml_view_4d(ctx, vb, o.d, htp_kv, o.nkvh, 1, L.nbk1, L.nbk2, kv_bytes, (size_t) gpu_kv * L.nbk1);
        ggml_tensor * mh = ggml_view_2d(ctx, mb, htp_kv, 1, (size_t) htp_kv * 2, (size_t) gpu_kv * 2);
        kh->buffer = vh->buffer = mh->buffer = buf;
        ggml_set_name(kh, "k_htp"); ggml_set_name(vh, "v_htp"); ggml_set_name(mh, "mask_htp");
        fa = ggml_flash_attn_ext(ctx, q, kh, vh, mh, 1.0f / sqrtf((float) o.d), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        place(buf, base, L.dst, fa, "fa_out");
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, probe1);
    if (fa) ggml_build_forward_expand(graph, fa);
    ggml_build_forward_expand(graph, probe2);
    if (!ggml_backend_supports_op(be, probe1) || (fa && !ggml_backend_supports_op(be, fa))) {
        fprintf(stderr, "HTP0 rejects the probe or FA op (probe %d, fa %d)\n", ggml_backend_supports_op(be, probe1), fa ? ggml_backend_supports_op(be, fa) : -1);
        return 1;
    }

    if (o.integrated) {
        // Full-range FA op; the backend tags it (src[7], op_params[8..10]) and runs the GPU share itself.
        ggml_tensor * kf = ggml_view_4d(ctx, kb, o.d, o.kv, o.nkvh, 1, L.nbk1, L.nbk2, kv_bytes, 0);
        ggml_tensor * vf = ggml_view_4d(ctx, vb, o.d, o.kv, o.nkvh, 1, L.nbk1, L.nbk2, kv_bytes, 0);
        ggml_tensor * mf = ggml_view_2d(ctx, mb, o.kv, 1, (size_t) o.kv * 2, 0);
        kf->buffer = vf->buffer = mf->buffer = buf;
        ggml_tensor * fa_full = ggml_flash_attn_ext(ctx, q, kf, vf, mf, 1.0f / sqrtf((float) o.d), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa_full, GGML_PREC_F32);
        place(buf, base, L.dst, fa_full, "fa_full");
        ggml_cgraph * gfull = ggml_new_graph_custom(ctx, 8, false);
        ggml_build_forward_expand(gfull, fa_full);
        if (!ggml_backend_supports_op(be, fa_full)) { fprintf(stderr, "HTP0 rejects the full-range FA op\n"); return 1; }
        printf("integrated: full-range FA on HTP0 with GGML_HEXAGON_HETERO_FRAC=%s, kv %d\n", o.gpu_frac > 0 ? getenv("GGML_HEXAGON_HETERO_FRAC") : "(off)", o.kv);
        stat_acc s_t;
        double worst = 0;
        for (int it = -1; it < o.iters; ++it) {
            memset(base + L.dst, 0, (size_t) o.nh * o.d * 4);
            const uint64_t t0 = cnt_now();
            ggml_backend_graph_compute(be, gfull);
            const uint64_t t1 = cnt_now();
            dc_civac(base + L.dst, (size_t) o.nh * o.d * 4);
            double w = 0;
            for (int h = 0; h < o.nh; ++h) {
                std::vector<double> ref; double Mref, Sref;
                ref_attn(o, base, L, h, 0, o.kv, ref, &Mref, &Sref);
                const float * out = (const float *) (base + L.dst + (size_t) h * o.d * 4);
                for (int dd = 0; dd < o.d; ++dd) w = std::max(w, fabs((double) out[dd] - (Sref > 0 ? ref[dd] / Sref : 0.0)));
            }
            worst = std::max(worst, w);
            if (it >= 0) s_t.add(ticks_us((double) (t1 - t0)));
            if (o.verbose) printf("  [integrated #%d] graph_compute %.1f us  max|err| %.3e\n", it, ticks_us((double) (t1 - t0)), w);
        }
        printf("integrated: max |err| vs CPU reference over %d runs: %.3e -> %s; graph_compute median %.1f us (min %.1f max %.1f)\n",
               o.iters, worst, worst < 2e-2 ? "OK" : "MISMATCH", s_t.med(), s_t.mn(), s_t.mx());
        ggml_free(ctx); ggml_backend_buffer_free(buf); ggml_backend_free(be);
        return worst < 2e-2 ? 0 : 2;
    }

    gpu_side g;
    const bool need_gpu = (o.run_gpu || o.run_both) && gpu_kv > 0;
    if (need_gpu && !gpu_init(g, o, base, L.total, fd)) return 1;

    printf("decode attention split: kv %d = GPU [0,%d) (%d splits of %d) + HTP [%d,%d) | nh %d nkvh %d d %d | K+V bytes: GPU %.1f MB, HTP %.1f MB\n",
           o.kv, gpu_kv, nsplit, o.span, gpu_kv, o.kv, o.nh, o.nkvh, o.d,
           2.0 * gpu_kv * L.nbk1 / 1048576.0, 2.0 * htp_kv * L.nbk1 / 1048576.0);

    volatile uint32_t * ready = (volatile uint32_t *) (base + L.ready);
    volatile uint32_t * done  = (volatile uint32_t *) (base + L.done);
    uint64_t * rec1 = (uint64_t *) (base + L.rec1);
    uint64_t * rec2 = (uint64_t *) (base + L.rec2);

    auto gpu_enqueue = [&](cl_uint want, cl_event * ev, cl_uint svm_off = 0, cl_uint max_spin = 2000000000u) {
        cl_uint a_q = L.q, a_k = L.k, a_v = L.v, a_m = L.mask, a_p = L.parts;
        cl_uint a_nbq2 = o.d * 4, a_nbk1 = L.nbk1, a_nbk2 = L.nbk2, a_nkv = gpu_kv, a_span = o.span, a_ns = nsplit, a_ps = L.part_stride;
        cl_float a_scale = 1.0f / sqrtf((float) o.d);
        int i = 0;
        clSetKernelArg(g.k_fa, i++, sizeof(cl_mem), &g.alias);
        clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &a_q); clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &a_k);
        clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &a_v); clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &a_m);
        clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &a_p);
        clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &a_nbq2);
        clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &a_nbk1); clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &a_nbk2);
        clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &a_nbk1); clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &a_nbk2);
        clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &a_nkv); clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &a_span);
        clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &a_ns); clSetKernelArg(g.k_fa, i++, sizeof(cl_float), &a_scale);
        clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &a_ps);
        clSetKernelArgSVMPointer(g.k_fa, i++, g.svm + svm_off);
        clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &want);
        clSetKernelArg(g.k_fa, i++, sizeof(cl_uint), &max_spin);
        size_t gsz[3] = { (size_t) o.d, (size_t) o.nkvh, (size_t) nsplit }, lsz[3] = { (size_t) o.d, 1, 1 };
        return clEnqueueNDRangeKernel(g.q, g.k_fa, 3, nullptr, gsz, lsz, 0, nullptr, ev);
    };

    // Check GPU partials once against the CPU reference over the GPU range (merged on host).
    auto check_gpu = [&]() -> bool {
        double worst = 0;
        for (int h = 0; h < o.nh; ++h) {
            std::vector<double> ref; double Mref, Sref;
            ref_attn(o, base, L, h, 0, gpu_kv, ref, &Mref, &Sref);
            double M = -INFINITY;
            for (int s = 0; s < nsplit; ++s) M = std::max(M, (double) ((float *) (base + L.parts + ((size_t) h * nsplit + s) * L.part_stride))[0]);
            std::vector<double> acc(o.d, 0.0); double S = 0;
            for (int s = 0; s < nsplit; ++s) {
                const float * part = (const float *) (base + L.parts + ((size_t) h * nsplit + s) * L.part_stride);
                const double d = (double) part[0] - M;
                if (d < -80.0) continue;
                const double w = exp(d);
                S += w * part[1];
                for (int dd = 0; dd < o.d; ++dd) acc[dd] += w * part[32 + dd];
            }
            for (int dd = 0; dd < o.d; ++dd) {
                const double got = S > 0 ? acc[dd] / S : 0.0, want = Sref > 0 ? ref[dd] / Sref : 0.0;
                worst = std::max(worst, fabs(got - want));
            }
        }
        printf("gpu: partials merged on host vs CPU reference over [0,%d): max |err| %.3e -> %s\n", gpu_kv, worst, worst < 2e-3 ? "OK" : "MISMATCH");
        return worst < 2e-3;
    };
    auto check_htp = [&]() -> bool {
        double worst = 0;
        for (int h = 0; h < o.nh; ++h) {
            std::vector<double> ref; double Mref, Sref;
            ref_attn(o, base, L, h, gpu_kv, o.kv, ref, &Mref, &Sref);
            const float * out = (const float *) (base + L.dst + (size_t) h * o.d * 4);
            for (int dd = 0; dd < o.d; ++dd) worst = std::max(worst, fabs((double) out[dd] - (Sref > 0 ? ref[dd] / Sref : 0.0)));
        }
        printf("htp: FA output vs CPU reference over [%d,%d): max |err| %.3e -> %s\n", gpu_kv, o.kv, worst, worst < 2e-2 ? "OK" : "MISMATCH");
        return worst < 2e-2;
    };

    enum { R_HTP, R_GPU, R_BOTH, R_N };
    const char * rname[R_N] = { "htp alone", "gpu alone", "both" };
    stat_acc s_htp[R_N], s_gpu[R_N], s_gpu_from_start[R_N], s_wall[R_N];
    bool checked_gpu = false, checked_htp = false;
    const size_t htp_bytes = 2 * (size_t) htp_kv * L.nbk1, gpu_bytes = 2 * (size_t) gpu_kv * L.nbk1;

    // Sustained token mode: pre-enqueue N GPU decode kernels on the in-order queue, each spinning
    // on its own SVM ready word; loop over N "layers" running the HTP FA op and relaying one word
    // per layer. Tests whether a chain of spinning kernels holds up, and whether per-layer time
    // drifts across a whole token (thermal / driver). Reuses the parts/dst regions each layer, so
    // the last layer's outputs remain for a correctness check.
    auto run_token = [&]() {
        const int N = o.layers;
        if ((size_t) N * 4 + 8 > 1024) { fprintf(stderr, "too many layers for the 4KB svm block\n"); return; }
        if (!(need_gpu && fa)) { fprintf(stderr, "token mode needs both htp and gpu (kv split with gpu-frac in (0,1))\n"); return; }
        stat_acc lay_htp, lay_gpu, tok_htp_wall, tok_gpu_wall, drift;
        std::vector<cl_event> evs(N);
        int alive_total = 0, alive_runs = 0;
        for (int pass = -1; pass < o.iters; ++pass) {
            for (int i = 0; i < N; ++i) { __atomic_store_n(&g.svm[i * 4], 0u, __ATOMIC_SEQ_CST); __atomic_store_n(&g.svm[i * 4 + 1], 0u, __ATOMIC_SEQ_CST); }
            memset(base + L.parts, 0, L.parts_size); dc_cvac(base + L.parts, L.parts_size);
            for (int i = 0; i < N; ++i) { evs[i] = nullptr; if (!cl_check(gpu_enqueue(1, &evs[i], (cl_uint) (i * 4), 2000000000u), "chain enqueue")) return; }
            clFlush(g.q);
            std::vector<double> htp_layer(N, 0.0);
            const uint64_t T0 = cnt_now();
            for (int i = 0; i < N; ++i) {
                *ready = 0; *done = 0; memset(rec1, 0, 512); memset(rec2, 0, 512); dc_cvac(base, L.flags_end);
                ggml_backend_graph_compute_async(be, graph);
                const uint64_t t0 = cnt_now();
                for (;;) { dc_civac((const void *) ready, 4); if (__atomic_load_n(ready, __ATOMIC_ACQUIRE) == 1u) break; if (cnt_now() - t0 > 3000ull * 19200ull) { fprintf(stderr, "layer %d ready timeout\n", i); return; } }
                __atomic_store_n(&g.svm[i * 4], 1u, __ATOMIC_SEQ_CST);   // release GPU kernel i
                __atomic_store_n(done, 1u, __ATOMIC_RELEASE);
                ggml_backend_synchronize(be);
                dc_civac(rec1, 512); dc_civac(rec2, 512);
                const uint64_t fa_start = rec1[HTP_SYNC_PROBE_R_T_DONE], fa_end = rec2[HTP_SYNC_PROBE_R_T_ENTRY];
                if (rec1[HTP_SYNC_PROBE_R_STATUS] != 0 || !fa_end) { fprintf(stderr, "layer %d probe bad\n", i); return; }
                htp_layer[i] = ticks_us((double) (fa_end - fa_start));
            }
            const uint64_t T_htp_end = cnt_now();
            clFinish(g.q);
            double first_start = 1e30, last_end = 0;
            std::vector<double> gpu_layer(N, 0.0);
            for (int i = 0; i < N; ++i) {
                cl_ulong ts = 0, te = 0;
                clGetEventProfilingInfo(evs[i], CL_PROFILING_COMMAND_START, sizeof(ts), &ts, nullptr);
                clGetEventProfilingInfo(evs[i], CL_PROFILING_COMMAND_END, sizeof(te), &te, nullptr);
                const double se = cnt_of_realtime(ts), ee = cnt_of_realtime(te);
                gpu_layer[i] = ticks_us(ee - se);
                first_start = std::min(first_start, se); last_end = std::max(last_end, ee);
                clReleaseEvent(evs[i]);
            }
            int alive = 0; for (int i = 0; i < N; ++i) alive += (__atomic_load_n(&g.svm[i * 4 + 1], __ATOMIC_ACQUIRE) == 1u);
            if (pass < 0) continue;
            alive_total += alive; alive_runs++;
            tok_htp_wall.add(ticks_us((double) (T_htp_end - T0)));
            tok_gpu_wall.add(ticks_us(last_end - first_start));
            double f3 = 0, l3 = 0; int nf = 0, nl = 0;
            for (int i = 0; i < N; ++i) { lay_htp.add(htp_layer[i]); lay_gpu.add(gpu_layer[i]); if (i < N / 3) { f3 += htp_layer[i]; nf++; } if (i >= 2 * N / 3) { l3 += htp_layer[i]; nl++; } }
            if (nf && nl && f3 > 0) drift.add((l3 / nl) / (f3 / nf));
        }
        dc_civac(base + L.parts, L.parts_size);
        const bool ok = check_gpu();
        printf("\nsustained token: %d layers x %d passes | GPU chain: %d/%d kernels ran per pass | last-pass partials %s\n",
               N, o.iters, alive_runs ? alive_total / alive_runs : 0, N, ok ? "OK" : "MISMATCH");
        printf("  %-32s %9.1f %9.1f %9.1f us  (median/min/max over %zu layer-instances)\n", "HTP FA op per layer", lay_htp.med(), lay_htp.mn(), lay_htp.mx(), lay_htp.v.size());
        printf("  %-32s %9.1f %9.1f %9.1f us\n", "GPU kernel per layer", lay_gpu.med(), lay_gpu.mn(), lay_gpu.mx());
        printf("  %-32s %9.1f us   GPU busy span %.1f us\n", "HTP token wall (attn only)", tok_htp_wall.med(), tok_gpu_wall.med());
        printf("  %-32s %.3f  (>~1.05 would mean thermal/driver drift over the token)\n", "per-layer HTP drift (last/first 3rd)", drift.med());
    };

    if (o.layers > 1) { run_token(); } else {
    for (int it = -1; it < o.iters; ++it) {
        for (int r = 0; r < R_N; ++r) {
            const bool use_htp = (r != R_GPU) && (o.run_htp || r == R_BOTH) && fa;
            const bool use_gpu = (r != R_HTP) && need_gpu;
            if (r == R_HTP && !o.run_htp) continue;
            if (r == R_GPU && !o.run_gpu) continue;
            if (r == R_BOTH && !o.run_both) continue;
            if (!use_htp && !use_gpu) continue;

            *ready = 0; *done = 0; memset(rec1, 0, 512); memset(rec2, 0, 512);
            dc_cvac(base, L.flags_end);
            cl_event ev = nullptr;
            if (use_gpu) {
                __atomic_store_n(&g.svm[0], 0u, __ATOMIC_SEQ_CST); __atomic_store_n(&g.svm[1], 0u, __ATOMIC_SEQ_CST);
                if (!cl_check(gpu_enqueue(1, &ev), "enqueue")) return 1;
                clFlush(g.q);
                const uint64_t t0 = cnt_now();
                while (__atomic_load_n(&g.svm[1], __ATOMIC_ACQUIRE) != 1u && cnt_now() - t0 < 500ull * 19200ull) {}
            }
            uint64_t T_rel = 0, T1 = 0;
            if (use_htp) {
                ggml_backend_graph_compute_async(be, graph);
                const uint64_t t0 = cnt_now();
                for (;;) {
                    dc_civac((const void *) ready, 4);
                    if (__atomic_load_n(ready, __ATOMIC_ACQUIRE) == 1u) break;
                    if (cnt_now() - t0 > 2000ull * 19200ull) { fprintf(stderr, "probe ready never seen\n"); return 1; }
                }
                if (use_gpu) __atomic_store_n(&g.svm[0], 1u, __ATOMIC_SEQ_CST);
                __atomic_store_n(done, 1u, __ATOMIC_RELEASE);
                T_rel = cnt_now();
                ggml_backend_synchronize(be);
                T1 = cnt_now();
            } else {
                T_rel = cnt_now();
                __atomic_store_n(&g.svm[0], 1u, __ATOMIC_SEQ_CST);
            }
            double gpu_start = 0, gpu_end = 0;
            if (use_gpu) {
                clWaitForEvents(1, &ev);
                cl_ulong ts = 0, te = 0;
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(ts), &ts, nullptr);
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(te), &te, nullptr);
                gpu_start = cnt_of_realtime(ts); gpu_end = cnt_of_realtime(te);
                clReleaseEvent(ev);
                if (!checked_gpu) { dc_civac(base + L.parts, L.parts_size); checked_gpu = check_gpu(); }
            }
            double htp_us = NAN, gpu_us = NAN, gpu_from_start_us = NAN, wall_us = NAN;
            if (use_htp) {
                dc_civac(rec1, 512); dc_civac(rec2, 512);
                const uint64_t fa_start = rec1[HTP_SYNC_PROBE_R_T_DONE], fa_end = rec2[HTP_SYNC_PROBE_R_T_ENTRY];
                if (rec1[HTP_SYNC_PROBE_R_STATUS] != 0 || !fa_end) { fprintf(stderr, "probe timeout / bad record\n"); return 1; }
                htp_us = ticks_us((double) (fa_end - fa_start));
                if (!checked_htp) { dc_civac(base + L.dst, o.nh * o.d * 4); checked_htp = check_htp(); }
                if (use_gpu) {
                    // the GPU was released ~1 us before the FA started; both measured from the release
                    gpu_from_start_us = ticks_us(gpu_end - (double) T_rel);
                    wall_us = std::max(htp_us, gpu_from_start_us);
                }
            }
            if (use_gpu) {
                gpu_us = ticks_us(gpu_end - std::max(gpu_start, (double) T_rel));   // pre-launched: time after release
                if (!use_htp) { gpu_from_start_us = ticks_us(gpu_end - (double) T_rel); wall_us = gpu_from_start_us; }
            }
            if (!use_gpu) wall_us = htp_us;
            if (it < 0) continue;   // warm-up
            if (use_htp) s_htp[r].add(htp_us);
            if (use_gpu) { s_gpu[r].add(gpu_us); s_gpu_from_start[r].add(gpu_from_start_us); }
            s_wall[r].add(wall_us);
            if (o.verbose) printf("  [%s #%d] htp %.1f us  gpu %.1f us (release->end %.1f)  wall %.1f us  host total %.1f us\n", rname[r], it, htp_us, gpu_us, gpu_from_start_us, wall_us, T1 ? ticks_us((double) (T1 - T_rel)) : NAN);
        }
    }

    printf("\n%-12s | %-28s %9s %9s %9s   %s\n", "run", "metric", "median", "min", "max", "GB/s (median)");
    for (int r = 0; r < R_N; ++r) {
        if (s_wall[r].empty()) continue;
        printf("%-12s |\n", rname[r]);
        if (!s_htp[r].empty()) printf("%-12s | %-28s %9.1f %9.1f %9.1f   %.1f\n", "", "HTP FA op (qtimer)", s_htp[r].med(), s_htp[r].mn(), s_htp[r].mx(), htp_bytes / (s_htp[r].med() * 1e-6) / 1e9);
        if (!s_gpu[r].empty()) printf("%-12s | %-28s %9.1f %9.1f %9.1f   %.1f\n", "", "GPU kernel after release", s_gpu[r].med(), s_gpu[r].mn(), s_gpu[r].mx(), gpu_bytes / (s_gpu[r].med() * 1e-6) / 1e9);
        if (!s_gpu_from_start[r].empty()) printf("%-12s | %-28s %9.1f %9.1f %9.1f\n", "", "GPU release -> end", s_gpu_from_start[r].med(), s_gpu_from_start[r].mn(), s_gpu_from_start[r].mx());
        printf("%-12s | %-28s %9.1f %9.1f %9.1f   %.1f\n", "", "attention wall (max of both)", s_wall[r].med(), s_wall[r].mn(), s_wall[r].mx(), (double) (htp_bytes + gpu_bytes) / (s_wall[r].med() * 1e-6) / 1e9);
    }
    if (!s_htp[R_HTP].empty() && !s_htp[R_BOTH].empty()) printf("\ncontention: HTP x%.2f slower with the GPU streaming", s_htp[R_BOTH].med() / s_htp[R_HTP].med());
    if (!s_gpu[R_GPU].empty() && !s_gpu[R_BOTH].empty()) printf(", GPU x%.2f slower with the HTP streaming", s_gpu[R_BOTH].med() / s_gpu[R_GPU].med());
    printf("\n");
    } // end single-shot

    if (g.svm) clSVMFree(g.ctx, g.svm);
    if (g.alias) clReleaseMemObject(g.alias);
    if (g.k_fa) clReleaseKernel(g.k_fa);
    if (g.prog) clReleaseProgram(g.prog);
    if (g.q) clReleaseCommandQueue(g.q);
    if (g.ctx) clReleaseContext(g.ctx);
    ggml_free(ctx); ggml_backend_buffer_free(buf); ggml_backend_free(be);
    return 0;
}

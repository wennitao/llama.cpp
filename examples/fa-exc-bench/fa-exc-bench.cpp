// llama-fa-exc-bench: the GPU exception kernel of the heterogeneous prefill split (fa_exc), timed alone.
//
// The kernels are the production ones: the OpenCL source is the raw string ggml_hexagon_hfold_cl_src of
// ggml/src/ggml-hexagon/ggml-hexagon.cpp, extracted at build time (extract-cl.cmake) or read at run time (--cl-src).
// Every argument is set as hfold_enqueue_chain / hfold_args_fa set it in the deployed mode (persistent work-groups,
// chain 2, GPU done words, (query tile, KV head) stages, f16 accumulator), over buffers laid out as the deployed ones:
// Q [128, Sq, 24] f32 permuted, K/V [128, 8, kv] f16 (one layer of the cache), the f16 causal mask, the f32 membership
// src[8] and the fold buffer (ggml_hexagon_hfold_ensure_buffer's layout). With --mem ion (default on Android) they are
// rpcmem buffers of the HTP0 buffer type seen through ION aliases with the host-IO-coherent policy, as deployed.
//
// Per KV length it draws exception lists per (KV head, 64-row sub-block) with the deployed op's statistics (lists empty
// on the head start's KV heads 0-1; elsewhere gamma-distributed lengths with a mean of --exc-frac of the op's blocks;
// the sub-block's own diagonal block, which c* = 3 always leaves to the GPU, included for sub-blocks 2-3 of a tile),
// runs the production stage_k_p (tables + K^T) and checks it, then times fa_exc for every variant (the
// GGML_HEXAGON_FA_FOLD_EXC bits, built as -D options) with device timestamps, arms interleaved in alternating order.
// Every variant's (m, l, acc) partial is compared bit for bit with the plain kernel's, and sampled rows against a
// float64 reference of attention over the exception blocks.

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "htp-ops.h"

#ifndef FA_EXC_BENCH_NO_HTP
#include "ggml-backend.h"
extern "C" int rpcmem_to_fd(void * po);
#endif

#include "hfold-cl-src.inc"   // fa_exc_bench_cl_src, generated from ggml-hexagon.cpp

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define DIE(...)      do { fprintf(stderr, "fa-exc-bench: " __VA_ARGS__); fputc('\n', stderr); exit(1); } while (0)
#define CHECK(c, ...) do { if (!(c)) DIE(__VA_ARGS__); } while (0)
#define CL_CHECK(e, what) do { cl_int _e = (e); if (_e != CL_SUCCESS) DIE("%s failed (%d)", (what), _e); } while (0)

static constexpr int D = 128, BK = 64;
static constexpr int EXC_TEX128 = 1, EXC_FSM = 2, EXC_QIMG = 4, EXC_P16 = 8, EXC_QK2B = 16;   // GGML_HEXAGON_FA_FOLD_EXC bits

// ---------------------------------------------------------------------------------------------------------------

struct options {
    std::vector<int> kv = { 1024, 2048, 3072, 4096, 5120, 6144, 7168 };
    int    sq = 1024, nh = 24, nkvh = 8, br = 256, kv_cap = 8192;
    double exc_frac = 0.075, len_shape = 2.0, qscale = 2.0;
    int    empty_kvh = 2, fixed_len = -1;
    std::vector<int> variants = { 0, 1, 2, 4, 8, 16, 3, 5, 9, 17, 21 };
    int bitref = 0;                    // the variant every arm's partial is compared with bit for bit
    std::vector<int> pwg = { 32 };
    std::vector<int> lds_pad;          // occupancy probe: each value adds -DFA_EXC_LDS_PAD=<bytes> arms of every variant
    int    pwg_stage = 64, iters = 20, warmup = 3, reps = 2, ref_rows = 64, seed = 1;
    bool   acc16 = true, empty_baseline = false, perf_hint = true, verbose = false, check_tables = true;
#ifdef FA_EXC_BENCH_NO_HTP
    std::string mem = "cl";
#else
    std::string mem = "ion";
#endif
    std::string cl_src_file, extra_opts, csv;
    int platform = 0, device = 0;
};

static void usage() {
    printf("usage: llama-fa-exc-bench [options]\n"
           "  --kv LIST          KV lengths, multiples of 64 and >= --sq (default 1024,2048,...,7168)\n"
           "  --sq N             query rows of the op (1024)      --nh N / --nkvh N   heads (24 / 8)\n"
           "  --br N             query tile rows per stage (256)  --kv-cap N          cache capacity for the K^T stride (8192)\n"
           "  --exc-frac F       mean exception blocks per list on the non-head-start KV heads, as a fraction of the op's blocks (0.075)\n"
           "  --len-shape K      gamma shape of the list lengths (2)   --fixed-len N   every list on those heads has N blocks\n"
           "  --empty-kvh N      KV heads 0..N-1 get no exceptions, as with the deployed head start (2)\n"
           "  --qscale F         std of Q entries (2; K and V are N(0,1)), sets the logit spread\n"
           "  --variants LIST    GGML_HEXAGON_FA_FOLD_EXC bit sets (1 TEX128, 2 FSM, 4 QIMG, 8 P16, 16 QK2B; 0 = plain)\n"
           "                     default 0,1,2,4,8,16,3,5,9,17,21\n"
           "  --bitref V         compare every arm bit for bit with variant V (default 0, the plain kernel)\n"
           "  --pwg LIST         persistent work-groups of fa_exc (32, as deployed)   --pwg-stage N  of stage_k_p (64)\n"
           "  --lds-pad LIST     occupancy probe: extra local bytes per work-group (-DFA_EXC_LDS_PAD), one arm per value and variant\n"
           "  --iters N          timed launches per arm and rep (20)  --warmup N (3)  --reps N (2, arm order alternates)\n"
           "  --mem ion|cl       rpcmem + ION alias as deployed (needs HTP0) or plain OpenCL buffers\n"
           "  --opts STR         extra build options for every arm   --cl-src FILE  kernels from FILE (ggml-hexagon.cpp or .cl)\n"
           "  --no-acc16         f32 partial accumulator (deployed: f16)\n"
           "  --empty-baseline   also time every arm with all lists empty: marginal us per pair = (t - t_empty) / pairs\n"
           "  --ref-rows N       rows per point checked against the float64 reference (64; 0 = off)\n"
           "  --no-check-tables  skip the host check of stage_k_p's tables and K^T\n"
           "  --no-perf-hint     no cl_qcom_perf_hint HIGH on the context (deployed: on)\n"
           "  --csv FILE         append one line per arm and KV   --seed N   --platform N --device N   -v\n");
}

static std::vector<int> parse_list(const char * s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back((int) strtol(tok.c_str(), nullptr, 0));
    return v;
}

// ---------------------------------------------------------------------------------------------------------------

static inline uint16_t f32_to_f16(float f) {   // round to nearest even
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t) (sign | 0x7c00u | (mant ? 0x200u : 0u));
    if (exp >= 31) return (uint16_t) (sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        mant |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t) (sign | h);
    }
    uint32_t h = ((uint32_t) exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;
    return (uint16_t) (sign | h);
}

static inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ffu, x;
    if (exp == 0) {
        if (mant == 0) x = sign;
        else { int e = -1; do { e++; mant <<= 1; } while (!(mant & 0x400u)); x = sign | ((uint32_t) (127 - 15 - e) << 23) | ((mant & 0x3ffu) << 13); }
    } else if (exp == 31) x = sign | 0x7f800000u | (mant << 13);
    else x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f; memcpy(&f, &x, 4); return f;
}

static inline void dc_civac(const void * p, size_t n) {
#if defined(__aarch64__)
    uintptr_t a = (uintptr_t) p & ~(uintptr_t) 63;
    const uintptr_t e = (uintptr_t) p + n;
    for (; a < e; a += 64) asm volatile("dc civac, %0" : : "r"(a) : "memory");
    asm volatile("dsb sy" : : : "memory");
#else
    (void) p; (void) n;
#endif
}

static size_t al(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

// ---------------------------------------------------------------------------------------------------------------
// OpenCL state and the shared buffers

struct cl_state {
    cl_platform_id plat = nullptr;
    cl_device_id   dev  = nullptr;
    cl_context     ctx  = nullptr;
    cl_command_queue q  = nullptr;
    size_t max_img_texels = 0;
    std::string ext;
    bool ion = false;
#ifndef FA_EXC_BENCH_NO_HTP
    ggml_backend_dev_t htp = nullptr;
    ggml_backend_t     htp_be = nullptr;
#endif
};

struct shared_buf {
    size_t    size = 0;
    uint8_t * host = nullptr;   // ion: the rpcmem mapping
    cl_mem    mem = nullptr, img_h = nullptr, img_u = nullptr;
#ifndef FA_EXC_BENCH_NO_HTP
    ggml_backend_buffer_t gbuf = nullptr;
#endif
};

static void buf_alloc(cl_state & cs, shared_buf & b, size_t size, const char * name) {
    b.size = al(size, 1u << 20);
    cl_int err = CL_SUCCESS;
    if (cs.ion) {
#ifndef FA_EXC_BENCH_NO_HTP
        b.gbuf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(cs.htp), b.size);
        CHECK(b.gbuf, "HTP0 buffer of %zu MB (%s) failed", b.size >> 20, name);
        b.host = (uint8_t *) ggml_backend_buffer_get_base(b.gbuf);
        const int fd = rpcmem_to_fd(b.host);
        CHECK(fd >= 0, "rpcmem_to_fd failed for %s", name);
        memset(b.host, 0, b.size);
        dc_civac(b.host, b.size);
        cl_mem_ion_host_ptr ion = {};
        ion.ext_host_ptr.allocation_type   = CL_MEM_ION_HOST_PTR_QCOM;
        ion.ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
        ion.ion_filedesc = fd;
        ion.ion_hostptr  = b.host;
        b.mem = clCreateBuffer(cs.ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM, b.size, &ion, &err);
        CL_CHECK(err, "ION alias");
#endif
    } else {
        b.mem = clCreateBuffer(cs.ctx, CL_MEM_READ_WRITE, b.size, nullptr, &err);
        CL_CHECK(err, "clCreateBuffer");
        const uint8_t zero = 0;
        CL_CHECK(clEnqueueFillBuffer(cs.q, b.mem, &zero, 1, 0, b.size, 0, nullptr, nullptr), "zero fill");
        CL_CHECK(clFinish(cs.q), "clFinish");
    }
    // whole-buffer views, as ggml_hexagon_hfold_image / _image_u32 make them
    cl_image_desc dk; memset(&dk, 0, sizeof(dk));
    dk.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    dk.buffer = b.mem;
    cl_image_format fh = { CL_RGBA, CL_HALF_FLOAT }, fu = { CL_RGBA, CL_UNSIGNED_INT32 };
    dk.image_width = b.size / 8;
    CHECK(dk.image_width <= cs.max_img_texels, "%s: %zu texels exceed the device's image buffer limit %zu", name, dk.image_width, cs.max_img_texels);
    b.img_h = clCreateImage(cs.ctx, CL_MEM_READ_WRITE, &fh, &dk, nullptr, &err); CL_CHECK(err, "RGBA half view");
    dk.image_width = b.size / 16;
    b.img_u = clCreateImage(cs.ctx, CL_MEM_READ_WRITE, &fu, &dk, nullptr, &err); CL_CHECK(err, "RGBA uint32 view");
}

static void buf_free(shared_buf & b) {
    if (b.img_u) clReleaseMemObject(b.img_u);
    if (b.img_h) clReleaseMemObject(b.img_h);
    if (b.mem) clReleaseMemObject(b.mem);
#ifndef FA_EXC_BENCH_NO_HTP
    if (b.gbuf) ggml_backend_buffer_free(b.gbuf);
#endif
    b = shared_buf();
}

// the GPU is idle when these are called
static void upload(cl_state & cs, shared_buf & b, size_t off, const void * src, size_t n) {
    if (cs.ion) { memcpy(b.host + off, src, n); dc_civac(b.host + off, n); }
    else CL_CHECK(clEnqueueWriteBuffer(cs.q, b.mem, CL_TRUE, off, n, src, 0, nullptr, nullptr), "write");
}
static void download(cl_state & cs, shared_buf & b, size_t off, void * dst, size_t n) {
    CL_CHECK(clFinish(cs.q), "clFinish");
    if (cs.ion) { dc_civac(b.host + off, n); memcpy(dst, b.host + off, n); }
    else CL_CHECK(clEnqueueReadBuffer(cs.q, b.mem, CL_TRUE, off, n, dst, 0, nullptr, nullptr), "read");
}

// ---------------------------------------------------------------------------------------------------------------
// the fold buffer, laid out as ggml_hexagon_hfold_ensure_buffer lays it out

struct fold_layout {
    size_t off_m, off_l, off_acc, off_ctl, off_idx, off_abs, off_cnt, off_lcnt, off_err, off_done, off_scnt, off_stamp, off_kt, size;
};

static fold_layout make_fold_layout(int rows, int nkvh, int nbk_cap, int num_sb_cap) {
    fold_layout L;
    const size_t ml  = al((size_t) rows * sizeof(float), 128);
    const size_t acc = al((size_t) rows * D * sizeof(float), 128);
    size_t off = 128;
    L.off_m = off; off += ml;
    L.off_l = off; off += ml;
    L.off_acc = off; off += acc;
    L.off_ctl = al(off, 4096); off = L.off_ctl + al(HTP_FA_HETERO_STATUS_OFF + HTP_FA_FOLD_ST_N * sizeof(uint32_t), 4096);
    L.off_idx  = off; off += al((size_t) nkvh * num_sb_cap * nbk_cap * sizeof(int32_t), 128);
    L.off_abs  = off; off += al((size_t) nkvh * nbk_cap * sizeof(int32_t), 128);
    L.off_cnt  = off; off += al((size_t) nkvh * sizeof(int32_t), 128);
    L.off_lcnt = off; off += al((size_t) nkvh * num_sb_cap * sizeof(int32_t), 128);
    L.off_err  = off; off += 256;
    L.off_done  = off; off += al(HTP_FA_FOLD_MAX_STAGES * sizeof(uint32_t), 128);
    L.off_scnt  = off; off += al((HTP_FA_FOLD_MAX_STAGES + 2) * sizeof(uint32_t), 128);
    L.off_stamp = off; off += al(HTP_FA_FOLD_MAX_STAGES * sizeof(uint32_t), 128);
    L.off_kt   = al(off, 4096); off = L.off_kt + (size_t) nkvh * D * ((size_t) nbk_cap * 64) * sizeof(uint16_t);
    L.size = al(off, 1u << 20);
    return L;
}

// ---------------------------------------------------------------------------------------------------------------
// programs: one per build-option set

struct arm {
    int variant = 0, lds_pad = 0;
    std::string opts, name;
    cl_program prog = nullptr;
    cl_kernel  k_stage_p = nullptr, k_fa = nullptr;
    cl_uint    fa_nargs = 0;
};

static std::string arm_name(int v, int pad) {
    std::string s = v == 0 ? "plain" : "";
    if (v & EXC_TEX128) s += s.empty() ? "tex128" : "+tex128";
    if (v & EXC_FSM)    s += s.empty() ? "fsm" : "+fsm";
    if (v & EXC_QIMG)   s += s.empty() ? "qimg" : "+qimg";
    if (v & EXC_P16)    s += s.empty() ? "p16" : "+p16";
    if (v & EXC_QK2B)   s += s.empty() ? "qk2b" : "+qk2b";
    if (pad) s += "/lds+" + std::to_string(pad);
    return s;
}

static void build_arm(cl_state & cs, const std::string & src, arm & a, const options & o) {
    a.opts = "-cl-std=CL2.0 -cl-fast-relaxed-math -DFA_D=128";
    if (o.acc16) a.opts += " -DACC_F16=1";
    if (a.variant & EXC_TEX128) a.opts += " -DFA_EXC_TEX128=1";
    if (a.variant & EXC_FSM)    a.opts += " -DFA_EXC_FSM=1";
    if (a.variant & EXC_QIMG)   a.opts += " -DFA_EXC_QIMG=1";
    if (a.variant & EXC_P16)    a.opts += " -DFA_EXC_P16=1";
    if (a.variant & EXC_QK2B)   a.opts += " -DFA_EXC_QK2B=1";
    // the fused softmax's row max: the extension macro the production host passes (Adreno does not predefine it)
    const char * rowmax = "-";
    if (a.variant & (EXC_FSM | EXC_P16 | EXC_QK2B)) {
        rowmax = "local-memory";
        if (cs.ext.find("cl_khr_subgroups") != std::string::npos) {
            if (cs.ext.find("cl_qcom_subgroup_shuffle") != std::string::npos) {
                a.opts += " -D cl_qcom_subgroup_shuffle=1"; rowmax = "qcom-shuffle";
            } else if (cs.ext.find("cl_khr_subgroup_shuffle") != std::string::npos) {
                a.opts += " -D cl_khr_subgroup_shuffle=1"; rowmax = "khr-shuffle";
            }
        }
        if (o.extra_opts.find("cl_qcom_subgroup_shuffle") != std::string::npos) rowmax = "qcom-shuffle(--opts)";
        else if (o.extra_opts.find("cl_khr_subgroup_shuffle") != std::string::npos) rowmax = "khr-shuffle(--opts)";
        if (o.extra_opts.find("FA_EXC_NO_SHFL") != std::string::npos) rowmax = "local-memory(forced)";
    }
    if (a.lds_pad)              a.opts += " -DFA_EXC_LDS_PAD=" + std::to_string(a.lds_pad);
    if (!o.extra_opts.empty())  a.opts += " " + o.extra_opts;
    a.name = arm_name(a.variant, a.lds_pad);
    const char * s = src.c_str(); const size_t n = src.size();
    cl_int err;
    a.prog = clCreateProgramWithSource(cs.ctx, 1, &s, &n, &err); CL_CHECK(err, "clCreateProgramWithSource");
    const auto t0 = std::chrono::steady_clock::now();
    err = clBuildProgram(a.prog, 1, &cs.dev, a.opts.c_str(), nullptr, nullptr);
    const double build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    size_t ln = 0; clGetProgramBuildInfo(a.prog, cs.dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ln);
    std::string log(ln, '\0'); clGetProgramBuildInfo(a.prog, cs.dev, CL_PROGRAM_BUILD_LOG, ln, &log[0], nullptr);
    if (err != CL_SUCCESS) DIE("build of arm %s ('%s') failed (%d):\n%s", a.name.c_str(), a.opts.c_str(), err, log.c_str());
    a.k_stage_p = clCreateKernel(a.prog, "stage_k_p", &err); CL_CHECK(err, "kernel stage_k_p");
    a.k_fa      = clCreateKernel(a.prog, "fa_exc", &err);    CL_CHECK(err, "kernel fa_exc");
    CL_CHECK(clGetKernelInfo(a.k_fa, CL_KERNEL_NUM_ARGS, sizeof(a.fa_nargs), &a.fa_nargs, nullptr), "num args");
    CHECK(a.fa_nargs == 49 || a.fa_nargs == 50, "fa_exc has %u arguments; this bench sets the 49 of the deployed kernel (+ q_img)", a.fa_nargs);
    CHECK(a.fa_nargs == 50 || !(a.variant & EXC_QIMG), "the kernel source has no q_img argument: variant %d needs the patched source", a.variant);
    size_t wg = 0, mult = 0, sg = 0;
    cl_ulong pm = 0, lm = 0;
    clGetKernelWorkGroupInfo(a.k_fa, cs.dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr);
    clGetKernelWorkGroupInfo(a.k_fa, cs.dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(pm), &pm, nullptr);
    clGetKernelWorkGroupInfo(a.k_fa, cs.dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(lm), &lm, nullptr);
    clGetKernelWorkGroupInfo(a.k_fa, cs.dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(mult), &mult, nullptr);
    const size_t lsz[3] = { 128, 1, 1 };
    if (clGetKernelSubGroupInfo(a.k_fa, cs.dev, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE, sizeof(lsz), lsz, sizeof(sg), &sg, nullptr) != CL_SUCCESS) sg = 0;
    printf("arm %-22s fa_exc: max wg %zu, local %llu B, private %llu B, wg multiple %zu, subgroup %zu, row max %s | build %.0f ms | %s\n",
           a.name.c_str(), wg, (unsigned long long) lm, (unsigned long long) pm, mult, sg, rowmax, build_ms, a.opts.c_str());
    if (o.verbose && log.find_first_not_of(" \n\r\t") != std::string::npos) printf("  build log: %s\n", log.c_str());
}

static void release_arm(arm & a) {
    if (a.k_fa) clReleaseKernel(a.k_fa);
    if (a.k_stage_p) clReleaseKernel(a.k_stage_p);
    if (a.prog) clReleaseProgram(a.prog);
}

// ---------------------------------------------------------------------------------------------------------------
// one op's inputs

struct op_inputs {
    int kv = 0, nbk = 0, num_sb = 0, n_qbt = 0, n_stages = 0, G = 0;
    size_t pairs = 0;
    std::vector<std::vector<int>> lists;   // [kvh * num_sb + sb] ascending absolute blocks
};

struct bench_bufs {
    shared_buf kvb, cmp, fold, svm;
    fold_layout L{};
    size_t k_off = 0, v_off = 0, kv_nb1 = 0, kv_nb2 = 0;
    size_t q_off = 0, q_nb1 = 0, q_nb2 = 0, m_off = 0, em_off = 0, em0_off = 0;
    int nbk_cap = 0, num_sb = 0;
    std::vector<uint16_t> k_host, v_host;   // [kv_max][nkvh][D]
    std::vector<float> q_host;              // [Sq][nh][D]
};

// the deployed arguments of stage_k_p (hfold_enqueue_chain, want 0, the op's own blocks)
static void set_stage_args(const arm & a, bench_bufs & B, const options & o, const op_inputs & op, bool empty) {
    const cl_kernel k = a.k_stage_p;
    const cl_mem kt = (a.variant & EXC_TEX128) ? B.fold.img_u : B.fold.img_h;
    const cl_int k_off = (cl_int) B.k_off, k_nb1 = (cl_int) B.kv_nb1, k_nb2 = (cl_int) B.kv_nb2;
    const cl_int kt_base_tex = (cl_int) (B.L.off_kt / 8), abs_off = (cl_int) B.L.off_abs, cnt_off = (cl_int) B.L.off_cnt;
    const cl_int abs_stride = B.nbk_cap, n_stage = B.nbk_cap * 64, nkvh = o.nkvh, scnt_off = (cl_int) B.L.off_scnt;
    const cl_uint want = 0;
    const cl_int em_off = (cl_int) (empty ? B.em0_off : B.em_off), nbk = op.nbk, num_sb = op.num_sb;
    const cl_int idx_off = (cl_int) B.L.off_idx, lcnt_off = (cl_int) B.L.off_lcnt, kb_per_head = op.nbk * 2;
    int i = 0;
    CL_CHECK(clSetKernelArg(k, i++, sizeof(cl_mem), &B.kvb.mem), "arg");
    clSetKernelArg(k, i++, sizeof(cl_int), &k_off); clSetKernelArg(k, i++, sizeof(cl_int), &k_nb1); clSetKernelArg(k, i++, sizeof(cl_int), &k_nb2);
    clSetKernelArg(k, i++, sizeof(cl_mem), &kt); clSetKernelArg(k, i++, sizeof(cl_int), &kt_base_tex);
    clSetKernelArg(k, i++, sizeof(cl_mem), &B.fold.mem);
    clSetKernelArg(k, i++, sizeof(cl_int), &abs_off); clSetKernelArg(k, i++, sizeof(cl_int), &cnt_off);
    clSetKernelArg(k, i++, sizeof(cl_int), &abs_stride); clSetKernelArg(k, i++, sizeof(cl_int), &n_stage);
    clSetKernelArg(k, i++, sizeof(cl_int), &nkvh); clSetKernelArg(k, i++, sizeof(cl_int), &scnt_off);
    clSetKernelArg(k, i++, sizeof(cl_mem), &B.svm.mem); clSetKernelArg(k, i++, sizeof(cl_uint), &want);
    clSetKernelArg(k, i++, sizeof(cl_mem), &B.cmp.mem); clSetKernelArg(k, i++, sizeof(cl_int), &em_off);
    clSetKernelArg(k, i++, sizeof(cl_int), &nbk); clSetKernelArg(k, i++, sizeof(cl_int), &num_sb);
    clSetKernelArg(k, i++, sizeof(cl_int), &idx_off); clSetKernelArg(k, i++, sizeof(cl_int), &lcnt_off);
    CL_CHECK(clSetKernelArg(k, i++, sizeof(cl_int), &kb_per_head), "stage_k_p args");
}

// the deployed arguments of fa_exc (hfold_args_fa with chain 2, gpu_done 1, (query tile, KV head) stages, want 0)
static void set_fa_args(const arm & a, bench_bufs & B, const options & o, const op_inputs & op) {
    const cl_kernel k = a.k_fa;
    const cl_mem kt = (a.variant & EXC_TEX128) ? B.fold.img_u : B.fold.img_h;
    const cl_mem vimg = (a.variant & EXC_TEX128) ? B.kvb.img_u : B.kvb.img_h;
    const cl_mem qimg = (a.variant & EXC_QIMG) ? B.cmp.img_u : kt;
    const cl_int q_off = (cl_int) B.q_off, q_nb1 = (cl_int) B.q_nb1, q_nb2 = (cl_int) B.q_nb2;
    const cl_int m_off = (cl_int) B.m_off, m_nb1 = (cl_int) (op.kv * 2);
    const cl_int idx = (cl_int) B.L.off_idx, abs = (cl_int) B.L.off_abs, cnt = (cl_int) B.L.off_cnt, lcnt = (cl_int) B.L.off_lcnt, err = (cl_int) B.L.off_err;
    const cl_int f_m = (cl_int) B.L.off_m, f_l = (cl_int) B.L.off_l, f_acc = (cl_int) B.L.off_acc;
    const cl_int Sq = o.sq, G = op.G, num_sb = op.num_sb, qb_per_sb = 2, top_k = B.nbk_cap, abs_stride = B.nbk_cap, n_stage = B.nbk_cap * 64;
    const cl_int kt_base_tex = (cl_int) (B.L.off_kt / 8), v_base_tex = (cl_int) (B.v_off / 8), v_tok_tex = (cl_int) (B.kv_nb1 / 8), v_head_tex = (cl_int) (B.kv_nb2 / 8);
    const float scale = 1.0f / sqrtf((float) D);
    const cl_uint want = 0;
    const cl_int chain = 2, gpu_done = 1, qbh = 1, nkvh = o.nkvh, qb32ps = o.br / 32;
    const cl_int scnt = (cl_int) B.L.off_scnt, done_off = (cl_int) B.L.off_done, stamp_off = (cl_int) B.L.off_stamp, n_stages = op.n_stages;
    const cl_int k_base_tex = (cl_int) (B.k_off / 8), k_tok_tex = (cl_int) (B.kv_nb1 / 8), k_head_tex = (cl_int) (B.kv_nb2 / 8);
    const cl_int em_off = (cl_int) B.em_off, nbk = op.nbk;
    int i = 0;
    clSetKernelArg(k, i++, sizeof(cl_mem), &kt); clSetKernelArg(k, i++, sizeof(cl_mem), &vimg);
    clSetKernelArg(k, i++, sizeof(cl_mem), &B.cmp.mem); clSetKernelArg(k, i++, sizeof(cl_int), &q_off);
    clSetKernelArg(k, i++, sizeof(cl_int), &q_nb1); clSetKernelArg(k, i++, sizeof(cl_int), &q_nb2);
    clSetKernelArg(k, i++, sizeof(cl_mem), &B.cmp.mem); clSetKernelArg(k, i++, sizeof(cl_int), &m_off); clSetKernelArg(k, i++, sizeof(cl_int), &m_nb1);
    clSetKernelArg(k, i++, sizeof(cl_mem), &B.fold.mem); clSetKernelArg(k, i++, sizeof(cl_mem), &B.fold.mem);
    clSetKernelArg(k, i++, sizeof(cl_int), &idx); clSetKernelArg(k, i++, sizeof(cl_int), &abs);
    clSetKernelArg(k, i++, sizeof(cl_int), &cnt); clSetKernelArg(k, i++, sizeof(cl_int), &lcnt); clSetKernelArg(k, i++, sizeof(cl_int), &err);
    clSetKernelArg(k, i++, sizeof(cl_int), &f_m); clSetKernelArg(k, i++, sizeof(cl_int), &f_l); clSetKernelArg(k, i++, sizeof(cl_int), &f_acc);
    clSetKernelArg(k, i++, sizeof(cl_int), &Sq); clSetKernelArg(k, i++, sizeof(cl_int), &G); clSetKernelArg(k, i++, sizeof(cl_int), &num_sb);
    clSetKernelArg(k, i++, sizeof(cl_int), &qb_per_sb); clSetKernelArg(k, i++, sizeof(cl_int), &top_k);
    clSetKernelArg(k, i++, sizeof(cl_int), &abs_stride); clSetKernelArg(k, i++, sizeof(cl_int), &n_stage);
    clSetKernelArg(k, i++, sizeof(cl_int), &kt_base_tex); clSetKernelArg(k, i++, sizeof(cl_int), &v_base_tex);
    clSetKernelArg(k, i++, sizeof(cl_int), &v_tok_tex); clSetKernelArg(k, i++, sizeof(cl_int), &v_head_tex);
    clSetKernelArg(k, i++, sizeof(cl_float), &scale);
    clSetKernelArg(k, i++, sizeof(cl_mem), &B.svm.mem); clSetKernelArg(k, i++, sizeof(cl_uint), &want);
    clSetKernelArg(k, i++, sizeof(cl_int), &chain); clSetKernelArg(k, i++, sizeof(cl_int), &gpu_done); clSetKernelArg(k, i++, sizeof(cl_int), &qbh);
    clSetKernelArg(k, i++, sizeof(cl_int), &nkvh); clSetKernelArg(k, i++, sizeof(cl_int), &qb32ps);
    clSetKernelArg(k, i++, sizeof(cl_int), &scnt); clSetKernelArg(k, i++, sizeof(cl_int), &done_off); clSetKernelArg(k, i++, sizeof(cl_int), &stamp_off);
    clSetKernelArg(k, i++, sizeof(cl_int), &n_stages);
    clSetKernelArg(k, i++, sizeof(cl_mem), &B.kvb.img_h);
    clSetKernelArg(k, i++, sizeof(cl_int), &k_base_tex); clSetKernelArg(k, i++, sizeof(cl_int), &k_tok_tex); clSetKernelArg(k, i++, sizeof(cl_int), &k_head_tex);
    clSetKernelArg(k, i++, sizeof(cl_mem), &B.cmp.mem); clSetKernelArg(k, i++, sizeof(cl_int), &em_off);
    CL_CHECK(clSetKernelArg(k, i++, sizeof(cl_int), &nbk), "fa_exc args");
    if (a.fa_nargs == 50) CL_CHECK(clSetKernelArg(k, i++, sizeof(cl_mem), &qimg), "fa_exc q_img");
}

// the gate kernel's job: this op's per-stage counters, the two persistent unit counters and the error word
static void zero_counters(cl_state & cs, bench_bufs & B) {
    const uint32_t zero = 0;
    CL_CHECK(clEnqueueFillBuffer(cs.q, B.fold.mem, &zero, 4, B.L.off_scnt, (HTP_FA_FOLD_MAX_STAGES + 2) * 4, 0, nullptr, nullptr), "fill counters");
    CL_CHECK(clEnqueueFillBuffer(cs.q, B.fold.mem, &zero, 4, B.L.off_err, 4, 0, nullptr, nullptr), "fill err");
}

static double event_ms(cl_event ev) {
    cl_ulong t0 = 0, t1 = 0;
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr), "profiling start");
    CL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr), "profiling end");
    return (double) (t1 - t0) * 1e-6;
}

static double run_stage(cl_state & cs, const arm & a, bench_bufs & B, const options & o, const op_inputs & op, bool empty) {
    zero_counters(cs, B);
    set_stage_args(a, B, o, op, empty);
    const size_t g = (size_t) o.pwg_stage * 128, l = 128;
    cl_event ev = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(cs.q, a.k_stage_p, 1, nullptr, &g, &l, 0, nullptr, &ev), "stage_k_p");
    CL_CHECK(clFinish(cs.q), "clFinish stage_k_p");
    const double ms = event_ms(ev);
    clReleaseEvent(ev);
    return ms;
}

// one timed batch of fa_exc launches; returns the device ms of each
static std::vector<double> run_fa(cl_state & cs, const arm & a, bench_bufs & B, const options & o, const op_inputs & op, int pwg, int n) {
    set_fa_args(a, B, o, op);
    const size_t n_units = (size_t) (o.br / 32) * op.G * op.n_stages;
    const size_t gsz[3] = { 128, std::min<size_t>((size_t) std::max(1, pwg), n_units), 1 }, lsz[3] = { 128, 1, 1 };
    std::vector<cl_event> ev((size_t) n, nullptr);
    for (int i = 0; i < n; ++i) {
        zero_counters(cs, B);
        CL_CHECK(clEnqueueNDRangeKernel(cs.q, a.k_fa, 3, nullptr, gsz, lsz, 0, nullptr, &ev[(size_t) i]), "fa_exc");
    }
    CL_CHECK(clFinish(cs.q), "clFinish fa_exc");
    std::vector<double> ms;
    for (auto e : ev) { ms.push_back(event_ms(e)); clReleaseEvent(e); }
    return ms;
}

// ---------------------------------------------------------------------------------------------------------------

static op_inputs make_op(const options & o, int kv, std::mt19937 & rng) {
    op_inputs op;
    op.kv = kv; op.nbk = kv / BK; op.num_sb = o.sq / 64; op.G = o.nh / o.nkvh;
    op.n_qbt = (o.sq + o.br - 1) / o.br; op.n_stages = op.n_qbt * o.nkvh;
    op.lists.assign((size_t) o.nkvh * op.num_sb, {});
    const int pos0 = kv - o.sq, R = o.br / 64;
    const double mean = o.exc_frac * op.nbk;
    std::gamma_distribution<double> gam(o.len_shape, mean / o.len_shape);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (int kvh = o.empty_kvh; kvh < o.nkvh; ++kvh) {
        for (int sb = 0; sb < op.num_sb; ++sb) {
            const int diag = (pos0 + sb * 64) / BK;      // the sub-block's own block, partly masked
            const int n_vis = diag + 1;
            int len = o.fixed_len >= 0 ? o.fixed_len : (int) std::lround(gam(rng));
            len = std::max(0, std::min(len, n_vis));
            std::vector<int> & l = op.lists[(size_t) kvh * op.num_sb + sb];
            const int j = sb % R;
            if (len > 0 && (j >= 2 || (j == 1 && u01(rng) < 0.5))) l.push_back(diag);
            std::vector<int> pool;
            for (int b = 0; b < diag; ++b) pool.push_back(b);
            std::shuffle(pool.begin(), pool.end(), rng);
            for (size_t i = 0; (int) l.size() < len && i < pool.size(); ++i) l.push_back(pool[i]);
            if ((int) l.size() < len) l.push_back(diag);
            std::sort(l.begin(), l.end());
            l.erase(std::unique(l.begin(), l.end()), l.end());
            op.pairs += l.size();
        }
    }
    return op;
}

static void upload_op(cl_state & cs, bench_bufs & B, const options & o, const op_inputs & op) {
    // causal mask [kv, Sq] f16: row t sees keys <= (kv - Sq) + t
    std::vector<uint16_t> mask((size_t) o.sq * op.kv);
    const uint16_t ninf = 0xfc00u, zero = 0;
    for (int t = 0; t < o.sq; ++t)
        for (int j = 0; j < op.kv; ++j) mask[(size_t) t * op.kv + j] = j > (op.kv - o.sq) + t ? ninf : zero;
    upload(cs, B.cmp, B.m_off, mask.data(), mask.size() * 2);
    // membership [nbk, R, NBq/R, nkvh] f32 (index (kvh * num_sb + sb) * nbk + b); an all-zero copy for the empty arm
    std::vector<float> em((size_t) o.nkvh * op.num_sb * op.nbk, 0.0f);
    for (size_t r = 0; r < op.lists.size(); ++r) for (int b : op.lists[r]) em[r * op.nbk + (size_t) b] = 1.0f;
    upload(cs, B.cmp, B.em_off, em.data(), em.size() * 4);
    std::fill(em.begin(), em.end(), 0.0f);
    upload(cs, B.cmp, B.em0_off, em.data(), em.size() * 4);
}

// stage_k_p's tables and K^T against the host's
static size_t check_stage(cl_state & cs, bench_bufs & B, const options & o, const op_inputs & op) {
    const int cap = B.nbk_cap, n_stage = cap * 64;
    std::vector<int32_t> absv((size_t) o.nkvh * cap), cnt(o.nkvh), lcnt((size_t) o.nkvh * op.num_sb), idx((size_t) o.nkvh * op.num_sb * cap);
    download(cs, B.fold, B.L.off_abs, absv.data(), absv.size() * 4);
    download(cs, B.fold, B.L.off_cnt, cnt.data(), cnt.size() * 4);
    download(cs, B.fold, B.L.off_lcnt, lcnt.data(), lcnt.size() * 4);
    download(cs, B.fold, B.L.off_idx, idx.data(), idx.size() * 4);
    std::vector<uint16_t> kt((size_t) o.nkvh * D * n_stage);
    download(cs, B.fold, B.L.off_kt, kt.data(), kt.size() * 2);
    size_t bad = 0;
    for (int kvh = 0; kvh < o.nkvh; ++kvh) {
        std::vector<int> u;
        for (int sb = 0; sb < op.num_sb; ++sb) for (int b : op.lists[(size_t) kvh * op.num_sb + sb]) u.push_back(b);
        std::sort(u.begin(), u.end()); u.erase(std::unique(u.begin(), u.end()), u.end());
        bad += cnt[kvh] != (int) u.size() * 64;
        for (int i = 0; i < cap; ++i) bad += absv[(size_t) kvh * cap + i] != (i < (int) u.size() ? u[i] : -1);
        for (int sb = 0; sb < op.num_sb; ++sb) {
            const auto & l = op.lists[(size_t) kvh * op.num_sb + sb];
            bad += lcnt[(size_t) kvh * op.num_sb + sb] != (int) l.size();
            for (size_t s = 0; s < l.size(); ++s) {
                const int c = (int) (std::lower_bound(u.begin(), u.end(), l[s]) - u.begin());
                bad += idx[((size_t) kvh * op.num_sb + sb) * cap + s] != c;
            }
        }
        for (size_t c = 0; c < u.size(); ++c)
            for (int i = 0; i < 64; ++i) {
                const size_t tok = (size_t) u[c] * 64 + i;
                for (int d = 0; d < D; ++d)
                    bad += kt[((size_t) kvh * D + d) * n_stage + c * 64 + i] != B.k_host[(tok * o.nkvh + kvh) * D + d];
            }
    }
    return bad;
}

struct partial {
    std::vector<float> m, l;
    std::vector<uint16_t> acc16;
    std::vector<float> acc32;
    int32_t err = 0;
};

static partial read_partial(cl_state & cs, bench_bufs & B, const options & o) {
    partial p;
    const size_t rows = (size_t) o.sq * o.nh;
    p.m.resize(rows); p.l.resize(rows);
    download(cs, B.fold, B.L.off_m, p.m.data(), rows * 4);
    download(cs, B.fold, B.L.off_l, p.l.data(), rows * 4);
    if (o.acc16) { p.acc16.resize(rows * D); download(cs, B.fold, B.L.off_acc, p.acc16.data(), rows * D * 2); }
    else         { p.acc32.resize(rows * D); download(cs, B.fold, B.L.off_acc, p.acc32.data(), rows * D * 4); }
    download(cs, B.fold, B.L.off_err, &p.err, 4);
    return p;
}

static float acc_at(const partial & p, size_t i) { return p.acc16.empty() ? p.acc32[i] : f16_to_f32(p.acc16[i]); }

struct bitdiff { size_t m = 0, l = 0, acc = 0; double m_abs = 0, l_rel = 0, acc_rel = 0; };

static bitdiff compare_bits(const partial & a, const partial & b) {
    bitdiff d;
    double amax = 0;
    for (size_t i = 0; i < a.m.size(); ++i) {
        d.m += memcmp(&a.m[i], &b.m[i], 4) != 0;
        d.l += memcmp(&a.l[i], &b.l[i], 4) != 0;
        d.m_abs = std::max(d.m_abs, (double) fabsf(a.m[i] - b.m[i]));
        if (b.l[i] > 0) d.l_rel = std::max(d.l_rel, fabs((double) a.l[i] / b.l[i] - 1.0));
    }
    const size_t n = a.acc16.empty() ? a.acc32.size() : a.acc16.size();
    for (size_t i = 0; i < n; ++i) amax = std::max(amax, (double) fabsf(acc_at(b, i)));
    for (size_t i = 0; i < n; ++i) {
        const bool same = a.acc16.empty() ? memcmp(&a.acc32[i], &b.acc32[i], 4) == 0 : a.acc16[i] == b.acc16[i];
        if (!same) { d.acc++; d.acc_rel = std::max(d.acc_rel, fabs((double) acc_at(a, i) - acc_at(b, i)) / (amax > 0 ? amax : 1.0)); }
    }
    return d;
}

struct ref_err { size_t rows = 0, empty_rows = 0, empty_bad = 0; double m_abs = 0, l_rel = 0, o_rel = 0; };

// float64 attention over the exception blocks of sampled rows: the unnormalised (m, l, acc) the fold consumes
static ref_err check_ref(const bench_bufs & B, const options & o, const op_inputs & op, const partial & p, std::mt19937 & rng) {
    ref_err e;
    const double scale = 1.0 / sqrt((double) D);
    std::uniform_int_distribution<int> pick_h(0, o.nh - 1), pick_t(0, o.sq - 1);
    std::vector<double> acc(D);
    for (int it = 0; it < o.ref_rows; ++it) {
        const int h = pick_h(rng), tk = pick_t(rng), kvh = h / op.G, sb = tk / 64;
        const size_t r = (size_t) h * o.sq + tk;
        const auto & l = op.lists[(size_t) kvh * op.num_sb + sb];
        const float * q = &B.q_host[((size_t) tk * o.nh + h) * D];
        std::vector<double> s; std::vector<int> keys;
        double m = -INFINITY;
        for (int b : l) for (int i = 0; i < 64; ++i) {
            const int key = b * 64 + i;
            if (key > (op.kv - o.sq) + tk) continue;   // causal mask
            const uint16_t * kr = &B.k_host[((size_t) key * o.nkvh + kvh) * D];
            double dot = 0; for (int d = 0; d < D; ++d) dot += (double) q[d] * f16_to_f32(kr[d]);
            s.push_back(dot * scale); keys.push_back(key); m = std::max(m, s.back());
        }
        if (s.empty()) {
            e.empty_rows++;
            bool ok = p.m[r] == -10000.0f && p.l[r] == 0.0f;
            for (int d = 0; d < D && ok; ++d) ok = acc_at(p, r * D + d) == 0.0f;
            e.empty_bad += !ok;
            continue;
        }
        double lsum = 0; std::fill(acc.begin(), acc.end(), 0.0);
        for (size_t i = 0; i < s.size(); ++i) {
            const double w = exp(s[i] - m); lsum += w;
            const uint16_t * vr = &B.v_host[((size_t) keys[i] * o.nkvh + kvh) * D];
            for (int d = 0; d < D; ++d) acc[d] += w * f16_to_f32(vr[d]);
        }
        e.rows++;
        e.m_abs = std::max(e.m_abs, fabs(p.m[r] - m));
        e.l_rel = std::max(e.l_rel, fabs(p.l[r] / lsum - 1.0));
        // normalised output, relative to the row's largest output (acc is stored in f16 when deployed)
        double omax = 0, dmax = 0;
        for (int d = 0; d < D; ++d) omax = std::max(omax, fabs(acc[d] / lsum));
        for (int d = 0; d < D; ++d) dmax = std::max(dmax, fabs(acc_at(p, r * D + d) / p.l[r] - acc[d] / lsum));
        e.o_rel = std::max(e.o_rel, dmax / (omax > 0 ? omax : 1.0));
    }
    return e;
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

static std::string load_cl_source(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    CHECK(f.good(), "cannot read %s", path.c_str());
    std::stringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    const std::string key = "ggml_hexagon_hfold_cl_src = R\"CL(";
    const size_t a = s.find(key);
    if (a == std::string::npos) return s;   // a bare .cl file
    const size_t b = s.find(")CL\";", a);
    CHECK(b != std::string::npos, "unterminated ggml_hexagon_hfold_cl_src in %s", path.c_str());
    return s.substr(a + key.size(), b - a - key.size());
}

// ---------------------------------------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char * { CHECK(i + 1 < argc, "%s needs a value", a.c_str()); return argv[++i]; };
        if      (a == "--kv")         o.kv = parse_list(val());
        else if (a == "--sq")         o.sq = atoi(val());
        else if (a == "--nh")         o.nh = atoi(val());
        else if (a == "--nkvh")       o.nkvh = atoi(val());
        else if (a == "--br")         o.br = atoi(val());
        else if (a == "--kv-cap")     o.kv_cap = atoi(val());
        else if (a == "--exc-frac")   o.exc_frac = atof(val());
        else if (a == "--len-shape")  o.len_shape = atof(val());
        else if (a == "--fixed-len")  o.fixed_len = atoi(val());
        else if (a == "--empty-kvh")  o.empty_kvh = atoi(val());
        else if (a == "--qscale")     o.qscale = atof(val());
        else if (a == "--variants")   o.variants = parse_list(val());
        else if (a == "--bitref")     o.bitref = atoi(val());
        else if (a == "--pwg")        o.pwg = parse_list(val());
        else if (a == "--pwg-stage")  o.pwg_stage = atoi(val());
        else if (a == "--lds-pad")    o.lds_pad = parse_list(val());
        else if (a == "--iters")      o.iters = atoi(val());
        else if (a == "--warmup")     o.warmup = atoi(val());
        else if (a == "--reps")       o.reps = atoi(val());
        else if (a == "--mem")        o.mem = val();
        else if (a == "--opts")       o.extra_opts = val();
        else if (a == "--cl-src")     o.cl_src_file = val();
        else if (a == "--no-acc16")   o.acc16 = false;
        else if (a == "--empty-baseline") o.empty_baseline = true;
        else if (a == "--ref-rows")   o.ref_rows = atoi(val());
        else if (a == "--no-check-tables") o.check_tables = false;
        else if (a == "--no-perf-hint") o.perf_hint = false;
        else if (a == "--csv")        o.csv = val();
        else if (a == "--seed")       o.seed = atoi(val());
        else if (a == "--platform")   o.platform = atoi(val());
        else if (a == "--device")     o.device = atoi(val());
        else if (a == "-v")           o.verbose = true;
        else { usage(); return a == "-h" || a == "--help" ? 0 : 1; }
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    CHECK(o.nh > 0 && o.nkvh > 0 && o.nh % o.nkvh == 0, "--nh must be a multiple of --nkvh");
    CHECK(o.sq % 64 == 0 && o.sq >= 64 && o.br % 64 == 0 && o.br >= 64, "--sq and --br must be multiples of 64");
    CHECK(((o.sq + o.br - 1) / o.br) * o.nkvh <= HTP_FA_FOLD_MAX_STAGES, "ceil(sq / br) * nkvh stages must fit %d", HTP_FA_FOLD_MAX_STAGES);
    CHECK(!o.kv.empty() && o.iters > 0 && o.reps > 0, "nothing to run");
    for (int kv : o.kv) CHECK(kv % 64 == 0 && kv >= o.sq && kv <= o.kv_cap, "--kv %d must be a multiple of 64 in [--sq, --kv-cap]", kv);
    CHECK(o.kv_cap % 64 == 0 && o.kv_cap / 64 <= 256, "--kv-cap must be a multiple of 64 and at most 16384 (stage_k_p's local table)");
    std::vector<int> vlist;
    for (int v : o.variants) { CHECK(v >= 0 && v <= 31, "variant %d is not a GGML_HEXAGON_FA_FOLD_EXC bit set", v); if (std::find(vlist.begin(), vlist.end(), v) == vlist.end()) vlist.push_back(v); }
    CHECK(o.bitref >= 0 && o.bitref <= 31, "--bitref %d is not a variant", o.bitref);
    if (std::find(vlist.begin(), vlist.end(), 0) == vlist.end()) vlist.insert(vlist.begin(), 0);             // the summary's baseline
    if (std::find(vlist.begin(), vlist.end(), o.bitref) == vlist.end()) vlist.insert(vlist.begin(), o.bitref);   // the bit reference

    const std::string src = o.cl_src_file.empty() ? std::string(fa_exc_bench_cl_src) : load_cl_source(o.cl_src_file);

    // ---- OpenCL ----
    cl_state cs;
    {
        cl_uint np = 0; CL_CHECK(clGetPlatformIDs(0, nullptr, &np), "clGetPlatformIDs");
        CHECK(np > (cl_uint) o.platform, "no OpenCL platform %d", o.platform);
        std::vector<cl_platform_id> ps(np); clGetPlatformIDs(np, ps.data(), nullptr);
        cs.plat = ps[(size_t) o.platform];
        cl_uint nd = 0; CL_CHECK(clGetDeviceIDs(cs.plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd), "clGetDeviceIDs");
        CHECK(nd > (cl_uint) o.device, "no device %d", o.device);
        std::vector<cl_device_id> ds(nd); clGetDeviceIDs(cs.plat, CL_DEVICE_TYPE_ALL, nd, ds.data(), nullptr);
        cs.dev = ds[(size_t) o.device];
        size_t n = 0; clGetDeviceInfo(cs.dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &n);
        cs.ext.resize(n); clGetDeviceInfo(cs.dev, CL_DEVICE_EXTENSIONS, n, &cs.ext[0], nullptr);
        clGetDeviceInfo(cs.dev, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, sizeof(cs.max_img_texels), &cs.max_img_texels, nullptr);
        char name[256] = {}, ver[256] = {}, drv[256] = {};
        clGetDeviceInfo(cs.dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        clGetDeviceInfo(cs.dev, CL_DEVICE_VERSION, sizeof(ver), ver, nullptr);
        clGetDeviceInfo(cs.dev, CL_DRIVER_VERSION, sizeof(drv), drv, nullptr);
        cl_uint cu = 0; cl_ulong lmem = 0; size_t mwg = 0; cl_uint clk = 0;
        clGetDeviceInfo(cs.dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
        clGetDeviceInfo(cs.dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
        clGetDeviceInfo(cs.dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(mwg), &mwg, nullptr);
        clGetDeviceInfo(cs.dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clk), &clk, nullptr);
        printf("device %s | %s | driver %s | %u CUs, %u MHz, local %llu B, max wg %zu, image buffer %zu texels\n",
               name, ver, drv, cu, clk, (unsigned long long) lmem, mwg, cs.max_img_texels);
        auto has = [&](const char * e) { return cs.ext.find(e) != std::string::npos; };
        printf("extensions: cl_khr_fp16 %d, cl_khr_subgroups %d, cl_khr_subgroup_shuffle %d, cl_qcom_subgroup_shuffle %d, cl_qcom_reqd_sub_group_size %d, cl_qcom_perf_hint %d, cl_qcom_ext_host_ptr %d\n",
               has("cl_khr_fp16"), has("cl_khr_subgroups"), has("cl_khr_subgroup_shuffle"), has("cl_qcom_subgroup_shuffle"),
               has("cl_qcom_reqd_sub_group_size"), has("cl_qcom_perf_hint"), has("cl_qcom_ext_host_ptr"));
        cl_int err;
        std::vector<cl_context_properties> props;
#ifdef CL_CONTEXT_PERF_HINT_QCOM
        if (o.perf_hint && has("cl_qcom_perf_hint")) {
            props = { CL_CONTEXT_PERF_HINT_QCOM, CL_PERF_HINT_HIGH_QCOM, 0 };
            printf("context: cl_qcom_perf_hint HIGH (as deployed)\n");
        }
#endif
        cs.ctx = clCreateContext(props.empty() ? nullptr : props.data(), 1, &cs.dev, nullptr, nullptr, &err); CL_CHECK(err, "clCreateContext");
        const cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        cs.q = clCreateCommandQueueWithProperties(cs.ctx, cs.dev, qp, &err); CL_CHECK(err, "clCreateCommandQueueWithProperties");
        if (o.mem == "ion") {
#ifdef FA_EXC_BENCH_NO_HTP
            DIE("--mem ion needs the HTP0 device: this build has no ggml (FA_EXC_BENCH_NO_HTP)");
#else
            CHECK(has("cl_qcom_ext_host_ptr"), "--mem ion needs cl_qcom_ext_host_ptr");
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
                if (!strcmp(ggml_backend_dev_name(ggml_backend_dev_get(i)), "HTP0")) cs.htp = ggml_backend_dev_get(i);
            CHECK(cs.htp, "--mem ion needs the HTP0 device (rpcmem); use --mem cl without it");
            cs.htp_be = ggml_backend_dev_init(cs.htp, nullptr);
            CHECK(cs.htp_be, "HTP0 init failed");
            cs.ion = true;
#endif
        } else {
            CHECK(o.mem == "cl", "--mem is ion or cl");
        }
    }

    // ---- buffers for the longest KV ----
    bench_bufs B;
    const int kv_max = *std::max_element(o.kv.begin(), o.kv.end());
    B.nbk_cap = o.kv_cap / 64; B.num_sb = o.sq / 64;
    B.kv_nb1 = (size_t) o.nkvh * D * 2; B.kv_nb2 = (size_t) D * 2;          // [D, nkvh, kv] f16, token-major
    B.k_off = 0; B.v_off = al((size_t) o.kv_cap * B.kv_nb1, 4096);         // cache_k_l / cache_v_l of one layer
    B.q_nb1 = (size_t) o.nh * D * 4; B.q_nb2 = (size_t) D * 4;              // Qcur permuted: token-major, heads inside
    B.q_off = 0;
    B.m_off = al((size_t) o.sq * B.q_nb1, 4096);
    B.em_off = al(B.m_off + (size_t) o.sq * kv_max * 2, 4096);
    const size_t em_bytes = (size_t) o.nkvh * B.num_sb * (kv_max / 64) * 4;
    B.em0_off = al(B.em_off + em_bytes, 4096);
    B.L = make_fold_layout(o.sq * o.nh, o.nkvh, B.nbk_cap, B.num_sb);
    buf_alloc(cs, B.kvb, B.v_off + (size_t) o.kv_cap * B.kv_nb1, "kv cache");
    buf_alloc(cs, B.cmp, B.em0_off + em_bytes, "compute");
    buf_alloc(cs, B.fold, B.L.size, "fold");
    buf_alloc(cs, B.svm, 4096, "svm stand-in");
    printf("buffers (%s): kv %zu MB, compute %zu MB, fold %zu MB (K^T at +%zu, %d blocks capacity)\n",
           cs.ion ? "rpcmem + ION alias, IO-coherent" : "OpenCL", B.kvb.size >> 20, B.cmp.size >> 20, B.fold.size >> 20, B.L.off_kt, B.nbk_cap);

    std::mt19937 rng((uint32_t) o.seed);
    {
        std::normal_distribution<float> nq(0.0f, (float) o.qscale), n1(0.0f, 1.0f);
        B.q_host.resize((size_t) o.sq * o.nh * D);
        for (auto & x : B.q_host) x = nq(rng);
        B.k_host.resize((size_t) o.kv_cap * o.nkvh * D); B.v_host.resize(B.k_host.size());
        for (auto & x : B.k_host) x = f32_to_f16(n1(rng));
        for (auto & x : B.v_host) x = f32_to_f16(n1(rng));
        upload(cs, B.cmp, B.q_off, B.q_host.data(), B.q_host.size() * 4);
        upload(cs, B.kvb, B.k_off, B.k_host.data(), B.k_host.size() * 2);
        upload(cs, B.kvb, B.v_off, B.v_host.data(), B.v_host.size() * 2);
    }

    // ---- arms ----
    std::vector<arm> arms;
    std::vector<int> pads = { 0 };
    for (int p : o.lds_pad) if (p > 0) pads.push_back(p);
    for (int pad : pads) for (int v : vlist) { arm a; a.variant = v; a.lds_pad = pad; build_arm(cs, src, a, o); arms.push_back(a); }

    FILE * csv = nullptr;
    if (!o.csv.empty()) {
        csv = fopen(o.csv.c_str(), "a");
        CHECK(csv, "cannot open %s", o.csv.c_str());
        if (ftell(csv) == 0) fprintf(csv, "kv,pairs,units,arm,variant,lds_pad,pwg,fa_ms_median,fa_ms_min,us_per_pair,us_per_wg_block,empty_ms,marginal_us_per_pair,stage_k_ms,"
                     "bitref,bits_m,bits_l,bits_acc,l_rel_vs_ref,acc_rel_vs_ref,ref_rows,ref_m_abs,ref_l_rel,ref_o_rel,ref_empty_bad,kernel_err\n");
    }

    struct agg { double pairs = 0, ms = 0; };
    std::map<std::pair<std::string, int>, agg> total;   // (arm, pwg) -> sums over KVs
    std::map<std::pair<std::string, int>, agg> total_marg;

    for (int kv : o.kv) {
        const op_inputs op = make_op(o, kv, rng);
        upload_op(cs, B, o, op);
        const size_t units = (size_t) (o.br / 32) * op.G * op.n_stages;
        size_t nonempty = 0, maxlen = 0;
        for (const auto & l : op.lists) { nonempty += !l.empty(); maxlen = std::max(maxlen, l.size()); }
        printf("\n== KV %d: %d blocks, %zu exception pairs in %zu non-empty lists of %zu (mean %.2f, max %zu), %zu units, %d stages\n",
               kv, op.nbk, op.pairs, nonempty, op.lists.size(), nonempty ? (double) op.pairs / nonempty : 0.0, maxlen, units, op.n_stages);

        // correctness first: each arm's tables, K^T and partial (the plain arm's partial is the bit reference)
        std::map<std::string, partial> parts;
        std::map<std::string, std::vector<double>> stage_ms;   // stage_k_p device ms, one per rep
        for (auto & a : arms) {
            run_stage(cs, a, B, o, op, false);
            if (o.check_tables) {
                const size_t bad = check_stage(cs, B, o, op);
                CHECK(bad == 0, "arm %s: stage_k_p's tables / K^T differ from the host's in %zu entries", a.name.c_str(), bad);
            }
            // poison the partial: a row the kernel does not write cannot pass
            const uint32_t poison = 0x7fc0deadu;
            CL_CHECK(clEnqueueFillBuffer(cs.q, B.fold.mem, &poison, 4, B.L.off_m, B.L.off_ctl - B.L.off_m, 0, nullptr, nullptr), "poison");
            run_fa(cs, a, B, o, op, o.pwg[0], 1);
            parts[a.name] = read_partial(cs, B, o);
            const partial & p = parts[a.name];
            size_t left = 0;
            for (size_t i = 0; i < p.m.size(); ++i) { uint32_t w; memcpy(&w, &p.m[i], 4); left += w == poison; memcpy(&w, &p.l[i], 4); left += w == poison; }
            CHECK(left == 0, "arm %s: %zu m/l words never written", a.name.c_str(), left);
            CHECK(p.err == 0, "arm %s: the kernel reported %d list errors", a.name.c_str(), p.err);
        }
        const partial & ref = parts[arm_name(o.bitref, 0)];

        // timing: reps x (arms in alternating order) x pwg, each a warmup plus o.iters launches
        std::map<std::pair<std::string, int>, std::vector<double>> times, times0;
        for (int rep = 0; rep < o.reps; ++rep) {
            std::vector<size_t> order(arms.size());
            for (size_t i = 0; i < arms.size(); ++i) order[i] = (rep % 2) ? arms.size() - 1 - i : i;
            for (size_t ai : order) {
                const arm & a = arms[ai];
                stage_ms[a.name].push_back(run_stage(cs, a, B, o, op, false));   // this arm's own K^T writes ahead of its own reads
                for (int pwg : o.pwg) {
                    if (o.warmup > 0) run_fa(cs, a, B, o, op, pwg, o.warmup);
                    auto ms = run_fa(cs, a, B, o, op, pwg, o.iters);
                    auto & v = times[{ a.name, pwg }]; v.insert(v.end(), ms.begin(), ms.end());
                }
                if (o.empty_baseline) {
                    run_stage(cs, a, B, o, op, true);
                    const op_inputs op0 = [&] { op_inputs e = op; for (auto & l : e.lists) l.clear(); e.pairs = 0; return e; }();
                    for (int pwg : o.pwg) {
                        // fa_exc reads the (now empty) tables; the membership pointer only matters with K_NAT
                        if (o.warmup > 0) run_fa(cs, a, B, o, op0, pwg, o.warmup);
                        auto ms = run_fa(cs, a, B, o, op0, pwg, o.iters);
                        auto & v = times0[{ a.name, pwg }]; v.insert(v.end(), ms.begin(), ms.end());
                    }
                }
            }
        }

        printf("%-24s %4s %9s %9s %8s %8s %9s %9s %7s | %-34s | %s\n", "arm", "pwg", "fa ms", "min ms", "us/pair", "us/wgblk",
               o.empty_baseline ? "empty ms" : "", o.empty_baseline ? "marg/pair" : "", "stage", ("bits differing vs " + arm_name(o.bitref, 0) + " (m l acc)").c_str(), "float64 reference");
        std::mt19937 rrng((uint32_t) (o.seed * 7919 + kv));
        for (auto & a : arms) {
            const partial & p = parts[a.name];
            const bitdiff bd = compare_bits(p, ref);
            std::mt19937 r2 = rrng;   // the same rows for every arm
            const ref_err re = o.ref_rows > 0 ? check_ref(B, o, op, p, r2) : ref_err();
            for (int pwg : o.pwg) {
                const auto & v = times[{ a.name, pwg }];
                const double med = median(v), mn = *std::min_element(v.begin(), v.end());
                const double upp = op.pairs ? med * 1e3 / op.pairs : 0.0;
                const double wgb = op.pairs ? med * 1e3 / ((double) op.pairs * 2 * op.G) : 0.0;
                double e_ms = 0, marg = 0;
                if (o.empty_baseline) { e_ms = median(times0[{ a.name, pwg }]); marg = op.pairs ? (med - e_ms) * 1e3 / op.pairs : 0.0; }
                char bits[96], e0[16] = "", e1[16] = "";
                snprintf(bits, sizeof(bits), "%zu %zu %zu (l %.1e acc %.1e)", bd.m, bd.l, bd.acc, bd.l_rel, bd.acc_rel);
                if (o.empty_baseline) { snprintf(e0, sizeof(e0), "%.3f", e_ms); snprintf(e1, sizeof(e1), "%.3f", marg); }
                printf("%-24s %4d %9.3f %9.3f %8.3f %8.3f %9s %9s %7.3f | %-34s | rows %zu: m %.1e l %.1e o %.1e, empty %zu/%zu bad\n",
                       a.name.c_str(), pwg, med, mn, upp, wgb, e0, e1,
                       median(stage_ms[a.name]), bits, re.rows, re.m_abs, re.l_rel, re.o_rel, re.empty_bad, re.empty_rows);
                total[{ a.name, pwg }].pairs += op.pairs; total[{ a.name, pwg }].ms += med;
                if (o.empty_baseline) { total_marg[{ a.name, pwg }].pairs += op.pairs; total_marg[{ a.name, pwg }].ms += med - e_ms; }
                if (csv) fprintf(csv, "%d,%zu,%zu,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%zu,%zu,%zu,%.3e,%.3e,%zu,%.3e,%.3e,%.3e,%zu,%d\n",
                                 kv, op.pairs, units, a.name.c_str(), a.variant, a.lds_pad, pwg, med, mn, upp, wgb, e_ms, marg, median(stage_ms[a.name]),
                                 o.bitref, bd.m, bd.l, bd.acc, bd.l_rel, bd.acc_rel, re.rows, re.m_abs, re.l_rel, re.o_rel, re.empty_bad, p.err);
            }
        }
    }

    printf("\n== summary over KV %s (sum of median ms / sum of pairs)\n", [&] { std::string s; for (int kv : o.kv) s += (s.empty() ? "" : ",") + std::to_string(kv); return s; }().c_str());
    for (int pwg : o.pwg) {
        const auto & t0 = total[{ arm_name(0, 0), pwg }];
        const double base = t0.pairs ? t0.ms * 1e3 / t0.pairs : 0.0;
        for (auto & a : arms) {
            const auto & t = total[{ a.name, pwg }];
            const double upp = t.pairs ? t.ms * 1e3 / t.pairs : 0.0;
            printf("%-24s pwg %3d: %.3f us/pair (%+.1f%% vs plain)", a.name.c_str(), pwg, upp, base > 0 ? 100.0 * (upp / base - 1.0) : 0.0);
            if (o.empty_baseline) {
                const auto & m = total_marg[{ a.name, pwg }];
                printf(", marginal %.3f us/pair", m.pairs ? m.ms * 1e3 / m.pairs : 0.0);
            }
            printf("\n");
        }
    }
    if (csv) fclose(csv);

    for (auto & a : arms) release_arm(a);
    buf_free(B.svm); buf_free(B.fold); buf_free(B.cmp); buf_free(B.kvb);
    clReleaseCommandQueue(cs.q); clReleaseContext(cs.ctx);
#ifndef FA_EXC_BENCH_NO_HTP
    if (cs.htp_be) ggml_backend_free(cs.htp_be);
#endif
    return 0;
}

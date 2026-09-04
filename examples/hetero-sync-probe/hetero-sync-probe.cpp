// hetero-sync-probe: how fast can an HTP op, mid-batch, hand a flag to the CPU/GPU and get one back?
//
// This is the de-risk measurement for a heterogeneous decode-attention design in which the GPU
// computes some KV splits and the HTP merge op consumes them through shared memory, with no
// graph barrier: the HTP op raises 'ready', the other agent writes partials + 'done', the HTP op
// spin-polls 'done'. The HTP side is HTP_OP_SYNC_PROBE (htp/sync-probe-ops.c), reached through a
// GGML_OP_CUSTOM node tagged with HTP_SYNC_PROBE_MAGIC (needs GGML_HEXAGON_SYNC_PROBE=1, set here).
//
// Modes (all inside one 1 MiB hexagon rpcmem buffer, aliased into OpenCL with the QCOM ION
// host-pointer extension so the GPU sees the same pages):
//   ping    - op stamps its entry time and returns: dispatch cost, qtimer-vs-cntvct offset bounds
//   cpu     - host polls 'ready', writes payload + 'done' itself: CPU<->HTP flag floor
//   gpu     - host polls 'ready', enqueues GPU fill + flag kernels: host-dispatched GPU path
//   gpu0    - same with no payload (flag only)
//   gpuspin - GPU kernel pre-enqueued, spins on 'ready' itself: no host in the loop
//
// Every HTP stamp is on the 19.2 MHz qtimer; host stamps are cntvct_el0 (same 19.2 MHz QTimer
// block, unknown constant offset D). The headline number, T_DONE - T_READY on the HTP clock, is
// D-free. Cross-domain splits use D bounds derived from the handshake itself.

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
#include <string>
#include <sched.h>
#include <time.h>
#include <unistd.h>
#include <vector>

extern "C" int rpcmem_to_fd(void * po);  // exported by libggml-hexagon.so (dlsym'd libcdsprpc wrapper)

// ---------------------------------------------------------------------------------------
// clocks and cache maintenance (aarch64 EL0)
// ---------------------------------------------------------------------------------------

static inline uint64_t cnt_now() {
    uint64_t v;
    asm volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(v) : : "memory");
    return v;
}

static inline double ticks_us(double t) { return t / 19.2; }

static inline uint64_t realtime_ns() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

// CLOCK_REALTIME ns  ->  cntvct ticks (the Adreno driver stamps CL events in CLOCK_REALTIME ns)
static double g_rt_off_ns = 0;  // realtime_ns - cnt_ns, sampled once
static void calibrate_realtime() {
    double best = 1e30, best_off = 0;
    for (int i = 0; i < 64; i++) {
        const uint64_t c0 = cnt_now();
        const uint64_t r  = realtime_ns();
        const uint64_t c1 = cnt_now();
        const double   w  = ticks_us((double) (c1 - c0)) * 1000.0;
        if (w < best) {
            best     = w;
            best_off = (double) r - ((double) c0 + (double) c1) * 0.5 * 1e9 / 19.2e6;
        }
    }
    g_rt_off_ns = best_off;
}
static inline double cnt_of_realtime(uint64_t ns) { return ((double) ns - g_rt_off_ns) * 19.2e6 / 1e9; }

static inline void dc_civac(const void * p, size_t n) {
    uintptr_t a = (uintptr_t) p & ~(uintptr_t) 63;
    const uintptr_t e = (uintptr_t) p + n;
    for (; a < e; a += 64) {
        asm volatile("dc civac, %0" : : "r"(a) : "memory");
    }
    asm volatile("dsb sy" : : : "memory");
}

static inline void dc_cvac(const void * p, size_t n) {
    uintptr_t a = (uintptr_t) p & ~(uintptr_t) 63;
    const uintptr_t e = (uintptr_t) p + n;
    for (; a < e; a += 64) {
        asm volatile("dc cvac, %0" : : "r"(a) : "memory");
    }
    asm volatile("dsb sy" : : : "memory");
}

// ---------------------------------------------------------------------------------------
// layout inside the shared buffer
// ---------------------------------------------------------------------------------------

static const size_t BUF_SIZE    = 1u << 20;
static const size_t READY_OFF   = 0;
static const size_t DONE_OFF    = 4096;
static const size_t ALIVE_OFF   = 8192;            // gpuspin: kernel is spinning
static const size_t ITERS_OFF   = 8192 + 128;      // gpuspin: spin iterations
static const size_t SENT_OFF    = 12288;           // alias sanity sentinel
static const size_t PAYLOAD_OFF = 16384;
static const size_t PAYLOAD_MAX = 512 * 1024;
static const size_t DSP_PAY_OFF = PAYLOAD_OFF + PAYLOAD_MAX;   // DSP-written data, read by CPU/GPU after ready
static const size_t DSP_PAY_MAX = 256 * 1024;
static const size_t REC_OFF     = BUF_SIZE - 4096; // dst record

enum probe_mode { M_PING, M_CPU, M_GPU, M_GPU0, M_GPU1, M_GPUSPIN, M_GPUSVM, M_COUNT };
static const char * mode_name[M_COUNT] = { "ping", "cpu", "gpu", "gpu0", "gpu1", "gpuspin", "gpusvm" };

struct options {
    int    iters       = 20;
    int    payload_kb  = 64;
    int    dsp_payload_kb = 8;   // DSP writes+flushes this much before raising ready
    int    b2b         = 5;      // back-to-back pre-enqueued kernel gap samples (0 = skip)
    int    inval       = 1;      // HTP poll read mode: 0 dcinva, 1 qurt invalidate, 2 none, 3 dma
    int    flush       = 0;      // HTP ready publish: 0 qurt flush, 1 dccleana+syncht
    int    timeout_ms  = 500;
    int    host_inval  = 1;      // dc civac before each host poll read
    int    host_clean  = 1;      // dc cvac after host writes into the shared buffer
    int    spin_max    = 5000000;   // gpuspin: spin iterations before the kernel gives up
    int    cl_policy   = 0;      // 0 = IOCOHERENT, else raw CL_MEM_HOST_*_QCOM value
    int    verbose     = 0;
    int    cpu         = -1;     // pin the main thread to this CPU
    int    no_flush    = 0;      // host-dispatched modes: skip clFlush after enqueue
    bool   modes[M_COUNT] = { true, true, true, true, true, true, true };
};

static void usage() {
    printf("usage: llama-hetero-sync-probe [--iters N] [--payload-kb K] [--inval 0|1|2|3] [--flush 0|1]\n"
           "         [--timeout-ms T] [--host-inval 0|1] [--host-clean 0|1] [--spin-max N]\n"
           "         [--cl-policy 0x40A4|0x40A5|0x40A9] [--cpu N] [--no-flush]\n"
           "         [--modes ping,cpu,gpu,gpu0,gpu1,gpuspin,gpusvm] [-v]\n");
}

static bool parse(int argc, char ** argv, options & o) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](int & v) { if (i + 1 >= argc) return false; v = (int) strtol(argv[++i], nullptr, 0); return true; };
        if (a == "--iters")            { if (!next(o.iters)) return false; }
        else if (a == "--payload-kb")  { if (!next(o.payload_kb)) return false; }
        else if (a == "--dsp-payload-kb") { if (!next(o.dsp_payload_kb)) return false; }
        else if (a == "--b2b")         { if (!next(o.b2b)) return false; }
        else if (a == "--inval")       { if (!next(o.inval)) return false; }
        else if (a == "--flush")       { if (!next(o.flush)) return false; }
        else if (a == "--timeout-ms")  { if (!next(o.timeout_ms)) return false; }
        else if (a == "--host-inval")  { if (!next(o.host_inval)) return false; }
        else if (a == "--host-clean")  { if (!next(o.host_clean)) return false; }
        else if (a == "--spin-max")    { if (!next(o.spin_max)) return false; }
        else if (a == "--cl-policy")   { if (!next(o.cl_policy)) return false; }
        else if (a == "--cpu")         { if (!next(o.cpu)) return false; }
        else if (a == "--no-flush")    { o.no_flush = 1; }
        else if (a == "-v")            { o.verbose = 1; }
        else if (a == "--modes") {
            if (i + 1 >= argc) return false;
            std::string m = argv[++i];
            for (int k = 0; k < M_COUNT; k++) o.modes[k] = false;
            size_t s = 0;
            while (s <= m.size()) {
                size_t e = m.find(',', s);
                if (e == std::string::npos) e = m.size();
                std::string tok = m.substr(s, e - s);
                bool found = false;
                for (int k = 0; k < M_COUNT; k++) if (tok == mode_name[k]) { o.modes[k] = true; found = true; }
                if (!found && !tok.empty()) { fprintf(stderr, "unknown mode %s\n", tok.c_str()); return false; }
                s = e + 1;
            }
        }
        else { usage(); return false; }
    }
    if (o.payload_kb < 0 || (size_t) o.payload_kb * 1024 > PAYLOAD_MAX) { fprintf(stderr, "payload too large\n"); return false; }
    if (o.dsp_payload_kb < 0 || (size_t) o.dsp_payload_kb * 1024 > DSP_PAY_MAX) { fprintf(stderr, "dsp payload too large\n"); return false; }
    return true;
}

// ---------------------------------------------------------------------------------------
// OpenCL side
// ---------------------------------------------------------------------------------------

static const char * cl_src_basic = R"CL(
__kernel void fill_payload(__global uchar * base, uint off, uint n, uint seed) {
    uint i = get_global_id(0);
    if (i < n) {
        ((__global uint *)(base + off))[i] = seed ^ i;
    }
}
__kernel void set_flag(__global uchar * base, uint off, uint v) {
    if (get_global_id(0) == 0) {
        *(__global volatile uint *)(base + off) = v;
    }
}
// one work-group: payload then flag, ordered by the barrier
__kernel void fill_and_flag(__global uchar * base, uint off, uint n, uint seed, uint done_off, uint done_val) {
    __global uint * p = (__global uint *)(base + off);
    for (uint i = get_local_id(0); i < n; i += get_local_size(0)) {
        p[i] = seed ^ i;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        *(__global volatile uint *)(base + done_off) = done_val;
    }
}
)CL";

// gpusvm: the flag lives in fine-grain SVM (host relays the HTP's ready), payload + done go to the ION alias.
static const char * cl_src_svm = R"CL(
__kernel void svm_spin_fill(__global uint * svm, uint want, __global uchar * base, uint payload_off, uint n, uint seed,
                            uint done_off, uint done_val, uint max_iters, uint dsp_off, uint dsp_n) {
    __global uint * p = (__global uint *)(base + payload_off);
    if (get_local_id(0) == 0) {
        atomic_store_explicit((volatile __global atomic_uint *)(svm + 16), 1u, memory_order_seq_cst, memory_scope_all_svm_devices);
        uint it = 0;
        while (atomic_load_explicit((volatile __global atomic_uint *) svm, memory_order_acquire, memory_scope_all_svm_devices) != want && it < max_iters) {
            it++;
        }
        atomic_store_explicit((volatile __global atomic_uint *)(svm + 32), it, memory_order_seq_cst, memory_scope_all_svm_devices);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    // Read what the DSP wrote while this kernel was already running (the Q / new-K/V direction).
    {
        __global const uint * d = (__global const uint *)(base + dsp_off);
        uint bad = 0;
        for (uint i = get_local_id(0); i < dsp_n; i += get_local_size(0)) {
            bad += (d[i] != (seed ^ 0xA5A5A5A5u ^ i)) ? 1u : 0u;
        }
        if (bad) {
            atomic_fetch_add_explicit((volatile __global atomic_uint *)(svm + 48), bad, memory_order_relaxed, memory_scope_all_svm_devices);
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (uint i = get_local_id(0); i < n; i += get_local_size(0)) {
        p[i] = seed ^ i;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        *(__global volatile uint *)(base + done_off) = done_val;
    }
}
)CL";

// One work-group. Item 0 announces itself, spins on 'ready', then the group fills the payload
// and item 0 releases 'done'.
static const char * cl_src_spin_atomic = R"CL(
__kernel void spin_fill(__global uchar * base, uint ready_off, uint ready_val, uint payload_off, uint n, uint seed,
                        uint done_off, uint done_val, uint alive_off, uint max_iters, uint iters_off) {
    __global volatile uint * ready = (__global volatile uint *)(base + ready_off);
    __global uint * p = (__global uint *)(base + payload_off);
    if (get_local_id(0) == 0) {
        atomic_store_explicit((volatile __global atomic_uint *)(base + alive_off), 1u, memory_order_seq_cst, memory_scope_device);
        uint it = 0;
        while (atomic_load_explicit((volatile __global atomic_uint *) ready, memory_order_acquire, memory_scope_device) != ready_val && it < max_iters) {
            it++;
        }
        *(__global uint *)(base + iters_off) = it;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (uint i = get_local_id(0); i < n; i += get_local_size(0)) {
        p[i] = seed ^ i;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        atomic_store_explicit((volatile __global atomic_uint *)(base + done_off), done_val, memory_order_seq_cst, memory_scope_device);
    }
}
)CL";

static const char * cl_src_spin_plain = R"CL(
__kernel void spin_fill(__global uchar * base, uint ready_off, uint ready_val, uint payload_off, uint n, uint seed,
                        uint done_off, uint done_val, uint alive_off, uint max_iters, uint iters_off) {
    __global volatile uint * ready = (__global volatile uint *)(base + ready_off);
    __global uint * p = (__global uint *)(base + payload_off);
    if (get_local_id(0) == 0) {
        *(__global volatile uint *)(base + alive_off) = 1u;
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        uint it = 0;
        while (*ready != ready_val && it < max_iters) {
            it++;
        }
        *(__global uint *)(base + iters_off) = it;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (uint i = get_local_id(0); i < n; i += get_local_size(0)) {
        p[i] = seed ^ i;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (get_local_id(0) == 0) {
        *(__global volatile uint *)(base + done_off) = done_val;
    }
}
)CL";

struct gpu_side {
    cl_platform_id   plat  = nullptr;
    cl_device_id     dev   = nullptr;
    cl_context       ctx   = nullptr;
    cl_command_queue q     = nullptr;
    cl_program       prog  = nullptr;
    cl_program       sprog = nullptr;
    cl_kernel        k_fill = nullptr, k_flag = nullptr, k_fused = nullptr, k_spin = nullptr, k_svm = nullptr;
    cl_program       vprog = nullptr;
    cl_mem           alias = nullptr;
    bool             spin_ok = false;
    bool             svm_ok  = false;
    uint32_t *       svm     = nullptr;   // fine-grain SVM block: [0] ready relay, [16] alive, [32] iterations
    std::string      spin_variant;
};

static bool cl_check(cl_int err, const char * what) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL: %s failed (%d)\n", what, err);
        return false;
    }
    return true;
}

static cl_program cl_build(gpu_side & g, const char * src, const char * opts, const char * label) {
    cl_int err;
    cl_program p = clCreateProgramWithSource(g.ctx, 1, &src, nullptr, &err);
    if (!cl_check(err, "clCreateProgramWithSource")) return nullptr;
    err = clBuildProgram(p, 1, &g.dev, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t n = 0;
        clGetProgramBuildInfo(p, g.dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        clGetProgramBuildInfo(p, g.dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        fprintf(stderr, "OpenCL: build of %s failed (%d):\n%s\n", label, err, log.c_str());
        clReleaseProgram(p);
        return nullptr;
    }
    return p;
}

static bool gpu_init(gpu_side & g, const options & o, uint8_t * base, int fd) {
    cl_int err;
    cl_uint np = 0;
    if (!cl_check(clGetPlatformIDs(1, &g.plat, &np), "clGetPlatformIDs") || np == 0) return false;
    cl_uint nd = 0;
    if (!cl_check(clGetDeviceIDs(g.plat, CL_DEVICE_TYPE_GPU, 1, &g.dev, &nd), "clGetDeviceIDs") || nd == 0) return false;

    char name[256] = {0}, ver[256] = {0};
    clGetDeviceInfo(g.dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    clGetDeviceInfo(g.dev, CL_DEVICE_VERSION, sizeof(ver), ver, nullptr);
    size_t elen = 0;
    clGetDeviceInfo(g.dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &elen);
    std::string ext(elen, '\0');
    clGetDeviceInfo(g.dev, CL_DEVICE_EXTENSIONS, elen, ext.data(), nullptr);
    printf("gpu: %s | %s\n", name, ver);
    printf("gpu: extensions: %s\n", ext.c_str());
    cl_device_svm_capabilities svm = 0;
    clGetDeviceInfo(g.dev, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm), &svm, nullptr);
    printf("gpu: svm caps: coarse-buffer %d fine-buffer %d fine-system %d atomics %d\n",
           !!(svm & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER), !!(svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER),
           !!(svm & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM), !!(svm & CL_DEVICE_SVM_ATOMICS));
    printf("gpu: ext_host_ptr %s  iocoherent %s  ion_host_ptr %s  dmabuf_host_ptr %s\n",
           ext.find("cl_qcom_ext_host_ptr ") != std::string::npos || ext.find("cl_qcom_ext_host_ptr\0") != std::string::npos ? "yes" : "no",
           ext.find("cl_qcom_ext_host_ptr_iocoherent") != std::string::npos ? "yes" : "no",
           ext.find("cl_qcom_ion_host_ptr") != std::string::npos ? "yes" : "no",
           ext.find("cl_qcom_dmabuf_host_ptr") != std::string::npos ? "yes" : "no");

    g.ctx = clCreateContext(nullptr, 1, &g.dev, nullptr, nullptr, &err);
    if (!cl_check(err, "clCreateContext")) return false;
    cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    g.q = clCreateCommandQueueWithProperties(g.ctx, g.dev, qp, &err);
    if (!cl_check(err, "clCreateCommandQueueWithProperties")) return false;

    g.prog = cl_build(g, cl_src_basic, "", "basic kernels");
    if (!g.prog) return false;
    g.k_fill = clCreateKernel(g.prog, "fill_payload", &err); if (!cl_check(err, "clCreateKernel fill_payload")) return false;
    g.k_flag = clCreateKernel(g.prog, "set_flag", &err);     if (!cl_check(err, "clCreateKernel set_flag")) return false;
    g.k_fused = clCreateKernel(g.prog, "fill_and_flag", &err); if (!cl_check(err, "clCreateKernel fill_and_flag")) return false;

    if ((svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) && (svm & CL_DEVICE_SVM_ATOMICS)) {
        g.vprog = cl_build(g, cl_src_svm, "-cl-std=CL2.0", "svm spin kernel");
        if (g.vprog) {
            g.k_svm = clCreateKernel(g.vprog, "svm_spin_fill", &err);
            if (err == CL_SUCCESS) {
                g.svm = (uint32_t *) clSVMAlloc(g.ctx, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, 4096, 64);
                g.svm_ok = g.svm != nullptr;
                if (!g.svm_ok) fprintf(stderr, "gpu: clSVMAlloc fine-grain+atomics failed\n");
            }
        }
    } else {
        printf("gpu: no fine-grain SVM with atomics -> gpusvm mode unavailable\n");
    }

    g.sprog = cl_build(g, cl_src_spin_atomic, "-cl-std=CL2.0", "spin kernel (atomics)");
    g.spin_variant = "atomic";
    if (!g.sprog) {
        g.sprog = cl_build(g, cl_src_spin_plain, "", "spin kernel (plain volatile)");
        g.spin_variant = "plain-volatile";
    }
    if (g.sprog) {
        g.k_spin = clCreateKernel(g.sprog, "spin_fill", &err);
        g.spin_ok = (err == CL_SUCCESS);
    }

    // Alias the hexagon rpcmem buffer: same pages, no copy (mllm het-pipeline recipe, validated on SM8750).
    cl_uint page = 0;
    clGetDeviceInfo(g.dev, CL_DEVICE_PAGE_SIZE_QCOM, sizeof(page), &page, nullptr);
    if (page == 0) page = 4096;
    cl_uint pad = 0;
    clGetDeviceInfo(g.dev, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof(pad), &pad, nullptr);
    printf("gpu: page %u pad %u  buffer base %p fd %d (%s aligned)\n", page, pad, (void *) base, fd,
           ((uintptr_t) base % page) == 0 ? "page" : "NOT page");

    const cl_uint policies[] = { o.cl_policy ? (cl_uint) o.cl_policy : (cl_uint) CL_MEM_HOST_IOCOHERENT_QCOM,
                                 CL_MEM_HOST_WRITEBACK_QCOM, CL_MEM_HOST_UNCACHED_QCOM };
    for (cl_uint pol : policies) {
        cl_mem_ion_host_ptr ion = {};
        ion.ext_host_ptr.allocation_type  = CL_MEM_ION_HOST_PTR_QCOM;
        ion.ext_host_ptr.host_cache_policy = pol;
        ion.ion_filedesc = fd;
        ion.ion_hostptr  = base;
        g.alias = clCreateBuffer(g.ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM, BUF_SIZE, &ion, &err);
        if (err == CL_SUCCESS) {
            printf("gpu: ION alias created with host_cache_policy 0x%x%s\n", pol, pol == CL_MEM_HOST_IOCOHERENT_QCOM ? " (IOCOHERENT)" : "");
            break;
        }
        fprintf(stderr, "gpu: ION alias with policy 0x%x failed (%d)\n", pol, err);
        g.alias = nullptr;
        if (o.cl_policy) break;
    }
    if (!g.alias) return false;

    // Sanity: GPU writes a sentinel, the host reads it through the rpcmem pointer.
    volatile uint32_t * sent = (volatile uint32_t *) (base + SENT_OFF);
    *sent = 0;
    dc_cvac(base + SENT_OFF, 64);
    cl_uint off = SENT_OFF, val = 0xC0FFEE01u;
    clSetKernelArg(g.k_flag, 0, sizeof(cl_mem), &g.alias);
    clSetKernelArg(g.k_flag, 1, sizeof(cl_uint), &off);
    clSetKernelArg(g.k_flag, 2, sizeof(cl_uint), &val);
    size_t one = 1;
    err = clEnqueueNDRangeKernel(g.q, g.k_flag, 1, nullptr, &one, nullptr, 0, nullptr, nullptr);
    if (!cl_check(err, "sentinel enqueue")) return false;
    clFinish(g.q);
    dc_civac(base + SENT_OFF, 64);
    const uint32_t got = *sent;
    printf("gpu: alias sentinel %s (host read 0x%08x)\n", got == val ? "OK -- GPU write visible through rpcmem pointer" : "FAILED", got);
    if (got != val) return false;
    return true;
}

// ---------------------------------------------------------------------------------------
// per-run bookkeeping
// ---------------------------------------------------------------------------------------

struct run_result {
    bool     ok        = false;   // record valid, no timeouts, payload correct
    bool     dsp_timeout = false;
    bool     host_timeout = false;
    uint64_t rec[HTP_SYNC_PROBE_R_N] = {0};
    uint64_t T0 = 0, T0b = 0, R = 0, W = 0, E = 0, T1 = 0;   // host cntvct
    uint64_t host_polls = 0;
    double   gpu_queued = 0, gpu_submit = 0, gpu_start = 0, gpu_end = 0;  // cntvct (double) via realtime map
    uint32_t spin_iters = 0;
    uint64_t t_alive = 0;
    bool     alive_seen = false;
    uint32_t dsp_pay_bad_gpu = 0;   // gpusvm: words of the DSP payload the GPU saw wrong
    uint32_t dsp_pay_bad_host = 0;  // cpu: words of the DSP payload the host saw wrong
};

struct stat_acc {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    bool empty() const { return v.empty(); }
    double med() { if (v.empty()) return NAN; std::vector<double> s = v; std::sort(s.begin(), s.end()); return s[s.size() / 2]; }
    double mn()  { return v.empty() ? NAN : *std::min_element(v.begin(), v.end()); }
    double mx()  { return v.empty() ? NAN : *std::max_element(v.begin(), v.end()); }
};

static void set_params(ggml_tensor * dst, const options & o, int mode, uint32_t n_payload, uint32_t seed) {
    int32_t * p = dst->op_params;
    memset(p, 0, sizeof(dst->op_params));
    p[HTP_SYNC_PROBE_P_MAGIC]         = (int32_t) HTP_SYNC_PROBE_MAGIC;
    p[HTP_SYNC_PROBE_P_MODE]          = mode == M_PING ? 1 : 0;
    p[HTP_SYNC_PROBE_P_TIMEOUT_US]    = o.timeout_ms * 1000;
    p[HTP_SYNC_PROBE_P_PAYLOAD_WORDS] = (int32_t) n_payload;
    p[HTP_SYNC_PROBE_P_INVAL]         = o.inval;
    p[HTP_SYNC_PROBE_P_READY_OFF]     = (int32_t) READY_OFF;
    p[HTP_SYNC_PROBE_P_DONE_OFF]      = (int32_t) DONE_OFF;
    p[HTP_SYNC_PROBE_P_PAYLOAD_OFF]   = (int32_t) PAYLOAD_OFF;
    p[HTP_SYNC_PROBE_P_SEED]          = (int32_t) seed;
    p[HTP_SYNC_PROBE_P_FLUSH]         = o.flush;
    p[HTP_SYNC_PROBE_P_READY_VAL]     = 1;
    p[HTP_SYNC_PROBE_P_DONE_VAL]      = 1;
    p[HTP_SYNC_PROBE_P_DSP_PAYLOAD_WORDS] = mode == M_PING ? 0 : o.dsp_payload_kb * 256;
    p[HTP_SYNC_PROBE_P_DSP_PAYLOAD_OFF]   = (int32_t) DSP_PAY_OFF;
}

static void noop_custom(ggml_tensor *, int, int, void *) {}

static bool host_wait_flag(const options & o, volatile uint32_t * f, uint32_t want, uint64_t timeout_ticks, uint64_t & polls, uint64_t & t_seen) {
    const uint64_t t0 = cnt_now();
    polls = 0;
    for (;;) {
        if (o.host_inval) dc_civac((const void *) f, 4);
        const uint32_t v = __atomic_load_n(f, __ATOMIC_ACQUIRE);
        polls++;
        if (v == want) { t_seen = cnt_now(); return true; }
        if (cnt_now() - t0 > timeout_ticks) { t_seen = cnt_now(); return false; }
    }
}

static run_result run_one(const options & o, int mode, int it, ggml_backend_t be, ggml_cgraph * graph, ggml_tensor * dst,
                          uint8_t * base, gpu_side & g) {
    run_result r;
    const uint32_t n_payload = (mode == M_GPU0 || mode == M_PING) ? 0 : (uint32_t) o.payload_kb * 256;
    const uint32_t seed      = 0x9E3779B9u + (uint32_t) it * 0x1001u + (uint32_t) mode;

    volatile uint32_t * ready = (volatile uint32_t *) (base + READY_OFF);
    volatile uint32_t * done  = (volatile uint32_t *) (base + DONE_OFF);
    volatile uint32_t * alive = (volatile uint32_t *) (base + ALIVE_OFF);
    uint32_t *          pay   = (uint32_t *) (base + PAYLOAD_OFF);
    uint64_t *          rec   = (uint64_t *) (base + REC_OFF);

    // Reset shared state. Clean it out of the CPU cache so no dirty CPU line can later overwrite
    // what the DSP/GPU write.
    memset(rec, 0, HTP_SYNC_PROBE_R_N * sizeof(uint64_t));
    *ready = 0; *done = 0; *alive = 0;
    *(uint32_t *) (base + ITERS_OFF) = 0;
    if (n_payload) memset(pay, 0xFF, n_payload * 4);
    const uint32_t dsp_n = mode == M_PING ? 0 : (uint32_t) o.dsp_payload_kb * 256;
    uint32_t * dsp_pay = (uint32_t *) (base + DSP_PAY_OFF);
    if (dsp_n) memset(dsp_pay, 0xEE, dsp_n * 4);
    if (o.host_clean) {
        if (dsp_n) dc_cvac(dsp_pay, dsp_n * 4);
        dc_cvac(base + READY_OFF, 64); dc_cvac(base + DONE_OFF, 64); dc_cvac(base + ALIVE_OFF, 256);
        dc_cvac(rec, 4096);
        if (n_payload) dc_cvac(pay, n_payload * 4);
    }

    set_params(dst, o, mode, n_payload, seed);
    if (mode == M_GPUSPIN) {
        // the GPU spin may be slow to notice the flag; give the DSP poll at least 2 s so a late GPU still lands
        dst->op_params[HTP_SYNC_PROBE_P_TIMEOUT_US] = std::max(o.timeout_ms, 2000) * 1000;
    }

    const uint64_t timeout_ticks = (uint64_t) o.timeout_ms * 19200ull;
    cl_event ev = nullptr;

    if (mode == M_GPUSVM) {
        __atomic_store_n(&g.svm[0], 0u, __ATOMIC_SEQ_CST);
        __atomic_store_n(&g.svm[16], 0u, __ATOMIC_SEQ_CST);
        __atomic_store_n(&g.svm[32], 0u, __ATOMIC_SEQ_CST);
        __atomic_store_n(&g.svm[48], 0u, __ATOMIC_SEQ_CST);
        cl_uint a_want = 1, a_pay = PAYLOAD_OFF, a_n = n_payload, a_seed = seed, a_done = DONE_OFF, a_dv = 1, a_max = (cl_uint) o.spin_max;
        cl_uint a_doff = DSP_PAY_OFF, a_dn = dsp_n;
        clSetKernelArg(g.k_svm, 9, sizeof(cl_uint), &a_doff);
        clSetKernelArg(g.k_svm, 10, sizeof(cl_uint), &a_dn);
        clSetKernelArgSVMPointer(g.k_svm, 0, g.svm);
        clSetKernelArg(g.k_svm, 1, sizeof(cl_uint), &a_want);
        clSetKernelArg(g.k_svm, 2, sizeof(cl_mem), &g.alias);
        clSetKernelArg(g.k_svm, 3, sizeof(cl_uint), &a_pay);
        clSetKernelArg(g.k_svm, 4, sizeof(cl_uint), &a_n);
        clSetKernelArg(g.k_svm, 5, sizeof(cl_uint), &a_seed);
        clSetKernelArg(g.k_svm, 6, sizeof(cl_uint), &a_done);
        clSetKernelArg(g.k_svm, 7, sizeof(cl_uint), &a_dv);
        clSetKernelArg(g.k_svm, 8, sizeof(cl_uint), &a_max);
        size_t gsz = 256, lsz = 256;
        cl_int err = clEnqueueNDRangeKernel(g.q, g.k_svm, 1, nullptr, &gsz, &lsz, 0, nullptr, &ev);
        if (!cl_check(err, "svm spin enqueue")) return r;
        clFlush(g.q);
        // wait for the kernel to be spinning (its alive store goes through fine-grain SVM)
        const uint64_t t0 = cnt_now();
        while (__atomic_load_n(&g.svm[16], __ATOMIC_ACQUIRE) != 1u) {
            if (cnt_now() - t0 > 200ull * 19200ull) break;
        }
        r.t_alive = cnt_now();
        r.alive_seen = __atomic_load_n(&g.svm[16], __ATOMIC_ACQUIRE) == 1u;
    }

    if (mode == M_GPUSPIN) {
        cl_uint a_ready = READY_OFF, a_rv = 1, a_pay = PAYLOAD_OFF, a_n = n_payload, a_seed = seed,
                a_done = DONE_OFF, a_dv = 1, a_alive = ALIVE_OFF, a_max = (cl_uint) o.spin_max, a_it = ITERS_OFF;
        clSetKernelArg(g.k_spin, 0, sizeof(cl_mem), &g.alias);
        clSetKernelArg(g.k_spin, 1, sizeof(cl_uint), &a_ready);
        clSetKernelArg(g.k_spin, 2, sizeof(cl_uint), &a_rv);
        clSetKernelArg(g.k_spin, 3, sizeof(cl_uint), &a_pay);
        clSetKernelArg(g.k_spin, 4, sizeof(cl_uint), &a_n);
        clSetKernelArg(g.k_spin, 5, sizeof(cl_uint), &a_seed);
        clSetKernelArg(g.k_spin, 6, sizeof(cl_uint), &a_done);
        clSetKernelArg(g.k_spin, 7, sizeof(cl_uint), &a_dv);
        clSetKernelArg(g.k_spin, 8, sizeof(cl_uint), &a_alive);
        clSetKernelArg(g.k_spin, 9, sizeof(cl_uint), &a_max);
        clSetKernelArg(g.k_spin, 10, sizeof(cl_uint), &a_it);
        size_t gsz = 256, lsz = 256;
        cl_int err = clEnqueueNDRangeKernel(g.q, g.k_spin, 1, nullptr, &gsz, &lsz, 0, nullptr, &ev);
        if (!cl_check(err, "spin enqueue")) return r;
        clFlush(g.q);
        uint64_t polls = 0;
        // 50 ms is plenty for a launch; if 'alive' never shows, the GPU's mid-kernel stores are not
        // reaching memory (or the launch is deferred). Recorded, not fatal: the run continues so the
        // spin kernel's end stamp still tells whether it saw 'ready'.
        r.alive_seen = host_wait_flag(o, alive, 1, 50ull * 19200ull, polls, r.t_alive);
    }

    r.T0 = cnt_now();
    ggml_backend_graph_compute_async(be, graph);
    r.T0b = cnt_now();

    if (mode == M_CPU || mode == M_GPU || mode == M_GPU0 || mode == M_GPU1 || mode == M_GPUSVM) {
        if (!host_wait_flag(o, ready, 1, timeout_ticks, r.host_polls, r.R)) {
            r.host_timeout = true;
        } else if (mode == M_GPUSVM) {
            __atomic_store_n(&g.svm[0], 1u, __ATOMIC_SEQ_CST);   // relay: the spinning kernel picks this up
            r.W = cnt_now();
        } else if (mode == M_GPU1) {
            cl_uint a_off = PAYLOAD_OFF, a_n = n_payload, a_seed = seed, f_off = DONE_OFF, f_val = 1;
            clSetKernelArg(g.k_fused, 0, sizeof(cl_mem), &g.alias);
            clSetKernelArg(g.k_fused, 1, sizeof(cl_uint), &a_off);
            clSetKernelArg(g.k_fused, 2, sizeof(cl_uint), &a_n);
            clSetKernelArg(g.k_fused, 3, sizeof(cl_uint), &a_seed);
            clSetKernelArg(g.k_fused, 4, sizeof(cl_uint), &f_off);
            clSetKernelArg(g.k_fused, 5, sizeof(cl_uint), &f_val);
            size_t gsz = 256, lsz = 256;
            clEnqueueNDRangeKernel(g.q, g.k_fused, 1, nullptr, &gsz, &lsz, 0, nullptr, &ev);
            if (!o.no_flush) clFlush(g.q);
            r.E = cnt_now();
        } else if (mode == M_CPU) {
            if (dsp_n) {
                if (o.host_inval) dc_civac(dsp_pay, dsp_n * 4);
                for (uint32_t i = 0; i < dsp_n; i++) r.dsp_pay_bad_host += (dsp_pay[i] != (seed ^ 0xA5A5A5A5u ^ i));
            }
            for (uint32_t i = 0; i < n_payload; i++) pay[i] = seed ^ i;
            if (o.host_clean && n_payload) dc_cvac(pay, n_payload * 4);
            __atomic_store_n(done, 1u, __ATOMIC_RELEASE);
            if (o.host_clean) dc_cvac(base + DONE_OFF, 64);
            r.W = cnt_now();
        } else {
            cl_uint a_off = PAYLOAD_OFF, a_n = n_payload, a_seed = seed, f_off = DONE_OFF, f_val = 1;
            if (n_payload) {
                clSetKernelArg(g.k_fill, 0, sizeof(cl_mem), &g.alias);
                clSetKernelArg(g.k_fill, 1, sizeof(cl_uint), &a_off);
                clSetKernelArg(g.k_fill, 2, sizeof(cl_uint), &a_n);
                clSetKernelArg(g.k_fill, 3, sizeof(cl_uint), &a_seed);
                size_t lsz = 256, gsz = ((n_payload + 255) / 256) * 256;
                clEnqueueNDRangeKernel(g.q, g.k_fill, 1, nullptr, &gsz, &lsz, 0, nullptr, nullptr);
            }
            clSetKernelArg(g.k_flag, 0, sizeof(cl_mem), &g.alias);
            clSetKernelArg(g.k_flag, 1, sizeof(cl_uint), &f_off);
            clSetKernelArg(g.k_flag, 2, sizeof(cl_uint), &f_val);
            size_t one = 1;
            clEnqueueNDRangeKernel(g.q, g.k_flag, 1, nullptr, &one, nullptr, 0, nullptr, &ev);
            if (!o.no_flush) clFlush(g.q);
            r.E = cnt_now();
        }
    }

    ggml_backend_synchronize(be);
    r.T1 = cnt_now();

    if (ev) {
        clWaitForEvents(1, &ev);
        cl_ulong tq = 0, ts = 0, t0 = 0, t1 = 0;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_QUEUED, sizeof(tq), &tq, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_SUBMIT, sizeof(ts), &ts, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START,  sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,    sizeof(t1), &t1, nullptr);
        r.gpu_queued = cnt_of_realtime(tq); r.gpu_submit = cnt_of_realtime(ts);
        r.gpu_start  = cnt_of_realtime(t0); r.gpu_end    = cnt_of_realtime(t1);
        clReleaseEvent(ev);
    }
    if (mode == M_GPUSPIN) {
        clFinish(g.q);
        dc_civac(base + ITERS_OFF, 64);
        r.spin_iters = *(volatile uint32_t *) (base + ITERS_OFF);
    }
    if (mode == M_GPUSVM) {
        clFinish(g.q);
        r.spin_iters = __atomic_load_n(&g.svm[32], __ATOMIC_ACQUIRE);
        r.dsp_pay_bad_gpu = __atomic_load_n(&g.svm[48], __ATOMIC_ACQUIRE);
    }

    dc_civac(rec, 4096);
    memcpy(r.rec, rec, sizeof(r.rec));
    r.dsp_timeout = r.rec[HTP_SYNC_PROBE_R_STATUS] != 0;
    const bool payload_ok = (mode == M_PING) || (r.rec[HTP_SYNC_PROBE_R_MISMATCH] == 0 && r.dsp_pay_bad_gpu == 0 && r.dsp_pay_bad_host == 0);
    r.ok = !r.dsp_timeout && !r.host_timeout && payload_ok && r.rec[HTP_SYNC_PROBE_R_T_ENTRY] != 0;

    if (o.verbose) {
        const uint64_t * q = r.rec;
        printf("  [%s #%d] %s entry->ready %.1f  rt(dsp) %.1f us  polls %" PRIu64 "  first-val %" PRIu64 "  mism %" PRIu64 " (first %" PRId64 ")%s%s  host: total %.1f us%s\n",
               mode_name[mode], it, r.ok ? "ok " : "BAD",
               ticks_us((double) (q[HTP_SYNC_PROBE_R_T_READY] - q[HTP_SYNC_PROBE_R_T_ENTRY])),
               ticks_us((double) (q[HTP_SYNC_PROBE_R_T_DONE] - q[HTP_SYNC_PROBE_R_T_READY])),
               q[HTP_SYNC_PROBE_R_POLLS], q[HTP_SYNC_PROBE_R_FIRST_VAL], q[HTP_SYNC_PROBE_R_MISMATCH], (int64_t) q[HTP_SYNC_PROBE_R_FIRST_BAD],
               r.dsp_timeout ? " DSP-TIMEOUT" : "", r.host_timeout ? " HOST-TIMEOUT" : "",
               ticks_us((double) (r.T1 - r.T0)),
               (mode == M_GPUSPIN || mode == M_GPUSVM) ? (std::string("  spin-iters ") + std::to_string(r.spin_iters) + (r.alive_seen ? " alive-seen" : " alive-NOT-seen")).c_str() : "");
        if (ev == nullptr && (mode == M_GPU || mode == M_GPU0 || mode == M_GPU1 || mode == M_GPUSPIN || mode == M_GPUSVM) && r.gpu_end != 0) {
            printf("      gpu: queued->submit %.1f  submit->start %.1f  start->end %.1f us\n",
                   ticks_us(r.gpu_submit - r.gpu_queued), ticks_us(r.gpu_start - r.gpu_submit), ticks_us(r.gpu_end - r.gpu_start));
        }
    }
    return r;
}

// ---------------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    options o;
    if (!parse(argc, argv, o)) return 1;
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (o.cpu >= 0) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(o.cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) perror("sched_setaffinity");
    }

    // Both must be set before the first ggml-hexagon call (read once at backend registration).
    setenv("GGML_HEXAGON_SYNC_PROBE", "1", 1);
    setenv("GGML_HEXAGON_ASYNC", "1", 1);

    calibrate_realtime();

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_dev_name(d), "HTP0") == 0) { dev = d; break; }
    }
    if (!dev) { fprintf(stderr, "HTP0 device not found\n"); return 1; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) { fprintf(stderr, "HTP0 init failed\n"); return 1; }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_t      buf  = ggml_backend_buft_alloc_buffer(buft, BUF_SIZE);
    if (!buf) { fprintf(stderr, "hexagon buffer alloc failed\n"); return 1; }
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
    const int fd   = rpcmem_to_fd(base);
    if (fd < 0) { fprintf(stderr, "rpcmem_to_fd failed (%d)\n", fd); return 1; }
    memset(base, 0, BUF_SIZE);

    // Graph: one CUSTOM node, src = flag area, dst = record, both hand-placed in the hexagon buffer.
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(8, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * flags = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (REC_OFF / 4));
    flags->buffer = buf; flags->data = base; ggml_set_name(flags, "sync_flags");
    ggml_backend_buffer_init_tensor(buf, flags);
    ggml_tensor * args[1] = { flags };
    ggml_tensor * dst = ggml_custom_4d(ctx, GGML_TYPE_F32, 64, 1, 1, 1, args, 1, noop_custom, 1, nullptr);
    dst->buffer = buf; dst->data = base + REC_OFF; ggml_set_name(dst, "sync_rec");
    ggml_backend_buffer_init_tensor(buf, dst);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, dst);

    set_params(dst, o, M_PING, 0, 0);  // the magic must be in op_params before the support check
    if (!ggml_backend_supports_op(be, dst)) {
        fprintf(stderr, "HTP0 does not accept the probe op (is the pushed libggml-hexagon.so / libggml-htp-v79.so current?)\n");
        return 1;
    }

    gpu_side g;
    const bool need_gpu = o.modes[M_GPU] || o.modes[M_GPU0] || o.modes[M_GPU1] || o.modes[M_GPUSPIN] || o.modes[M_GPUSVM];
    if (need_gpu && !gpu_init(g, o, base, fd)) {
        fprintf(stderr, "GPU side unavailable; running CPU/ping modes only\n");
        o.modes[M_GPU] = o.modes[M_GPU0] = o.modes[M_GPU1] = o.modes[M_GPUSPIN] = o.modes[M_GPUSVM] = false;
    }
    if (o.modes[M_GPUSPIN] && !g.spin_ok) {
        fprintf(stderr, "spin kernel unavailable; skipping gpuspin\n");
        o.modes[M_GPUSPIN] = false;
    }
    if (o.modes[M_GPUSVM] && !g.svm_ok) {
        fprintf(stderr, "fine-grain SVM unavailable; skipping gpusvm\n");
        o.modes[M_GPUSVM] = false;
    }
    if (o.modes[M_GPUSPIN]) printf("gpu: spin kernel variant: %s\n", g.spin_variant.c_str());

    printf("probe: iters %d payload %d KB  dsp-inval %d dsp-flush %d  host-inval %d host-clean %d  timeout %d ms  cpu %d%s\n",
           o.iters, o.payload_kb, o.inval, o.flush, o.host_inval, o.host_clean, o.timeout_ms, sched_getcpu(), o.no_flush ? "  no-flush" : "");

    // Warm-up: one of each so first-use costs (skel load, CL first launch) stay out of the stats.
    for (int m = 0; m < M_COUNT; m++) if (o.modes[m]) run_one(o, m, -1, be, graph, dst, base, g);

    std::vector<run_result> res[M_COUNT];
    for (int it = 0; it < o.iters; it++) {
        for (int m = 0; m < M_COUNT; m++) {
            if (!o.modes[m]) continue;
            res[m].push_back(run_one(o, m, it, be, graph, dst, base, g));
        }
    }

    // D = qtimer - cntvct bounds. Ping: entry in [T0, T1]. Handshake: T_READY - D <= R, W <= T_DONE - D.
    double D_lo = -1e30, D_hi = 1e30;
    for (auto & r : res[M_PING]) {
        if (!r.ok) continue;
        D_lo = std::max(D_lo, (double) r.rec[HTP_SYNC_PROBE_R_T_ENTRY] - (double) r.T1);
        D_hi = std::min(D_hi, (double) r.rec[HTP_SYNC_PROBE_R_T_ENTRY] - (double) r.T0);
    }
    for (auto & r : res[M_CPU]) {
        if (!r.ok) continue;
        D_lo = std::max(D_lo, (double) r.rec[HTP_SYNC_PROBE_R_T_READY] - (double) r.R);
        D_hi = std::min(D_hi, (double) r.rec[HTP_SYNC_PROBE_R_T_DONE] - (double) r.W);
    }
    // a sub-5us inversion is stamp-ordering noise, not an unbounded offset
    const bool have_D = D_lo > -1e29 && D_hi < 1e29 && (D_hi - D_lo) > -5.0 * 19.2;
    const double D = have_D ? 0.5 * (D_lo + D_hi) : 0.0;
    printf("\nclock offset D = qtimer - cntvct: %s [%.1f, %.1f] us -> using %.1f us (+-%.1f)\n",
           have_D ? "bounded" : "UNBOUNDED (cross-domain splits below are unreliable)",
           ticks_us(D_lo), ticks_us(D_hi), ticks_us(D), have_D ? ticks_us(0.5 * (D_hi - D_lo)) : 0.0);

    auto host_of = [&](uint64_t q) { return (double) q - D; };  // DSP qtimer -> host cntvct axis

    printf("\n%-8s %5s | %-30s %9s %9s %9s\n", "mode", "ok", "metric (us)", "median", "min", "max");
    for (int m = 0; m < M_COUNT; m++) {
        if (res[m].empty()) continue;
        int n_ok = 0, n_dsp_to = 0, n_host_to = 0, n_bad_payload = 0;
        int n_dsp_pay_bad = 0;
        stat_acc s_dsp_pay;
        stat_acc s_total, s_dispatch, s_entry_ready, s_rt, s_polls, s_poll_us, s_ready_host, s_dwell, s_oneway_sum, s_host_dsp,
                 s_gpu_enq_start, s_gpu_kernel, s_gpu_end_dsp, s_enq_to_end, s_check, s_spin_iters, s_alive_gap;
        for (auto & r : res[m]) {
            n_ok += r.ok; n_dsp_to += r.dsp_timeout; n_host_to += r.host_timeout;
            n_bad_payload += (!r.dsp_timeout && r.rec[HTP_SYNC_PROBE_R_MISMATCH] != 0);
            const uint64_t * q = r.rec;
            s_total.add(ticks_us((double) (r.T1 - r.T0)));
            if (q[HTP_SYNC_PROBE_R_T_ENTRY]) s_dispatch.add(ticks_us(host_of(q[HTP_SYNC_PROBE_R_T_ENTRY]) - (double) r.T0));
            if (m == M_PING || !r.ok) continue;
            s_entry_ready.add(ticks_us((double) (q[HTP_SYNC_PROBE_R_T_READY] - q[HTP_SYNC_PROBE_R_T_ENTRY])));
            s_dsp_pay.add(ticks_us((double) (q[HTP_SYNC_PROBE_R_T_DSP_PAYLOAD] - q[HTP_SYNC_PROBE_R_T_ENTRY])));
            n_dsp_pay_bad += (r.dsp_pay_bad_gpu != 0 || r.dsp_pay_bad_host != 0);
            const double rt = ticks_us((double) (q[HTP_SYNC_PROBE_R_T_DONE] - q[HTP_SYNC_PROBE_R_T_READY]));
            s_rt.add(rt);
            s_polls.add((double) q[HTP_SYNC_PROBE_R_POLLS]);
            if (q[HTP_SYNC_PROBE_R_POLLS]) s_poll_us.add(rt / (double) q[HTP_SYNC_PROBE_R_POLLS]);
            s_check.add(ticks_us((double) (q[HTP_SYNC_PROBE_R_T_CHECKED] - q[HTP_SYNC_PROBE_R_T_DONE])));
            if (m == M_CPU) {
                s_ready_host.add(ticks_us((double) r.R - host_of(q[HTP_SYNC_PROBE_R_T_READY])));
                s_dwell.add(ticks_us((double) (r.W - r.R)));
                s_oneway_sum.add(rt - ticks_us((double) (r.W - r.R)));
                s_host_dsp.add(ticks_us(host_of(q[HTP_SYNC_PROBE_R_T_DONE]) - (double) r.W));
            } else if (m == M_GPUSVM) {
                s_ready_host.add(ticks_us((double) r.R - host_of(q[HTP_SYNC_PROBE_R_T_READY])));
                s_dwell.add(ticks_us((double) (r.W - r.R)));
                s_spin_iters.add((double) r.spin_iters);
                s_enq_to_end.add(ticks_us(r.gpu_end - (double) r.W));
                s_gpu_end_dsp.add(ticks_us(host_of(q[HTP_SYNC_PROBE_R_T_DONE]) - r.gpu_end));
                s_gpu_kernel.add(ticks_us(r.gpu_end - r.gpu_start));
            } else if (m == M_GPU || m == M_GPU0 || m == M_GPU1) {
                s_ready_host.add(ticks_us((double) r.R - host_of(q[HTP_SYNC_PROBE_R_T_READY])));
                s_dwell.add(ticks_us((double) (r.E - r.R)));
                s_gpu_enq_start.add(ticks_us(r.gpu_start - (double) r.E));
                s_gpu_kernel.add(ticks_us(r.gpu_end - r.gpu_start));
                s_enq_to_end.add(ticks_us(r.gpu_end - (double) r.E));
                s_gpu_end_dsp.add(ticks_us(host_of(q[HTP_SYNC_PROBE_R_T_DONE]) - r.gpu_end));
                s_oneway_sum.add(rt - ticks_us((double) (r.E - r.R)) - ticks_us(r.gpu_end - (double) r.E));
            } else if (m == M_GPUSPIN) {
                s_spin_iters.add((double) r.spin_iters);
                if (r.alive_seen) s_alive_gap.add(ticks_us((double) r.T0 - (double) r.t_alive));
                s_gpu_kernel.add(ticks_us(r.gpu_end - r.gpu_start));
                s_gpu_end_dsp.add(ticks_us(host_of(q[HTP_SYNC_PROBE_R_T_DONE]) - r.gpu_end));
                s_enq_to_end.add(ticks_us(r.gpu_end - host_of(q[HTP_SYNC_PROBE_R_T_READY])));
            }
        }
        auto row = [&](const char * name, stat_acc & s) {
            if (s.empty()) return;
            printf("%-8s %5s | %-30s %9.1f %9.1f %9.1f\n", "", "", name, s.med(), s.mn(), s.mx());
        };
        char okbuf[32];
        snprintf(okbuf, sizeof(okbuf), "%d/%zu", n_ok, res[m].size());
        printf("%-8s %5s | %-30s\n", mode_name[m], okbuf,
               m == M_PING ? "" : (n_dsp_to || n_host_to || n_bad_payload ? "(dsp-timeout/host-timeout/bad-payload below)" : "payload correct on every run"));
        if (n_dsp_to || n_host_to || n_bad_payload) {
            printf("%-8s %5s |   dsp-timeouts %d  host-timeouts %d  bad-payload %d\n", "", "", n_dsp_to, n_host_to, n_bad_payload);
        }
        row("host: compute_async->sync total", s_total);
        row("dispatch: T0 -> op entry [D]", s_dispatch);
        if (m == M_PING) continue;
        row("dsp: write+flush its payload", s_dsp_pay);
        row("dsp: entry -> ready published", s_entry_ready);
        if (m == M_CPU || m == M_GPUSVM) {
            printf("%-8s %5s |   DSP-written payload (%d KB) read %s after ready: %d/%zu runs wrong\n", "", "",
                   o.dsp_payload_kb, m == M_CPU ? "by host" : "by pre-launched GPU kernel", n_dsp_pay_bad, res[m].size());
        }
        row("DSP ROUND TRIP ready->done seen", s_rt);
        row("dsp: poll iterations", s_polls);
        row("dsp: us per poll iteration", s_poll_us);
        row("dsp: payload check", s_check);
        if (m == M_CPU) {
            row("ready -> host saw it [D]", s_ready_host);
            row("host dwell (write payload+done)", s_dwell);
            row("host done -> dsp saw it [D]", s_host_dsp);
            row("sum of one-way latencies", s_oneway_sum);
        } else if (m == M_GPUSVM) {
            int n_alive = 0; for (auto & r : res[m]) n_alive += r.alive_seen;
            printf("%-8s %5s |   kernel 'alive' via SVM seen by host: %d/%zu\n", "", "", n_alive, res[m].size());
            row("ready -> host saw it [D]", s_ready_host);
            row("host dwell (svm relay store)", s_dwell);
            row("gpu spin iterations", s_spin_iters);
            row("svm relay -> gpu kernel end", s_enq_to_end);
            row("gpu: spin kernel start -> end", s_gpu_kernel);
            row("gpu end -> dsp saw done [D]", s_gpu_end_dsp);
        } else if (m == M_GPU || m == M_GPU0 || m == M_GPU1) {
            row("ready -> host saw it [D]", s_ready_host);
            row("host dwell (enqueue+flush)", s_dwell);
            row("gpu: enqueue -> kernel start", s_gpu_enq_start);
            row("gpu: flag kernel start -> end", s_gpu_kernel);
            row("gpu: enqueue -> kernel end", s_enq_to_end);
            row("gpu end -> dsp saw done [D]", s_gpu_end_dsp);
            row("residual: ready->host + end->dsp", s_oneway_sum);
        } else if (m == M_GPUSPIN) {
            int n_alive = 0; for (auto & r : res[m]) n_alive += r.alive_seen;
            printf("%-8s %5s |   'alive' store seen mid-kernel by host: %d/%zu\n", "", "", n_alive, res[m].size());
            row("gpu spin iterations", s_spin_iters);
            row("gpu alive -> compute_async gap", s_alive_gap);
            row("gpu: spin kernel start -> end", s_gpu_kernel);
            row("dsp ready -> gpu kernel end [D]", s_enq_to_end);
            row("gpu end -> dsp saw done [D]", s_gpu_end_dsp);
        }
    }
    printf("\n[D] = uses the clock offset estimate above; everything else is single-clock.\n");

    // Back-to-back pre-enqueued kernels on the in-order queue: how soon after kernel A ends does
    // kernel B (already submitted) start? This is the per-layer re-arm cost of the pre-launch design.
    if (o.b2b > 0 && g.svm_ok) {
        stat_acc s_gap, s_relay_a, s_relay_b;
        int n_ok = 0;
        for (int i = 0; i < o.b2b + 1; i++) {
            __atomic_store_n(&g.svm[0], 0u, __ATOMIC_SEQ_CST);
            __atomic_store_n(&g.svm[16], 0u, __ATOMIC_SEQ_CST);
            volatile uint32_t * done_a = (volatile uint32_t *) (base + DONE_OFF);
            volatile uint32_t * done_b = (volatile uint32_t *) (base + DONE_OFF + 64);
            *done_a = 0; *done_b = 0;
            dc_cvac(base + DONE_OFF, 128);
            cl_uint zero = 0, pay = PAYLOAD_OFF, seed = 1, off_a = DONE_OFF, off_b = DONE_OFF + 64, dv = 1, mx = (cl_uint) o.spin_max, doff = DSP_PAY_OFF;
            cl_uint want_a = 1, want_b = 2;
            cl_event ea = nullptr, eb = nullptr;
            size_t gsz = 256, lsz = 256;
            clSetKernelArgSVMPointer(g.k_svm, 0, g.svm);
            clSetKernelArg(g.k_svm, 1, sizeof(cl_uint), &want_a);
            clSetKernelArg(g.k_svm, 2, sizeof(cl_mem), &g.alias);
            clSetKernelArg(g.k_svm, 3, sizeof(cl_uint), &pay);
            clSetKernelArg(g.k_svm, 4, sizeof(cl_uint), &zero);
            clSetKernelArg(g.k_svm, 5, sizeof(cl_uint), &seed);
            clSetKernelArg(g.k_svm, 6, sizeof(cl_uint), &off_a);
            clSetKernelArg(g.k_svm, 7, sizeof(cl_uint), &dv);
            clSetKernelArg(g.k_svm, 8, sizeof(cl_uint), &mx);
            clSetKernelArg(g.k_svm, 9, sizeof(cl_uint), &doff);
            clSetKernelArg(g.k_svm, 10, sizeof(cl_uint), &zero);
            clEnqueueNDRangeKernel(g.q, g.k_svm, 1, nullptr, &gsz, &lsz, 0, nullptr, &ea);
            clSetKernelArg(g.k_svm, 1, sizeof(cl_uint), &want_b);
            clSetKernelArg(g.k_svm, 6, sizeof(cl_uint), &off_b);
            clEnqueueNDRangeKernel(g.q, g.k_svm, 1, nullptr, &gsz, &lsz, 0, nullptr, &eb);
            clFlush(g.q);
            // wait for A to be spinning, release it, wait for its done word, release B at once
            const uint64_t t0 = cnt_now();
            while (__atomic_load_n(&g.svm[16], __ATOMIC_ACQUIRE) != 1u && cnt_now() - t0 < 500ull * 19200ull) {}
            const uint64_t ra = cnt_now();
            __atomic_store_n(&g.svm[0], 1u, __ATOMIC_SEQ_CST);
            uint64_t polls = 0, seen_a = 0;
            const bool ok_a = host_wait_flag(o, done_a, 1, 500ull * 19200ull, polls, seen_a);
            const uint64_t rb = cnt_now();
            __atomic_store_n(&g.svm[0], 2u, __ATOMIC_SEQ_CST);
            uint64_t seen_b = 0;
            const bool ok_b = host_wait_flag(o, done_b, 1, 500ull * 19200ull, polls, seen_b);
            clFinish(g.q);
            cl_ulong a_end = 0, b_start = 0, b_end = 0;
            clGetEventProfilingInfo(ea, CL_PROFILING_COMMAND_END,   sizeof(a_end),   &a_end,   nullptr);
            clGetEventProfilingInfo(eb, CL_PROFILING_COMMAND_START, sizeof(b_start), &b_start, nullptr);
            clGetEventProfilingInfo(eb, CL_PROFILING_COMMAND_END,   sizeof(b_end),   &b_end,   nullptr);
            clReleaseEvent(ea); clReleaseEvent(eb);
            if (i == 0) continue;  // warm-up
            if (ok_a && ok_b) {
                n_ok++;
                s_gap.add(ticks_us(cnt_of_realtime(b_start) - cnt_of_realtime(a_end)));
                s_relay_a.add(ticks_us((double) seen_a - (double) ra));
                s_relay_b.add(ticks_us((double) seen_b - (double) rb));
            }
        }
        printf("\nback-to-back pre-enqueued kernels (%d/%d ok):\n", n_ok, o.b2b);
        printf("  %-40s %9.1f %9.1f %9.1f us (median/min/max)\n", "kernel A end -> kernel B start", s_gap.med(), s_gap.mn(), s_gap.mx());
        printf("  %-40s %9.1f %9.1f %9.1f us\n", "host relay -> A done visible to host", s_relay_a.med(), s_relay_a.mn(), s_relay_a.mx());
        printf("  %-40s %9.1f %9.1f %9.1f us\n", "host relay -> B done visible to host", s_relay_b.med(), s_relay_b.mn(), s_relay_b.mx());
    }

    if (g.svm) clSVMFree(g.ctx, g.svm);
    if (g.k_svm) clReleaseKernel(g.k_svm);
    if (g.k_fused) clReleaseKernel(g.k_fused);
    if (g.vprog) clReleaseProgram(g.vprog);
    if (g.alias) clReleaseMemObject(g.alias);
    if (g.k_spin) clReleaseKernel(g.k_spin);
    if (g.k_fill) clReleaseKernel(g.k_fill);
    if (g.k_flag) clReleaseKernel(g.k_flag);
    if (g.sprog) clReleaseProgram(g.sprog);
    if (g.prog) clReleaseProgram(g.prog);
    if (g.q) clReleaseCommandQueue(g.q);
    if (g.ctx) clReleaseContext(g.ctx);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    return 0;
}

// Why is CPU computation over a ggml-hexagon rpcmem buffer far slower than over ordinary
// memory, and which access patterns survive it?
//
// The number this tool exists to explain: the same antidiagonal permutation gather over a
// 4 MiB F32 tensor, on the CPU backend at 8 threads, cost 105 us over malloc'd memory,
// 242 us over a hexagon buffer the DSP had never run on, and 793 us over the same kind of
// buffer after ONE ggml_backend_graph_compute on the DSP -- and the 793 us was persistent,
// not a one-time cache invalidation (docs/backend/snapdragon/xattn-scoring.md:426-431).
//
// That 7.5x is not one effect. It is a chain of four independent changes, and the existing
// dataset changes all four at once:
//
//   M-def --(A: page size)--> M-4k --(B: ION/dmabuf)--> R-raw --(C: fastrpc_mmap)-->
//   R-virgin --(D: DSP HAP_mmap2)--> R-dsp
//
// Link C has never been measured by anyone: ggml_hexagon_shared_buffer::alloc() calls
// mmap() unconditionally (ggml-hexagon.cpp:402), so a "virgin" hexagon buffer has ALREADY
// been fastrpc_mmap'd with FASTRPC_MAP_FD_DELAYED (:352-354). If the 2.3x lives in link C
// and not link B, the TLB-reach hypothesis is dead and the fix is "do not register CPU-side
// working buffers with FastRPC", which is a completely different conclusion.
//
// So the probe measures every link separately, in ONE process, after HTP0 has been
// initialised -- htp_iface_start votes bus and core DCVS to HAP_DCVS_VCORNER_MAX with
// sleep_disable (htp/main.c:437-460) at session construction (ggml-hexagon.cpp:1874), so a
// malloc baseline taken in a CPU-only process is at a different DDR corner and is not a
// valid denominator. The old 105 us is suspect on exactly that ground.
//
// Every element here exists to discriminate between two hypotheses:
//   H1 TLB reach / 4 KiB pages        decided by M-4k vs M-def, and page-chase vs line-chase
//   H2 the mapping changes on DSP use decided by the persistence timeline (--run persist)
//   H3 access-pattern sensitivity     decided by the granularity curve, ratio(g)
//   H4 the process got slower         decided by M-def interleaved INSIDE every round
//
// R-unc (uncached rpcmem) is the positive control that no earlier pass had: it says what an
// attribute change to uncached actually costs on this exact memory, so the observed step can
// be compared against a known attribute change instead of guessed at.
//
// Nothing here changes backend behaviour. GGML_HEXAGON_VMEM / OPBATCH / OPSTAGE are set by
// the probe into its own environment before the backend is initialised, because
// scripts/snapdragon/adb/run-tool.sh does not forward them.
//
// Run (scripts/snapdragon/adb/run-tool.sh forwards nothing this needs, so it works as-is):
//   run-tool.sh llama-rpcmem-probe --run gonogo    ~1 min, one number: does anything survive
//   run-tool.sh llama-rpcmem-probe --run links     no DSP at all: locates the 2.3x in A/B/C
//   run-tool.sh llama-rpcmem-probe --run persist   the crux: one-time, permanent, reversible
//   run-tool.sh llama-rpcmem-probe --csv /data/local/tmp/probe.csv    everything
//
// Take the baseline BEFORE fixing the uninitialised remote_rpc_control_latency.latency at
// ggml-hexagon.cpp:1815-1824. RPC_PM_QOS controls CPU low-power modes keyed on RPC activity
// in a 100 ms window (remote.h:283-304), i.e. it acts on exactly the S0->S1 transition under
// study; changing it first would confound the measurement it is meant to be judged against.
//
// R6 -- allocation luck: R-dsp and R-virgin are different physical allocations, and the whole
// of link D could be an artifact of which one got the DSP role. Run --run all twice, the
// second time with --swap-roles. If the degraded label follows the ROLE it is real.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/utsname.h>
#include <unistd.h>

#ifdef __ANDROID__
#    include <sys/system_properties.h>
#endif

#ifndef MADV_HUGEPAGE
#    define MADV_HUGEPAGE 14
#endif
#ifndef MADV_NOHUGEPAGE
#    define MADV_NOHUGEPAGE 15
#endif

// rpcmem.h:85-89. Redefined locally so the example does not need the Hexagon SDK include
// path; these are the exact values ggml-hexagon.cpp:387 passes.
static const int RPCMEM_HEAP_ID_SYSTEM_ = 25;
// rpcmem.h:46-52: RPCMEM_DEFAULT_FLAGS is ION_FLAG_CACHED when that is defined and 1
// otherwise. ION_FLAG_CACHED is not defined anywhere in the SDK tree or the NDK sysroot, so
// the #else branch is what the backend actually compiles.
static const uint32_t RPCMEM_DEFAULT_FLAGS_ = 1;
// rpcmem.h:93-95.
static const uint32_t RPCMEM_FLAG_UNCACHED_ = 0;

// examples/xattn-split/xattn-split.cpp:4-6 -- the same gather costs HTP 272 us and the CPU
// 107 us. A CPU access pattern survives only if it stays under that multiple of its own
// malloc baseline; 1.0x is not the line and reporting against 1.0x is what makes a 2.3x
// look fatal when it is actually a win.
static const double SURVIVAL_LINE = 272.0 / 107.0;

static const size_t MiB = 1024 * 1024;

// ---------------------------------------------------------------------------------------
// libcdsprpc trampolines
//
// R-raw and R-unc must be rpcmem allocations that NO FastRPC call has ever seen, so they
// cannot come from the backend -- ggml_hexagon_shared_buffer::alloc() fastrpc_mmap's every
// buffer it makes (ggml-hexagon.cpp:402). Same library, same symbols the backend resolves
// (htp-drv.cpp:328, :352-355), resolved independently here.
// ---------------------------------------------------------------------------------------

typedef void * (*rpcmem_alloc_fn) (int, uint32_t, int);
typedef void * (*rpcmem_alloc2_fn)(int, uint32_t, size_t);
typedef void   (*rpcmem_free_fn)  (void *);
typedef int    (*rpcmem_to_fd_fn) (void *);

struct rpcmem_api {
    void *           lib    = nullptr;
    rpcmem_alloc_fn  alloc  = nullptr;
    rpcmem_alloc2_fn alloc2 = nullptr;
    rpcmem_free_fn   free   = nullptr;
    rpcmem_to_fd_fn  to_fd  = nullptr;

    bool ok() const { return (alloc || alloc2) && free && to_fd; }

    bool open() {
        lib = dlopen("libcdsprpc.so", RTLD_NOW);
        if (!lib) {
            return false;
        }
        alloc  = (rpcmem_alloc_fn)  dlsym(lib, "rpcmem_alloc");
        alloc2 = (rpcmem_alloc2_fn) dlsym(lib, "rpcmem_alloc2");   // htp-drv.cpp:353 dlsyms this with ignore=true
        free   = (rpcmem_free_fn)   dlsym(lib, "rpcmem_free");
        to_fd  = (rpcmem_to_fd_fn)  dlsym(lib, "rpcmem_to_fd");
        return ok();
    }

    void * get(uint32_t flags, size_t size) {
        if (alloc2) {
            return alloc2(RPCMEM_HEAP_ID_SYSTEM_, flags, size);
        }
        return alloc(RPCMEM_HEAP_ID_SYSTEM_, flags, (int) size);
    }
};

static rpcmem_api g_rpc;

// ---------------------------------------------------------------------------------------
// The kernels
//
// Exactly one copy loop exists in the binary and `seq` is the identity permutation through
// it. Pattern is a property of the DATA, not of the generated code, so the compiler cannot
// specialise one arm's loop differently from another's (R7). The explicit memcpy arm is a
// separate, clearly labelled kernel because it is a genuinely different code path -- it is
// mllm's path (ShaBlockSparsePromptProcessorSplit.cpp:878, :886, :1090).
// ---------------------------------------------------------------------------------------

#define LAUNDER(p) __asm__ volatile("" : "+r"(p))

__attribute__((noinline))
static uint64_t kern_copy(uint8_t * dst, const uint8_t * src, const uint32_t * perm,
                          size_t i0, size_t i1, size_t g) {
    LAUNDER(dst);
    LAUNDER(src);
    LAUNDER(perm);

    uint64_t acc = 0;
    for (size_t i = i0; i < i1; i++) {
        const uint8_t * s = src + (size_t) perm[i] * g;
        uint8_t *       d = dst + i * g;

        uint32_t w;
        memcpy(&w, s, sizeof(w));
        acc += w;

        for (size_t b = 0; b < g; b += 8) {
            uint64_t v;
            memcpy(&v, s + b, sizeof(v));
            memcpy(d + b, &v, sizeof(v));
        }
    }
    __asm__ volatile("" ::: "memory");
    return acc;
}

__attribute__((noinline))
static uint64_t kern_memcpy(uint8_t * dst, const uint8_t * src, const uint32_t * perm,
                            size_t i0, size_t i1, size_t g) {
    LAUNDER(dst);
    LAUNDER(src);
    LAUNDER(perm);

    uint64_t acc = 0;
    for (size_t i = i0; i < i1; i++) {
        const uint8_t * s = src + (size_t) perm[i] * g;
        uint8_t *       d = dst + i * g;

        uint32_t w;
        memcpy(&w, s, sizeof(w));
        acc += w;

        memcpy(d, s, g);
    }
    __asm__ volatile("" ::: "memory");
    return acc;
}

__attribute__((noinline))
static uint64_t kern_read(const uint8_t * src, const uint32_t * perm,
                          size_t i0, size_t i1, size_t g) {
    LAUNDER(src);
    LAUNDER(perm);

    uint64_t acc = 0;
    for (size_t i = i0; i < i1; i++) {
        const uint8_t * s = src + (size_t) perm[i] * g;
        for (size_t b = 0; b < g; b += 4) {
            uint32_t v;
            memcpy(&v, s + b, sizeof(v));
            acc += v;
        }
    }
    __asm__ volatile("" ::: "memory");
    return acc;
}

// A dependent chain: the load's result is the next load's address, so nothing overlaps and
// the cost is (number of accesses) x (latency of one access). page-chase and line-chase run
// the same number of steps; only the stride differs. That is the whole H1 tiebreaker -- a
// translation mechanism punishes the 4 KiB stride and leaves the 64 B stride alone, any
// bandwidth or memory-attribute mechanism punishes both.
__attribute__((noinline))
static uint64_t kern_chase(const uint8_t * base, uint32_t start, size_t steps, size_t stride) {
    LAUNDER(base);

    uint32_t i = start;
    for (size_t s = 0; s < steps; s++) {
        memcpy(&i, base + (size_t) i * stride, sizeof(i));
    }
    __asm__ volatile("" ::: "memory");
    return i;
}

static volatile uint64_t g_sink;

enum pattern_id {
    PAT_SEQ = 0,
    PAT_ANTIDIAG,
    PAT_RAND,
    PAT_SEQ_READ,
    PAT_MEMCPY_SEQ,
    PAT_PAGE_CHASE,
    PAT_LINE_CHASE,
    PAT_COUNT
};

static const char * pat_name(int p) {
    switch (p) {
        case PAT_SEQ:        return "seq";
        case PAT_ANTIDIAG:   return "antidiag";
        case PAT_RAND:       return "rand";
        case PAT_SEQ_READ:   return "seq-read";
        case PAT_MEMCPY_SEQ: return "memcpy-seq";
        case PAT_PAGE_CHASE: return "page-chase";
        case PAT_LINE_CHASE: return "line-chase";
    }
    return "?";
}

static bool pat_is_chase(int p) { return p == PAT_PAGE_CHASE || p == PAT_LINE_CHASE; }
static bool pat_is_copy (int p) { return p == PAT_SEQ || p == PAT_ANTIDIAG || p == PAT_RAND || p == PAT_MEMCPY_SEQ; }

struct job_desc {
    int              pat   = PAT_SEQ;
    uint8_t *        dst   = nullptr;
    const uint8_t *  src   = nullptr;
    const uint32_t * perm  = nullptr;
    size_t           n     = 0;   // chunks, or chase steps
    size_t           g     = 0;   // chunk bytes, or chase stride
};

static uint64_t run_job(const job_desc & j, int t, int nt) {
    if (pat_is_chase(j.pat)) {
        return kern_chase(j.src, 0, j.n, j.g);
    }

    const size_t i0 = (size_t) ((uint64_t) j.n * t       / nt);
    const size_t i1 = (size_t) ((uint64_t) j.n * (t + 1) / nt);

    switch (j.pat) {
        case PAT_SEQ:
        case PAT_ANTIDIAG:
        case PAT_RAND:        return kern_copy  (j.dst, j.src, j.perm, i0, i1, j.g);
        case PAT_MEMCPY_SEQ:  return kern_memcpy(j.dst, j.src, j.perm, i0, i1, j.g);
        case PAT_SEQ_READ:    return kern_read  (j.src, j.perm, i0, i1, j.g);
    }
    return 0;
}

// ---------------------------------------------------------------------------------------
// Worker pool
//
// The probe's own, not ggml's threadpool, so dispatch is under the probe's control. A
// condvar round trip costs tens of microseconds and the reference number is 105 us, so the
// pool spins for the duration of a burst and parks between bursts.
// ---------------------------------------------------------------------------------------

static inline void cpu_relax() {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("yield");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

struct worker_pool {
    std::vector<std::thread> th;
    std::mutex               m;
    std::condition_variable  cv;

    std::atomic<uint32_t> gen{ 0 };
    std::atomic<uint32_t> done{ 0 };
    std::atomic<bool>     spinning{ false };
    std::atomic<bool>     quit{ false };

    job_desc               job;
    int                    nt = 1;
    std::vector<uint64_t>  acc;
    uint64_t               affinity = 0;

    void start(int max_threads, uint64_t aff) {
        affinity = aff;
        acc.resize(max_threads);
        apply_affinity();
        for (int i = 1; i < max_threads; i++) {
            th.emplace_back([this, i] { loop(i); });
        }
    }

    void apply_affinity() const {
        if (!affinity) {
            return;
        }
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int c = 0; c < 64 && c < CPU_SETSIZE; c++) {
            if (affinity & (1ull << c)) {
                CPU_SET(c, &set);
            }
        }
        sched_setaffinity(0, sizeof(set), &set);
    }

    void loop(int id) {
        apply_affinity();
        uint32_t my_gen = 0;
        for (;;) {
            for (;;) {
                if (quit.load(std::memory_order_relaxed)) {
                    return;
                }
                if (gen.load(std::memory_order_acquire) != my_gen) {
                    break;
                }
                if (spinning.load(std::memory_order_relaxed)) {
                    cpu_relax();
                    continue;
                }
                std::unique_lock<std::mutex> lk(m);
                if (!spinning.load(std::memory_order_relaxed) &&
                    gen.load(std::memory_order_acquire) == my_gen &&
                    !quit.load(std::memory_order_relaxed)) {
                    cv.wait(lk);
                }
            }
            my_gen = gen.load(std::memory_order_acquire);
            if (id < nt) {
                acc[id] = run_job(job, id, nt);
            }
            done.fetch_add(1, std::memory_order_release);
        }
    }

    void begin_burst() {
        {
            std::lock_guard<std::mutex> lk(m);
            spinning.store(true, std::memory_order_relaxed);
        }
        cv.notify_all();
    }

    void end_burst() {
        std::lock_guard<std::mutex> lk(m);
        spinning.store(false, std::memory_order_relaxed);
    }

    // One timed pass. Returns the folded accumulator so the caller can sink it.
    uint64_t dispatch(const job_desc & j, int threads) {
        job = j;
        nt  = threads;
        if (threads == 1) {
            return run_job(job, 0, 1);
        }
        const uint32_t n_helpers = (uint32_t) th.size();
        done.store(0, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_release);

        acc[0] = run_job(job, 0, nt);

        while (done.load(std::memory_order_acquire) != n_helpers) {
            cpu_relax();
        }
        uint64_t a = 0;
        for (int i = 0; i < nt; i++) {
            a += acc[i];
        }
        return a;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(m);
            quit.store(true, std::memory_order_relaxed);
            spinning.store(true, std::memory_order_relaxed);
        }
        cv.notify_all();
        for (auto & t : th) {
            t.join();
        }
    }
};

static worker_pool g_pool;

// ---------------------------------------------------------------------------------------
// Arms
// ---------------------------------------------------------------------------------------

enum arm_id {
    ARM_M_DEF = 0,
    ARM_M_HUGE,
    ARM_M_4K,
    ARM_R_RAW,
    ARM_R_UNC,
    ARM_R_VIRGIN,
    ARM_R_DSP,
    ARM_COUNT      // R-evict is not an arm: it is never timed
};

struct arm {
    const char *          name    = "";
    const char *          how     = "";
    uint8_t *             alloc   = nullptr;   // raw allocation base
    size_t                alloc_n = 0;
    uint8_t *             w       = nullptr;   // 2 MiB-aligned working base
    size_t                half    = 0;         // src half size == dst half size
    int                   fd      = -1;
    bool                  avail   = false;
    bool                  is_rpc  = false;
    ggml_backend_buffer_t buf     = nullptr;   // hexagon arms only
};

struct region {
    const char * name;
    size_t       src_off;
    size_t       dst_off;
};

// Every DSP tensor lives in P. Q is the same fd, the same HAP_mmap2 mapping and zero DSP
// data traffic -- P vs Q is "mapping-wide or touched-region?" and it costs nothing.
static region g_reg_P{ "P", 0, 0 };
static region g_reg_Q{ "Q", 0, 0 };

// ---------------------------------------------------------------------------------------
// Reference fill and checksums
//
// The universal guard: the destination checksum after a burst must match the value the FIRST
// arm produced for the same (pattern, g, size, region). It catches elision, a mis-sized
// region, and an arm that silently measured the wrong memory.
// ---------------------------------------------------------------------------------------

static inline uint32_t smix32(uint32_t x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x85ebca6bu;
    x = (x ^ (x >> 13)) * 0xc2b2ae35u;
    return x ^ (x >> 16);
}

static uint64_t fnv(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    size_t   i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t v;
        memcpy(&v, p + i, sizeof(v));
        h = (h ^ v) * 1099511628211ull;
    }
    for (; i < n; i++) {
        h = (h ^ p[i]) * 1099511628211ull;
    }
    return h;
}

// SRC half gets a deterministic word pattern, DST half gets zeros. Both halves are written
// in full so that no first-touch fault can ever land inside a timed region (R9).
static void fill_arm(const arm & a) {
    uint32_t * s = (uint32_t *) a.w;
    const size_t nw = a.half / 4;
    for (size_t i = 0; i < nw; i++) {
        s[i] = smix32((uint32_t) i);
    }
    memset(a.w + a.half, 0, a.half);
}

static std::map<std::string, uint64_t> g_ref_ck;
static bool                            g_ck_failed = false;
static std::string                     g_ck_detail;

// ---------------------------------------------------------------------------------------
// Permutations, cached so the cost of building them never lands in a measurement
// ---------------------------------------------------------------------------------------

static std::map<std::string, std::vector<uint32_t>> g_perms;

static const uint32_t * get_perm(int pat, size_t n) {
    char key[64];
    snprintf(key, sizeof(key), "%d:%zu", pat == PAT_MEMCPY_SEQ ? PAT_SEQ : pat, n);
    auto it = g_perms.find(key);
    if (it != g_perms.end()) {
        return it->second.data();
    }

    std::vector<uint32_t> p(n);
    if (pat == PAT_ANTIDIAG) {
        // The harness permutation, xattn-split.cpp:263-271: reverse within blocks of S=16.
        const size_t S = 16;
        for (size_t i = 0; i < n; i++) {
            p[i] = (uint32_t) ((i / S) * S + (S - 1 - i % S));
        }
    } else if (pat == PAT_RAND) {
        for (size_t i = 0; i < n; i++) {
            p[i] = (uint32_t) i;
        }
        // One fixed seed, so the permutation is byte-identical on every arm.
        uint64_t st = 0x243f6a8885a308d3ull;
        for (size_t i = n; i > 1; i--) {
            st += 0x9e3779b97f4a7c15ull;
            uint64_t z = st;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
            z ^= z >> 31;
            std::swap(p[i - 1], p[z % i]);
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            p[i] = (uint32_t) i;
        }
    }
    auto & v = g_perms[key];
    v = std::move(p);
    return v.data();
}

// The chase chain is written INTO the src half, which destroys the reference fill. Every
// caller of this must re-fill the arm afterwards.
static void build_chain(const arm & a, const region & rg, int pat, size_t bytes, size_t & steps) {
    uint8_t * base = a.w + rg.src_off;
    if (pat == PAT_PAGE_CHASE) {
        const size_t np = bytes / 4096;
        const uint32_t * perm = get_perm(PAT_RAND, np);
        // perm is a permutation, so next[perm[i]] = perm[i+1] is a single cycle over pages.
        for (size_t i = 0; i < np; i++) {
            const uint32_t cur = perm[i];
            const uint32_t nxt = perm[(i + 1) % np];
            memcpy(base + (size_t) cur * 4096, &nxt, sizeof(nxt));
        }
        steps = np;
    } else {
        // Same access count, but every access is inside one 4 KiB page: 64 lines, cycled.
        const size_t nl = 4096 / 64;
        const uint32_t * perm = get_perm(PAT_RAND, nl);
        for (size_t i = 0; i < nl; i++) {
            const uint32_t cur = perm[i];
            const uint32_t nxt = perm[(i + 1) % nl];
            memcpy(base + (size_t) cur * 64, &nxt, sizeof(nxt));
        }
        steps = bytes / 4096;
    }
}

// ---------------------------------------------------------------------------------------
// Bursts
// ---------------------------------------------------------------------------------------

struct burst {
    bool                valid   = false;
    int                 passes  = 0;
    double              first   = 0;
    double              p50     = 0;
    double              mn      = 0;
    double              mx      = 0;
    double              min20   = 0;
    double              bytes   = 0;   // bytes moved per pass
    uint64_t            ck      = 0;
    bool                low_n   = false;
    bool                unstable = false;   // max/min > 2 -> big.LITTLE migration (R8)
    std::vector<double> series;
};

struct burst_opts {
    int max_passes = 40;
    int min_passes = 6;
    int budget_ms  = 150;   // 0 = no budget, always run max_passes
};

static double p50_of(std::vector<double> v) {
    if (v.empty()) {
        return 0;
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static burst run_burst(const arm & a, const region & rg, int pat, size_t g, size_t bytes,
                       int threads, const burst_opts & bo) {
    burst b;
    if (!a.avail) {
        return b;
    }

    job_desc j;
    j.pat = pat;

    size_t chain_steps = 0;
    if (pat_is_chase(pat)) {
        build_chain(a, rg, pat, bytes, chain_steps);
        j.src = a.w + rg.src_off;
        j.n   = chain_steps;
        j.g   = (pat == PAT_PAGE_CHASE) ? 4096 : 64;
        threads = 1;                       // a dependent chain has nothing to parallelise
        b.bytes = (double) chain_steps * 4;
    } else {
        const size_t n = bytes / g;
        if (n < 16) {
            return b;                      // the S=16 antidiagonal is undefined below this
        }
        j.perm = get_perm(pat, n);
        j.src  = a.w + rg.src_off;
        j.dst  = a.w + rg.dst_off;
        j.n    = n;
        j.g    = g;
        b.bytes = (pat == PAT_SEQ_READ) ? (double) bytes : (double) bytes * 2;
    }

    g_pool.begin_burst();

    // acc is the value of ONE pass, not a fold across passes: every pass computes the same
    // number, and the time budget stops different arms after different pass counts, so any
    // fold would make the checksum depend on the pass count instead of on the memory.
    uint64_t acc = 0, sink = 0;
    const int64_t t_start = ggml_time_us();
    for (int i = 0; i < bo.max_passes; i++) {
        const int64_t t0 = ggml_time_us();
        const uint64_t pass_acc = g_pool.dispatch(j, threads);
        b.series.push_back((double) (ggml_time_us() - t0));
        acc   = pass_acc;
        sink ^= pass_acc;

        if (bo.budget_ms > 0 && (int) b.series.size() >= bo.min_passes &&
            ggml_time_us() - t_start > (int64_t) bo.budget_ms * 1000) {
            break;
        }
    }

    g_pool.end_burst();
    g_sink = sink;

    b.passes = (int) b.series.size();
    b.first  = b.series[0];

    std::vector<double> tail(b.series.begin() + std::min<size_t>(3, b.series.size() - 1), b.series.end());
    b.p50 = p50_of(tail);
    b.mn  = *std::min_element(tail.begin(), tail.end());
    b.mx  = *std::max_element(tail.begin(), tail.end());

    // min20 exists only so the new figures can be read against the 105/242/793 dataset,
    // which was min-of-20. It is never used for a verdict.
    b.min20 = b.series[0];
    for (int i = 1; i < std::min(20, b.passes); i++) {
        b.min20 = std::min(b.min20, b.series[i]);
    }

    b.low_n    = b.passes < 10;
    b.unstable = b.mn > 0 && (b.mx / b.mn) > 2.0;

    b.ck = pat_is_copy(pat) ? (fnv(a.w + rg.dst_off, bytes) ^ acc) : acc;

    char key[128];
    snprintf(key, sizeof(key), "%d:%zu:%zu:%s", pat, g, bytes, rg.name);
    auto it = g_ref_ck.find(key);
    if (it == g_ref_ck.end()) {
        g_ref_ck[key] = b.ck;
    } else if (it->second != b.ck && !pat_is_chase(pat)) {
        // A chase mismatch is expected: the chain is rebuilt per arm from the same
        // permutation, so it does agree, but a copy mismatch means an arm measured the
        // wrong memory and every number in the run is suspect.
        g_ck_failed = true;
        char d[256];
        snprintf(d, sizeof(d), "%s %s %s g=%zu n=%zu: got %016" PRIx64 " want %016" PRIx64,
                 a.name, rg.name, pat_name(pat), g, bytes, b.ck, it->second);
        g_ck_detail = d;
    }

    b.valid = true;
    return b;
}

// ---------------------------------------------------------------------------------------
// Result store
// ---------------------------------------------------------------------------------------

struct cell {
    std::string arm, region, pat, state;
    size_t      g = 0, bytes = 0;
    int         threads = 0, round = -1;
    burst       b;
};

static std::vector<cell>               g_cells;
static std::map<std::string, size_t>   g_index;

static std::string cell_key(const char * arm, const char * rg, int pat, size_t g, size_t bytes,
                            int threads, const char * state) {
    char k[192];
    snprintf(k, sizeof(k), "%s|%s|%s|%zu|%zu|%d|%s", arm, rg, pat_name(pat), g, bytes, threads, state);
    return k;
}

static void store(const arm & a, const region & rg, int pat, size_t g, size_t bytes, int threads,
                  const char * state, int round, const burst & b) {
    if (!b.valid) {
        return;
    }
    cell c;
    c.arm = a.name; c.region = rg.name; c.pat = pat_name(pat); c.state = state;
    c.g = g; c.bytes = bytes; c.threads = threads; c.round = round; c.b = b;
    g_index[cell_key(a.name, rg.name, pat, g, bytes, threads, state)] = g_cells.size();
    g_cells.push_back(std::move(c));
}

static const burst * lookup(const char * arm, const char * rg, int pat, size_t g, size_t bytes,
                            int threads, const char * state) {
    auto it = g_index.find(cell_key(arm, rg, pat, g, bytes, threads, state));
    return it == g_index.end() ? nullptr : &g_cells[it->second].b;
}

// measure + store in one step
static burst measure(const arm & a, const region & rg, int pat, size_t g, size_t bytes,
                     int threads, const char * state, int round, const burst_opts & bo) {
    burst b = run_burst(a, rg, pat, g, bytes, threads, bo);
    store(a, rg, pat, g, bytes, threads, state, round, b);
    return b;
}

// ---------------------------------------------------------------------------------------
// /proc/self/smaps
//
// This is the only non-timing instrument available without root, and it settles H1's
// premise outright: if M-def shows AnonHugePages: 0 kB, H1 is dead before any timing.
// ---------------------------------------------------------------------------------------

struct smaps_row {
    bool        found = false;
    std::string backing;
    std::string vmflags;
    std::map<std::string, long> kv;
};

static smaps_row smaps_for(uintptr_t addr) {
    smaps_row r;
    FILE * f = fopen("/proc/self/smaps", "r");
    if (!f) {
        return r;
    }
    char line[1024];
    bool in = false;
    while (fgets(line, sizeof(line), f)) {
        unsigned long long lo = 0, hi = 0;
        char perms[8] = { 0 }, rest[768] = { 0 };
        if (sscanf(line, "%llx-%llx %7s %*s %*s %*s %767[^\n]", &lo, &hi, perms, rest) >= 3) {
            if (in) {
                break;                       // finished the VMA we wanted
            }
            in = (addr >= lo && addr < hi);
            if (in) {
                const char * p = rest;
                while (*p == ' ') {
                    p++;
                }
                r.found   = true;
                r.backing = *p ? p : "anonymous";
            }
            continue;
        }
        if (!in) {
            continue;
        }
        char key[64];
        long val = 0;
        if (sscanf(line, "%63[^:]: %ld kB", key, &val) == 2) {
            r.kv[key] = val;
        } else if (strncmp(line, "VmFlags:", 8) == 0) {
            r.vmflags = line + 9;
            while (!r.vmflags.empty() && (r.vmflags.back() == '\n' || r.vmflags.back() == ' ')) {
                r.vmflags.pop_back();
            }
        } else if (strncmp(line, "THPeligible:", 12) == 0) {
            r.kv["THPeligible"] = atol(line + 12);
        }
    }
    fclose(f);
    return r;
}

static std::map<std::string, smaps_row> g_prev_smaps;

static void dump_smaps(const arm * arms, const char * when) {
    printf("\nSMAPS @ %s   (* = changed since the previous dump)\n", when);
    printf("  %-9s %-14s %10s %10s %6s %6s %10s %4s %10s %8s\n",
           "arm", "backing", "Size", "Rss", "KPS", "MPS", "AnonHuge", "THPe", "Shared_Dy", "Locked");
    for (int i = 0; i < ARM_COUNT; i++) {
        const arm & a = arms[i];
        if (!a.avail) {
            printf("  %-9s (unavailable)\n", a.name);
            continue;
        }
        smaps_row r = smaps_for((uintptr_t) a.w);
        if (!r.found) {
            printf("  %-9s (no smaps entry)\n", a.name);
            continue;
        }
        auto get = [&](const char * k) { auto it = r.kv.find(k); return it == r.kv.end() ? -1L : it->second; };
        auto chg = [&](const char * k) -> const char * {
            auto p = g_prev_smaps.find(a.name);
            if (p == g_prev_smaps.end()) {
                return " ";
            }
            auto o = p->second.kv.find(k);
            auto n = r.kv.find(k);
            const long ov = o == p->second.kv.end() ? -1 : o->second;
            const long nv = n == r.kv.end() ? -1 : n->second;
            return ov == nv ? " " : "*";
        };
        std::string back = r.backing.substr(0, 14);
        printf("  %-9s %-14s %9ld%s %9ld%s %5ld%s %5ld%s %9ld%s %3ld%s %9ld%s %7ld%s\n",
               a.name, back.c_str(),
               get("Size"), chg("Size"), get("Rss"), chg("Rss"),
               get("KernelPageSize"), chg("KernelPageSize"), get("MMUPageSize"), chg("MMUPageSize"),
               get("AnonHugePages"), chg("AnonHugePages"), get("THPeligible"), chg("THPeligible"),
               get("Shared_Dirty"), chg("Shared_Dirty"), get("Locked"), chg("Locked"));
        if (!r.vmflags.empty()) {
            printf("  %-9s   VmFlags: %s\n", "", r.vmflags.c_str());
        }
        g_prev_smaps[a.name] = r;
    }
}

// ---------------------------------------------------------------------------------------
// DSP events
//
// The smallest possible real graph_compute: a 1-element F32 ADD with src0, src1 and dst all
// inside R-dsp. add_buffer dedupes by fd (ggml-hexagon.cpp:1250-1268), so three tensors in
// one buffer yield exactly ONE buffer descriptor, and the DSP HAP_mmap2's the WHOLE buffer
// (htp/main.c:844) regardless. One element touched, 130 MiB mapped -- that asymmetry is what
// makes the size-independence test work.
// ---------------------------------------------------------------------------------------

struct dsp_event {
    const char *  label = "";
    ggml_cgraph * graph = nullptr;
    float *       sentinel_at = nullptr;   // read AFTER the next round, see below
    float         expect = 0;
    bool          expect_write = true;
    double        call_us = 0;
    bool          checked = false;
    bool          ok = true;
};

struct dsp_setup {
    ggml_context * ctx = nullptr;
    dsp_event      tiny;      // 1 element in R-dsp/P
    dsp_event      full;      // all 16 MiB of R-dsp/P
    dsp_event      evict;     // 1 element in R-evict, forces drop_mmap of R-dsp
    ggml_backend_buffer_t evict_buf = nullptr;
};

static ggml_tensor * place(ggml_context * ctx, ggml_backend_buffer_t buf, uint8_t * base,
                           size_t off, int64_t ne, const char * name) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
    t->buffer = buf;
    t->data   = base + off;
    ggml_set_name(t, name);
    ggml_backend_buffer_init_tensor(buf, t);
    return t;
}

static ggml_cgraph * make_add(ggml_context * ctx, ggml_backend_buffer_t buf, uint8_t * base,
                              size_t a_off, size_t b_off, size_t d_off, int64_t ne,
                              const char * tag, ggml_tensor ** out_dst) {
    char n[64];
    snprintf(n, sizeof(n), "%s_a", tag);
    ggml_tensor * a = place(ctx, buf, base, a_off, ne, n);
    snprintf(n, sizeof(n), "%s_b", tag);
    ggml_tensor * b = place(ctx, buf, base, b_off, ne, n);

    ggml_tensor * d = ggml_add(ctx, a, b);
    d->buffer = buf;
    d->data   = base + d_off;
    snprintf(n, sizeof(n), "%s_d", tag);
    ggml_set_name(d, n);
    ggml_backend_buffer_init_tensor(buf, d);

    ggml_cgraph * g = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(g, d);

    *out_dst = d;
    return g;
}

// The sentinel sits in the LAST page of region P's 16 MiB dst extent. It has to survive the
// round that follows the event, because checking it immediately after graph_compute would
// either (a) read back this thread's own dirty line and prove nothing, or (b) need a
// cache-evicting sweep that would destroy the "first pass after the event" signal the whole
// persistence protocol depends on. Every timed burst writes the FIRST 4 MiB of its region's
// dst half, and region Q's dst half starts at half + half/2 -- so the only offset inside P
// that no burst can ever reach, at any arena size, is P's own tail.
static const size_t SENTINEL_OFF = 16 * MiB - 4096;

static void arm_sentinel(dsp_event & e, float * at, float expect, bool expect_write) {
    e.sentinel_at  = at;
    e.expect       = expect;
    e.expect_write = expect_write;
    e.checked      = false;
    *at            = -12345.0f;
}

static bool check_sentinel(dsp_event & e) {
    if (!e.sentinel_at || e.checked) {
        return true;
    }
    e.checked = true;
    const float got = *e.sentinel_at;
    if (e.expect_write) {
        e.ok = std::fabs(got - e.expect) <= 1e-3f * (1.0f + std::fabs(e.expect));
    } else {
        // At OPSTAGE=1 the kernels bail on HTP_OPFLAGS_SKIP_COMPUTE (htp/binary-ops.c:807)
        // and at OPSTAGE=0 nothing is enqueued at all (ggml-hexagon.cpp:4052), so the
        // assertion inverts: the sentinel MUST still be there.
        e.ok = got == -12345.0f;
    }
    return e.ok;
}

static void fire(ggml_backend_t dev, dsp_event & e) {
    const int64_t t0 = ggml_time_us();
    ggml_backend_graph_compute(dev, e.graph);
    e.call_us = (double) (ggml_time_us() - t0);
}

// ---------------------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------------------

struct options {
    std::string dev        = "HTP0";
    std::string run        = "all";
    std::string csv;
    size_t      arena_mib  = 130;    // 2 x 64 MiB working halves + 2 MiB alignment slack
    size_t      evict_mib  = 64;
    int         threads    = 8;
    uint64_t    affinity   = 0;
    int         burst      = 40;
    int         budget_ms  = 150;
    double      aa_tol     = 0.05;
    bool        swap_roles = false;  // R6: exchange the R-dsp and R-virgin allocations
    bool        no_unc     = false;
    int         opstage    = -1;     // p5child only
    std::string perf_arm   = "R-dsp";
    std::string perf_state = "touched";
    int         perf_secs  = 5;
};

static void usage(const char * prog) {
    printf("usage: %s [options]\n", prog);
    printf("  --dev NAME        hexagon backend to drive (default HTP0)\n");
    printf("  --run WHAT        all | gonogo | links | persist | perf   (default all)\n");
    printf("  --csv PATH        machine-readable output\n");
    printf("  --arena-mib N     per-arm arena size (default 130 = 2x64 MiB halves + slack)\n");
    printf("  --evict-mib N     R-evict size (default 64)\n");
    printf("  --threads N       high thread count (default 8)\n");
    printf("  --affinity HEX    sched_setaffinity mask for every probe thread\n");
    printf("  --burst N         timed passes per burst (default 40)\n");
    printf("  --budget-ms N     per-burst time budget, 0 = always run --burst passes (default 150)\n");
    printf("  --aa-tol F        A/A gate tolerance (default 0.05)\n");
    printf("  --swap-roles      give the DSP role to the other allocation (R6)\n");
    printf("  --no-unc          skip the uncached rpcmem arm\n");
    printf("  --perf-arm NAME   --run perf: which arm (default R-dsp)\n");
    printf("  --perf-state S    --run perf: virgin | touched (default touched)\n");
    printf("  --perf-secs N     --run perf: seconds to loop (default 5)\n");
}

static bool parse_args(int argc, char ** argv, options & o) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s needs a value\n", what);
                exit(1);
            }
            return argv[++i];
        };
        if      (a == "--dev")        { o.dev        = next("--dev"); }
        else if (a == "--run")        { o.run        = next("--run"); }
        else if (a == "--csv")        { o.csv        = next("--csv"); }
        else if (a == "--arena-mib")  { o.arena_mib  = (size_t) atoll(next("--arena-mib")); }
        else if (a == "--evict-mib")  { o.evict_mib  = (size_t) atoll(next("--evict-mib")); }
        else if (a == "--threads")    { o.threads    = atoi(next("--threads")); }
        else if (a == "--affinity")   { o.affinity   = strtoull(next("--affinity"), nullptr, 0); }
        else if (a == "--burst")      { o.burst      = atoi(next("--burst")); }
        else if (a == "--budget-ms")  { o.budget_ms  = atoi(next("--budget-ms")); }
        else if (a == "--aa-tol")     { o.aa_tol     = atof(next("--aa-tol")); }
        else if (a == "--swap-roles") { o.swap_roles = true; }
        else if (a == "--no-unc")     { o.no_unc     = true; }
        else if (a == "--opstage")    { o.opstage    = atoi(next("--opstage")); }
        else if (a == "--perf-arm")   { o.perf_arm   = next("--perf-arm"); }
        else if (a == "--perf-state") { o.perf_state = next("--perf-state"); }
        else if (a == "--perf-secs")  { o.perf_secs  = atoi(next("--perf-secs")); }
        else if (a == "-h" || a == "--help") { usage(argv[0]); exit(0); }
        else {
            fprintf(stderr, "error: unknown argument %s\n", a.c_str());
            usage(argv[0]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------

static void hr() {
    printf("--------------------------------------------------------------------------------------\n");
}

static std::string read_first_line(const char * path) {
    FILE * f = fopen(path, "r");
    if (!f) {
        return "unreadable";
    }
    char b[512] = { 0 };
    if (!fgets(b, sizeof(b), f)) {
        b[0] = 0;
    }
    fclose(f);
    std::string s = b;
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) {
        s.pop_back();
    }
    return s.empty() ? "unreadable" : s;
}

static std::string sysprop(const char * key) {
#ifdef __ANDROID__
    char v[PROP_VALUE_MAX] = { 0 };
    if (__system_property_get(key, v) > 0) {
        return v;
    }
#else
    (void) key;
#endif
    return "?";
}

// A busy spin, not a sleep: a 50 ms nanosleep lets the CPU drop its DVFS corner, and the
// whole point of the round spacing is that only the buffer state changes between rounds.
static void spin_until(int64_t t_us) {
    volatile uint64_t x = 0;
    while (ggml_time_us() < t_us) {
        for (int i = 0; i < 256; i++) {
            x += (uint64_t) i;
        }
    }
    (void) x;
}

static double ratio(const burst * a, const burst * b) {
    if (!a || !b || !a->valid || !b->valid || b->p50 <= 0) {
        return 0;
    }
    return a->p50 / b->p50;
}

static void print_ratio(double r) {
    if (r <= 0) {
        printf("     --  ");
    } else {
        printf(" %7.2fx", r);
    }
}

// ---------------------------------------------------------------------------------------
// Phase implementations
// ---------------------------------------------------------------------------------------

static const size_t DEPLOY_BYTES = 4 * MiB;   // 128 * 512 * 16 * 4 == 4194304, exactly 4 MiB

struct probe_ctx {
    options   o;
    arm       arms[ARM_COUNT];
    burst_opts fast;     // budgeted
    burst_opts fixed;    // exactly --burst passes, used by the persistence rounds
    size_t    half = 0;
};

// P1: the GO/NO-GO. One cell decides whether streaming CPU work over rpcmem has a path.
static void phase_p1(probe_ctx & c, const char * state) {
    printf("\nP1  GO/NO-GO -- N=4 MiB, tail p50 us, state %s. 1T is the primary number (R8):\n"
           "    without --affinity one thread landing on a little core dominates every 8T pass.\n", state);
    printf("  %-9s %10s %9s %10s %9s | %10s %9s %10s %9s\n", "arm",
           "seq64K 1T", "vs M-def", "rand64 1T", "vs M-def",
           "seq64K NT", "vs M-def", "rand64 NT", "vs M-def");
    for (int i = 0; i < ARM_COUNT; i++) {
        const arm & a = c.arms[i];
        if (!a.avail) {
            printf("  %-9s (unavailable)\n", a.name);
            continue;
        }
        burst s1 = measure(a, g_reg_P, PAT_SEQ,  65536, DEPLOY_BYTES, 1, state, -1, c.fast);
        burst r1 = measure(a, g_reg_P, PAT_RAND, 64,    DEPLOY_BYTES, 1, state, -1, c.fast);
        burst sn = measure(a, g_reg_P, PAT_SEQ,  65536, DEPLOY_BYTES, c.o.threads, state, -1, c.fast);
        burst rn = measure(a, g_reg_P, PAT_RAND, 64,    DEPLOY_BYTES, c.o.threads, state, -1, c.fast);

        printf("  %-9s %10.1f ", a.name, s1.p50);
        print_ratio(ratio(&s1, lookup("M-def", "P", PAT_SEQ, 65536, DEPLOY_BYTES, 1, state)));
        printf(" %10.1f ", r1.p50);
        print_ratio(ratio(&r1, lookup("M-def", "P", PAT_RAND, 64, DEPLOY_BYTES, 1, state)));
        printf(" | %10.1f ", sn.p50);
        print_ratio(ratio(&sn, lookup("M-def", "P", PAT_SEQ, 65536, DEPLOY_BYTES, c.o.threads, state)));
        printf(" %10.1f ", rn.p50);
        print_ratio(ratio(&rn, lookup("M-def", "P", PAT_RAND, 64, DEPLOY_BYTES, c.o.threads, state)));
        if (s1.unstable || r1.unstable || sn.unstable || rn.unstable) {
            printf("  UNSTABLE(max/min>2)");
        }
        printf("\n");
    }
}

// P2: the link decomposition. No DSP is involved at all, so this is the one phase that can
// be run on its own and still answer "where does the 2.3x live".
static void phase_p2(probe_ctx & c) {
    const std::vector<size_t> sizes = { 256 * 1024, 512 * 1024, 1 * MiB, 2 * MiB, 4 * MiB,
                                        8 * MiB, 16 * MiB, 32 * MiB, 64 * MiB };
    const int pats[2] = { PAT_ANTIDIAG, PAT_RAND };
    const size_t gs[2] = { 512, 64 };

    for (int pi = 0; pi < 2; pi++) {
        for (int t : { 1, c.o.threads }) {
            printf("\nP2  %s g=%zu, %d thread%s, tail p50 us / ratio vs M-def\n",
                   pat_name(pats[pi]), gs[pi], t, t == 1 ? "" : "s");
            printf("  %-10s", "size");
            for (int i = 0; i < ARM_COUNT; i++) {
                printf(" %-17s", c.arms[i].name);
            }
            printf("\n");

            for (size_t n : sizes) {
                if (n > c.half) {
                    continue;
                }
                printf("  %-10s", n >= MiB ? (std::to_string(n / MiB) + "M").c_str()
                                           : (std::to_string(n / 1024) + "K").c_str());
                const burst * ref = nullptr;
                for (int i = 0; i < ARM_COUNT; i++) {
                    const arm & a = c.arms[i];
                    // The uncached control is a calibration point, not a sweep: at 64 MiB a
                    // random 64-byte gather over uncached memory runs for seconds a pass.
                    const bool skip = !a.avail || (i == ARM_R_UNC && n > DEPLOY_BYTES);
                    if (skip) {
                        printf(" %-17s", "        --       ");
                        continue;
                    }
                    burst b = measure(a, g_reg_P, pats[pi], gs[pi], n, t, "S0", -1, c.fast);
                    if (i == ARM_M_DEF) {
                        ref = lookup("M-def", "P", pats[pi], gs[pi], n, t, "S0");
                    }
                    const double r = ratio(&b, ref);
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%8.1f %6.2fx%s", b.p50, r, b.unstable ? "!" : " ");
                    printf(" %-17s", buf);
                }
                printf("\n");
            }
        }
    }
}

// P3: per-page cost vs per-byte cost. H1 punishes page-chase and leaves line-chase alone.
static void phase_p3(probe_ctx & c, const char * state) {
    printf("\nP3  dependent chains, 1 thread, state %s -- same access COUNT, different stride.\n"
           "    The tiebreaker is each arm's PENALTY vs M-def, not the page/line ratio: line-chase\n"
           "    is L1-resident by construction, so its absolute time says nothing on its own.\n", state);
    printf("  %-9s %19s %19s %19s %19s\n",
           "arm", "page-chase 4M", "line-chase 4M", "page-chase 64M", "line-chase 64M");
    double ref[4] = { 0, 0, 0, 0 };
    for (int i = 0; i < ARM_COUNT; i++) {
        const arm & a = c.arms[i];
        if (!a.avail) {
            printf("  %-9s (unavailable)\n", a.name);
            continue;
        }
        double v[4] = { 0, 0, 0, 0 };
        int k = 0;
        for (size_t n : { DEPLOY_BYTES, (size_t) (64 * MiB) }) {
            for (int p : { PAT_PAGE_CHASE, PAT_LINE_CHASE }) {
                if (n > c.half || (i == ARM_R_UNC && n > DEPLOY_BYTES)) {
                    v[k++] = 0;
                    continue;
                }
                burst b = measure(a, g_reg_P, p, 0, n, 1, state, -1, c.fast);
                v[k++] = b.p50;
            }
        }
        if (i == ARM_M_DEF) {
            memcpy(ref, v, sizeof(ref));
        }
        printf("  %-9s", a.name);
        for (int j = 0; j < 4; j++) {
            char buf[32];
            if (v[j] <= 0) {
                snprintf(buf, sizeof(buf), "%18s", "--");
            } else {
                snprintf(buf, sizeof(buf), "%10.1f %6.2fx", v[j], ref[j] > 0 ? v[j] / ref[j] : 0.0);
            }
            printf(" %19s", buf);
        }
        printf("\n");
    }
    // The chains overwrote the src halves. Put the reference pattern back before anything
    // else is measured.
    for (int i = 0; i < ARM_COUNT; i++) {
        if (c.arms[i].avail) {
            fill_arm(c.arms[i]);
        }
    }
}

// P6: the deliverable. Where ratio(g) crosses the survival line is the answer to "which
// access patterns survive".
static void phase_p6(probe_ctx & c) {
    const std::vector<size_t> sizes = { 256 * 1024, 1 * MiB, 4 * MiB, 16 * MiB, 64 * MiB };
    const std::vector<size_t> gs    = { 16, 64, 512, 4096, 65536 };
    const int pats[4] = { PAT_SEQ, PAT_ANTIDIAG, PAT_RAND, PAT_MEMCPY_SEQ };

    for (int t : { 1, c.o.threads }) {
        printf("\nP6  R-dsp@S7 / M-def, %d thread%s  --  SURVIVES iff <= %.2fx\n",
               t, t == 1 ? "" : "s", SURVIVAL_LINE);
        printf("  %-10s %-8s", "size", "g");
        for (int p : pats) {
            printf(" %-16s", pat_name(p));
        }
        printf("\n");
        for (size_t n : sizes) {
            if (n > c.half) {
                continue;
            }
            for (size_t g : gs) {
                if (n / g < 16) {
                    continue;
                }
                printf("  %-10s %-8zu", n >= MiB ? (std::to_string(n / MiB) + "M").c_str()
                                                 : (std::to_string(n / 1024) + "K").c_str(), g);
                for (int p : pats) {
                    burst bm = measure(c.arms[ARM_M_DEF], g_reg_P, p, g, n, t, "S7", -1, c.fast);
                    burst bd = measure(c.arms[ARM_R_DSP], g_reg_P, p, g, n, t, "S7", -1, c.fast);
                    const double r = ratio(&bd, &bm);
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%6.2fx %-8s", r,
                             r <= 0 ? "" : (r <= SURVIVAL_LINE ? "SURVIVES" : "LOSES"));
                    printf(" %-16s", buf);
                }
                printf("\n");
            }
        }
    }

    // Different code path, same work: this is the shape mllm actually runs, and it is the
    // reason mllm never observed any of this.
    printf("\nP6b seq-read family (moves N bytes, NOT comparable to the copy rows above), %d threads\n",
          c.o.threads);
    printf("  %-10s %-8s %12s %12s %10s\n", "size", "g", "M-def", "R-dsp@S7", "ratio");
    for (size_t n : { (size_t) (4 * MiB), (size_t) (64 * MiB) }) {
        if (n > c.half) {
            continue;
        }
        for (size_t g : { (size_t) 512, (size_t) 65536 }) {
            burst bm = measure(c.arms[ARM_M_DEF], g_reg_P, PAT_SEQ_READ, g, n, c.o.threads, "S7", -1, c.fast);
            burst bd = measure(c.arms[ARM_R_DSP], g_reg_P, PAT_SEQ_READ, g, n, c.o.threads, "S7", -1, c.fast);
            printf("  %-10s %-8zu %12.1f %12.1f %9.2fx\n",
                   (std::to_string(n / MiB) + "M").c_str(), g, bm.p50, bd.p50, ratio(&bd, &bm));
        }
    }
}

// ---------------------------------------------------------------------------------------
// P4 -- the persistence protocol
// ---------------------------------------------------------------------------------------

struct round_cell {
    const arm *  a  = nullptr;
    const region * rg = nullptr;
    const char * label = "";
};

struct round_rec {
    int         idx = 0;
    double      t_ms = 0;
    std::string state;
    burst       b1[5];    // 1 thread
    burst       bT[5];    // --threads
};

static const int N_ROUND_CELLS = 5;

static void run_round(probe_ctx & c, round_rec & r, const round_cell * cells, int64_t t0_us) {
    r.t_ms = (double) (ggml_time_us() - t0_us) / 1000.0;
    for (int i = 0; i < N_ROUND_CELLS; i++) {
        char st[32];
        snprintf(st, sizeof(st), "%s.r%d", r.state.c_str(), r.idx);
        r.b1[i] = measure(*cells[i].a, *cells[i].rg, PAT_ANTIDIAG, 512, DEPLOY_BYTES, 1,
                          st, r.idx, c.fixed);
        r.bT[i] = measure(*cells[i].a, *cells[i].rg, PAT_ANTIDIAG, 512, DEPLOY_BYTES, c.o.threads,
                          st, r.idx, c.fixed);
    }
}

struct p4_result {
    std::vector<round_rec> rounds;
    bool   aa_ok = true;
    std::string aa_detail;
    bool   sentinel_ok = true;
    std::string sentinel_detail;
    double e1_us = 0, e7_us = 0, v9_us = 0, e13_us = 0;
    std::vector<double> evict_us, retouch_us;
    // The index into rounds[] of the FIRST round after each evict / re-touch. Reading the
    // cycle off the state label instead would mix cycle 1's three rounds with cycle 2's two.
    std::vector<int> evict_round, touch_round;
    bool evict_available = true;
};

static void phase_p4(probe_ctx & c, ggml_backend_t dev, dsp_setup & ds, p4_result & res) {
    arm * A = c.arms;
    const round_cell cells[N_ROUND_CELLS] = {
        { &c.arms[ARM_M_DEF],    &g_reg_P, "M-def"    },
        { &c.arms[ARM_M_HUGE],   &g_reg_P, "M-huge"   },
        { &c.arms[ARM_R_VIRGIN], &g_reg_P, "R-virgin" },
        { &c.arms[ARM_R_DSP],    &g_reg_P, "R-dsp/P"  },
        { &c.arms[ARM_R_DSP],    &g_reg_Q, "R-dsp/Q"  },
    };

    const int64_t t0 = ggml_time_us();
    auto do_round = [&](int idx, const char * state, int64_t at_us) {
        if (at_us > 0) {
            spin_until(t0 + at_us);
        }
        round_rec r;
        r.idx   = idx;
        r.state = state;
        run_round(c, r, cells, t0);
        res.rounds.push_back(r);
    };

    // rounds 0 and 1 -- the A/A gate. Nothing happens between them.
    do_round(0, "S0", 0);
    do_round(1, "S0", 0);

    const int64_t t_e1 = ggml_time_us();
    fire(dev, ds.tiny);
    res.e1_us = ds.tiny.call_us;
    do_round(2, "S1", 0);
    check_sentinel(ds.tiny);
    dump_smaps(A, "S1 (after the 1-element DSP op)");
    do_round(3, "S1", t_e1 - t0 +      50 * 1000);
    do_round(4, "S1", t_e1 - t0 +     250 * 1000);
    do_round(5, "S1", t_e1 - t0 +    1000 * 1000);
    do_round(6, "S1", t_e1 - t0 +    5000 * 1000);

    const int64_t t_e7 = ggml_time_us();
    fire(dev, ds.full);
    res.e7_us = ds.full.call_us;
    do_round(7, "S7", 0);
    check_sentinel(ds.full);
    dump_smaps(A, "S7 (after the whole-region DSP op)");
    do_round(8, "S7", t_e7 - t0 +   50 * 1000);
    do_round(9, "S7", t_e7 - t0 + 1000 * 1000);

    // Three eviction cycles, not one. A single slow->evict->fast transition is one
    // coincidence away from being an artifact of the evicting graph_compute itself.
    // R3: drop_mmap fires only when m_vmem + e_vmem > ctx->max_vmem AND the buffer is absent
    // from m_reuse (htp/main.c:890-896). Without R-evict there is no lever at all, and a
    // silent no-op here would read as "irreversible", which is a conclusion, not a gap.
    res.evict_available = ds.evict.graph != nullptr;
    int idx = 10;
    for (int cyc = 0; cyc < 3 && res.evict_available; cyc++) {
        fire(dev, ds.evict);
        res.evict_us.push_back(ds.evict.call_us);
        if (cyc == 0) {
            res.v9_us = ds.evict.call_us;
        }
        const int nr = (cyc == 0) ? 3 : 2;
        res.evict_round.push_back((int) res.rounds.size());
        for (int i = 0; i < nr; i++) {
            do_round(idx++, "S9", 0);
        }
        if (cyc == 0) {
            dump_smaps(A, "S9 (after forced drop_mmap)");
        }

        fire(dev, ds.tiny);
        res.retouch_us.push_back(ds.tiny.call_us);
        if (cyc == 0) {
            res.e13_us = ds.tiny.call_us;
        }
        res.touch_round.push_back((int) res.rounds.size());
        for (int i = 0; i < nr; i++) {
            do_round(idx++, "S13", 0);
        }
    }

    // A/A gate, per arm and per thread count. Hard: it blocks every verdict line.
    for (int i = 0; i < N_ROUND_CELLS; i++) {
        for (int tt = 0; tt < 2; tt++) {
            const burst & b0 = tt ? res.rounds[0].bT[i] : res.rounds[0].b1[i];
            const burst & b1 = tt ? res.rounds[1].bT[i] : res.rounds[1].b1[i];
            if (!b0.valid || !b1.valid || b0.p50 <= 0) {
                continue;
            }
            const double d = std::fabs(b1.p50 - b0.p50) / b0.p50;
            if (d >= c.o.aa_tol) {
                res.aa_ok = false;
                char buf[256];
                snprintf(buf, sizeof(buf), "%s @%dT (tail0=%.1f tail1=%.1f delta=%.1f%%)",
                         cells[i].label, tt ? c.o.threads : 1, b0.p50, b1.p50, d * 100.0);
                if (!res.aa_detail.empty()) {
                    res.aa_detail += "; ";
                }
                res.aa_detail += buf;
            }
        }
    }

    res.sentinel_ok = ds.tiny.ok && ds.full.ok;
    if (!res.sentinel_ok) {
        res.sentinel_detail = std::string(ds.tiny.ok ? "" : "tiny-ADD ") +
                              (ds.full.ok ? "" : "whole-region-ADD ");
    }
}

static void print_p4(const probe_ctx & c, const p4_result & res) {
    const char * labels[N_ROUND_CELLS] = { "M-def", "M-huge", "R-virgin", "R-dsp/P", "R-dsp/Q" };

    for (int tt = 0; tt < 2; tt++) {
        printf("\nP4  persistence, antidiag g=512 N=4 MiB, %d thread%s -- cells are first/tail (us)\n",
               tt ? c.o.threads : 1, tt ? "s" : "");
        printf("  %-5s %-5s %8s", "round", "state", "t_ms");
        for (int i = 0; i < N_ROUND_CELLS; i++) {
            printf(" %-18s", labels[i]);
        }
        printf("\n");
        for (const auto & r : res.rounds) {
            printf("  %-5d %-5s %8.0f", r.idx, r.state.c_str(), r.t_ms);
            for (int i = 0; i < N_ROUND_CELLS; i++) {
                const burst & b = tt ? r.bT[i] : r.b1[i];
                char buf[32];
                if (b.valid) {
                    snprintf(buf, sizeof(buf), "%.0f/%.0f", b.first, b.p50);
                } else {
                    snprintf(buf, sizeof(buf), "--");
                }
                printf(" %-18s", buf);
            }
            printf("\n");
        }
    }

    // The transitions, pass by pass. A one-time invalidation lives entirely in pass 0 and
    // is invisible in any aggregate; this is the only place it can be seen.
    printf("\nP4b raw per-pass series for R-dsp/P at 1 thread (the transitions)\n");
    for (int want : { 1, 2, 3, 10, 13 }) {
        for (const auto & r : res.rounds) {
            if (r.idx != want) {
                continue;
            }
            printf("  round %-2d %-4s :", r.idx, r.state.c_str());
            const burst & b = r.b1[3];
            for (size_t i = 0; i < b.series.size() && i < 40; i++) {
                printf(" %.0f", b.series[i]);
            }
            printf("\n");
        }
    }

    printf("\nP4c DSP event call times (us) -- a cold HAP_mmap2 of a %zu MiB buffer is not free,\n"
           "    so a remap-after-evict is distinguishable from a reuse_buf hit with no debug build\n",
           c.o.arena_mib);
    printf("  E1  tiny ADD (first map)   %10.1f\n", res.e1_us);
    printf("  E7  whole-region ADD       %10.1f\n", res.e7_us);
    if (!res.evict_available) {
        printf("  EVICTION LEVER NOT ARMED: R-evict could not be allocated, so drop_mmap was never\n"
               "  forced and reversibility is UNTESTED. Do not read the rounds above as irreversible.\n");
    }
    for (size_t i = 0; i < res.evict_us.size(); i++) {
        printf("  cycle %zu: evict %8.1f   re-touch %8.1f\n", i + 1, res.evict_us[i],
               i < res.retouch_us.size() ? res.retouch_us[i] : 0.0);
    }
}

// ---------------------------------------------------------------------------------------
// P5 -- the OPSTAGE ladder, in child processes
//
// GGML_HEXAGON_OPSTAGE is read once by ggml_hexagon_init, so it cannot be changed inside a
// running process. Each child is a fresh process, which also means its R-dsp is genuinely
// virgin -- exactly what the S0->S1 step needs.
// ---------------------------------------------------------------------------------------

struct p5_row {
    int    opstage = -1;
    double step = 0;
    bool   ran = false;
    bool   sentinel_ok = false;
};

static p5_row run_p5_child(const options & o, int opstage) {
    p5_row row;
    row.opstage = opstage;

    char self[512] = { 0 };
    const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0) {
        return row;
    }

    char cmd[1400];
    snprintf(cmd, sizeof(cmd),
             "GGML_HEXAGON_OPSTAGE=%d '%s' --dev %s --run p5child --opstage %d "
             "--arena-mib 66 --evict-mib 16 --threads %d --burst %d --no-unc 2>&1",
             opstage, self, o.dev.c_str(), opstage, o.threads, o.burst);

    FILE * p = popen(cmd, "r");
    if (!p) {
        return row;
    }
    char line[1024];
    while (fgets(line, sizeof(line), p)) {
        if (strncmp(line, "P5RESULT", 8) == 0) {
            double step = 0;
            int sok = 0;
            if (sscanf(line, "P5RESULT step=%lf sentinel=%d", &step, &sok) == 2) {
                row.step        = step;
                row.ran         = true;
                row.sentinel_ok = sok != 0;
            }
        } else {
            printf("    [opstage=%d] %s", opstage, line);
        }
    }
    pclose(p);
    return row;
}

// ---------------------------------------------------------------------------------------
// Verdicts
// ---------------------------------------------------------------------------------------

struct verdict_inputs {
    bool   have_p2 = false;
    bool   have_p3 = false;
    bool   have_p4 = false;
    bool   have_p6 = false;
    bool   have_rand = false;
    bool   have_gsweep = false;

    bool   thp_enabled = false;
    long   mdef_anonhuge = -1;

    double m4k_over_mdef  = 0;
    double mhuge_over_mdef = 0;
    double birth = 0;
    double link_b = 0;
    double link_c = 0;
    double unc_over_mdef = 0;
    double page_pen = 0;
    double line_pen = 0;
    double ratio_1t = 0, ratio_8t = 0;

    double step = 0;
    double step_full = 0;
    double step_q = 0;
    double first_over_tail_r2 = 0;
    double mdef_drift = 0;
    double aa_delta = 0;
    bool   aa_ok = true;
    bool   sentinel_ok = true;
    std::vector<double> cyc_evict, cyc_touch;
    double step_seq = 0, step_rand = 0;

    double seq_survive = 0;
    double memcpy_survive = 0;
    double antidiag_deploy = 0;

    p5_row p5_stage1, p5_stage0;
};

static void print_verdicts(const probe_ctx & c, const verdict_inputs & v) {
    hr();
    if (!v.aa_ok) {
        printf("INVALID RUN: A/A gate failed -- rounds 0 and 1 differ by more than %.0f%% with no\n"
               "             DSP activity between them, so drift cannot be told from a real step and\n"
               "             no verdict is licensed. The failing cells are named above.\n"
               "             Remedy: --affinity <mask> to pin one cluster, or --threads 1, or raise\n"
               "             --burst so the tail has more samples. Do NOT raise --aa-tol to get past\n"
               "             it without saying so -- the tolerance is what makes the ratios mean anything.\n",
               c.o.aa_tol * 100.0);
        return;
    }
    if (!v.sentinel_ok) {
        printf("INVALID RUN: DSP op did not execute (sentinel assertion failed). Every 'no step'\n"
               "             below would be an artifact of nothing having run. No verdicts.\n");
        return;
    }
    if (g_ck_failed) {
        printf("INVALID RUN: destination checksum mismatch -- %s\n", g_ck_detail.c_str());
        return;
    }

    // H1
    printf("VERDICT H1  TLB reach / 4KB pages          : ");
    if (!v.have_p2 || !v.have_rand || !v.have_gsweep) {
        printf("UNDECIDED\n   reason: H1 cannot be read off the antidiagonal permutation alone -- at g=512 it\n"
               "   reverses 16 rows = 8 KiB and touches every byte of every page it visits in\n"
               "   page-sequential order, so it is not TLB-hostile. The rand arm and the g sweep\n"
               "   are required inputs and this run does not have them.\n");
    } else if (!v.thp_enabled || v.mdef_anonhuge == 0) {
        printf("KILLED\n   premise  THP %s, M-def AnonHugePages = %ld kB -> the malloc baseline is not\n"
               "            huge-paged, so 'malloc gets THP and rpcmem does not' is false.\n",
               v.thp_enabled ? "on" : "off (/sys/kernel/mm/transparent_hugepage/enabled)", v.mdef_anonhuge);
    } else if (v.m4k_over_mdef >= 1.80) {
        printf("CONFIRMED\n   decider  M-4k/M-def = %.2f   (CONFIRM needs >=1.80, KILL needs <1.15)\n",
               v.m4k_over_mdef);
        printf("   support  M-huge/M-def=%.2f  page-chase pen=%.2f  line-chase pen=%.2f\n",
               v.mhuge_over_mdef, v.page_pen, v.line_pen);
    } else if (v.m4k_over_mdef > 0 && v.m4k_over_mdef < 1.15) {
        printf("KILLED\n   decider  M-4k/M-def = %.2f   (CONFIRM needs >=1.80, KILL needs <1.15)\n",
               v.m4k_over_mdef);
        printf("   support  M-huge/M-def=%.2f  page-chase pen=%.2f  line-chase pen=%.2f\n",
               v.mhuge_over_mdef, v.page_pen, v.line_pen);
        printf("   failed prediction: H1 requires <=1.10 for antidiag@g=512 at 4 MiB; observed %.2f\n",
               v.birth);
    } else {
        printf("UNDECIDED\n   decider  M-4k/M-def = %.2f falls between the KILL (<1.15) and CONFIRM (>=1.80)\n"
               "            thresholds fixed before the run.\n", v.m4k_over_mdef);
    }
    if (v.birth > 0) {
        printf("   links    A(page size) %.2f  B(ION/dmabuf) %.2f  C(fastrpc_mmap) %.2f  -> birth %.2f\n",
               v.m4k_over_mdef, v.link_b, v.link_c, v.birth);
    }

    // H2
    printf("\nVERDICT H2  mapping changes on DSP use     : ");
    const bool h2_have = v.have_p4;
    const bool one_time = v.first_over_tail_r2 > 2.0 && v.step < 1.2;
    bool reversible = false;
    if (v.cyc_evict.size() >= 3 && v.cyc_touch.size() >= 3) {
        reversible = true;
        for (size_t i = 0; i < 3; i++) {
            if (!(v.cyc_evict[i] < 1.3 && v.cyc_touch[i] > 2.0)) {
                reversible = false;
            }
        }
    }
    if (!h2_have) {
        printf("UNDECIDED\n   reason: the persistence timeline (--run persist) was not run.\n");
    } else if (v.mdef_drift > 1.3 && std::fabs(v.mdef_drift - v.step) / std::max(0.01, v.step) < 0.15) {
        // Only when M-def actually MOVED. "Nothing moved anywhere" is a missing step, not a
        // process that got slower, and attributing it to H4 would hide the real result.
        printf("KILLED\n   M-def moved by the same factor (%.2f) as R-dsp (%.2f) at S0->S1: the buffer is\n"
               "   not the subject. See H4.\n", v.mdef_drift, v.step);
    } else if (one_time) {
        printf("KILLED\n   the step lives entirely in pass 0 (first/tail at round 2 = %.2f) and tail(2)\n"
               "   is back at tail(1). That is the per-batch DSP dcache FLUSH_INVALIDATE_ALL\n"
               "   (htp/main.c:1022, :1070) making the CPU refetch from DDR once, not a mapping change.\n",
               v.first_over_tail_r2);
    } else if (v.step >= 2.0) {
        printf("%s\n", reversible ? "CONFIRMED" : "CONFIRMED (weak: irreversible)");
        printf("   step %.2f | tiny-vs-full %.2f | untouched-region Q %.2f", v.step,
               v.step_full > 0 ? v.step / v.step_full : 0.0, v.step_q);
        if (v.p5_stage1.ran) {
            printf(" | OPSTAGE=1 %.2f", v.p5_stage1.step);
        }
        if (v.p5_stage0.ran) {
            printf(" | OPSTAGE=0 %.2f", v.p5_stage0.step);
        }
        printf("\n   reversible:");
        for (size_t i = 0; i < v.cyc_evict.size(); i++) {
            printf(" cycle%zu evict %.2f / touch %.2f ;", i + 1, v.cyc_evict[i],
                   i < v.cyc_touch.size() ? v.cyc_touch[i] : 0.0);
        }
        printf("\n");
        if (!reversible) {
            printf("   NOT reversible: the pages themselves changed. Once the DSP has ever touched a\n"
                   "   buffer, that memory is degraded for the CPU for the buffer's lifetime.\n");
        }
        printf("   sub-mechanism: ");
        if (v.step_seq <= 0 || v.step_rand <= 0) {
            printf("UNDECIDED (need step(seq g=64K) and step(rand g=64) at both S0 and S7)\n");
        } else if (v.unc_over_mdef > 0 && v.antidiag_deploy > 0.7 * v.unc_over_mdef) {
            printf("ATTRIBUTE\n      R-dsp@S7 lands on top of R-unc (%.2f vs %.2f): the CPU mapping effectively\n"
                   "      stopped being cacheable. NO access pattern escapes this.\n",
                   v.antidiag_deploy, v.unc_over_mdef);
        } else if (v.step_rand > 1.8 * v.step_seq) {
            printf("TRANSLATION\n      step(seq g=64K)=%.2f vs step(rand g=64)=%.2f -> sequential amortizes the\n"
                   "      walk, gathers do not. Streaming CPU work survives; gathers do not.\n"
                   "      Corroborate with dTLB-load-misses (see the counters line in the header).\n",
                   v.step_seq, v.step_rand);
        } else {
            printf("ATTRIBUTE (short of uncaching)\n      step(seq g=64K)=%.2f ~ step(rand g=64)=%.2f, both far below R-unc (%.2f):\n"
                   "      shareability / prefetch / write-combining, not a translation change.\n",
                   v.step_seq, v.step_rand, v.unc_over_mdef);
        }
    } else {
        printf("KILLED\n   step = %.2f (needs >=2.0). ", v.step);
        if (v.birth > 1.5) {
            printf("R-virgin is already %.2fx at birth and there is no\n   further step: the whole effect is birth, H2 has nothing to explain.\n", v.birth);
        } else {
            printf("No step at S0->S1 at all.\n");
        }
    }

    // H3
    printf("\nVERDICT H3  access-pattern sensitivity     : ");
    if (!v.have_p6) {
        printf("UNDECIDED\n   reason: the granularity curve was not run.\n");
    } else if (v.seq_survive > 0 && v.seq_survive <= 1.20) {
        printf("CONFIRMED\n   seq g>=4096 @S7 = %.2fx (needs <=1.20)  memcpy-seq = %.2fx\n",
               v.seq_survive, v.memcpy_survive);
        printf("   mllm's std::memcpy path (ShaBlockSparsePromptProcessorSplit.cpp:878, :886, :1090)\n"
               "   is genuinely unaffected, which is why mllm never observed this.\n");
    } else if (v.seq_survive > 0) {
        printf("REFUTED\n   seq g>=4096 @S7 = %.2fx (CONFIRM needs <=1.20): sequential carries the step too,\n"
               "   so nothing survives for any working set that does not fit in cache.\n", v.seq_survive);
    } else {
        printf("UNDECIDED\n   reason: the seq g>=4096 cell is missing.\n");
    }
    if (v.antidiag_deploy > 0) {
        printf("   crossover: the deployment case (antidiag g=512, 4 MiB) is %.2fx -> %s the %.2fx line\n",
               v.antidiag_deploy, v.antidiag_deploy <= SURVIVAL_LINE ? "WINS" : "LOSES", SURVIVAL_LINE);
    }

    // H4
    printf("\nVERDICT H4  process/system got slower      : ");
    if (!v.have_p4) {
        printf("UNDECIDED\n   reason: the persistence timeline was not run.\n");
    } else if (v.mdef_drift > 1.5) {
        printf("CONFIRMED\n   M-def itself moved %.2fx across the rounds. The buffer is innocent and the\n"
               "   framing is wrong: no other verdict above may be relied on.\n", v.mdef_drift);
    } else {
        printf("KILLED\n   M-def drift across all rounds = %+.1f%% ; A/A gate delta %.1f%%\n",
               (v.mdef_drift - 1.0) * 100.0, v.aa_delta * 100.0);
    }

    // Deliverable
    printf("\nDELIVERABLE  : ");
    if (v.seq_survive > 0 && v.seq_survive <= SURVIVAL_LINE && v.antidiag_deploy > SURVIVAL_LINE) {
        printf("STREAMING ONLY\n   sequential work at g>=4096 costs %.2fx over the malloc baseline and clears the\n"
               "   %.2fx break-even, but the deployment gather costs %.2fx and does not.\n",
               v.seq_survive, SURVIVAL_LINE, v.antidiag_deploy);
    } else if (v.antidiag_deploy > 0 && v.antidiag_deploy <= SURVIVAL_LINE) {
        printf("SPLIT VIABLE\n   the deployment gather over R-dsp costs %.2fx over the malloc baseline, under the\n"
               "   %.2fx break-even set by HTP 272 us vs CPU 107 us.\n", v.antidiag_deploy, SURVIVAL_LINE);
    } else if (v.seq_survive > SURVIVAL_LINE) {
        printf("NO SPLIT VIABLE\n   even pure sequential streaming costs %.2fx, above the %.2fx break-even. Nothing\n"
               "   the CPU can do over a DSP-touched buffer beats leaving the work on HTP.\n",
               v.seq_survive, SURVIVAL_LINE);
    } else {
        printf("UNDECIDED\n   the run did not produce both the streaming cell and the deployment cell.\n");
    }
    if (v.antidiag_deploy > 0) {
        printf("   the xattn reversal specifically: %s at %.2fx vs the %.2fx line\n",
               v.antidiag_deploy <= SURVIVAL_LINE ? "WINS" : "LOSES", v.antidiag_deploy, SURVIVAL_LINE);
    }
}

// ---------------------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------------------

static void write_csv(const std::string & path) {
    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "warning: cannot write %s\n", path.c_str());
        return;
    }
    fprintf(f, "arm,region,pattern,g,size,threads,round,state,passes,"
               "first_us,p50_tail_us,min_tail_us,max_tail_us,min20_us,bytes,bytes_per_s,checksum\n");
    for (const auto & c : g_cells) {
        const double bps = c.b.p50 > 0 ? c.b.bytes / (c.b.p50 * 1e-6) : 0.0;
        fprintf(f, "%s,%s,%s,%zu,%zu,%d,%d,%s,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.0f,%.0f,%016" PRIx64 "\n",
                c.arm.c_str(), c.region.c_str(), c.pat.c_str(), c.g, c.bytes, c.threads, c.round,
                c.state.c_str(), c.b.passes, c.b.first, c.b.p50, c.b.mn, c.b.mx, c.b.min20,
                c.b.bytes, bps, c.b.ck);
    }
    fclose(f);
    printf("\ncsv: %s (%zu rows)\n", path.c_str(), g_cells.size());
}

// ---------------------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------------------

static bool alloc_anon(arm & a, size_t bytes, int advice) {
    void * p = nullptr;
    if (posix_memalign(&p, 2 * MiB, bytes) != 0 || !p) {
        return false;
    }
    if (advice) {
        madvise(p, bytes, advice);
    }
    a.alloc   = (uint8_t *) p;
    a.alloc_n = bytes;
    return true;
}

static bool alloc_rpc(arm & a, size_t bytes, uint32_t flags) {
    void * p = g_rpc.get(flags, bytes);
    if (!p) {
        return false;
    }
    a.alloc   = (uint8_t *) p;
    a.alloc_n = bytes;
    a.fd      = g_rpc.to_fd(p);
    a.is_rpc  = true;
    return true;
}

static void finish_arm(arm & a, size_t half) {
    // Never let an allocator choose the intra-page offset: hexagon aligns to 128
    // (ggml-hexagon.cpp:1123) and adds a 4 KiB guard page (:1100), the CPU backend uses
    // ggml_aligned_malloc at 64 (ggml/src/ggml.c:331-336). Differing offsets change how many
    // pages a chunk spans, which is the exact variable H1 is about.
    const uintptr_t base = (uintptr_t) a.alloc;
    const uintptr_t al   = (base + (2 * MiB - 1)) & ~(uintptr_t) (2 * MiB - 1);
    a.w     = (uint8_t *) al;
    a.half  = half;
    a.avail = true;
}

int main(int argc, char ** argv) {
    options o;
    if (!parse_args(argc, argv, o)) {
        return 1;
    }
    o.burst = std::max(4, std::min(4096, o.burst));
    const bool child = (o.run == "p5child");

    // Set before any backend call: ggml_hexagon_init reads these once.
    //   VMEM   -- so a graph_compute on R-evict pushes m_vmem + e_vmem over ctx->max_vmem and
    //             prep_op_bufs drops R-dsp's mapping (htp/main.c:890-896). The default is
    //             3355443200 (htp/htp-ops.h:116) and the path essentially never fires.
    //   OPBATCH-- the op-queue shm is shm_blk_size * opt_opqueue (ggml-hexagon.cpp:1440-1455);
    //             at the default 1024 that is tens of MiB allocated for a one-node graph.
    // scripts/snapdragon/adb/run-tool.sh does not forward either, so the probe sets its own.
    if (!getenv("GGML_HEXAGON_VMEM")) {
        setenv("GGML_HEXAGON_VMEM", std::to_string(o.arena_mib + 8).c_str(), 1);
    }
    if (!getenv("GGML_HEXAGON_OPBATCH")) {
        setenv("GGML_HEXAGON_OPBATCH", "8", 1);
    }
    // Never run the eviction cycle with async: a batch that reads the buffer can still be in
    // flight while drop_mmap tears the DSP mapping down.
    setenv("GGML_HEXAGON_ASYNC", "0", 1);

    ggml_time_init();
    ggml_backend_load_all();

    ggml_backend_t dev = ggml_backend_init_by_name(o.dev.c_str(), nullptr);
    if (!dev) {
        fprintf(stderr, "error: no backend named '%s'. available:\n", o.dev.c_str());
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            fprintf(stderr, "  %s\n", ggml_backend_dev_name(ggml_backend_dev_get(i)));
        }
        return 1;
    }
    ggml_backend_dev_t devdev = ggml_backend_get_device(dev);

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        fprintf(stderr, "error: no CPU backend\n");
        return 1;
    }
    // No probe kernel runs on the CPU backend -- the point of one noinline kernel over raw
    // pointers is that ggml's threadpool cannot vary between arms. It is initialised anyway
    // so that its threadpool exists in the same process and at the same DVFS state as
    // everything else, and set to the same width as the probe's own pool.
    {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
        auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (fn) {
            fn(cpu, o.threads);
        }
    }

    probe_ctx c;
    c.o = o;
    c.fast.max_passes  = o.burst;
    c.fast.budget_ms   = o.budget_ms;
    c.fixed.max_passes = o.burst;
    c.fixed.budget_ms  = 0;

    // Region P is 16 MiB and the DSP's whole-region ADD covers all of it, the sentinel sits
    // 8 MiB into P's dst half, and the copy family needs src and dst halves of equal size.
    if (o.arena_mib < 36) {
        fprintf(stderr, "error: --arena-mib must be >= 36 (region P is 16 MiB in each half)\n");
        return 1;
    }
    const size_t arena = o.arena_mib * MiB;
    c.half = (arena - 2 * MiB) / 2;
    c.half &= ~(2 * MiB - 1);

    g_reg_P = { "P", 0, c.half };
    g_reg_Q = { "Q", c.half / 2, c.half + c.half / 2 };

    // ---- arms -------------------------------------------------------------------------
    arm * A = c.arms;
    A[ARM_M_DEF]    = { "M-def",    "posix_memalign(2MiB), default THP" };
    A[ARM_M_HUGE]   = { "M-huge",   "+ MADV_HUGEPAGE" };
    A[ARM_M_4K]     = { "M-4k",     "+ MADV_NOHUGEPAGE" };
    A[ARM_R_RAW]    = { "R-raw",    "rpcmem_alloc2, never fastrpc_mmap'd" };
    A[ARM_R_UNC]    = { "R-unc",    "rpcmem_alloc2 RPCMEM_FLAG_UNCACHED" };
    A[ARM_R_VIRGIN] = { "R-virgin", "hexagon buft, never in any DSP graph" };
    A[ARM_R_DSP]    = { "R-dsp",    "hexagon buft, subject of every DSP event" };

    if (alloc_anon(A[ARM_M_DEF],  arena, 0))                 finish_arm(A[ARM_M_DEF],  c.half);
    if (alloc_anon(A[ARM_M_HUGE], arena, MADV_HUGEPAGE))     finish_arm(A[ARM_M_HUGE], c.half);
    if (alloc_anon(A[ARM_M_4K],   arena, MADV_NOHUGEPAGE))   finish_arm(A[ARM_M_4K],   c.half);

    const bool have_rpc = g_rpc.open();
    if (have_rpc) {
        if (alloc_rpc(A[ARM_R_RAW], arena, RPCMEM_DEFAULT_FLAGS_)) {
            finish_arm(A[ARM_R_RAW], c.half);
        }
        if (!o.no_unc && alloc_rpc(A[ARM_R_UNC], arena, RPCMEM_FLAG_UNCACHED_)) {
            finish_arm(A[ARM_R_UNC], c.half);
        }
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(devdev);
    ggml_backend_buffer_t hb[2] = { ggml_backend_buft_alloc_buffer(buft, arena),
                                    ggml_backend_buft_alloc_buffer(buft, arena) };
    // R6: the degraded label must follow the DSP ROLE, not the allocation. Run once each way.
    const int i_virgin = o.swap_roles ? 1 : 0;
    const int i_dsp    = o.swap_roles ? 0 : 1;
    if (hb[i_virgin]) {
        A[ARM_R_VIRGIN].buf   = hb[i_virgin];
        A[ARM_R_VIRGIN].alloc = (uint8_t *) ggml_backend_buffer_get_base(hb[i_virgin]);
        A[ARM_R_VIRGIN].alloc_n = arena;
        A[ARM_R_VIRGIN].is_rpc = true;
        if (g_rpc.to_fd) {
            A[ARM_R_VIRGIN].fd = g_rpc.to_fd(A[ARM_R_VIRGIN].alloc);
        }
        finish_arm(A[ARM_R_VIRGIN], c.half);
    }
    if (hb[i_dsp]) {
        A[ARM_R_DSP].buf   = hb[i_dsp];
        A[ARM_R_DSP].alloc = (uint8_t *) ggml_backend_buffer_get_base(hb[i_dsp]);
        A[ARM_R_DSP].alloc_n = arena;
        A[ARM_R_DSP].is_rpc = true;
        if (g_rpc.to_fd) {
            A[ARM_R_DSP].fd = g_rpc.to_fd(A[ARM_R_DSP].alloc);
        }
        finish_arm(A[ARM_R_DSP], c.half);
    }
    if (!A[ARM_M_DEF].avail || !A[ARM_R_DSP].avail) {
        fprintf(stderr, "error: could not allocate the reference arm or the DSP arm at %zu MiB\n",
                o.arena_mib);
        return 1;
    }

    ggml_backend_buffer_t evict_buf = ggml_backend_buft_alloc_buffer(buft, o.evict_mib * MiB);

    // ---- pool, affinity, warm-up ------------------------------------------------------
    g_pool.start(std::max(1, o.threads), o.affinity);

    for (int i = 0; i < ARM_COUNT; i++) {
        if (A[i].avail) {
            fill_arm(A[i]);
        }
    }
    spin_until(ggml_time_us() + 200000);   // raise CPU DVFS before anything is timed

    // ---- DSP graphs -------------------------------------------------------------------
    dsp_setup ds;
    {
        ggml_init_params ip = {};
        ip.mem_size   = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(8, false) * 4;
        ip.no_alloc   = true;
        ds.ctx        = ggml_init(ip);
        ds.evict_buf  = evict_buf;

        uint8_t * base = A[ARM_R_DSP].w;
        ggml_tensor * d = nullptr;

        // 1 element: the DSP touches 12 bytes and HAP_mmap2's all %zu MiB.
        ds.tiny.graph = make_add(ds.ctx, A[ARM_R_DSP].buf, base,
                                 g_reg_P.src_off, g_reg_P.src_off + 128,
                                 g_reg_P.dst_off + SENTINEL_OFF, 1, "tiny", &d);
        ds.tiny.label = "E-tiny";
        arm_sentinel(ds.tiny, (float *) d->data, 0.0f, true);

        // The whole of region P: 16 MiB read, 16 MiB written. src0 and src1 are the same
        // bytes, which add_tensor dedupes into one descriptor -- fine for a binary op.
        const int64_t ne_full = (int64_t) (16 * MiB / 4);
        ds.full.graph = make_add(ds.ctx, A[ARM_R_DSP].buf, base,
                                 g_reg_P.src_off, g_reg_P.src_off,
                                 g_reg_P.dst_off, ne_full, "full", &d);
        ds.full.label = "E-full";

        if (evict_buf) {
            uint8_t * eb = (uint8_t *) ggml_backend_buffer_get_base(evict_buf);
            ggml_tensor * ed = nullptr;
            ds.evict.graph = make_add(ds.ctx, evict_buf, eb, 0, 128, 4096, 1, "evict", &ed);
            ds.evict.label = "V-evict";
        }

        // R1: if the ADD is not supported, flush_batch early-returns (ggml-hexagon.cpp:1636)
        // and "no step" would be reported for a run in which nothing ever reached the DSP.
        for (ggml_cgraph * g : { ds.tiny.graph, ds.full.graph, ds.evict.graph }) {
            if (!g) {
                continue;
            }
            if (!ggml_backend_dev_supports_op(devdev, ggml_graph_node(g, 0))) {
                fprintf(stderr, "error: %s does not support the probe's 1-node ADD; nothing would reach the DSP\n",
                        ggml_backend_name(dev));
                return 1;
            }
        }
        // R2: three tensors, ONE buffer descriptor. If a tensor landed in R-virgin, links C
        // and D would merge silently.
        ggml_tensor * tn = ggml_graph_node(ds.tiny.graph, 0);
        GGML_ASSERT(tn->buffer == A[ARM_R_DSP].buf);
        GGML_ASSERT(tn->src[0]->buffer == A[ARM_R_DSP].buf);
        GGML_ASSERT(tn->src[1]->buffer == A[ARM_R_DSP].buf);
    }

    // ---- header -----------------------------------------------------------------------
    if (!child) {
        struct utsname un;
        uname(&un);
        printf("=== rpcmem-probe ===\n");
        printf("device        : %s  soc %s  kernel %s\n",
               sysprop("ro.product.model").c_str(), sysprop("ro.board.platform").c_str(), un.release);
        printf("build         : ggml %s commit %s\n", ggml_version(), ggml_commit());
        printf("backends      : %s + %s\n", ggml_backend_name(dev), ggml_backend_name(cpu));
        printf("env           : OPSTAGE=%s VMEM=%s(MiB) OPBATCH=%s ASYNC=0  "
               "(the backend's own 'op batching' log line above carries the effective vmem)\n",
               getenv("GGML_HEXAGON_OPSTAGE") ? getenv("GGML_HEXAGON_OPSTAGE") : "0x3(default)",
               getenv("GGML_HEXAGON_VMEM"), getenv("GGML_HEXAGON_OPBATCH"));
        printf("arena         : %zu MiB per arm x %d arms + %zu MiB R-evict = %zu MiB resident\n"
               "                (every byte is pre-written so no first-touch fault can land in a timed\n"
               "                region; use --arena-mib to shrink, minimum 36)\n",
               o.arena_mib, ARM_COUNT, o.evict_mib, o.arena_mib * ARM_COUNT + o.evict_mib);
        printf("                working halves %zu MiB, region P @+0, region Q @+%zu MiB\n",
               c.half / MiB, g_reg_Q.src_off / MiB);
        if (g_reg_Q.src_off < 16 * MiB) {
            printf("                WARNING: region Q starts inside the 16 MiB the DSP's whole-region ADD\n"
                   "                touches, so P-vs-Q is not a clean touched/untouched comparison here.\n"
                   "                Use --arena-mib 130 or more.\n");
        }
        printf("threads       : 1, %d      affinity: ", o.threads);
        if (o.affinity) {
            printf("0x%llx\n", (unsigned long long) o.affinity);
        } else {
            printf("none  (1-thread is the primary number; see R8)\n");
        }
        printf("thp           : enabled=%s  hpage_pmd_size=%s\n",
               read_first_line("/sys/kernel/mm/transparent_hugepage/enabled").c_str(),
               read_first_line("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size").c_str());
        const char * sp = access("/system/bin/simpleperf", X_OK) == 0 ? "/system/bin/simpleperf"
                        : (access("/data/local/tmp/simpleperf", X_OK) == 0 ? "/data/local/tmp/simpleperf" : "ABSENT");
        printf("counters      : simpleperf=%s  perf_event_paranoid=%s\n", sp,
               read_first_line("/proc/sys/kernel/perf_event_paranoid").c_str());
        if (strcmp(sp, "ABSENT") != 0) {
            printf("                %s stat -e dTLB-load-misses,L1-dcache-load-misses,cache-misses,"
                   "cpu-cycles,instructions -- %s --run perf --perf-arm R-dsp --perf-state touched\n",
                   sp, argv[0]);
        }
        printf("clock         : ggml_time_us; per-pass series, bursts of %d, tail = p50 of passes 3..n\n"
               "                (min20 also reported, matching the 105/242/793 dataset's discipline)\n", o.burst);
        printf("survival line : %.2fx over M-def  (HTP 272us vs CPU 107us, xattn-split.cpp:4-6)\n", SURVIVAL_LINE);
        printf("rpcmem        : libcdsprpc %s, rpcmem_alloc2 %s\n",
               have_rpc ? "loaded" : "NOT LOADED (R-raw/R-unc unavailable)",
               g_rpc.alloc2 ? "present" : "absent, using rpcmem_alloc");
        printf("fds           : ");
        for (int i = 0; i < ARM_COUNT; i++) {
            printf("%s=%d ", A[i].name, A[i].fd);
        }
        printf(" (R-virgin and R-dsp must differ; that is what makes link C and link D separable)\n");
        for (int i = 0; i < ARM_COUNT; i++) {
            printf("arm %-9s : %-42s %s\n", A[i].name, A[i].how, A[i].avail ? "" : "UNAVAILABLE");
        }
    }

    verdict_inputs vi;
    p4_result      p4;

    // ---- run --------------------------------------------------------------------------
    //
    // The order below is DSP-state-monotone, not the contract's numeric phase order: every
    // cell that needs R-dsp virgin has to run before the first graph_compute, and every cell
    // that needs it mapped has to run after. There is no other order that produces all of
    // them in one process.

    const bool want_all     = (o.run == "all");
    const bool want_gonogo  = want_all || o.run == "gonogo";
    const bool want_links   = want_all || o.run == "links";
    const bool want_persist = want_all || o.run == "persist" || child;

    if (o.run == "perf") {
        // One cell, looped, for external counter attribution. Still inside an
        // HTP0-initialised process so the DDR corner matches (htp/main.c:437-460).
        arm * pa = nullptr;
        for (int i = 0; i < ARM_COUNT; i++) {
            if (o.perf_arm == A[i].name) {
                pa = &A[i];
            }
        }
        if (!pa || !pa->avail) {
            fprintf(stderr, "error: --perf-arm %s unavailable\n", o.perf_arm.c_str());
            return 1;
        }
        if (o.perf_state == "touched") {
            fire(dev, ds.tiny);
        }
        printf("perf: arm=%s state=%s antidiag g=512 N=4MiB threads=1 for %d s\n",
               pa->name, o.perf_state.c_str(), o.perf_secs);
        const int64_t until = ggml_time_us() + (int64_t) o.perf_secs * 1000000;
        burst_opts bo;
        bo.budget_ms = 200;
        while (ggml_time_us() < until) {
            run_burst(*pa, g_reg_P, PAT_ANTIDIAG, 512, DEPLOY_BYTES, 1, bo);
        }
        g_pool.stop();
        return 0;
    }

    if (!child) {
        dump_smaps(A, "P0 startup (R-dsp virgin)");
        {
            smaps_row r = smaps_for((uintptr_t) A[ARM_M_DEF].w);
            auto it = r.kv.find("AnonHugePages");
            vi.mdef_anonhuge = it == r.kv.end() ? -1 : it->second;
        }
        const std::string thp = read_first_line("/sys/kernel/mm/transparent_hugepage/enabled");
        vi.thp_enabled = thp.find("[always]") != std::string::npos ||
                         thp.find("[madvise]") != std::string::npos;
    }

    // --- pre-DSP: everything that needs R-dsp virgin -----------------------------------
    if (want_links && !child) {
        phase_p2(c);
        vi.have_p2 = true;
        vi.have_rand = true;
        vi.have_gsweep = true;
    }
    if (want_gonogo && !child) {
        phase_p1(c, "S0");
        phase_p3(c, "S0");
        vi.have_p3 = true;
    }
    if (want_persist && !want_gonogo && !want_links && !child) {
        // persist alone still needs the S0 denominators the step is measured against
        phase_p1(c, "S0");
    }

    // --- the DSP timeline ---------------------------------------------------------------
    if (want_persist) {
        if (child) {
            // The child's job is one number: does the step appear at this OPSTAGE?
            const bool expect_write = (o.opstage < 0) || (o.opstage & 2);
            arm_sentinel(ds.tiny, (float *) ggml_graph_node(ds.tiny.graph, 0)->data,
                         *(float *) ((uint8_t *) A[ARM_R_DSP].w + g_reg_P.src_off) +
                         *(float *) ((uint8_t *) A[ARM_R_DSP].w + g_reg_P.src_off + 128),
                         expect_write);
            burst b0 = run_burst(A[ARM_R_DSP], g_reg_P, PAT_ANTIDIAG, 512, DEPLOY_BYTES, 1, c.fixed);
            fire(dev, ds.tiny);
            burst b1 = run_burst(A[ARM_R_DSP], g_reg_P, PAT_ANTIDIAG, 512, DEPLOY_BYTES, 1, c.fixed);
            check_sentinel(ds.tiny);
            printf("P5RESULT step=%.4f sentinel=%d\n",
                   b0.p50 > 0 ? b1.p50 / b0.p50 : 0.0, ds.tiny.ok ? 1 : 0);
            g_pool.stop();
            return 0;
        }

        // The sentinel expectation has to be computed from the live src bytes.
        const float * s = (const float *) (A[ARM_R_DSP].w + g_reg_P.src_off);
        arm_sentinel(ds.tiny, (float *) ggml_graph_node(ds.tiny.graph, 0)->data,
                     s[0] + s[32], true);
        arm_sentinel(ds.full, (float *) ((uint8_t *) A[ARM_R_DSP].w + g_reg_P.dst_off + SENTINEL_OFF),
                     2.0f * ((const float *) (A[ARM_R_DSP].w + g_reg_P.src_off + SENTINEL_OFF))[0], true);

        phase_p4(c, dev, ds, p4);
        print_p4(c, p4);
        vi.have_p4 = true;
        vi.aa_ok = p4.aa_ok;
        vi.sentinel_ok = p4.sentinel_ok;
        if (!p4.aa_ok) {
            printf("\nA/A GATE FAILED: %s\n", p4.aa_detail.c_str());
        }
        if (!p4.sentinel_ok) {
            printf("\nSENTINEL FAILED: %s did not write its destination\n", p4.sentinel_detail.c_str());
        }
        dump_smaps(A, "after the DSP timeline");
    }

    // --- post-DSP: everything that needs R-dsp mapped ------------------------------------
    //
    // R1 again: with --run gonogo there is no persistence timeline, so nothing has touched
    // R-dsp yet and an "S7" table would silently be another S0 table. Fire the whole-region
    // ADD here, and check its sentinel only after P1 has moved enough memory to be sure the
    // read comes from DDR rather than this thread's own dirty line.
    if (want_gonogo && !want_persist) {
        const float * sp = (const float *) (A[ARM_R_DSP].w + g_reg_P.src_off + SENTINEL_OFF);
        arm_sentinel(ds.full, (float *) (A[ARM_R_DSP].w + g_reg_P.dst_off + SENTINEL_OFF),
                     2.0f * sp[0], true);
        fire(dev, ds.full);
        printf("\nE-full (whole region P) took %.1f us -- this is the DSP event the S7 rows below\n"
               "  are taken after; a cold HAP_mmap2 of %zu MiB (htp/main.c:844) is most of it\n",
               ds.full.call_us, o.arena_mib);
    }
    if (want_gonogo || want_persist) {
        phase_p1(c, "S7");

        // The one number the project turns on. Taken at 1 thread: without --affinity the
        // 8-thread cell is one little-core migration away from being noise (R8).
        const double go = ratio(lookup("R-dsp", "P", PAT_SEQ, 65536, DEPLOY_BYTES, 1, "S7"),
                                lookup("M-def", "P", PAT_SEQ, 65536, DEPLOY_BYTES, 1, "S7"));
        printf("\nGO/NO-GO  R-dsp@S7 / M-def on pure streaming (seq g=64K, 4 MiB, 1T) = %.2fx\n", go);
        if (go <= 0) {
            printf("  UNDECIDED: the cell is missing.\n");
        } else if (go <= 1.3) {
            printf("  GO -- streaming CPU work over a DSP-touched buffer is essentially free, so the\n"
                   "  project has a path and the remaining question is only which patterns keep it.\n");
        } else if (go >= SURVIVAL_LINE) {
            printf("  NO-GO -- even pure sequential streaming costs %.2fx, past the %.2fx break-even.\n"
                   "  Nothing the CPU can do over this memory beats leaving the work on HTP. Stop here.\n",
                   go, SURVIVAL_LINE);
        } else {
            printf("  MARGINAL -- streaming costs %.2fx, under the %.2fx break-even but well above the\n"
                   "  1.30x that would make the split unconditional. The granularity curve decides it.\n",
                   go, SURVIVAL_LINE);
        }
    }
    // Before P3, not after: P3 rebuilds the pointer chain in the src half and then restores
    // the reference fill over BOTH halves, which zeroes the sentinel.
    if (want_gonogo && !want_persist) {
        check_sentinel(ds.full);
        vi.sentinel_ok = ds.full.ok;
    }
    if (want_gonogo) {
        phase_p3(c, "S7");
    }
    if (want_all) {
        phase_p6(c);
        vi.have_p6 = true;

        printf("\nP5  OPSTAGE ladder, in child processes (a fresh process is the only way to get\n"
               "    both a different OPSTAGE and a genuinely virgin R-dsp)\n");
        vi.p5_stage1 = run_p5_child(o, 1);
        vi.p5_stage0 = run_p5_child(o, 0);
        printf("  OPSTAGE=0x1 (batch sent, HAP_mmap2 runs, kernels bail on SKIP_COMPUTE) step = %.2f %s\n",
               vi.p5_stage1.step, vi.p5_stage1.ran ? "" : "(child failed)");
        printf("  OPSTAGE=0x0 (nothing enqueued at all, ggml-hexagon.cpp:4052)             step = %.2f %s\n",
               vi.p5_stage0.step, vi.p5_stage0.ran ? "" : "(child failed)");
        printf("  step at 0x1 -> it is the mapping, not DSP data movement.\n"
               "  step at 0x0 -> the cause is on the HOST side of graph_compute and the DSP is\n"
               "                 not involved at all.\n");
    }

    // ---- the CHAIN table ---------------------------------------------------------------
    if (!child && (want_links || want_gonogo)) {
        hr();
        printf("LINK DECOMPOSITION  (antidiag g=512, N=4 MiB, 1 thread, tail p50)\n");
        printf("  %-12s %9s %10s   %-14s %s\n", "arm", "t_us", "vs M-def", "link", "what the link is");
        const burst * ref = lookup("M-def", "P", PAT_ANTIDIAG, 512, DEPLOY_BYTES, 1, "S0");
        if (!ref) {
            printf("  not taken: the S0 half of this table comes from P2, which only --run links\n"
                   "  and --run all execute. A table of zeros would be worse than no table.\n");
        }
        struct { const char * arm; const char * st; const char * link; const char * what; } rows[] = {
            { "M-def",    "S0", "--",           "reference" },
            { "M-huge",   "S0", "--",           "THP forced on" },
            { "M-4k",     "S0", "A page size",  "H1 lives here and nowhere else" },
            { "R-raw",    "S0", "B ION/dmabuf", "rpcmem, never fastrpc_mmap'd" },
            { "R-virgin", "S0", "C fastrpc_mmap", "ggml-hexagon.cpp:402" },
            { "R-dsp",    "S0", "(=R-virgin)",  "if these differ the arms are not comparable" },
            { "R-dsp",    "S7", "D HAP_mmap2",  "htp/main.c:844" },
            { "R-unc",    "S0", "--",           "POSITIVE CONTROL: uncached rpcmem" },
        };
        double t[8] = { 0 };
        for (int i = 0; ref && i < 8; i++) {
            const burst * b = lookup(rows[i].arm, "P", PAT_ANTIDIAG, 512, DEPLOY_BYTES, 1, rows[i].st);
            t[i] = b ? b->p50 : 0;
            printf("  %-8s @%-2s %9.1f ", rows[i].arm, rows[i].st, t[i]);
            print_ratio(b && ref ? b->p50 / ref->p50 : 0);
            printf("   %-14s %s\n", rows[i].link, rows[i].what);
        }
        if (t[0] > 0 && t[2] > 0 && t[3] > 0 && t[4] > 0 && t[6] > 0) {
            const double A_ = t[2] / t[0], B_ = t[3] / t[2], C_ = t[4] / t[3], D_ = t[6] / t[4];
            printf("  product of A*B*C*D = %.2f   vs measured end-to-end %.2f  (must agree within 10%%)\n",
                   A_ * B_ * C_ * D_, t[6] / t[0]);
            vi.m4k_over_mdef = A_;
            vi.link_b = B_;
            vi.link_c = C_;
            vi.birth  = t[4] / t[0];
            vi.have_p2 = true;
        }
        if (t[1] > 0 && t[0] > 0) {
            vi.mhuge_over_mdef = t[1] / t[0];
        }
        if (t[7] > 0 && t[0] > 0) {
            vi.unc_over_mdef = t[7] / t[0];
        }
    }

    // ---- fill in the verdict inputs ----------------------------------------------------
    if (vi.have_p4 && p4.rounds.size() > 2) {
        auto tail = [&](int r, int cell) { return p4.rounds[r].b1[cell].p50; };
        auto firstv = [&](int r, int cell) { return p4.rounds[r].b1[cell].first; };
        const double base_dsp = tail(1, 3);
        vi.step   = base_dsp > 0 ? tail(2, 3) / base_dsp : 0;
        vi.step_q = tail(1, 4) > 0 ? tail(2, 4) / tail(1, 4) : 0;
        vi.first_over_tail_r2 = tail(2, 3) > 0 ? firstv(2, 3) / tail(2, 3) : 0;
        if (p4.rounds.size() > 7) {
            vi.step_full = base_dsp > 0 ? tail(7, 3) / base_dsp : 0;
        }
        double mdef_lo = tail(0, 0), mdef_hi = tail(0, 0);
        for (const auto & r : p4.rounds) {
            mdef_lo = std::min(mdef_lo, r.b1[0].p50);
            mdef_hi = std::max(mdef_hi, r.b1[0].p50);
        }
        vi.mdef_drift = mdef_lo > 0 ? mdef_hi / mdef_lo : 0;
        vi.aa_delta   = tail(0, 0) > 0 ? std::fabs(tail(1, 0) - tail(0, 0)) / tail(0, 0) : 0;

        // The cycles, read off the recorded round index of the FIRST round after each
        // event. Grouping by state label instead would mix cycle 1's three rounds with
        // cycle 2's two and the "repeatable" claim would be unearned.
        for (int ri : p4.evict_round) {
            if (base_dsp > 0 && ri >= 0 && ri < (int) p4.rounds.size()) {
                vi.cyc_evict.push_back(p4.rounds[ri].b1[3].p50 / base_dsp);
            }
        }
        for (int ri : p4.touch_round) {
            if (base_dsp > 0 && ri >= 0 && ri < (int) p4.rounds.size()) {
                vi.cyc_touch.push_back(p4.rounds[ri].b1[3].p50 / base_dsp);
            }
        }
    }
    {
        const burst * s0 = lookup("R-dsp", "P", PAT_SEQ, 65536, DEPLOY_BYTES, c.o.threads, "S0");
        const burst * s7 = lookup("R-dsp", "P", PAT_SEQ, 65536, DEPLOY_BYTES, c.o.threads, "S7");
        vi.step_seq = ratio(s7, s0);
        const burst * r0 = lookup("R-dsp", "P", PAT_RAND, 64, DEPLOY_BYTES, c.o.threads, "S0");
        const burst * r7 = lookup("R-dsp", "P", PAT_RAND, 64, DEPLOY_BYTES, c.o.threads, "S7");
        vi.step_rand = ratio(r7, r0);

        vi.seq_survive = ratio(lookup("R-dsp", "P", PAT_SEQ, 65536, DEPLOY_BYTES, c.o.threads, "S7"),
                               lookup("M-def", "P", PAT_SEQ, 65536, DEPLOY_BYTES, c.o.threads, "S7"));
        vi.memcpy_survive = ratio(lookup("R-dsp", "P", PAT_MEMCPY_SEQ, 65536, DEPLOY_BYTES, c.o.threads, "S7"),
                                  lookup("M-def", "P", PAT_MEMCPY_SEQ, 65536, DEPLOY_BYTES, c.o.threads, "S7"));

        const burst * pc = lookup("R-dsp", "P", PAT_PAGE_CHASE, 0, DEPLOY_BYTES, 1, "S0");
        const burst * lc = lookup("R-dsp", "P", PAT_LINE_CHASE, 0, DEPLOY_BYTES, 1, "S0");
        const burst * pm = lookup("M-def", "P", PAT_PAGE_CHASE, 0, DEPLOY_BYTES, 1, "S0");
        const burst * lm = lookup("M-def", "P", PAT_LINE_CHASE, 0, DEPLOY_BYTES, 1, "S0");
        vi.page_pen = ratio(pc, pm);
        vi.line_pen = ratio(lc, lm);
    }
    if (vi.have_p4 && p4.rounds.size() > 7) {
        const double m = p4.rounds[7].b1[0].p50;
        const double d = p4.rounds[7].b1[3].p50;
        vi.antidiag_deploy = m > 0 ? d / m : 0;
    } else {
        vi.antidiag_deploy = ratio(lookup("R-dsp", "P", PAT_ANTIDIAG, 512, DEPLOY_BYTES, 1, "S7"),
                                   lookup("M-def", "P", PAT_ANTIDIAG, 512, DEPLOY_BYTES, 1, "S7"));
    }

    if (!child) {
        print_verdicts(c, vi);
        if (!o.csv.empty()) {
            write_csv(o.csv);
        }
    }

    g_pool.stop();
    ggml_free(ds.ctx);
    return 0;
}

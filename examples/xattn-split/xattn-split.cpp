// Move the XAttention antidiagonal reversal off HTP and measure what that is worth.
//
// The scorer splits into two very different halves. The reduced matmul is regular and HTP
// wins it 3.7x over the GPU. The reversal is a pure permutation gather (GGML_OP_GET_ROWS,
// no arithmetic) and HTP LOSES it: at Lq=512 the same gather costs HTP 272 us, the GPU
// 177 us and the CPU 107 us. It is 25-43% of the scoring pass.
//
// The point of this harness is that the win does not need concurrency. A hexagon buffer is
// plain host memory -- rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM) and the buffer type reports
// is_host -- so the CPU can compute over an HTP-allocated tensor with no copy and no
// scheduler. Simply RELOCATING the gather, strictly sequential, banks 272-107 us.
//
// ggml_backend_sched cannot be used for this. It would place the gather's output in a CPU
// buffer and pay a single-threaded 4.19 MB ggml_backend_tensor_copy at the boundary, which
// costs more than the whole win. So the two backends are driven by hand here, with every
// tensor living in ONE hexagon buffer.
//
// Measurements, in the order they gate each other:
//   R1  reversal on CPU over rpcmem vs over malloc. If rpcmem is not CPU-cacheable the
//       CPU number collapses and the whole idea is void. THIS GATES EVERYTHING ELSE.
//   A   cpu(reversal) then htp(rest), sequential. The deployable number.
//   B   the overlap ceiling, dependency deliberately broken. Results are garbage, the
//       timing is the ceiling nothing can beat.
//   P   a correct 3-stage head-split pipeline. Heads are independent, so the CPU can
//       reverse the top half while HTP scores the bottom half.
//
// The graph is the one tests/test-backend-ops.cpp test_xattn_score builds at stage 0.
// Run:  llama-xattn-split [--dev HTP0] [--iters 20] [--threads N] [--async] [--S 16]
// --dev CPU runs both roles on the CPU backend, which validates the graphs and the split
// arithmetic on any host.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct scorer_shape {
    int64_t d   = 128;
    int64_t Hq  = 16;
    int64_t Hkv = 8;
    int64_t Lq  = 512;
    int64_t Lk  = 512;
    int64_t S   = 16;
    int64_t Bl  = 64;

    int64_t Nq()  const { return Lq / S;  }
    int64_t Nk()  const { return Lk / S;  }
    int64_t P()   const { return Bl / S;  }   // reduced rows per block
    int64_t NBq() const { return Lq / Bl; }
    int64_t NBk() const { return Lk / Bl; }
};

// A parked worker for the CPU half of the overlap tests. Every hexagon call stays on the
// main thread on purpose: the session's op queue, its free list and its op cache are plain
// non-atomic containers, so only one thread may ever touch a session.
struct cpu_worker {
    std::thread             th;
    std::mutex              m;
    std::condition_variable cv;
    std::function<void()>   job;
    bool                    busy = false;
    bool                    quit = false;

    cpu_worker() {
        th = std::thread([this] {
            for (;;) {
                std::function<void()> j;
                {
                    std::unique_lock<std::mutex> lk(m);
                    cv.wait(lk, [this] { return busy || quit; });
                    if (quit) {
                        return;
                    }
                    j = job;
                }
                j();
                {
                    std::lock_guard<std::mutex> lk(m);
                    busy = false;
                }
                cv.notify_all();
            }
        });
    }

    ~cpu_worker() {
        {
            std::lock_guard<std::mutex> lk(m);
            quit = true;
        }
        cv.notify_all();
        th.join();
    }

    void submit(std::function<void()> j) {
        {
            std::lock_guard<std::mutex> lk(m);
            job  = std::move(j);
            busy = true;
        }
        cv.notify_all();
    }

    void wait() {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [this] { return !busy; });
    }
};

// One head range of the scorer, cut at the reversal.
struct chain {
    int64_t       nh    = 0;
    ggml_tensor * qa    = nullptr;  // reversal output -- whoever runs `rev` writes this
    ggml_tensor * qa_in = nullptr;  // same memory, a plain input to `rest`
    ggml_tensor * sum   = nullptr;  // attn_sum of `rest`
    ggml_tensor * ref   = nullptr;  // attn_sum of `full`, the all-HTP baseline
    ggml_cgraph * rev   = nullptr;  // q, idx -> qa
    ggml_cgraph * rest  = nullptr;  // qa_in, k, cmask -> sum
    ggml_cgraph * full  = nullptr;  // q, idx, k, cmask -> ref  (only for the full range)
};

// Everything after the reversal. qa is [d, Lq, nh] and enters as an input, which is what
// lets a different backend produce it.
static ggml_tensor * build_tail(ggml_context * ctx, const scorer_shape & sh,
                                ggml_tensor * qa, ggml_tensor * k, ggml_tensor * cmask, int64_t nh) {
    const int64_t nkh = nh * sh.Hkv / sh.Hq;

    // Both reshapes are free views. They need contiguity, which a head range of a
    // contiguous tensor has -- heads are the slowest axis.
    ggml_tensor * rq = ggml_reshape_3d(ctx, qa, sh.S * sh.d, sh.Nq(), nh);
    ggml_tensor * rk = ggml_reshape_3d(ctx, k,  sh.S * sh.d, sh.Nk(), nkh);

    ggml_tensor * mm = ggml_mul_mat(ctx, rk, rq);          // F32 [Nk, Nq, nh]
    ggml_mul_mat_set_prec(mm, GGML_PREC_F32);

    // Scale + mask + softmax in one node, exactly as test_xattn_score does.
    ggml_tensor * sm = ggml_soft_max_ext(ctx, mm, cmask,
                                         1.0f / (sqrtf((float) sh.d) * (float) sh.S), 0.0f);

    ggml_tensor * t6  = ggml_reshape_4d(ctx, sm, sh.P(), sh.NBk(), sh.Nq(), nh);
    ggml_tensor * t7  = ggml_sum_rows(ctx, t6);            // [1, NBk, Nq, nh]
    ggml_tensor * t8  = ggml_reshape_4d(ctx, t7, sh.NBk(), sh.P(), sh.NBq(), nh);
    ggml_tensor * t9  = ggml_permute(ctx, t8, 1, 0, 2, 3); // [P, NBk, NBq, nh]
    ggml_tensor * t10 = ggml_cont(ctx, t9);
    ggml_tensor * t11 = ggml_sum_rows(ctx, t10);           // [1, NBk, NBq, nh]

    return ggml_reshape_3d(ctx, t11, sh.NBk(), sh.NBq(), nh);
}

static chain build_chain(ggml_context * ctx, const scorer_shape & sh,
                         ggml_tensor * q, ggml_tensor * k, ggml_tensor * idx, ggml_tensor * cmask,
                         int64_t h0, int64_t h1, bool want_full, int graph_size) {
    chain c;
    c.nh = h1 - h0;

    const int64_t r   = sh.Hq / sh.Hkv;
    const int64_t kh0 = h0 / r;
    const int64_t nkh = c.nh / r;

    ggml_tensor * qv = q, * iv = idx, * kv = k;
    if (c.nh != sh.Hq) {
        // A head range is contiguous, so these views stay reshape-able, which the tail needs.
        qv = ggml_view_3d(ctx, q,   sh.d, sh.Lq, c.nh, q->nb[1], q->nb[2], h0 * q->nb[2]);
        iv = ggml_view_2d(ctx, idx, sh.Lq, c.nh, idx->nb[1], h0 * idx->nb[1]);
        kv = ggml_view_3d(ctx, k,   sh.d, sh.Lk, nkh,  k->nb[1], k->nb[2], kh0 * k->nb[2]);
    }

    c.qa = ggml_get_rows(ctx, qv, iv);                     // F32 [d, Lq, nh]
    ggml_set_name(c.qa, "q_antidiag");

    // The split point as the tail sees it: the same bytes, entering the tail as a LEAF.
    // ggml_view_tensor leaves op == GGML_OP_NONE with no srcs, and graph building follows
    // src[] only, so expanding the tail stops here instead of pulling the reversal back in.
    // The alias has to be a real view, not a patched data pointer: allocation resolves the
    // tail's own reshape views through view_src, and it does that once.
    c.qa_in = ggml_view_tensor(ctx, c.qa);
    ggml_set_name(c.qa_in, "q_antidiag_in");

    c.sum = build_tail(ctx, sh, c.qa_in, kv, cmask, c.nh);
    ggml_set_name(c.sum, "attn_sum");

    c.rev = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(c.rev, c.qa);

    c.rest = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(c.rest, c.sum);

    if (want_full) {
        // The baseline: the whole pass in one graph, the reversal included. Built over its
        // own tail so it can be compared against the split result without a save/restore.
        c.ref = build_tail(ctx, sh, c.qa, kv, cmask, c.nh);
        ggml_set_name(c.ref, "attn_sum_ref");
        c.full = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(c.full, c.ref);
    }

    return c;
}

// Min of N whole-graph runs. Min, not mean: the thing being compared is a fixed amount of
// work, so the fastest run is the one least polluted by scheduling noise.
template <typename F>
static double timeit(int warmup, int iters, F && f) {
    for (int i = 0; i < warmup; i++) {
        f();
    }
    double best = 0;
    for (int i = 0; i < iters; i++) {
        const int64_t t0 = ggml_time_us();
        f();
        const double us = (double) (ggml_time_us() - t0);
        if (i == 0 || us < best) {
            best = us;
        }
    }
    return best;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) a[i] * (double) a[i];
    }
    return den > 0 ? num / den : num;
}

static std::vector<float> read_tensor(ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

static void fill_uniform(ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int64_t n = ggml_nelements(t);
    std::vector<float> f(n);
    for (int64_t i = 0; i < n; i++) {
        f[i] = dist(rng);
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, n * sizeof(float));
    } else {
        std::vector<ggml_fp16_t> h(n);
        ggml_fp32_to_fp16_row(f.data(), h.data(), n);
        ggml_backend_tensor_set(t, h.data(), 0, n * sizeof(ggml_fp16_t));
    }
}

// idx[l] = (l/S)*S + (S-1 - l%S), repeated per head. get_rows has no broadcast over ne1.
static void fill_idx(ggml_tensor * t, const scorer_shape & sh) {
    std::vector<int32_t> perm(sh.Lq * sh.Hq);
    for (int64_t h = 0; h < sh.Hq; h++) {
        for (int64_t l = 0; l < sh.Lq; l++) {
            perm[h * sh.Lq + l] = (int32_t) ((l / sh.S) * sh.S + (sh.S - 1 - l % sh.S));
        }
    }
    ggml_backend_tensor_set(t, perm.data(), 0, perm.size() * sizeof(int32_t));
}

// rk <= rq + (Nk - Nq): the reduced-grid causal mask. -1e30f, not -INFINITY, because
// hexagon's fast softmax prep multiplies the mask in a non-IEEE format.
static void fill_cmask(ggml_tensor * t, const scorer_shape & sh) {
    const int64_t nk = sh.Nk(), nq = sh.Nq(), off = sh.Nk() - sh.Nq();
    std::vector<float> m(nk * nq);
    for (int64_t rq = 0; rq < nq; rq++) {
        for (int64_t rk = 0; rk < nk; rk++) {
            m[rq * nk + rk] = (rk <= rq + off) ? 0.0f : -1e30f;
        }
    }
    ggml_backend_tensor_set(t, m.data(), 0, m.size() * sizeof(float));
}

struct options {
    std::string dev     = "HTP0";
    int         iters   = 20;
    int         warmup  = 3;
    int         threads = 0;   // 0 = hardware_concurrency
    int         S       = 16;
    bool        async   = false;
    std::vector<int64_t> lks { 512, 1024, 2048 };
};

static void usage(const char * prog) {
    printf("usage: %s [options]\n", prog);
    printf("  --dev NAME      device backend to relocate work AWAY from (default HTP0; CPU validates on any host)\n");
    printf("  --iters N       timed iterations per measurement (default 20)\n");
    printf("  --warmup N      untimed iterations (default 3)\n");
    printf("  --threads N     CPU backend threads (default hardware_concurrency)\n");
    printf("  --S N           antidiagonal stride (default 16)\n");
    printf("  --lk A,B,C      key lengths to sweep (default 512,1024,2048)\n");
    printf("  --async         set GGML_HEXAGON_ASYNC=1, so HTP graph_compute only submits\n");
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
        if (a == "--dev")          { o.dev     = next("--dev"); }
        else if (a == "--iters")   { o.iters   = atoi(next("--iters")); }
        else if (a == "--warmup")  { o.warmup  = atoi(next("--warmup")); }
        else if (a == "--threads") { o.threads = atoi(next("--threads")); }
        else if (a == "--S")       { o.S       = atoi(next("--S")); }
        else if (a == "--async")   { o.async   = true; }
        else if (a == "--lk") {
            o.lks.clear();
            std::string s = next("--lk");
            size_t p = 0;
            while (p <= s.size()) {
                const size_t c = s.find(',', p);
                const std::string tok = s.substr(p, c == std::string::npos ? std::string::npos : c - p);
                if (!tok.empty()) {
                    o.lks.push_back(atoll(tok.c_str()));
                }
                if (c == std::string::npos) {
                    break;
                }
                p = c + 1;
            }
        }
        else if (a == "-h" || a == "--help") { usage(argv[0]); exit(0); }
        else {
            fprintf(stderr, "error: unknown argument %s\n", a.c_str());
            usage(argv[0]);
            return false;
        }
    }
    return true;
}

static void set_n_threads(ggml_backend_t backend, int n) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn) {
        fn(backend, n);
    } else {
        fprintf(stderr, "warning: %s has no set_n_threads, it will run at its default\n", ggml_backend_name(backend));
    }
}

// One (shape, allocation backend) worth of tensors, all in a single buffer.
struct arena {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    ggml_tensor * q     = nullptr;
    ggml_tensor * k     = nullptr;
    ggml_tensor * idx   = nullptr;
    ggml_tensor * cmask = nullptr;

    chain full;   // all Hq heads
    chain lo;     // heads [0, Hq/2)
    chain hi;     // heads [Hq/2, Hq)

    ~arena() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

static const int GRAPH_SIZE = 64;

static bool arena_build(arena & a, const scorer_shape & sh, ggml_backend_t alloc_backend, bool halves) {
    // The reference implementation asserts none of these and S | Bl in particular fails
    // SILENTLY there -- the whole block grid comes out the wrong size with no error.
    GGML_ASSERT(sh.Lq % sh.S == 0 && sh.Lk % sh.S == 0);
    GGML_ASSERT(sh.Bl % sh.S == 0 && sh.S <= sh.Bl);
    GGML_ASSERT(sh.Lq % sh.Bl == 0 && sh.Lk % sh.Bl == 0);
    GGML_ASSERT(sh.Hq % sh.Hkv == 0 && sh.Lk >= sh.Lq);
    // the head split has to keep whole GQA groups on each side
    GGML_ASSERT(!halves || (sh.Hkv % 2 == 0));

    const size_t mem = ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(GRAPH_SIZE, false) * 16;
    ggml_init_params ip = { mem, nullptr, /*no_alloc =*/ true };
    a.ctx = ggml_init(ip);
    if (!a.ctx) {
        return false;
    }

    // llama.cpp's flash-attention layout: q is [d, Lq, Hq], k is [d, Lk, Hkv].
    a.q = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, sh.d, sh.Lq, sh.Hq, 1);
    ggml_set_name(a.q, "q");
    a.k = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F16, sh.d, sh.Lk, sh.Hkv, 1);
    ggml_set_name(a.k, "k");
    a.idx = ggml_new_tensor_2d(a.ctx, GGML_TYPE_I32, sh.Lq, sh.Hq);
    ggml_set_name(a.idx, "xattn_idx");
    a.cmask = ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, sh.Nk(), sh.Nq());
    ggml_set_name(a.cmask, "xattn_cmask");

    a.full = build_chain(a.ctx, sh, a.q, a.k, a.idx, a.cmask, 0, sh.Hq, /*want_full =*/ true, GRAPH_SIZE);
    if (halves) {
        a.lo = build_chain(a.ctx, sh, a.q, a.k, a.idx, a.cmask, 0,          sh.Hq / 2, false, GRAPH_SIZE);
        a.hi = build_chain(a.ctx, sh, a.q, a.k, a.idx, a.cmask, sh.Hq / 2,  sh.Hq,     false, GRAPH_SIZE);
    }

    // One buffer for everything, so the CPU/device boundary is a shared address and not a
    // ggml_backend_tensor_copy. This is the whole point of the design.
    a.buf = ggml_backend_alloc_ctx_tensors(a.ctx, alloc_backend);
    return a.buf != nullptr;
}

static void arena_fill(arena & a, const scorer_shape & sh, uint32_t seed) {
    std::mt19937 rng(seed);
    fill_uniform(a.q, rng);
    fill_uniform(a.k, rng);
    fill_idx(a.idx, sh);
    fill_cmask(a.cmask, sh);
}

// Poison the reversal output so a stale value cannot fake a correct split result. This is
// the check that CPU writes are actually visible to the DSP.
static void poison(ggml_tensor * t) {
    std::vector<float> z(ggml_nelements(t), 0.0f);
    ggml_backend_tensor_set(t, z.data(), 0, z.size() * sizeof(float));
}

int main(int argc, char ** argv) {
    options o;
    if (!parse_args(argc, argv, o)) {
        return 1;
    }
    if (o.async) {
        // Read once by ggml_hexagon_init, so it has to be set before any backend call.
#ifdef _WIN32
        _putenv_s("GGML_HEXAGON_ASYNC", "1");
#else
        setenv("GGML_HEXAGON_ASYNC", "1", 1);
#endif
    }
    if (o.threads <= 0) {
        o.threads = (int) std::thread::hardware_concurrency();
    }

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
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        fprintf(stderr, "error: no CPU backend\n");
        return 1;
    }
    // Mandatory. The CPU backend defaults to GGML_DEFAULT_N_THREADS = 4, and the 107 us
    // reference figure is 8-threaded, so forgetting this makes the split look ~2x worse.
    set_n_threads(cpu, o.threads);

    cpu_worker worker;

    scorer_shape sh;
    sh.S = o.S;

    printf("xattn-split: relocating the XAttention antidiagonal reversal off %s\n", ggml_backend_name(dev));
    printf("  device backend : %s\n", ggml_backend_name(dev));
    printf("  host backend   : %s (%d threads)\n", ggml_backend_name(cpu), o.threads);
    printf("  shape          : d=%lld Hq=%lld Hkv=%lld Lq=%lld S=%lld Bl=%lld\n",
           (long long) sh.d, (long long) sh.Hq, (long long) sh.Hkv,
           (long long) sh.Lq, (long long) sh.S, (long long) sh.Bl);
    printf("  boundary       : q_antidiag %lld bytes\n", (long long) (sh.d * sh.Lq * sh.Hq * 4));
    printf("  timing         : min of %d iters (%d warmup), whole graph, no node replication\n",
           o.iters, o.warmup);
    printf("  hexagon async  : %s\n", o.async ? "requested (GGML_HEXAGON_ASYNC=1)" : "off (default)");
    printf("\n");

    // ---- R1 GATE ------------------------------------------------------------------
    // If rpcmem is not CPU-cacheable the CPU reversal is 5-20x slower than its malloc
    // baseline and the entire approach is void. Everything below is meaningless until
    // this ratio is ~1. The reversal only touches Lq, so it does not depend on Lk.
    {
        scorer_shape sh1 = sh;
        sh1.Lk = sh.Lq;

        arena a_dev, a_cpu;
        if (!arena_build(a_dev, sh1, dev, false) || !arena_build(a_cpu, sh1, cpu, false)) {
            fprintf(stderr, "error: allocation failed\n");
            return 1;
        }
        arena_fill(a_dev, sh1, 1234);
        arena_fill(a_cpu, sh1, 1234);

        const double t_rpc = timeit(o.warmup, o.iters, [&] { ggml_backend_graph_compute(cpu, a_dev.full.rev); });
        const double t_mal = timeit(o.warmup, o.iters, [&] { ggml_backend_graph_compute(cpu, a_cpu.full.rev); });
        const double t_dsp = timeit(o.warmup, o.iters, [&] { ggml_backend_graph_compute(dev, a_dev.full.rev); });

        printf("R1 GATE -- reversal alone, same graph, different buffers\n");
        printf("  %-28s %8.1f us\n", "cpu over device buffer",  t_rpc);
        printf("  %-28s %8.1f us\n", "cpu over cpu buffer",     t_mal);
        printf("  %-28s %8.1f us\n", "device over device buffer", t_dsp);
        printf("  ratio cpu(device)/cpu(cpu) = %.2f  %s\n", t_rpc / t_mal,
               t_rpc / t_mal < 1.5 ? "-> device memory is CPU-cacheable, proceed"
                                           : "-> STOP: device memory is slow for the CPU");
        printf("\n");
    }

    // ---- the sweep ----------------------------------------------------------------
    printf("PER-Lk (us, min). full = all-device baseline. A = cpu(rev) then device(rest), sequential.\n");
    printf("B = overlap ceiling, DEPENDENCY BROKEN so results are garbage. P = correct head-split pipeline.\n\n");
    printf("  %6s %9s %9s %9s %9s | %9s %6s | %9s %9s %6s | %9s %6s | %10s %10s\n",
           "Lk", "full", "dev_rev", "dev_rest", "cpu_rev",
           "A", "x", "B_thread", "B_async", "x",
           "P", "x", "nmse_A", "nmse_P");

    for (int64_t lk : o.lks) {
        scorer_shape s = sh;
        s.Lk = lk;

        arena a;
        if (!arena_build(a, s, dev, true)) {
            fprintf(stderr, "error: allocation failed at Lk=%lld\n", (long long) lk);
            return 1;
        }
        arena_fill(a, s, 1234);

        const double t_full = timeit(o.warmup, o.iters, [&] { ggml_backend_graph_compute(dev, a.full.full); });
        const double t_drev = timeit(o.warmup, o.iters, [&] { ggml_backend_graph_compute(dev, a.full.rev);  });
        // R3: measured standalone, never derived by subtracting full - rev.
        const double t_rest = timeit(o.warmup, o.iters, [&] { ggml_backend_graph_compute(dev, a.full.rest); });
        const double t_crev = timeit(o.warmup, o.iters, [&] { ggml_backend_graph_compute(cpu, a.full.rev);  });

        const double t_a = timeit(o.warmup, o.iters, [&] {
            ggml_backend_graph_compute(cpu, a.full.rev);
            ggml_backend_graph_compute(dev, a.full.rest);
        });

        // R4: the pure overlap ceiling. rest reads q_antidiag while rev rewrites it, so the
        // numbers below are wrong on purpose -- if this does not beat A, nothing will.
        const double t_bt = timeit(o.warmup, o.iters, [&] {
            worker.submit([&] { ggml_backend_graph_compute(cpu, a.full.rev); });
            ggml_backend_graph_compute(dev, a.full.rest);
            worker.wait();
        });
        // Same ceiling on one thread. Needs --async, otherwise graph_compute still blocks
        // and this just reproduces A.
        const double t_ba = timeit(o.warmup, o.iters, [&] {
            ggml_backend_graph_compute_async(dev, a.full.rest);
            ggml_backend_graph_compute(cpu, a.full.rev);
            ggml_backend_synchronize(dev);
        });

        // The correct pipeline. Heads are independent, so stage 2 overlaps the CPU reversal
        // of the top half with the device scoring the bottom half. The CPU only ever writes
        // a tensor no in-flight batch reads.
        const double t_p = timeit(o.warmup, o.iters, [&] {
            ggml_backend_graph_compute(cpu, a.lo.rev);
            worker.submit([&] { ggml_backend_graph_compute(cpu, a.hi.rev); });
            ggml_backend_graph_compute(dev, a.lo.rest);
            worker.wait();
            ggml_backend_graph_compute(dev, a.hi.rest);
        });

        // ---- correctness -----------------------------------------------------------
        ggml_backend_graph_compute(dev, a.full.full);
        const std::vector<float> ref = read_tensor(a.full.ref);

        poison(a.full.qa);
        ggml_backend_graph_compute(cpu, a.full.rev);
        ggml_backend_graph_compute(dev, a.full.rest);
        const double e_a = nmse(ref, read_tensor(a.full.sum));

        poison(a.lo.qa);
        poison(a.hi.qa);
        ggml_backend_graph_compute(cpu, a.lo.rev);
        worker.submit([&] { ggml_backend_graph_compute(cpu, a.hi.rev); });
        ggml_backend_graph_compute(dev, a.lo.rest);
        worker.wait();
        ggml_backend_graph_compute(dev, a.hi.rest);

        std::vector<float> got = read_tensor(a.lo.sum);
        const std::vector<float> got_hi = read_tensor(a.hi.sum);
        got.insert(got.end(), got_hi.begin(), got_hi.end());
        const double e_p = nmse(ref, got);

        printf("  %6lld %9.1f %9.1f %9.1f %9.1f | %9.1f %6.2f | %9.1f %9.1f %6.2f | %9.1f %6.2f | %10.2e %10.2e\n",
               (long long) lk, t_full, t_drev, t_rest, t_crev,
               t_a, t_full / t_a,
               t_bt, t_ba, t_full / t_bt,
               t_p, t_full / t_p,
               e_a, e_p);
        fflush(stdout);
    }

    printf("\n");
    printf("notes\n");
    printf("  * dev_rev + dev_rest - full is the extra per-graph_compute constant the split pays.\n");
    printf("  * nmse_A > 0 with a correct graph means the device did not see the CPU's writes.\n");
    printf("  * B is an upper bound only. The real chain q -> reversal -> matmul cannot overlap;\n");
    printf("    P is what a correct pipeline gets, and it costs one un-overlapped half at each end.\n");

    ggml_backend_free(cpu);
    ggml_backend_free(dev);
    return 0;
}

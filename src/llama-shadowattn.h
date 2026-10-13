#pragma once

// ShadowAttn (Yin et al., arXiv 2508.16703) in the prefill graph, for the phone's end-to-end prefill comparison. Its
// estimation Q.K^T runs on the NPU (one MUL_MAT per layer: the HMX F16 path, F32 scores); its per-row top-k and the
// exact attention over the selected keys run on the CPU as one custom op, with the NEON kernels of the operator
// benchmark (heterogeneous-accuracy-20260917/shadowattn-20260928/phone/ShadowAttnBench.cpp). A row's scores become
// 8-bit over the row's range, as the 8-bit scores of ShadowAttn's INT8 NPU graph; a binary search on the 8-bit cut
// selects ceil(r * visible) keys, lowest index first on ties; F16 products with F32 accumulation (FMLAL). The accuracy
// reference is docs/papers/heterogeneous-inference/accuracy/cuda_attention.py::shadowattn_mask with one ratio for
// every head.
//
//   LLAMA_SHADOWATTN=r          per-row key ratio, 0 < r <= 1 (unset: off)
//   LLAMA_SHADOWATTN_TIMING=1   one line per op call and thread: wall, 8-bit scores, top-k and attention time, CPU
//   LLAMA_SHADOWATTN_CHECK=n    every n-th row of every head again in double: a stable sort of the same 8-bit scores
//                               must give the same keys, and the NPU scores of the row are recomputed from Q and K;
//                               the largest output and score errors are reported at exit
//
// A ubatch is one chunk of the accuracy protocol: aligned, on a cleared cache, keys 0..n_kv-1 with the ubatch at the
// end (build_attn_mha checks it), so row i sees keys 0..n_kv-n_tokens+i. Ubatches of at most 256 queries (the tail)
// stay dense, as in the reference.

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_FML)
#    include <arm_neon.h>
#    define LLAMA_SHADOWATTN_NEON 1
#endif
#if defined(__linux__)
#    include <sched.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#    include <unistd.h>
#endif

static double llama_shadowattn_ratio() {
    static const double r = [] {
        const char * s = getenv("LLAMA_SHADOWATTN");
        if (!s || !*s) {
            return 0.0;
        }
        char *       end = nullptr;
        const double v   = strtod(s, &end);
        GGML_ASSERT(end != s && *end == '\0' && v > 0.0 && v <= 1.0 && "LLAMA_SHADOWATTN must be in (0, 1]");
        return v;
    }();
    return r;
}

static int llama_shadowattn_env_int(const char * name) {
    const char * s = getenv(name);
    return s && *s ? atoi(s) : 0;
}

struct llama_shadowattn_layer {
    double ratio;
    float  scale;
    int    il;
};

// the reference's count, in double: min(visible, max(1, ceil(r * visible)))
static int llama_shadowattn_row_keys(double ratio, int n) {
    return std::min(n, std::max(1, (int) std::ceil(ratio * n)));
}

struct llama_shadowattn_stats {
    std::atomic<long long> rows{0}, pairs{0}, checked{0}, select_mismatch{0};
    std::atomic<unsigned>  cpus{0};
    std::atomic<int>       out_err_ppm{0}, score_err_ppm{0};  // largest relative errors, parts per million
    std::atomic<long long> score_bad_rows{0};                 // checked rows with a score error above 1% of the row's max
    std::mutex             worst_mutex;
    char                   worst[256] = "";                   // where the largest score error was seen
    std::atomic<long long> bad_keys[64][2] = {};              // sampled keys by 256-key block: [block][0] bad (>1%), [1] all
    std::atomic<long long> bad_rows_kv[64][2] = {};           // checked rows by n_kv / 256: [0] with a bad key, [1] all

    ~llama_shadowattn_stats() {
        if (rows.load() == 0) {
            return;
        }
        fprintf(stderr, "shadowattn: %lld rows, %lld selected pairs, cpus 0x%x", rows.load(), pairs.load(), cpus.load());
        if (checked.load()) {
            fprintf(stderr, "; checked %lld rows: selections differing %lld, max output error %.1f ppm of the row's max |o|, "
                            "max NPU score error %.1f ppm of the row's max |score| (%lld rows above 1%%; worst: %s)",
                    checked.load(), select_mismatch.load(), (double) out_err_ppm.load(), (double) score_err_ppm.load(),
                    score_bad_rows.load(), worst);
            fprintf(stderr, "\nshadowattn-check rows by n_kv/256 (bad/all):");
            for (int b = 0; b < 64; ++b) {
                if (bad_rows_kv[b][1].load()) {
                    fprintf(stderr, " %d:%lld/%lld", b, bad_rows_kv[b][0].load(), bad_rows_kv[b][1].load());
                }
            }
            fprintf(stderr, "\nshadowattn-check sampled keys by key/256 (bad/all):");
            for (int b = 0; b < 64; ++b) {
                if (bad_keys[b][1].load()) {
                    fprintf(stderr, " %d:%lld/%lld", b, bad_keys[b][0].load(), bad_keys[b][1].load());
                }
            }
        }
        fprintf(stderr, "\n");
    }
};

static llama_shadowattn_stats & llama_shadowattn_global_stats() {
    static llama_shadowattn_stats s;
    return s;
}

static void llama_shadowattn_atomic_max(std::atomic<int> & a, int v) {
    for (int cur = a.load(); v > cur && !a.compare_exchange_weak(cur, v);) {
    }
}

// --- scores -> 8 bit, top-k on 8-bit scores ---

// u8 = round((s - min) * 255 / (max - min)) over the n visible scores
static void llama_shadowattn_to_u8(const float * s, int n, uint8_t * y) {
#ifdef LLAMA_SHADOWATTN_NEON
    float32x4_t vmx = vdupq_n_f32(-INFINITY), vmn = vdupq_n_f32(INFINITY);
    int         i   = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t x = vld1q_f32(s + i);
        vmx                 = vmaxq_f32(vmx, x);
        vmn                 = vminq_f32(vmn, x);
    }
    float mx = vmaxvq_f32(vmx), mn = vminvq_f32(vmn);
    for (; i < n; ++i) {
        mx = std::max(mx, s[i]);
        mn = std::min(mn, s[i]);
    }
    const float       inv  = mx > mn ? 255.0f / (mx - mn) : 0.0f;
    const float32x4_t vinv = vdupq_n_f32(inv), vmin = vdupq_n_f32(mn);
    for (i = 0; i + 8 <= n; i += 8) {
        const uint32x4_t a = vcvtnq_u32_f32(vmulq_f32(vsubq_f32(vld1q_f32(s + i), vmin), vinv));
        const uint32x4_t b = vcvtnq_u32_f32(vmulq_f32(vsubq_f32(vld1q_f32(s + i + 4), vmin), vinv));
        vst1_u8(y + i, vqmovn_u16(vcombine_u16(vqmovn_u32(a), vqmovn_u32(b))));
    }
    for (; i < n; ++i) {
        y[i] = (uint8_t) std::min(255.0f, nearbyintf((s[i] - mn) * inv));
    }
#else
    float mx = -INFINITY, mn = INFINITY;
    for (int i = 0; i < n; ++i) {
        mx = std::max(mx, s[i]);
        mn = std::min(mn, s[i]);
    }
    const float inv = mx > mn ? 255.0f / (mx - mn) : 0.0f;
    for (int i = 0; i < n; ++i) {
        y[i] = (uint8_t) std::min(255.0f, nearbyintf((s[i] - mn) * inv));
    }
#endif
}

static int llama_shadowattn_count_ge(const uint8_t * s, int n, int c) {
#ifdef LLAMA_SHADOWATTN_NEON
    // 16 lanes of u8 counters, widened every 128 vectors
    const uint8x16_t vc    = vdupq_n_u8((uint8_t) c);
    uint32x4_t       total = vdupq_n_u32(0);
    int              i     = 0;
    while (i + 16 <= n) {
        uint8x16_t acc  = vdupq_n_u8(0);
        const int  stop = std::min(n - 15, i + 16 * 128);
        for (; i < stop; i += 16) {
            acc = vsubq_u8(acc, vcgeq_u8(vld1q_u8(s + i), vc));
        }
        total = vpadalq_u16(total, vpaddlq_u8(acc));
    }
    int r = (int) vaddvq_u32(total);
    for (; i < n; ++i) {
        r += s[i] >= c;
    }
    return r;
#else
    int r = 0;
    for (int i = 0; i < n; ++i) {
        r += s[i] >= c;
    }
    return r;
#endif
}

// the k largest of n 8-bit scores, lowest index first on ties at the cut; returns k indices (two ascending runs)
static int llama_shadowattn_select(const uint8_t * s, int n, int k, int32_t * out) {
    int lo = 0, hi = 255;  // invariant: count_ge(lo) >= k
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (llama_shadowattn_count_ge(s, n, mid) >= k) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const int b = lo, above = b == 255 ? 0 : llama_shadowattn_count_ge(s, n, b + 1);
    int       m = 0, ties = k - above;
    int       i = 0;
#ifdef LLAMA_SHADOWATTN_NEON
    const uint8x16_t vb = vdupq_n_u8((uint8_t) b);
    for (; i + 16 <= n; i += 16) {
        uint64_t bits = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(vcgtq_u8(vld1q_u8(s + i), vb)), 4)), 0);
        while (bits) {
            const int z = __builtin_ctzll(bits);
            out[m++]    = i + (z >> 2);
            bits &= ~(0xFull << (z & ~3));
        }
    }
#endif
    for (; i < n; ++i) {
        if (s[i] > b) {
            out[m++] = i;
        }
    }
    // ties at the cut: lowest indices first
    i = 0;
#ifdef LLAMA_SHADOWATTN_NEON
    for (; ties > 0 && i + 16 <= n; i += 16) {
        uint64_t bits = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(vceqq_u8(vld1q_u8(s + i), vb)), 4)), 0);
        while (bits && ties > 0) {
            const int z = __builtin_ctzll(bits);
            out[m++]    = i + (z >> 2);
            --ties;
            bits &= ~(0xFull << (z & ~3));
        }
    }
#endif
    for (; ties > 0 && i < n; ++i) {
        if (s[i] == b) {
            out[m++] = i;
            --ties;
        }
    }
    return m;
}

// --- exact attention over the selected keys ---

#ifdef LLAMA_SHADOWATTN_NEON
// ggml_v_expf (ggml-cpu/vec.h, MIT)
static inline float32x4_t llama_shadowattn_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n, vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t  e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t  c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
                                    vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                                              vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u),
                                    u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c))) {
        return vfmaq_f32(k, j, k);
    }
    const uint32x4_t  dd = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(dd, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, dd));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

// softmax(q.K_sel^T * scale) V_sel for D = 128: F16 products, F32 accumulation (FMLAL); K and V rows at a byte stride
static void llama_shadowattn_row(const __fp16 * q, const char * K, const char * V, size_t stride, const int32_t * idx, int m,
                                 float scale, float * sc, float * out) {
    constexpr int D  = 128;
    float         mx = -INFINITY;
    int           t  = 0;
    // Q.K: four keys at a time, 16 FMLAL chains (even/odd dims x low/high x key)
    for (; t + 4 <= m; t += 4) {
        if (t + 12 <= m) {
            for (int u = 8; u < 12; ++u) {
                const char * pf = K + (size_t) idx[t + u] * stride;
                __builtin_prefetch(pf);
                __builtin_prefetch(pf + 64);
                __builtin_prefetch(pf + 128);
                __builtin_prefetch(pf + 192);
            }
        }
        const __fp16 * kp[4] = {
            (const __fp16 *) (K + (size_t) idx[t] * stride), (const __fp16 *) (K + (size_t) idx[t + 1] * stride),
            (const __fp16 *) (K + (size_t) idx[t + 2] * stride), (const __fp16 *) (K + (size_t) idx[t + 3] * stride)
        };
        float32x4_t a[16];
        for (auto & x : a) {
            x = vdupq_n_f32(0);
        }
        for (int i = 0; i < D / 8; i += 2) {
            const float16x8_t q0 = vld1q_f16(q + 8 * i), q1 = vld1q_f16(q + 8 * i + 8);
            for (int j = 0; j < 4; ++j) {
                const float16x8_t k0 = vld1q_f16(kp[j] + 8 * i), k1 = vld1q_f16(kp[j] + 8 * i + 8);
                a[4 * j]             = vfmlalq_low_f16(a[4 * j], q0, k0);
                a[4 * j + 1]         = vfmlalq_high_f16(a[4 * j + 1], q0, k0);
                a[4 * j + 2]         = vfmlalq_low_f16(a[4 * j + 2], q1, k1);
                a[4 * j + 3]         = vfmlalq_high_f16(a[4 * j + 3], q1, k1);
            }
        }
        float32x4_t r = vdupq_n_f32(0);
        r = vsetq_lane_f32(vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3]))), r, 0);
        r = vsetq_lane_f32(vaddvq_f32(vaddq_f32(vaddq_f32(a[4], a[5]), vaddq_f32(a[6], a[7]))), r, 1);
        r = vsetq_lane_f32(vaddvq_f32(vaddq_f32(vaddq_f32(a[8], a[9]), vaddq_f32(a[10], a[11]))), r, 2);
        r = vsetq_lane_f32(vaddvq_f32(vaddq_f32(vaddq_f32(a[12], a[13]), vaddq_f32(a[14], a[15]))), r, 3);
        r = vmulq_n_f32(r, scale);
        vst1q_f32(sc + t, r);
        mx = std::max(mx, vmaxvq_f32(r));
    }
    for (; t < m; ++t) {
        const __fp16 * k  = (const __fp16 *) (K + (size_t) idx[t] * stride);
        float32x4_t    a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        for (int i = 0; i < D / 8; ++i) {
            const float16x8_t qv = vld1q_f16(q + 8 * i), kv = vld1q_f16(k + 8 * i);
            a0                   = vfmlalq_low_f16(a0, qv, kv);
            a1                   = vfmlalq_high_f16(a1, qv, kv);
        }
        sc[t] = vaddvq_f32(vaddq_f32(a0, a1)) * scale;
        mx    = std::max(mx, sc[t]);
    }
    float32x4_t sum = vdupq_n_f32(0), vm = vdupq_n_f32(mx);
    for (t = 0; t + 4 <= m; t += 4) {
        const float32x4_t e = llama_shadowattn_expf(vsubq_f32(vld1q_f32(sc + t), vm));
        vst1q_f32(sc + t, e);
        sum = vaddq_f32(sum, e);
    }
    float total = vaddvq_f32(sum);
    for (; t < m; ++t) {
        total += (sc[t] = expf(sc[t] - mx));
    }
    const float inv = 1.0f / total;
    // P.V: 16 accumulators per 64-dim half, one broadcast probability per key
    for (int half = 0; half < 2; ++half) {
        float32x4_t o[16];
        for (auto & x : o) {
            x = vdupq_n_f32(0);
        }
        for (int u = 0; u < m; ++u) {
            if (u + 8 < m) {
                const char * pf = V + (size_t) idx[u + 8] * stride + half * 128;
                __builtin_prefetch(pf);
                __builtin_prefetch(pf + 64);
            }
            const float16x8_t p = vdupq_n_f16((__fp16) (sc[u] * inv));
            const __fp16 *    v = (const __fp16 *) (V + (size_t) idx[u] * stride) + half * 64;
            for (int i = 0; i < 8; ++i) {
                const float16x8_t vv = vld1q_f16(v + 8 * i);
                o[2 * i]             = vfmlalq_low_f16(o[2 * i], p, vv);
                o[2 * i + 1]         = vfmlalq_high_f16(o[2 * i + 1], p, vv);
            }
        }
        for (int i = 0; i < 16; ++i) {
            vst1q_f32(out + half * 64 + 4 * i, o[i]);
        }
    }
}
#endif

// any D, F32 arithmetic (hosts without FMLAL)
static void llama_shadowattn_row_ref(const float * q, const char * K, const char * V, size_t stride, int D, const int32_t * idx,
                                     int m, float scale, float * sc, float * out) {
    float mx = -INFINITY;
    for (int t = 0; t < m; ++t) {
        const ggml_fp16_t * k = (const ggml_fp16_t *) (K + (size_t) idx[t] * stride);
        float               d = 0.0f;
        for (int i = 0; i < D; ++i) {
            d += q[i] * ggml_fp16_to_fp32(k[i]);
        }
        sc[t] = d * scale;
        mx    = std::max(mx, sc[t]);
    }
    float total = 0.0f;
    for (int t = 0; t < m; ++t) {
        total += (sc[t] = expf(sc[t] - mx));
    }
    std::fill(out, out + D, 0.0f);
    for (int t = 0; t < m; ++t) {
        const ggml_fp16_t * v = (const ggml_fp16_t *) (V + (size_t) idx[t] * stride);
        const float         p = sc[t] / total;
        for (int i = 0; i < D; ++i) {
            out[i] += p * ggml_fp16_to_fp32(v[i]);
        }
    }
}

// --- the op: dst [D, Hq, L] F32; src kq [n_kv, L, Hq] F32 (raw NPU scores), q [D, L, Hq] F32, k/v [D, n_kv, Hkv] F16 ---

static void llama_shadowattn_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    using clk                       = std::chrono::steady_clock;
    const auto                 t_op = clk::now();
    const auto *               p    = (const llama_shadowattn_layer *) userdata;
    const ggml_tensor *        kq   = dst->src[0];
    const ggml_tensor *        q    = dst->src[1];
    const ggml_tensor *        k    = dst->src[2];
    const ggml_tensor *        v    = dst->src[3];
    const int                  D = (int) q->ne[0], L = (int) q->ne[1], Hq = (int) q->ne[2];
    const int                  n_kv = (int) k->ne[1], G = Hq / (int) k->ne[2], off = n_kv - L;
    static const int           timing = llama_shadowattn_env_int("LLAMA_SHADOWATTN_TIMING");
    static const int           check  = llama_shadowattn_env_int("LLAMA_SHADOWATTN_CHECK");
    GGML_ASSERT(k->nb[1] == v->nb[1] && k->nb[2] == v->nb[2] && kq->nb[0] == sizeof(float) && q->nb[0] == sizeof(float));

    thread_local std::vector<float>   sc;
    thread_local std::vector<uint8_t> s8;
    thread_local std::vector<int32_t> idx, order;
    sc.resize(n_kv);
    s8.resize(n_kv);
    idx.resize(n_kv);

    constexpr int ROWS  = 32;  // one work item: 32 rows of one query head; items interleave over the threads
    const int     items = Hq * ((L + ROWS - 1) / ROWS);
    double        t_u8 = 0, t_sel = 0, t_att = 0;
    long long     rows = 0, pairs = 0;
    for (int it = ith; it < items; it += nth) {
        const int    h = it / ((L + ROWS - 1) / ROWS), r0 = (it % ((L + ROWS - 1) / ROWS)) * ROWS;
        const char * K = (const char *) k->data + (size_t) (h / G) * k->nb[2];
        const char * V = (const char *) v->data + (size_t) (h / G) * v->nb[2];
        for (int r = r0; r < std::min(L, r0 + ROWS); ++r) {
            const int     n  = off + r + 1, m = llama_shadowattn_row_keys(p->ratio, n);
            const float * s  = (const float *) ((const char *) kq->data + (size_t) r * kq->nb[1] + (size_t) h * kq->nb[2]);
            const float * qr = (const float *) ((const char *) q->data + (size_t) r * q->nb[1] + (size_t) h * q->nb[2]);
            float *       o  = (float *) ((char *) dst->data + (size_t) r * dst->nb[2] + (size_t) h * dst->nb[1]);
            const auto    t0 = timing ? clk::now() : clk::time_point();
            llama_shadowattn_to_u8(s, n, s8.data());
            const auto t1 = timing ? clk::now() : clk::time_point();
            GGML_ASSERT(llama_shadowattn_select(s8.data(), n, m, idx.data()) == m);
            const auto t2 = timing ? clk::now() : clk::time_point();
#ifdef LLAMA_SHADOWATTN_NEON
            if (D == 128) {
                __fp16 qh[128];
                for (int i = 0; i < 128; i += 4) {
                    vst1_f16(qh + i, vcvt_f16_f32(vld1q_f32(qr + i)));
                }
                llama_shadowattn_row(qh, K, V, k->nb[1], idx.data(), m, p->scale, sc.data(), o);
            } else
#endif
            {
                llama_shadowattn_row_ref(qr, K, V, k->nb[1], D, idx.data(), m, p->scale, sc.data(), o);
            }
            if (timing) {
                const auto t3 = clk::now();
                t_u8 += std::chrono::duration<double, std::micro>(t1 - t0).count();
                t_sel += std::chrono::duration<double, std::micro>(t2 - t1).count();
                t_att += std::chrono::duration<double, std::micro>(t3 - t2).count();
            }
            rows++;
            pairs += m;

            if (check > 0 && r % check == 0) {
                auto & st = llama_shadowattn_global_stats();
                // the selection: a stable descending sort of the same 8-bit scores
                order.resize(n);
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return s8[a] > s8[b]; });
                std::vector<int32_t> want(order.begin(), order.begin() + m), got(idx.begin(), idx.begin() + m);
                std::sort(want.begin(), want.end());
                std::sort(got.begin(), got.end());
                st.select_mismatch += want != got;
                // the output, in double over the selected keys with the query the kernel used (F16 on the NEON path)
                std::vector<double> e(m), acc(D, 0.0), qd(qr, qr + D);
#ifdef LLAMA_SHADOWATTN_NEON
                if (D == 128) {
                    for (auto & x : qd) {
                        x = (double) (__fp16) (float) x;
                    }
                }
#endif
                double mx = -INFINITY, tot = 0.0, omax = 0.0, err = 0.0;
                for (int t = 0; t < m; ++t) {
                    const ggml_fp16_t * kr = (const ggml_fp16_t *) (K + (size_t) want[t] * k->nb[1]);
                    double              d  = 0.0;
                    for (int i = 0; i < D; ++i) {
                        d += qd[i] * ggml_fp16_to_fp32(kr[i]);
                    }
                    e[t] = d * p->scale;
                    mx   = std::max(mx, e[t]);
                }
                for (int t = 0; t < m; ++t) {
                    tot += (e[t] = std::exp(e[t] - mx));
                }
                for (int t = 0; t < m; ++t) {
                    const ggml_fp16_t * vr = (const ggml_fp16_t *) (V + (size_t) want[t] * v->nb[1]);
                    for (int i = 0; i < D; ++i) {
                        acc[i] += e[t] / tot * ggml_fp16_to_fp32(vr[i]);
                    }
                }
                for (int i = 0; i < D; ++i) {
                    omax = std::max(omax, std::fabs(acc[i]));
                    err  = std::max(err, std::fabs(acc[i] - o[i]));
                }
                llama_shadowattn_atomic_max(st.out_err_ppm, (int) std::lround(1e6 * err / std::max(omax, 1e-30)));
                // the NPU scores of this row against Q.K in double
                double smax = 0.0, serr = 0.0, wexact = 0.0, wnpu = 0.0;
                int    wj = -1;
                std::vector<std::pair<int, double>> errs;
                for (int j = 0; j < n; j += std::max(1, n / 64)) {
                    const ggml_fp16_t * kr = (const ggml_fp16_t *) (K + (size_t) j * k->nb[1]);
                    double              d  = 0.0;
                    for (int i = 0; i < D; ++i) {
                        d += (double) qr[i] * ggml_fp16_to_fp32(kr[i]);
                    }
                    smax = std::max(smax, std::fabs(d));
                    errs.push_back({ j, std::fabs(d - s[j]) });
                    if (std::fabs(d - s[j]) > serr) {
                        serr   = std::fabs(d - s[j]);
                        wj     = j;
                        wexact = d;
                        wnpu   = s[j];
                    }
                }
                const int ppm = (int) std::lround(1e6 * serr / std::max(smax, 1e-30));
                st.score_bad_rows += ppm > 10000;
                for (const auto & e : errs) {
                    st.bad_keys[std::min(63, e.first / 256)][0] += e.second > 0.01 * smax;
                    st.bad_keys[std::min(63, e.first / 256)][1] += 1;
                }
                st.bad_rows_kv[std::min(63, n_kv / 256)][0] += ppm > 10000;
                st.bad_rows_kv[std::min(63, n_kv / 256)][1] += 1;
                if (ppm > st.score_err_ppm.load()) {
                    std::lock_guard<std::mutex> lock(st.worst_mutex);
                    if (ppm > st.score_err_ppm.load()) {
                        snprintf(st.worst, sizeof(st.worst), "il %d n_kv %d L %d head %d row %d key %d npu %.4g exact %.4g row max %.4g",
                                 p->il, n_kv, L, h, r, wj, wnpu, wexact, smax);
                    }
                }
                llama_shadowattn_atomic_max(st.score_err_ppm, ppm);
                st.checked++;
            }
        }
    }
    auto & st = llama_shadowattn_global_stats();
    st.rows += rows;
    st.pairs += pairs;
    int cpu = -1;
#if defined(__linux__)
    cpu = sched_getcpu();
    if (cpu >= 0 && cpu < 32) {
        st.cpus |= 1u << cpu;
    }
#endif
    if (timing) {
        // one write per line: the op's threads finish together and their lines must not interleave
        char      line[256];
        const int len = snprintf(line, sizeof(line),
                                 "shadowattn-op il %d ith %d L %d n_kv %d wall_us %.0f u8_us %.0f select_us %.0f attend_us %.0f rows %lld pairs %lld cpu %d\n",
                                 p->il, ith, L, n_kv, std::chrono::duration<double, std::micro>(clk::now() - t_op).count(), t_u8, t_sel, t_att,
                                 rows, pairs, cpu);
        GGML_ASSERT(len > 0 && len < (int) sizeof(line));
        fflush(stderr);
#if defined(__unix__) || defined(__APPLE__)
        GGML_ASSERT(write(2, line, len) == len);
#else
        fputs(line, stderr);
#endif
    }
}

// q [D, L, Hq] F32 (the permuted view), k/v [D, n_kv, Hkv] F16 cache views; returns [D, Hq, L] F32 as flash attention
static ggml_tensor * llama_shadowattn_build(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, float kq_scale,
                                            int il) {
    static llama_shadowattn_layer layers[1024];
    GGML_ASSERT(il >= 0 && il < 1024);
    GGML_ASSERT(q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
    GGML_ASSERT(q->ne[0] == k->ne[0] && v->ne[0] == k->ne[0] && q->ne[2] % k->ne[2] == 0 && k->ne[1] >= q->ne[1] &&
                q->ne[3] == 1 && k->ne[3] == 1);
    layers[il] = { llama_shadowattn_ratio(), kq_scale, il };

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);  // [n_kv, L, Hq]: the estimation, on the NPU
    ggml_format_name(kq, "shadow_kq-%d", il);
    ggml_tensor * args[] = { kq, q, k, v };
    ggml_tensor * out    = ggml_custom_4d(ctx, GGML_TYPE_F32, q->ne[0], q->ne[2], q->ne[1], 1, args, 4, llama_shadowattn_op,
                                          GGML_N_TASKS_MAX, &layers[il]);
    ggml_format_name(out, "shadow_attn-%d", il);
    return out;
}

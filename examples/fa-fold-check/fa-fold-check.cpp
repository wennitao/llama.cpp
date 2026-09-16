// Exactness and cost of the heterogeneous PREFILL fold (htp-ops.h, HTP_FA_FOLD_MAGIC).
//
// The HMX prefill kernel normalises O by an HMX multiply with diag(1/l). With a partial from
// another engine the correct diagonal is w_htp/S_total instead, so the HTP's own rescale is free
// and the only new work is one scaled add of the other engine's accumulator while the store
// thread de-tiles. This measures both halves of that claim:
//
//   dense  mask lets every key through, no fold            -> checks the harness itself
//   fold   mask hides [split, kv) from the kernel, and a
//          CPU-computed partial over [split, kv) is folded -> must equal the SAME reference
//
// Both arms are timed, so (fold - dense) at the same total key count is the fold's cost.
// A third arm folds an EMPTY partial, which must reproduce the dense result bit for bit.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "htp-ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <ctime>

static double now_ms() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

struct options {
    int d = 128, nh = 16, nkvh = 8, lq = 512, kv = 2048, split = -1, iters = 20;
    int probe = 0;   // HTP_FA_FOLD_F_PROBE_* bits added to a timing-only in-store run
    bool verbose = false;
};

static void usage() {
    printf("usage: llama-fa-fold-check [--d 128] [--nh 16] [--nkvh 8] [--lq 512] [--kv 2048] [--split N] [--iters 20] [-v]\n"
           "  --split  first key the fold owns (default kv/2); the kernel sees [0, split)\n");
}

struct layout {
    size_t q = 0, k = 0, v = 0, mask = 0, dst = 0, fold = 0, total = 0;
    size_t part_stride = 0, off_m = 0, off_l = 0, off_acc = 0, off_hm = 0, off_hl = 0, off_hacc = 0;
};

static ggml_tensor * place(ggml_backend_buffer_t buf, uint8_t * base, size_t off, ggml_tensor * t, const char * name) {
    t->buffer = buf;
    t->data   = base + off;
    ggml_set_name(t, name);
    ggml_backend_buffer_init_tensor(buf, t);
    return t;
}

static inline float h2f(uint16_t h) { return ggml_fp16_to_fp32(h); }
static inline uint16_t f2h(float f) { return ggml_fp32_to_fp16(f); }

int main(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](int & dst) { if (i + 1 < argc) dst = atoi(argv[++i]); };
        if      (a == "--d")     next(o.d);
        else if (a == "--nh")    next(o.nh);
        else if (a == "--nkvh")  next(o.nkvh);
        else if (a == "--lq")    next(o.lq);
        else if (a == "--kv")    next(o.kv);
        else if (a == "--split") next(o.split);
        else if (a == "--iters") next(o.iters);
        else if (a == "--probe") next(o.probe);
        else if (a == "-v")      o.verbose = true;
        else { usage(); return 1; }
    }
    if (o.split < 0) o.split = o.kv / 2;
    if (o.split % 64 != 0 || o.split <= 0 || o.split >= o.kv) { fprintf(stderr, "--split must be a multiple of 64 inside (0, kv)\n"); return 1; }
    if (o.nh % o.nkvh != 0) { fprintf(stderr, "nh must be a multiple of nkvh\n"); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    setenv("GGML_HEXAGON_ASYNC", "0", 1);

    const int    G     = o.nh / o.nkvh;
    const float  scale = 1.0f / sqrtf((float) o.d);
    const size_t D     = (size_t) o.d;

    layout L;
    L.part_stride = D * sizeof(float);   // acc row stride; m and l live in their own arrays
    auto bump = [](size_t & off, size_t bytes) { size_t at = off; off = (off + bytes + 4095) & ~(size_t) 4095; return at; };
    size_t off = 0;
    L.q    = bump(off, (size_t) o.lq * o.nh   * D * sizeof(float));
    L.k    = bump(off, (size_t) o.kv * o.nkvh * D * sizeof(uint16_t));
    L.v    = bump(off, (size_t) o.kv * o.nkvh * D * sizeof(uint16_t));
    L.mask = bump(off, (size_t) GGML_PAD(o.lq, 64) * o.kv * sizeof(uint16_t));
    L.dst  = bump(off, (size_t) o.lq * o.nh   * D * sizeof(float));
    const size_t rows    = (size_t) o.lq * o.nh;
    const size_t ml_bytes = (rows * sizeof(float) + 127) & ~(size_t) 127;
    L.off_m   = 128;
    L.off_l   = L.off_m + ml_bytes;
    L.off_acc = L.off_l + ml_bytes;
    // SPILL arm: the HTP's own partial, same shapes, after the other engine's acc
    L.off_hm   = L.off_acc + ((rows * L.part_stride + 127) & ~(size_t) 127);
    L.off_hl   = L.off_hm + ml_bytes;
    L.off_hacc = L.off_hl + ml_bytes;
    L.fold = bump(off, L.off_hacc + rows * L.part_stride);
    L.total = (off + (1u << 20) - 1) & ~((size_t) (1u << 20) - 1);

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("HTP0");
    if (!dev) { fprintf(stderr, "HTP0 device not found\n"); return 1; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) { fprintf(stderr, "failed to init HTP0\n"); return 1; }
    ggml_backend_buffer_type_t bt = ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(bt, L.total);
    if (!buf) { fprintf(stderr, "failed to allocate %zu bytes on HTP0\n", L.total); return 1; }
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
    memset(base, 0, L.total);

    // ---- inputs ----
    float *    q    = (float *)    (base + L.q);
    uint16_t * k    = (uint16_t *) (base + L.k);
    uint16_t * v    = (uint16_t *) (base + L.v);
    uint16_t * mask = (uint16_t *) (base + L.mask);
    float *    dst  = (float *)    (base + L.dst);
    uint8_t *  fold = base + L.fold;

    srand(1234);
    auto rnd = []() { return (float) ((rand() % 2001) - 1000) / 2000.0f; };
    for (size_t i = 0; i < (size_t) o.lq * o.nh * D; i++) q[i] = rnd();
    for (size_t i = 0; i < (size_t) o.kv * o.nkvh * D; i++) { k[i] = f2h(rnd()); v[i] = f2h(rnd()); }

    const int mask_rows = GGML_PAD(o.lq, 64);

    // ---- host reference over ALL keys, and the partial over [split, kv) ----
    std::vector<float> ref((size_t) o.lq * o.nh * D);
    std::vector<float> ref_head((size_t) o.lq * o.nh * D);   // attention over [0, split) only
    struct htp_fa_fold_hdr hdr = {};
    hdr.magic = HTP_FA_FOLD_MAGIC; hdr.rows = (uint32_t) rows; hdr.neq1 = (uint32_t) o.lq; hdr.dv = (uint32_t) D;
    hdr.off_m = (uint32_t) L.off_m; hdr.off_l = (uint32_t) L.off_l; hdr.off_acc = (uint32_t) L.off_acc;
    hdr.off_hm = (uint32_t) L.off_hm; hdr.off_hl = (uint32_t) L.off_hl; hdr.off_hacc = (uint32_t) L.off_hacc;
    memcpy(fold, &hdr, sizeof(hdr));
    auto set_flags = [&](uint32_t f) { ((struct htp_fa_fold_hdr *) fold)->flags = f; };
    float * fm = (float *) (fold + L.off_m);
    float * fl = (float *) (fold + L.off_l);

    std::vector<double> acc(D), s(o.kv);
    for (int h = 0; h < o.nh; h++) {
        const int kvh = h / G;
        for (int t = 0; t < o.lq; t++) {
            const float * qr = q + ((size_t) h * o.lq + t) * D;
            for (int j = 0; j < o.kv; j++) {
                const uint16_t * kr = k + ((size_t) kvh * o.kv + j) * D;
                double dot = 0;
                for (size_t e = 0; e < D; e++) dot += (double) qr[e] * h2f(kr[e]);
                s[j] = dot * scale;
            }
            // reference: softmax over every key
            double m_all = *std::max_element(s.begin(), s.end()), l_all = 0;
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int j = 0; j < o.kv; j++) {
                const double p = exp(s[j] - m_all);
                l_all += p;
                const uint16_t * vr = v + ((size_t) kvh * o.kv + j) * D;
                for (size_t e = 0; e < D; e++) acc[e] += p * h2f(vr[e]);
            }
            float * rr = ref.data() + ((size_t) t * o.nh + h) * D;
            for (size_t e = 0; e < D; e++) rr[e] = (float) (acc[e] / l_all);

            // reference for the masked-only arm: softmax over [0, split)
            double m_h0 = -INFINITY, l_h0 = 0;
            for (int j = 0; j < o.split; j++) m_h0 = std::max(m_h0, s[j]);
            std::vector<double> acc_h(D, 0.0);
            for (int j = 0; j < o.split; j++) {
                const double p = exp(s[j] - m_h0);
                l_h0 += p;
                const uint16_t * vr = v + ((size_t) kvh * o.kv + j) * D;
                for (size_t e = 0; e < D; e++) acc_h[e] += p * h2f(vr[e]);
            }
            float * rh = ref_head.data() + ((size_t) t * o.nh + h) * D;
            for (size_t e = 0; e < D; e++) rh[e] = (float) (acc_h[e] / l_h0);

            // partial the fold supplies: unnormalised (m, l, acc) over [split, kv)
            double m_p = -INFINITY, l_p = 0;
            for (int j = o.split; j < o.kv; j++) m_p = std::max(m_p, s[j]);
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int j = o.split; j < o.kv; j++) {
                const double p = exp(s[j] - m_p);
                l_p += p;
                const uint16_t * vr = v + ((size_t) kvh * o.kv + j) * D;
                for (size_t e = 0; e < D; e++) acc[e] += p * h2f(vr[e]);
            }
            if (o.verbose && t < 2 && h < 2) {
                double m_full = *std::max_element(s.begin(), s.end());
                printf("  [host t%d h%d] m_head %.6f (x log2e %.6f)  l_head %.6f | m_full %.6f (x log2e %.6f)\n",
                       t, h, m_h0, m_h0 * 1.44269504, l_h0, m_full, m_full * 1.44269504);
            }
            if (o.verbose && h == 0 && t == 0) {
                // what the device must produce for this row, from the same two halves
                const double M  = std::max(m_h0, m_p);
                const double wh = exp(m_h0 - M), wp = exp(m_p - M);
                const double SS = wh * l_h0 + wp * l_p;
                printf("  [dbg h0 t0] m_head %.6f l_head %.6f | m_part %.6f l_part %.6f\n", m_h0, l_h0, m_p, l_p);
                printf("  [dbg h0 t0] w_head %.6e w_part %.6e S %.6e\n", wh, wp, SS);
                for (int e = 0; e < 4; e++) {
                    const double want = (wh * acc_h[e] + wp * acc[e]) / SS;
                    printf("  [dbg h0 t0] e%d  head %.6f part %.6f  combined %.6f  ref %.6f\n",
                           e, acc_h[e] / l_h0, acc[e] / l_p, want, ref[((size_t) t * o.nh + h) * D + e]);
                }
            }
            // row index: (ib3 * neq2 + head) * neq1 + token, with ib3 = 0
            const size_t hrow = (size_t) h * o.lq + t;
            fm[hrow] = (float) m_p;
            fl[hrow] = (float) l_p;
            float * pa = (float *) (fold + L.off_acc + hrow * L.part_stride);
            for (size_t e = 0; e < D; e++) pa[e] = (float) acc[e];
        }
    }

    // ---- graph ----
    auto run = [&](bool hide_tail, bool with_fold, double * ms, uint32_t flags = 0) {
        set_flags(flags);
        for (int r = 0; r < mask_rows; r++) {
            for (int j = 0; j < o.kv; j++) {
                const bool hidden = hide_tail && j >= o.split;
                mask[(size_t) r * o.kv + j] = f2h(hidden ? -INFINITY : 0.0f);
            }
        }
        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, o.d, o.lq, o.nh, 1);
        ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, o.d, o.kv, o.nkvh, 1);
        ggml_tensor * tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, o.d, o.kv, o.nkvh, 1);
        ggml_tensor * tm = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, o.kv, mask_rows, 1, 1);
        place(buf, base, L.q, tq, "q"); place(buf, base, L.k, tk, "k");
        place(buf, base, L.v, tv, "v"); place(buf, base, L.mask, tm, "mask");
        ggml_tensor * out = ggml_flash_attn_ext(ctx, tq, tk, tv, tm, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        out->buffer = buf; out->data = base + L.dst; ggml_set_name(out, "dst");
        ggml_backend_buffer_init_tensor(buf, out);
        ggml_tensor * tf = nullptr;
        if (with_fold) {
            tf = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (L.total - L.fold) / 4);
            place(buf, base, L.fold, tf, "fold");
            out->src[7] = tf;
        }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
        double t0 = now_ms();
        for (int i = 0; i < o.iters; i++) ggml_backend_graph_compute(be, gf);
        *ms = (now_ms() - t0) / o.iters;
        ggml_free(ctx);
    };

    auto compare_to = [&](const char * name, const std::vector<float> & r) {
        double max_abs = 0, sum2 = 0, ref2 = 0;
        for (size_t i = 0; i < (size_t) o.lq * o.nh * D; i++) {
            const double d0 = (double) dst[i] - r[i];
            max_abs = std::max(max_abs, fabs(d0));
            sum2 += d0 * d0; ref2 += (double) r[i] * r[i];
        }
        printf("  %-14s max_abs %.3e  nmse %.3e\n", name, max_abs, sum2 / (ref2 > 0 ? ref2 : 1));
    };

    auto compare = [&](const char * name) {
        double max_abs = 0, sum2 = 0, ref2 = 0;
        for (size_t i = 0; i < (size_t) o.lq * o.nh * D; i++) {
            const double d0 = (double) dst[i] - ref[i];
            max_abs = std::max(max_abs, fabs(d0));
            sum2 += d0 * d0; ref2 += (double) ref[i] * ref[i];
        }
        printf("  %-14s max_abs %.3e  nmse %.3e\n", name, max_abs, sum2 / (ref2 > 0 ? ref2 : 1));
        return max_abs;
    };

    double ms_dense = 0, ms_fold = 0, ms_empty = 0;
    printf("shape d=%d nh=%d nkvh=%d lq=%d kv=%d split=%d (fold owns %d keys)\n",
           o.d, o.nh, o.nkvh, o.lq, o.kv, o.split, o.kv - o.split);

    memset(dst, 0, (size_t) o.lq * o.nh * D * sizeof(float));
    run(/*hide_tail=*/false, /*with_fold=*/false, &ms_dense);
    compare("dense");
    std::vector<float> dense_out(dst, dst + (size_t) o.lq * o.nh * D);

    // mask alone: the kernel must reproduce attention over [0, split) with no fold in play
    double ms_masked = 0;
    memset(dst, 0, (size_t) o.lq * o.nh * D * sizeof(float));
    run(/*hide_tail=*/true, /*with_fold=*/false, &ms_masked);
    compare_to("masked-only", ref_head);

    memset(dst, 0, (size_t) o.lq * o.nh * D * sizeof(float));
    run(/*hide_tail=*/true, /*with_fold=*/true, &ms_fold);
    compare("fold");

    // in-store fold: no diagonal, no norm pass; the store threads merge the raw accumulator
    double ms_instore = 0;
    memset(dst, 0, (size_t) o.lq * o.nh * D * sizeof(float));
    run(/*hide_tail=*/true, /*with_fold=*/true, &ms_instore, HTP_FA_FOLD_F_INSTORE);
    compare("fold-instore");

    // explicit merge: HTP spills its own partial, then a separate pass merges both from DDR
    double ms_spill = 0;
    memset(dst, 0, (size_t) o.lq * o.nh * D * sizeof(float));
    run(/*hide_tail=*/true, /*with_fold=*/true, &ms_spill, HTP_FA_FOLD_F_SPILL);
    compare("spill-merge");

    // empty partial: m = HTP_FA_M_INITIAL_VAL, l = 0, acc = 0 -> must reproduce dense exactly
    for (size_t hrow = 0; hrow < rows; hrow++) {
        ((float *) (fold + L.off_m))[hrow] = -10000.0f;
        ((float *) (fold + L.off_l))[hrow] = 0.0f;
        memset(fold + L.off_acc + hrow * L.part_stride, 0, D * sizeof(float));
    }
    memset(dst, 0, (size_t) o.lq * o.nh * D * sizeof(float));
    run(/*hide_tail=*/false, /*with_fold=*/true, &ms_empty);
    double empty_diff = 0;
    for (size_t i = 0; i < dense_out.size(); i++) empty_diff = std::max(empty_diff, (double) fabsf(dst[i] - dense_out[i]));
    printf("  %-14s max_abs vs dense %.3e (must be 0)\n", "empty-fold", empty_diff);

    printf("time  dense %.3f ms   fold %.3f ms (%+.1f%%)   spill-merge %.3f ms (%+.1f%%)   empty-fold %.3f ms (%+.1f%%)\n",
           ms_dense, ms_fold, 100.0 * (ms_fold / ms_dense - 1.0), ms_spill, 100.0 * (ms_spill / ms_dense - 1.0),
           ms_empty, 100.0 * (ms_empty / ms_dense - 1.0));
    printf("merge cost  fold %+.0f us   fold-instore %+.0f us   explicit-on-NPU %+.0f us   (over dense, same shape)\n",
           1e3 * (ms_fold - ms_dense), 1e3 * (ms_instore - ms_dense), 1e3 * (ms_spill - ms_dense));

    // Interleaved repeats: the arms alternate so thermal drift hits them equally, and the min is
    // reported. spill-only skips the merge pass (dst unwritten), so the difference to spill-merge
    // is the merge pass itself.
    double t_d = 1e9, t_f = 1e9, t_fi = 1e9, t_s = 1e9, t_so = 1e9, ms = 0;
    for (int rep = 0; rep < 3; rep++) {
        run(false, false, &ms);                                                 t_d  = std::min(t_d,  ms);
        run(true,  true,  &ms);                                                 t_f  = std::min(t_f,  ms);
        run(true,  true,  &ms, HTP_FA_FOLD_F_INSTORE);                          t_fi = std::min(t_fi, ms);
        run(true,  true,  &ms, HTP_FA_FOLD_F_SPILL);                            t_s  = std::min(t_s,  ms);
        run(true,  true,  &ms, HTP_FA_FOLD_F_SPILL | HTP_FA_FOLD_F_NOMERGE);    t_so = std::min(t_so, ms);
    }
    printf("min-of-3 interleaved  dense %.3f ms   fold %.3f (%+.0f us)   fold-instore %.3f (%+.0f us)   spill+merge %.3f (%+.0f us)   spill-only %.3f (%+.0f us)   => merge pass %.0f us\n",
           t_d, t_f, 1e3 * (t_f - t_d), t_fi, 1e3 * (t_fi - t_d), t_s, 1e3 * (t_s - t_d), t_so, 1e3 * (t_so - t_d), 1e3 * (t_s - t_so));
    if (o.probe) {
        // timing only (NOWEIGHTS / NOACC give wrong results by design), interleaved with the plain in-store run
        double t_p = 1e9, t_r = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            run(true, true, &ms, HTP_FA_FOLD_F_INSTORE);                                 t_r = std::min(t_r, ms);
            run(true, true, &ms, HTP_FA_FOLD_F_INSTORE | (uint32_t) o.probe);           t_p = std::min(t_p, ms);
        }
        printf("probe 0x%x   in-store %.3f ms (%+.0f us)   probed %.3f ms (%+.0f us)   delta %+.0f us\n",
               o.probe, t_r, 1e3 * (t_r - t_d), t_p, 1e3 * (t_p - t_d), 1e3 * (t_p - t_r));
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    return 0;
}

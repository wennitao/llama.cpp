#pragma once

// Existing training-free sparse-prefill selectors as NPU graphs (study: heterogeneous-accuracy-20260917/
// selectors-npu-20261002). Each turns one ubatch's Q/K into a 0/1 block membership that llama-block-sparse.h runs;
// the references are docs/papers/heterogeneous-inference/accuracy/existing_selectors.py, which the CUDA accuracy runs
// use. A ubatch is one chunk of the accuracy protocol: aligned, on a cleared cache, keys 0..n_kv-1 with the ubatch at
// the end (llm_graph_context::build_attn_mha checks it), so "the last queries" of a method's probe are the ubatch's.
//
//   LLAMA_SELECTOR=minference_vs    MInference vertical-slash, adaptive budget = LLAMA_SELECTOR_PARAM * keys (20% columns,
//                                   80% diagonals), run as the smallest 64x64-block cover of its token pattern
//   LLAMA_SELECTOR=flexprefill      FlexPrefill, gamma = LLAMA_SELECTOR_PARAM, tau 0.1, 128-token blocks (64-key lists
//                                   per 128-query block)
//   LLAMA_SELECTOR=minference_bs    MInference block-sparse, top-k = LLAMA_SELECTOR_PARAM 64-blocks per 64-query row
//   LLAMA_SELECTOR=spargeattn       SpargeAttn TopCdf, tau = LLAMA_SELECTOR_PARAM, theta 0.1, 128-query x 64-key blocks
//   LLAMA_SELECTOR=flashprefill     FlashPrefill, alpha = LLAMA_SELECTOR_PARAM, 2 sink + 4 window blocks of 128
//   LLAMA_SELECTOR=sampleattn       SampleAttention v3 (chunk_n 1), alpha = LLAMA_SELECTOR_PARAM, 128-token blocks
//   LLAMA_SELECTOR_GQA=head|union   one list per query head (two GQA lanes, default) or their OR per K/V head
//   LLAMA_SELECTOR_SELECT=cpu|npu   the selection rule as a CPU custom op (the reference, default) or on the DSP
//
// The scoring (probe products, softmax, column and diagonal sums, pooled products) uses standard ops; each method's
// selection rule -- top-k / cumulative cover, its forced entries, the pattern decision and the expansion to blocks --
// is one custom op, as XAttention's cumulative rule is (llama-xattention.h). Its op_params: [6] LLAMA_SEL_MAGIC when
// routed to the DSP, [7] mode, [8..11] the mode's parameters.

#include "ggml.h"
#include "llama-block-sparse.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define LLAMA_SEL_MAGIC 0x534c4354

enum llama_sel_mode {
    LLAMA_SEL_MODE_VS     = 1,
    LLAMA_SEL_MODE_FLEX   = 2,
    LLAMA_SEL_MODE_BS     = 3,
    LLAMA_SEL_MODE_SPARGE = 4,
    LLAMA_SEL_MODE_FLASH  = 5,
    LLAMA_SEL_MODE_SAMPLE = 6,
};

static const char * llama_selector_name() {
    const char * s = getenv("LLAMA_SELECTOR");
    if (!s || !*s) { return nullptr; }
    GGML_ASSERT(strcmp(s, "minference_vs") == 0 || strcmp(s, "flexprefill") == 0 || strcmp(s, "minference_bs") == 0 ||
                strcmp(s, "spargeattn") == 0 || strcmp(s, "flashprefill") == 0 || strcmp(s, "sampleattn") == 0);
    return s;
}

// double, as the Python reference's arithmetic (int(keys * ratio) must round the same way)
static double llama_selector_param() {
    const char * s = getenv("LLAMA_SELECTOR_PARAM");
    GGML_ASSERT(s && "LLAMA_SELECTOR needs LLAMA_SELECTOR_PARAM");
    char * end = nullptr;
    const double v = strtod(s, &end);
    GGML_ASSERT(end != s && *end == '\0' && v > 0.0f);
    return v;
}

static bool llama_selector_union() {
    const char * s = getenv("LLAMA_SELECTOR_GQA");
    GGML_ASSERT(!s || strcmp(s, "head") == 0 || strcmp(s, "union") == 0);
    return s && strcmp(s, "union") == 0;
}

static bool llama_selector_npu() {
    const char * s = getenv("LLAMA_SELECTOR_SELECT");
    GGML_ASSERT(!s || strcmp(s, "cpu") == 0 || strcmp(s, "npu") == 0);
    return s && strcmp(s, "npu") == 0;
}

// ---- selection rules (CPU reference; the DSP runs the same in htp/argsort-ops.c) ----

// the k largest of x[0..n) by a stable descending sort (ties: lower index first), as a 0/1 mask; -inf is never kept
static void llama_sel_topk(const float * x, int64_t n, int64_t k, std::vector<int64_t> & order, uint8_t * keep) {
    order.resize(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return x[a] > x[b]; });
    std::fill(keep, keep + n, 0);
    for (int64_t r = 0; r < std::min(k, n); ++r) {
        if (x[order[r]] != -INFINITY) { keep[order[r]] = 1; }
    }
}

// FlexPrefill score_cover_topk: entries (descending) whose cumulative sum stays <= score, plus one
static int64_t llama_sel_cover_count(const float * x, int64_t n, double score, std::vector<float> & sorted) {
    sorted.assign(x, x + n);
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    double cumulative = 0.0;
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        cumulative += sorted[i];
        if (cumulative <= score) { ++count; } else { break; }
    }
    return count + 1;
}

// MInference vertical-slash on one ubatch -> the 64-block cover of its token pattern, causal.
//   src0 vertical column sums [nk, 1, hq]; src1 diagonal sums indexed t = nk - 1 - offset [nk, 1, hq]
//   params [8] vertical_size, [9] slash_size; dst [nbk, nbq, hq]
// Columns 0..29 and offsets 0..99 are forced (+inf, counted in the budgets, as the reference's torch.topk). A selected
// column j keeps block j/64 for every query block at or after it; a selected offset o keeps block (A, b) when
// o in [64(A-b) - 63, 64(A-b) + 63].
static void llama_sel_vs(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * vertical = dst->src[0];
    const ggml_tensor * slash = dst->src[1];
    const int64_t nk = vertical->ne[0], hq = vertical->ne[2];
    const int64_t nbk = dst->ne[0], nbq = dst->ne[1];
    const int64_t vertical_size = dst->op_params[8], slash_size = dst->op_params[9];
    GGML_ASSERT(nk == nbk*64 && dst->ne[2] == hq);
    std::vector<float> v(nk), s(nk);
    std::vector<int64_t> order;
    std::vector<uint8_t> vk(nk), sk(nk);
    std::vector<uint8_t> vb(nbk), g(nbk), g2(nbk);
    for (int64_t h = ith; h < hq; h += nth) {
        memcpy(v.data(), (const char *) vertical->data + h*vertical->nb[2], nk*sizeof(float));
        memcpy(s.data(), (const char *) slash->data + h*slash->nb[2], nk*sizeof(float));
        for (int64_t j = 0; j < std::min<int64_t>(30, nk); ++j) { v[j] = INFINITY; }
        for (int64_t t = std::max<int64_t>(0, nk - 100); t < nk; ++t) { s[t] = INFINITY; }
        llama_sel_topk(v.data(), nk, vertical_size, order, vk.data());
        llama_sel_topk(s.data(), nk, slash_size, order, sk.data());
        for (int64_t b = 0; b < nbk; ++b) {
            vb[b] = g[b] = g2[b] = 0;
            for (int64_t i = 0; i < 64; ++i) {
                vb[b] |= vk[b*64 + i];
                g[b]  |= sk[b*64 + i];
                g2[b] |= i < 63 ? sk[b*64 + i] : 0;
            }
        }
        for (int64_t a = 0; a < nbq; ++a) {
            const int64_t A = nbk - nbq + a;
            float * out = (float *) ((char *) dst->data + a*dst->nb[1] + h*dst->nb[2]);
            for (int64_t b = 0; b < nbk; ++b) {
                const int64_t dd = A - b;
                // offsets [64dd-63, 64dd+63] are t in [64(nbk-dd-1), 64(nbk-dd+1)-2]: all of group nbk-dd-1 and the
                // first 63 of group nbk-dd
                const bool slash_hit = dd >= 0 && (g[nbk - dd - 1] || (nbk - dd < nbk && g2[nbk - dd]));
                out[b] = dd >= 0 && (vb[b] || slash_hit) ? 1.0f : 0.0f;
            }
        }
    }
}

// FlexPrefill get_active_blocks on one ubatch at 128-token blocks -> [nb, nq, hq], causal.
//   src0 vertical (probe mean) [nk, 1, hq]; src1 diagonal sums / probe size, t = nk - 1 - offset [nk, 1, hq];
//   src2 pooled-probe x pooled-key logits [nb, 1, hq]; src3 pooled-query x pooled-key logits [nb, nq, hq]
//   params [8] gamma (f32), [9] tau (f32), [10] min budget, [11] max budget (blocks)
static void llama_sel_flex(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * vertical = dst->src[0];
    const ggml_tensor * slash = dst->src[1];
    const ggml_tensor * probe = dst->src[2];
    const ggml_tensor * pooled = dst->src[3];
    const int64_t nk = vertical->ne[0], hq = vertical->ne[2], nb = dst->ne[0], nq = dst->ne[1];
    const int64_t block = nk/nb;
    float gamma, tau;
    memcpy(&gamma, &dst->op_params[8], sizeof(float));
    memcpy(&tau, &dst->op_params[9], sizeof(float));
    const int64_t low = dst->op_params[10], high = std::min<int64_t>(dst->op_params[11], nb);
    GGML_ASSERT(nk == nb*block && pooled->ne[0] == nb && pooled->ne[1] == nq);
    std::vector<float> v(nk), s(nk), vbk(nb), sbk(nb), sorted, p(nb), flat(nq*nb);
    std::vector<int64_t> order;
    std::vector<uint8_t> vsel(nb), ssel(nb);
    for (int64_t h = ith; h < hq; h += nth) {
        memcpy(v.data(), (const char *) vertical->data + h*vertical->nb[2], nk*sizeof(float));
        memcpy(s.data(), (const char *) slash->data + h*slash->nb[2], nk*sizeof(float));
        int64_t num_v = std::clamp<int64_t>(llama_sel_cover_count(v.data(), nk, gamma, sorted)/128 + 1, low, high);
        int64_t num_s = std::clamp<int64_t>(llama_sel_cover_count(s.data(), nk, gamma, sorted)/128 + 1, low, high);
        std::fill(vbk.begin(), vbk.end(), 0.0f);
        std::fill(sbk.begin(), sbk.end(), 0.0f);
        for (int64_t j = 0; j < nk; ++j) {
            vbk[j/block] += v[j];
            sbk[(nk - 1 - j)/block] += s[j];      // s is t-indexed: offset nk-1-t, block offset (nk-1-t)/block
        }
        // pattern decision: sqrt JS divergence between softmax(pooled probe x pooled keys) and the probe's block mass
        const float * pl = (const float *) ((const char *) probe->data + h*probe->nb[2]);
        const float mx = *std::max_element(pl, pl + nb);
        double z = 0.0;
        for (int64_t c = 0; c < nb; ++c) { p[c] = expf(pl[c] - mx); z += p[c]; }
        double js = 0.0;
        for (int64_t c = 0; c < nb; ++c) {
            const double pc = p[c]/z, qc = vbk[c], m = (pc + qc)/2;
            js += .5*pc*log(pc/m) + .5*qc*log(qc/m);
        }
        const bool query_aware = sqrt(js) < tau;
        if (query_aware) { num_v = num_s = low; }
        vbk[0] = INFINITY;
        sbk[0] = INFINITY;
        llama_sel_topk(vbk.data(), nb, num_v, order, vsel.data());
        llama_sel_topk(sbk.data(), nb, num_s, order, ssel.data());
        std::vector<uint8_t> extra(nq*nb, 0);
        if (query_aware) {
            // causal softmax of every query block's pooled logits, then FlexPrefill's score_cover_idx over the
            // flattened map: entries (descending) whose cumulative sum stays <= gamma * nq
            for (int64_t a = 0; a < nq; ++a) {
                const int64_t A = nb - nq + a;
                const float * row = (const float *) ((const char *) pooled->data + a*pooled->nb[1] + h*pooled->nb[2]);
                float m2 = -INFINITY;
                for (int64_t c = 0; c <= A; ++c) { m2 = std::max(m2, row[c]); }
                double zz = 0.0;
                for (int64_t c = 0; c <= A; ++c) { zz += exp(row[c] - m2); }
                for (int64_t c = 0; c < nb; ++c) { flat[a*nb + c] = c <= A ? (float) (exp(row[c] - m2)/zz) : 0.0f; }
            }
            order.resize(nq*nb);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int64_t x, int64_t y) { return flat[x] > flat[y]; });
            double cumulative = 0.0;
            for (int64_t r = 0; r < nq*nb; ++r) {
                cumulative += flat[order[r]];
                if (cumulative > (double) gamma*nq) { break; }
                extra[order[r]] = 1;
            }
        }
        for (int64_t a = 0; a < nq; ++a) {
            const int64_t A = nb - nq + a;
            float * out = (float *) ((char *) dst->data + a*dst->nb[1] + h*dst->nb[2]);
            for (int64_t c = 0; c < nb; ++c) {
                out[c] = c <= A && (vsel[c] || ssel[A - c] || extra[a*nb + c]) ? 1.0f : 0.0f;
            }
        }
    }
}

// MInference block-sparse: top-k of each 64-query row's raw pooled dot products over the visible blocks.
//   src0 [nbk, nbq, hq]; params [8] top-k; dst [nbk, nbq, hq]
static void llama_sel_bs(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * p = dst->src[0];
    const int64_t nbk = dst->ne[0], nbq = dst->ne[1], hq = dst->ne[2], topk = dst->op_params[8];
    std::vector<float> row(nbk);
    std::vector<int64_t> order;
    std::vector<uint8_t> keep(nbk);
    for (int64_t r = ith; r < nbq*hq; r += nth) {
        const int64_t a = r % nbq, h = r / nbq, A = nbk - nbq + a;
        memcpy(row.data(), (const char *) p->data + a*p->nb[1] + h*p->nb[2], nbk*sizeof(float));
        for (int64_t b = A + 1; b < nbk; ++b) { row[b] = -INFINITY; }
        llama_sel_topk(row.data(), nbk, topk, order, keep.data());
        float * out = (float *) ((char *) dst->data + a*dst->nb[1] + h*dst->nb[2]);
        for (int64_t b = 0; b < nbk; ++b) { out[b] = keep[b] && b <= A ? 1.0f : 0.0f; }
    }
}

// SpargeAttn TopCdf on 128-query rows over 64-key blocks.
//   src0 pooled scores / sqrt(d) [nbk, nq, hq]; src1 query-block self-similarity [1, hq, nq]; src2 key-block
//   self-similarity after smoothing [1, nbk, hkv] (both |mean unit vector|^2); params [8] tau, [9] theta (f32)
// A row keeps the longest descending prefix whose cumulative softmax (over self-similar, causal key blocks) stays
// <= tau, at least one block; non-self-similar key columns and query rows are kept whole; causal: key block <
// (A + 1) * 2 for 128-query block A.
static void llama_sel_sparge(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * s = dst->src[0];
    const ggml_tensor * simq = dst->src[1];
    const ggml_tensor * simk = dst->src[2];
    const int64_t nbk = dst->ne[0], nq = dst->ne[1], hq = dst->ne[2], hkv = simk->ne[2], groups = hq/hkv;
    float tau, theta;
    memcpy(&tau, &dst->op_params[8], sizeof(float));
    memcpy(&theta, &dst->op_params[9], sizeof(float));
    std::vector<float> row(nbk);
    std::vector<int64_t> order(nbk);
    for (int64_t r = ith; r < nq*hq; r += nth) {
        const int64_t a = r % nq, h = r / nq, A = nbk/2 - nq + a;
        const int64_t visible = std::min<int64_t>(nbk, (A + 1)*2);
        const float * sk = (const float *) ((const char *) simk->data + (h/groups)*simk->nb[2]);
        const bool q_similar = *(const float *) ((const char *) simq->data + h*simq->nb[1] + a*simq->nb[2]) > theta;
        const float * src = (const float *) ((const char *) s->data + a*s->nb[1] + h*s->nb[2]);
        float mx = -INFINITY;
        for (int64_t b = 0; b < nbk; ++b) {
            row[b] = b < visible && sk[b] > theta ? src[b] : -INFINITY;
            mx = std::max(mx, row[b]);
        }
        double z = 0.0;
        for (int64_t b = 0; b < nbk; ++b) { row[b] = row[b] == -INFINITY ? 0.0f : expf(row[b] - mx); z += row[b]; }
        for (int64_t b = 0; b < nbk; ++b) { row[b] = z > 0 ? (float) (row[b]/z) : 0.0f; }
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int64_t x, int64_t y) { return row[x] > row[y]; });
        double cumulative = 0.0;
        int64_t count = 0;
        for (int64_t i = 0; i < nbk; ++i) {
            cumulative += row[order[i]];
            if (cumulative <= tau) { ++count; } else { break; }
        }
        count = std::max<int64_t>(count, 1);
        float * out = (float *) ((char *) dst->data + a*dst->nb[1] + h*dst->nb[2]);
        for (int64_t b = 0; b < nbk; ++b) { out[b] = 0.0f; }
        for (int64_t i = 0; i < count; ++i) { out[order[i]] = 1.0f; }
        for (int64_t b = 0; b < nbk; ++b) {
            if (!q_similar || !(sk[b] > theta)) { out[b] = 1.0f; }
            if (b >= visible) { out[b] = 0.0f; }
        }
        // a query block whose first 64 queries would see no key keeps the key block at its start (SpargeAttn's kernel
        // masks causally with a finite -5e6 and would average the block's future keys there)
        bool seen = false;
        for (int64_t b = 0; b <= 2*A; ++b) { seen = seen || out[b] > 0.5f; }
        if (!seen) { out[2*A] = 1.0f; }
    }
}

// FlashPrefill deal_output_score on 128-blocks: score >= alpha * the row max, the first `sink` blocks, the `window`
// blocks ending at the diagonal, causal.  src0 normalized scores [nb, nq, hq]; params [8] alpha (f32), [9] sink,
// [10] window; dst [nb, nq, hq]
static void llama_sel_flash(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * s = dst->src[0];
    const int64_t nb = dst->ne[0], nq = dst->ne[1], hq = dst->ne[2];
    float alpha;
    memcpy(&alpha, &dst->op_params[8], sizeof(float));
    const int64_t sink = dst->op_params[9], window = dst->op_params[10];
    for (int64_t r = ith; r < nq*hq; r += nth) {
        const int64_t a = r % nq, h = r / nq, A = nb - nq + a;
        const float * src = (const float *) ((const char *) s->data + a*s->nb[1] + h*s->nb[2]);
        float mx = -INFINITY;
        for (int64_t c = 0; c < nb; ++c) { mx = std::max(mx, src[c]); }
        float * out = (float *) ((char *) dst->data + a*dst->nb[1] + h*dst->nb[2]);
        for (int64_t c = 0; c < nb; ++c) {
            const int64_t dist = A - c;
            out[c] = dist >= 0 && (src[c] >= alpha*mx || c < sink || dist < window) ? 1.0f : 0.0f;
        }
    }
}

// SampleAttention v3 on 128-blocks: per head the fewest top column blocks (and slash blocks) whose normalized
// cumulative score reaches alpha.  src0 column sums of the probe [nk, 1, hq]; src1 diagonal sums, t = nk-1-offset
// [nk, 1, hq]; params [8] alpha (f32); dst [nb, nq, hq]
static void llama_sel_sample(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * vertical = dst->src[0];
    const ggml_tensor * slash = dst->src[1];
    const int64_t nk = vertical->ne[0], nb = dst->ne[0], nq = dst->ne[1], hq = dst->ne[2], block = nk/nb;
    float alpha;
    memcpy(&alpha, &dst->op_params[8], sizeof(float));
    std::vector<float> cb(nb), sb(nb);
    std::vector<int64_t> order;
    std::vector<uint8_t> csel(nb), ssel(nb);
    auto cover = [&](std::vector<float> & x, uint8_t * keep) {
        double total = 0.0;
        for (float v : x) { total += v; }
        for (float & v : x) { v = (float) (v/total); }
        order.resize(nb);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int64_t i, int64_t j) { return x[i] > x[j]; });
        double cumulative = 0.0;
        int64_t count = 0;
        for (int64_t i = 0; i < nb; ++i) {
            cumulative += x[order[i]];
            if (cumulative < alpha) { ++count; } else { break; }
        }
        count = std::min<int64_t>(count + 1, nb);
        std::fill(keep, keep + nb, 0);
        for (int64_t i = 0; i < count; ++i) { keep[order[i]] = 1; }
    };
    for (int64_t h = ith; h < hq; h += nth) {
        const float * v = (const float *) ((const char *) vertical->data + h*vertical->nb[2]);
        const float * sl = (const float *) ((const char *) slash->data + h*slash->nb[2]);
        std::fill(cb.begin(), cb.end(), 0.0f);
        std::fill(sb.begin(), sb.end(), 0.0f);
        for (int64_t j = 0; j < nk; ++j) {
            cb[j/block] += v[j];
            sb[(nk - 1 - j)/block] += sl[j];
        }
        cover(cb, csel.data());
        cover(sb, ssel.data());
        for (int64_t a = 0; a < nq; ++a) {
            const int64_t A = nb - nq + a;
            float * out = (float *) ((char *) dst->data + a*dst->nb[1] + h*dst->nb[2]);
            bool seen = false;
            for (int64_t c = 0; c < nb; ++c) {
                out[c] = c <= A && (csel[c] || ssel[A - c]) ? 1.0f : 0.0f;
                seen = seen || out[c] > 0.5f;
            }
            if (!seen) { out[A] = 1.0f; }    // a query block left with no block keeps its diagonal
        }
    }
}

// ---- scoring graphs ----

// sum over ne1 (a power of two) by halving adds of views -> [ne0, 1, ne2]
static ggml_tensor * llama_sel_sum1(ggml_context * ctx, ggml_tensor * a) {
    for (int64_t n = a->ne[1]; n > 1; n /= 2) {
        ggml_tensor * lo = ggml_view_3d(ctx, a, a->ne[0], n/2, a->ne[2], a->nb[1], a->nb[2], 0);
        ggml_tensor * hi = ggml_view_3d(ctx, a, a->ne[0], n/2, a->ne[2], a->nb[1], a->nb[2], (size_t)(n/2)*a->nb[1]);
        a = ggml_add(ctx, lo, hi);
    }
    return a;
}

// the last n queries' softmax over all keys, [nk, n, hq] (q [d, lq, hq], k [d, nk, hkv] F16 views, mask [nk, lq])
static ggml_tensor * llama_sel_probe(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * mask, int64_t n, float scale) {
    const int64_t d = q->ne[0], lq = q->ne[1], hq = q->ne[2], nk = k->ne[1];
    ggml_tensor * qp = ggml_view_3d(ctx, q, d, n, hq, q->nb[1], q->nb[2], (size_t)(lq - n)*q->nb[1]);
    ggml_tensor * s = ggml_mul_mat(ctx, k, qp);
    ggml_mul_mat_set_prec(s, GGML_PREC_F32);
    ggml_tensor * mp = ggml_view_2d(ctx, mask, nk, n, mask->nb[1], (size_t)(lq - n)*mask->nb[1]);
    return ggml_soft_max_ext(ctx, s, mp, scale, 0.0f);
}

// diagonal sums of probe probabilities a [nk, n, hq] (query i at position nk - n + i), indexed t = nk - 1 - offset:
// with n zero columns in front, row i of a skewed view (row stride W + 1, W = n + nk) reads exactly the keys at offset
// nk - 1 - t from query i, so the halving sum over rows is the diagonal sum
static ggml_tensor * llama_sel_diagonals(ggml_context * ctx, ggml_tensor * a) {
    const int64_t nk = a->ne[0], n = a->ne[1], hq = a->ne[2], w = n + nk;
    GGML_ASSERT(ggml_is_contiguous(a));
    ggml_tensor * zeros = ggml_fill(ctx, ggml_reshape_3d(ctx, ggml_view_1d(ctx, a, n*n*hq, 0), n, n, hq), 0.0f);
    ggml_tensor * padded = ggml_concat(ctx, zeros, a, 0);          // [w, n, hq]
    ggml_tensor * skew = ggml_view_3d(ctx, padded, nk, n, hq, (size_t)(w + 1)*sizeof(float), (size_t)n*w*sizeof(float), sizeof(float));
    return llama_sel_sum1(ctx, skew);
}

static ggml_tensor * llama_sel_rule(ggml_context * ctx, enum llama_sel_mode mode, ggml_custom_op_t fun, ggml_tensor ** args, int n_args,
                                    int64_t ne0, int64_t ne1, int64_t ne2, const int32_t * params, int layer, const char * name) {
    ggml_tensor * m = ggml_custom_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, 1, args, n_args, fun, GGML_N_TASKS_MAX, nullptr);
    m->op_params[6] = llama_selector_npu() ? LLAMA_SEL_MAGIC : 0;
    m->op_params[7] = mode;
    memcpy(&m->op_params[8], params, 4*sizeof(int32_t));
    ggml_format_name(m, "%s-%d", name, layer);
    return m;
}

static ggml_tensor * llama_selector_minference_vs(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * mask, double ratio, int layer) {
    const int64_t lq = q->ne[1], hq = q->ne[2], nk = k->ne[1];
    const int64_t n = std::min<int64_t>(64, lq);
    // the reference scales by 1/sqrt(128) (the head dimension of its models; Qwen3's too)
    ggml_tensor * a = llama_sel_probe(ctx, q, k, mask, n, 1.0f/sqrtf(128.0f));
    ggml_tensor * vertical = llama_sel_sum1(ctx, a);
    ggml_tensor * slash = llama_sel_diagonals(ctx, a);
    ggml_format_name(vertical, "sel_vertical-%d", layer);
    ggml_format_name(slash, "sel_slash-%d", layer);
    const int64_t budget = (int64_t) ((double) nk*ratio);
    const int32_t params[4] = {(int32_t) std::min<int64_t>(nk, std::max<int64_t>((int64_t) (budget*.2), 30)),
                               (int32_t) std::min<int64_t>(nk, std::max<int64_t>((int64_t) (budget*.8), 50)), 0, 0};
    ggml_tensor * args[] = {vertical, slash};
    return llama_sel_rule(ctx, LLAMA_SEL_MODE_VS, llama_sel_vs, args, 2, nk/64, lq/64, hq, params, layer, "sel_membership");
}

// Block means pool WHOLE TOKEN ROWS: q [d, lq, hq] and k [d, nk, hkv] are permuted views whose token rows hold every
// head, and a per-head [d, tokens, blocks, heads] view is permuted -- a backend that rejects permuted operands (the
// HTP's binary ops) runs every halving add of it on the CPU. Halving [d*heads, tokens, blocks] row views instead keeps
// every add on the accelerator, with each element summed by the same pairwise tree; only the small result is permuted.

// block means of the queries, [d, lq/bs, hq]
static ggml_tensor * llama_sel_pool_q(ggml_context * ctx, ggml_tensor * q, int64_t bs) {
    const int64_t d = q->ne[0], lq = q->ne[1], hq = q->ne[2], n = lq/bs;
    GGML_ASSERT(q->nb[2] == (size_t) d*ggml_element_size(q) && q->nb[1] == (size_t) hq*q->nb[2]);
    ggml_tensor * x = llama_sel_sum1(ctx, ggml_view_3d(ctx, q, d*hq, bs, n, q->nb[1], bs*q->nb[1], 0));   // [d*hq, 1, n]
    x = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, x, d, hq, n), 0, 2, 1, 3));                // [d, n, hq]
    return ggml_scale(ctx, x, 1.0f/bs);
}

// block means of the F16 keys, [d, nk/bs, hkv]: F16 halvings down to sums of 16 keys (far below the F16 range), then F32
static ggml_tensor * llama_sel_pool_k(ggml_context * ctx, ggml_tensor * k, int64_t bs) {
    const int64_t d = k->ne[0], nk = k->ne[1], hkv = k->ne[2], n = nk/bs;
    GGML_ASSERT(k->nb[2] == (size_t) d*ggml_element_size(k) && k->nb[1] == (size_t) hkv*k->nb[2]);
    ggml_tensor * x = ggml_view_3d(ctx, k, d*hkv, bs, n, k->nb[1], bs*k->nb[1], 0);
    int64_t m = bs;
    for (; m > std::max<int64_t>(bs/16, 1); m /= 2) {
        ggml_tensor * lo = ggml_view_3d(ctx, x, d*hkv, m/2, n, x->nb[1], x->nb[2], 0);
        ggml_tensor * hi = ggml_view_3d(ctx, x, d*hkv, m/2, n, x->nb[1], x->nb[2], (size_t)(m/2)*x->nb[1]);
        x = ggml_add(ctx, lo, hi);
    }
    x = llama_sel_sum1(ctx, ggml_cast(ctx, x, GGML_TYPE_F32));                                            // [d*hkv, 1, n]
    x = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, x, d, hkv, n), 0, 2, 1, 3));               // [d, n, hkv]
    return ggml_scale(ctx, x, 1.0f/bs);
}

// pooled keys [d, n, h] as the F16 operand of a product with every query: the HTP's matrix unit (HMX) takes F16 rows
// in 32-row tiles, anything else runs on HVX at ~1/38 the rate per MAC, so the rows are zero-padded to a multiple of 32
// (the caller drops the padded scores); the official selectors compute these products from fp16/bf16 tensors too
static ggml_tensor * llama_sel_keys_f16(ggml_context * ctx, ggml_tensor * k) {
    const int64_t n = k->ne[1], np = GGML_PAD(n, 32);
    return ggml_cast(ctx, np == n ? k : ggml_pad(ctx, k, 0, (int) (np - n), 0, 0), GGML_TYPE_F16);
}

// FlexPrefill; returns 128-block memberships [nb, nq, hq]
static ggml_tensor * llama_selector_flexprefill(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * mask, double gamma_d, int layer) {
    const float gamma = (float) gamma_d;
    const int64_t block = 128, d = q->ne[0], lq = q->ne[1], hq = q->ne[2], nk = k->ne[1];
    const int64_t nb = nk/block, nq = lq/block;
    GGML_ASSERT(nk % block == 0 && lq % block == 0);
    const float scale = 1.0f/sqrtf((float) d);
    ggml_tensor * a = llama_sel_probe(ctx, q, k, mask, block, scale);
    ggml_tensor * vertical = ggml_scale(ctx, llama_sel_sum1(ctx, a), 1.0f/block);
    ggml_tensor * slash = ggml_scale(ctx, llama_sel_diagonals(ctx, a), 1.0f/block);
    // block means of K, the probe's mean query and every query block's mean (the queries scaled by 1/sqrt(d))
    ggml_tensor * avg_k = llama_sel_pool_k(ctx, k, block);                                       // [d, nb, hkv]
    ggml_tensor * avg_q = ggml_scale(ctx, llama_sel_pool_q(ctx, q, block), scale);              // [d, nq, hq]
    ggml_tensor * q_mean = ggml_view_3d(ctx, avg_q, d, 1, hq, avg_q->nb[1], avg_q->nb[2], (size_t)(nq - 1)*avg_q->nb[1]);
    ggml_tensor * probe_logits = ggml_mul_mat(ctx, avg_k, q_mean);                  // [nb, 1, hq]
    ggml_tensor * pooled_logits = ggml_mul_mat(ctx, avg_k, avg_q);                  // [nb, nq, hq]
    ggml_format_name(vertical, "sel_vertical-%d", layer);
    ggml_format_name(slash, "sel_slash-%d", layer);
    int32_t params[4];
    const float tau = 0.1f;
    memcpy(&params[0], &gamma, sizeof(float));
    memcpy(&params[1], &tau, sizeof(float));
    params[2] = 1;                       // min_budget: the reference's default, 1 token = 1 block
    params[3] = (int32_t) nb;            // max_budget: unbounded
    ggml_tensor * args[] = {vertical, slash, probe_logits, pooled_logits};
    return llama_sel_rule(ctx, LLAMA_SEL_MODE_FLEX, llama_sel_flex, args, 4, nb, nq, hq, params, layer, "sel_membership");
}

// sum of squares over ne0 of a contiguous [d, ...] tensor -> [1, ...]
static ggml_tensor * llama_sel_sqnorm(ggml_context * ctx, ggml_tensor * x) {
    return ggml_sum_rows(ctx, ggml_sqr(ctx, x));
}

static ggml_tensor * llama_selector_minference_bs(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, double topk, int layer) {
    const int64_t lq = q->ne[1], hq = q->ne[2], nk = k->ne[1];
    ggml_tensor * p = ggml_mul_mat(ctx, llama_sel_pool_k(ctx, k, 64), llama_sel_pool_q(ctx, q, 64));   // raw dot, no scale
    ggml_mul_mat_set_prec(p, GGML_PREC_F32);
    const int32_t params[4] = {(int32_t) llround(topk), 0, 0, 0};
    ggml_tensor * args[] = {p};
    return llama_sel_rule(ctx, LLAMA_SEL_MODE_BS, llama_sel_bs, args, 1, nk/64, lq/64, hq, params, layer, "sel_membership");
}

// SpargeAttn; returns [nk/64, lq/128, hq]
static ggml_tensor * llama_selector_spargeattn(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, double tau, int layer) {
    const int64_t d = q->ne[0], lq = q->ne[1], hq = q->ne[2], nk = k->ne[1], hkv = k->ne[2];
    const int64_t nbk = nk/64, nq = lq/128;
    const float theta = 0.1f;
    // key blocks: means, the mean key (smooth_k, over every key), and the self-similarity of the smoothed tokens
    ggml_tensor * kb = llama_sel_pool_k(ctx, k, 64);                                               // [d, nbk, hkv]
    ggml_tensor * km = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, kb, 1, 0, 2, 3))), 1.0f/nbk);   // [1, d, hkv]
    ggml_tensor * kf = ggml_cast(ctx, k, GGML_TYPE_F32);
    ggml_tensor * ks = ggml_add(ctx, kf, ggml_scale(ctx, ggml_reshape_3d(ctx, km, d, 1, hkv), -1.0f));
    ggml_tensor * ku = ggml_l2_norm(ctx, ks, 1e-12f);                                              // [d, nk, hkv]
    ggml_format_name(kf, "sel_kf32-%d", layer);
    ggml_format_name(ks, "sel_ksmooth-%d", layer);
    ggml_format_name(ku, "sel_kunit-%d", layer);
    ku = ggml_reshape_4d(ctx, ku, d, 64, nbk, hkv);
    for (int64_t m = 64; m > 1; m /= 2) {
        ggml_tensor * lo = ggml_view_4d(ctx, ku, d, m/2, nbk, hkv, ku->nb[1], ku->nb[2], ku->nb[3], 0);
        ggml_tensor * hi = ggml_view_4d(ctx, ku, d, m/2, nbk, hkv, ku->nb[1], ku->nb[2], ku->nb[3], (size_t)(m/2)*ku->nb[1]);
        ku = ggml_add(ctx, lo, hi);
    }
    ggml_tensor * simk = llama_sel_sqnorm(ctx, ggml_scale(ctx, ggml_reshape_3d(ctx, ku, d, nbk, hkv), 1.0f/64));   // [1, nbk, hkv]
    // query blocks: self-similarity of the unit queries (q_cur's own [d, hq, lq] layout), and means
    ggml_tensor * qu = ggml_l2_norm(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), 1e-12f);                 // [d, hq, lq]
    qu = ggml_reshape_3d(ctx, qu, d*hq, 128, nq);
    ggml_tensor * qs = ggml_reshape_3d(ctx, ggml_scale(ctx, llama_sel_sum1(ctx, qu), 1.0f/128), d, hq, nq);
    ggml_tensor * simq = llama_sel_sqnorm(ctx, qs);                                                // [1, hq, nq]
    ggml_tensor * s = ggml_mul_mat(ctx, kb, llama_sel_pool_q(ctx, q, 128));                       // [nbk, nq, hq]
    ggml_mul_mat_set_prec(s, GGML_PREC_F32);
    s = ggml_scale(ctx, s, 1.0f/sqrtf((float) d));
    ggml_format_name(km, "sel_kmean-%d", layer);
    ggml_format_name(simk, "sel_simk-%d", layer);
    ggml_format_name(simq, "sel_simq-%d", layer);
    ggml_format_name(s, "sel_scores-%d", layer);
    int32_t params[4] = {0, 0, 0, 0};
    const float tauf = (float) tau;
    memcpy(&params[0], &tauf, sizeof(float));
    memcpy(&params[1], &theta, sizeof(float));
    ggml_tensor * args[] = {s, simq, simk};
    return llama_sel_rule(ctx, LLAMA_SEL_MODE_SPARGE, llama_sel_sparge, args, 3, nbk, nq, hq, params, layer, "sel_membership");
}

// FlashPrefill; returns [nk/128, lq/128, hq]. Its per-tile normalized score is a softmax over each tile's flattened
// (128 queries x key blocks) logits, summed over the queries: the tile maximum cancels exactly as in the reference's
// normalize_scores. A query counts for a key block only at or past the block's last key: the KQ mask at that key.
static ggml_tensor * llama_selector_flashprefill(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * mask, double alpha, int layer) {
    const int64_t d = q->ne[0], lq = q->ne[1], hq = q->ne[2], nk = k->ne[1];
    const int64_t nb = nk/128, nq = lq/128;
    ggml_tensor * s = ggml_mul_mat(ctx, llama_sel_keys_f16(ctx, llama_sel_pool_k(ctx, k, 128)), q);   // [nb padded, lq, hq]
    ggml_mul_mat_set_prec(s, GGML_PREC_F32);
    if (s->ne[0] != nb) {
        s = ggml_cont(ctx, ggml_view_3d(ctx, s, nb, lq, hq, s->nb[1], s->nb[2], 0));                 // [nb, lq, hq]
    }
    ggml_tensor * mlast = ggml_cont(ctx, ggml_view_3d(ctx, mask, 1, nb, lq, 128*ggml_element_size(mask), mask->nb[1], 127*ggml_element_size(mask)));
    ggml_tensor * p = ggml_soft_max_ext(ctx, ggml_reshape_3d(ctx, s, nb*128, nq, hq), ggml_reshape_2d(ctx, mlast, nb*128, nq),
                                        1.0f/sqrtf((float) d), 0.0f);
    ggml_tensor * score = llama_sel_sum1(ctx, ggml_reshape_3d(ctx, p, nb, 128, nq*hq));             // [nb, 1, nq*hq]
    score = ggml_reshape_3d(ctx, score, nb, nq, hq);
    int32_t params[4] = {0, 2, 4, 0};
    const float alphaf = (float) alpha;
    memcpy(&params[0], &alphaf, sizeof(float));
    ggml_tensor * args[] = {score};
    return llama_sel_rule(ctx, LLAMA_SEL_MODE_FLASH, llama_sel_flash, args, 1, nb, nq, hq, params, layer, "sel_membership");
}

// SampleAttention v3; returns [nk/128, lq/128, hq]
static ggml_tensor * llama_selector_sampleattn(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * mask, double alpha, int layer) {
    const int64_t d = q->ne[0], lq = q->ne[1], hq = q->ne[2], nk = k->ne[1];
    ggml_tensor * a = llama_sel_probe(ctx, q, k, mask, 128, 1.0f/sqrtf((float) d));
    ggml_tensor * vertical = llama_sel_sum1(ctx, a);
    ggml_tensor * slash = llama_sel_diagonals(ctx, a);
    int32_t params[4] = {0, 0, 0, 0};
    const float alphaf = (float) alpha;
    memcpy(&params[0], &alphaf, sizeof(float));
    ggml_tensor * args[] = {vertical, slash};
    return llama_sel_rule(ctx, LLAMA_SEL_MODE_SAMPLE, llama_sel_sample, args, 2, nk/128, lq/128, hq, params, layer, "sel_membership");
}

// [nb, nq, hq] at 128-key blocks -> [2*nb, nq, hq] at 64-key blocks (each 128-block as its pair)
static ggml_tensor * llama_sel_keys128(ggml_context * ctx, ggml_tensor * m) {
    return ggml_reshape_3d(ctx, ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, m, 1, m->ne[0], m->ne[1], m->ne[2]), 2, m->ne[0], m->ne[1], m->ne[2]),
                           2*m->ne[0], m->ne[1], m->ne[2]);
}

static ggml_tensor * llama_selector_build(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                          ggml_tensor * mask, float scale, int layer) {
    const char * name = llama_selector_name();
    const double param = llama_selector_param();
    const int64_t hkv = k->ne[2];
    ggml_tensor * m;
    int64_t bq = 64;
    if (strcmp(name, "minference_vs") == 0) {
        m = llama_selector_minference_vs(ctx, q, k, mask, param, layer);
    } else if (strcmp(name, "minference_bs") == 0) {
        m = llama_selector_minference_bs(ctx, q, k, param, layer);
    } else if (strcmp(name, "spargeattn") == 0) {
        m = llama_selector_spargeattn(ctx, q, k, param, layer);          // 64-key blocks, 128-query rows
        bq = 128;
    } else {
        // 128-key blocks as pairs of 64-key blocks, lists per 128-query block
        m = strcmp(name, "flexprefill") == 0  ? llama_selector_flexprefill(ctx, q, k, mask, param, layer)
          : strcmp(name, "flashprefill") == 0 ? llama_selector_flashprefill(ctx, q, k, mask, param, layer)
          :                                     llama_selector_sampleattn(ctx, q, k, mask, param, layer);
        m = llama_sel_keys128(ctx, m);
        bq = 128;
    }
    if (llama_selector_union()) {
        m = llama_block_sparse_union(ctx, m, hkv);
    }
    return llama_block_sparse_attn(ctx, q, k, v, mask, m, 64, bq, scale, layer, "sel");
}

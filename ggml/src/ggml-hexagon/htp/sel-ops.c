#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"

// HTP_OP_SEL: the selection rules of the existing sparse-prefill selectors (src/llama-selectors.h, whose CPU custom
// ops are the reference), so a selector's graph never leaves the NPU.
//
// Long rows (one head's column or diagonal scores over every key, up to 16k) are never sorted: the k-th largest key
// and the cumulative-cover threshold are found by bisection over order-preserving 32-bit keys, 32 passes of HVX
// compares/adds over the row in VTCM. Ties at the threshold keep the lowest indices first, as the reference's stable
// sort. Short per-head work (block sums, the JS divergence, top-k of <= 256 blocks) runs scalar on the stack;
// FlexPrefill's <= 1024-entry flattened map uses the same bisection. Tensor rows move between DDR and VTCM/stack only
// by HVX copies (scalar DDR loads leave L1D lines that DMA writes do not invalidate; see argsort-ops.c).
//
//   mode 1, VS:   src0 column sums [nk, 1, hq], src1 diagonal sums (t = nk-1-offset) [nk, 1, hq];
//                 params [8] vertical_size, [9] slash_size; dst [nk/64, nbq, hq] 64-block cover
//   mode 2, FLEX: src0 probe mean [nk, 1, hq], src1 diagonal sums / probe (t-indexed) [nk, 1, hq],
//                 src2 pooled probe logits [nb, 1, hq], src3 pooled logits [nb, nq, hq];
//                 params [8] gamma, [9] tau (f32), [10] min, [11] max budget (blocks); dst [nb, nq, hq]

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "hvx-reduce.h"
#include "hex-dma.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#define SEL_MAX_ROW   16384   // keys per row
#define SEL_MAX_NB    256     // blocks per row
#define SEL_MAX_FLAT  1024    // query blocks x key blocks of FlexPrefill's flattened map

struct sel_context {
    struct htp_ops_context * octx;
    uint8_t * vtcm_base;
    size_t    vtcm_per_thread;
    uint32_t  rows_per_thread;
};

// ---- vector helpers over a VTCM row of n (multiple of 32) uint32 keys ----

// order-preserving keys: flip every bit of negatives, the sign bit of the rest
static inline HVX_Vector sel_key(HVX_Vector bits) {
    HVX_Vector sign = Q6_Vw_vasr_VwR(bits, 31);
    return Q6_V_vxor_VV(bits, Q6_V_vor_VV(sign, Q6_V_vsplat_R(0x80000000)));
}

static inline float sel_key_value(uint32_t key) {
    union { uint32_t u; float f; } v;
    v.u = (key & 0x80000000u) ? key ^ 0x80000000u : ~key;
    return v.f;
}

// number of keys >= c
static uint32_t sel_count_ge(const HVX_Vector * keys, uint32_t nv, uint32_t c) {
    const HVX_Vector cv = Q6_V_vsplat_R(c), one = Q6_V_vsplat_R(1), zero = Q6_V_vzero();
    HVX_Vector acc = zero;
    for (uint32_t i = 0; i < nv; i++) {
        HVX_VectorPred lt = Q6_Q_vcmp_gt_VuwVuw(cv, keys[i]);
        acc = Q6_Vw_vadd_VwVw(acc, Q6_V_vmux_QVV(lt, zero, one));
    }
    return (uint32_t) hvx_vec_get_i32(hvx_vec_reduce_sum_i32(acc));
}

// sum of the values whose key is >= c
static float sel_sum_ge(const HVX_Vector * keys, const HVX_Vector * values, uint32_t nv, uint32_t c) {
    const HVX_Vector cv = Q6_V_vsplat_R(c), zero = Q6_V_vzero();
    HVX_Vector acc = zero;
    for (uint32_t i = 0; i < nv; i++) {
        HVX_VectorPred lt = Q6_Q_vcmp_gt_VuwVuw(cv, keys[i]);
        acc = hvx_vec_add_f32_f32(acc, Q6_V_vmux_QVV(lt, zero, values[i]));
    }
    return hvx_vec_get_f32(hvx_vec_reduce_sum_n_f32(acc, 32));
}

// the k-th largest key (1 <= k <= n): the largest c with count(keys >= c) >= k
static uint32_t sel_kth_key(const HVX_Vector * keys, uint32_t nv, uint32_t k) {
    uint32_t c = 0;
    for (int bit = 31; bit >= 0; bit--) {
        const uint32_t t = c | (1u << bit);
        if (sel_count_ge(keys, nv, t) >= k) { c = t; }
    }
    return c;
}

// the top-k selection as 0/1 words in `mask` (VTCM): every key > c, and the lowest-index `need` keys == c
static void sel_topk_mask(const HVX_Vector * keys, uint32_t nv, uint32_t k, HVX_Vector * mask) {
    const HVX_Vector one = Q6_V_vsplat_R(1), zero = Q6_V_vzero();
    if (k == 0) {
        for (uint32_t i = 0; i < nv; i++) { mask[i] = zero; }
        return;
    }
    k = MIN(k, nv*32);
    const uint32_t c = sel_kth_key(keys, nv, k);
    const uint32_t gt = c == 0xffffffffu ? 0 : sel_count_ge(keys, nv, c + 1);
    uint32_t need = k - gt;
    const uint32_t eq = sel_count_ge(keys, nv, c) - gt;
    const HVX_Vector cv = Q6_V_vsplat_R(c);
    for (uint32_t i = 0; i < nv; i++) {
        HVX_VectorPred lt = Q6_Q_vcmp_gt_VuwVuw(cv, keys[i]);        // key < c
        HVX_VectorPred e = Q6_Q_vcmp_eq_VwVw(keys[i], cv);
        HVX_Vector m = Q6_V_vmux_QVV(lt, zero, one);                  // key >= c
        if (need < eq) {
            // keep only the first `need` equal keys, in index order
            HVX_Vector ev = Q6_V_vmux_QVV(e, one, zero);
            const uint32_t here = (uint32_t) hvx_vec_get_i32(hvx_vec_reduce_sum_i32(ev));
            if (here > 0) {
                if (here <= need) {
                    need -= here;
                } else {
                    int32_t lanes[32] __attribute__((aligned(128)));
                    int32_t keep[32] __attribute__((aligned(128)));
                    *(HVX_Vector *) lanes = ev;
                    *(HVX_Vector *) keep = m;
                    asm volatile(" syncht\n" ::: "memory");
                    for (int l = 0; l < 32; l++) {
                        if (lanes[l]) {
                            if (need) { need--; } else { keep[l] = 0; }
                        }
                    }
                    asm volatile(" syncht\n" ::: "memory");
                    m = *(HVX_Vector *) keep;
                }
            }
            if (need == 0) {
                // the remaining equal keys are dropped
                for (uint32_t j = i + 1; j < nv; j++) {
                    HVX_VectorPred gtj = Q6_Q_vcmp_gt_VuwVuw(keys[j], cv);
                    mask[j] = Q6_V_vmux_QVV(gtj, one, zero);
                }
                mask[i] = m;
                return;
            }
        }
        mask[i] = m;
    }
}

// on non-negative values, the entries (descending) whose cumulative sum stays <= score (strict: < score), + 1:
// FlexPrefill's score_cover_topk (<=), SampleAttention's minimal count reaching alpha (<)
static uint32_t sel_cover_count_ex(const HVX_Vector * keys, const HVX_Vector * values, uint32_t nv, float score, bool strict) {
    const uint32_t n = nv*32;
    const float total = sel_sum_ge(keys, values, nv, 0);
    if (strict ? total < score : total <= score) { return n + 1; }
    uint32_t c = 0;                                   // largest c with sum(keys >= c) > score (strict: >= score)
    for (int bit = 31; bit >= 0; bit--) {
        const uint32_t t = c | (1u << bit);
        const float s = sel_sum_ge(keys, values, nv, t);
        if (strict ? s >= score : s > score) { c = t; }
    }
    const uint32_t n1 = c == 0xffffffffu ? 0 : sel_count_ge(keys, nv, c + 1);
    const float s1 = c == 0xffffffffu ? 0.0f : sel_sum_ge(keys, values, nv, c + 1);
    const uint32_t eq = sel_count_ge(keys, nv, c) - n1;
    const float v = sel_key_value(c);
    uint32_t fit = 0;                                 // tied entries that still fit under the score
    if (v > 0.0f) {
        float acc = s1;
        while (fit < eq && (strict ? acc + v < score : acc + v <= score)) { acc += v; fit++; }
    } else {
        fit = eq;
    }
    return n1 + fit + 1;
}

static uint32_t sel_cover_count(const HVX_Vector * keys, const HVX_Vector * values, uint32_t nv, float score) {
    return sel_cover_count_ex(keys, values, nv, score, false);
}

// load row (n floats, DDR) into VTCM values and keys
static void sel_load(const uint8_t * src, uint32_t n, HVX_Vector * values, HVX_Vector * keys) {
    hvx_copy_f32_au((uint8_t *) values, src, n);
    for (uint32_t i = 0; i < n/32; i++) { keys[i] = sel_key(values[i]); }
}

// any selected word in each group of 64 (two vectors), and in its first 63 lanes
static void sel_groups(const HVX_Vector * mask, uint32_t ng, uint8_t * any, uint8_t * any63) {
    const HVX_VectorPred last = Q6_Q_vsetq_R(31*4);  // lanes 0..30 of the second vector
    for (uint32_t g = 0; g < ng; g++) {
        HVX_Vector a = Q6_V_vor_VV(mask[2*g], mask[2*g + 1]);
        HVX_Vector b = Q6_V_vor_VV(mask[2*g], Q6_V_vmux_QVV(last, mask[2*g + 1], Q6_V_vzero()));
        any[g]   = hvx_vec_get_i32(hvx_vec_reduce_sum_i32(a)) != 0;
        any63[g] = hvx_vec_get_i32(hvx_vec_reduce_sum_i32(b)) != 0;
    }
}

// write one membership row from the stack (scalar stores) to DDR (vector copy)
static void sel_store_row(uint8_t * dst, const float * row, uint32_t n) {
    asm volatile(" syncht\n" ::: "memory");
    hvx_copy_f32_uu(dst, (const uint8_t *) row, n);
}

static void sel_vs_head(struct sel_context * sctx, uint32_t h, uint8_t * spad) {
    struct htp_ops_context * octx = sctx->octx;
    const struct htp_tensor * vertical = octx->src[0];
    const struct htp_tensor * slash = octx->src[1];
    const struct htp_tensor * dst = octx->dst;
    const uint32_t nk = vertical->ne[0], nbk = dst->ne[0], nbq = dst->ne[1];
    const uint32_t nv = nk/32;
    HVX_Vector * values = (HVX_Vector *) spad;
    HVX_Vector * keys = values + nv;
    HVX_Vector * mask = keys + nv;
    uint8_t vb[SEL_MAX_NB], vb63[SEL_MAX_NB], g[SEL_MAX_NB], g2[SEL_MAX_NB];
    float row[SEL_MAX_NB] __attribute__((aligned(128)));

    // columns: 0..29 forced (the largest keys, counted in the budget as the reference's +inf)
    sel_load((const uint8_t *) vertical->data + h*vertical->nb[2], nk, values, keys);
    keys[0] = Q6_V_vmux_QVV(Q6_Q_vsetq_R(30*4), Q6_V_vsplat_R(0xffffffff), keys[0]);
    sel_topk_mask(keys, nv, (uint32_t) octx->op_params[8], mask);
    sel_groups(mask, nbk, vb, vb63);
    // diagonals: t >= nk - 100 forced
    sel_load((const uint8_t *) slash->data + h*slash->nb[2], nk, values, keys);
    for (uint32_t i = (nk - 100)/32; i < nv; i++) {
        const int32_t first = (int32_t) (nk - 100) - (int32_t) (32*i);
        HVX_VectorPred before = Q6_Q_vsetq_R(first > 0 ? first*4 : 0);   // lanes < first are not forced
        keys[i] = Q6_V_vmux_QVV(before, keys[i], Q6_V_vsplat_R(0xffffffff));
    }
    sel_topk_mask(keys, nv, (uint32_t) octx->op_params[9], mask);
    sel_groups(mask, nbk, g, g2);
    for (uint32_t a = 0; a < nbq; a++) {
        const int32_t A = (int32_t) (nbk - nbq + a);
        for (uint32_t b = 0; b < nbk; b++) {
            const int32_t d = A - (int32_t) b;
            const bool hit = d >= 0 && (vb[b] || g[nbk - d - 1] || (d >= 1 && g2[nbk - d]));
            row[b] = hit ? 1.0f : 0.0f;
        }
        sel_store_row((uint8_t *) dst->data + a*dst->nb[1] + h*dst->nb[2], row, nbk);
    }
}

// stable top-k of a short row (n <= SEL_MAX_NB) into keep[]; -inf never kept
static void sel_small_topk(const float * x, uint32_t n, uint32_t k, uint8_t * keep) {
    for (uint32_t i = 0; i < n; i++) { keep[i] = 0; }
    for (uint32_t r = 0; r < MIN(k, n); r++) {
        int32_t best = -1;
        for (uint32_t i = 0; i < n; i++) {
            if (!keep[i] && (best < 0 || x[i] > x[best])) { best = (int32_t) i; }
        }
        if (best < 0 || x[best] == -INFINITY) { break; }
        keep[best] = 1;
    }
}

static void sel_flex_head(struct sel_context * sctx, uint32_t h, uint8_t * spad) {
    struct htp_ops_context * octx = sctx->octx;
    const struct htp_tensor * vertical = octx->src[0];
    const struct htp_tensor * slash = octx->src[1];
    const struct htp_tensor * probe = octx->src[2];
    const struct htp_tensor * pooled = octx->src[3];
    const struct htp_tensor * dst = octx->dst;
    const uint32_t nk = vertical->ne[0], nb = dst->ne[0], nq = dst->ne[1], block = nk/nb, nv = nk/32;
    float gamma, tau;
    memcpy(&gamma, &octx->op_params[8], sizeof(float));
    memcpy(&tau, &octx->op_params[9], sizeof(float));
    const int32_t low = octx->op_params[10], high = MIN(octx->op_params[11], (int32_t) nb);
    HVX_Vector * values = (HVX_Vector *) spad;
    HVX_Vector * keys = values + nv;
    float vbk[SEL_MAX_NB], sbk[SEL_MAX_NB], p[SEL_MAX_NB] __attribute__((aligned(128)));
    uint8_t vsel[SEL_MAX_NB], ssel[SEL_MAX_NB];
    float row[SEL_MAX_NB] __attribute__((aligned(128)));

    // vertical: cover count, then 128-block sums (the row's own block size)
    sel_load((const uint8_t *) vertical->data + h*vertical->nb[2], nk, values, keys);
    int32_t num_v = (int32_t) (sel_cover_count(keys, values, nv, gamma)/128 + 1);
    for (uint32_t c = 0; c < nb; c++) {
        HVX_Vector acc = Q6_V_vzero();
        for (uint32_t i = c*block/32; i < (c + 1)*block/32; i++) { acc = hvx_vec_add_f32_f32(acc, values[i]); }
        vbk[c] = hvx_vec_get_f32(hvx_vec_reduce_sum_n_f32(acc, 32));
    }
    // slash: t-indexed, block offset d sums t in block nb-1-d
    sel_load((const uint8_t *) slash->data + h*slash->nb[2], nk, values, keys);
    int32_t num_s = (int32_t) (sel_cover_count(keys, values, nv, gamma)/128 + 1);
    for (uint32_t c = 0; c < nb; c++) {
        HVX_Vector acc = Q6_V_vzero();
        for (uint32_t i = c*block/32; i < (c + 1)*block/32; i++) { acc = hvx_vec_add_f32_f32(acc, values[i]); }
        sbk[nb - 1 - c] = hvx_vec_get_f32(hvx_vec_reduce_sum_n_f32(acc, 32));
    }
    num_v = MAX(low, MIN(high, num_v));
    num_s = MAX(low, MIN(high, num_s));

    // pattern decision
    hvx_copy_f32_uu((uint8_t *) p, (const uint8_t *) probe->data + h*probe->nb[2], nb);
    asm volatile(" syncht\n" ::: "memory");
    float mx = -INFINITY;
    for (uint32_t c = 0; c < nb; c++) { mx = MAX(mx, p[c]); }
    double z = 0.0;
    for (uint32_t c = 0; c < nb; c++) { p[c] = expf(p[c] - mx); z += p[c]; }
    double js = 0.0;
    for (uint32_t c = 0; c < nb; c++) {
        const double pc = p[c]/z, qc = vbk[c], m = (pc + qc)/2;
        js += .5*pc*log(pc/m) + .5*qc*log(qc/m);
    }
    const bool query_aware = sqrt(js) < tau;
    if (query_aware) { num_v = num_s = low; }
    vbk[0] = INFINITY;
    sbk[0] = INFINITY;
    sel_small_topk(vbk, nb, (uint32_t) num_v, vsel);
    sel_small_topk(sbk, nb, (uint32_t) num_s, ssel);

    // query aware: causal softmax of each query block's pooled logits, then the flattened map's cover -- the
    // entries (descending) whose cumulative sum stays <= gamma*nq are its top-M, M = cover count - 1
    const uint32_t nflat = nq*nb, nfv = (nflat + 31)/32;
    HVX_Vector * fval = keys + nv;                                    // VTCM: map, keys, mask
    HVX_Vector * fkeys = fval + nfv;
    HVX_Vector * fmask = fkeys + nfv;
    if (query_aware) {
        for (uint32_t i = 0; i < nfv; i++) { fval[i] = Q6_V_vzero(); }   // zeros never enter the cover
        for (uint32_t a = 0; a < nq; a++) {
            const uint32_t A = nb - nq + a;
            hvx_copy_f32_uu((uint8_t *) row, (const uint8_t *) pooled->data + a*pooled->nb[1] + h*pooled->nb[2], nb);
            asm volatile(" syncht\n" ::: "memory");
            float m2 = -INFINITY;
            for (uint32_t c = 0; c <= A; c++) { m2 = MAX(m2, row[c]); }
            double zz = 0.0;
            for (uint32_t c = 0; c <= A; c++) { zz += exp(row[c] - m2); }
            for (uint32_t c = 0; c < nb; c++) { p[c] = c <= A ? (float) (exp(row[c] - m2)/zz) : 0.0f; }
            asm volatile(" syncht\n" ::: "memory");
            hvx_copy_f32_uu((uint8_t *) fval + a*nb*sizeof(float), (const uint8_t *) p, nb);
        }
        for (uint32_t i = 0; i < nfv; i++) { fkeys[i] = sel_key(fval[i]); }
        const uint32_t m = sel_cover_count(fkeys, fval, nfv, gamma*nq) - 1;
        sel_topk_mask(fkeys, nfv, m, fmask);
    }
    int32_t flags[SEL_MAX_NB] __attribute__((aligned(128)));
    for (uint32_t a = 0; a < nq; a++) {
        const uint32_t A = nb - nq + a;
        if (query_aware) {
            hvx_copy_f32_uu((uint8_t *) flags, (const uint8_t *) fmask + a*nb*sizeof(int32_t), nb);
            asm volatile(" syncht\n" ::: "memory");
        }
        for (uint32_t c = 0; c < nb; c++) {
            row[c] = c <= A && (vsel[c] || ssel[A - c] || (query_aware && flags[c])) ? 1.0f : 0.0f;
        }
        sel_store_row((uint8_t *) dst->data + a*dst->nb[1] + h*dst->nb[2], row, nb);
    }
}

// one short row (n <= SEL_MAX_NB floats from DDR) into VTCM values + keys, padded to whole vectors with key 0 (below
// every real key, never selected while the budget is within the real entries)
static uint32_t sel_load_short(const uint8_t * src, uint32_t n, HVX_Vector * values, HVX_Vector * keys) {
    const uint32_t nv = (n + 31)/32;
    for (uint32_t i = 0; i < nv; i++) { values[i] = Q6_V_vzero(); }
    hvx_copy_f32_uu((uint8_t *) values, src, n);
    for (uint32_t i = 0; i < nv; i++) { keys[i] = sel_key(values[i]); }
    if (n % 32) {
        keys[nv - 1] = Q6_V_vmux_QVV(Q6_Q_vsetq_R((n % 32)*4), keys[nv - 1], Q6_V_vzero());
    }
    return nv;
}

// keep keys[i] only for the first `n` entries (later ones become key 0)
static void sel_limit(HVX_Vector * keys, uint32_t nv, uint32_t n) {
    for (uint32_t i = 0; i < nv; i++) {
        const int32_t first = (int32_t) n - (int32_t) (32*i);
        if (first >= 32) { continue; }
        keys[i] = first <= 0 ? Q6_V_vzero() : Q6_V_vmux_QVV(Q6_Q_vsetq_R(first*4), keys[i], Q6_V_vzero());
    }
}

// 0/1 words of a VTCM mask into a stack float row
static void sel_mask_row(const HVX_Vector * mask, uint32_t n, float * row) {
    int32_t words[SEL_MAX_NB] __attribute__((aligned(128)));
    hvx_copy_f32_uu((uint8_t *) words, (const uint8_t *) mask, n);
    asm volatile(" syncht\n" ::: "memory");
    for (uint32_t i = 0; i < n; i++) { row[i] = words[i] ? 1.0f : 0.0f; }
}

// MInference block-sparse: one (query block, head) row
static void sel_bs_row(struct sel_context * sctx, uint32_t a, uint32_t h, uint8_t * spad) {
    struct htp_ops_context * octx = sctx->octx;
    const struct htp_tensor * p = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    const uint32_t nbk = dst->ne[0], nbq = dst->ne[1], A = nbk - nbq + a;
    HVX_Vector * values = (HVX_Vector *) spad;
    HVX_Vector * keys = values + SEL_MAX_NB/32;
    HVX_Vector * mask = keys + SEL_MAX_NB/32;
    float row[SEL_MAX_NB] __attribute__((aligned(128)));
    const uint32_t nv = sel_load_short((const uint8_t *) p->data + a*p->nb[1] + h*p->nb[2], nbk, values, keys);
    sel_limit(keys, nv, A + 1);                                      // causal: blocks after the row's own
    sel_topk_mask(keys, nv, MIN((uint32_t) octx->op_params[8], A + 1), mask);
    sel_mask_row(mask, nbk, row);
    sel_store_row((uint8_t *) dst->data + a*dst->nb[1] + h*dst->nb[2], row, nbk);
}

// SpargeAttn TopCdf: one (128-query block, head) row over 64-key blocks
static void sel_sparge_row(struct sel_context * sctx, uint32_t a, uint32_t h, uint8_t * spad) {
    struct htp_ops_context * octx = sctx->octx;
    const struct htp_tensor * s = octx->src[0];
    const struct htp_tensor * simq = octx->src[1];
    const struct htp_tensor * simk = octx->src[2];
    const struct htp_tensor * dst = octx->dst;
    const uint32_t nbk = dst->ne[0], nq = dst->ne[1], hq = dst->ne[2], groups = hq/simk->ne[2];
    const uint32_t A = nbk/2 - nq + a, visible = MIN(nbk, (A + 1)*2);
    float tau, theta;
    memcpy(&tau, &octx->op_params[8], sizeof(float));
    memcpy(&theta, &octx->op_params[9], sizeof(float));
    HVX_Vector * values = (HVX_Vector *) spad;
    HVX_Vector * keys = values + SEL_MAX_NB/32;
    HVX_Vector * mask = keys + SEL_MAX_NB/32;
    float row[SEL_MAX_NB] __attribute__((aligned(128)));
    float sk[SEL_MAX_NB] __attribute__((aligned(128)));
    float sq[32] __attribute__((aligned(128)));
    hvx_copy_f32_uu((uint8_t *) sk, (const uint8_t *) simk->data + (h/groups)*simk->nb[2], nbk);
    hvx_copy_f32_uu((uint8_t *) sq, (const uint8_t *) simq->data + h*simq->nb[1] + a*simq->nb[2], 1);
    hvx_copy_f32_uu((uint8_t *) row, (const uint8_t *) s->data + a*s->nb[1] + h*s->nb[2], nbk);
    asm volatile(" syncht\n" ::: "memory");
    const bool q_similar = sq[0] > theta;
    // softmax over the causal, self-similar key blocks (scalar: <= 256 entries)
    float mx = -INFINITY;
    for (uint32_t b = 0; b < nbk; b++) {
        if (!(b < visible && sk[b] > theta)) { row[b] = -INFINITY; }
        mx = MAX(mx, row[b]);
    }
    double z = 0.0;
    for (uint32_t b = 0; b < nbk; b++) { row[b] = row[b] == -INFINITY ? 0.0f : expf(row[b] - mx); z += row[b]; }
    for (uint32_t b = 0; b < nbk; b++) { row[b] = z > 0 ? (float) (row[b]/z) : 0.0f; }
    asm volatile(" syncht\n" ::: "memory");
    const uint32_t nv = (nbk + 31)/32;
    for (uint32_t i = 0; i < nv; i++) { values[i] = Q6_V_vzero(); }
    hvx_copy_f32_uu((uint8_t *) values, (const uint8_t *) row, nbk);
    for (uint32_t i = 0; i < nv; i++) { keys[i] = sel_key(values[i]); }
    sel_limit(keys, nv, nbk);
    // the longest descending prefix with cumulative <= tau, at least one block
    uint32_t count = sel_cover_count(keys, values, nv, tau) - 1;
    count = MAX(count, 1);
    sel_topk_mask(keys, nv, count, mask);
    sel_mask_row(mask, nbk, row);
    bool seen = false;
    for (uint32_t b = 0; b < nbk; b++) {
        if (!q_similar || !(sk[b] > theta)) { row[b] = 1.0f; }
        if (b >= visible) { row[b] = 0.0f; }
        seen = seen || (b <= 2*A && row[b] > .5f);
    }
    // a query block whose first 64 queries would see no key keeps the key block at its start
    if (!seen) { row[2*A] = 1.0f; }
    sel_store_row((uint8_t *) dst->data + a*dst->nb[1] + h*dst->nb[2], row, nbk);
}

// FlashPrefill: one (128-query block, head) row
static void sel_flash_row(struct sel_context * sctx, uint32_t a, uint32_t h) {
    struct htp_ops_context * octx = sctx->octx;
    const struct htp_tensor * s = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    const uint32_t nb = dst->ne[0], nq = dst->ne[1];
    const int32_t A = (int32_t) (nb - nq + a), sink = octx->op_params[9], window = octx->op_params[10];
    float alpha;
    memcpy(&alpha, &octx->op_params[8], sizeof(float));
    float row[SEL_MAX_NB] __attribute__((aligned(128)));
    hvx_copy_f32_uu((uint8_t *) row, (const uint8_t *) s->data + a*s->nb[1] + h*s->nb[2], nb);
    asm volatile(" syncht\n" ::: "memory");
    float mx = -INFINITY;
    for (uint32_t c = 0; c < nb; c++) { mx = MAX(mx, row[c]); }
    for (uint32_t c = 0; c < nb; c++) {
        const int32_t dist = A - (int32_t) c;
        row[c] = dist >= 0 && (row[c] >= alpha*mx || (int32_t) c < sink || dist < window) ? 1.0f : 0.0f;
    }
    sel_store_row((uint8_t *) dst->data + a*dst->nb[1] + h*dst->nb[2], row, nb);
}

// SampleAttention v3: one head
static void sel_sample_head(struct sel_context * sctx, uint32_t h, uint8_t * spad) {
    struct htp_ops_context * octx = sctx->octx;
    const struct htp_tensor * vertical = octx->src[0];
    const struct htp_tensor * slash = octx->src[1];
    const struct htp_tensor * dst = octx->dst;
    const uint32_t nk = vertical->ne[0], nb = dst->ne[0], nq = dst->ne[1], block = nk/nb, nv = nk/32;
    float alpha;
    memcpy(&alpha, &octx->op_params[8], sizeof(float));
    HVX_Vector * values = (HVX_Vector *) spad;
    HVX_Vector * keys = values + nv;
    HVX_Vector * bvals = keys + nv;                                  // block sums: values, keys, mask
    HVX_Vector * bkeys = bvals + SEL_MAX_NB/32;
    HVX_Vector * bmask = bkeys + SEL_MAX_NB/32;
    float sums[SEL_MAX_NB] __attribute__((aligned(128)));
    float csel[SEL_MAX_NB] __attribute__((aligned(128)));
    float ssel[SEL_MAX_NB] __attribute__((aligned(128)));
    float row[SEL_MAX_NB] __attribute__((aligned(128)));
    for (int pass = 0; pass < 2; pass++) {
        const struct htp_tensor * src = pass == 0 ? vertical : slash;
        sel_load((const uint8_t *) src->data + h*src->nb[2], nk, values, keys);
        double total = 0.0;
        for (uint32_t c = 0; c < nb; c++) {
            HVX_Vector acc = Q6_V_vzero();
            for (uint32_t i = c*block/32; i < (c + 1)*block/32; i++) { acc = hvx_vec_add_f32_f32(acc, values[i]); }
            const float v = hvx_vec_get_f32(hvx_vec_reduce_sum_n_f32(acc, 32));
            sums[pass == 0 ? c : nb - 1 - c] = v;                     // slash: t-block c is offset block nb-1-c
            total += v;
        }
        for (uint32_t c = 0; c < nb; c++) { sums[c] = (float) (sums[c]/total); }
        asm volatile(" syncht\n" ::: "memory");
        const uint32_t bnv = (nb + 31)/32;
        for (uint32_t i = 0; i < bnv; i++) { bvals[i] = Q6_V_vzero(); }
        hvx_copy_f32_uu((uint8_t *) bvals, (const uint8_t *) sums, nb);
        for (uint32_t i = 0; i < bnv; i++) { bkeys[i] = sel_key(bvals[i]); }
        sel_limit(bkeys, bnv, nb);
        const uint32_t count = MIN(sel_cover_count_ex(bkeys, bvals, bnv, alpha, true), nb);
        sel_topk_mask(bkeys, bnv, count, bmask);
        sel_mask_row(bmask, nb, pass == 0 ? csel : ssel);
    }
    for (uint32_t a = 0; a < nq; a++) {
        const int32_t A = (int32_t) (nb - nq + a);
        bool seen = false;
        for (uint32_t c = 0; c < nb; c++) {
            const int32_t d = A - (int32_t) c;
            row[c] = d >= 0 && (csel[c] > .5f || ssel[d] > .5f) ? 1.0f : 0.0f;
            seen = seen || row[c] > .5f;
        }
        if (!seen) { row[A] = 1.0f; }      // a query block left with no block keeps its diagonal
        sel_store_row((uint8_t *) dst->data + a*dst->nb[1] + h*dst->nb[2], row, nb);
    }
}

static void sel_job(unsigned int n, unsigned int ith, void * data) {
    struct sel_context * sctx = (struct sel_context *) data;
    struct htp_ops_context * octx = sctx->octx;
    const int32_t mode = octx->op_params[HTP_SEL_P_MODE];
    const bool per_head = mode == HTP_SEL_MODE_VS || mode == HTP_SEL_MODE_FLEX || mode == HTP_SEL_MODE_SAMPLE;
    const uint32_t nbq = octx->dst->ne[1], units = per_head ? octx->dst->ne[2] : nbq*octx->dst->ne[2];
    uint8_t * spad = sctx->vtcm_base + sctx->vtcm_per_thread * ith;
    const uint32_t start = sctx->rows_per_thread * ith, end = MIN(start + sctx->rows_per_thread, units);
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, start);
    for (uint32_t u = start; u < end; u++) {
        switch (mode) {
            case HTP_SEL_MODE_VS:     sel_vs_head(sctx, u, spad); break;
            case HTP_SEL_MODE_FLEX:   sel_flex_head(sctx, u, spad); break;
            case HTP_SEL_MODE_SAMPLE: sel_sample_head(sctx, u, spad); break;
            case HTP_SEL_MODE_BS:     sel_bs_row(sctx, u % nbq, u / nbq, spad); break;
            case HTP_SEL_MODE_SPARGE: sel_sparge_row(sctx, u % nbq, u / nbq, spad); break;
            case HTP_SEL_MODE_FLASH:  sel_flash_row(sctx, u % nbq, u / nbq); break;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, start);
}

int op_sel(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst = octx->dst;
    const int32_t mode = octx->op_params[HTP_SEL_P_MODE];
    const bool per_head = mode == HTP_SEL_MODE_VS || mode == HTP_SEL_MODE_FLEX || mode == HTP_SEL_MODE_SAMPLE;
    if (mode < HTP_SEL_MODE_VS || mode > HTP_SEL_MODE_SAMPLE || src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (per_head && (src0->ne[0] % 32 || src0->ne[0] > SEL_MAX_ROW)) || dst->ne[0] > SEL_MAX_NB ||
        (mode == HTP_SEL_MODE_FLEX && dst->ne[0]*dst->ne[1] > SEL_MAX_FLAT)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    const uint32_t heads = per_head ? dst->ne[2] : dst->ne[1]*dst->ne[2];
    const uint32_t n_threads = MIN(heads, octx->n_threads);
    const size_t row = per_head ? src0->ne[0] : 0;
    const size_t per_thread = hex_round_up(3*row*sizeof(float) + 3*SEL_MAX_FLAT*sizeof(float), 256);
    if (octx->ctx->vtcm_size < per_thread * n_threads) {
        FARF(ERROR, "sel: VTCM size too small. Needed %zu, have %zu", per_thread * n_threads, octx->ctx->vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }
    struct sel_context sctx;
    sctx.octx = octx;
    sctx.vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    sctx.vtcm_per_thread = per_thread;
    sctx.rows_per_thread = (heads + n_threads - 1) / n_threads;
    worker_pool_run_func(octx->ctx->worker_pool, sel_job, &sctx, n_threads);
    return HTP_STATUS_OK;
}

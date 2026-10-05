#pragma once

#include "ggml.h"
#include "llama-block-sparse.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

static double llama_xattention_threshold() {
    const char * value = getenv("LLAMA_XATTN_THRESHOLD");
    if (!value) { return 0.0f; }
    char * end = nullptr;
    const double threshold = strtod(value, &end);
    GGML_ASSERT(end != value && *end == '\0' && threshold > 0.0f && threshold <= 1.0f);
    return threshold;
}

static int llama_xattention_block_size() {
    const char * value = getenv("LLAMA_XATTN_BLOCK_SIZE");
    const int size = value ? atoi(value) : 128;
    GGML_ASSERT(size == 64 || size == 128 || size == 256);
    return size;
}

static float llama_xattention_mask_at(const ggml_tensor * mask, int64_t key, int64_t query) {
    const auto * p = static_cast<const char *>(mask->data) + key*mask->nb[0] + query*mask->nb[1];
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    return mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(p)) : *reinterpret_cast<const float *>(p);
}

static void llama_xattention_reverse_indices(ggml_tensor * dst, int ith, int nth, void *) {
    auto * out = static_cast<int32_t *>(dst->data);
    for (int64_t h = ith; h < dst->ne[1]; h += nth) {
        for (int64_t i = 0; i < dst->ne[0]; ++i) {
            out[h*dst->ne[0]+i] = int32_t(i/16*16 + 15-i%16);
        }
    }
}

static void llama_xattention_reduce_mask(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * mask = dst->src[0];
    auto * out = static_cast<float *>(dst->data);
    for (int64_t q = ith; q < dst->ne[1]; q += nth) {
        for (int64_t k = 0; k < dst->ne[0]; ++k) {
            // columns past the mask are the zero keys that pad the product to whole 32-row tiles
            out[q*dst->ne[0]+k] = k*16 < mask->ne[0] && std::isfinite(llama_xattention_mask_at(mask, k*16, q*16+15)) ? 0.0f : -1.0e30f;
        }
    }
}

// LLAMA_XATTN_SELECT=npu tags the cumulative selection so the Hexagon backend runs it on the DSP (HTP_OP_XATTN_SELECT,
// the same walk over HVX-sorted priorities) instead of the CPU; the CPU op stays the reference and the fallback.
static bool llama_xattention_select_npu() {
    const char * value = getenv("LLAMA_XATTN_SELECT");
    GGML_ASSERT(!value || strcmp(value, "npu") == 0 || strcmp(value, "cpu") == 0);
    return value && strcmp(value, "npu") == 0;
}

static bool llama_xattention_gqa_union() {
    // LLAMA_XATTN_GQA=union shares one list per KV head (the OR of its query heads' masks) so a single
    // GQA FlashAttention runs; the default keeps upstream's independent query-head masks.
    const char * value = getenv("LLAMA_XATTN_GQA");
    GGML_ASSERT(!value || strcmp(value, "union") == 0 || strcmp(value, "head") == 0);
    return value && strcmp(value, "union") == 0;
}

// dst is [nk, nq, H]; scores carries H*groups heads, and a GQA group's masks are OR-ed when groups > 1.
static void llama_xattention_select(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * scores = dst->src[0];
    const auto * mask = dst->src[1];
    const int64_t nk = scores->ne[0], nq = scores->ne[1];
    const int64_t bs = mask->ne[1]/nq, groups = scores->ne[2]/dst->ne[2];
    const double threshold = llama_xattention_threshold();
    GGML_ASSERT(nk >= 2 && mask->ne[1] % nq == 0 && scores->ne[2] == groups*dst->ne[2]);
    std::vector<int64_t> ranked(nk);
    std::vector<double> other(nk), priority(nk);
    for (int64_t row = ith; row < nq*dst->ne[2]; row += nth) {
        const int64_t q = row%nq, j = row/nq;
        auto * keep = static_cast<float *>(dst->data) + row*nk;
        std::fill(keep, keep+nk, 0.0f);
        int64_t diagonal = -1;
        for (int64_t b = 0; b < nk; ++b) {
            if (std::isfinite(llama_xattention_mask_at(mask, b*bs, (q+1)*bs-1))) { diagonal = b; }
        }
        GGML_ASSERT(diagonal >= 0);
        keep[0] = keep[diagonal] = 1.0f;
        for (int64_t h = j*groups; h < (j+1)*groups; ++h) {
            const auto * values = reinterpret_cast<const float *>(static_cast<const char *>(scores->data) + q*scores->nb[1] + h*scores->nb[2]);
            double total = 0.0, forced = 0.0;
            for (int64_t b = 0; b < nk; ++b) {
                const double mass = values[b];
                GGML_ASSERT(std::isfinite(mass) && mass >= 0.0f);
                // At offset zero, upstream's identity assignment replaces the sink column.
                const bool mandatory = b == diagonal || (b == 0 && diagonal > q);
                total += mass;
                forced += mandatory ? mass : 0.0f;
                other[b] = mandatory ? 0.0f : mass;
                priority[b] = mandatory ? 100000.0*(1.0+mass) : mass;
            }
            std::iota(ranked.begin(), ranked.end(), 0);
            std::stable_sort(ranked.begin(), ranked.end(), [&](int64_t a, int64_t b) { return priority[a] > priority[b]; });
            std::sort(other.begin(), other.end(), std::greater<double>());
            // Upstream reserves two leading slots, including when sink and diagonal coincide.
            double cumulative = 0.0;
            for (int64_t rank = 0; rank < nk; ++rank) {
                if (cumulative < total*threshold && ranked[rank] <= diagonal) { keep[ranked[rank]] = 1.0f; }
                cumulative += rank == 0 ? 0.0f : rank == 1 ? forced : other[rank-2];
            }
        }
    }
}

// Keys padded so the reduced product has lk/16 % 32 == 0 rows: Hexagon's HMX batched product
// takes only whole 32-row tiles of src0 and its softmax only 32-aligned rows; the padding is masked.
static int64_t llama_xattention_padded_keys(int64_t lk) {
    return GGML_PAD(lk, 512);
}

// The reversed-row indices and the reduced causal mask depend only on the ubatch, so a graph
// builds them once and every layer shares them.
static std::pair<ggml_tensor *, ggml_tensor *> llama_xattention_inputs(ggml_context * ctx, int64_t lq, int64_t hq, int64_t lk, ggml_tensor * mask) {
    auto * indices = ggml_custom_4d(ctx, GGML_TYPE_I32, lq, hq, 1, 1, nullptr, 0, llama_xattention_reverse_indices, 1, nullptr);
    ggml_tensor * mask_args[] = {mask};
    auto * reduced = ggml_custom_4d(ctx, GGML_TYPE_F32, llama_xattention_padded_keys(lk)/16, lq/16, 1, 1, mask_args, 1,
                                    llama_xattention_reduce_mask, 1, nullptr);
    return {indices, reduced};
}

static ggml_tensor * llama_xattention_build(
        ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * mask, ggml_tensor * indices, ggml_tensor * reduced, float scale, int layer) {
    const int64_t d = q->ne[0], lq = q->ne[1], lk = k->ne[1];
    const int64_t hq = q->ne[2], hkv = k->ne[2], groups = hq/hkv;
    const int bs = llama_xattention_block_size();
    GGML_ASSERT(lq % bs == 0 && lk % bs == 0 && hq % hkv == 0);
    GGML_ASSERT(q->ne[3] == 1 && k->ne[3] == 1 && mask && mask->ne[2] == 1 && mask->ne[3] == 1);
    const int64_t lkp = llama_xattention_padded_keys(lk);
    const int64_t nbq = lq/bs, nbk = lk/bs, nbkp = lkp/bs, pool = bs/16;
    GGML_ASSERT(reduced->ne[0] == lkp/16 && reduced->ne[1] == lq/16);
    auto * reversed = ggml_get_rows(ctx, q, indices);
    auto * rq = ggml_reshape_3d(ctx, reversed, 16*d, lq/16, hq);
    auto * kc = ggml_cont(ctx, k);
    if (lkp != lk) {
        auto * zeros = ggml_fill(ctx, ggml_reshape_3d(ctx, ggml_view_1d(ctx, kc, d*(lkp-lk)*hkv, 0), d, lkp-lk, hkv), 0.0f);
        kc = ggml_concat(ctx, kc, zeros, 1);
    }
    auto * rk = ggml_reshape_3d(ctx, kc, 16*d, lkp/16, hkv);
    auto * product = ggml_mul_mat(ctx, rk, rq);
    ggml_mul_mat_set_prec(product, GGML_PREC_F32);
    auto * probabilities = ggml_soft_max_ext(ctx, product, reduced, scale/16.0f, 0.0f);
    auto * key_sum = ggml_sum_rows(ctx, ggml_reshape_4d(ctx, probabilities, pool, nbkp, lq/16, hq));
    auto * query_rows = ggml_permute(ctx, ggml_reshape_4d(ctx, key_sum, nbkp, pool, nbq, hq), 1, 0, 2, 3);
    auto * query_sum = ggml_sum_rows(ctx, ggml_cont(ctx, query_rows));
    ggml_format_name(query_sum, "xattn_scores-%d", layer);
    auto * scores = ggml_reshape_3d(ctx, query_sum, nbkp, nbq, hq);
    if (nbkp != nbk) {
        scores = ggml_cont(ctx, ggml_view_3d(ctx, scores, nbk, nbq, hq, scores->nb[1], scores->nb[2], 0));
    }
    const bool shared = groups > 1 && llama_xattention_gqa_union();
    const int64_t lists = shared ? hkv : hq;
    ggml_tensor * selection_args[] = {scores, mask};
    // rows are independent: the cumulative selection uses every CPU thread of the backend
    auto * membership = ggml_custom_4d(ctx, GGML_TYPE_F32, nbk, nbq, lists, 1, selection_args, 2, llama_xattention_select, GGML_N_TASKS_MAX, nullptr);
    if (llama_xattention_select_npu()) {
        // HTP_XATTN_SELECT_P_MAGIC / _THRESHOLD (ggml-hexagon htp-ops.h): slots past the custom-op header
        const float threshold = (float) llama_xattention_threshold();
        GGML_ASSERT((double) threshold == llama_xattention_threshold());
        membership->op_params[12] = 0x58534c54;
        memcpy(&membership->op_params[13], &threshold, sizeof(threshold));
    }
    ggml_format_name(membership, "xattn_membership-%d", layer);
    return llama_block_sparse_attn(ctx, q, k, v, mask, membership, bs, bs, scale, layer, "xattn");
}

static ggml_tensor * llama_xattention_build(
        ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * mask, float scale, int layer) {
    const auto inputs = llama_xattention_inputs(ctx, q->ne[1], q->ne[2], k->ne[1], mask);
    return llama_xattention_build(ctx, q, k, v, mask, inputs.first, inputs.second, scale, layer);
}

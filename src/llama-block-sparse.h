#pragma once

#include "ggml.h"

#include <cstring>

// Block-sparse attention from 0/1 block memberships, as the Hexagon kernel runs it: every selector that hands the
// kernel per-block lists (XAttention, src/llama-xattention.h) ends here.
//
// membership [nbk, nbq, lists] F32 0/1, one row per bq-query block: lists == hq is one list per query head and runs one
// FlashAttention per GQA lane (lane g holds heads g, g+groups, ...; the kernel takes one list per K/V head); lists ==
// hkv is one list per K/V head and one GQA FlashAttention. The kernel takes ranked lists (members first) plus counts:
// members sort as 2, other blocks as 1 and padding as 0 over a power-of-two width (the HTP sort is much slower at other
// lengths), and the first nbk entries are kept.
static ggml_tensor * llama_block_sparse_attn(
        ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask,
        ggml_tensor * membership, int64_t bs, int64_t bq, float scale, int layer, const char * prefix) {
    const int64_t d = q->ne[0], lq = q->ne[1], hq = q->ne[2], hkv = k->ne[2], groups = hq/hkv;
    const int64_t nbk = membership->ne[0], nbq = membership->ne[1], lists = membership->ne[2];
    GGML_ASSERT(lists == hq || lists == hkv);
    GGML_ASSERT(nbq*bq == lq);
    int64_t width = 32;
    while (width < nbk) { width *= 2; }
    auto * key = ggml_scale_bias(ctx, membership, 1.0f, 1.0f);
    if (width != nbk) { key = ggml_pad(ctx, key, int(width-nbk), 0, 0, 0); }
    auto * ranked = ggml_argsort(ctx, key, GGML_SORT_ORDER_DESC);
    ranked = ggml_view_3d(ctx, ranked, nbk, nbq, lists, ranked->nb[1], ranked->nb[2], 0);
    auto * counts = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, membership), nbq, lists);
    ggml_format_name(counts, "%s_counts-%d", prefix, layer);
    if (lists == hkv) {
        auto * out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_sparse(out, ranked, bs, bq);
        ggml_flash_attn_ext_set_sparse_cnt(out, counts);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        ggml_format_name(out, "%s_fa-%d", prefix, layer);
        return out;
    }
    // each lane's output viewed as [DV, 1, hkv, lq], so concatenating on dim 1 restores head order
    ggml_tensor * joined = nullptr;
    for (int64_t lane = 0; lane < groups; ++lane) {
        auto * q_lane = ggml_view_3d(ctx, q, d, lq, hkv, q->nb[1], groups*q->nb[2], lane*q->nb[2]);
        auto * selection = ggml_view_3d(ctx, ranked, nbk, nbq, hkv, ranked->nb[1], groups*ranked->nb[2], lane*ranked->nb[2]);
        auto * count = ggml_view_2d(ctx, counts, nbq, hkv, groups*counts->nb[1], lane*counts->nb[1]);
        auto * partial = ggml_flash_attn_ext(ctx, q_lane, k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_sparse(partial, selection, bs, bq);
        ggml_flash_attn_ext_set_sparse_cnt(partial, count);
        ggml_flash_attn_ext_set_prec(partial, GGML_PREC_F32);
        ggml_format_name(partial, "%s_fa_%d-%d", prefix, int(lane), layer);
        auto * part = ggml_reshape_4d(ctx, partial, v->ne[0], 1, hkv, lq);
        joined = joined ? ggml_concat(ctx, joined, part, 1) : part;
    }
    return ggml_reshape_3d(ctx, joined, v->ne[0], hq, lq);
}

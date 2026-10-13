#pragma once

// The workload-aware GPU budget of the heterogeneous prefill split, applied per (256-query tile, KV head) inside the
// kernels instead of as graph ops. Plain C (no HVX) so the host can test it against the graph formulation.
//
// One tile's exception pairs (j, b): sub-block j (of R) selected key block b and the tile's shared list did not. The GPU
// takes at most `budget` of them, ranked singles first (blocks exactly one sub-block of the tile wants: on the NPU each
// would cost a whole 256 x 64 block for one pair), then the rest, each in the order below; the others run on the NPU,
// masked to the sub-blocks that selected them. budget = number of ranks r with r <= k * R * |shared| + 0.5 in exact
// arithmetic. Both the rule and the graph count a block's wants over the exception membership itself, so they agree with
// LLAMA_SPARSE_ATTN_EXC_THR too.
//
// Orders (fold header budget_order, GGML_HEXAGON_FA_FOLD_BUDGET_ORDER):
//   SB:          (sub-block, block). src/llama-graph.cpp computes this one with cumsums (LLAMA_SPARSE_ATTN_GPU_SHARE).
//                A cap that ends inside the non-singles leaves one pair of many blocks on the GPU while the NPU runs
//                each of those blocks whole for the other pairs.
//   BLOCK:       (block, sub-block): a block's pairs get consecutive ranks, so the cap cuts at most one block per tile.
//   BLOCK_WHOLE: BLOCK, and the block the cap cuts goes to the NPU whole. Its GPU pairs cost the NPU nothing (it runs
//                the whole 256 x 64 block anyway), so the GPU keeps fewer pairs than the budget.

#include <stdbool.h>
#include <stdint.h>

#define FA_BUDGET_MAX_R   4
#define FA_BUDGET_MAX_NBK 512   // key blocks of 64: 32k keys

#define FA_BUDGET_ORDER_SB          0u
#define FA_BUDGET_ORDER_BLOCK       1u
#define FA_BUDGET_ORDER_BLOCK_WHOLE 2u

typedef struct {
    uint16_t j;   // sub-block within the tile
    uint16_t b;   // key block
} fa_budget_pair;

// round(k * r * n_shared) with ties going down, k in thousandths: what "rank <= cap + 0.5" admits for an exact cap.
static inline uint32_t fa_budget_pairs(uint32_t budget_milli, uint32_t r, uint32_t n_shared) {
    const uint64_t x = (uint64_t) budget_milli * r * n_shared;
    return x < 500u ? 0u : (uint32_t) ((x + 499u) / 1000u);
}

// Split one tile. em_rows: row j of the membership at em_rows + j * row_stride (32-bit words; any nonzero bit pattern is a
// member, as the DSP's and the GPU's readers test it). Writes the NPU pairs to npu[] (capacity r * nbk) in ranking order
// ((j, b) for SB, (b, j) otherwise) and returns how many; *n_gpu gets the number of pairs the GPU keeps. An unknown order
// ranks as SB. Returns 0 with *n_gpu = 0 on bad sizes.
static inline uint32_t fa_budget_split(const uint32_t * em_rows, uint32_t row_stride, uint32_t r, uint32_t nbk,
                                       uint32_t budget, uint32_t order, fa_budget_pair * npu, uint32_t * n_gpu) {
    *n_gpu = 0;
    if (r == 0 || r > FA_BUDGET_MAX_R || nbk == 0 || nbk > FA_BUDGET_MAX_NBK) {
        return 0;
    }
    uint8_t  wants[FA_BUDGET_MAX_NBK];   // sub-blocks of the tile that want block b
    uint32_t n1 = 0;                     // singles: blocks exactly one sub-block wants (one pair each)
    for (uint32_t b = 0; b < nbk; ++b) {
        uint32_t c = 0;
        for (uint32_t j = 0; j < r; ++j) {
            c += em_rows[(size_t) j * row_stride + b] != 0u;
        }
        wants[b] = (uint8_t) c;
        n1 += c == 1u;
    }
    const uint32_t b1 = budget < n1 ? budget : n1;   // singles the GPU keeps
    const uint32_t b2 = budget > n1 ? budget - n1 : 0u;   // other pairs the GPU keeps
    uint32_t rank1 = 0, rank2 = 0, n_npu = 0, kept = 0;
    if (order == FA_BUDGET_ORDER_BLOCK || order == FA_BUDGET_ORDER_BLOCK_WHOLE) {
        for (uint32_t b = 0; b < nbk; ++b) {
            const uint32_t w = wants[b];
            if (w == 0u) {
                continue;
            }
            // the GPU keeps the block's first g pairs: ranks rank2 + 1 .. rank2 + w, those <= b2
            uint32_t g;
            if (w == 1u) {
                g = ++rank1 <= b1 ? 1u : 0u;
            } else {
                g = rank2 >= b2 ? 0u : (b2 - rank2 < w ? b2 - rank2 : w);
                if (g < w && order == FA_BUDGET_ORDER_BLOCK_WHOLE) {
                    g = 0u;
                }
                rank2 += w;
            }
            for (uint32_t j = 0; j < r; ++j) {
                if (em_rows[(size_t) j * row_stride + b] == 0u) {
                    continue;
                }
                if (g) {
                    g--;
                    kept++;
                } else {
                    npu[n_npu].j = (uint16_t) j;
                    npu[n_npu].b = (uint16_t) b;
                    n_npu++;
                }
            }
        }
        *n_gpu = kept;
        return n_npu;
    }
    for (uint32_t j = 0; j < r; ++j) {
        const uint32_t * row = em_rows + (size_t) j * row_stride;
        for (uint32_t b = 0; b < nbk; ++b) {
            if (row[b] == 0u) {
                continue;
            }
            const bool gpu = wants[b] == 1u ? ++rank1 <= b1 : ++rank2 <= b2;
            if (gpu) {
                kept++;
            } else {
                npu[n_npu].j = (uint16_t) j;
                npu[n_npu].b = (uint16_t) b;
                n_npu++;
            }
        }
    }
    *n_gpu = kept;
    return n_npu;
}

#ifndef HTP_FLASH_ATTN_OPS_H
#define HTP_FLASH_ATTN_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "hex-fastdiv.h"
#include "hex-common.h"
#include "htp-vtcm.h"

#ifdef __cplusplus
extern "C" {
#endif

// Tile constants (mirrored from hmx-utils.h for use on host side if needed)
#define HTP_FA_HMX_TILE_SIZE   2048
#define HMX_FP16_TILE_SIZE     2048
#define HMX_FP16_TILE_N_ROWS   32
#define HMX_FP16_TILE_N_COLS   32
#define HMX_FP16_TILE_N_ELMS   1024

#define HVX_FA_DMA_CACHE_SIZE  128
#define HMX_FA_DMA_CACHE_SIZE  4

// Upper bound on the KV block count a residency map can cover. Bounds the two
// push-side bitmaps carried in hmx_fa_context (FA_RES_MAX_BLOCKS/8 bytes each).
// At the usual sparse_bs of 64 this is a KV length of 32768.
#define FA_RES_MAX_BLOCKS      512

// How the device treats the KV block residency map. Rides in kernel params so the
// same binary can be A/B'd; 0 is OFF so a host that never sets it keeps the old path.
enum htp_fa_res_mode {
    HTP_FA_RES_OFF  = 0,  // never build the map
    HTP_FA_RES_AUTO = 1,  // build it when the selection actually repeats blocks
    HTP_FA_RES_ON   = 2,  // build it whenever it fits, skipping the duplication probe
    HTP_FA_RES_MISS = 3   // debug: keep the slot plumbing, never claim a hit
};


#define HTP_FA_M_INITIAL_VAL  -10000.0f

enum htp_fa_kernel_type {
    HTP_FA_KERNEL_UNSUPPORTED = 0,
    HTP_FA_KERNEL_HVX,
    HTP_FA_KERNEL_HMX
};

struct htp_fa_kernel_params {
    uint8_t  kernel_type;        // enum htp_fa_kernel_type
    uint8_t  is_q_fp32;          // 1 = Q type is F32, 0 = F16
    uint8_t  is_dst_fp32;        // 1 = dst type is F32, 0 = F16
    uint8_t  n_threads;          // Number of threads to run

    // Common parameters
    uint16_t Br;
    uint16_t Bc;
    uint16_t n_kv_blocks;        // also HVX's n_blocks
    uint16_t G;                  // GQA factor (n_heads / n_kv_heads)

    float    scale;
    float    max_bias;
    float    logit_softcap;
    uint32_t vtcm_size;

    uint32_t qrows;
    uint32_t qrows_per_thread;
    float    m0;
    float    m1;
    uint32_t n_head_log2;

    struct fastdiv_values src3_div2;
    struct fastdiv_values src3_div3;

    struct fastdiv_values broadcast_rk2;
    struct fastdiv_values broadcast_rk3;
    struct fastdiv_values broadcast_rv2;
    struct fastdiv_values broadcast_rv3;

    union {
        struct {
            uint32_t g_br;
            // row_buf_stride / mask_buf_row_stride used to live here. Both were written
            // by the host and never read on device -- the kernel takes them from the
            // layout it rebuilds itself (factx.row_buf_stride / .mask_buf_row_stride).
            // Reclaimed to carry the sparse selection geometry, which the kernel needs
            // in order to address m selected blocks inside one Bc-wide chunk.
            uint16_t sparse_bs;          // selection block size; 0 = dense
            uint16_t n_sel;              // selected blocks per selection row; 0 = dense
            // Query-block size the sel[] rows are expressed in (op_params[5]). This is
            // the SCORER's block size, not the kernel's Br -- the kernel picks row
            // q_start/sel_bq. 0 = shared selection (sel->ne[1] == 1), the legacy layout.
            uint16_t sel_bq;
            uint16_t res_mode;           // enum htp_fa_res_mode (KV block residency)
            int32_t  mask_broadcast;
            int32_t  pipeline;
            struct fastdiv_values div_G;
        } hmx;
        struct {
            uint32_t size_q_row_padded;
            uint32_t size_k_row_padded;
            uint32_t size_v_row_padded;
            struct fastdiv_values src0_div21;
            struct fastdiv_values src0_div1;
        } hvx;
    } u;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_fa_kernel_params) <= 128, "htp_fa_kernel_params is too large for kernel_params blob");
#endif

// VTCM region layout for the HMX flash-attention kernel.
//
// Single source of truth for both the host (which needs the total size to pick a
// (Br, Bc) tiling that fits the VTCM budget) and the device (which needs the actual
// byte offsets to place each scratch buffer). Building the layout once and reading
// offsets/total from it makes host estimate and device allocation impossible to
// desync -- previously they were duplicated formulas in two files and drifted.
//
// All fields are byte offsets / byte sizes -- no HVX_Vector type is named here so the
// header stays host-includable. The device casts (base + off_*) to the proper type.
// An offset of 0 marks a region that is not allocated for this configuration (only
// off_v_tiles[1], which exists only when pipelining); the device sets such pointers NULL.
struct hmx_fa_vtcm_layout {
    // Byte offsets from vtcm_base for each region.
    size_t off_q_tiles;
    size_t off_q_dma;
    size_t off_o_tiles[2];
    size_t off_k_fp16[2];
    size_t off_v_fp16[2];
    size_t off_k_tiles[2];
    size_t off_v_tiles[2];
    size_t off_s_tiles[2];
    size_t off_p_tiles[2];
    size_t off_d_tiles[2];
    size_t off_d_inv_l;
    size_t off_m_vec;
    size_t off_l_vec;
    size_t off_s_rowmax;
    size_t off_p_rowsum;
    size_t off_row_bufs;
    size_t off_hmx_scales_id;
    size_t off_hmx_scales_qk;
    size_t off_mask_buf;
    size_t off_slopes;

    // Region byte sizes reused by the device at runtime (not just for allocation).
    size_t q_tile_bytes;
    size_t o_tile_bytes;
    size_t s_tile_bytes;       // S and P tiles (same size)
    size_t d_tile_bytes;       // d_tiles[0..1] + d_inv_l, allocated back to back
    size_t m_line_bytes;       // one mask row
    size_t m_buf_slot_bytes;   // one dma_cache slot = align_up(Br * m_line_bytes, 4096)
    size_t col_vec_bytes;

    // Derived strides.
    size_t row_buf_stride;       // HVX vectors (128B) per row buffer
    size_t mask_buf_row_stride;  // __fp16 elements per row in the mask buffer

    bool   pipeline;
    size_t total_bytes;
};

// Build the VTCM layout.

static inline void hmx_fa_vtcm_layout_build(struct hmx_fa_vtcm_layout * L,
                                       size_t gqa_factor, size_t DK, size_t DV,
                                       size_t Br, size_t Bc, size_t n_threads, bool pipeline, bool is_q_fp32,
                                       bool mask_per_head) {
    const size_t g_br         = hex_align_up(gqa_factor * Br, HMX_FP16_TILE_N_ROWS);
    const size_t q_tile_size  = hex_align_up(g_br * DK   * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
    const size_t o_tile_size  = hex_align_up(g_br * DV   * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
    const size_t k_tile_size  = hex_align_up(Bc   * DK   * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
    const size_t v_tile_size  = hex_align_up(Bc   * DV   * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
    const size_t s_tile_size  = hex_align_up(g_br * Bc   * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);

    // The rescale matrices are diagonal: the HMX kernels only ever load the g_br/32
    // tiles that sit on the diagonal, so store just those, packed back to back with
    // a stride of one tile.  The old [g_br, g_br] square layout allocated g_br/32
    // times more than it used, which is also why a second D buffer was unaffordable.
    const size_t d_tile_size = (g_br / HMX_FP16_TILE_N_ROWS) * HTP_FA_HMX_TILE_SIZE;

    const size_t q_dma_size   = hex_align_up(g_br * DK * (is_q_fp32 ? sizeof(float) : sizeof(__fp16)), 128);
    const size_t k_dma_size   = hex_align_up(Bc * hex_round_up(DK * sizeof(__fp16), 128), 128);
    const size_t v_dma_size   = hex_align_up(Bc * hex_round_up(DV * sizeof(__fp16), 128), 128);
    const size_t col_vec_size = hex_align_up(g_br * sizeof(float),  256);
    const size_t row_vec_size = hex_align_up(Bc   * sizeof(__fp16), 256);
    const size_t m_line_size  = hex_align_up(Bc   * sizeof(__fp16), 128);
    const size_t m_buf_slot   = hex_align_up(Br * m_line_size, 256);
    // A broadcast mask stages one line per query row and is cycled through the
    // dma_cache's slots. A per-head mask stages gqa_factor lines per query row, and
    // the pipelined loop prefetches one block ahead, so it needs two such buffers --
    // otherwise the prefetch lands on the block the softmax is still reading.
    const size_t m_buf_slots  = mask_per_head ? (2 * gqa_factor) : HMX_FA_DMA_CACHE_SIZE;
    const size_t m_buf_size   = m_buf_slot * m_buf_slots;
    const size_t slopes_size  = hex_align_up(g_br * sizeof(__fp16), 128);

    size_t off = 0;

    // Group A (Part 1 - HMX Tiled buffers)
    VTCM_LAYOUT_ALLOC(off, off_q_tiles,       q_tile_size);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[0],    o_tile_size);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[1],    o_tile_size);
    VTCM_LAYOUT_ALLOC(off, off_d_tiles[0],    d_tile_size);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_d_tiles[1], d_tile_size, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_d_inv_l,       d_tile_size);

    // Group B & C share start offset (Group B tiles must be 2KB aligned)
    size_t off_group_b_c = hex_align_up(off, HTP_FA_HMX_TILE_SIZE);

    // Group B: Compute-only buffers
    size_t off_group_b = off_group_b_c;
    VTCM_LAYOUT_ALLOC(off_group_b, off_k_tiles[0],    k_tile_size);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_k_tiles[1], k_tile_size, pipeline);
    VTCM_LAYOUT_ALLOC(off_group_b, off_v_tiles[0],    v_tile_size);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_v_tiles[1], v_tile_size, pipeline);
    VTCM_LAYOUT_ALLOC(off_group_b, off_s_tiles[0],    s_tile_size);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_s_tiles[1], s_tile_size, pipeline);
    VTCM_LAYOUT_ALLOC(off_group_b, off_p_tiles[0],    s_tile_size);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_p_tiles[1], s_tile_size, pipeline);
    VTCM_LAYOUT_ALLOC(off_group_b, off_s_rowmax,      col_vec_size);
    VTCM_LAYOUT_ALLOC(off_group_b, off_p_rowsum,      col_vec_size);
    VTCM_LAYOUT_ALLOC(off_group_b, off_row_bufs,      row_vec_size * 2 * n_threads);

    const size_t group_b_size = off_group_b - off_group_b_c;

    // Group C: Q fetch DMA buffer
    size_t off_group_c = off_group_b_c;
    VTCM_LAYOUT_ALLOC(off_group_c, off_q_dma,         q_dma_size);

    const size_t group_c_size = off_group_c - off_group_b_c;

    off = off_group_b_c + hex_smax(group_b_size, group_c_size);

    // Group A (Part 2 - remaining non-HMX buffers)
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[0],     k_dma_size);
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[1],     k_dma_size);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[0],     v_dma_size);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[1],     v_dma_size);
    VTCM_LAYOUT_ALLOC(off, off_m_vec,         col_vec_size);
    VTCM_LAYOUT_ALLOC(off, off_l_vec,         col_vec_size);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_id, 256);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_qk, 256);
    VTCM_LAYOUT_ALLOC(off, off_mask_buf,      m_buf_size);
    VTCM_LAYOUT_ALLOC(off, off_slopes,        slopes_size);

    L->q_tile_bytes        = q_tile_size;
    L->o_tile_bytes        = o_tile_size;
    L->col_vec_bytes       = col_vec_size;
    L->s_tile_bytes        = s_tile_size;
    // Measured from the actual offsets rather than assumed to be N * d_tile_size, so
    // that inserting a region between them (or adding padding to VTCM_LAYOUT_ALLOC)
    // cannot silently leave the tail of the run unzeroed.
    L->d_tile_bytes        = (L->off_d_inv_l + d_tile_size) - L->off_d_tiles[0];
    L->m_line_bytes        = m_line_size;
    L->m_buf_slot_bytes    = m_buf_slot;
    L->row_buf_stride      = row_vec_size / 128;
    L->mask_buf_row_stride = m_line_size / sizeof(__fp16);
    L->pipeline            = pipeline;
    L->total_bytes         = off;
}

// Exact VTCM usage for a given (gqa_factor, DK, DV, Br, Bc) configuration.
static inline size_t hmx_fa_compute_vtcm_usage(size_t gqa_factor, size_t DK, size_t DV, size_t Br, size_t Bc, size_t n_threads, bool pipeline, bool is_q_fp32, bool mask_per_head) {
    struct hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, gqa_factor, DK, DV, Br, Bc, n_threads, pipeline, is_q_fp32, mask_per_head);
    return L.total_bytes;
}

// Optional KV block residency region: one dedicated slot per SELECTED KV block, so a
// block that several query blocks of the same KV head select is DMA'd once per head
// instead of once per query block. The slots hold raw DMA'd fp16 rows with exactly the
// staging buffers' row stride -- not HMX tiles -- so nothing downstream of the
// scatter-transpose changes.
//
// Deliberately NOT part of hmx_fa_vtcm_layout_build / hmx_fa_compute_vtcm_usage. The
// (Br, Bc) search and the host's admissibility test must stay bit-identical to what
// they were without residency, so this region is carved RESIDUALLY out of whatever
// VTCM the chosen tiling happened to leave over. If it does not fit, n_slots comes
// back 0, the feature turns itself off, and every code path is the one that ran
// before -- which is what keeps a shape that cannot afford the map from silently
// getting a different tiling.
#define FA_RES_ALIGN  2048   // symmetry with the HMX tile regions; 128 is the hard floor
#define FA_RES_SLACK  4096   // keep a page clear of the VTCM end

struct hmx_fa_res_region {
    size_t off_k;
    size_t off_v;
    size_t k_slot_bytes;   // sparse_bs * hex_round_up(DK * 2, 128)
    size_t v_slot_bytes;   // sparse_bs * hex_round_up(DV * 2, 128)
    size_t n_slots;        // 0 = disabled / does not fit
};

static inline void hmx_fa_res_region_build(struct hmx_fa_res_region * R,
                                           size_t layout_total,
                                           size_t vtcm_budget,
                                           size_t n_blocks,
                                           size_t bs,
                                           size_t k_row_padded,
                                           size_t v_row_padded) {
    R->off_k        = 0;
    R->off_v        = 0;
    R->k_slot_bytes = bs * k_row_padded;
    R->v_slot_bytes = bs * v_row_padded;
    R->n_slots      = 0;

    const size_t slot = R->k_slot_bytes + R->v_slot_bytes;
    const size_t off  = hex_align_up(layout_total, FA_RES_ALIGN);

    if (n_blocks == 0 || slot == 0 || off + FA_RES_SLACK >= vtcm_budget) {
        return;
    }

    // All-or-nothing: a partially resident map would mix slot addresses with staging
    // addresses in one chunk, and that mixed path has no test coverage yet.
    const size_t free_bytes = vtcm_budget - off - FA_RES_SLACK;
    if (free_bytes / slot < n_blocks) {
        return;
    }

    R->off_k   = off;
    R->off_v   = off + n_blocks * R->k_slot_bytes;
    R->n_slots = n_blocks;
}

#define FA_HVX_BLOCK_SIZE 64

static inline size_t hvx_fa_compute_vtcm_usage(size_t DK, size_t DV, bool is_q_fp32, bool has_mask, size_t n_threads) {
    const size_t size_q_row_padded = hex_round_up(DK * (is_q_fp32 ? 4 : 2), 128);
    const size_t size_k_row_padded = hex_round_up(DK * sizeof(__fp16), 128);
    const size_t size_v_row_padded = hex_round_up(DV * sizeof(__fp16), 128);

    const size_t size_q_block = size_q_row_padded * 1;
    const size_t size_k_block = size_k_row_padded * FA_HVX_BLOCK_SIZE;
    const size_t size_v_block = size_v_row_padded * FA_HVX_BLOCK_SIZE;
    const size_t size_m_block = hex_round_up(FA_HVX_BLOCK_SIZE * sizeof(__fp16), 128);
    const size_t size_vkq_acc = hex_round_up(DV * sizeof(float), 128);

    const size_t size_per_thread = size_q_block * 1
                                 + size_k_block * 2
                                 + size_v_block * 2
                                 + (has_mask ? size_m_block * HVX_FA_DMA_CACHE_SIZE : 0)
                                 + size_vkq_acc;

    return size_per_thread * n_threads;
}

#define FA_MIN_KV_BLOCKS 3

// Max selected blocks folded into one FA chunk. Bounded so the extra in-flight DMA
// descriptors (m per tensor) stay well inside the 256-entry queue.
#define FA_SPARSE_MAX_M  8

// Cost-based (Br, Bc) search for flash attention with pipeline constraint.
//
// Bc_fixed pins the KV chunk size instead of searching for it (0 = search).
// Block-sparse attention needs this: its KV block indices are expressed in
// units chosen by the graph, so the kernel must chunk on exactly that unit.
// Only Br is searched in that case, and the call fails if the pinned Bc does
// not fit the VTCM budget for any Br.
//
// br_align is the symmetric constraint on the QUERY axis, needed only when the
// selection has a query-block axis: a kernel query tile [q_start, q_start+Br) must
// not straddle a scorer query block, which holds exactly when Br divides the block
// size. Note the direction -- Br | Bq, not Bq | Br: a tile that spans several query
// blocks has aligned edges but still gives all its rows one selection, which is the
// same silent failure. The search returns -1 when no legal Br fits, so the op is
// rejected and drops to a dense backend rather than computing the wrong attention.
static inline int hmx_fa_find_chunk_size(size_t * Br_out,
                                  size_t * Bc_out,
                                  size_t   gqa_factor,
                                  size_t   DK,
                                  size_t   DV,
                                  size_t   qo_len,
                                  size_t   kv_len,
                                  size_t   vtcm_budget,
                                  size_t   n_threads,
                                  bool     is_q_fp32,
                                  size_t   bc_step,   // Bc must be a multiple of this (0 = bc_unit)
                                  size_t   bc_cap,    // upper bound on Bc (0 = none)
                                  size_t   sel_blocks,// selected blocks; Bc/bc_step must divide it (0 = n/a)
                                  size_t   br_align,  // Br must DIVIDE this (0 = unconstrained)
                                  bool     mask_per_head) {
    const size_t T       = HMX_FP16_TILE_N_ROWS;  // 32
    const size_t br_unit = hmx_ceil_div(T, gqa_factor);
    const size_t bc_unit = HMX_FP16_TILE_N_COLS * 2;  // 64
    const bool   can_pipeline = (kv_len >= FA_MIN_KV_BLOCKS * bc_unit && n_threads >= 2);

    // Br_max: largest Br aligned to br_unit that does not exceed qo_len.
    const size_t Br_max = qo_len >= br_unit ? hex_align_down(qo_len, br_unit) : br_unit;
    // With a query-block axis, Br | br_align implies Br <= br_align, so start the walk
    // there instead of scanning candidates that can only be rejected below.
    const size_t Br_start = br_align ? hex_smin(Br_max, hex_align_down(br_align, br_unit)) : Br_max;

    // Pipeline constraint: cap Bc so n_kv_blocks >= FA_MIN_KV_BLOCKS. The exact bound
    // is ceil(kv_len/Bc) >= N  <=>  Bc < kv_len/(N-1), so take the largest bc_unit
    // multiple strictly below that. Deriving it from kv_len/N instead overshoots the
    // block count whenever kv_len/N is not bc_unit-aligned -- kv=512 yielded Bc=128
    // and 4 blocks where Bc=192 gives 3 -- and each extra KV block costs a
    // near-constant ~190us at nb=512, independent of Bc. The search below still walks
    // downward from this cap, so a candidate that does not fit VTCM just falls back to
    // the next smaller one.
    // Only relax when kv_len is too short to form enough blocks.
    const size_t Bc_search    = can_pipeline ? hex_align_down((kv_len - 1) / (FA_MIN_KV_BLOCKS - 1), bc_unit) :
                                               (kv_len >= bc_unit ? hex_align_down(kv_len, bc_unit) : bc_unit);
    // Block-sparse pins the SELECTION granularity, not the tiling: Bc may cover m
    // selected blocks, so it is searched in multiples of bs rather than fixed to it.
    const size_t step         = bc_step ? bc_step : bc_unit;
    const size_t Bc_capped    = bc_cap ? hex_smin(Bc_search, bc_cap) : Bc_search;
    // Clamp up to one step: when the pipeline cap lands below the selection block size
    // there is still exactly one legal candidate (Bc == bs, i.e. m == 1). Without this
    // the range collapses to empty, find_chunk_size fails, and the op silently falls
    // back to a dense CPU backend instead of running ungrouped.
    // Sparse only: when the pipeline cap lands below the selection block size there is
    // still exactly one legal candidate (Bc == bs, m == 1), and without the clamp the
    // range collapses to empty and the op silently falls back to CPU. Dense keeps the
    // original bound -- forcing a candidate there changes which kernel gets chosen for
    // short KV and measurably regresses the dense suite.
    const size_t Bc_limit     = bc_step ? hex_smax(step, (Bc_capped / step) * step)
                                        : (Bc_capped / step) * step;
    const size_t Bc_floor     = step;
    // Cost coefficients calibrated from profiling
    const size_t c_q_fixed    = 800;   // per-Q-block: q_load + epilogue o_update + o_norm + o_store
    const size_t c_iter_base  = 200;   // per-KV-iter base (HMX dot/update + DMA)
    const size_t c_softmax    = 600;   // per 64-row vector chunk on HVX

    size_t best_cost = SIZE_MAX, best_mn = 0, best_padded = SIZE_MAX;
    size_t best_Br = 0, best_Bc = 0;

    for (size_t Br = Br_start; Br >= br_unit; Br -= br_unit) {
        // A query tile must lie inside one scorer query block. br_unit is ceil(32/G),
        // so for G in {3,5,6,7} nothing divides a 64- or 128-wide query block and the
        // search legitimately finds no candidate at all -- the caller then rejects the
        // op, which is the fail-closed outcome.
        const bool br_ok = (br_align == 0) || ((br_align % Br) == 0);

        // Try all Bc candidates from Bc_limit down to Bc_floor
        for (size_t Bc = Bc_limit; br_ok && Bc >= Bc_floor; Bc -= step) {
            // Require m = Bc/bs to divide the selected-block count, so every chunk is
            // full. A ragged final chunk leaves stale rows in the K/V/mask staging
            // buffers past its true width, and nothing downstream re-derives that
            // width per chunk -- so allow only the evenly-dividing factorisations.
            if (bc_step && sel_blocks && ((sel_blocks % (Bc / bc_step)) != 0)) {
                continue;
            }
            size_t vtcm_needed = hmx_fa_compute_vtcm_usage(gqa_factor, DK, DV, Br, Bc, n_threads, can_pipeline, is_q_fp32, mask_per_head);
            if (vtcm_needed <= vtcm_budget) {
                // This Bc fits for this Br!
                const size_t q_blocks       = (qo_len + Br - 1) / Br;
                const size_t kv_blocks      = (kv_len + Bc - 1) / Bc;
                const size_t actual_threads = (kv_blocks >= 3 && n_threads >= 2) ? n_threads : 1;
                const size_t n_rows_g       = Br * gqa_factor;
                const size_t n_row_vec_cnt  = (n_rows_g + 63) / 64;
                const size_t n_use          = n_row_vec_cnt < actual_threads ? n_row_vec_cnt : actual_threads;
                const size_t vecs_per_t     = n_use > 0 ? (n_row_vec_cnt + n_use - 1) / n_use : 1;

                const size_t c_iter_actual  = c_iter_base + c_softmax * vecs_per_t;
                const size_t cost           = q_blocks * (c_q_fixed + kv_blocks * c_iter_actual);
                const size_t mn             = Br * Bc;

                // The KV width actually processed, ragged tail included: a partial last
                // block still costs a full Bc of softmax and DMA. The cost model counts
                // blocks but is blind to Bc, so two candidates with the same block count
                // tie -- and picking the larger Bc then loses to padding. At kv=768,
                // Bc=320 gives blocks of 320/320/128 (960 processed for 768 of work)
                // while Bc=256 gives 256/256/256 and measured 6% faster.
                const size_t kv_padded = kv_blocks * Bc;

                const bool better = (cost < best_cost) ||
                                    (cost == best_cost && kv_padded <  best_padded) ||
                                    (cost == best_cost && kv_padded == best_padded && mn > best_mn);
                if (better) {
                    best_cost   = cost;
                    best_mn     = mn;
                    best_padded = kv_padded;
                    best_Br     = Br;
                    best_Bc     = Bc;
                }
                // Keep scanning smaller Bc for this Br: they can never cost less (fewer
                // rows per block means more blocks) but they can tie with less padding.
            }
        }

        if (Br == br_unit) {
            break;
        }
    }

    if (best_Br == 0 || best_Bc == 0) {
        return -1;
    }

    *Br_out = best_Br;
    *Bc_out = best_Bc;
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* HTP_FLASH_ATTN_OPS_H */

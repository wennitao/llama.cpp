#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <assert.h>
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <math.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "hex-dma.h"
#include "hex-fastdiv.h"
#include "hex-profile.h"
#include "hmx-queue.h"
#include "hmx-utils.h"
#include "hvx-utils.h"
#include "hvx-dump.h"
#include "hvx-copy.h"
#include "hvx-reduce.h"
#include "hvx-flash-attn.h"
#include "htp-vtcm.h"
#include "work-queue.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"

#include "flash-attn-ops.h"
#include "hvx-fa-kernels.h"
#include "hmx-fa-kernels.h"

// Must be multiple of 32
#define FLASH_ATTN_BLOCK_SIZE (32 * 2)

struct htp_fa_context {
    const struct htp_ops_context * octx;

    struct fastdiv_values src0_div21;
    struct fastdiv_values src0_div1;

    struct fastdiv_values broadcast_rk2;
    struct fastdiv_values broadcast_rk3;
    struct fastdiv_values broadcast_rv2;
    struct fastdiv_values broadcast_rv3;

    struct fastdiv_values src3_div2;
    struct fastdiv_values src3_div3;

    float scale;
    float max_bias;
    __fp16 logit_softcap;

    uint32_t n_head_log2;
    float m0;
    float m1;
    __fp16 slopes[512];

    uint32_t n_blocks;

    size_t size_q_row_padded;
    size_t size_k_row_padded;
    size_t size_v_row_padded;

    size_t size_k_block;
    size_t size_v_block;
    size_t size_m_block;

    uint32_t qrows;
    uint32_t qrows_per_thread;

    bool is_q_fp32;

    size_t size_q_block;
    size_t size_vkq_acc;

    uint8_t * spad_q;
    uint8_t * spad_k;
    uint8_t * spad_v;
    uint8_t * spad_m;
    uint8_t * spad_a;

    uint64_t t_start;
};

struct hmx_fa_context {
    const struct htp_ops_context * octx;
    const struct htp_tensor *      sinks;  // attention sinks (src[4]), NULL if absent
    bool         pipeline;  // true when n_kv_blocks >= FA_MIN_KV_BLOCKS && n_threads >= 2
    uint32_t     n_threads;

    // Op parameters
    __fp16       scale;
    float        max_bias;
    __fp16       logit_softcap;
    uint32_t     n_head_log2;
    float        m0, m1;

    // Dimensions
    uint32_t     DK, DV;
    uint32_t     n_kv;        // kv_len
    uint32_t     n_kv_heads;  // number of KV heads
    uint32_t     n_heads;     // number of Q heads
    uint32_t     G;           // GQA factor = n_heads / n_kv_heads
    struct fastdiv_values div_G;
    struct fastdiv_values src3_div2;
    struct fastdiv_values src3_div3;
    uint32_t     n_kv_blocks;
    uint32_t     neq1;        // Q token count

    // Block-sparse selection (src[5]): list of KV block indices to attend to,
    // in units of Bc. NULL for dense attention, where block b is simply b.
    const int32_t * sel;
    uint32_t     sel_nb1;     // byte stride between per-query-block lists; 0 = shared
    uint32_t     sel_nb2;     // byte stride between per-KV-head lists
    uint32_t     sel_nb3;     // byte stride between per-sequence lists
    // Per-row selection length (src[6]; NULL = every row uses n_sel). F32 because it
    // is the natural output of an in-graph reduction; read scalar, truncated, clamped.
    const float * cnt;
    uint32_t     cnt_nb_qb;   // byte stride between per-query-block counts; 0 = shared
    uint32_t     cnt_nb_head; // byte stride between per-KV-head counts
    uint32_t     cnt_nb_seq;  // byte stride between per-sequence counts
    uint32_t     sel_nq;      // query-block rows in sel[] (sel->ne[1])
    struct fastdiv_values div_sel_bq;  // q_start -> query-block row, divides by sel_bq
    uint32_t     sparse_bs;   // selection block size (== Bc when dense)
    uint32_t     n_sel;       // selected blocks per (query block, kv_head, seq); 0 when dense
    uint32_t     m;           // selected blocks per chunk: sel ? Bc/sparse_bs : 1
    uint32_t     n_blk_total; // ceil(n_kv / sparse_bs) -- bound for clamping sel[]
    uint32_t     mask_slot_stride; // __fp16 elements per mask double-buffer slot
    bool         mask_use_cache;   // m == 1 && broadcast: keep the dma_cache fast path

    // ---- KV block residency (per-query-block sparse only; NULL disables everything) ----
    //
    // One VTCM slot per KV block INDEX, holding that block's raw DMA'd fp16 rows with
    // the same row stride as the staging buffers. Direct-mapped (slot == block index),
    // so within one (sequence, KV head) a slot is never reused for a different block:
    // there is no eviction, no replacement policy, and no way for an in-flight job to
    // reference a slot that is about to be rewritten.
    __fp16 *     k_res;
    __fp16 *     v_res;
    size_t       k_res_slot;       // bytes per K slot = sparse_bs * size_k_row_padded
    size_t       v_res_slot;       // bytes per V slot = sparse_bs * size_v_row_padded
    bool         res_force_miss;   // debug: exercise the slot plumbing, never claim a hit
    uint32_t     res_epoch_ib3;    // the (sequence, KV head) the valid bits describe
    uint32_t     res_epoch_kv_head;
    // Set at PUSH time, and read only by the pusher. K and V need SEPARATE bitmaps:
    // fa_push_chunk runs its K loop first, so one shared bitmap would let the V loop
    // see the K loop's bit and skip a transfer that never happened.
    uint32_t     k_res_valid[FA_RES_MAX_BLOCKS / 32];
    uint32_t     v_res_valid[FA_RES_MAX_BLOCKS / 32];

    // Types
    bool         is_q_fp32;
    bool         is_dst_fp32;

    // Dynamic block sizes
    uint32_t     Br;    // Q tokens per block (before GQA expansion)
    uint32_t     Bc;
    uint32_t     g_br;  // hex_align_up(G * Br, 32) - actual tile row dim

    // VTCM buffers (allocated by vtcm_seq_alloc)
    __fp16 *     vtcm_q_dma;           // Q DMA fetch buffer
    __fp16 *     vtcm_q_tiles;         // Q tile format [g_br, D]
    __fp16 *     vtcm_o_tiles[2];      // O ping-pong [g_br, D]
    __fp16 *     vtcm_k_fp16[2];       // K DMA double-buffer [Bc, D]
    __fp16 *     vtcm_v_fp16[2];       // V DMA double-buffer [Bc, D]
    __fp16 *     vtcm_k_tiles[2];      // K tiles (transposed, double-buffered)
    __fp16 *     vtcm_v_tiles[2];      // V tiles (column-major, double-buffered)
    __fp16 *     vtcm_s_tiles[2];      // S = QK^T [g_br, Bc] (double-buffered)
    __fp16 *     vtcm_p_tiles[2];      // P = softmax(S) [g_br, Bc]
    __fp16 *     vtcm_d_tiles[2];      // Diagonal rescale, g_br/32 packed diagonal tiles (double-buffered)
    __fp16 *     vtcm_d_inv_l;         // Diagonal rescale (1/l), same packed layout
    HVX_Vector * vtcm_m_vec;           // Row max [g_br]
    HVX_Vector * vtcm_l_vec;           // Row sum [g_br]
    HVX_Vector * vtcm_s_rowmax;        // Softmax intermediate [g_br]
    HVX_Vector * vtcm_p_rowsum;        // Softmax intermediate [g_br]
    HVX_Vector * vtcm_row_bufs;        // Per-thread softmax row scratch [n_threads][2][Bc/64]
    uint8_t *    vtcm_hmx_scales_id;   // HMX output scales (identity)
    uint8_t *    vtcm_hmx_scales_qk;   // HMX output scales (qk_scale)
    __fp16 *     vtcm_mask_buf;        // VTCM mask buffer [Br * m_line], DMA'd per KV block
    __fp16 *     vtcm_slopes;          // ALiBi slopes [g_br]
    size_t       row_buf_stride;       // HVX vectors per row buffer (Bc/64)
    size_t       mask_buf_row_stride;  // elements (__fp16) per row in mask buffer
    size_t       mask_buf_gqa_stride;  // __fp16 elements per per-head mask buffer (double-buffered)
    size_t       q_tile_bytes;
    size_t       o_tile_bytes;
    size_t       col_vec_bytes;
    size_t       d_tile_bytes;
    bool         mask_broadcast;       // true when mask->ne[2] == 1 (head-independent, single 2D DMA)
    dma_cache    m_cache;
};

// A "chunk" is one iteration of the KV loop: Bc rows staged into VTCM. Dense: the
// chunk is one contiguous run at c*Bc. Sparse: the chunk stitches together m selected
// blocks of sparse_bs rows each, named by sel[] and scattered anywhere in the KV range.
//
// INVARIANT: the kernel only ever works in chunk-LOCAL column indices. A chunk's m
// blocks are not contiguous in KV, so absolute KV position reaches the softmax solely
// through which mask columns were staged alongside them. Never reintroduce a
// (kv_start + column) term -- that is exactly the assumption chunking invalidates.

// Query-block row of sel[] for the query tile starting at q_start.
//
// sel_nb1 == 0 covers both dense and the shared-selection layout (sel->ne[1] == 1),
// so the fast path never touches div_sel_bq -- which is only initialised when the
// selection actually has a query axis.
//
// The clamp mirrors the index clamp in fa_chunk_block_start: the host validates
// sel->ne[1] against ceil(neq1/Bq), but a host/device disagreement must degrade to
// re-reading the last row, never to an out-of-bounds load.
static inline uint32_t fa_sel_row(const struct hmx_fa_context * factx, uint32_t q_start) {
    if (__builtin_expect(factx->sel_nb1 == 0, true)) {
        return 0;
    }
    return (uint32_t) hex_smin(fastdiv(q_start, &factx->div_sel_bq), factx->sel_nq - 1);
}

// Selection length of one (query block, KV head, sequence) row. Without a count
// tensor every row uses n_sel; with one, n_sel is the upper bound (u_max) and the row
// reads its own length. The count is device memory the host validates only by SHAPE,
// so it is truncated and clamped here -- an out-of-range value must degrade to a legal
// length, never to a chunk count the DMA FIFO and the loop bound disagree about.
static inline uint32_t fa_row_nsel(const struct hmx_fa_context * factx,
                                   uint32_t qb, uint32_t kv_head, uint32_t ib3) {
    if (__builtin_expect(factx->cnt == NULL, true)) {
        return factx->n_sel;
    }
    const float c = *(const float *) ((const uint8_t *) factx->cnt +
                                      qb      * factx->cnt_nb_qb +
                                      kv_head * factx->cnt_nb_head +
                                      ib3     * factx->cnt_nb_seq);
    int32_t n = (int32_t) c;
    if (n < 1) {
        n = 1;
    }
    if (n > (int32_t) factx->n_sel) {
        n = (int32_t) factx->n_sel;
    }
    return (uint32_t) n;
}

// Number of selected blocks in chunk c of one row. Dense: the chunk is itself one block.
//
// This IS a function of the query-block row (via row_nsel), which is safe for the
// untagged DMA FIFO for the same reason fa_push_chunk derives its own sel row: every
// push and pop site computes the length from the q_start of the TILE THE CHUNK SERVES,
// so producer and consumer agree by construction. What must never happen is one site
// using a cached "current" length while staging another tile's chunk -- the tail
// prefetch stages the next tile's chunk 0 while the consumer still pops the current
// tile's -- which is why the length is always re-derived from (qb, kv_head, ib3) and
// never carried across the loop seam.
static inline uint32_t fa_chunk_nblk(const struct hmx_fa_context * factx, uint32_t c,
                                     uint32_t row_nsel) {
    if (__builtin_expect(factx->sel == NULL, true)) {
        return 1;
    }
    const uint32_t base = c * factx->m;
    return base < row_nsel ? (uint32_t) hex_smin(factx->m, row_nsel - base) : 0;
}

// Index of the j'th selected block of chunk c, for query block qb.
//
// qb picks the row of the selection list: with a query axis every query block retrieves
// from its own set of KV blocks, which is the whole point of per-query-block selection.
// sel[] is device memory the host validates only by LENGTH, so the index is clamped
// here -- an out-of-range entry must not turn into an out-of-bounds DMA source.
//
// Sparse only; the dense path has no list. Split out of fa_chunk_block_start because
// the residency map keys on the INDEX, and dividing the row offset back down to
// recover it is both slower and a lie about where the number came from.
static inline uint32_t fa_chunk_block_idx(const struct hmx_fa_context * factx,
                                          uint32_t                      c,
                                          uint32_t                      j,
                                          uint32_t                      qb,
                                          uint32_t                      kv_head,
                                          uint32_t                      ib3) {
    const int32_t * list = (const int32_t *) ((const uint8_t *) factx->sel +
                                              qb      * factx->sel_nb1 +
                                              kv_head * factx->sel_nb2 +
                                              ib3     * factx->sel_nb3);
    uint32_t idx = (uint32_t) list[c * factx->m + j];
    if (idx >= factx->n_blk_total) {
        idx = factx->n_blk_total - 1;   // clamp the INDEX, never the row count: a zero
    }                                   // row count would break n_col_tiles > 0 downstream
    return idx;
}

// KV row offset of the j'th selected block of chunk c. Dense: blocks are walked in
// order, so chunk c starts at c * Bc.
static inline uint32_t fa_chunk_block_start(const struct hmx_fa_context * factx,
                                            uint32_t                      c,
                                            uint32_t                      j,
                                            uint32_t                      qb,
                                            uint32_t                      kv_head,
                                            uint32_t                      ib3) {
    if (__builtin_expect(factx->sel == NULL, true)) {
        return c * factx->Bc;
    }
    return fa_chunk_block_idx(factx, c, j, qb, kv_head, ib3) * factx->sparse_bs;
}

// Rows a block starting at KV row `start` contributes, clamped for a KV length that is
// not a multiple of the block size.
static inline uint32_t fa_rows_at(const struct hmx_fa_context * factx, uint32_t start, uint32_t n_kv) {
    const uint32_t span = factx->sel ? factx->sparse_bs : factx->Bc;
    return start < n_kv ? (uint32_t) hex_smin(span, n_kv - start) : 0;
}

// Rows contributed by the j'th block of chunk c, clamped for a KV length that is not
// a multiple of the block size.
static inline uint32_t fa_block_rows(const struct hmx_fa_context * factx,
                                     uint32_t c, uint32_t j, uint32_t qb, uint32_t kv_head,
                                     uint32_t ib3, uint32_t n_kv) {
    return fa_rows_at(factx, fa_chunk_block_start(factx, c, j, qb, kv_head, ib3), n_kv);
}

// ---- KV block residency bookkeeping -------------------------------------------------
//
// The bitmaps live entirely on the PUSH side. The consumer never consults them: it
// learns a block's VTCM address only from dma_queue_pop().dst, which carries the slot
// address verbatim for a hit and the freshly written slot for a miss.

// Returns true when this block is already staged in its slot, and marks it staged
// either way. A second push of the same block inside one epoch therefore emits a
// zero-work descriptor; FIFO order guarantees that descriptor is popped only after the
// real transfer ahead of it has been waited on, so no extra synchronisation is needed.
static inline bool fa_res_mark(uint32_t * bitmap, uint32_t idx, bool force_miss) {
    const uint32_t w   = idx >> 5;
    const uint32_t bit = 1u << (idx & 31);
    const bool     hit = (bitmap[w] & bit) != 0;
    bitmap[w] |= bit;
    return hit && !force_miss;
}

// A residency epoch is one (sequence, KV head): slots are keyed by block index alone,
// so they must be invalidated when the head changes.
//
// Cleared by the PUSHER, not at the top of the loop body. The tail prefetch stages the
// NEXT iteration's first chunk before that iteration begins, so a clear that ran when
// the consumer reached the new head would arrive one push too late and let those
// descriptors claim hits against the previous head's slots.
static inline void fa_res_begin_epoch(struct hmx_fa_context * factx, uint32_t ib3, uint32_t kv_head) {
    if (factx->k_res == NULL) {
        return;
    }
    if (factx->res_epoch_ib3 == ib3 && factx->res_epoch_kv_head == kv_head) {
        return;
    }
    memset(factx->k_res_valid, 0, sizeof(factx->k_res_valid));
    memset(factx->v_res_valid, 0, sizeof(factx->v_res_valid));
    factx->res_epoch_ib3     = ib3;
    factx->res_epoch_kv_head = kv_head;
}

// Total rows staged for chunk c, i.e. the KV width the kernel actually computes.
static inline uint32_t fa_chunk_rows(const struct hmx_fa_context * factx,
                                     uint32_t c, uint32_t qb, uint32_t kv_head,
                                     uint32_t ib3, uint32_t n_kv) {
    const uint32_t nblk = fa_chunk_nblk(factx, c, fa_row_nsel(factx, qb, kv_head, ib3));
    uint32_t rows = 0;
    for (uint32_t j = 0; j < nblk; ++j) {
        rows += fa_block_rows(factx, c, j, qb, kv_head, ib3, n_kv);
    }
    return rows;
}

// Chunk-level shims. fa_kv_block_start returns the FIRST selected block's offset, so it
// is only valid where a KV *position* is wanted (trace tags, mask/DMA base of block 0).
// Never derive a chunk's width from it: with m > 1 the chunk spans m scattered blocks
// and nek1 - first_start under-reports whenever that first block sits near the KV end.
// Use fa_chunk_rows for widths.
static inline uint32_t fa_kv_block_start(const struct hmx_fa_context * factx,
                                         uint32_t b, uint32_t qb, uint32_t kv_head, uint32_t ib3) {
    return fa_chunk_block_start(factx, b, 0, qb, kv_head, ib3);
}

static inline uint32_t fa_kv_block_rows(const struct hmx_fa_context * factx,
                                        uint32_t b, uint32_t qb, uint32_t kv_head, uint32_t ib3,
                                        uint32_t n_kv) {
    return fa_chunk_rows(factx, b, qb, kv_head, ib3, n_kv);
}

static void flash_attn_ext_f16_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_fa_context * factx = (struct htp_fa_context *) data;
    const struct htp_ops_context * octx = factx->octx;
    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * mask  = octx->src[3];
    const struct htp_tensor * sinks = octx->src[4];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t neq0 = q->ne[0];
    const uint32_t neq1 = q->ne[1];
    const uint32_t neq2 = q->ne[2];
    const uint32_t neq3 = q->ne[3];

    const uint32_t nek0 = k->ne[0];
    const uint32_t nek1 = k->ne[1];
    const uint32_t nek2 = k->ne[2];
    const uint32_t nek3 = k->ne[3];

    const uint32_t nev0 = v->ne[0];
    const uint32_t nev1 = v->ne[1];
    const uint32_t nev2 = v->ne[2];
    const uint32_t nev3 = v->ne[3];

    const uint32_t nbq1 = q->nb[1];
    const uint32_t nbq2 = q->nb[2];
    const uint32_t nbq3 = q->nb[3];

    const uint32_t nbk1 = k->nb[1];
    const uint32_t nbk2 = k->nb[2];
    const uint32_t nbk3 = k->nb[3];

    const uint32_t nbv1 = v->nb[1];
    const uint32_t nbv2 = v->nb[2];
    const uint32_t nbv3 = v->nb[3];

    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];
    const uint32_t ne3 = dst->ne[3];

    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    // total rows in q
    const uint32_t nr = factx->qrows;
    const uint32_t dr = factx->qrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, nr);

    if (ir0 >= ir1) return;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    dma_queue * dma = octx->ctx->dma[ith];

    const uint32_t DK = nek0;
    const uint32_t DV = nev0;

    const size_t size_q_row = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    const size_t size_k_row = DK * sizeof(__fp16);
    const size_t size_v_row = DV * sizeof(__fp16);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = factx->spad_q + factx->size_q_block * ith;
    uint8_t * spad_k = factx->spad_k + factx->size_k_block * 2 * ith;
    uint8_t * spad_v = factx->spad_v + factx->size_v_block * 2 * ith;
    uint8_t * spad_m = factx->spad_m + (mask ? factx->size_m_block * HVX_FA_DMA_CACHE_SIZE : 0) * ith;
    uint8_t * spad_a = factx->spad_a + factx->size_vkq_acc * ith;

    dma_cache m_cache;
    dma_cache_init(&m_cache, spad_m, factx->size_m_block, HVX_FA_DMA_CACHE_SIZE);

    for (uint32_t ir = ir0; ir < ir1; ++ir) {
        const uint32_t iq3 = fastdiv(ir, &factx->src0_div21);
        const uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &factx->src0_div1);
        const uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        const uint32_t ik3 = fastdiv(iq3, &factx->broadcast_rk3);
        const uint32_t ik2 = fastdiv(iq2, &factx->broadcast_rk2);

        const uint32_t iv3 = fastdiv(iq3, &factx->broadcast_rv3);
        const uint32_t iv2 = fastdiv(iq2, &factx->broadcast_rv2);

        const __fp16 * mp_base = NULL;
        if (mask) {
            const uint32_t im2 = fastmodulo(iq2, mask->ne[2], &factx->src3_div2);
            const uint32_t im3 = fastmodulo(iq3, mask->ne[3], &factx->src3_div3);
            mp_base = (const __fp16 *) ((const uint8_t *) mask->data + iq1*mask->nb[1] + im2*mask->nb[2] + im3*mask->nb[3]);
        }

        // Precalculate next row variables if there is a next row
        bool has_next_ir = (ir + 1 < ir1);
        uint32_t next_ik2 = 0, next_ik3 = 0, next_iv2 = 0, next_iv3 = 0;
        const uint8_t * next_q_row_ptr = NULL;
        const __fp16 * next_mp_base = NULL;

        const uint8_t * next_k_src0 = NULL;
        const uint8_t * next_v_src0 = NULL;
        const uint8_t * next_m_src0 = NULL;
        uint32_t next_block_size0 = 0;

        const uint8_t * next_k_src1 = NULL;
        const uint8_t * next_v_src1 = NULL;
        const uint8_t * next_m_src1 = NULL;
        uint32_t next_block_size1 = 0;

        if (has_next_ir) {
            const uint32_t next_ir = ir + 1;
            const uint32_t next_iq3 = fastdiv(next_ir, &factx->src0_div21);
            const uint32_t next_iq2 = fastdiv(next_ir - next_iq3*neq2*neq1, &factx->src0_div1);
            const uint32_t next_iq1 = (next_ir - next_iq3*neq2*neq1 - next_iq2 * neq1);

            next_ik3 = fastdiv(next_iq3, &factx->broadcast_rk3);
            next_ik2 = fastdiv(next_iq2, &factx->broadcast_rk2);

            next_iv3 = fastdiv(next_iq3, &factx->broadcast_rv3);
            next_iv2 = fastdiv(next_iq2, &factx->broadcast_rv2);

            next_q_row_ptr = (const uint8_t *) q->data + (next_iq1*nbq1 + next_iq2*nbq2 + next_iq3*nbq3);

            if (mask) {
                const uint32_t next_im2 = fastmodulo(next_iq2, mask->ne[2], &factx->src3_div2);
                const uint32_t next_im3 = fastmodulo(next_iq3, mask->ne[3], &factx->src3_div3);
                next_mp_base = (const __fp16 *) ((const uint8_t *) mask->data + next_iq1*mask->nb[1] + next_im2*mask->nb[2] + next_im3*mask->nb[3]);
            }

            // Precalculate next K/V block 0 source pointers
            {
                const uint32_t ic_start = 0;
                next_block_size0 = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);
                next_k_src0 = (const uint8_t *) k->data + (ic_start*nbk1 + next_ik2*nbk2 + next_ik3*nbk3);
                next_v_src0 = (const uint8_t *) v->data + (ic_start*nbv1 + next_iv2*nbv2 + next_iv3*nbv3);
                if (mask) {
                    next_m_src0 = (const uint8_t *) (next_mp_base + ic_start);
                }
            }

            // Precalculate next K/V block 1 source pointers (if n_blocks > 1)
            if (factx->n_blocks > 1) {
                const uint32_t ic_start = 1 * FLASH_ATTN_BLOCK_SIZE;
                next_block_size1 = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);
                next_k_src1 = (const uint8_t *) k->data + (ic_start*nbk1 + next_ik2*nbk2 + next_ik3*nbk3);
                next_v_src1 = (const uint8_t *) v->data + (ic_start*nbv1 + next_iv2*nbv2 + next_iv3*nbv3);
                if (mask) {
                    next_m_src1 = (const uint8_t *) (next_mp_base + ic_start);
                }
            }
        }

        if (ir == ir0) {
            // Fetch Q row
            const uint8_t * q_row_ptr = (const uint8_t *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3);
            dma_queue_push(dma, dma_make_ptr(spad_q, q_row_ptr), factx->size_q_row_padded, nbq1, size_q_row, 1);

            // Prefetch first two blocks
            for (uint32_t ib = 0; ib < MIN(factx->n_blocks, 2); ++ib) {
                const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
                const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

                // K
                const uint8_t * k_src = (const uint8_t *) k->data + (ic_start*nbk1 + ik2*nbk2 + ik3*nbk3);
                uint8_t * k_dst = spad_k + (ib % 2) * factx->size_k_block;
                dma_queue_push(dma, dma_make_ptr(k_dst, k_src), factx->size_k_row_padded, nbk1, size_k_row, current_block_size);

                // V
                const uint8_t * v_src = (const uint8_t *) v->data + (ic_start*nbv1 + iv2*nbv2 + iv3*nbv3);
                uint8_t * v_dst = spad_v + (ib % 2) * factx->size_v_block;
                dma_queue_push(dma, dma_make_ptr(v_dst, v_src), factx->size_v_row_padded, nbv1, size_v_row, current_block_size);

                // Mask
                if (mask) {
                    const uint8_t * m_src = (const uint8_t *) (mp_base + ic_start);
                    // Mask is 1D contiguous for this row
                    dma_cache_push(dma, &m_cache, m_src, current_block_size * 2, current_block_size * 2, current_block_size * 2, 1);
                }
            }
        }

        const uint32_t h = iq2; // head index
        const __fp16 slope = factx->slopes[h];

        HVX_Vector S_vec = hvx_vec_splat_f32(0.0f);
        HVX_Vector M_vec = hvx_vec_splat_f32(HTP_FA_M_INITIAL_VAL);

        // Clear accumulator
        hvx_splat_f32_a(spad_a, 0, DV);
        float * VKQ32 = (float *) (spad_a + 0);

        uint8_t * q_ptr_vtcm = dma_queue_pop(dma).dst;
        if (factx->is_q_fp32) {
            hvx_copy_f16_f32_aa(q_ptr_vtcm, q_ptr_vtcm, DK);  // inplace convert f32 to f16
        }

        const HVX_Vector slope_vec = hvx_vec_splat_f16(slope);
        const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
        const HVX_Vector v_cap     = (factx->logit_softcap != 0.0f) ? hvx_vec_splat_f16(factx->logit_softcap) : Q6_V_vzero();
        const HVX_Vector vinf      = Q6_Vh_vsplat_R(0xFC00);
        const HVX_Vector vmin      = Q6_Vh_vsplat_R(0xFBFF);
        const HVX_Vector v_log2e   = hvx_vec_splat_f16(EXP_LOG2E_F);
        const uint32_t stride_v2   = factx->size_v_row_padded * 2;
        for (uint32_t ib = 0; ib < factx->n_blocks; ++ib) {
            const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
            const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

            // Wait for DMA
            uint8_t * k_base = dma_queue_pop(dma).dst; // K
            uint8_t * v_base = dma_queue_pop(dma).dst; // V
            __fp16  * m_base = mask ? dma_queue_pop(dma).dst : NULL; // M

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_QK, ir);

            // Inner loop processing the block from VTCM
            // 1. Compute scores (64 elements FP16)
            HVX_Vector scores_f16 = Q6_V_vzero();
            if (current_block_size > 0) {
                HVX_Vector scores0 = hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_base, factx->size_k_row_padded, DK, factx->scale);
                HVX_Vector scores1 = (current_block_size > 32) ? hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_base + 32 * factx->size_k_row_padded, factx->size_k_row_padded, DK, factx->scale) : Q6_V_vzero();
                scores_f16 = hvx_vec_f32_to_f16(scores0, scores1);
            }

            // 2. Softcap (in FP16)
            if (factx->logit_softcap != 0.0f) {
                scores_f16 = hvx_vec_tanh_f16(scores_f16);
                scores_f16 = hvx_vec_mul_f16_f16(scores_f16, v_cap);
            }

            HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(current_block_size * sizeof(__fp16));

            // 3. Mask (in FP16)
            if (mask) {
                HVX_Vector m_vals_f16 = *(const HVX_UVector *) m_base;
                HVX_VectorPred is_inf = Q6_Q_vcmp_eq_VhVh(m_vals_f16, vinf);
                m_vals_f16 = Q6_V_vmux_QVV(is_inf, vmin, m_vals_f16);

                HVX_Vector m_scaled = hvx_vec_mul_f16_f16(m_vals_f16, slope_vec);
                scores_f16 = Q6_V_vmux_QVV(q_tail_keep, hvx_vec_add_f16_f16(scores_f16, m_scaled), v_neg_inf);
            } else {
                scores_f16 = Q6_V_vmux_QVV(q_tail_keep, scores_f16, v_neg_inf);
            }

            // Compute block max in FP16
            HVX_Vector v_max_f16 = hvx_vec_reduce_max_f16(scores_f16);
            HVX_Vector v_max     = Q6_V_lo_W(hvx_vec_f16_to_f32(v_max_f16)); // splat block max in FP32
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_QK, ir);

            if (ib + 1 == factx->n_blocks && has_next_ir) {
                // Queue next row's Q row!
                dma_queue_push(dma, dma_make_ptr(spad_q, next_q_row_ptr), factx->size_q_row_padded, nbq1, size_q_row, 1);

                if (factx->n_blocks % 2 == 0) {
                    // Queue next row's block 0 (into buffer slot 0)
                    uint8_t * k_dst = spad_k + 0 * factx->size_k_block;
                    uint8_t * v_dst = spad_v + 0 * factx->size_v_block;

                    // K (block 0 of next row)
                    dma_queue_push(dma, dma_make_ptr(k_dst, next_k_src0), factx->size_k_row_padded, nbk1, size_k_row, next_block_size0);

                    // V (block 0 of next row)
                    dma_queue_push(dma, dma_make_ptr(v_dst, next_v_src0), factx->size_v_row_padded, nbv1, size_v_row, next_block_size0);

                    // Mask (block 0 of next row)
                    if (mask) {
                        dma_cache_push(dma, &m_cache, next_m_src0, next_block_size0 * 2, next_block_size0 * 2, next_block_size0 * 2, 1);
                    }
                }
            }

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_SFM, ir);
            {
                // 4. Online Softmax Update
                HVX_Vector M_new_vec = Q6_Vsf_vmax_VsfVsf(v_max, M_vec);
                HVX_Vector diff_vec  = HVX_OP_SUB_F32(M_vec, M_new_vec);

                HVX_Vector diff_f16   = hvx_vec_f32_to_f16(diff_vec, diff_vec);
                HVX_Vector diff_base2 = hvx_vec_mul_f16_f16(diff_f16, v_log2e);
                HVX_Vector ms_f16     = hvx_vec_exp2_f16(diff_base2);
                HVX_Vector ms_vec     = Q6_V_lo_W(hvx_vec_f16_to_f32(ms_f16));

                M_vec = M_new_vec;

                hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms_vec);

                // Compute P = exp2((S - M) * log2(e)) in FP16
                HVX_Vector v_m_vec_f16 = hvx_vec_f32_to_f16(M_vec, M_vec);
                HVX_Vector v_s_minus_m = Q6_Vqf16_vsub_VhfVhf(scores_f16, v_m_vec_f16);

                HVX_Vector v_s_minus_m_base2 = hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m), v_log2e);

                HVX_Vector P = hvx_vec_exp2_f16(v_s_minus_m_base2);
                P = Q6_V_vmux_QVV(q_tail_keep, P, Q6_V_vzero());

                // Convert P to FP32 to update the running sum S_vec
                HVX_VectorPair P_pair = hvx_vec_f16_to_f32(P);
                HVX_Vector P0 = Q6_V_lo_W(P_pair);
                HVX_Vector P1 = Q6_V_hi_W(P_pair);
                HVX_Vector p_sum_vec = hvx_vec_reduce_sum_f32(HVX_OP_ADD_F32(P0, P1));

                S_vec = HVX_OP_ADD_F32(HVX_OP_MUL_F32(S_vec, ms_vec), p_sum_vec);

                // 5. Accumulate V (F16 * F16 -> F32 accumulator)
                const uint8_t * v_ptr = v_base;

                for (uint32_t j = 0; j < current_block_size; j += 2) {
                    if (j + 1 == current_block_size) {
                        HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                        hvx_mad_f32_f16_aa_vec(VKQ32, v_ptr, S0, DV);
                        break;
                    }

                    HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                    HVX_Vector S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));

                    hvx_mad_f32_f16_aa_rx2_vec(VKQ32, v_ptr, v_ptr + factx->size_v_row_padded, S0, S1, DV);
                    v_ptr += stride_v2;
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, ir);

            // Issue DMA for next+1 block (if exists)
            if (ib + 2 < factx->n_blocks) {
                const uint32_t next_ib = ib + 2;
                const uint32_t next_ic_start = next_ib * FLASH_ATTN_BLOCK_SIZE;
                const uint32_t next_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - next_ic_start);

                // K
                const uint8_t * k_src = (const uint8_t *) k->data + (next_ic_start*nbk1 + ik2*nbk2 + ik3*nbk3);
                dma_queue_push(dma, dma_make_ptr(k_base, k_src), factx->size_k_row_padded, nbk1, size_k_row, next_block_size);

                // V
                const uint8_t * v_src = (const uint8_t *) v->data + (next_ic_start*nbv1 + iv2*nbv2 + iv3*nbv3);
                dma_queue_push(dma, dma_make_ptr(v_base, v_src), factx->size_v_row_padded, nbv1, size_v_row, next_block_size);

                // Mask
                if (mask) {
                    const uint8_t * m_src = (const uint8_t *) (mp_base + next_ic_start);
                    dma_cache_push(dma, &m_cache, m_src, next_block_size * 2, next_block_size * 2, next_block_size * 2, 1);
                }
            }
        }

        if (has_next_ir) {
            if (factx->n_blocks % 2 == 0) {
                // Queue next row's block 1 (into buffer slot 1, if n_blocks > 1)
                if (factx->n_blocks > 1) {
                    uint8_t * k_dst = spad_k + 1 * factx->size_k_block;
                    uint8_t * v_dst = spad_v + 1 * factx->size_v_block;

                    // K (block 1 of next row)
                    dma_queue_push(dma, dma_make_ptr(k_dst, next_k_src1), factx->size_k_row_padded, nbk1, size_k_row, next_block_size1);

                    // V (block 1 of next row)
                    dma_queue_push(dma, dma_make_ptr(v_dst, next_v_src1), factx->size_v_row_padded, nbv1, size_v_row, next_block_size1);

                    // Mask (block 1 of next row)
                    if (mask) {
                        dma_cache_push(dma, &m_cache, next_m_src1, next_block_size1 * 2, next_block_size1 * 2, next_block_size1 * 2, 1);
                    }
                }
            } else {
                // Queue next row's block 0 (into buffer slot 0)
                {
                    uint8_t * k_dst = spad_k + 0 * factx->size_k_block;
                    uint8_t * v_dst = spad_v + 0 * factx->size_v_block;

                    // K (block 0 of next row)
                    dma_queue_push(dma, dma_make_ptr(k_dst, next_k_src0), factx->size_k_row_padded, nbk1, size_k_row, next_block_size0);

                    // V (block 0 of next row)
                    dma_queue_push(dma, dma_make_ptr(v_dst, next_v_src0), factx->size_v_row_padded, nbv1, size_v_row, next_block_size0);

                    // Mask (block 0 of next row)
                    if (mask) {
                        dma_cache_push(dma, &m_cache, next_m_src0, next_block_size0 * 2, next_block_size0 * 2, next_block_size0 * 2, 1);
                    }
                }

                // Queue next row's block 1 (into buffer slot 1, if n_blocks > 1)
                if (factx->n_blocks > 1) {
                    uint8_t * k_dst = spad_k + 1 * factx->size_k_block;
                    uint8_t * v_dst = spad_v + 1 * factx->size_v_block;

                    // K (block 1 of next row)
                    dma_queue_push(dma, dma_make_ptr(k_dst, next_k_src1), factx->size_k_row_padded, nbk1, size_k_row, next_block_size1);

                    // V (block 1 of next row)
                    dma_queue_push(dma, dma_make_ptr(v_dst, next_v_src1), factx->size_v_row_padded, nbv1, size_v_row, next_block_size1);

                    // Mask (block 1 of next row)
                    if (mask) {
                        dma_cache_push(dma, &m_cache, next_m_src1, next_block_size1 * 2, next_block_size1 * 2, next_block_size1 * 2, 1);
                    }
                }
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, ir);
        // sinks
        float M = hvx_vec_get_f32(M_vec);
        float S = hvx_vec_get_f32(S_vec);

        if (sinks) {
            const float s = ((float *)((char *) sinks->data))[h];

            float vs = 1.0f;

            if (s > M) {
                HVX_Vector diff_vec = hvx_vec_splat_f32(M - s);
                HVX_Vector ms_vec   = hvx_vec_exp_f32(diff_vec);
                hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms_vec);

                float ms = hvx_vec_get_f32(ms_vec);
                S = S * ms + vs;
            } else {
                HVX_Vector diff_vec = hvx_vec_splat_f32(s - M);
                vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                S += vs;
            }
        }

        const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
        hvx_scale_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, S_inv);

        // Store result
        // dst indices
        const uint32_t i1 = iq1;
        const uint32_t i2 = iq2;
        const uint32_t i3 = iq3;

        // dst is permuted: [DV, n_heads, n_tokens, n_seq]
        // head stride is nb[1], token stride is nb[2], batch stride is nb[3]
        uint8_t * dst_ptr = (uint8_t *) dst->data + i2 * dst->nb[1] + i1 * dst->nb[2] + i3 * dst->nb[3];

        if (dst->type == HTP_TYPE_F32) {
            hvx_copy_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
        } else if (dst->type == HTP_TYPE_F16) {
            hvx_copy_f16_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, ir);
    }
}

// ============================================================================
// HMX Phase args and thread logic
// ============================================================================

// A chunk's rows are tiled from `nblk` source blocks laid out consecutively at
// `block_rows` intervals in the CHUNK's row space. nblk == 1 with block_rows ==
// kv_rows is the pre-residency form: one contiguous staging buffer, one call, exactly
// the tiles the single-call version produced. nblk > 1 is the residency form, where
// each block lives in its own VTCM slot and only its source address differs.
//
// Splitting the destination per block is free: block b starts at chunk row
// b*block_rows, block_rows is a multiple of the 32-row tile height, and both interleave
// kernels derive the destination tile from row/32 and the in-tile row from row%32. So
// tiling a block's LOCAL rows into a destination shifted by (b*block_rows/32) tiles
// lands every element exactly where the whole-chunk call put it.
typedef struct {
    struct hmx_fa_context * factx;
    uint32_t                kv_rows;
    size_t                  src_stride;
    void * const *          bases;       // per-block VTCM source, taken from the DMA descriptors
    uint32_t                nblk;
    uint32_t                block_rows;
    void *                  k_tiles_dst;
    uint32_t                kv_start;
    uint32_t                rows_per_t;
} fa_k_int_args_t;

static void fa_k_interleave_thread(unsigned int n, unsigned int i, void * data) {
    fa_k_int_args_t *       args  = (fa_k_int_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const uint32_t total_rows = args->kv_rows;
    const uint32_t rows_per_t = args->rows_per_t;
    const uint32_t start      = i * rows_per_t;
    const uint32_t end        = (uint32_t) hex_smin(start + rows_per_t, total_rows);

    if (start >= total_rows) {
        return;
    }

    // Bytes one block's tiles occupy in the K tile buffer: DK/32 tiles per 32 rows.
    const size_t   k_tile_run = (size_t) (factx->DK / HMX_FP16_TILE_N_COLS) * HMX_FP16_TILE_N_ELMS;
    const uint32_t blk_rows   = args->block_rows;

    struct htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (args->kv_start + start));
    for (uint32_t r = start; r < end;) {
        // Row ranges are even-aligned and blk_rows is a multiple of the tile height, so
        // a row PAIR never straddles a block boundary -- which is what lets each block
        // be tiled independently without changing the odd-row zero fill.
        const uint32_t b      = (args->nblk == 1) ? 0 : (r / blk_rows);
        const uint32_t bstart = b * blk_rows;
        const uint32_t brows  = (uint32_t) hex_smin(blk_rows, total_rows - bstart);
        const uint32_t r_end  = (uint32_t) hex_smin(end, bstart + brows);

        __fp16 * dst = (__fp16 *) args->k_tiles_dst + (size_t) (bstart / HMX_FP16_TILE_N_ROWS) * k_tile_run;
        hmx_interleave_rows_to_tiles(dst, (const __fp16 *) args->bases[b], brows, factx->DK,
                                     args->src_stride, r - bstart, r_end - bstart);
        r = r_end;
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (args->kv_start + start));
}

static void fa_phase_k_interleave(struct hmx_fa_context * factx, uint32_t kv_rows, size_t src_stride,
                                  void * const * bases, uint32_t nblk, uint32_t block_rows,
                                  uint32_t kv_start, void * k_tiles_dst) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    if (factx->n_threads > 1 && kv_rows >= factx->n_threads * 2) {
        n = factx->n_threads;
    }
    uint32_t rows_per_t = hex_align_up(hmx_ceil_div(kv_rows, n), 2);
    fa_k_int_args_t args = { factx, kv_rows, src_stride, bases, nblk, block_rows, k_tiles_dst, kv_start, rows_per_t };
    if (n > 1) {
        work_queue_run(wp, fa_k_interleave_thread, &args, n);
    } else {
        fa_k_interleave_thread(1, 0, &args);
    }
}

// Same per-block source split as the K phase. The V tile layout is [dim_tile][row_tile]
// with a dim-tile stride of n_col_tiles (the CHUNK's width in tiles), and that stride is
// passed through untouched -- only the destination's row-tile origin is shifted per
// block, so the chunk's V tile buffer comes out exactly as the whole-chunk call built
// it and hmx_fa_o_update_worker needs no change at all.
typedef struct {
    struct hmx_fa_context * factx;
    uint32_t                kv_rows;
    size_t                  src_stride;
    void * const *          bases;
    uint32_t                nblk;
    uint32_t                block_rows;
    void *                  v_tiles_dst;
    size_t                  n_col_tiles;
    uint32_t                kv_start;
    uint32_t                rows_per_t;
} fa_v_int_args_t;

static void fa_v_interleave_thread(unsigned int n, unsigned int i, void * data) {
    fa_v_int_args_t *       args  = (fa_v_int_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const uint32_t total_rows = args->kv_rows;
    const uint32_t rows_per_t = args->rows_per_t;
    const uint32_t start      = i * rows_per_t;
    const uint32_t end        = (uint32_t) hex_smin(start + rows_per_t, total_rows);

    if (start >= total_rows) {
        return;
    }

    const uint32_t blk_rows = args->block_rows;

    struct htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (args->kv_start + start));
    for (uint32_t r = start; r < end;) {
        const uint32_t b      = (args->nblk == 1) ? 0 : (r / blk_rows);
        const uint32_t bstart = b * blk_rows;
        const uint32_t brows  = (uint32_t) hex_smin(blk_rows, total_rows - bstart);
        const uint32_t r_end  = (uint32_t) hex_smin(end, bstart + brows);

        __fp16 * dst = (__fp16 *) args->v_tiles_dst +
                       (size_t) (bstart / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ELMS;
        hmx_interleave_cols_to_tiles(dst, (const __fp16 *) args->bases[b], brows, factx->DV,
                                     args->src_stride, (uint32_t) args->n_col_tiles,
                                     r - bstart, r_end - bstart);
        r = r_end;
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (args->kv_start + start));
}

static void fa_phase_v_interleave(struct hmx_fa_context * factx,
                                  uint32_t                kv_rows,
                                  size_t                  src_stride,
                                  void * const *          bases,
                                  uint32_t                nblk,
                                  uint32_t                block_rows,
                                  void *                  v_tiles_dst,
                                  size_t                  n_col_tiles,
                                  uint32_t                kv_start) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    if (factx->n_threads > 1 && kv_rows >= factx->n_threads * 2) {
        n = factx->n_threads;
    }
    uint32_t rows_per_t = hex_align_up(hmx_ceil_div(kv_rows, n), 2);
    fa_v_int_args_t args = { factx, kv_rows, src_stride, bases, nblk, block_rows,
                             v_tiles_dst, n_col_tiles, kv_start, rows_per_t };
    if (n > 1) {
        work_queue_run(wp, fa_v_interleave_thread, &args, n);
    } else {
        fa_v_interleave_thread(1, 0, &args);
    }
}

typedef struct {
    struct hmx_fa_context *   factx;
    const struct htp_tensor * q;
    uint32_t                  q_start;
    uint32_t                  kv_head;
    uint32_t                  ib3;
    size_t                    n_rows_g;
    size_t                    rows_per_t;
    size_t                    n_rows_q;
    bool                      q_transposed;
    atomic_uint               barrier;
} fa_q_load_args_t;

static void fa_q_load_thread(unsigned int n, unsigned int i, void * data) {
    fa_q_load_args_t *      args  = (fa_q_load_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows_g = args->n_rows_g;
    const size_t G        = factx->G;
    const size_t DK       = factx->DK;

    // Partition the padded Q rows (g_br) across threads.
    // Keep start/end even so r and r+1 are always in the same thread's range.
    const size_t rows_per_t = args->rows_per_t;
    const size_t start      = (size_t) i * rows_per_t;
    const size_t end        = hex_smin(start + rows_per_t, factx->g_br);

    struct htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (args->q_start * G + start));

    // Parallel initialization of per-block state
    {
        const uint32_t g_br = factx->g_br;
        const uint32_t DV   = factx->DV;

        const size_t col_vec_bytes = factx->col_vec_bytes;
        const size_t d_tile_bytes  = factx->d_tile_bytes;

        // Initialize vtcm_l_vec & vtcm_m_vec
        const size_t l_bytes_per_t = hex_align_up(col_vec_bytes / n, 128);
        const size_t l_start       = i * l_bytes_per_t;
        const size_t l_end         = hex_smin(l_start + l_bytes_per_t, col_vec_bytes);

        const size_t m_bytes_per_t = hex_align_up(col_vec_bytes / n, 128);
        const size_t m_start       = i * m_bytes_per_t;
        const size_t m_end         = hex_smin(m_start + m_bytes_per_t, col_vec_bytes);

        if (factx->sinks) {
            const float * sinks_data = (const float *) (uintptr_t) factx->sinks->data;
            float *       m_vec      = (float *) factx->vtcm_m_vec;
            const size_t  r_start    = l_start / sizeof(float);
            const size_t  r_end      = l_end / sizeof(float);
            const float   scale_factor = EXP_LOG2E_F;

            const HVX_Vector v_scale = hvx_vec_splat_f32(scale_factor);

            for (size_t r = r_start; r < r_end; r += 32) {
                HVX_VectorAlias local_m;
                for (size_t j = 0; j < 32; ++j) {
                    size_t curr_r = r + j;
                    if (curr_r < n_rows_g) {
                        const size_t h_idx = fastmodulo(curr_r, G, &factx->div_G);
                        const size_t head  = args->kv_head * G + h_idx;
                        local_m.fp32[j] = sinks_data[head];
                    } else {
                        local_m.fp32[j] = HTP_FA_M_INITIAL_VAL;
                    }
                }
                HVX_Vector v_scaled = HVX_OP_MUL_F32(local_m.v, v_scale);
                *(HVX_Vector *) (m_vec + r) = v_scaled;
            }
            if (l_start < col_vec_bytes) {
                hvx_splat_u8_a((char *) factx->vtcm_l_vec + l_start, 0, l_end - l_start);
            }
        } else {
            if (l_start < col_vec_bytes) {
                hvx_splat_u8_a((char *) factx->vtcm_l_vec + l_start, 0, l_end - l_start);
            }
            if (m_start < col_vec_bytes) {
                hvx_splat_f32_a((char *) factx->vtcm_m_vec + m_start, HTP_FA_M_INITIAL_VAL, (m_end - m_start) / sizeof(float));
            }
        }

        // Zero the whole rescale region: vtcm_d_tiles[0], the optional vtcm_d_tiles[1]
        // and vtcm_d_inv_l are equal-sized and allocated back to back, so one run covers
        // them all.  The scatter only ever writes the diagonal, ignore the rest.
        const size_t d_bytes_per_t = hex_align_up(d_tile_bytes / n, 128);
        const size_t d_start       = i * d_bytes_per_t;
        const size_t d_end         = hex_smin(d_start + d_bytes_per_t, d_tile_bytes);
        if (d_start < d_tile_bytes) {
            hvx_splat_u8_a((char *) factx->vtcm_d_tiles[0] + d_start, 0, d_end - d_start);
        }
    }

    if (start < factx->g_br) {
        const struct htp_tensor * q       = args->q;
        const uint32_t            q_start = args->q_start;
        const uint32_t            kv_head = args->kv_head;
        const uint32_t            ib3     = args->ib3;

        assert(factx->DK == factx->DV);

        const bool use_q_dma = (factx->vtcm_q_dma != NULL);

        __fp16 * q_tiles = factx->vtcm_q_tiles;
        if (use_q_dma) {
            const size_t g_rows_end = hex_smin(end, n_rows_g);
            const uint32_t d_limit = factx->is_q_fp32 ? DK / 32 : DK / 64;

            uint8_t * q_flat  = (uint8_t *) factx->vtcm_q_dma;
            if (factx->is_q_fp32) {
                switch (d_limit) {
                case 2:  hmx_fa_q_prep_fp32_d2(q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, args->q_transposed); break;
                case 4:  hmx_fa_q_prep_fp32_d4(q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, args->q_transposed); break;
                default: hmx_fa_q_prep_fp32(   q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, d_limit, args->q_transposed); break;
                }
            } else {
                switch (d_limit) {
                case 1:  hmx_fa_q_prep_fp16_d1(q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, args->q_transposed); break;
                case 2:  hmx_fa_q_prep_fp16_d2(q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, args->q_transposed); break;
                default: hmx_fa_q_prep_fp16(   q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, d_limit, args->q_transposed); break;
                }
            }
        } else {
            // Fallback: direct-from-DDR/L2 path
            hmx_fa_q_prep_fallback(q_tiles, q->data, q->nb[1], q->nb[2], q->nb[3],
                                   q_start, kv_head, ib3, start, end, n_rows_g, G, DK, factx->is_q_fp32, &factx->div_G);
        }
    }

    // Synchronize threads before zeroing out vtcm_o_tiles[0] to prevent race condition
    if (n > 1) {
        atomic_fetch_sub(&args->barrier, 1);
        while (atomic_load(&args->barrier) > 0) {
            // spin wait
        }
    }

    // Zero out vtcm_o_tiles[0] as it was used as temp_q_vtcm
    {
        const uint32_t g_br = factx->g_br;
        const uint32_t DV   = factx->DV;
        const size_t o_tile_bytes  = factx->o_tile_bytes;
        const size_t o_bytes_per_t = hex_align_up(o_tile_bytes / n, 128);
        const size_t o_start       = i * o_bytes_per_t;
        const size_t o_end         = hex_smin(o_start + o_bytes_per_t, o_tile_bytes);
        if (o_start < o_tile_bytes) {
            hvx_splat_u8_a((char *) factx->vtcm_o_tiles[0] + o_start, 0, o_end - o_start);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (args->q_start * G + start));
}

static void fa_phase_q_load(struct hmx_fa_context *   factx,
                            const struct htp_tensor * q,
                            uint32_t                  q_start,
                            uint32_t                  kv_head,
                            uint32_t                  ib3,
                            size_t                    n_rows_g) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    if (factx->n_threads > 1 && n_rows_g >= (size_t) (factx->n_threads * 2)) {
        n = factx->n_threads;
    }
    size_t rows_per_t = hex_align_up(hmx_ceil_div(factx->g_br, n), 2);
    const uint32_t n_rows_q = hex_smin(factx->Br, factx->neq1 - q_start);
    fa_q_load_args_t args;
    args.factx = factx;
    args.q = q;
    args.q_start = q_start;
    args.kv_head = kv_head;
    args.ib3 = ib3;
    args.n_rows_g = n_rows_g;
    args.rows_per_t = rows_per_t;
    args.n_rows_q = n_rows_q;
    args.q_transposed = q->nb[1] < q->nb[2];
    atomic_init(&args.barrier, n);
    if (n > 1) {
        work_queue_run(wp, fa_q_load_thread, &args, n);
    } else {
        fa_q_load_thread(1, 0, &args);
    }
}

typedef struct {
    struct hmx_fa_context *   factx;
    const struct htp_tensor * dst;
    const __fp16 *            o_tile_src;
    uint32_t                  q_start;
    uint32_t                  kv_head;
    uint32_t                  ib3;
    size_t                    n_rows_g;
    size_t                    rows_per_t;
} fa_o_store_args_t;

static void fa_o_store_thread_f32(unsigned int n, unsigned int i, void * data) {
    fa_o_store_args_t *     args  = (fa_o_store_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows_g = args->n_rows_g;
    const size_t G        = factx->G;
    const size_t DV       = factx->DV;

    const size_t rows_per_t = args->rows_per_t;
    const size_t start      = (size_t) i * rows_per_t;
    const size_t end        = hex_smin(start + rows_per_t, n_rows_g);

    if (start >= n_rows_g) {
        return;
    }

    struct htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));

    const struct htp_tensor * dst        = args->dst;
    const __fp16 *            o_tile_src = args->o_tile_src;
    const uint32_t            q_start    = args->q_start;
    const uint32_t            kv_head    = args->kv_head;
    const uint32_t            ib3        = args->ib3;

    size_t q_idx = fastdiv(start, &factx->div_G);
    size_t h_idx = fastmodulo(start, G, &factx->div_G);

    for (size_t r = start; r < end; ++r) {
        float * out = (float *) ((uint8_t *) dst->data + (kv_head * G + h_idx) * dst->nb[1] +
                                 (q_start + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_ROWS;
        size_t         r1            = r % HMX_FP16_TILE_N_ROWS;
        const __fp16 * tile_row_base = o_tile_src + r0 * HMX_FP16_TILE_N_ROWS * DV;

        for (uint32_t d = 0; d < DV / 32; ++d) {
            const HVX_Vector * in_tile = (const HVX_Vector *) (tile_row_base + d * HMX_FP16_TILE_N_ELMS);
            HVX_VectorPair     vp      = hvx_vec_f16_to_f32_shuff(in_tile[r1 / 2]);
            if (r1 % 2 == 0) {
                *(HVX_UVector *) (out + d * 32) = Q6_V_lo_W(vp);
            } else {
                *(HVX_UVector *) (out + d * 32) = Q6_V_hi_W(vp);
            }
        }

        h_idx++;
        if (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));
}

static void fa_o_store_thread_f16(unsigned int n, unsigned int i, void * data) {
    fa_o_store_args_t *     args  = (fa_o_store_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows_g   = args->n_rows_g;
    const size_t rows_per_t = args->rows_per_t;
    const size_t G          = factx->G;
    const size_t DV         = factx->DV;
    const size_t start      = (size_t) i * rows_per_t;
    const size_t end        = hex_smin(start + rows_per_t, n_rows_g);

    if (start >= n_rows_g) {
        return;
    }

    struct htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));

    const struct htp_tensor * dst        = args->dst;
    const __fp16 *            o_tile_src = args->o_tile_src;
    const uint32_t            q_start    = args->q_start;
    const uint32_t            kv_head    = args->kv_head;
    const uint32_t            ib3        = args->ib3;

    size_t q_idx = fastdiv(start, &factx->div_G);
    size_t h_idx = fastmodulo(start, G, &factx->div_G);

    for (size_t r = start; r < end; ++r) {
        __fp16 * out = (__fp16 *) ((uint8_t *) dst->data + (kv_head * G + h_idx) * dst->nb[1] +
                                   (q_start + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_ROWS;
        size_t         r1            = r % HMX_FP16_TILE_N_ROWS;
        const __fp16 * tile_row_base = o_tile_src + r0 * HMX_FP16_TILE_N_ROWS * DV;

        for (uint32_t d = 0; d < DV / 64; ++d) {
            const __fp16 *     in_dtile = tile_row_base + d * HMX_FP16_TILE_N_ELMS * 2;
            const HVX_Vector * pv_in0   = ((const HVX_Vector *) in_dtile) + r1 / 2;
            const HVX_Vector * pv_in1   = pv_in0 + 16;
            HVX_VectorPair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            if (r1 % 2 == 0) {
                *(HVX_UVector *) (out + d * 64) = Q6_V_lo_W(vp);
            } else {
                *(HVX_UVector *) (out + d * 64) = Q6_V_hi_W(vp);
            }
        }

        h_idx++;
        if (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));
}

static void fa_phase_o_store(struct hmx_fa_context *   factx,
                             const struct htp_tensor * dst,
                             const __fp16 *            o_tile_src,
                             uint32_t                  q_start,
                             uint32_t                  kv_head,
                             uint32_t                  ib3,
                             size_t                    n_rows_g) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    if (factx->n_threads > 1 && n_rows_g >= (size_t) (factx->n_threads * 2)) {
        n = factx->n_threads;
    }
    size_t rows_per_t = hmx_ceil_div(n_rows_g, n);
    fa_o_store_args_t args = { factx, dst, o_tile_src, q_start, kv_head, ib3, n_rows_g, rows_per_t };
    worker_callback_t store_fn = factx->is_dst_fp32 ? fa_o_store_thread_f32 : fa_o_store_thread_f16;
    if (n > 1) {
        work_queue_run(wp, store_fn, &args, n);
    } else {
        store_fn(1, 0, &args);
    }
}

typedef struct {
    struct hmx_fa_context *   factx;
    size_t                    buf_idx;
    size_t                    kv_rows;
    size_t                    n_rows_g;
    size_t                    n_col_tiles;
    size_t                    n_tiles_per_bc;
    size_t                    n_row_tiles;
    size_t                    n_row_tiles_g_br;
    uint32_t                  Bc;
    uint32_t                  G;
    uint32_t                  kv_head;
    uint32_t                  kv_start;
    uint32_t                  q_start;
    uint32_t                  ib3;
    bool                      is_first_block;  // first KV block processed for this Q block
    bool                      has_alibi;  // true when max_bias != 0 (need slope * mask + add)
    __fp16 *                  slopes;
    const struct htp_tensor * mask;
    const __fp16 *            mask_vtcm;             // VTCM mask buffer base (NULL = DDR fallback)
    size_t                    mask_vtcm_row_stride;  // elements (__fp16) per row in VTCM mask buffer
    struct fastdiv_values     thread_div;
} fa_softmax_args_t;

static inline void fa_softmax_impl(
    unsigned int n, unsigned int i, void * data,
    const bool has_mask,
    const bool mask_broadcast,
    const bool is_g1,
    const bool has_alibi,
    const bool has_softcap
) {
    fa_softmax_args_t *     args  = (fa_softmax_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows_g       = args->n_rows_g;
    const size_t kv_rows        = args->kv_rows;
    const size_t Bc             = args->Bc;
    const size_t G              = args->G;
    const size_t n_tiles_per_bc = args->n_tiles_per_bc;
    const size_t n_row_vec_cnt  = hmx_ceil_div(n_rows_g, 32);
    const uint32_t im3          = has_mask ? fastmodulo(args->ib3, args->mask->ne[3], &factx->src3_div3) : 0;

    size_t vec_start = 0;
    size_t vec_end   = n_row_vec_cnt;
    if (n > 1) {
        const size_t vecs_per_t = fastdiv(n_row_vec_cnt + n - 1, &args->thread_div);
        vec_start = i * vecs_per_t;
        vec_end   = hex_smin(vec_start + vecs_per_t, n_row_vec_cnt);
    }

    if (vec_start >= n_row_vec_cnt) {
        return;
    }

    struct htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (args->q_start * G + vec_start * 32));

    // Per-thread row scratch: thread i uses bufs at offset i * 2 * stride
    const size_t row_buf_stride = factx->row_buf_stride;
    HVX_Vector * my_row_buf0    = factx->vtcm_row_bufs + i * 2 * row_buf_stride;
    HVX_Vector * my_row_buf1    = my_row_buf0 + row_buf_stride;

    const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);

    for (size_t r_vec_idx = vec_start; r_vec_idx < vec_end; ++r_vec_idx) {
        HVX_Vector rowmax_acc_v = v_neg_inf;
        HVX_Vector rowsum_acc_v = Q6_V_vzero();
        HVX_Vector m_prev_v0    = factx->vtcm_m_vec[r_vec_idx];

        // A 32-row unit starts 64 B into the fp16 slopes array on odd units, but
        // hvx_vmem is an ALIGNED load, and hvx_vmemu would read 64 B past the end of
        // vtcm_slopes -- the last VTCM allocation, which may end exactly at
        // ctx->vtcm_size. Load the enclosing 128 B block (always in bounds) and fold
        // the odd-unit half into the per-row rotate below.
        HVX_Vector v_slopes = Q6_V_vzero();
        if (has_alibi) {
            v_slopes = hvx_vmem(args->slopes + (r_vec_idx & ~(size_t) 1) * 32);
        }
        const uint32_t slope_lane0 = (uint32_t) (r_vec_idx & 1) * 32;

        for (uint32_t r_vec_off = 0; r_vec_off < 32; r_vec_off += 2) {
            uint32_t r = r_vec_idx * 32 + r_vec_off;
            if (r >= hex_align_up(n_rows_g, 2)) {
                break;
            }

            uint32_t r0 = r / HMX_FP16_TILE_N_ROWS;
            uint32_t r1 = r % HMX_FP16_TILE_N_ROWS;

            const __fp16 * s_ld_base = factx->vtcm_s_tiles[args->buf_idx] + r0 * HMX_FP16_TILE_N_ROWS * Bc;
            __fp16 *       p_st_base = factx->vtcm_p_tiles[args->buf_idx] + r0 * HMX_FP16_TILE_N_ROWS * Bc;

            // Decode 2 rows from S tiles into per-thread row buffers
            if (has_softcap) {
                const HVX_Vector v_cap = hvx_vec_splat_f16(factx->logit_softcap);
                for (size_t c = 0; c < kv_rows; c += 64) {
                    size_t             ci       = c / 64;
                    const __fp16 *     in_dtile = s_ld_base + ci * HMX_FP16_TILE_N_ELMS * 2;
                    const HVX_Vector * pv_s_in0 = ((const HVX_Vector *) in_dtile) + r1 / 2;
                    const HVX_Vector * pv_s_in1 = pv_s_in0 + 16;

                    HVX_VectorPair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    HVX_Vector     v_s_row0  = Q6_V_lo_W(vp_s_drow);
                    HVX_Vector     v_s_row1  = Q6_V_hi_W(vp_s_drow);

                    HVX_Vector t0   = hvx_vec_tanh_f16(v_s_row0);
                    my_row_buf0[ci] = hvx_vec_mul_f16_f16(t0, v_cap);

                    HVX_Vector t1   = hvx_vec_tanh_f16(v_s_row1);
                    my_row_buf1[ci] = hvx_vec_mul_f16_f16(t1, v_cap);
                }
            } else {
                size_t c = 0;
                for (; c + 64 < kv_rows; c += 128) {
                    size_t             ci0       = c / 64;
                    size_t             ci1       = ci0 + 1;
                    const __fp16 *     in_dtile0 = s_ld_base + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                    const __fp16 *     in_dtile1 = s_ld_base + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                    const HVX_Vector * pv_s_in0_0 = ((const HVX_Vector *) in_dtile0) + r1 / 2;
                    const HVX_Vector * pv_s_in1_0 = pv_s_in0_0 + 16;
                    const HVX_Vector * pv_s_in0_1 = ((const HVX_Vector *) in_dtile1) + r1 / 2;
                    const HVX_Vector * pv_s_in1_1 = pv_s_in0_1 + 16;

                    HVX_VectorPair vp_s_drow0 = Q6_W_vdeal_VVR(*pv_s_in1_0, *pv_s_in0_0, -2);
                    my_row_buf0[ci0]          = Q6_V_lo_W(vp_s_drow0);
                    my_row_buf1[ci0]          = Q6_V_hi_W(vp_s_drow0);

                    HVX_VectorPair vp_s_drow1 = Q6_W_vdeal_VVR(*pv_s_in1_1, *pv_s_in0_1, -2);
                    my_row_buf0[ci1]          = Q6_V_lo_W(vp_s_drow1);
                    my_row_buf1[ci1]          = Q6_V_hi_W(vp_s_drow1);
                }
                for (; c < kv_rows; c += 64) {
                    size_t             ci       = c / 64;
                    const __fp16 *     in_dtile = s_ld_base + ci * HMX_FP16_TILE_N_ELMS * 2;
                    const HVX_Vector * pv_s_in0 = ((const HVX_Vector *) in_dtile) + r1 / 2;
                    const HVX_Vector * pv_s_in1 = pv_s_in0 + 16;

                    HVX_VectorPair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    my_row_buf0[ci]          = Q6_V_lo_W(vp_s_drow);
                    my_row_buf1[ci]          = Q6_V_hi_W(vp_s_drow);
                }
            }

            // Apply mask & compute rowmax(S)
            HVX_Vector v_slope0 = Q6_V_vzero();
            HVX_Vector v_slope1 = Q6_V_vzero();
            if (has_alibi) {
                v_slope0 = hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, (slope_lane0 + r_vec_off) * 2));
                v_slope1 = (r + 1 < n_rows_g) ? hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, (slope_lane0 + r_vec_off + 1) * 2)) : Q6_V_vzero();
            }

            const HVX_Vector v_threshold = Q6_Vh_vsplat_R(0xcc00);  // fp16 -16.0

            HVX_Vector v_s_rowmax0 = v_neg_inf;
            HVX_Vector v_s_rowmax1 = v_neg_inf;
            if (has_mask) {
                for (size_t c = 0; c < kv_rows; c += 64) {
                    size_t         ci          = c / 64;
                    const size_t   ne          = hex_smin(kv_rows - c, 64);
                    HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));

                    HVX_Vector v_mask0, v_mask1;

                    if (mask_broadcast) {
                        if (is_g1) {
                            const size_t qi0 = r + 0;
                            v_mask0 = *(const HVX_Vector *) (args->mask_vtcm + qi0 * args->mask_vtcm_row_stride + c);
                            v_mask1 = v_neg_inf;
                            if (r + 1 < n_rows_g) {
                                const size_t qi1 = r + 1;
                                v_mask1 = *(const HVX_Vector *) (args->mask_vtcm + qi1 * args->mask_vtcm_row_stride + c);
                            }
                        } else {
                            const size_t qi0 = fastdiv(r + 0, &factx->div_G);
                            v_mask0 = *(const HVX_Vector *) (args->mask_vtcm + qi0 * args->mask_vtcm_row_stride + c);
                            v_mask1 = v_neg_inf;
                            if (r + 1 < n_rows_g) {
                                const size_t qi1 = fastdiv(r + 1, &factx->div_G);
                                if (qi1 == qi0) {
                                    v_mask1 = v_mask0;
                                } else {
                                    v_mask1 = *(const HVX_Vector *) (args->mask_vtcm + qi1 * args->mask_vtcm_row_stride + c);
                                }
                            }
                        }
                    } else {
                        // Head-dependent mask: pre-interleaved per row r.
                        const size_t r0 = r + 0;
                        v_mask0 = *(const HVX_Vector *) (args->mask_vtcm + r0 * args->mask_vtcm_row_stride + c);
                        v_mask1 = v_neg_inf;
                        if (r + 1 < n_rows_g) {
                            const size_t r1 = r + 1;
                            v_mask1 = *(const HVX_Vector *) (args->mask_vtcm + r1 * args->mask_vtcm_row_stride + c);
                        }
                    }

                    // Threshold: mask values below -16.0 are treated as -inf (causal mask).
                    HVX_VectorPred q_keep0 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_threshold), q_tail_keep);
                    HVX_VectorPred q_keep1 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_threshold), q_tail_keep);

                    // Scale mask values by log2(e) for base-2 calculations
                    const HVX_Vector v_log2e = hvx_vec_splat_f16(EXP_LOG2E_F);
                    HVX_Vector v_mask0_scaled = hvx_vec_mul_f16_f16(v_mask0, v_log2e);
                    HVX_Vector v_mask1_scaled = hvx_vec_mul_f16_f16(v_mask1, v_log2e);

                    if (has_alibi) {
                        HVX_Vector v_sm0 = hvx_vec_mul_f16_f16(v_mask0_scaled, v_slope0);
                        HVX_Vector v_sm1 = hvx_vec_mul_f16_f16(v_mask1_scaled, v_slope1);
                        my_row_buf0[ci]  = Q6_V_vmux_QVV(q_keep0, hvx_vec_add_f16_f16(my_row_buf0[ci], v_sm0), v_neg_inf);
                        my_row_buf1[ci]  = Q6_V_vmux_QVV(q_keep1, hvx_vec_add_f16_f16(my_row_buf1[ci], v_sm1), v_neg_inf);
                    } else {
                        my_row_buf0[ci] = Q6_V_vmux_QVV(q_keep0, hvx_vec_add_f16_f16(my_row_buf0[ci], v_mask0_scaled), v_neg_inf);
                        my_row_buf1[ci] = Q6_V_vmux_QVV(q_keep1, hvx_vec_add_f16_f16(my_row_buf1[ci], v_mask1_scaled), v_neg_inf);
                    }

                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci]);
                }
            } else {
                size_t c = 0;
                for (; c + 64 < kv_rows; c += 128) {
                    size_t ci0 = c / 64;
                    size_t ci1 = ci0 + 1;
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci0]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci0]);
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci1]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci1]);
                }
                for (; c < kv_rows; c += 64) {
                    size_t         ci          = c / 64;
                    const size_t   ne          = hex_smin(kv_rows - c, 64);
                    HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
                    if (ne < 64) {
                        my_row_buf0[ci] = Q6_V_vmux_QVV(q_tail_keep, my_row_buf0[ci], v_neg_inf);
                        my_row_buf1[ci] = Q6_V_vmux_QVV(q_tail_keep, my_row_buf1[ci], v_neg_inf);
                    }
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci]);
                }
            }

            v_s_rowmax0 = hvx_vec_reduce_max_f16(v_s_rowmax0);
            v_s_rowmax1 = hvx_vec_reduce_max_f16(v_s_rowmax1);

            // Splat m_prev[r], m_prev[r+1] from the float per-row accumulators and convert to fp16 vectors
            // One 32-lane f32 vector now covers the whole unit, so the r_vec_off >= 32
            // arm is unreachable and m_prev_v1 no longer exists.
            HVX_Vector v_m_prev0, v_m_prev1;
            {
                HVX_Vector v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, r_vec_off * 4));
                v_m_prev0 = hvx_vec_f32_to_f16(v0, v0);
                if (r + 1 < n_rows_g) {
                    HVX_Vector v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, (r_vec_off + 1) * 4));
                    v_m_prev1 = hvx_vec_f32_to_f16(v1, v1);
                } else {
                    v_m_prev1 = Q6_V_vzero();
                }
            }

            HVX_Vector v_dup_m0 = Q6_Vhf_vmax_VhfVhf(v_m_prev0, v_s_rowmax0);
            HVX_Vector v_dup_m1 = Q6_Vhf_vmax_VhfVhf(v_m_prev1, v_s_rowmax1);

            // Insert row r, r+1 rowmax into rowmax_acc_v
            {
                HVX_VectorPred p_start = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_VectorPred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_VectorPred p_end   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_VectorPred p_lane0 = Q6_Q_and_QQn(p_mid, p_start);
                HVX_VectorPred p_lane1 = Q6_Q_and_QQn(p_end, p_mid);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane0, v_dup_m0, rowmax_acc_v);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane1, v_dup_m1, rowmax_acc_v);
            }

            // Compute P = exp(S - m_new)
            const HVX_Vector v_zero      = Q6_V_vzero();
            HVX_Vector       v_p_rowsum0 = v_zero;
            HVX_Vector       v_p_rowsum1 = v_zero;

            size_t c = 0;
            for (; c + 64 < kv_rows; c += 128) {
                size_t     ci0          = c / 64;
                size_t     ci1          = ci0 + 1;

                HVX_Vector v_s_minus_m0_0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci0], v_dup_m0);
                HVX_Vector v_s_minus_m1_0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci0], v_dup_m1);
                HVX_Vector v_s_minus_m0_1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci1], v_dup_m0);
                HVX_Vector v_s_minus_m1_1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci1], v_dup_m1);

                HVX_Vector v_p_row0_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_0));
                HVX_Vector v_p_row1_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_0));
                HVX_Vector v_p_row0_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_1));
                HVX_Vector v_p_row1_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_1));

                __fp16 *     out_dtile0  = p_st_base + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                __fp16 *     out_dtile1  = p_st_base + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_p_out0_0  = ((HVX_Vector *) out_dtile0) + r1 / 2;
                HVX_Vector * pv_p_out1_0  = pv_p_out0_0 + 16;
                HVX_Vector * pv_p_out0_1  = ((HVX_Vector *) out_dtile1) + r1 / 2;
                HVX_Vector * pv_p_out1_1  = pv_p_out0_1 + 16;

                HVX_VectorPair vp_p_dual0 = Q6_W_vshuff_VVR(v_p_row1_hf_0, v_p_row0_hf_0, -2);
                *pv_p_out0_0               = Q6_V_lo_W(vp_p_dual0);
                *pv_p_out1_0               = Q6_V_hi_W(vp_p_dual0);

                HVX_VectorPair vp_p_dual1 = Q6_W_vshuff_VVR(v_p_row1_hf_1, v_p_row0_hf_1, -2);
                *pv_p_out0_1               = Q6_V_lo_W(vp_p_dual1);
                *pv_p_out1_1               = Q6_V_hi_W(vp_p_dual1);

                HVX_VectorPair vp_p0_0 = hvx_vec_f16_to_f32_shuff(v_p_row0_hf_0);
                HVX_VectorPair vp_p1_0 = hvx_vec_f16_to_f32_shuff(v_p_row1_hf_0);
                HVX_VectorPair vp_p0_1 = hvx_vec_f16_to_f32_shuff(v_p_row0_hf_1);
                HVX_VectorPair vp_p1_1 = hvx_vec_f16_to_f32_shuff(v_p_row1_hf_1);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_0), Q6_V_hi_W(vp_p0_0)));
                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_1), Q6_V_hi_W(vp_p0_1)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_0), Q6_V_hi_W(vp_p1_0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_1), Q6_V_hi_W(vp_p1_1)));
            }
            for (size_t c_rem = c; c_rem < kv_rows; c_rem += 64) {
                size_t     ci           = c_rem / 64;
                HVX_Vector v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci], v_dup_m0);
                HVX_Vector v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci], v_dup_m1);

                HVX_Vector v_p_row0_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0));
                HVX_Vector v_p_row1_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1));
                __fp16 *     out_dtile  = p_st_base + ci * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_p_out0  = ((HVX_Vector *) out_dtile) + r1 / 2;
                HVX_Vector * pv_p_out1  = pv_p_out0 + 16;

                HVX_VectorPair vp_p_dual = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
                *pv_p_out0               = Q6_V_lo_W(vp_p_dual);
                *pv_p_out1               = Q6_V_hi_W(vp_p_dual);

                HVX_VectorPair vp_p0 = hvx_vec_f16_to_f32_shuff(v_p_row0_hf);
                HVX_VectorPair vp_p1 = hvx_vec_f16_to_f32_shuff(v_p_row1_hf);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0), Q6_V_hi_W(vp_p0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1), Q6_V_hi_W(vp_p1)));
            }

            HVX_Vector rowsum0_sf = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum0));
            HVX_Vector rowsum1_sf = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum1));
            {
                HVX_Vector rv0_v = hvx_vec_f32_to_f16(rowsum0_sf, rowsum0_sf);
                HVX_Vector rv1_v = hvx_vec_f32_to_f16(rowsum1_sf, rowsum1_sf);

                HVX_VectorPred p_start = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_VectorPred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_VectorPred p_end   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_VectorPred p_lane0 = Q6_Q_and_QQn(p_mid, p_start);
                HVX_VectorPred p_lane1 = Q6_Q_and_QQn(p_end, p_mid);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane0, rv0_v, rowsum_acc_v);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane1, rv1_v, rowsum_acc_v);
            }
        }

        // Inline fa_ml_update_and_build_d for this vector (lock-free and in parallel)
        // Only fp16 lanes 0..31 of the accumulators are live at 32-row granularity, so
        // every f32 "hi" half below is padding and is dropped. Feeding v_m_diff0 twice
        // to hvx_vec_f32_to_f16 keeps the pack lane-wise and leaves lanes 0..31 -- the
        // ones the D scatter and the l update read -- bit-identical to today's.
        HVX_VectorPair rowmax_acc_pair    = hvx_vec_f16_to_f32(rowmax_acc_v);
        HVX_Vector     v_rowmax_acc_f32_0 = Q6_V_lo_W(rowmax_acc_pair);

        HVX_Vector v_m_curr0 = Q6_Vsf_vmax_VsfVsf(m_prev_v0, v_rowmax_acc_f32_0);

        HVX_Vector v_m_diff0 = HVX_OP_SUB_F32(m_prev_v0, v_m_curr0);

        HVX_Vector v_m_diff_f16   = hvx_vec_f32_to_f16(v_m_diff0, v_m_diff0);
        HVX_Vector exp_m_diff_f16 = hvx_vec_exp2_f16(v_m_diff_f16);

        HVX_VectorPair exp_m_diff_pair = hvx_vec_f16_to_f32(exp_m_diff_f16);
        HVX_Vector exp_m_diff0 = Q6_V_lo_W(exp_m_diff_pair);

        HVX_VectorPair rowsum_acc_pair = hvx_vec_f16_to_f32(rowsum_acc_v);
        HVX_Vector     v_rowsum_acc_f32_0 = Q6_V_lo_W(rowsum_acc_pair);

        HVX_Vector v_l_curr0;
        if (args->is_first_block && factx->sinks != NULL) {
            // First KV block with sinks: m_prev holds the seeded sink value (not -inf),
            // so exp_m_diff = exp2(sink - m_curr) is the sink's contribution to the
            // denominator. l_prev is 0 here, so add exp_m_diff directly instead of
            // multiplying the (uninitialized) l_prev term.
            v_l_curr0 = HVX_OP_ADD_F32(exp_m_diff0, v_rowsum_acc_f32_0);
        } else {
            HVX_Vector l_prev_v0 = factx->vtcm_l_vec[r_vec_idx];
            v_l_curr0 = HVX_OP_ADD_F32(HVX_OP_MUL_F32(l_prev_v0, exp_m_diff0), v_rowsum_acc_f32_0);
        }

        factx->vtcm_m_vec[r_vec_idx] = v_m_curr0;
        factx->vtcm_l_vec[r_vec_idx] = v_l_curr0;

        // Build diagonal tile D = diag(exp(m_diff))
        const HVX_Vector     v_offsets = *(const HVX_Vector *) d_tile_scatter_offsets;
        const HVX_VectorPred q_32_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
        HVX_Vector           v_exp_m_diff = exp_m_diff_f16;

        __fp16 * const d_tiles_out = factx->vtcm_d_tiles[args->buf_idx];

        // n_row_vec_cnt == ceil(n_rows_g/32) == n_row_tiles, so a unit maps 1:1 onto a
        // D tile and the second scatter (with its 64-BYTE = 32-lane ror) disappears.
        // The bound check is kept as belt and braces.
        if (r_vec_idx < args->n_row_tiles) {
            __fp16 * out_base = d_tiles_out + r_vec_idx * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_exp_m_diff);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (args->q_start * G + vec_start * 32));
}

static void fa_softmax_thread_nomask(unsigned int n, unsigned int i, void * data) {
    fa_softmax_impl(n, i, data,
                    /*has_mask=*/false,
                    /*mask_broadcast=*/false,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

static void fa_softmax_thread_mask_broadcast_g1(unsigned int n, unsigned int i, void * data) {
    fa_softmax_impl(n, i, data,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/true,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

static void fa_softmax_thread_mask_broadcast_gn(unsigned int n, unsigned int i, void * data) {
    fa_softmax_impl(n, i, data,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

static void fa_softmax_thread(unsigned int n, unsigned int i, void * data) {
    fa_softmax_args_t *     args  = (fa_softmax_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const bool has_mask       = (args->mask != NULL);
    const bool mask_broadcast = factx->mask_broadcast;
    const bool is_g1          = (args->G == 1);
    const bool has_alibi      = args->has_alibi;
    const bool has_softcap    = (factx->logit_softcap != 0.0f);

    fa_softmax_impl(n, i, data, has_mask, mask_broadcast, is_g1, has_alibi, has_softcap);
}

static __attribute__((noinline)) void fa_build_d_diag_inv_l(struct hmx_fa_context * factx,
                                                            size_t                  n_row_tiles,
                                                            size_t                  n_row_tiles_g_br) {
    const HVX_Vector     v_offsets = *(const HVX_Vector *) d_tile_scatter_offsets;
    const HVX_VectorPred q_32_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
    const HVX_Vector     one       = hvx_vec_splat_f32(1.0f);

    HVX_Vector v_content = Q6_V_vzero();
    for (size_t i = 0; i < n_row_tiles; ++i) {
        if ((i % 2) == 0) {
            HVX_Vector inv_lo = HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i]));
            HVX_Vector inv_hi = (i + 1 < n_row_tiles) ? HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i + 1])) : Q6_V_vzero();
            v_content = hvx_vec_f32_to_f16(inv_lo, inv_hi);
        } else {
            v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 * out_base = factx->vtcm_d_inv_l + i * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
    }
}

static void fa_phase_softmax_and_build_d(struct hmx_fa_context * factx,
                                         fa_softmax_args_t *     sargs,
                                         size_t                  n_row_tiles,
                                         size_t                  n_row_tiles_g_br) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    const size_t n_row_vec_cnt = hmx_ceil_div(sargs->n_rows_g, 32);

    worker_callback_t softmax_fn = fa_softmax_thread;
    if (sargs->mask == NULL && factx->logit_softcap == 0.0f && !sargs->has_alibi) {
        softmax_fn = fa_softmax_thread_nomask;
    } else if (sargs->mask != NULL && factx->mask_broadcast && factx->logit_softcap == 0.0f && !sargs->has_alibi) {
        if (sargs->G == 1) {
            softmax_fn = fa_softmax_thread_mask_broadcast_g1;
        } else {
            softmax_fn = fa_softmax_thread_mask_broadcast_gn;
        }
    }

    // Fork on the same work threshold as the 64-row scheme did, not the same unit
    // count. Halving the granularity doubled n_row_vec_cnt, so a bare ">= 2" would
    // newly fork the band n_rows_g in [33,64] -- one old unit's worth of work -- and
    // pay a fork/join per KV block for it. No shape in the perf suite lands in that
    // band, so the trade is unmeasured; keep the old behaviour there and let the
    // change be strictly "same threads or more".
    if (factx->n_threads > 1 && n_row_vec_cnt >= 3) {
        uint32_t n_use = (uint32_t) hex_smin((size_t) factx->n_threads, n_row_vec_cnt);
        sargs->thread_div = init_fastdiv_values(n_use);
        work_queue_run(wp, softmax_fn, sargs, n_use);
    } else {
        softmax_fn(1, 0, sargs);
    }
}

// ============================================================================
// HMX job structs and worker functions
// ============================================================================

typedef struct {
    const __fp16 * q_tiles;
    const __fp16 * k_tiles;
    __fp16 *       s_tiles;
    size_t         n_row_tiles;
    size_t         n_col_tiles;
    size_t         n_dot_tiles;  // DK / 32
    size_t         n_tiles_per_bc;
    uint8_t *      hmx_scales;
} hmx_fa_qk_job_t;

static void hmx_fa_qk_dot_worker(void * data) {
    hmx_fa_qk_job_t * job            = (hmx_fa_qk_job_t *) data;
    const size_t      n_row_tiles    = job->n_row_tiles;
    const size_t      n_col_tiles    = job->n_col_tiles;
    const size_t      n_dot_tiles    = job->n_dot_tiles;
    const size_t      n_tiles_per_bc = job->n_tiles_per_bc;
    const __fp16 * restrict q_tiles  = job->q_tiles;
    const __fp16 * restrict k_tiles  = job->k_tiles;
    __fp16 * restrict s_tiles        = job->s_tiles;
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)job->hmx_scales));
    const size_t dot_stride = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    for (size_t r = 0; r < n_row_tiles; ++r) {
        const __fp16 * row_tiles = q_tiles + r * dot_stride;
        const __fp16 * col_tiles = k_tiles;
        __fp16 *       out_tile  = s_tiles + r * n_tiles_per_bc * HMX_FP16_TILE_N_ELMS;

        for (size_t c = 0; c < n_col_tiles; ++c) {
            hmx_fa_qk_dot_tile(row_tiles, col_tiles, out_tile, n_dot_tiles);
            col_tiles += dot_stride;
            out_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

typedef struct {
    __fp16 *       o_curr;
    const __fp16 * o_prev;
    const __fp16 * p_tiles;
    const __fp16 * v_tiles;
    const __fp16 * d_tiles;
    uint8_t *      hmx_scales;
    size_t         n_row_tiles;
    size_t         n_col_tiles;
    size_t         n_row_tiles_g_br;
    size_t         n_tiles_per_bc;
    size_t         DV;
} hmx_fa_o_update_job_t;

static void hmx_fa_o_update_worker(void * data) {
    hmx_fa_o_update_job_t * job              = (hmx_fa_o_update_job_t *) data;
    const size_t            n_row_tiles      = job->n_row_tiles;
    const size_t            n_col_tiles      = job->n_col_tiles;
    const size_t            n_row_tiles_g_br = job->n_row_tiles_g_br;
    const size_t            n_tiles_per_bc   = job->n_tiles_per_bc;
    const size_t            DV_tiles         = job->DV / 32;
    const __fp16 * restrict d_tiles          = job->d_tiles;
    const __fp16 * restrict p_tiles          = job->p_tiles;
    const __fp16 * restrict v_tiles          = job->v_tiles;
    const __fp16 * restrict o_prev           = job->o_prev;
    __fp16 * restrict o_curr                 = job->o_curr;
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)job->hmx_scales));
    const size_t o_stride = n_row_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    const size_t v_stride = n_tiles_per_bc * HMX_FP16_TILE_N_ELMS;
    for (size_t r = 0; r < n_row_tiles; ++r) {
        const __fp16 * d_diag     = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        const __fp16 * p_tile_in  = p_tiles + (r * n_tiles_per_bc) * HMX_FP16_TILE_N_ELMS;
        const __fp16 * o_rc       = o_prev + r * HMX_FP16_TILE_N_ELMS;
        const __fp16 * v_tile_in  = v_tiles;
        __fp16       * o_tile_out = o_curr + r * HMX_FP16_TILE_N_ELMS;

        for (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_update_tile(d_diag, o_rc, p_tile_in, v_tile_in, o_tile_out, n_col_tiles);
            o_rc       += o_stride;
            v_tile_in  += v_stride;
            o_tile_out += o_stride;
        }
    }
}

typedef struct {
    __fp16 *       o_curr;   // output (row-major tile layout)
    const __fp16 * o_prev;   // input (column-major tile layout)
    const __fp16 * d_tiles;  // diag(1/l) tiles
    uint8_t *      hmx_scales;
    size_t         n_row_tiles;
    size_t         n_row_tiles_g_br;
    size_t         DV;
} hmx_fa_o_norm_job_t;

static void hmx_fa_o_norm_worker(void * data) {
    hmx_fa_o_norm_job_t * job              = (hmx_fa_o_norm_job_t *) data;
    const size_t          n_row_tiles      = job->n_row_tiles;
    const size_t          n_row_tiles_g_br = job->n_row_tiles_g_br;
    const size_t          DV_tiles         = job->DV / 32;
    const __fp16 * restrict d_tiles        = job->d_tiles;
    const __fp16 * restrict o_prev         = job->o_prev;
    __fp16 * restrict o_curr               = job->o_curr;
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)job->hmx_scales));
    const size_t o_stride = n_row_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    for (size_t r = 0; r < n_row_tiles; ++r) {
        const __fp16 * d_diag = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        const __fp16 * o_rc = o_prev + r * HMX_FP16_TILE_N_ELMS;
        __fp16 *       o_out = o_curr + r * DV_tiles * HMX_FP16_TILE_N_ELMS;

        for (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_norm_tile(d_diag, o_rc, o_out);
            o_rc  += o_stride;
            o_out += HMX_FP16_TILE_N_ELMS;
        }
    }
}

// Populate per-GQA-row ALiBi slopes for a given KV head.
static __attribute__((noinline)) void fa_compute_slopes(
                              const struct hmx_fa_context * factx,
                              uint32_t                      kv_head,
                              size_t                        n_rows_g) {
    __fp16 * slopes = factx->vtcm_slopes;
    if (factx->max_bias == 0.0f) {
        hvx_splat_f16_a(slopes, 1.0f, n_rows_g);
        return;
    }

    const uint32_t G           = factx->G;
    const uint32_t n_head_log2 = factx->n_head_log2;
    const float    m0          = factx->m0;
    const float    m1          = factx->m1;

    __fp16 temp_slopes[512] __attribute__((aligned(128)));
    if (G <= 32) {
        // Fast path: Compute G unique slope values in vector registers
        HVX_Vector v_val = hvx_alibi_slopes(kv_head, G, n_head_log2, m0, m1);

        __fp16 temp_slopes_aligned[64] __attribute__((aligned(128)));
        hvx_vmem(temp_slopes_aligned) = hvx_vec_f32_to_f16(v_val, Q6_V_vzero());

        for (uint32_t i = 0; i < G; ++i) {
            temp_slopes[i] = temp_slopes_aligned[i];
        }
    } else {
        // Fallback path: G > 32 (rare configurations)
        for (uint32_t i = 0; i < G; ++i) {
            temp_slopes[i] = (__fp16)alibi_slope(kv_head * G + i, n_head_log2, m0, m1);
        }
    }

    // Allocate stack buffer to avoid scalar writes to VTCM (which generates L2 misses)
    __fp16 local_slopes[n_rows_g] __attribute__((aligned(128)));
    for (size_t r = 0; r < n_rows_g; ++r) {
        local_slopes[r] = temp_slopes[fastmodulo(r, G, &factx->div_G)];
    }

    // Copy to VTCM slopes using HVX block copy (both are aligned to 128 bytes)
    hvx_copy_f16_aa((uint8_t *)slopes, (const uint8_t *)local_slopes, n_rows_g);
}

static void fa_push_mask_dma_gqa(
    dma_queue *               dma,
    const struct htp_tensor * mask,
    uint32_t                  q_start,
    uint32_t                  im3,
    uint32_t                  kv_start,
    uint32_t                  kv_head,
    uint32_t                  G,
    uint32_t                  m_line_bytes,
    uint32_t                  kv_rows,
    uint32_t                  n_rows_q,
    uint32_t                  buf_idx,
    struct hmx_fa_context *   factx
) {
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t h_idx = kv_head * G + g;
        const uint32_t im2 = fastmodulo(h_idx, mask->ne[2], &factx->src3_div2);
        const uint8_t * ms_src = (const uint8_t *) mask->data + q_start * mask->nb[1] +
                                 im2 * mask->nb[2] + im3 * mask->nb[3] + kv_start * sizeof(__fp16);
        uint8_t * ms_dst = (uint8_t *) (factx->vtcm_mask_buf + buf_idx * factx->mask_buf_gqa_stride)
                           + g * m_line_bytes;
        dma_queue_push(dma, dma_make_ptr(ms_dst, ms_src), G * m_line_bytes, mask->nb[1], kv_rows * sizeof(__fp16), n_rows_q);
    }
}

static void fa_pop_mask_dma_gqa(dma_queue * dma, uint32_t G) {
    for (uint32_t g = 0; g < G; ++g) {
        dma_queue_pop(dma);
    }
}

// Stage one chunk: its nblk selected blocks are scattered in KV, so each needs its
// own 2D descriptor into consecutive slots of the same Bc-wide VTCM buffer.
//
// Push order is grouped PER TENSOR (K*nblk, V*nblk, mask*nblk), never per sub-block.
// The queue is FIFO and K's pop site sits one loop iteration away from V's, so a run
// of nblk K pops happens at a single site. Interleaving would not assert or hang --
// it would silently read K out of the V buffer.
//
// q_start names the query tile whose data is being staged: it selects the mask rows
// AND, with a per-query-block selection, the sel[] row. The row is derived HERE from
// that same q_start rather than passed in, because the tail prefetch at the bottom of
// the loop nest stages the NEXT iteration's chunk 0 (passing next_q_start) while the
// consumer one iteration later re-derives the row from its own q_start. Deriving it
// in one place makes producer and consumer agree by construction -- a mismatch would
// DMA one block's rows and tile them as another's, with no assert and no NaN.
static inline uint32_t fa_push_chunk(dma_queue * dma, const struct htp_tensor * k, const struct htp_tensor * v,
                                     const struct htp_tensor * mask, uint32_t c,
                                     size_t size_k_row_padded, size_t size_k_row,
                                     size_t size_v_row_padded, size_t size_v_row,
                                     uint32_t ik2, uint32_t ik3, uint32_t iv2, uint32_t iv3,
                                     uint32_t q_start, uint32_t im3, uint32_t kv_head, uint32_t G,
                                     size_t m_line_bytes, size_t n_rows_q, size_t nek1,
                                     size_t buf, uint32_t ib3, bool push_mask,
                                     struct hmx_fa_context * factx) {
    const uint32_t bs   = factx->sel ? factx->sparse_bs : (uint32_t) factx->Bc;
    const uint32_t qb   = fa_sel_row(factx, q_start);
    const uint32_t nblk = fa_chunk_nblk(factx, c, fa_row_nsel(factx, qb, kv_head, ib3));

    for (uint32_t j = 0; j < nblk; ++j) {
        uint32_t  start;
        uint8_t * dst;
        bool      resident = false;
        if (factx->k_res) {
            const uint32_t idx = fa_chunk_block_idx(factx, c, j, qb, kv_head, ib3);
            start    = idx * factx->sparse_bs;
            dst      = (uint8_t *) factx->k_res + (size_t) idx * factx->k_res_slot;
            resident = fa_res_mark(factx->k_res_valid, idx, factx->res_force_miss);
        } else {
            start = fa_chunk_block_start(factx, c, j, qb, kv_head, ib3);
            dst   = (uint8_t *) factx->vtcm_k_fp16[buf] + (size_t) j * bs * size_k_row_padded;
        }
        const uint32_t rows = fa_rows_at(factx, start, nek1);
        const uint8_t * src = (const uint8_t *) k->data + start * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
        if (resident) {
            // One descriptor per block whether or not a transfer is needed. A zero-size
            // 1D push sets done and skips dmlink, so it moves no bytes -- but it still
            // advances push_idx and records dst, which is what keeps the pop counts one
            // loop iteration away a function of fa_chunk_nblk alone and hands the
            // consumer the slot address through the same channel as a real transfer.
            dma_queue_push_single_1d(dma, dma_make_ptr(dst, src), 0);
        } else {
            dma_queue_push(dma, dma_make_ptr(dst, src), size_k_row_padded, k->nb[1], size_k_row, rows);
        }
    }
    for (uint32_t j = 0; j < nblk; ++j) {
        uint32_t  start;
        uint8_t * dst;
        bool      resident = false;
        if (factx->v_res) {
            const uint32_t idx = fa_chunk_block_idx(factx, c, j, qb, kv_head, ib3);
            start    = idx * factx->sparse_bs;
            dst      = (uint8_t *) factx->v_res + (size_t) idx * factx->v_res_slot;
            resident = fa_res_mark(factx->v_res_valid, idx, factx->res_force_miss);
        } else {
            start = fa_chunk_block_start(factx, c, j, qb, kv_head, ib3);
            dst   = (uint8_t *) factx->vtcm_v_fp16[buf] + (size_t) j * bs * size_v_row_padded;
        }
        const uint32_t rows = fa_rows_at(factx, start, nek1);
        const uint8_t * src = (const uint8_t *) v->data + start * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
        if (resident) {
            dma_queue_push_single_1d(dma, dma_make_ptr(dst, src), 0);
        } else {
            dma_queue_push(dma, dma_make_ptr(dst, src), size_v_row_padded, v->nb[1], size_v_row, rows);
        }
    }

    if (mask && push_mask) {
        if (__builtin_expect(factx->mask_use_cache, true)) {
            // m == 1 broadcast: the cache picks its own slot and pushes exactly one
            // descriptor, so it can only serve a chunk that is a single block.
            const uint32_t start = fa_chunk_block_start(factx, c, 0, qb, kv_head, ib3);
            const uint32_t rows  = fa_block_rows(factx, c, 0, qb, kv_head, ib3, nek1);
            const uint8_t * ms_src = (const uint8_t *) mask->data + q_start * mask->nb[1] +
                                     im3 * mask->nb[3] + start * sizeof(__fp16);
            dma_cache_push(dma, &factx->m_cache, ms_src, m_line_bytes, mask->nb[1], rows * sizeof(__fp16), n_rows_q);
        } else if (__builtin_expect(factx->mask_broadcast, true)) {
            __fp16 * base = factx->vtcm_mask_buf + buf * factx->mask_slot_stride;
            for (uint32_t j = 0; j < nblk; ++j) {
                const uint32_t start = fa_chunk_block_start(factx, c, j, qb, kv_head, ib3);
                const uint32_t rows  = fa_block_rows(factx, c, j, qb, kv_head, ib3, nek1);
                const uint8_t * ms_src = (const uint8_t *) mask->data + q_start * mask->nb[1] +
                                         im3 * mask->nb[3] + start * sizeof(__fp16);
                uint8_t * ms_dst = (uint8_t *) base + (size_t) j * bs * sizeof(__fp16);
                dma_queue_push(dma, dma_make_ptr(ms_dst, ms_src), m_line_bytes, mask->nb[1],
                               rows * sizeof(__fp16), n_rows_q);
            }
        } else {
            // Per-head mask: grouping is not enabled for this case (m == 1), so one block.
            const uint32_t start = fa_chunk_block_start(factx, c, 0, qb, kv_head, ib3);
            const uint32_t rows  = fa_block_rows(factx, c, 0, qb, kv_head, ib3, nek1);
            fa_push_mask_dma_gqa(dma, mask, q_start, im3, start, kv_head, G, m_line_bytes, rows, n_rows_q, buf, factx);
        }
    }
    return nblk;
}

// Pop a chunk's n descriptors and return its VTCM base. The n descriptors were pushed
// consecutively with dst = base + j*block_bytes, so the FIRST one's dst is the base --
// which lets the fallback path stay agnostic about which double-buffer slot it landed in.
static inline void * fa_pop_chunk_base(dma_queue * dma, uint32_t n) {
    void * base = dma_queue_pop(dma).dst;
    for (uint32_t j = 1; j < n; ++j) {
        dma_queue_pop(dma);
    }
    return base;
}

static inline void fa_pop_n(dma_queue * dma, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        dma_queue_pop(dma);
    }
}

// Pop a chunk's n K (or V) descriptors, recording each block's VTCM address.
//
// With residency the blocks land in scattered slots, so the address is knowable ONLY
// from the descriptor -- fa_pop_n's "discard the return" form cannot serve here, and
// neither can fa_pop_chunk_base's "first dst is the base" assumption. A hit's zero-work
// descriptor carries the same slot address a real transfer would have, so the consumer
// needs no hit/miss channel at all.
static inline void fa_pop_bases(dma_queue * dma, uint32_t n, void ** bases) {
    for (uint32_t i = 0; i < n; ++i) {
        bases[i] = dma_queue_pop(dma).dst;
    }
}

static inline void fa_prefetch_block(dma_queue * dma, const struct htp_tensor * k, const struct htp_tensor * v, const struct htp_tensor * mask,
                                     uint32_t b, size_t Bc, size_t size_k_row_padded, size_t size_k_row, size_t size_v_row_padded, size_t size_v_row,
                                     uint32_t ik2, uint32_t ik3, uint32_t iv2, uint32_t iv3, uint32_t q_start, uint32_t im3, uint32_t kv_head, uint32_t G,
                                     size_t m_line_bytes, size_t n_rows_q, size_t nek1, size_t prefetch_buf, uint32_t ib3, struct hmx_fa_context * factx) {
    (void) Bc;
    fa_push_chunk(dma, k, v, mask, b, size_k_row_padded, size_k_row, size_v_row_padded, size_v_row,
                  ik2, ik3, iv2, iv3, q_start, im3, kv_head, G, m_line_bytes, n_rows_q, nek1,
                  prefetch_buf, ib3, /*push_mask=*/mask != NULL, factx);
}

// ============================================================================
// Iteration order over (sequence, query block, KV head)
// ============================================================================
//
// The nest is linearised so that producer and consumer agree BY CONSTRUCTION: the loop
// body runs fa_iter_at(it) and the tail prefetch stages fa_iter_at(it + 1). Two
// hand-written successor formulas that must be kept in lockstep is exactly the shape of
// bug that cannot be caught downstream -- fa_chunk_nblk is deliberately blind to the
// query block, a cache hit's dummy descriptor preserves mask parity, and wrong-but-real
// K rows dotted with a real Q tile produce plausible finite numbers. No assert, no hang,
// no NaN.
//
// kv_head_outer swaps which of the two inner axes varies fastest. KV block residency
// needs the KV head OUTSIDE the query block loop, because its slots are keyed by block
// index alone: with the head inside, a fixed head's query blocks are visited in
// n_kv_heads-separated iterations and every slot is overwritten in between. Dense and
// shared-selection keep kv_head_outer = false, i.e. the original order, byte for byte.
struct fa_iter_order {
    uint32_t n_q_blocks;
    uint32_t n_kv_heads;
    bool     kv_head_outer;
};

struct fa_iter {
    uint32_t ib3;
    uint32_t qb_idx;
    uint32_t kv_head;
};

static inline struct fa_iter fa_iter_at(const struct fa_iter_order * o, uint32_t it) {
    const uint32_t per_seq = o->n_q_blocks * o->n_kv_heads;
    struct fa_iter r;
    r.ib3 = it / per_seq;
    const uint32_t rem = it - r.ib3 * per_seq;
    if (o->kv_head_outer) {
        r.kv_head = rem / o->n_q_blocks;
        r.qb_idx  = rem - r.kv_head * o->n_q_blocks;
    } else {
        r.qb_idx  = rem / o->n_kv_heads;
        r.kv_head = rem - r.qb_idx * o->n_kv_heads;
    }
    return r;
}

// Does any KV block appear in more than one query block's selection row?
//
// Residency only pays when it does, and it is not free: it forces the KV-head loop
// outside the query-block loop, which costs a broadcast mask its dma_cache reuse across
// heads. The eval suite deliberately builds pairwise-DISJOINT per-query-block
// selections, so without this probe those shapes would pay the reorder for a guaranteed
// 0% hit rate. Head 0 / sequence 0 is a representative sample, not a correctness input:
// a wrong answer here only turns the optimisation on or off.
static bool fa_sel_repeats_blocks(const struct hmx_fa_context * factx, uint32_t n_q_blocks) {
    uint32_t seen[FA_RES_MAX_BLOCKS / 32];
    memset(seen, 0, sizeof(seen));

    uint32_t distinct = 0;
    uint32_t total    = 0;
    for (uint32_t i = 0; i < n_q_blocks; ++i) {
        const uint32_t  qb   = fa_sel_row(factx, i * factx->Br);
        const int32_t * list = (const int32_t *) ((const uint8_t *) factx->sel + qb * factx->sel_nb1);
        const uint32_t  ns   = fa_row_nsel(factx, qb, 0, 0);
        for (uint32_t s = 0; s < ns; ++s) {
            uint32_t idx = (uint32_t) list[s];
            if (idx >= factx->n_blk_total) {
                idx = factx->n_blk_total - 1;
            }
            const uint32_t w   = idx >> 5;
            const uint32_t bit = 1u << (idx & 31);
            if (!(seen[w] & bit)) {
                seen[w] |= bit;
                distinct++;
            }
            total++;
        }
    }
    return distinct < total;
}

// ============================================================================
// Core HMX flash attention algorithm (GQA-merged)
// ============================================================================

int hmx_flash_attn_ext(struct htp_ops_context * octx) {
    struct htp_thread_trace * tr_hvx = &octx->ctx->trace[0];
    struct htp_thread_trace * tr_hmx = &octx->ctx->trace[HTP_MAX_NTHREADS];
    const struct htp_tensor * q    = octx->src[0];
    const struct htp_tensor * k    = octx->src[1];
    const struct htp_tensor * v    = octx->src[2];
    const struct htp_tensor * mask = (octx->src[3] && octx->src[3]->data) ? octx->src[3] : NULL;
    const struct htp_tensor * dst  = octx->dst;

    struct htp_context * const ctx = octx->ctx;

    if (!ctx->hmx_enabled) {
        return HTP_STATUS_NO_SUPPORT;
    }

    // Dimensions
    const uint32_t neq0 = q->ne[0];  // head_dim (DK)
    const uint32_t neq1 = q->ne[1];  // n_tokens
    const uint32_t neq2 = q->ne[2];  // n_heads
    const uint32_t neq3 = q->ne[3];  // n_seqs

    const uint32_t nek0 = k->ne[0];  // head_dim
    const uint32_t nek1 = k->ne[1];  // kv_len

    const uint32_t nev0 = v->ne[0];  // head_dim (DV)

    const uint32_t DK = neq0;
    const uint32_t DV = nev0;

    // HMX requires head_dim to be multiple of 32
    if (DK % 32 != 0 || DV % 32 != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_fa_kernel_params * kparams = (const struct htp_fa_kernel_params *) octx->kernel_params;
    const uint32_t n_kv_heads = k->ne[2];

    // ======== Build context ========
    struct hmx_fa_context factx;
    memset(&factx, 0, sizeof(factx));
    factx.octx           = octx;
    factx.sinks          = octx->src[4];  // NULL if this op has no attention sinks
    factx.n_threads      = kparams->n_threads;
    factx.DK             = DK;
    factx.DV             = DV;
    factx.n_kv           = nek1;
    factx.n_kv_heads     = n_kv_heads;
    factx.n_heads        = neq2;
    factx.G              = kparams->G;
    factx.div_G          = kparams->u.hmx.div_G;
    factx.neq1           = neq1;
    factx.Br             = kparams->Br;
    factx.Bc             = kparams->Bc;
    factx.g_br           = kparams->u.hmx.g_br;
    factx.n_kv_blocks    = kparams->n_kv_blocks;
    factx.is_q_fp32      = (kparams->is_q_fp32 != 0);
    factx.is_dst_fp32    = (kparams->is_dst_fp32 != 0);
    factx.pipeline       = (kparams->u.hmx.pipeline != 0);
    factx.mask_broadcast = (kparams->u.hmx.mask_broadcast != 0);
    if (mask) {
        factx.src3_div2  = kparams->src3_div2;
        factx.src3_div3  = kparams->src3_div3;
    }

    // Block-sparse selection list (optional). The host guarantees the layout:
    // I32, rows unit-strided, [n_sel, NBq or 1, n_kv_heads or 1, n_seqs or 1], and
    // pins Bc to the block size the indices are expressed in. Only nb[0] is pinned
    // (a strided ggml_argsort_top_k view is legal), so nb[1..3] are honoured as
    // strides. A broadcast dim has ne == 1, so its stride is zeroed here and the
    // same list serves every query block / head / sequence.
    {
        const struct htp_tensor * sel = octx->src[5];
        if (sel && sel->data) {
            factx.sel       = (const int32_t *) sel->data;
            factx.sel_nb1   = (sel->ne[1] > 1) ? sel->nb[1] : 0;
            factx.sel_nb2   = (sel->ne[2] > 1) ? sel->nb[2] : 0;
            factx.sel_nb3   = (sel->ne[3] > 1) ? sel->nb[3] : 0;
            factx.sel_nq    = sel->ne[1];
            factx.sparse_bs = kparams->u.hmx.sparse_bs;
            factx.n_sel     = kparams->u.hmx.n_sel;
            factx.m         = factx.sparse_bs ? (factx.Bc / factx.sparse_bs) : 1;
            // The per-chunk block-address arrays in the KV loop are FA_SPARSE_MAX_M
            // deep, and the staging buffers are sized for the same bound. The host
            // caps Bc so this holds; reject rather than overrun if it ever does not.
            if (factx.m > FA_SPARSE_MAX_M) {
                return HTP_STATUS_NO_SUPPORT;
            }
            factx.n_blk_total = factx.sparse_bs ? ((nek1 + factx.sparse_bs - 1) / factx.sparse_bs) : 0;
            // The unit of the query axis is the SCORER's query-block size, which has no
            // relation to the kernel's Br -- it rides in op_params[5] and reaches here
            // via kparams. Zero means "one row per selection block", which is what the
            // XAttention scorer emits (one Bl for both the query and the key axis).
            const uint32_t sel_bq = kparams->u.hmx.sel_bq ? kparams->u.hmx.sel_bq : factx.sparse_bs;
            factx.div_sel_bq = init_fastdiv_values(sel_bq ? sel_bq : 1);

            // Per-row selection length. The host promises the shape mirrors sel's row
            // axes (ne[0] == sel->ne[1] and so on); broadcast dims zero their stride so
            // one count can serve every head or sequence, exactly like sel itself.
            if (kparams->u.hmx.dyn_sel) {
                const struct htp_tensor * cnt = octx->src[6];
                if (!cnt || !cnt->data) {
                    return HTP_STATUS_NO_SUPPORT;
                }
                factx.cnt         = (const float *) cnt->data;
                factx.cnt_nb_qb   = (cnt->ne[0] > 1) ? cnt->nb[0] : 0;
                factx.cnt_nb_head = (cnt->ne[1] > 1) ? cnt->nb[1] : 0;
                factx.cnt_nb_seq  = (cnt->ne[2] > 1) ? cnt->nb[2] : 0;
            }
        }
    }

    if (kparams->logit_softcap == 0.0f) {
        factx.scale = (__fp16) (kparams->scale * EXP_LOG2E_F);  // log2(e)
    } else {
        factx.scale = (__fp16) kparams->scale;
    }
    factx.max_bias      = kparams->max_bias;
    factx.logit_softcap = (__fp16) (kparams->logit_softcap * EXP_LOG2E_F);

    factx.n_head_log2 = kparams->n_head_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    const uint32_t Br = factx.Br;
    const uint32_t Bc = factx.Bc;
    const uint32_t g_br = factx.g_br;
    const bool pipeline = factx.pipeline;
    const uint32_t n_threads = factx.n_threads;
    const uint32_t G = factx.G;

    // ======== VTCM allocation (GQA-aware) ========
    // K/V row sizes drive the DMA descriptors (not the VTCM layout) and are used
    // throughout the KV loop below.
    const size_t size_k_row        = DK * sizeof(__fp16);
    const size_t size_v_row        = DV * sizeof(__fp16);
    const size_t size_k_row_padded = hex_round_up(size_k_row, 128);
    const size_t size_v_row_padded = hex_round_up(size_v_row, 128);

    // Build the VTCM layout once (shared with the host estimator) and place every
    // scratch buffer at its computed offset.
    struct hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, G, DK, DV, Br, Bc, n_threads, pipeline, factx.is_q_fp32,
                             (mask != NULL) && !factx.mask_broadcast);

    if (L.total_bytes > ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = ctx->vtcm_base;

    factx.vtcm_q_dma          = VTCM_LAYOUT_PTR(__fp16, base, L.off_q_dma);
    factx.vtcm_q_tiles        = VTCM_LAYOUT_PTR(__fp16, base, L.off_q_tiles);
    factx.vtcm_o_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_o_tiles[0]);
    factx.vtcm_o_tiles[1]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_o_tiles[1]);
    factx.vtcm_k_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_k_fp16[0]);
    factx.vtcm_k_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_k_fp16[1]);
    factx.vtcm_v_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_v_fp16[0]);
    factx.vtcm_v_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_v_fp16[1]);
    factx.vtcm_k_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_k_tiles[0]);
    factx.vtcm_k_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_k_tiles[1], pipeline);
    factx.vtcm_v_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_v_tiles[0]);
    factx.vtcm_v_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_v_tiles[1], pipeline);
    factx.vtcm_s_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_s_tiles[0]);
    factx.vtcm_s_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_s_tiles[1], pipeline);
    factx.vtcm_p_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_p_tiles[0]);
    factx.vtcm_p_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_p_tiles[1], pipeline);
    factx.vtcm_d_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_d_tiles[0]);
    factx.vtcm_d_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_d_tiles[1], pipeline);
    factx.vtcm_d_inv_l        = VTCM_LAYOUT_PTR(__fp16, base, L.off_d_inv_l);
    factx.vtcm_m_vec          = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_m_vec);
    factx.vtcm_l_vec          = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_l_vec);
    factx.vtcm_s_rowmax       = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_s_rowmax);
    factx.vtcm_p_rowsum       = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_p_rowsum);
    factx.vtcm_row_bufs       = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_row_bufs);
    factx.row_buf_stride      = L.row_buf_stride;
    factx.vtcm_hmx_scales_id  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_hmx_scales_id);
    factx.vtcm_hmx_scales_qk  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_hmx_scales_qk);
    factx.vtcm_mask_buf       = VTCM_LAYOUT_PTR(__fp16, base, L.off_mask_buf);
    factx.mask_buf_row_stride = L.mask_buf_row_stride;
    factx.mask_buf_gqa_stride = (G * L.m_buf_slot_bytes) / sizeof(__fp16);
    factx.mask_slot_stride    = L.m_buf_slot_bytes / sizeof(__fp16);
    // dma_cache picks its own slot and pushes exactly one descriptor, so it can only
    // serve a chunk that is a single block.
    factx.mask_use_cache      = (factx.m <= 1) && factx.mask_broadcast;
    factx.q_tile_bytes        = L.q_tile_bytes;
    factx.o_tile_bytes        = L.o_tile_bytes;
    factx.col_vec_bytes       = L.col_vec_bytes;
    factx.d_tile_bytes        = L.d_tile_bytes;
    factx.vtcm_slopes         = VTCM_LAYOUT_PTR(__fp16, base, L.off_slopes);

    const size_t m_line_bytes = L.m_line_bytes;  // used by the mask DMAs in the KV loop

    dma_cache_init(&factx.m_cache, (uint8_t *) factx.vtcm_mask_buf, L.m_buf_slot_bytes, HMX_FA_DMA_CACHE_SIZE);

    // ======== Initialize HMX output scales ========
    hmx_init_column_scales(factx.vtcm_hmx_scales_id, Q6_V_vsplat_R(0x3c00)); // 1.0
    hmx_init_column_scales(factx.vtcm_hmx_scales_qk, hvx_vec_splat_f16(factx.scale));

    // ======== Skip compute if profiling ========
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    // ======== KV block residency map ========
    //
    // Per-query-block selection re-stages the SAME KV block once per query block: the
    // loop visits (query block, KV head) pairs, and a block that several query blocks of
    // one head select is DMA'd once per pair. Giving every KV block index its own VTCM
    // slot collapses that to once per head.
    //
    // Scoped hard to sparse + per-query-block + pipelined. The other arms are excluded
    // on purpose, not by omission:
    //   - shared selection: all query blocks read the same list, so the reuse the map
    //     would exploit is already exploited; and that arm is not DMA-bound.
    //   - dense: blocks are walked in order with no reuse across query tiles at a fixed
    //     head, and the loop swap the map requires would destroy the broadcast mask
    //     cache's cross-head reuse, which is the whole reason it hits there.
    //   - non-pipelined fallback: fa_pop_chunk_base returns the FIRST descriptor's dst
    //     as the chunk base and then reads kv_rows contiguous rows from it, which is
    //     silently the wrong memory once the blocks are in scattered slots.
    struct fa_iter_order iter_order;
    iter_order.n_q_blocks    = (neq1 + Br - 1) / Br;
    iter_order.n_kv_heads    = n_kv_heads;
    iter_order.kv_head_outer = false;
    {
        const uint32_t res_mode = kparams->u.hmx.res_mode;
        const bool     eligible =
            res_mode != HTP_FA_RES_OFF &&
            factx.sel != NULL && factx.sel_nb1 != 0 && factx.sparse_bs != 0 &&
            factx.pipeline &&
            iter_order.n_q_blocks > 1 &&
            factx.n_blk_total > 0 && factx.n_blk_total <= FA_RES_MAX_BLOCKS &&
            // A slot's rows are tiled from a shifted tile origin, which is only the same
            // thing as tiling the whole chunk when a block spans whole 32-row tiles.
            (factx.sparse_bs % HMX_FP16_TILE_N_ROWS) == 0;

        struct hmx_fa_res_region R;
        hmx_fa_res_region_build(&R, L.total_bytes, ctx->vtcm_size, eligible ? factx.n_blk_total : 0,
                                factx.sparse_bs, size_k_row_padded, size_v_row_padded);

        if (R.n_slots != 0 &&
            (res_mode != HTP_FA_RES_AUTO || fa_sel_repeats_blocks(&factx, iter_order.n_q_blocks))) {
            factx.k_res            = VTCM_LAYOUT_PTR(__fp16, base, R.off_k);
            factx.v_res            = VTCM_LAYOUT_PTR(__fp16, base, R.off_v);
            factx.k_res_slot       = R.k_slot_bytes;
            factx.v_res_slot       = R.v_slot_bytes;
            factx.res_force_miss   = (res_mode == HTP_FA_RES_MISS);
            // No epoch yet: the first fa_res_begin_epoch must clear.
            factx.res_epoch_ib3     = UINT32_MAX;
            factx.res_epoch_kv_head = UINT32_MAX;
            iter_order.kv_head_outer = true;
        }
    }

    // ======== DMA setup ========
    dma_queue * const dma = ctx->dma[0];

    const size_t n_row_tiles_g_br = g_br / HMX_FP16_TILE_N_ROWS;
    const size_t n_tiles_per_bc   = Bc / HMX_FP16_TILE_N_COLS;
    // Rows one staged block contributes to a chunk. Only meaningful with residency, where
    // the chunk's m blocks sit in separate slots; without it the chunk is one contiguous
    // run and the interleave is told nblk == 1.
    const uint32_t res_blk_rows   = factx.sparse_bs;

    const size_t qo_element_size = factx.is_q_fp32 ? sizeof(float) : sizeof(__fp16);

    const bool q_transposed                 = q->nb[1] < q->nb[2];
    const size_t q_src_stride               = q_transposed ? q->nb[2] : q->nb[1];
    const size_t q_row_bytes_untransposed   = factx.G * factx.DK * qo_element_size;
    const size_t q_row_bytes_trans_factor   = factx.DK * qo_element_size;
    const uint32_t kv_rows0                 = hex_smin(Bc, nek1);

    // ======== Reusable job descriptors for pipeline ========
    hmx_fa_qk_job_t       qk_job;
    hmx_fa_o_update_job_t ou_job;
    hmx_fa_o_norm_job_t   on_job;

    // ======== Main loop ========
    //
    // One linear index over (sequence, query block, KV head); fa_iter_at decides which
    // of the inner two varies fastest. The tail prefetch at the bottom uses the SAME
    // function on it + 1, so the successor can never drift out of step with the loop.
    const uint32_t n_iter = neq3 * iter_order.n_q_blocks * iter_order.n_kv_heads;
    for (uint32_t it = 0; it < n_iter; ++it) {
        {
            const struct fa_iter cur = fa_iter_at(&iter_order, it);
            const uint32_t ib3     = cur.ib3;
            const uint32_t q_start = cur.qb_idx * Br;
            const uint32_t kv_head = cur.kv_head;
            const uint32_t im3     = mask ? fastmodulo(ib3, mask->ne[3], &factx.src3_div3) : 0;

            const uint32_t n_rows_q    = hex_smin(Br, neq1 - q_start);
            const size_t   n_rows_g    = n_rows_q * G;
            const size_t   g_br_actual = hex_align_up(n_rows_g, HMX_FP16_TILE_N_ROWS);
            const size_t   n_row_tiles = g_br_actual / HMX_FP16_TILE_N_ROWS;

            // sel[] row for this query tile. The host constrains the chunk-size search
            // so Br divides the selection's query-block size, hence the whole tile sits
            // inside one query block and a single row serves all of its rows. Zero for
            // dense and for a shared (ne[1] == 1) selection.
            const uint32_t qb = fa_sel_row(&factx, q_start);

            // This tile's own KV work: its row's selection length and the chunk count it
            // decomposes into. Without a count tensor row_nsel == n_sel and row_chunks ==
            // n_kv_blocks, so the fixed-length path is bit-identical. Everything inside
            // this iteration -- loop bound, prefetch horizons, epilogue -- runs on
            // row_chunks; factx.n_kv_blocks is only the across-tiles upper bound.
            const uint32_t row_nsel   = factx.sel ? fa_row_nsel(&factx, qb, kv_head, ib3) : 0;
            const uint32_t row_chunks = factx.sel ? (row_nsel + factx.m - 1) / factx.m
                                                  : factx.n_kv_blocks;

            // Trace tag. Both inner axes are in it, because with the KV head outside the
            // query block loop a bare q_start recurs once per head and the phases become
            // impossible to attribute in a trace.
            const uint16_t iter_tag = (uint16_t) (kv_head * iter_order.n_q_blocks + cur.qb_idx);

            {
                const uint32_t ik2 = kv_head;
                const uint32_t ik3 = fastdiv(ib3, &kparams->broadcast_rk3);
                const uint32_t iv2 = kv_head;
                const uint32_t iv3 = fastdiv(ib3, &kparams->broadcast_rv3);

                // First KV block of this (sequence, q-block, kv-head) iteration.
                // Sparse selection can differ per query block, head and sequence, so
                // this is recomputed here rather than hoisted out of the loop nest.
                const uint32_t blk0_start = fa_kv_block_start(&factx, 0, qb, kv_head, ib3);
                const uint32_t blk0_rows  = fa_chunk_rows(&factx, 0, qb, kv_head, ib3, nek1);

                // 1. Push Q and KV DMAs for the very first iteration.
                // Subsequent iterations are enqueued early at the end of the previous iteration.
                if (it == 0) {
                    const uint8_t * q_ptr = (const uint8_t *) q->data;
                    const size_t q_row_bytes = q_transposed ? n_rows_q * q_row_bytes_trans_factor : q_row_bytes_untransposed;
                    const size_t n_rows      = q_transposed ? factx.G : n_rows_q;
                    dma_queue_push(dma, dma_make_ptr(factx.vtcm_q_dma, q_ptr), q_row_bytes, hex_smax(q_src_stride, q_row_bytes), q_row_bytes, n_rows);

                    if (factx.n_kv_blocks > 0) {
                        fa_res_begin_epoch(&factx, ib3, kv_head);
                        fa_push_chunk(dma, k, v, mask, 0, size_k_row_padded, size_k_row, size_v_row_padded, size_v_row,
                                      ik2, ik3, iv2, iv3, q_start, im3, kv_head, G, m_line_bytes, n_rows_q, nek1,
                                      0, ib3, /*push_mask=*/(factx.pipeline && mask), &factx);
                    }
                }

                // 2. Pop Q DMA (blocks until Q is loaded)
                dma_queue_pop(dma);

                // ---- Load Q block & Initialize per-block state ----
                fa_phase_q_load(&factx, q, q_start, kv_head, ib3, n_rows_g);

                __fp16 * o_tile_prev = factx.vtcm_o_tiles[0];
                __fp16 * o_tile_curr = factx.vtcm_o_tiles[1];

                // ---- KV block loop with DMA double-buffering ----
                size_t buf_idx = 0;

                htp_trace_event_start(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, iter_tag);
                fa_compute_slopes(&factx, kv_head, n_rows_g);
                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, iter_tag);

                const size_t k_src_stride = size_k_row_padded / sizeof(__fp16);
                const size_t v_src_stride = size_v_row_padded / sizeof(__fp16);

                hmx_queue_t hmx_q = ctx->hmx_queue;

                if (factx.pipeline) {
                    // Double-buffered job structs because HMX queue runs asynchronously
                    hmx_fa_qk_job_t qk_job[2];
                    hmx_fa_o_update_job_t ou_job[2];

                    // Prefetch block 1 early if there are multiple blocks
                    if (row_chunks > 1) {
                        fa_prefetch_block(dma, k, v, mask, 1, Bc, size_k_row_padded, size_k_row, size_v_row_padded, size_v_row,
                                          ik2, ik3, iv2, iv3, q_start, im3, kv_head, G, m_line_bytes, n_rows_q, nek1, 1, ib3, &factx);
                    }

                    // Prep and start QK-dot(0)
                    //
                    // The chunk's per-block VTCM addresses come from the descriptors,
                    // never from the double-buffer index: with residency they are
                    // scattered slots, and without it they are exactly
                    // vtcm_k_fp16[buf] + j*bs*stride, so bases[0] is the old base.
                    void * k_bases[FA_SPARSE_MAX_M];
                    const uint32_t nblk0 = fa_chunk_nblk(&factx, 0, row_nsel);
                    fa_pop_bases(dma, nblk0, k_bases);
                    fa_phase_k_interleave(&factx, blk0_rows, k_src_stride, k_bases,
                                          factx.k_res ? nblk0 : 1,
                                          factx.k_res ? res_blk_rows : blk0_rows,
                                          blk0_start, factx.vtcm_k_tiles[0]);

                    qk_job[0].q_tiles        = factx.vtcm_q_tiles;
                    qk_job[0].k_tiles        = factx.vtcm_k_tiles[0];
                    qk_job[0].s_tiles        = factx.vtcm_s_tiles[0];
                    qk_job[0].n_row_tiles    = n_row_tiles;
                    qk_job[0].n_col_tiles    = hmx_ceil_div(blk0_rows, HMX_FP16_TILE_N_COLS);
                    qk_job[0].n_dot_tiles    = DK / 32;
                    qk_job[0].n_tiles_per_bc = n_tiles_per_bc;
                    qk_job[0].hmx_scales     = factx.vtcm_hmx_scales_qk;
                    hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_qk_dot_worker, &qk_job[0]));

                    for (uint32_t kv_blk = 0; kv_blk < row_chunks; ++kv_blk) {
                        const uint32_t kv_start    = fa_kv_block_start(&factx, kv_blk, qb, kv_head, ib3);
                        const uint32_t kv_rows     = fa_chunk_rows(&factx, kv_blk, qb, kv_head, ib3, nek1);
                        const size_t   n_col_tiles = hmx_ceil_div(kv_rows, HMX_FP16_TILE_N_COLS);

                        // ---- 1. Pop and run V-prep for current block ----
                        void * v_bases[FA_SPARSE_MAX_M];
                        const uint32_t cur_nblk = fa_chunk_nblk(&factx, kv_blk, row_nsel);
                        fa_pop_bases(dma, cur_nblk, v_bases);
                        fa_phase_v_interleave(&factx, kv_rows, v_src_stride, v_bases,
                                              factx.v_res ? cur_nblk : 1,
                                              factx.v_res ? res_blk_rows : kv_rows,
                                              factx.vtcm_v_tiles[buf_idx], n_tiles_per_bc, kv_start);

                        // ---- 2. Pop and run mask-prep for current block ----
                        __fp16 * current_mask_vtcm = NULL;
                        if (mask) {
                            if (__builtin_expect(factx.mask_use_cache, true)) {
                                current_mask_vtcm = (__fp16 *) dma_queue_pop(dma).dst;
                            } else if (__builtin_expect(factx.mask_broadcast, true)) {
                                fa_pop_n(dma, fa_chunk_nblk(&factx, kv_blk, row_nsel));
                                current_mask_vtcm = factx.vtcm_mask_buf + buf_idx * factx.mask_slot_stride;
                            } else {
                                fa_pop_mask_dma_gqa(dma, G);
                                current_mask_vtcm = factx.vtcm_mask_buf + buf_idx * factx.mask_buf_gqa_stride;
                            }
                        }

                        // ---- 3. Start HMX O update for block kv_blk - 1 (reads P[1 - buf_idx], V[1 - buf_idx], D) ----
                        // O update relys on the previous block's P and V tiles.
                        // O update MUST be pushed before the next block's QK-dot: hmx_queue_pop() retires the
                        // oldest descriptor, so push order alone decides which pop waits for which job.
                        // If OU went in after QK(i+1), the pop below would retire QK(i+1) and leave
                        // OU(i-1) in flight into the next iteration, where V-prep overwrites V[prev_buf].
                        if (kv_blk > 0) {
                            const size_t prev_buf        = 1 - buf_idx;
                            ou_job[prev_buf].o_curr      = o_tile_curr;
                            ou_job[prev_buf].o_prev      = o_tile_prev;
                            ou_job[prev_buf].p_tiles     = factx.vtcm_p_tiles[prev_buf];
                            ou_job[prev_buf].v_tiles     = factx.vtcm_v_tiles[prev_buf];
                            ou_job[prev_buf].d_tiles     = factx.vtcm_d_tiles[prev_buf];
                            ou_job[prev_buf].hmx_scales  = factx.vtcm_hmx_scales_id;
                            ou_job[prev_buf].n_row_tiles = n_row_tiles;
                            ou_job[prev_buf].n_col_tiles = hmx_ceil_div(
                                fa_kv_block_rows(&factx, kv_blk - 1, qb, kv_head, ib3, nek1), HMX_FP16_TILE_N_COLS);
                            ou_job[prev_buf].n_row_tiles_g_br = n_row_tiles_g_br;
                            ou_job[prev_buf].n_tiles_per_bc   = n_tiles_per_bc;
                            ou_job[prev_buf].DV               = DV;
                            hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job[prev_buf]));
                        }

                        // ---- 4. Pop and run K-prep for next block & push next QK-dot ----
                        if (kv_blk + 1 < row_chunks) {
                            const uint32_t next_start = fa_kv_block_start(&factx, kv_blk + 1, qb, kv_head, ib3);
                            const uint32_t next_rows  = fa_chunk_rows(&factx, kv_blk + 1, qb, kv_head, ib3, nek1);
                            const size_t   next_buf   = 1 - buf_idx;

                            void * next_k_bases[FA_SPARSE_MAX_M];
                            const uint32_t next_nblk = fa_chunk_nblk(&factx, kv_blk + 1, row_nsel);
                            fa_pop_bases(dma, next_nblk, next_k_bases);
                            fa_phase_k_interleave(&factx, next_rows, k_src_stride, next_k_bases,
                                                  factx.k_res ? next_nblk : 1,
                                                  factx.k_res ? res_blk_rows : next_rows,
                                                  next_start, factx.vtcm_k_tiles[next_buf]);

                            qk_job[next_buf].q_tiles        = factx.vtcm_q_tiles;
                            qk_job[next_buf].k_tiles        = factx.vtcm_k_tiles[next_buf];
                            qk_job[next_buf].s_tiles        = factx.vtcm_s_tiles[next_buf];
                            qk_job[next_buf].n_row_tiles    = n_row_tiles;
                            qk_job[next_buf].n_col_tiles    = hmx_ceil_div(next_rows, HMX_FP16_TILE_N_COLS);
                            qk_job[next_buf].n_dot_tiles    = DK / 32;
                            qk_job[next_buf].n_tiles_per_bc = n_tiles_per_bc;
                            qk_job[next_buf].hmx_scales     = factx.vtcm_hmx_scales_qk;
                            hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_qk_dot_worker, &qk_job[next_buf]));
                        }

                        // ---- 5. Wait for current block's QK-dot to finish ----
                        hmx_queue_pop(hmx_q);

                        // ---- 6. Phase 2: softmax + build_D ----
                        fa_softmax_args_t sargs;
                        memset(&sargs, 0, sizeof(sargs));
                        sargs.factx                = &factx;
                        sargs.buf_idx              = buf_idx;
                        sargs.kv_rows              = kv_rows;
                        sargs.n_rows_g             = n_rows_g;
                        sargs.n_col_tiles          = n_col_tiles;
                        sargs.n_tiles_per_bc       = n_tiles_per_bc;
                        sargs.n_row_tiles          = n_row_tiles;
                        sargs.n_row_tiles_g_br     = n_row_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_head              = kv_head;
                        sargs.kv_start             = kv_start;
                        sargs.is_first_block       = (kv_blk == 0);
                        sargs.q_start              = q_start;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.mask                 = mask;
                        sargs.mask_vtcm            = current_mask_vtcm;
                        sargs.mask_vtcm_row_stride = factx.mask_buf_row_stride;
                        sargs.slopes               = factx.vtcm_slopes;

                        // Run Softmax on HVX (blocking call)
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_row_tiles, n_row_tiles_g_br);

                        // Wait for HMX O update for block kv_blk - 1 to finish
                        if (kv_blk > 0) {
                            hmx_queue_pop(hmx_q);
                            hex_swap_ptr((void **) &o_tile_curr, (void **) &o_tile_prev);
                        }

                        // Prefetch block kv_blk + 2
                        if (kv_blk + 2 < row_chunks) {
                            fa_prefetch_block(dma, k, v, mask, kv_blk + 2, Bc, size_k_row_padded, size_k_row, size_v_row_padded, size_v_row,
                                              ik2, ik3, iv2, iv3, q_start, im3, kv_head, G, m_line_bytes, n_rows_q, nek1, buf_idx, ib3, &factx);
                        }

                        buf_idx = 1 - buf_idx;
                    }

                    // Epilogue
                    if (row_chunks > 0) {
                        const uint32_t last_blk = row_chunks - 1;
                        const size_t last_cols  = hmx_ceil_div(fa_kv_block_rows(&factx, last_blk, qb, kv_head, ib3, nek1), HMX_FP16_TILE_N_COLS);
                        ou_job[0].o_curr           = o_tile_curr;
                        ou_job[0].o_prev           = o_tile_prev;
                        ou_job[0].p_tiles          = factx.vtcm_p_tiles[1 - buf_idx];
                        ou_job[0].v_tiles          = factx.vtcm_v_tiles[1 - buf_idx];
                        ou_job[0].d_tiles          = factx.vtcm_d_tiles[1 - buf_idx];
                        ou_job[0].hmx_scales       = factx.vtcm_hmx_scales_id;
                        ou_job[0].n_row_tiles      = n_row_tiles;
                        ou_job[0].n_col_tiles      = last_cols;
                        ou_job[0].n_row_tiles_g_br = n_row_tiles_g_br;
                        ou_job[0].n_tiles_per_bc   = n_tiles_per_bc;
                        ou_job[0].DV               = DV;
                        hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job[0]));

                        // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                        htp_trace_event_start(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, iter_tag);
                        fa_build_d_diag_inv_l(&factx, n_row_tiles, n_row_tiles_g_br);
                        htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, iter_tag);
                        hmx_queue_pop(hmx_q);

                        hex_swap_ptr((void **) &o_tile_curr, (void **) &o_tile_prev);
                    }

                } else {
                    // Fallback path
                    for (uint32_t kv_blk = 0; kv_blk < row_chunks; ++kv_blk) {
                        const uint32_t kv_start    = fa_kv_block_start(&factx, kv_blk, qb, kv_head, ib3);
                        const uint32_t kv_rows     = fa_chunk_rows(&factx, kv_blk, qb, kv_head, ib3, nek1);
                        const uint32_t cur_nblk    = fa_chunk_nblk(&factx, kv_blk, row_nsel);
                        const size_t   n_col_tiles = hmx_ceil_div(kv_rows, HMX_FP16_TILE_N_COLS);
                        const uint32_t chunk_bs    = factx.sel ? factx.sparse_bs : (uint32_t) Bc;

                        if (mask) {
                            if (__builtin_expect(factx.mask_use_cache, true)) {
                                const uint8_t * ms_src = (const uint8_t *) mask->data + q_start * mask->nb[1] + im3 * mask->nb[3] + kv_start * sizeof(__fp16);
                                dma_cache_push(dma, &factx.m_cache, ms_src, m_line_bytes, mask->nb[1], kv_rows * sizeof(__fp16), n_rows_q);
                            } else if (__builtin_expect(factx.mask_broadcast, true)) {
                                for (uint32_t j = 0; j < cur_nblk; ++j) {
                                    const uint32_t bstart = fa_chunk_block_start(&factx, kv_blk, j, qb, kv_head, ib3);
                                    const uint32_t brows  = fa_block_rows(&factx, kv_blk, j, qb, kv_head, ib3, nek1);
                                    const uint8_t * ms_src = (const uint8_t *) mask->data + q_start * mask->nb[1] + im3 * mask->nb[3] + bstart * sizeof(__fp16);
                                    uint8_t * ms_dst = (uint8_t *) factx.vtcm_mask_buf + (size_t) j * chunk_bs * sizeof(__fp16);
                                    dma_queue_push(dma, dma_make_ptr(ms_dst, ms_src), m_line_bytes, mask->nb[1], brows * sizeof(__fp16), n_rows_q);
                                }
                            } else {
                                fa_push_mask_dma_gqa(dma, mask, q_start, im3, kv_start, kv_head, G, m_line_bytes, kv_rows, n_rows_q, 0, &factx);
                            }
                        }

                        if (kv_blk + 1 < row_chunks) {
                            const size_t    prefetch_buf   = 1 - buf_idx;
                            const uint32_t  nxt_nblk       = fa_chunk_nblk(&factx, kv_blk + 1, row_nsel);
                            for (uint32_t j = 0; j < nxt_nblk; ++j) {
                                const uint32_t bstart = fa_chunk_block_start(&factx, kv_blk + 1, j, qb, kv_head, ib3);
                                const uint32_t brows  = fa_block_rows(&factx, kv_blk + 1, j, qb, kv_head, ib3, nek1);
                                const uint8_t * src = (const uint8_t *) k->data + bstart * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                                uint8_t * dst = (uint8_t *) factx.vtcm_k_fp16[prefetch_buf] + (size_t) j * chunk_bs * size_k_row_padded;
                                dma_queue_push(dma, dma_make_ptr(dst, src), size_k_row_padded, k->nb[1], size_k_row, brows);
                            }
                            for (uint32_t j = 0; j < nxt_nblk; ++j) {
                                const uint32_t bstart = fa_chunk_block_start(&factx, kv_blk + 1, j, qb, kv_head, ib3);
                                const uint32_t brows  = fa_block_rows(&factx, kv_blk + 1, j, qb, kv_head, ib3, nek1);
                                const uint8_t * src = (const uint8_t *) v->data + bstart * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                                uint8_t * dst = (uint8_t *) factx.vtcm_v_fp16[prefetch_buf] + (size_t) j * chunk_bs * size_v_row_padded;
                                dma_queue_push(dma, dma_make_ptr(dst, src), size_v_row_padded, v->nb[1], size_v_row, brows);
                            }
                        }

                        // Wait for current K DMA and interleave. The fallback path never
                        // runs with residency (fa_pop_chunk_base assumes the chunk's
                        // blocks are contiguous from the first descriptor's dst), so the
                        // chunk is always one contiguous run: nblk == 1.
                        void * curr_k = fa_pop_chunk_base(dma, cur_nblk);
                        fa_phase_k_interleave(&factx, kv_rows, k_src_stride, &curr_k, 1, kv_rows,
                                              kv_start, factx.vtcm_k_tiles[0]);

                        {
                            qk_job.q_tiles        = factx.vtcm_q_tiles;
                            qk_job.k_tiles        = factx.vtcm_k_tiles[0];
                            qk_job.s_tiles        = factx.vtcm_s_tiles[0];
                            qk_job.n_row_tiles    = n_row_tiles;
                            qk_job.n_col_tiles    = n_col_tiles;
                            qk_job.n_dot_tiles    = (size_t) (DK / 32);
                            qk_job.n_tiles_per_bc = n_tiles_per_bc;
                            qk_job.hmx_scales     = factx.vtcm_hmx_scales_qk;

                            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_qk_dot_worker, &qk_job));
                            hmx_queue_pop(ctx->hmx_queue);
                        }

                        // Wait for current V DMA and interleave
                        void * curr_v = fa_pop_chunk_base(dma, cur_nblk);
                        fa_phase_v_interleave(&factx, kv_rows, v_src_stride, &curr_v, 1, kv_rows,
                                              factx.vtcm_v_tiles[0], n_tiles_per_bc, kv_start);

                        // ---- Phase 3: softmax + build_D ----
                        __fp16 * current_mask_vtcm = NULL;
                        if (mask) {
                            if (__builtin_expect(factx.mask_broadcast, true)) {
                                current_mask_vtcm = (__fp16 *) fa_pop_chunk_base(dma, factx.mask_use_cache ? 1 : cur_nblk);
                            } else {
                                fa_pop_mask_dma_gqa(dma, G);
                                current_mask_vtcm = factx.vtcm_mask_buf;
                            }
                        }

                        fa_softmax_args_t sargs;
                        memset(&sargs, 0, sizeof(sargs));
                        sargs.factx                = &factx;
                        sargs.kv_rows              = kv_rows;
                        sargs.n_rows_g             = n_rows_g;
                        sargs.n_col_tiles          = n_col_tiles;
                        sargs.n_tiles_per_bc       = n_tiles_per_bc;
                        sargs.n_row_tiles          = n_row_tiles;
                        sargs.n_row_tiles_g_br     = n_row_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_head              = kv_head;
                        sargs.kv_start             = kv_start;
                        sargs.is_first_block       = (kv_blk == 0);
                        sargs.q_start              = q_start;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.mask                 = mask;
                        sargs.mask_vtcm            = current_mask_vtcm;
                        sargs.mask_vtcm_row_stride = factx.mask_buf_row_stride;
                        sargs.slopes               = factx.vtcm_slopes;
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_row_tiles, n_row_tiles_g_br);

                        {
                            ou_job.o_curr           = o_tile_curr;
                            ou_job.o_prev           = o_tile_prev;
                            ou_job.p_tiles          = factx.vtcm_p_tiles[0];
                            ou_job.v_tiles          = factx.vtcm_v_tiles[0];
                            ou_job.d_tiles          = factx.vtcm_d_tiles[0];
                            ou_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                            ou_job.n_row_tiles      = n_row_tiles;
                            ou_job.n_col_tiles      = n_col_tiles;
                            ou_job.n_row_tiles_g_br = n_row_tiles_g_br;
                            ou_job.n_tiles_per_bc   = n_tiles_per_bc;
                            ou_job.DV               = DV;

                            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job));
                            if (kv_blk + 1 == row_chunks) {
                                // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                                htp_trace_event_start(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, iter_tag);
                                fa_build_d_diag_inv_l(&factx, n_row_tiles, n_row_tiles_g_br);
                                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, iter_tag);
                            }
                            hmx_queue_pop(ctx->hmx_queue);

                            hex_swap_ptr((void **) &o_tile_curr, (void **) &o_tile_prev);
                        }

                        buf_idx = 1 - buf_idx;
                    }
                }

                // Enqueue DMAs for the next iteration early so they overlap with O-PROC.
                // The successor is the SAME decomposition the loop itself runs, one index
                // later -- never a hand-rolled copy of the nesting order.
                const struct fa_iter nxt = fa_iter_at(&iter_order, it + 1);
                const uint32_t next_kv_head = nxt.kv_head;
                const uint32_t next_q_start = nxt.qb_idx * Br;
                const uint32_t next_ib3     = nxt.ib3;
                const bool     has_next     = (next_ib3 < neq3);

                if (has_next) {
                    const uint32_t next_n_rows_q = hex_smin(Br, neq1 - next_q_start);
                    const uint8_t * next_q_ptr = (const uint8_t *) q->data + next_q_start * q->nb[1] + (next_kv_head * factx.G) * q->nb[2] + next_ib3 * q->nb[3];
                    const size_t next_q_row_bytes = q_transposed ? next_n_rows_q * q_row_bytes_trans_factor : q_row_bytes_untransposed;
                    const size_t next_n_rows      = q_transposed ? factx.G : next_n_rows_q;
                    dma_queue_push(dma, dma_make_ptr(factx.vtcm_q_dma, next_q_ptr), next_q_row_bytes, hex_smax(q_src_stride, next_q_row_bytes), next_q_row_bytes, next_n_rows);

                    if (factx.n_kv_blocks > 0) {
                        const uint32_t next_ik2 = next_kv_head;
                        const uint32_t next_iv2 = next_kv_head;
                        uint32_t next_ik3 = ik3;
                        uint32_t next_iv3 = iv3;
                        if (next_ib3 != ib3) {
                            next_ik3 = fastdiv(next_ib3, &kparams->broadcast_rk3);
                            next_iv3 = fastdiv(next_ib3, &kparams->broadcast_rv3);
                        }

                        // The next iteration may be a different query block, head or
                        // sequence, so its first chunk comes from that iteration's own
                        // list row. fa_push_chunk resolves all three from the next_*
                        // values passed below -- there is deliberately nothing to
                        // precompute here, because the consumer one iteration later
                        // re-derives the same row from its own q_start/kv_head/ib3.
                        uint32_t next_im3 = im3;
                        if (mask && next_ib3 != ib3) {
                            next_im3 = fastmodulo(next_ib3, mask->ne[3], &factx.src3_div3);
                        }
                        // Residency slots are keyed by block index alone, so they belong
                        // to one (sequence, KV head). Retire the epoch HERE, where the
                        // push crosses into the next head -- the consumer reaches that
                        // head one iteration later, which is one push too late.
                        fa_res_begin_epoch(&factx, next_ib3, next_kv_head);
                        fa_push_chunk(dma, k, v, mask, 0, size_k_row_padded, size_k_row, size_v_row_padded, size_v_row,
                                      next_ik2, next_ik3, next_iv2, next_iv3, next_q_start, next_im3,
                                      next_kv_head, G, m_line_bytes, next_n_rows_q, nek1,
                                      0, next_ib3, /*push_mask=*/(factx.pipeline && mask), &factx);
                    }
                }

                // ---- Final normalization ----
                {
                    on_job.o_curr           = o_tile_curr;
                    on_job.o_prev           = o_tile_prev;
                    on_job.d_tiles          = factx.vtcm_d_inv_l;
                    on_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                    on_job.n_row_tiles      = n_row_tiles;
                    on_job.n_row_tiles_g_br = n_row_tiles_g_br;
                    on_job.DV               = DV;
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_o_norm_worker, &on_job));
                    hmx_queue_pop(ctx->hmx_queue);
                }

                // ---- Store O block ----
                fa_phase_o_store(&factx, dst, o_tile_curr, q_start, kv_head, ib3, n_rows_g);
            }
        }
    }

    return HTP_STATUS_OK;
}

int op_flash_attn_ext(struct htp_ops_context * octx) {
    const struct htp_tensor * q    = octx->src[0];
    const struct htp_tensor * k    = octx->src[1];
    const struct htp_tensor * v    = octx->src[2];
    const struct htp_tensor * mask = octx->src[3];
    const struct htp_tensor * dst  = octx->dst;

    // Check support
    if ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) || k->type != HTP_TYPE_F16 || v->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_fa_kernel_params * kparams = (const struct htp_fa_kernel_params *) octx->kernel_params;

    if (kparams->kernel_type == HTP_FA_KERNEL_UNSUPPORTED) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (kparams->kernel_type == HTP_FA_KERNEL_HMX) {
        return hmx_flash_attn_ext(octx);
    }

    struct htp_fa_context factx;
    factx.octx = octx;

    factx.t_start = HAP_perf_get_qtimer_count();

    factx.src0_div21 = kparams->u.hvx.src0_div21;
    factx.src0_div1  = kparams->u.hvx.src0_div1;

    factx.broadcast_rk2 = kparams->broadcast_rk2;
    factx.broadcast_rk3 = kparams->broadcast_rk3;
    factx.broadcast_rv2 = kparams->broadcast_rv2;
    factx.broadcast_rv3 = kparams->broadcast_rv3;

    if (mask) {
        factx.src3_div2 = kparams->src3_div2;
        factx.src3_div3 = kparams->src3_div3;
    }

    factx.is_q_fp32 = (kparams->is_q_fp32 != 0);
    factx.size_q_row_padded = kparams->u.hvx.size_q_row_padded;
    factx.size_k_row_padded = kparams->u.hvx.size_k_row_padded;
    factx.size_v_row_padded = kparams->u.hvx.size_v_row_padded;

    size_t size_q_block = factx.size_q_row_padded * 1; // single row for now
    factx.size_k_block = factx.size_k_row_padded * FLASH_ATTN_BLOCK_SIZE;
    factx.size_v_block = factx.size_v_row_padded * FLASH_ATTN_BLOCK_SIZE;
    factx.size_m_block = hex_round_up(FLASH_ATTN_BLOCK_SIZE * sizeof(__fp16), 128);

    factx.n_blocks = kparams->n_kv_blocks;

    factx.scale = kparams->scale;
    factx.max_bias = kparams->max_bias;
    factx.logit_softcap = (__fp16) kparams->logit_softcap;

    factx.n_head_log2 = kparams->n_head_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    const uint32_t n_head = q->ne[2];
    if (n_head > 512) {
        return HTP_STATUS_NO_SUPPORT;
    }
    for (uint32_t h = 0; h < n_head; ++h) {
        factx.slopes[h] = (__fp16) ((kparams->max_bias > 0.0f) ? alibi_slope(h, factx.n_head_log2, factx.m0, factx.m1) : 1.0f);
    }

    // total rows in q
    factx.qrows = kparams->qrows;
    factx.qrows_per_thread = kparams->qrows_per_thread;

    size_t size_vkq_acc = hex_round_up(v->ne[0] * sizeof(float), 128); // VKQ32

    factx.size_q_block = size_q_block;
    factx.size_vkq_acc = size_vkq_acc;

    uint8_t * vtcm_cur = octx->ctx->vtcm_base;

    factx.spad_q = vtcm_seq_alloc(&vtcm_cur, size_q_block * octx->n_threads);
    factx.spad_k = vtcm_seq_alloc(&vtcm_cur, factx.size_k_block * 2 * octx->n_threads);
    factx.spad_v = vtcm_seq_alloc(&vtcm_cur, factx.size_v_block * 2 * octx->n_threads);
    factx.spad_m = vtcm_seq_alloc(&vtcm_cur, (mask ? factx.size_m_block * HVX_FA_DMA_CACHE_SIZE : 0) * octx->n_threads);
    factx.spad_a = vtcm_seq_alloc(&vtcm_cur, size_vkq_acc * octx->n_threads);

    if ((size_t) (vtcm_cur - octx->ctx->vtcm_base) > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        work_queue_run(octx->ctx->work_queue, flash_attn_ext_f16_thread, &factx, octx->n_threads);
    }

    return HTP_STATUS_OK;
}

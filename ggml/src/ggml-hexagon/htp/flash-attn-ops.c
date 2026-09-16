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

    // Split-KV decode path (flash_attn_ext_f16_dec_thread); zero when the row-per-thread
    // path runs. The unit of work is (sequence, KV head, KV range) and every query row of
    // that KV head -- G GQA heads x n_tokens -- consumes each DMA'd block from VTCM.
    uint32_t  dec_G;           // query heads per KV head
    uint32_t  dec_R;           // rows per unit = n_tokens * G
    uint32_t  dec_n_split;     // KV ranges per (sequence, KV head)
    uint32_t  dec_bps;         // blocks per split
    uint32_t  dec_n_units;     // n_seqs * n_kv_heads * n_split
    uint32_t  dec_n_mseg;      // mask segments per block: n_tokens (broadcast mask) or R
    size_t    dec_stride_part; // bytes per (row, split) partial: 128 (M, S) + size_vkq_acc
    uint8_t * dec_partials;    // shared VTCM: [rows_total][n_split] partials
    uint32_t  dec_b_base;      // first KV block this op computes (hetero: the GPU owns [0, dec_b_base))
    uint32_t  dec_nslots;      // staging slots per thread (2 today; cluster mode may deepen the ring)
    uint32_t  het_gpu_nsplit;  // hetero: GPU partials per row appended to the merge (0 = none)
    const uint8_t * het_parts; // hetero: GPU partials in DDR, [row][gpu_split] x dec_stride_part

    // Cluster-selected pages (htp-ops.h, HTP_FA_CLUSTER_*). When cl_on, a unit (KV head, split)
    // walks a per-head list: cl_sel_n[kvh] shadow pages (contiguous 16 KB per tensor, no mask)
    // followed by the dense positional blocks [cl_dense_b0, n_blocks) with the mask.
    bool            cl_on;
    bool            cl_inplace;        // pages are page_keys consecutive rows of the positional K/V (HTP_FA_CLUSTER_HDR_INPLACE)
    bool            cl_runs;           // whole-cluster runs (HTP_FA_CLUSTER_HDR_RUNS): lists are DMA blocks of selected runs
    struct htp_fa_cluster_blk * cl_sel_blk;   // VTCM: [nek2][cl_max_blk] blocks (runs mode)
    const uint8_t * cl_runs_tab;       // this layer's run table: [kvh][cl_n_pages_max(=n_runs_max)] x struct htp_fa_cluster_run
    uint32_t        cl_max_blk;        // blocks per head the list can hold
    uint32_t        cl_rows_max;       // shadow rows per head
    uint32_t        cl_row_bytes;      // D * 2
    uint32_t        cl_budget_rows;    // runs mode: rows per head charged to the budget
    uint8_t *       cl_sel_r;          // VTCM per thread: run table staging (runs mode)
    size_t          cl_sel_r_stride;
    uint32_t        cl_flags;
    const uint8_t * cl_k_pages;        // shadow K pages of this layer: [kvh][page] x page_bytes
    const uint8_t * cl_v_pages;
    size_t          cl_page_bytes;
    uint32_t        cl_page_keys;      // rows per shadow page: 16, 32 or 64 (dense-tail blocks stay 64)
    size_t          cl_head_stride;    // n_pages_max * page_bytes
    uint32_t        cl_n_pages_max;
    uint32_t        cl_n_cand;         // candidate pages (per head) this op may select from
    uint32_t        cl_dense_b0;       // first dense positional block
    uint32_t        cl_n_dense_blocks;
    uint16_t *      cl_sel_pages;      // VTCM: [nek2][cl_n_pages_max]
    uint32_t        cl_sel_n[HTP_FA_CLUSTER_MAX_HEADS];
    uint8_t         cl_head_order[HTP_FA_CLUSTER_MAX_HEADS];   // heads sorted by descending list length
    uint32_t        cl_next_unit;      // dynamic unit dispatch (atomic)
    // on-device selection (density > 0): score the query against the page descriptors, take the
    // top cl_budget pages per KV head (flash_attn_ext_f16_select_thread)
    uint32_t        cl_density;        // permille of the candidates; 0 = host-written lists
    uint32_t        cl_budget;         // pages per head selected on device
    const uint8_t * cl_cent;           // this layer's page descriptors: [kvh][n_pages_max] x cl_cent_bytes
    uint32_t        cl_cent_bytes;
    uint32_t        cl_sel_rows;       // candidate rows staged per head, rounded up to 32
    uint8_t *       cl_sel_c;          // VTCM per thread: cl_sel_rows x cl_cent_bytes (descriptor staging)
    uint8_t *       cl_sel_s;          // VTCM per thread: cl_sel_rows/32 score vectors
    size_t          cl_sel_c_stride, cl_sel_s_stride;

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

    // Hetero prefill fold (dev prototype, src[7] + HTP_FA_FOLD_MAGIC; see htp-ops.h).
    // het_parts NULL leaves every path below bit-identical to the non-folded kernel.
    const float *   het_m;       // [rows], natural-log running max (NULL = fold off)
    const float *   het_l;       // [rows]
    const uint8_t * het_acc;     // [rows][DV] f32
    size_t          het_acc_stride;
    // SPILL measurement mode: the HTP's own partial goes here and a separate pass merges.
    float *         het_hm;
    float *         het_hl;
    uint8_t *       het_hacc;
    bool            fold_spill;
    bool            fold_nomerge;   // SPILL timing aid: no merge pass
    uint32_t        het_neq1;
    float *         het_g;       // per-row weight for the other engine's accumulator, [g_br]

    // Live-producer handshake (HTP_FA_FOLD_F_LIVE). fold_ready NULL = the partial is already
    // resident when the op starts, which is the no-wait path examples/fa-fold-check runs.
    volatile uint32_t * fold_ready;    // ctl + slot * stride      (this op publishes its sequence here)
    volatile uint32_t * fold_done;     // ctl + slot * stride + 128 (producer echoes the sequence back)
    volatile uint32_t * fold_status;   // ctl + HTP_FA_HETERO_STATUS_OFF
    size_t          fold_m_bytes;      // the three C1 regions, for the pre-fold invalidate
    size_t          fold_l_bytes;
    size_t          fold_acc_bytes;
    uint32_t        fold_seq;
    uint32_t        fold_slot;
    uint32_t        fold_timeout_us;
    bool            fold_flush_kv;
    bool            fold_noflush;      // HTP_FA_FOLD_F_NOFLUSH: rely on the op-start dirty flush
    bool            fold_need_inval;   // this op's L2 may hold partial lines from an earlier fold: invalidate per stage
    bool            fold_waited;       // the op waits at most once, before its FIRST fold
    bool            fold_ok;

    // STAGED fold (HTP_FA_FOLD_F_STAGED): per-KV-head delivery, per-tile fold decision.
    bool             fold_staged;
    bool             fold_stage_qb;     // stages are query blocks (else KV heads)
    bool             fold_stage_qbh;    // stages are (query block, KV head) pairs: qb * n_kv_heads + kvh (QB order)
    uint32_t         fold_n_kv_heads;
    bool             fold_instore;      // HTP_FA_FOLD_F_INSTORE: merge in the store threads, no diag, no norm
    bool             fold_acc_f16;      // HTP_FA_FOLD_F_ACC_F16: the partial's accumulator rows are f16
    uint32_t         fold_probe;        // HTP_FA_FOLD_F_PROBE_* bits (dev timing only)
    bool             fold_pf_early;     // INSTORE: this tile's partial is valid at tile start, softmax threads prefetch it
    bool             fold_tile;         // THIS tile folds (it has exceptions); false = plain path
    const uint32_t * fold_exc;          // [n_kv_heads][fold_num_sb] x (exc_nbk ? exc_nbk : 1), nonzero = exception
    uint32_t         fold_num_sb;
    uint32_t         fold_exc_nbk;      // 0: one flag per (head, sub-block); else membership row length
    // Exact-mask mode (src[8] without a fold buffer): the FULL per-sub-block membership,
    // [n_kv_heads][num_sb] rows of xmask_nbk f32 (nonzero = the sub-block selected the block).
    // The softmax sets every (32-row group, KV block) the group's sub-block did not select to -inf,
    // so the large tile computes exactly the per-64-block policy.
    const float *    xmask;
    uint32_t         xmask_nbk;
    uint32_t         xmask_num_sb;
    uint64_t         fold_stage_seen;   // bit k: done word k confirmed and its rows invalidated
    uint32_t         fold_wait_us_acc;  // this op's blocked time over all stages
    uint32_t         fold_tiles_blocked;
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
    uint32_t                  neq2;
    size_t                    n_rows_g;
    size_t                    rows_per_t;
} fa_o_store_args_t;

static inline HVX_Vector fa_fold_weights_vec(struct hmx_fa_context * factx, size_t i, size_t n_rows_g, uint32_t q_start,
                                             uint32_t kv_head, uint32_t ib3, uint32_t neq2, uint32_t G, HVX_Vector v_l2,
                                             HVX_Vector v_ln2, HVX_Vector v_lo, float * m_o, float * l_o, HVX_Vector * w_other);
static void fa_fold_prefetch_slice(struct hmx_fa_context * factx, unsigned int ith, size_t n_rows_g, uint32_t q_start,
                                   uint32_t kv_head, uint32_t ib3);
static void fa_fold_prefetch_tile_piece(struct hmx_fa_context * factx, unsigned int ith, unsigned int nth, size_t n_rows_g,
                                        uint32_t q_start, uint32_t kv_head, uint32_t ib3);

static inline void fa_o_store_impl_f32(unsigned int n, unsigned int i, void * data, const bool fold, const bool spill, const bool instore) {
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

    // Fold: the diagonal already carried w_htp/S_total into the tile, so all that is left is the
    // other engine's term. het_g[r] is w_other/S_total, built alongside the diagonal.
    const uint8_t * het_acc    = fold ? factx->het_acc : NULL;
    const size_t    het_stride = fold ? factx->het_acc_stride : 0;
    const float *   het_g      = fold ? factx->het_g : NULL;
    const uint32_t  neq2       = args->neq2;

    // Spill: the tile handed in is the UNNORMALISED accumulator (the norm was skipped), and it
    // goes to the HTP-side partial with its (m, l) instead of to dst. vtcm_m_vec holds
    // (m + scale) * log2(e) -- see fa_fold_diag_vec -- so m is unwound to natural units here,
    // matching what the other engine writes, so the merge pass compares like with like.
    const float * m_vec_f = spill ? (const float *) factx->vtcm_m_vec : NULL;
    const float * l_vec_f = spill ? (const float *) factx->vtcm_l_vec : NULL;
    const float   m_bias  = spill ? (float) factx->scale : 0.0f;
    const float   k_ln2   = 0.6931471805599453f;
    // The raw accumulator is in COLUMN-major tile order (tile (r, c) at c * n_row_tiles_g_br + r,
    // see hmx_fa_o_update_worker); the norm pass is what re-lays it row-major for this walk.
    const bool    raw          = spill || instore;
    const size_t  o_col_stride = raw ? (size_t) (factx->g_br / HMX_FP16_TILE_N_ROWS) * HMX_FP16_TILE_N_ELMS : 0;

    // In-store fold: the merge weights are formed per 32-row group from the VTCM (m, l) and the
    // partial's (m, l), exactly as the diagonal builder would, but here in every store thread and
    // with the diagonal itself never built. The thread's rows are one token run per GQA head, so
    // ONE 2D l2fetch (width = run, height = G, stride = head stride) stages its slice of the
    // partial's accumulator while the first rows are de-tiled; the (m, l) lines are dcfetch hints,
    // which do not cancel it.
    float      wa[HMX_FP16_TILE_N_ROWS] __attribute__((aligned(128)));
    float      wb[HMX_FP16_TILE_N_ROWS] __attribute__((aligned(128)));
    float      m_o[HMX_FP16_TILE_N_ROWS] __attribute__((aligned(128)));
    float      l_o[HMX_FP16_TILE_N_ROWS] __attribute__((aligned(128)));
    size_t     grp_cur   = (size_t) -1;
    bool       grp_empty = false;   // no row of this 32-row group has a partial: skip its accumulator
    const bool acc_f16   = instore && factx->fold_acc_f16;
    HVX_Vector v_wh    = Q6_V_vzero();
    const HVX_Vector v_l2  = hvx_vec_splat_f32(EXP_LOG2E_F);
    const HVX_Vector v_ln2 = hvx_vec_splat_f32(0.6931471805599453f);
    const HVX_Vector v_lo  = hvx_vec_splat_f32(-80.0f);
    if (instore && !factx->fold_pf_early && !(factx->fold_probe & HTP_FA_FOLD_F_PROBE_NOPF)) {
        // late: the partial only became valid at this tile's fold site (first tile of a head in
        // STAGED / LIVE), so the fetch and the reads start together
        fa_fold_prefetch_slice(factx, i, n_rows_g, q_start, kv_head, ib3);
    }

    for (size_t r = start; r < end; ++r) {
        float * out = (float *) ((uint8_t *) dst->data + (kv_head * G + h_idx) * dst->nb[1] +
                                 (q_start + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);
        if (spill) {
            const size_t hrow = ((size_t) ib3 * neq2 + (kv_head * G + h_idx)) * factx->het_neq1 + (q_start + q_idx);
            out = (float *) (factx->het_hacc + hrow * factx->het_acc_stride);
            factx->het_hm[hrow] = (m_vec_f[r] - m_bias) * k_ln2;
            factx->het_hl[hrow] = l_vec_f[r];
        }

        size_t         r0            = r / HMX_FP16_TILE_N_ROWS;
        size_t         r1            = r % HMX_FP16_TILE_N_ROWS;
        const __fp16 * tile_row_base = o_tile_src + r0 * HMX_FP16_TILE_N_ROWS * DV;

        const uint8_t * acc_o = NULL;   // this row of the other engine's accumulator (f32, or f16 with ACC_F16)
        HVX_Vector      v_wo  = Q6_V_vzero();
        if (fold) {
            const size_t hrow = ((size_t) ib3 * neq2 + (kv_head * G + h_idx)) * factx->het_neq1 + (q_start + q_idx);
            acc_o = het_acc + hrow * het_stride;
            v_wo  = hvx_vec_splat_f32(het_g[r]);
        }
        if (instore) {
            if (r0 != grp_cur) {
                grp_cur = r0;
                if (factx->fold_probe & HTP_FA_FOLD_F_PROBE_NOWEIGHTS) {
                    hvx_vmem(wa) = hvx_vec_splat_f32(0.5f);
                    hvx_vmem(wb) = hvx_vec_splat_f32(0.5f);
                    grp_empty = false;
                } else {
                    HVX_Vector vb;
                    HVX_Vector va = fa_fold_weights_vec(factx, r0, n_rows_g, q_start, kv_head, ib3, neq2, (uint32_t) G,
                                                        v_l2, v_ln2, v_lo, m_o, l_o, &vb);
                    hvx_vmem(wa) = va;
                    hvx_vmem(wb) = vb;
                    // an empty partial has l == 0 and a zero accumulator: its term is exactly 0
                    bool empty = true;
                    for (size_t j = 0; j < HMX_FP16_TILE_N_ROWS && empty; ++j) empty = l_o[j] == 0.0f;
                    grp_empty = empty;
                }
            }
            const size_t hrow = ((size_t) ib3 * neq2 + (kv_head * G + h_idx)) * factx->het_neq1 + (q_start + q_idx);
            acc_o = factx->het_acc + hrow * factx->het_acc_stride;
            v_wh  = hvx_vec_splat_f32(wa[r1]);
            v_wo  = hvx_vec_splat_f32(wb[r1]);
        }
        const bool skip_acc = instore && (grp_empty || (factx->fold_probe & HTP_FA_FOLD_F_PROBE_NOACC));

        HVX_VectorPair vp_o16 = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
        for (uint32_t d = 0; d < DV / 32; ++d) {
            const HVX_Vector * in_tile = raw
                ? (const HVX_Vector *) (o_tile_src + d * o_col_stride + r0 * HMX_FP16_TILE_N_ELMS)
                : (const HVX_Vector *) (tile_row_base + d * HMX_FP16_TILE_N_ELMS);
            HVX_VectorPair     vp      = hvx_vec_f16_to_f32_shuff(in_tile[r1 / 2]);
            HVX_Vector         v_out   = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            if (instore || fold) {
                HVX_Vector v_acc = Q6_V_vzero();
                if (!skip_acc) {
                    if (acc_f16) {
                        // 64 f16 dims per vector: convert once, use the low half on even d, the high on odd
                        if ((d & 1) == 0) {
                            vp_o16 = hvx_vec_f16_to_f32(*(const HVX_UVector *) (acc_o + (size_t) d * 32 * sizeof(__fp16)));
                        }
                        v_acc = (d & 1) ? Q6_V_hi_W(vp_o16) : Q6_V_lo_W(vp_o16);
                    } else {
                        v_acc = *(const HVX_UVector *) (acc_o + (size_t) d * 32 * sizeof(float));
                    }
                }
                if (instore) {
                    v_out = skip_acc ? HVX_OP_MUL_F32(v_wh, v_out)
                                     : HVX_OP_ADD_F32(HVX_OP_MUL_F32(v_wh, v_out), HVX_OP_MUL_F32(v_wo, v_acc));
                } else {
                    v_out = HVX_OP_ADD_F32(v_out, HVX_OP_MUL_F32(v_wo, v_acc));
                }
            }
            *(HVX_UVector *) (out + d * 32) = v_out;
        }

        h_idx++;
        if (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));
}

static void fa_o_store_thread_f32(unsigned int n, unsigned int i, void * data) {
    fa_o_store_impl_f32(n, i, data, /*fold=*/false, /*spill=*/false, /*instore=*/false);
}

static void fa_o_store_thread_f32_fold(unsigned int n, unsigned int i, void * data) {
    fa_o_store_impl_f32(n, i, data, /*fold=*/true, /*spill=*/false, /*instore=*/false);
}

static void fa_o_store_thread_f32_spill(unsigned int n, unsigned int i, void * data) {
    fa_o_store_impl_f32(n, i, data, /*fold=*/false, /*spill=*/true, /*instore=*/false);
}

static void fa_o_store_thread_f32_instore(unsigned int n, unsigned int i, void * data) {
    fa_o_store_impl_f32(n, i, data, /*fold=*/false, /*spill=*/false, /*instore=*/true);
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
                             uint32_t                  neq2,
                             size_t                    n_rows_g) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    if (factx->n_threads > 1 && n_rows_g >= (size_t) (factx->n_threads * 2)) {
        n = factx->n_threads;
    }
    size_t rows_per_t = hmx_ceil_div(n_rows_g, n);
    fa_o_store_args_t args = { factx, dst, o_tile_src, q_start, kv_head, ib3, neq2, n_rows_g, rows_per_t };
    worker_callback_t store_fn = factx->is_dst_fp32
                                     ? (factx->fold_spill ? fa_o_store_thread_f32_spill
                                        : (factx->fold_tile ? (factx->fold_instore ? fa_o_store_thread_f32_instore : fa_o_store_thread_f32_fold)
                                                            : fa_o_store_thread_f32))
                                     : fa_o_store_thread_f16;
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
    uint32_t                  kv_blk;          // chunk index (exact-mask mode maps its blocks back to KV block ids)
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
    // INSTORE fold: stage this thread's slice of the partial into L2 a whole tile ahead of the
    // store that reads it. Issued once per tile, from every worker (one l2fetch engine per thread).
    if (args->is_first_block && factx->fold_pf_early) {
        fa_fold_prefetch_tile_piece(factx, i, n, args->n_rows_g, args->q_start, args->kv_head, args->ib3);
    }

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

    // exact-mask mode: this chunk's KV block per 64-key vector, once per call
    uint32_t xm_blk[16];
    uint32_t xm_nblk = 0;
    if (factx->xmask) {
        const uint32_t qb = fa_sel_row(factx, args->q_start);
        for (size_t c = 0; c < kv_rows && xm_nblk < 16; c += 64) {
            xm_blk[xm_nblk] = fa_chunk_block_idx(factx, args->kv_blk, xm_nblk, qb, args->kv_head, args->ib3);
            xm_nblk++;
        }
    }

    for (size_t r_vec_idx = vec_start; r_vec_idx < vec_end; ++r_vec_idx) {
        HVX_Vector rowmax_acc_v = v_neg_inf;
        HVX_Vector rowsum_acc_v = Q6_V_vzero();
        HVX_Vector m_prev_v0    = factx->vtcm_m_vec[r_vec_idx];

        // exact-mask mode: the group's 32 rows are 32/G tokens of one 64-token sub-block; one bit
        // per 64-key vector says the sub-block did not select that block
        uint32_t xm_bits = 0;
        if (factx->xmask) {
            const uint32_t tok = args->q_start + (uint32_t) fastdiv(r_vec_idx * 32, &factx->div_G);
            const uint32_t sb  = tok / HTP_FA_FOLD_EXC_SB;
            const float *  row = factx->xmask + ((size_t) args->kv_head * factx->xmask_num_sb + sb) * factx->xmask_nbk;
            for (uint32_t ci = 0; ci < xm_nblk; ++ci) {
                if (xm_blk[ci] >= factx->xmask_nbk || row[xm_blk[ci]] == 0.0f) {
                    xm_bits |= 1u << ci;
                }
            }
        }

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

            if (xm_bits) {
                // exact-mask mode: whole 64-key vectors of blocks this group's sub-block did not select
                for (uint32_t ci = 0; ci < xm_nblk; ++ci) {
                    if (xm_bits & (1u << ci)) {
                        my_row_buf0[ci] = v_neg_inf;
                        my_row_buf1[ci] = v_neg_inf;
                    }
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

// ---- Live-producer handshake for the prefill fold (HTP_FA_FOLD_F_LIVE) ----
//
// Same shape as the hetero DECODE handshake below: the DSP publishes an incrementing sequence in
// the slot's ready word, the producer echoes it into the done word once the whole partial is
// written. Everything the producer reads is flushed BEFORE ready goes up, everything it writes is
// invalidated AFTER done comes back.

// Invalidate one region another engine wrote. Rounded out to whole lines: the C1 regions are
// 128-aligned but their lengths are not, and the DSP never has dirty lines here (it only ever
// writes the control region, which is a separate 128-aligned region), so over-invalidating a tail
// line cannot lose a store.
static inline void fa_fold_inval_range(const void * p, size_t bytes) {
    if (!bytes) {
        return;
    }
    const uint32_t s = (uint32_t) (uintptr_t) p & ~(uint32_t) (HEX_L2_LINE_SIZE - 1);
    const uint32_t e = ((uint32_t) (uintptr_t) p + (uint32_t) bytes + HEX_L2_LINE_SIZE - 1) & ~(uint32_t) (HEX_L2_LINE_SIZE - 1);
    qurt_mem_cache_clean((qurt_addr_t) s, e - s, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
}

// Publish ready. Called BEFORE the KV loop: the producer's blocks do not depend on anything this
// op computes, so every microsecond it starts earlier comes straight off the wait below.
static void fa_fold_publish_ready(struct hmx_fa_context *   factx,
                                  const struct htp_tensor * q,
                                  const struct htp_tensor * k,
                                  const struct htp_tensor * v) {
    // The batch flushes its dirty ranges when it EXITS, which is long after the producer reads Q.
    // Flush here or the producer attends over whatever DDR still holds -- unless the header says
    // the op-start input flush (htp_tensor_flush_all) already covered every DSP-written input.
    if (!factx->fold_noflush) {
        hex_l2flush((void *) (uintptr_t) q->data, q->size);
        // The producer also reads the mask (a CPY on this DSP wrote its f16 copy) and, in the
        // prefill split, the exception membership (src[8], written by the selection ops before this).
        if (factx->octx->src[3] && factx->octx->src[3]->data) {
            hex_l2flush((void *) (uintptr_t) factx->octx->src[3]->data, factx->octx->src[3]->size);
        }
        if (factx->octx->src[8] && factx->octx->src[8]->data) {
            hex_l2flush((void *) (uintptr_t) factx->octx->src[8]->data, factx->octx->src[8]->size);
        }
        if (factx->fold_flush_kv) {
            hex_l2flush((void *) (uintptr_t) k->data, k->size);
            hex_l2flush((void *) (uintptr_t) v->data, v->size);
        }
    }

    volatile uint32_t * ready = factx->fold_ready;
    Q6_dcinva_A((void *) ready);
    factx->fold_seq = *ready + 1;
    *ready          = factx->fold_seq;
    qurt_mem_cache_clean((qurt_addr_t) ready, sizeof(uint32_t), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
}

// The op's ONE wait, taken just before the first fa_build_d_diag_fold. false = the producer never
// published; the caller must fail the op, because no correct partial exists and the fold would
// otherwise merge stale memory that looks like a plausible tensor.
static bool fa_fold_wait_done(struct hmx_fa_context * factx) {
    if (factx->fold_waited) {
        return factx->fold_ok;
    }
    factx->fold_waited = true;

    volatile uint32_t * done = factx->fold_done;
    const uint64_t t0        = HAP_perf_get_qtimer_count();
    const uint64_t timeout   = (uint64_t) factx->fold_timeout_us * 192ull / 10ull;   // 19.2 MHz qtimer
    uint32_t       seen      = 0;
    bool           ok        = true;
    for (;;) {
        Q6_dcinva_A((void *) done);
        seen = *done;
        if (seen == factx->fold_seq) {
            break;
        }
        if (HAP_perf_get_qtimer_count() - t0 > timeout) {
            ok = false;
            break;
        }
    }
    const uint32_t wait_us = (uint32_t) ((HAP_perf_get_qtimer_count() - t0) * 10ull / 192ull);

    // Feedback for the host: how long this op idled, every time -- a fold that is correct but
    // always 2 ms late is a performance bug the host can only see from here.
    {
        volatile uint32_t * wait_w = factx->fold_ready + 2;   // slot line + 8, the decode convention
        *wait_w = wait_us;
        qurt_mem_cache_clean((qurt_addr_t) wait_w, sizeof(uint32_t), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    }
    {
        volatile uint32_t * st = factx->fold_status;
        qurt_mem_cache_clean((qurt_addr_t) st, HTP_FA_FOLD_ST_N * sizeof(uint32_t), QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
        st[HTP_FA_FOLD_ST_TIMEOUTS]  += ok ? 0u : 1u;
        st[HTP_FA_FOLD_ST_WAITS]     += 1u;
        st[HTP_FA_FOLD_ST_DONE_SEEN]  = seen;
        st[HTP_FA_FOLD_ST_SEQ_WANTED] = factx->fold_seq;
        st[HTP_FA_FOLD_ST_SLOT]       = factx->fold_slot;
        st[HTP_FA_FOLD_ST_WAIT_US]    = wait_us;
        if (wait_us > st[HTP_FA_FOLD_ST_WAIT_US_MAX]) {
            st[HTP_FA_FOLD_ST_WAIT_US_MAX] = wait_us;
        }
        qurt_mem_cache_clean((qurt_addr_t) st, HTP_FA_FOLD_ST_N * sizeof(uint32_t), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    }

    if (!ok) {
        FARF(ERROR, "fa fold: slot %u seq %u: producer done not seen within %u us (saw %u)", factx->fold_slot,
             factx->fold_seq, factx->fold_timeout_us, seen);
        return false;
    }

    // The producer wrote all three C1 regions from another engine. Lines this DSP kept from an
    // EARLIER op on the same buffer (the previous layer folds through the same allocation) are
    // stale, and a stale line is a silently wrong softmax anchor.
    if (factx->fold_need_inval) {
        fa_fold_inval_range(factx->het_m, factx->fold_m_bytes);
        fa_fold_inval_range(factx->het_l, factx->fold_l_bytes);
        fa_fold_inval_range(factx->het_acc, factx->fold_acc_bytes);
    }

    factx->fold_ok = true;
    return true;
}

// Does the tile starting at q_start for this KV head have any exception sub-block? (STAGED)
static inline bool fa_fold_tile_has_exc(const struct hmx_fa_context * factx, uint32_t kv_head, uint32_t q_start, uint32_t neq1) {
    const uint32_t   q_end = hex_smin(q_start + factx->Br, neq1);
    const uint32_t   sb0   = q_start / HTP_FA_FOLD_EXC_SB;
    const uint32_t   sb1   = (q_end + HTP_FA_FOLD_EXC_SB - 1) / HTP_FA_FOLD_EXC_SB;
    const uint32_t   nbk   = factx->fold_exc_nbk ? factx->fold_exc_nbk : 1;   // flags, or membership rows
    const uint32_t * row   = factx->fold_exc + ((size_t) kv_head * factx->fold_num_sb + sb0) * nbk;
    for (uint32_t k = 0; k < (sb1 - sb0) * nbk; ++k) {
        if (row[k]) {   // nonzero bits: 1.0f as f32 or a uint32 flag, both nonzero
            return true;
        }
    }
    return false;
}

// STAGED: head kv_head is delivered -- record it and drop the lines this DSP may hold of its rows
// (rows of the G query heads of one KV head are contiguous).
// The stage a tile belongs to: its KV head, or its query block (HTP_FA_FOLD_F_STAGE_QB).
static inline uint32_t fa_fold_stage_of(const struct hmx_fa_context * factx, uint32_t q_start, uint32_t kv_head) {
    if (factx->fold_stage_qbh) {
        return (q_start / factx->Br) * factx->fold_n_kv_heads + kv_head;
    }
    return factx->fold_stage_qb ? q_start / factx->Br : kv_head;
}

static void fa_fold_stage_mark(struct hmx_fa_context * factx, uint32_t stage) {
    factx->fold_stage_seen |= 1ull << stage;
    if (!factx->fold_need_inval || (factx->fold_probe & HTP_FA_FOLD_F_PROBE_NOINVAL)) {
        return;   // no stale lines possible (see the epoch check at setup)
    }
    if (factx->fold_stage_qb) {
        // rows [qb*Br, +Br) of every query head (QB), or of the G heads of one KV head (QBH)
        const uint32_t neq2 = factx->octx->src[0]->ne[2];
        const uint32_t qb   = factx->fold_stage_qbh ? stage / factx->fold_n_kv_heads : stage;
        const uint32_t h0   = factx->fold_stage_qbh ? (stage % factx->fold_n_kv_heads) * factx->G : 0;
        const uint32_t h1   = factx->fold_stage_qbh ? h0 + factx->G : neq2;
        const size_t   tok0 = (size_t) qb * factx->Br;
        const size_t   nrow = hex_smin((size_t) factx->Br, (size_t) factx->het_neq1 - tok0);
        for (uint32_t h = h0; h < h1; ++h) {
            const size_t row0 = (size_t) h * factx->het_neq1 + tok0;
            fa_fold_inval_range(factx->het_m + row0, nrow * sizeof(float));
            fa_fold_inval_range(factx->het_l + row0, nrow * sizeof(float));
            fa_fold_inval_range(factx->het_acc + row0 * factx->het_acc_stride, nrow * factx->het_acc_stride);
        }
        return;
    }
    const size_t row0 = (size_t) stage * factx->G * factx->het_neq1;
    const size_t nrow = (size_t) factx->G * factx->het_neq1;
    fa_fold_inval_range(factx->het_m + row0, nrow * sizeof(float));
    fa_fold_inval_range(factx->het_l + row0, nrow * sizeof(float));
    fa_fold_inval_range(factx->het_acc + row0 * factx->het_acc_stride, nrow * factx->het_acc_stride);
}

// The per-tile wait. Single-wait mode defers to fa_fold_wait_done. STAGED waits for THIS tile's KV
// head the first time a tile of that head folds, then invalidates that head's rows of the partial
// (rows of the G query heads of one KV head are contiguous), and never blocks on the head again.
static bool fa_fold_wait_tile(struct hmx_fa_context * factx, uint32_t q_start, uint32_t kv_head) {
    if (!factx->fold_staged) {
        return fa_fold_wait_done(factx);
    }
    const uint32_t stage = fa_fold_stage_of(factx, q_start, kv_head);
    if (factx->fold_stage_seen & (1ull << stage)) {
        return true;
    }
    volatile uint32_t * done    = factx->fold_done + stage;
    const uint64_t      t0      = HAP_perf_get_qtimer_count();
    const uint64_t      timeout = (uint64_t) factx->fold_timeout_us * 192ull / 10ull;
    bool                spun    = false;
    for (;;) {
        Q6_dcinva_A((void *) done);
        const uint32_t seen = *done;
        if (seen == factx->fold_seq) {
            break;
        }
        spun = true;
        if (HAP_perf_get_qtimer_count() - t0 > timeout) {
            volatile uint32_t * st = factx->fold_status;
            qurt_mem_cache_clean((qurt_addr_t) st, HTP_FA_FOLD_ST_N * sizeof(uint32_t), QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
            st[HTP_FA_FOLD_ST_TIMEOUTS] += 1u;
            st[HTP_FA_FOLD_ST_DONE_SEEN]  = seen;
            st[HTP_FA_FOLD_ST_SEQ_WANTED] = factx->fold_seq;
            qurt_mem_cache_clean((qurt_addr_t) st, HTP_FA_FOLD_ST_N * sizeof(uint32_t), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
            FARF(ERROR, "fa fold: staged: stage %u seq %u not delivered within %u us (saw %u)", stage, factx->fold_seq,
                 factx->fold_timeout_us, seen);
            return false;
        }
    }
    factx->fold_wait_us_acc   += (uint32_t) ((HAP_perf_get_qtimer_count() - t0) * 10ull / 192ull);
    factx->fold_tiles_blocked += spun ? 1u : 0u;
    fa_fold_stage_mark(factx, stage);
    return true;
}

// STAGED, non-blocking: is head kv_head delivered already? Marks it (and invalidates its rows) if so.
static bool fa_fold_try_stage(struct hmx_fa_context * factx, uint32_t stage) {
    if (factx->fold_stage_seen & (1ull << stage)) {
        return true;
    }
    volatile uint32_t * done = factx->fold_done + stage;
    Q6_dcinva_A((void *) done);
    if (*done != factx->fold_seq) {
        return false;
    }
    fa_fold_stage_mark(factx, stage);
    return true;
}

// Can this tile's partial be read NOW (i.e. prefetched from the start of the tile)? Resident: always.
// Single-wait LIVE: once the op's one wait has passed. STAGED: once the head is delivered.
static bool fa_fold_partial_valid(struct hmx_fa_context * factx, uint32_t q_start, uint32_t kv_head) {
    if (!factx->fold_ready) {
        return true;
    }
    if (factx->fold_staged) {
        return fa_fold_try_stage(factx, fa_fold_stage_of(factx, q_start, kv_head));
    }
    return factx->fold_waited && factx->fold_ok;
}

// The l2fetch descriptor's stride is a 16-BIT field (PRM: Rtt[47:32]), so a fetch can never span
// GQA heads of the partial (head stride = neq1 * 512 B) or the two accumulators of the merge: every
// fetch here is LINEAR (hex_l2fetch_block, 16 KB rows), and one thread issues one fetch per call
// site, because a second l2fetch on the same thread replaces the first.
//
// Early form (from the first chunk's softmax workers, a whole tile ahead of the store): the tile's
// partial is G head runs of n_rows_q records; worker ith takes head (ith % G), piece (ith / G) of
// that run, so the n workers between them cover the tile with one linear fetch each. The (m, l)
// lines of the piece are dcfetch hints, which do not cancel it.
static void fa_fold_prefetch_tile_piece(struct hmx_fa_context * factx, unsigned int ith, unsigned int nth, size_t n_rows_g,
                                        uint32_t q_start, uint32_t kv_head, uint32_t ib3) {
    const uint32_t G        = factx->G;
    const uint32_t neq2     = factx->octx->src[0]->ne[2];
    const size_t   n_rows_q = (n_rows_g + G - 1) / G;
    const uint32_t h        = ith % G;
    const uint32_t pieces   = (nth + G - 1) / G;
    const uint32_t piece    = ith / G;
    const size_t   per      = (n_rows_q + pieces - 1) / pieces;
    const size_t   t0       = (size_t) piece * per;
    const size_t   t1       = hex_smin(t0 + per, n_rows_q);
    if (t0 >= t1) {
        return;
    }
    const size_t row0 = ((size_t) ib3 * neq2 + (kv_head * G + h)) * factx->het_neq1 + q_start + t0;
    hex_l2fetch_rows(factx->het_acc + row0 * factx->het_acc_stride, (uint32_t) factx->het_acc_stride, (uint32_t) (t1 - t0));
    for (size_t t = 0; t < t1 - t0; t += 32) {
        Q6_dcfetch_A((void *) (factx->het_m + row0 + t));
        Q6_dcfetch_A((void *) (factx->het_l + row0 + t));
    }
}

// Late form (from the store thread itself, when the partial only became valid at the fold site):
// the thread's own rows are one token run per head; head runs are fetched in turn, and on this
// thread only the last is guaranteed to survive, so this is the fallback and not the design.
static void fa_fold_prefetch_slice(struct hmx_fa_context * factx, unsigned int ith, size_t n_rows_g, uint32_t q_start,
                                   uint32_t kv_head, uint32_t ib3) {
    const uint32_t G       = factx->G;
    const uint32_t neq2    = factx->octx->src[0]->ne[2];
    const uint32_t n_store = (factx->n_threads > 1 && n_rows_g >= (size_t) factx->n_threads * 2) ? factx->n_threads : 1;
    if (ith >= n_store) {
        return;
    }
    const size_t rows_per_t = hmx_ceil_div(n_rows_g, n_store);
    const size_t start      = (size_t) ith * rows_per_t;
    const size_t end        = hex_smin(start + rows_per_t, n_rows_g);
    if (start >= end) {
        return;
    }
    const size_t tok0 = q_start + fastdiv(start, &factx->div_G);
    const size_t tok1 = q_start + fastdiv(end - 1, &factx->div_G) + 1;
    for (uint32_t h = 0; h < G; ++h) {
        const size_t hr = ((size_t) ib3 * neq2 + kv_head * G + h) * factx->het_neq1 + tok0;
        hex_l2fetch_rows(factx->het_acc + hr * factx->het_acc_stride, (uint32_t) factx->het_acc_stride, (uint32_t) (tok1 - tok0));
        for (size_t t = 0; t < tok1 - tok0; t += 32) {
            Q6_dcfetch_A((void *) (factx->het_m + hr + t));
            Q6_dcfetch_A((void *) (factx->het_l + hr + t));
        }
    }
}

// STAGED: the op's handshake feedback, written once at the end (the single-wait path writes it
// from inside its one wait).
static void fa_fold_staged_status(struct hmx_fa_context * factx) {
    volatile uint32_t * wait_w = factx->fold_ready + 2;
    *wait_w = factx->fold_wait_us_acc;
    qurt_mem_cache_clean((qurt_addr_t) wait_w, sizeof(uint32_t), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

    volatile uint32_t * st = factx->fold_status;
    qurt_mem_cache_clean((qurt_addr_t) st, HTP_FA_FOLD_ST_N * sizeof(uint32_t), QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    st[HTP_FA_FOLD_ST_WAITS]        += factx->fold_tiles_blocked ? 1u : 0u;
    st[HTP_FA_FOLD_ST_TILES_BLOCKED] += factx->fold_tiles_blocked;
    st[HTP_FA_FOLD_ST_SEQ_WANTED]    = factx->fold_seq;
    st[HTP_FA_FOLD_ST_SLOT]          = factx->fold_slot;
    st[HTP_FA_FOLD_ST_WAIT_US]       = factx->fold_wait_us_acc;
    st[HTP_FA_FOLD_ST_WAIT_US_SUM]  += factx->fold_wait_us_acc;
    if (factx->fold_wait_us_acc > st[HTP_FA_FOLD_ST_WAIT_US_MAX]) {
        st[HTP_FA_FOLD_ST_WAIT_US_MAX] = factx->fold_wait_us_acc;
    }
    qurt_mem_cache_clean((qurt_addr_t) st, HTP_FA_FOLD_ST_N * sizeof(uint32_t), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
}

// SPILL mode's second half: the explicit merge over DDR. Both partials are read from DDR --
// the HTP's own side included, which is exactly the stream the fold avoids by keeping its
// accumulator in VTCM. Structured to be memory-bound, so the number it produces is the
// design's floor and not this loop's overhead:
//  - weights are built 32 rows at a time with vector math (the four (m, l) arrays are
//    contiguous in row order, so they load as vectors) and parked in a small scratch, so
//    the per-row work is two scalar loads, two splats and the combine;
//  - both accumulators are prefetched two groups ahead with ONE 2D l2fetch (width = the
//    group, height 2, stride = the distance between the regions). One l2fetch per thread is
//    all the hardware keeps in flight, so two calls per group would cancel each other, and a
//    whole-range fetch (the previous version) thrashes: each thread's range is larger than L2.
#define FA_MERGE_GROUP 64
#define FA_MERGE_LEAD  3    // groups of accumulator prefetch kept in flight per thread
static void fa_fold_merge_thread(unsigned int nth, unsigned int ith, void * data) {
    struct hmx_fa_context * factx = (struct hmx_fa_context *) data;
    const struct htp_tensor * dst = factx->octx->dst;
    const struct htp_tensor * q   = factx->octx->src[0];
    const uint32_t neq1 = q->ne[1], neq2 = q->ne[2], neq3 = q->ne[3];
    const size_t   DV   = factx->DV;
    const size_t   rows = (size_t) neq1 * neq2 * neq3;
    const size_t   per  = hex_align_up((rows + nth - 1) / nth, FA_MERGE_GROUP);   // groups stay 128 B aligned
    const size_t   r0   = (size_t) ith * per;
    const size_t   r1   = hex_smin(r0 + per, rows);
    if (r0 >= r1) {
        return;
    }
    const size_t    stride = factx->het_acc_stride;
    const uint8_t * hacc   = factx->het_hacc;
    const uint8_t * oacc   = factx->het_acc;

    const float * hm = factx->het_hm;
    const float * hl = factx->het_hl;
    const float * om = factx->het_m;
    const float * ol = factx->het_l;

    const HVX_Vector v_lo   = hvx_vec_splat_f32(-80.0f);
    const HVX_Vector v_zero = Q6_V_vzero();

    float wa[FA_MERGE_GROUP] __attribute__((aligned(128)));
    float wb[FA_MERGE_GROUP] __attribute__((aligned(128)));

    // Lead-in: the other engine's rows for the first groups (the HTP's own rows follow inside the loop).
    // The l2fetch stride field is 16 bits, so the two regions (8 MB apart) need separate fetches.
    {
        const size_t n = hex_smin(FA_MERGE_LEAD * FA_MERGE_GROUP, r1 - r0);
        hex_l2fetch_rows(oacc + r0 * stride, (uint32_t) stride, (uint32_t) n);
    }
    // the (m, l) lines of the first group; later groups' lines are loaded one group early
    HVX_Vector vm_h[2] = { hvx_vmem(hm + r0), hvx_vmem(hm + r0 + 32) }, vl_h[2] = { hvx_vmem(hl + r0), hvx_vmem(hl + r0 + 32) };
    HVX_Vector vm_o[2] = { hvx_vmem(om + r0), hvx_vmem(om + r0 + 32) }, vl_o[2] = { hvx_vmem(ol + r0), hvx_vmem(ol + r0 + 32) };

    // row -> (iq1, iq2, iq3), advanced incrementally
    size_t iq3 = r0 / ((size_t) neq2 * neq1);
    size_t rem = r0 - iq3 * neq2 * neq1;
    size_t iq2 = rem / neq1;
    size_t iq1 = rem - iq2 * neq1;

    for (size_t g = r0; g < r1; g += FA_MERGE_GROUP) {
        const size_t n = hex_smin(FA_MERGE_GROUP, r1 - g);
        const size_t pf  = g + FA_MERGE_LEAD * FA_MERGE_GROUP;
        const size_t npf = pf < r1 ? hex_smin(FA_MERGE_GROUP, r1 - pf) : 0;
        // this group's HTP rows were fetched a lead ago from the same thread; refresh the lead on
        // the HTP side now and on the other engine's side after the weights, so the two fetches
        // do not replace each other on this thread's single l2fetch engine
        hex_l2fetch_rows(hacc + g * stride, (uint32_t) stride, (uint32_t) n);

        // both partials are natural-log anchored; weights via the vector exp, clamped like the fold
        for (int h = 0; h < FA_MERGE_GROUP / 32; ++h) {
            const HVX_Vector vM   = Q6_Vsf_vmax_VsfVsf(vm_h[h], vm_o[h]);
            const HVX_Vector vw_h = hvx_vec_exp_f32(Q6_Vsf_vmax_VsfVsf(HVX_OP_SUB_F32(vm_h[h], vM), v_lo));
            const HVX_Vector vw_o = hvx_vec_exp_f32(Q6_Vsf_vmax_VsfVsf(HVX_OP_SUB_F32(vm_o[h], vM), v_lo));
            const HVX_Vector vS   = HVX_OP_ADD_F32(HVX_OP_MUL_F32(vw_h, vl_h[h]), HVX_OP_MUL_F32(vw_o, vl_o[h]));
            const HVX_VectorPred q_pos = Q6_Q_vcmp_gt_VsfVsf(vS, v_zero);
            const HVX_Vector vinv = Q6_V_vmux_QVV(q_pos, hvx_vec_inverse_f32(vS), v_zero);   // S == 0: nothing to merge
            hvx_vmem(wa + 32 * h) = HVX_OP_MUL_F32(vw_h, vinv);
            hvx_vmem(wb + 32 * h) = HVX_OP_MUL_F32(vw_o, vinv);
        }

        if (npf) {
            hex_l2fetch_rows(oacc + pf * stride, (uint32_t) stride, (uint32_t) npf);
        }

        // next group's (m, l) lines: issued here so they land during this group's combine
        if (g + FA_MERGE_GROUP < r1) {
            for (int h = 0; h < FA_MERGE_GROUP / 32; ++h) {
                vm_h[h] = hvx_vmem(hm + g + FA_MERGE_GROUP + 32 * h);
                vl_h[h] = hvx_vmem(hl + g + FA_MERGE_GROUP + 32 * h);
                vm_o[h] = hvx_vmem(om + g + FA_MERGE_GROUP + 32 * h);
                vl_o[h] = hvx_vmem(ol + g + FA_MERGE_GROUP + 32 * h);
            }
        }

        for (size_t j = 0; j < n; ++j) {
            const size_t r = g + j;
            float * out = (float *) ((uint8_t *) dst->data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);
            const float * a = (const float *) (hacc + r * stride);
            const float * b = (const float *) (oacc + r * stride);
            const HVX_Vector v_a = hvx_vec_splat_f32(wa[j]);
            const HVX_Vector v_b = hvx_vec_splat_f32(wb[j]);
            for (size_t d = 0; d < DV; d += 32) {
                HVX_Vector v = HVX_OP_ADD_F32(HVX_OP_MUL_F32(v_a, hvx_vmem(a + d)), HVX_OP_MUL_F32(v_b, hvx_vmem(b + d)));
                *(HVX_UVector *) (out + d) = v;
            }
            if (++iq1 == neq1) {
                iq1 = 0;
                if (++iq2 == neq2) {
                    iq2 = 0;
                    ++iq3;
                }
            }
        }
    }
}

// Bail out of the HMX prefill op without leaving the DSP in a state later ops inherit. The KV
// pipeline runs one iteration ahead, so at it > 0 the ring already holds this iteration's Q and
// first chunk; a bare return hands those descriptors to whatever op runs next.
static int fa_fold_bail(dma_queue * dma, const char * why, uint32_t a, uint32_t b) {
    FARF(ERROR, "fa fold: %s (%u, %u)", why, a, b);
    dma_queue_flush(dma);
    return HTP_STATUS_INTERNAL_ERR;
}

// One 32-row group of the fold: gather the partial's (m, l) for these rows, combine with the
// HTP's own running (m, l), and leave w_other/S_total in het_g. Returns w_htp/S_total, the
// diagonal the HMX normalization multiplies O by.
//
// Rows of a tile are (token, GQA head) pairs with the head index moving fastest -- the same
// order fa_o_store_impl_f32 walks, so the two agree on which partial belongs to which row.
static inline HVX_Vector fa_fold_weights_vec(struct hmx_fa_context * factx,
                                          size_t                  i,
                                          size_t                  n_rows_g,
                                          uint32_t                q_start,
                                          uint32_t                kv_head,
                                          uint32_t                ib3,
                                          uint32_t                neq2,
                                          uint32_t                G,
                                          HVX_Vector              v_l2,
                                          HVX_Vector              v_ln2,
                                          HVX_Vector              v_lo,
                                          float *                 m_o,
                                          float *                 l_o,
                                          HVX_Vector *            w_other) {
    const size_t r0 = i * 32;
    for (size_t j = 0; j < 32; ++j) {
        const size_t r = r0 + j;
        if (r >= n_rows_g) {
            m_o[j] = HTP_FA_M_INITIAL_VAL;
            l_o[j] = 0.0f;
            continue;
        }
        const size_t q_idx = fastdiv(r, &factx->div_G);
        const size_t h_idx = fastmodulo(r, G, &factx->div_G);
        const size_t hrow  = ((size_t) ib3 * neq2 + (kv_head * G + h_idx)) * factx->het_neq1 + (q_start + q_idx);
        m_o[j] = factx->het_m[hrow];
        l_o[j] = factx->het_l[hrow];
    }

    const HVX_Vector vm_h = factx->vtcm_m_vec[i];                              // base-2 already
    const HVX_Vector vl_h = factx->vtcm_l_vec[i];
    // vtcm_m_vec does NOT hold m: it holds (m + scale) * log2(e). The extra scale term cancels in
    // the O/l normalization, so nothing in the non-folded kernel can see it -- but an external
    // partial must carry the same bias or the two anchors are compared on different footings.
    // Verified on device at DK=128 and DK=64, where the measured offset tracks scale * log2(e).
    const HVX_Vector v_bias = hvx_vec_splat_f32((float) factx->scale);
    const HVX_Vector vm_o = HVX_OP_ADD_F32(HVX_OP_MUL_F32(hvx_vmem(m_o), v_l2), v_bias);
    const HVX_Vector vl_o = hvx_vmem(l_o);

    const HVX_Vector vM  = Q6_Vsf_vmax_VsfVsf(vm_h, vm_o);
    const HVX_Vector dh  = Q6_Vsf_vmax_VsfVsf(HVX_OP_SUB_F32(vm_h, vM), v_lo);
    const HVX_Vector don = Q6_Vsf_vmax_VsfVsf(HVX_OP_SUB_F32(vm_o, vM), v_lo);
    const HVX_Vector w_h = hvx_vec_exp_f32(HVX_OP_MUL_F32(dh, v_ln2));
    const HVX_Vector w_o = hvx_vec_exp_f32(HVX_OP_MUL_F32(don, v_ln2));

    const HVX_Vector S    = HVX_OP_ADD_F32(HVX_OP_MUL_F32(w_h, vl_h), HVX_OP_MUL_F32(w_o, vl_o));
    const HVX_Vector invS = hvx_vec_inverse_f32(S);

    *w_other = HVX_OP_MUL_F32(w_o, invS);


    return HVX_OP_MUL_F32(w_h, invS);
}

// Diagonal-builder form: w_other/S_total goes to het_g[r] for the store thread, w_htp/S_total is returned.
static inline HVX_Vector fa_fold_diag_vec(struct hmx_fa_context * factx, size_t i, size_t n_rows_g, uint32_t q_start,
                                          uint32_t kv_head, uint32_t ib3, uint32_t neq2, uint32_t G, HVX_Vector v_l2,
                                          HVX_Vector v_ln2, HVX_Vector v_lo, float * m_o, float * l_o) {
    HVX_Vector w_other;
    const HVX_Vector w_h = fa_fold_weights_vec(factx, i, n_rows_g, q_start, kv_head, ib3, neq2, G, v_l2, v_ln2, v_lo, m_o, l_o, &w_other);
    hvx_vmem(factx->het_g + i * 32) = w_other;
    return w_h;
}

// Fold variant of fa_build_d_diag_inv_l. With a partial from another engine the correct final
// diagonal is w_htp/S_total, not 1/l -- so the rescale of the HTP's own accumulator costs nothing
// beyond a different diagonal, and only the other engine's term needs new work in the store.
//
// The two domains differ and the mismatch is SILENT: the HMX path folds log2(e) into scale
// (factx.scale *= EXP_LOG2E_F) so its m is base-2, while the partial's m is natural. Only m
// converts; l is a sum of exp(x - m) terms and has the same value in either base.
// The diagonal builder's prefetch, on its own: G runs of the tile's m, l and acc records into L2.
static void fa_fold_prefetch_tile(struct hmx_fa_context * factx, size_t n_rows_g, uint32_t q_start, uint32_t kv_head,
                                  uint32_t ib3, uint32_t neq2) {
    const uint32_t G        = factx->G;
    const size_t   n_rows_q = (n_rows_g + G - 1) / G;
    for (uint32_t h = 0; h < G; ++h) {
        const size_t row0 = ((size_t) ib3 * neq2 + (kv_head * G + h)) * factx->het_neq1 + q_start;
        hex_l2fetch_rows(factx->het_m + row0, (uint32_t) (n_rows_q * sizeof(float)), 1);
        hex_l2fetch_rows(factx->het_l + row0, (uint32_t) (n_rows_q * sizeof(float)), 1);
        hex_l2fetch_rows(factx->het_acc + row0 * factx->het_acc_stride, (uint32_t) factx->het_acc_stride, (uint32_t) n_rows_q);
    }
}

static __attribute__((noinline)) void fa_build_d_diag_fold(struct hmx_fa_context * factx,
                                                           size_t                  n_row_tiles,
                                                           size_t                  n_rows_g,
                                                           uint32_t                q_start,
                                                           uint32_t                kv_head,
                                                           uint32_t                ib3,
                                                           uint32_t                neq2) {
    // The fold reads its partials from DDR. Tile rows interleave the G GQA heads, but for ONE head
    // the records are contiguous in the row index, so the tile's partials are G runs of
    // n_rows_q records. Pull them into L2 before the gather touches them one scalar at a time --
    // without this the gather misses to DRAM on every row and costs more than the whole op.
    {
        const uint32_t G       = factx->G;
        const size_t   n_rows_q = (n_rows_g + G - 1) / G;
        for (uint32_t h = 0; h < G; ++h) {
            const size_t row0 = ((size_t) ib3 * neq2 + (kv_head * G + h)) * factx->het_neq1 + q_start;
            hex_l2fetch_rows(factx->het_m + row0, (uint32_t) (n_rows_q * sizeof(float)), 1);
            hex_l2fetch_rows(factx->het_l + row0, (uint32_t) (n_rows_q * sizeof(float)), 1);
            hex_l2fetch_rows(factx->het_acc + row0 * factx->het_acc_stride, (uint32_t) factx->het_acc_stride, (uint32_t) n_rows_q);
        }
    }

    const HVX_Vector     v_offsets = *(const HVX_Vector *) d_tile_scatter_offsets;
    const HVX_VectorPred q_32_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));

    const uint32_t G      = factx->G;
    const HVX_Vector v_l2 = hvx_vec_splat_f32(EXP_LOG2E_F);
    const HVX_Vector v_ln2  = hvx_vec_splat_f32(0.6931471805599453f);
    // exp2(x) = exp(x * ln2). Clamped so an empty partial (m = HTP_FA_M_INITIAL_VAL) cannot
    // reach the exp as -inf and return NaN, which would then survive the NaN * 0 acc term.
    const HVX_Vector v_lo = hvx_vec_splat_f32(-80.0f);

    HVX_Vector v_content = Q6_V_vzero();
    float m_o[32] __attribute__((aligned(128)));
    float l_o[32] __attribute__((aligned(128)));
    HVX_Vector d_hi = Q6_V_vzero();

    for (size_t i = 0; i < n_row_tiles; ++i) {
        if ((i % 2) == 0) {
            HVX_Vector d_lo = fa_fold_diag_vec(factx, i, n_rows_g, q_start, kv_head, ib3, neq2, G,
                                               v_l2, v_ln2, v_lo, m_o, l_o);
            d_hi = (i + 1 < n_row_tiles)
                       ? fa_fold_diag_vec(factx, i + 1, n_rows_g, q_start, kv_head, ib3, neq2, G,
                                          v_l2, v_ln2, v_lo, m_o, l_o)
                       : Q6_V_vzero();
            v_content = hvx_vec_f32_to_f16(d_lo, d_hi);
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
    uint32_t         n_q_blocks;
    uint32_t         n_kv_heads;
    bool             kv_head_outer;
    const uint16_t * perm;   // optional: it -> linear index in the base order (STAGED fold)
    uint32_t         n_iter; // perm's length; it >= n_iter is the loop's "one past the end" probe
};
#define FA_ITER_PERM_MAX 1024

struct fa_iter {
    uint32_t ib3;
    uint32_t qb_idx;
    uint32_t kv_head;
};

static inline struct fa_iter fa_iter_at(const struct fa_iter_order * o, uint32_t it) {
    // The tail prefetch asks for it + 1 on the LAST iteration too, and reads ib3 >= neq3 as "no
    // successor". Past the table that probe must stay unpermuted, or it indexes stack garbage and
    // can invent a next tile whose DMAs land on the tile still being stored.
    if (o->perm && it < o->n_iter) {
        it = o->perm[it];
    }
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
    // Per-row weight for the other engine's accumulator, built with the diagonal and consumed by
    // the store. Stack rather than VTCM so the fold cannot shift the tile-size search and
    // invalidate every measurement taken without it.
    float het_g_scratch[HTP_FA_FOLD_MAX_G_BR + 32];

    // Hetero prefill fold (dev prototype; see htp-ops.h). src[7] describes itself, so a buffer
    // whose magic does not match leaves every path below bit-identical to the non-folded kernel.
    // A buffer whose magic DOES match is a promise that another engine owns part of this softmax,
    // so from here on nothing may quietly drop it: a shape the prototype cannot serve fails the op
    // instead of falling back to an HTP-only answer that still looks like a plausible tensor.
    factx.het_m          = NULL;
    factx.het_l          = NULL;
    factx.het_acc        = NULL;
    factx.het_acc_stride = 0;
    factx.het_neq1       = 0;
    factx.het_g          = NULL;
    factx.het_hm         = NULL;
    factx.het_hl         = NULL;
    factx.het_hacc       = NULL;
    factx.fold_spill     = false;
    factx.fold_nomerge   = false;
    factx.fold_staged    = false;
    factx.fold_stage_qb  = false;
    factx.fold_stage_qbh = false;
    factx.fold_need_inval = true;
    factx.xmask          = NULL;
    factx.xmask_nbk      = 0;
    factx.xmask_num_sb   = 0;
    factx.fold_n_kv_heads = 0;
    factx.fold_instore   = false;
    factx.fold_probe     = 0;
    factx.fold_pf_early  = false;
    factx.fold_tile      = false;
    factx.fold_exc       = NULL;
    factx.fold_num_sb    = 0;
    factx.fold_exc_nbk   = 0;
    factx.fold_stage_seen   = 0;
    factx.fold_wait_us_acc  = 0;
    factx.fold_tiles_blocked = 0;
    if (octx->src[7] && octx->src[7]->data) {
        uint8_t * const fb = (uint8_t *) (uintptr_t) octx->src[7]->data;
        // The host writes this header from the CPU. A line left over from an earlier op on the
        // same buffer would hand us the previous graph's offsets.
        Q6_dcinva_A(fb);
        Q6_dcinva_A(fb + HEX_L2_LINE_SIZE);
        const struct htp_fa_fold_hdr * fh = (const struct htp_fa_fold_hdr *) fb;
        if (fh->magic == HTP_FA_FOLD_MAGIC) {
            const uint64_t rows       = (uint64_t) neq1 * neq2 * neq3;
            const bool     acc_f16    = (fh->flags & HTP_FA_FOLD_F_ACC_F16) != 0;
            const uint64_t acc_stride = (uint64_t) DV * (acc_f16 ? sizeof(__fp16) : sizeof(float));
            const uint64_t ml_bytes   = rows * sizeof(float);
            const uint64_t bufsz      = octx->src[7]->size;
            // Reject every configuration whose softmax is not in the domain the partial is written
            // in. With a softcap factx.scale keeps its NATURAL value (see the branch above), so the
            // fold's bias would be scale rather than scale*log2(e), and the HTP's logits are
            // tanh-capped while the producer's are not. ALiBi adds a per-head slope the producer
            // does not apply, and sinks are folded in by the HTP alone. Each of those is a
            // plausible-looking tensor, not an error, so they fail the op here.
            const bool bad_domain = kparams->logit_softcap != 0.0f || kparams->max_bias != 0.0f || octx->src[4];
            if (bad_domain || !factx.is_dst_fp32 || fh->neq1 != neq1 || fh->dv != DV || (uint64_t) fh->rows != rows ||
                factx.g_br > HTP_FA_FOLD_MAX_G_BR ||
                (fh->off_m % 128) != 0 || (fh->off_l % 128) != 0 || (fh->off_acc % 128) != 0 ||
                (uint64_t) fh->off_m + ml_bytes > bufsz ||
                (uint64_t) fh->off_l + ml_bytes > bufsz ||
                (uint64_t) fh->off_acc + rows * acc_stride > bufsz) {
                FARF(ERROR, "fa fold: unusable: softcap %d alibi %d sinks %d | rows %u want %u, neq1 %u want %u, dv %u want %u, g_br %u, dst_f32 %u, size %u",
                     kparams->logit_softcap != 0.0f, kparams->max_bias != 0.0f, octx->src[4] != NULL,
                     fh->rows, (unsigned) rows, fh->neq1, neq1, fh->dv, DV, factx.g_br,
                     (unsigned) factx.is_dst_fp32, octx->src[7]->size);
                return HTP_STATUS_INVAL_PARAMS;
            }
            factx.het_m          = (const float *) (fb + fh->off_m);
            factx.het_l          = (const float *) (fb + fh->off_l);
            factx.het_acc        = fb + fh->off_acc;
            factx.het_acc_stride = (size_t) acc_stride;
            factx.het_neq1       = fh->neq1;
            factx.het_g          = (float *) (((uintptr_t) het_g_scratch + 127) & ~(uintptr_t) 127);
            // Exactly the ranges the fold reads: m[rows], l[rows], acc[rows][dv].
            factx.fold_m_bytes   = (size_t) ml_bytes;
            factx.fold_l_bytes   = (size_t) ml_bytes;
            factx.fold_acc_bytes = (size_t) (rows * acc_stride);
            factx.fold_instore   = (fh->flags & HTP_FA_FOLD_F_INSTORE) != 0;
            factx.fold_acc_f16   = acc_f16;
            if (acc_f16 && (!factx.fold_instore || (fh->flags & HTP_FA_FOLD_F_SPILL))) {
                FARF(ERROR, "fa fold: ACC_F16 needs INSTORE and excludes SPILL (flags %u)", fh->flags);
                return HTP_STATUS_INVAL_PARAMS;
            }
            factx.fold_probe     = fh->flags & (HTP_FA_FOLD_F_PROBE_NOPF | HTP_FA_FOLD_F_PROBE_TILEPF |
                                                HTP_FA_FOLD_F_PROBE_NOWEIGHTS | HTP_FA_FOLD_F_PROBE_NOACC |
                                                HTP_FA_FOLD_F_PROBE_NOINVAL | HTP_FA_FOLD_F_PROBE_INVAL);

            // A live producer fills the partial while this op runs. Without the flag the partial
            // is already resident and the op behaves exactly as it did before the handshake.
            if (fh->flags & HTP_FA_FOLD_F_LIVE) {
                const uint64_t ctl_end = (uint64_t) fh->off_ctl + HTP_FA_HETERO_STATUS_OFF +
                                         HTP_FA_FOLD_ST_N * sizeof(uint32_t);
                // >= 256 keeps the control region clear of the two header lines invalidated above,
                // which would otherwise discard the ready word of slot 0.
                // The control region is 96 KB + change (HTP_FA_HETERO_STATUS_OFF), so "it fits in
                // the buffer" is not enough: overlapping m, l or acc would have the producer's
                // ready/done words land inside a partial the fold then reads as data.
                const uint64_t m_end   = (uint64_t) fh->off_m   + ml_bytes;
                const uint64_t l_end   = (uint64_t) fh->off_l   + ml_bytes;
                const uint64_t a_end   = (uint64_t) fh->off_acc + rows * acc_stride;
                const uint64_t c_beg   = fh->off_ctl;
                const bool overlaps = (c_beg < m_end && fh->off_m   < ctl_end) ||
                                      (c_beg < l_end && fh->off_l   < ctl_end) ||
                                      (c_beg < a_end && fh->off_acc < ctl_end);
                if (fh->slot >= HTP_FA_FOLD_MAX_SLOTS || fh->off_ctl < 256 || (fh->off_ctl % 128) != 0 ||
                    ctl_end > bufsz || overlaps) {
                    FARF(ERROR, "fa fold: live producer with a bad control region: slot %u off_ctl %u size %u overlaps %d",
                         fh->slot, fh->off_ctl, octx->src[7]->size, (int) overlaps);
                    return HTP_STATUS_INVAL_PARAMS;
                }
                uint8_t * const ctl   = fb + fh->off_ctl;
                factx.fold_ready      = (volatile uint32_t *) (ctl + (size_t) fh->slot * HTP_FA_HETERO_SLOT_STRIDE);
                factx.fold_done       = (volatile uint32_t *) (ctl + (size_t) fh->slot * HTP_FA_HETERO_SLOT_STRIDE + 128);
                if (fh->off_done) {
                    // a longer done line outside the slot: no overlap with the partial or the control region
                    const uint64_t d_beg = fh->off_done, d_end = d_beg + HTP_FA_FOLD_MAX_STAGES * sizeof(uint32_t);
                    const bool d_bad = (fh->off_done % 128) != 0 || d_beg < 256 || d_end > bufsz ||
                                       (d_beg < m_end && fh->off_m < d_end) || (d_beg < l_end && fh->off_l < d_end) ||
                                       (d_beg < a_end && fh->off_acc < d_end) || (d_beg < ctl_end && c_beg < d_end);
                    if (d_bad) {
                        FARF(ERROR, "fa fold: bad done line off_done %u", fh->off_done);
                        return HTP_STATUS_INVAL_PARAMS;
                    }
                    factx.fold_done = (volatile uint32_t *) (fb + fh->off_done);
                }
                factx.fold_status     = (volatile uint32_t *) (ctl + HTP_FA_HETERO_STATUS_OFF);
                factx.fold_slot       = fh->slot;
                factx.fold_timeout_us = fh->timeout_us ? fh->timeout_us : HTP_FA_FOLD_DONE_TIMEOUT_US;
                factx.fold_flush_kv   = (fh->flags & HTP_FA_FOLD_F_FLUSH_KV) != 0;
                factx.fold_noflush    = (fh->flags & HTP_FA_FOLD_F_NOFLUSH) != 0;
                // Stale partial lines can only be ones this DSP read in an earlier fold. If a whole-L2
                // flush-invalidate ran since (the op-start flush is one whenever > 4 MB of inputs are
                // dirty; Q alone is 8 MB at ub 1024) nothing stale is left and the per-stage range
                // invalidation (~8 MB of dcinva per op, ~115 us) is skipped; otherwise one whole-L2
                // invalidate here replaces it. (Doing the whole-L2 flush unconditionally cost ~18% of
                // pp4096 and fixed nothing: the producer's stale reads were on the GPU's side, see
                // the submission trigger in ggml-hexagon.cpp.)
                {
                    struct htp_context * hctx = octx->ctx;
                    if (hctx->l2_inval_epoch == hctx->fold_l2_epoch_seen) {
                        qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
                        hctx->l2_inval_epoch++;
                        FARF(HIGH, "fa fold: whole-L2 invalidate before ready (no flush since the last fold)");
                    }
                    hctx->fold_l2_epoch_seen = hctx->l2_inval_epoch;
                    factx.fold_need_inval = (factx.fold_probe & HTP_FA_FOLD_F_PROBE_INVAL) != 0;
                }
            }

            if (fh->flags & HTP_FA_FOLD_F_STAGED) {
                const uint32_t num_sb    = (neq1 + HTP_FA_FOLD_EXC_SB - 1) / HTP_FA_FOLD_EXC_SB;
                if (fh->off_exc == 0) {
                    // The membership is src[8] itself: F32 [NBk, R, NBq/R, n_kv_heads], one 0/1 row of
                    // NBk per (head, 64-token sub-block) in exactly the (head * num_sb + sb) order.
                    const struct htp_tensor * em = octx->src[8];
                    const bool bad = !em || !em->data || em->type != HTP_TYPE_F32 || em->nb[0] != sizeof(float) ||
                                     (uint64_t) em->ne[1] * em->ne[2] != num_sb || em->ne[3] != n_kv_heads ||
                                     !(fh->flags & HTP_FA_FOLD_F_LIVE) || neq3 != 1 || n_kv_heads > HTP_FA_FOLD_MAX_STAGES;
                    if (bad) {
                        FARF(ERROR, "fa fold: STAGED with src[8] membership unusable: em %p heads %u num_sb %u", em, n_kv_heads, num_sb);
                        return HTP_STATUS_INVAL_PARAMS;
                    }
                    factx.fold_staged  = true;
                    factx.fold_exc     = (const uint32_t *) (uintptr_t) em->data;   // this DSP wrote it: coherent
                    factx.fold_num_sb  = num_sb;
                    factx.fold_exc_nbk = em->ne[0];
                } else {
                const uint64_t exc_bytes = (uint64_t) n_kv_heads * num_sb * (fh->exc_nbk ? fh->exc_nbk : 1) * sizeof(uint32_t);
                const uint64_t exc_beg   = fh->off_exc, exc_end = exc_beg + exc_bytes;
                const uint64_t m_end  = (uint64_t) fh->off_m + ml_bytes, l_end = (uint64_t) fh->off_l + ml_bytes;
                const uint64_t a_end  = (uint64_t) fh->off_acc + rows * acc_stride;
                const uint64_t c_end  = (uint64_t) fh->off_ctl + HTP_FA_HETERO_STATUS_OFF + HTP_FA_FOLD_ST_N * sizeof(uint32_t);
                const bool overlaps = (exc_beg < m_end && fh->off_m < exc_end) || (exc_beg < l_end && fh->off_l < exc_end) ||
                                      (exc_beg < a_end && fh->off_acc < exc_end) || (exc_beg < c_end && fh->off_ctl < exc_end);
                if (!(fh->flags & HTP_FA_FOLD_F_LIVE) || neq3 != 1 || n_kv_heads > HTP_FA_FOLD_MAX_STAGES ||
                    (fh->off_exc % 128) != 0 || exc_beg < 256 || exc_end > bufsz || overlaps) {
                    FARF(ERROR, "fa fold: STAGED unusable: flags %u neq3 %u heads %u off_exc %u size %u overlaps %d",
                         fh->flags, neq3, n_kv_heads, fh->off_exc, octx->src[7]->size, (int) overlaps);
                    return HTP_STATUS_INVAL_PARAMS;
                }
                factx.fold_staged = true;
                factx.fold_exc    = (const uint32_t *) (fb + fh->off_exc);
                factx.fold_num_sb = num_sb;
                factx.fold_exc_nbk = fh->exc_nbk;
                fa_fold_inval_range(factx.fold_exc, (size_t) exc_bytes);   // host-written before the op
                }
            }

            if (factx.fold_staged) {
                factx.fold_stage_qbh   = (fh->flags & HTP_FA_FOLD_F_STAGE_QBH) != 0;
                factx.fold_stage_qb    = factx.fold_stage_qbh || (fh->flags & HTP_FA_FOLD_F_STAGE_QB) != 0;
                factx.fold_n_kv_heads  = n_kv_heads;
                const uint32_t n_qb     = (neq1 + factx.Br - 1) / factx.Br;
                const uint32_t n_stages = factx.fold_stage_qbh ? n_qb * n_kv_heads : factx.fold_stage_qb ? n_qb : n_kv_heads;
                const uint32_t max_st   = fh->off_done ? HTP_FA_FOLD_MAX_STAGES : HTP_FA_FOLD_LINE_STAGES;
                if (n_stages > max_st) {
                    FARF(ERROR, "fa fold: STAGED: %u stages exceed the done line (%u)", n_stages, max_st);
                    return HTP_STATUS_INVAL_PARAMS;
                }
            }

            if (fh->flags & HTP_FA_FOLD_F_SPILL) {
                // Same shape rules as the other engine's regions, plus disjointness from them: the
                // merge reads both sides, so an overlap would read a row of one as the other.
                const uint64_t hm_end = (uint64_t) fh->off_hm + ml_bytes, hl_end = (uint64_t) fh->off_hl + ml_bytes;
                const uint64_t ha_end = (uint64_t) fh->off_hacc + rows * acc_stride;
                const uint64_t m_end  = (uint64_t) fh->off_m + ml_bytes,  l_end = (uint64_t) fh->off_l + ml_bytes;
                const uint64_t a_end  = (uint64_t) fh->off_acc + rows * acc_stride;
                const bool bad = (fh->off_hm % 128) || (fh->off_hl % 128) || (fh->off_hacc % 128) ||
                                 hm_end > bufsz || hl_end > bufsz || ha_end > bufsz ||
                                 (fh->flags & (HTP_FA_FOLD_F_LIVE | HTP_FA_FOLD_F_INSTORE)) ||
                                 (fh->off_hm < a_end && fh->off_acc < hm_end) || (fh->off_hl < a_end && fh->off_acc < hl_end) ||
                                 (fh->off_hacc < a_end && fh->off_acc < ha_end) ||
                                 (fh->off_hacc < m_end && fh->off_m < ha_end) || (fh->off_hacc < l_end && fh->off_l < ha_end);
                if (bad) {
                    FARF(ERROR, "fa fold: SPILL regions unusable: hm %u hl %u hacc %u size %u", fh->off_hm, fh->off_hl, fh->off_hacc, octx->src[7]->size);
                    return HTP_STATUS_INVAL_PARAMS;
                }
                factx.het_hm     = (float *) (fb + fh->off_hm);
                factx.het_hl     = (float *) (fb + fh->off_hl);
                factx.het_hacc   = fb + fh->off_hacc;
                factx.fold_spill = true;
                factx.fold_nomerge = (fh->flags & HTP_FA_FOLD_F_NOMERGE) != 0;
            }

            // The partial came from another engine either way. With a live producer fa_fold_wait_done
            // invalidates after `done`; with flags = 0 nobody else will, and a line this DSP kept
            // from an earlier op through the same allocation is a silently wrong anchor. Must run
            // after the block above, which is where fold_ready and the three sizes are settled.
            if (!factx.fold_ready) {
                fa_fold_inval_range(factx.het_m, factx.fold_m_bytes);
                fa_fold_inval_range(factx.het_l, factx.fold_l_bytes);
                fa_fold_inval_range(factx.het_acc, factx.fold_acc_bytes);
            }
        }
    }

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

    // ======== Start the live producer ========
    // As early as the op can: it reads Q, not anything the KV loop below produces, so this is
    // pure overlap. The matching wait is at the first fold, at the bottom of the first tile.
    if (factx.fold_ready) {
        fa_fold_publish_ready(&factx, q, k, v);
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
    iter_order.perm          = NULL;
    iter_order.n_iter        = 0;
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

    // STAGED fold: exception-free tiles first (plain path, nothing to wait for), then the
    // exception tiles in the producer's delivery order, KV head ascending and query block
    // ascending within it. Indices are in the base (query-block-outer) order, so the
    // permutation is the only order-dependent thing here.
    // Exact-mask mode: a membership in src[8] with NO fold buffer. Same tensor shape as the
    // split's exception membership (F32 [NBk, R, NBq/R, n_kv_heads]) but carrying every selected
    // (sub-block, block) pair; the tile's list is the union and the softmax masks the rest.
    if (!factx.het_m && octx->src[8] && octx->src[8]->data) {
        const struct htp_tensor * em     = octx->src[8];
        const uint32_t            num_sb = (neq1 + HTP_FA_FOLD_EXC_SB - 1) / HTP_FA_FOLD_EXC_SB;
        const uint32_t            nkvh   = k->ne[2];
        if (!factx.sel || em->type != HTP_TYPE_F32 || em->nb[0] != sizeof(float) ||
            (uint64_t) em->ne[1] * em->ne[2] != num_sb || em->ne[3] != nkvh) {
            FARF(ERROR, "fa exact-mask: membership unusable (sel %d type %u ne %u %u %u %u, want num_sb %u heads %u)",
                 factx.sel != NULL, em->type, em->ne[0], em->ne[1], em->ne[2], em->ne[3], num_sb, nkvh);
            return HTP_STATUS_INVAL_PARAMS;
        }
        factx.xmask        = (const float *) (uintptr_t) em->data;   // this DSP wrote it: coherent
        factx.xmask_nbk    = em->ne[0];
        factx.xmask_num_sb = num_sb;
    }

    uint16_t iter_perm[FA_ITER_PERM_MAX];
    if (factx.fold_staged) {
        const uint32_t n_it = neq3 * iter_order.n_q_blocks * iter_order.n_kv_heads;
        if (n_it > FA_ITER_PERM_MAX) {
            FARF(ERROR, "fa fold: STAGED: %u tiles exceed the permutation table (%u)", n_it, FA_ITER_PERM_MAX);
            return HTP_STATUS_INVAL_PARAMS;
        }
        // Exception tiles follow the producer's delivery order: KV head outer when it stages by
        // head, query block outer (the default order, which keeps the mask cache's cross-head
        // reuse) when it stages by query block.
        uint32_t n = 0;
        for (int pass = 0; pass < 2; ++pass) {
            const uint32_t n_outer = factx.fold_stage_qb ? iter_order.n_q_blocks : iter_order.n_kv_heads;
            const uint32_t n_inner = factx.fold_stage_qb ? iter_order.n_kv_heads : iter_order.n_q_blocks;
            for (uint32_t o = 0; o < n_outer; ++o) {
                for (uint32_t i = 0; i < n_inner; ++i) {
                    const uint32_t qb = factx.fold_stage_qb ? o : i, kvh = factx.fold_stage_qb ? i : o;
                    if (fa_fold_tile_has_exc(&factx, kvh, qb * Br, neq1) == (pass == 1)) {
                        iter_perm[n++] = (uint16_t) (qb * iter_order.n_kv_heads + kvh);
                    }
                }
            }
        }
        iter_order.kv_head_outer = false;
        iter_order.perm          = iter_perm;
        iter_order.n_iter        = n_it;
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

            // Which path this tile takes. Staged: only tiles with exceptions fold (and wait);
            // the rest run the kernel exactly as without a partial.
            factx.fold_tile = factx.het_m && !factx.fold_spill &&
                              (!factx.fold_staged || fa_fold_tile_has_exc(&factx, kv_head, q_start, neq1));
            // In-store fold: if the partial is already readable, the softmax threads of the first
            // chunk prefetch it a whole tile ahead of the store; otherwise the store fetches late.
            factx.fold_pf_early = factx.fold_tile && factx.fold_instore && !(factx.fold_probe & HTP_FA_FOLD_F_PROBE_NOPF) &&
                                  fa_fold_partial_valid(&factx, q_start, kv_head);

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

            // The fold builds its diagonal and its het_g weights in the KV-loop epilogue. With no
            // chunks there is no epilogue, and the store would de-tile against an undefined
            // het_g -- a wrong tile with no error anywhere.
            if (factx.het_m && row_chunks == 0) {
                return fa_fold_bail(dma, "tile has no KV chunks", q_start, kv_head);
            }

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
                    // The first tile is (0, 0, 0) only in the base order; a permuted order (STAGED
                    // fold) may start anywhere, so address Q the way the tail prefetch does.
                    const uint8_t * q_ptr = (const uint8_t *) q->data + q_start * q->nb[1] + (kv_head * factx.G) * q->nb[2] + ib3 * q->nb[3];
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
                        sargs.kv_blk               = kv_blk;
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
                        if (factx.fold_spill) {
                            // spill: the tile leaves unnormalised, the merge pass normalises
                        } else if (factx.fold_tile) {
                            // The latest point the fold allows: everything above this line ran
                            // while the producer worked.
                            if (factx.fold_ready && !fa_fold_wait_tile(&factx, q_start, kv_head)) {
                                // No partial exists. Retire the O update so the HMX queue is not
                                // left with a job in flight, then fail: dst is still untouched,
                                // because this wait is taken before the op's first store.
                                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, iter_tag);
                                hmx_queue_pop(hmx_q);
                                return fa_fold_bail(dma, "producer late", q_start, kv_head);
                            }
                            if (!factx.fold_instore) {   // INSTORE: the store threads merge, no diagonal
                                fa_build_d_diag_fold(&factx, n_row_tiles, n_rows_g, q_start, kv_head, ib3, neq2);
                            } else if (factx.fold_probe & HTP_FA_FOLD_F_PROBE_TILEPF) {
                                fa_fold_prefetch_tile(&factx, n_rows_g, q_start, kv_head, ib3, neq2);
                            }
                        } else {
                            fa_build_d_diag_inv_l(&factx, n_row_tiles, n_row_tiles_g_br);
                        }
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
                        sargs.kv_blk               = kv_blk;
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
                                if (factx.fold_spill) {
                                    // spill: the tile leaves unnormalised, the merge pass normalises
                                } else if (factx.fold_tile) {
                                    if (factx.fold_ready && !fa_fold_wait_tile(&factx, q_start, kv_head)) {
                                        // See the pipelined path: drain the HMX queue, then fail
                                        // rather than fold memory no producer wrote.
                                        htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, iter_tag);
                                        hmx_queue_pop(ctx->hmx_queue);
                                        return fa_fold_bail(dma, "producer late", q_start, kv_head);
                                    }
                                    if (!factx.fold_instore) {   // INSTORE: the store threads merge, no diagonal
                                        fa_build_d_diag_fold(&factx, n_row_tiles, n_rows_g, q_start, kv_head, ib3, neq2);
                                    } else if (factx.fold_probe & HTP_FA_FOLD_F_PROBE_TILEPF) {
                                        fa_fold_prefetch_tile(&factx, n_rows_g, q_start, kv_head, ib3, neq2);
                                    }
                                } else {
                                    fa_build_d_diag_inv_l(&factx, n_row_tiles, n_row_tiles_g_br);
                                }
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
                // Spill mode keeps the accumulator unnormalised: the merge pass divides by the
                // combined S, so the diag(1/l) multiply would be wrong here, not merely wasted.
                const bool raw_store = factx.fold_spill || (factx.fold_instore && factx.fold_tile);
                if (!raw_store) {
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

                // ---- Store O block ---- (spill: the unnormalised accumulator is in o_tile_prev)
                fa_phase_o_store(&factx, dst, raw_store ? o_tile_prev : o_tile_curr, q_start, kv_head, ib3, neq2, n_rows_g);
            }
        }
    }

    if (factx.fold_spill && !factx.fold_nomerge) {
        // The explicit merge, as a separate pass over both partials in DDR.
        work_queue_run(ctx->work_queue, fa_fold_merge_thread, &factx, factx.n_threads);
    }
    if (factx.fold_staged) {
        fa_fold_staged_status(&factx);
    }

    return HTP_STATUS_OK;
}


// ============================================================================
// HVX decode path: split-KV, GQA-grouped
// ============================================================================
//
// The row-per-thread kernel above fetches a KV head once per QUERY row. With GQA both
// query heads of a group therefore stream the same K/V (2x the bytes), and 16 rows over
// 6 threads leaves one thread a single row. Measured on Qwen3-1.7B at kv=4096 (per-thread
// trace): threads are ~65% busy on HVX compute and finish at 100/100/100/100/100/36% of
// the span. Here the unit of work is (sequence, KV head, KV range): the range's K/V
// blocks are DMA'd once and every query row of that KV head -- all G GQA heads x all
// n_tokens -- consumes them from VTCM. n_split ranges per KV head give the pool enough
// units to balance. Each unit leaves an UNNORMALISED partial (M, S, acc[DV]) per row in
// shared VTCM; flash_attn_ext_f16_merge_thread combines the splits with the standard
// w_s = exp(M_s - M) weighting, applies sinks, normalises and stores -- the same
// flash-decoding structure as a split-K GPU kernel, on the NPU, with no crossings.
//
// Everything numeric (dot, softcap, mask, online softmax, P*V) is the SAME HVX sequence
// as the row-per-thread kernel, run once per row against a block staged once.

#define HVX_FA_DEC_PART_HDR 128     // bytes: M, S, then acc[DV] f32 follows
#define HVX_FA_DEC_R_MAX    64      // rows per unit (n_tokens * G); larger falls back

static inline uint8_t * hvx_fa_dec_partial(const struct htp_fa_context * factx, uint32_t row_global, uint32_t sp) {
    return factx->dec_partials + ((size_t) row_global * factx->dec_n_split + sp) * factx->dec_stride_part;
}

// A unit's work is an ordered list of 64-key blocks. Dense: ordinal j IS the KV block index.
// Cluster mode (factx->cl_on): the first cl_sel_n[kvh] ordinals are shadow pages -- rows
// contiguous at row_size stride, no mask -- and the rest are the dense positional blocks
// starting at cl_dense_b0, addressed and masked exactly as the dense kernel does.
struct hvx_fa_dec_blk {
    const uint8_t * k;
    const uint8_t * v;
    uint32_t        bsz;      // rows in this block
    uint32_t        pos;      // absolute KV position of row 0 (mask column); UINT32_MAX = no mask
    bool            contig;   // rows contiguous at row_size stride (shadow page) vs nb[1] stride
};

static inline void hvx_fa_dec_blk_src(const struct htp_fa_context * factx, const struct htp_tensor * k, const struct htp_tensor * v,
                                      uint32_t kvh, uint32_t ik2, uint32_t ik3, uint32_t iv2, uint32_t iv3, uint32_t j,
                                      struct hvx_fa_dec_blk * b) {
    uint32_t ib = j;
    if (factx->cl_on) {
        const uint32_t nsel = factx->cl_sel_n[kvh];
        if (j < nsel) {
            if (factx->cl_runs) {
                // a DMA block of a selected cluster: contiguous rows of the head's shadow, no mask
                const struct htp_fa_cluster_blk * e = factx->cl_sel_blk + (size_t) kvh * factx->cl_max_blk + j;
                uint32_t row = e->row, bsz = e->bsz;
                if (bsz == 0 || bsz > FLASH_ATTN_BLOCK_SIZE || row + bsz > factx->cl_rows_max) {
                    row = 0; bsz = 1;   // never a wild address
                }
                const size_t off = (size_t) kvh * factx->cl_head_stride + (size_t) row * factx->cl_row_bytes;
                b->k      = factx->cl_k_pages + off;
                b->v      = factx->cl_v_pages + off;
                b->bsz    = bsz;
                b->pos    = UINT32_MAX;
                b->contig = true;
                return;
            }
            uint32_t page = factx->cl_sel_pages[(size_t) kvh * factx->cl_n_pages_max + j];
            if (page >= factx->cl_n_cand) {
                page = factx->cl_n_cand - 1;   // clamp: a bad index must never form a wild address
            }
            if (factx->cl_inplace) {
                // the page is rows [page*PK, page*PK + PK) of the positional cache: the dense block fetch
                // (nb[1] stride) without the mask -- every candidate row lies below dense_start <= nek1
                const uint32_t ic = page * factx->cl_page_keys;
                b->k      = (const uint8_t *) k->data + ((size_t) ic * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3]);
                b->v      = (const uint8_t *) v->data + ((size_t) ic * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3]);
                b->bsz    = factx->cl_page_keys;
                b->pos    = UINT32_MAX;
                b->contig = false;
                return;
            }
            const size_t   off  = (size_t) kvh * factx->cl_head_stride + (size_t) page * factx->cl_page_bytes;
            b->k      = factx->cl_k_pages + off;
            b->v      = factx->cl_v_pages + off;
            b->bsz    = factx->cl_page_keys;
            b->pos    = UINT32_MAX;
            b->contig = true;
            return;
        }
        ib = factx->cl_dense_b0 + (j - nsel);
    }
    const uint32_t nek1     = k->ne[1];
    const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
    b->k      = (const uint8_t *) k->data + (ic_start * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3]);
    b->v      = (const uint8_t *) v->data + (ic_start * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3]);
    b->bsz    = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);
    b->pos    = ic_start;
    b->contig = false;
}

// Stage one block into (k_dst, v_dst) and, for masked blocks, its mask columns through the mask
// cache. Pushes are grouped K, V, then mask -- the pop order in the consumer. Returns whether a
// mask pop is pending for this block.
static inline bool hvx_fa_dec_stage(dma_queue * dma, dma_cache * mc, const struct htp_fa_context * factx,
                                    const struct hvx_fa_dec_blk * b, uint8_t * k_dst, uint8_t * v_dst,
                                    size_t size_k_row, size_t size_v_row, uint32_t nbk1, uint32_t nbv1,
                                    const __fp16 * const * mp_base, uint32_t n_mseg, bool mask) {
    if (b->contig && (factx->cl_flags & HTP_FA_CLUSTER_FLAG_DESC1D) &&
        factx->size_k_row_padded == size_k_row && factx->size_v_row_padded == size_v_row) {
        dma_queue_push_single_1d(dma, dma_make_ptr(k_dst, b->k), (size_t) b->bsz * size_k_row);
        dma_queue_push_single_1d(dma, dma_make_ptr(v_dst, b->v), (size_t) b->bsz * size_v_row);
    } else {
        dma_queue_push(dma, dma_make_ptr(k_dst, b->k), factx->size_k_row_padded, b->contig ? size_k_row : nbk1, size_k_row, b->bsz);
        dma_queue_push(dma, dma_make_ptr(v_dst, b->v), factx->size_v_row_padded, b->contig ? size_v_row : nbv1, size_v_row, b->bsz);
    }
    const bool has_mask = mask && b->pos != UINT32_MAX;
    if (has_mask) {
        for (uint32_t m = 0; m < n_mseg; ++m) {
            dma_cache_push(dma, mc, (const uint8_t *) (mp_base[m] + b->pos), b->bsz * 2, b->bsz * 2, b->bsz * 2, 1);
        }
    }
    return has_mask;
}

static void flash_attn_ext_f16_dec_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_fa_context * factx = (struct htp_fa_context *) data;
    const struct htp_ops_context * octx = factx->octx;
    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * mask  = (octx->src[3] && octx->src[3]->data) ? octx->src[3] : NULL;

    const uint32_t neq1 = q->ne[1];
    const uint32_t neq2 = q->ne[2];
    const uint32_t nek2 = k->ne[2];

    const uint32_t nbq1 = q->nb[1], nbq2 = q->nb[2], nbq3 = q->nb[3];
    const uint32_t nbk1 = k->nb[1];
    const uint32_t nbv1 = v->nb[1];

    const uint32_t DK = k->ne[0];
    const uint32_t DV = v->ne[0];

    const size_t size_q_row = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    const size_t size_k_row = DK * sizeof(__fp16);
    const size_t size_v_row = DV * sizeof(__fp16);

    const uint32_t G       = factx->dec_G;
    const uint32_t R       = factx->dec_R;
    const uint32_t n_split = factx->dec_n_split;
    const uint32_t bps     = factx->dec_bps;
    const uint32_t n_mseg  = factx->dec_n_mseg;
    const uint32_t nslots  = factx->dec_nslots;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    dma_queue * dma = octx->ctx->dma[ith];

    uint8_t * spad_q = factx->spad_q + factx->size_q_block * R * ith;
    uint8_t * spad_k = factx->spad_k + factx->size_k_block * nslots * ith;
    uint8_t * spad_v = factx->spad_v + factx->size_v_block * nslots * ith;
    uint8_t * spad_m = factx->spad_m + (mask ? factx->size_m_block * HVX_FA_DMA_CACHE_SIZE : 0) * ith;
    uint8_t * spad_a = factx->spad_a + factx->size_vkq_acc * R * ith;

    dma_cache m_cache;
    dma_cache_init(&m_cache, spad_m, factx->size_m_block, HVX_FA_DMA_CACHE_SIZE);

    const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
    const HVX_Vector v_cap     = (factx->logit_softcap != 0.0f) ? hvx_vec_splat_f16(factx->logit_softcap) : Q6_V_vzero();
    const HVX_Vector vinf      = Q6_Vh_vsplat_R(0xFC00);
    const HVX_Vector vmin      = Q6_Vh_vsplat_R(0xFBFF);
    const HVX_Vector v_log2e   = hvx_vec_splat_f16(EXP_LOG2E_F);
    const uint32_t stride_v2   = factx->size_v_row_padded * 2;

    float Mr[HVX_FA_DEC_R_MAX];
    float Sr[HVX_FA_DEC_R_MAX];

    // Staging ring: block j of a unit lands in slot j % nslots; the slot remembers what was
    // pushed for it so the consumer pops exactly what the pusher pushed.
    bool     slot_has_mask[HTP_FA_CLUSTER_MAX_SLOTS];
    uint32_t slot_bsz[HTP_FA_CLUSTER_MAX_SLOTS];
    struct hvx_fa_dec_blk blk;

    uint32_t next_static = ith;
    for (;;) {
        uint32_t unit;
        if (factx->cl_on) {
            // Per-head lists differ in length: hand units out dynamically, longest heads first.
            unit = __atomic_fetch_add(&factx->cl_next_unit, 1u, __ATOMIC_RELAXED);
        } else {
            unit = next_static;
            next_static += nth;
        }
        if (unit >= factx->dec_n_units) {
            break;
        }
        const uint32_t per_seq = nek2 * n_split;
        const uint32_t iq3 = unit / per_seq;
        const uint32_t rem = unit - iq3 * per_seq;
        uint32_t       kvh = rem / n_split;
        const uint32_t sp  = rem - kvh * n_split;

        // Ordinals [j0, j1) of this unit within the head's block list (dense: KV block indices).
        uint32_t j0, j1;
        if (factx->cl_on) {
            kvh = factx->cl_head_order[kvh];
            const uint32_t total = factx->cl_sel_n[kvh] + factx->cl_n_dense_blocks;
            const uint32_t bps_h = (total + n_split - 1) / n_split;
            j0 = sp * bps_h;
            j1 = (j0 + bps_h < total) ? j0 + bps_h : total;
        } else {
            j0 = factx->dec_b_base + sp * bps;
            j1 = (j0 + bps < factx->n_blocks) ? j0 + bps : factx->n_blocks;
        }

        // Row r = t*G + g of this unit is query row (iq1 = t, iq2 = kvh*G + g), i.e. global
        // row (iq3*neq2 + kvh*G + g)*neq1 + t.
        const uint32_t row_global0 = (iq3 * neq2 + kvh * G) * neq1;

        if (j0 >= j1) {
            // Empty range: neutral partials so the merge sees M very negative, S = 0, acc = 0.
            for (uint32_t r = 0; r < R; ++r) {
                const uint32_t t = r / G, g = r - t * G;
                uint8_t * part = hvx_fa_dec_partial(factx, row_global0 + g * neq1 + t, sp);
                ((float *) part)[0] = HTP_FA_M_INITIAL_VAL;
                ((float *) part)[1] = 0.0f;
                hvx_splat_f32_a(part + HVX_FA_DEC_PART_HDR, 0.0f, factx->size_vkq_acc / sizeof(float));
            }
            continue;
        }

        const uint32_t ik3 = fastdiv(iq3, &factx->broadcast_rk3);
        const uint32_t iv3 = fastdiv(iq3, &factx->broadcast_rv3);
        const uint32_t ik2 = kvh;
        const uint32_t iv2 = kvh;

        // Mask row base per segment; segment of row r is t when the mask broadcasts over
        // heads, else r itself.
        const __fp16 * mp_base[HVX_FA_DEC_R_MAX];
        if (mask) {
            const uint32_t im3 = fastmodulo(iq3, mask->ne[3], &factx->src3_div3);
            for (uint32_t m = 0; m < n_mseg; ++m) {
                const uint32_t t   = (n_mseg == neq1) ? m : m / G;
                const uint32_t g   = (n_mseg == neq1) ? 0 : m - t * G;
                const uint32_t iq2 = kvh * G + g;
                const uint32_t im2 = fastmodulo(iq2, mask->ne[2], &factx->src3_div2);
                mp_base[m] = (const __fp16 *) ((const uint8_t *) mask->data + t * mask->nb[1] + im2 * mask->nb[2] + im3 * mask->nb[3]);
            }
        }

        // Q rows into VTCM, in small batches so the DMA ring is never loaded with many rows.
        for (uint32_t r0 = 0; r0 < R; r0 += 4) {
            const uint32_t r1 = (r0 + 4 < R) ? r0 + 4 : R;
            for (uint32_t r = r0; r < r1; ++r) {
                const uint32_t t = r / G, g = r - t * G;
                const uint32_t iq2 = kvh * G + g;
                const uint8_t * q_row_ptr = (const uint8_t *) q->data + (t * nbq1 + iq2 * nbq2 + iq3 * nbq3);
                dma_queue_push(dma, dma_make_ptr(spad_q + r * factx->size_q_block, q_row_ptr), factx->size_q_row_padded, nbq1, size_q_row, 1);
            }
            for (uint32_t r = r0; r < r1; ++r) {
                uint8_t * qv = dma_queue_pop(dma).dst;
                if (factx->is_q_fp32) {
                    hvx_copy_f16_f32_aa(qv, qv, DK);   // in place f32 -> f16
                }
            }
        }

        // Stage the first nslots blocks of the range. Block j of the unit lands in slot j % nslots.
        for (uint32_t j = j0; j < j1 && j < j0 + nslots; ++j) {
            const uint32_t slot = (j - j0) % nslots;
            hvx_fa_dec_blk_src(factx, k, v, kvh, ik2, ik3, iv2, iv3, j, &blk);
            slot_has_mask[slot] = hvx_fa_dec_stage(dma, &m_cache, factx, &blk, spad_k + slot * factx->size_k_block, spad_v + slot * factx->size_v_block,
                                                   size_k_row, size_v_row, nbk1, nbv1, mp_base, n_mseg, mask != NULL);
            slot_bsz[slot] = blk.bsz;
        }

        for (uint32_t r = 0; r < R; ++r) {
            Mr[r] = HTP_FA_M_INITIAL_VAL;
            Sr[r] = 0.0f;
            hvx_splat_f32_a(spad_a + r * factx->size_vkq_acc, 0.0f, factx->size_vkq_acc / sizeof(float));
        }

        for (uint32_t j = j0; j < j1; ++j) {
            const uint32_t slot     = (j - j0) % nslots;
            const uint32_t bsz      = slot_bsz[slot];
            const bool     blk_mask = slot_has_mask[slot];

            uint8_t * k_base = dma_queue_pop(dma).dst;
            uint8_t * v_base = dma_queue_pop(dma).dst;
            const __fp16 * m_base[HVX_FA_DEC_R_MAX];
            if (blk_mask) {
                for (uint32_t m = 0; m < n_mseg; ++m) {
                    m_base[m] = (const __fp16 *) dma_queue_pop(dma).dst;
                }
            }

            const HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(bsz * sizeof(__fp16));

            for (uint32_t r = 0; r < R; ++r) {
                const uint32_t t = r / G, g = r - t * G;
                const uint32_t h = kvh * G + g;
                const uint8_t * q_ptr_vtcm = spad_q + r * factx->size_q_block;
                float * VKQ32 = (float *) (spad_a + r * factx->size_vkq_acc);

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_QK, unit);

                HVX_Vector scores_f16 = Q6_V_vzero();
                if (bsz > 0) {
                    HVX_Vector scores0 = hvx_dot_f16_f16_aa_rx32_tree(q_ptr_vtcm, k_base, factx->size_k_row_padded, DK, factx->scale);
                    HVX_Vector scores1 = (bsz > 32) ? hvx_dot_f16_f16_aa_rx32_tree(q_ptr_vtcm, k_base + 32 * factx->size_k_row_padded, factx->size_k_row_padded, DK, factx->scale) : Q6_V_vzero();
                    scores_f16 = hvx_vec_f32_to_f16(scores0, scores1);
                }

                if (factx->logit_softcap != 0.0f) {
                    scores_f16 = hvx_vec_tanh_f16(scores_f16);
                    scores_f16 = hvx_vec_mul_f16_f16(scores_f16, v_cap);
                }

                if (blk_mask) {
                    const uint32_t m = (n_mseg == neq1) ? t : r;
                    HVX_Vector m_vals_f16 = *(const HVX_UVector *) m_base[m];
                    HVX_VectorPred is_inf = Q6_Q_vcmp_eq_VhVh(m_vals_f16, vinf);
                    m_vals_f16 = Q6_V_vmux_QVV(is_inf, vmin, m_vals_f16);
                    HVX_Vector m_scaled = hvx_vec_mul_f16_f16(m_vals_f16, hvx_vec_splat_f16(factx->slopes[h]));
                    scores_f16 = Q6_V_vmux_QVV(q_tail_keep, hvx_vec_add_f16_f16(scores_f16, m_scaled), v_neg_inf);
                } else {
                    scores_f16 = Q6_V_vmux_QVV(q_tail_keep, scores_f16, v_neg_inf);
                }

                HVX_Vector v_max_f16 = hvx_vec_reduce_max_f16(scores_f16);
                HVX_Vector v_max     = Q6_V_lo_W(hvx_vec_f16_to_f32(v_max_f16));
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_QK, unit);

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_SFM, unit);
                {
                    HVX_Vector M_vec     = hvx_vec_splat_f32(Mr[r]);
                    HVX_Vector M_new_vec = Q6_Vsf_vmax_VsfVsf(v_max, M_vec);
                    HVX_Vector diff_vec  = HVX_OP_SUB_F32(M_vec, M_new_vec);

                    HVX_Vector diff_f16   = hvx_vec_f32_to_f16(diff_vec, diff_vec);
                    HVX_Vector diff_base2 = hvx_vec_mul_f16_f16(diff_f16, v_log2e);
                    HVX_Vector ms_f16     = hvx_vec_exp2_f16(diff_base2);
                    HVX_Vector ms_vec     = Q6_V_lo_W(hvx_vec_f16_to_f32(ms_f16));

                    hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, factx->size_vkq_acc / sizeof(float), ms_vec);

                    HVX_Vector v_m_vec_f16 = hvx_vec_f32_to_f16(M_new_vec, M_new_vec);
                    HVX_Vector v_s_minus_m = Q6_Vqf16_vsub_VhfVhf(scores_f16, v_m_vec_f16);
                    HVX_Vector v_s_minus_m_base2 = hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m), v_log2e);

                    HVX_Vector P = hvx_vec_exp2_f16(v_s_minus_m_base2);
                    P = Q6_V_vmux_QVV(q_tail_keep, P, Q6_V_vzero());

                    HVX_VectorPair P_pair = hvx_vec_f16_to_f32(P);
                    HVX_Vector p_sum_vec  = hvx_vec_reduce_sum_f32(HVX_OP_ADD_F32(Q6_V_lo_W(P_pair), Q6_V_hi_W(P_pair)));

                    HVX_Vector S_vec = HVX_OP_ADD_F32(HVX_OP_MUL_F32(hvx_vec_splat_f32(Sr[r]), ms_vec), p_sum_vec);
                    Mr[r] = hvx_vec_get_f32(M_new_vec);
                    Sr[r] = hvx_vec_get_f32(S_vec);

                    const uint8_t * v_ptr = v_base;
                    const uint32_t nvec_v = factx->size_v_row_padded / VLEN;
                    for (uint32_t j = 0; j < bsz; j += 2) {
                        if (j + 1 == bsz) {
                            HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            hvx_mad_f32_f16_aa_vec_noshuff(VKQ32, v_ptr, S0, nvec_v);
                            break;
                        }
                        HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                        HVX_Vector S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                        hvx_mad_f32_f16_aa_rx2_vec_noshuff(VKQ32, v_ptr, v_ptr + factx->size_v_row_padded, S0, S1, nvec_v);
                        v_ptr += stride_v2;
                    }
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, unit);
            }

            // Prefetch block j + nslots of this range into the slot just consumed.
            if (j + nslots < j1) {
                hvx_fa_dec_blk_src(factx, k, v, kvh, ik2, ik3, iv2, iv3, j + nslots, &blk);
                slot_has_mask[slot] = hvx_fa_dec_stage(dma, &m_cache, factx, &blk, k_base, v_base,
                                                       size_k_row, size_v_row, nbk1, nbv1, mp_base, n_mseg, mask != NULL);
                slot_bsz[slot] = blk.bsz;
            }
        }

        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t t = r / G, g = r - t * G;
            uint8_t * part = hvx_fa_dec_partial(factx, row_global0 + g * neq1 + t, sp);
            ((float *) part)[0] = Mr[r];
            ((float *) part)[1] = Sr[r];
            hvx_unshuff_copy_f32_aa(part + HVX_FA_DEC_PART_HDR, spad_a + r * factx->size_vkq_acc, factx->size_v_row_padded / VLEN);
        }
    }
}

// Merge the n_split partials of each row, apply sinks, normalise, store. Rows go
// round-robin to threads; a row's acc is assembled in the thread's first accumulator slot.
static void flash_attn_ext_f16_merge_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_fa_context * factx = (struct htp_fa_context *) data;
    const struct htp_ops_context * octx = factx->octx;
    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * sinks = octx->src[4];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t neq1 = q->ne[1], neq2 = q->ne[2], neq3 = q->ne[3];
    const uint32_t DV   = v->ne[0];
    const uint32_t rows_total = neq1 * neq2 * neq3;
    const uint32_t n_split    = factx->dec_n_split;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    float * acc = (float *) (factx->spad_a + factx->size_vkq_acc * factx->dec_R * ith);

    for (uint32_t row = ith; row < rows_total; row += nth) {
        const uint32_t iq3 = row / (neq2 * neq1);
        const uint32_t rem = row - iq3 * neq2 * neq1;
        const uint32_t iq2 = rem / neq1;
        const uint32_t iq1 = rem - iq2 * neq1;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, row);

        // Partials sp < n_split are the HVX units' (VTCM); the rest are the GPU's (DDR, hetero).
        const uint32_t n_all = n_split + factx->het_gpu_nsplit;
        #define HVX_FA_PART(sp) ((sp) < n_split ? (const uint8_t *) hvx_fa_dec_partial(factx, row, (sp)) \
                                                : factx->het_parts + ((size_t) row * factx->het_gpu_nsplit + ((sp) - n_split)) * factx->dec_stride_part)
        float M = HTP_FA_M_INITIAL_VAL;
        for (uint32_t sp = 0; sp < n_all; ++sp) {
            const float m = ((const float *) HVX_FA_PART(sp))[0];
            M = (m > M) ? m : M;
        }

        // The accumulators are padded to size_vkq_acc (a multiple of 128 B); work on the
        // whole padded span so head dims that are not multiples of 32 (40, 72, 80, ...)
        // keep their tail lanes -- only DV floats are stored below.
        const uint32_t nvec_acc = factx->size_vkq_acc / sizeof(HVX_Vector);
        float S = 0.0f;
        hvx_splat_f32_a(acc, 0.0f, factx->size_vkq_acc / sizeof(float));
        for (uint32_t sp = 0; sp < n_all; ++sp) {
            const uint8_t * part = HVX_FA_PART(sp);
            const float m_s = ((const float *) part)[0];
            const float s_s = ((const float *) part)[1];
            // An empty split carries M = HTP_FA_M_INITIAL_VAL; its weight must be exactly 0
            // rather than whatever the f32 exp does with an argument of -1e4 (the row kernel
            // only ever routes that case through the saturating f16 exp2).
            const float d_s = m_s - M;
            if (d_s < -80.0f) {
                continue;
            }
            HVX_Vector w_vec = hvx_vec_exp_f32(hvx_vec_splat_f32(d_s));
            const float w = hvx_vec_get_f32(w_vec);
            S += w * s_s;
            const HVX_Vector * pv = (const HVX_Vector *) (part + HVX_FA_DEC_PART_HDR);
            HVX_Vector * av = (HVX_Vector *) acc;
            for (uint32_t i = 0; i < nvec_acc; ++i) {
                av[i] = HVX_OP_ADD_F32(av[i], HVX_OP_MUL_F32(w_vec, pv[i]));
            }
        }

        if (sinks) {
            const float s = ((const float *) sinks->data)[iq2];
            float vs = 1.0f;
            if (s > M) {
                HVX_Vector ms_vec = hvx_vec_exp_f32(hvx_vec_splat_f32(M - s));
                hvx_scale_vec_f32_aa((uint8_t *) acc, (const uint8_t *) acc, DV, ms_vec);
                S = S * hvx_vec_get_f32(ms_vec) + vs;
            } else {
                vs = hvx_vec_get_f32(hvx_vec_exp_f32(hvx_vec_splat_f32(s - M)));
                S += vs;
            }
        }

        const float S_inv = (S == 0.0f) ? 0.0f : 1.0f / S;
        hvx_scale_f32_aa((uint8_t *) acc, (const uint8_t *) acc, DV, S_inv);

        uint8_t * dst_ptr = (uint8_t *) dst->data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3];
        if (dst->type == HTP_TYPE_F32) {
            hvx_copy_f32_ua(dst_ptr, (const uint8_t *) acc, DV);
        } else if (dst->type == HTP_TYPE_F16) {
            hvx_copy_f16_f32_ua(dst_ptr, (const uint8_t *) acc, DV);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, row);
        #undef HVX_FA_PART
    }
}

// Pick n_split so the (sequence, KV head, split) units balance across the pool: minimise
// the makespan ceil(units / nth) * ceil(n_blocks / n_split); smallest n_split on ties.
static uint32_t hvx_fa_dec_pick_split(uint32_t n_seq_heads, uint32_t n_blocks, uint32_t nth, uint32_t max_split) {
    uint32_t best = 1, best_ms = UINT32_MAX;
    const uint32_t lim = (n_blocks < max_split) ? n_blocks : max_split;
    for (uint32_t s = 1; s <= lim; ++s) {
        const uint32_t units  = n_seq_heads * s;
        const uint32_t rounds = (units + nth - 1) / nth;
        const uint32_t bps    = (n_blocks + s - 1) / s;
        const uint32_t ms     = rounds * bps;
        if (ms < best_ms) {
            best_ms = ms;
            best    = s;
        }
    }
    return best;
}

// ---- cluster-selected pages: read the shadow buffer and lay out this op's work ------------------
// The header, directory and chunk table are CPU/GPU-written DDR the DSP reads through its cache,
// so every read is preceded by an invalidate; the pages themselves only move by DMA.

struct hvx_fa_cl_setup {
    const struct htp_fa_cluster_header * hdr;
    const uint8_t * layer_base;
    const uint8_t * host_sel;     // per-head lists in DDR (host-list mode)
    uint32_t        n_cand, dense_start, max_total;
};

static inline void hvx_fa_cl_inval(const void * p, size_t n) {
    uintptr_t a = (uintptr_t) p & ~(uintptr_t) (HEX_L2_LINE_SIZE - 1);
    const uintptr_t e = (uintptr_t) p + n;
    for (; a < e; a += HEX_L2_LINE_SIZE) {
        Q6_dcinva_A((void *) a);
    }
}

// ---- in-place descriptors computed on the DSP -------------------------------------------------
// Page p of a layer is rows [p*PK, p*PK+PK) of the positional K cache. The host publishes how many
// rows are complete (htp_fa_cluster_hdir.rows_valid, written before the graph is submitted); the op
// computes the f16 mean of every complete page not yet described -- one 2D descriptor per page brings
// all heads of its rows into VTCM -- and publishes the count in the DSP-owned directory. The rows
// come through the DMA engine like every other K row, so the CPU never reads DSP-written data.
struct hvx_fa_desc_job {
    const struct htp_ops_context * octx;
    const uint8_t * kdata;
    uint32_t  nbk1, nek2, DK, PK, p0, p1, n_pages_max, cbytes;
    size_t    row_bytes;      // nek2 * DK * 2: the heads of one row, contiguous
    uint8_t * cent;           // this layer's descriptors: [nek2][n_pages_max] x cbytes
    uint8_t * vtcm;
    size_t    vtcm_stride;    // PK * row_bytes per thread
};

static void flash_attn_ext_f16_desc_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct hvx_fa_desc_job * j = (const struct hvx_fa_desc_job *) data;
    dma_queue * dma = j->octx->ctx->dma[ith];
    uint8_t * stage = j->vtcm + (size_t) ith * j->vtcm_stride;
    const HVX_Vector inv = hvx_vec_splat_f32(1.0f / (float) j->PK);
    const uint32_t nvec = j->DK / 64;   // f16 vectors per head row (DK 128 -> 2)
    for (uint32_t p = j->p0 + ith; p < j->p1; p += nth) {
        dma_queue_push(dma, dma_make_ptr(stage, j->kdata + (size_t) p * j->PK * j->nbk1), j->row_bytes, j->nbk1, j->row_bytes, j->PK);
        dma_queue_pop(dma);
        for (uint32_t h = 0; h < j->nek2; ++h) {
            HVX_Vector acc[2][2];
            for (uint32_t v = 0; v < nvec; ++v) { acc[v][0] = Q6_V_vzero(); acc[v][1] = Q6_V_vzero(); }
            for (uint32_t i = 0; i < j->PK; ++i) {
                const HVX_Vector * row = (const HVX_Vector *) (stage + (size_t) i * j->row_bytes + (size_t) h * j->DK * 2);
                for (uint32_t v = 0; v < nvec; ++v) {
                    HVX_VectorPair w = hvx_vec_f16_to_f32(row[v]);
                    acc[v][0] = HVX_OP_ADD_F32(acc[v][0], Q6_V_lo_W(w));
                    acc[v][1] = HVX_OP_ADD_F32(acc[v][1], Q6_V_hi_W(w));
                }
            }
            HVX_Vector * out = (HVX_Vector *) (j->cent + ((size_t) h * j->n_pages_max + p) * j->cbytes);
            for (uint32_t v = 0; v < nvec; ++v) {
                out[v] = hvx_vec_f32_to_f16(HVX_OP_MUL_F32(acc[v][0], inv), HVX_OP_MUL_F32(acc[v][1], inv));
            }
        }
    }
}

// Bring the DSP-owned directory of layer il up to hdir.rows_valid: shrink after a rewind, or describe
// the newly complete pages. Returns false when the op cannot run in cluster mode.
static bool hvx_fa_cl_dsp_desc(const struct htp_ops_context * octx, const uint8_t * base, const struct htp_fa_cluster_header * hdr,
                               uint32_t il, const uint8_t * layer_base, struct htp_fa_cluster_dir * dir, uint32_t nek2, uint32_t DK) {
    const struct htp_fa_cluster_hdir * hd = (const struct htp_fa_cluster_hdir *) (base + HTP_FA_CLUSTER_HDIR_OFF + (size_t) il * HTP_FA_CLUSTER_DIR_STRIDE);
    hvx_fa_cl_inval(hd, sizeof(*hd));
    uint32_t want = hd->rows_valid / hdr->page_keys;
    if (want > hdr->n_pages_max) {
        want = hdr->n_pages_max;
    }
    if (want < dir->n_pages_pub) {
        dir->n_pages_pub = want;
        dir->covered_end = want * hdr->page_keys;
        qurt_mem_cache_clean((qurt_addr_t) dir, sizeof(*dir), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    } else if (want > dir->n_pages_pub) {
        const struct htp_tensor * k = octx->src[1];
        const size_t row_bytes = (size_t) nek2 * DK * 2;
        if (k->nb[2] != DK * 2 || k->nb[1] < row_bytes || (DK % 64) != 0 || DK > 128) {
            return false;   // heads must be contiguous within a row
        }
        struct hvx_fa_desc_job j;
        j.octx = octx; j.kdata = (const uint8_t *) (uintptr_t) k->data; j.nbk1 = k->nb[1]; j.nek2 = nek2; j.DK = DK; j.PK = hdr->page_keys;
        j.p0 = dir->n_pages_pub; j.p1 = want; j.n_pages_max = hdr->n_pages_max; j.cbytes = hdr->centroid_bytes; j.row_bytes = row_bytes;
        j.cent = (uint8_t *) (uintptr_t) (layer_base + hdr->off_centroids);
        j.vtcm = octx->ctx->vtcm_base; j.vtcm_stride = (size_t) hdr->page_keys * row_bytes;
        uint32_t nth = octx->n_threads;
        while (nth > 1 && j.vtcm_stride * nth > octx->ctx->vtcm_size) {
            nth--;
        }
        if (j.vtcm_stride > octx->ctx->vtcm_size) {
            return false;
        }
        work_queue_run(octx->ctx->work_queue, flash_attn_ext_f16_desc_thread, &j, nth);
        // publish: descriptors first (the select pass reads them through the DMA engine), then the count
        for (uint32_t h = 0; h < nek2; ++h) {
            qurt_mem_cache_clean((qurt_addr_t) (j.cent + ((size_t) h * hdr->n_pages_max + j.p0) * hdr->centroid_bytes),
                                 (size_t) (want - j.p0) * hdr->centroid_bytes, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        }
        dir->n_pages_pub = want;
        dir->covered_end = want * hdr->page_keys;
        qurt_mem_cache_clean((qurt_addr_t) dir, sizeof(*dir), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    }
    return true;
}

// Returns true when this op runs in cluster mode; fills factx->cl_* except the VTCM list copy
// (hvx_fa_cl_copy_lists, after the VTCM allocation). Any inconsistency degrades to dense.
static bool hvx_fa_cl_setup(struct htp_fa_context * factx, const struct htp_ops_context * octx,
                            uint32_t nek1, uint32_t nek2, uint32_t DK, uint32_t DV, struct hvx_fa_cl_setup * s) {
    const uint8_t * base = (const uint8_t *) (uintptr_t) octx->src[7]->data;
    hvx_fa_cl_inval(base, sizeof(struct htp_fa_cluster_header));
    const struct htp_fa_cluster_header * hdr = (const struct htp_fa_cluster_header *) base;
    if (hdr->magic != HTP_FA_CLUSTER_MAGIC || hdr->version != HTP_FA_CLUSTER_VERSION) {
        return false;
    }
    if (hdr->n_kv_heads != nek2 || hdr->D != DK || DK != DV || nek2 > HTP_FA_CLUSTER_MAX_HEADS ||
        (hdr->page_keys != 16 && hdr->page_keys != 32 && hdr->page_keys != 64) ||
        hdr->page_bytes != hdr->page_keys * DK * 2) {
        return false;
    }
    const uint32_t il = (uint32_t) octx->op_params[HTP_FA_CLUSTER_OPP_LAYER];
    if (il >= hdr->n_layers) {
        return false;
    }
    struct htp_fa_cluster_dir * dir = (struct htp_fa_cluster_dir *) (uintptr_t) (base + HTP_FA_CLUSTER_DIR_OFF + (size_t) il * HTP_FA_CLUSTER_DIR_STRIDE);
    hvx_fa_cl_inval(dir, sizeof(*dir));
    const bool inplace = (hdr->flags & HTP_FA_CLUSTER_HDR_INPLACE) != 0;
    const uint8_t * layer_base = base + hdr->layer0_off + (size_t) il * hdr->layer_stride;
    if (inplace && (hdr->flags & HTP_FA_CLUSTER_HDR_DSP_DESC)) {
        if (!hvx_fa_cl_dsp_desc(octx, base, hdr, il, layer_base, dir, nek2, DK)) {
            return false;
        }
    }
    if (dir->stale || dir->covered_end == 0 || (!inplace && dir->n_chunks == 0)) {
        return false;
    }
    // Rows the host vouches for before this graph was submitted. The sidecar publishes asynchronously:
    // after a reset or rewind its directory is stale until it catches up, and the first graph of a new
    // context (position 0, the attention sink) must not attend pages of the previous one.
    uint32_t rows_ok = UINT32_MAX;
    if (hdr->flags & HTP_FA_CLUSTER_HDR_HOST_ROWS) {
        const struct htp_fa_cluster_hdir * hd = (const struct htp_fa_cluster_hdir *) (base + HTP_FA_CLUSTER_HDIR_OFF + (size_t) il * HTP_FA_CLUSTER_DIR_STRIDE);
        hvx_fa_cl_inval(hd, sizeof(*hd));
        rows_ok = hd->rows_valid;
    }
    const struct htp_fa_cluster_chunk * chunks = (const struct htp_fa_cluster_chunk *) (layer_base + hdr->off_chunks);
    const uint32_t n_chunks = MIN(dir->n_chunks, hdr->max_chunks);
    hvx_fa_cl_inval(chunks, (size_t) n_chunks * sizeof(*chunks));

    // dense_start = the last chunk boundary that leaves at least W positions to the dense tail;
    // candidates are the pages of every chunk up to it. Chunks are appended in position order.
    const uint32_t W = (uint32_t) octx->op_params[HTP_FA_CLUSTER_OPP_WINDOW];
    const bool     runs = (hdr->flags & HTP_FA_CLUSTER_HDR_RUNS) != 0;
    const uint32_t units_max = runs ? hdr->n_runs_max : hdr->n_pages_max;   // descriptor units per head (runs or pages)
    uint32_t dense_start = 0, n_cand = 0;
    if (inplace) {
        // In-place pages: the candidates are the complete published pages that leave at least W
        // positions to the dense tail; the boundary is rounded down to a 64-key block so the tail
        // is addressed exactly as the dense kernel addresses it (tail = W .. W+63 keys).
        const uint32_t avail = MIN(MIN(dir->covered_end, dir->n_pages_pub * hdr->page_keys), rows_ok);
        if (nek1 <= W) {
            return false;
        }
        dense_start = MIN(avail, nek1 - W) & ~(FLASH_ATTN_BLOCK_SIZE - 1);
        n_cand      = dense_start / hdr->page_keys;
    }
    for (uint32_t c = 0; !inplace && c < n_chunks; ++c) {
        const struct htp_fa_cluster_chunk ch = chunks[c];
        if (ch.pos_end > dir->covered_end || ch.pos_end > rows_ok || ch.pos_end + W > nek1 || (ch.pos_end % FLASH_ATTN_BLOCK_SIZE) != 0) {
            break;
        }
        if (ch.page_first + ch.n_pages > dir->n_pages_pub || ch.page_first + ch.n_pages > units_max) {
            break;
        }
        dense_start = ch.pos_end;
        n_cand      = ch.page_first + ch.n_pages;
    }
    if (n_cand == 0 || dense_start / FLASH_ATTN_BLOCK_SIZE > factx->n_blocks) {
        return false;
    }

    factx->cl_flags          = (uint32_t) octx->op_params[HTP_FA_CLUSTER_OPP_FLAGS];
    factx->cl_inplace        = inplace;
    factx->cl_runs           = runs;
    factx->cl_k_pages        = inplace ? NULL : layer_base + hdr->off_k_pages;
    factx->cl_v_pages        = inplace ? NULL : layer_base + hdr->off_v_pages;
    factx->cl_page_bytes     = hdr->page_bytes;
    factx->cl_page_keys      = hdr->page_keys;
    factx->cl_head_stride    = (size_t) hdr->n_pages_max * hdr->page_bytes;
    factx->cl_n_pages_max    = runs ? hdr->n_runs_max : hdr->n_pages_max;   // per-head descriptor / list stride (units)
    factx->cl_rows_max       = hdr->n_pages_max * hdr->page_keys;
    factx->cl_row_bytes      = DK * 2;
    factx->cl_runs_tab       = layer_base + hdr->off_runs;
    factx->cl_max_blk        = runs ? hdr->n_pages_max + hdr->n_runs_max : 0;
    factx->cl_budget_rows    = 0;
    factx->cl_n_cand         = n_cand;
    factx->cl_dense_b0       = dense_start / FLASH_ATTN_BLOCK_SIZE;
    factx->cl_n_dense_blocks = factx->n_blocks - factx->cl_dense_b0;

    s->hdr = hdr; s->layer_base = layer_base; s->n_cand = n_cand; s->dense_start = dense_start; s->host_sel = NULL;

    factx->cl_density    = (uint32_t) octx->op_params[HTP_FA_CLUSTER_OPP_DENSITY];
    factx->cl_budget     = 0;
    factx->cl_cent       = layer_base + hdr->off_centroids;
    factx->cl_cent_bytes = hdr->centroid_bytes;
    factx->cl_sel_rows   = (n_cand + 31) & ~31u;
    if (factx->cl_density != 0) {
        // On-device selection: every head gets the same budget; the lists are built by the select
        // pass after the VTCM allocation. The descriptor must be exactly one DK-wide f16 row.
        if (hdr->centroid_bytes != hex_round_up((size_t) DK * 2, 128) || DK % 64 != 0) {
            return false;
        }
        if (runs) {
            // Row budget over the candidate rows minus the forced (sink) rows; the select pass takes whole
            // runs in score order until it is filled. The block count is only known after selection, so
            // the split pick gets an upper bound.
            const struct htp_fa_cluster_run * r0 = (const struct htp_fa_cluster_run *) factx->cl_runs_tab;
            hvx_fa_cl_inval(r0, sizeof(*r0));
            const uint32_t forced_rows = (n_cand > 0 && (r0->flags & HTP_FA_CLUSTER_RUN_FORCED)) ? r0->n_rows : 0;
            const uint32_t cand_rows   = dense_start > forced_rows ? dense_start - forced_rows : 0;
            factx->cl_budget_rows = (uint32_t) (((uint64_t) cand_rows * factx->cl_density + 999) / 1000);
            uint32_t ub = factx->cl_budget_rows / FLASH_ATTN_BLOCK_SIZE + n_cand + 1;
            if (ub > factx->cl_max_blk) ub = factx->cl_max_blk;
            factx->cl_budget = ub;
            for (uint32_t h = 0; h < nek2; ++h) {
                factx->cl_sel_n[h]      = ub;
                factx->cl_head_order[h] = (uint8_t) h;
            }
            s->max_total = ub + factx->cl_n_dense_blocks;
            return true;
        }
        uint32_t b = (uint32_t) (((uint64_t) n_cand * factx->cl_density + 999) / 1000);
        const uint32_t min_pages = HTP_FA_CLUSTER_FLAG_MINPAGES(factx->cl_flags);
        if (b < min_pages) b = min_pages;
        if (b > n_cand)    b = n_cand;
        factx->cl_budget = b;
        for (uint32_t h = 0; h < nek2; ++h) {
            factx->cl_sel_n[h]      = b;
            factx->cl_head_order[h] = (uint8_t) h;
        }
        s->max_total = b + factx->cl_n_dense_blocks;
        return true;
    }
    if (runs) {
        return false;   // host-written lists are not defined for runs; density 0 falls back to dense
    }
    // Host-written lists: lengths now (the split pick needs them), pages after the VTCM alloc.
    s->host_sel = layer_base + hdr->off_host_sel;
    hvx_fa_cl_inval(s->host_sel, (size_t) nek2 * hdr->host_sel_stride);
    s->max_total = 0;
    for (uint32_t h = 0; h < nek2; ++h) {
        uint32_t n = *(const uint32_t *) (s->host_sel + (size_t) h * hdr->host_sel_stride);
        if (n > n_cand) {
            n = n_cand;
        }
        factx->cl_sel_n[h] = n;
        const uint32_t total = n + factx->cl_n_dense_blocks;
        if (total > s->max_total) {
            s->max_total = total;
        }
    }
    // Heads by descending list length: dynamic dispatch hands the long ones out first.
    for (uint32_t h = 0; h < nek2; ++h) {
        factx->cl_head_order[h] = (uint8_t) h;
    }
    for (uint32_t i = 1; i < nek2; ++i) {
        const uint8_t x = factx->cl_head_order[i];
        uint32_t j = i;
        while (j > 0 && factx->cl_sel_n[factx->cl_head_order[j - 1]] < factx->cl_sel_n[x]) {
            factx->cl_head_order[j] = factx->cl_head_order[j - 1];
            --j;
        }
        factx->cl_head_order[j] = x;
    }
    return true;
}

static void hvx_fa_cl_copy_lists(struct htp_fa_context * factx, const struct hvx_fa_cl_setup * s, uint32_t nek2) {
    const uint32_t stride = s->hdr->host_sel_stride;
    if (s->host_sel) {
        for (uint32_t h = 0; h < nek2; ++h) {
            const uint16_t * pages = (const uint16_t *) (s->host_sel + (size_t) h * stride + 128);
            uint16_t * dst = factx->cl_sel_pages + (size_t) h * factx->cl_n_pages_max;
            const uint32_t n = factx->cl_sel_n[h];
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t p = pages[i];
                if (p >= factx->cl_n_cand) {
                    p = factx->cl_n_cand - 1;   // clamp the INDEX, never the count (sparse-HMX convention)
                }
                dst[i] = (uint16_t) p;
            }
        }
    }
    if (factx->cl_flags & HTP_FA_CLUSTER_FLAG_ECHO) {
        uint8_t * echo = (uint8_t *) (uintptr_t) (s->layer_base + s->hdr->off_echo_sel);
        for (uint32_t h = 0; h < nek2; ++h) {
            uint8_t * e = echo + (size_t) h * stride;
            *(uint32_t *) e = factx->cl_sel_n[h];
            if (factx->cl_runs) {
                memcpy(e + 128, factx->cl_sel_blk + (size_t) h * factx->cl_max_blk, (size_t) factx->cl_sel_n[h] * sizeof(struct htp_fa_cluster_blk));
            } else {
                memcpy(e + 128, factx->cl_sel_pages + (size_t) h * factx->cl_n_pages_max, (size_t) factx->cl_sel_n[h] * 2);
            }
        }
        qurt_mem_cache_clean((qurt_addr_t) echo, (size_t) nek2 * stride, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    }
}

// On-device page selection: one unit per KV head. Stage the head's page descriptors and its G
// query rows in VTCM, score every descriptor against every query row of the group
// (hvx_dot_f16_f16_aa_rx32_tree, 32 descriptors per call), fold the group by max (or sum), then
// take the top cl_budget pages by repeated masked arg-max (B passes over n_cand/32 vectors; a few
// microseconds for the budgets that matter). The list comes out in score order.
static void flash_attn_ext_f16_select_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_fa_context * factx = (struct htp_fa_context *) data;
    const struct htp_ops_context * octx = factx->octx;
    const struct htp_tensor * q = octx->src[0];
    const struct htp_tensor * k = octx->src[1];

    const uint32_t nek2 = k->ne[2];
    const uint32_t DK   = k->ne[0];
    const uint32_t G    = factx->dec_G;
    const uint32_t nbq1 = q->nb[1], nbq2 = q->nb[2];
    const size_t   size_q_row = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    const uint32_t n_cand  = factx->cl_n_cand;
    const uint32_t n_vec   = (n_cand + 31) / 32;
    const uint32_t B       = factx->cl_budget;
    const uint32_t cbytes  = factx->cl_cent_bytes;
    const bool     fold_sum = (factx->cl_flags & HTP_FA_CLUSTER_FLAG_FOLD_SUM) != 0;

    dma_queue * dma = octx->ctx->dma[ith];
    uint8_t * spad_q = factx->spad_q + factx->size_q_block * factx->dec_R * ith;
    uint8_t * spad_c = factx->cl_sel_c + factx->cl_sel_c_stride * ith;
    HVX_Vector * scores = (HVX_Vector *) (factx->cl_sel_s + factx->cl_sel_s_stride * ith);

    int32_t __attribute__((aligned(128))) ramp[32];
    for (int i = 0; i < 32; ++i) ramp[i] = i;
    const HVX_Vector iota    = *(const HVX_Vector *) ramp;
    const HVX_Vector neg_inf = hvx_vec_splat_f32(-INFINITY);
    const int32_t    BIG     = 0x40000000;

    for (uint32_t kvh = ith; kvh < nek2; kvh += nth) {
        // descriptors of this head, in chunks of <= 128 rows per descriptor
        const uint8_t * cent = factx->cl_cent + (size_t) kvh * factx->cl_n_pages_max * cbytes;
        for (uint32_t r0 = 0; r0 < n_cand; r0 += 128) {
            const uint32_t nr = MIN(128u, n_cand - r0);
            dma_queue_push(dma, dma_make_ptr(spad_c + (size_t) r0 * cbytes, cent + (size_t) r0 * cbytes), cbytes, cbytes, cbytes, nr);
        }
        struct htp_fa_cluster_run * runs = NULL;
        if (factx->cl_runs) {
            runs = (struct htp_fa_cluster_run *) (factx->cl_sel_r + factx->cl_sel_r_stride * ith);
            const uint8_t * src = factx->cl_runs_tab + (size_t) kvh * factx->cl_n_pages_max * sizeof(struct htp_fa_cluster_run);
            dma_queue_push_single_1d(dma, dma_make_ptr((uint8_t *) runs, src), hex_round_up((size_t) n_cand * sizeof(struct htp_fa_cluster_run), 128));
        }
        // the G query rows of this KV head (decode: neq1 == 1, so row g is head kvh*G + g)
        for (uint32_t g = 0; g < G; ++g) {
            const uint8_t * q_row = (const uint8_t *) q->data + (size_t) (kvh * G + g) * nbq2;
            dma_queue_push(dma, dma_make_ptr(spad_q + g * factx->size_q_block, q_row), factx->size_q_row_padded, nbq1, size_q_row, 1);
        }
        for (uint32_t r0 = 0; r0 < n_cand; r0 += 128) {
            dma_queue_pop(dma);
        }
        if (factx->cl_runs) {
            dma_queue_pop(dma);
        }
        for (uint32_t g = 0; g < G; ++g) {
            uint8_t * qv = dma_queue_pop(dma).dst;
            if (factx->is_q_fp32) {
                hvx_copy_f16_f32_aa(qv, qv, DK);
            }
        }

        // scores: lane j of vector i is the folded score of page 32*i + j
        for (uint32_t i = 0; i < n_vec; ++i) {
            const uint8_t * rows = spad_c + (size_t) i * 32 * cbytes;
            HVX_Vector s = hvx_dot_f16_f16_aa_rx32_tree(spad_q, rows, cbytes, DK, 1.0f);
            for (uint32_t g = 1; g < G; ++g) {
                HVX_Vector sg = hvx_dot_f16_f16_aa_rx32_tree(spad_q + g * factx->size_q_block, rows, cbytes, DK, 1.0f);
                s = fold_sum ? HVX_OP_ADD_F32(s, sg) : Q6_Vsf_vmax_VsfVsf(s, sg);
            }
            const uint32_t valid = (i + 1) * 32 <= n_cand ? 32 : n_cand - i * 32;
            if (valid < 32) {
                s = Q6_V_vmux_QVV(Q6_Q_vsetq_R(valid * 4), s, neg_inf);
            }
            scores[i] = s;
        }

        if (factx->cl_runs) {
            // Whole clusters: forced runs (the sinks) score +inf and are free; empty runs never score;
            // take runs in score order while they fit the row budget (the first non-forced run always
            // fits), expanding each into 64-row DMA blocks.
            float * sf = (float *) scores;
            for (uint32_t r = 0; r < n_cand; ++r) {
                if (runs[r].flags & HTP_FA_CLUSTER_RUN_FORCED) sf[r] = INFINITY;
                else if (runs[r].n_rows == 0)                   sf[r] = -INFINITY;
            }
            struct htp_fa_cluster_blk * out = factx->cl_sel_blk + (size_t) kvh * factx->cl_max_blk;
            uint32_t nb = 0, cum = 0, taken = 0;
            const uint32_t budget = factx->cl_budget_rows;
            for (;;) {
                HVX_Vector m = scores[0];
                for (uint32_t i = 1; i < n_vec; ++i) {
                    m = Q6_Vsf_vmax_VsfVsf(m, scores[i]);
                }
                const HVX_Vector mx = hvx_vec_reduce_max_f32(m);
                if (hvx_vec_get_f32(mx) == -INFINITY) {
                    break;
                }
                HVX_Vector best = Q6_V_vzero();
                for (uint32_t i = 0; i < n_vec; ++i) {
                    const HVX_VectorPred eq = Q6_Q_vcmp_eq_VwVw(scores[i], mx);
                    const HVX_Vector cand = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(BIG), Q6_Vw_vadd_VwVw(iota, Q6_V_vsplat_R((int32_t) (i * 32))));
                    best = Q6_Vw_vmax_VwVw(best, Q6_V_vmux_QVV(eq, cand, Q6_V_vzero()));
                }
                const int32_t  idx = BIG - hvx_vec_get_i32(hvx_vec_reduce_max_i32(best));
                const uint32_t vi = (uint32_t) idx / 32, li = (uint32_t) idx % 32;
                scores[vi] = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(iota, Q6_V_vsplat_R((int32_t) li)), neg_inf, scores[vi]);
                if ((uint32_t) idx >= n_cand) {
                    break;
                }
                const struct htp_fa_cluster_run rr = runs[idx];
                const bool forced = (rr.flags & HTP_FA_CLUSTER_RUN_FORCED) != 0;
                if (!forced) {
                    if (cum + rr.n_rows > budget && taken > 0) {
                        break;
                    }
                    cum += rr.n_rows;
                    taken++;
                }
                for (uint32_t r0 = 0; r0 < rr.n_rows && nb < factx->cl_max_blk; r0 += FLASH_ATTN_BLOCK_SIZE) {
                    out[nb].row = rr.row_first + r0;
                    out[nb].bsz = (uint16_t) MIN(FLASH_ATTN_BLOCK_SIZE, rr.n_rows - r0);
                    out[nb].pad = 0;
                    nb++;
                }
                if (nb >= factx->cl_max_blk || (!forced && cum >= budget)) {
                    break;
                }
            }
            factx->cl_sel_n[kvh] = nb;
            continue;
        }

        // forced pages: the first F candidate pages of the layer (the sink tokens live in page 0)
        // score +inf so the arg-max takes them first; they count against the budget
        {
            uint32_t F = HTP_FA_CLUSTER_FLAG_FORCE(factx->cl_flags);
            if (F > B) F = B;
            float * sf = (float *) scores;
            for (uint32_t f = 0; f < F; ++f) sf[f] = INFINITY;
        }

        // top-B by repeated masked arg-max
        uint16_t * out = factx->cl_sel_pages + (size_t) kvh * factx->cl_n_pages_max;
        for (uint32_t b = 0; b < B; ++b) {
            HVX_Vector m = scores[0];
            for (uint32_t i = 1; i < n_vec; ++i) {
                m = Q6_Vsf_vmax_VsfVsf(m, scores[i]);
            }
            const HVX_Vector mx = hvx_vec_reduce_max_f32(m);
            // first lane equal to the max, as BIG - global_index so the max reduction finds the min index
            HVX_Vector best = Q6_V_vzero();
            for (uint32_t i = 0; i < n_vec; ++i) {
                const HVX_VectorPred eq = Q6_Q_vcmp_eq_VwVw(scores[i], mx);
                const HVX_Vector cand = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(BIG), Q6_Vw_vadd_VwVw(iota, Q6_V_vsplat_R((int32_t) (i * 32))));
                best = Q6_Vw_vmax_VwVw(best, Q6_V_vmux_QVV(eq, cand, Q6_V_vzero()));
            }
            const int32_t  idx = BIG - hvx_vec_get_i32(hvx_vec_reduce_max_i32(best));
            const uint32_t vi = (uint32_t) idx / 32, li = (uint32_t) idx % 32;
            scores[vi] = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(iota, Q6_V_vsplat_R((int32_t) li)), neg_inf, scores[vi]);
            out[b] = (uint16_t) idx;
        }
    }
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

    // Split-KV decode path (flash_attn_ext_f16_dec_thread). Host opt-out via
    // kparams->u.hvx.split_kv == 0; falls back when rows-per-unit exceed the scratch bound,
    // when query heads do not divide evenly over KV heads, or when there is nothing to do.
    const uint32_t neq1 = q->ne[1], neq2 = q->ne[2], neq3 = q->ne[3];
    const uint32_t nek2 = k->ne[2];
    const uint32_t G    = (nek2 && (neq2 % nek2) == 0) ? neq2 / nek2 : 0;
    const uint32_t R    = neq1 * G;
    bool dec = kparams->u.hvx.split_kv != 0 && G > 0 && R >= 1 && R <= HVX_FA_DEC_R_MAX &&
               v->ne[2] == nek2 && factx.n_blocks > 0;

    // Heterogeneous split: the host tagged this node and attached the control buffer as src[7].
    // The GPU computes KV blocks [0, het_gpu_blocks) and writes its partials into the buffer.
    uint32_t het_gpu_blocks = 0, het_slot = 0, het_nsplit = 0;
    uint8_t * het_base = NULL;
    if (dec && octx->src[7] && octx->src[7]->data && octx->op_params[HTP_FA_HETERO_OPP_GPU_BLOCKS] > 0 && neq1 == 1 && neq3 == 1) {
        het_gpu_blocks = (uint32_t) octx->op_params[HTP_FA_HETERO_OPP_GPU_BLOCKS];
        het_slot       = (uint32_t) octx->op_params[HTP_FA_HETERO_OPP_SLOT];
        het_nsplit     = (uint32_t) octx->op_params[HTP_FA_HETERO_OPP_GPU_NSPLIT];
        if (het_gpu_blocks < factx.n_blocks && het_nsplit > 0 && het_slot < HTP_FA_HETERO_MAX_SLOTS) {
            het_base = (uint8_t *) (uintptr_t) octx->src[7]->data;
        } else {
            het_gpu_blocks = 0;
        }
    }
    factx.dec_b_base     = het_gpu_blocks;
    factx.het_gpu_nsplit = 0;
    factx.het_parts      = NULL;
    const uint32_t n_blocks_htp = factx.n_blocks - het_gpu_blocks;

    // Cluster-selected pages: the host tagged this node (op_params[11..15]) and attached the shadow
    // buffer as src[7]. Exclusive with the hetero split in this prototype. Anything that does not
    // check out degrades to the dense kernel.
    factx.dec_nslots = 2;
    factx.cl_on      = false;
    factx.cl_inplace = false;
    factx.cl_runs    = false;
    factx.cl_flags   = 0;
    struct hvx_fa_cl_setup cls;
    if (dec && !het_base && octx->src[7] && octx->src[7]->data &&
        octx->op_params[HTP_FA_CLUSTER_OPP_MAGIC] == (int32_t) HTP_FA_CLUSTER_MAGIC &&
        neq1 == 1 && neq3 == 1 && kparams->max_bias == 0.0f && !(octx->src[4] && octx->src[4]->data)) {
        factx.cl_on = hvx_fa_cl_setup(&factx, octx, k->ne[1], nek2, k->ne[0], v->ne[0], &cls);
        if (factx.cl_on) {
            uint32_t ns = 1u << HTP_FA_CLUSTER_FLAG_NSLOTS(factx.cl_flags);
            if (ns < 2) ns = 2;
            if (ns > HTP_FA_CLUSTER_MAX_SLOTS) ns = HTP_FA_CLUSTER_MAX_SLOTS;
            factx.dec_nslots = ns;
        }
    }

    if (dec) {
        const uint32_t rows_total = neq1 * neq2 * neq3;
        // The dec thread accumulates P*V without shuffling V, so its accumulator is the
        // widened (even, odd) pair per padded V vector: two f32 vectors per V vector.
        size_vkq_acc       = factx.size_v_row_padded * 2;
        factx.size_vkq_acc = size_vkq_acc;
        factx.dec_G      = G;
        factx.dec_R      = R;
        factx.dec_n_mseg = (mask && mask->ne[2] != 1) ? R : neq1;
        factx.dec_stride_part = HVX_FA_DEC_PART_HDR + size_vkq_acc;
        factx.dec_n_split = hvx_fa_dec_pick_split(neq3 * nek2, factx.cl_on ? cls.max_total : n_blocks_htp, octx->n_threads, 8);
        const size_t lists_bytes = !factx.cl_on ? 0
                                 : factx.cl_runs ? hex_round_up((size_t) nek2 * factx.cl_max_blk * sizeof(struct htp_fa_cluster_blk), 128)
                                 : hex_round_up((size_t) nek2 * factx.cl_n_pages_max * sizeof(uint16_t), 128);
        const bool   cl_select   = factx.cl_on && factx.cl_density != 0;
        factx.cl_sel_c_stride = cl_select ? (size_t) factx.cl_sel_rows * factx.cl_cent_bytes : 0;
        factx.cl_sel_s_stride = cl_select ? (size_t) (factx.cl_sel_rows / 32) * VLEN : 0;
        factx.cl_sel_r_stride = (cl_select && factx.cl_runs) ? hex_round_up((size_t) factx.cl_n_pages_max * sizeof(struct htp_fa_cluster_run), 128) : 0;
        for (;;) {
            factx.dec_bps     = (n_blocks_htp + factx.dec_n_split - 1) / factx.dec_n_split;
            factx.dec_n_units = neq3 * nek2 * factx.dec_n_split;
            vtcm_cur = octx->ctx->vtcm_base;
            factx.spad_q = vtcm_seq_alloc(&vtcm_cur, size_q_block * R * octx->n_threads);
            factx.spad_k = vtcm_seq_alloc(&vtcm_cur, factx.size_k_block * factx.dec_nslots * octx->n_threads);
            factx.spad_v = vtcm_seq_alloc(&vtcm_cur, factx.size_v_block * factx.dec_nslots * octx->n_threads);
            factx.spad_m = vtcm_seq_alloc(&vtcm_cur, (mask ? factx.size_m_block * HVX_FA_DMA_CACHE_SIZE : 0) * octx->n_threads);
            factx.spad_a = vtcm_seq_alloc(&vtcm_cur, size_vkq_acc * R * octx->n_threads);
            factx.dec_partials = vtcm_seq_alloc(&vtcm_cur, factx.dec_stride_part * rows_total * factx.dec_n_split);
            factx.cl_sel_pages = (uint16_t *) vtcm_seq_alloc(&vtcm_cur, lists_bytes);
            factx.cl_sel_blk   = (struct htp_fa_cluster_blk *) factx.cl_sel_pages;
            factx.cl_sel_c     = vtcm_seq_alloc(&vtcm_cur, factx.cl_sel_c_stride * octx->n_threads);
            factx.cl_sel_s     = vtcm_seq_alloc(&vtcm_cur, factx.cl_sel_s_stride * octx->n_threads);
            factx.cl_sel_r     = vtcm_seq_alloc(&vtcm_cur, factx.cl_sel_r_stride * octx->n_threads);
            if ((size_t) (vtcm_cur - octx->ctx->vtcm_base) <= octx->ctx->vtcm_size) {
                break;
            }
            if (factx.dec_n_split == 1) {
                dec = false;              // even unsplit does not fit: use the row path
                vtcm_cur = octx->ctx->vtcm_base;
                break;
            }
            factx.dec_n_split = 1;        // retry once without splitting
        }
        if (dec && factx.cl_on) {
            if (cl_select && !(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
                work_queue_run(octx->ctx->work_queue, flash_attn_ext_f16_select_thread, &factx, octx->n_threads);
            }
            hvx_fa_cl_copy_lists(&factx, &cls, nek2);   // host lists (density 0) and/or the echo
        }
        factx.cl_next_unit = 0;
    }

    if (dec) {
        if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
            uint32_t het_seq = 0;
            if (het_base) {
                // Q was flushed by the batch's per-op dirty-range flush; flush again explicitly so the
                // GPU reads are never a stale-line question, then publish this op's sequence number.
                const size_t q_row_bytes = q->ne[0] * ((q->type == HTP_TYPE_F32) ? 4 : 2);
                for (uint32_t h = 0; h < neq2; ++h) {
                    hex_l2flush((uint8_t *) (uintptr_t) q->data + h * q->nb[2], q_row_bytes);
                }
                volatile uint32_t * ready = (volatile uint32_t *) (het_base + het_slot * HTP_FA_HETERO_SLOT_STRIDE);
                Q6_dcinva_A((void *) ready);
                het_seq = *ready + 1;
                *ready  = het_seq;
                qurt_mem_cache_clean((qurt_addr_t) ready, sizeof(uint32_t), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
            }

            work_queue_run(octx->ctx->work_queue, flash_attn_ext_f16_dec_thread, &factx, octx->n_threads);

            if (het_base) {
                volatile uint32_t * done = (volatile uint32_t *) (het_base + het_slot * HTP_FA_HETERO_SLOT_STRIDE + 128);
                const uint64_t t0 = HAP_perf_get_qtimer_count();
                const uint64_t timeout = (uint64_t) HTP_FA_HETERO_DONE_TIMEOUT_US * 192ull / 10ull;
                bool ok = true;
                for (;;) {
                    Q6_dcinva_A((void *) done);
                    if (*done == het_seq) break;
                    if (HAP_perf_get_qtimer_count() - t0 > timeout) { ok = false; break; }
                }
                // Feedback for the host's share controller: how long this op idled waiting for the GPU.
                {
                    volatile uint32_t * wait_w = (volatile uint32_t *) (het_base + het_slot * HTP_FA_HETERO_SLOT_STRIDE + 8);
                    *wait_w = (uint32_t) ((HAP_perf_get_qtimer_count() - t0) * 10ull / 192ull);
                    qurt_mem_cache_clean((qurt_addr_t) wait_w, sizeof(uint32_t), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
                }
                if (ok) {
                    const uint8_t * parts = het_base + HTP_FA_HETERO_PARTS_OFF + (size_t) het_slot * HTP_FA_HETERO_PART_SLOT;
                    const size_t parts_bytes = (size_t) neq1 * neq2 * neq3 * het_nsplit * factx.dec_stride_part;
                    for (size_t off = 0; off < parts_bytes; off += HEX_L2_LINE_SIZE) {
                        Q6_dcinva_A((void *) (parts + off));
                    }
                    factx.het_parts      = parts;
                    factx.het_gpu_nsplit = het_nsplit;
                } else {
                    // Late GPU: merge the HTP partials only (wrong result, no hang) and count it.
                    volatile uint32_t * st = (volatile uint32_t *) (het_base + HTP_FA_HETERO_STATUS_OFF);
                    Q6_dcinva_A((void *) st);
                    st[0] = st[0] + 1;
                    Q6_dcinva_A((void *) done);
                    st[1] = *done;      // what the DSP last saw at the done word
                    st[2] = het_seq;    // what it wanted
                    st[3] = het_slot;
                    qurt_mem_cache_clean((qurt_addr_t) st, 16, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
                    FARF(ERROR, "hetero fa: slot %u seq %u: GPU done not seen within %u us", het_slot, het_seq, (unsigned) HTP_FA_HETERO_DONE_TIMEOUT_US);
                }
            }

            work_queue_run(octx->ctx->work_queue, flash_attn_ext_f16_merge_thread, &factx, octx->n_threads);
        }
        return HTP_STATUS_OK;
    }

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

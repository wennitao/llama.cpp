#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

// Fused XAttention block-selection scoring.
//
// Replaces this chain, which the block-sparse prefill path runs once per layer:
//
//   scores = mul_mat(K, Q_reps)            [kv, N, nh]  f32
//   sm     = soft_max(scores, scale)       [kv, N, nh]  f32
//   blk    = sum_rows(reshape(sm, bs, ...))[nblk, N, nh]f32
//
// Unfused, the [kv, N, nh] intermediate is written once by the matmul, read and
// rewritten by the softmax, and read again by sum_rows -- 4x its size in DRAM
// traffic for a result that is bs-times smaller than the intermediate. Measured on
// SM8750 that chain is ~90% of the marginal cost of the scoring pass; the matmul
// itself is only ~10% (docs/backend/snapdragon/xattn-block-selection.md).
//
// Here the intermediate never leaves VTCM. Each worker owns one (head, query-rep)
// row at a time, computes the whole kv-length score vector into a scratchpad, and
// emits only the nblk block sums. DRAM traffic drops to K plus the small output.
//
// The matmul runs on HVX rather than HMX deliberately. At the shapes that matter
// (N = R*nr is 16..128 against kv rows) the pass is bound by streaming K, not by
// MACs -- a least-squares fit over R puts 84-100% of the standalone matmul's time
// in a fixed K-streaming term. HVX keeps K in its native f16 layout and skips the
// tile/transpose staging HMX would need. Moving the dot to HMX only becomes
// interesting once N is large enough to be compute-bound.

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <string.h>
#include <math.h>

#include "hex-dma.h"
#include "hmx-queue.h"
#include "hmx-utils.h"
#include "hvx-utils.h"
#include "hvx-reduce.h"
#include "hvx-fa-kernels.h"
#include "hmx-fa-kernels.h"
#include "htp-vtcm.h"
#include "work-queue.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"

struct xattn_score_context {
    struct htp_ops_context * octx;

    const uint8_t * k_data;    // f16 [hs, kv, nh]
    const uint8_t * q_data;    // f32 [hs, N,  nh]
    uint8_t *       d_data;    // f32 [nblk, N, nh]

    size_t k_nb1, k_nb2;
    size_t q_nb1, q_nb2;
    size_t d_nb1, d_nb2;

    uint32_t hs;
    uint32_t kv;
    uint32_t nh;
    uint32_t nq;               // N == R * nr, query representatives per head
    uint32_t nblk;
    uint32_t bs;

    float    scale;

    uint32_t rows_per_thread;  // partition is over nh * nq rows
    uint32_t total_rows;
};

// f32 -> f16 for one query row. hs is a multiple of VLEN_FP16 (checked by the caller),
// so this is a straight walk with no tail.
static inline void xattn_cvt_q_row_f16(__fp16 * restrict dst, const float * restrict src, uint32_t hs) {
    const HVX_Vector * restrict vsrc = (const HVX_Vector * restrict) src;

    for (uint32_t i = 0, v = 0; i < hs; i += VLEN_FP16, v += 2) {
        hvx_vec_f32_to_f16_a((uint8_t *) dst + i * SIZEOF_FP16, vsrc[v], vsrc[v + 1]);
    }
}

static void xattn_score_thread_f32(unsigned int nth, unsigned int ith, void * data) {
    const struct xattn_score_context * xctx = (const struct xattn_score_context *) data;
    struct htp_ops_context *           octx = xctx->octx;

    const uint32_t start_row = xctx->rows_per_thread * ith;
    const uint32_t end_row   = MIN(start_row + xctx->rows_per_thread, xctx->total_rows);

    if (start_row >= end_row) {
        return;
    }

    // sp holds the kv-length score vector, ep the exponentials and then the nblk
    // block sums. qp holds the f16 copy of the query row.
    float  * restrict sp = (float *)  (octx->src0_spad.data + ith * octx->src0_spad.size_per_thread);
    float  * restrict ep = (float *)  (octx->src1_spad.data + ith * octx->src1_spad.size_per_thread);
    __fp16 * restrict qp = (__fp16 *) (octx->dst_spad.data  + ith * octx->dst_spad.size_per_thread);

    const uint32_t hs   = xctx->hs;
    const uint32_t kv   = xctx->kv;
    const uint32_t nq   = xctx->nq;
    const uint32_t nblk = xctx->nblk;
    const uint32_t bs   = xctx->bs;
    const size_t   k_nb1 = xctx->k_nb1;

    uint64_t qt = HAP_perf_get_qtimer_count();

    for (uint32_t r = start_row; r < end_row; ++r) {
        const uint32_t h = r / nq;
        const uint32_t n = r - h * nq;

        const uint8_t * restrict kbase = xctx->k_data + h * xctx->k_nb2;
        const float   * restrict qrow  = (const float *) (xctx->q_data + h * xctx->q_nb2 + n * xctx->q_nb1);
        float         * restrict drow  = (float *)       (xctx->d_data + h * xctx->d_nb2 + n * xctx->d_nb1);

        xattn_cvt_q_row_f16(qp, qrow, hs);

        // 32 keys per call; the scale folds into the dot so no separate pass is needed.
        for (uint32_t j = 0; j < kv; j += VLEN_FP32) {
            hex_l2fetch(kbase + (j + VLEN_FP32) * k_nb1, k_nb1, k_nb1, VLEN_FP32);

            HVX_Vector d = hvx_dot_f16_f16_aa_rx32(qp, kbase + j * k_nb1, k_nb1, hs, xctx->scale);
            *(HVX_Vector *) (sp + j) = d;
        }

        // softmax over the full kv axis, then collapse each bs-wide block. The
        // normalizer is the sum of the block sums, so it costs nothing extra.
        const float mx = hvx_reduce_max_f32((const uint8_t *) sp, kv);

        hvx_sub_scalar_f32((uint8_t *) ep, (const uint8_t *) sp, mx, kv);
        hvx_exp_f32((uint8_t *) sp, (const uint8_t *) ep, kv, false);

        float z = 0.0f;
        for (uint32_t b = 0; b < nblk; ++b) {
            const float acc = hvx_reduce_sum_f32((const uint8_t *) (sp + b * bs), bs);
            ep[b] = acc;
            z    += acc;
        }

        const float inv_z = (z > 0.0f) ? (1.0f / z) : 1.0f;
        hvx_scale_f32((uint8_t *) drow, (const uint8_t *) ep, nblk, inv_z);
    }

    qt = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - qt);
    FARF(HIGH, "xattn-score %d/%d: hs %u kv %u nh %u nq %u bs %u nblk %u rows %u:%u usec %u\n", ith, nth, hs, kv,
         xctx->nh, nq, bs, nblk, start_row, end_row, (unsigned) qt);
}

static int xattn_score_hvx(struct htp_ops_context * octx) {
    const struct htp_tensor * k   = octx->src[0];
    const struct htp_tensor * q   = octx->src[1];
    const struct htp_tensor * dst = octx->dst;

    if (k->type != HTP_TYPE_F16 || q->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t hs = k->ne[0];
    const uint32_t kv = k->ne[1];
    const uint32_t nh = k->ne[2];
    const uint32_t nq = q->ne[1];

    // dst is the SUM_ROWS output of the fused chain, so it is [1, nblk, nq, nh]:
    // the reduced axis survives as a degenerate ne[0]. Head and query-rep strides
    // therefore come from nb[3] and nb[2], not nb[2] and nb[1].
    const uint32_t nblk = dst->ne[1];

    if (dst->ne[0] != 1 || dst->ne[2] != nq || dst->ne[3] != nh) {
        return HTP_STATUS_NO_SUPPORT;
    }

    // Last-resort guard, and it must stay unreachable. try_fuse_xattn_score is a
    // superset of these checks, and it has to be: a fused op has no supports_op
    // gate, and returning HTP_STATUS_NO_SUPPORT here would abort every remaining op
    // in the batch rather than just this one (main.c stops the loop on the first
    // non-OK status). Failing loudly still beats computing garbage, so the check
    // stays -- but any shape that trips it is a host-side gating bug.
    if ((hs % VLEN_FP16) || (kv % VLEN_FP32) || (nblk == 0) || (kv % nblk) ||
        (k->nb[1] & (VLEN - 1)) || !hex_is_aligned((void *) k->data, VLEN) ||
        !hex_is_aligned((void *) q->data, VLEN)) {
        FARF(ERROR, "xattn-score: shape not gated by the host: hs %u kv %u nblk %u k.nb1 %u\n", hs, kv, nblk,
             (unsigned) k->nb[1]);
        return HTP_STATUS_NO_SUPPORT;
    }

    // Fused nodes carry no ggml op_params of their own -- the host stages the
    // softmax temperature into kernel_params instead.
    float scale;
    memcpy(&scale, &octx->kernel_params[0], sizeof(float));

    const uint32_t total_rows = nh * nq;
    const uint32_t n_threads  = MIN(octx->n_threads, total_rows);

    // Two kv-length f32 scratchpads plus one f16 query row, per thread.
    const size_t score_size = hex_round_up(kv * SIZEOF_FP32, 128);

    octx->src0_spad.size_per_thread = score_size;
    octx->src1_spad.size_per_thread = score_size;
    octx->dst_spad.size_per_thread  = hex_round_up(hs * SIZEOF_FP16, 128);

    octx->src0_spad.size = octx->src0_spad.size_per_thread * n_threads;
    octx->src1_spad.size = octx->src1_spad.size_per_thread * n_threads;
    octx->dst_spad.size  = octx->dst_spad.size_per_thread  * n_threads;

    const size_t spad_size = octx->src0_spad.size + octx->src1_spad.size + octx->dst_spad.size;

    FARF(HIGH, "xattn-score: %ux%ux%ux%u x %ux%ux%ux%u -> %ux%ux%ux%u : spad %zu\n", k->ne[0], k->ne[1], k->ne[2],
         k->ne[3], q->ne[0], q->ne[1], q->ne[2], q->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], spad_size);

    if (octx->ctx->vtcm_size < spad_size) {
        FARF(ERROR, "xattn-score : current VTCM reservation %zu is too small, needed %zu\n", octx->ctx->vtcm_size,
             spad_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;                        octx->src0_spad.src = NULL;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size; octx->src1_spad.src = NULL;
    octx->dst_spad.data  = octx->src1_spad.data + octx->src1_spad.size; octx->dst_spad.src  = NULL;

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    struct xattn_score_context xctx = {
        .octx            = octx,
        .k_data          = (const uint8_t *) k->data,
        .q_data          = (const uint8_t *) q->data,
        .d_data          = (uint8_t *) dst->data,
        .k_nb1           = k->nb[1],
        .k_nb2           = k->nb[2],
        .q_nb1           = q->nb[1],
        .q_nb2           = q->nb[2],
        .d_nb1           = dst->nb[2],
        .d_nb2           = dst->nb[3],
        .hs              = hs,
        .kv              = kv,
        .nh              = nh,
        .nq              = nq,
        .nblk            = nblk,
        .bs              = kv / nblk,
        .scale           = scale,
        .rows_per_thread = (total_rows + n_threads - 1) / n_threads,
        .total_rows      = total_rows,
    };

    worker_pool_run_func(octx->ctx->worker_pool, xattn_score_thread_f32, &xctx, n_threads);

    return HTP_STATUS_OK;
}

// ============================================================================
// HMX path: tuned matmul + fused epilogue
//
// The HVX version above fuses correctly but replaces the HMX matmul with HVX dot
// products, and at hs=128 a dot is two f16 vectors whose horizontal reduction costs
// more than the two multiply-accumulates it reduces -- 42.8 GFLOP/s against HMX's
// 1.55 TFLOP/s. Trading a 36x slower matmul to remove a chain worth 41-76% of the
// pass loses. So this path keeps the matmul on HMX exactly as flash attention drives
// it, and fuses only the epilogue: S lands in VTCM as fp16 tiles and the softmax and
// block-sum run over it in place, so the [kv, nq, nh] intermediate never reaches DRAM.
//
// Operand orientation follows FA's QK^T: Q is the activation (output rows = query
// reps), K is the weight (output columns = key index), S = [query row][key].
// ============================================================================

struct xattn_hmx_layout {
    size_t off_scales;
    size_t off_q_tiles;
    size_t off_k_tiles;
    size_t off_s_tiles;
    size_t off_score;    // per-thread, kv f32
    size_t off_exp;      // per-thread, kv f32 (also holds the nblk block sums)
    size_t score_stride;
    size_t total_bytes;
};

static void xattn_hmx_layout_build(struct xattn_hmx_layout * L, uint32_t hs, uint32_t kv, uint32_t m_chunk,
                                   uint32_t n_threads) {
    // Every mxmem operand must be 2048-aligned. Nothing asserts that anywhere, so the
    // ordering below is the enforcement: all four HMX regions come first, each sized
    // to a 2048 multiple, so each one starts aligned by construction.
    size_t off = 0;
    L->off_scales  = off; off += 2048;
    L->off_q_tiles = off; off += hex_align_up((size_t) m_chunk * hs * SIZEOF_FP16, 2048);
    L->off_k_tiles = off; off += hex_align_up((size_t) kv * hs * SIZEOF_FP16, 2048);
    L->off_s_tiles = off; off += hex_align_up((size_t) m_chunk * kv * SIZEOF_FP16, 2048);

    L->score_stride = hex_round_up((size_t) kv * SIZEOF_FP32, 128);
    L->off_score = off; off += L->score_stride * n_threads;
    L->off_exp   = off; off += L->score_stride * n_threads;
    L->total_bytes = off;
}

// --- K: DRAM f16 row-major -> transposed HMX weight tiles -------------------

typedef struct {
    __fp16 *       k_tiles;
    const __fp16 * k_src;
    size_t         src_stride;
    uint32_t       kv;
    uint32_t       hs;
    uint32_t       rows_per_t;
} xattn_k_args_t;

static void xattn_k_interleave_thread(unsigned int n, unsigned int i, void * data) {
    xattn_k_args_t * a = (xattn_k_args_t *) data;

    const uint32_t start = i * a->rows_per_t;
    const uint32_t end   = (uint32_t) hex_smin(start + a->rows_per_t, a->kv);
    if (start >= a->kv) {
        return;
    }

    // Read K straight from DRAM. FA stages through a DMA buffer because it needs the
    // descriptor pipeline for prefetch overlap across KV chunks; here K is walked once
    // per head, so the extra VTCM crossing would buy nothing.
    hmx_interleave_rows_to_tiles(a->k_tiles, a->k_src, a->kv, a->hs, a->src_stride, start, end);
}

// --- Q: DRAM f32 rows -> HMX activation tiles -------------------------------

typedef struct {
    __fp16 *        q_tiles;
    const uint8_t * q_src;
    size_t          q_nb1;
    uint32_t        m_rows;      // valid rows in this chunk
    uint32_t        m_padded;    // rows to write, rounded to a tile
    uint32_t        hs;
    uint32_t        rows_per_t;
} xattn_q_args_t;

static void xattn_q_prep_thread(unsigned int n, unsigned int i, void * data) {
    xattn_q_args_t * a = (xattn_q_args_t *) data;

    const uint32_t start = i * a->rows_per_t;
    const uint32_t end   = (uint32_t) hex_smin(start + a->rows_per_t, a->m_padded);
    if (start >= a->m_padded) {
        return;
    }

    const size_t n_dot = a->hs / HMX_FP16_TILE_N_COLS;

    // Rows are written in pairs because one 128-byte tile vector holds two rows
    // interleaved; rows_per_t is rounded to 2 so a thread never starts mid-pair.
    for (uint32_t r = start; r < end; r += 2) {
        const size_t r0 = r / HMX_FP16_TILE_N_ROWS;
        const size_t r1 = r % HMX_FP16_TILE_N_ROWS;
        __fp16 *     out_base = a->q_tiles + r0 * n_dot * HMX_FP16_TILE_N_ELMS;

        const HVX_Vector * p0 = (r < a->m_rows)     ? (const HVX_Vector *) (a->q_src + (size_t) r * a->q_nb1)       : NULL;
        const HVX_Vector * p1 = (r + 1 < a->m_rows) ? (const HVX_Vector *) (a->q_src + (size_t) (r + 1) * a->q_nb1) : NULL;

        // Padded rows are zeroed, not skipped: HMX would otherwise multiply stale tile
        // bytes and the softmax over that row would look plausible while being garbage.
        for (size_t d = 0; d < n_dot; ++d) {
            HVX_Vector v0 = p0 ? p0[d] : Q6_V_vzero();
            HVX_Vector v1 = p1 ? p1[d] : Q6_V_vzero();
            ((HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = hvx_vec_f32_to_f16_shuff(v0, v1);
        }
    }
}

// --- the HMX job (byte-for-byte FA's QK^T worker) ---------------------------

typedef struct {
    const __fp16 * q_tiles;
    const __fp16 * k_tiles;
    __fp16 *       s_tiles;
    size_t         n_row_tiles;
    size_t         n_col_tiles;
    size_t         n_dot_tiles;
    uint8_t *      hmx_scales;
} xattn_qk_job_t;

// Runs on the dedicated HMX thread, which unlocks HVX while spinning and never
// re-locks -- so this must contain only HMX asm and pointer arithmetic.
static void xattn_qk_dot_worker(void * data) {
    xattn_qk_job_t * job         = (xattn_qk_job_t *) data;
    const size_t     n_row_tiles = job->n_row_tiles;
    const size_t     n_col_tiles = job->n_col_tiles;
    const size_t     n_dot_tiles = job->n_dot_tiles;

    const __fp16 * restrict q_tiles = job->q_tiles;
    const __fp16 * restrict k_tiles = job->k_tiles;
    __fp16 * restrict       s_tiles = job->s_tiles;

    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) job->hmx_scales));

    const size_t dot_stride = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    for (size_t r = 0; r < n_row_tiles; ++r) {
        const __fp16 * row_tiles = q_tiles + r * dot_stride;
        const __fp16 * col_tiles = k_tiles;
        // n_tiles_per_bc == n_col_tiles here: S is allocated exactly kv wide, so the
        // allocated-vs-actual row stride split that FA needs does not arise.
        __fp16 *       out_tile  = s_tiles + r * n_col_tiles * HMX_FP16_TILE_N_ELMS;

        for (size_t c = 0; c < n_col_tiles; ++c) {
            hmx_fa_qk_dot_tile(row_tiles, col_tiles, out_tile, n_dot_tiles);
            col_tiles += dot_stride;
            out_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

// --- fused epilogue: de-interleave S, softmax, block-sum --------------------

typedef struct {
    struct htp_ops_context * octx;
    const __fp16 * s_tiles;
    uint8_t *      dst_base;   // already offset to this head
    size_t         d_nb1;      // per query rep
    float *        score_base;
    float *        exp_base;
    size_t         score_stride;
    uint32_t       kv;
    uint32_t       nblk;
    uint32_t       bs;
    uint32_t       m_rows;
    uint32_t       rows_per_t;
} xattn_epi_args_t;

static void xattn_epilogue_thread(unsigned int n, unsigned int i, void * data) {
    xattn_epi_args_t * a = (xattn_epi_args_t *) data;

    const uint32_t start = i * a->rows_per_t;
    const uint32_t end   = (uint32_t) hex_smin(start + a->rows_per_t, a->m_rows);
    if (start >= a->m_rows) {
        return;
    }

    float * restrict sp = (float *) ((uint8_t *) a->score_base + i * a->score_stride);
    float * restrict ep = (float *) ((uint8_t *) a->exp_base   + i * a->score_stride);

    const uint32_t kv   = a->kv;
    const uint32_t nblk = a->nblk;
    const uint32_t bs   = a->bs;

    // Two rows come out of one de-interleave, so walk in pairs. rows_per_t is rounded
    // to 2 so a thread never starts on the odd half of a pair.
    for (uint32_t r = start; r < end; r += 2) {
        const uint32_t r0 = r / HMX_FP16_TILE_N_ROWS;
        const uint32_t r1 = r % HMX_FP16_TILE_N_ROWS;

        const __fp16 * s_ld = a->s_tiles + (size_t) r0 * HMX_FP16_TILE_N_ROWS * kv;

        for (uint32_t pair = 0; pair < 2; ++pair) {
            if (r + pair >= a->m_rows) {
                break;
            }

            for (uint32_t c = 0; c < kv; c += 64) {
                const uint32_t     ci = c / 64;
                const __fp16 *     dt = s_ld + (size_t) ci * HMX_FP16_TILE_N_ELMS * 2;
                const HVX_Vector * p0 = ((const HVX_Vector *) dt) + r1 / 2;
                const HVX_Vector * p1 = p0 + 16;

                HVX_VectorPair pr  = Q6_W_vdeal_VVR(*p1, *p0, -2);
                HVX_Vector     row = pair ? Q6_V_hi_W(pr) : Q6_V_lo_W(pr);

                // hvx_vec_f16_to_f32, NOT the _shuff variant: _shuff leaves the f32
                // lanes interleaved. FA only uses it where it feeds a sum and order is
                // irrelevant. Here the lanes ARE the key index, so a permutation would
                // leave the softmax total at 1.0 and silently corrupt the block sums.
                HVX_VectorPair w = hvx_vec_f16_to_f32(row);
                ((HVX_Vector *) (sp + c))[0] = Q6_V_lo_W(w);
                ((HVX_Vector *) (sp + c))[1] = Q6_V_hi_W(w);
            }

            const float mx = hvx_reduce_max_f32((const uint8_t *) sp, kv);

            hvx_sub_scalar_f32((uint8_t *) ep, (const uint8_t *) sp, mx, kv);
            hvx_exp_f32((uint8_t *) sp, (const uint8_t *) ep, kv, false);

            float z = 0.0f;
            for (uint32_t b = 0; b < nblk; ++b) {
                const float acc = hvx_reduce_sum_f32((const uint8_t *) (sp + b * bs), bs);
                ep[b] = acc;
                z    += acc;
            }

            const float inv_z = (z > 0.0f) ? (1.0f / z) : 1.0f;
            float * drow = (float *) (a->dst_base + (size_t) (r + pair) * a->d_nb1);
            hvx_scale_f32((uint8_t *) drow, (const uint8_t *) ep, nblk, inv_z);
        }
    }
}

// --- driver ----------------------------------------------------------------

// Largest 32-row chunk that fits VTCM. Returns false if even one tile row does not,
// in which case the caller falls back to the HVX path rather than failing the batch.
static bool xattn_hmx_pick_chunk(const struct htp_ops_context * octx, struct xattn_hmx_layout * L,
                                 uint32_t * m_chunk, uint32_t hs, uint32_t kv, uint32_t nq, uint32_t n_threads) {
    uint32_t m = hex_align_up(nq, HMX_FP16_TILE_N_ROWS);

    for (;;) {
        xattn_hmx_layout_build(L, hs, kv, m, n_threads);
        if (L->total_bytes <= octx->ctx->vtcm_size) {
            *m_chunk = m;
            return true;
        }
        if (m <= HMX_FP16_TILE_N_ROWS) {
            return false;
        }
        m = hex_align_up(m / 2, HMX_FP16_TILE_N_ROWS);
    }
}

static int xattn_score_hmx(struct htp_ops_context * octx, const struct xattn_hmx_layout * L, uint32_t m_chunk,
                           uint32_t n_threads) {
    const struct htp_tensor * k   = octx->src[0];
    const struct htp_tensor * q   = octx->src[1];
    const struct htp_tensor * dst = octx->dst;

    const uint32_t hs   = k->ne[0];
    const uint32_t kv   = k->ne[1];
    const uint32_t nh   = k->ne[2];
    const uint32_t nq   = q->ne[1];
    const uint32_t nblk = dst->ne[1];
    const uint32_t bs   = kv / nblk;

    float scale;
    memcpy(&scale, &octx->kernel_params[0], sizeof(float));

    uint8_t * const vb = octx->ctx->vtcm_base;

    uint8_t * hmx_scales = vb + L->off_scales;
    __fp16 *  q_tiles    = (__fp16 *) (vb + L->off_q_tiles);
    __fp16 *  k_tiles    = (__fp16 *) (vb + L->off_k_tiles);
    __fp16 *  s_tiles    = (__fp16 *) (vb + L->off_s_tiles);
    float *   score_base = (float *)  (vb + L->off_score);
    float *   exp_base   = (float *)  (vb + L->off_exp);

    // The softmax temperature rides the HMX output stage as a per-column scale, so no
    // separate pass over S is needed. Plain `scale`, not scale*log2(e): the epilogue
    // uses hvx_exp_f32, which is base e.
    hmx_init_column_scales(hmx_scales, hvx_vec_splat_f16((__fp16) scale));

    work_queue_t wq   = octx->ctx->work_queue;
    hmx_queue_t  hmxq = octx->ctx->hmx_queue;

    const size_t n_dot_tiles = hs / HMX_FP16_TILE_N_COLS;
    const size_t n_col_tiles = kv / HMX_FP16_TILE_N_COLS;

    uint64_t qt = HAP_perf_get_qtimer_count();

    for (uint32_t h = 0; h < nh; ++h) {
        xattn_k_args_t ka = {
            .k_tiles    = k_tiles,
            .k_src      = (const __fp16 *) ((const uint8_t *) k->data + (size_t) h * k->nb[2]),
            // hmx_interleave_rows_to_tiles indexes its source as __fp16*, so the stride
            // is in elements, not bytes (FA divides by sizeof(__fp16) for the same reason).
            .src_stride = k->nb[1] / sizeof(__fp16),
            .kv         = kv,
            .hs         = hs,
            .rows_per_t = hex_align_up(hmx_ceil_div(kv, n_threads), 2),
        };
        work_queue_run(wq, xattn_k_interleave_thread, &ka, n_threads);

        for (uint32_t m0 = 0; m0 < nq; m0 += m_chunk) {
            const uint32_t m_rows   = (uint32_t) hex_smin(m_chunk, nq - m0);
            const uint32_t m_padded = hex_align_up(m_rows, HMX_FP16_TILE_N_ROWS);

            xattn_q_args_t qa = {
                .q_tiles    = q_tiles,
                .q_src      = (const uint8_t *) q->data + (size_t) h * q->nb[2] + (size_t) m0 * q->nb[1],
                .q_nb1      = q->nb[1],
                .m_rows     = m_rows,
                .m_padded   = m_padded,
                .hs         = hs,
                .rows_per_t = hex_align_up(hmx_ceil_div(m_padded, n_threads), 2),
            };
            work_queue_run(wq, xattn_q_prep_thread, &qa, n_threads);

            xattn_qk_job_t job = {
                .q_tiles     = q_tiles,
                .k_tiles     = k_tiles,
                .s_tiles     = s_tiles,
                .n_row_tiles = m_padded / HMX_FP16_TILE_N_ROWS,
                .n_col_tiles = n_col_tiles,
                .n_dot_tiles = n_dot_tiles,
                .hmx_scales  = hmx_scales,
            };
            if (!hmx_queue_push(hmxq, hmx_queue_make_desc(xattn_qk_dot_worker, &job))) {
                FARF(ERROR, "xattn-score: hmx queue full\n");
                return HTP_STATUS_INTERNAL_ERR;
            }
            hmx_queue_pop(hmxq);

            xattn_epi_args_t ea = {
                .octx         = octx,
                .s_tiles      = s_tiles,
                .dst_base     = (uint8_t *) dst->data + (size_t) h * dst->nb[3] + (size_t) m0 * dst->nb[2],
                .d_nb1        = dst->nb[2],
                .score_base   = score_base,
                .exp_base     = exp_base,
                .score_stride = L->score_stride,
                .kv           = kv,
                .nblk         = nblk,
                .bs           = bs,
                .m_rows       = m_rows,
                .rows_per_t   = hex_align_up(hmx_ceil_div(m_rows, n_threads), 2),
            };
            work_queue_run(wq, xattn_epilogue_thread, &ea, n_threads);
        }
    }

    // Push/pop must balance: idx_pop is global to the queue, so an op that leaves one
    // outstanding makes the NEXT op's first pop return while a job is still running.
    if (!hmx_queue_empty(hmxq)) {
        FARF(ERROR, "xattn-score: hmx queue not drained\n");
    }

    qt = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - qt);
    FARF(HIGH, "xattn-score-hmx: hs %u kv %u nh %u nq %u nblk %u m_chunk %u vtcm %zu usec %u\n", hs, kv, nh, nq, nblk,
         m_chunk, L->total_bytes, (unsigned) qt);

    return HTP_STATUS_OK;
}

int op_xattn_score(struct htp_ops_context * octx) {
    const struct htp_tensor * k   = octx->src[0];
    const struct htp_tensor * q   = octx->src[1];
    const struct htp_tensor * dst = octx->dst;

    if (k->type != HTP_TYPE_F16 || q->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        return xattn_score_hvx(octx);  // reports the real error there
    }

    const uint32_t hs = k->ne[0];
    const uint32_t kv = k->ne[1];
    const uint32_t nq = q->ne[1];

    // The epilogue de-interleaves 64 key columns at a time and the tile grid needs a
    // whole number of 32-row/32-column tiles. Anything else falls back rather than
    // failing: HTP_STATUS_NO_SUPPORT from a fused op aborts the whole batch.
    const bool hmx_ok = octx->ctx->hmx_enabled && octx->ctx->hmx_queue != NULL && octx->ctx->work_queue != NULL &&
                        (hs % HMX_FP16_TILE_N_COLS) == 0 && (kv % 64) == 0 && hs >= HMX_FP16_TILE_N_COLS;

    if (hmx_ok) {
        const uint32_t n_threads = MIN(octx->n_threads, HTP_MAX_NTHREADS);

        struct xattn_hmx_layout L;
        uint32_t                m_chunk = 0;
        if (xattn_hmx_pick_chunk(octx, &L, &m_chunk, hs, kv, nq, n_threads)) {
            if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
                return HTP_STATUS_OK;
            }
            return xattn_score_hmx(octx, &L, m_chunk, n_threads);
        }
    }

    return xattn_score_hvx(octx);
}

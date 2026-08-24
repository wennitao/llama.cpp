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
#include "hvx-utils.h"
#include "hvx-fa-kernels.h"

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

int op_xattn_score(struct htp_ops_context * octx) {
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

    // The HVX path walks 32 keys and 64 features at a time with aligned loads.
    if ((hs % VLEN_FP16) || (kv % VLEN_FP32) || (nblk == 0) || (kv % nblk) ||
        (k->nb[1] & (VLEN - 1)) || !hex_is_aligned((void *) k->data, VLEN)) {
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

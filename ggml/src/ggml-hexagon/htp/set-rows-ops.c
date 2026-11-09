#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "dma-queue.h"
#include "work-queue.h"
#include "hvx-utils.h"
#include "hex-utils.h"
#include "hvx-copy.h"
#include "hvx-quant.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "hex-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp/set-rows-ops.h"

#define set_rows_preamble                      \
    const uint32_t ne00 = octx->src[0]->ne[0]; \
    const uint32_t ne01 = octx->src[0]->ne[1]; \
    const uint32_t ne02 = octx->src[0]->ne[2]; \
    const uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    const uint32_t ne10 = octx->src[1]->ne[0]; \
    const uint32_t ne11 = octx->src[1]->ne[1]; \
    const uint32_t ne12 = octx->src[1]->ne[2]; \
    const uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    const uint32_t nb01 = octx->src[0]->nb[1]; \
    const uint32_t nb02 = octx->src[0]->nb[2]; \
    const uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    const uint32_t nb10 = octx->src[1]->nb[0]; \
    const uint32_t nb11 = octx->src[1]->nb[1]; \
    const uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    const uint32_t nb1 = octx->dst->nb[1];     \
    const uint32_t nb2 = octx->dst->nb[2];     \
    const uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    const uint32_t ne0 = octx->dst->ne[0];     \
    const uint32_t ne1 = octx->dst->ne[1];     \
    const uint32_t ne2 = octx->dst->ne[2];     \
    const uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    const uint32_t nr  = ne01;

struct set_rows_context {
    struct htp_ops_context * octx;
    const struct htp_set_rows_kernel_params * kparams;
    struct htp_set_rows_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;
    uint32_t task_start;
    uint32_t tasks;
    uint32_t tasks_per_thread;
};

#define SET_ROWS_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_EXPR)                                                \
static void set_rows_thread_dma_##TYPE_NAME##_##IDX_TYPE(unsigned int nth, unsigned int ith, void *data) {       \
    struct set_rows_context * srctx = (struct set_rows_context *)data;                                           \
    struct htp_ops_context * octx = srctx->octx;                                                                 \
    const struct htp_set_rows_kernel_params * kparams = srctx->kparams;                                          \
    set_rows_preamble;                                                                                           \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    const uint32_t dr  = srctx->tasks_per_thread;                                                                \
    const uint32_t ir0 = srctx->task_start + dr * ith;                                                           \
    if (ir0 >= srctx->task_start + srctx->tasks) {                                                               \
        return;                                                                                                  \
    }                                                                                                            \
    const uint32_t ir1 = MIN(ir0 + dr, srctx->task_start + srctx->tasks);                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                     \
    const struct htp_set_rows_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, srctx->vtcm_base, vtcm_layout->off_src0) +                    \
                          ith * vtcm_layout->src0_bytes_per_thread;                                              \
    uint8_t * vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, srctx->vtcm_base, vtcm_layout->off_dst) +                     \
                          ith * vtcm_layout->dst_bytes_per_thread;                                               \
    const uint32_t packed_channels = kparams->packed_channels;                                                     \
    const uint32_t copy_ne00 = ne00 * packed_channels;                                                          \
    const uint32_t channel_groups = ne02 / packed_channels;                                                    \
    const uint32_t dst_row_size  = htp_tensor_get_row_size(octx->dst->type, copy_ne00);                               \
    const uint32_t nrows_per_thread = ir1 - ir0;                                                                 \
    const uint32_t total_steps = ne03 * channel_groups * nrows_per_thread;                                                 \
    uint32_t pi_step = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    for (uint32_t step = 0, spad_idx = 0; step < total_steps && spad_idx < 2; ++step, spad_idx++) {              \
        uint32_t i = ir0 + pi_step;                                                                              \
        const dma_addr_t src0_data = octx->src[0]->data + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_queue_push(dma_q,                                                                                    \
                       dma_make_data(octx->dst->data,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_size),                     \
                       dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 0);                          \
        dma_queue_push(dma_q,                                                                                    \
                       dma_make_data(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_size,                    \
                                     src0_data),                                                                 \
                       ne00 * sizeof(float), nb02, ne00 * sizeof(float), packed_channels);                       \
        pi_step++;                                                                                               \
        if (pi_step == nrows_per_thread) {                                                                       \
            pi_step = 0;                                                                                         \
            pi02++;                                                                                              \
            if (pi02 == channel_groups) {                                                                                  \
                pi02 = 0;                                                                                        \
                pi03++;                                                                                          \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    uint32_t ci_step = 0;                                                                                        \
    uint32_t ci02 = 0;                                                                                           \
    uint32_t ci03 = 0;                                                                                           \
    uint32_t ci11_base = 0;                                                                                      \
    uint32_t ci12_base = 0;                                                                                      \
    for (uint32_t step = 0; step < total_steps; ++step) {                                                        \
        void * dst_spad = (void *) dma_queue_pop(dma_q).src;                                                     \
        void * src_spad = (void *) dma_queue_pop(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_step;                                                                              \
        const uintptr_t src1_addr = octx->src[1]->data + i*nb10 + ci11_base*nb11 + ci12_base*nb12;               \
        const IDX_TYPE i1 = *(const IDX_TYPE *)src1_addr;                                                        \
        const bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        const uint32_t target_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                 \
        if (valid_i1) {                                                                                          \
            COMPUTE_EXPR;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                  \
        if (valid_i1) {                                                                                          \
            const dma_addr_t dst_data = octx->dst->data + target_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_queue_push(dma_q,                                                                                \
                           dma_make_data(dst_data, dst_spad),                                                    \
                           dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 1);                      \
        } else {                                                                                                 \
            dma_queue_push(dma_q,                                                                                \
                           dma_make_data(octx->dst->data, dst_spad),                                             \
                           dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 0);                      \
        }                                                                                                        \
        const uint32_t next_step = step + 2;                                                                     \
        if (next_step < total_steps) {                                                                           \
            uint32_t ni = ir0 + pi_step;                                                                         \
            const dma_addr_t psrc0_data = octx->src[0]->data + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_queue_push(dma_q,                                                                                \
                           dma_make_data(src_spad, psrc0_data),                                                  \
                           ne00 * sizeof(float), nb02, ne00 * sizeof(float), packed_channels);                   \
            pi_step++;                                                                                           \
            if (pi_step == nrows_per_thread) {                                                                   \
                pi_step = 0;                                                                                     \
                pi02++;                                                                                          \
                if (pi02 == channel_groups) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_step++;                                                                                               \
        if (ci_step == nrows_per_thread) {                                                                       \
            ci_step = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_base++;                                                                                         \
            if (ci11_base == ne11) {                                                                             \
                ci11_base = 0;                                                                                   \
            }                                                                                                    \
            if (ci02 == channel_groups) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_base++;                                                                                     \
                if (ci12_base == ne12) {                                                                         \
                    ci12_base = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_queue_flush(dma_q);                                                                                      \
}

SET_ROWS_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, copy_ne00); })
SET_ROWS_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, copy_ne00); })

SET_ROWS_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, copy_ne00); })
SET_ROWS_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (const uint8_t *)src_spad, copy_ne00); })

SET_ROWS_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_row_q8_0_f32(dst_spad, (const float *)src_spad, copy_ne00); })
SET_ROWS_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_row_q8_0_f32(dst_spad, (const float *)src_spad, copy_ne00); })

struct set_rows_win_context {
    struct htp_ops_context * octx;
    struct htp_set_rows_win_layout layout;
    uint8_t * vtcm_base;
    uint32_t ch_per_thread;
};

// Gather lane l reads column l % ne00 of channel l / ne00
static inline HVX_Vector set_rows_win_offsets(uint32_t ne00, uint32_t nb02) {
    int32_t offsets[32] __attribute__((aligned(128)));
    for (uint32_t l = 0; l < 32; l++) {
        offsets[l] = (l / ne00) * nb02 + (l % ne00) * sizeof(float);
    }
    return *(HVX_Vector *) offsets;
}

// f32 windows path: the src0 rows of a channel overlap, so each thread loads the span of its channels once,
// builds each window of 32 / ne00 channels with one VTCM gather, and stores its part of the dst rows with large DMAs
static void set_rows_thread_win_f32(unsigned int nth, unsigned int ith, void * data) {
    const struct set_rows_win_context * wctx = (const struct set_rows_win_context *) data;
    struct htp_ops_context * octx = wctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t ne00 = src0->ne[0];
    const uint32_t nw   = src0->ne[1];
    const uint32_t nc   = src0->ne[2];
    const uint32_t nb01 = src0->nb[1];
    const uint32_t nb02 = src0->nb[2];

    const uint32_t c0 = ith * wctx->ch_per_thread;
    if (c0 >= nc) {
        return;
    }
    const uint32_t nch        = MIN(wctx->ch_per_thread, nc - c0);
    const uint32_t ch_per_vec = 32 / ne00;
    const uint32_t row_size   = nch * ne00 * sizeof(float);
    const uint32_t row_stride = wctx->layout.dst_row_stride;

    dma_queue * dma_q = octx->ctx->dma[ith];
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    uint8_t * vtcm_src = wctx->vtcm_base + ith * wctx->layout.bytes_per_thread;
    uint8_t * vtcm_dst = vtcm_src + wctx->layout.src_size;

    // load only the bytes that the windows use, never past the end of src0
    const uint32_t src_size = (nch - 1) * nb02 + (nw - 1) * nb01 + ne00 * sizeof(float);
    dma_queue_push(dma_q, dma_make_data(vtcm_src, src0->data + c0 * nb02), src_size, src_size, src_size, 1);

    uint32_t slot[HTP_SET_ROWS_WIN_MAX_ROWS];
    uint32_t s0 = UINT32_MAX;
    for (uint32_t w = 0; w < nw; w++) {
        const uint8_t * p  = (const uint8_t *) src1->data + w * src1->nb[0];
        const int64_t   i1 = src1->type == HTP_TYPE_I32 ? *(const int32_t *) p : *(const int64_t *) p;
        slot[w] = (uint64_t) i1 < (uint64_t) dst->ne[1] ? (uint32_t) i1 : UINT32_MAX;
        s0      = MIN(s0, slot[w]);
    }

    // when the slots are a permutation of s0 .. s0 + nw - 1, stage window w at row slot - s0 and store all rows with one 2D DMA
    bool     contig = true;
    uint64_t seen   = 0;
    for (uint32_t w = 0; w < nw && contig; w++) {
        const uint32_t r = slot[w] - s0;
        if (slot[w] == UINT32_MAX || r >= nw || ((seen >> r) & 1)) {
            contig = false;
        } else {
            seen |= (uint64_t) 1 << r;
        }
    }

    const HVX_Vector offsets = set_rows_win_offsets(ne00, nb02);

    dma_queue_pop(dma_q);

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    for (uint32_t w = 0; w < nw; w++) {
        uint8_t *       out = vtcm_dst + (contig ? slot[w] - s0 : w) * row_stride;
        const uint8_t * in  = vtcm_src + w * nb01;
        for (uint32_t c = 0; c < nch; c += ch_per_vec) {
            const uint32_t mu = (MIN(ch_per_vec, nch - c) - 1) * nb02 + ne00 * sizeof(float) - 1;
            Q6_vgather_ARMVw((HVX_Vector *) out, (size_t) (in + c * nb02), mu, offsets);
            out += VLEN;
        }
    }
    hvx_gather_sync(vtcm_dst);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    const dma_addr_t dst_c0 = dst->data + c0 * dst->nb[2];
    if (contig) {
        dma_queue_push(dma_q, dma_make_data(dst_c0 + s0 * dst->nb[1], vtcm_dst), dst->nb[1], row_stride, row_size, nw);
    } else {
        // skip invalid slots; of windows with the same slot the last one wins
        for (uint32_t w = 0; w < nw; w++) {
            bool last = slot[w] != UINT32_MAX;
            for (uint32_t k = w + 1; k < nw && last; k++) {
                last = slot[k] != slot[w];
            }
            if (last) {
                dma_queue_push(dma_q, dma_make_data(dst_c0 + slot[w] * dst->nb[1], vtcm_dst + w * row_stride), row_size, row_size, row_size, 1);
            }
        }
    }
    dma_queue_flush(dma_q);
}

int op_set_rows(struct htp_ops_context * octx) {
    const struct htp_set_rows_kernel_params * kparams = (const struct htp_set_rows_kernel_params *)octx->kernel_params;
    set_rows_preamble;

    if (htp_tensor_is_extended(octx->src[1])) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_tensor * dst = octx->dst;
    const uint32_t total_tasks    = kparams->total_tasks;

    uint32_t task_start = 0;
    uint32_t tasks      = total_tasks;

    if (octx->ctx->mdev.count > 1) {
        const bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_LINE_SIZE - 1)) == 0 && !htp_tensor_is_permuted(dst);
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_tasks, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        task_start = range.start;
        tasks      = range.count;
    }

    if (tasks == 0) {
        return HTP_STATUS_OK;
    }

    if (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t n_threads = octx->n_threads;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_block((const void *)octx->src[1]->data, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    if (kparams->win) {
        struct set_rows_win_context wctx;
        wctx.octx          = octx;
        wctx.vtcm_base     = (uint8_t *)octx->ctx->vtcm_base;
        wctx.ch_per_thread = kparams->win_ch_per_thread;
        htp_set_rows_win_layout_build(&wctx.layout, ne00, ne01, nb01, nb02, wctx.ch_per_thread, n_threads);
        work_queue_run(octx->ctx->work_queue, set_rows_thread_win_f32, &wctx, n_threads);
        return HTP_STATUS_OK;
    }

    struct set_rows_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_start = task_start;
    srctx.tasks = tasks;
    srctx.tasks_per_thread = fastdiv(tasks + n_threads - 1, &octx->n_threads_div);

    htp_set_rows_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00 * kparams->packed_channels, n_threads);
    srctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;

    work_queue_func_t q_func = NULL;
    const bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    switch (octx->dst->type) {
        case HTP_TYPE_F32:  q_func = is_i32 ? set_rows_thread_dma_f32_int32_t  : set_rows_thread_dma_f32_int64_t;  break;
        case HTP_TYPE_F16:  q_func = is_i32 ? set_rows_thread_dma_f16_int32_t  : set_rows_thread_dma_f16_int64_t;  break;
        case HTP_TYPE_Q8_0: q_func = is_i32 ? set_rows_thread_dma_q8_0_int32_t : set_rows_thread_dma_q8_0_int64_t; break;
        default:            return HTP_STATUS_NO_SUPPORT;
    }

    FARF(HIGH, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_bytes_per_thread * n_threads,
         srctx.vtcm_layout.dst_bytes_per_thread  * n_threads,
         n_threads);

    work_queue_run(octx->ctx->work_queue, q_func, &srctx, n_threads);

    return HTP_STATUS_OK;
}

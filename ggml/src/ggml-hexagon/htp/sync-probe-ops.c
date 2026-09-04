// HTP <-> host/GPU flag round-trip probe. Dev-only: the host gates it behind GGML_HEXAGON_SYNC_PROBE=1.
//
// The op runs inside a normal op batch. It publishes a 'ready' word in src[0], then spin-polls a
// 'done' word that another agent (CPU or GPU) writes into the same rpcmem buffer, checks a payload
// that agent wrote, and records qtimer stamps of each step into dst. All stamps are on the DSP
// qtimer, so the round trip (T_DONE - T_READY) needs no cross-domain clock offset.
//
// See enum htp_sync_probe_param / htp_sync_probe_rec in htp-ops.h for the layouts.

#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <qurt.h>
#include <qurt_memory.h>
#include <stdint.h>
#include <string.h>

#include "hex-utils.h"
#include "hex-dma.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"

static inline uint64_t sync_probe_us_to_ticks(uint32_t us) {
    return (uint64_t) us * 192ull / 10ull;  // 19.2 MHz qtimer
}

// Read the done word through the requested path. Returns the value read.
static inline uint32_t sync_probe_read_done(uint32_t inval, volatile uint32_t * done, dma_queue * q, uint8_t * stage) {
    switch (inval) {
        case 0:
            Q6_dcinva_A((void *) done);
            return *done;
        case 1:
            qurt_mem_cache_clean((qurt_addr_t) done, sizeof(uint32_t), QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
            return *done;
        case 3:
            // DMA the line into VTCM. ctx->dma is the nocache alias, so the DDR side bypasses L2.
            dma_queue_push_single_1d(q, dma_make_ptr(stage, (const void *) done), HEX_L2_LINE_SIZE);
            dma_queue_pop(q);
            return *(volatile uint32_t *) stage;
        default:
            return *done;
    }
}

int op_sync_probe(struct htp_ops_context * octx) {
    const uint64_t t_entry = HAP_perf_get_qtimer_count();

    const int32_t * p = octx->op_params;
    if ((uint32_t) p[HTP_SYNC_PROBE_P_MAGIC] != HTP_SYNC_PROBE_MAGIC) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;
    if (!src0 || !dst || dst->size < HTP_SYNC_PROBE_R_N * sizeof(uint64_t)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    uint8_t *  area = (uint8_t *) (uintptr_t) src0->data;
    uint64_t * rec  = (uint64_t *) (uintptr_t) dst->data;

    memset(rec, 0, HTP_SYNC_PROBE_R_N * sizeof(uint64_t));
    rec[HTP_SYNC_PROBE_R_T_ENTRY]  = t_entry;
    rec[HTP_SYNC_PROBE_R_FIRST_BAD] = ~0ull;

    if (p[HTP_SYNC_PROBE_P_MODE] == 1) {
        return HTP_STATUS_OK;  // ping: entry stamp only
    }

    volatile uint32_t * ready   = (volatile uint32_t *) (area + p[HTP_SYNC_PROBE_P_READY_OFF]);
    volatile uint32_t * done    = (volatile uint32_t *) (area + p[HTP_SYNC_PROBE_P_DONE_OFF]);
    const uint32_t *    payload = (const uint32_t *)    (area + p[HTP_SYNC_PROBE_P_PAYLOAD_OFF]);

    const uint32_t n_payload     = (uint32_t) p[HTP_SYNC_PROBE_P_PAYLOAD_WORDS];
    uint32_t       inval         = (uint32_t) p[HTP_SYNC_PROBE_P_INVAL];
    const uint64_t timeout_ticks = sync_probe_us_to_ticks((uint32_t) p[HTP_SYNC_PROBE_P_TIMEOUT_US]);
    const uint32_t done_val      = (uint32_t) p[HTP_SYNC_PROBE_P_DONE_VAL];
    const uint32_t seed          = (uint32_t) p[HTP_SYNC_PROBE_P_SEED];

    // VTCM staging for the DMA read path: one line for the flag, then the payload.
    dma_queue * q     = octx->ctx->dma[0];
    uint8_t *   stage = octx->ctx->vtcm_base;
    if (inval == 3 && (!q || !stage || octx->ctx->vtcm_size < HEX_L2_LINE_SIZE + (size_t) n_payload * 4)) {
        inval = 1;
    }

    // Data the other agent will read after it sees ready (stands in for Q and the new K/V rows).
    // The batch-exit flush is too late for this too, so flush the range explicitly.
    const uint32_t dsp_n = (uint32_t) p[HTP_SYNC_PROBE_P_DSP_PAYLOAD_WORDS];
    if (dsp_n) {
        uint32_t * dp = (uint32_t *) (area + p[HTP_SYNC_PROBE_P_DSP_PAYLOAD_OFF]);
        for (uint32_t i = 0; i < dsp_n; i++) {
            dp[i] = seed ^ 0xA5A5A5A5u ^ i;
        }
        qurt_mem_cache_clean((qurt_addr_t) dp, dsp_n * 4, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    }
    rec[HTP_SYNC_PROBE_R_T_DSP_PAYLOAD] = HAP_perf_get_qtimer_count();

    // Publish ready. The batch-exit flush is too late: the other agent polls this word now.
    *ready = (uint32_t) p[HTP_SYNC_PROBE_P_READY_VAL];
    if (p[HTP_SYNC_PROBE_P_FLUSH] == 0) {
        qurt_mem_cache_clean((qurt_addr_t) ready, sizeof(uint32_t), QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    } else {
        Q6_dccleana_A((void *) ready);
        asm volatile(" syncht\n");
    }
    const uint64_t t_ready = HAP_perf_get_qtimer_count();
    rec[HTP_SYNC_PROBE_R_T_READY] = t_ready;

    // Poll done.
    uint64_t polls  = 0;
    uint32_t v      = 0;
    uint64_t status = 0;
    for (;;) {
        v = sync_probe_read_done(inval, done, q, stage);
        polls++;
        if (polls == 1) {
            rec[HTP_SYNC_PROBE_R_T_FIRST_POLL] = HAP_perf_get_qtimer_count();
            rec[HTP_SYNC_PROBE_R_FIRST_VAL]    = v;
        }
        if (v == done_val) {
            break;
        }
        if (timeout_ticks) {
            const uint64_t now = HAP_perf_get_qtimer_count();
            if (now - t_ready > timeout_ticks) {
                status = 1;
                break;
            }
        }
    }
    rec[HTP_SYNC_PROBE_R_T_DONE]   = HAP_perf_get_qtimer_count();
    rec[HTP_SYNC_PROBE_R_POLLS]    = polls;
    rec[HTP_SYNC_PROBE_R_DONE_RAW] = v;

    // Check the payload the other agent wrote.
    uint64_t mism      = 0;
    uint64_t first_bad = ~0ull;
    if (status == 0 && n_payload) {
        const uint32_t * src = payload;
        if (inval == 3) {
            uint8_t * pst = stage + HEX_L2_LINE_SIZE;
            dma_queue_push_single_1d(q, dma_make_ptr(pst, payload), (size_t) n_payload * 4);
            dma_queue_pop(q);
            src = (const uint32_t *) pst;
        } else if (inval != 2) {
            qurt_mem_cache_clean((qurt_addr_t) payload, n_payload * 4, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
        }
        for (uint32_t i = 0; i < n_payload; i++) {
            if (src[i] != (seed ^ i)) {
                if (mism == 0) {
                    first_bad = i;
                }
                mism++;
            }
        }
    }
    rec[HTP_SYNC_PROBE_R_T_CHECKED] = HAP_perf_get_qtimer_count();
    rec[HTP_SYNC_PROBE_R_MISMATCH]  = mism;
    rec[HTP_SYNC_PROBE_R_FIRST_BAD] = first_bad;
    rec[HTP_SYNC_PROBE_R_STATUS]    = status;

    FARF(HIGH, "sync-probe: inval %u polls %u status %u mism %u rt %u us", inval, (unsigned) polls, (unsigned) status,
         (unsigned) mism, (unsigned) HAP_perf_qtimer_count_to_us(rec[HTP_SYNC_PROBE_R_T_DONE] - t_ready));

    return HTP_STATUS_OK;
}

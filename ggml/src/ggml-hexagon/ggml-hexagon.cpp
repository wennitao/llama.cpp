#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <regex>
#include <queue>
#include <algorithm>
#include <random>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#       define NOMINMAX
#    endif
#    include <windows.h>
#    include <sal.h>
#else
#    include <semaphore.h>
#    include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"

#include <AEEStdErr.h>
#include <dspqueue.h>
#include <rpcmem.h>

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-hexagon.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "htp-opnode.h"
#include "htp-ops.h"
#include "htp/matmul-ops.h"
#include "htp/flash-attn-ops.h"
#include "htp/unary-ops.h"
#include "htp_iface.h"
#include "htp-drv.h"

#ifdef GGML_HEXAGON_HETERO
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <condition_variable>
#include <deque>
#include <functional>
#include <sched.h>
#endif

using intvec  = std::vector<int>;
using uintvec = std::vector<unsigned int>;
using u32vec  = std::vector<uint32_t>;

static int    opt_arch    = 0; // autodetect
static size_t opt_ndev    = 1;
static size_t opt_nhvx    = 0; // use all
static int    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
static size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
static size_t opt_mbuf    = 1ul * 1024 * 1024 * 1024; // max buffer size
static int    opt_etm     = 0;
static int    opt_verbose = 0;
static int    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
static int    opt_hostbuf = 1; // hostbuf ON by default
// Let the lm-head (src0->ne[1] > 32768) run on the HTP. Off by default: the cut below was
// 'refuse the lm-head for now'; GGML_HEXAGON_LM_HEAD=1 lifts it so the cost can be measured.
static int    opt_lm_head = 0;
// Split-KV, GQA-grouped HVX decode attention (default on); GGML_HEXAGON_FA_DECODE=0 restores
// the row-per-thread kernel for A/B.
static int    opt_fa_decode = 1;
// Dev-only: let a GGML_OP_CUSTOM node tagged with HTP_SYNC_PROBE_MAGIC run as HTP_OP_SYNC_PROBE
// (flag round-trip probe, examples/hetero-sync-probe). Off by default.
static int    opt_sync_probe = 0;
// Heterogeneous decode attention prototype: fraction of each decode FLASH_ATTN_EXT's KV range the
// GPU computes (0 = off). The HTP merge consumes the GPU's partials through a shared buffer
// (src[7]) with a flag handshake relayed by a host thread; see docs heterogeneous-npu-gpu.md 4g-4h.
static float  opt_hetero_frac = 0.0f;
static int    opt_hetero_span = 512;   // KV keys per GPU work-group
static int    opt_hetero_cpu  = 7;     // relay thread affinity (-1 = none)
// Feedback control of each layer's GPU share (GGML_HEXAGON_HETERO_ADAPT=1, default on): the HTP
// reports how long it idled waiting for the GPU; the relay moves the share one 64-key block per
// token toward a few microseconds of wait. Keeps the split useful when the GPU throttles.
static int    opt_hetero_adapt = 1;
// Heterogeneous PREFILL split (GGML_HEXAGON_FA_FOLD=1): GPU computes the exception blocks of a
// sparse prefill FLASH_ATTN_EXT, the HMX op folds the partial (see ggml_hexagon_hfold_*).
static int    opt_fa_fold     = 0;
static int    opt_fa_fold_cpu = 7;     // relay thread affinity (-1 = none)
static int    opt_fa_fold_check = 0;   // GGML_HEXAGON_FA_FOLD_CHECK=1: host-reference a few GPU partial rows per batch
static int    opt_fa_fold_svmtab = 0;  // GGML_HEXAGON_FA_FOLD_SVMTAB=1: per-op tables in fine-grain SVM instead of the alias (staleness probe)
static int    opt_fa_fold_svmq = 0;    // GGML_HEXAGON_FA_FOLD_SVMQ=1: copy Q into SVM per op (staleness probe)
static int    opt_fa_fold_sync = 0;    // GGML_HEXAGON_FA_FOLD_SYNC=1: map/unmap the GPU's inputs per op (probe; not needed on Adreno 830)
static int    opt_fa_fold_perf = 1;    // GGML_HEXAGON_FA_FOLD_PERF=0: do not pin the GPU clock (cl_qcom_perf_hint)
static int    opt_fa_fold_flush = 0;   // GGML_HEXAGON_FA_FOLD_FLUSH=1: explicit Q/K/V/mask/membership flush before ready (else the op-start flush)
static int    opt_fa_fold_stage_qb = 1; // GGML_HEXAGON_FA_FOLD_STAGE_QB=0: stage by KV head instead of query block
// Control arm: GGML_HEXAGON_FA_FOLD_KEEPALIVE=<busy us> runs a dummy GPU kernel for that long every
// GGML_HEXAGON_FA_FOLD_KEEPALIVE_PERIOD_US (default 18000) with NO attention work on the GPU, to
// separate the split's own gain from the bus-clock lift a periodically busy GPU gives the HTP.
static int    opt_fa_fold_keepalive = 0;
static int    opt_fa_fold_keepalive_period = 18000;
// GPU-side chain (GGML_HEXAGON_FA_FOLD_GPUDONE=1, default): gate -> compact -> stage_k -> fa_exc are
// enqueued as one in-order chain that needs nothing from the host after `ready` (the relay only
// mirrors ready into SVM); the membership is compacted on the GPU, and the kernel signals each
// (query block, KV head) stage itself: an ION done word plus a stamp (direct-visibility probe)
// and a fine-grain SVM word the relay copies to the DSP's done line. PREQ says when the chain is
// enqueued: 0 = at ready (a fresh launch), -1 = right after the previous op's chain, -2 = every
// chain of the graph at batch start (the in-order queue serialises them and the gates open on the
// DSP's own ready word, so a descheduled relay thread costs nothing), N > 0 = N us before the
// predicted ready (EMA of the ready-to-ready period). CHAIN=2 = one persistent fa_exc launch per
// op whose work-groups pull (stage, unit) items in stage order; 1 = one plain launch per op; 0 =
// one launch per stage.
static int    opt_fa_fold_gpudone = 1;
static int    opt_fa_fold_preq = -2;
static int    opt_fa_fold_chain = 2;
static int    opt_fa_fold_qbh = 1;         // GGML_HEXAGON_FA_FOLD_STAGE_QBH=0: stages stay whole query blocks
static int    opt_fa_fold_gate_spins = 1000000;   // ~280 ms at ~0.28 us per SVM poll
static int    opt_fa_fold_pwg = 32;        // CHAIN=2: persistent work-groups per op (each pulls stage-ordered units)
static int    opt_fa_fold_pwg_stage = 64;  // persistent work-groups for the K^T staging kernel
static int    opt_fa_fold_trace = 0;       // GGML_HEXAGON_FA_FOLD_TRACE=N: per-stage timeline of the first N jobs of every 16th batch
static int    opt_fa_fold_ion_ready = 1;   // GGML_HEXAGON_FA_FOLD_ION_READY=0: the gate polls only the SVM mirror, not the DSP's word
static int    opt_fa_fold_ktbuf = 0;       // GGML_HEXAGON_FA_FOLD_KTBUF=1: K^T through plain buffer loads/stores instead of the image
static int    opt_fa_fold_probe = 0;       // GGML_HEXAGON_FA_FOLD_PROBE=<bits>: HTP_FA_FOLD_F_PROBE_* bits OR-ed into the header flags (timing only)
// A GPU kernel that reads DSP-written memory through the alias sees stale lines of what an EARLIER
// op left at the same address unless a SUBMISSION referencing the buffer is made after the write:
// fences, atomic loads, migrations, map/unmap and idle time did not help, a one-work-item kernel
// submitted at ready did (reference check 0 of 204 rows wrong vs 13-26 before). TRIG=2 (default):
// the relay submits it on a second queue at ready and the gate opens on the relay's SVM word only.
// TRIG=3 (gate on the trigger's COMPLETION) deadlocks: the second queue does not run concurrently
// with the first on this device, the trigger lands behind the pre-enqueued chain, and every op
// times out. The submission itself is what matters (TRIG=2), not the kernel's execution.
static int    opt_fa_fold_trig = 2;        // GGML_HEXAGON_FA_FOLD_TRIG: 0 = off, 1 = submit but do not wait (probe), 2 = submit and gate on the submission, 3 = gate on its completion (deadlocks; probe)
static int    opt_fa_fold_acq = 0;         // GGML_HEXAGON_FA_FOLD_ACQ=1: acquire fence (all SVM devices) at kernel entry
static int    opt_fa_fold_qatomic = 0;     // GGML_HEXAGON_FA_FOLD_QATOMIC=1: Q through atomic loads
static int    opt_fa_fold_wgfence = 0;     // GGML_HEXAGON_FA_FOLD_WGFENCE=1: per-work-group all-SVM-devices release fence
static int    opt_fa_fold_acc16 = 1;       // GGML_HEXAGON_FA_FOLD_ACC16=0: the GPU writes the partial accumulator as f32 instead of f16 (f16 halves the fold's reads; exact within f16 rounding)
// KNAT=1 (K in its natural layout through an image, no staging, first stage ~115 us after ready)
// is 80-100 us per op faster but its perplexity did not reproduce across runs (19.6350 x2 /
// 19.7070) with the reference check clean, so it stays opt-in until that is understood.
static int    opt_fa_fold_knat = 0;        // GGML_HEXAGON_FA_FOLD_KNAT=1: read K in its natural layout through an image of the cache (no K^T staging, no tables)
static int    opt_fa_fold_preq_idle = 0;   // GGML_HEXAGON_FA_FOLD_PREQ_IDLE_US: PREQ=-3 waits this long after the previous chain before enqueuing
static int    opt_fa_fold_qmap = 0;        // GGML_HEXAGON_FA_FOLD_QMAP=1: enqueue a map/unmap of Q and the membership in the chain after the gate
static int    opt_fa_fold_gate_delay = 0;  // GGML_HEXAGON_FA_FOLD_GATE_DELAY=<polls>: diagnostic hold after ready (~30 us per poll)
static int    opt_fa_fold_notag = 0;       // GGML_HEXAGON_FA_FOLD_NOTAG=1: init the sidecar (keep-alive) but tag no node: the NPU runs the pool alone (timing probe, output lacks the exceptions)
// Cluster-selected sparse decode attention prototype (GGML_HEXAGON_CLUSTER_ATTN=<density_permille>,<window>[,flags]).
// The backend owns a cluster-ordered shadow of the KV cache (htp-ops.h, HTP_FA_CLUSTER_*) and tags decode
// FLASH_ATTN_EXT nodes so the HVX kernel attends over selected 64-key pages plus a dense recent window.
// -1 = off. Density 0 = host-written page lists (measurement mode); > 0 = on-device selection.
static int      opt_cluster_density = -1;
static int      opt_cluster_window  = 256;
static uint32_t opt_cluster_flags   = 0;
static int      opt_cluster_cpu     = -1;   // sidecar thread affinity (-1 = none)
static int      opt_cluster_verify  = 0;    // CPU-check every published chunk against the cache
static int      opt_cluster_chunk   = 1024; // keys per clustering chunk (multiple of 64)
static int      opt_cluster_positional = 0; // 1 = baseline: pages are 64 consecutive positions (no k-means), descriptor = mean
static int      opt_cluster_mem_mb   = 0;    // shadow memory budget in MB; 0 = what is left of the cDSP vmem budget after weights + KV
static int      opt_cluster_page     = 64;   // keys per shadow page: 16, 32 or 64 (GGML_HEXAGON_CLUSTER_PAGE)
static int      opt_cluster_runs     = 0;    // 1 = whole-cluster selection: variable-size clusters as runs in the shadow, sinks a forced run (GGML_HEXAGON_CLUSTER_RUNS)
static int      opt_cluster_avg      = 32;   // runs mode: average cluster size in keys (GGML_HEXAGON_CLUSTER_AVG)
static int      opt_cluster_threads  = 4;    // sidecar worker threads (layers in parallel) (GGML_HEXAGON_CLUSTER_THREADS)
static int      opt_cluster_inplace  = 0;    // 1 = pages are consecutive rows of llama's cache; the shadow holds only the page means (GGML_HEXAGON_CLUSTER_INPLACE)
static int      opt_cluster_desc_cpu = 0;    // in-place mode: 1 = the CPU sidecar computes the page means (diagnostic); 0 = the FA op does (GGML_HEXAGON_CLUSTER_DESC_CPU)
static int      opt_cluster_lag      = 0;    // rows held back from publication (diagnostic: CPU visibility of DSP-written rows) (GGML_HEXAGON_CLUSTER_LAG)
static int      opt_cluster_sink     = 0;    // sink keys (positions [0, n)): packed into page 0 by the sidecar, which the kernel always selects (GGML_HEXAGON_CLUSTER_SINK)

static int    opt_mm_select = 3; // 3 = HMX -> Tiled -> Flat -> CPU, 2 = Tiled -> Flat -> CPU, 1 = Flat -> CPU
static int    opt_fa_select = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
// Block-sparse flash attention. 1 = honour src[5], 0 = ignore it and run the op densely.
// This is the ONLY clean sparse-off A/B on a single binary: FA_SELECT cannot serve,
// because 0 and 1 disable the HMX path entirely (:2006) and sparse is HMX-only, so
// lowering it sends the whole op to CPU rather than to a dense HTP kernel. Without a
// same-binary A/B every speedup number needs two builds and stops being trustworthy.
static int    opt_fa_sparse = 1;
// KV block residency map in the sparse HMX flash-attention kernel; see enum
// htp_fa_res_mode. Default AUTO. Exposed so the feature can be A/B'd on one binary --
// without that the perf comparison needs two builds and stops being trustworthy.
// Default OFF, not AUTO. The feature is correct (36/36 across all four modes) but
// measured zero gain: it saves KV DMA bytes, and DMA bytes are not what the
// per-query-block path is bound by. See flash-attn-htp-anatomy.md. AUTO would still
// swap the loop order for nothing, so it is not a safe default.
static int    opt_fa_kv_residency = HTP_FA_RES_OFF;

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
static u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

// Enable all stages by default
static int opt_opstage  = HTP_OPSTAGE_QUEUE | HTP_OPSTAGE_COMPUTE;
static int opt_opbatch  = 1024; // max number of ops in a batch
static int opt_opqueue  = 16;   // max number of pending batches
static int opt_optrace  = 0;    // trace buffer size per thread (0 means default)
static int opt_oppoll   = 0;    // polling for batch completions
static int opt_opfusion = 1;    // enable/disable op fusion

// Async graph submission. graph_compute then only SUBMITS the batch, and the join moves
// to ggml_backend_hexagon_synchronize -- which is exactly the contract of
// ggml_backend_graph_compute_async (ggml-backend.cpp:444-453). It lets the caller run CPU
// work between the submit and the join.
// Default OFF: every recorded perf baseline was taken with graph_compute blocking, and
// deferring the join arms two hazards only the caller can rule out -- a buffer freed while
// the DSP still reads it, and (with opt_profile) a ggml_tensor freed before the deferred
// pop dereferences it. ggml_backend_sched gains nothing from this: it synchronizes on
// every backend transition anyway (ggml-backend.cpp:1611-1617).
static int opt_async    = 0;

// XAttention scoring fusion. Off by default: the fused kernel is correct but the
// HVX dot it uses for the score matmul is slower than the unfused HMX MUL_MAT at
// every shape measured, so enabling it is currently a regression. See
// docs/backend/snapdragon/xattn-block-selection.md.
static int opt_xattn_fusion = 0;

static std::regex* opt_opfilter = NULL; // regex of ops to not claim

#define HEX_VERBOSE(...) \
    if (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

static const char * status_to_str(uint32_t status) {
    switch (status) {
        case HTP_STATUS_OK:
            return "OK";
        case HTP_STATUS_NO_SUPPORT:
            return "NO-SUPPORT";
        case HTP_STATUS_INVAL_PARAMS:
            return "INVAL-PARAMS";
        case HTP_STATUS_VTCM_TOO_SMALL:
            return "VTCM-TOO-SMALL";
        case HTP_STATUS_INTERNAL_ERR:
            return "INTERNAL-ERROR";
        default:
            return "UNKNOWN";
    }
}

// ** debug helpers

static void ggml_hexagon_dump_op_exec(const std::string &sess_name, const htp_opnode & node, const uint32_t req_flags) {
    if (!opt_verbose) return;

    htp_opformat fmt(node);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                node.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

static void ggml_hexagon_dump_op_supp(const std::string &sess_name, const struct ggml_tensor * op, bool supp) {
    if (!opt_verbose) return;

    htp_opformat fmt(htp_opformat(htp_opnode{const_cast<ggml_tensor*>(op), {}, HTP_OP_INVALID}));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

static const char * htp_event_name(uint16_t id) {
    switch (id) {
        case HTP_TRACE_EVT_DMA:            return "DMA";
        case HTP_TRACE_EVT_HVX_COMP:       return "HVX_COMP";
        case HTP_TRACE_EVT_HVX_A_QUANT:    return "HVX_A_QUANT";
        case HTP_TRACE_EVT_HVX_A_PREP:     return "HVX_A_PREP";
        case HTP_TRACE_EVT_HVX_W_DEQUANT:  return "HVX_W_DEQUANT";
        case HTP_TRACE_EVT_HVX_W_PREP:     return "HVX_W_PREP";
        case HTP_TRACE_EVT_HVX_O_PROC:     return "HVX_O_PROC";
        case HTP_TRACE_EVT_HVX_FA_QK:      return "HVX_QK_FA";
        case HTP_TRACE_EVT_HVX_FA_SFM:     return "HVX_SFM_FA";
        case HTP_TRACE_EVT_HVX_FA_Q_PREP:  return "HVX_Q_PREP";
        case HTP_TRACE_EVT_HVX_FA_K_PREP:  return "HVX_K_PREP";
        case HTP_TRACE_EVT_HVX_FA_V_PREP:  return "HVX_V_PREP";
        case HTP_TRACE_EVT_HMX_COMP:       return "HMX_COMP";
        case HTP_TRACE_EVT_L2FLUSH:        return "L2FLUSH";
        case HTP_TRACE_EVT_INIT:           return "INIT";
        case HTP_TRACE_EVT_BUFF:           return "BUFF";
        default:                           return "UNKNOWN";
    }
}

static void ggml_hexagon_dump_op_prof(const std::string &sess_name, const htp_opnode & node, const htp_prof_desc & pd) {
    if (!opt_profile) return;

    uint32_t op_usec = pd.usecs;
    uint32_t op_cycles = pd.cycles_stop - pd.cycles_start;
    const uint32_t * pmu = pd.pmu;

    char pmu_str[256] = "";
    if (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, sizeof(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(node);
    float mhz = op_usec > 0 ? (float) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            node.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

static void ggml_hexagon_dump_batch_prof(const std::string & sess_name, const htp_opbatch_rsp & rsp) {
    uint64_t batch_cycles = rsp.cycles_stop - rsp.cycles_start;
    float batch_mhz = rsp.usecs > 0 ? (float) batch_cycles / rsp.usecs : 0.0f;

    char evt_str[256] = "----";
    if (opt_profile == 3) {
        snprintf(evt_str, sizeof(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (unsigned long long) batch_cycles, (unsigned long long) rsp.cycles_start, batch_mhz);
}

static void ggml_hexagon_dump_trace_events(const std::string & sess_name, const htp_opbatch_rsp & rsp,
                                           const htp_trace_desc * trace_events, uint32_t n_traces) {
    if (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        for (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        for (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            for (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                const auto & e = trace_events[t * n_traces + idx];
                bool is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.cycles);
            }
        }
    }
}

// **

static inline bool ggml_hexagon_is_repack_type(enum ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
           type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4;
}

static inline bool ggml_hexagon_is_hmx_weight_type(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || ggml_hexagon_is_repack_type(type);
}

struct ggml_hexagon_session;

static void ggml_hexagon_precompute_matmul_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
);

static void ggml_hexagon_precompute_unary_params(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_unary_kernel_params * kparams
);

static void ggml_hexagon_precompute_fused_qkv_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct htp_mm_kernel_params * kparams
);

static void ggml_hexagon_precompute_fused_ffn_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct htp_mm_kernel_params * kparams
);

// ** backend sessions

struct ggml_hexagon_opbatch;
struct ggml_hexagon_opqueue;
struct htp_opnode;
struct ggml_hexagon_hetero;
struct ggml_hexagon_hfold;
static void ggml_hexagon_hfold_free(ggml_hexagon_session * sess);
static void ggml_hexagon_hfold_buffer_freed(ggml_hexagon_session * sess, ggml_backend_buffer_t buffer);
struct ggml_hexagon_cluster;
struct ggml_hexagon_session;
#ifdef GGML_HEXAGON_HETERO
static void ggml_hexagon_hetero_free(ggml_hexagon_session * sess);
#endif
static void ggml_hexagon_cluster_free(ggml_hexagon_session * sess);
static void ggml_hexagon_cluster_buffer_freed(ggml_hexagon_session * sess, ggml_backend_buffer_t buffer);

struct ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    int              dev_id;
    bool             valid_session;
    bool             valid_handle;
    bool             valid_queue;
    bool             valid_iface;

    std::atomic<int>      op_pending;
    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    ggml_backend_buffer_type buffer_type        = {};
    ggml_backend_buffer_type repack_buffer_type = {};

    uint32_t n_threads = 0;
    uint32_t n_hvx     = 0;
    uint32_t n_hmx     = 0;
    uint64_t vtcm_size = 0;
    size_t   max_vmem  = 0;
    size_t   max_bufsize = 0;

    struct {
        uint64_t uid = 0;
        std::vector<htp_opnode> htp_nodes;
    } cached_graph;

    ggml_hexagon_hetero *  hetero  = nullptr;
    ggml_hexagon_hfold *   hfold   = nullptr;
    ggml_hexagon_cluster * cluster = nullptr;

    ggml_hexagon_session(int dev_id, ggml_backend_dev_t dev) noexcept(false);
    ~ggml_hexagon_session() noexcept(true);

    const char* c_name() const { return name.c_str(); }

    void allocate(int dev_id) noexcept(false);
    void release() noexcept(true);

    void enqueue_op(const htp_opnode & node);
    void flush(bool all = true);

    void flush_pending(bool all = false);
    void flush_batch();
};

// ** backend buffers

struct ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(const std::string & name, ggml_hexagon_session * sess) {
        this->sess = sess;
        this->name = name;
    }

    ggml_hexagon_session * sess;
    std::string            name;
};

struct ggml_hexagon_shared_buffer {
    ggml_hexagon_session * sess;
    uint8_t *              base;
    size_t                 size;
    int                    fd;
    bool                   mapped;
    bool                   pinned;

    void mmap() {
        fastrpc_map_flags flags = this->pinned ? FASTRPC_MAP_FD : FASTRPC_MAP_FD_DELAYED;

        int err = fastrpc_mmap(sess->domain_id, this->fd, (void *) this->base, 0, this->size, flags);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, this->size, this->fd, (unsigned) err);
            throw std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u\n",
                sess->c_name(), (void *) this->base, this->size, this->fd, pinned);

        this->mapped = true;
    }

    void unmap() {
        if (!this->mapped) return;

        if (!this->pinned) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, this->fd);
        }

        fastrpc_munmap(sess->domain_id, this->fd, (void *) this->base, this->size);

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (void *) this->base, size, this->fd);

        this->mapped = false;
        this->fd     = -1;
    }

    void alloc(size_t size) {
        if (this->base) return;

        this->base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        if (!this->base) {
            GGML_LOG_ERROR("ggml-hex: %s failed to allocate buffer : size %zu\n", sess->c_name(), size);
            throw std::runtime_error("ggml-hex: rpcmem_alloc failed (see log for details)");
        }

        this->fd = rpcmem_to_fd(this->base);
        if (this->fd < 0) {
            GGML_LOG_ERROR("ggml-hex: %s failed to get FD for buffer %p\n", sess->c_name(), (void *) this->base);
            throw std::runtime_error("ggml-hex: rpcmem_to_fd failed (see log for details)");
        }
        this->size = size;

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (void *) this->base, this->size, this->fd, (int) pinned);
        mmap();
    }

    void free() {
        if (!this->base) return;

        unmap();
        rpcmem_free(this->base);

        HEX_VERBOSE("ggml-hex: %s freed buffer: base %p size %zu fd %d\n", sess->c_name(),
                (void *) this->base, size, this->fd);

        this->base = NULL;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, bool pinned = false) {
        this->sess   = sess;
        this->size   = 0;
        this->base   = nullptr;
        this->fd     = -1;
        this->mapped = false;
        this->pinned = pinned;

        alloc(size);
    }

    ~ggml_hexagon_shared_buffer() {
        free();
    }
};

static ggml_hexagon_session * ggml_backend_hexagon_buffer_get_sess(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer->buft->context)->sess;
}

static void ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    {
        auto sbuf_ = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
        ggml_hexagon_cluster_buffer_freed(sbuf_->sess, buffer);
#ifdef GGML_HEXAGON_HETERO
        ggml_hexagon_hfold_buffer_freed(sbuf_->sess, buffer);
#endif
    }
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    // In async mode a batch that reads this buffer can still be in flight, and the dtor
    // unmaps it from the DSP. Join first. Nothing is ever in flight in the default
    // synchronous mode, and op_queue is already gone once the session is released.
    if (opt_async && sbuf->sess->op_queue) {
        sbuf->sess->flush();
    }
    delete sbuf;
}

static void * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    return sbuf->base;
}

static enum ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu usage %d\n", sess->c_name(),
                tensor->name, (void *) sbuf->base, tensor->data, ggml_nbytes(tensor), (int) buffer->usage);

    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        return GGML_STATUS_SUCCESS; // nothing to do for the view
    }

    return GGML_STATUS_SUCCESS;
}

// ** Repack helpers for tiled quantized weights

static void unpack_q4_0_quants(uint8_t * qs, const block_q4_0 * x, unsigned int bi) {
    static const int qk = QK4_0;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const int x0             = (x->qs[i] & 0x0F);
        const int x1             = (x->qs[i] >> 4);
        qs[bi * qk + i + 0]      = x0;
        qs[bi * qk + i + qk / 2] = x1;
    }
}

static void pack_q4_0_quants(block_q4_0 * x, const uint8_t * qs, unsigned int bi) {
    static const int qk = QK4_0;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const uint8_t x0 = qs[bi * qk + i + 0];
        const uint8_t x1 = qs[bi * qk + i + qk / 2];
        x->qs[i]         = x0 | (x1 << 4);
    }
}

static void unpack_q4_1_quants(uint8_t * qs, const block_q4_1 * x, unsigned int bi) {
    static const int qk = QK4_1;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const int x0             = (x->qs[i] & 0x0F);
        const int x1             = (x->qs[i] >> 4);
        qs[bi * qk + i + 0]      = x0;
        qs[bi * qk + i + qk / 2] = x1;
    }
}

static void pack_q4_1_quants(block_q4_1 * x, const uint8_t * qs, unsigned int bi) {
    static const int qk = QK4_1;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const uint8_t x0 = qs[bi * qk + i + 0];
        const uint8_t x1 = qs[bi * qk + i + qk / 2];
        x->qs[i]         = x0 | (x1 << 4);
    }
}

static void unpack_mxfp4_quants(uint8_t * qs, const block_mxfp4 * x, unsigned int bi) {
    static const int qk = QK_MXFP4;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const int x0             = (x->qs[i] & 0x0F);
        const int x1             = (x->qs[i] >> 4);
        qs[bi * qk + i + 0]      = x0;
        qs[bi * qk + i + qk / 2] = x1;
    }
}

static void pack_mxfp4_quants(block_mxfp4 * x, const uint8_t * qs, unsigned int bi) {
    static const int qk = QK_MXFP4;

    for (unsigned int i = 0; i < qk / 2; ++i) {
        const uint8_t x0 = qs[bi * qk + i + 0];
        const uint8_t x1 = qs[bi * qk + i + qk / 2];
        x->qs[i]         = x0 | (x1 << 4);
    }
}

// repack q4_0 data into q4_0_tiled tensor
static void repack_q4_0_tiled(ggml_tensor * t, const void * data, size_t size) {
    const block_q4_0 * src_matrix = (const block_q4_0 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q4_0 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            unpack_q4_0_quants(tile_quants[row], &src_expert[r * (ne0 / 32) + kt], 0);
                        } else {
                            memset(tile_quants[row], 8, 32);
                        }
                    }

                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_expert[r * (ne0 / 32) + kt].d : 0;
                    }
                }
            }
        }
    }

    GGML_UNUSED(size);
}

// repack q4_0_tiled tensor into q4_0 data
static void repack_tiled_q4_0(void * data, const ggml_tensor * t, size_t size) {
    block_q4_0 * dst_matrix = (block_q4_0 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_q4_0 * dst_expert = dst_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            uint8_t val = tile_src[cp * 32 + row];
                            tile_quants[row][2 * cp + 0] = val & 0x0F;
                            tile_quants[row][2 * cp + 1] = val >> 4;
                        }
                    }

                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            pack_q4_0_quants(&dst_expert[r * (ne0 / 32) + kt], tile_quants[row], 0);
                        }
                    }

                    const ggml_half * scale_src = (const ggml_half *)(tile_src + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            dst_expert[r * (ne0 / 32) + kt].d = scale_src[row];
                        }
                    }
                }
            }
        }
    }

    GGML_UNUSED(size);
}

// repack q4_1 data into q4_1_tiled tensor
static void repack_q4_1_tiled(ggml_tensor * t, const void * data, size_t size) {
    const block_q4_1 * src_matrix = (const block_q4_1 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q4_1 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            unpack_q4_1_quants(tile_quants[row], &src_expert[r * (ne0 / 32) + kt], 0);
                        } else {
                            memset(tile_quants[row], 0, 32);
                        }
                    }

                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            scale_dst[2 * row + 0] = src_expert[r * (ne0 / 32) + kt].d;
                            scale_dst[2 * row + 1] = src_expert[r * (ne0 / 32) + kt].m;
                        } else {
                            scale_dst[2 * row + 0] = 0;
                            scale_dst[2 * row + 1] = 0;
                        }
                    }
                }
            }
        }
    }

    GGML_UNUSED(size);
}

// repack q4_1_tiled tensor into q4_1 data
static void repack_tiled_q4_1(void * data, const ggml_tensor * t, size_t size) {
    block_q4_1 * dst_matrix = (block_q4_1 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_q4_1 * dst_expert = dst_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            uint8_t val = tile_src[cp * 32 + row];
                            tile_quants[row][2 * cp + 0] = val & 0x0F;
                            tile_quants[row][2 * cp + 1] = val >> 4;
                        }
                    }

                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            pack_q4_1_quants(&dst_expert[r * (ne0 / 32) + kt], tile_quants[row], 0);
                        }
                    }

                    const ggml_half * scale_src = (const ggml_half *)(tile_src + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            dst_expert[r * (ne0 / 32) + kt].d = scale_src[2 * row];
                            dst_expert[r * (ne0 / 32) + kt].m = scale_src[2 * row + 1];
                        }
                    }
                }
            }
        }
    }

    GGML_UNUSED(size);
}

// repack q8_0 data into q8_0_tiled tensor
static void repack_q8_0_tiled(ggml_tensor * t, const void * data, size_t size) {
    const block_q8_0 * src_matrix = (const block_q8_0 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q8_0 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    for (int cp = 0; cp < 16; cp++) {
                        int col0 = cp * 2;
                        int col1 = col0 + 1;
                        for (int row = 0; row < 32; row++) {
                            int64_t r = ct * 32 + row;
                            const block_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_expert[r * (ne0 / 32) + kt] : NULL;
                            tile_dst[cp * 64 + 2 * row + 0] = b ? b->qs[col0] : 0;
                            tile_dst[cp * 64 + 2 * row + 1] = b ? b->qs[col1] : 0;
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 1024);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_expert[r * (ne0 / 32) + kt].d : 0;
                    }
                }
            }
        }
    }

    GGML_UNUSED(size);
}

// repack q8_0_tiled tensor into q8_0 data
static void repack_tiled_q8_0(void * data, const ggml_tensor * t, size_t size) {
    block_q8_0 * dst_matrix = (block_q8_0 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_q8_0 * dst_expert = dst_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                    for (int cp = 0; cp < 16; cp++) {
                        int col0 = cp * 2;
                        int col1 = col0 + 1;
                        for (int row = 0; row < 32; row++) {
                            int64_t r = ct * 32 + row;
                            if (r < ne1 && kt < ne0 / 32) {
                                block_q8_0 & b = dst_expert[r * (ne0 / 32) + kt];
                                b.qs[col0] = tile_src[cp * 64 + 2 * row + 0];
                                b.qs[col1] = tile_src[cp * 64 + 2 * row + 1];
                            }
                        }
                    }

                    const ggml_half * scale_src = (const ggml_half *)(tile_src + 1024);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            dst_expert[r * (ne0 / 32) + kt].d = scale_src[row];
                        }
                    }
                }
            }
        }
    }

    GGML_UNUSED(size);
}

// repack mxfp4 data into mxfp4_tiled tensor
static void repack_mxfp4_tiled(ggml_tensor * t, const void * data, size_t size) {
    const block_mxfp4 * src_matrix = (const block_mxfp4 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_MXFP4;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_mxfp4 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            unpack_mxfp4_quants(tile_quants[row], &src_expert[r * (ne0 / 32) + kt], 0);
                        } else {
                            memset(tile_quants[row], 0, 32);
                        }
                    }

                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                        }
                    }

                    uint8_t * scale_dst = tile_dst + 512;
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_expert[r * (ne0 / 32) + kt].e : 0;
                    }
                }
            }
        }
    }

    GGML_UNUSED(size);
}

// repack mxfp4_tiled tensor into mxfp4 data
static void repack_tiled_mxfp4(void * data, const ggml_tensor * t, size_t size) {
    block_mxfp4 * dst_matrix = (block_mxfp4 *) data;
    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    int64_t ne2 = t->ne[2];
    int64_t ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up(ne0, 32);
    int64_t ne1_padded = hex_round_up(ne1, 32);

    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_MXFP4;
    const size_t matrix_size = n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_mxfp4 * dst_expert = dst_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            uint8_t val = tile_src[cp * 32 + row];
                            tile_quants[row][2 * cp + 0] = val & 0x0F;
                            tile_quants[row][2 * cp + 1] = val >> 4;
                        }
                    }

                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            pack_mxfp4_quants(&dst_expert[r * (ne0 / 32) + kt], tile_quants[row], 0);
                        }
                    }

                    const uint8_t * scale_src = tile_src + 512;
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            dst_expert[r * (ne0 / 32) + kt].e = scale_src[row];
                        }
                    }
                }
            }
        }
    }

    GGML_UNUSED(size);
}

static void ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   const void *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu\n", sess->c_name(), tensor->name, data, offset, size);

    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_q4_0_tiled(tensor, data, size);
            break;

        case GGML_TYPE_Q4_1:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_q4_1_tiled(tensor, data, size);
            break;

        case GGML_TYPE_Q8_0:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_q8_0_tiled(tensor, data, size);
            break;

        case GGML_TYPE_IQ4_NL:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            // IQ4_NL has identical block layout to Q4_0 (ggml_half d + uint8_t qs[16])
            repack_q4_0_tiled(tensor, data, size);
            break;

        case GGML_TYPE_MXFP4:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_mxfp4_tiled(tensor, data, size);
            break;

        default:
            memcpy((char *) tensor->data + offset, data, size);
            break;
    }
}

static void ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   const ggml_tensor *   tensor,
                                                   void *                data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu\n", sess->c_name(), tensor->name, data, offset, size);

    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q4_0(data, tensor, size);
            break;

        case GGML_TYPE_Q4_1:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q4_1(data, tensor, size);
            break;

        case GGML_TYPE_Q8_0:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q8_0(data, tensor, size);
            break;

        case GGML_TYPE_IQ4_NL:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_q4_0(data, tensor, size);
            break;

        case GGML_TYPE_MXFP4:
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
            repack_tiled_mxfp4(data, tensor, size);
            break;

        default:
            memcpy(data, (const char *) tensor->data + offset, size);
            break;
    }
}

static bool ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   const struct ggml_tensor * src,
                                                   struct ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    return false;

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

static void ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (void *) sbuf->base, sbuf->size);
    memset(sbuf->base, value, sbuf->size);
}

static ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ NULL,
};

// ** backend buffer type

static const char * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    return static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto sess = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->sess;
    try {
        size += 4 * 1024;  // guard page
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, size);
        return ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate buffer context (host): %s\n", sess->c_name(), exc.what());
        return nullptr;
    }
}

static ggml_backend_buffer_t ggml_backend_hexagon_repack_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto sess = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->sess;
    try {
        size += 4 * 1024;  // guard page
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, size);
        return ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate buffer context (repack): %s\n", sess->c_name(), exc.what());
        return nullptr;
    }
}

static size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;  // HVX alignment
    GGML_UNUSED(buft);
}

static size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * t) {
    if (t->type == GGML_TYPE_Q4_0 || t->type == GGML_TYPE_Q4_1 || t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_IQ4_NL || t->type == GGML_TYPE_MXFP4) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        return ggml_row_size(t->type, ne0) * ne1 * ne2 * ne3;
    }
    return ggml_nbytes(t);

    GGML_UNUSED(buft);
}

static size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * context = static_cast<ggml_backend_hexagon_buffer_type_context *>(buft->context);
    return context->sess->max_bufsize;
}

static bool ggml_backend_hexagon_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return opt_hostbuf;

    GGML_UNUSED(buft);
}

static bool ggml_backend_hexagon_repack_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_hexagon_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_hexagon_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_hexagon_buffer_type_is_host,
};

static ggml_backend_buffer_type_i ggml_backend_hexagon_repack_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_hexagon_repack_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_hexagon_repack_buffer_type_is_host,
};

static bool ggml_backend_buffer_is_hexagon(const struct ggml_backend_buffer * b) {
    return b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

static inline bool ggml_backend_buffer_is_hexagon_repack(const struct ggml_backend_buffer * b) {
    if (!opt_hostbuf) {
        return ggml_backend_buffer_is_hexagon(b);
    }
    return b->buft->iface.alloc_buffer == ggml_backend_hexagon_repack_buffer_type_alloc_buffer;
}

struct ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<int, int>                b_map; // buffer fd   to index
    std::unordered_map<const ggml_tensor*, int> t_map; // tensor ptr  to index
    std::unordered_multimap<void*, int>         d_map; // tensor data to index



    unsigned int n_bufs;     // num buffers in the batch
    unsigned int n_tens;     // num tensors ...
    unsigned int n_ops;      // num ops ...
    size_t       b_vmem;     // sum of all buffer sizes

    unsigned int n_bufs_max;
    unsigned int n_tens_max;
    unsigned int n_ops_max;
    size_t       b_vmem_max;

    void reset() {
        n_bufs = 0;
        n_tens = 0;
        n_ops  = 0;
        b_vmem = 0;

        b_map.clear();
        t_map.clear();
        d_map.clear();
    }

    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t batch_size, size_t max_vmem) {
        this->sess = sess;

        n_bufs_max = HTP_OP_MAX_BUFS;
        n_ops_max  = batch_size;
        n_tens_max = std::min<size_t>(n_ops_max + n_ops_max * HTP_OP_MAX_INPUTS, HTP_OP_MAX_TENSORS);

        b_vmem_max = max_vmem;

        ops.resize(n_ops_max);

        h_bufs.resize(n_bufs_max);
        h_tens.resize(n_tens_max);
        h_ops.resize(n_ops_max);

        b_map.reserve(n_bufs_max);
        t_map.reserve(n_tens_max);
        d_map.reserve(n_tens_max);

        GGML_LOG_INFO("ggml-hex: %s op batching: n-bufs %u n-tensors %u n-ops %u vmem %zu\n",
                sess->c_name(), n_bufs_max, n_tens_max, n_ops_max, b_vmem_max);

        reset();
    }

    bool empty() const { return n_ops == 0; }

    // add buffer and return its index
    int add_buffer(ggml_hexagon_shared_buffer * sbuf) {
        // Lookup by fd
        auto it = b_map.find(sbuf->fd);
        if (it != b_map.end()) { return it->second; }

        // Add new buffer to the batch
        int bi = n_bufs++;
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);

        b_map.insert({sbuf->fd, bi});

        htp_buf_desc &b = h_bufs[bi];
        b.base = (uint64_t) sbuf->base;
        b.fd   = sbuf->fd;
        b.size = sbuf->size;

        b_vmem += b.size;

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (void*) sbuf->base, (size_t) b.size, b_vmem);

        return bi;
    }



    bool same_shape(const htp_tensor * h, const ggml_tensor * t) const {
        int64_t ne0 = t->ne[0];
        int64_t ne1 = t->ne[1];
        const bool is_repack = ggml_backend_buffer_is_hexagon_repack(t->buffer) && ggml_hexagon_is_repack_type(t->type);
        if (is_repack) {
            ne0 = hex_round_up(ne0, 32);
            ne1 = hex_round_up(ne1, 32);
        }
        int64_t nb1 = is_repack ? ggml_row_size(t->type, ne0) : t->nb[1];
        int64_t nb2 = is_repack ? nb1 * ne1 : t->nb[2];
        int64_t nb3 = is_repack ? nb2 * t->ne[2] : t->nb[3];

        return (h->type == t->type) &&
               (h->ne[0] == ne0) && (h->ne[1] == ne1) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == t->nb[0]) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    int add_tensor(const ggml_tensor * t) {
        auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // First lookup by tensor data
        auto range = d_map.equal_range(t->data);
        for (auto it = range.first; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            if (same_shape(h, t)) { return it->second; }
        }

        // Lookup by tensor ptr
        auto it = t_map.find(t);
        if (it != t_map.end()) { return it->second; }

        // Add new tensor to the batch
        int ti = n_tens++;
        GGML_ASSERT(n_tens <= n_tens_max);

        t_map.insert({t,       ti});
        d_map.insert({t->data, ti});

        uint64_t t_offset = (uint8_t *) t->data - sbuf->base;
        size_t   t_size   = ggml_nbytes(t);

        htp_tensor &h = h_tens[ti];
        h.bi    = add_buffer(sbuf);
        h.ti    = ti;
        h.data  = t_offset;
        h.type  = t->type;

        const bool is_repack = ggml_backend_buffer_is_hexagon_repack(t->buffer) && ggml_hexagon_is_repack_type(t->type);
        if (is_repack) {
            h.ne[0] = hex_round_up(t->ne[0], 32);
            h.ne[1] = hex_round_up(t->ne[1], 32);
            h.ne[2] = t->ne[2];
            h.ne[3] = t->ne[3];

            h.nb[0] = t->nb[0];
            h.nb[1] = ggml_row_size(t->type, h.ne[0]);
            h.nb[2] = h.nb[1] * h.ne[1];
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_size  = h.size;
        } else {
            h.size  = t_size;
            h.ne[0] = t->ne[0]; h.ne[1] = t->ne[1]; h.ne[2] = t->ne[2]; h.ne[3] = t->ne[3];
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }



        h.flags = 0;
        if (ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            h.flags |= HTP_TENSOR_COMPUTE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (void*) t->data, (size_t) t_offset, t_size, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        return ti;
    }

    bool fit_op(const htp_opnode & node) const {
        if (n_ops >= n_ops_max ) return false;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        auto fit_tensor = [&](const ggml_tensor *t) {
            if (!t) return;
            if (!t_map.count(t)) {
                extra_tens++;

                auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
                if (!b_map.count(sbuf->fd)) {
                    extra_vmem += sbuf->size;
                    extra_bufs += 1;
                }
            }
        };

        for (const auto * src : node.get_inputs()) {
            fit_tensor(src);
        }
        for (const auto * output : node.get_outputs()) {
            fit_tensor(output);
        }

        if ((extra_bufs + n_bufs) > n_bufs_max) return false;
        if ((extra_tens + n_tens) > n_tens_max) return false;
        if ((extra_vmem + b_vmem) > b_vmem_max) return false;

        return true;
    }

    // assumes that fit_op() was called first and returned true
    void add_op(const htp_opnode & node) {
        // Add new op

        unsigned int n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = node;

        htp_op_desc &o = h_ops[n];
        memcpy(o.params,        node.node->op_params, sizeof(node.node->op_params));
        memcpy(o.kernel_params, node.kernel_params,   sizeof(o.kernel_params));
        o.opcode = node.opcode;
        o.flags  = 0;

        if (!(opt_opstage & HTP_OPSTAGE_COMPUTE)) {
            o.flags |= HTP_OPFLAGS_SKIP_COMPUTE;
        }

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        auto inputs = node.get_inputs();
        for (unsigned int i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? add_tensor(inputs[i]) : 0xffff;
        }

        auto outputs = node.get_outputs();
        for (unsigned int i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }

    void finalize_ranges() {
    }
};

struct ggml_hexagon_opqueue {
    // Shared buffer for storing batches
    ggml_hexagon_shared_buffer *shm_buf;
    size_t                      shm_blk_size;

    using opvec = std::vector<htp_opnode>;

    std::queue<unsigned int>    done;           // completed batch ids
    std::vector<opvec>          op_cache;       // per batch op cache
    std::vector<uint64_t>       start_usec;     // per batch start time

    ggml_hexagon_opqueue(ggml_hexagon_session *sess, size_t batch_size, size_t depth) {
        size_t n_bufs    = HTP_OP_MAX_BUFS;
        size_t n_ops     = batch_size;
        size_t n_tensors = n_ops * HTP_OP_MAX_OUTPUTS + n_ops * HTP_OP_MAX_INPUTS;

        size_t tr_size = 0;
        if (opt_profile == 3) {
            tr_size = (HTP_MAX_NTHREADS + 1) * opt_optrace * sizeof(htp_trace_desc);
        }

        shm_blk_size = sizeof(htp_buf_desc)  * n_bufs    +
                       sizeof(htp_tensor)    * n_tensors +
                       sizeof(htp_op_desc)   * n_ops     +
                       sizeof(htp_prof_desc) * n_ops     +
                       tr_size;

        shm_buf = new ggml_hexagon_shared_buffer(sess, shm_blk_size * depth, true /* pinned */);

        op_cache.resize(depth);
        start_usec.resize(depth, 0);

        // init done queue
        for (unsigned int i = 0; i < depth; i++) { done.push(i); }

        if (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated op-queue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_size, depth, shm_buf->size, shm_blk_size);
        }
    }

    ~ggml_hexagon_opqueue() {
        delete shm_buf;
    }

    // push new batch
    bool push(htp_opbatch_req& req, dspqueue_buffer& dbuf, ggml_hexagon_opbatch* op_batch) {
        static_assert(sizeof(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(sizeof(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(sizeof(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(sizeof(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(sizeof(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(sizeof(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        if (done.empty()) { return false; }

        req.id        = done.front(); done.pop(); // batch id
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[req.id]   = op_batch->ops;
        start_usec[req.id] = ggml_time_us();

        const size_t b_size = sizeof(htp_buf_desc)  * req.n_bufs;
        const size_t t_size = sizeof(htp_tensor)    * req.n_tensors;
        const size_t o_size = sizeof(htp_op_desc)   * req.n_ops;
        const size_t p_size = sizeof(htp_prof_desc) * req.n_ops;

        size_t tr_size = 0;
        if (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * req.n_traces * sizeof(htp_trace_desc);
        } else {
            req.n_traces = 0;
        }

        dbuf.ptr      = shm_buf->base + (req.id * shm_blk_size);
        dbuf.fd       = shm_buf->fd;
        dbuf.flags    = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
        dbuf.offset   = (uint8_t*) dbuf.ptr - (uint8_t*) shm_buf->base;
        dbuf.size     = b_size + t_size + o_size + p_size + tr_size;

        GGML_ASSERT(dbuf.size <= shm_blk_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * b_ptr = m_ptr; m_ptr += b_size;
        uint8_t * t_ptr = m_ptr; m_ptr += t_size;
        uint8_t * o_ptr = m_ptr;

        memcpy(b_ptr, (void *) op_batch->h_bufs.data(), b_size);
        memcpy(t_ptr, (void *) op_batch->h_tens.data(), t_size);
        memcpy(o_ptr, (void *) op_batch->h_ops.data(),  o_size);

        HEX_VERBOSE("ggml-hex: %s op-queue push batch #%u : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), req.id, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_size, t_size, o_size, (size_t) dbuf.size);

        op_batch->reset();

        if (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            for (unsigned int i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (void *) b[i].base, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            for (unsigned int i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %u size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, t[i].data, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        return true;
    }

    void pop(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
        GGML_ASSERT(rsp.id < op_cache.size());

        done.push(rsp.id);

        const size_t b_size = sizeof(htp_buf_desc)  * rsp.n_bufs;
        const size_t t_size = sizeof(htp_tensor)    * rsp.n_tensors;
        const size_t o_size = sizeof(htp_op_desc)   * rsp.n_ops;
        const size_t p_size = sizeof(htp_prof_desc) * rsp.n_ops;

        size_t tr_size = 0;
        uint32_t n_traces = 0;
        if (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * n_traces * sizeof(htp_trace_desc);
        }

        const size_t m_size = b_size + t_size + o_size + p_size + tr_size;
        GGML_ASSERT(m_size <= shm_blk_size);

        HEX_VERBOSE("ggml-hex: %s op-queue pop batch #%u : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), rsp.id, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_size, t_size, o_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);

        if (opt_profile && rsp.n_ops > 0) {
            auto & ops = op_cache[rsp.id];

            GGML_ASSERT(rsp.n_ops <= ops.size());

            const htp_prof_desc * pd = (const htp_prof_desc *) p_ptr;

            const htp_trace_desc * trace_events = nullptr;

            if (opt_profile == 3) {
                trace_events = (const htp_trace_desc *) (p_ptr + p_size);
            }

            ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);

            for (uint32_t i = 0; i < rsp.n_ops; i++) {
                ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
            }

            ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
        }
    }
};

// Flush HTP response queue i.e wait for all outstanding requests to complete
void ggml_hexagon_session::flush_pending(bool all) {
    while (this->op_pending) {
        struct htp_opbatch_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        struct dspqueue_buffer dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        const uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        int err = dspqueue_read(this->queue, &flags, 1, &n_dbufs, &dbuf, sizeof(rsp), &rsp_size, (uint8_t *) &rsp, timeo);
        if (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            continue;
        }

        if (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (unsigned) err);
        }

        // Basic sanity checks
        if (rsp_size != sizeof(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", this->c_name(), rsp_size, n_dbufs);
        }

        if (rsp.status != HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp: %s\n", this->c_name(), status_to_str(rsp.status));
            // TODO: handle errors
        }

        op_queue->pop(rsp, dbuf);

        this->op_pending--;  // atomic dec

        if (!all) break;
    }
}

void ggml_hexagon_session::flush_batch() {
    if (op_batch->empty()) { return; }

    op_batch->finalize_ranges();

    htp_opbatch_req req {};
    dspqueue_buffer dbuf{};

    if (!op_queue->push(req, dbuf, op_batch)) {
        flush_pending(false);
        op_queue->push(req, dbuf, op_batch);
    }

    // Bump pending flag (cleared in the session::flush once we get the response)
    this->op_pending++;  // atomic inc

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", this->c_name(), dbuf.ptr, dbuf.size);

    int err = dspqueue_write(this->queue, 0, 1, &dbuf, sizeof(req), (const uint8_t*) &req, DSPQUEUE_TIMEOUT);
    if (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", this->c_name(), (unsigned) err);
    }
}

void ggml_hexagon_session::enqueue_op(const htp_opnode & node) {
    if (!op_batch->fit_op(node)) {
        flush_batch();
    }
    op_batch->add_op(node);
}

// Flush HTP response queue i.e wait for all outstanding requests to complete
void ggml_hexagon_session::flush(bool all) {
    flush_batch();
    flush_pending(all);
}

static size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_buffer *> sbufs;

    const size_t MiB = 1024 * 1024;
    const size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t step = 256u * MiB;

    try {
        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;
        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;
        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;

        while (1) {
            sbufs.push_back(new ggml_hexagon_shared_buffer(sess, step, true));
            vmem += step;
        }
    } catch (...) { }

    for (auto b : sbufs) { delete b; }

    return vmem - step; // backoff to account for overhead from internal mappings
}

void ggml_hexagon_session::allocate(int dev_id) noexcept(false) {
    this->valid_session = false;
    this->valid_handle  = false;
    this->valid_queue   = false;
    this->valid_iface   = false;

    this->domain_id  = 3;  // Default for CDSP, updated after the session is created
    this->session_id = 0;  // Default for CDSP, updated after the session is created
    this->dev_id     = dev_id;
    this->name       = std::string("HTP") + std::to_string(dev_id);

    this->op_pending  = 0;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session\n", this->name.c_str());

    domain * my_domain = htpdrv_get_domain(this->domain_id);
    if (my_domain == NULL) {
        GGML_LOG_ERROR("ggml-hex: unable to get domain struct for CDSP\n");
        throw std::runtime_error("ggml-hex: failed to get CDSP domain (see log for details)");
    }

    // Create new session
    if (dev_id != 0) {
        struct remote_rpc_reserve_new_session n;
        n.domain_name_len  = strlen(CDSP_DOMAIN_NAME);
        n.domain_name      = const_cast<char *>(CDSP_DOMAIN_NAME);
        n.session_name     = const_cast<char *>(this->name.c_str());
        n.session_name_len = this->name.size();

        int err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (void *) &n, sizeof(n));
        if (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: failed to reserve new session %d : error 0x%x\n", dev_id, err);
            throw std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        this->session_id    = n.session_id;
        this->domain_id     = n.effective_domain_id;
        this->valid_session = true;
    }

    // Get session URI

    char session_uri[256];
    {
        char htp_uri[256];
        snprintf(htp_uri, sizeof(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        struct remote_rpc_get_uri u = {};
        u.session_id      = this->session_id;
        u.domain_name     = const_cast<char *>(CDSP_DOMAIN_NAME);
        u.domain_name_len = strlen(CDSP_DOMAIN_NAME);
        u.module_uri      = const_cast<char *>(htp_uri);
        u.module_uri_len  = strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = sizeof(session_uri);

        int err = remote_session_control(FASTRPC_GET_URI, (void *) &u, sizeof(u));
        if (err != AEE_SUCCESS) {
            // fallback to single session uris
            int htp_URI_domain_len = strlen(htp_uri) + MAX_DOMAIN_NAMELEN;

            snprintf(session_uri, htp_URI_domain_len, "%s%s", htp_uri, my_domain->uri);

            GGML_LOG_WARN("ggml-hex: failed to get URI for session %d : error 0x%x. Falling back to single session URI: %s\n", dev_id, err, session_uri);
        }
    }

    // Enable Unsigned PD
    {
        struct remote_rpc_control_unsigned_module u;
        u.domain = this->domain_id;
        u.enable = 1;
        int err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *) &u, sizeof(u));
        if (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: failed to enable unsigned PD for session %d : error 0x%x\n", dev_id, err);
            throw std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    // Open session
    int err = htp_iface_open(session_uri, &this->handle);
    if (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: failed to open session %d : error 0x%x\n", dev_id, err);
        throw std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    this->valid_handle = true;

    // Query HW info and resolve session options
    this->max_bufsize = opt_mbuf;
    {
        unsigned int hw_n_threads = 0;
        unsigned int hw_n_hvx     = 0;
        unsigned int hw_n_hmx     = 0;
        unsigned long long hw_vtcm_size = 0;
        int hw_err = htp_iface_hwinfo(this->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_size);
        if (hw_err == 0) {
            this->n_threads = opt_nhvx > 0 ? (uint32_t)opt_nhvx : (uint32_t)hw_n_threads;
            this->n_hvx     = opt_nhvx > 0 ? (uint32_t)opt_nhvx : (uint32_t)hw_n_hvx;
            this->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            this->vtcm_size = (uint64_t)hw_vtcm_size;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          this->c_name(), this->n_threads, this->n_hvx, this->n_hmx,
                          (unsigned long long)(this->vtcm_size / (1024 * 1024)));
        } else {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", this->c_name(), hw_err);
            this->n_threads = opt_nhvx > 0 ? (uint32_t)opt_nhvx : 8;
            this->n_hvx     = opt_nhvx > 0 ? (uint32_t)opt_nhvx : 8;
            this->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            this->vtcm_size = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        struct remote_rpc_control_latency l;
        l.enable = 1;

        int err = remote_handle64_control(this->handle, DSPRPC_CONTROL_LATENCY, (void *) &l, sizeof(l));
        if (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (unsigned) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", this->c_name(),
                  this->session_id, this->domain_id, session_uri, (unsigned long) this->handle);

    const size_t req_q_size = (sizeof(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    const size_t rsp_q_size = (sizeof(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(this->domain_id,
                          0,              // Flags
                          req_q_size,     // Request  queue size (in bytes)
                          rsp_q_size,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (void *) this,  // Callback context
                          &queue);
    if (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", this->name.c_str(), (unsigned) err);
        throw std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    this->valid_queue = true;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &this->queue_id);
    if (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (unsigned) err);
        throw std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    if (opt_etm) {
        err = htp_iface_etm(this->handle, 1);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    // Allocate buffers and state for op batching
    this->op_queue = new ggml_hexagon_opqueue(this, opt_opbatch, opt_opqueue);

    if (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(this);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", this->c_name(), opt_vmem);
    }
    this->max_vmem = opt_vmem;

    this->op_batch = new ggml_hexagon_opbatch(this, opt_opbatch, this->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_start(this->handle, dev_id, this->queue_id, opt_nhvx, opt_nhmx, this->max_vmem);
    if (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", this->c_name(), (unsigned) err);
        throw std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    this->valid_iface = true;

    if (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(this->handle, opt_profile, &pmu_conf);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (unsigned) err);
        }
    }
}

void ggml_hexagon_session::release() noexcept(true) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", this->name.c_str());

    int err;

#ifdef GGML_HEXAGON_HETERO
    ggml_hexagon_hetero_free(this);
    ggml_hexagon_hfold_free(this);
#endif
    ggml_hexagon_cluster_free(this);

    if (this->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(this->handle);
        if (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (unsigned) err);
        }
    }

    delete this->op_batch;
    delete this->op_queue;
    this->op_batch = nullptr;
    this->op_queue = nullptr;

    if (opt_etm) {
        err = htp_iface_etm(this->handle, 0);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    if (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(this->handle, 0, &pmu_conf);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (unsigned) err);
        }
    }

    if (this->valid_queue) {
        err = dspqueue_close(queue);
        if (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (unsigned) err);
        }
    }

    if (this->valid_handle) {
        htp_iface_close(this->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(int dev_id, ggml_backend_dev_t dev) noexcept(false) {
    buffer_type.device        = dev;
    repack_buffer_type.device = dev;

    op_batch = nullptr;
    op_queue = nullptr;

    try {
        allocate(dev_id);

        buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
        buffer_type.context = new ggml_backend_hexagon_buffer_type_context(this->name, this);

        repack_buffer_type.iface   = ggml_backend_hexagon_repack_buffer_type_interface;
        repack_buffer_type.context = new ggml_backend_hexagon_buffer_type_context(this->name + "-REPACK", this);
    } catch (const std::exception & exc) {
        release();
        throw;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(true) {
    release();

    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(repack_buffer_type.context);
}

// ** backend interface

// Sparse flash-attention block size, carried in op_params[4] (see
// ggml_hexagon_supported_fa_sparse for the full src[5] contract). Zero when the
// op is dense.
#define HTP_FA_PARAM_SPARSE_BS 4

// Query-block size the src[5] rows are expressed in, carried in op_params[5].
// Zero means "same as the selection block size", which is what an XAttention
// scorer emits: it uses one block size Bl for both the query and the key axis.
#define HTP_FA_PARAM_SPARSE_BQ 5

static int64_t ggml_hexagon_fa_sparse_bs(const struct ggml_tensor * op) {
    return op->src[5] ? (int64_t) ggml_get_op_params_i32(op, HTP_FA_PARAM_SPARSE_BS) : 0;
}

static int64_t ggml_hexagon_fa_sparse_bq(const struct ggml_tensor * op) {
    if (!op->src[5]) {
        return 0;
    }
    const int64_t bq = (int64_t) ggml_get_op_params_i32(op, HTP_FA_PARAM_SPARSE_BQ);
    return bq ? bq : ggml_hexagon_fa_sparse_bs(op);
}

// True when the selection carries a query-block axis, i.e. the kernel must pick a
// different sel[] row per query block instead of broadcasting row 0 to all of them.
static bool ggml_hexagon_fa_sparse_per_qblock(const struct ggml_tensor * op) {
    return op->src[5] && op->src[5]->ne[1] > 1;
}

static bool ggml_hexagon_flash_attn_is_hmx_eligible(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * q,
    const struct ggml_tensor * k,
    const struct ggml_tensor * v,
    const struct ggml_tensor * sinks
) {
    if (sess->n_hmx == 0) {
        return false;
    }

    if (opt_fa_select < 2) {
        return false;
    }

    if (k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        return false;
    }

    const uint32_t DK = q->ne[0];
    const uint32_t DV = v->ne[0];

    if (DK % 64 != 0 || DV % 64 != 0) {
        return false;
    }

    // Fall back to HVX for small token counts if head dimension is small (DK <= 128)
    const uint32_t neq1 = q->ne[1];
    if (DK <= 128 && neq1 < 5) {
        return false;
    }

    return true;

    GGML_UNUSED(sinks);
}

static bool ggml_hexagon_precompute_flash_attn_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_fa_kernel_params * kparams
) {
    if (opt_fa_select < 1) {
        return false;
    }

    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * q    = op->src[0];
    const struct ggml_tensor * k    = op->src[1];
    const struct ggml_tensor * v    = op->src[2];
    const struct ggml_tensor * mask = op->src[3];
    const struct ggml_tensor * dst  = op;

    const uint32_t neq0 = q->ne[0];  // head_dim (DK)
    const uint32_t neq1 = q->ne[1];  // n_tokens
    const uint32_t neq2 = q->ne[2];  // n_heads

    const uint32_t nek1 = k->ne[1];  // kv_len

    const uint32_t nev0 = v->ne[0];  // head_dim (DV)

    const uint32_t DK = neq0;
    const uint32_t DV = nev0;

    const uint32_t n_kv_heads = k->ne[2];
    const uint32_t G          = neq2 / n_kv_heads;

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         &op->op_params[0], sizeof(float));
    memcpy(&max_bias,      &op->op_params[1], sizeof(float));
    memcpy(&logit_softcap, &op->op_params[2], sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->is_q_fp32 = (q->type == GGML_TYPE_F32) ? 1 : 0;
    kparams->is_dst_fp32 = (dst->type == GGML_TYPE_F32) ? 1 : 0;
    kparams->G = G;

    const uint32_t n_head = q->ne[2];
    kparams->n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
    kparams->m0 = std::pow(2.0f, -(max_bias) / kparams->n_head_log2);
    kparams->m1 = std::pow(2.0f, -(max_bias / 2.0f) / kparams->n_head_log2);

    // Check HMX eligibility
    const struct ggml_tensor * sinks = op->src[4];
    const struct ggml_tensor * sel   = op->src[5];
    if (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // Sparse: pin the chunk size to the graph's block size and count only
        // the selected blocks, so every KV-length-derived quantity below (block
        // count, pipelining, VTCM) reflects the work actually performed.
        const size_t   bs           = sel ? (size_t) ggml_hexagon_fa_sparse_bs(op) : 0;
        const uint32_t kv_effective = sel ? (uint32_t) (sel->ne[0] * bs) : nek1;
        const bool     mask_per_head = (mask != nullptr && mask->ne[2] != 1);

        // Grouping: fold up to m selected blocks into one Bc-wide chunk, so the kernel
        // tiles wide while the graph keeps a fine selection granularity. Gated on
        //   - every selected block being full (no interior holes to stitch around),
        //   - a broadcast/absent mask (per-head grouping is not implemented),
        //   - and enough chunks left to keep the pipelined path (>= FA_MIN_KV_BLOCKS),
        //     which is what lets the non-pipelined loop stay a strict m == 1 path.
        size_t m_max = 1;
        if (sel && bs && !mask_per_head && (nek1 % bs) == 0) {
            // The fallback (non-pipelined) loop now handles m > 1 too, so grouping no
            // longer has to preserve >= FA_MIN_KV_BLOCKS chunks. The chunk-size search
            // already prices losing threads below that threshold, so let it decide.
            m_max = (size_t) sel->ne[0];
            m_max = hex_smin(m_max, (size_t) FA_SPARSE_MAX_M);
            if (m_max < 1) {
                m_max = 1;
            }
        }

        // Per-query-block selection: the kernel picks the sel[] row as q_start/bq, so a
        // query tile must not straddle a query block. Constrain the search rather than
        // validating after it -- Br is chosen from VTCM pressure, and a validate-only
        // rule would silently disable the backend whenever the search moved.
        const size_t br_align = ggml_hexagon_fa_sparse_per_qblock(op) ? (size_t) ggml_hexagon_fa_sparse_bq(op) : 0;

        // Per-row selection length (src[6]): sel->ne[0] becomes the upper bound and
        // rows decompose into their own chunk counts, so the "m divides n_sel" search
        // constraint is meaningless -- the kernel derives every chunk's width per row.
        const struct ggml_tensor * cnt = sel ? op->src[6] : nullptr;

        size_t Br = 0, Bc = 0;
        int ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK, DV, neq1, kv_effective, sess->vtcm_size, sess->n_threads,
                                         kparams->is_q_fp32 != 0, /*bc_step=*/bs, /*bc_cap=*/sel ? m_max * bs : 0,
                                         /*sel_blocks=*/(sel && !cnt) ? (size_t) sel->ne[0] : 0, br_align, mask_per_head);
        if (ret != 0 && br_align) {
            // br_unit is ceil(32/G), so for G in {3,5,6,7} nothing divides a 64- or
            // 128-wide query block. Fail-closed (the op drops to a dense backend), but
            // it is a silent capability cliff, so name it.
            HEX_VERBOSE("ggml-hex: fa-params no (Br, Bc) with Br dividing bq %zu (G %u neq1 %u kv_eff %u)\n",
                        br_align, G, neq1, kv_effective);
        }
        if (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            kparams->Br = Br;
            kparams->Bc = Bc;
            const uint32_t m = (sel && bs) ? (uint32_t) (Bc / bs) : 1;
            kparams->n_kv_blocks = sel ? (uint32_t) (((size_t) sel->ne[0] + m - 1) / m) : (nek1 + Bc - 1) / Bc;
            kparams->n_threads = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? sess->n_threads : 1;

            kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK, DV, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, kparams->is_q_fp32 != 0, mask_per_head);

            HEX_VERBOSE("ggml-hex: fa-params neq1 %u kv %u kv_eff %u : Br %zu Bc %zu n_kv_blocks %u "
                        "n_threads %u pipeline %u g_br %u vtcm %d/%zu sel_nq %lld bq %zu\n",
                        neq1, nek1, kv_effective, Br, Bc, kparams->n_kv_blocks, kparams->n_threads,
                        kparams->u.hmx.pipeline, kparams->u.hmx.g_br, kparams->vtcm_size, sess->vtcm_size,
                        sel ? (long long) sel->ne[1] : 0LL, br_align);

            // Selection geometry. The kernel derives its own VTCM strides from the layout
            // it rebuilds, so what it needs from here is only how the chunk decomposes:
            // Bc covers m = Bc/sparse_bs selected blocks, and n_sel bounds the last one.
            kparams->u.hmx.sparse_bs = sel ? (uint16_t) bs : 0;
            kparams->u.hmx.n_sel     = sel ? (uint16_t) sel->ne[0] : 0;
            // Only meaningful with a query axis; the shared-selection path leaves it 0
            // so the device keeps taking row 0 for every query block, as before.
            kparams->u.hmx.sel_bq    = (uint16_t) br_align;
            kparams->u.hmx.dyn_sel   = cnt ? 1 : 0;
            // The device still decides: it needs the VTCM the chosen (Br, Bc) left over,
            // and the selection itself, neither of which is known here.
            kparams->u.hmx.res_mode  = (uint16_t) opt_fa_kv_residency;
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_values(G);
            if (mask) {
                kparams->src3_div2 = init_fastdiv_values(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_values(mask->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            return true;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads = sess->n_threads;

    const size_t size_q_row_padded = hex_round_up(q->ne[0] * (kparams->is_q_fp32 ? 4 : 2), 128);
    const size_t size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    const size_t size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, kparams->is_q_fp32 != 0, mask != nullptr, sess->n_threads);

    kparams->u.hvx.size_q_row_padded = size_q_row_padded;
    kparams->u.hvx.size_k_row_padded = size_k_row_padded;
    kparams->u.hvx.size_v_row_padded = size_v_row_padded;
    kparams->u.hvx.split_kv          = opt_fa_decode ? 1 : 0;
    kparams->u.hvx.src0_div21 = init_fastdiv_values(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_values(q->ne[1]);
    kparams->broadcast_rk2 = init_fastdiv_values(q->ne[2]/k->ne[2]);
    kparams->broadcast_rk3 = init_fastdiv_values(q->ne[3]/k->ne[3]);
    kparams->broadcast_rv2 = init_fastdiv_values(q->ne[2]/v->ne[2]);
    kparams->broadcast_rv3 = init_fastdiv_values(q->ne[3]/v->ne[3]);
    if (mask) {
        kparams->src3_div2 = init_fastdiv_values(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_values(mask->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + sess->n_threads - 1) / sess->n_threads;

    return true;
}

// Block-sparse flash attention (experimental).
//
// A sparse FLASH_ATTN_EXT carries an extra input in src[5]: an I32 list of KV
// *block* indices to attend to, instead of the full [0, kv_len) range. The
// block size is op_params[4] and the kernel chunks KV on exactly that unit, so
// index i selects rows [idx*BS, idx*BS + BS) of K/V (and the matching mask
// columns). Selection may vary per query block, per KV head and per sequence.
//
//   src[5]: I32, nb[0] == 4, ne = [n_sel, NBq or 1, n_kv_heads or 1, n_seqs or 1]
//   op_params[4]: BS, multiple of 64, <= kv_len
//   op_params[5]: BQ, the query-block size the sel rows are expressed in (0 => BS).
//                 NBq = ceil(q->ne[1] / BQ). The kernel reads row q_start/BQ, so a
//                 query tile must not straddle a query block: the chunk-size search
//                 is constrained to Br | BQ (see ggml_hexagon_precompute_...).
//
// ne[1] == 1 is the shared-selection layout: every query row in the op sees the
// same list. That is what an attn_sum reduced over the query axis produces, and it
// stays the cheap path -- the kernel zeroes the row stride and never divides.
//
// Without src[6], n_sel is FIXED across query blocks (top-k, not a tau threshold).
// With src[6] -- F32 per-row lengths, ne mirroring sel->ne[1..3] -- sel->ne[0] is the
// upper bound u_max: each row attends to the first cnt entries of its list, and the
// kernel derives loop bound, chunk widths and DMA push/pop counts per row from the
// q_start of the tile a chunk serves, which is what keeps its untagged FIFO in sync.
// Host-side sizing (pipelining, thread count, VTCM) stays a function of u_max alone;
// VTCM in particular holds one Bc-wide chunk, not the selection, so it is unaffected.
//
// Only the HMX kernel implements the indirection; anything else rejects the op
// so it falls back to a dense backend rather than silently ignoring src[5].
static bool ggml_hexagon_supported_fa_sparse(const struct ggml_tensor * op) {
    if (!opt_fa_sparse) {
        // Reject the sparse op so the node drops to a dense backend, which ignores
        // src[5] and computes the exact same result. This is the A/B baseline.
        HEX_VERBOSE("ggml-hex: fa-sparse no : disabled by GGML_HEXAGON_FA_SPARSE=0\n");
        return false;
    }

    const struct ggml_tensor * k   = op->src[1];
    const struct ggml_tensor * q   = op->src[0];
    const struct ggml_tensor * sel = op->src[5];

    // The kernel indexes sel as base + kv_head*nb[2] + ib3*nb[3], then list[i] --
    // so it needs unit stride within a row and nothing more. Requiring full
    // contiguity rejects the natural output of ggml_argsort_top_k, which is a
    // strided view of the argsort result, and would silently drop the whole FA op
    // to a dense CPU fallback.
    if (sel->type != GGML_TYPE_I32 || sel->nb[0] != sizeof(int32_t)) {
        HEX_VERBOSE("ggml-hex: fa-sparse no : sel type %s nb0 %zu (want I32, 4)\n",
                    ggml_type_name(sel->type), (size_t) sel->nb[0]);
        return false;
    }

    const int64_t bs = ggml_hexagon_fa_sparse_bs(op);
    if (bs < 64 || (bs % 64) != 0 || bs > k->ne[1]) {
        HEX_VERBOSE("ggml-hex: fa-sparse no : bs %lld (want multiple of 64, <= kv %lld)\n",
                    (long long) bs, (long long) k->ne[1]);
        return false;
    }

    // Query axis: either broadcast (one list for the whole op) or exactly one list
    // per query block. The 32-alignment keeps Br | BQ reachable -- Br is quantised to
    // ceil(32/G), so an unaligned BQ could only ever be divided by a smaller G's unit.
    // The upper bound is what fits kparams.Br (uint16_t): Br divides BQ, so a BQ the
    // field cannot hold would silently truncate on the way to the device.
    const int64_t bq = ggml_hexagon_fa_sparse_bq(op);
    if (bq < 32 || (bq % 32) != 0 || bq > UINT16_MAX) {
        HEX_VERBOSE("ggml-hex: fa-sparse no : bq %lld (want multiple of 32, <= %u)\n",
                    (long long) bq, (unsigned) UINT16_MAX);
        return false;
    }
    if (sel->ne[1] != 1 && sel->ne[1] != (q->ne[1] + bq - 1) / bq) {
        HEX_VERBOSE("ggml-hex: fa-sparse no : sel nq %lld (want 1 or ceil(%lld/%lld) = %lld)\n",
                    (long long) sel->ne[1], (long long) q->ne[1], (long long) bq,
                    (long long) ((q->ne[1] + bq - 1) / bq));
        return false;
    }
    if (sel->ne[2] != 1 && sel->ne[2] != k->ne[2]) {
        HEX_VERBOSE("ggml-hex: fa-sparse no : sel nh %lld (want 1 or n_kv_heads %lld)\n",
                    (long long) sel->ne[2], (long long) k->ne[2]);
        return false;
    }
    if (sel->ne[3] != 1 && sel->ne[3] != q->ne[3]) {
        HEX_VERBOSE("ggml-hex: fa-sparse no : sel ns %lld (want 1 or n_seqs %lld)\n",
                    (long long) sel->ne[3], (long long) q->ne[3]);
        return false;
    }
    if (sel->ne[0] < 1 || sel->ne[0] > (k->ne[1] + bs - 1) / bs) {
        HEX_VERBOSE("ggml-hex: fa-sparse no : n_sel %lld (want 1..%lld = ceil(kv %lld / bs %lld))\n",
                    (long long) sel->ne[0], (long long) ((k->ne[1] + bs - 1) / bs),
                    (long long) k->ne[1], (long long) bs);
        return false;
    }

    const struct ggml_tensor * cnt = op->src[6];
    if (cnt) {
        // Shape must mirror sel's row axes one-for-one: a count row is meaningless
        // without the list row it measures. Broadcast follows sel's broadcast.
        if (cnt->type != GGML_TYPE_F32 || cnt->nb[0] != sizeof(float)) {
            HEX_VERBOSE("ggml-hex: fa-sparse no : cnt type %s nb0 %zu (want F32, 4)\n",
                        ggml_type_name(cnt->type), (size_t) cnt->nb[0]);
            return false;
        }
        if (cnt->ne[0] != sel->ne[1] || cnt->ne[1] != sel->ne[2] ||
            cnt->ne[2] != sel->ne[3] || cnt->ne[3] != 1) {
            HEX_VERBOSE("ggml-hex: fa-sparse no : cnt ne [%lld,%lld,%lld,%lld] (want [%lld,%lld,%lld,1])\n",
                        (long long) cnt->ne[0], (long long) cnt->ne[1],
                        (long long) cnt->ne[2], (long long) cnt->ne[3],
                        (long long) sel->ne[1], (long long) sel->ne[2], (long long) sel->ne[3]);
            return false;
        }
    }

    return true;
}

static bool ggml_hexagon_supported_flash_attn_ext(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * src3 = op->src[3];
    const struct ggml_tensor * src4 = op->src[4];
    const struct ggml_tensor * dst  = op;

    // Check for F16 support only as requested
    if ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) || src1->type != GGML_TYPE_F16 || src2->type != GGML_TYPE_F16) {
        return false;
    }

    if (src3 && src3->type != GGML_TYPE_F16) {  // mask
        return false;
    }

    if (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        return false;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        return false;
    }

    if (dst->ne[3] != 1) {
        return false;
    }

    if (op->src[5] && !ggml_hexagon_supported_fa_sparse(op)) {
        return false;
    }

    struct htp_fa_kernel_params kparams;
    if (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        return false;
    }

    // The block-index indirection lives only in the HMX kernel.
    if (op->src[5] && kparams.kernel_type != HTP_FA_KERNEL_HMX) {
        return false;
    }

    // Belt and braces on the query-tile alignment. The chunk-size search is already
    // constrained to Br | BQ, so this cannot fire unless the search regressed -- and
    // a straddling tile is silent (the kernel would give every row of the tile one
    // query block's selection and still produce a plausible-looking result).
    if (ggml_hexagon_fa_sparse_per_qblock(op)) {
        const size_t bq = (size_t) ggml_hexagon_fa_sparse_bq(op);
        if (kparams.Br == 0 || (bq % kparams.Br) != 0) {
            HEX_VERBOSE("ggml-hex: skip flash_attn_ext, Br %u does not divide bq %zu\n", (unsigned) kparams.Br, bq);
            return false;
        }
    }

    if ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_size, sess->vtcm_size);
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_gated_delta_net(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * q     = op->src[0];
    const struct ggml_tensor * k     = op->src[1];
    const struct ggml_tensor * v     = op->src[2];
    const struct ggml_tensor * g     = op->src[3];
    const struct ggml_tensor * beta  = op->src[4];
    const struct ggml_tensor * state = op->src[5];
    const struct ggml_tensor * dst   = op;

    if (!q || !k || !v || !g || !beta || !state) {
        return false;
    }

    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t S_v      = v->ne[0];
    const int64_t H        = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs   = v->ne[3];
    const int64_t K        = ggml_get_op_params_i32(op, 0);

    if (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        return false;
    }
    if (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        return false;
    }
    if ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        return false;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    if (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        return false;
    }
    if (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_matmul_is_hmx_eligible(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int ne01_padded,
    bool is_matmul_id,
    bool is_batched
) {
    const int ne00  = src0->ne[0];
    const int ne11  = src1->ne[1];
    const int ne12  = src1->ne[2];
    const int wtype = src0->type;

    // HMX weight tile requires N to be 32-aligned.
    if (ne01_padded % 32 != 0) {
        return false;
    }

    // HMX supports F16, F32, and repack quantized types.
    if (!ggml_hexagon_is_hmx_weight_type((ggml_type) wtype)) {
        return false;
    }

    // HMX paths require K aligned to 32.
    if (ne00 % 32 != 0) {
        return false;
    }

    // Quantized HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    if (!is_matmul_id && is_batched && wtype != GGML_TYPE_F16) {
        return false;
    }

    // HMX assumes contiguous row-major layout.
    if (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        return false;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS
    const int m = is_matmul_id ? ne12 : ne11;
    if (m <= HTP_MM_HMX_MIN_NROWS) {
        return false;
    }

    return true;

    GGML_UNUSED(dst);
}

static bool ggml_hexagon_precompute_hmx_mm_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int wtype,
    int ne00_padded,
    int ne01_padded,
    int ne02,
    int ne11,
    int ne12,
    int ne11_padded,
    bool is_matmul_id,
    bool is_batched,
    size_t vtcm_budget,
    struct htp_mm_kernel_params * kparams
) {
    const int aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
    const bool pipeline = is_matmul_id ? false : htp_mm_hmx_pipeline(ne11);
    const int n_threads = (int)sess->n_threads;
    const int ne10 = src1->ne[0];

    const bool is_batched_val = is_matmul_id ? false : is_batched;
    const int group_size = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_size = 0;
    bool use_grouped = false;
    int act_threads_selected = 0;

    if (is_batched_val && wtype == GGML_TYPE_F16 && group_size > 1) {
        // Try grouped path first
        const bool use_dma_activation = (src1->nb[1]/sizeof(float) > (size_t)ne00_padded);
        if (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, ne11, group_size, use_dma_activation, n_threads, pipeline, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            use_grouped = true;
        }
    }

    if (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        const int m_id_rows = (int) ((size_t) dst->ne[1] * dst->ne[2]);
        if (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, ne11_padded, ne11, n_threads, pipeline, is_matmul_id, aligned_tile_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            return false;
        }
    }

    kparams->n_hmx = 1;
    kparams->pipeline = pipeline ? 1 : 0;
    kparams->m_chunk = m_chunk;
    kparams->n_chunk = n_chunk;
    kparams->n_threads = n_threads;
    kparams->n_act_threads = act_threads_selected;
    kparams->tile_size = htp_mm_get_weight_tile_size(wtype);
    kparams->aligned_tile_size = aligned_tile_size;
    kparams->src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
    kparams->vtcm_size = vtcm_size;
    kparams->vtcm_src0_size = 0;
    kparams->div_n_act_threads = init_fastdiv_values(act_threads_selected);
    kparams->div_ne00_padded   = init_fastdiv_values(ne00_padded);
    kparams->vtcm_src1_size = 0;
    kparams->vtcm_dst_size = 0;

    if (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_F16_BATCHED;
    } else {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    return true;

    GGML_UNUSED(src0);
}

static void ggml_hexagon_precompute_hvx_mm_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int wtype,
    int ne02,
    int ne03,
    int ne10,
    int ne11,
    int ne12,
    int ne13,
    bool is_matmul_id,
    const size_t src2_row_size,
    size_t vtcm_budget,
    struct htp_mm_kernel_params * kparams
) {
    kparams->n_hmx = 0;

    const bool is_quant = (wtype != GGML_TYPE_F16 && wtype != GGML_TYPE_F32);
    const int src1_nrows = ne11 * ne12 * ne13;

    if (is_quant) {
        // Quantized HVX
        kparams->tile_size = htp_mm_get_weight_tile_size(wtype);
        kparams->aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);

        const bool k_align = (ne10 % 32 == 0);

        if (is_matmul_id) {
            kparams->kernel_type   = (src1_nrows < (int) sess->n_threads) ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);

            struct htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], 0, src2_row_size, d, true, false, false
                );
                if (L.total_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    break;
                }
            }
            if (best_n_prefetch == 2 && L.total_bytes > vtcm_budget) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], 0, src2_row_size, 2, true, false, false
                );
            }
            kparams->n_prefetch = best_n_prefetch;
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_src1_size = L.src1_bytes;
            kparams->vtcm_dst_size  = L.dst_bytes;
        } else {
            bool try_tiled = (k_align && opt_mm_select >= 2);
            if (try_tiled) {
                kparams->src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
                if (src1_nrows < (int)sess->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } else {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                struct htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, d, false, false, false
                    );
                    if (L.total_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        break;
                    }
                }
                if (best_n_prefetch == 2 && L.total_bytes > vtcm_budget) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 2, false, false, false
                    );
                }

                kparams->n_prefetch = best_n_prefetch;

                if (L.total_bytes <= vtcm_budget) {
                    kparams->vtcm_size = L.total_bytes;
                    kparams->vtcm_src0_size = L.src0_bytes;
                    kparams->vtcm_src1_size = L.src1_bytes;
                    kparams->vtcm_dst_size = L.dst_bytes;
                    goto done_quant;
                }
                HEX_VERBOSE("ggml-hex: %s HVX tiled path VTCM size needed (%zu) > budget (%zu), falling back to HVX flat\n", sess->name.c_str(), L.total_bytes, vtcm_budget);
            }

            // Flat HVX fallback
            {
                kparams->src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);
                kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;

                struct htp_mm_hvx_vtcm_layout L;
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 16, false, false, false
                );

                kparams->n_prefetch = 16;
                kparams->vtcm_size = L.total_bytes;
                kparams->vtcm_src0_size = L.src0_bytes;
                kparams->vtcm_src1_size = L.src1_bytes;
                kparams->vtcm_dst_size = L.dst_bytes;
            }
        }

    done_quant:;
    } else if (wtype == GGML_TYPE_F16) {
        // F16 HVX
        const bool is_batched  = (ne02 > 1) || (ne03 > 1);
        const bool is_permuted = ggml_is_permuted(src0) || ggml_is_permuted(src1);

        struct htp_mm_hvx_vtcm_layout L;
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
            dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 16, false, false, false
        );

        if (!is_batched && !is_permuted && L.total_bytes <= vtcm_budget) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->src1_row_size = hex_round_up(ne10 * 2, 128);
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_src1_size = L.src1_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
        } else {
            if (src1->type == GGML_TYPE_F32) {
                kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F32_DDR;
            } else {
                kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_DDR;
            }
            kparams->src1_row_size = src1->nb[1];
            htp_mm_hvx_vtcm_layout_build(
                &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 16, false, false, false
            );
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_src1_size = L.src1_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
        }
    } else {
        // F32 HVX
        const bool is_batched  = (ne02 > 1) || (ne03 > 1);
        const bool is_permuted = ggml_is_permuted(src0) || ggml_is_permuted(src1);

        struct htp_mm_hvx_vtcm_layout L;
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
            dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 16, false, false, false
        );

        if (!is_batched && !is_permuted && L.total_bytes <= vtcm_budget) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->src1_row_size = hex_round_up(ne10 * 4, 128);
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_src1_size = L.src1_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
        } else {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_DDR;
            kparams->src1_row_size = src1->nb[1];
            htp_mm_hvx_vtcm_layout_build(
                &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 16, false, false, false
            );
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_src1_size = L.src1_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
        }
    }
}

static void ggml_hexagon_precompute_matmul_params_impl(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    const size_t src2_row_size,
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int wtype = src0->type;
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);
    const int ne00_padded = is_repack ? hex_round_up(ne00, 32) : ne00;
    const int ne01_padded = is_repack ? hex_round_up(ne01, 32) : ne01;
    const int ne11_padded = hex_round_up(ne11, 32);

    const bool is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    const bool is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    const size_t vtcm_budget = sess->vtcm_size;

    // Check HMX eligibility and try precomputing HMX parameters
    bool hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 3);
    if (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, ne01_padded, is_matmul_id, is_batched)) {
        if (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_padded, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, vtcm_budget, kparams)) {
            goto finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_row_size, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_values(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_values(ne11);
    kparams->div_r2       = init_fastdiv_values(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_values(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne11     = init_fastdiv_values(ne11);
}

static void ggml_hexagon_precompute_matmul_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, kparams);
}

static void ggml_hexagon_precompute_fused_matmul_add_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * src2,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2->nb[1], kparams);
}

static void ggml_hexagon_precompute_unary_params(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = (std::min)((uint32_t)sess->n_threads, src0_nrows);

    kparams->n_threads = n_threads;

    const size_t src0_data_row_size = src0->ne[0] * sizeof(float);
    const size_t dst_data_row_size  = dst->ne[0]  * sizeof(float);

    const size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    const size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = 0;
    bool broadcast_weight = false;

    if (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != nullptr);
        src1_data_row_size = src1->ne[0] * sizeof(float);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
        broadcast_weight = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->broadcast_weight      = broadcast_weight;

    struct htp_unary_vtcm_layout L;
    uint32_t col_tile = 0;
    uint32_t vtcm_row_per_thread = 0;

    htp_unary_vtcm_layout_build(&L, op, src0->ne[0], dst->ne[0],
                                op == HTP_OP_RMS_NORM_MUL ? src1->ne[0] : 0,
                                broadcast_weight, n_threads, sess->vtcm_size,
                                &col_tile, &vtcm_row_per_thread);

    kparams->col_tile = col_tile;
    kparams->vtcm_row_per_thread = vtcm_row_per_thread;
    kparams->vtcm_size = L.total_bytes;

    kparams->vtcm_src0_size_per_thread = L.src0_bytes;
    kparams->vtcm_src1_size_per_thread = L.src1_bytes;
    kparams->vtcm_dst_size_per_thread  = L.dst_bytes;

    kparams->vtcm_src0_size = L.src0_bytes * n_threads;
    kparams->vtcm_src1_size = L.src1_bytes * n_threads;
    kparams->vtcm_dst_size  = L.dst_bytes * n_threads;

    kparams->block = col_tile ? 0 : ((L.src0_bytes / 2) / src0_row_size_aligned);

    const uint32_t tiles_per_row = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kparams->div_ne01  = init_fastdiv_values(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_values(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_values(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_values(tiles_per_row);
}

static void ggml_hexagon_precompute_fused_qkv_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0, // Wk
    const struct ggml_tensor * src1, // x
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const int wtype = src0->type;
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);

    const int ne10 = src1->ne[0];
    const int src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
    const size_t src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
    const size_t src0_row_size = src0->nb[1];

    uint32_t best_n_prefetch = 16;

    if (is_repack) {
        const uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        best_n_prefetch = 2;
        for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            struct htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, src1_nrows, sess->n_threads,
                0, src0_row_size, src1_row_size, 0, d, false, true, false
            );
            if (L.total_bytes <= sess->vtcm_size) {
                best_n_prefetch = d;
                break;
            }
        }
    }

    struct htp_mm_hvx_vtcm_layout L;
    bool try_tiled = (opt_mm_select >= 2);

    // Test tiled first
    htp_mm_hvx_vtcm_layout_build(
        &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, src1_nrows, sess->n_threads,
        0, src0_row_size, src1_row_size, 0, best_n_prefetch, false, true, false
    );

    if (try_tiled && L.total_bytes <= sess->vtcm_size) {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
        kparams->vtcm_src0_size = L.src0_bytes;
        kparams->vtcm_src1_size = L.src1_bytes;
        kparams->vtcm_src2_size = L.src2_bytes;
        kparams->vtcm_src3_size = L.src3_bytes;
        kparams->vtcm_dst_size  = L.dst_bytes;
        kparams->vtcm_size      = L.total_bytes;
        kparams->n_prefetch     = best_n_prefetch;
    } else {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;
        size_t flat_src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);

        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT, wtype, ne10, src1_nrows, sess->n_threads,
            0, src0_row_size, flat_src1_row_size, 0, best_n_prefetch, false, true, false
        );
        kparams->vtcm_src0_size = L.src0_bytes;
        kparams->vtcm_src1_size = L.src1_bytes;
        kparams->vtcm_src2_size = L.src2_bytes;
        kparams->vtcm_src3_size = L.src3_bytes;
        kparams->vtcm_dst_size  = L.dst_bytes;
        kparams->vtcm_size      = L.total_bytes;
        kparams->n_prefetch     = best_n_prefetch;
    }
}

static void ggml_hexagon_precompute_fused_ffn_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0, // Wgate
    const struct ggml_tensor * src1, // y
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const int wtype = src0->type;
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);

    const int ne10 = src1->ne[0];
    const int src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
    const size_t src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
    const size_t src0_row_size = src0->nb[1];

    uint32_t best_n_prefetch = 16;

    if (is_repack) {
        const uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        best_n_prefetch = 2;
        for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            struct htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, src1_nrows, sess->n_threads,
                0, src0_row_size, src1_row_size, 0, d, false, false, true
            );
            if (L.total_bytes <= sess->vtcm_size) {
                best_n_prefetch = d;
                break;
            }
        }
    }

    struct htp_mm_hvx_vtcm_layout L;
    bool try_tiled = (opt_mm_select >= 2);

    // Test tiled first
    htp_mm_hvx_vtcm_layout_build(
        &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, src1_nrows, sess->n_threads,
        0, src0_row_size, src1_row_size, 0, best_n_prefetch, false, false, true
    );

    if (try_tiled && L.total_bytes <= sess->vtcm_size) {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
        kparams->vtcm_src0_size = L.src0_bytes;
        kparams->vtcm_src1_size = L.src1_bytes;
        kparams->vtcm_src2_size = L.src2_bytes;
        kparams->vtcm_dst_size  = L.dst_bytes;
        kparams->vtcm_size      = L.total_bytes;
        kparams->n_prefetch     = best_n_prefetch;
    } else {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;
        size_t flat_src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);

        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT, wtype, ne10, src1_nrows, sess->n_threads,
            0, src0_row_size, flat_src1_row_size, 0, best_n_prefetch, false, false, true
        );
        kparams->vtcm_src0_size = L.src0_bytes;
        kparams->vtcm_src1_size = L.src1_bytes;
        kparams->vtcm_src2_size = L.src2_bytes;
        kparams->vtcm_dst_size  = L.dst_bytes;
        kparams->vtcm_size      = L.total_bytes;
        kparams->n_prefetch     = best_n_prefetch;
    }
}

static bool ggml_hexagon_supported_mul_mat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            if (src0->ne[0] % 32) {
                return false;
            }

            // hardcoded limit to refuse the lm-head for now (GGML_HEXAGON_LM_HEAD=1 lifts it)
            if (src0->ne[1] > 32768 && !opt_lm_head) {
                return false;
            }

            if (src1->ne[2] != 1 || src1->ne[3] != 1) {
                return false;  // no broadcasting (for now)
            }

            // src0 (weights) must be repacked
            if (src0->buffer && !ggml_backend_buffer_is_hexagon_repack(src0->buffer)) {
                return false;
            }
            break;

        case GGML_TYPE_F16:
            if (src0->nb[1] < src0->nb[0]) {
                return false;
            }
            if (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                return false;
            }
            break;

        case GGML_TYPE_F32:
            if (src1->type != GGML_TYPE_F32) {
                return false;
            }
            if (src0->nb[1] < src0->nb[0]) {
                return false;
            }
            if (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                return false;
            }
            break;

        default:
            return false;
    }

    struct htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    if ((size_t)kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_mul_mat_id(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            if ((src0->ne[0] % 32)) {
                return false;
            }

            // src0 (weights) must be repacked
            if (src0->buffer && !ggml_backend_buffer_is_hexagon_repack(src0->buffer)) {
                return false;
            }
            break;

        default:
            return false;
    }

    struct htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    if ((size_t)kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        return false;
    }

    return true;
}

static bool ggml_hexagon_supported_binary(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (src0->type == GGML_TYPE_F32) {
        if (src1->type != GGML_TYPE_F32) {
            return false;
        }
        if (dst->type != GGML_TYPE_F32) {
            return false;
        }
    }
    else if (src0->type == GGML_TYPE_F16) {
        if (src1->type != GGML_TYPE_F16) {
            return false;
        }
        if (dst->type != GGML_TYPE_F16) {
            return false;
        }
    }
    else {
        return false;
    }

    if (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        return false;
    }
    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }
    if (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_add_id(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }

    // REVISIT: add support for non-contigiuos tensors
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_unary(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (ggml_is_permuted(src0)) {
        return false;
    }
    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }

    // dst must be contiguous; src0 may be non-contiguous
    if (!ggml_is_contiguous(dst)) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_sum_rows(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    // TODO: add support for non-contigiuos tensors
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_activations(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous_1(src0)) {
        return false;
    }
    if (!ggml_is_contiguous(dst)) {
        return false;
    }

    if (src1) {
        if (src1->type != GGML_TYPE_F32) {
            return false;
        }
        if (!ggml_are_same_shape(src0, src1)) {
            return false;
        }
        if (!ggml_is_contiguous_1(src1)) {
            return false;
        }
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_softmax(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    if (src2) {
        return false;  // FIXME: add support for sinks
    }

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src1) {
        if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            return false;
        }
        if (src0->ne[0] != src1->ne[0]) {
            return false;
        }
        if (src1->ne[1] < src0->ne[1]) {
            return false;
        }
        if (src0->ne[2] % src1->ne[2] != 0) {
            return false;
        }
        if (src0->ne[3] % src1->ne[3] != 0) {
            return false;
        }
    }

    if (src1) {
        if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            return false;
        }
    } else {
        if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            return false;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    const int64_t ne0 = src0->ne[0];
    if (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        return false;
    }

    // HVX vector size constraints for softmax
    #define SOFTMAX_MAX_ROW_SIZE 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    if (ne0 > SOFTMAX_MAX_ROW_SIZE) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_set_rows(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // values
    const struct ggml_tensor * src1 = op->src[1]; // indices
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }

    if (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        return false;
    }

    if (dst->type != GGML_TYPE_F16) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_get_rows(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // values
    const struct ggml_tensor * src1 = op->src[1]; // indices
    const struct ggml_tensor * dst  = op;

    // F16 sources are gathered with an on-the-fly convert to the F32 result that
    // ggml_get_rows always produces. Supporting it keeps a F16 KV cache gatherable
    // without a F32 copy of the whole cache.
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        return false;
    }

    if (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        return false;
    }

    // A F16 destination lets a GET_ROWS+CPY pair collapse to a single op, and when it
    // matches a F16 source the gather degenerates to a raw copy with no conversion.
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_argsort(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // values
    const struct ggml_tensor * dst  = op;         // indices

    if (src0->type != GGML_TYPE_F32) {
        return false;
    }

    if (dst->type != GGML_TYPE_I32) {
        return false;
    }

    if (src0->ne[0] > (16*1024)) {
        // reject tensors with huge rows for now
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_rope(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const int32_t * op_params = &op->op_params[0];

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    if (op_params[15] % 32 != 0) {
        return false;
    }

    int mode = op_params[2];

    // n_dims == ne0/2, so the rotation spans the full row
    if (mode == GGML_ROPE_TYPE_VISION) {
        const int n_dims = op_params[1];
        if (n_dims != (int) (op->src[0]->ne[0] / 2)) {
            return false;
        }
    }
    if (mode & 1) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) {
        return false;  // FIXME: add support for GGML_TYPE_F16 for src0
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->type != GGML_TYPE_I32) {
        return false;
    }
    if (src2) {
        if (src2->type != GGML_TYPE_F32) {
            return false;
        }
        int n_dims = op_params[1];
        if (src2->ne[0] < (n_dims / 2)) {
            return false;
        }
    }

    if (src2) {
        if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(src2)) {
            return false;
        }
    } else {
        if (!ggml_is_contiguous(src1)) {
            return false;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    if (src0->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->nb[1] < src0->ne[0] * sizeof(float) || dst->nb[1] < dst->ne[0] * sizeof(float)) {
        return false;
    }
    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_ssm_conv(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    // Only support FP32 for now
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Check IO tensor shapes and dims
    if (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        return false; // src0 should be effectively 3D
    }

    const int d_conv = src1->ne[0];
    const int d_inner = src0->ne[1];
    const int n_t = dst->ne[1];
    const int n_s = dst->ne[2];

    if (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        return false;
    }
    if (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        return false;
    }
    if (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        return false;
    }
    if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->nb[1] != src0->ne[0] * sizeof(float) || src1->nb[1] != src1->ne[0] * sizeof(float)) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_im2col(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    const bool is_2D = ((const int32_t *) op->op_params)[6] == 1;
    if (!is_2D) {
        return false;
    }

    // For now support F32->F32 and F32->F16 only.
    if (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        return false;
    }

    if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    // For now keep padded OPs on CPU. Will revisit once we expand coverage past patch-embed OPs.
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    if (p0 != 0 || p1 != 0) {
        return false;
    }

    GGML_UNUSED(sess);
    return true;
}

static bool ggml_hexagon_supported_pad(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_cumsum(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_diag(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    // diag only supports F32 currently
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Input must have ne[1] == 1 (vector input)
    if (src0->ne[1] != 1) {
        return false;
    }

    // Output must be square in first two dimensions
    if (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_solve_tri(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // A
    const struct ggml_tensor * src1 = op->src[1]; // B
    const struct ggml_tensor * dst  = op;         // X

    if (!src0 || !src1) {
        return false;
    }

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->ne[0] != src0->ne[1]) {
        return false;
    }

    if (src0->ne[1] != src1->ne[1]) {
        return false;
    }

    if (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        return false;
    }

    if (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        return false;
    }

    return true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_tri(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    if (src0->type != GGML_TYPE_F32) { return false; }
    if (dst->type  != GGML_TYPE_F32) { return false; }
    if (!ggml_are_same_shape(src0, dst)) { return false; }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { return false; }

    return true;

    GGML_UNUSED(sess);
}

static const char * ggml_backend_hexagon_name(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    return sess->c_name();
}

static void ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

// ---------------------------------------------------------------------------------------------
// Heterogeneous decode attention (prototype). One shared 8 MB rpcmem buffer per session holds the
// per-slot ready/done sequence words and the GPU's partials; it is attached to each tagged decode
// FLASH_ATTN_EXT node as src[7]. A relay thread pre-enqueues one GPU kernel per tagged node for
// every graph_compute (kernels spin on a fine-grain SVM word), then copies each slot's ready word,
// raised by the HTP op, into SVM. The GPU never polls rpcmem (Adreno does not see mid-kernel
// updates there), the HTP polls the GPU's done word with a cache invalidate per read.
// ---------------------------------------------------------------------------------------------
#ifdef GGML_HEXAGON_HETERO

static const char * ggml_hexagon_hetero_cl_src = R"CL(
#ifndef FA_D
#define FA_D 128
#endif
#ifndef FA_G
#define FA_G 2
#endif
#define HTP_M_INIT (-10000.0f)

// One work-group per (KV head, KV split), FA_D lanes; the FA_G query heads of the KV head share
// every K/V read. Q comes from and the partials go to fine-grain SVM: inside a long in-order
// chain, writes through the ION alias only reach memory at command-buffer boundaries, SVM is
// coherent immediately. Partial per (row, split): 128-byte header (M, S), then acc[FA_D] f32.
__kernel void fa_dec_gqa(__global const float * q_svm,
                         __global const uchar * kb, uint k_off, __global const uchar * vb, uint v_off,
                         __global const uchar * mb, uint m_off, uint has_mask,
                         __global float * parts_svm,
                         uint nbk1, uint nbk2, uint nbv1, uint nbv2,
                         uint n_kv, uint span, uint nsplit, float scale, uint part_stride_f,
                         __global uint * ctl, uint want, uint max_spin, uint n_groups) {
    const int t   = get_local_id(0);
    const int kvh = get_group_id(1);
    const int sp  = get_group_id(2);

    __local float Q_l[FA_G][FA_D];
    __local float S_l[FA_G][FA_D];
    __local float sh_a[FA_G];
    __local float sh_m[FA_G];
    __local float sh_l[FA_G];

    if (t == 0) {
        uint it = 0;
        while (atomic_load_explicit((volatile __global atomic_uint *) ctl, memory_order_acquire, memory_scope_all_svm_devices) != want && it < max_spin) { it++; }
        if (kvh == 0 && sp == 0) {
            atomic_store_explicit((volatile __global atomic_uint *)(ctl + 3), it, memory_order_relaxed, memory_scope_all_svm_devices);
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int g = 0; g < FA_G; ++g) {
        Q_l[g][t] = q_svm[(kvh * FA_G + g) * FA_D + t] * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int kv_begin = sp * span;
    const int kv_end   = min((int) n_kv, kv_begin + (int) span);

    float m_run = -INFINITY, l_run = 0.0f;
    float o_acc[FA_G];
    for (int g = 0; g < FA_G; ++g) o_acc[g] = 0.0f;

    if (kv_begin < kv_end) {
        for (int bs = kv_begin; bs < kv_end; bs += FA_D) {
            const int blk_n = min(FA_D, kv_end - bs);
            float s[FA_G];
            for (int g = 0; g < FA_G; ++g) s[g] = -INFINITY;
            if (t < blk_n) {
                __global const half * krow = (__global const half *)(kb + k_off + (ulong)(bs + t) * nbk1 + kvh * nbk2);
                float8 acc[FA_G];
                for (int g = 0; g < FA_G; ++g) acc[g] = (float8)(0.0f);
                #pragma unroll
                for (int d8 = 0; d8 < FA_D / 8; ++d8) {
                    const float8 k8 = convert_float8(vload8(d8, krow));
                    for (int g = 0; g < FA_G; ++g) acc[g] += k8 * vload8(d8, Q_l[g]);
                }
                float mval = 0.0f;
                if (has_mask) mval = (float)((__global const half *)(mb + m_off))[bs + t];
                for (int g = 0; g < FA_G; ++g) {
                    const float8 a = acc[g];
                    s[g] = (a.s0 + a.s1 + a.s2 + a.s3 + a.s4 + a.s5 + a.s6 + a.s7) + mval;
                }
            }
            for (int g = 0; g < FA_G; ++g) S_l[g][t] = s[g];
            barrier(CLK_LOCAL_MEM_FENCE);

            if (t < FA_G) {
                float m_t = -INFINITY;
                for (int c = 0; c < FA_D; ++c) m_t = fmax(m_t, S_l[t][c]);
                const float m_new = fmax(m_run, m_t);
                const float a = (m_run == -INFINITY) ? 0.0f : native_exp(m_run - m_new);
                float l = 0.0f;
                for (int c = 0; c < FA_D; ++c) {
                    const float sc = S_l[t][c];
                    const float p = (isinf(sc) || m_new == -INFINITY) ? 0.0f : native_exp(sc - m_new);
                    S_l[t][c] = p;
                    l += p;
                }
                l_run = a * l_run + l;
                m_run = m_new;
                sh_a[t] = a;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            {
                float pv[FA_G];
                for (int g = 0; g < FA_G; ++g) pv[g] = 0.0f;
                __global const half * vrow = (__global const half *)(vb + v_off + (ulong) bs * nbv1 + kvh * nbv2) + t;
                const int vstep = nbv1 / 2;
                #pragma unroll 8
                for (int c = 0; c < blk_n; ++c) {
                    const float vf = (float) vrow[(ulong) c * vstep];
                    for (int g = 0; g < FA_G; ++g) pv[g] += S_l[g][c] * vf;
                }
                for (int g = 0; g < FA_G; ++g) o_acc[g] = o_acc[g] * sh_a[g] + pv[g];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (t < FA_G) { sh_m[t] = (kv_begin < kv_end) ? m_run : HTP_M_INIT; sh_l[t] = l_run; }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int g = 0; g < FA_G; ++g) {
        const int row = kvh * FA_G + g;
        __global float * part = parts_svm + ((ulong) row * nsplit + sp) * part_stride_f;
        if (t == 0) { part[0] = sh_m[g]; part[1] = sh_l[g]; }
        part[32 + t] = o_acc[g];
    }
    // last work-group of the kernel publishes done (SVM, host-visible immediately)
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (t == 0) {
        const uint old = atomic_fetch_add_explicit((volatile __global atomic_uint *)(ctl + 2), 1u, memory_order_acq_rel, memory_scope_all_svm_devices);
        if (old == n_groups - 1) {
            atomic_store_explicit((volatile __global atomic_uint *)(ctl + 1), want, memory_order_seq_cst, memory_scope_all_svm_devices);
        }
    }
}
)CL";

struct ggml_hexagon_hetero_job {
    const ggml_tensor * node = nullptr;
    uint32_t n_blocks = 0, gpu_blocks = 0;   // KV blocks total / owned by the GPU (adapted per token)
    uint32_t slot = 0, want = 0;
    cl_mem q = nullptr, k = nullptr, v = nullptr, m = nullptr;
    uint32_t q_off = 0, k_off = 0, v_off = 0, m_off = 0, has_mask = 0, p_off = 0;
    uint32_t nbq2 = 0, nbk1 = 0, nbk2 = 0, nbv1 = 0, nbv2 = 0;
    uint32_t n_kv_gpu = 0, span = 0, nsplit = 0, part_stride = 0, done_off = 0;
    uint32_t n_kv_heads = 0, D = 0, n_heads = 0;
    const uint8_t * q_host = nullptr;   // Q rows in the compute buffer (host VA), copied into SVM per token
    float    scale = 0.0f;
};

// SVM block layout (fine-grain, coherent): control words, then per-slot Q, then per-slot partials
#define HETERO_SVM_CTL_BYTES   4096
#define HETERO_SVM_Q_BYTES     16384
#define HETERO_SVM_PARTS_BYTES HTP_FA_HETERO_PART_SLOT
#define HETERO_SVM_BYTES       (HETERO_SVM_CTL_BYTES + HTP_FA_HETERO_MAX_SLOTS * (HETERO_SVM_Q_BYTES + HETERO_SVM_PARTS_BYTES))
static inline uint32_t * hetero_svm_ctl(ggml_hexagon_hetero * h, uint32_t slot);
static inline float *    hetero_svm_q(ggml_hexagon_hetero * h, uint32_t slot);
static inline float *    hetero_svm_parts(ggml_hexagon_hetero * h, uint32_t slot);

struct ggml_hexagon_hetero {
    ggml_hexagon_session *  sess = nullptr;
    ggml_backend_buffer_t   buf  = nullptr;   // control words + GPU partials, attached as src[7]
    uint8_t *               base = nullptr;
    int                     fd   = -1;
    ggml_context *          tctx = nullptr;
    ggml_tensor *           ctrl = nullptr;

    cl_platform_id   plat = nullptr;
    cl_device_id     dev  = nullptr;
    cl_context       ctx  = nullptr;
    cl_command_queue q    = nullptr;
    cl_program       prog = nullptr;
    cl_kernel        k_fa = nullptr;
    uint8_t *        svm  = nullptr;          // HETERO_SVM_BYTES, see hetero_svm_*
    int              G = 0, D = 0;            // shape the program was compiled for

    std::unordered_map<int, cl_mem>                                  aliases;   // by rpcmem fd
    std::unordered_map<const ggml_tensor *, ggml_hexagon_hetero_job> jobs;
    std::vector<uint32_t>                                            seq;       // per slot

    std::thread             relay;
    std::mutex              mu;
    std::condition_variable cv;
    std::deque<std::vector<ggml_hexagon_hetero_job>> batches;
    bool                    stop = false;
    uint64_t                n_batches = 0, n_jobs = 0, n_ready_timeouts = 0, n_gpu_timeouts = 0;
};

static inline uint32_t * hetero_svm_ctl(ggml_hexagon_hetero * h, uint32_t slot)   { return (uint32_t *) (h->svm + slot * 16); }
static inline float *    hetero_svm_q(ggml_hexagon_hetero * h, uint32_t slot)     { return (float *) (h->svm + HETERO_SVM_CTL_BYTES + (size_t) slot * HETERO_SVM_Q_BYTES); }
static inline float *    hetero_svm_parts(ggml_hexagon_hetero * h, uint32_t slot) { return (float *) (h->svm + HETERO_SVM_CTL_BYTES + (size_t) HTP_FA_HETERO_MAX_SLOTS * HETERO_SVM_Q_BYTES + (size_t) slot * HETERO_SVM_PARTS_BYTES); }

static inline void hetero_dc_civac(const void * p, size_t n) {
#if defined(__aarch64__)
    uintptr_t a = (uintptr_t) p & ~(uintptr_t) 63;
    const uintptr_t e = (uintptr_t) p + n;
    for (; a < e; a += 64) asm volatile("dc civac, %0" : : "r"(a) : "memory");
    asm volatile("dsb sy" : : : "memory");
#else
    (void) p; (void) n;
#endif
}

static bool hetero_cl_check(cl_int err, const char * what) {
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: hetero: %s failed (%d)\n", what, err);
        return false;
    }
    return true;
}

static cl_mem ggml_hexagon_hetero_alias(ggml_hexagon_hetero * h, ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto it = h->aliases.find(sbuf->fd);
    if (it != h->aliases.end()) {
        return it->second;
    }
    cl_mem_ion_host_ptr ion = {};
    ion.ext_host_ptr.allocation_type  = CL_MEM_ION_HOST_PTR_QCOM;
    ion.ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
    ion.ion_filedesc = sbuf->fd;
    ion.ion_hostptr  = sbuf->base;
    const size_t size = sbuf->size & ~(size_t) 4095;   // never past the allocation
    cl_int err;
    cl_mem mem = clCreateBuffer(h->ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM, size, &ion, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: hetero: ION alias of buffer fd %d size %zu failed (%d)\n", sbuf->fd, size, err);
        return nullptr;
    }
    HEX_VERBOSE("ggml-hex: hetero: aliased buffer fd %d base %p size %zu into OpenCL\n", sbuf->fd, (void *) sbuf->base, size);
    h->aliases[sbuf->fd] = mem;
    return mem;
}

static void ggml_hexagon_hetero_relay_main(ggml_hexagon_hetero * h);

static bool ggml_hexagon_hetero_init(ggml_hexagon_session * sess) {
    auto h = new ggml_hexagon_hetero();
    h->sess = sess;

    h->buf = ggml_backend_buft_alloc_buffer(&sess->buffer_type, HTP_FA_HETERO_BUF_SIZE);
    if (!h->buf) {
        GGML_LOG_ERROR("ggml-hex: hetero: control buffer alloc failed\n");
        delete h;
        return false;
    }
    ggml_backend_buffer_set_usage(h->buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(h->buf->context);
    h->base = sbuf->base;
    h->fd   = sbuf->fd;
    memset(h->base, 0, HTP_FA_HETERO_BUF_SIZE);
    hetero_dc_civac(h->base, HTP_FA_HETERO_BUF_SIZE);

    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    h->tctx = ggml_init(ip);
    h->ctrl = ggml_new_tensor_1d(h->tctx, GGML_TYPE_F32, HTP_FA_HETERO_BUF_SIZE / 4);
    h->ctrl->buffer = h->buf;
    h->ctrl->data   = h->base;
    ggml_set_name(h->ctrl, "hexagon_hetero_ctrl");

    cl_int err; cl_uint n = 0;
    if (!hetero_cl_check(clGetPlatformIDs(1, &h->plat, &n), "clGetPlatformIDs") || !n) { delete h; return false; }
    if (!hetero_cl_check(clGetDeviceIDs(h->plat, CL_DEVICE_TYPE_GPU, 1, &h->dev, &n), "clGetDeviceIDs") || !n) { delete h; return false; }
    cl_device_svm_capabilities svm = 0;
    clGetDeviceInfo(h->dev, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm), &svm, nullptr);
    if (!(svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) || !(svm & CL_DEVICE_SVM_ATOMICS)) {
        GGML_LOG_ERROR("ggml-hex: hetero: GPU lacks fine-grain SVM atomics\n");
        delete h; return false;
    }
    h->ctx = clCreateContext(nullptr, 1, &h->dev, nullptr, nullptr, &err);
    if (!hetero_cl_check(err, "clCreateContext")) { delete h; return false; }
    cl_queue_properties qprops[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    h->q = clCreateCommandQueueWithProperties(h->ctx, h->dev, qprops, &err);
    if (!hetero_cl_check(err, "clCreateCommandQueueWithProperties")) { delete h; return false; }
    h->svm = (uint8_t *) clSVMAlloc(h->ctx, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, HETERO_SVM_BYTES, 0);
    if (!h->svm) { GGML_LOG_ERROR("ggml-hex: hetero: clSVMAlloc(%zu) failed\n", (size_t) HETERO_SVM_BYTES); delete h; return false; }
    memset(h->svm, 0, HETERO_SVM_BYTES);
    h->seq.assign(HTP_FA_HETERO_MAX_SLOTS, 0);

    if (!ggml_hexagon_hetero_alias(h, h->buf)) { delete h; return false; }

    h->relay = std::thread(ggml_hexagon_hetero_relay_main, h);
    sess->hetero = h;
    GGML_LOG_INFO("ggml-hex: %s hetero decode attention enabled: GPU share %.2f, span %d, relay cpu %d\n",
                  sess->c_name(), opt_hetero_frac, opt_hetero_span, opt_hetero_cpu);
    return true;
}

static bool ggml_hexagon_hetero_compile(ggml_hexagon_hetero * h, int D, int G) {
    if (h->prog) {
        return h->D == D && h->G == G;
    }
    cl_int err;
    h->prog = clCreateProgramWithSource(h->ctx, 1, &ggml_hexagon_hetero_cl_src, nullptr, &err);
    if (!hetero_cl_check(err, "clCreateProgramWithSource")) return false;
    char opts[128];
    snprintf(opts, sizeof(opts), "-cl-std=CL2.0 -DFA_D=%d -DFA_G=%d", D, G);
    err = clBuildProgram(h->prog, 1, &h->dev, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t ln = 0;
        clGetProgramBuildInfo(h->prog, h->dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ln);
        std::string log(ln, '\0');
        clGetProgramBuildInfo(h->prog, h->dev, CL_PROGRAM_BUILD_LOG, ln, log.data(), nullptr);
        GGML_LOG_ERROR("ggml-hex: hetero: kernel build failed (%d):\n%s\n", err, log.c_str());
        clReleaseProgram(h->prog); h->prog = nullptr;
        return false;
    }
    h->k_fa   = clCreateKernel(h->prog, "fa_dec_gqa", &err); if (!hetero_cl_check(err, "clCreateKernel fa_dec_gqa")) return false;
    h->D = D; h->G = G;
    return true;
}

// Tag a decode FLASH_ATTN_EXT node for the split and build its GPU job. Returns false (node left
// untouched) when the shape or the resources do not fit the prototype.
static bool ggml_hexagon_hetero_prepare(ggml_hexagon_session * sess, ggml_tensor * n, const struct htp_fa_kernel_params * kp, int slot) {
    ggml_hexagon_hetero * h = sess->hetero;
    const ggml_tensor * q = n->src[0], * k = n->src[1], * v = n->src[2], * m = n->src[3];
    n->op_params[HTP_FA_HETERO_OPP_GPU_BLOCKS] = 0;

    if (kp->kernel_type != HTP_FA_KERNEL_HVX || !kp->u.hvx.split_kv) return false;
    if (q->ne[1] != 1 || q->ne[3] != 1 || slot >= HTP_FA_HETERO_MAX_SLOTS) return false;
    const int DK = (int) q->ne[0], DV = (int) v->ne[0];
    if (DK != DV || (DK != 128 && DK != 64)) return false;
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || (m && m->type != GGML_TYPE_F16)) return false;
    if (n->src[4] || n->src[5] || n->src[6]) return false;   // sinks / sparse: not in the prototype
    const int nek2 = (int) k->ne[2], neq2 = (int) q->ne[2];
    if (nek2 == 0 || neq2 % nek2 != 0) return false;
    const int G = neq2 / nek2;
    const uint32_t n_blocks = kp->n_kv_blocks;
    const uint32_t gpu_blocks = (uint32_t) (opt_hetero_frac * (float) n_blocks);
    if (gpu_blocks < 2 || gpu_blocks >= n_blocks) return false;
    const uint32_t gpu_kv = gpu_blocks * 64;
    if (gpu_kv > (uint32_t) k->ne[1]) return false;
    const uint32_t nsplit = (gpu_kv + opt_hetero_span - 1) / opt_hetero_span;
    const uint32_t part_stride = 128 + hex_round_up(DV * 2, 128) * 2;
    if ((size_t) neq2 * nsplit * part_stride > HTP_FA_HETERO_PART_SLOT) return false;
    if ((size_t) neq2 * DK * sizeof(float) > HETERO_SVM_Q_BYTES) return false;
    if (!ggml_backend_buffer_is_hexagon(q->buffer) || !ggml_backend_buffer_is_hexagon(k->buffer) ||
        !ggml_backend_buffer_is_hexagon(v->buffer) || (m && !ggml_backend_buffer_is_hexagon(m->buffer))) return false;
    if (!ggml_hexagon_hetero_compile(h, DK, G)) return false;

    ggml_hexagon_hetero_job job;
    job.node = n; job.slot = (uint32_t) slot;
    job.q = ggml_hexagon_hetero_alias(h, q->buffer);
    job.k = ggml_hexagon_hetero_alias(h, k->buffer);
    job.v = ggml_hexagon_hetero_alias(h, v->buffer);
    job.m = m ? ggml_hexagon_hetero_alias(h, m->buffer) : h->aliases[h->fd];
    if (!job.q || !job.k || !job.v || !job.m) return false;
    auto off = [](const ggml_tensor * t) { return (uint32_t) ((const uint8_t *) t->data - static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context)->base); };
    job.q_off = off(q); job.k_off = off(k); job.v_off = off(v); job.m_off = m ? off(m) : 0; job.has_mask = m ? 1 : 0;
    job.p_off = HTP_FA_HETERO_PARTS_OFF + (uint32_t) slot * HTP_FA_HETERO_PART_SLOT;
    job.nbq2 = (uint32_t) q->nb[2]; job.nbk1 = (uint32_t) k->nb[1]; job.nbk2 = (uint32_t) k->nb[2];
    job.nbv1 = (uint32_t) v->nb[1]; job.nbv2 = (uint32_t) v->nb[2];
    job.n_kv_gpu = gpu_kv; job.span = (uint32_t) opt_hetero_span; job.nsplit = nsplit; job.part_stride = part_stride;
    job.n_blocks = n_blocks; job.gpu_blocks = gpu_blocks;
    job.done_off = (uint32_t) slot * HTP_FA_HETERO_SLOT_STRIDE + 128;
    job.n_kv_heads = (uint32_t) nek2; job.D = (uint32_t) DK; job.n_heads = (uint32_t) neq2;
    job.q_host = (const uint8_t *) q->data;
    memcpy(&job.scale, &n->op_params[0], sizeof(float));

    n->op_params[HTP_FA_HETERO_OPP_GPU_BLOCKS] = (int32_t) gpu_blocks;
    n->op_params[HTP_FA_HETERO_OPP_SLOT]       = slot;
    n->op_params[HTP_FA_HETERO_OPP_GPU_NSPLIT] = (int32_t) nsplit;
    n->src[7] = h->ctrl;
    h->jobs[n] = job;
    if (opt_verbose || slot == 0) {
        GGML_LOG_INFO("ggml-hex: hetero: node %s slot %d: GPU blocks %u of %u (%u keys, %u splits), HTP blocks %u, D %d G %d\n",
                      n->name, slot, gpu_blocks, n_blocks, gpu_kv, nsplit, n_blocks - gpu_blocks, DK, G);
    }
    return true;
}

// Called once per graph_compute with the nodes about to be queued, in execution order.
static void ggml_hexagon_hetero_post(ggml_hexagon_session * sess, const std::vector<htp_opnode> & nodes) {
    ggml_hexagon_hetero * h = sess->hetero;
    std::vector<ggml_hexagon_hetero_job> batch;
    {
        std::lock_guard<std::mutex> lk(h->mu);
        for (const auto & node : nodes) {
            if (node.opcode != HTP_OP_FLASH_ATTN_EXT || node.node->op_params[HTP_FA_HETERO_OPP_GPU_BLOCKS] <= 0) continue;
            auto it = h->jobs.find(node.node);
            if (it == h->jobs.end()) continue;
            ggml_hexagon_hetero_job job = it->second;
            job.want = ++h->seq[job.slot];
            batch.push_back(job);
        }
        if (batch.empty()) return;
        h->batches.push_back(std::move(batch));
    }
    h->cv.notify_one();
}

static void ggml_hexagon_hetero_relay_main(ggml_hexagon_hetero * h) {
#if defined(__linux__)
    if (opt_hetero_cpu >= 0) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(opt_hetero_cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
#endif
    for (;;) {
        std::vector<ggml_hexagon_hetero_job> batch;
        {
            std::unique_lock<std::mutex> lk(h->mu);
            h->cv.wait(lk, [&] { return h->stop || !h->batches.empty(); });
            if (h->stop && h->batches.empty()) return;
            batch = std::move(h->batches.front());
            h->batches.pop_front();
        }
        // 1. the whole chain goes on the in-order queue up front; each kernel spins on its slot's SVM word
        const auto t_batch0 = std::chrono::steady_clock::now();
        for (const auto & j : batch) {
            uint32_t * ctl = hetero_svm_ctl(h, j.slot);
            __atomic_store_n(&ctl[2], 0u, __ATOMIC_SEQ_CST);   // work-group counter (kernel not yet released)
            __atomic_store_n(&ctl[3], 0u, __ATOMIC_SEQ_CST);
            cl_uint a_max = 300000u;   // ~85 ms of spin; then compute anyway (never trips the KGSL hang watchdog)
            cl_uint part_stride_f = j.part_stride / 4;
            cl_uint n_groups = j.n_kv_heads * j.nsplit;
            int i = 0;
            clSetKernelArgSVMPointer(h->k_fa, i++, hetero_svm_q(h, j.slot));
            clSetKernelArg(h->k_fa, i++, sizeof(cl_mem), &j.k);  clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &j.k_off);
            clSetKernelArg(h->k_fa, i++, sizeof(cl_mem), &j.v);  clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &j.v_off);
            clSetKernelArg(h->k_fa, i++, sizeof(cl_mem), &j.m);  clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &j.m_off);
            clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &j.has_mask);
            clSetKernelArgSVMPointer(h->k_fa, i++, hetero_svm_parts(h, j.slot));
            clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &j.nbk1); clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &j.nbk2);
            clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &j.nbv1); clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &j.nbv2);
            clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &j.n_kv_gpu); clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &j.span);
            clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &j.nsplit);   clSetKernelArg(h->k_fa, i++, sizeof(cl_float), &j.scale);
            clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &part_stride_f);
            clSetKernelArgSVMPointer(h->k_fa, i++, ctl);
            clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &j.want);
            clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &a_max);
            clSetKernelArg(h->k_fa, i++, sizeof(cl_uint), &n_groups);
            size_t gsz[3] = { (size_t) j.D, (size_t) j.n_kv_heads, (size_t) j.nsplit }, lsz[3] = { (size_t) j.D, 1, 1 };
            cl_int err = clEnqueueNDRangeKernel(h->q, h->k_fa, 3, nullptr, gsz, lsz, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) { GGML_LOG_ERROR("ggml-hex: hetero: enqueue fa failed (%d)\\n", err); }
        }
        clFlush(h->q);
        const auto t_enq = std::chrono::steady_clock::now();

        // 2. relay, in op order: HTP raises ready (rpcmem) -> copy Q into SVM, release the kernel ->
        //    GPU sets done (SVM) -> copy partials into rpcmem, set the HTP's done word.
        for (const auto & j : batch) {
            uint32_t * ctl = hetero_svm_ctl(h, j.slot);
            volatile uint32_t * ready = (volatile uint32_t *) (h->base + j.slot * HTP_FA_HETERO_SLOT_STRIDE);
            volatile uint32_t * done  = (volatile uint32_t *) (h->base + j.done_off);
            const auto t0 = std::chrono::steady_clock::now();
            bool seen = false;
            for (;;) {
                hetero_dc_civac((const void *) ready, 4);
                if (__atomic_load_n(ready, __ATOMIC_ACQUIRE) == j.want) { seen = true; break; }
                if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(500)) break;
            }
            if (!seen) h->n_ready_timeouts++;
            const auto t_ready = std::chrono::steady_clock::now();
            // Q rows -> SVM (the HTP flushed them before raising ready; the CPU<->HTP path is IO-coherent)
            float * qs = hetero_svm_q(h, j.slot);
            for (uint32_t hh = 0; hh < j.n_heads; ++hh) {
                memcpy(qs + (size_t) hh * j.D, j.q_host + (size_t) hh * j.nbq2, (size_t) j.D * sizeof(float));
            }
            __atomic_store_n(&ctl[0], j.want, __ATOMIC_SEQ_CST);   // release the kernel
            // GPU done (SVM)
            bool gdone = false;
            for (;;) {
                if (__atomic_load_n(&ctl[1], __ATOMIC_ACQUIRE) == j.want) { gdone = true; break; }
                if (std::chrono::steady_clock::now() - t_ready > std::chrono::milliseconds(200)) break;
            }
            const auto t_gdone = std::chrono::steady_clock::now();
            if (!gdone) h->n_gpu_timeouts++;
            // partials SVM -> rpcmem, then the HTP's done word
            uint8_t * dst = h->base + HTP_FA_HETERO_PARTS_OFF + (size_t) j.slot * HTP_FA_HETERO_PART_SLOT;
            const size_t parts_bytes = (size_t) j.n_heads * j.nsplit * j.part_stride;
            memcpy(dst, hetero_svm_parts(h, j.slot), parts_bytes);
            hetero_dc_civac(dst, parts_bytes);
            __atomic_store_n(done, j.want, __ATOMIC_SEQ_CST);
            hetero_dc_civac((const void *) done, 4);
            if (opt_hetero_adapt && seen && gdone) {
                // The HTP wrote how long it waited for the previous token's done (this token's value
                // is not there yet). Move the share one block toward a small positive wait.
                volatile uint32_t * wait_w = (volatile uint32_t *) (h->base + j.slot * HTP_FA_HETERO_SLOT_STRIDE + 8);
                hetero_dc_civac((const void *) wait_w, 4);
                const uint32_t wait_us = __atomic_load_n(wait_w, __ATOMIC_ACQUIRE);
                std::lock_guard<std::mutex> lk(h->mu);
                auto it = h->jobs.find(j.node);
                if (it != h->jobs.end()) {
                    ggml_hexagon_hetero_job & jj = it->second;
                    uint32_t gb = jj.gpu_blocks;
                    if (wait_us > 40) {
                        // GPU late by wait_us; an HTP block costs ~8.5 us, so moving wait/17 blocks
                        // roughly halves the imbalance per token. Converges in a handful of tokens.
                        uint32_t step = std::min<uint32_t>(std::max<uint32_t>(1, wait_us / 17), 16);
                        gb = (gb > step + 2) ? gb - step : 2;
                    } else if (wait_us < 5 && gb + 2 < jj.n_blocks) {
                        gb += 1;                                                     // GPU early: grow slowly
                    }
                    if (gb != jj.gpu_blocks) {
                        const uint32_t gpu_kv = gb * 64;
                        const uint32_t nsplit = (gpu_kv + jj.span - 1) / jj.span;
                        if ((size_t) jj.n_heads * nsplit * jj.part_stride <= HTP_FA_HETERO_PART_SLOT) {
                            jj.gpu_blocks = gb; jj.n_kv_gpu = gpu_kv; jj.nsplit = nsplit;
                            ggml_tensor * n = const_cast<ggml_tensor *>(jj.node);
                            n->op_params[HTP_FA_HETERO_OPP_GPU_BLOCKS] = (int32_t) gb;
                            n->op_params[HTP_FA_HETERO_OPP_GPU_NSPLIT] = (int32_t) nsplit;
                        }
                    }
                }
            }
            if (opt_verbose > 1 || (!seen || !gdone) ) {
                const auto t_end = std::chrono::steady_clock::now();
                GGML_LOG_DEBUG("ggml-hex: hetero relay: batch %llu slot %u want %u: ready %s after %.0f us; gpu %s %.0f us after release (spin %u); copy+publish %.0f us\n",
                               (unsigned long long) h->n_batches, j.slot, j.want, seen ? "seen" : "TIMEOUT",
                               std::chrono::duration<double, std::micro>(t_ready - t0).count(), gdone ? "done" : "TIMEOUT",
                               std::chrono::duration<double, std::micro>(t_gdone - t_ready).count(), __atomic_load_n(&ctl[3], __ATOMIC_ACQUIRE),
                               std::chrono::duration<double, std::micro>(t_end - t_gdone).count());
            }
        }
        if (opt_verbose) {
            hetero_dc_civac(h->base + HTP_FA_HETERO_STATUS_OFF, 64);
            const uint32_t dsp_to = *(volatile uint32_t *) (h->base + HTP_FA_HETERO_STATUS_OFF);
            uint32_t gb_min = ~0u, gb_max = 0, wait_sum = 0;
            {
                std::lock_guard<std::mutex> lk(h->mu);
                for (const auto & j : batch) {
                    auto it = h->jobs.find(j.node);
                    if (it != h->jobs.end()) { gb_min = std::min(gb_min, it->second.gpu_blocks); gb_max = std::max(gb_max, it->second.gpu_blocks); }
                    volatile uint32_t * wait_w = (volatile uint32_t *) (h->base + j.slot * HTP_FA_HETERO_SLOT_STRIDE + 8);
                    hetero_dc_civac((const void *) wait_w, 4);
                    wait_sum += __atomic_load_n(wait_w, __ATOMIC_ACQUIRE);
                }
            }
            GGML_LOG_DEBUG("ggml-hex: hetero relay: batch %llu: %zu jobs in %.0f us (enqueue %.0f us); GPU blocks %u..%u of %u; HTP wait for GPU avg %u us; ready/GPU/DSP timeouts %llu/%llu/%u\n",
                           (unsigned long long) h->n_batches, batch.size(),
                           std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_batch0).count(),
                           std::chrono::duration<double, std::micro>(t_enq - t_batch0).count(),
                           gb_min, gb_max, batch.empty() ? 0u : batch[0].n_blocks, batch.empty() ? 0u : wait_sum / (uint32_t) batch.size(),
                           (unsigned long long) h->n_ready_timeouts, (unsigned long long) h->n_gpu_timeouts, dsp_to);
        }
        h->n_batches++;
        h->n_jobs += batch.size();
    }
}

static void ggml_hexagon_hetero_free(ggml_hexagon_session * sess) {
    ggml_hexagon_hetero * h = sess->hetero;
    if (!h) return;
    {
        std::lock_guard<std::mutex> lk(h->mu);
        h->stop = true;
    }
    h->cv.notify_one();
    if (h->relay.joinable()) h->relay.join();
    if (h->q) clFinish(h->q);
    uint32_t dsp_timeouts = 0;
    if (h->base) {
        hetero_dc_civac(h->base + HTP_FA_HETERO_STATUS_OFF, 64);
        dsp_timeouts = *(volatile uint32_t *) (h->base + HTP_FA_HETERO_STATUS_OFF);
    }
    GGML_LOG_INFO("ggml-hex: %s hetero: %llu graphs, %llu GPU jobs, ready timeouts %llu, GPU done timeouts %llu, DSP done timeouts %u\n",
                  sess->c_name(), (unsigned long long) h->n_batches, (unsigned long long) h->n_jobs,
                  (unsigned long long) h->n_ready_timeouts, (unsigned long long) h->n_gpu_timeouts, dsp_timeouts);
    for (auto & kv : h->aliases) clReleaseMemObject(kv.second);
    if (h->svm) clSVMFree(h->ctx, h->svm);
    if (h->k_fa) clReleaseKernel(h->k_fa);
    if (h->prog) clReleaseProgram(h->prog);
    if (h->q) clReleaseCommandQueue(h->q);
    if (h->ctx) clReleaseContext(h->ctx);
    if (h->tctx) ggml_free(h->tctx);
    if (h->buf) ggml_backend_buffer_free(h->buf);
    delete h;
    sess->hetero = nullptr;
}

// ---------------------------------------------------------------------------------------------
// Heterogeneous PREFILL split (prototype, GGML_HEXAGON_FA_FOLD=1; the graph side is
// LLAMA_SPARSE_ATTN=thr:<c> with LLAMA_SPARSE_ATTN_CSTAR>0, optionally LLAMA_SPARSE_ATTN_HEADSTART).
//
// The graph hands every prefill FLASH_ATTN_EXT its shared block list (src[5]/src[6]) and the
// exception membership (src[8], F32 0/1 [NBk, R, NBq/R, n_kv_heads]). This sidecar attaches ONE
// fold buffer as src[7] (header + the GPU's (m, l, acc) partial + handshake words + the GPU's
// tables and staged K^T) and runs a relay thread: when the HMX op raises `ready` (after flushing Q,
// K, V, the mask and the membership), the relay scans the membership into per-head compact block
// tables, stages the exception blocks' K rows transposed, and launches the exception kernel
// (examples/fa-exc-gpu's, over ION aliases of the hexagon buffers) once per KV head in ascending
// order, writing done[head] as each completes. The op folds the partial in its store threads
// (HTP_FA_FOLD_F_INSTORE), runs exception-free tiles first and waits once per head
// (HTP_FA_FOLD_F_STAGED). One slot: the DSP runs one op at a time and the next op's ready comes
// after this op's fold, so the buffer is never live for two ops.
// ---------------------------------------------------------------------------------------------

static const char * ggml_hexagon_hfold_cl_src = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define FA2_BQ  32
#define FA2_BK  64
#ifndef FA_D
#define FA_D 128
#endif
#define M_EMPTY (-10000.0f)

// Submission trigger: a one-work-item kernel that references the DSP-written inputs. Submitted at
// ready on a second queue: it is the SUBMISSION that makes the driver's cache maintenance for
// these buffers happen (pre-enqueued kernels read stale lines of the previous op's Q otherwise).
__kernel void touch(__global const uchar * a, __global const uchar * b, __global const uchar * c,
                    __global const uchar * d, __global const uchar * e, __global uint * out) {
    if (get_global_id(0) == 0) out[8] = (uint) a[0] + b[0] + c[0] + d[0] + e[0];
}

// Keep-alive control: touch a buffer for a while. Memory-bound like the exception kernel.
__kernel void keepalive(__global float4 * p, const int n4, const int iters, __global float * out) {
    const int g = get_global_id(0), gs = get_global_size(0);
    float4 acc = (float4)0;
    for (int it = 0; it < iters; ++it) {
        for (int i = g; i < n4; i += gs) acc += p[i];
    }
    if (acc.s0 == 12345.678f) out[g] = acc.s1 + acc.s2 + acc.s3;   // never true; keeps the loads alive
}

// The chain's gate: one work-item spins on the SVM ready mirror until it reads `want`, then
// publishes svm[1] = want (or want | GATE_TIMEOUT after max_spins). The kernels behind it on the
// in-order queue run only when svm[1] == want, so a chain whose op never came does nothing.
#define GATE_TIMEOUT 0x80000000u
#define STAGE_NBK_MAX 256
#ifndef KT_BUF
#define KT_BUF 0    // 1: K^T is written and read through the buffer alias (plain loads), not the image
#endif
#ifndef ACQ_FENCE
#define ACQ_FENCE 0 // 1: acquire fence at all-SVM-devices scope at kernel entry (drop stale cached lines of DSP-written inputs)
#endif
#if ACQ_FENCE
#define ENTRY_ACQUIRE() atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE, memory_order_acquire, memory_scope_all_svm_devices)
#else
#define ENTRY_ACQUIRE() do {} while (0)
#endif
#ifndef Q_ATOMIC
#define Q_ATOMIC 0  // 1: Q rows through device-scope atomic loads (bypass stale cached lines of an earlier op's Q)
#endif
#ifndef ACC_F16
#define ACC_F16 0   // 1: the partial's accumulator rows are written as half (HTP_FA_FOLD_F_ACC_F16)
#endif
#ifndef K_NAT
#define K_NAT 0     // 1: read K from the cache in its natural [key][d] layout through an image (no K^T staging, no tables)
#endif
#ifndef WG_FENCE
#define WG_FENCE 0  // 1: every work-group releases at all-SVM-devices scope before its stage count (else device scope; the last group's acq_rel fence covers the rest)
#endif
#if Q_ATOMIC
inline float4 q_load4(__global const float * p) {
    volatile __global atomic_uint * a = (volatile __global atomic_uint *) p;
    return (float4)(as_float(atomic_load_explicit(a + 0, memory_order_relaxed, memory_scope_device)),
                    as_float(atomic_load_explicit(a + 1, memory_order_relaxed, memory_scope_device)),
                    as_float(atomic_load_explicit(a + 2, memory_order_relaxed, memory_scope_device)),
                    as_float(atomic_load_explicit(a + 3, memory_order_relaxed, memory_scope_device)));
}
#else
inline float4 q_load4(__global const float * p) { return vload4(0, p); }
#endif
// Polls the SVM mirror the relay writes AND the DSP's own ready word through the ION alias
// (ion_ready, may be null): whichever shows `want` first opens the gate; svm[6] records which.
__kernel void gate(__global uint * svm, const uint want, const int max_spins,
                   __global uchar * tabs, const int scnt_off, const int err_off, const int n_stages,
                   __global uint * ion_ready, const int delay_polls) {
    if (get_global_id(0) != 0) return;
    int it = 0;
    uint won = 1u;   // 1 = SVM mirror, 2 = ION word
    for (;;) {
        if (atomic_load_explicit((volatile __global atomic_uint *) svm, memory_order_acquire, memory_scope_all_svm_devices) == want) break;
        if (ion_ready && atomic_load_explicit((volatile __global atomic_uint *) ion_ready, memory_order_acquire, memory_scope_device) == want) { won = 2u; break; }
        if (++it >= max_spins) {
            atomic_store_explicit((volatile __global atomic_uint *)(svm + 2), (uint) it, memory_order_seq_cst, memory_scope_all_svm_devices);
            atomic_store_explicit((volatile __global atomic_uint *)(svm + 1), want | GATE_TIMEOUT, memory_order_seq_cst, memory_scope_all_svm_devices);
            return;
        }
    }
    // diagnostic: hold the chain back for delay_polls more polls (~30 us each) after ready; the
    // loads feed a store so they cannot be dropped
    if (delay_polls > 0) {
        uint acc = 0;
        for (int d = 0; d < delay_polls; ++d) {
            acc += atomic_load_explicit((volatile __global atomic_uint *) svm, memory_order_acquire, memory_scope_all_svm_devices);
            if (ion_ready) acc += atomic_load_explicit((volatile __global atomic_uint *) ion_ready, memory_order_acquire, memory_scope_device);
        }
        atomic_store_explicit((volatile __global atomic_uint *)(svm + 7), acc, memory_order_relaxed, memory_scope_all_svm_devices);
    }
    atomic_store_explicit((volatile __global atomic_uint *)(svm + 6), won, memory_order_seq_cst, memory_scope_all_svm_devices);
    // this op's counters: per-stage work-group counts, the two persistent unit counters, the error word
    __global int * sc = (__global int *)(tabs + scnt_off);
    for (int st = 0; st < n_stages; ++st) sc[st] = 0;
    sc[64] = 0; sc[65] = 0;
    *(__global int *)(tabs + err_off) = 0;
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    atomic_store_explicit((volatile __global atomic_uint *)(svm + 2), (uint) it, memory_order_seq_cst, memory_scope_all_svm_devices);
    atomic_store_explicit((volatile __global atomic_uint *)(svm + 1), want, memory_order_seq_cst, memory_scope_all_svm_devices);
}
inline int gate_open(__global const uint * svm, const uint want) {
    return want == 0u || atomic_load_explicit((volatile __global atomic_uint *)(svm + 1), memory_order_acquire, memory_scope_all_svm_devices) == want;
}
// timeline stamp: the first work-group of a kernel marks its start (trace only)
inline void mark_start(__global uint * svm, const uint want, const int word) {
    if (want != 0u && get_group_id(0) == 0 && get_group_id(1) == 0 && get_group_id(2) == 0 && get_local_id(0) == 0) {
        atomic_store_explicit((volatile __global atomic_uint *)(svm + word), want, memory_order_seq_cst, memory_scope_all_svm_devices);
    }
}

// The membership (one 0/1 f32 row of nbk per (KV head, sub-block)) -> the per-head compact
// tables the two kernels below read, exactly as the host relay used to build them. One
// work-group per KV head. Also zeroes the kernel error word and the per-stage counters.
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void compact(__global const uint * svm, const uint want,
             __global const uchar * embuf, const int em_off, const int nbk, const int num_sb,
             __global uchar * tabs, const int idx_off, const int abs_off, const int cnt_off, const int lcnt_off, const int err_off,
             const int scnt_off, const int n_stages, const int nbk_cap) {
    if (!gate_open(svm, want)) return;
    mark_start((__global uint *) svm, want, 3);
    const int kvh = get_group_id(0);
    const int t   = get_local_id(0);
    __local int tc[1024];
    __global const float * em = (__global const float *)(embuf + em_off) + (long) kvh * num_sb * nbk;
    for (int b = t; b < nbk; b += 256) {
        int u = 0;
        for (int sb = 0; sb < num_sb; ++sb) u |= (em[(long) sb * nbk + b] != 0.0f);
        tc[b] = u ? 0 : -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __global int * abs_tab = (__global int *)(tabs + abs_off) + (long) kvh * nbk_cap;
    __global int * cnt     = (__global int *)(tabs + cnt_off);
    if (t == 0) {
        int c = 0;
        for (int b = 0; b < nbk; ++b) if (tc[b] == 0) { abs_tab[c] = b; tc[b] = c++; }
        for (int i = c; i < nbk_cap; ++i) abs_tab[i] = -1;
        cnt[kvh] = c * 64;
        if (kvh == 0) {
            *(__global int *)(tabs + err_off) = 0;
            __global int * sc = (__global int *)(tabs + scnt_off);
            for (int st = 0; st < n_stages; ++st) sc[st] = 0;
            sc[64] = 0;   // the persistent fa_exc unit counter
            sc[65] = 0;   // the persistent stage_k unit counter
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __global int * idx  = (__global int *)(tabs + idx_off);
    __global int * lcnt = (__global int *)(tabs + lcnt_off);
    for (int sb = t; sb < num_sb; sb += 256) {
        __global int *         irow = idx + ((long) kvh * num_sb + sb) * nbk_cap;
        __global const float * row  = em + (long) sb * nbk;
        int s = 0;
        for (int b = 0; b < nbk; ++b) if (row[b] != 0.0f) irow[s++] = tc[b];
        lcnt[kvh * num_sb + sb] = s;
    }
}

inline void stage_k_item(__global const uchar * kbuf, const int k_off, const int k_nb1, const int k_nb2,
                         __write_only image1d_buffer_t kt_img, const int kt_base_tex,
                         __global const uchar * tabs, const int abs_off, const int abs_stride, const int n_stage,
                         const int k4, const int dg, const int kvh) {
    __global const int * abs_tab = (__global const int *)(tabs + abs_off);
    half8 kr[4];
    for (int c = 0; c < 4; ++c) {
        const int ki  = k4 * 4 + c;
        const int tok = abs_tab[kvh * abs_stride + (ki >> 6)] * FA2_BK + (ki & (FA2_BK - 1));
        __global const half * src = (__global const half *)(kbuf + k_off + (long) tok * k_nb1 + (long) kvh * k_nb2);
        kr[c] = vload8(0, src + dg * 8);
    }
    const int M_4 = n_stage >> 2;
    const int tex = kt_base_tex + kvh * FA_D * M_4 + (dg * 8) * M_4 + k4;
    write_imageh(kt_img, tex,           (half4)(kr[0].s0, kr[1].s0, kr[2].s0, kr[3].s0));
    write_imageh(kt_img, tex + 1 * M_4, (half4)(kr[0].s1, kr[1].s1, kr[2].s1, kr[3].s1));
    write_imageh(kt_img, tex + 2 * M_4, (half4)(kr[0].s2, kr[1].s2, kr[2].s2, kr[3].s2));
    write_imageh(kt_img, tex + 3 * M_4, (half4)(kr[0].s3, kr[1].s3, kr[2].s3, kr[3].s3));
    write_imageh(kt_img, tex + 4 * M_4, (half4)(kr[0].s4, kr[1].s4, kr[2].s4, kr[3].s4));
    write_imageh(kt_img, tex + 5 * M_4, (half4)(kr[0].s5, kr[1].s5, kr[2].s5, kr[3].s5));
    write_imageh(kt_img, tex + 6 * M_4, (half4)(kr[0].s6, kr[1].s6, kr[2].s6, kr[3].s6));
    write_imageh(kt_img, tex + 7 * M_4, (half4)(kr[0].s7, kr[1].s7, kr[2].s7, kr[3].s7));
}

// Kt[kv_head][d][key] over the compacted exception blocks of each KV head, from the K cache.
// Written THROUGH THE IMAGE (one texel = 4 consecutive keys at one d): a K^T written through the
// buffer alias and read through an image view returned the previous op's keys, because the
// texture cache is only invalidated for data the driver knows was written as an image.
__kernel void stage_k(__global const uchar * kbuf, const int k_off, const int k_nb1, const int k_nb2,
                      __write_only image1d_buffer_t kt_img, const int kt_base_tex,
                      __global const uchar * tabs, const int abs_off, const int cnt_off,
                      const int abs_stride, const int n_stage,
                      __global const uint * svm, const uint want) {
    if (!gate_open(svm, want)) return;
    mark_start((__global uint *) svm, want, 4);
    const int k4  = get_global_id(0);   // 4 keys
    const int dg  = get_global_id(1);   // 8 dims
    const int kvh = get_global_id(2);
    __global const int * cnt = (__global const int *)(tabs + cnt_off);
    if (k4 * 4 >= cnt[kvh]) return;
    stage_k_item(kbuf, k_off, k_nb1, k_nb2, kt_img, kt_base_tex, tabs, abs_off, abs_stride, n_stage, k4, dg, kvh);
}

// Persistent form: n_pwg work-groups of 128 pull units of (KV head, 8 consecutive k4) from a
// counter (head-major); a unit past its head's real count costs one atomic. Each work-group
// compacts the membership of the head it is on in local memory (1K floats), so nothing is
// needed from the host or an earlier kernel; the work-group that takes a head's first unit
// also publishes that head's tables for fa_exc, which runs behind the kernel boundary.
__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void stage_k_p(__global const uchar * kbuf, const int k_off, const int k_nb1, const int k_nb2,
               __write_only image1d_buffer_t kt_img, const int kt_base_tex,
               __global uchar * tabs, const int abs_off, const int cnt_off,
               const int abs_stride, const int n_stage, const int nkvh, const int scnt_off,
               __global const uint * svm, const uint want,
               __global uchar * embuf, const int em_off, const int nbk, const int num_sb,
               const int idx_off, const int lcnt_off) {
    if (!gate_open(svm, want)) return;
    ENTRY_ACQUIRE();
    mark_start((__global uint *) svm, want, 4);
    const int t = get_local_id(0);
    __local int unit_sh, cur_head, cnt_l;
    __local int tc[STAGE_NBK_MAX], abs_l[STAGE_NBK_MAX];   // 2 KB: keeps the persistent work-groups resident
    if (t == 0) cur_head = -1;
    const int kb_per_head = (n_stage / 4 + 7) / 8;
    const int n_units = nkvh * kb_per_head;
    for (;;) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (t == 0) unit_sh = atomic_add((volatile __global int *)(tabs + scnt_off) + 65, 1);
        barrier(CLK_LOCAL_MEM_FENCE);
        const int unit = unit_sh;
        if (unit >= n_units) break;
        const int kvh = unit / kb_per_head;
        const int kb  = unit % kb_per_head;
        // membership rows as uint bits through device-scope atomic loads: a plain load may hit a
        // stale UCHE line left by an earlier op's membership at the same address (the graph
        // allocator reuses it); atomic loads see the DSP's writes, as the gate's ready poll does
        volatile __global atomic_uint * em = (volatile __global atomic_uint *)(embuf + em_off) + (long) kvh * num_sb * nbk;
        if (kvh != cur_head) {
            for (int b = t; b < nbk; b += 128) {
                int u = 0;
                for (int sb = 0; sb < num_sb; ++sb) u |= (atomic_load_explicit(em + (long) sb * nbk + b, memory_order_relaxed, memory_scope_device) != 0u);
                tc[b] = u ? 0 : -1;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (t == 0) {
                int c = 0;
                for (int b = 0; b < nbk; ++b) if (tc[b] == 0) { abs_l[c] = b; tc[b] = c++; }
                cnt_l = c * 64; cur_head = kvh;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (kb == 0) {
            // publish this head's tables (abs, cnt, idx rows, lcnt) exactly as the host relay did
            __global int * abs_tab = (__global int *)(tabs + abs_off) + (long) kvh * abs_stride;
            const int c = cnt_l / 64;
            for (int i = t; i < abs_stride; i += 128) abs_tab[i] = i < c ? abs_l[i] : -1;
            if (t == 0) ((__global int *)(tabs + cnt_off))[kvh] = cnt_l;
            __global int * idx  = (__global int *)(tabs + idx_off);
            __global int * lcnt = (__global int *)(tabs + lcnt_off);
            for (int sb = t; sb < num_sb; sb += 128) {
                __global int * irow = idx + ((long) kvh * num_sb + sb) * abs_stride;
                volatile __global atomic_uint * row = em + (long) sb * nbk;
                int s = 0;
                for (int b = 0; b < nbk; ++b) if (atomic_load_explicit(row + b, memory_order_relaxed, memory_scope_device) != 0u) irow[s++] = tc[b];
                lcnt[kvh * num_sb + sb] = s;
            }
        }
        const int k4 = kb * 8 + t / 16;
        const int dg = t % 16;
        if (k4 * 4 < cnt_l) {
            half8 kr[4];
            for (int cc = 0; cc < 4; ++cc) {
                const int ki  = k4 * 4 + cc;
                const int tok = abs_l[ki >> 6] * FA2_BK + (ki & (FA2_BK - 1));
                __global const half * src = (__global const half *)(kbuf + k_off + (long) tok * k_nb1 + (long) kvh * k_nb2);
                kr[cc] = vload8(0, src + dg * 8);
            }
            const int M_4 = n_stage >> 2;
            const int tex = kt_base_tex + kvh * FA_D * M_4 + (dg * 8) * M_4 + k4;
#if KT_BUF
            __global half * kt = (__global half *) tabs;   // tabs == the fold buffer alias; texel tex = 4 halves at tex * 4
            vstore4((half4)(kr[0].s0, kr[1].s0, kr[2].s0, kr[3].s0), 0, kt + (long) (tex          ) * 4);
            vstore4((half4)(kr[0].s1, kr[1].s1, kr[2].s1, kr[3].s1), 0, kt + (long) (tex + 1 * M_4) * 4);
            vstore4((half4)(kr[0].s2, kr[1].s2, kr[2].s2, kr[3].s2), 0, kt + (long) (tex + 2 * M_4) * 4);
            vstore4((half4)(kr[0].s3, kr[1].s3, kr[2].s3, kr[3].s3), 0, kt + (long) (tex + 3 * M_4) * 4);
            vstore4((half4)(kr[0].s4, kr[1].s4, kr[2].s4, kr[3].s4), 0, kt + (long) (tex + 4 * M_4) * 4);
            vstore4((half4)(kr[0].s5, kr[1].s5, kr[2].s5, kr[3].s5), 0, kt + (long) (tex + 5 * M_4) * 4);
            vstore4((half4)(kr[0].s6, kr[1].s6, kr[2].s6, kr[3].s6), 0, kt + (long) (tex + 6 * M_4) * 4);
            vstore4((half4)(kr[0].s7, kr[1].s7, kr[2].s7, kr[3].s7), 0, kt + (long) (tex + 7 * M_4) * 4);
#else
            write_imageh(kt_img, tex,           (half4)(kr[0].s0, kr[1].s0, kr[2].s0, kr[3].s0));
            write_imageh(kt_img, tex + 1 * M_4, (half4)(kr[0].s1, kr[1].s1, kr[2].s1, kr[3].s1));
            write_imageh(kt_img, tex + 2 * M_4, (half4)(kr[0].s2, kr[1].s2, kr[2].s2, kr[3].s2));
            write_imageh(kt_img, tex + 3 * M_4, (half4)(kr[0].s3, kr[1].s3, kr[2].s3, kr[3].s3));
            write_imageh(kt_img, tex + 4 * M_4, (half4)(kr[0].s4, kr[1].s4, kr[2].s4, kr[3].s4));
            write_imageh(kt_img, tex + 5 * M_4, (half4)(kr[0].s5, kr[1].s5, kr[2].s5, kr[3].s5));
            write_imageh(kt_img, tex + 6 * M_4, (half4)(kr[0].s6, kr[1].s6, kr[2].s6, kr[3].s6));
            write_imageh(kt_img, tex + 7 * M_4, (half4)(kr[0].s7, kr[1].s7, kr[2].s7, kr[3].s7));
#endif
        }
    }
}

// One work-group = 32 query rows of one head over that sub-block's exception blocks. Emits the
// UNNORMALISED (m, l, acc) per row; a row without exceptions leaves the empty partial.
// Work-group mapping: chain == 0: (qb, bh) = (gid1, gid2), a launch per stage or per op with the
// host's offsets. chain == 1: one launch per op, gid2 = stage (dispatched in stage order), gid1
// = (32-row block within the stage, head within the stage). gpu_done != 0: the last work-group
// of a stage (per-stage counter) writes the ION stamp and done word, then the SVM done word.
__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void fa_exc(__read_only image1d_buffer_t kt_img, __read_only image1d_buffer_t v_img,
            __global const uchar * qbuf, const int q_off, const int q_nb1, const int q_nb2,
            __global const uchar * mbuf, const int mask_off, const int mask_nb1,
            __global uchar * hbuf, __global uchar * tabs, const int idx_off, const int abs_off, const int cnt_off, const int lcnt_off, const int err_off,
            const int f_m, const int f_l, const int f_acc,
            const int Sq, const int G, const int num_sb, const int qb_per_sb, const int top_k,
            const int abs_stride, const int n_stage,
            const int kt_base_tex, const int v_base_tex, const int v_tok_tex, const int v_head_tex,
            const float scale,
            __global uint * svm, const uint want, const int chain, const int gpu_done, const int qbh,
            const int nkvh, const int qb32_per_stage, const int scnt_off, const int done_off, const int stamp_off, const int n_stages_arg,
            __read_only image1d_buffer_t k_img, const int k_base_tex, const int k_tok_tex, const int k_head_tex,
            __global uchar * embuf, const int em_off, const int nbk) {
    if (!gate_open(svm, want)) return;
    ENTRY_ACQUIRE();
    mark_start(svm, want, 5);
    const int t  = get_local_id(0);
    __local float S_lds[FA2_BQ][FA2_BK];
    __local float m_run[FA2_BQ]; __local float l_run[FA2_BQ]; __local float a_sh[FA2_BQ];
    __local int   unit_sh;
#if K_NAT
    __local int   list_sh[STAGE_NBK_MAX]; __local int n_list_sh;
#endif
    const int hpl     = qbh ? G : G * nkvh;            // heads per stage
    const int n_units = n_stages_arg * qb32_per_stage * hpl;
    // chain == 2: persistent work-groups pull units (stage-major) from a counter until none are left
    for (;;) {
    int qb, bh;
    if (chain == 2) {
        if (t == 0) unit_sh = atomic_add((volatile __global int *)(tabs + scnt_off) + 64, 1);
        barrier(CLK_LOCAL_MEM_FENCE);
        const int unit = unit_sh;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (unit >= n_units) break;
        const int stage = unit / (qb32_per_stage * hpl);
        const int y     = unit % (qb32_per_stage * hpl);
        qb = (qbh ? stage / nkvh : stage) * qb32_per_stage + y / hpl;
        bh = (qbh ? (stage % nkvh) * G : 0) + y % hpl;
    } else if (chain == 1) {
        const int stage = get_global_id(2);
        const int y     = get_global_id(1);
        qb = (qbh ? stage / nkvh : stage) * qb32_per_stage + y / hpl;
        bh = (qbh ? (stage % nkvh) * G : 0) + y % hpl;
    } else {
        qb = get_global_id(1);
        bh = get_global_id(2);
    }
    const int q0 = qb * FA2_BQ;
    const int kvh = bh / G;
    const int sb  = qb / qb_per_sb;
    if (q0 < Sq) {

#if K_NAT
    // the exception list of (kvh, sb) straight from the membership row, ascending block order;
    // the barrier publishes it to the other waves (without it they read the previous unit's list)
    if (t == 0) {
        volatile __global atomic_uint * emrow = (volatile __global atomic_uint *)(embuf + em_off) + ((long) kvh * num_sb + sb) * nbk;
        int s = 0;
        for (int b = 0; b < nbk && s < STAGE_NBK_MAX; ++b) if (atomic_load_explicit(emrow + b, memory_order_relaxed, memory_scope_device) != 0u) list_sh[s++] = b;
        n_list_sh = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int k_head = k_base_tex + kvh * k_head_tex;
#else
    __global const int * idx_row = (__global const int *)(tabs + idx_off) + ((long) kvh * num_sb + sb) * top_k;
    const int cnt_row = ((__global const int *)(tabs + lcnt_off))[(long) kvh * num_sb + sb];
    __global const int * abs_tab = (__global const int *)(tabs + abs_off);
    __global const int * cnt     = (__global const int *)(tabs + cnt_off);
    const int n_blk_kvh = cnt[kvh] >> 6;
#endif

    const int pv_mt = t % (FA_D / 8); const int pv_nt = t / (FA_D / 8); const int pv_d0 = pv_mt * 8;
    float8 o0 = (float8)0, o1 = (float8)0, o2 = (float8)0, o3 = (float8)0;
    if (t < FA2_BQ) { m_run[t] = -INFINITY; l_run[t] = 0.0f; }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int M_4kv   = n_stage >> 2;
    const int kt_head = kt_base_tex + kvh * FA_D * M_4kv;
    const int v_head  = v_base_tex + kvh * v_head_tex;
    __global const float * Qh = (__global const float *)(qbuf + q_off + (long) bh * q_nb2);
    const int q_row_f = q_nb1 >> 2;
    __global const half * Mk = (__global const half *)(mbuf + mask_off);
    const int mask_row_h = mask_nb1 >> 1;

#if K_NAT
    const int n_list = n_list_sh;   // visible: the barrier above followed thread 0's scan
#else
    const int n_list = (cnt_row < top_k) ? cnt_row : top_k;
#endif
    for (int s = 0; s < n_list; ++s) {
#if K_NAT
        const int cb  = list_sh[s];          // absolute block
        const int vk0 = cb * FA2_BK;
#else
        const int cb = idx_row[s];
        if (cb < 0) break;
        if (cb >= n_blk_kvh) {
            if (t == 0) atomic_inc((volatile __global int *)(tabs + err_off));
            break;
        }
        const int kt0 = cb * FA2_BK;
        const int vk0 = abs_tab[kvh * abs_stride + cb] * FA2_BK;
#endif
        {
            const int mt = t % (FA2_BK / 8); const int nt = t / (FA2_BK / 8);
            // rows past Sq (a ragged last ubatch) read a clamped row and are never written
            const int qr0 = min(q0 + nt * 2,     Sq - 1);
            const int qr1 = min(q0 + nt * 2 + 1, Sq - 1);
            float8 c0 = (float8)0, c1 = (float8)0;
#if K_NAT
            // natural layout: one texel = 4 dims of one key; this thread's 8 keys are 8 rows of 32 texels
            {
                const int kt_row0 = k_head + (vk0 + mt * 8) * k_tok_tex;
                float c0a[8] = {0, 0, 0, 0, 0, 0, 0, 0}, c1a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                for (int i = 0; i < FA_D; i += 4) {
                    const float4 wa = q_load4(Qh + (long) qr0 * q_row_f + i);
                    const float4 wb = q_load4(Qh + (long) qr1 * q_row_f + i);
                    #pragma unroll
                    for (int jj = 0; jj < 8; ++jj) {
                        const float4 kq = convert_float4(read_imageh(k_img, kt_row0 + jj * k_tok_tex + (i >> 2)));
                        c0a[jj] += dot(kq, wa); c1a[jj] += dot(kq, wb);
                    }
                }
                c0 = (float8)(c0a[0], c0a[1], c0a[2], c0a[3], c0a[4], c0a[5], c0a[6], c0a[7]);
                c1 = (float8)(c1a[0], c1a[1], c1a[2], c1a[3], c1a[4], c1a[5], c1a[6], c1a[7]);
            }
#else
            const int kbase = kt_head + (kt0 >> 2) + mt * 2;
            for (int i = 0; i < FA_D; i += 4) {
                const int tb = kbase + i * M_4kv;
                half8 B0, B1, B2, B3;
#if KT_BUF
                __global const half * kt = (__global const half *) hbuf;
                B0 = vload8(0, kt + (long) (tb            ) * 4);
                B1 = vload8(0, kt + (long) (tb +     M_4kv) * 4);
                B2 = vload8(0, kt + (long) (tb + 2 * M_4kv) * 4);
                B3 = vload8(0, kt + (long) (tb + 3 * M_4kv) * 4);
#else
                B0.s0123 = read_imageh(kt_img, tb);              B0.s4567 = read_imageh(kt_img, tb + 1);
                B1.s0123 = read_imageh(kt_img, tb + M_4kv);      B1.s4567 = read_imageh(kt_img, tb + M_4kv + 1);
                B2.s0123 = read_imageh(kt_img, tb + 2 * M_4kv);  B2.s4567 = read_imageh(kt_img, tb + 2 * M_4kv + 1);
                B3.s0123 = read_imageh(kt_img, tb + 3 * M_4kv);  B3.s4567 = read_imageh(kt_img, tb + 3 * M_4kv + 1);
#endif
                const float8 b0 = convert_float8(B0), b1 = convert_float8(B1), b2 = convert_float8(B2), b3 = convert_float8(B3);
                const float4 wa = q_load4(Qh + (long) qr0 * q_row_f + i);
                const float4 wb = q_load4(Qh + (long) qr1 * q_row_f + i);
                c0 += b0*wa.s0; c0 += b1*wa.s1; c0 += b2*wa.s2; c0 += b3*wa.s3;
                c1 += b0*wb.s0; c1 += b1*wb.s1; c1 += b2*wb.s2; c1 += b3*wb.s3;
            }
#endif
            const int kb = vk0 + mt * 8;
            #define EMITS(NN, CV, QR) { float8 v = (CV) * scale; \
                v += convert_float8(vload8(0, Mk + (long) (QR) * mask_row_h + kb)); \
                vstore8(v, 0, &S_lds[nt * 2 + (NN)][mt * 8]); }
            EMITS(0, c0, qr0); EMITS(1, c1, qr1);
            #undef EMITS
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (t < FA2_BQ) {
            const int q = t; __local float * row = S_lds[q];
            float8 mx = (float8)(-INFINITY);
            for (int k = 0; k < FA2_BK; k += 8) mx = fmax(mx, vload8(0, row + k));
            float4 m4 = fmax(mx.lo, mx.hi); float2 m2 = fmax(m4.lo, m4.hi); float m_tile = fmax(m2.s0, m2.s1);
            const float m_old = m_run[q]; const float m_new = fmax(m_old, m_tile);
            const float a = (m_old == -INFINITY) ? 0.0f : native_exp(m_old - m_new);
            float8 l8 = (float8)0;
            for (int k = 0; k < FA2_BK; k += 8) {
                const float8 s8 = vload8(0, row + k);
                const float8 p = select(native_exp(s8 - m_new), (float8)0, isinf(s8));
                vstore8(p, 0, row + k); l8 += p;
            }
            float4 l4 = l8.lo + l8.hi; float2 l2 = l4.lo + l4.hi;
            l_run[q] = a * l_run[q] + (l2.s0 + l2.s1); m_run[q] = m_new; a_sh[q] = a;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        o0 *= a_sh[pv_nt * 4 + 0]; o1 *= a_sh[pv_nt * 4 + 1]; o2 *= a_sh[pv_nt * 4 + 2]; o3 *= a_sh[pv_nt * 4 + 3];
        for (int kk = 0; kk < FA2_BK; kk += 4) {
            const int vt = v_head + (vk0 + kk) * v_tok_tex + pv_mt * 2;
            half8 V0, V1, V2, V3;
            V0.s0123 = read_imageh(v_img, vt);                   V0.s4567 = read_imageh(v_img, vt + 1);
            V1.s0123 = read_imageh(v_img, vt + v_tok_tex);       V1.s4567 = read_imageh(v_img, vt + v_tok_tex + 1);
            V2.s0123 = read_imageh(v_img, vt + 2 * v_tok_tex);   V2.s4567 = read_imageh(v_img, vt + 2 * v_tok_tex + 1);
            V3.s0123 = read_imageh(v_img, vt + 3 * v_tok_tex);   V3.s4567 = read_imageh(v_img, vt + 3 * v_tok_tex + 1);
            const float8 v0 = convert_float8(V0), v1 = convert_float8(V1), v2 = convert_float8(V2), v3 = convert_float8(V3);
            float4 p0 = vload4(0, &S_lds[pv_nt * 4 + 0][kk]); float4 p1 = vload4(0, &S_lds[pv_nt * 4 + 1][kk]);
            float4 p2 = vload4(0, &S_lds[pv_nt * 4 + 2][kk]); float4 p3 = vload4(0, &S_lds[pv_nt * 4 + 3][kk]);
            o0 += v0*p0.s0; o0 += v1*p0.s1; o0 += v2*p0.s2; o0 += v3*p0.s3;
            o1 += v0*p1.s0; o1 += v1*p1.s1; o1 += v2*p1.s2; o1 += v3*p1.s3;
            o2 += v0*p2.s0; o2 += v1*p2.s1; o2 += v2*p2.s2; o2 += v3*p2.s3;
            o3 += v0*p3.s0; o3 += v1*p3.s1; o3 += v2*p3.s2; o3 += v3*p3.s3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    const long r0 = (long) bh * Sq + q0;
    __global float * Fm = (__global float *)(hbuf + f_m);
    __global float * Fl = (__global float *)(hbuf + f_l);
#if ACC_F16
    __global half * Fa = (__global half *)(hbuf + f_acc);
#else
    __global float * Fa = (__global float *)(hbuf + f_acc);
#endif
    if (t < FA2_BQ && q0 + t < Sq) {
        const float l = l_run[t];
        Fm[r0 + t] = (l > 0.0f) ? m_run[t] : M_EMPTY;
        Fl[r0 + t] = l;
    }
#if ACC_F16
    #define OWRITE(NN, OV) { const int q = pv_nt * 4 + (NN); const float l = l_run[q]; \
        if (q0 + q < Sq) vstore_half8((l > 0.0f) ? (OV) : (float8)0, 0, Fa + (r0 + q) * FA_D + pv_d0); }
#else
    #define OWRITE(NN, OV) { const int q = pv_nt * 4 + (NN); const float l = l_run[q]; \
        if (q0 + q < Sq) vstore8((l > 0.0f) ? (OV) : (float8)0, 0, Fa + (r0 + q) * FA_D + pv_d0); }
#endif
    OWRITE(0, o0); OWRITE(1, o1); OWRITE(2, o2); OWRITE(3, o3);
    #undef OWRITE
    }   // q0 < Sq
    if (gpu_done) {
        // every work-group of the stage counts, ragged ones included, so the host's per-stage
        // count is simply qb32_per_stage * heads-per-stage
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
        if (t == 0) {
            const int stage = (qb / qb32_per_stage) * (qbh ? nkvh : 1) + (qbh ? kvh : 0);
            const int n_wg  = qb32_per_stage * (qbh ? G : G * nkvh);
            // this work-group's rows are in L2 (the barrier above); the last work-group's acq_rel
            // fence at all-SVM-devices scope pushes every group's rows out before the done word
#if WG_FENCE
            atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
#else
            atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
#endif
            const int old = atomic_fetch_add_explicit((volatile __global atomic_int *)(tabs + scnt_off) + stage, 1, memory_order_acq_rel, memory_scope_device);
            if (old == n_wg - 1) {
                atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acq_rel, memory_scope_all_svm_devices);
                atomic_store_explicit((volatile __global atomic_uint *)(hbuf + stamp_off) + stage, want, memory_order_seq_cst, memory_scope_all_svm_devices);
                atomic_store_explicit((volatile __global atomic_uint *)(hbuf + done_off) + stage, want, memory_order_seq_cst, memory_scope_all_svm_devices);
                atomic_store_explicit((volatile __global atomic_uint *)(svm + 64 + stage), want, memory_order_seq_cst, memory_scope_all_svm_devices);
            }
        }
    }
    if (chain != 2) break;
    barrier(CLK_LOCAL_MEM_FENCE);
    }   // persistent loop
}
)CL";

struct hfold_chain { bool enq = false; std::chrono::steady_clock::time_point t_enq; std::vector<cl_event> ev; };

struct ggml_hexagon_hfold_job {
    const ggml_tensor * node = nullptr;
    uint32_t want = 0;
    cl_mem   q = nullptr, k = nullptr, v_img = nullptr, m = nullptr, k_img = nullptr;
    uint32_t q_off = 0, q_nb1 = 0, q_nb2 = 0, k_off = 0, k_nb1 = 0, k_nb2 = 0, v_off = 0, v_nb1 = 0, v_nb2 = 0, m_off = 0, m_nb1 = 0;
    uint32_t Sq = 0, nh = 0, nkvh = 0, G = 0, nbk = 0, num_sb = 0, num_qb = 0;
    uint32_t br = 0, n_stages = 0;      // the kernel's query tile rows; stages = query blocks or KV heads
    bool     qbh = false;               // stages are (query block, KV head) pairs
    cl_mem   em = nullptr;              // src[8] through the compute buffer's alias (the GPU compacts it)
    uint32_t em_off = 0;
    const float * em_host = nullptr;   // src[8] in the compute buffer (host VA)
    size_t   em_bytes = 0;
    float    scale = 0.0f;
    const uint8_t * q_base = nullptr, * k_base = nullptr, * v_base = nullptr, * m_base = nullptr;   // host VAs (FOLD_CHECK)
    hfold_chain chain;                 // GPU chain state (enqueued when; events to release)
};

struct ggml_hexagon_hfold {
    ggml_hexagon_session *  sess = nullptr;
    ggml_backend_buffer_t   buf  = nullptr;   // header + partial + handshake + GPU tables + staged K^T
    uint8_t *               base = nullptr;
    size_t                  size = 0;
    ggml_context *          tctx = nullptr;
    ggml_tensor *           ctrl = nullptr;   // hand-wired tensor over buf, attached as src[7]
    // layout (bytes from base)
    size_t   off_m = 0, off_l = 0, off_acc = 0, off_ctl = 0, off_idx = 0, off_abs = 0, off_cnt = 0, off_lcnt = 0, off_err = 0, off_kt = 0;
    size_t   off_done = 0, off_scnt = 0, off_stamp = 0;   // GPU-written done words, per-stage counters, visibility stamps
    uint32_t rows = 0, neq1 = 0, nbk_cap = 0, num_sb_cap = 0, nkvh = 0, D = 0;

    cl_platform_id   plat = nullptr;
    cl_device_id     dev  = nullptr;
    cl_context       ctx  = nullptr;
    cl_command_queue q    = nullptr;
    cl_program       prog = nullptr;
    cl_kernel        k_stage = nullptr, k_fa = nullptr, k_keep = nullptr, k_gate = nullptr, k_compact = nullptr, k_stage_p = nullptr, k_touch = nullptr;
    cl_command_queue q2 = nullptr;                               // trigger submissions at ready (TRIG)
    uint32_t *       svm_ctl = nullptr;                          // fine-grain SVM: [0] ready mirror, [1] gate status, [2] gate spins, [64..] done words
    bool             svm_ok = false;
    // chain path statistics
    uint64_t n_chain = 0, n_gate_timeouts = 0, n_fallback = 0, n_direct = 0, n_stamp_seen = 0, n_stamp_total = 0, n_preq = 0, n_late = 0;
    uint64_t n_kernel_err = 0, n_check_rows = 0, n_check_bad = 0;   // GPU list-bound errors; CHECK=2 reference rows / mismatches
    double   us_first_done = 0, us_last_done = 0, us_gate = 0, us_lead = 0;   // summed over chain jobs (ready -> first/last done, ready -> gate open, enqueue lead before ready)
    double   period_us = 0;                                                    // EMA of ready-to-ready within a batch
    cl_mem           keep_buf = nullptr, keep_out = nullptr;
    std::thread      keeper;
    std::atomic<bool> keep_stop{false};
    uint64_t         n_keep = 0;
    size_t           max_img_texels = 0;
    cl_mem           hbuf = nullptr, kt_img = nullptr;           // alias + K^T texture view of buf
    cl_mem           ready_sub = nullptr;                        // sub-buffer of hbuf at the ready word (gate polls it)
    uint64_t         n_gate_ion = 0;                             // gates opened by the ION word before the SVM mirror
    uint8_t *        svm_tabs = nullptr;                         // SVMTAB probe: tables here instead of buf
    uint8_t *        svm_q = nullptr;                            // SVMQ probe: Q copied here per op
    size_t           svm_tabs_bytes = 0, svm_q_bytes = 0;
    std::unordered_map<int, cl_mem> aliases;                     // by rpcmem fd
    std::unordered_map<int, cl_mem> images;                      // RGBA-half view of a whole alias, by fd
    std::unordered_map<const ggml_tensor *, ggml_hexagon_hfold_job> jobs;
    uint32_t seq = 0;

    std::thread             relay;
    std::mutex              mu;
    std::condition_variable cv;
    std::deque<std::vector<ggml_hexagon_hfold_job>> batches;
    bool                    stop = false, busy = false;
    uint64_t n_batches = 0, n_jobs = 0, n_ready_timeouts = 0, n_gpu_err = 0;
    double   us_ready = 0, us_gpu = 0;   // summed over jobs: ready wait, GPU (first enqueue -> last head done)
};

static bool hfold_cl_check(cl_int err, const char * what) {
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: fa-fold: %s failed (%d)\n", what, err);
        return false;
    }
    return true;
}

static cl_mem ggml_hexagon_hfold_alias(ggml_hexagon_hfold * h, ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto it = h->aliases.find(sbuf->fd);
    if (it != h->aliases.end()) {
        return it->second;
    }
    cl_mem_ion_host_ptr ion = {};
    ion.ext_host_ptr.allocation_type   = CL_MEM_ION_HOST_PTR_QCOM;
    ion.ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
    ion.ion_filedesc = sbuf->fd;
    ion.ion_hostptr  = sbuf->base;
    const size_t size = sbuf->size & ~(size_t) 4095;
    cl_int err;
    cl_mem mem = clCreateBuffer(h->ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM, size, &ion, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: fa-fold: ION alias of buffer fd %d size %zu failed (%d)\n", sbuf->fd, size, err);
        return nullptr;
    }
    HEX_VERBOSE("ggml-hex: fa-fold: aliased buffer fd %d base %p size %zu into OpenCL\n", sbuf->fd, (void *) sbuf->base, size);
    h->aliases[sbuf->fd] = mem;
    return mem;
}

// RGBA-half texture view over a whole alias (the V cache and the staged K^T are read through it).
static cl_mem ggml_hexagon_hfold_image(ggml_hexagon_hfold * h, ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto it = h->images.find(sbuf->fd);
    if (it != h->images.end()) {
        return it->second;
    }
    cl_mem alias = ggml_hexagon_hfold_alias(h, buffer);
    if (!alias) return nullptr;
    const size_t texels = (sbuf->size & ~(size_t) 4095) / 8;
    if (texels > h->max_img_texels) {
        GGML_LOG_ERROR("ggml-hex: fa-fold: buffer fd %d needs %zu texels, the device caps a 1D image buffer at %zu (~%zu MB); context too long for the alias\n",
                       sbuf->fd, texels, h->max_img_texels, h->max_img_texels * 8 >> 20);
        return nullptr;
    }
    cl_image_format fmt = { CL_RGBA, CL_HALF_FLOAT };
    cl_image_desc   dk; memset(&dk, 0, sizeof(dk));
    dk.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    dk.image_width = texels;
    dk.buffer      = alias;
    cl_int err;
    cl_mem img = clCreateImage(h->ctx, CL_MEM_READ_WRITE, &fmt, &dk, nullptr, &err);
    if (!hfold_cl_check(err, "clCreateImage (RGBA half view)")) return nullptr;
    h->images[sbuf->fd] = img;
    return img;
}

static void ggml_hexagon_hfold_relay_main(ggml_hexagon_hfold * h);
static void ggml_hexagon_hfold_log_summary(ggml_hexagon_hfold * h);
static void hfold_enqueue_chain(ggml_hexagon_hfold * h, const ggml_hexagon_hfold_job & j, hfold_chain & c);

static bool ggml_hexagon_hfold_init(ggml_hexagon_session * sess) {
    auto h = new ggml_hexagon_hfold();
    h->sess = sess;
    cl_uint n = 0; cl_int err;
    if (!hfold_cl_check(clGetPlatformIDs(1, &h->plat, &n), "clGetPlatformIDs") || !n) { delete h; return false; }
    if (!hfold_cl_check(clGetDeviceIDs(h->plat, CL_DEVICE_TYPE_GPU, 1, &h->dev, &n), "clGetDeviceIDs") || !n) { delete h; return false; }
    {
        size_t ext_len = 0;
        clGetDeviceInfo(h->dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_len);
        std::string ext(ext_len, '\0');
        clGetDeviceInfo(h->dev, CL_DEVICE_EXTENSIONS, ext_len, ext.data(), nullptr);
        if (ext.find("cl_qcom_ext_host_ptr") == std::string::npos) {
            GGML_LOG_ERROR("ggml-hex: fa-fold: GPU lacks cl_qcom_ext_host_ptr (no ION alias)\n");
            delete h; return false;
        }
        clGetDeviceInfo(h->dev, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, sizeof(h->max_img_texels), &h->max_img_texels, nullptr);
    }
    {
        // The GPU idles between ops and would otherwise run each op's kernels on a ramping clock.
        // cl_qcom_perf_hint pins it high for this context; ignored where the extension is absent.
        size_t ext_len = 0;
        clGetDeviceInfo(h->dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_len);
        std::string ext(ext_len, '\0');
        clGetDeviceInfo(h->dev, CL_DEVICE_EXTENSIONS, ext_len, ext.data(), nullptr);
        const bool perf_hint = ext.find("cl_qcom_perf_hint") != std::string::npos && opt_fa_fold_perf;
#ifndef CL_CONTEXT_PERF_HINT_QCOM
#define CL_CONTEXT_PERF_HINT_QCOM 0x40C2
#define CL_PERF_HINT_HIGH_QCOM    0x40C3
#endif
        const cl_context_properties props[] = { CL_CONTEXT_PERF_HINT_QCOM, CL_PERF_HINT_HIGH_QCOM, 0 };
        h->ctx = clCreateContext(perf_hint ? props : nullptr, 1, &h->dev, nullptr, nullptr, &err);
        if (!hfold_cl_check(err, "clCreateContext")) { delete h; return false; }
        if (perf_hint) GGML_LOG_INFO("ggml-hex: fa-fold: GPU perf hint HIGH set on the context\n");
    }
    h->q = clCreateCommandQueueWithProperties(h->ctx, h->dev, nullptr, &err);
    if (!hfold_cl_check(err, "clCreateCommandQueueWithProperties")) { delete h; return false; }
    h->prog = clCreateProgramWithSource(h->ctx, 1, &ggml_hexagon_hfold_cl_src, nullptr, &err);
    if (!hfold_cl_check(err, "clCreateProgramWithSource")) { delete h; return false; }
    {
        std::string bopts = "-cl-std=CL2.0 -cl-fast-relaxed-math -DFA_D=128";
        if (opt_fa_fold_ktbuf) bopts += " -DKT_BUF=1";
        if (opt_fa_fold_acq)   bopts += " -DACQ_FENCE=1";
        if (opt_fa_fold_qatomic) bopts += " -DQ_ATOMIC=1";
        if (opt_fa_fold_wgfence) bopts += " -DWG_FENCE=1";
        if (opt_fa_fold_acc16)   bopts += " -DACC_F16=1";
        if (opt_fa_fold_knat)    bopts += " -DK_NAT=1";
        err = clBuildProgram(h->prog, 1, &h->dev, bopts.c_str(), nullptr, nullptr);
    }
    if (err != CL_SUCCESS) {
        size_t len = 0;
        clGetProgramBuildInfo(h->prog, h->dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, '\0');
        clGetProgramBuildInfo(h->prog, h->dev, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
        GGML_LOG_ERROR("ggml-hex: fa-fold: kernel build failed (%d):\n%s\n", err, log.c_str());
        delete h; return false;
    }
    h->k_stage = clCreateKernel(h->prog, "stage_k", &err); if (!hfold_cl_check(err, "clCreateKernel stage_k")) { delete h; return false; }
    h->k_fa    = clCreateKernel(h->prog, "fa_exc",  &err); if (!hfold_cl_check(err, "clCreateKernel fa_exc"))  { delete h; return false; }
    h->k_gate  = clCreateKernel(h->prog, "gate",    &err); if (!hfold_cl_check(err, "clCreateKernel gate"))    { delete h; return false; }
    h->k_compact = clCreateKernel(h->prog, "compact", &err); if (!hfold_cl_check(err, "clCreateKernel compact")) { delete h; return false; }
    h->k_stage_p = clCreateKernel(h->prog, "stage_k_p", &err); if (!hfold_cl_check(err, "clCreateKernel stage_k_p")) { delete h; return false; }
    h->k_touch   = clCreateKernel(h->prog, "touch", &err); if (!hfold_cl_check(err, "clCreateKernel touch")) { delete h; return false; }
    h->q2 = clCreateCommandQueueWithProperties(h->ctx, h->dev, nullptr, &err);
    if (!hfold_cl_check(err, "clCreateCommandQueueWithProperties (trigger)")) { delete h; return false; }
    {
        cl_device_svm_capabilities svm = 0;
        clGetDeviceInfo(h->dev, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm), &svm, nullptr);
        if ((svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) && (svm & CL_DEVICE_SVM_ATOMICS)) {
            h->svm_ctl = (uint32_t *) clSVMAlloc(h->ctx, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, 4096, 64);
            if (h->svm_ctl) { memset(h->svm_ctl, 0, 4096); h->svm_ok = true; }
        }
        if (!h->svm_ok && opt_fa_fold_gpudone) {
            GGML_LOG_WARN("ggml-hex: fa-fold: no fine-grain SVM atomics; GPU-side chain disabled\n");
            opt_fa_fold_gpudone = 0;
        }
    }
    if (opt_fa_fold_keepalive > 0) {
        h->k_keep = clCreateKernel(h->prog, "keepalive", &err); if (!hfold_cl_check(err, "clCreateKernel keepalive")) { delete h; return false; }
        const size_t bytes = 32u << 20;
        h->keep_buf = clCreateBuffer(h->ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err); if (!hfold_cl_check(err, "keepalive buffer")) { delete h; return false; }
        h->keep_out = clCreateBuffer(h->ctx, CL_MEM_WRITE_ONLY, 1u << 20, nullptr, &err); if (!hfold_cl_check(err, "keepalive out")) { delete h; return false; }
        h->keeper = std::thread([h]() {
#if defined(__linux__)
            if (opt_fa_fold_cpu >= 0) { cpu_set_t set; CPU_ZERO(&set); CPU_SET(opt_fa_fold_cpu, &set); sched_setaffinity(0, sizeof(set), &set); }
#endif
            cl_command_queue q = clCreateCommandQueueWithProperties(h->ctx, h->dev, nullptr, nullptr);
            const cl_int n4 = (cl_int) ((32u << 20) / 16);
            cl_int iters = 1;
            clSetKernelArg(h->k_keep, 0, sizeof(cl_mem), &h->keep_buf);
            clSetKernelArg(h->k_keep, 1, sizeof(cl_int), &n4);
            clSetKernelArg(h->k_keep, 3, sizeof(cl_mem), &h->keep_out);
            const size_t gsz = 4096 * 64, lsz = 64;
            // calibrate iters so one launch takes ~the requested busy time
            for (int c = 0; c < 4 && !h->keep_stop.load(); ++c) {
                clSetKernelArg(h->k_keep, 2, sizeof(cl_int), &iters);
                const auto t0 = std::chrono::steady_clock::now();
                clEnqueueNDRangeKernel(q, h->k_keep, 1, nullptr, &gsz, &lsz, 0, nullptr, nullptr); clFinish(q);
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                if (us > 50.0) iters = std::max<cl_int>(1, (cl_int) (iters * opt_fa_fold_keepalive / us));
            }
            GGML_LOG_INFO("ggml-hex: fa-fold keep-alive: %d iterations per launch, period %d us\n", iters, opt_fa_fold_keepalive_period);
            clSetKernelArg(h->k_keep, 2, sizeof(cl_int), &iters);
            while (!h->keep_stop.load()) {
                const auto t0 = std::chrono::steady_clock::now();
                clEnqueueNDRangeKernel(q, h->k_keep, 1, nullptr, &gsz, &lsz, 0, nullptr, nullptr); clFinish(q);
                h->n_keep++;
                const auto el = std::chrono::steady_clock::now() - t0;
                const auto period = std::chrono::microseconds(opt_fa_fold_keepalive_period);
                if (el < period) std::this_thread::sleep_for(period - el);
            }
            clReleaseCommandQueue(q);
        });
        GGML_LOG_INFO("ggml-hex: fa-fold: keep-alive control ON (%d us busy every %d us; no attention work on the GPU)\n",
                      opt_fa_fold_keepalive, opt_fa_fold_keepalive_period);
    }
    h->relay = std::thread(ggml_hexagon_hfold_relay_main, h);
    sess->hfold = h;
    GGML_LOG_INFO("ggml-hex: %s hetero prefill split enabled (GPU exceptions folded in-store, staged per KV head; relay cpu %d)\n",
                  sess->c_name(), opt_fa_fold_cpu);
    return true;
}

static void ggml_hexagon_hfold_wait_idle(ggml_hexagon_hfold * h) {
    std::unique_lock<std::mutex> lk(h->mu);
    h->cv.wait(lk, [&] { return h->batches.empty() && !h->busy; });
}

// (Re)build the fold buffer for a node shape. Regions: header | m | l | acc | ctl | idx | abs | cnt | lcnt | err | Kt.
static bool ggml_hexagon_hfold_ensure_buffer(ggml_hexagon_hfold * h, uint32_t rows, uint32_t neq1, uint32_t nkvh, uint32_t D,
                                             uint32_t nbk_cap, uint32_t num_sb_cap, bool qbh) {
    auto al = [](size_t v, size_t a) { return (v + a - 1) & ~(a - 1); };
    const bool fits = h->buf && rows <= h->rows && neq1 <= h->neq1 && nkvh == h->nkvh && D == h->D &&
                      nbk_cap <= h->nbk_cap && num_sb_cap <= h->num_sb_cap;
    if (!fits) {
        ggml_hexagon_hfold_wait_idle(h);
        // the old buffer's alias and view go with it
        if (h->buf) {
            auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(h->buf->context);
            if (h->ready_sub) { clReleaseMemObject(h->ready_sub); h->ready_sub = nullptr; }
            auto ia = h->aliases.find(sbuf->fd); if (ia != h->aliases.end()) { clReleaseMemObject(ia->second); h->aliases.erase(ia); }
            auto ii = h->images.find(sbuf->fd);  if (ii != h->images.end())  { clReleaseMemObject(ii->second); h->images.erase(ii); }
            if (h->tctx) { ggml_free(h->tctx); h->tctx = nullptr; }
            ggml_backend_buffer_free(h->buf); h->buf = nullptr;
        }
        rows = std::max(rows, h->rows); neq1 = std::max(neq1, h->neq1);
        nbk_cap = std::max(nbk_cap, h->nbk_cap); num_sb_cap = std::max(num_sb_cap, h->num_sb_cap);
        const size_t ml  = al((size_t) rows * sizeof(float), 128);
        const size_t acc = al((size_t) rows * D * sizeof(float), 128);
        size_t off = 128;
        h->off_m = off; off += ml;
        h->off_l = off; off += ml;
        h->off_acc = off; off += acc;
        h->off_ctl = al(off, 4096); off = h->off_ctl + al(HTP_FA_HETERO_STATUS_OFF + HTP_FA_FOLD_ST_N * sizeof(uint32_t), 4096);
        h->off_idx  = off; off += al((size_t) nkvh * num_sb_cap * nbk_cap * sizeof(int32_t), 128);
        h->off_abs  = off; off += al((size_t) nkvh * nbk_cap * sizeof(int32_t), 128);
        h->off_cnt  = off; off += al((size_t) nkvh * sizeof(int32_t), 128);
        h->off_lcnt = off; off += al((size_t) nkvh * num_sb_cap * sizeof(int32_t), 128);
        h->off_err  = off; off += 256;
        h->off_done  = off; off += al(HTP_FA_FOLD_MAX_STAGES * sizeof(uint32_t), 128);
        h->off_scnt  = off; off += al((HTP_FA_FOLD_MAX_STAGES + 2) * sizeof(uint32_t), 128);   // + the two persistent unit counters
        h->off_stamp = off; off += al(HTP_FA_FOLD_MAX_STAGES * sizeof(uint32_t), 128);
        h->off_kt   = al(off, 4096); off = h->off_kt + (size_t) nkvh * D * ((size_t) nbk_cap * 64) * sizeof(uint16_t);
        h->size = al(off, 1u << 20);
        h->buf = ggml_backend_buft_alloc_buffer(&h->sess->buffer_type, h->size);
        if (!h->buf) { GGML_LOG_ERROR("ggml-hex: fa-fold: fold buffer alloc (%zu MB) failed\n", h->size >> 20); return false; }
        h->base = (uint8_t *) ggml_backend_buffer_get_base(h->buf);
        memset(h->base, 0, h->size);
        hetero_dc_civac(h->base, h->size);
        ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, true };
        h->tctx = ggml_init(ip);
        h->ctrl = ggml_new_tensor_1d(h->tctx, GGML_TYPE_F32, (int64_t) (h->size / 4));
        h->ctrl->buffer = h->buf;
        h->ctrl->data   = h->base;
        ggml_set_name(h->ctrl, "hexagon_fa_fold");
        h->hbuf   = ggml_hexagon_hfold_alias(h, h->buf);
        h->kt_img = h->hbuf ? ggml_hexagon_hfold_image(h, h->buf) : nullptr;
        if (!h->hbuf || !h->kt_img) return false;
        if (h->ready_sub) { clReleaseMemObject(h->ready_sub); h->ready_sub = nullptr; }
        {
            cl_int err2 = CL_SUCCESS;
            cl_buffer_region reg = { h->off_ctl, 4096 };
            h->ready_sub = clCreateSubBuffer(h->hbuf, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err2);
            if (err2 != CL_SUCCESS) { GGML_LOG_WARN("ggml-hex: fa-fold: ready sub-buffer failed (%d); the gate polls SVM only\n", err2); h->ready_sub = nullptr; opt_fa_fold_ion_ready = 0; }
        }
        if (opt_fa_fold_svmtab) {
            if (h->svm_tabs) clSVMFree(h->ctx, h->svm_tabs);
            h->svm_tabs_bytes = h->off_kt - h->off_idx;
            h->svm_tabs = (uint8_t *) clSVMAlloc(h->ctx, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, h->svm_tabs_bytes, 0);
            if (!h->svm_tabs) { GGML_LOG_ERROR("ggml-hex: fa-fold: clSVMAlloc tables failed\n"); return false; }
            memset(h->svm_tabs, 0, h->svm_tabs_bytes);
        }
        if (opt_fa_fold_svmq) {
            if (h->svm_q) clSVMFree(h->ctx, h->svm_q);
            h->svm_q_bytes = (size_t) rows * D * sizeof(float);
            h->svm_q = (uint8_t *) clSVMAlloc(h->ctx, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, h->svm_q_bytes, 0);
            if (!h->svm_q) { GGML_LOG_ERROR("ggml-hex: fa-fold: clSVMAlloc Q failed\n"); return false; }
        }
        h->rows = rows; h->neq1 = neq1; h->nkvh = nkvh; h->D = D; h->nbk_cap = nbk_cap; h->num_sb_cap = num_sb_cap;
        GGML_LOG_INFO("ggml-hex: fa-fold: fold buffer %zu MB (partial %u rows x %u, %u KV heads, up to %u blocks, K^T staging %zu MB)\n",
                      h->size >> 20, rows, D, nkvh, nbk_cap, ((size_t) nkvh * D * nbk_cap * 64 * 2) >> 20);
    }
    // header for THIS shape (the same buffer serves every node of the graph; all share the shape)
    struct htp_fa_fold_hdr hdr = {};
    hdr.magic = HTP_FA_FOLD_MAGIC; hdr.rows = rows; hdr.neq1 = neq1; hdr.dv = D;
    hdr.off_m = (uint32_t) h->off_m; hdr.off_l = (uint32_t) h->off_l; hdr.off_acc = (uint32_t) h->off_acc;
    hdr.flags = HTP_FA_FOLD_F_LIVE | HTP_FA_FOLD_F_FLUSH_KV | HTP_FA_FOLD_F_STAGED | HTP_FA_FOLD_F_INSTORE |
                (opt_fa_fold_flush ? 0u : HTP_FA_FOLD_F_NOFLUSH) | (opt_fa_fold_stage_qb ? HTP_FA_FOLD_F_STAGE_QB : 0u) |
                (qbh ? HTP_FA_FOLD_F_STAGE_QBH : 0u) | (opt_fa_fold_acc16 ? HTP_FA_FOLD_F_ACC_F16 : 0u) | (uint32_t) opt_fa_fold_probe;
    hdr.slot = 0; hdr.off_ctl = (uint32_t) h->off_ctl; hdr.timeout_us = 2000000;
    hdr.off_exc = 0; hdr.exc_nbk = 0;   // the membership is src[8]
    hdr.off_done = (uint32_t) h->off_done;
    const struct htp_fa_fold_hdr * cur = (const struct htp_fa_fold_hdr *) h->base;
    if (memcmp(cur, &hdr, sizeof(hdr)) != 0) {
        ggml_hexagon_hfold_wait_idle(h);
        memcpy(h->base, &hdr, sizeof(hdr));
        hetero_dc_civac(h->base, 256);
    }
    return true;
}

static bool ggml_hexagon_hfold_prepare(ggml_hexagon_session * sess, ggml_tensor * n, const struct htp_fa_kernel_params * kp) {
    ggml_hexagon_hfold * h = sess->hfold;
    const ggml_tensor * q = n->src[0], * k = n->src[1], * v = n->src[2], * m = n->src[3];
    const ggml_tensor * sel = n->src[5], * cnt = n->src[6], * em = n->src[8];
    if (kp->kernel_type != HTP_FA_KERNEL_HMX) return false;
    if (!sel || !cnt || !em || !m) return false;                       // the graph did not build the split
    if (q->ne[1] < 2 || q->ne[3] != 1) return false;
    const int DK = (int) q->ne[0], DV = (int) v->ne[0];
    if (DK != 128 || DV != 128) return false;
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || m->type != GGML_TYPE_F16 || em->type != GGML_TYPE_F32) return false;
    const uint32_t nkvh = (uint32_t) k->ne[2], nh = (uint32_t) q->ne[2];
    if (nkvh == 0 || nh % nkvh != 0 || nkvh > HTP_FA_FOLD_MAX_STAGES) return false;
    const uint32_t G = nh / nkvh, Sq = (uint32_t) q->ne[1], nkv = (uint32_t) k->ne[1];
    if (nkv % 64 != 0 || (uint32_t) v->ne[1] != nkv) return false;
    const uint32_t nbk = nkv / 64, num_sb = (Sq + 63) / 64;
    if ((uint32_t) em->ne[0] != nbk || (uint32_t) (em->ne[1] * em->ne[2]) != num_sb || (uint32_t) em->ne[3] != nkvh || !ggml_is_contiguous(em)) return false;
    if ((uint32_t) m->ne[0] != nkv || (uint32_t) m->ne[1] < Sq || m->ne[2] != 1) return false;
    if (q->nb[0] != 4 || q->nb[1] % 16 || q->nb[2] % 16 || k->nb[0] != 2 || k->nb[1] % 8 || k->nb[2] % 8 ||
        v->nb[0] != 2 || v->nb[1] % 8 || v->nb[2] % 8 || m->nb[1] % 16) return false;
    if (!ggml_backend_buffer_is_hexagon(q->buffer) || !ggml_backend_buffer_is_hexagon(k->buffer) || !ggml_backend_buffer_is_hexagon(v->buffer) ||
        !ggml_backend_buffer_is_hexagon(m->buffer) || !ggml_backend_buffer_is_hexagon(em->buffer)) return false;
    // K^T staging capacity: the whole cache the view sits in, so a longer context later needs no realloc
    const ggml_tensor * root = k->view_src ? k->view_src : k;
    const uint32_t kv_cap = (uint32_t) (k->view_src ? root->ne[1] : k->ne[1]);
    const uint32_t nbk_cap = std::max(nbk, (kv_cap + 63) / 64);
    if (nbk_cap > 256) return false;    // the staging kernel's local compaction table (kv <= 16384)
    const uint32_t n_qbt = (Sq + kp->Br - 1) / kp->Br;
    // (query block, KV head) stages when GPU-side done words are on and they fit the done line
    const bool qbh = opt_fa_fold_gpudone && opt_fa_fold_qbh && opt_fa_fold_stage_qb && n_qbt * nkvh <= HTP_FA_FOLD_MAX_STAGES;
    if (!ggml_hexagon_hfold_ensure_buffer(h, Sq * nh, Sq, nkvh, (uint32_t) DK, nbk_cap, num_sb, qbh)) return false;

    ggml_hexagon_hfold_job job;
    job.node = n;
    job.q = ggml_hexagon_hfold_alias(h, q->buffer);
    job.k = ggml_hexagon_hfold_alias(h, k->buffer);
    job.v_img = ggml_hexagon_hfold_image(h, v->buffer);
    job.m = ggml_hexagon_hfold_alias(h, m->buffer);
    job.em = ggml_hexagon_hfold_alias(h, em->buffer);
    job.k_img = ggml_hexagon_hfold_image(h, k->buffer);
    if (!job.q || !job.k || !job.v_img || !job.m || !job.em || !job.k_img) return false;
    auto off = [](const ggml_tensor * t) { return (uint32_t) ((const uint8_t *) t->data - static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context)->base); };
    job.q_off = off(q); job.q_nb1 = (uint32_t) q->nb[1]; job.q_nb2 = (uint32_t) q->nb[2];
    job.k_off = off(k); job.k_nb1 = (uint32_t) k->nb[1]; job.k_nb2 = (uint32_t) k->nb[2];
    job.v_off = off(v); job.v_nb1 = (uint32_t) v->nb[1]; job.v_nb2 = (uint32_t) v->nb[2];
    job.m_off = off(m); job.m_nb1 = (uint32_t) m->nb[1];
    job.em_off = off(em);
    if (job.v_off % 8 || job.k_off % 8) return false;
    job.Sq = Sq; job.nh = nh; job.nkvh = nkvh; job.G = G; job.nbk = nbk; job.num_sb = num_sb; job.num_qb = (Sq + 31) / 32;
    job.br = kp->Br; job.qbh = qbh;
    job.n_stages = qbh ? n_qbt * nkvh : opt_fa_fold_stage_qb ? n_qbt : nkvh;
    if (job.br % 32 || job.n_stages > (qbh ? HTP_FA_FOLD_MAX_STAGES : HTP_FA_FOLD_LINE_STAGES)) return false;
    job.em_host = (const float *) em->data; job.em_bytes = ggml_nbytes(em);
    memcpy(&job.scale, &n->op_params[0], sizeof(float));
    auto hbase = [](const ggml_tensor * t) { return (const uint8_t *) static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context)->base; };
    job.q_base = hbase(q); job.k_base = hbase(k); job.v_base = hbase(v); job.m_base = hbase(m);
    n->src[7] = h->ctrl;
    h->jobs[n] = job;
    if (opt_verbose || h->jobs.size() == 1) {
        GGML_LOG_INFO("ggml-hex: fa-fold: tagged %s: %u tokens x %u heads (%u KV heads, G %u), %u KV blocks, %u sub-blocks\n",
                      n->name, Sq, nh, nkvh, G, nbk, num_sb);
    }
    return true;
}

// Once per graph_compute with the nodes about to be queued, in execution order.
static void ggml_hexagon_hfold_post(ggml_hexagon_session * sess, const std::vector<htp_opnode> & nodes) {
    ggml_hexagon_hfold * h = sess->hfold;
    std::vector<ggml_hexagon_hfold_job> batch;
    {
        std::lock_guard<std::mutex> lk(h->mu);
        for (const auto & node : nodes) {
            if (node.opcode != HTP_OP_FLASH_ATTN_EXT) continue;
            auto it = h->jobs.find(node.node);
            if (it == h->jobs.end() || node.node->src[7] != h->ctrl) continue;
            ggml_hexagon_hfold_job job = it->second;
            job.want = ++h->seq;
            batch.push_back(job);
        }
        if (batch.empty()) return;
        // The first op's chain goes in from here, before the DSP is even dispatched: the relay
        // thread's wake-up can lag by milliseconds when the CPU is busy (perplexity's logits),
        // and the first attention op of a graph is only a few ops of DSP work away.
        if (opt_fa_fold_gpudone && opt_fa_fold_preq != 0 && h->hbuf) {
            hfold_enqueue_chain(h, batch[0], batch[0].chain);
        }
        h->batches.push_back(std::move(batch));
    }
    h->cv.notify_all();
}

static void hfold_set_tabs(ggml_hexagon_hfold * h, cl_kernel kk, int idx) {
    if (opt_fa_fold_svmtab) clSetKernelArgSVMPointer(kk, idx, h->svm_tabs); else clSetKernelArg(kk, idx, sizeof(cl_mem), &h->hbuf);
}
// the SVM control block; kernels that get want == 0 never read it, so any buffer will do
static void hfold_set_svm(ggml_hexagon_hfold * h, cl_kernel kk, int idx) {
    if (h->svm_ok) clSetKernelArgSVMPointer(kk, idx, h->svm_ctl); else clSetKernelArg(kk, idx, sizeof(cl_mem), &h->hbuf);
}
struct hfold_tab_offs { cl_int idx, abs, cnt, lcnt, err, scnt; };
static hfold_tab_offs hfold_tabs_of(ggml_hexagon_hfold * h, bool svm_tabs) {
    const size_t tab0 = svm_tabs ? h->off_idx : 0;
    hfold_tab_offs o;
    o.idx = (cl_int) (h->off_idx - tab0); o.abs = (cl_int) (h->off_abs - tab0); o.cnt = (cl_int) (h->off_cnt - tab0);
    o.lcnt = (cl_int) (h->off_lcnt - tab0); o.err = (cl_int) (h->off_err - tab0); o.scnt = (cl_int) (h->off_scnt - tab0);
    return o;
}
static void hfold_args_stage_k(ggml_hexagon_hfold * h, const ggml_hexagon_hfold_job & j, const hfold_tab_offs & to, cl_int n_stage, cl_uint want) {
    const cl_int k_off = (cl_int) j.k_off, k_nb1 = (cl_int) j.k_nb1, k_nb2 = (cl_int) j.k_nb2;
    const cl_int abs_stride = (cl_int) h->nbk_cap, kt_base_tex = (cl_int) (h->off_kt / 8);
    int a = 0;
    clSetKernelArg(h->k_stage, a++, sizeof(cl_mem), &j.k);
    clSetKernelArg(h->k_stage, a++, sizeof(cl_int), &k_off); clSetKernelArg(h->k_stage, a++, sizeof(cl_int), &k_nb1); clSetKernelArg(h->k_stage, a++, sizeof(cl_int), &k_nb2);
    clSetKernelArg(h->k_stage, a++, sizeof(cl_mem), &h->kt_img); clSetKernelArg(h->k_stage, a++, sizeof(cl_int), &kt_base_tex);
    hfold_set_tabs(h, h->k_stage, a++);
    clSetKernelArg(h->k_stage, a++, sizeof(cl_int), &to.abs); clSetKernelArg(h->k_stage, a++, sizeof(cl_int), &to.cnt);
    clSetKernelArg(h->k_stage, a++, sizeof(cl_int), &abs_stride); clSetKernelArg(h->k_stage, a++, sizeof(cl_int), &n_stage);
    hfold_set_svm(h, h->k_stage, a++); clSetKernelArg(h->k_stage, a++, sizeof(cl_uint), &want);
}
static void hfold_args_fa(ggml_hexagon_hfold * h, const ggml_hexagon_hfold_job & j, const hfold_tab_offs & to, cl_int n_stage, cl_uint want, cl_int chain, cl_int gpu_done) {
    const cl_int q_off = opt_fa_fold_svmq ? 0 : (cl_int) j.q_off, q_nb1 = (cl_int) j.q_nb1, q_nb2 = (cl_int) j.q_nb2;
    const cl_int m_off = (cl_int) j.m_off, m_nb1 = (cl_int) j.m_nb1;
    const cl_int f_m = (cl_int) h->off_m, f_l = (cl_int) h->off_l, f_acc = (cl_int) h->off_acc;
    const cl_int Sq = (cl_int) j.Sq, G = (cl_int) j.G, num_sb = (cl_int) j.num_sb, qb_per_sb = 2, top_k = (cl_int) h->nbk_cap;
    const cl_int abs_stride = (cl_int) h->nbk_cap;
    const cl_int kt_base_tex = (cl_int) (h->off_kt / 8), v_base_tex = (cl_int) (j.v_off / 8);
    const cl_int v_tok_tex = (cl_int) (j.v_nb1 / 8), v_head_tex = (cl_int) (j.v_nb2 / 8);
    const cl_int qbh = j.qbh ? 1 : 0, nkvh = (cl_int) j.nkvh, qb32ps = (cl_int) (j.br / 32);
    const cl_int done_off = (cl_int) h->off_done, stamp_off = (cl_int) h->off_stamp;
    int a = 0;
    clSetKernelArg(h->k_fa, a++, sizeof(cl_mem), &h->kt_img); clSetKernelArg(h->k_fa, a++, sizeof(cl_mem), &j.v_img);
    if (opt_fa_fold_svmq) clSetKernelArgSVMPointer(h->k_fa, a++, h->svm_q); else clSetKernelArg(h->k_fa, a++, sizeof(cl_mem), &j.q);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &q_off);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &q_nb1); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &q_nb2);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_mem), &j.m); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &m_off); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &m_nb1);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_mem), &h->hbuf); hfold_set_tabs(h, h->k_fa, a++);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &to.idx); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &to.abs);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &to.cnt); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &to.lcnt); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &to.err);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &f_m); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &f_l); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &f_acc);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &Sq); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &G); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &num_sb);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &qb_per_sb); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &top_k);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &abs_stride); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &n_stage);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &kt_base_tex); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &v_base_tex);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &v_tok_tex); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &v_head_tex);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_float), &j.scale);
    hfold_set_svm(h, h->k_fa, a++); clSetKernelArg(h->k_fa, a++, sizeof(cl_uint), &want);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &chain); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &gpu_done); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &qbh);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &nkvh); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &qb32ps);
    const cl_int n_stages = (cl_int) j.n_stages;
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &to.scnt); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &done_off); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &stamp_off);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &n_stages);
    const cl_int k_base_tex = (cl_int) (j.k_off / 8), k_tok_tex = (cl_int) (j.k_nb1 / 8), k_head_tex = (cl_int) (j.k_nb2 / 8);
    const cl_int em_off = (cl_int) j.em_off, nbk = (cl_int) j.nbk;
    clSetKernelArg(h->k_fa, a++, sizeof(cl_mem), &j.k_img);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &k_base_tex); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &k_tok_tex); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &k_head_tex);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_mem), &j.em); clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &em_off);
    clSetKernelArg(h->k_fa, a++, sizeof(cl_int), &nbk);
}
// one fa_exc launch per stage with the host's offsets (legacy mapping); events out
static void hfold_launch_fa_stages(ggml_hexagon_hfold * h, const ggml_hexagon_hfold_job & j, std::vector<cl_event> & ev, bool full_qb) {
    const uint32_t qb32_per_stage = j.br / 32;
    const size_t lsz[3] = { 128, 1, 1 };
    for (uint32_t st = 0; st < j.n_stages; ++st) {
        size_t off3[3], gsz[3];
        if (j.qbh) {
            const uint32_t qb = st / j.nkvh, kvh = st % j.nkvh;
            off3[0] = 0; off3[1] = (size_t) qb * qb32_per_stage; off3[2] = (size_t) kvh * j.G;
            gsz[0] = 128; gsz[1] = full_qb ? qb32_per_stage : std::min<size_t>(qb32_per_stage, j.num_qb - off3[1]); gsz[2] = j.G;
        } else if (opt_fa_fold_stage_qb) {
            off3[0] = 0; off3[1] = (size_t) st * qb32_per_stage; off3[2] = 0;
            gsz[0] = 128; gsz[1] = full_qb ? qb32_per_stage : std::min<size_t>(qb32_per_stage, j.num_qb - off3[1]); gsz[2] = j.nh;
        } else {
            off3[0] = 0; off3[1] = 0; off3[2] = (size_t) st * j.G;
            gsz[0] = 128; gsz[1] = j.num_qb; gsz[2] = j.G;
        }
        const cl_int err = clEnqueueNDRangeKernel(h->q, h->k_fa, 3, off3, gsz, lsz, 0, nullptr, &ev[st]);
        if (err != CL_SUCCESS) { h->n_gpu_err++; GGML_LOG_ERROR("ggml-hex: fa-fold: enqueue fa_exc stage %u failed (%d)\n", st, err); ev[st] = nullptr; }
        // one submission per launch: batched into one command buffer the driver reports every
        // event complete together at the END, and the staging degenerates to a single wait
        clFlush(h->q);
    }
}
// the membership -> per-head compact tables, on the host (legacy path and the chain's fallback)

static void hfold_enqueue_chain(ggml_hexagon_hfold * h, const ggml_hexagon_hfold_job & j, hfold_chain & c) {
    const hfold_tab_offs to = hfold_tabs_of(h, false);
    const cl_uint want = j.want;
    const cl_int max_spins = opt_fa_fold_gate_spins;
    const cl_int n_stage = (cl_int) h->nbk_cap * 64;   // capacity: the host does not know the list yet
    cl_int err;
    {
        const cl_int n_stages = (cl_int) j.n_stages;
        int a = 0;
        hfold_set_svm(h, h->k_gate, a++); clSetKernelArg(h->k_gate, a++, sizeof(cl_uint), &want); clSetKernelArg(h->k_gate, a++, sizeof(cl_int), &max_spins);
        clSetKernelArg(h->k_gate, a++, sizeof(cl_mem), &h->hbuf); clSetKernelArg(h->k_gate, a++, sizeof(cl_int), &to.scnt);
        clSetKernelArg(h->k_gate, a++, sizeof(cl_int), &to.err); clSetKernelArg(h->k_gate, a++, sizeof(cl_int), &n_stages);
        // the ready word as a sub-buffer of the alias (the kernel takes a bare pointer)
        if (opt_fa_fold_ion_ready) clSetKernelArg(h->k_gate, a++, sizeof(cl_mem), &h->ready_sub); else { cl_mem nul = nullptr; clSetKernelArg(h->k_gate, a++, sizeof(cl_mem), &nul); }
        const cl_int delay_polls = opt_fa_fold_gate_delay;
        clSetKernelArg(h->k_gate, a++, sizeof(cl_int), &delay_polls);
        const size_t one = 1;
        err = clEnqueueNDRangeKernel(h->q, h->k_gate, 1, nullptr, &one, &one, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { h->n_gpu_err++; GGML_LOG_ERROR("ggml-hex: fa-fold: enqueue gate failed (%d)\n", err); }
    }
    if (opt_fa_fold_qmap == 3) {
        // In-order migration of the DSP-written inputs to the host and (implicitly, at next use)
        // back: the runtime must then treat them as host-modified and invalidate its caches.
        cl_mem mems[2] = { j.q, j.em };
        cl_int e2 = clEnqueueMigrateMemObjects(h->q, 2, mems, CL_MIGRATE_MEM_OBJECT_HOST, 0, nullptr, nullptr);
        if (e2 != CL_SUCCESS) { h->n_gpu_err++; GGML_LOG_ERROR("ggml-hex: fa-fold: chain migrate failed (%d)\n", e2); }
    } else if (opt_fa_fold_qmap) {
        // In-order map/unmap of the DSP-written inputs the GPU reads through the alias: executed
        // on the queue after the gate, so the driver's cache maintenance for these ranges happens
        // right before the kernels that read them, with no host involvement at run time.
        auto qmap = [&](cl_mem mem, size_t off, size_t bytes) {
            cl_int e2 = CL_SUCCESS;
            void * p = clEnqueueMapBuffer(h->q, mem, CL_FALSE, CL_MAP_WRITE_INVALIDATE_REGION, off, bytes, 0, nullptr, nullptr, &e2);
            if (e2 != CL_SUCCESS || !p) { h->n_gpu_err++; GGML_LOG_ERROR("ggml-hex: fa-fold: chain map failed (%d)\n", e2); return; }
            e2 = clEnqueueUnmapMemObject(h->q, mem, p, 0, nullptr, nullptr);
            if (e2 != CL_SUCCESS) { h->n_gpu_err++; GGML_LOG_ERROR("ggml-hex: fa-fold: chain unmap failed (%d)\n", e2); }
        };
        qmap(j.q, j.q_off, (size_t) (j.Sq - 1) * j.q_nb1 + (size_t) (j.nh - 1) * j.q_nb2 + h->D * sizeof(float));
        qmap(j.em, j.em_off, j.em_bytes);
        if (opt_fa_fold_qmap > 1) qmap(j.m, j.m_off, (size_t) j.Sq * j.m_nb1);
    }
    if (!opt_fa_fold_knat) {
        const cl_int k_off = (cl_int) j.k_off, k_nb1 = (cl_int) j.k_nb1, k_nb2 = (cl_int) j.k_nb2;
        const cl_int abs_stride = (cl_int) h->nbk_cap, kt_base_tex = (cl_int) (h->off_kt / 8), nkvh = (cl_int) j.nkvh;
        const cl_int em_off = (cl_int) j.em_off, nbk = (cl_int) j.nbk, num_sb = (cl_int) j.num_sb;
        int a = 0;
        clSetKernelArg(h->k_stage_p, a++, sizeof(cl_mem), &j.k);
        clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &k_off); clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &k_nb1); clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &k_nb2);
        clSetKernelArg(h->k_stage_p, a++, sizeof(cl_mem), &h->kt_img); clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &kt_base_tex);
        clSetKernelArg(h->k_stage_p, a++, sizeof(cl_mem), &h->hbuf);
        clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &to.abs); clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &to.cnt);
        clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &abs_stride); clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &n_stage);
        clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &nkvh); clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &to.scnt);
        hfold_set_svm(h, h->k_stage_p, a++); clSetKernelArg(h->k_stage_p, a++, sizeof(cl_uint), &want);
        clSetKernelArg(h->k_stage_p, a++, sizeof(cl_mem), &j.em); clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &em_off);
        clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &nbk); clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &num_sb);
        clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &to.idx); clSetKernelArg(h->k_stage_p, a++, sizeof(cl_int), &to.lcnt);
        const size_t g = (size_t) std::max(1, opt_fa_fold_pwg_stage) * 128, l = 128;
        err = clEnqueueNDRangeKernel(h->q, h->k_stage_p, 1, nullptr, &g, &l, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { h->n_gpu_err++; GGML_LOG_ERROR("ggml-hex: fa-fold: enqueue stage_k_p failed (%d)\n", err); }
    }
    if (opt_fa_fold_chain) {
        hfold_args_fa(h, j, to, n_stage, want, opt_fa_fold_chain, 1);
        c.ev.assign(1, nullptr);
        const size_t n_units = (size_t) (j.br / 32) * (j.qbh ? j.G : j.nh) * j.n_stages;
        const size_t n_pwg   = std::min<size_t>(std::max(1, opt_fa_fold_pwg), n_units);
        const size_t gsz[3]  = { 128, opt_fa_fold_chain == 2 ? n_pwg : (size_t) (j.br / 32) * (j.qbh ? j.G : j.nh), opt_fa_fold_chain == 2 ? 1 : (size_t) j.n_stages };
        const size_t lsz[3]  = { 128, 1, 1 };
        err = clEnqueueNDRangeKernel(h->q, h->k_fa, 3, nullptr, gsz, lsz, 0, nullptr, &c.ev[0]);
        if (err != CL_SUCCESS) { h->n_gpu_err++; GGML_LOG_ERROR("ggml-hex: fa-fold: enqueue fa_exc chain failed (%d)\n", err); c.ev[0] = nullptr; }
        clFlush(h->q);
    } else {
        hfold_args_fa(h, j, to, n_stage, want, 0, 1);
        c.ev.assign(j.n_stages, nullptr);
        hfold_launch_fa_stages(h, j, c.ev, true);
    }
    c.enq = true; c.t_enq = std::chrono::steady_clock::now();
}

static void ggml_hexagon_hfold_relay_main(ggml_hexagon_hfold * h) {
#if defined(__linux__)
    if (opt_fa_fold_cpu >= 0) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(opt_fa_fold_cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
#endif
    using clock_t = std::chrono::steady_clock;
    auto us_between = [](clock_t::time_point a, clock_t::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
    std::vector<int32_t> to_compact;
    // the buffer is allocated (and may be reallocated) after this thread starts: refreshed per batch
    volatile uint32_t * ready = nullptr, * done = nullptr, * stamp = nullptr;

    // ---- kernel argument setters shared by the legacy and the chain path ----
    auto host_tables = [&](const ggml_hexagon_hfold_job & j, uint32_t & n_stage_blk) {
        hetero_dc_civac(j.em_host, j.em_bytes);
        uint8_t * tabs_base = opt_fa_fold_svmtab ? h->svm_tabs : h->base;
        const size_t tab0   = opt_fa_fold_svmtab ? h->off_idx : 0;
        int32_t * idx_h  = (int32_t *) (tabs_base + h->off_idx - tab0);
        int32_t * abs_h  = (int32_t *) (tabs_base + h->off_abs - tab0);
        int32_t * cnt_h  = (int32_t *) (tabs_base + h->off_cnt - tab0);
        int32_t * lcnt_h = (int32_t *) (tabs_base + h->off_lcnt - tab0);
        to_compact.assign(j.nbk, -1);
        n_stage_blk = 0;
        for (uint32_t kvh = 0; kvh < j.nkvh; ++kvh) {
            std::fill(to_compact.begin(), to_compact.end(), -1);
            for (uint32_t sb = 0; sb < j.num_sb; ++sb) {
                const float * row = j.em_host + ((size_t) kvh * j.num_sb + sb) * j.nbk;
                for (uint32_t b = 0; b < j.nbk; ++b) if (row[b] != 0.0f) to_compact[b] = 0;
            }
            int32_t c = 0;
            for (uint32_t b = 0; b < j.nbk; ++b) {
                if (to_compact[b] == 0) { abs_h[(size_t) kvh * h->nbk_cap + c] = (int32_t) b; to_compact[b] = c++; }
            }
            for (int32_t i = c; i < (int32_t) h->nbk_cap; ++i) abs_h[(size_t) kvh * h->nbk_cap + i] = -1;
            cnt_h[kvh] = c * 64;
            n_stage_blk = std::max<uint32_t>(n_stage_blk, (uint32_t) c);
            for (uint32_t sb = 0; sb < j.num_sb; ++sb) {
                const float * row = j.em_host + ((size_t) kvh * j.num_sb + sb) * j.nbk;
                int32_t * irow = idx_h + ((size_t) kvh * j.num_sb + sb) * h->nbk_cap;
                int32_t s = 0;
                for (uint32_t b = 0; b < j.nbk; ++b) if (row[b] != 0.0f) irow[s++] = to_compact[b];
                lcnt_h[(size_t) kvh * j.num_sb + sb] = s;
            }
        }
        *(int32_t *) (tabs_base + h->off_err - tab0) = 0;
        if (opt_fa_fold_check) {
            uint32_t nz = 0, lsum = 0;
            for (size_t i = 0; i < (size_t) j.nkvh * j.num_sb * j.nbk; ++i) nz += j.em_host[i] != 0.0f;
            for (size_t i = 0; i < (size_t) j.nkvh * j.num_sb; ++i) lsum += (uint32_t) lcnt_h[i];
            GGML_LOG_INFO("ggml-hex: fa-fold relay: want %u: em %p (%u blocks x %u sb x %u heads) nonzero %u -> list entries %u, staged blocks/head max %u, Sq %u nh %u k_off %u q_off %u m_off %u\n",
                          j.want, (const void *) j.em_host, j.nbk, j.num_sb, j.nkvh, nz, lsum, n_stage_blk, j.Sq, j.nh, j.k_off, j.q_off, j.m_off);
        }
        if (!opt_fa_fold_svmtab) {
            hetero_dc_civac(h->base + h->off_idx, h->off_kt - h->off_idx);
        }
        if (opt_fa_fold_svmq) {
            const size_t qbytes = (size_t) (j.Sq - 1) * j.q_nb1 + (size_t) (j.nh - 1) * j.q_nb2 + h->D * sizeof(float);
            hetero_dc_civac(j.q_base + j.q_off, qbytes);
            memcpy(h->svm_q, j.q_base + j.q_off, std::min(qbytes, h->svm_q_bytes));
        }
    };
    // host-driven GPU work after ready: tables, cache sync probe, stage_k, one fa_exc per stage,
    // done words written here as each event completes (held until the reference ran in check mode)
    auto legacy_gpu = [&](const ggml_hexagon_hfold_job & j, double & b_tab, double & b_gpu, uint32_t & b_blk, clock_t::time_point t_ready) {
        uint32_t n_stage_blk = 0;
        host_tables(j, n_stage_blk);
        const auto t_tab = clock_t::now();
        b_tab += us_between(t_ready, t_tab);
        cl_int err = CL_SUCCESS;
        auto sync = [&](cl_mem mem, size_t off, size_t bytes) {
            if (!bytes || !opt_fa_fold_sync) return;
            void * p = clEnqueueMapBuffer(h->q, mem, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, off, bytes, 0, nullptr, nullptr, &err);
            if (err != CL_SUCCESS || !p) { h->n_gpu_err++; GGML_LOG_ERROR("ggml-hex: fa-fold: map for sync failed (%d)\n", err); return; }
            err = clEnqueueUnmapMemObject(h->q, mem, p, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) { h->n_gpu_err++; GGML_LOG_ERROR("ggml-hex: fa-fold: unmap for sync failed (%d)\n", err); }
        };
        sync(j.q, j.q_off, (size_t) (j.Sq - 1) * j.q_nb1 + (size_t) (j.nh - 1) * j.q_nb2 + h->D * sizeof(float));
        sync(j.m, j.m_off, (size_t) j.Sq * j.m_nb1);
        sync(j.k, j.k_off, (size_t) (j.nbk * 64 - 1) * j.k_nb1 + (size_t) (j.nkvh - 1) * j.k_nb2 + h->D * sizeof(uint16_t));
        {
            auto it = h->aliases.find(static_cast<ggml_hexagon_shared_buffer *>(const_cast<ggml_tensor *>(j.node)->src[2]->buffer->context)->fd);
            if (it != h->aliases.end()) sync(it->second, j.v_off, (size_t) (j.nbk * 64 - 1) * j.v_nb1 + (size_t) (j.nkvh - 1) * j.v_nb2 + h->D * sizeof(uint16_t));
        }
        sync(h->hbuf, h->off_idx, h->off_kt - h->off_idx);
        sync(h->hbuf, h->off_m, h->off_ctl - h->off_m);
        const cl_int n_stage = (cl_int) std::max<uint32_t>(n_stage_blk, 1) * 64;
        const hfold_tab_offs to = hfold_tabs_of(h, opt_fa_fold_svmtab);
        std::vector<cl_event> ev(j.n_stages, nullptr);
        if (n_stage_blk > 0) {
            hfold_args_stage_k(h, j, to, n_stage, 0u);
            const size_t g_stage[3] = { (size_t) n_stage / 4, (size_t) h->D / 8, (size_t) j.nkvh };
            err = clEnqueueNDRangeKernel(h->q, h->k_stage, 3, nullptr, g_stage, nullptr, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) { h->n_gpu_err++; GGML_LOG_ERROR("ggml-hex: fa-fold: enqueue stage_k failed (%d)\n", err); }
            hfold_args_fa(h, j, to, n_stage, 0u, 0, 0);
            hfold_launch_fa_stages(h, j, ev, false);
        }
        for (uint32_t st = 0; st < j.n_stages; ++st) {
            if (ev[st]) { clWaitForEvents(1, &ev[st]); clReleaseEvent(ev[st]); }
            if (!opt_fa_fold_check) {
                __atomic_store_n(done + st, j.want, __ATOMIC_RELEASE);
                hetero_dc_civac((const void *) (done + st), sizeof(uint32_t));
            }
        }
        h->us_gpu += us_between(t_ready, clock_t::now());
        b_gpu += us_between(t_tab, clock_t::now());
        b_blk += n_stage_blk;
        if (opt_fa_fold_check) {
            // Host reference for a few rows: the first sub-block with exceptions of KV heads 0 and 1,
            // query heads kvh*G, tokens sb*64 + {0, 17}. Reads the same hexagon buffers the GPU did.
            hetero_dc_civac(h->base + h->off_m, (size_t) j.Sq * j.nh * sizeof(float));
            hetero_dc_civac(h->base + h->off_l, (size_t) j.Sq * j.nh * sizeof(float));
            hetero_dc_civac(h->base + h->off_acc, (size_t) j.Sq * j.nh * h->D * sizeof(float));
            for (uint32_t kvh = 0; kvh < std::min<uint32_t>(2, j.nkvh); ++kvh) {
                int sb_pick = -1;
                for (uint32_t sb = 0; sb < j.num_sb && sb_pick < 0; ++sb) {
                    const float * row = j.em_host + ((size_t) kvh * j.num_sb + sb) * j.nbk;
                    for (uint32_t b = 0; b < j.nbk; ++b) if (row[b] != 0.0f) { sb_pick = (int) sb; break; }
                }
                if (sb_pick < 0) { GGML_LOG_INFO("ggml-hex: fa-fold check: kvh %u has no exceptions\n", kvh); continue; }
                const float * row = j.em_host + ((size_t) kvh * j.num_sb + sb_pick) * j.nbk;
                const uint32_t bh = kvh * j.G;
                for (int tt = 0; tt < 2; ++tt) {
                    const uint32_t t = (uint32_t) sb_pick * 64 + (tt ? 17 : 0);
                    if (t >= j.Sq) continue;
                    const float * qr = (const float *) (j.q_base + j.q_off + (size_t) t * j.q_nb1 + (size_t) bh * j.q_nb2);
                    hetero_dc_civac(qr, h->D * sizeof(float));
                    const uint16_t * mr = (const uint16_t *) (j.m_base + j.m_off + (size_t) t * j.m_nb1);
                    hetero_dc_civac(mr, (size_t) j.nbk * 64 * sizeof(uint16_t));
                    double m_ref = -INFINITY, l_ref = 0; std::vector<double> acc(h->D, 0.0), sc;
                    std::vector<uint32_t> keys;
                    for (uint32_t b = 0; b < j.nbk; ++b) if (row[b] != 0.0f) for (uint32_t i = 0; i < 64; ++i) keys.push_back(b * 64 + i);
                    for (uint32_t key : keys) {
                        const float mk = ggml_fp16_to_fp32(mr[key]);
                        if (std::isinf(mk)) { sc.push_back(-INFINITY); continue; }
                        const uint16_t * kr = (const uint16_t *) (j.k_base + j.k_off + (size_t) key * j.k_nb1 + (size_t) kvh * j.k_nb2);
                        hetero_dc_civac(kr, h->D * sizeof(uint16_t));
                        double dot = 0; for (uint32_t e = 0; e < h->D; ++e) dot += (double) qr[e] * ggml_fp16_to_fp32(kr[e]);
                        sc.push_back(dot * j.scale + mk); m_ref = std::max(m_ref, sc.back());
                    }
                    for (size_t i = 0; i < keys.size(); ++i) {
                        if (std::isinf(sc[i])) continue;
                        const uint16_t * vr = (const uint16_t *) (j.v_base + j.v_off + (size_t) keys[i] * j.v_nb1 + (size_t) kvh * j.v_nb2);
                        hetero_dc_civac(vr, h->D * sizeof(uint16_t));
                        const double pw = exp(sc[i] - m_ref); l_ref += pw;
                        for (uint32_t e = 0; e < h->D; ++e) acc[e] += pw * ggml_fp16_to_fp32(vr[e]);
                    }
                    const size_t r = (size_t) bh * j.Sq + t;
                    const float m_g = ((const float *) (h->base + h->off_m))[r], l_g = ((const float *) (h->base + h->off_l))[r];
                    std::vector<float> a_g(h->D);
                    for (uint32_t e = 0; e < h->D; ++e) a_g[e] = opt_fa_fold_acc16 ? ggml_fp16_to_fp32(((const uint16_t *) (h->base + h->off_acc))[r * h->D + e]) : ((const float *) (h->base + h->off_acc))[r * h->D + e];
                    double da = 0, na = 0; for (uint32_t e = 0; e < h->D; ++e) { da = std::max(da, fabs(a_g[e] - acc[e])); na = std::max(na, fabs(acc[e])); }
                    GGML_LOG_INFO("ggml-hex: fa-fold check: want %u kvh %u sb %d tok %u (%zu exc keys, %zu unmasked): m gpu %.5f ref %.5f | l gpu %.4f ref %.4f | acc max|diff| %.3e of %.3e%s\n",
                                  j.want, kvh, sb_pick, t, keys.size(), (size_t) std::count_if(sc.begin(), sc.end(), [](double x) { return !std::isinf(x); }),
                                  m_g, m_ref, l_g, l_ref, da, na, (fabs(m_g - m_ref) > 1e-3 || fabs(l_g - l_ref) > 1e-3 * std::max(1.0, l_ref)) ? "  MISMATCH" : "");
                }
            }
            for (uint32_t st = 0; st < j.n_stages; ++st) {
                __atomic_store_n(done + st, j.want, __ATOMIC_RELEASE);
                hetero_dc_civac((const void *) (done + st), sizeof(uint32_t));
            }
        }
        if (opt_verbose > 1) {
            if (!opt_fa_fold_svmtab) hetero_dc_civac(h->base + h->off_err, 4);
            GGML_LOG_DEBUG("ggml-hex: fa-fold relay: job want %u: %u staged blocks, GPU %u stages done %.0f us after ready, kernel errors %d\n",
                           j.want, n_stage_blk, j.n_stages, us_between(t_ready, clock_t::now()),
                           *(volatile int32_t *) ((opt_fa_fold_svmtab ? h->svm_tabs - h->off_idx : h->base) + h->off_err));
        }
    };
    // poll the DSP's ready word up to 3 s; tight for 200 us, then 20 us naps. `poll_hook` runs
    // each turn (the chain path enqueues the next chain from here when its time comes).
    // `t_tight`: from this time on (the predicted ready minus a margin) poll without napping
    auto wait_ready = [&](uint32_t want, const std::function<void()> & poll_hook, clock_t::time_point t_tight) {
        const auto t0 = clock_t::now();
        for (;;) {
            hetero_dc_civac((const void *) ready, 4);
            if (__atomic_load_n(ready, __ATOMIC_ACQUIRE) == want) return true;
            const auto now = clock_t::now();
            const auto el  = now - t0;
            if (el > std::chrono::milliseconds(3000)) return false;
            if (poll_hook) poll_hook();
            if (el > std::chrono::microseconds(200) && now < t_tight) std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
    };

    // CHECK=2 (chain path): at `ready`, snapshot the membership row and the Q row of a few
    // exception rows (the DSP does not touch them again during the op, the next layer's ops do);
    // after the last stage, recompute those rows on the host from the snapshot plus K, V and the
    // mask (stable for the graph) and compare with the GPU's (m, l, acc).
    struct check_pick { uint32_t kvh, sb, tok, bh; std::vector<float> em_row, q_row; };
    std::vector<check_pick> picks;
    auto check_snapshot = [&](const ggml_hexagon_hfold_job & j, int n_rows) {
        picks.clear();
        hetero_dc_civac(j.em_host, j.em_bytes);
        for (uint32_t kvh = 0; kvh < j.nkvh && (int) picks.size() < n_rows; kvh += 3) {
            for (uint32_t sb = 1; sb < j.num_sb; sb += 2) {
                const float * row = j.em_host + ((size_t) kvh * j.num_sb + sb) * j.nbk;
                bool any = false; for (uint32_t b = 0; b < j.nbk; ++b) any |= row[b] != 0.0f;
                if (!any) continue;
                // two rows per pick, one in each wave of the kernel's 32-row block (offsets 37 and 53)
                for (uint32_t off = 37; off <= 53; off += 16) {
                    check_pick p; p.kvh = kvh; p.sb = sb; p.tok = sb * 64 + off; p.bh = kvh * j.G + 1;
                    if (p.tok >= j.Sq) break;
                    p.em_row.assign(row, row + j.nbk);
                    const float * qr = (const float *) (j.q_base + j.q_off + (size_t) p.tok * j.q_nb1 + (size_t) p.bh * j.q_nb2);
                    hetero_dc_civac(qr, h->D * sizeof(float));
                    p.q_row.assign(qr, qr + h->D);
                    picks.push_back(std::move(p));
                }
                break;
            }
        }
    };
    auto check_verify = [&](const ggml_hexagon_hfold_job & j) {
        for (const auto & p : picks) {
            const uint16_t * mr = (const uint16_t *) (j.m_base + j.m_off + (size_t) p.tok * j.m_nb1);
            hetero_dc_civac(mr, (size_t) j.nbk * 64 * sizeof(uint16_t));
            double m_ref = -INFINITY, l_ref = 0; std::vector<double> acc(h->D, 0.0), sc; std::vector<uint32_t> keys;
            for (uint32_t b = 0; b < j.nbk; ++b) if (p.em_row[b] != 0.0f) for (uint32_t i = 0; i < 64; ++i) keys.push_back(b * 64 + i);
            for (uint32_t key : keys) {
                const float mk = ggml_fp16_to_fp32(mr[key]);
                if (std::isinf(mk)) { sc.push_back(-INFINITY); continue; }
                const uint16_t * kr = (const uint16_t *) (j.k_base + j.k_off + (size_t) key * j.k_nb1 + (size_t) p.kvh * j.k_nb2);
                hetero_dc_civac(kr, h->D * sizeof(uint16_t));
                double dot = 0; for (uint32_t e = 0; e < h->D; ++e) dot += (double) p.q_row[e] * ggml_fp16_to_fp32(kr[e]);
                sc.push_back(dot * j.scale + mk); m_ref = std::max(m_ref, sc.back());
            }
            for (size_t i = 0; i < keys.size(); ++i) {
                if (std::isinf(sc[i])) continue;
                const uint16_t * vr = (const uint16_t *) (j.v_base + j.v_off + (size_t) keys[i] * j.v_nb1 + (size_t) p.kvh * j.v_nb2);
                hetero_dc_civac(vr, h->D * sizeof(uint16_t));
                const double pw = exp(sc[i] - m_ref); l_ref += pw;
                for (uint32_t e = 0; e < h->D; ++e) acc[e] += pw * ggml_fp16_to_fp32(vr[e]);
            }
            const size_t r = (size_t) p.bh * j.Sq + p.tok;
            hetero_dc_civac(h->base + h->off_m + r * 4, 4); hetero_dc_civac(h->base + h->off_l + r * 4, 4);
            const size_t acc_esz = opt_fa_fold_acc16 ? 2 : 4;
            hetero_dc_civac(h->base + h->off_acc + r * h->D * acc_esz, h->D * acc_esz);
            const float m_g = ((const float *) (h->base + h->off_m))[r], l_g = ((const float *) (h->base + h->off_l))[r];
            std::vector<float> a_g(h->D);
            for (uint32_t e = 0; e < h->D; ++e) a_g[e] = opt_fa_fold_acc16 ? ggml_fp16_to_fp32(((const uint16_t *) (h->base + h->off_acc))[r * h->D + e]) : ((const float *) (h->base + h->off_acc))[r * h->D + e];
            double da = 0, na = 0; for (uint32_t e = 0; e < h->D; ++e) { da = std::max(da, fabs(a_g[e] - acc[e])); na = std::max(na, fabs(acc[e])); }
            // f16 rows carry ~1e-3 relative rounding of the accumulator itself
            const double acc_tol = opt_fa_fold_acc16 ? 4e-3 : 1e-2;
            const bool bad = fabs(m_g - m_ref) > 1e-3 || fabs(l_g - l_ref) > 1e-3 * std::max(1.0, l_ref) || da > acc_tol * std::max(1.0, na);
            h->n_check_rows++; h->n_check_bad += bad;
            if (bad || opt_verbose > 1) {
                double qn = 0; for (uint32_t e = 0; e < h->D; ++e) qn += (double) p.q_row[e] * p.q_row[e];
                size_t imax = 0; for (size_t i = 1; i < sc.size(); ++i) if (sc[i] > sc[imax]) imax = i;
                // the same Q row as the host sees it NOW (vs the snapshot at ready)
                const float * qr_now = (const float *) (j.q_base + j.q_off + (size_t) p.tok * j.q_nb1 + (size_t) p.bh * j.q_nb2);
                hetero_dc_civac(qr_now, h->D * sizeof(float));
                double dq = 0; for (uint32_t e = 0; e < h->D; ++e) dq = std::max(dq, (double) fabs(qr_now[e] - p.q_row[e]));
                GGML_LOG_INFO("ggml-hex: fa-fold check2: want %u kvh %u sb %u tok %u (%zu keys): m gpu %.5f ref %.5f | l gpu %.4f ref %.4f | acc max|diff| %.3e of %.3e | |q|^2 %.1f argmax key %u (sq %.2f) | Q now vs snapshot max|diff| %.3e%s\n",
                              j.want, p.kvh, p.sb, p.tok, keys.size(), m_g, m_ref, l_g, l_ref, da, na, qn, keys.empty() ? 0u : keys[imax], sc.empty() ? 0.0 : sc[imax], dq, bad ? "  MISMATCH" : "");
            }
        }
        picks.clear();
    };

    for (;;) {
        std::vector<ggml_hexagon_hfold_job> batch;
        {
            std::unique_lock<std::mutex> lk(h->mu);
            h->cv.wait(lk, [&] { return h->stop || !h->batches.empty(); });
            if (h->stop && h->batches.empty()) return;
            batch = std::move(h->batches.front());
            h->batches.pop_front();
            h->busy = true;
        }
        double b_ready = 0, b_gpu = 0, b_tab = 0; uint32_t b_blk = 0;
        ready = (volatile uint32_t *) (h->base + h->off_ctl);
        done  = (volatile uint32_t *) (h->base + h->off_done);
        stamp = (volatile uint32_t *) (h->base + h->off_stamp);

        if (!opt_fa_fold_gpudone) {
            // ---- legacy: everything after ready is driven from here ----
            for (const auto & j : batch) {
                const auto t0 = clock_t::now();
                const bool seen = wait_ready(j.want, nullptr, clock_t::time_point::max());
                const auto t_ready = clock_t::now();
                h->us_ready += us_between(t0, t_ready);
                b_ready += us_between(t0, t_ready);
                if (!seen) { h->n_ready_timeouts++; continue; }   // the op times out on its own and fails loudly
                if (opt_fa_fold_check == 2 && (h->n_jobs % 2) == 0) check_snapshot(j, 2);
                legacy_gpu(j, b_tab, b_gpu, b_blk, t_ready);
                if (opt_fa_fold_check == 2 && !picks.empty()) check_verify(j);
                h->n_chain++;   // counts checked jobs for the summary line
            }
        } else {
            // ---- chain: gate -> compact -> stage_k -> fa_exc, enqueued ahead of ready ----
            auto release_chain = [&](hfold_chain & c) {
                for (auto & e : c.ev) if (e) { clWaitForEvents(1, &e); clReleaseEvent(e); e = nullptr; }
            };
            // the next chain's enqueue time: -1 = at once, N = N us before the predicted ready
            clock_t::time_point t_ready_prev; bool have_prev = false;
            size_t next_ji = 0; clock_t::time_point t_next = clock_t::now(); bool next_armed = false;
            auto arm_next = [&](size_t ji, clock_t::time_point t_ref) {
                if (ji >= batch.size() || batch[ji].chain.enq || opt_fa_fold_preq == 0 || opt_fa_fold_preq == -3) { next_armed = false; return; }
                next_ji = ji; next_armed = true;
                t_next = t_ref;
                if (opt_fa_fold_preq > 0 && h->period_us > 0) {
                    const double lead = (double) opt_fa_fold_preq;
                    if (h->period_us > lead) t_next = t_ref + std::chrono::microseconds((int64_t) (h->period_us - lead));
                }
            };
            auto poll_hook = [&]() {
                if (next_armed && clock_t::now() >= t_next) { hfold_enqueue_chain(h, batch[next_ji], batch[next_ji].chain); next_armed = false; }
            };
            if (opt_fa_fold_preq == -2) {
                for (size_t ji = 0; ji < batch.size(); ++ji) if (!batch[ji].chain.enq) hfold_enqueue_chain(h, batch[ji], batch[ji].chain);
            } else if (opt_fa_fold_preq != 0) {   // -1, -3, N: the first chain now, the rest as their turn comes
                if (!batch[0].chain.enq) hfold_enqueue_chain(h, batch[0], batch[0].chain);   // the first op is several ops of DSP work away
            }

            for (size_t ji = 0; ji < batch.size(); ++ji) {
                const ggml_hexagon_hfold_job & j = batch[ji];
                hfold_chain & c = batch[ji].chain;
                if (!c.enq) arm_next(ji, clock_t::now());
                const auto t0 = clock_t::now();
                clock_t::time_point t_tight = clock_t::time_point::max();
                if (have_prev && h->period_us > 500) t_tight = t_ready_prev + std::chrono::microseconds((int64_t) (h->period_us - 400));
                const bool seen = wait_ready(j.want, poll_hook, t_tight);
                const auto t_ready = clock_t::now();
                h->us_ready += us_between(t0, t_ready);
                b_ready += us_between(t0, t_ready);
                if (!seen) {
                    h->n_ready_timeouts++;
                    if (c.enq) release_chain(c);   // the gate times out on its own
                    continue;
                }
                if (!c.enq) {
                    hfold_enqueue_chain(h, j, c);
                    if (opt_fa_fold_preq != 0) h->n_late++;
                } else {
                    h->n_preq++; h->us_lead += us_between(c.t_enq, t_ready);
                }
                next_armed = false;
                if (have_prev) {
                    const double p = us_between(t_ready_prev, t_ready);
                    h->period_us = h->period_us > 0 ? 0.7 * h->period_us + 0.3 * p : p;
                }
                t_ready_prev = t_ready; have_prev = true;
                if (opt_fa_fold_check == 2 && (h->n_chain % 2) == 0) check_snapshot(j, 2);
                if (opt_fa_fold_trig) {
                    // a submission that references the DSP-written inputs, after they were written
                    int a = 0;
                    clSetKernelArg(h->k_touch, a++, sizeof(cl_mem), &j.q); clSetKernelArg(h->k_touch, a++, sizeof(cl_mem), &j.em);
                    clSetKernelArg(h->k_touch, a++, sizeof(cl_mem), &j.k); clSetKernelArg(h->k_touch, a++, sizeof(cl_mem), &j.m);
                    auto itv = h->aliases.find(static_cast<ggml_hexagon_shared_buffer *>(const_cast<ggml_tensor *>(j.node)->src[2]->buffer->context)->fd);
                    cl_mem valias = itv != h->aliases.end() ? itv->second : j.k;
                    clSetKernelArg(h->k_touch, a++, sizeof(cl_mem), &valias);
                    hfold_set_svm(h, h->k_touch, a++);
                    const size_t one = 1;
                    cl_event ev_trig = nullptr;
                    cl_int e2 = clEnqueueNDRangeKernel(h->q2, h->k_touch, 1, nullptr, &one, &one, 0, nullptr, opt_fa_fold_trig >= 3 ? &ev_trig : nullptr);
                    if (e2 != CL_SUCCESS) { h->n_gpu_err++; GGML_LOG_ERROR("ggml-hex: fa-fold: trigger enqueue failed (%d)\n", e2); }
                    clFlush(h->q2);
                    if (ev_trig) {
                        // TRIG=3: the maintenance the submission carries runs on the GPU's own
                        // timeline; a kernel that starts reading 60 us after the gate can still
                        // precede it (bimodal perplexity with the natural-layout kernel). Wait for
                        // the trigger to complete before the word that opens the gate.
                        clWaitForEvents(1, &ev_trig); clReleaseEvent(ev_trig);
                    }
                }
                // open the gate
                __atomic_store_n(h->svm_ctl + 0, j.want, __ATOMIC_SEQ_CST);
                arm_next(ji + 1, t_ready);
                bool gate_to = false, failed = false;
                for (;;) {
                    const uint32_t v = __atomic_load_n(h->svm_ctl + 1, __ATOMIC_ACQUIRE);
                    if (v == j.want) break;
                    if (v == (j.want | 0x80000000u)) { gate_to = true; break; }
                    poll_hook();
                    if (us_between(t_ready, clock_t::now()) > 3e6) { failed = true; break; }
                }
                const auto t_gate = clock_t::now();
                h->us_gate += us_between(t_ready, t_gate);
                if (!gate_to && !failed && __atomic_load_n(h->svm_ctl + 6, __ATOMIC_ACQUIRE) == 2u) h->n_gate_ion++;
                const bool trace_job = opt_fa_fold_trace > 0 && (h->n_batches % 16) == 0 && ji < (size_t) opt_fa_fold_trace;
                double trace_t[HTP_FA_FOLD_MAX_STAGES] = {};
                if (gate_to || failed) {
                    // the chain did nothing (every kernel behind the gate checks it): host path
                    h->n_gate_timeouts += gate_to; h->n_fallback++;
                    release_chain(c);
                    legacy_gpu(j, b_tab, b_gpu, b_blk, t_ready);
                    h->n_chain++;
                    continue;
                }
                // stages arrive in order; relay each SVM done word to the DSP's done line unless the
                // kernel's own ION write is already visible there (the direct-visibility probe)
                double t_k[3] = { -1, -1, -1 };   // compact / stage_k / fa_exc start, us after ready
                for (uint32_t st = 0; st < j.n_stages; ++st) {
                    for (;;) {
                        if (__atomic_load_n(h->svm_ctl + 64 + st, __ATOMIC_ACQUIRE) == j.want) break;
                        if (trace_job) {
                            for (int k = 0; k < 3; ++k) if (t_k[k] < 0 && __atomic_load_n(h->svm_ctl + 3 + k, __ATOMIC_ACQUIRE) == j.want) t_k[k] = us_between(t_ready, clock_t::now());
                        }
                        poll_hook();
                        if (us_between(t_ready, clock_t::now()) > 3e6) { failed = true; break; }
                    }
                    if (failed) break;
                    const auto t_st = clock_t::now();
                    if (st == 0) h->us_first_done += us_between(t_ready, t_st);
                    if (st + 1 == j.n_stages) h->us_last_done += us_between(t_ready, t_st);
                    hetero_dc_civac((const void *) (done + st), sizeof(uint32_t));
                    if (__atomic_load_n(done + st, __ATOMIC_ACQUIRE) == j.want) {
                        h->n_direct++;
                    } else {
                        __atomic_store_n(done + st, j.want, __ATOMIC_RELEASE);
                        hetero_dc_civac((const void *) (done + st), sizeof(uint32_t));
                    }
                    hetero_dc_civac((const void *) (stamp + st), sizeof(uint32_t));
                    h->n_stamp_total++;
                    if (__atomic_load_n(stamp + st, __ATOMIC_ACQUIRE) == j.want) h->n_stamp_seen++;
                    if (trace_job) trace_t[st] = us_between(t_ready, t_st);
                }
                if (trace_job) {
                    std::string line;
                    char tmp[32];
                    for (uint32_t st = 0; st < j.n_stages; ++st) { snprintf(tmp, sizeof(tmp), " %.0f", trace_t[st]); line += tmp; }
                    GGML_LOG_INFO("ggml-hex: fa-fold trace: want %u: enqueued %.0f us before ready, gate open +%.0f us (%u gate polls), stage_k starts +%.0f (compact %.0f), fa_exc +%.0f; stage done (us after ready):%s\n",
                                  j.want, us_between(c.t_enq, t_ready), us_between(t_ready, t_gate), __atomic_load_n(h->svm_ctl + 2, __ATOMIC_ACQUIRE), t_k[1], t_k[0], t_k[2], line.c_str());
                }
                if (failed) {
                    GGML_LOG_ERROR("ggml-hex: fa-fold chain: want %u: stages not delivered within 3 s (gate %u)\n", j.want, __atomic_load_n(h->svm_ctl + 1, __ATOMIC_ACQUIRE));
                    h->n_gpu_err++;
                }
                release_chain(c);
                // -3: the next chain goes in only now, with nothing of ours left on the GPU: a
                // submission after idle is what makes the driver drop the previous op's cached
                // lines of Q (same address, new content) -- pre-enqueued chains read them stale
                if (opt_fa_fold_preq == -3 && ji + 1 < batch.size() && !batch[ji + 1].chain.enq) {
                    if (opt_fa_fold_preq_idle > 0) std::this_thread::sleep_for(std::chrono::microseconds(opt_fa_fold_preq_idle));
                    hfold_enqueue_chain(h, batch[ji + 1], batch[ji + 1].chain);
                }
                h->us_gpu += us_between(t_ready, clock_t::now());
                b_gpu += us_between(t_gate, clock_t::now());
                h->n_chain++;
                hetero_dc_civac(h->base + h->off_err, 4);
                h->n_kernel_err += (uint64_t) *(volatile int32_t *) (h->base + h->off_err);
                if (opt_fa_fold_check == 2 && !picks.empty()) check_verify(j);
                if (opt_verbose > 1) {
                    hetero_dc_civac(h->base + h->off_err, 4);
                    GGML_LOG_DEBUG("ggml-hex: fa-fold chain: want %u: enqueued %.0f us before ready, gate open +%.0f us, last done +%.0f us, kernel errors %d\n",
                                   j.want, us_between(c.t_enq, t_ready), us_between(t_ready, t_gate), us_between(t_ready, clock_t::now()),
                                   *(volatile int32_t *) (h->base + h->off_err));
                }
            }
            for (auto & jb : batch) release_chain(jb.chain);
        }
        if (opt_verbose) {
            volatile uint32_t * st = (volatile uint32_t *) (h->base + h->off_ctl + HTP_FA_HETERO_STATUS_OFF);
            hetero_dc_civac((const void *) st, 64);
            GGML_LOG_DEBUG("ggml-hex: fa-fold relay: batch %llu: %zu jobs; per job: ready wait %.0f us, tables %.0f us, GPU %.0f us, staged blocks/head %.1f; DSP: ops that blocked %u, stage waits that spun %u, last op blocked %u us (max %u), timeouts %u; relay ready timeouts %llu\n",
                           (unsigned long long) h->n_batches, batch.size(), b_ready / batch.size(), b_tab / batch.size(), b_gpu / batch.size(), (double) b_blk / batch.size(),
                           st[HTP_FA_FOLD_ST_WAITS], st[HTP_FA_FOLD_ST_TILES_BLOCKED],
                           st[HTP_FA_FOLD_ST_WAIT_US], st[HTP_FA_FOLD_ST_WAIT_US_MAX], st[HTP_FA_FOLD_ST_TIMEOUTS], (unsigned long long) h->n_ready_timeouts);
        }
        {
            std::lock_guard<std::mutex> lk(h->mu);
            h->n_batches++; h->n_jobs += batch.size(); h->busy = false;
        }
        h->cv.notify_all();
        if ((h->n_batches % 8) == 0) ggml_hexagon_hfold_log_summary(h);
    }
}

// A hexagon buffer is going away. Its ION alias and texture view must go with it: the dmabuf fd
// (the alias cache key) is recycled by the next allocation, and an alias keeps the OLD pages alive
// -- so a graph whose compute buffer grew (a longer ubatch of the same prompt) would otherwise
// read Q and the mask from the freed allocation. Seen as NaNs from the first layer of the third
// ubatch on.
static void ggml_hexagon_hfold_buffer_freed(ggml_hexagon_session * sess, ggml_backend_buffer_t buffer) {
    ggml_hexagon_hfold * h = sess->hfold;
    if (!h || !buffer || buffer == h->buf) return;
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    const bool had = h->aliases.count(sbuf->fd) || h->images.count(sbuf->fd);
    if (!had) return;
    ggml_hexagon_hfold_wait_idle(h);
    clFinish(h->q);
    auto ii = h->images.find(sbuf->fd);  if (ii != h->images.end())  { clReleaseMemObject(ii->second); h->images.erase(ii); }
    auto ia = h->aliases.find(sbuf->fd); if (ia != h->aliases.end()) { clReleaseMemObject(ia->second); h->aliases.erase(ia); }
    // jobs that referenced this buffer are stale; the next graph build re-prepares them
    for (auto it = h->jobs.begin(); it != h->jobs.end();) {
        const ggml_tensor * n = it->first;
        bool uses = false;
        for (int i = 0; i < GGML_MAX_SRC && !uses; ++i) uses = n->src[i] && n->src[i]->buffer == buffer;
        it = uses ? h->jobs.erase(it) : std::next(it);
    }
    HEX_VERBOSE("ggml-hex: fa-fold: dropped the alias of freed buffer fd %d base %p\n", sbuf->fd, (void *) sbuf->base);
}

static void ggml_hexagon_hfold_log_summary(ggml_hexagon_hfold * h) {
    uint32_t st[HTP_FA_FOLD_ST_N] = {};
    if (h->base) {
        hetero_dc_civac(h->base + h->off_ctl + HTP_FA_HETERO_STATUS_OFF, sizeof(st));
        memcpy(st, h->base + h->off_ctl + HTP_FA_HETERO_STATUS_OFF, sizeof(st));
    }
    GGML_LOG_INFO("ggml-hex: fa-fold: %llu graphs, %llu GPU jobs; relay ready timeouts %llu, GPU errors %llu; DSP ops that blocked %u, stage waits that spun %u, max blocked %u us, mean blocked %.0f us/op, DSP timeouts %u; per job: ready wait %.0f us, GPU %.0f us\n",
                  (unsigned long long) h->n_batches, (unsigned long long) h->n_jobs,
                  (unsigned long long) h->n_ready_timeouts, (unsigned long long) h->n_gpu_err,
                  st[HTP_FA_FOLD_ST_WAITS], st[HTP_FA_FOLD_ST_TILES_BLOCKED], st[HTP_FA_FOLD_ST_WAIT_US_MAX],
                  h->n_jobs ? (double) st[HTP_FA_FOLD_ST_WAIT_US_SUM] / (double) h->n_jobs : 0.0, st[HTP_FA_FOLD_ST_TIMEOUTS],
                  h->n_jobs ? h->us_ready / (double) h->n_jobs : 0.0, h->n_jobs ? h->us_gpu / (double) h->n_jobs : 0.0);
    if (h->n_chain) {
        const double n = (double) h->n_chain;
        GGML_LOG_INFO("ggml-hex: fa-fold chain: %llu jobs (preq %d, chain %d): pre-enqueued %llu, late %llu, gate timeouts %llu, fallbacks %llu; per job: enqueue lead %.0f us before ready, gate open %.0f us after ready, first done %.0f us, last done %.0f us; ION done already visible at the SVM word %llu of %llu stages, stamp visible %llu; gates opened by the DSP's own word %llu; kernel list errors %llu; check rows %llu bad %llu; period %.0f us\n",
                      (unsigned long long) h->n_chain, opt_fa_fold_preq, opt_fa_fold_chain, (unsigned long long) h->n_preq, (unsigned long long) h->n_late,
                      (unsigned long long) h->n_gate_timeouts, (unsigned long long) h->n_fallback,
                      h->us_lead / n, h->us_gate / n, h->us_first_done / n, h->us_last_done / n,
                      (unsigned long long) h->n_direct, (unsigned long long) h->n_stamp_total, (unsigned long long) h->n_stamp_seen, (unsigned long long) h->n_gate_ion,
                      (unsigned long long) h->n_kernel_err, (unsigned long long) h->n_check_rows, (unsigned long long) h->n_check_bad, h->period_us);
    }
}

static void ggml_hexagon_hfold_free(ggml_hexagon_session * sess) {
    ggml_hexagon_hfold * h = sess->hfold;
    if (!h) return;
    {
        std::lock_guard<std::mutex> lk(h->mu);
        h->stop = true;
    }
    h->cv.notify_all();
    if (h->relay.joinable()) h->relay.join();
    h->keep_stop = true;
    if (h->keeper.joinable()) h->keeper.join();
    if (h->k_keep) { clReleaseKernel(h->k_keep); clReleaseMemObject(h->keep_buf); clReleaseMemObject(h->keep_out); GGML_LOG_INFO("ggml-hex: fa-fold keep-alive: %llu launches\n", (unsigned long long) h->n_keep); }
    if (h->q) clFinish(h->q);
    ggml_hexagon_hfold_log_summary(h);
    if (h->ready_sub) clReleaseMemObject(h->ready_sub);
    for (auto & kv : h->images)  clReleaseMemObject(kv.second);
    for (auto & kv : h->aliases) clReleaseMemObject(kv.second);
    if (h->svm_tabs) clSVMFree(h->ctx, h->svm_tabs);
    if (h->svm_q) clSVMFree(h->ctx, h->svm_q);
    if (h->svm_ctl) clSVMFree(h->ctx, h->svm_ctl);
    if (h->k_fa) clReleaseKernel(h->k_fa);
    if (h->k_stage) clReleaseKernel(h->k_stage);
    if (h->k_gate) clReleaseKernel(h->k_gate);
    if (h->k_compact) clReleaseKernel(h->k_compact);
    if (h->k_stage_p) clReleaseKernel(h->k_stage_p);
    if (h->k_touch) clReleaseKernel(h->k_touch);
    if (h->q2) { clFinish(h->q2); clReleaseCommandQueue(h->q2); }
    if (h->prog) clReleaseProgram(h->prog);
    if (h->q) clReleaseCommandQueue(h->q);
    if (h->ctx) clReleaseContext(h->ctx);
    if (h->tctx) ggml_free(h->tctx);
    if (h->buf) ggml_backend_buffer_free(h->buf);
    delete h;
    sess->hfold = nullptr;
}

#endif // GGML_HEXAGON_HETERO

// ---------------------------------------------------------------------------------------
// Cluster-selected sparse decode attention (research prototype; docs cluster-sparse-decode.md).
// Stage 1: the backend owns the shadow buffer (htp-ops.h, struct htp_fa_cluster_header) and tags
// decode FLASH_ATTN_EXT nodes; the per-head page lists are host-written (density 0) or, later,
// selected on the DSP. The clustering sidecar that fills the shadow comes with a later stage.
// ---------------------------------------------------------------------------------------

struct ggml_hexagon_cluster_layer {
    const ggml_tensor * k_root = nullptr, * v_root = nullptr;   // llama's cache tensors (cache_k_l%d / cache_v_l%d)
    uint32_t written_end = 0;    // positions [0, written_end) hold rows written so far (contiguous prefix)
    uint32_t covered_end = 0;    // positions [0, covered_end) are clustered and published
    uint32_t n_chunks = 0, n_pages = 0;
    bool     stale = false;      // non-contiguous writes: this layer stays dense until a reset
};

struct ggml_hexagon_cluster_job {
    uint64_t seq = 0;
    int32_t  n_tokens = 0;
    int64_t  pos_min = 0, pos_max = -1;
    bool     contiguous = true;
    int64_t  t_post_ns = 0;
};

struct ggml_hexagon_cluster_sidecar;   // GPU k-means + packing thread (needs OpenCL)

struct ggml_hexagon_cluster {
    ggml_hexagon_session *       sess = nullptr;
    ggml_backend_buffer_t        buf  = nullptr;
    uint8_t *                    base = nullptr;
    size_t                       size = 0;
    ggml_context *               tctx = nullptr;
    ggml_tensor *                ctrl = nullptr;   // hand-wired tensor over the buffer, attached as src[7]
    struct htp_fa_cluster_header hdr  = {};
    uint32_t                     kv_cache_size = 0;     // positions in llama's cache (the shadow may cover fewer)
    uint32_t                     n_tagged = 0;

    // SET_ROWS tracking (which rows the last computed graph wrote)
    std::vector<ggml_hexagon_cluster_layer> layers;
    const ggml_tensor *          idx_tensor = nullptr;   // the graph's KV row-index input (shared by all layers)
    bool                         pending = false;       // a graph with SET_ROWS was submitted and not yet harvested
    ggml_hexagon_cluster_job     pending_job;           // positions captured before compute recycled the index buffer
    int32_t                      last_n_tokens = 0;
    uint64_t                     seq = 0;
    ggml_hexagon_cluster_sidecar * side = nullptr;
};

static void ggml_hexagon_cluster_after_compute(ggml_hexagon_session * sess);
static void ggml_hexagon_cluster_track_graph(ggml_hexagon_session * sess, const ggml_cgraph * graph);

static inline void cluster_dc_civac(const void * p, size_t n) {
#if defined(__aarch64__)
    uintptr_t a = (uintptr_t) p & ~(uintptr_t) 63;
    const uintptr_t e = (uintptr_t) p + n;
    for (; a < e; a += 64) asm volatile("dc civac, %0" : : "r"(a) : "memory");
    asm volatile("dsb sy" : : : "memory");
#else
    (void) p; (void) n;
#endif
}

// ---- the clustering sidecar -------------------------------------------------------------------
// A host thread with its own OpenCL queue. For every layer whose positional cache has a chunk of
// unclustered rows it: copies the chunk's K rows to the GPU, runs k-means (C = keys/64 centroids,
// k-means++ init on a subsample, Lloyd iterations with an early stop), reads the assignment back,
// sorts the keys by (cluster, distance) on the CPU, gathers K and V rows into the shadow pages in
// that order, computes the page descriptors (f16 means), and publishes the chunk (chunk entry and
// page count first, covered_end last, every write followed by a cache clean). Pages already
// published are never rewritten except after a reset (positions written from 0 again).

#ifdef GGML_HEXAGON_HETERO
static const char * ggml_hexagon_cluster_cl_src = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef KM_D
#define KM_D 128
#endif
#ifndef KM_TILE
#define KM_TILE 16
#endif
#ifndef KM_NMAX
#define KM_NMAX 1024
#endif
inline __global const half * km_row(__global const uchar * K, uint n, uint h, uint nbk_row, uint nbk_head) {
    return (__global const half *) (K + (ulong) n * nbk_row + (ulong) h * nbk_head);
}
__kernel void km_assign(__global const uchar * K, uint nbk_row, uint nbk_head,
                        __global const float * mu, __global const float * mu2,
                        __global uchar * assign, __global float * dist, __global uint * n_changed, uint N, uint C) {
    const uint n = get_global_id(0), h = get_global_id(1), lid = get_local_id(0), lsz = get_local_size(0);
    __local float lmu[KM_TILE * KM_D];
    __local float lmu2[KM_TILE];
    float8 k[KM_D / 8];
    if (n < N) { __global const half * kr = km_row(K, n, h, nbk_row, nbk_head); for (uint i = 0; i < KM_D / 8; ++i) k[i] = convert_float8(vload8(i, kr)); }
    float best = INFINITY; uint besti = 0;
    for (uint c0 = 0; c0 < C; c0 += KM_TILE) {
        const uint tc = min((uint) KM_TILE, C - c0);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = lid; i < tc * KM_D; i += lsz) lmu[i] = mu[((ulong) h * C + c0) * KM_D + i];
        for (uint i = lid; i < tc; i += lsz) lmu2[i] = mu2[h * C + c0 + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (n < N) {
            for (uint c = 0; c < tc; ++c) {
                float8 acc = (float8)(0.0f);
                for (uint i = 0; i < KM_D / 8; ++i) acc += k[i] * vload8(i, lmu + c * KM_D);
                const float dt = acc.s0 + acc.s1 + acc.s2 + acc.s3 + acc.s4 + acc.s5 + acc.s6 + acc.s7;
                const float dd = lmu2[c] - 2.0f * dt;
                if (dd < best) { best = dd; besti = c0 + c; }
            }
        }
    }
    if (n < N) { const uint idx = h * N + n; if (assign[idx] != (uchar) besti) atomic_inc(n_changed); assign[idx] = (uchar) besti; dist[idx] = best; }
}
__kernel void km_update(__global const uchar * K, uint nbk_row, uint nbk_head,
                        __global const uchar * assign, __global const float * dist,
                        __global float * mu, __global float * mu2, __global uint * counts, uint N, uint C) {
    const uint d = get_local_id(0), c = get_group_id(1), h = get_group_id(2);
    __local uchar la[KM_NMAX];
    __local float red[KM_D];
    __local uint  lfar;
    for (uint i = d; i < N; i += KM_D) la[i] = assign[h * N + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0f; uint cnt = 0;
    for (uint n = 0; n < N; ++n) { if (la[n] == (uchar) c) { sum += vload_half(d, km_row(K, n, h, nbk_row, nbk_head)); cnt++; } }
    float m;
    if (cnt > 0) { m = sum / (float) cnt; }
    else {
        if (d == 0) { float bd = -INFINITY; uint bi = 0; for (uint n = 0; n < N; ++n) { const float v = dist[h * N + n]; if (v > bd) { bd = v; bi = n; } } lfar = bi; }
        barrier(CLK_LOCAL_MEM_FENCE);
        m = vload_half(d, km_row(K, lfar, h, nbk_row, nbk_head));
    }
    mu[((ulong) h * C + c) * KM_D + d] = m;
    red[d] = m * m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = KM_D / 2; s > 0; s >>= 1) { if (d < s) red[d] += red[d + s]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (d == 0) { mu2[h * C + c] = red[0]; counts[h * C + c] = cnt; }
}
)CL";

struct ggml_hexagon_cluster_sidecar {
    ggml_hexagon_cluster * c = nullptr;
    cl_platform_id plat = nullptr; cl_device_id dev = nullptr; cl_context ctx = nullptr; cl_command_queue q = nullptr;
    cl_program prog = nullptr; cl_kernel k_assign = nullptr, k_update = nullptr;
    cl_mem bK = nullptr, bMu = nullptr, bMu2 = nullptr, bCnt = nullptr, bAs = nullptr, bDist = nullptr, bChg = nullptr;
    uint32_t N = 0, H = 0, D = 0, C = 0;    // chunk rows, heads, dims, centroids (N/64)
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<ggml_hexagon_cluster_job> jobs;
    bool stop = false;
    bool busy = false;                 // a job is being processed (reads llama's cache rows)
    std::condition_variable cv_idle;   // signalled when busy drops
    // per-worker scratch (the CPU parts of a chunk run in parallel across layers; the OpenCL k-means is serialized by cl_mu)
    struct scratch {
        std::vector<float> mu0, mu2;
        std::vector<uint8_t> assign;
        std::vector<float> dist;
        std::vector<int> perm;
        std::vector<float> acc;
    };
    std::vector<scratch> scr;
    std::mutex cl_mu;
    std::mutex stat_mu;
    int n_workers = 1;
    // delayed re-verification (GGML_HEXAGON_CLUSTER_VERIFY): the previous chunk's pages vs the cache rows, re-read later
    std::vector<uint32_t> last_pos0, last_pos1, last_page_first;   // per layer
    uint64_t n_stale_rows = 0, n_checked_rows = 0;
    // stats
    uint64_t n_jobs = 0, n_chunks = 0, n_resets = 0, n_verify_fail = 0;
    double   t_gpu_ms = 0, t_cpu_ms = 0, lag_max_ms = 0;
};

static bool cluster_cl_ok(cl_int e, const char * w) { if (e != CL_SUCCESS) { GGML_LOG_ERROR("ggml-hex: cluster: %s failed (%d)\n", w, e); return false; } return true; }

static inline float cluster_f16_to_f32(uint16_t h) { return ggml_fp16_to_fp32(h); }

// k-means++ init on a 256-row subsample of the chunk, per head (CPU, reads the cache rows directly)
static void cluster_init_centroids(ggml_hexagon_cluster_sidecar * s, ggml_hexagon_cluster_sidecar::scratch * sc, const uint8_t * krows, uint32_t nbk_row, uint32_t N, uint32_t C, std::mt19937 & rng) {
    const uint32_t H = s->H, D = s->D;
    auto key = [&](uint32_t h, uint32_t n, uint32_t d) { return cluster_f16_to_f32(((const uint16_t *) (krows + (size_t) n * nbk_row + (size_t) h * D * 2))[d]); };
    const uint32_t ns = std::min<uint32_t>(N, 256);
    std::vector<uint32_t> sub(ns);
    for (uint32_t i = 0; i < ns; ++i) sub[i] = (uint32_t) (((uint64_t) i * N) / ns);
    std::vector<double> dmin(ns);
    for (uint32_t h = 0; h < H; ++h) {
        std::vector<uint32_t> chosen;
        std::uniform_int_distribution<uint32_t> first(0, ns - 1);
        chosen.push_back(sub[first(rng)]);
        std::fill(dmin.begin(), dmin.end(), 1e30);
        while (chosen.size() < C) {
            const uint32_t last = chosen.back(); double tot = 0;
            for (uint32_t i = 0; i < ns; ++i) { double sd = 0; for (uint32_t d = 0; d < D; ++d) { const double t = key(h, sub[i], d) - key(h, last, d); sd += t * t; } dmin[i] = std::min(dmin[i], sd); tot += dmin[i]; }
            std::uniform_real_distribution<double> u(0.0, tot > 0 ? tot : 1.0); double r = u(rng); uint32_t pick = ns - 1;
            for (uint32_t i = 0; i < ns; ++i) { r -= dmin[i]; if (r <= 0) { pick = i; break; } }
            chosen.push_back(sub[pick]);
        }
        for (uint32_t c = 0; c < C; ++c) {
            double sq = 0;
            for (uint32_t d = 0; d < D; ++d) { const float v = key(h, chosen[c], d); sc->mu0[((size_t) h * C + c) * D + d] = v; sq += (double) v * v; }
            sc->mu2[(size_t) h * C + c] = (float) sq;
        }
    }
}

// GPU k-means over N rows (krows, row stride nbk_row) with C centroids per head, initial centroids in
// sc->mu0/mu2; fills sc->assign / sc->dist (row-major [h][n]).
static bool cluster_kmeans_gpu(ggml_hexagon_cluster_sidecar * s, ggml_hexagon_cluster_sidecar::scratch * sc, const uint8_t * krows, uint32_t nbk_row, uint32_t N, uint32_t C) {
    std::lock_guard<std::mutex> lk(s->cl_mu);   // one OpenCL queue: the GPU phase is serialized, the CPU phases are not
    const uint32_t H = s->H, D = s->D;
    cl_int err = clEnqueueWriteBuffer(s->q, s->bK, CL_FALSE, 0, (size_t) N * nbk_row, krows, 0, nullptr, nullptr);
    if (!cluster_cl_ok(err, "write K chunk")) return false;
    clEnqueueWriteBuffer(s->q, s->bMu, CL_FALSE, 0, (size_t) H * C * D * 4, sc->mu0.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(s->q, s->bMu2, CL_FALSE, 0, (size_t) H * C * 4, sc->mu2.data(), 0, nullptr, nullptr);
    std::fill(sc->assign.begin(), sc->assign.end(), 0xff);
    clEnqueueWriteBuffer(s->q, s->bAs, CL_FALSE, 0, (size_t) H * N, sc->assign.data(), 0, nullptr, nullptr);
    const cl_uint uN = N, uC = C, uNb = nbk_row, uNh = D * 2;
    for (int it = 0; it < 8; ++it) {
        const cl_uint zero = 0;
        clEnqueueWriteBuffer(s->q, s->bChg, CL_FALSE, 0, 4, &zero, 0, nullptr, nullptr);
        int a = 0;
        clSetKernelArg(s->k_assign, a++, sizeof(cl_mem), &s->bK); clSetKernelArg(s->k_assign, a++, 4, &uNb); clSetKernelArg(s->k_assign, a++, 4, &uNh);
        clSetKernelArg(s->k_assign, a++, sizeof(cl_mem), &s->bMu); clSetKernelArg(s->k_assign, a++, sizeof(cl_mem), &s->bMu2);
        clSetKernelArg(s->k_assign, a++, sizeof(cl_mem), &s->bAs); clSetKernelArg(s->k_assign, a++, sizeof(cl_mem), &s->bDist); clSetKernelArg(s->k_assign, a++, sizeof(cl_mem), &s->bChg);
        clSetKernelArg(s->k_assign, a++, 4, &uN); clSetKernelArg(s->k_assign, a++, 4, &uC);
        size_t g1[2] = { (size_t) ((N + 63) / 64 * 64), (size_t) H }, l1[2] = { 64, 1 };
        if (!cluster_cl_ok(clEnqueueNDRangeKernel(s->q, s->k_assign, 2, nullptr, g1, l1, 0, nullptr, nullptr), "km_assign")) return false;
        a = 0;
        clSetKernelArg(s->k_update, a++, sizeof(cl_mem), &s->bK); clSetKernelArg(s->k_update, a++, 4, &uNb); clSetKernelArg(s->k_update, a++, 4, &uNh);
        clSetKernelArg(s->k_update, a++, sizeof(cl_mem), &s->bAs); clSetKernelArg(s->k_update, a++, sizeof(cl_mem), &s->bDist);
        clSetKernelArg(s->k_update, a++, sizeof(cl_mem), &s->bMu); clSetKernelArg(s->k_update, a++, sizeof(cl_mem), &s->bMu2); clSetKernelArg(s->k_update, a++, sizeof(cl_mem), &s->bCnt);
        clSetKernelArg(s->k_update, a++, 4, &uN); clSetKernelArg(s->k_update, a++, 4, &uC);
        size_t g2[3] = { (size_t) D, (size_t) C, (size_t) H }, l2[3] = { (size_t) D, 1, 1 };
        if (!cluster_cl_ok(clEnqueueNDRangeKernel(s->q, s->k_update, 3, nullptr, g2, l2, 0, nullptr, nullptr), "km_update")) return false;
        cl_uint changed = 0;
        clEnqueueReadBuffer(s->q, s->bChg, CL_TRUE, 0, 4, &changed, 0, nullptr, nullptr);
        if (changed == 0) break;
    }
    clEnqueueReadBuffer(s->q, s->bAs, CL_FALSE, 0, (size_t) H * N, sc->assign.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(s->q, s->bDist, CL_TRUE, 0, (size_t) H * N * 4, sc->dist.data(), 0, nullptr, nullptr);
    return true;
}

// Runs mode: cluster positions [pos0, pos1) of layer il into whole variable-size clusters. The sink keys
// (chunk 0 only) form a forced run and are never clustered; the rest is k-means with N_mid / avg clusters
// per KV head. Rows are stored at [pos0, pos1) of each head's shadow rows in (sinks, cluster, distance)
// order, one run-table entry and one centroid (mean of its keys) per run. Publish order: rows, pos_map,
// runs, centroids, chunk entry, unit count, covered_end.
static bool cluster_run_chunk_runs(ggml_hexagon_cluster_sidecar * s, ggml_hexagon_cluster_sidecar::scratch * sc, uint32_t il, uint32_t pos0, uint32_t pos1) {
    ggml_hexagon_cluster * c = s->c;
    ggml_hexagon_cluster_layer & L = c->layers[il];
    const htp_fa_cluster_header & hdr = c->hdr;
    const uint32_t N = pos1 - pos0, H = s->H, D = s->D, avg = hdr.avg_cluster;
    const uint32_t n_sink = (pos0 == 0) ? std::min<uint32_t>((uint32_t) std::max(opt_cluster_sink, 0), N) : 0;
    const uint32_t N_mid = N - n_sink;
    const uint32_t C = N_mid > 0 ? std::max<uint32_t>(1, N_mid / avg) : 0;
    const uint32_t n_runs = C + (n_sink ? 1 : 0);
    const uint32_t rows_max = hdr.n_pages_max * hdr.page_keys;
    if (N == 0 || N > s->N || (N % 64) || C > 255 || C > s->C || !L.k_root || !L.v_root) return false;
    if (L.n_chunks >= hdr.max_chunks || L.n_pages + n_runs > hdr.n_runs_max || pos1 > rows_max) return false;
    const uint32_t nbk_row = (uint32_t) L.k_root->nb[1], nbv_row = (uint32_t) L.v_root->nb[1], nbk_head = D * 2;
    const uint8_t * krows = (const uint8_t *) L.k_root->data + (size_t) pos0 * nbk_row;
    const uint8_t * vrows = (const uint8_t *) L.v_root->data + (size_t) pos0 * nbv_row;
    const auto t0 = std::chrono::steady_clock::now();
    cluster_dc_civac(krows, (size_t) N * nbk_row);
    cluster_dc_civac(vrows, (size_t) N * nbv_row);
    std::mt19937 rng(1234u + il * 7919u + pos0);
    const auto t1 = std::chrono::steady_clock::now();
    if (C > 0) {
        const uint8_t * kmid = krows + (size_t) n_sink * nbk_row;
        if (opt_cluster_positional) {
            for (uint32_t h = 0; h < H; ++h) for (uint32_t i = 0; i < N_mid; ++i) { sc->assign[(size_t) h * N_mid + i] = (uint8_t) std::min(i / avg, C - 1); sc->dist[(size_t) h * N_mid + i] = (float) (i % avg); }
        } else {
            cluster_init_centroids(s, sc, kmid, nbk_row, N_mid, C, rng);
            if (!cluster_kmeans_gpu(s, sc, kmid, nbk_row, N_mid, C)) return false;
        }
    }
    const auto t2 = std::chrono::steady_clock::now();

    uint8_t * lb = c->base + hdr.layer0_off + (size_t) il * hdr.layer_stride;
    const size_t head_stride = (size_t) hdr.n_pages_max * hdr.page_bytes;
    const uint32_t run_first = L.n_pages;
    std::vector<uint32_t> cnt(C), start(C);
    for (uint32_t h = 0; h < H; ++h) {
        const uint8_t * as = sc->assign.data() + (size_t) h * N_mid;
        const float *   ds = sc->dist.data() + (size_t) h * N_mid;
        sc->perm.resize(N_mid);
        for (uint32_t i = 0; i < N_mid; ++i) sc->perm[i] = (int) i;
        std::stable_sort(sc->perm.begin(), sc->perm.end(), [&](int x, int y) { return as[x] != as[y] ? as[x] < as[y] : ds[x] < ds[y]; });
        std::fill(cnt.begin(), cnt.end(), 0u);
        for (uint32_t i = 0; i < N_mid; ++i) cnt[std::min<uint32_t>(as[i], C - 1)]++;
        for (uint32_t k = 0, acc = n_sink; k < C; ++k) { start[k] = acc; acc += cnt[k]; }
        uint8_t *  kp = lb + hdr.off_k_pages + (size_t) h * head_stride + (size_t) pos0 * nbk_head;
        uint8_t *  vp = lb + hdr.off_v_pages + (size_t) h * head_stride + (size_t) pos0 * nbk_head;
        uint32_t * pm = (uint32_t *) (lb + hdr.off_pos_map) + (size_t) h * rows_max + pos0;
        for (uint32_t i = 0; i < n_sink; ++i) {
            memcpy(kp + (size_t) i * nbk_head, krows + (size_t) i * nbk_row + (size_t) h * nbk_head, nbk_head);
            memcpy(vp + (size_t) i * nbk_head, vrows + (size_t) i * nbv_row + (size_t) h * nbk_head, nbk_head);
            pm[i] = pos0 + i;
        }
        for (uint32_t i = 0; i < N_mid; ++i) {
            const uint32_t n = n_sink + (uint32_t) sc->perm[i], r = n_sink + i;
            memcpy(kp + (size_t) r * nbk_head, krows + (size_t) n * nbk_row + (size_t) h * nbk_head, nbk_head);
            memcpy(vp + (size_t) r * nbk_head, vrows + (size_t) n * nbv_row + (size_t) h * nbk_head, nbk_head);
            pm[r] = pos0 + n;
        }
        auto * rt = (htp_fa_cluster_run *) (lb + hdr.off_runs) + (size_t) h * hdr.n_runs_max + run_first;
        uint8_t * cent = lb + hdr.off_centroids + ((size_t) h * hdr.n_runs_max + run_first) * hdr.centroid_bytes;
        uint32_t r = 0;
        auto emit = [&](uint32_t row_first, uint32_t n_rows, uint16_t flags) {
            rt[r] = { pos0 + row_first, (uint16_t) n_rows, flags };
            std::fill(sc->acc.begin(), sc->acc.end(), 0.0f);
            for (uint32_t i = 0; i < n_rows; ++i) {
                const uint16_t * row = (const uint16_t *) (kp + (size_t) (row_first + i) * nbk_head);
                for (uint32_t d = 0; d < D; ++d) sc->acc[d] += cluster_f16_to_f32(row[d]);
            }
            uint16_t * cp = (uint16_t *) (cent + (size_t) r * hdr.centroid_bytes);
            const float inv = n_rows ? 1.0f / (float) n_rows : 0.0f;
            for (uint32_t d = 0; d < D; ++d) cp[d] = ggml_fp32_to_fp16(sc->acc[d] * inv);
            r++;
        };
        if (n_sink) emit(0, n_sink, HTP_FA_CLUSTER_RUN_FORCED);
        for (uint32_t k = 0; k < C; ++k) emit(start[k], cnt[k], 0);
        cluster_dc_civac(kp, (size_t) N * nbk_head);
        cluster_dc_civac(vp, (size_t) N * nbk_head);
        cluster_dc_civac(pm, (size_t) N * 4);
        cluster_dc_civac(rt, (size_t) n_runs * sizeof(htp_fa_cluster_run));
        cluster_dc_civac(cent, (size_t) n_runs * hdr.centroid_bytes);
    }
    auto * chunk = (htp_fa_cluster_chunk *) (lb + hdr.off_chunks) + L.n_chunks;
    *chunk = { pos0, pos1, run_first, n_runs };
    cluster_dc_civac(chunk, sizeof(*chunk));
    auto * dir = (htp_fa_cluster_dir *) (c->base + HTP_FA_CLUSTER_DIR_OFF + (size_t) il * HTP_FA_CLUSTER_DIR_STRIDE);
    L.n_chunks += 1; L.n_pages += n_runs;
    dir->n_chunks = L.n_chunks; dir->n_pages_pub = L.n_pages;
    cluster_dc_civac(dir, sizeof(*dir));
    dir->covered_end = pos1;
    cluster_dc_civac(dir, sizeof(*dir));
    L.covered_end = pos1;
    const auto t3 = std::chrono::steady_clock::now();
    { std::lock_guard<std::mutex> lk(s->stat_mu);
      s->t_gpu_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
      s->t_cpu_ms += std::chrono::duration<double, std::milli>(t1 - t0).count() + std::chrono::duration<double, std::milli>(t3 - t2).count();
      s->n_chunks++; }
    if (opt_verbose && opt_cluster_verify) {
        uint32_t cmin = N, cmax = 0;
        for (uint32_t k = 0; k < C; ++k) { cmin = std::min(cmin, cnt[k]); cmax = std::max(cmax, cnt[k]); }
        GGML_LOG_INFO("ggml-hex: cluster: layer %u chunk [%u,%u) -> runs %u..%u (%u clusters of %u..%u keys%s) gpu %.2f ms cpu %.2f ms\n", il, pos0, pos1,
                      run_first, run_first + n_runs - 1, C, cmin, cmax, n_sink ? ", + sink run" : "",
                      std::chrono::duration<double, std::milli>(t2 - t1).count(), std::chrono::duration<double, std::milli>(t3 - t2 + t1 - t0).count());
    }
    return true;
}

// Cluster positions [pos0, pos1) of layer il into pages page_first.. and publish. Returns false on error.
static bool cluster_run_chunk(ggml_hexagon_cluster_sidecar * s, ggml_hexagon_cluster_sidecar::scratch * sc, uint32_t il, uint32_t pos0, uint32_t pos1) {
    ggml_hexagon_cluster * c = s->c;
    ggml_hexagon_cluster_layer & L = c->layers[il];
    const htp_fa_cluster_header & hdr = c->hdr;
    const uint32_t N = pos1 - pos0, H = s->H, D = s->D, PK = hdr.page_keys;
    const uint32_t C = N / PK, n_pages = C;
    if (N == 0 || N > s->N || (N % PK) || C > 255 || !L.k_root || !L.v_root) return false;   // assignments are uchar
    if (L.n_chunks >= hdr.max_chunks || L.n_pages + n_pages > hdr.n_pages_max) return false;
    const uint32_t nbk_row = (uint32_t) L.k_root->nb[1];
    const uint32_t nbk_head = D * 2;
    const uint8_t * krows = (const uint8_t *) L.k_root->data + (size_t) pos0 * nbk_row;
    const uint8_t * vrows = (const uint8_t *) L.v_root->data + (size_t) pos0 * L.v_root->nb[1];
    const auto t0 = std::chrono::steady_clock::now();
    if (opt_cluster_verify && s->last_pos1.size() == c->layers.size() && s->last_pos1[il] > s->last_pos0[il] && s->last_pos1[il] <= pos0 && opt_cluster_positional && !opt_cluster_runs) {
        // Delayed re-verification: the previous chunk's pages were copied right after its rows were written.
        // Re-read those cache rows now and count the rows that differ: any difference means this core read
        // stale data the first time (positional pages: page row i of the chunk is position pos0 + i).
        const uint32_t q0 = s->last_pos0[il], q1 = s->last_pos1[il], pf = s->last_page_first[il];
        const uint8_t * kq = (const uint8_t *) L.k_root->data + (size_t) q0 * nbk_row;
        const uint8_t * vq = (const uint8_t *) L.v_root->data + (size_t) q0 * L.v_root->nb[1];
        cluster_dc_civac(kq, (size_t) (q1 - q0) * nbk_row);
        cluster_dc_civac(vq, (size_t) (q1 - q0) * L.v_root->nb[1]);
        const uint8_t * lb0 = c->base + hdr.layer0_off + (size_t) il * hdr.layer_stride;
        uint32_t stale = 0, stale_v = 0, stale_c = 0, first_bad = UINT32_MAX;
        for (uint32_t i = 0; i < q1 - q0; ++i) {
            for (uint32_t h = 0; h < H; ++h) {
                const uint8_t * page_row = lb0 + hdr.off_k_pages + (size_t) h * ((size_t) hdr.n_pages_max * hdr.page_bytes) + ((size_t) pf * PK + i) * D * 2;
                const uint8_t * vpage_row = lb0 + hdr.off_v_pages + (size_t) h * ((size_t) hdr.n_pages_max * hdr.page_bytes) + ((size_t) pf * PK + i) * D * 2;
                if (memcmp(page_row, kq + (size_t) i * nbk_row + (size_t) h * D * 2, (size_t) D * 2)) { stale++; if (first_bad == UINT32_MAX) first_bad = q0 + i; break; }
                if (memcmp(vpage_row, vq + (size_t) i * L.v_root->nb[1] + (size_t) h * D * 2, (size_t) D * 2)) { stale_v++; if (first_bad == UINT32_MAX) first_bad = q0 + i; break; }
            }
        }
        // descriptors: recompute the mean of each page from the cache rows as they are now
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t p = 0; p < (q1 - q0) / PK; ++p) {
                const uint16_t * cp = (const uint16_t *) (lb0 + hdr.off_centroids + ((size_t) h * hdr.n_pages_max + pf + p) * hdr.centroid_bytes);
                bool bad = false;
                for (uint32_t d = 0; d < D && !bad; d += 17) {
                    double m = 0; for (uint32_t i = 0; i < PK; ++i) m += cluster_f16_to_f32(((const uint16_t *) (kq + (size_t) (p * PK + i) * nbk_row + (size_t) h * D * 2))[d]);
                    m /= PK;
                    if (fabs(m - cluster_f16_to_f32(cp[d])) > 2e-2 + 2e-2 * fabs(m)) bad = true;
                }
                stale_c += bad;
            }
        }
        s->n_checked_rows += q1 - q0; s->n_stale_rows += stale + stale_v + stale_c;
        if (stale || stale_v || stale_c) GGML_LOG_WARN("ggml-hex: cluster: layer %u chunk [%u,%u) re-read at [%u,%u): K rows differ %u, V rows differ %u, descriptors off %u (first row %u): STALE\n",
                                                        il, q0, q1, pos0, pos1, stale, stale_v, stale_c, first_bad);
    }
    // the DSP wrote these rows; make sure this core reads them from memory
    cluster_dc_civac(krows, (size_t) N * nbk_row);
    cluster_dc_civac(vrows, (size_t) N * L.v_root->nb[1]);

    std::mt19937 rng(1234u + il * 7919u + pos0);
    auto t1 = std::chrono::steady_clock::now();
    if (opt_cluster_positional) {
        // baseline: positional pages; every key is its own "cluster" in position order
        for (uint32_t h = 0; h < H; ++h) for (uint32_t i = 0; i < N; ++i) { sc->assign[(size_t) h * N + i] = (uint8_t) (i / PK); sc->dist[(size_t) h * N + i] = (float) (i % PK); }
    } else {
    cluster_init_centroids(s, sc, krows, nbk_row, N, C, rng);
    t1 = std::chrono::steady_clock::now();
    if (!cluster_kmeans_gpu(s, sc, krows, nbk_row, N, C)) return false;
    }
    const auto t2 = std::chrono::steady_clock::now();
#if 0
    cl_int err = clEnqueueWriteBuffer(s->q, s->bK, CL_FALSE, 0, (size_t) N * nbk_row, krows, 0, nullptr, nullptr);
    if (!cluster_cl_ok(err, "write K chunk")) return false;
    clEnqueueWriteBuffer(s->q, s->bMu, CL_FALSE, 0, (size_t) H * C * D * 4, sc->mu0.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(s->q, s->bMu2, CL_FALSE, 0, (size_t) H * C * 4, sc->mu2.data(), 0, nullptr, nullptr);
    std::fill(sc->assign.begin(), sc->assign.end(), 0xff);
    clEnqueueWriteBuffer(s->q, s->bAs, CL_FALSE, 0, (size_t) H * N, sc->assign.data(), 0, nullptr, nullptr);
    const cl_uint uN = N, uC = C, uNb = nbk_row, uNh = D * 2;
    for (int it = 0; it < 8; ++it) {
        const cl_uint zero = 0;
        clEnqueueWriteBuffer(s->q, s->bChg, CL_FALSE, 0, 4, &zero, 0, nullptr, nullptr);
        int a = 0;
        clSetKernelArg(s->k_assign, a++, sizeof(cl_mem), &s->bK); clSetKernelArg(s->k_assign, a++, 4, &uNb); clSetKernelArg(s->k_assign, a++, 4, &uNh);
        clSetKernelArg(s->k_assign, a++, sizeof(cl_mem), &s->bMu); clSetKernelArg(s->k_assign, a++, sizeof(cl_mem), &s->bMu2);
        clSetKernelArg(s->k_assign, a++, sizeof(cl_mem), &s->bAs); clSetKernelArg(s->k_assign, a++, sizeof(cl_mem), &s->bDist); clSetKernelArg(s->k_assign, a++, sizeof(cl_mem), &s->bChg);
        clSetKernelArg(s->k_assign, a++, 4, &uN); clSetKernelArg(s->k_assign, a++, 4, &uC);
        size_t g1[2] = { (size_t) N, (size_t) H }, l1[2] = { 64, 1 };
        if (!cluster_cl_ok(clEnqueueNDRangeKernel(s->q, s->k_assign, 2, nullptr, g1, l1, 0, nullptr, nullptr), "km_assign")) return false;
        a = 0;
        clSetKernelArg(s->k_update, a++, sizeof(cl_mem), &s->bK); clSetKernelArg(s->k_update, a++, 4, &uNb); clSetKernelArg(s->k_update, a++, 4, &uNh);
        clSetKernelArg(s->k_update, a++, sizeof(cl_mem), &s->bAs); clSetKernelArg(s->k_update, a++, sizeof(cl_mem), &s->bDist);
        clSetKernelArg(s->k_update, a++, sizeof(cl_mem), &s->bMu); clSetKernelArg(s->k_update, a++, sizeof(cl_mem), &s->bMu2); clSetKernelArg(s->k_update, a++, sizeof(cl_mem), &s->bCnt);
        clSetKernelArg(s->k_update, a++, 4, &uN); clSetKernelArg(s->k_update, a++, 4, &uC);
        size_t g2[3] = { (size_t) D, (size_t) C, (size_t) H }, l2[3] = { (size_t) D, 1, 1 };
        if (!cluster_cl_ok(clEnqueueNDRangeKernel(s->q, s->k_update, 3, nullptr, g2, l2, 0, nullptr, nullptr), "km_update")) return false;
        cl_uint changed = 0;
        clEnqueueReadBuffer(s->q, s->bChg, CL_TRUE, 0, 4, &changed, 0, nullptr, nullptr);
        if (changed == 0) break;
    }
    clEnqueueReadBuffer(s->q, s->bAs, CL_FALSE, 0, (size_t) H * N, sc->assign.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(s->q, s->bDist, CL_TRUE, 0, (size_t) H * N * 4, sc->dist.data(), 0, nullptr, nullptr);
#endif

    // CPU: per head, sort by (cluster, distance), gather rows into the shadow, page means
    uint8_t * lb = c->base + hdr.layer0_off + (size_t) il * hdr.layer_stride;
    const size_t head_stride = (size_t) hdr.n_pages_max * hdr.page_bytes;
    const uint32_t page_first = L.n_pages;
    for (uint32_t h = 0; h < H; ++h) {
        const uint8_t * as = sc->assign.data() + (size_t) h * N;
        const float *   ds = sc->dist.data() + (size_t) h * N;
        sc->perm.resize(N);
        for (uint32_t i = 0; i < N; ++i) sc->perm[i] = (int) i;
        // sink keys (positions < opt_cluster_sink, chunk 0 only) sort before every cluster: they land at the
        // start of page 0 of the layer, which the kernel force-selects (HTP_FA_CLUSTER_FLAG_FORCE)
        const uint32_t n_sink = (pos0 == 0 && opt_cluster_sink > 0) ? std::min((uint32_t) opt_cluster_sink, N) : 0;
        auto ckey = [&](int x) { return (uint32_t) x < n_sink ? -1 : (int) as[x]; };
        std::stable_sort(sc->perm.begin(), sc->perm.end(), [&](int x, int y) { const int kx = ckey(x), ky = ckey(y); return kx != ky ? kx < ky : ds[x] < ds[y]; });
        uint8_t *  kp = lb + hdr.off_k_pages + (size_t) h * head_stride + (size_t) page_first * hdr.page_bytes;
        uint8_t *  vp = lb + hdr.off_v_pages + (size_t) h * head_stride + (size_t) page_first * hdr.page_bytes;
        uint32_t * pm = (uint32_t *) (lb + hdr.off_pos_map) + ((size_t) h * hdr.n_pages_max + page_first) * PK;
        for (uint32_t i = 0; i < N; ++i) {
            const uint32_t n = (uint32_t) sc->perm[i];
            memcpy(kp + (size_t) i * D * 2, krows + (size_t) n * nbk_row + (size_t) h * nbk_head, (size_t) D * 2);
            memcpy(vp + (size_t) i * D * 2, vrows + (size_t) n * L.v_root->nb[1] + (size_t) h * nbk_head, (size_t) D * 2);
            pm[i] = pos0 + n;
        }
        // descriptors: f16 mean of each page's keys
        uint16_t * cent = (uint16_t *) (lb + hdr.off_centroids + ((size_t) h * hdr.n_pages_max + page_first) * hdr.centroid_bytes);
        for (uint32_t p = 0; p < n_pages; ++p) {
            std::fill(sc->acc.begin(), sc->acc.end(), 0.0f);
            const uint16_t * rows = (const uint16_t *) (kp + (size_t) p * hdr.page_bytes);
            for (uint32_t i = 0; i < PK; ++i) for (uint32_t d = 0; d < D; ++d) sc->acc[d] += cluster_f16_to_f32(rows[i * D + d]);
            uint16_t * cp = (uint16_t *) ((uint8_t *) cent + (size_t) p * hdr.centroid_bytes);
            for (uint32_t d = 0; d < D; ++d) cp[d] = ggml_fp32_to_fp16(sc->acc[d] * (1.0f / (float) PK));
        }
        cluster_dc_civac(kp, (size_t) n_pages * hdr.page_bytes);
        cluster_dc_civac(vp, (size_t) n_pages * hdr.page_bytes);
        cluster_dc_civac(pm, (size_t) N * 4);
        cluster_dc_civac(cent, (size_t) n_pages * hdr.centroid_bytes);
    }
    // chunk entry, then the directory (page count, then covered_end last)
    auto * chunk = (htp_fa_cluster_chunk *) (lb + hdr.off_chunks) + L.n_chunks;
    *chunk = { pos0, pos1, page_first, n_pages };
    cluster_dc_civac(chunk, sizeof(*chunk));
    auto * dir = (htp_fa_cluster_dir *) (c->base + HTP_FA_CLUSTER_DIR_OFF + (size_t) il * HTP_FA_CLUSTER_DIR_STRIDE);
    L.n_chunks += 1; L.n_pages += n_pages;
    dir->n_chunks = L.n_chunks; dir->n_pages_pub = L.n_pages;
    cluster_dc_civac(dir, sizeof(*dir));
    dir->covered_end = pos1;
    cluster_dc_civac(dir, sizeof(*dir));
    L.covered_end = pos1;
    if (opt_cluster_verify) {
        if (s->last_pos1.size() != c->layers.size()) { s->last_pos0.assign(c->layers.size(), 0); s->last_pos1.assign(c->layers.size(), 0); s->last_page_first.assign(c->layers.size(), 0); }
        s->last_pos0[il] = pos0; s->last_pos1[il] = pos1; s->last_page_first[il] = page_first;
    }
    const auto t3 = std::chrono::steady_clock::now();
    { std::lock_guard<std::mutex> lk(s->stat_mu);
      s->t_gpu_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
      s->t_cpu_ms += std::chrono::duration<double, std::milli>(t1 - t0).count() + std::chrono::duration<double, std::milli>(t3 - t2).count();
      s->n_chunks++; }

    if (opt_cluster_verify) {
        // pos_map is a permutation of [pos0, pos1); every page row equals the cache row it names;
        // every descriptor is the mean of its page within 1e-2
        int bad = 0;
        for (uint32_t h = 0; h < H && bad == 0; ++h) {
            const uint32_t * pm = (const uint32_t *) (lb + hdr.off_pos_map) + ((size_t) h * hdr.n_pages_max + page_first) * PK;
            std::vector<uint8_t> seen(N, 0);
            const uint8_t * kp = lb + hdr.off_k_pages + (size_t) h * head_stride + (size_t) page_first * hdr.page_bytes;
            for (uint32_t i = 0; i < N; ++i) {
                const uint32_t pos = pm[i];
                if (pos < pos0 || pos >= pos1 || seen[pos - pos0]++) { bad = 1; break; }
                if (memcmp(kp + (size_t) i * D * 2, krows + (size_t) (pos - pos0) * nbk_row + (size_t) h * nbk_head, (size_t) D * 2)) { bad = 2; break; }
            }
            const uint16_t * cent = (const uint16_t *) (lb + hdr.off_centroids + ((size_t) h * hdr.n_pages_max + page_first) * hdr.centroid_bytes);
            for (uint32_t p = 0; p < n_pages && !bad; ++p) {
                for (uint32_t d = 0; d < D; d += 37) {
                    double m = 0; for (uint32_t i = 0; i < PK; ++i) m += cluster_f16_to_f32(((const uint16_t *) (kp + (size_t) p * hdr.page_bytes))[i * D + d]);
                    m /= PK;
                    if (fabs(m - cluster_f16_to_f32(((const uint16_t *) ((const uint8_t *) cent + (size_t) p * hdr.centroid_bytes))[d])) > 1e-2 + 1e-2 * fabs(m)) { bad = 3; break; }
                }
            }
        }
        if (bad) s->n_verify_fail++;
        if (bad || opt_verbose) {
            GGML_LOG_INFO("ggml-hex: cluster: layer %u chunk [%u,%u) -> pages %u..%u: %s (gpu %.2f ms, cpu %.2f ms)\n", il, pos0, pos1, page_first, page_first + n_pages - 1,
                          bad == 0 ? "verified" : bad == 1 ? "POS_MAP NOT A PERMUTATION" : bad == 2 ? "PAGE ROW MISMATCH" : "DESCRIPTOR MISMATCH",
                          std::chrono::duration<double, std::milli>(t2 - t1).count(), std::chrono::duration<double, std::milli>(t3 - t2 + t1 - t0).count());
        }
    }
    return true;
}

// ---- in-place pages: no shadow rows, only descriptors -----------------------------------------
// Page p of a layer is rows [p*PK, p*PK+PK) of llama's cache. The sidecar publishes the f16 mean of
// every complete page as its descriptor (the same centroid region the select pass reads) and the
// count of published pages; the kernel fetches the selected pages from the cache itself with the
// dense block's strided descriptor. A rewind below covered_end shrinks the published prefix.
static void cluster_inplace_shrink(ggml_hexagon_cluster * c, uint32_t il, uint32_t pos) {
    ggml_hexagon_cluster_layer & L = c->layers[il];
    const uint32_t PK = c->hdr.page_keys;
    const uint32_t new_cov = pos / PK * PK;
    if (new_cov >= L.covered_end) return;
    auto * dir = (htp_fa_cluster_dir *) (c->base + HTP_FA_CLUSTER_DIR_OFF + (size_t) il * HTP_FA_CLUSTER_DIR_STRIDE);
    dir->covered_end = new_cov;
    cluster_dc_civac(dir, sizeof(*dir));
    dir->n_pages_pub = new_cov / PK;
    cluster_dc_civac(dir, sizeof(*dir));
    L.covered_end = new_cov; L.n_pages = new_cov / PK;
}

static bool cluster_run_inplace(ggml_hexagon_cluster_sidecar * s, ggml_hexagon_cluster_sidecar::scratch * sc, uint32_t il) {
    ggml_hexagon_cluster * c = s->c;
    ggml_hexagon_cluster_layer & L = c->layers[il];
    const htp_fa_cluster_header & hdr = c->hdr;
    const uint32_t PK = hdr.page_keys, H = s->H, D = s->D;
    const uint32_t p0 = L.covered_end / PK;
    const uint32_t we = L.written_end > (uint32_t) opt_cluster_lag ? L.written_end - (uint32_t) opt_cluster_lag : 0;   // rows old enough to publish
    uint32_t p1 = we / PK;
    if (p1 > hdr.n_pages_max) p1 = hdr.n_pages_max;
    if (p1 <= p0 || !L.k_root) return true;
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t nbk_row = (uint32_t) L.k_root->nb[1];
    const uint8_t * krows = (const uint8_t *) L.k_root->data;
    cluster_dc_civac(krows + (size_t) p0 * PK * nbk_row, (size_t) (p1 - p0) * PK * nbk_row);   // the DSP wrote these rows
    uint8_t * lb = c->base + hdr.layer0_off + (size_t) il * hdr.layer_stride;
    const float inv = 1.0f / (float) PK;
    for (uint32_t h = 0; h < H; ++h) {
        uint8_t * cent = lb + hdr.off_centroids + ((size_t) h * hdr.n_pages_max + p0) * hdr.centroid_bytes;
        for (uint32_t p = p0; p < p1; ++p) {
            std::fill(sc->acc.begin(), sc->acc.end(), 0.0f);
            for (uint32_t i = 0; i < PK; ++i) {
                const uint16_t * row = (const uint16_t *) (krows + (size_t) (p * PK + i) * nbk_row + (size_t) h * D * 2);
                for (uint32_t d = 0; d < D; ++d) sc->acc[d] += cluster_f16_to_f32(row[d]);
            }
            uint16_t * cp = (uint16_t *) (cent + (size_t) (p - p0) * hdr.centroid_bytes);
            for (uint32_t d = 0; d < D; ++d) cp[d] = ggml_fp32_to_fp16(sc->acc[d] * inv);
        }
        cluster_dc_civac(cent, (size_t) (p1 - p0) * hdr.centroid_bytes);
    }
    auto * dir = (htp_fa_cluster_dir *) (c->base + HTP_FA_CLUSTER_DIR_OFF + (size_t) il * HTP_FA_CLUSTER_DIR_STRIDE);
    dir->n_pages_pub = p1;
    cluster_dc_civac(dir, sizeof(*dir));
    dir->covered_end = p1 * PK;
    cluster_dc_civac(dir, sizeof(*dir));
    L.covered_end = p1 * PK; L.n_pages = p1;
    { std::lock_guard<std::mutex> lk(s->stat_mu);
      s->t_cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      s->n_chunks++; }
    if (opt_verbose && opt_cluster_verify) {
        GGML_LOG_INFO("ggml-hex: cluster: layer %u in-place pages %u..%u published (covered %u)\n", il, p0, p1 - 1, L.covered_end);
    }
    return true;
}

static void cluster_reset_layer(ggml_hexagon_cluster * c, uint32_t il) {
    ggml_hexagon_cluster_layer & L = c->layers[il];
    auto * dir = (htp_fa_cluster_dir *) (c->base + HTP_FA_CLUSTER_DIR_OFF + (size_t) il * HTP_FA_CLUSTER_DIR_STRIDE);
    dir->covered_end = 0;
    cluster_dc_civac(dir, sizeof(*dir));
    dir->n_pages_pub = 0; dir->n_chunks = 0; dir->stale = 0;
    cluster_dc_civac(dir, sizeof(*dir));
    L.covered_end = 0; L.n_chunks = 0; L.n_pages = 0; L.written_end = 0; L.stale = false;
}

static void ggml_hexagon_cluster_sidecar_main(ggml_hexagon_cluster_sidecar * s) {
    if (opt_cluster_cpu >= 0) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(opt_cluster_cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
    ggml_hexagon_cluster * c = s->c;
    for (;;) {
        ggml_hexagon_cluster_job job;
        {
            std::unique_lock<std::mutex> lk(s->mu);
            s->cv.wait(lk, [&] { return s->stop || !s->jobs.empty(); });
            if (s->stop && s->jobs.empty()) return;
            job = s->jobs.front();
            s->jobs.pop_front();
            s->busy = true;
        }
        s->n_jobs++;
        const bool prefill = job.n_tokens > 1;
        const bool end_of_prefill = !prefill && c->last_n_tokens > 1;
        c->last_n_tokens = job.n_tokens;
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> plan(c->layers.size());
        for (uint32_t il = 0; il < c->layers.size(); ++il) {
            ggml_hexagon_cluster_layer & L = c->layers[il];
            if (!L.k_root || !L.v_root) continue;
            if (job.pos_min == 0 && (L.written_end > 0 || L.covered_end > 0)) {
                cluster_reset_layer(c, il);   // the cache is being filled from position 0 again
                s->n_resets++;
            }
            if (opt_cluster_inplace) {
                // descriptors above a rewritten position are stale: shrink the published prefix, then
                // publish every complete page below the new contiguous end
                if (!job.contiguous) {
                    cluster_inplace_shrink(c, il, (uint32_t) job.pos_min);
                    L.written_end = std::min<uint32_t>(L.written_end, (uint32_t) job.pos_min);
                    continue;
                }
                if ((uint64_t) job.pos_min > L.written_end) continue;   // a gap: nothing is published across it
                if ((uint64_t) job.pos_min < L.covered_end) cluster_inplace_shrink(c, il, (uint32_t) job.pos_min);
                L.written_end = (uint32_t) job.pos_max + 1;
                cluster_run_inplace(s, &s->scr[0], il);
                continue;
            }
            if (!job.contiguous || (uint64_t) job.pos_min < L.covered_end) {
                if (!L.stale) {
                    L.stale = true;
                    auto * dir = (htp_fa_cluster_dir *) (c->base + HTP_FA_CLUSTER_DIR_OFF + (size_t) il * HTP_FA_CLUSTER_DIR_STRIDE);
                    dir->stale = 1; cluster_dc_civac(dir, sizeof(*dir));
                    GGML_LOG_WARN("ggml-hex: cluster: layer %u: rows written below covered_end or non-contiguous; layer runs dense until a reset\n", il);
                }
                continue;
            }
            if (L.stale) continue;
            if ((uint64_t) job.pos_min > L.written_end) { L.stale = true; continue; }   // a gap: no clustering across it
            L.written_end = std::max<uint32_t>(L.written_end, (uint32_t) job.pos_max + 1);
            const uint32_t chunk = (uint32_t) opt_cluster_chunk;
            const uint32_t we = L.written_end > (uint32_t) opt_cluster_lag ? L.written_end - (uint32_t) opt_cluster_lag : 0;   // rows old enough to publish
            // Plan this layer's chunks (executed below, layers in parallel). The shadow may cover fewer
            // positions than the cache (memory budget): stop at its capacity.
            const uint32_t rows_max = c->hdr.n_pages_max * c->hdr.page_keys;
            uint32_t cov = L.covered_end, units = L.n_pages;
            auto cap_ok = [&](uint32_t n_keys) {
                if (opt_cluster_runs) return cov + n_keys <= rows_max && units + n_keys / c->hdr.avg_cluster + 2 <= c->hdr.n_runs_max;
                return (c->hdr.n_pages_max - units) * c->hdr.page_keys >= n_keys;
            };
            auto take = [&](uint32_t n_keys) {
                plan[il].push_back({ cov, cov + n_keys });
                units += opt_cluster_runs ? n_keys / c->hdr.avg_cluster + 2 : n_keys / c->hdr.page_keys;
                cov += n_keys;
            };
            while (we > cov && we - cov >= chunk && cap_ok(chunk)) take(chunk);
            if (end_of_prefill) {
                // the prompt ended: cluster the remaining tail down to a block boundary, within the shadow capacity
                uint32_t tail = (L.written_end - cov) & ~(HTP_FA_CLUSTER_PAGE_KEYS - 1);
                while (tail >= 2 * HTP_FA_CLUSTER_PAGE_KEYS && !cap_ok(tail)) tail -= HTP_FA_CLUSTER_PAGE_KEYS;
                if (tail >= 2 * HTP_FA_CLUSTER_PAGE_KEYS) take(tail);
            }
        }
        // Execute the plans: each layer's chunks in order, layers spread over the workers (thread-local
        // scratch; the OpenCL phase is serialized inside cluster_kmeans_gpu).
        {
            bool any = false;
            for (auto & pl : plan) any |= !pl.empty();
            if (any) {
                const int nw = std::max(1, std::min<int>(s->n_workers, (int) c->layers.size()));
                auto worker = [&](int w) {
                    ggml_hexagon_cluster_sidecar::scratch * sc = &s->scr[w];
                    for (uint32_t il = (uint32_t) w; il < c->layers.size(); il += (uint32_t) nw) {
                        ggml_hexagon_cluster_layer & L = c->layers[il];
                        for (auto & pr : plan[il]) {
                            if (L.stale) break;
                            const bool ok = opt_cluster_runs ? cluster_run_chunk_runs(s, sc, il, pr.first, pr.second)
                                                             : cluster_run_chunk(s, sc, il, pr.first, pr.second);
                            if (!ok) { L.stale = true; break; }
                        }
                    }
                };
                std::vector<std::thread> th;
                for (int w = 1; w < nw; ++w) th.emplace_back(worker, w);
                worker(0);
                for (auto & t : th) t.join();
            }
        }
        { std::lock_guard<std::mutex> lk(s->mu); s->busy = false; }
        s->cv_idle.notify_all();
        const double lag = (std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count() - (double) job.t_post_ns) * 1e-6;
        s->lag_max_ms = std::max(s->lag_max_ms, lag);
        if (opt_verbose) {
            GGML_LOG_DEBUG("ggml-hex: cluster: job %llu (%d tokens, pos %lld..%lld) done: chunks so far %llu, gpu %.1f ms, cpu %.1f ms, lag %.1f ms\n",
                           (unsigned long long) job.seq, job.n_tokens, (long long) job.pos_min, (long long) job.pos_max, (unsigned long long) s->n_chunks, s->t_gpu_ms, s->t_cpu_ms, lag);
        }
    }
}

static bool ggml_hexagon_cluster_sidecar_start(ggml_hexagon_cluster * c) {
    auto s = new ggml_hexagon_cluster_sidecar();
    s->c = c; s->N = (uint32_t) opt_cluster_chunk; s->H = c->hdr.n_kv_heads; s->D = c->hdr.D;
    s->C = opt_cluster_runs ? std::max<uint32_t>(1, s->N / (uint32_t) opt_cluster_avg) : s->N / c->hdr.page_keys;
    if (opt_cluster_inplace && !opt_cluster_desc_cpu) {
        // the FA op computes the descriptors itself: no sidecar thread at all
        delete s;
        return true;
    }
    if (opt_cluster_inplace) {
        // descriptors only: a CPU thread, no OpenCL
        s->scr.resize(1); s->scr[0].acc.resize(s->D);
        c->side = s;
        s->th = std::thread(ggml_hexagon_cluster_sidecar_main, s);
        return true;
    }
    cl_int err; cl_uint n = 0;
    if (!cluster_cl_ok(clGetPlatformIDs(1, &s->plat, &n), "clGetPlatformIDs") || !n) { delete s; return false; }
    if (!cluster_cl_ok(clGetDeviceIDs(s->plat, CL_DEVICE_TYPE_GPU, 1, &s->dev, &n), "clGetDeviceIDs") || !n) { delete s; return false; }
    s->ctx = clCreateContext(nullptr, 1, &s->dev, nullptr, nullptr, &err);
    if (!cluster_cl_ok(err, "clCreateContext")) { delete s; return false; }
    s->q = clCreateCommandQueueWithProperties(s->ctx, s->dev, nullptr, &err);
    if (!cluster_cl_ok(err, "clCreateCommandQueue")) { delete s; return false; }
    char opts[128]; snprintf(opts, sizeof(opts), "-cl-std=CL2.0 -DKM_D=%u -DKM_TILE=16 -DKM_NMAX=%u", s->D, s->N);
    s->prog = clCreateProgramWithSource(s->ctx, 1, &ggml_hexagon_cluster_cl_src, nullptr, &err);
    if (clBuildProgram(s->prog, 1, &s->dev, opts, nullptr, nullptr) != CL_SUCCESS) {
        size_t ln = 0; clGetProgramBuildInfo(s->prog, s->dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ln); std::string log(ln, 0);
        clGetProgramBuildInfo(s->prog, s->dev, CL_PROGRAM_BUILD_LOG, ln, log.data(), nullptr);
        GGML_LOG_ERROR("ggml-hex: cluster: k-means program build failed:\n%s\n", log.c_str()); delete s; return false;
    }
    s->k_assign = clCreateKernel(s->prog, "km_assign", &err); s->k_update = clCreateKernel(s->prog, "km_update", &err);
    const uint32_t nbk_row = s->H * s->D * 2;
    s->bK   = clCreateBuffer(s->ctx, CL_MEM_READ_ONLY,  (size_t) s->N * nbk_row, nullptr, &err);
    s->bMu  = clCreateBuffer(s->ctx, CL_MEM_READ_WRITE, (size_t) s->H * s->C * s->D * 4, nullptr, &err);
    s->bMu2 = clCreateBuffer(s->ctx, CL_MEM_READ_WRITE, (size_t) s->H * s->C * 4, nullptr, &err);
    s->bCnt = clCreateBuffer(s->ctx, CL_MEM_READ_WRITE, (size_t) s->H * s->C * 4, nullptr, &err);
    s->bAs  = clCreateBuffer(s->ctx, CL_MEM_READ_WRITE, (size_t) s->H * s->N, nullptr, &err);
    s->bDist = clCreateBuffer(s->ctx, CL_MEM_READ_WRITE, (size_t) s->H * s->N * 4, nullptr, &err);
    s->bChg = clCreateBuffer(s->ctx, CL_MEM_READ_WRITE, 4, nullptr, &err);
    if (!s->k_assign || !s->k_update || !s->bK || !s->bMu || !s->bMu2 || !s->bCnt || !s->bAs || !s->bDist || !s->bChg) {
        GGML_LOG_ERROR("ggml-hex: cluster: OpenCL objects failed\n"); delete s; return false;
    }
    s->n_workers = std::max(1, std::min(opt_cluster_threads, 16));
    s->scr.resize(s->n_workers);
    for (auto & sc : s->scr) {
        sc.mu0.resize((size_t) s->H * s->C * s->D); sc.mu2.resize((size_t) s->H * s->C);
        sc.assign.resize((size_t) s->H * s->N); sc.dist.resize((size_t) s->H * s->N); sc.acc.resize(s->D);
    }
    c->side = s;
    s->th = std::thread(ggml_hexagon_cluster_sidecar_main, s);
    return true;
}

static void ggml_hexagon_cluster_sidecar_stop(ggml_hexagon_cluster * c) {
    ggml_hexagon_cluster_sidecar * s = c->side;
    if (!s) return;
    { std::lock_guard<std::mutex> lk(s->mu); s->stop = true; s->jobs.clear(); }
    s->cv.notify_one();
    if (s->th.joinable()) s->th.join();
    if (opt_cluster_verify) GGML_LOG_WARN("ggml-hex: cluster sidecar delayed re-verify: %llu of %llu rows were stale when first read\n", (unsigned long long) s->n_stale_rows, (unsigned long long) s->n_checked_rows);
    GGML_LOG_INFO("ggml-hex: cluster sidecar: %llu jobs, %llu chunks, %llu resets, gpu %.1f ms, cpu %.1f ms, max lag %.1f ms, verify failures %llu\n",
                  (unsigned long long) s->n_jobs, (unsigned long long) s->n_chunks, (unsigned long long) s->n_resets, s->t_gpu_ms, s->t_cpu_ms, s->lag_max_ms,
                  (unsigned long long) s->n_verify_fail);
    if (s->q) clFinish(s->q);
    for (cl_mem m : { s->bK, s->bMu, s->bMu2, s->bCnt, s->bAs, s->bDist, s->bChg }) if (m) clReleaseMemObject(m);
    if (s->k_assign) clReleaseKernel(s->k_assign);
    if (s->k_update) clReleaseKernel(s->k_update);
    if (s->prog) clReleaseProgram(s->prog);
    if (s->q) clReleaseCommandQueue(s->q);
    if (s->ctx) clReleaseContext(s->ctx);
    delete s;
    c->side = nullptr;
}
#endif // GGML_HEXAGON_HETERO

// A hexagon buffer is being freed. If the sidecar reads it (llama's KV cache: llama_free runs before
// the backend session is released), drop the queued jobs, let a chunk in flight finish its reads,
// and forget the roots so nothing touches the freed memory.
static void ggml_hexagon_cluster_buffer_freed(ggml_hexagon_session * sess, ggml_backend_buffer_t buffer) {
    if (!sess || !sess->cluster) return;
    ggml_hexagon_cluster * c = sess->cluster;
#ifdef GGML_HEXAGON_HETERO
    ggml_hexagon_cluster_sidecar * s = c->side;
    if (s) {
        std::unique_lock<std::mutex> lk(s->mu);
        bool refs = false;
        for (auto & L : c->layers) {
            refs |= (L.k_root && L.k_root->buffer == buffer) || (L.v_root && L.v_root->buffer == buffer);
        }
        if (!refs) return;
        s->jobs.clear();
        s->cv_idle.wait(lk, [&] { return !s->busy; });
        for (auto & L : c->layers) {
            if (L.k_root && L.k_root->buffer == buffer) L.k_root = nullptr;
            if (L.v_root && L.v_root->buffer == buffer) L.v_root = nullptr;
        }
        return;
    }
#endif
    for (auto & L : c->layers) {
        if (L.k_root && L.k_root->buffer == buffer) L.k_root = nullptr;
        if (L.v_root && L.v_root->buffer == buffer) L.v_root = nullptr;
    }
}

// Record which KV rows the graph is about to write. Called at the TOP of graph_compute, after
// llama's set_inputs and BEFORE the ops run: the SET_ROWS index leaf (src[1]) is a transient whose
// buffer the scheduler recycles for activations during compute, so its positions must be read now,
// not afterwards. The k/v cache roots (src[2]) are stable. All layers of a batch share one index.
static void ggml_hexagon_cluster_track_graph(ggml_hexagon_session * sess, const ggml_cgraph * graph) {
    ggml_hexagon_cluster * c = sess->cluster;
    if (!c) return;
    const ggml_tensor * idx = nullptr;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * n = graph->nodes[i];
        if (n->op != GGML_OP_SET_ROWS || !n->src[2] || !n->src[1]) continue;
        int il = -1; bool is_v = false;
        if (sscanf(n->src[2]->name, "cache_k_l%d", &il) == 1) { is_v = false; }
        else if (sscanf(n->src[2]->name, "cache_v_l%d", &il) == 1) { is_v = true; }
        else continue;
        if (il < 0 || (size_t) il >= c->layers.size()) continue;
        if (is_v) c->layers[il].v_root = n->src[2]; else c->layers[il].k_root = n->src[2];
        idx = n->src[1];
    }
    c->idx_tensor = idx;
    c->pending = false;
    if (!idx || !idx->data || idx->ne[0] <= 0) return;
    if (idx->type != GGML_TYPE_I64 && idx->type != GGML_TYPE_I32) return;
    const int64_t n = idx->ne[0];
    int64_t mn = INT64_MAX, mx = -1; bool bad = false;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t v = idx->type == GGML_TYPE_I64 ? ((const int64_t *) idx->data)[i] : (int64_t) ((const int32_t *) idx->data)[i];
        if (v < 0 || v >= (int64_t) c->kv_cache_size) { bad = true; break; }
        mn = std::min(mn, v); mx = std::max(mx, v);
    }
    if (c->hdr.flags & (HTP_FA_CLUSTER_HDR_DSP_DESC | HTP_FA_CLUSTER_HDR_HOST_ROWS)) {
        // rows [0, mn) were written by earlier graphs and are in the sequence (a write from 0 is a reset,
        // a write below the previous end a rewind). Written on the dispatch thread BEFORE the graph is
        // submitted, so the kernel never uses pages the asynchronous sidecar has not yet retired; in
        // DSP-descriptor mode the FA op also describes pages up to here.
        const uint32_t rows_valid = (bad || mx < 0) ? 0u : (uint32_t) mn;
        for (uint32_t il = 0; il < c->layers.size(); ++il) {
            auto * hd = (htp_fa_cluster_hdir *) (c->base + HTP_FA_CLUSTER_HDIR_OFF + (size_t) il * HTP_FA_CLUSTER_DIR_STRIDE);
            if (hd->rows_valid != rows_valid) {
                hd->rows_valid = rows_valid;
                cluster_dc_civac(hd, sizeof(*hd));
            }
        }
    }
    if (bad || mx < 0) return;   // out-of-range (rewind, multi-stream, recycled buffer): skip this batch
    ggml_hexagon_cluster_job & j = c->pending_job;
    j.seq = ++c->seq; j.n_tokens = (int32_t) n; j.pos_min = mn; j.pos_max = mx;
    j.contiguous = (mx - mn + 1 == n);
    c->pending = true;
}

// The graph completed (the DSP has written the KV rows): hand the batch captured before compute to
// the sidecar, which can now read those cache rows.
static void ggml_hexagon_cluster_after_compute(ggml_hexagon_session * sess) {
    ggml_hexagon_cluster * c = sess->cluster;
    if (!c || !c->pending) return;
    c->pending = false;
#ifdef GGML_HEXAGON_HETERO
    if (!c->side) return;
    ggml_hexagon_cluster_job job = c->pending_job;
    job.t_post_ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count();
    { std::lock_guard<std::mutex> lk(c->side->mu); c->side->jobs.push_back(job); }
    c->side->cv.notify_one();
#endif
}

static bool ggml_hexagon_cluster_init(ggml_hexagon_session * sess, uint32_t n_layers, uint32_t kv_size, uint32_t n_kv_heads, uint32_t D) {
    // Memory budget. The shadow is a second copy of the KV cache, and the cDSP maps weights, cache and
    // shadow into one virtual space of opt_vmem bytes (~3.35 GB): a 16k context on Qwen3-1.7B is already
    // 1.9 GB of cache + 1.3 GB of weights, so a full shadow cannot be mapped and the DSP faults on first
    // touch (the fastrpc mapping is delayed). Cluster only as many leading positions as fit; the rest of
    // the context stays in the dense tail. Weight buffers are not accounted here: a fixed margin stands in
    // (GGML_HEXAGON_CLUSTER_MEM_MB overrides the budget).
    const uint64_t vmem     = sess->max_vmem ? (uint64_t) sess->max_vmem : (uint64_t) opt_vmem;
    const uint64_t kv_bytes = (uint64_t) n_layers * 2 * kv_size * n_kv_heads * D * 2;
    const uint64_t weights_margin = 1536ull << 20;
    const uint64_t budget = opt_cluster_mem_mb > 0 ? ((uint64_t) opt_cluster_mem_mb << 20)
                          : (vmem > kv_bytes + weights_margin ? vmem - kv_bytes - weights_margin : 0);
    struct htp_fa_cluster_header hdr;
    uint32_t kv_shadow = kv_size;
    const uint32_t hdr_flags = HTP_FA_CLUSTER_HDR_HOST_ROWS | (opt_cluster_runs ? HTP_FA_CLUSTER_HDR_RUNS : 0u)
                             | (opt_cluster_inplace ? (HTP_FA_CLUSTER_HDR_INPLACE | (opt_cluster_desc_cpu ? 0u : HTP_FA_CLUSTER_HDR_DSP_DESC)) : 0u);
    if (n_layers > HTP_FA_CLUSTER_MAX_LAYERS) {
        GGML_LOG_WARN("ggml-hex: cluster attention disabled: %u layers exceed the directory (%d)\n", n_layers, HTP_FA_CLUSTER_MAX_LAYERS);
        opt_cluster_density = -1;
        return false;
    }
    uint64_t size = htp_fa_cluster_layout_v2(&hdr, n_layers, kv_shadow, n_kv_heads, D, (uint32_t) opt_cluster_page, (uint32_t) opt_cluster_avg, hdr_flags);
    // In-place pages keep only descriptors (1/32 of the KV with 16-key pages, 1/128 with 64-key): the
    // budget is for the shadow copy, which does not exist here.
    if (!opt_cluster_inplace && size > budget) {
        kv_shadow = ((uint32_t) ((double) kv_size * (double) budget / (double) size)) & ~63u;
        while (kv_shadow >= 1024 && (size = htp_fa_cluster_layout_v2(&hdr, n_layers, kv_shadow, n_kv_heads, D, (uint32_t) opt_cluster_page, (uint32_t) opt_cluster_avg, hdr_flags)) > budget) {
            kv_shadow -= 1024;
        }
        if (kv_shadow < 1024 || size > budget) {
            GGML_LOG_WARN("ggml-hex: cluster attention disabled: no room for a shadow (KV cache %llu MB + weights margin %llu MB of the %llu MB cDSP vmem budget); decode runs dense\n",
                          (unsigned long long) (kv_bytes >> 20), (unsigned long long) (weights_margin >> 20), (unsigned long long) (vmem >> 20));
            opt_cluster_density = -1;
            return false;
        }
        GGML_LOG_WARN("ggml-hex: cluster attention: shadow capped to the first %u of %u positions (%llu MB fit the %llu MB left of the cDSP vmem budget after %llu MB of KV cache); the rest stays dense\n",
                      kv_shadow, kv_size, (unsigned long long) (size >> 20), (unsigned long long) (budget >> 20), (unsigned long long) (kv_bytes >> 20));
    }
    auto c = new ggml_hexagon_cluster();
    c->sess = sess;
    c->layers.resize(n_layers);
    c->kv_cache_size = kv_size;
    c->hdr  = hdr;
    c->size = (size_t) size;
    c->buf  = ggml_backend_buft_alloc_buffer(&sess->buffer_type, c->size);
    if (!c->buf) {
        GGML_LOG_ERROR("ggml-hex: cluster: shadow buffer alloc (%zu MB) failed\n", c->size >> 20);
        delete c;
        return false;
    }
    ggml_backend_buffer_set_usage(c->buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(c->buf->context);
    c->base = sbuf->base;
    // Header + directories: zero (covered_end 0 => every layer runs dense until pages are published).
    const size_t hdr_bytes = (size_t) c->hdr.layer0_off;
    memset(c->base, 0, hdr_bytes);
    memcpy(c->base, &c->hdr, sizeof(c->hdr));
    cluster_dc_civac(c->base, hdr_bytes);

    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    c->tctx = ggml_init(ip);
    c->ctrl = ggml_new_tensor_1d(c->tctx, GGML_TYPE_F32, (int64_t) (c->size / 4));
    c->ctrl->buffer = c->buf;
    c->ctrl->data   = c->base;
    ggml_set_name(c->ctrl, "hexagon_cluster_shadow");
    sess->cluster = c;
#ifdef GGML_HEXAGON_HETERO
    if (!ggml_hexagon_cluster_sidecar_start(c)) {
        GGML_LOG_WARN("ggml-hex: cluster: sidecar unavailable (OpenCL); the shadow stays empty and decode runs dense\n");
    }
#else
    GGML_LOG_WARN("ggml-hex: cluster: built without OpenCL; no clustering sidecar, decode runs dense\n");
#endif
    GGML_LOG_INFO("ggml-hex: %s cluster attention: %s %zu MB (%u layers x %u heads x %u pages of %u keys), density %d permille, window %d, flags 0x%x, sink %d\n",
                  sess->c_name(), opt_cluster_inplace ? (opt_cluster_desc_cpu ? "in-place pages, CPU descriptors" : "in-place pages, DSP descriptors") : opt_cluster_runs ? "shadow, whole-cluster runs" : "shadow", c->size >> 20, n_layers, n_kv_heads, c->hdr.n_pages_max, c->hdr.page_keys,
                  opt_cluster_density, opt_cluster_window, opt_cluster_flags, opt_cluster_sink);
    return true;
}

// Tag a decode FLASH_ATTN_EXT node for the page-list kernel. Returns false (node untouched) when the
// shape does not fit the prototype. The layer index comes from the KV cache tensor name
// (cache_k_l%d) and falls back to the node's ordinal among the graph's FA nodes.
static bool ggml_hexagon_cluster_prepare(ggml_hexagon_session * sess, ggml_tensor * n, const struct htp_fa_kernel_params * kp, int fa_ordinal) {
    ggml_hexagon_cluster * c = sess->cluster;
    n->op_params[HTP_FA_CLUSTER_OPP_MAGIC] = 0;
    const ggml_tensor * q = n->src[0], * k = n->src[1], * v = n->src[2], * m = n->src[3];
    if (kp->kernel_type != HTP_FA_KERNEL_HVX || !kp->u.hvx.split_kv) return false;
    if (q->ne[1] != 1 || q->ne[3] != 1) return false;
    const int DK = (int) q->ne[0], DV = (int) v->ne[0];
    if (DK != DV || (DK != 128 && DK != 64)) return false;
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) return false;
    if (m && (m->type != GGML_TYPE_F16 || m->ne[2] != 1)) return false;
    if (n->src[4] || n->src[5] || n->src[6] || n->src[7]) return false;   // sinks / sparse / hetero: not in the prototype
    float max_bias = 0.0f;
    memcpy(&max_bias, &n->op_params[1], sizeof(float));
    if (max_bias != 0.0f) return false;
    if (k->ne[2] == 0 || q->ne[2] % k->ne[2] != 0) return false;
    if ((uint32_t) k->ne[2] != c->hdr.n_kv_heads || (uint32_t) DK != c->hdr.D) return false;
    if (!ggml_backend_buffer_is_hexagon(q->buffer) || !ggml_backend_buffer_is_hexagon(k->buffer) ||
        !ggml_backend_buffer_is_hexagon(v->buffer) || (m && !ggml_backend_buffer_is_hexagon(m->buffer))) return false;

    int il = -1;
    const ggml_tensor * root = k->view_src ? k->view_src : k;
    if (sscanf(root->name, "cache_k_l%d", &il) != 1) {
        il = fa_ordinal;
    }
    if (il < 0 || (uint32_t) il >= c->hdr.n_layers) return false;

    n->op_params[HTP_FA_CLUSTER_OPP_MAGIC]   = (int32_t) HTP_FA_CLUSTER_MAGIC;
    n->op_params[HTP_FA_CLUSTER_OPP_LAYER]   = il;
    n->op_params[HTP_FA_CLUSTER_OPP_DENSITY] = opt_cluster_density;
    n->op_params[HTP_FA_CLUSTER_OPP_WINDOW]  = opt_cluster_window;
    n->op_params[HTP_FA_CLUSTER_OPP_FLAGS]   = (int32_t) (opt_cluster_flags | (opt_cluster_sink > 0 ? (1u << 16) : 0u));   // HTP_FA_CLUSTER_FLAG_FORCE = 1 page
    n->src[7] = c->ctrl;
    c->n_tagged++;
    if (opt_verbose || c->n_tagged == 1) {
        GGML_LOG_INFO("ggml-hex: cluster: tagged node %s as layer %d (kv %lld, %d heads, D %d)\n",
                      n->name, il, (long long) k->ne[1], (int) k->ne[2], DK);
    }
    return true;
}

static void ggml_hexagon_cluster_free(ggml_hexagon_session * sess) {
    ggml_hexagon_cluster * c = sess->cluster;
    if (!c) return;
    GGML_LOG_INFO("ggml-hex: %s cluster: %u nodes tagged\n", sess->c_name(), c->n_tagged);
#ifdef GGML_HEXAGON_HETERO
    ggml_hexagon_cluster_sidecar_stop(c);
#endif
    if (c->tctx) ggml_free(c->tctx);
    if (c->buf) ggml_backend_buffer_free(c->buf);
    delete c;
    sess->cluster = nullptr;
}

static htp_op_code op_remap_to_htp(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_FLASH_ATTN_EXT:  return HTP_OP_FLASH_ATTN_EXT;
        case GGML_OP_MUL_MAT:         return HTP_OP_MUL_MAT;
        case GGML_OP_MUL_MAT_ID:      return HTP_OP_MUL_MAT_ID;
        case GGML_OP_MUL:             return HTP_OP_MUL;
        case GGML_OP_ADD:             return HTP_OP_ADD;
        case GGML_OP_ADD_ID:          return HTP_OP_ADD_ID;
        case GGML_OP_SUB:             return HTP_OP_SUB;
        case GGML_OP_DIV:             return HTP_OP_DIV;
        case GGML_OP_CPY:             return HTP_OP_CPY;
        case GGML_OP_CONT:            return HTP_OP_CPY;
        case GGML_OP_GET_ROWS:        return HTP_OP_GET_ROWS;
        case GGML_OP_SET_ROWS:        return HTP_OP_SET_ROWS;
        case GGML_OP_SUM_ROWS:        return HTP_OP_SUM_ROWS;
        case GGML_OP_ARGSORT:         return HTP_OP_ARGSORT;
        case GGML_OP_NORM:            return HTP_OP_NORM;
        case GGML_OP_L2_NORM:         return HTP_OP_L2_NORM;
        case GGML_OP_RMS_NORM:        return HTP_OP_RMS_NORM;
        case GGML_OP_CONCAT:          return HTP_OP_CONCAT;
        case GGML_OP_SCALE:           return HTP_OP_SCALE;
        case GGML_OP_CLAMP:           return HTP_OP_CLAMP;
        case GGML_OP_SQR:             return HTP_OP_SQR;
        case GGML_OP_SQRT:            return HTP_OP_SQRT;
        case GGML_OP_SOFT_MAX:        return HTP_OP_SOFTMAX;
        case GGML_OP_SSM_CONV:        return HTP_OP_SSM_CONV;
        case GGML_OP_GATED_DELTA_NET: return HTP_OP_GATED_DELTA_NET;
        case GGML_OP_ROPE:            return HTP_OP_ROPE;
        case GGML_OP_REPEAT:          return HTP_OP_REPEAT;
        case GGML_OP_CUMSUM:          return HTP_OP_CUMSUM;
        case GGML_OP_FILL:            return HTP_OP_FILL;
        case GGML_OP_DIAG:            return HTP_OP_DIAG;
        case GGML_OP_SOLVE_TRI:       return HTP_OP_SOLVE_TRI;
        case GGML_OP_TRI:             return HTP_OP_TRI;
        case GGML_OP_PAD:             return HTP_OP_PAD;
        case GGML_OP_IM2COL:          return HTP_OP_IM2COL;
        case GGML_OP_CUSTOM:
            if (opt_sync_probe && (uint32_t) t->op_params[HTP_SYNC_PROBE_P_MAGIC] == HTP_SYNC_PROBE_MAGIC) {
                return HTP_OP_SYNC_PROBE;
            }
            break;

        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_SILU:       return HTP_OP_UNARY_SILU;
                case GGML_UNARY_OP_GELU:       return HTP_OP_UNARY_GELU;
                case GGML_UNARY_OP_GELU_QUICK: return HTP_OP_UNARY_GELU;
                case GGML_UNARY_OP_SIGMOID:    return HTP_OP_UNARY_SIGMOID;
                case GGML_UNARY_OP_NEG:        return HTP_OP_UNARY_NEG;
                case GGML_UNARY_OP_EXP:        return HTP_OP_UNARY_EXP;
                case GGML_UNARY_OP_SOFTPLUS:   return HTP_OP_UNARY_SOFTPLUS;
                case GGML_UNARY_OP_TANH:       return HTP_OP_UNARY_TANH;
            default:
                break;
            }
            break;

        case GGML_OP_GLU:
            switch (ggml_get_glu_op(t)) {
                case GGML_GLU_OP_SWIGLU:     return HTP_OP_GLU_SWIGLU;
                case GGML_GLU_OP_SWIGLU_OAI: return HTP_OP_GLU_SWIGLU_OAI;
                case GGML_GLU_OP_GEGLU:      return HTP_OP_GLU_GEGLU;
                default: break;
            }
            break;

        default:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    return HTP_OP_INVALID;
}

static inline bool op_is_compute(ggml_tensor *node)
{
    return !ggml_op_is_empty(node->op) && !ggml_is_empty(node) && (node->flags & GGML_TENSOR_FLAG_COMPUTE);
}

static bool mm_is_hmx_eligible(const ggml_tensor * t) {
    if (opt_nhmx == 0) { return false; }

    const ggml_tensor * src0 = t->src[0];
    const ggml_tensor * src1 = t->src[1];

    const int wtype = src0->type;
    const bool is_repack    = ggml_hexagon_is_repack_type((ggml_type) wtype);
    const bool is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    const bool is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    const int ne01_padded = is_repack ? hex_round_up(src0->ne[1], 32) : src0->ne[1];

    return ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, ne01_padded, is_matmul_id, is_batched);
}

static bool is_mergeable_mul_mat(const ggml_tensor * t) {
    if (!t || t->op != GGML_OP_MUL_MAT)   return false;
    if (t->src[1]->type != GGML_TYPE_F32) return false;
    return ggml_is_quantized(t->src[0]->type) && !mm_is_hmx_eligible(t);
}

static bool is_mergeable_mul_mat_pair(const ggml_tensor * n1, const ggml_tensor * n2) {
    if (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        return false;
    }
    if (n1->src[1] != n2->src[1]) {
        return false;
    }
    if (n1->src[0]->ne[0] != n2->src[0]->ne[0] ||
        n1->src[0]->ne[1] != n2->src[0]->ne[1]) {
        return false;
    }
    if (n1->src[0]->type != n2->src[0]->type) {
        return false;
    }
    return true;
}

static bool is_qkv_mergeable(const ggml_tensor * n_q, const ggml_tensor * n_k, const ggml_tensor * n_v) {
    if (!is_mergeable_mul_mat(n_q) || !is_mergeable_mul_mat(n_k) || !is_mergeable_mul_mat(n_v)) {
        return false;
    }
    if (n_q->src[1] != n_k->src[1] || n_q->src[1] != n_v->src[1]) {
        return false;
    }
    if (n_q->src[0]->type != n_k->src[0]->type || n_q->src[0]->type != n_v->src[0]->type) {
        return false;
    }
    if (n_k->src[0]->ne[0] != n_v->src[0]->ne[0] ||
        n_k->src[0]->ne[1] != n_v->src[0]->ne[1]) {
        return false;
    }
    if (n_q->src[0]->ne[0] != n_k->src[0]->ne[0]) {
        return false;
    }
    return true;
}

// XAttention block-selection scoring: MUL_MAT -> SOFT_MAX -> RESHAPE -> SUM_ROWS.
//
// Unfused, the [kv, nq, nh] score tensor is written by the matmul, read and
// rewritten by the softmax, and read again by sum_rows -- four passes over a
// tensor that is `bs` times larger than the result. Measured on SM8750 that chain
// is ~90% of the marginal cost of the scoring pass while the matmul is ~10%
// (docs/backend/snapdragon/xattn-block-selection.md). Fused, the intermediate
// stays in VTCM and only K plus the small output cross DRAM.
//
// The RESHAPE has to join the fused set even though it computes nothing: it is
// SUM_ROWS's src[0], and htp_opnode::get_inputs() treats any src that is not
// itself a member of the group as a real input to be marshalled.
static bool try_fuse_xattn_score(const ggml_hexagon_session * sess, const ggml_cgraph * graph, int & i,
                                 std::vector<htp_opnode> & nodes) {
    if (!opt_xattn_fusion) {
        return false;
    }
    if (i + 3 >= graph->n_nodes) {
        return false;
    }

    ggml_tensor * mm = graph->nodes[i];
    ggml_tensor * sm = graph->nodes[i + 1];
    ggml_tensor * rs = graph->nodes[i + 2];
    ggml_tensor * sr = graph->nodes[i + 3];

    if (mm->op != GGML_OP_MUL_MAT || sm->op != GGML_OP_SOFT_MAX ||
        rs->op != GGML_OP_RESHAPE || sr->op != GGML_OP_SUM_ROWS) {
        return false;
    }

    // strict producer -> consumer chain
    if (sm->src[0] != mm || rs->src[0] != sm || sr->src[0] != rs) {
        return false;
    }

    // Every intermediate must be dead outside the group. The RESHAPE needs the
    // raw use count rather than ggml_node_has_n_uses, which rejects anything with
    // a view_src -- true of every reshape. Its view source is the SOFT_MAX output,
    // and that tensor having exactly one use is already checked below, so a second
    // consumer reaching in through the view cannot hide from us.
    if (!ggml_node_has_n_uses(graph, i, 1) || !ggml_node_has_n_uses(graph, i + 1, 1)) {
        return false;
    }
    if (ggml_node_get_use_count(graph, i + 2) != 1 || (rs->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    // no mask and no ALiBi slope -- the kernel implements a plain softmax
    if (sm->src[1] != nullptr) {
        return false;
    }
    float scale, max_bias;
    memcpy(&scale,    &sm->op_params[0], sizeof(float));
    memcpy(&max_bias, &sm->op_params[1], sizeof(float));
    if (max_bias != 0.0f) {
        return false;
    }

    const ggml_tensor * k = mm->src[0];
    const ggml_tensor * q = mm->src[1];

    if (k->type != GGML_TYPE_F16 || q->type != GGML_TYPE_F32 || sr->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(k) || !ggml_is_contiguous(q) || !ggml_is_contiguous(sr)) {
        return false;
    }
    if (k->ne[3] != 1 || q->ne[3] != 1 || k->ne[0] != q->ne[0] || k->ne[2] != q->ne[2]) {
        return false;
    }

    const int64_t hs   = k->ne[0];
    const int64_t kv   = k->ne[1];
    const int64_t nh   = k->ne[2];
    const int64_t nq   = q->ne[1];
    const int64_t nblk = sr->ne[1];

    // the reshape must only be regrouping the kv axis into [bs, nblk]
    if (rs->ne[2] != nq || rs->ne[3] != nh || nblk == 0 || rs->ne[0] * nblk != kv) {
        return false;
    }
    if (sr->ne[0] != 1 || sr->ne[2] != nq || sr->ne[3] != nh) {
        return false;
    }

    // Matches the DSP kernel's aligned-load constraints. This gate must stay a
    // SUPERSET of the checks in op_xattn_score: a fused node never reaches
    // ggml_backend_hexagon_supports_op, so this is the only gate there is, and a
    // HTP_STATUS_NO_SUPPORT returned from the device aborts every remaining op in
    // the batch (main.c:1044 stops the loop on the first non-OK status). Anything
    // rejected here falls back to the unfused four-op chain, which is safe.
    if ((hs % 64) != 0 || (kv % 32) != 0 || (k->nb[1] % 128) != 0) {
        return false;
    }

    // The kernel reads K and Q rows with aligned HVX loads. Contiguity alone does
    // not imply a 128-byte-aligned base or row stride, and q is the output of a
    // cont/view in the XAttention graph, so check both explicitly.
    if ((reinterpret_cast<uintptr_t>(k->data) % 128) != 0 ||
        (reinterpret_cast<uintptr_t>(q->data) % 128) != 0 ||
        (q->nb[1] % 128) != 0 || (q->nb[2] % 128) != 0 || (k->nb[2] % 128) != 0) {
        HEX_VERBOSE("ggml-hex: xattn-score align reject : k.data %% 128 = %u q.data %% 128 = %u "
                    "q.nb1 %zu q.nb2 %zu k.nb2 %zu\n",
                    (unsigned) (reinterpret_cast<uintptr_t>(k->data) % 128),
                    (unsigned) (reinterpret_cast<uintptr_t>(q->data) % 128),
                    (size_t) q->nb[1], (size_t) q->nb[2], (size_t) k->nb[2]);
        return false;
    }

    htp_opnode node(mm, {}, HTP_OP_XATTN_SCORE);
    node.add_fused(sm);
    node.add_fused(rs);
    node.add_fused(sr);
    memcpy(&node.kernel_params[0], &scale, sizeof(float));

    nodes.push_back(std::move(node));
    i += 3;  // consume SOFT_MAX, RESHAPE and SUM_ROWS

    HEX_VERBOSE("ggml-hex: fused xattn-score : hs %lld kv %lld nh %lld nq %lld nblk %lld\n", (long long) hs,
                (long long) kv, (long long) nh, (long long) nq, (long long) nblk);
    return true;
}

static bool try_fuse_node(const ggml_hexagon_session * sess, const ggml_cgraph * graph, int & i, std::vector<htp_opnode> & nodes) {
    if (!opt_opfusion) {
        return false;
    }

    if (try_fuse_xattn_score(sess, graph, i, nodes)) {
        return true;
    }

    ggml_tensor * n = graph->nodes[i];
    ggml_tensor * next_node = (i + 1 < graph->n_nodes) ? graph->nodes[i + 1] : nullptr;

    if (n->op == GGML_OP_RMS_NORM && next_node) {
        if (next_node->op == GGML_OP_MUL && op_is_compute(next_node) && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            htp_opnode node(n, {}, HTP_OP_RMS_NORM_MUL);
            node.add_fused(next_node);

            auto inputs = node.get_inputs();
            const struct ggml_tensor * src0 = inputs[0];
            const struct ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : nullptr;
            ggml_hexagon_precompute_unary_params(sess,
                node.opcode, src0, src1, node.dst(),
                (struct htp_unary_kernel_params *)node.kernel_params
            );

            nodes.push_back(std::move(node));
            i++; // skip the fused MUL node
            return true;
        }
    }

    if (is_mergeable_mul_mat(n)) {
        ggml_tensor * n1 = (i + 1 < graph->n_nodes) ? graph->nodes[i + 1] : nullptr;
        ggml_tensor * n2 = (i + 2 < graph->n_nodes) ? graph->nodes[i + 2] : nullptr;
        if (is_qkv_mergeable(n, n1, n2)) {
            struct htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_qkv_params(sess, n1->src[0], n1->src[1], &kparams);
            if ((size_t)kparams.vtcm_size <= sess->vtcm_size) {
                // Reorder to KVQ: K (n1), V (n2), Q (n)
                htp_opnode node(n1, {}, HTP_OP_MUL_MAT_QKV);
                node.add_fused(n2, true);
                node.add_fused(n, true);
                memcpy(node.kernel_params, &kparams, sizeof(kparams));
                nodes.push_back(std::move(node));
                i += 2;
                return true;
            } else {
                HEX_VERBOSE("ggml-hex: skip QKV fusion because VTCM needed (%d) > budget (%zu)\n",
                            kparams.vtcm_size, sess->vtcm_size);
            }
        }
        if (is_mergeable_mul_mat_pair(n, n1)) {
            struct htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_ffn_params(sess, n->src[0], n->src[1], &kparams);
            if ((size_t)kparams.vtcm_size <= sess->vtcm_size) {
                htp_opnode node(n, {}, HTP_OP_MUL_MAT_FFN);
                node.add_fused(n1, true);
                memcpy(node.kernel_params, &kparams, sizeof(kparams));
                nodes.push_back(std::move(node));
                i += 1;
                return true;
            } else {
                HEX_VERBOSE("ggml-hex: skip FFN fusion because VTCM needed (%d) > budget (%zu)\n",
                            kparams.vtcm_size, sess->vtcm_size);
            }
        }
    }

    if (n->op == GGML_OP_MUL_MAT && next_node) {
        if (next_node->op == GGML_OP_ADD && op_is_compute(next_node) && ggml_can_fuse(graph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
            if (next_node->src[0] == n || next_node->src[1] == n) {
                const struct ggml_tensor * src2 = (next_node->src[0] == n) ? next_node->src[1] : next_node->src[0];
                struct htp_mm_kernel_params kparams;
                ggml_hexagon_precompute_fused_matmul_add_params(sess, n->src[0], n->src[1], src2, next_node, &kparams);
                const int src1_nrows = n->src[1]->ne[1] * n->src[1]->ne[2] * n->src[1]->ne[3];
                const bool can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
                if (can_fuse && (size_t)kparams.vtcm_size <= sess->vtcm_size) {
                    htp_opnode node(n, {}, HTP_OP_MUL_MAT_ADD);
                    node.add_fused(next_node);
                    memcpy(node.kernel_params, &kparams, sizeof(kparams));
                    nodes.push_back(std::move(node));
                    i += 1;
                    return true;
                } else if (can_fuse) {
                    HEX_VERBOSE("ggml-hex: skip MUL_MAT_ADD fusion because VTCM needed (%d) > budget (%zu)\n",
                                kparams.vtcm_size, sess->vtcm_size);
                }
            }
        }
    }

    return false;
}

static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    const std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    bool cache_hit = (graph->uid != 0 && sess->cached_graph.uid == graph->uid);
    if (cache_hit) {
        nodes_ptr = &sess->cached_graph.htp_nodes;
    } else {
        computed_nodes.reserve(graph->n_nodes);
        int hetero_slot = 0;
        int cluster_fa_ordinal = 0;

        // Fuse and finalize
        for (int i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            if (!op_is_compute(n)) {
                continue;
            }

            if (try_fuse_node(sess, graph, i, computed_nodes)) {
                continue;
            }

            htp_opnode node(n, {}, HTP_OP_INVALID);
            node.opcode = op_remap_to_htp(n);
            if (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    node.node->src[0], node.node->src[1], node.node,
                    (struct htp_mm_kernel_params *)node.kernel_params
                );
            } else if (node.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    node.node,
                    (struct htp_fa_kernel_params *)node.kernel_params
                );
#ifdef GGML_HEXAGON_HETERO
                if (opt_hetero_frac > 0.0f) {
                    if (!sess->hetero && !ggml_hexagon_hetero_init(sess)) {
                        GGML_LOG_WARN("ggml-hex: hetero decode attention disabled (init failed)\n");
                        opt_hetero_frac = 0.0f;
                    }
                    if (sess->hetero && ggml_hexagon_hetero_prepare(sess, n, (const struct htp_fa_kernel_params *) node.kernel_params, hetero_slot)) {
                        hetero_slot++;
                    }
                }
                if (opt_fa_fold > 0 && opt_hetero_frac <= 0.0f) {
                    // (init happens on the first FA node even if it is not tagged: the keep-alive control needs it)
                    if (!sess->hfold && !ggml_hexagon_hfold_init(sess)) {
                        GGML_LOG_WARN("ggml-hex: hetero prefill split disabled (init failed)\n");
                        opt_fa_fold = 0;
                    }
                    if (sess->hfold && !opt_fa_fold_notag) {
                        // a node the graph built for the split (src[8] set) that the sidecar cannot
                        // take would run the pool alone and silently lose its exceptions: say so
                        if (!ggml_hexagon_hfold_prepare(sess, n, (const struct htp_fa_kernel_params *) node.kernel_params) && n->src[8]) {
                            static int warned = 0;
                            if (!warned++) GGML_LOG_WARN("ggml-hex: fa-fold: %s has an exception membership but cannot be tagged (shape/alignment): its output lacks the exceptions\n", n->name);
                        }
                    }
                }
#endif
                if (opt_cluster_density >= 0 && opt_hetero_frac <= 0.0f) {
                    if (!sess->cluster) {
                        // Shadow dims from the first FA node: kv_size from the root KV cache tensor
                        // ([n_embd_k_gqa, kv_size, n_stream]), n_layers = FA nodes in this graph.
                        const ggml_tensor * k    = n->src[1];
                        const ggml_tensor * root = k->view_src ? k->view_src : k;
                        const uint32_t kv_size   = (uint32_t) (k->view_src ? root->ne[1] : k->ne[1]);
                        uint32_t n_layers = 0;
                        for (int j = 0; j < graph->n_nodes; ++j) {
                            n_layers += graph->nodes[j]->op == GGML_OP_FLASH_ATTN_EXT;
                        }
                        if (!ggml_hexagon_cluster_init(sess, n_layers, kv_size, (uint32_t) k->ne[2], (uint32_t) k->ne[0])) {
                            GGML_LOG_WARN("ggml-hex: cluster attention disabled (init failed)\n");
                            opt_cluster_density = -1;
                        }
                    }
                    if (sess->cluster && n->src[0]->ne[1] == 1) {
                        ggml_hexagon_cluster_prepare(sess, n, (const struct htp_fa_kernel_params *) node.kernel_params, cluster_fa_ordinal);
                    }
                    cluster_fa_ordinal++;
                }
            } else if (htp_op_is_unary(node.opcode)) {
                auto inputs = node.get_inputs();
                const struct ggml_tensor * src0 = inputs[0];
                const struct ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : nullptr;
                ggml_hexagon_precompute_unary_params(sess,
                    node.opcode, src0, src1, node.dst(),
                    (struct htp_unary_kernel_params *)node.kernel_params
                );
            }
            computed_nodes.push_back(std::move(node));
        }

        if (graph->uid != 0) {
            sess->cached_graph.uid = graph->uid;
            sess->cached_graph.htp_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_graph.htp_nodes;
        } else {
            nodes_ptr = &computed_nodes;
        }
    }
    // Read the KV write positions now, before the ops recycle the index leaf's buffer (runs on
    // cache hit and miss; the k/v roots are refreshed here too when the graph was rebuilt).
    if (sess->cluster) {
        ggml_hexagon_cluster_track_graph(sess, graph);
    }

#ifdef GGML_HEXAGON_HETERO
    if (sess->hetero) {
        ggml_hexagon_hetero_post(sess, *nodes_ptr);
    }
    if (sess->hfold) {
        ggml_hexagon_hfold_post(sess, *nodes_ptr);
    }
#endif

    // Queue and execute
    if (opt_opstage & HTP_OPSTAGE_QUEUE) {
        for (const auto & node : *nodes_ptr) {
            sess->enqueue_op(node);
        }
    }

    if (opt_async) {
        // Submit only. The join is ggml_backend_hexagon_synchronize, which
        // ggml_backend_graph_compute calls right after this.
        sess->flush_batch();
    } else {
        // Wait until all pending ops complete
        sess->flush();
        ggml_hexagon_cluster_after_compute(sess);
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush();
    ggml_hexagon_cluster_after_compute(sess);
}

static std::vector<int> ggml_hexagon_graph_optimize_reorder(const std::vector<htp_opnode> & nodes) {
    const int n = nodes.size();

    std::vector<int> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows use to reuse dynamically quantized src1 in VTCM.

    // TODO: the current version might do incorrect reordering in cases where quantized src0
    //       input is an output of another Op.

    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) {
            continue;
        }

        res.push_back(i0);

        const auto & node0 = nodes[i0];

        if (!node0.stackable()) {
            continue;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr int N_FORWARD = 16;

        for (int i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            if (used[i1]) {
                continue;
            }

            const auto & node1 = nodes[i1];

            if (node1.stackable() && node1.same_input(node0)) {
                res.push_back(i1);
                used[i1] = true;
            }
        }
    }

    return res;
}

static void ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf) {
    const int n = gf->n_nodes;

    constexpr int MAX_FUSE = 16;

    enum ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    for (int i = 0; i < n; i++) {
        htp_opnode node = {
            /*.node =*/gf->nodes[i],
            /*.fused =*/{},
        };

        // fuse only ops that start with these operations
        // can be expanded when needed
        if (node.op() == GGML_OP_ADD ||
            node.op() == GGML_OP_NORM ||
            node.op() == GGML_OP_RMS_NORM) {
            ops[0] = node.op();

            int f = i + 1;
            while (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                if (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    break;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            for (; f > 1; f--) {
                if (ggml_can_fuse(gf, i, ops, f)) {
                    break;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            for (int k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                node.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(node));
    }

    const auto order = ggml_hexagon_graph_optimize_reorder(nodes);

    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_UNUSED(backend);
}

static struct ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_hexagon_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ ggml_backend_hexagon_graph_optimize,
};

static ggml_guid_t ggml_backend_hexagon_guid() {
    static ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    return &guid;
}

bool ggml_backend_is_hexagon(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

static ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, const char * params) {
    auto sess = static_cast<ggml_hexagon_session *>(dev->context);

    return new ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

static const char * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    auto sess = static_cast<ggml_hexagon_session *>(dev->context);
    return sess->c_name();

    GGML_UNUSED(dev);
}

static const char * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    return "Hexagon";
    GGML_UNUSED(dev);
}

static void ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = *free;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ (bool) opt_hostbuf,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto sess = static_cast<ggml_hexagon_session *>(dev->context);
    return &sess->buffer_type;
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_repack_buffer_type(ggml_backend_dev_t dev) {
    auto sess = static_cast<ggml_hexagon_session *>(dev->context);
    return &sess->repack_buffer_type;
}

static bool ggml_hexagon_supported_buffer(ggml_hexagon_session *sess, const struct ggml_tensor * t) {
    if (t && t->buffer) {
        if (ggml_backend_buffer_is_hexagon(t->buffer)      == false) return false; // not our buffer
        if (ggml_backend_hexagon_buffer_get_sess(t->buffer) != sess) return false; // wrong session
    }
    return true;
}

static bool ggml_hexagon_supported_buffers(ggml_hexagon_session *sess, const struct ggml_tensor * t) {
    // all srcs & dsts must be mapped to the same session
    if (!ggml_hexagon_supported_buffer(sess, t)) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (!ggml_hexagon_supported_buffer(sess, t->src[i])) {
            return false;
        }
    }

    return true;
}

static bool ggml_hexagon_supported_cpy(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    GGML_UNUSED(sess);

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    // for now we can do f32 -> f16 and f16 -> f32 (without reshaping)
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;
    if ( dst->type != GGML_TYPE_F32 &&  dst->type != GGML_TYPE_F16) return false;

    const bool sametype   = (src0->type == dst->type);
    const bool transposed = ggml_is_transposed(src0) || ggml_is_transposed(dst);
    const bool sameshape  = !transposed && ggml_are_same_shape(src0, dst);

    // can handle any shape and any same-type (pretty slow if reshaping is required)
    if (sametype) return true;

    // cannot handle re-shaping and type conversion at the same time
    if (!sameshape) return false;

    return true;
}

static bool ggml_hexagon_supported_cont(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    GGML_UNUSED(sess);
    const struct ggml_tensor * src0 = op->src[0];

    // CONT is same-type only, supports f32 and f16
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;

    return true;
}

static bool ggml_hexagon_supported_repeat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    GGML_UNUSED(sess);
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    // Support f32 and f16
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;

    // src and dst must be the same type
    if (src0->type != dst->type) return false;

    // dst dims must be multiples of src dims
    if (dst->ne[0] % src0->ne[0] != 0) return false;
    if (dst->ne[1] % src0->ne[1] != 0) return false;
    if (dst->ne[2] % src0->ne[2] != 0) return false;
    if (dst->ne[3] % src0->ne[3] != 0) return false;

    // require contiguous tensors (no transposition)
    if (ggml_is_transposed(src0) || ggml_is_transposed(dst)) return false;

    return true;
}

static bool ggml_hexagon_supported_concat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    int dim = ((const int32_t *) op->op_params)[0];
    if (dim < 0 || dim >= GGML_MAX_DIMS) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const struct ggml_tensor * src = op->src[i];
        if (!src) {
            continue;
        }
        if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_I32 && src->type != GGML_TYPE_F16) {
            return false;
        }
    }

    return true;
    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_fill(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * dst = op;

    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        return false;
    }

    return true;
    GGML_UNUSED(sess);
}

static bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    auto sess = static_cast<ggml_hexagon_session *>(dev->context);

    // reject ops that match the filter
    if (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        return false;
    }

    // all srcs & dsts must be mapped to the same session
    if (!ggml_hexagon_supported_buffers(sess, op)) {
        ggml_hexagon_dump_op_supp(sess->name, op, false);
        return false;
    }

    bool supp = false;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            supp = true;
            break;

        case GGML_OP_MUL:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
            supp = ggml_hexagon_supported_binary(sess, op);
            break;

        case GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            break;

        case GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            break;

        case GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            break;

        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
            supp = ggml_hexagon_supported_unary(sess, op);
            break;

        case GGML_OP_SQR:
        case GGML_OP_SQRT:
            supp = ggml_hexagon_supported_unary(sess, op);
            break;

        case GGML_OP_SUM_ROWS:
            supp = ggml_hexagon_supported_sum_rows(sess, op);
            break;

        case GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            break;

        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    break;
                default:
                    break;
            }
            break;

        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    break;
                default:
                    break;
            }
            break;

        case GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            break;

        case GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            break;

        case GGML_OP_CUSTOM:
            supp = opt_sync_probe && (uint32_t) op->op_params[HTP_SYNC_PROBE_P_MAGIC] == HTP_SYNC_PROBE_MAGIC;
            break;

        case GGML_OP_SET_ROWS:
            supp = ggml_hexagon_supported_set_rows(sess, op);
            break;

        case GGML_OP_GET_ROWS:
            supp = ggml_hexagon_supported_get_rows(sess, op);
            break;

        case GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            break;

        case GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            break;

        case GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            break;

        case GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            break;

        case GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            break;

        case GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            break;

        case GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            break;

        case GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            break;

        case GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            break;

        case GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            break;

        case GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            break;

        case GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            break;

        case GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            break;

        case GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            break;

        default:
            break;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    return supp;
}

static bool ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_alignment != ggml_backend_hexagon_buffer_type_get_alignment) {
        return false;
    }

    auto s0 = static_cast<ggml_hexagon_session *>(dev->context);
    auto s1 = static_cast<ggml_backend_hexagon_buffer_type_context *>(buft->context)->sess;

    // Need session/domain-id for buffers to be compatible
    bool supp = (s0->session_id == s1->session_id);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s (%d)\n", s0->name.c_str(), s1->name.c_str(), (int) supp);

    return supp;
}

static ggml_backend_buffer_type_t * ggml_backend_hexagon_device_get_extra_buffers_type(ggml_backend_dev_t dev) {
    auto s0 = static_cast<ggml_hexagon_session *>(dev->context);
    HEX_VERBOSE("ggml-hex: device-get-extra-buft : %s \n", s0->name.c_str());

    static ggml_backend_buffer_type_t bufts[2];
    bufts[0] = ggml_backend_hexagon_device_get_repack_buffer_type(dev);
    bufts[1] = NULL;
    return bufts;
}

static const struct ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,  // ggml_backend_hexagon_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ NULL,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//** backend registry

#define GGML_HEXAGON_MAX_SESSIONS 16

struct ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d\n", opt_arch);

    // Create devices / sessions
    for (size_t i = 0; i < opt_ndev; i++) {
        devices[i].iface = ggml_backend_hexagon_device_i;
        devices[i].reg   = reg;
        try {
            devices[i].context = new ggml_hexagon_session(i, &devices[i]);
        } catch (const std::exception & exc) {
            GGML_LOG_ERROR("ggml-hex: failed to create device/session %zu\n", i);
            devices[i].context = nullptr;
        }
    }
}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices / sessions
    for (size_t i = 0; i < opt_ndev; i++) {
        auto sess = static_cast<ggml_hexagon_session *>(devices[i].context);
        delete sess;
    }
}

static const char * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    return "HTP";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_hexagon_reg_get_device_count(ggml_backend_reg_t reg) {
    return opt_ndev;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    if (index >= opt_ndev || !hreg->devices[index].context) {
        return nullptr;
    }

    return &hreg->devices[index];
}

static void * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0 && opt_hostbuf) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_hexagon_device_get_extra_buffers_type;
        return (void *) fct;
    }

    return NULL;
    GGML_UNUSED(reg);
}

template<typename T> std::vector<T> str_to_vec(const char* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    while (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, nullptr, 0));
    }

    return v;
}

template<typename T, int BASE=10> std::string vec_to_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(BASE) << std::showbase;
    for (auto i : v) { ss << i << ','; }
    auto str = ss.str(); str.pop_back(); // drop last comma
    return str;
}

static void ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((unsigned int) HTP_TYPE_Q4_0 == (unsigned int) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q4_1 == (unsigned int) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_Q8_0 == (unsigned int) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_MXFP4 == (unsigned int) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned int) HTP_TYPE_IQ4_NL == (unsigned int) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");

    const char * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    const char * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    const char * str_opstage  = getenv("GGML_HEXAGON_OPSTAGE");
    const char * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    const char * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    const char * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    const char * str_async    = getenv("GGML_HEXAGON_ASYNC");
    const char * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    const char * str_xattn_fusion = getenv("GGML_HEXAGON_XATTN_FUSION");
    const char * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    const char * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    const char * str_etm      = getenv("GGML_HEXAGON_ETM");
    const char * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    const char * str_use_hmx  = getenv("GGML_HEXAGON_USE_HMX");
    const char * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    const char * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    const char * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    const char * str_fa_kvres = getenv("GGML_HEXAGON_FA_KV_RESIDENCY");
    const char * str_lm_head   = getenv("GGML_HEXAGON_LM_HEAD");
    const char * str_fa_decode = getenv("GGML_HEXAGON_FA_DECODE");
    const char * str_fa_sparse = getenv("GGML_HEXAGON_FA_SPARSE");
    const char * str_sync_probe = getenv("GGML_HEXAGON_SYNC_PROBE");
    const char * str_hetero_frac = getenv("GGML_HEXAGON_HETERO_FRAC");
    const char * str_hetero_span = getenv("GGML_HEXAGON_HETERO_SPAN");
    const char * str_hetero_cpu  = getenv("GGML_HEXAGON_HETERO_CPU");
    const char * str_hetero_adapt = getenv("GGML_HEXAGON_HETERO_ADAPT");
    const char * str_fa_fold      = getenv("GGML_HEXAGON_FA_FOLD");
    const char * str_fa_fold_cpu  = getenv("GGML_HEXAGON_FA_FOLD_CPU");
    const char * str_fa_fold_check = getenv("GGML_HEXAGON_FA_FOLD_CHECK");
    const char * str_fa_fold_svmtab = getenv("GGML_HEXAGON_FA_FOLD_SVMTAB");
    const char * str_fa_fold_svmq   = getenv("GGML_HEXAGON_FA_FOLD_SVMQ");
    const char * str_fa_fold_sync   = getenv("GGML_HEXAGON_FA_FOLD_SYNC");
    const char * str_fa_fold_perf   = getenv("GGML_HEXAGON_FA_FOLD_PERF");
    const char * str_fa_fold_flush  = getenv("GGML_HEXAGON_FA_FOLD_FLUSH");
    const char * str_fa_fold_stage  = getenv("GGML_HEXAGON_FA_FOLD_STAGE_QB");
    const char * str_fa_fold_keep   = getenv("GGML_HEXAGON_FA_FOLD_KEEPALIVE");
    const char * str_fa_fold_keepp  = getenv("GGML_HEXAGON_FA_FOLD_KEEPALIVE_PERIOD_US");
    const char * str_fa_fold_gpudone = getenv("GGML_HEXAGON_FA_FOLD_GPUDONE");
    const char * str_fa_fold_preq   = getenv("GGML_HEXAGON_FA_FOLD_PREQ");
    const char * str_fa_fold_chain  = getenv("GGML_HEXAGON_FA_FOLD_CHAIN");
    const char * str_fa_fold_qbh    = getenv("GGML_HEXAGON_FA_FOLD_STAGE_QBH");
    const char * str_fa_fold_gspins = getenv("GGML_HEXAGON_FA_FOLD_GATE_SPINS");
    const char * str_fa_fold_pwg    = getenv("GGML_HEXAGON_FA_FOLD_PWG");
    const char * str_fa_fold_pwgs   = getenv("GGML_HEXAGON_FA_FOLD_PWG_STAGE");
    const char * str_fa_fold_trace  = getenv("GGML_HEXAGON_FA_FOLD_TRACE");
    const char * str_fa_fold_ionr   = getenv("GGML_HEXAGON_FA_FOLD_ION_READY");
    const char * str_fa_fold_ktbuf  = getenv("GGML_HEXAGON_FA_FOLD_KTBUF");
    const char * str_fa_fold_notag  = getenv("GGML_HEXAGON_FA_FOLD_NOTAG");
    const char * str_fa_fold_probe  = getenv("GGML_HEXAGON_FA_FOLD_PROBE");
    const char * str_fa_fold_gdelay = getenv("GGML_HEXAGON_FA_FOLD_GATE_DELAY");
    const char * str_fa_fold_acq    = getenv("GGML_HEXAGON_FA_FOLD_ACQ");
    const char * str_fa_fold_trig   = getenv("GGML_HEXAGON_FA_FOLD_TRIG");
    const char * str_fa_fold_qatom  = getenv("GGML_HEXAGON_FA_FOLD_QATOMIC");
    const char * str_fa_fold_wgf    = getenv("GGML_HEXAGON_FA_FOLD_WGFENCE");
    const char * str_fa_fold_acc16  = getenv("GGML_HEXAGON_FA_FOLD_ACC16");
    const char * str_fa_fold_knat   = getenv("GGML_HEXAGON_FA_FOLD_KNAT");
    const char * str_fa_fold_pidle  = getenv("GGML_HEXAGON_FA_FOLD_PREQ_IDLE_US");
    const char * str_fa_fold_qmap   = getenv("GGML_HEXAGON_FA_FOLD_QMAP");
    const char * str_cluster  = getenv("GGML_HEXAGON_CLUSTER_ATTN");
    const char * str_cluster_cpu = getenv("GGML_HEXAGON_CLUSTER_CPU");
    const char * str_cluster_verify = getenv("GGML_HEXAGON_CLUSTER_VERIFY");
    const char * str_cluster_chunk = getenv("GGML_HEXAGON_CLUSTER_CHUNK");
    const char * str_cluster_pos   = getenv("GGML_HEXAGON_CLUSTER_POSITIONAL");
    const char * str_cluster_mem   = getenv("GGML_HEXAGON_CLUSTER_MEM_MB");
    const char * str_cluster_page  = getenv("GGML_HEXAGON_CLUSTER_PAGE");
    const char * str_cluster_sink  = getenv("GGML_HEXAGON_CLUSTER_SINK");
    const char * str_cluster_inpl  = getenv("GGML_HEXAGON_CLUSTER_INPLACE");
    const char * str_cluster_runs  = getenv("GGML_HEXAGON_CLUSTER_RUNS");
    const char * str_cluster_avg   = getenv("GGML_HEXAGON_CLUSTER_AVG");
    const char * str_cluster_thr   = getenv("GGML_HEXAGON_CLUSTER_THREADS");
    const char * str_cluster_lag   = getenv("GGML_HEXAGON_CLUSTER_LAG");
    const char * str_cluster_dcpu  = getenv("GGML_HEXAGON_CLUSTER_DESC_CPU");
    const char * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    const char * str_arch     = getenv("GGML_HEXAGON_ARCH");
    const char * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    const char * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    const char * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");

    // Init Arch first since it affects other defaults
    if (!str_arch) {
        int err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } else {
            if (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } else if (opt_arch > 81) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v81\n", opt_arch);
                opt_arch = 81;
            }
        }
    } else {
        if (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, NULL, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;

    auto RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? new std::regex(str_opfilter, RE_ICASE) : NULL;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_hostbuf   = str_hostbuf  ? atoi(str_hostbuf)                      : opt_hostbuf;
    opt_opstage   = str_opstage  ? strtoul(str_opstage, NULL, 0)          : opt_opstage;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, NULL, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, NULL, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, NULL, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  NULL, 0)          : opt_oppoll;
    opt_async     = str_async    ? atoi(str_async)                        : opt_async;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_xattn_fusion = str_xattn_fusion ? atoi(str_xattn_fusion)          : opt_xattn_fusion;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, NULL, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : (str_use_hmx ? atoi(str_use_hmx) : opt_nhmx);
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_kv_residency = str_fa_kvres ? atoi(str_fa_kvres)               : opt_fa_kv_residency;
    opt_fa_sparse = str_fa_sparse ? atoi(str_fa_sparse)                   : opt_fa_sparse;
    opt_lm_head   = str_lm_head   ? atoi(str_lm_head)                     : opt_lm_head;
    opt_fa_decode = str_fa_decode ? atoi(str_fa_decode)                   : opt_fa_decode;
    opt_sync_probe = str_sync_probe ? atoi(str_sync_probe)                  : opt_sync_probe;
    opt_hetero_frac = str_hetero_frac ? strtof(str_hetero_frac, nullptr)     : opt_hetero_frac;
    opt_hetero_span = str_hetero_span ? atoi(str_hetero_span)                : opt_hetero_span;
    opt_hetero_cpu  = str_hetero_cpu  ? atoi(str_hetero_cpu)                 : opt_hetero_cpu;
    opt_hetero_adapt = str_hetero_adapt ? atoi(str_hetero_adapt)             : opt_hetero_adapt;
    opt_fa_fold      = str_fa_fold      ? atoi(str_fa_fold)                  : opt_fa_fold;
    opt_fa_fold_cpu  = str_fa_fold_cpu  ? atoi(str_fa_fold_cpu)              : opt_fa_fold_cpu;
    opt_fa_fold_check = str_fa_fold_check ? atoi(str_fa_fold_check)          : opt_fa_fold_check;
    opt_fa_fold_svmtab = str_fa_fold_svmtab ? atoi(str_fa_fold_svmtab)       : opt_fa_fold_svmtab;
    opt_fa_fold_svmq   = str_fa_fold_svmq   ? atoi(str_fa_fold_svmq)         : opt_fa_fold_svmq;
    opt_fa_fold_sync   = str_fa_fold_sync   ? atoi(str_fa_fold_sync)         : opt_fa_fold_sync;
    opt_fa_fold_perf   = str_fa_fold_perf   ? atoi(str_fa_fold_perf)         : opt_fa_fold_perf;
    opt_fa_fold_flush  = str_fa_fold_flush  ? atoi(str_fa_fold_flush)        : opt_fa_fold_flush;
    opt_fa_fold_stage_qb = str_fa_fold_stage ? atoi(str_fa_fold_stage)       : opt_fa_fold_stage_qb;
    opt_fa_fold_keepalive = str_fa_fold_keep ? atoi(str_fa_fold_keep)        : opt_fa_fold_keepalive;
    opt_fa_fold_keepalive_period = str_fa_fold_keepp ? atoi(str_fa_fold_keepp) : opt_fa_fold_keepalive_period;
    opt_fa_fold_gpudone = str_fa_fold_gpudone ? atoi(str_fa_fold_gpudone)     : opt_fa_fold_gpudone;
    opt_fa_fold_preq    = str_fa_fold_preq    ? atoi(str_fa_fold_preq)        : opt_fa_fold_preq;
    opt_fa_fold_chain   = str_fa_fold_chain   ? atoi(str_fa_fold_chain)       : opt_fa_fold_chain;
    opt_fa_fold_qbh     = str_fa_fold_qbh     ? atoi(str_fa_fold_qbh)         : opt_fa_fold_qbh;
    opt_fa_fold_gate_spins = str_fa_fold_gspins ? atoi(str_fa_fold_gspins)    : opt_fa_fold_gate_spins;
    opt_fa_fold_pwg     = str_fa_fold_pwg     ? atoi(str_fa_fold_pwg)         : opt_fa_fold_pwg;
    opt_fa_fold_pwg_stage = str_fa_fold_pwgs  ? atoi(str_fa_fold_pwgs)        : opt_fa_fold_pwg_stage;
    opt_fa_fold_trace   = str_fa_fold_trace   ? atoi(str_fa_fold_trace)       : opt_fa_fold_trace;
    opt_fa_fold_ion_ready = str_fa_fold_ionr  ? atoi(str_fa_fold_ionr)        : opt_fa_fold_ion_ready;
    opt_fa_fold_ktbuf   = str_fa_fold_ktbuf   ? atoi(str_fa_fold_ktbuf)       : opt_fa_fold_ktbuf;
    opt_fa_fold_notag   = str_fa_fold_notag   ? atoi(str_fa_fold_notag)       : opt_fa_fold_notag;
    opt_fa_fold_probe   = str_fa_fold_probe   ? atoi(str_fa_fold_probe)       : opt_fa_fold_probe;
    opt_fa_fold_gate_delay = str_fa_fold_gdelay ? atoi(str_fa_fold_gdelay)    : opt_fa_fold_gate_delay;
    opt_fa_fold_acq     = str_fa_fold_acq     ? atoi(str_fa_fold_acq)         : opt_fa_fold_acq;
    opt_fa_fold_trig    = str_fa_fold_trig    ? atoi(str_fa_fold_trig)        : opt_fa_fold_trig;
    if (opt_fa_fold_trig >= 2) opt_fa_fold_ion_ready = 0;   // the gate must wait for the relay's word, written after the trigger
    opt_fa_fold_qatomic = str_fa_fold_qatom   ? atoi(str_fa_fold_qatom)       : opt_fa_fold_qatomic;
    opt_fa_fold_wgfence = str_fa_fold_wgf     ? atoi(str_fa_fold_wgf)         : opt_fa_fold_wgfence;
    opt_fa_fold_acc16   = str_fa_fold_acc16   ? atoi(str_fa_fold_acc16)       : opt_fa_fold_acc16;
    opt_fa_fold_knat    = str_fa_fold_knat    ? atoi(str_fa_fold_knat)        : opt_fa_fold_knat;
    opt_fa_fold_preq_idle = str_fa_fold_pidle ? atoi(str_fa_fold_pidle)       : opt_fa_fold_preq_idle;
    opt_fa_fold_qmap    = str_fa_fold_qmap    ? atoi(str_fa_fold_qmap)        : opt_fa_fold_qmap;
    if (opt_fa_fold_check == 1) opt_fa_fold_gpudone = 0;   // the check holds the done words until the reference ran (CHECK=2: post-hoc rows on the chain path)
#ifndef GGML_HEXAGON_HETERO
    if (opt_hetero_frac > 0.0f) {
        GGML_LOG_WARN("ggml-hex: GGML_HEXAGON_HETERO_FRAC set but the backend was built without OpenCL; ignored\n");
        opt_hetero_frac = 0.0f;
    }
    if (opt_fa_fold > 0) {
        GGML_LOG_WARN("ggml-hex: GGML_HEXAGON_FA_FOLD set but the backend was built without OpenCL; ignored\n");
        opt_fa_fold = 0;
    }
#endif
    if (str_cluster) {
        int density = -1, window = opt_cluster_window;
        unsigned flags = 0;
        if (sscanf(str_cluster, "%d,%d,%i", &density, &window, &flags) >= 1 && density >= 0) {
            opt_cluster_density = density;
            opt_cluster_window  = window;
            opt_cluster_flags   = flags;
        } else {
            GGML_LOG_WARN("ggml-hex: GGML_HEXAGON_CLUSTER_ATTN='%s' not understood (want <density_permille>,<window>[,flags]); ignored\n", str_cluster);
        }
    }
    opt_cluster_cpu    = str_cluster_cpu    ? atoi(str_cluster_cpu)    : opt_cluster_cpu;
    opt_cluster_verify = str_cluster_verify ? atoi(str_cluster_verify) : opt_cluster_verify;
    opt_cluster_chunk  = str_cluster_chunk  ? atoi(str_cluster_chunk)  : opt_cluster_chunk;
    opt_cluster_positional = str_cluster_pos ? atoi(str_cluster_pos)  : opt_cluster_positional;
    opt_cluster_mem_mb     = str_cluster_mem ? atoi(str_cluster_mem)  : opt_cluster_mem_mb;
    opt_cluster_page       = str_cluster_page ? atoi(str_cluster_page) : opt_cluster_page;
    if (opt_cluster_page != 16 && opt_cluster_page != 32 && opt_cluster_page != 64) {
        GGML_LOG_WARN("ggml-hex: GGML_HEXAGON_CLUSTER_PAGE=%d not 16/32/64; using 64\n", opt_cluster_page);
        opt_cluster_page = 64;
    }
    opt_cluster_sink = str_cluster_sink ? atoi(str_cluster_sink) : opt_cluster_sink;
    opt_cluster_inplace = str_cluster_inpl ? atoi(str_cluster_inpl) : opt_cluster_inplace;
    opt_cluster_lag     = str_cluster_lag  ? atoi(str_cluster_lag)  : opt_cluster_lag;
    opt_cluster_desc_cpu = str_cluster_dcpu ? atoi(str_cluster_dcpu) : opt_cluster_desc_cpu;
    if (opt_cluster_inplace) opt_cluster_positional = 1;   // in-place pages are positional by construction
    opt_cluster_runs = str_cluster_runs ? atoi(str_cluster_runs) : opt_cluster_runs;
    opt_cluster_avg  = str_cluster_avg  ? atoi(str_cluster_avg)  : opt_cluster_avg;
    opt_cluster_threads = str_cluster_thr ? atoi(str_cluster_thr) : opt_cluster_threads;
    if (opt_cluster_runs) {
        // whole clusters live in the shadow; rows are addressed individually (page 64 keeps the row region aligned)
        opt_cluster_inplace = 0;
        opt_cluster_page    = 64;
        if (opt_cluster_sink <= 0) opt_cluster_sink = 4;
        if (opt_cluster_avg < 8 || opt_cluster_avg > 256) {
            GGML_LOG_WARN("ggml-hex: GGML_HEXAGON_CLUSTER_AVG=%d out of range (8..256); using 32\n", opt_cluster_avg);
            opt_cluster_avg = 32;
        }
    }
    if (opt_cluster_sink < 0 || opt_cluster_sink > opt_cluster_page) {
        GGML_LOG_WARN("ggml-hex: GGML_HEXAGON_CLUSTER_SINK=%d out of range (0..page keys %d); using %d\n", opt_cluster_sink, opt_cluster_page, opt_cluster_page);
        opt_cluster_sink = opt_cluster_page;
    }
    if (opt_cluster_chunk < 128 || opt_cluster_chunk > 1024 || (opt_cluster_chunk % 64)) {
        GGML_LOG_WARN("ggml-hex: GGML_HEXAGON_CLUSTER_CHUNK=%d out of range (128..1024, multiple of 64); using 1024\n", opt_cluster_chunk);
        opt_cluster_chunk = 1024;
    }
    opt_ndev      = str_ndev     ? strtoul(str_ndev, NULL, 0)             : opt_ndev;
    opt_hostbuf   = str_hostbuf  ? atoi(str_hostbuf)                      : opt_hostbuf;
    opt_mbuf      = str_mbuf     ? strtoul(str_mbuf, NULL, 0) * MiB       : opt_mbuf;
    opt_vmem      = str_vmem     ? strtoul(str_vmem, NULL, 0) * MiB       : opt_vmem;

    if (opt_ndev > GGML_HEXAGON_MAX_SESSIONS) {
        opt_ndev = GGML_HEXAGON_MAX_SESSIONS;
    }

#if defined(__ANDROID__)
    if (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

    if (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            auto v  = str_to_vec<uint32_t>(str_profile);
            switch (v.size()) {
                case 1:  opt_profile = v[0]; return opt_pmu_evt; // mode with default pmu events
                case 8:  opt_profile = 2;    return v;           // mode with custom  pmu events
                default: opt_profile = 0;    return {};          // garbage input
            }}();
        if (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = new ggml_hexagon_registry(reg);
}

static const struct ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(void) {
    static bool initialized = false;

    static ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ NULL };

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            auto nErr = htpdrv_init();
            if (nErr != AEE_SUCCESS) {
                return NULL;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)

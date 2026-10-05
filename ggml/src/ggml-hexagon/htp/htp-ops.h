#ifndef HTP_OPS_H
#define HTP_OPS_H

#include <assert.h>
#include <stdint.h>
#include <string.h>

// ggml-common.h must be included prio to this header

enum htp_status {
    HTP_STATUS_OK             = 1,
    HTP_STATUS_INTERNAL_ERR   = 2,
    HTP_STATUS_NO_SUPPORT     = 3,
    HTP_STATUS_INVAL_PARAMS   = 4,
    HTP_STATUS_VTCM_TOO_SMALL = 5,
};

// First set of values must match the ggml_type.
// Duplicated here because we can't include full ggml.h in the htp build.
// We have some static_asserts in the cpp code to ensure things are in sync.
enum htp_data_type {
    HTP_TYPE_F32    = 0,
    HTP_TYPE_F16    = 1,
    HTP_TYPE_Q4_0   = 2,
    HTP_TYPE_Q4_1   = 3,
    HTP_TYPE_Q8_0   = 8,
    HTP_TYPE_IQ4_NL = 20,
    HTP_TYPE_I32    = 26,
    HTP_TYPE_I64    = 27,
    HTP_TYPE_MXFP4  = 39,

    // types used internally for repack, dyn.quant, etc
    HTP_TYPE_Q4_0_TILED = 200,
    HTP_TYPE_Q4_1_TILED,
    HTP_TYPE_Q8_0_TILED,
    HTP_TYPE_MXFP4_TILED,

    HTP_TYPE_INVALID
};

// Constats for internal types
#define QK_Q4_0_TILED  256  // 32x32 Q4_0 tiled layout
#define QK_Q8_0_TILED  128  // 32x32 Q8_0 tiled layout
#define QK_MXFP4_TILED 256  // 32x32 MXFP4 tiled layout



// Mask to enable various stages of the Ops.
// Used for debugging and profiling.
enum htp_op_stage {
    HTP_OPSTAGE_QUEUE    = (1 << 0),  // Enable Queueing (ie calls into NPU)
    HTP_OPSTAGE_COMPUTE  = (1 << 1),  // Enable Compute
};

// Do not reorder first 4 (used as an index)
enum htp_op_code {
    HTP_OP_MUL = 0,
    HTP_OP_ADD = 1,
    HTP_OP_SUB = 2,
    HTP_OP_DIV = 3,
    HTP_OP_MUL_MAT,
    HTP_OP_MUL_MAT_ID,
    HTP_OP_MUL_MAT_QKV,
    HTP_OP_MUL_MAT_FFN,
    HTP_OP_MUL_MAT_ADD,
    HTP_OP_RMS_NORM,
    HTP_OP_RMS_NORM_MUL,
    HTP_OP_UNARY_SILU,
    HTP_OP_UNARY_GELU,
    HTP_OP_UNARY_SIGMOID,
    HTP_OP_UNARY_EXP,
    HTP_OP_UNARY_NEG,
    HTP_OP_UNARY_SOFTPLUS,
    HTP_OP_UNARY_TANH,
    HTP_OP_GLU_SWIGLU,
    HTP_OP_GLU_SWIGLU_OAI,
    HTP_OP_GLU_GEGLU,
    HTP_OP_SOFTMAX,
    HTP_OP_ADD_ID,
    HTP_OP_ROPE,
    HTP_OP_FLASH_ATTN_EXT,
    HTP_OP_SET_ROWS,
    HTP_OP_GET_ROWS,
    HTP_OP_SCALE,
    HTP_OP_CPY,
    HTP_OP_ARGSORT,
    HTP_OP_SQR,
    HTP_OP_SQRT,
    HTP_OP_SUM_ROWS,
    HTP_OP_SSM_CONV,
    HTP_OP_REPEAT,
    HTP_OP_CUMSUM,
    HTP_OP_FILL,
    HTP_OP_DIAG,
    HTP_OP_SOLVE_TRI,
    HTP_OP_L2_NORM,
    HTP_OP_GATED_DELTA_NET,
    HTP_OP_TRI,
    HTP_OP_PAD,
    HTP_OP_NORM,
    HTP_OP_CONCAT,
    HTP_OP_CLAMP,
    HTP_OP_IM2COL,
    HTP_OP_XATTN_SCORE,
    HTP_OP_SYNC_PROBE,
    HTP_OP_XATTN_SELECT,
    HTP_OP_SEL,

    HTP_OP_INVALID
};

// HTP_OP_SEL: the existing selectors' selection rules (sel-ops.c), a GGML_OP_CUSTOM node that src/llama-selectors.h
// tags with LLAMA_SEL_MAGIC in op_params[6]; [7] mode, [8..11] the mode's parameters.
enum htp_sel_params {
    HTP_SEL_P_MAGIC = 6,
    HTP_SEL_P_MODE  = 7,
};
enum htp_sel_mode {
    HTP_SEL_MODE_VS     = 1,
    HTP_SEL_MODE_FLEX   = 2,
    HTP_SEL_MODE_BS     = 3,
    HTP_SEL_MODE_SPARGE = 4,
    HTP_SEL_MODE_FLASH  = 5,
    HTP_SEL_MODE_SAMPLE = 6,
};
#define HTP_SEL_MAGIC 0x534c4354

// HTP_OP_XATTN_SELECT: XAttention's cumulative block selection (argsort-ops.c), a GGML_OP_CUSTOM node that
// src/llama-xattention.h tags in op_params slots the custom-op header (function, n_tasks, userdata) leaves free.
enum htp_xattn_select_params {
    HTP_XATTN_SELECT_P_MAGIC     = 12,
    HTP_XATTN_SELECT_P_THRESHOLD = 13,
};
#define HTP_XATTN_SELECT_MAGIC 0x58534c54

// HTP_OP_SYNC_PROBE: dev-only flag round-trip probe (host gates it behind GGML_HEXAGON_SYNC_PROBE=1).
// The op raises a 'ready' word in src[0], spin-polls a 'done' word another agent (CPU or GPU) writes,
// checks a payload that agent wrote, and records qtimer stamps into dst. op_params layout:
enum htp_sync_probe_param {
    HTP_SYNC_PROBE_P_MAGIC = 0,       // HTP_SYNC_PROBE_MAGIC
    HTP_SYNC_PROBE_P_MODE,            // 0 handshake, 1 ping (stamp entry and return)
    HTP_SYNC_PROBE_P_TIMEOUT_US,      // give up polling after this (0 = never)
    HTP_SYNC_PROBE_P_PAYLOAD_WORDS,   // uint32 words to check at PAYLOAD_OFF (0 = none)
    HTP_SYNC_PROBE_P_INVAL,           // poll read: 0 dcinva, 1 qurt invalidate, 2 none, 3 dma (cache bypass)
    HTP_SYNC_PROBE_P_READY_OFF,       // byte offsets into src[0]
    HTP_SYNC_PROBE_P_DONE_OFF,
    HTP_SYNC_PROBE_P_PAYLOAD_OFF,
    HTP_SYNC_PROBE_P_SEED,            // payload word i must equal seed ^ i
    HTP_SYNC_PROBE_P_FLUSH,           // ready publish: 0 qurt flush, 1 dccleana + syncht
    HTP_SYNC_PROBE_P_READY_VAL,
    HTP_SYNC_PROBE_P_DONE_VAL,
    HTP_SYNC_PROBE_P_DSP_PAYLOAD_WORDS, // words the DSP writes (seed ^ 0xA5A5A5A5 ^ i) and flushes before ready
    HTP_SYNC_PROBE_P_DSP_PAYLOAD_OFF,
};

// dst record, uint64 words
enum htp_sync_probe_rec {
    HTP_SYNC_PROBE_R_T_ENTRY = 0,     // qtimer at op entry
    HTP_SYNC_PROBE_R_T_READY,         // qtimer after the ready word was published
    HTP_SYNC_PROBE_R_T_DONE,          // qtimer when the done word was observed (or timeout)
    HTP_SYNC_PROBE_R_T_CHECKED,       // qtimer after the payload check
    HTP_SYNC_PROBE_R_POLLS,           // poll iterations
    HTP_SYNC_PROBE_R_MISMATCH,        // payload words that did not match
    HTP_SYNC_PROBE_R_FIRST_BAD,       // index of the first mismatch (~0 if none)
    HTP_SYNC_PROBE_R_STATUS,          // 0 ok, 1 timeout
    HTP_SYNC_PROBE_R_DONE_RAW,        // last value read from the done word
    HTP_SYNC_PROBE_R_T_FIRST_POLL,    // qtimer of the first poll read
    HTP_SYNC_PROBE_R_FIRST_VAL,       // value of the first poll read
    HTP_SYNC_PROBE_R_T_DSP_PAYLOAD,   // qtimer after the DSP payload was written and flushed
    HTP_SYNC_PROBE_R_N
};

#define HTP_SYNC_PROBE_MAGIC 0x53594e43

// Heterogeneous decode attention (dev prototype, host env GGML_HEXAGON_HETERO_FRAC). The GPU owns the
// leading KV blocks of a decode FLASH_ATTN_EXT and hands (M, S, acc) partials to the HTP merge through
// a shared control buffer the host attaches as src[7]. The parameters ride in the node's op_params.
#define HTP_FA_HETERO_OPP_GPU_BLOCKS 8      // op_params[8]: leading 64-key blocks owned by the GPU (0 = off)
#define HTP_FA_HETERO_OPP_SLOT       9      // op_params[9]: control slot (per FA node in the graph)
#define HTP_FA_HETERO_OPP_GPU_NSPLIT 10     // op_params[10]: GPU partials per row
// control buffer layout (bytes)
#define HTP_FA_HETERO_SLOT_STRIDE    256            // slot s: ready_seq @ s*256 (DSP writes), done_seq @ s*256+128 (GPU writes)
#define HTP_FA_HETERO_STATUS_OFF     (96 * 1024)    // [0] DSP-side done timeouts
#define HTP_FA_HETERO_PARTS_OFF      (128 * 1024)   // partials: slot * HTP_FA_HETERO_PART_SLOT, [row][gpu_split] x part stride
#define HTP_FA_HETERO_PART_SLOT      (256 * 1024)
#define HTP_FA_HETERO_MAX_SLOTS      30
#define HTP_FA_HETERO_BUF_SIZE       (8 * 1024 * 1024)
#define HTP_FA_HETERO_DONE_TIMEOUT_US 50000
// Cluster GPU_RUNS hand-off inside the same control region: per KV head the blocks the GPU attends
// ({ uint32 n; pad to 128 B; htp_fa_cluster_blk blk[MAX_BLK] }, DSP-written before ready) and the
// work-group counter the GPU's last group uses to publish done (never reset: it reads want * n_groups).
#define HTP_FA_HETERO_SEL_OFF        4096
#define HTP_FA_HETERO_SEL_MAX_BLK    64
#define HTP_FA_HETERO_SEL_STRIDE     (128 + 16 * HTP_FA_HETERO_SEL_MAX_BLK)
#define HTP_FA_HETERO_SLOT_COUNTER   132            // slot s: s * HTP_FA_HETERO_SLOT_STRIDE + 132, the done line
#define HTP_FA_HETERO_SLOT_LIST      140            // GPU_SELECT: the GPU wrote the records (== want), same line

// Heterogeneous PREFILL fold (dev prototype; host env GGML_HEXAGON_FA_FOLD).
//
// A second engine computes some of the KV blocks of a PREFILL FLASH_ATTN_EXT and leaves one
// unnormalised online-softmax partial per (sequence, head, token) row. The HMX kernel folds that
// partial into its FINAL NORMALIZATION instead of running a separate merge pass: the normalization
// is already an HMX multiply by diag(1/l), so re-aiming that diagonal at w_htp/S_total costs
// nothing, and the only new work is one scaled add of the other engine's accumulator while the
// store thread de-tiles. The HTP's own (m, l, O) never leaves VTCM.
//
// The buffer describes itself -- op_params 0..15 are all taken -- and rides in src[7]. A magic
// that does not match leaves the op bit-identical to the non-folded kernel.
#define HTP_FA_FOLD_MAGIC     0x464f4c44  // 'FOLD' at word 0 of src[7]; anything else = off
#define HTP_FA_FOLD_MAX_G_BR  1024        // rows per tile the prototype accepts (stack scratch)

// hdr.flags. Zero means the partial is ALREADY RESIDENT when the op starts (a host-written
// partial, e.g. examples/fa-fold-check): no handshake, no waiting, bit-identical to before.
#define HTP_FA_FOLD_F_LIVE      (1u << 0)  // a producer fills the partial DURING this op: publish ready, wait for done
#define HTP_FA_FOLD_F_FLUSH_KV  (1u << 1)  // flush K and V with Q before ready (the producer also reads the KV cache)
// Dev measurement: instead of folding the other engine's partial into the final normalization,
// write the HTP's OWN unnormalised (m, l, acc) to the off_h* regions and merge the two partials
// into dst in a SEPARATE pass over DDR. Exists to price the explicit merge against the fold;
// it is strictly more traffic (the fold keeps the HTP side in VTCM) and is not a deployment mode.
#define HTP_FA_FOLD_F_SPILL     (1u << 2)
// With SPILL: skip the merge pass (dst is left unwritten). Timing-only, to price the two halves
// of SPILL separately.
#define HTP_FA_FOLD_F_NOMERGE   (1u << 3)
// STAGED (requires LIVE, neq3 == 1, <= HTP_FA_FOLD_MAX_STAGES KV heads): the producer delivers the
// partial one KV head at a time in ascending head order and echoes the sequence into done word k
// (slot line + 128 + 4k) as head k completes. The op then (1) skips the fold and the wait on tiles
// whose 64-token sub-blocks have no exceptions, (2) runs those tiles FIRST so the producer works in
// their shadow, and (3) takes one wait per KV head, at its first exception tile. hdr.off_exc names
// a uint32 per (KV head, sub-block), row-major [n_kv_heads][ceil(neq1 / 64)], nonzero = that
// sub-block has exceptions -- or, with hdr.exc_nbk != 0, the graph's own membership rows
// (sparse_exc_mem: exc_nbk f32 per (head, sub-block), same row order, nonzero = member), which
// the producer reads too, so nothing has to be packed for either engine. hdr.off_exc == 0 means
// the membership is the op's src[8] itself (F32 [NBk, R, NBq/R, n_kv_heads], the graph's
// sparse_exc_mem), which the op also FLUSHES before ready so the producer can read it.
#define HTP_FA_FOLD_F_STAGED    (1u << 4)
// INSTORE: fold in the store threads instead of the HMX normalisation. The tile leaves the KV loop
// as the raw accumulator, the store threads form the two merge weights per 32-row group from the
// VTCM (m, l) and the partial's (m, l), and emit w_htp * acc_htp + w_other * acc_other while
// de-tiling. No diagonal build, no norm pass; each thread prefetches its slice of the partial with
// one 2D l2fetch. Not with SPILL.
#define HTP_FA_FOLD_F_INSTORE   (1u << 5)
// NOFLUSH (with LIVE): do not flush Q / K / V / mask / membership before ready. The op's own input
// flush at op start (htp_tensor_flush_all, whole-L2 above 4 MB dirty) already put every DSP-written
// input in DDR, so the explicit line loops here -- ~32 MB per op -- were redundant and cost ~0.5 ms.
#define HTP_FA_FOLD_F_NOFLUSH   (1u << 6)
// STAGE_QB (with STAGED): the producer delivers the partial one QUERY BLOCK (Br rows, all heads) at
// a time, done word k = query block k, and the op keeps its default query-block-outer order, which
// keeps the mask DMA cache's reuse across the KV heads of a query block (KV-head-outer staging
// refetches every mask rectangle per head, ~12 MB per op instead of ~1.6). <= 32 query blocks.
#define HTP_FA_FOLD_F_STAGE_QB  (1u << 7)
// Dev probes for INSTORE timing (results may be WRONG with PROBE_NOWEIGHTS): where the cost sits.
#define HTP_FA_FOLD_F_PROBE_NOPF      (1u << 8)   // no per-thread prefetch of the partial
#define HTP_FA_FOLD_F_PROBE_TILEPF    (1u << 9)   // whole-tile prefetch from the main thread at the fold site instead
#define HTP_FA_FOLD_F_PROBE_NOWEIGHTS (1u << 10)  // skip the (m, l) gather and weights: constant 0.5/0.5
#define HTP_FA_FOLD_F_PROBE_NOACC     (1u << 11)  // skip the partial's accumulator read: normalise only
#define HTP_FA_FOLD_F_PROBE_NOINVAL   (1u << 13)  // skip the per-stage L2 invalidation of the partial's rows (timing only)
#define HTP_FA_FOLD_F_PROBE_INVAL     (1u << 14)  // force the per-stage L2 invalidation even when the epoch says it is redundant
// ACC_F16 (with INSTORE, not SPILL): the producer writes the partial's accumulator rows as f16
// (row stride dv * 2). Halves the 7 MB per op the fold reads; the HTP's own accumulator is f16.
#define HTP_FA_FOLD_F_ACC_F16         (1u << 15)
// STAGE_QBH (with STAGED, implies STAGE_QB ordering): one stage per (query block, KV head), stage
// k = qb * n_kv_heads + kv_head, so the first exception tile of a query block waits for one KV
// head's partial rather than the whole block's. Needs hdr.off_done (the slot line holds 32 words).
#define HTP_FA_FOLD_F_STAGE_QBH (1u << 12)
#define HTP_FA_FOLD_EXC_SB      64
#define HTP_FA_FOLD_MAX_STAGES  64          // with hdr.off_done; the legacy slot line holds HTP_FA_FOLD_LINE_STAGES
#define HTP_FA_FOLD_LINE_STAGES 32

// Handshake control region at hdr.off_ctl, laid out like the hetero DECODE control region so the
// host drives both with one helper:
//   slot s: ready_seq @ s * HTP_FA_HETERO_SLOT_STRIDE        (DSP writes, producer polls)
//           wait_us   @ s * HTP_FA_HETERO_SLOT_STRIDE + 8    (DSP writes: us this op idled)
//           done_seq  @ s * HTP_FA_HETERO_SLOT_STRIDE + 128  (producer writes; == ready_seq means the partial is complete)
//   status  @ HTP_FA_HETERO_STATUS_OFF, uint32 words below
// The region must therefore be at least HTP_FA_HETERO_STATUS_OFF + 32 bytes long.
#define HTP_FA_FOLD_MAX_SLOTS       HTP_FA_HETERO_MAX_SLOTS
#define HTP_FA_FOLD_DONE_TIMEOUT_US 50000

enum htp_fa_fold_status_word {
    HTP_FA_FOLD_ST_TIMEOUTS = 0,   // ops that gave up on the producer (each one FAILED the op)
    HTP_FA_FOLD_ST_WAITS,          // ops that waited at all
    HTP_FA_FOLD_ST_DONE_SEEN,      // last value read from the done word
    HTP_FA_FOLD_ST_SEQ_WANTED,     // what it wanted
    HTP_FA_FOLD_ST_SLOT,
    HTP_FA_FOLD_ST_WAIT_US,        // last wait (STAGED: this op's blocked time summed over its stages)
    HTP_FA_FOLD_ST_WAIT_US_MAX,
    HTP_FA_FOLD_ST_TILES_BLOCKED,  // STAGED: stage waits that actually spun, summed over ops
    HTP_FA_FOLD_ST_WAIT_US_SUM,    // STAGED: blocked time summed over ops (mean = / ops)
    HTP_FA_FOLD_ST_N
};

// src[7] holds THREE regions, not one record per row. m and l are read one value per row while
// the diagonal is built, so interleaving them with acc made every gather touch a 640 B line for
// 8 bytes of payload; split out, a tile's m and l are 1 KB of contiguous stream per GQA head.
// All offsets are 128-aligned byte offsets from the header. Row r of (sequence ib3, head iq2,
// token iq1) is r = (ib3 * neq2 + iq2) * neq1 + iq1, and holds:
//   m[r]        f32, the running max in NATURAL log units (-INFINITY or very negative = empty)
//   l[r]        f32, the running sum
//   acc[r][dv]  f32, the UNNORMALISED accumulator
struct htp_fa_fold_hdr {
    uint32_t magic;
    uint32_t rows;      // neq1 * neq2 * neq3
    uint32_t neq1;
    uint32_t dv;
    uint32_t off_m;
    uint32_t off_l;
    uint32_t off_acc;   // row stride is dv * sizeof(float)
    uint32_t flags;     // HTP_FA_FOLD_F_*
    uint32_t slot;      // handshake slot, < HTP_FA_FOLD_MAX_SLOTS (LIVE only)
    uint32_t off_ctl;   // handshake control region, 128-aligned bytes from the header, >= 256 (LIVE only)
    uint32_t timeout_us; // done-word deadline; 0 = HTP_FA_FOLD_DONE_TIMEOUT_US
    uint32_t off_hm;     // SPILL only: the HTP's own partial, same shapes as off_m / off_l / off_acc
    uint32_t off_hl;
    uint32_t off_hacc;
    uint32_t off_exc;    // STAGED only: per (KV head, 64-token sub-block) exception table, 128-aligned
    uint32_t exc_nbk;    // STAGED only: 0 = one uint32 flag per (head, sub-block); else exc_nbk f32 per
                         // (head, sub-block), the graph's sparse_exc_mem rows, nonzero = member
    uint32_t off_done;   // STAGED only: done words (HTP_FA_FOLD_MAX_STAGES uint32, 128-aligned); 0 = slot line + 128
};

// Cluster-selected sparse decode attention (dev prototype; host env GGML_HEXAGON_CLUSTER_ATTN).
//
// The backend owns a "shadow" copy of the KV cache in cluster order: per layer and KV head the
// keys are packed into 64-key pages (16 KB per page per tensor), each page described by the f16
// mean of its keys. A decode FLASH_ATTN_EXT tagged with these op_params (static per node) and the
// shadow buffer as src[7] attends over a per-head LIST of pages plus a dense tail of the positional
// cache [dense_start, n_kv) with the normal mask. Everything that changes per token is DATA in the
// shadow buffer (the host graph cache makes op_params static): the layer directory publishes how
// far the clustering reaches (covered_end), and the kernel derives the dense tail from the chunk
// table so no key is attended twice. covered_end == 0 reproduces the dense kernel exactly.
#define HTP_FA_CLUSTER_OPP_MAGIC    11     // op_params[11]: HTP_FA_CLUSTER_MAGIC enables the mode
#define HTP_FA_CLUSTER_OPP_LAYER    12     // op_params[12]: layer index (directory / per-layer region)
#define HTP_FA_CLUSTER_OPP_DENSITY  13     // op_params[13]: page budget, permille of the candidates; 0 = host-written lists
#define HTP_FA_CLUSTER_OPP_WINDOW   14     // op_params[14]: recent window W (keys) always attended densely
#define HTP_FA_CLUSTER_OPP_FLAGS    15     // op_params[15]: HTP_FA_CLUSTER_FLAG_*
#define HTP_FA_CLUSTER_MAGIC        0x434c4b56u
#define HTP_FA_CLUSTER_VERSION      2u
#define HTP_FA_CLUSTER_FLAG_DESC1D      (1u << 0)              // fetch shadow pages with one 1D descriptor (else 64 x 256 B rows)
#define HTP_FA_CLUSTER_FLAG_NSLOTS(f)   (((f) >> 1) & 7u)      // log2 of the staging slots per thread; 0 = default (2)
#define HTP_FA_CLUSTER_FLAG_ECHO        (1u << 4)              // write the lists the kernel used into echo_sel
#define HTP_FA_CLUSTER_FLAG_COALESCE    (1u << 5)              // merge consecutive pages into one descriptor (reserved)
#define HTP_FA_CLUSTER_FLAG_FOLD_SUM    (1u << 6)              // GQA score fold: sum over the group (else max)
#define HTP_FA_CLUSTER_FLAG_POLLFIRST   (1u << 7)              // hetero timing: poll the GPU's done right after ready, before the dec pass
#define HTP_FA_CLUSTER_FLAG_MINPAGES(f) (((f) >> 8) & 0xffu)   // minimum pages per head when a budget is used
#define HTP_FA_CLUSTER_FLAG_FORCE(f)    (((f) >> 16) & 0xffu)  // always select the first F candidate pages of the layer (sink pages)
#define HTP_FA_CLUSTER_FLAG_KEEP(f)     (((f) >> 24) & 0xffu)  // GPU_RUNS: percent of the selected blocks the HTP keeps (shadow rows); the rest go to the GPU
#define HTP_FA_CLUSTER_PAGE_KEYS    64     // default and maximum page size; 16 and 32 are also valid (header.page_keys)
#define HTP_FA_CLUSTER_HDR_INPLACE  (1u << 0)  // header.flags: pages are page_keys consecutive rows of the positional cache itself;
                                               // the shadow holds only descriptors (no K/V pages, pos_map or n_valid)
#define HTP_FA_CLUSTER_HDR_DSP_DESC (1u << 1)  // header.flags: the FA op computes the descriptors itself from hdir.rows_valid
                                               // (host writes hdir only; the DSP owns the directory)
#define HTP_FA_CLUSTER_HDR_HOST_ROWS (1u << 2) // header.flags: the kernel never uses pages beyond hdir.rows_valid (host-written before
                                               // each graph; a reset or rewind lowers it synchronously while the sidecar catches up)
#define HTP_FA_CLUSTER_HDR_RUNS     (1u << 3)  // header.flags: the selection unit is a whole variable-size cluster -- a run of rows in
                                               // cluster order (off_runs table); the kernel takes runs whole until a row budget
#define HTP_FA_CLUSTER_HDR_SCATTER  (1u << 4)  // header.flags (with RUNS): a run's rows are fetched from the positional cache at
                                               // pos_map[row], one linked 1D descriptor per K and V row; the shadow rows are unused
#define HTP_FA_CLUSTER_HDR_GPU_RUNS (1u << 5)  // header.flags (with RUNS): the GPU attends the selected runs from the positional cache
                                               // through pos_map. The op writes its per-head block lists (minus a KEEP share) into the
                                               // hetero control region at hdr.hetero_off (bytes from the header), publishes ready
                                               // (op_params[9] = slot, [10] = GPU partials per row), attends the dense tail and its
                                               // kept runs, waits for done and merges the GPU partials.
#define HTP_FA_CLUSTER_HDR_GPU_SELECT (1u << 6) // header.flags (with GPU_RUNS): the GPU also selects the runs (scores the centroids,
                                               // greedy fill to the row budget) and writes the records itself; the op runs no select
                                               // pass, keeps no runs, publishes ready right after setup and attends the dense tail.
#define HTP_FA_CLUSTER_HDR_POS_RUNS (1u << 7) // no K/V shadow: HTP takes positional runs of at least KEEP rows (0 = GPU all)
#define HTP_FA_CLUSTER_HDR_HOST_RUNS (1u << 8) // density 0: host_sel contains replay blocks, not page indices
#define HTP_FA_CLUSTER_HDR_PREPARED (1u << 9) // position-sorted members and routing spans prepared per cluster
#define HTP_FA_CLUSTER_HDR_GPU_ROWS (1u << 10) // KEEP limits GPU rows per head in units of 64
#define HTP_FA_CLUSTER_HDR_OVERLAP  (1u << 11) // publish GPU records after each head's selection
#define HTP_FA_CLUSTER_HDR_REORDER  (1u << 12) // K/V prefix is physically in cluster order; no shadow pages
#define HTP_FA_CLUSTER_SPAN_HTP     (1u << 31)
#define HTP_FA_CLUSTER_RUN_FORCED   (1u << 0)  // run.flags: always attended (the sink tokens), not charged to the budget
#define HTP_FA_CLUSTER_MAX_LAYERS   128

// One cluster of one KV head (runs mode): rows [row_first, row_first + n_rows) of the head's shadow rows.
struct htp_fa_cluster_run {
    uint32_t row_first;
    uint16_t n_rows;
    uint16_t flags;
};

// One DMA block of the kernel's per-head list in runs mode (also the echo format): rows [row, row + bsz) of the
// head's shadow and, when bsz2 != 0, a second run's rows [row2, row2 + bsz2) packed into the same 64-row block.
struct htp_fa_cluster_blk {
    uint32_t row;
    uint32_t row2;
    uint16_t bsz;
    uint16_t bsz2;
    uint32_t pad;
};
#define HTP_FA_CLUSTER_MAX_CHUNKS   256
#define HTP_FA_CLUSTER_MAX_HEADS    64
#define HTP_FA_CLUSTER_MAX_SLOTS    8      // staging ring depth cap (per thread)
#define HTP_FA_CLUSTER_DIR_OFF      4096
#define HTP_FA_CLUSTER_DIR_STRIDE   128

// Shadow buffer, byte 0. Written once by the host; the DSP invalidates before reading.
struct htp_fa_cluster_header {
    uint32_t magic, version, n_layers, kv_size;
    uint32_t n_kv_heads, D, page_keys, page_bytes;      // page_bytes = page_keys * D * 2
    uint32_t n_pages_max, chunk_keys, max_chunks, flags;
    uint32_t centroid_bytes, host_sel_stride, avg_cluster, n_runs_max;   // runs mode: average cluster size, run-table entries per head
    uint64_t layer0_off, layer_stride, hetero_off;      // hetero_off: 0 = no hetero control region
    // per-layer sub-region offsets, bytes from the layer base (all 128-B aligned, pages 16 KB aligned)
    uint64_t off_chunks;      // [max_chunks] x struct htp_fa_cluster_chunk
    uint64_t off_host_sel;    // [n_kv_heads] x host_sel_stride: { uint32_t n; uint32_t pad[31]; uint16_t pages[]; }
    uint64_t off_echo_sel;    // same shape, written by the kernel under HTP_FA_CLUSTER_FLAG_ECHO
    uint64_t off_centroids;   // [n_kv_heads][n_pages_max] x f16[D] (centroid_bytes rows)
    uint64_t off_pos_map;     // [n_kv_heads][n_pages_max] x uint32_t[page_keys]: positional row of each key
    uint64_t off_n_valid;     // reserved; POS_RUNS: GPU records [count, HTP rows, HTP blocks, indexed], stride 128 + rows_max * 4
    uint64_t off_k_pages;     // [n_kv_heads][n_pages_max] x [page_keys][D] f16  (runs mode: [n_kv_heads][kv_size] rows in cluster order)
    uint64_t off_v_pages;     // same
    uint64_t off_runs;        // runs mode: [n_kv_heads][n_runs_max] x struct htp_fa_cluster_run (centroids are per run in that mode)
};

// Per-layer directory entry @ HTP_FA_CLUSTER_DIR_OFF + il * HTP_FA_CLUSTER_DIR_STRIDE.
struct htp_fa_cluster_dir {
    uint32_t covered_end;     // positions [0, covered_end) are clustered; a multiple of 64; written LAST
    uint32_t n_pages_pub;     // pages readable (immutable once published)
    uint32_t n_chunks;
    uint32_t stale;           // 1: the layer was rewritten below covered_end -> kernel runs dense
    uint32_t t_gpu_us, t_publish_us;
    uint32_t pad[26];
};

// Host-written per-layer entry @ HTP_FA_CLUSTER_HDIR_OFF + il * HTP_FA_CLUSTER_DIR_STRIDE (DSP-descriptor mode).
// Its own 128-B line: the host writes it before a graph is submitted, the DSP only reads it.
#define HTP_FA_CLUSTER_HDIR_OFF     (HTP_FA_CLUSTER_DIR_OFF + HTP_FA_CLUSTER_MAX_LAYERS * HTP_FA_CLUSTER_DIR_STRIDE)
struct htp_fa_cluster_hdir {
    uint32_t rows_valid;      // rows [0, rows_valid) of the positional cache are complete and in the sequence
    uint32_t pad[31];
};

struct htp_fa_cluster_chunk {
    uint32_t pos_begin, pos_end;   // positional range this chunk clustered; pos_end % 64 == 0
    uint32_t page_first, n_pages;  // its pages, per KV head
};

// Single source of truth for the shadow layout (host backend, tools, and the DSP read the same
// offsets from the header this fills). Returns the buffer size in bytes. Sub-regions are 128-B
// aligned, page regions 16 KB aligned, layers 64 KB aligned.
static inline uint64_t htp_fa_cluster_align(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

static inline uint64_t htp_fa_cluster_meta_offset(const struct htp_fa_cluster_header * h) {
    return h->off_n_valid + ((h->flags & HTP_FA_CLUSTER_HDR_POS_RUNS)
        ? (uint64_t) h->n_kv_heads * (128 + (uint64_t) h->n_pages_max * h->page_keys * 4)
        : htp_fa_cluster_align((uint64_t) h->n_pages_max * 2, 128));
}

// Metadata starts with the minimum run length in a 128-byte header, then [head][positions, span ends].
// Positions are sorted within each whole cluster. Adjacent short runs share one GPU span.
static inline void htp_fa_cluster_prepare_spans(const uint32_t * pos, uint32_t * spans, uint32_t first, uint32_t count, uint32_t min_run) {
    const uint32_t end = first + count;
    uint32_t pending = first;
    for (uint32_t i = first; i < end;) {
        uint32_t j = i + 1;
        while (j < end && pos[j] == pos[j - 1] + 1) j++;
        if (min_run && j - i >= min_run) {
            for (; pending < i; ++pending) spans[pending] = i;
            for (; pending < j; ++pending) spans[pending] = j | HTP_FA_CLUSTER_SPAN_HTP;
        }
        i = j;
    }
    for (; pending < end; ++pending) spans[pending] = end;
}

static inline uint64_t htp_fa_cluster_layout_v2(struct htp_fa_cluster_header * h, uint32_t n_layers, uint32_t kv_size,
                                                uint32_t n_kv_heads, uint32_t D, uint32_t page_keys, uint32_t avg_cluster, uint32_t hdr_flags) {
    memset(h, 0, sizeof(*h));
    const int inplace = (hdr_flags & HTP_FA_CLUSTER_HDR_INPLACE) != 0;
    const int runs    = (hdr_flags & HTP_FA_CLUSTER_HDR_RUNS) != 0;
    const int pos_runs = runs && (hdr_flags & HTP_FA_CLUSTER_HDR_POS_RUNS) != 0;
    const int no_pages = inplace || pos_runs || (hdr_flags & HTP_FA_CLUSTER_HDR_REORDER);
    h->flags       = hdr_flags;
    if (runs) {
        page_keys = HTP_FA_CLUSTER_PAGE_KEYS;   // rows are addressed individually; 64 keeps the row region 16 KB aligned
    }
    if (page_keys != 16 && page_keys != 32 && page_keys != 64) {
        page_keys = HTP_FA_CLUSTER_PAGE_KEYS;
    }
    if (avg_cluster < 4) {
        avg_cluster = 4;
    }
    h->magic       = HTP_FA_CLUSTER_MAGIC;
    h->version     = HTP_FA_CLUSTER_VERSION;
    h->n_layers    = n_layers;
    h->kv_size     = kv_size;
    h->n_kv_heads  = n_kv_heads;
    h->D           = D;
    h->page_keys   = page_keys;
    h->page_bytes  = page_keys * D * 2;
    h->n_pages_max = (kv_size + page_keys - 1) / page_keys;
    h->chunk_keys  = 1024;
    h->max_chunks  = HTP_FA_CLUSTER_MAX_CHUNKS;
    h->avg_cluster = runs ? avg_cluster : page_keys;
    // runs: one entry per cluster plus slack for the forced sink run and a short tail run per chunk; multiple of 16 keeps
    // every head's table 128-B aligned
    h->n_runs_max  = runs ? (uint32_t) htp_fa_cluster_align((kv_size + avg_cluster - 1) / avg_cluster + 2ull * HTP_FA_CLUSTER_MAX_CHUNKS, 16) : 0;
    const uint64_t n_units = runs ? h->n_runs_max : h->n_pages_max;   // descriptor rows per head
    h->centroid_bytes  = (uint32_t) htp_fa_cluster_align((uint64_t) D * 2, 128);
    h->host_sel_stride = (uint32_t) htp_fa_cluster_align(128 + (runs ? (uint64_t) sizeof(struct htp_fa_cluster_blk) * (h->n_pages_max + h->n_runs_max) : 2ull * h->n_pages_max), 128);
    uint64_t off = 0;
    h->off_chunks    = off;                            off += (uint64_t) h->max_chunks * sizeof(struct htp_fa_cluster_chunk);
    h->off_host_sel  = htp_fa_cluster_align(off, 128); off  = h->off_host_sel  + (uint64_t) n_kv_heads * h->host_sel_stride;
    h->off_echo_sel  = htp_fa_cluster_align(off, 128); off  = h->off_echo_sel  + (uint64_t) n_kv_heads * h->host_sel_stride;
    h->off_centroids = htp_fa_cluster_align(off, 128); off  = h->off_centroids + (uint64_t) n_kv_heads * n_units * h->centroid_bytes;
    h->off_pos_map   = htp_fa_cluster_align(off, 128); off  = h->off_pos_map   + (inplace ? 0 : (uint64_t) n_kv_heads * h->n_pages_max * page_keys * 4);
    h->off_n_valid   = htp_fa_cluster_align(off, 128); off  = h->off_n_valid + (pos_runs ? (uint64_t) n_kv_heads * (128 + (uint64_t) h->n_pages_max * page_keys * 4) : inplace ? 0 : htp_fa_cluster_align((uint64_t) h->n_pages_max * 2, 128));
    if (runs && (hdr_flags & HTP_FA_CLUSTER_HDR_PREPARED)) off += 128 + (uint64_t) n_kv_heads * h->n_pages_max * page_keys * 8;
    h->off_k_pages   = htp_fa_cluster_align(off, no_pages ? 128 : 16384); off = h->off_k_pages + (no_pages ? 0 : (uint64_t) n_kv_heads * h->n_pages_max * h->page_bytes);
    h->off_v_pages   = htp_fa_cluster_align(off, no_pages ? 128 : 16384); off = h->off_v_pages + (no_pages ? 0 : (uint64_t) n_kv_heads * h->n_pages_max * h->page_bytes);
    h->off_runs      = htp_fa_cluster_align(off, 128); off  = h->off_runs + (runs ? (uint64_t) n_kv_heads * h->n_runs_max * sizeof(struct htp_fa_cluster_run) : 0);
    h->layer_stride  = htp_fa_cluster_align(off, 65536);
    h->layer0_off    = htp_fa_cluster_align(HTP_FA_CLUSTER_DIR_OFF + (uint64_t) n_layers * HTP_FA_CLUSTER_DIR_STRIDE, 65536);
    return h->layer0_off + (uint64_t) n_layers * h->layer_stride;
}

static inline uint64_t htp_fa_cluster_layout_ex(struct htp_fa_cluster_header * h, uint32_t n_layers, uint32_t kv_size,
                                                uint32_t n_kv_heads, uint32_t D, uint32_t page_keys, uint32_t hdr_flags) {
    return htp_fa_cluster_layout_v2(h, n_layers, kv_size, n_kv_heads, D, page_keys, page_keys, hdr_flags);
}

static inline uint64_t htp_fa_cluster_layout(struct htp_fa_cluster_header * h, uint32_t n_layers, uint32_t kv_size,
                                             uint32_t n_kv_heads, uint32_t D, uint32_t page_keys) {
    return htp_fa_cluster_layout_ex(h, n_layers, kv_size, n_kv_heads, D, page_keys, 0);
}

#define HTP_OP_MAX_DIMS    4    // aka GGML_MAX_DIMS
#define HTP_OP_MAX_INPUTS  12   // sparse flash-attention carries sel (src 5) and its per-row count (src 6); hetero FA carries its
                                // control / fold buffer (src 7) and the prefill split its exception membership (src 8);
                                // 12 + 4 output slots keep htp_op_desc a multiple of 8 bytes
#define HTP_OP_MAX_OUTPUTS 4
#define HTP_OP_MAX_PARAMS  16   // aka GGML_MAX_OP_PARAMS
#define HTP_OP_MAX_KERN_PARAMS 32

#define HTP_OP_MAX_BUFS    16
#define HTP_OP_MAX_TENSORS 8192 // must stay under 64K (uint16)

#define HTP_OP_MAX_VMEM_DEFAULT (3355443200u)

#define HTP_MMAP_MAX_VMEM  (2147483648u)

enum htp_tensor_flags {
    HTP_TENSOR_COMPUTE = (1U << 0), // Tensor buffer temporal compute data (not weights)
    HTP_TENSOR_DIRTY   = (1U << 1)  // Tensor buffer is dirty and needs to be flushed
};

// Tensor descriptor
struct htp_tensor {
    uint32_t data;                 // Buffer offset in the messages, and data pointer on the NPU
    uint32_t reserved;             // Reserved for alignment padding (must be multiple of 8)
    uint32_t size;                 // Data size in bytes
    uint32_t flags;                // Buffer / tensor flags
    uint32_t type;                 // Data type
    uint16_t bi;                   // Buffer index
    uint16_t ti;                   // Tensor index
    uint32_t ne[HTP_OP_MAX_DIMS];  // Number of elements
    uint32_t nb[HTP_OP_MAX_DIMS];  // Stride in bytes (see ggml.h ggml_tensor)
};

// Buffer descriptor
struct htp_buf_desc {
    uint64_t base;     // base address
    uint64_t size;     // total size
    uint32_t flags;    // buffer flags (unused)
    uint32_t fd;       // file descriptor
};

enum htp_op_flags {
    HTP_OPFLAGS_SKIP_COMPUTE  = (1U << 0), // Skip actual computation (used for profiling)
};

// Op descriptor
struct htp_op_desc {
    uint32_t opcode;                    // GGML/HTP Op
    uint32_t flags;                     // Op flags
    int32_t  params[HTP_OP_MAX_PARAMS]; // Params for the op, e.g. epsilon of RMS norm
    int32_t  kernel_params[HTP_OP_MAX_KERN_PARAMS]; // generic blob for host-precomputed parameters
    uint16_t src[HTP_OP_MAX_INPUTS];    // Input tensors indices
    uint16_t dst[HTP_OP_MAX_OUTPUTS];   // Output tensor indices (12 + 4 uint16 = 32 B, 64-bit aligned)
};

#ifndef HTP_MAX_NTHREADS
#define HTP_MAX_NTHREADS 10
#endif

#define HTP_TRACE_MAX_EVENTS 256

enum htp_profiler_mode {
    HTP_PROF_DISABLED = 0,
    HTP_PROF_BASIC    = 1,
    HTP_PROF_PMU      = 2,
    HTP_PROF_TRACE    = 3,
};

enum htp_trace_event_id {
    HTP_TRACE_EVT_DMA                 = 0,
    HTP_TRACE_EVT_L2FLUSH             = 1,
    HTP_TRACE_EVT_INIT                = 2,
    HTP_TRACE_EVT_BUFF                = 3,

    HTP_TRACE_EVT_HVX_COMP            = 20,
    HTP_TRACE_EVT_HVX_A_QUANT         = 21,
    HTP_TRACE_EVT_HVX_A_PREP          = 22,
    HTP_TRACE_EVT_HVX_W_DEQUANT       = 23,
    HTP_TRACE_EVT_HVX_W_PREP          = 24,
    HTP_TRACE_EVT_HVX_O_PROC          = 25,
    HTP_TRACE_EVT_HVX_FA_QK           = 26,
    HTP_TRACE_EVT_HVX_FA_SFM          = 27,
    HTP_TRACE_EVT_HVX_FA_Q_PREP       = 28,
    HTP_TRACE_EVT_HVX_FA_K_PREP       = 29,
    HTP_TRACE_EVT_HVX_FA_V_PREP       = 30,

    HTP_TRACE_EVT_HMX_COMP            = 40,
};

struct htp_trace_desc {
    uint32_t cycles;  // lower 32-bits of cycle counter
    uint16_t id;      // Event ID
    uint16_t info;    // bit 15: is_stop. bits 14-0: tile/chunk index or other metadata.
};

#define HTP_PROF_PMU_NCNT 8

// Profile descriptor
struct htp_prof_desc {
    uint32_t opcode;                 // GGML/HTP Op
    uint32_t usecs;                  // Number of usec
    uint32_t cycles_start;           // Start cycle counter
    uint32_t cycles_stop;            // Stop cycle counter
    uint32_t pmu[HTP_PROF_PMU_NCNT]; // PMU counters
};

struct htp_opbatch_req {
    uint32_t id;          // Batch id
    uint32_t n_bufs;      // Number of buffers
    uint32_t n_tensors;   // Number of tensors
    uint32_t n_ops;       // Number of ops
    uint32_t n_traces;    // Number of trace descriptors per thread
    uint32_t pad;         // unused
    // struct htp_buf_desc  bufs[];    -- dspqueue buf 0
    // struct htp_tensor    tensors[]; -- dspqueue buf 0
    // struct htp_op_desc   ops[];     -- dspqueue buf 0
};

struct htp_opbatch_rsp {
    uint32_t id;         // Batch id
    uint32_t status;     // HTP_STATUS_...
    uint32_t n_bufs;     // Number of buffers
    uint32_t n_tensors;  // Number of tensors
    uint32_t n_ops;      // Number of op profile descriptors
    uint32_t n_traces[HTP_MAX_NTHREADS + 1];
    uint32_t usecs;          // Number of usec
    uint32_t pad;            // align to 8 bytes
    uint64_t cycles_start;   // Start cycle counter
    uint64_t cycles_stop;    // Stop cycle counter
    // struct htp_prof_desc profs[];  -- dspqueue buf 0
};

#endif /* HTP_OPS_H */

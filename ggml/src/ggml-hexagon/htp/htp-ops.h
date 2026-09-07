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

    HTP_OP_INVALID
};

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
#define HTP_FA_CLUSTER_VERSION      1u
#define HTP_FA_CLUSTER_FLAG_DESC1D      (1u << 0)              // fetch shadow pages with one 1D descriptor (else 64 x 256 B rows)
#define HTP_FA_CLUSTER_FLAG_NSLOTS(f)   (((f) >> 1) & 7u)      // log2 of the staging slots per thread; 0 = default (2)
#define HTP_FA_CLUSTER_FLAG_ECHO        (1u << 4)              // write the lists the kernel used into echo_sel
#define HTP_FA_CLUSTER_FLAG_COALESCE    (1u << 5)              // merge consecutive pages into one descriptor (reserved)
#define HTP_FA_CLUSTER_FLAG_FOLD_SUM    (1u << 6)              // GQA score fold: sum over the group (else max)
#define HTP_FA_CLUSTER_FLAG_MINPAGES(f) (((f) >> 8) & 0xffu)   // minimum pages per head when a budget is used
#define HTP_FA_CLUSTER_FLAG_FORCE(f)    (((f) >> 16) & 0xffu)  // always select the first F candidate pages of the layer (sink pages)
#define HTP_FA_CLUSTER_PAGE_KEYS    64     // default and maximum page size; 16 and 32 are also valid (header.page_keys)
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
    uint32_t centroid_bytes, host_sel_stride, pad0, pad1;
    uint64_t layer0_off, layer_stride, hetero_off;      // hetero_off: 0 = no hetero control region
    // per-layer sub-region offsets, bytes from the layer base (all 128-B aligned, pages 16 KB aligned)
    uint64_t off_chunks;      // [max_chunks] x struct htp_fa_cluster_chunk
    uint64_t off_host_sel;    // [n_kv_heads] x host_sel_stride: { uint32_t n; uint32_t pad[31]; uint16_t pages[]; }
    uint64_t off_echo_sel;    // same shape, written by the kernel under HTP_FA_CLUSTER_FLAG_ECHO
    uint64_t off_centroids;   // [n_kv_heads][n_pages_max] x f16[D] (centroid_bytes rows)
    uint64_t off_pos_map;     // [n_kv_heads][n_pages_max] x uint32_t[page_keys]: positional row of each key
    uint64_t off_n_valid;     // [n_pages_max] x uint16_t (reserved: pages are full in v1)
    uint64_t off_k_pages;     // [n_kv_heads][n_pages_max] x [page_keys][D] f16
    uint64_t off_v_pages;     // same
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

struct htp_fa_cluster_chunk {
    uint32_t pos_begin, pos_end;   // positional range this chunk clustered; pos_end % 64 == 0
    uint32_t page_first, n_pages;  // its pages, per KV head
};

// Single source of truth for the shadow layout (host backend, tools, and the DSP read the same
// offsets from the header this fills). Returns the buffer size in bytes. Sub-regions are 128-B
// aligned, page regions 16 KB aligned, layers 64 KB aligned.
static inline uint64_t htp_fa_cluster_align(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

static inline uint64_t htp_fa_cluster_layout(struct htp_fa_cluster_header * h, uint32_t n_layers, uint32_t kv_size,
                                             uint32_t n_kv_heads, uint32_t D, uint32_t page_keys) {
    memset(h, 0, sizeof(*h));
    if (page_keys != 16 && page_keys != 32 && page_keys != 64) {
        page_keys = HTP_FA_CLUSTER_PAGE_KEYS;
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
    h->centroid_bytes  = (uint32_t) htp_fa_cluster_align((uint64_t) D * 2, 128);
    h->host_sel_stride = (uint32_t) htp_fa_cluster_align(128 + 2ull * h->n_pages_max, 128);
    uint64_t off = 0;
    h->off_chunks    = off;                            off += (uint64_t) h->max_chunks * sizeof(struct htp_fa_cluster_chunk);
    h->off_host_sel  = htp_fa_cluster_align(off, 128); off  = h->off_host_sel  + (uint64_t) n_kv_heads * h->host_sel_stride;
    h->off_echo_sel  = htp_fa_cluster_align(off, 128); off  = h->off_echo_sel  + (uint64_t) n_kv_heads * h->host_sel_stride;
    h->off_centroids = htp_fa_cluster_align(off, 128); off  = h->off_centroids + (uint64_t) n_kv_heads * h->n_pages_max * h->centroid_bytes;
    h->off_pos_map   = htp_fa_cluster_align(off, 128); off  = h->off_pos_map   + (uint64_t) n_kv_heads * h->n_pages_max * page_keys * 4;
    h->off_n_valid   = htp_fa_cluster_align(off, 128); off  = h->off_n_valid   + htp_fa_cluster_align((uint64_t) h->n_pages_max * 2, 128);
    h->off_k_pages   = htp_fa_cluster_align(off, 16384); off = h->off_k_pages  + (uint64_t) n_kv_heads * h->n_pages_max * h->page_bytes;
    h->off_v_pages   = htp_fa_cluster_align(off, 16384); off = h->off_v_pages  + (uint64_t) n_kv_heads * h->n_pages_max * h->page_bytes;
    h->layer_stride  = htp_fa_cluster_align(off, 65536);
    h->layer0_off    = htp_fa_cluster_align(HTP_FA_CLUSTER_DIR_OFF + (uint64_t) n_layers * HTP_FA_CLUSTER_DIR_STRIDE, 65536);
    return h->layer0_off + (uint64_t) n_layers * h->layer_stride;
}

#define HTP_OP_MAX_DIMS    4    // aka GGML_MAX_DIMS
#define HTP_OP_MAX_INPUTS  8    // sparse flash-attention carries sel (src 5) and its per-row count (src 6); hetero decode FA carries its control buffer (src 7)
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
    uint16_t dst[HTP_OP_MAX_OUTPUTS];   // Output tensor indices (8 + 4 uint16 = 24 B, 64-bit aligned)
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

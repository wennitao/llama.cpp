#pragma once

// Block-sparse flash attention: the host-side policy.
//
// EXPERIMENTAL and off by default. Enabled with LLAMA_SPARSE_ATTN=<density percent>,
// e.g. LLAMA_SPARSE_ATTN=25. Backends that do not implement the src[5] indirection
// reject the op and it falls back to dense, so this can only ever cost speed.
//
// The numbers here are properties of the Hexagon HMX kernel (docs/backend/snapdragon/
// sparse-attention.md). They are in generic code because the SELECTION has to be built
// where the KV cache lives; a backend that wants different ones will want a different
// selection too.

#include <cstdint>
#include <cstdlib>
#include <cstring>

#define LLAMA_SPARSE_ATTN_BS 64   // KV block the selection names
#define LLAMA_SPARSE_ATTN_BQ 256  // query block one list serves (k = 4 blocks of 64)

// Rows actually averaged per block, out of BS/BQ. Both sides subsample; neither is free
// but they are not equally cheap. A query block's rows are a redundant view of the same
// ranking question, so Q64 -> Q4 costs 0.001 of recall. A key block's CENTROID is the
// object being ranked, so a sample estimates it rather than repeating it -- K is 5-9x more
// sensitive. Measured over 11 RULER/LongBench datasets x 2 models at k=4, 25%:
// K16 costs 0.2 points, K8 costs 0.5, K4 costs 0.9, K1 costs 3.6 and falls below having no
// scorer at all. K8 is the knee: 2.6x cheaper than a full K mean on device (742 vs 1917 us
// whole-graph at Lq=2048, Lk=4096) for half a point.
#define LLAMA_SPARSE_ATTN_QSUB 4
#define LLAMA_SPARSE_ATTN_KSUB 8

// Density percent from the environment; 0 = off. "thr:<c>" parses as 0 here.
static inline uint32_t llama_sparse_attn_density() {
    const char * s = getenv("LLAMA_SPARSE_ATTN");
    if (!s) {
        return 0;
    }
    const int v = atoi(s);
    return (v > 0 && v < 100) ? (uint32_t) v : 0;
}

// Threshold mode: LLAMA_SPARSE_ATTN=thr:<c> replaces fixed top-u with a per-row adaptive
// rule computed on device. Each 64-query block keeps KV block j iff softmax(scores)_j *
// avail > c, the four blocks of one 256-query tile union their memberships, and the tile
// serves that union with its own per-row length (FLASH_ATTN_EXT src[6]).
//
// Measured against fixed-u on wikitext-2 at ctx=4096 (bf16 reference, per-head scores):
// c=1.0 gives the deployed fixed-u pipeline's quality at ~0.68x its kernel cost, and
// c=0.3 gives DENSE quality at the deployed cost. The rule is per-element rather than
// XAttention's cumulative-mass cut because it needs no sort/scatter in the graph, and
// the two sit on the same measured quality/cost curve.
static inline float llama_sparse_attn_thr() {
    const char * s = getenv("LLAMA_SPARSE_ATTN");
    if (!s || strncmp(s, "thr:", 4) != 0) {
        return 0.0f;
    }
    const float c = (float) atof(s + 4);
    return (c > 0.0f && c < 100.0f) ? c : 0.0f;
}

// Heterogeneous split: LLAMA_SPARSE_ATTN_CSTAR=<c>, 0 (the default) = off.
//
// One 256-query tile is R = BQ/BS fine 64-query sub-blocks, and c(b) counts how many of
// them picked KV block b. The tile today runs the UNION, i.e. every b with c(b) >= 1.
// With the split on, only c(b) >= c_star stays on the NPU; a block that just one or two
// sub-blocks wanted is an EXCEPTION of those sub-blocks and goes to the GPU, which
// returns its partial through the FLASH_ATTN_EXT fold buffer (src[7]).
//
// c_star = 1 is the control arm, not a no-op: the shared list is the whole union again
// and every exception list is empty, so the GPU side must measure as a no-op. c_star > R
// would leave nothing shared, so it is rejected here.
//
// WHAT THE SPLIT CHANGES NUMERICALLY, because it is easy to assert the wrong thing here.
// Under the union every row of a tile attends to {b : c(b) >= 1}. Under the split, row j
// attends to {b : c(b) >= c_star} together with its own exceptions, which is sel_j union
// {c >= c_star} -- a strict SUBSET of the union whenever some block was wanted by another
// sub-block and not by j. So c_star > 0 is NOT bit-equal to c_star = 0 and must not be
// tested as if it were: it moves the policy from the union toward the exact per-sub-block
// selection, which is a quality change to measure (perplexity, RULER), not a regression.
// The invariant that DOES hold exactly is c_star = 0: node for node, bit for bit, today.
//
// Only threshold mode (LLAMA_SPARSE_ATTN=thr:<c>) has the per-sub-block memberships the
// split is cut from; fixed-u mode never forms c(b) and ignores this.
static inline uint32_t llama_sparse_attn_cstar() {
    const char * s = getenv("LLAMA_SPARSE_ATTN_CSTAR");
    if (!s) {
        return 0;
    }
    const int v = atoi(s);
    const int r = LLAMA_SPARSE_ATTN_BQ / LLAMA_SPARSE_ATTN_BS;
    return (v >= 1 && v <= r) ? (uint32_t) v : 0;   // caller warns on a set-but-rejected value
}

// Head start for the split: LLAMA_SPARSE_ATTN_HEADSTART=K tiles (256-query block x KV head, KV
// head 0's tiles first) run the union on the NPU with no GPU work, so the GPU's launch latency
// lands in their shadow. Measured on device: 6 tiles (~0.6 ms of NPU work) hide the ~0.5 ms
// transient completely at the deployed operating point. 0 = off.
static inline uint32_t llama_sparse_attn_headstart() {
    const char * s = getenv("LLAMA_SPARSE_ATTN_HEADSTART");
    if (!s) {
        return 0;
    }
    const int v = atoi(s);
    return v > 0 ? (uint32_t) v : 0;
}

// Per-(tile, KV head) cap on the exception pairs handed to the GPU (LLAMA_SPARSE_ATTN_GPU_CAP=B,
// 0 = off). Pairs past the cap rejoin that tile's shared list. Each pair costs the GPU ~12 us and
// saves the NPU ~3 us per op-tile, so beyond ~4 the GPU falls behind the NPU's pool pass and the
// NPU waits; the cap keeps every stage's GPU work inside the NPU's shadow with a small head start.
// Exact-mask experiment: LLAMA_SPARSE_ATTN_EXACT=1 (threshold mode, CSTAR unset) hands the FA op
// the FULL per-sub-block membership as src[8]; the HMX kernel keeps the 256-query tile and its
// union list but masks every (32-row group, KV block) the group's sub-block did not select, so the
// large tile computes exactly the per-64-block policy. The NPU-only arm of the "same pairs" A/B
// against the bq=64 build and the NPU/GPU split.
static inline bool llama_sparse_attn_exact() {
    const char * s = getenv("LLAMA_SPARSE_ATTN_EXACT");
    return s && atoi(s) > 0;
}

static inline uint32_t llama_sparse_attn_gpu_cap() {
    const char * s = getenv("LLAMA_SPARSE_ATTN_GPU_CAP");
    if (!s) {
        return 0;
    }
    const int v = atoi(s);
    return v > 0 ? (uint32_t) v : 0;
}

// Extra in-graph invariant checks for the split. Off by default because each one is a
// real node in every layer.
static inline bool llama_sparse_attn_debug() {
    const char * s = getenv("LLAMA_SPARSE_ATTN_DEBUG");
    return s && atoi(s) != 0;
}

// Block scorer for threshold selection. LLAMA_SPARSE_ATTN_SCORER=reps4 scores every query
// head's four sampled rows against four K representatives per block; the default is the
// sampled mean.
//
// LLAMA_SPARSE_ATTN_SCORER=pooled keeps that recipe but makes every representative the mean of
// 16 consecutive rows, so all 64 query and key rows of a block are read: each score is then the
// exact mean logit of a 16x16 tile, the quantity XAttention's antidiagonal samples. Measured on
// exact attention over RULER 4k/8k (prerope-selector-20261001): +3.2 points of the non-forced
// mass and -20% sparse-output error against reps4 at equal density.
//
// LLAMA_SPARSE_ATTN_SCORER=pooledk pools only the keys and keeps reps4's sampled query rows: on the same exact-attention
// audit it keeps nearly all of the gain (+2.9 of the +3.2 points) for about half the pooling cost.
static inline bool llama_sparse_attn_pooled() {
    const char * s = getenv("LLAMA_SPARSE_ATTN_SCORER");
    return s && (strcmp(s, "pooled") == 0 || strcmp(s, "pooledk") == 0);
}

static inline bool llama_sparse_attn_pooled_q() {
    const char * s = getenv("LLAMA_SPARSE_ATTN_SCORER");
    return s && strcmp(s, "pooled") == 0;
}

static inline bool llama_sparse_attn_reps4() {
    const char * s = getenv("LLAMA_SPARSE_ATTN_SCORER");
    return s && (strcmp(s, "reps4") == 0 || llama_sparse_attn_pooled());
}

// n_kv_blocks the HMX kernel will choose for a given u. Transcribed from
// hmx_fa_find_chunk_size (ggml/src/ggml-hexagon/htp/flash-attn-ops.h:370+) and checked
// against the device's own fa-params line at 20 values of u: 20/20 exact.
//
// The shorthand "largest divisor of u <= 8" is NOT this function -- it drops the
// pipeline cap, which usually binds first. At u=16 the cap gives m=4 and four chunks,
// not m=8 and two.
static inline uint32_t llama_sparse_attn_nkvb(uint32_t u, uint32_t bs) {
    const uint32_t kv_eff = u * bs;
    const uint32_t search = ((kv_eff - 1) / 2) / bs * bs;      // pipeline cap
    const uint32_t capped = search < (u < 8 ? u : 8) * bs ? search : (u < 8 ? u : 8) * bs;
    const uint32_t limit  = capped < bs ? bs : capped / bs * bs;
    uint32_t m = 1;
    for (uint32_t d = 1; d * bs <= limit; ++d) {
        if (u % d == 0) {
            m = d;
        }
    }
    return (u + m - 1) / m;
}

// Pick u for a target density.
//
// The measured (kv, u) surface says two things that decide this.
//
// First, the kernel's cost does not depend on kv at all -- only on u and the query count.
// u=16 costs 3112 / 3105 / 3110 us at kv = 1024 / 2048 / 4096 (0.2% spread). It touches u
// blocks and nothing else. So the speedup against dense grows linearly with context, and
// the right question is never "what fraction" but "how many blocks can I afford".
//
// Second, cost is dominated by the CHUNK COUNT, not by u. At kv=4096, nb=1024: every u
// with three chunks lands between 1764 and 3771 us, while u=17 (seventeen chunks, m=1)
// costs 7969 and u=59 costs 26965 -- three times DENSE. A chunk is worth roughly five
// blocks.
//
// So: minimise n_kv_blocks over a small window above the naive u, tie-breaking to the
// smallest u. This never selects fewer blocks than the target, and when it moves it
// usually moves to something both cheaper and larger. The one case in the measured
// surface where it moves without needing to (u0=16 -> 18) costs 2.7% and buys two extra
// blocks of recall, which is a trade worth making.
//
// The previous rule -- keep u0 unless its chunk count exceeded twice the window minimum --
// was too lax and left up to 1.44x on the table (u0=5 kept 5 chunks at 2632 us when u=6
// runs in three at 1832).
static inline uint32_t llama_sparse_attn_round_u(uint32_t u0, uint32_t n_blocks, uint32_t bs) {
    const uint32_t hi = u0 + 8 < n_blocks ? u0 + 8 : n_blocks - 1;
    uint32_t best = u0, best_c = llama_sparse_attn_nkvb(u0, bs);
    for (uint32_t u = u0 + 1; u <= hi; ++u) {
        const uint32_t c = llama_sparse_attn_nkvb(u, bs);
        if (c < best_c) {          // strict: a tie keeps the smaller u, which is cheaper
            best_c = c;
            best   = u;
        }
    }
    return best;
}

static inline uint32_t llama_sparse_attn_pick_u(uint32_t n_blocks, uint32_t density_pct, uint32_t bs) {
    const uint32_t u0 = (n_blocks * density_pct + 99) / 100;
    if (u0 < 3 || u0 >= n_blocks) {
        return 0;                                  // nothing to gain; caller runs dense
    }
    return llama_sparse_attn_round_u(u0, n_blocks, bs);
}

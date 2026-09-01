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

// Density percent from the environment; 0 = off.
static inline uint32_t llama_sparse_attn_density() {
    const char * s = getenv("LLAMA_SPARSE_ATTN");
    if (!s) {
        return 0;
    }
    const int v = atoi(s);
    return (v > 0 && v < 100) ? (uint32_t) v : 0;
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

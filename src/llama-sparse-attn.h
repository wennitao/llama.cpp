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
// Deliberately NOT "minimise the cost model": that fit is good to 3% but its 16% worst
// case is the size of the gap between adjacent good u, and it ranks u=16 against u=18
// backwards (it predicts 18 is cheaper; measured, 16 is -- 6339 vs 6451 us). The model
// is only trusted for what it gets right by an order of magnitude: a u whose chunk count
// is several times the neighbourhood minimum costs 2-3x, because every chunk is a fixed
// ~161 us regardless of how much KV it holds.
//
// So: keep the naive u unless its chunk count is anomalous. Against the 20 measured
// points this fires 8 times, gains 1.32x-3.00x every time, and never fires on a u that
// was already good -- it cannot lose.
//
// This is load-bearing rather than insurance. n_kv is padded to a multiple of 256, so
// NBk is a multiple of 4 and u0 = NBk/4 walks every integer, primes included: 28 of the
// 64 padded n_kv values up to 16384 have an anomalous u0. n_kv=4352 gives u0=17, which
// measured 2.49x the cost of u=18.
static inline uint32_t llama_sparse_attn_pick_u(uint32_t n_blocks, uint32_t density_pct, uint32_t bs) {
    const uint32_t u0 = (n_blocks * density_pct + 99) / 100;
    if (u0 < 3 || u0 >= n_blocks) {
        return 0;                                  // nothing to gain; caller runs dense
    }
    const uint32_t hi = u0 + 8 < n_blocks ? u0 + 8 : n_blocks - 1;
    uint32_t lo_nkvb = llama_sparse_attn_nkvb(u0, bs);
    for (uint32_t u = u0; u <= hi; ++u) {
        const uint32_t c = llama_sparse_attn_nkvb(u, bs);
        if (c < lo_nkvb) {
            lo_nkvb = c;
        }
    }
    if (llama_sparse_attn_nkvb(u0, bs) <= 2 * lo_nkvb) {
        return u0;
    }
    for (uint32_t u = u0; u <= hi; ++u) {
        if (llama_sparse_attn_nkvb(u, bs) == lo_nkvb) {
            return u;
        }
    }
    return u0;
}

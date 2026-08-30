// The block-sparse attention u policy, checked against the device.
//
// llama_sparse_attn_nkvb transcribes hmx_fa_find_chunk_size, which lives in a
// backend-private header this file cannot include -- so the reference is the Hexagon
// kernel's own fa-params log line, read off an HTP0 run at kv=4096, nb=2048, bq=256.
// If the kernel's chunk-size search changes, this test is what notices.

#include "../src/llama-sparse-attn.h"

#include <cstdio>
#include <initializer_list>

int main() {
    int bad = 0;

    // (u, n_kv_blocks) as reported by the device.
    static const int dev[][2] = {
        { 9, 3}, {11, 11}, {12,  3}, {13, 13}, {15,  3}, {16,  4}, {17, 17},
        {18, 3}, {19, 19}, {20,  4}, {21,  3}, {22, 11}, {23, 23}, {24,  3},
        {25, 5}, {27,  9}, {28,  4}, {30,  5}, {31, 31}, {32,  4},
    };
    const int n = (int) (sizeof(dev) / sizeof(dev[0]));

    for (int i = 0; i < n; ++i) {
        const uint32_t got = llama_sparse_attn_nkvb((uint32_t) dev[i][0], LLAMA_SPARSE_ATTN_BS);
        if ((int) got != dev[i][1]) {
            printf("FAIL nkvb: u=%d host says %u, device says %d\n", dev[i][0], got, dev[i][1]);
            bad++;
        }
    }

    // The policy may only ever trade UP: never fewer blocks than the naive u, never out
    // of range, and never a worse chunk count than the u it replaced. A policy that can
    // lose is worse than no policy, because it loses silently.
    for (uint32_t n_bk = 8; n_bk <= 512; ++n_bk) {
        for (uint32_t d : {6u, 12u, 25u, 50u}) {
            const uint32_t u0 = (n_bk * d + 99) / 100;
            const uint32_t u  = llama_sparse_attn_pick_u(n_bk, d, LLAMA_SPARSE_ATTN_BS);
            if (u == 0) {
                continue;                       // caller runs dense
            }
            if (u < u0 || u >= n_bk) {
                printf("FAIL range: NBk=%u d=%u u0=%u u=%u\n", n_bk, d, u0, u);
                bad++;
            }
            if (llama_sparse_attn_nkvb(u, LLAMA_SPARSE_ATTN_BS) >
                llama_sparse_attn_nkvb(u0, LLAMA_SPARSE_ATTN_BS)) {
                printf("FAIL chunks: NBk=%u d=%u u0=%u u=%u made it worse\n", n_bk, d, u0, u);
                bad++;
            }
            // The kernel drops to n_threads=1, pipeline=0 below FA_MIN_KV_BLOCKS = 3
            // (ggml-hexagon.cpp:2139). It cannot be reached from u -- the chunk-size
            // search caps Bc at align_down((kv_eff-1)/2, bs) precisely to prevent it --
            // but that is a property of the search, so assert it rather than assume it.
            if (llama_sparse_attn_nkvb(u, LLAMA_SPARSE_ATTN_BS) < 3) {
                printf("FAIL cliff: NBk=%u d=%u u=%u gives %u chunks\n", n_bk, d, u,
                       llama_sparse_attn_nkvb(u, LLAMA_SPARSE_ATTN_BS));
                bad++;
            }
        }
    }

    // NOT asserted: that u is monotone in n_kv. It is not, and deliberately so. At
    // NBk=28 the naive u0=7 has seven chunks, so the policy moves it to 9 (three
    // chunks); at NBk=32 the naive u0=8 is already fine and stays. So u dips 9 -> 8 as
    // the cache grows past 1792 -> 2048, eight times over n_kv up to 16384. The dip is
    // always from a BOOSTED u back to a naive one -- u >= u0 always holds, which is the
    // invariant that matters -- and the alternative is either giving up a 2x on the
    // anomalous shapes or carrying the previous u across ubatches, which would stop u
    // being a pure function of n_kv and break the can_reuse shape check.

    printf("%s\n", bad ? "FAILED" : "OK");
    return bad != 0;
}
